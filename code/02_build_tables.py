#!/usr/bin/env python
"""Phase 3+4 — clinical-care variables, then the manuscript tables.

Adds the Table 3 variables that need tables 01 does not read (procedures,
medications, microbiology, position), then builds every aggregate table the
manuscript needs. Output is aggregate only and lands in
<site>_upload_to_box/ — the folder that may leave the site. Small cells are masked in
the formatted tables only when study.suppress_small_cells is true in
config/donor_criteria.yaml; the raw-count files are never masked, so they pool.

Tables produced:
    table2_characteristics.csv   demographics + labs + contraindications by definition
    table3_clinical_care.csv     procedures, medications, organ support by definition
    tableS2_missingness.csv      per-variable missingness
    hospital_level_counts.csv    per analytic hospital (Will, 2026-08-20)

    uv run python code/02_build_tables.py      # reads config/config.json
"""
from __future__ import annotations

import sys

# Print Unicode (≤, →) on any console, Windows included
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path

import duckdb
import polars as pl
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from utils.audit import StageAudit
from utils.dtypes import write_parquet                      
from utils.config import config                    
from utils.criteria import (DEFS, contraindication_prefix,
                            contraindication_sql, load_contraindications)

SITE = config["site_name"]
TABLES = Path(config["tables_path"])
FT = config.get("file_type", "parquet")
INTER = Path(config["output_intermediate"])
FINAL = Path(config["output_final"])
FINAL.mkdir(parents=True, exist_ok=True)

CRIT = yaml.safe_load((REPO / "config/donor_criteria.yaml").read_text())
CARE = CRIT["clinical_care"]
SUPPRESS = CRIT["study"]["small_cell_suppression_n"]
SUPPRESS_ON = CRIT["study"].get("suppress_small_cells", True)
if not SUPPRESS_ON:
    print("NOTE: small-cell masking is OFF (config study.suppress_small_cells=false)")

_CD = CRIT["clif_donor"]

audit = StageAudit(SITE, FINAL, label="tables")
con = duckdb.connect()


def tbl(name: str) -> Path:
    return TABLES / f"clif_{name}.{FT}"


def have(name: str) -> bool:
    return tbl(name).exists()


cohort = pl.read_parquet(INTER / "cohort_with_definitions.parquet")
print(f"[{SITE}] {cohort.height:,} decedents")
# Every stay in each decedent's encounter block (written by step 01), so the
# clinical-care variables cover the same stays as the definitions.
MEMBERS = pl.read_parquet(INTER / "encounter_block_members.parquet").with_columns(
    pl.col("hospitalization_id").cast(pl.Utf8))
ids = MEMBERS.select("hospitalization_id").unique().to_pandas()
con.register("ids_df", ids)

missing_tables: list[str] = []


def add_flag(name: str, table: str, sql: str) -> None:
    """Attach a per-hospitalization boolean. Absent table -> null column, recorded."""
    global cohort
    if not have(table):
        missing_tables.append(table)
        cohort = cohort.with_columns(pl.lit(None, pl.Boolean).alias(name))
        audit.note(f"var_{name}", f"UNAVAILABLE — clif_{table} absent at this site")
        print(f"  {name:34s} SKIPPED (no clif_{table})")
        return
    try:
        f = con.sql(sql.format(p=tbl(table))).pl()
    except duckdb.BinderException as e:          # a missing or renamed column; anything else stops the run
        cohort = cohort.with_columns(pl.lit(None, pl.Boolean).alias(name))
        audit.note(f"var_{name}", f"UNAVAILABLE — {type(e).__name__}: {str(e)[:120]}")
        print(f"  {name:34s} SKIPPED ({type(e).__name__})")
        return
    per_patient = (f.rename({f.columns[1]: name})
                   .with_columns(pl.col("hospitalization_id").cast(pl.Utf8))
                   .join(MEMBERS, on="hospitalization_id", how="inner")
                   .group_by("patient_id").agg(pl.col(name).cast(pl.Boolean).any()))
    cohort = (cohort.join(per_patient, on="patient_id", how="left")
                    .with_columns(pl.col(name).fill_null(False).cast(pl.Boolean)))
    n = int(cohort.filter(pl.col(name))["patient_id"].n_unique())
    print(f"  {name:34s} {n:>7,}")


print("\n-- clinical care variables --")

# Neurologic procedures — CPT and ICD-10-PCS OR-ed together.
# Sites code inpatient procedures in different vocabularies (some PCS only,
# some CPT only for these concepts), and spell the format label differently
# ("icd10pcs" / "ICD10PCS" / "ICD-10-PCS"), so the label is normalised by
# lowercasing and stripping non-alphanumerics before comparison.
PROC = pl.read_csv(REPO / CARE["procedures_file"], comment_prefix="#", infer_schema_length=0)
FMT_NORM = "regexp_replace(lower(CAST(procedure_code_format AS VARCHAR)),'[^a-z0-9]','','g')"
CODE = "upper(CAST(procedure_code AS VARCHAR))"

for concept in sorted(PROC["concept"].unique()):
    sub = PROC.filter(pl.col("concept") == concept)
    clauses = []
    exact = [r["pattern"] for r in sub.iter_rows(named=True)
             if r["vocabulary"] == "cpt" and r["match_type"] == "exact"]
    if exact:
        clauses.append(f"({FMT_NORM}='cpt' AND {CODE} IN ("
                       + ",".join(f"'{c}'" for c in exact) + "))")
    for r in sub.iter_rows(named=True):
        if r["vocabulary"] == "icd10pcs":
            clauses.append(f"({FMT_NORM}='icd10pcs' AND regexp_matches({CODE}, '{r['pattern']}'))")
    add_flag(f"proc_{concept}", "patient_procedures", f"""
        SELECT CAST(hospitalization_id AS VARCHAR) hospitalization_id, TRUE flag
        FROM read_parquet('{{p}}')
        WHERE CAST(hospitalization_id AS VARCHAR) IN (SELECT hospitalization_id FROM ids_df)
          AND ({' OR '.join(clauses)})
        GROUP BY 1""")

# ── donor administrative codes (v6 Tables 1 and 2) ───────────────────────────
# Diagnosis codes go through hospital_diagnosis, CPT
# through patient_procedures. Codes and vocabulary labels are both normalised —
# the sites spell the vocabulary three different ways.
DAC_PATH = REPO / CRIT["reporting"]["donor_administrative_codes_file"]
if DAC_PATH.exists():
    DAC = pl.read_csv(DAC_PATH, comment_prefix="#", infer_schema_length=0)
    _nrm = lambda s: "".join(ch for ch in str(s) if ch.isalnum()).upper()
    DXC = "upper(regexp_replace(CAST(diagnosis_code AS VARCHAR),'[^0-9A-Za-z]','','g'))"
    PRC = "upper(regexp_replace(CAST(procedure_code AS VARCHAR),'[^0-9A-Za-z]','','g'))"
    for r in DAC.filter(pl.col("match_type") == "exact").iter_rows(named=True):
        code = _nrm(r["code"])
        if r["vocabulary"] == "icd10cm":
            add_flag(f"dx_{code.lower()}", "hospital_diagnosis", f"""
                SELECT CAST(hospitalization_id AS VARCHAR) hospitalization_id, TRUE flag
                FROM read_parquet('{{p}}')
                WHERE CAST(hospitalization_id AS VARCHAR) IN (SELECT hospitalization_id FROM ids_df)
                  AND {DXC} = '{code}' GROUP BY 1""")
        else:
            add_flag(f"cpt_{code.lower()}", "patient_procedures", f"""
                SELECT CAST(hospitalization_id AS VARCHAR) hospitalization_id, TRUE flag
                FROM read_parquet('{{p}}')
                WHERE CAST(hospitalization_id AS VARCHAR) IN (SELECT hospitalization_id FROM ids_df)
                  AND {PRC} = '{code}' GROUP BY 1""")

# Cancer alone, separated from the pooled contraindication flag, because v6
# Table 1 lists "Cancer" and "Positive blood culture" as separate rows. Same
# list and matching as step 01.
CONTRA_CODES = load_contraindications(REPO / _CD["contraindications"]["icd10_file"])
_DX_NORM = "lower(regexp_replace(CAST(diagnosis_code AS VARCHAR),'[^0-9A-Za-z]','','g'))"
_DX_SYS = "regexp_replace(lower(CAST(diagnosis_code_format AS VARCHAR)),'[^a-z0-9]','','g')"
add_flag("icd10_cancer_only", "hospital_diagnosis", f"""
    SELECT CAST(hospitalization_id AS VARCHAR) hospitalization_id, TRUE flag
    FROM read_parquet('{{p}}')
    WHERE CAST(hospitalization_id AS VARCHAR) IN (SELECT hospitalization_id FROM ids_df)
      AND {_DX_SYS} IN ('icd10','icd10cm')
      AND {contraindication_sql(_DX_NORM, CONTRA_CODES)}
    GROUP BY 1""")

# Infectious screening, from microbiology_culture. This is culture positivity,
# NOT serostatus — serologies live in microbiology_nonculture, which not every site has.
for _org, _rx in [("cmv", "cytomegalo|cmv"), ("ebv", "epstein|ebv"),
                  ("tuberculosis", "tubercul|mycobact"), ("aspergillus", "aspergill")]:
    add_flag(f"micro_{_org}", "microbiology_culture", f"""
        SELECT CAST(hospitalization_id AS VARCHAR) hospitalization_id, TRUE flag
        FROM read_parquet('{{p}}')
        WHERE CAST(hospitalization_id AS VARCHAR) IN (SELECT hospitalization_id FROM ids_df)
          AND regexp_matches(lower(CAST(organism_category AS VARCHAR)), '{_rx}')
        GROUP BY 1""")

# Medications
for group, cats in CARE["medications"].items():
    lst = ",".join(f"'{c}'" for c in cats)
    for table in ("medication_admin_continuous", "medication_admin_intermittent"):
        col = f"med_{group}_{'cont' if 'continuous' in table else 'intermit'}"
        add_flag(col, table, f"""
            SELECT CAST(hospitalization_id AS VARCHAR) hospitalization_id, TRUE flag
            FROM read_parquet('{{p}}')
            WHERE CAST(hospitalization_id AS VARCHAR) IN (SELECT hospitalization_id FROM ids_df)
              AND lower(CAST(med_category AS VARCHAR)) IN ({lst})
            GROUP BY 1""")

# Fungemia — the one clinical contraindication CLIF can actually see (DECISIONS F-11)
add_flag("contra_fungemia", "microbiology_culture", f"""
    SELECT CAST(hospitalization_id AS VARCHAR) hospitalization_id, TRUE flag
    FROM read_parquet('{{p}}')
    WHERE CAST(hospitalization_id AS VARCHAR) IN (SELECT hospitalization_id FROM ids_df)
      AND lower(CAST(fluid_category AS VARCHAR))='{CARE["fungemia"]["fluid_category"]}'
      AND regexp_matches(lower(COALESCE(CAST(organism_category AS VARCHAR),'')),
                         '{CARE["fungemia"]["organism_regex"]}')
    GROUP BY 1""")

# Denominator for Joel's "% tested" — was a blood culture drawn at all?
add_flag("tested_blood_culture", "microbiology_culture", """
    SELECT CAST(hospitalization_id AS VARCHAR) hospitalization_id, TRUE flag
    FROM read_parquet('{p}')
    WHERE CAST(hospitalization_id AS VARCHAR) IN (SELECT hospitalization_id FROM ids_df)
      AND lower(CAST(fluid_category AS VARCHAR))='blood_buffy'
    GROUP BY 1""")

add_flag("prone_position", "position", """
    SELECT CAST(hospitalization_id AS VARCHAR) hospitalization_id, TRUE flag
    FROM read_parquet('{p}')
    WHERE CAST(hospitalization_id AS VARCHAR) IN (SELECT hospitalization_id FROM ids_df)
      -- exact match: the mCIDE vocabulary is prone / not_prone, so LIKE
      -- '%prone%' also matches not_prone and reports ~everyone as proned.
      AND lower(trim(COALESCE(CAST(position_category AS VARCHAR),''))) = 'prone'
    GROUP BY 1""")

cohort = cohort.with_columns(
    (pl.col("contra_fungemia").fill_null(False) | (~pl.col("no_positive_culture_48hrs"))
     | pl.col("icd10_contraindication")).alias("contra_any"))

write_parquet(cohort, INTER / "cohort_with_variables.parquet", config["timezone"])

# ── definition columns: DEFS, from utils.criteria ────────────────────────────


def mask(col: str) -> pl.Series:
    return cohort[col].fill_null(False)


def suppress(n: int) -> str:
    return f"<{SUPPRESS}" if SUPPRESS_ON and 0 < n < SUPPRESS else f"{n:,}"


def pct(n: int, d: int) -> str:
    if not d:
        return "—"
    if SUPPRESS_ON and 0 < n < SUPPRESS:
        return f"<{SUPPRESS}"
    return f"{n:,} ({100*n/d:.1f}%)"


def median_iqr(col: str, m: pl.Series) -> str:
    s = cohort.filter(m)[col].drop_nulls()
    if s.len() == 0 or (SUPPRESS_ON and s.len() < SUPPRESS):
        return "—"
    return f"{s.median():.1f} ({s.quantile(.25):.1f}-{s.quantile(.75):.1f})"


rows: list[dict] = []


def row(label: str, fn) -> None:
    rows.append({"variable": label, **{d: fn(mask(c)) for d, c in DEFS}})


N = {d: int(cohort.filter(mask(c))["patient_id"].n_unique()) for d, c in DEFS}
rows.append({"variable": "N patients (% of decedents)",
             **{d: pct(n, cohort.height) for d, n in N.items()}})
for lbl, col in [("Age at death, median (IQR)", "age_at_death"),
                 ("Weight, median (IQR) [kg]", "last_weight_kg"),
                 ("Height, median (IQR) [cm]", "last_height_cm"),
                 ("BMI, median (IQR)", "bmi"),
                 ("Terminal creatinine, median (IQR) [mg/dL]", "creatinine_value"),
                 ("Terminal total bilirubin, median (IQR) [mg/dL]", "bilirubin_total_value"),
                 ("Terminal AST, median (IQR) [U/L]", "ast_value"),
                 ("Terminal ALT, median (IQR) [U/L]", "alt_value"),
                 ("Terminal GCS, median (IQR)", "gcs_total_value"),
                 ("Terminal RASS, median (IQR)", "rass_value"),
                 ("Hospital LOS, median (IQR) [days]", "hospital_length_of_stay_days"),
                 ("First ICU LOS, median (IQR) [days]", "first_icu_los_days")]:
    if col in cohort.columns:
        row(lbl, lambda m, c=col: median_iqr(c, m))

row("Male sex, n (%)", lambda m: pct(int(cohort.filter(m & (pl.col("sex_category").str.to_lowercase() == "male")).height), int(m.sum())))
for lbl, col in [("HCV infection", "icd10_hcv"), ("Hypertension", "icd10_htn"),
                 ("Diabetes", "icd10_dm"), ("History of CVA", "icd10_cva")]:
    if col in cohort.columns:
        row(lbl, lambda m, c=col: pct(int(cohort.filter(m & pl.col(c).fill_null(False)).height), int(m.sum())))

for cat in ("race_category", "ethnicity_category"):
    if cat in cohort.columns:
        for v in sorted(x for x in cohort[cat].unique().to_list() if x):
            row(f"{cat.replace('_category','').title()}: {v}",
                lambda m, c=cat, val=v: pct(int(cohort.filter(m & (pl.col(c) == val)).height), int(m.sum())))

# Contraindications — % positive, and % positive among tested (Joel, 2026-08-17)
row("Positive blood culture <=48h before death",
    lambda m: pct(int(cohort.filter(m & (~pl.col("no_positive_culture_48hrs"))).height), int(m.sum())))
if cohort["contra_fungemia"].null_count() < cohort.height:
    row("Fungemia (blood culture)",
        lambda m: pct(int(cohort.filter(m & pl.col("contra_fungemia").fill_null(False)).height), int(m.sum())))
    row("  ... blood culture obtained (denominator)",
        lambda m: pct(int(cohort.filter(m & pl.col("tested_blood_culture").fill_null(False)).height), int(m.sum())))
    row("  ... fungemia among those cultured",
        lambda m: pct(int(cohort.filter(m & pl.col("contra_fungemia").fill_null(False)).height),
                      int(cohort.filter(m & pl.col("tested_blood_culture").fill_null(False)).height)))
row(">=1 relative contraindication", lambda m: pct(int(cohort.filter(m & pl.col("contra_any")).height), int(m.sum())))

# ── machine-readable long form, so the aggregator can pool exactly ───────────
# The display tables above hold formatted strings ("3,246 (60.7%)"), which cannot
# be summed across sites. Every statistic is therefore ALSO emitted with its raw
# numerator, denominator and quantiles, so coordinating/01_combined_report.py can pool counts
# exactly and recompute proportions on the pooled denominator rather than
# averaging percentages.
raw: list[dict] = []


def raw_count(table: str, variable: str, dname: str, m: pl.Series, expr) -> None:
    d = int(m.sum())
    raw.append({"table": table, "variable": variable, "definition": dname, "stat": "count",
                "n": int(cohort.filter(m & expr).height), "denom": d,
                "median": None, "q25": None, "q75": None, "n_nonnull": None})


def raw_median(table: str, variable: str, dname: str, m: pl.Series, col: str) -> None:
    s = cohort.filter(m)[col].drop_nulls()
    raw.append({"table": table, "variable": variable, "definition": dname, "stat": "median",
                "n": None, "denom": int(m.sum()),
                "median": float(s.median()) if s.len() else None,
                "q25": float(s.quantile(.25)) if s.len() else None,
                "q75": float(s.quantile(.75)) if s.len() else None,
                "n_nonnull": int(s.len())})


for _d, _c in DEFS:
    _m = mask(_c)
    raw.append({"table": "table2", "variable": "N patients", "definition": _d, "stat": "count",
                "n": int(_m.sum()), "denom": cohort.height,
                "median": None, "q25": None, "q75": None, "n_nonnull": None})
    for _lbl, _col in [("Age at death", "age_at_death"), ("Weight [kg]", "last_weight_kg"),
                       ("Height [cm]", "last_height_cm"), ("BMI", "bmi"),
                       ("Terminal creatinine [mg/dL]", "creatinine_value"),
                       ("Terminal total bilirubin [mg/dL]", "bilirubin_total_value"),
                       ("Terminal AST [U/L]", "ast_value"), ("Terminal ALT [U/L]", "alt_value"),
                       ("Terminal BUN [mg/dL]", "bun_value"),
                       ("Terminal sodium [mmol/L]", "sodium_value"),
                       ("Terminal GCS", "gcs_total_value"), ("Terminal RASS", "rass_value"),
                       ("Hospital LOS [days]", "hospital_length_of_stay_days"),
                       ("First ICU LOS [days]", "first_icu_los_days")]:
        if _col in cohort.columns:
            raw_median("table2", _lbl, _d, _m, _col)
    raw_count("table2", "Male sex", _d, _m, pl.col("sex_category").str.to_lowercase() == "male")
    for _lbl, _col in [("HCV infection", "icd10_hcv"), ("Hypertension", "icd10_htn"),
                       ("Diabetes", "icd10_dm"), ("History of CVA", "icd10_cva")]:
        if _col in cohort.columns:
            raw_count("table2", _lbl, _d, _m, pl.col(_col).fill_null(False))
    for _cat in ("race_category", "ethnicity_category"):
        if _cat in cohort.columns:
            for _v in sorted(x for x in cohort[_cat].unique().to_list() if x):
                raw_count("table2", f"{_cat.replace('_category','').title()}: {_v}", _d, _m,
                          pl.col(_cat) == _v)
    # v6 Table 1 lists the CALC cause ranges and the donor/brain-death codes as
    # their own rows, so they are emitted individually rather than only as the
    # combined calc_flag.
    for _lbl, _col in [("Ischemic heart disease", "icd10_ischemic"),
                       ("Cerebrovascular disease", "icd10_cerebro"),
                       ("External causes", "icd10_external"),
                       ("Z52.9 donor of organs", "dx_z529"),
                       ("Cancer", "icd10_cancer_only")]:
        if _col in cohort.columns:
            raw_count("table2", _lbl, _d, _m, pl.col(_col).fill_null(False))
    if "ever_icu" in cohort.columns:
        raw_count("table2", "Admitted to an ICU", _d, _m, pl.col("ever_icu").fill_null(False))
    if "imv_48hr_expire" in cohort.columns:
        raw_count("table2", "Invasive mechanical ventilation", _d, _m,
                  pl.col("imv_48hr_expire").fill_null(False))
    raw_count("table2", "Positive blood culture <=48h", _d, _m, ~pl.col("no_positive_culture_48hrs"))
    if "contra_fungemia" in cohort.columns and cohort["contra_fungemia"].null_count() < cohort.height:
        raw_count("table2", "Fungemia", _d, _m, pl.col("contra_fungemia").fill_null(False))
        raw_count("table2", "Blood culture obtained", _d, _m, pl.col("tested_blood_culture").fill_null(False))
    raw_count("table2", ">=1 relative contraindication", _d, _m, pl.col("contra_any"))
    # Brain death (G93.82). Reporting only — see 01; it is the face-validity
    # check, so it deliberately does not feed any eligibility criterion.
    if "icd10_brain_death" in cohort.columns:
        raw_count("table2", "Brain death (G93.82)", _d, _m,
                  pl.col("icd10_brain_death").fill_null(False))
    for _col in [c for c in cohort.columns
                 if c.startswith(("proc_", "med_", "cpt_", "micro_"))] + \
                ["on_crrt_48h_before_death", "prone_position"]:
        if _col in cohort.columns and cohort[_col].null_count() < cohort.height:
            raw_count("table3", _col, _d, _m, pl.col(_col).fill_null(False))

pl.DataFrame(raw).write_csv(FINAL / "table_stats_raw.csv")
print(f"raw long-form stats: {len(raw)} rows -> table_stats_raw.csv")

# ── binned counts, so the coordinating centre can pool medians exactly ───────
# A median of the pooled patients is not a function of the site medians. Counts
# per fixed-width bin are, to the width of the bin, and they add across sites.
_BINS = [("Age at death", "age_at_death", 1), ("Weight [kg]", "last_weight_kg", 1),
         ("Height [cm]", "last_height_cm", 1), ("BMI", "bmi", .5),
         ("Terminal creatinine [mg/dL]", "creatinine_value", .1),
         ("Terminal total bilirubin [mg/dL]", "bilirubin_total_value", .1),
         ("Terminal AST [U/L]", "ast_value", 10), ("Terminal ALT [U/L]", "alt_value", 10),
         ("Terminal BUN [mg/dL]", "bun_value", 1), ("Terminal sodium [mmol/L]", "sodium_value", 1),
         ("Terminal GCS", "gcs_total_value", 1), ("Terminal RASS", "rass_value", 1),
         ("Hospital LOS [days]", "hospital_length_of_stay_days", .5),
         ("First ICU LOS [days]", "first_icu_los_days", .5)]
_bins: list[dict] = []
for _d, _c in DEFS:
    _m = mask(_c)
    for _lbl, _col, _w in _BINS:
        if _col not in cohort.columns:
            continue
        _s = cohort.filter(_m)[_col].drop_nulls().cast(pl.Float64)
        _g = (_s.to_frame("v").with_columns(((pl.col("v") / _w + 1e-9).floor() * _w).round(6).alias("bin_lo"))
              .group_by("bin_lo").len().sort("bin_lo"))
        assert int(_g["len"].sum()) == _s.len()
        _bins += [{"site": SITE, "table": "table2", "variable": _lbl, "definition": _d,
                   "bin_lo": float(_b), "width": float(_w), "n": int(_n)} for _b, _n in _g.iter_rows()]
_bins_df = pl.DataFrame(_bins, schema={"site": pl.Utf8, "table": pl.Utf8, "variable": pl.Utf8, "definition": pl.Utf8,
                                       "bin_lo": pl.Float64, "width": pl.Float64, "n": pl.Int64})
_bins_df.write_csv(INTER / "value_bins.csv")
if not SUPPRESS_ON:          # bins are small cells; with suppression on they stay local
    _bins_df.write_csv(FINAL / "value_bins.csv")
print(f"value bins: {_bins_df.height} rows -> value_bins.csv" + (" (kept local: suppression on)" if SUPPRESS_ON else ""))

t2 = pl.DataFrame(rows)
t2.write_csv(FINAL / "table2_characteristics.csv")
print(f"\nTable 2: {t2.height} rows -> table2_characteristics.csv")

# ── Table 3 ──────────────────────────────────────────────────────────────────
care_rows = [{"variable": "N patients", **{d: suppress(n) for d, n in N.items()}}]
for col in [c for c in cohort.columns if c.startswith(("proc_", "med_"))] + \
           ["on_crrt_48h_before_death", "prone_position"]:
    if col not in cohort.columns:
        continue
    if cohort[col].null_count() == cohort.height:
        care_rows.append({"variable": f"{col} — NOT CAPTURED AT THIS SITE",
                          **{d: "—" for d, _ in DEFS}})
        continue
    care_rows.append({"variable": col,
                      **{d: pct(int(cohort.filter(mask(c) & pl.col(col).fill_null(False)).height),
                                int(mask(c).sum())) for d, c in DEFS}})
pl.DataFrame(care_rows).write_csv(FINAL / "table3_clinical_care.csv")
print(f"Table 3: {len(care_rows)} rows -> table3_clinical_care.csv")

# ── Table S2 missingness ─────────────────────────────────────────────────────
miss = [{"variable": c, "n_missing": int(cohort[c].null_count()),
         "pct_missing": round(100 * cohort[c].null_count() / cohort.height, 2)}
        for c in ["age_at_death", "bmi", "last_weight_kg", "last_height_cm", "creatinine_value",
                  "bilirubin_total_value", "ast_value", "alt_value", "gcs_total_value",
                  "rass_value", "hospital_length_of_stay_days", "first_icu_los_days",
                  "race_category", "ethnicity_category"] if c in cohort.columns]
for t in sorted(set(missing_tables)):
    miss.append({"variable": f"[table] clif_{t}", "n_missing": cohort.height, "pct_missing": 100.0})
pl.DataFrame(miss).write_csv(FINAL / "tableS2_missingness.csv")
print(f"Table S2: {len(miss)} rows -> tableS2_missingness.csv")

# ── hospital-level (Will, 2026-08-20) ────────────────────────────────────────
hosp = (cohort.filter(pl.col("hospital_label").is_not_null())
        .group_by(["hospital_label", "srtr_ccn_id"])
        .agg([pl.col("patient_id").n_unique().alias("n_decedents"),
              # one row per hospital even when some ADT rows lack the type
              pl.col("hospital_type").drop_nulls().first().alias("hospital_type"),
              *[pl.col(c).fill_null(False).sum().alias(d.replace(" ", "_").replace("-", "_"))
                for d, c in DEFS],
              # CALC under each diagnosis-position reading, so the pooled figures
              # can be drawn for a reading other than the configured one
              # (calc.diagnosis_position) without a re-run. CALC above is the
              # configured reading.
              *[(pl.col(c).fill_null(False) & pl.col("age_75_less").fill_null(False)).sum().alias(n)
                for n, c in (("CALC_any", "calc_cause_any"), ("CALC_primary", "calc_cause_primary"),
                             ("CALC_primary_poa", "calc_cause_primary_poa"))]])
        .sort("n_decedents", descending=True)
        .with_columns(pl.lit(SITE).alias("site")))
# Counts are written UNSUPPRESSED. Suppressing here silently broke pooling:
# a cell of 1-10 became null, and summing nulls across sites read as zero, which
# understated the pooled denominator. These are hospital-level
# aggregate decedent counts, not patient-identifying; masking belongs at the
# publication layer, which is where the dashboard applies it. A `low_cell`
# column marks the rows a published table must mask.
low = pl.any_horizontal([pl.col(c).is_between(1, SUPPRESS - 1)
                         for c in hosp.columns if hosp.schema[c] in (pl.Int64, pl.UInt32, pl.Int32)])
hosp = hosp.with_columns(low.alias("has_low_cell"))
hosp.write_csv(FINAL / "hospital_level_counts.csv")
print(f"Hospital-level: {hosp.height} analytic hospitals -> hospital_level_counts.csv")

audit.record("40_tables_built", cohort, cohort, key="patient_id",
             rule="aggregate tables written to <site>_upload_to_box",
             missing_tables=sorted(set(missing_tables)) or "none")
audit.write()
print(f"\nall outputs -> {FINAL}")

# ── academic vs community ────────────────────────────────────────────────────
# The manuscript compares academic and community hospitals but nothing was ever
# computed for it. hospital_type comes from the terminal ADT record (CLIF field),
# not the crosswalk, which carries no such column. Counts only, so they pool.
_by_type = (cohort.with_columns(pl.col("hospital_type").fill_null("unknown"))
            .group_by("hospital_type")
            .agg([pl.col("patient_id").n_unique().alias("n_decedents"),
                  pl.col("hospital_label").n_unique().alias("n_hospitals"),
                  *[pl.col(c).fill_null(False).sum().alias(d.replace(" ", "_").replace("-", "_"))
                    for d, c in DEFS]])
            .with_columns(pl.lit(SITE).alias("site"))
            .sort("n_decedents", descending=True))
_by_type.write_csv(FINAL / "definition_counts_by_hospital_type.csv")
print("hospital_type: " + ", ".join(
    f"{r['hospital_type']}={r['n_decedents']:,} ({r['n_hospitals']} hosp)"
    for r in _by_type.iter_rows(named=True)))

# ── the cancer codes behind the contraindication criterion ───────────────────
# The criteria are applied in parallel, so the question is not which codes
# dropped patients at a step but which codes the decedents with a
# contraindication carry: per code, decedents carrying it, and decedents
# carrying no other contraindicating code.
con.register("dx_all", con.sql(f"""
    SELECT CAST(hospitalization_id AS VARCHAR) hid, {_DX_NORM} code
    FROM read_parquet('{TABLES}/clif_hospital_diagnosis.{FT}')
    WHERE {_DX_SYS} IN ('icd10','icd10cm')""").df())
_contra_pts = cohort.filter(pl.col("icd10_contraindication").fill_null(False)).select("patient_id")
con.register("contra_ids", MEMBERS.join(_contra_pts, on="patient_id", how="inner").to_pandas())
_hit = con.sql(f"""
    WITH h AS (SELECT DISTINCT d.patient_id, x.code
               FROM contra_ids d JOIN dx_all x ON x.hid = d.hospitalization_id
               WHERE {contraindication_sql("x.code", CONTRA_CODES)})
    SELECT code, count(DISTINCT patient_id) n_decedents,
           count(DISTINCT CASE WHEN NOT EXISTS
                (SELECT 1 FROM h y WHERE y.patient_id = h.patient_id AND y.code <> h.code)
                THEN patient_id END) n_only_this_code
    FROM h GROUP BY 1 ORDER BY n_decedents DESC, code""").pl()   # code breaks ties: stable across runs
_codes = (_hit.with_columns(
    pl.Series("description", [CONTRA_CODES[contraindication_prefix(c, CONTRA_CODES)] for c in _hit["code"]],
              dtype=pl.Utf8),
    pl.col("code").str.to_uppercase(), pl.lit(SITE).alias("site"))
    .select("site", "code", "description", "n_decedents", "n_only_this_code"))
_codes.write_csv(INTER / "contraindication_codes.csv")
# Per-code counts are small cells. Unless study.ship_exclusion_code_detail is
# true the shipped file carries the total only; the per-code table stays local.
if CRIT["study"].get("ship_exclusion_code_detail", False):
    _codes.write_csv(FINAL / "contraindication_codes.csv")
else:
    pl.DataFrame({"site": [SITE], "code": [None], "n_decedents": [int(_contra_pts.height)],
                  "n_only_this_code": [None],
                  "description": [f"{_codes.height} distinct codes; the per-code table is in the site's local output folder"]},
                 schema={"site": pl.Utf8, "code": pl.Utf8, "n_decedents": pl.Int64, "n_only_this_code": pl.Int64,
                         "description": pl.Utf8}
                 ).select("site", "code", "description", "n_decedents", "n_only_this_code").write_csv(FINAL / "contraindication_codes.csv")
print(f"Contraindication codes: {_contra_pts.height:,} decedents, {_codes.height} codes -> contraindication_codes.csv")
