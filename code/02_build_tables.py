#!/usr/bin/env python
"""Phase 3+4 — clinical-care variables, then the manuscript tables.

Adds the Table 3 variables that need tables 01 does not read (procedures,
medications, microbiology, position), then builds every aggregate table the
manuscript needs. Output is aggregate only and lands in
<site>_upload_to_box/ — the folder that may leave the site. Small cells are masked in
the formatted tables only when study.suppress_small_cells is true in
config/donor_criteria.yaml; the raw-count files are never masked, so they pool.

Tables produced, matching CLIF-donor manuscript_4_WFP:
    table2_characteristics.csv   demographics + labs + contraindications by definition
    table3_clinical_care.csv     procedures, medications, organ support by definition
    tableS2_missingness.csv      per-variable missingness
    hospital_level_counts.csv    per analytic hospital (Will, 2026-08-20)

    CLIF_DONOR_SITE=ucmc python code/02_build_tables.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import polars as pl
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from utils.audit import StageAudit
from utils.dtypes import write_parquet                      # noqa: E402
from utils.config import config                         # noqa: E402

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

# Values named in the CONSORT labels come from the YAML that step 01 applied,
# so a label cannot state a threshold the counts did not use.
_CD = CRIT["clif_donor"]
_AGE, _IMV_H = f'{_CD["age_at_death_max"]:g}', f'{_CD["imv_hours_before_death"]:g}'
_CX_H = f'{_CD["contraindications"]["positive_blood_culture_hours"]:g}'
_AGE_IN, _AGE_OUT = ("\u2264", ">") if _CD["age_operator"] == "<=" else ("<", "\u2265")
_CAUSES = ", ".join(f"{lo}-{hi}" for lo, hi in CRIT["calc"]["cause_icd10_prefixes"].values())

audit = StageAudit(SITE, FINAL, label="tables")
con = duckdb.connect()


def tbl(name: str) -> Path:
    return TABLES / f"clif_{name}.{FT}"


def have(name: str) -> bool:
    return tbl(name).exists()


cohort = pl.read_parquet(INTER / "cohort_with_definitions.parquet")
print(f"[{SITE}] {cohort.height:,} decedents")
ids = pl.DataFrame({"hospitalization_id": cohort["hospitalization_id"].cast(pl.Utf8)}).to_pandas()
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
    except Exception as e:                       # a missing column, not a missing table
        cohort = cohort.with_columns(pl.lit(None, pl.Boolean).alias(name))
        audit.note(f"var_{name}", f"UNAVAILABLE — {type(e).__name__}: {str(e)[:120]}")
        print(f"  {name:34s} SKIPPED ({type(e).__name__})")
        return
    cohort = (cohort.join(f.rename({f.columns[1]: name}), on="hospitalization_id", how="left")
                    .with_columns(pl.col(name).fill_null(False).cast(pl.Boolean)))
    n = int(cohort.filter(pl.col(name))["patient_id"].n_unique())
    print(f"  {name:34s} {n:>7,}")


print("\n-- clinical care variables --")

# Neurologic procedures — CPT and ICD-10-PCS OR-ed together.
# Sites code inpatient procedures in different vocabularies (UCMC: PCS only;
# RUSH: CPT only for these concepts), and spell the format label differently
# ("icd10pcs" / "ICD10PCS" / "ICD-10-PCS"), so the label is normalised by
# lowercasing and stripping non-alphanumerics before comparison.
PROC = pl.read_csv(REPO / "utils/codes/neuro_procedures.csv", comment_prefix="#",
                   infer_schema_length=0)
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
# Emily's face-validity codes, one flag per code because the manuscript lists
# each CPT as its own row. Diagnosis codes go through hospital_diagnosis, CPT
# through patient_procedures. Codes and vocabulary labels are both normalised —
# the sites spell the vocabulary three different ways.
DAC_PATH = REPO / "utils/codes/donor_administrative_codes.csv"
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
# Table 1 lists "Cancer" and "Positive blood culture" as separate rows.
_cancer_codes = (pl.read_csv(REPO / "utils/icd10_contraindications.csv", infer_schema_length=0)
                 .rename({"ICD-10-CM": "code"})
                 .filter(pl.col("dx_broad").is_in(["cancer", "other"])))
_cc = ",".join("'" + _nrm(c) + "'" for c in _cancer_codes["code"])
add_flag("icd10_cancer_only", "hospital_diagnosis", f"""
    SELECT CAST(hospitalization_id AS VARCHAR) hospitalization_id, TRUE flag
    FROM read_parquet('{{p}}')
    WHERE CAST(hospitalization_id AS VARCHAR) IN (SELECT hospitalization_id FROM ids_df)
      AND {DXC} IN ({_cc}) GROUP BY 1""")

# Infectious screening, from microbiology_culture. This is culture positivity,
# NOT serostatus — serologies live in microbiology_nonculture, which RUSH lacks.
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

write_parquet(cohort, INTER / "cohort_with_variables.parquet")

# ── definition columns ───────────────────────────────────────────────────────
DEFS = [("CLIF-donor", "clif_eligible_donors"), ("CALC", "calc_flag"),
        ("Ventilated Patient", "ventilated_patient_no_age_limit")]


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
                       ("Sepsis (ICD-10)", "icd10_sepsis"),
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
hosp = (cohort.filter(pl.col("in_srtr_denominator") & pl.col("hospital_label").is_not_null())
        .group_by(["hospital_label", "srtr_ccn_id", "ccn_facility_type", "hospital_type"])
        .agg([pl.col("patient_id").n_unique().alias("n_decedents"),
              *[pl.col(c).fill_null(False).sum().alias(d.replace(" ", "_").replace("-", "_"))
                for d, c in DEFS]])
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

# ── CONSORT cascades, one per definition ─────────────────────────────────────
# The strobe_counts flags are population-wide, not sequential: organ_check_pass
# and no_positive_culture_48hrs are evaluated on every decedent, so reading them
# as successive steps makes the funnel go up. Applied cumulatively here instead,
# for each definition, so the report can show a CONSORT per definition.
_CALC_CAUSE_COL = {"any": "calc_cause_any", "primary": "calc_cause_primary",
                   "primary_poa": "calc_cause_primary_poa"}[
    str(CRIT["calc"].get("diagnosis_position", "any")).lower()]

cohort = cohort.with_columns(
    (~pl.col("icd10_contraindication").fill_null(False)).alias("no_icd10_contraindication"),
    # CLIF-donor excludes on the cancer and "other" arms only; sepsis is never
    # an exclusion (2026-09-19). Built in 01 as icd10_contraindication_clif.
    (~pl.col("icd10_contraindication_clif").fill_null(False)
     ).alias("no_icd10_contraindication_clif"),
    # CALC's cause flag is built in 01 in three diagnosis-position variants;
    # calc.diagnosis_position in config/donor_criteria.yaml picks which one is
    # the analysis. Falls back to the any-position flag for older cohorts.
    (pl.col(_CALC_CAUSE_COL) if _CALC_CAUSE_COL in cohort.columns
     else (pl.col("icd10_ischemic") | pl.col("icd10_cerebro") | pl.col("icd10_external"))
     ).fill_null(False).alias("calc_cause"),
)

# Order, per Emily Vail 2026-08-25: the criteria the definitions SHARE come
# first and in the same order (deaths -> ventilation -> age), so the cascades are
# readable side by side and the definition-specific steps are isolated at the
# end. Set intersection is commutative, so reordering changes no endpoint.
# Ventilated Patient IS the ventilation step: it is IMV within 48 h of death with
# no age limit, so step 1 of the clinical cascades equals that definition exactly
# (verified: 10,013 pooled, and no CLIF-donor or PD patient falls outside it).
# CALC has NO ventilation criterion, so its cascade legitimately lacks that step.
CASCADES = {
    "CLIF-donor": [
        ("All in-hospital deaths", None, None),
        (f"Received IMV within {_IMV_H} h of death\n(= Ventilated Patient)", "imv_48hr_expire",
         f"No IMV within {_IMV_H} h of death"),
        (f"Age {_AGE_IN} {_AGE} at death", "age_75_less", f"Age {_AGE_OUT} {_AGE} at death"),
        ("No cancer diagnosis", "no_icd10_contraindication_clif",
         "Cancer diagnosis"),
        (f"No positive blood culture within {_CX_H} h", "no_positive_culture_48hrs",
         f"Positive blood culture within {_CX_H} h"),
        ("Passed organ quality assessment\n(CLIF-donor eligible)", "organ_check_pass",
         "Failed kidney, liver, and BMI assessment"),
    ],
    # Kallan SAS seven-step cascade (Goldberg AJT 2017), reordered to share the
    # prefix above; the SAS step numbers are unchanged in DECISIONS.
    "CALC": [
        ("All in-hospital deaths", None, None),
        (f"Age {_AGE_IN} {_AGE} at death", "age_75_less", f"Age {_AGE_OUT} {_AGE} at death"),
        # CALC applies no contraindication codes (42 CFR 486.302; D-43), so the
        # cascade ends here.
        (f"Cause of death consistent with donation\n({_CAUSES})\n(CALC qualified)",
         "calc_cause", "No qualifying cause of death"),
    ],
    "Ventilated Patient": [
        ("All in-hospital deaths", None, None),
        (f"Received IMV within {_IMV_H} h of death\n(Ventilated Patient)", "imv_48hr_expire",
         f"No IMV within {_IMV_H} h of death"),
    ],
}

_flow = []
for _defn, _steps in CASCADES.items():
    _keep = pl.Series([True] * cohort.height)
    for _i, (_label, _col, _excl) in enumerate(_steps):
        if _col is not None:
            _keep = _keep & cohort[_col].fill_null(False)
        _flow.append({"site": SITE, "definition": _defn, "step": _i, "label": _label,
                      "n": cohort.filter(_keep)["patient_id"].n_unique(),
                      "excluded_label": _excl})
    print(f"CONSORT {_defn}: " + " -> ".join(
        f"{r['n']:,}" for r in _flow if r["definition"] == _defn))
pl.DataFrame(_flow).write_csv(FINAL / "consort_counts.csv")

# ── academic vs community ────────────────────────────────────────────────────
# The manuscript compares academic and community hospitals but nothing was ever
# computed for it. hospital_type comes from the terminal ADT record (CLIF field),
# not the crosswalk, which carries no such column. Counts only, so they pool.
_by_type = (cohort.filter(pl.col("in_srtr_denominator")
                          & pl.col("hospital_type").is_not_null())
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

# ── which codes actually caused each exclusion, step by step ─────────────────
# The CONSORT says how many were dropped at each step. This says WHY: for every
# ICD-driven step, the specific codes carried by the patients dropped there.
#
# Two kinds of criterion, and they need different reporting:
#   PRESENCE  the code disqualifies you (contraindications, MSOF, cancer).
#             Report each code and how many of the dropped patients carried it.
#   ABSENCE   you are dropped for NOT having a qualifying code (cause of death).
#             There is no code to name, so report the count only, alongside the
#             codes the SURVIVORS had, which is the useful diagnostic there.
_CODE_STEPS = {
    # Keyed on the flag the CLIF-donor cascade actually uses. It was keyed on
    # no_icd10_contraindication, which no cascade step carries, so the cancer
    # step was reported as "non-ICD" and no code was ever attributed.
    ("CLIF-donor", "no_icd10_contraindication_clif"):
        ("presence", "contraindication_clif", None),
    ("CALC", "calc_cause"):
        ("absence", "calc_cause", None),
}

_contra_codes = (pl.read_csv(REPO / _CD["contraindications"]["icd10_file"], infer_schema_length=0)
                 .rename({"ICD-10-CM": "code"}))
# CLIF-donor excludes on the cancer and "other" arms only, as in step 01. The
# sepsis arm is reporting only and must not be named as a reason for exclusion.
_clif_arm_codes = _contra_codes.filter(pl.col("dx_broad").is_in(["cancer", "other"]))
_norm = lambda s: sorted({str(x).upper().replace(".", "").strip() for x in s})


def _desc_map(df, code_col, desc_col) -> dict[str, str]:
    """Code -> description, built row-wise. Zipping a sorted set of codes against
    an unsorted description column silently mismatches every pair."""
    out = {}
    for c, d in zip(df[code_col], df[desc_col]):
        k = str(c).upper().replace(".", "").strip()
        out.setdefault(k, d)
    return out


_contra_desc = _desc_map(_contra_codes, "code", "description")

_SETS = {
    "contraindication_clif": (_norm(_clif_arm_codes["code"]), _contra_desc),
    "calc_cause": ([], {}),
}

con.register("dx_all", con.sql(f"""
    SELECT CAST(hospitalization_id AS VARCHAR) hid,
           UPPER(REPLACE(CAST(diagnosis_code AS VARCHAR), '.', '')) code
    FROM read_parquet('{TABLES}/clif_hospital_diagnosis.{FT}')""").df())

_excl = []
for _defn, _steps in CASCADES.items():
    _keep = pl.Series([True] * cohort.height)
    for _label, _col, _excl_label in _steps:
        if _col is None:
            continue
        _entering = _keep.clone()
        _keep = _keep & cohort[_col].fill_null(False)
        _dropped = _entering & ~_keep
        n_drop = cohort.filter(_dropped)["patient_id"].n_unique()
        n_in = cohort.filter(_entering)["patient_id"].n_unique()
        spec = _CODE_STEPS.get((_defn, _col))
        base = {"site": SITE, "definition": _defn, "step_label": _label.replace("\n", " "),
                "flag": _col, "n_entering_step": n_in, "n_excluded_at_step": n_drop}
        if spec is None:
            _excl.append({**base, "criterion_type": "non-ICD",
                          "code": None, "description": None,
                          "n_excluded_carrying_code": None, "n_excluded_only_this_code": None})
            continue
        kind, setname, _ = spec
        if kind == "absence" or not n_drop:
            _excl.append({**base, "criterion_type": f"{kind} (no code to attribute)",
                          "code": None, "description": None,
                          "n_excluded_carrying_code": None, "n_excluded_only_this_code": None})
            continue
        codes, desc = _SETS[setname]
        ids = (cohort.filter(_dropped).select("hospitalization_id")
               .with_columns(pl.col("hospitalization_id").cast(pl.Utf8)).drop_nulls().unique())
        pat = (cohort.filter(_dropped).select(["patient_id", "hospitalization_id"])
               .with_columns(pl.col("hospitalization_id").cast(pl.Utf8)))
        con.register("drop_ids", pat.to_pandas())
        lst = "','".join(codes)
        hit = con.sql(f"""
            WITH h AS (SELECT DISTINCT d.patient_id, x.code
                       FROM drop_ids d JOIN dx_all x ON x.hid = d.hospitalization_id
                       WHERE x.code IN ('{lst}'))
            SELECT code, count(DISTINCT patient_id) n,
                   count(DISTINCT CASE WHEN NOT EXISTS
                        (SELECT 1 FROM h y WHERE y.patient_id = h.patient_id AND y.code <> h.code)
                        THEN patient_id END) n_sole
            FROM h GROUP BY 1 ORDER BY n DESC, code""").pl()   # code breaks ties: stable across runs
        for r in hit.iter_rows(named=True):
            _excl.append({**base, "criterion_type": "presence",
                          "code": r["code"], "description": desc.get(r["code"], ""),
                          "n_excluded_carrying_code": r["n"],
                          "n_excluded_only_this_code": r["n_sole"]})
        if not hit.height:
            _excl.append({**base, "criterion_type": "presence",
                          "code": None, "description": "no code matched the dropped patients",
                          "n_excluded_carrying_code": 0, "n_excluded_only_this_code": 0})

pl.DataFrame(_excl).write_csv(FINAL / "exclusion_codes_by_step.csv")
print(f"Exclusion attribution: {len(_excl)} rows -> exclusion_codes_by_step.csv")
