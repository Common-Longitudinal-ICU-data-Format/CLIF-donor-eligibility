#!/usr/bin/env python
"""Data-quality diagnostics: can the numbers from 01 and 02 be trusted?

Four questions this answers:

  1. For each definition, which of ITS OWN required fields are missing, and how
     often? A definition can only be as complete as the fields it depends on, so
     missingness is reported against each definition's own requirement list
     rather than as one cohort-wide table.

  2. How is missingness distributed across PATIENTS? A field that is 8% missing
     could be 8% of patients missing one value, or 2% of patients missing four.
     Those imply very different things about whether the cohort is comparable.

  3. Which patients do the definitions actually share? An UpSet plot over the
     membership combinations, because the discordant cells are the interesting
     ones and a Venn diagram cannot show them legibly.

  4. Do this run's own artifacts agree? The STROBE counts, the CONSORT, the
     definition counts and the Table 2 denominators all count the same patients.
     Reconciliation compares them and stops the run if they disagree.

MEASUREMENT vs EVENT. A lab value can be missing. An event flag derived from a
timestamped table (was the patient on IMV, was a culture positive) cannot be —
absence of a record means the event did not happen, not that it is unknown.
Conflating the two overstates missingness, so only measurement fields are
counted and event-derived fields are listed separately as "not applicable".

    uv run python code/03_diagnostics.py       # reads config/config.json
"""
from __future__ import annotations

import sys

# Print Unicode (≤, →) on any console, Windows included
sys.stdout.reconfigure(encoding="utf-8")
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          
import polars as pl                    

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from utils.audit import StageAudit       
from utils.config import config
from utils.criteria import DEFS

SITE = config["site_name"]
INTER = Path(config["output_intermediate"])
FINAL = Path(config["output_final"])
FINAL.mkdir(parents=True, exist_ok=True)
audit = StageAudit(SITE, FINAL, label="missingness")

cohort = pl.read_parquet(INTER / "cohort_with_variables.parquet")
N = cohort.height
print(f"[{SITE}] {N:,} decedents")

# ── what each definition actually requires ───────────────────────────────────
# 'measured' fields can be null. 'event' fields are derived from timestamped
# tables where absence means the event did not occur.
REQUIREMENTS = {
    "CLIF-donor": dict(
        measured=["age_at_death", "last_weight_kg", "last_height_cm", "bmi",
                  "creatinine_value", "bilirubin_total_value", "ast_value", "alt_value"],
        event=["imv_48hr_expire", "on_crrt_48h_before_death",
               "no_positive_culture_48hrs", "icd10_contraindication"]),
    "CALC": dict(
        measured=["age_at_death"],
        event=["icd10_ischemic", "icd10_cerebro", "icd10_external", "icd10_contraindication"]),
    "Ventilated Patient": dict(measured=[], event=["imv_48hr_expire"]),
}
DEFCOL = dict(DEFS)

# Reporting fields that are not definition inputs but appear in Table 2.
REPORTING = ["gcs_total_value", "rass_value", "first_icu_los_days", "pf_ratio",
             "race_category", "ethnicity_category", "sex_category"]

# ── 1. missingness by definition ─────────────────────────────────────────────
rows = []
for dname, req in REQUIREMENTS.items():
    elig = cohort.filter(pl.col(DEFCOL[dname]).fill_null(False))
    for f in req["measured"]:
        if f not in cohort.columns:
            continue
        rows.append({
            "definition": dname, "field": f, "field_role": "measured",
            "n_missing_all_decedents": int(cohort[f].null_count()),
            "pct_missing_all_decedents": round(100 * cohort[f].null_count() / N, 2),
            "n_missing_among_eligible": int(elig[f].null_count()),
            "pct_missing_among_eligible": round(100 * elig[f].null_count() / elig.height, 2) if elig.height else None,
        })
    for f in req["event"]:
        if f not in cohort.columns:
            continue
        rows.append({"definition": dname, "field": f, "field_role": "event",
                     "n_missing_all_decedents": 0, "pct_missing_all_decedents": 0.0,
                     "n_missing_among_eligible": 0, "pct_missing_among_eligible": 0.0})
for f in REPORTING:
    if f in cohort.columns:
        rows.append({"definition": "(reporting only)", "field": f, "field_role": "measured",
                     "n_missing_all_decedents": int(cohort[f].null_count()),
                     "pct_missing_all_decedents": round(100 * cohort[f].null_count() / N, 2),
                     "n_missing_among_eligible": None, "pct_missing_among_eligible": None})
by_def = pl.DataFrame(rows)
by_def.write_csv(FINAL / "missingness_by_definition.csv")
print(f"\n1. missingness by definition -> {by_def.height} field-rows")
print(by_def.filter(pl.col("field_role") == "measured")
      .select(["definition", "field", "pct_missing_all_decedents", "pct_missing_among_eligible"])
      .to_pandas().to_string(index=False))

# ── 2. patient-level missingness ─────────────────────────────────────────────
# CLIF-donor has the largest requirement set, so completeness is scored against
# it: a patient missing any of these cannot be assessed for eligibility.
CLIF_MEAS = [f for f in REQUIREMENTS["CLIF-donor"]["measured"] if f in cohort.columns]
pat = cohort.with_columns(
    sum((pl.col(f).is_null().cast(pl.Int32) for f in CLIF_MEAS), pl.lit(0)).alias("n_missing"))
dist = (pat.group_by("n_missing").agg(pl.len().alias("n_patients"))
        .sort("n_missing")
        .with_columns((100 * pl.col("n_patients") / N).round(2).alias("pct_patients")))
dist.write_csv(FINAL / "missingness_patient_level.csv")
complete = int(dist.filter(pl.col("n_missing") == 0)["n_patients"].sum())
print(f"\n2. patient-level: {complete:,}/{N:,} ({100*complete/N:.1f}%) complete on all "
      f"{len(CLIF_MEAS)} CLIF-donor measurement fields")
print(dist.to_pandas().to_string(index=False))

# which fields co-occur as missing
patterns = (pat.filter(pl.col("n_missing") > 0)
            .with_columns(pl.concat_str(
                [pl.when(pl.col(f).is_null()).then(pl.lit(f)).otherwise(pl.lit(""))
                 for f in CLIF_MEAS], separator="|").alias("pattern"))
            .group_by("pattern").agg(pl.len().alias("n_patients"))
            .sort(["n_patients", "pattern"], descending=[True, False]).head(15)   # pattern breaks ties
            .with_columns(pl.col("pattern").str.replace_all(r"\|+", " + ").str.strip_chars(" +")))
patterns.write_csv(FINAL / "missingness_patterns.csv")

fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=150)
ax.bar(dist["n_missing"], dist["n_patients"], color="#0e6b61", width=.72)
for x, y in zip(dist["n_missing"], dist["n_patients"]):
    ax.text(x, y, f"{y:,}\n{100*y/N:.1f}%", ha="center", va="bottom", fontsize=7.5, color="#16262a")
ax.set_xlabel(f"number of the {len(CLIF_MEAS)} CLIF-donor measurement fields missing")
ax.set_ylabel("decedents")
ax.set_title(f"{SITE.upper()} — patient-level missingness", loc="left", fontsize=11, weight="bold")
ax.set_xticks(list(dist["n_missing"]))
ax.set_ylim(0, dist["n_patients"].max() * 1.22)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(FINAL / "missingness_patient_level.png"); plt.close(fig)

# ── 2b. missingness by hospital (Table S2 heatmap) ───────────────────────────
# Fields on one axis, hospitals on the other. Emitted as aggregate counts so the
# heatmap can be assembled from site CSVs without patient-level data.
HEAT_FIELDS = CLIF_MEAS + [f for f in REPORTING if f in cohort.columns]
hrows = []
if "hospital_label" in cohort.columns:
    for hid, g in cohort.group_by("hospital_label"):
        hid = hid[0] if isinstance(hid, tuple) else hid
        if hid is None:
            continue
        for f in HEAT_FIELDS:
            hrows.append({"site": SITE, "hospital_label": hid, "field": f,
                          "n_total": g.height, "n_missing": int(g[f].null_count()),
                          "pct_missing": round(100 * g[f].null_count() / g.height, 2)})
pl.DataFrame(hrows).sort(["hospital_label", "field"]).write_csv(FINAL / "missingness_by_hospital.csv")
print(f"\n2b. missingness by hospital -> {len(hrows)} field-hospital rows")

# ── 2c. element coverage: absent from the data, or absent from the cohort? ───
# A zero in Table 2 or Table 3 has two very different causes. Either the site
# never records that concept at all (a vocabulary gap — the row should be
# footnoted, not reported as 0%), or the site records it and no decedent had it
# (a real clinical finding). Those are indistinguishable in the tables, so each
# concept is probed site-wide and in the cohort (element_coverage.csv), and, for
# the missingness heatmap, as the share of cohort decedents with a record at
# each hospital (element_coverage_by_hospital.csv): every lab, vital, assessment,
# IMV and CRRT record, blood cultures, each diagnosis arm, each medication
# category, each procedure concept and prone positioning.
import duckdb  # noqa: E402
import re as _re                                                     # noqa: E402
import yaml as _yaml                                                 # noqa: E402
from utils.checks import usable_number_sql                           # noqa: E402
from utils.criteria import contraindication_sql, icd_range_sql, load_contraindications   # noqa: E402

TABLES = Path(config["tables_path"])
FT = config.get("file_type", "parquet")
con = duckdb.connect()
_CRIT = _yaml.safe_load((REPO / "config/donor_criteria.yaml").read_text())
_DX_NORM = "lower(regexp_replace(CAST(diagnosis_code AS VARCHAR),'[^0-9A-Za-z]','','g'))"
_DX_ICD10 = "regexp_replace(lower(CAST(diagnosis_code_format AS VARCHAR)),'[^a-z0-9]','','g') IN ('icd10','icd10cm')"
elem_rows: list[dict] = []
elem_hosp_rows: list[dict] = []
_hosp_n: dict[str, int] = {}
if "hospital_label" in cohort.columns and "hospitalization_id" in cohort.columns:
    # Every stay in each decedent's encounter block, with the decedent's hospital.
    ids = (pl.read_parquet(INTER / "encounter_block_members.parquet")
           .with_columns(pl.col("hospitalization_id").cast(pl.Utf8))
           .join(cohort.select(["patient_id", "hospital_label"]), on="patient_id", how="inner")
           .to_pandas())
    con.register("ids_df", ids)
    _hosp_n = dict(cohort.drop_nulls("hospital_label").group_by("hospital_label").len().iter_rows())


def _probe(concept: str, group: str, table: str, col: str | None = None,
           values: list[str] | None = None, regexes: list[str] | None = None,
           extra: str = "", cond: str | None = None) -> None:
    """Count hospitalizations matching a concept site-wide and in the cohort, and
    cohort decedents with one at each hospital. The concept is `col` in `values`
    (case and spaces ignored), `col` matching a regex once punctuation is stripped,
    either, or a `cond` written out; `extra` narrows any of them."""
    f = TABLES / f"clif_{table}.{FT}"
    if cond is None:
        parts = []
        if values:
            vals = ", ".join("'" + str(v).lower().replace("'", "''") + "'" for v in values)
            parts.append(f"lower(trim(CAST({col} AS VARCHAR))) IN ({vals})")
        if regexes:
            parts += ["regexp_matches(lower(REGEXP_REPLACE(CAST("
                      f"{col} AS VARCHAR), '[^0-9A-Za-z]', '', 'g')), "
                      "'" + r.lower().replace("'", "''") + "')" for r in regexes]
        cond = " OR ".join(parts)
    if extra:
        cond = f"({cond}) AND ({extra})"
    q, status = None, "table absent"
    if f.exists():
        try:
            q = con.sql(f"""
                SELECT i.hospital_label, count(DISTINCT CAST(t.hospitalization_id AS VARCHAR)) AS n_hosp,
                       count(DISTINCT i.patient_id) AS n_pat
                FROM read_parquet('{f}') t
                LEFT JOIN ids_df i ON CAST(t.hospitalization_id AS VARCHAR) = i.hospitalization_id
                WHERE {cond} GROUP BY 1""").df()
        except duckdb.BinderException:          # a column this extract does not carry
            status = "column absent"
    sw = int(q.n_hosp.sum()) if q is not None else 0
    ic = int(q.n_hosp[q.hospital_label.notna()].sum()) if q is not None else 0
    by = {r.hospital_label: int(r.n_pat) for r in q.itertuples() if r.hospital_label} if q is not None else {}
    if q is not None:
        status = ("not recorded at this site" if sw == 0 else
                  "recorded, absent in decedents" if ic == 0 else "present")
    elem_rows.append({"site": SITE, "group": group, "concept": concept, "source_table": table,
                      "status": status, "n_hosp_site_wide": sw, "n_hosp_in_cohort": ic})
    elem_hosp_rows.extend({"site": SITE, "hospital_label": h, "group": group, "concept": concept,
                           "source_table": table, "n_decedents": n, "n_with_record": by.get(h, 0),
                           "pct_with_record": round(100 * by.get(h, 0) / n, 1)} for h, n in _hosp_n.items())


if "hospitalization_id" in cohort.columns:
    # neuro procedures, by concept, across both vocabularies (exact codes and
    # anchored PCS patterns are OR-ed)
    _np = pl.read_csv(REPO / _CRIT["clinical_care"]["procedures_file"], comment_prefix="#",
                      infer_schema_length=0)
    for concept in sorted(set(_np["concept"])):
        rows = _np.filter(pl.col("concept") == concept)
        _probe(concept, "neurologic procedure", "patient_procedures", "procedure_code",
               values=[str(v) for v in rows.filter(pl.col("match_type") == "exact")["pattern"]] or None,
               regexes=[str(v) for v in rows.filter(pl.col("match_type") != "exact")["pattern"]] or None)

    # medications, per mCIDE category, in whichever table carries them
    for grp, meds in _CRIT["clinical_care"]["medications"].items():
        for route, tb in [("continuous", "medication_admin_continuous"),
                          ("intermittent", "medication_admin_intermittent")]:
            _probe(f"{grp} [{route}]", "medication", tb, "med_category", values=meds)

    # cultures: fungemia and the contraindication organisms, and blood cultures at all
    _fg = _CRIT["clinical_care"]["fungemia"]
    _probe("fungemia organisms", "contraindication", "microbiology_culture", "organism_category",
           regexes=[_fg["organism_regex"]])
    for org in _CRIT["clinical_care"]["contraindications_not_in_clif"]:
        _probe(org, "contraindication", "microbiology_culture", "organism_category",
               regexes=[org.split("_")[0]])
    _probe("blood culture obtained", "culture", "microbiology_culture", "fluid_category", values=["blood_buffy"])

    # the diagnosis arms every definition reads
    for name, (lo, hi) in _CRIT["calc"]["cause_icd10_prefixes"].items():
        _probe(f"{name.replace('_', ' ')} ({lo}-{hi})", "diagnosis", "hospital_diagnosis",
               cond=icd_range_sql(_DX_NORM, lo, hi), extra=_DX_ICD10)
    _CONTRA = load_contraindications(REPO / _CRIT["clif_donor"]["contraindications"]["icd10_file"])
    _probe("contraindicating cancer code", "diagnosis", "hospital_diagnosis",
           cond=contraindication_sql(_DX_NORM, _CONTRA), extra=_DX_ICD10)
    _probe("any ICD-10 code", "diagnosis", "hospital_diagnosis", cond="diagnosis_code IS NOT NULL", extra=_DX_ICD10)

    # events and measurements the criteria and Table 2 depend on
    _probe("invasive mechanical ventilation", "respiratory support", "respiratory_support", "device_category",
           values=["imv"])
    _probe("fio2_set", "respiratory support", "respiratory_support", cond=usable_number_sql("fio2_set"))
    _probe("any CRRT record", "renal replacement", "crrt_therapy", cond="TRUE")
    _probe("prone", "positioning", "position", "position_category", values=["prone"])
    for v in ["weight_kg", "height_cm"]:
        _probe(v, "vitals", "vitals", "vital_category", values=[v])
    for lab in ["creatinine", "bilirubin_total", "ast", "alt", "bun", "sodium", "po2_arterial",
                "platelet_count", "lactate"]:
        _probe(lab, "laboratory", "labs", "lab_category", values=[lab])
    for asm in ["gcs_total", "RASS"]:
        _probe(asm, "assessment", "patient_assessments", "assessment_category", values=[asm])

_ec = pl.DataFrame(elem_rows)
if _ec.height:
    _ec = _ec.sort(["group", "concept"])
    _ec.write_csv(FINAL / "element_coverage.csv")
    _bad = _ec.filter(pl.col("status") != "present")
    print(f"\n2c. element coverage -> {_ec.height} concepts; "
          f"{_bad.height} not present ("
          f"{_ec.filter(pl.col('status') == 'not recorded at this site').height} vocabulary gaps)")
    for r in _bad.iter_rows(named=True):
        print(f"     {r['status']:30s} {r['group']}: {r['concept']}")
pl.DataFrame(elem_hosp_rows, schema={"site": pl.Utf8, "hospital_label": pl.Utf8, "group": pl.Utf8, "concept": pl.Utf8,
                                     "source_table": pl.Utf8, "n_decedents": pl.Int64, "n_with_record": pl.Int64,
                                     "pct_with_record": pl.Float64}
             ).sort(["hospital_label", "group", "concept"]).write_csv(FINAL / "element_coverage_by_hospital.csv")
print(f"    by hospital -> {len(elem_hosp_rows)} concept-hospital rows")

# ── 3. UpSet over the definitions ────────────────────────────────────────────
NAMES = list(DEFCOL)
flags = {d: cohort[DEFCOL[d]].fill_null(False) for d in NAMES}
combos = []
for k in range(1, len(NAMES) + 1):
    for members in combinations(NAMES, k):
        m = pl.Series([True] * N)
        for d in NAMES:
            m = m & (flags[d] if d in members else ~flags[d])
        n = int(m.sum())
        if n:
            combos.append({"members": " ∩ ".join(members), "n_definitions": k,
                           "n_patients": n, "pct_of_decedents": round(100 * n / N, 2),
                           **{d: (d in members) for d in NAMES}})
combos.sort(key=lambda r: -r["n_patients"])
upset = pl.DataFrame(combos)
upset.write_csv(FINAL / "definition_overlap_upset.csv")
none_n = N - int(upset["n_patients"].sum())
print(f"\n3. UpSet: {upset.height} non-empty intersections; {none_n:,} decedents meet no definition")
print(upset.select(["members", "n_patients", "pct_of_decedents"]).head(10).to_pandas().to_string(index=False))

# classic UpSet: intersection bars on top, membership matrix below, set sizes left
top = upset.head(14)
nrows, ncols = len(NAMES), top.height
fig = plt.figure(figsize=(max(7.5, ncols * .62), 5.6), dpi=150)
gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 5], height_ratios=[2.5, 1.35],
                      wspace=.06, hspace=.06)
axb = fig.add_subplot(gs[0, 1])
axm = fig.add_subplot(gs[1, 1], sharex=axb)
# NOT sharey: clearing ticks on the set-size axis would clear the matrix labels too.
axs = fig.add_subplot(gs[1, 0])

x = range(ncols)
axb.bar(x, top["n_patients"], color="#0e6b61", width=.66)
for i, v in enumerate(top["n_patients"]):
    axb.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=7.5, color="#16262a")
axb.set_ylabel("patients in intersection")
axb.set_ylim(0, top["n_patients"].max() * 1.18)
for s in ("top", "right", "bottom"):
    axb.spines[s].set_visible(False)
axb.tick_params(bottom=False, labelbottom=False)
axb.set_title(f"{SITE.upper()} — patients shared between definitions   (n={N:,} decedents)",
              loc="left", fontsize=11, weight="bold", pad=10)

for r, d in enumerate(NAMES):
    y = nrows - 1 - r
    axm.axhline(y, color="#eef2f0", lw=11, zorder=0)
    on = [i for i in range(ncols) if top[d][i]]
    axm.scatter(list(range(ncols)), [y] * ncols, s=52, color="#d5ddda", zorder=1)
    if on:
        axm.scatter(on, [y] * len(on), s=52, color="#16262a", zorder=2)
for i in range(ncols):
    ys = [nrows - 1 - r for r, d in enumerate(NAMES) if top[d][i]]
    if len(ys) > 1:
        axm.plot([i, i], [min(ys), max(ys)], color="#16262a", lw=1.7, zorder=1)
# Names go on the RIGHT of the matrix: the set-size bars occupy the left gutter,
# so left-hand tick labels would be drawn underneath them and disappear.
axm.set_yticks(range(nrows)); axm.set_yticklabels(NAMES[::-1], fontsize=9.5)
axm.yaxis.tick_right(); axm.yaxis.set_label_position("right")
axm.set_xticks([]); axm.set_ylim(-.6, nrows - .4)
for s in axm.spines.values():
    s.set_visible(False)
axm.tick_params(left=False, right=False, pad=8)

sizes = [int(flags[d].sum()) for d in NAMES][::-1]
axs.barh(range(nrows), sizes, color="#8fb3ac", height=.5)
for i, v in enumerate(sizes):
    axs.text(v, i, f"{v:,} ", va="center", ha="right", fontsize=8, color="#16262a")
axs.set_ylim(-.6, nrows - .4)
axs.invert_xaxis()
axs.set_xlabel("total in definition", fontsize=8.5, labelpad=2)
axs.set_yticks([]); axs.set_xticks([])
for s in axs.spines.values():
    s.set_visible(False)
fig.savefig(FINAL / "definition_overlap_upset.png", bbox_inches="tight"); plt.close(fig)

audit.record("50_missingness_upset", cohort, cohort, key="patient_id",
             rule="missingness by definition, patient-level, and definition overlap",
             complete_cases=complete, intersections=upset.height, meets_no_definition=none_n)
audit.write()



# ── 2e. donor administrative codes (Emily Vail, 2026-08-26) ──────────────────
# Face-validity check: do administrative markers of an actual donation, or of
# brain death, appear at all? Reported three ways per code, because a zero has
# three different meanings:
#   n_hosp_site_wide   the code exists in this site's data at all
#   n_decedents        it appears on a cohort decedent
#   n_clif_donor       it appears on someone CLIF-donor already identifies
# Codes and vocabulary labels are normalised on BOTH sides (upper-case, strip
# non-alphanumerics) because the sites spell the vocabulary three different ways.
_dac_path = REPO / _CRIT["reporting"]["donor_administrative_codes_file"]
if _dac_path.exists() and "hospitalization_id" in cohort.columns:
    _dac = pl.read_csv(_dac_path, comment_prefix="#", infer_schema_length=0)
    _nrm = lambda s: _re.sub(r"[^0-9A-Za-z]", "", str(s)).upper()

    con.register("coh_dac", pl.read_parquet(INTER / "encounter_block_members.parquet")
        .with_columns(pl.col("hospitalization_id").cast(pl.Utf8))
        .join(cohort.select(["patient_id", "clif_eligible_donors"]), on="patient_id", how="inner")
        .to_pandas())

    # Normalised views of both source tables, vocabulary label included so a hit
    # can be attributed to the vocabulary it came from.
    _srcs = {
        "icd10cm": (f"clif_hospital_diagnosis.{FT}", "diagnosis_code", "diagnosis_code_format"),
        "cpt": (f"clif_patient_procedures.{FT}", "procedure_code", "procedure_code_format"),
    }
    dac_rows = []
    for _vocab, (_file, _code_col, _fmt_col) in _srcs.items():
        f = TABLES / _file
        if not f.exists():
            continue
        con.execute(f"""
            CREATE OR REPLACE TEMP VIEW src AS
            SELECT CAST(hospitalization_id AS VARCHAR) hid,
                   upper(regexp_replace(CAST({_code_col} AS VARCHAR), '[^0-9A-Za-z]', '', 'g')) code,
                   upper(regexp_replace(COALESCE(CAST({_fmt_col} AS VARCHAR), ''),
                         '[^0-9A-Za-z]', '', 'g')) fmt
            FROM read_parquet('{f}')""")
        for r in _dac.filter(pl.col("vocabulary") == _vocab).iter_rows(named=True):
            pat = _nrm(r["code"])
            cond = (f"code = '{pat}'" if r["match_type"] == "exact"
                    else f"code LIKE '{pat}%'")
            q = con.sql(f"""
                SELECT count(DISTINCT s.hid) AS site_wide,
                       count(DISTINCT c.patient_id) AS n_dec,
                       count(DISTINCT CASE WHEN c.clif_eligible_donors
                             THEN c.patient_id END) AS n_clif,
                       string_agg(DISTINCT s.fmt, '|') AS formats
                FROM src s LEFT JOIN coh_dac c ON c.hospitalization_id = s.hid
                WHERE {cond}""").df().iloc[0]
            dac_rows.append({
                "site": SITE, "code": r["code"], "vocabulary": _vocab,
                "match_type": r["match_type"], "concept": r["concept"],
                "organ_or_service": r["organ_or_service"],
                "description": r["description"],
                "n_hospitalizations_site_wide": int(q.site_wide),
                "n_cohort_decedents": int(q.n_dec),
                "n_among_clif_donors": int(q.n_clif),
                "vocabulary_labels_seen": q.formats or "",
                "status": ("not present at this site" if not q.site_wide else
                           "present at site, absent in decedents" if not q.n_dec
                           else "present in decedents"),
            })
    if dac_rows:
        _dacdf = pl.DataFrame(dac_rows).with_columns(
            (100 * pl.col("n_cohort_decedents") / N).round(2).alias("pct_of_decedents"))
        _dacdf.write_csv(FINAL / "donor_administrative_code_availability.csv")
        print(f"\n2e. donor administrative codes -> {_dacdf.height} codes")
        for r in _dacdf.iter_rows(named=True):
            print(f"     {r['code']:8s} {r['status']:38s} "
                  f"site-wide={r['n_hospitalizations_site_wide']:6,} "
                  f"decedents={r['n_cohort_decedents']:5,} "
                  f"clif-donors={r['n_among_clif_donors']:4,}")

# ── Figure 1 for this site: the cohort flow, from this run's own counts ──────
# The counts behind it ship in figure1_counts.csv, so the coordinating centre
# pools the same figure over sites without patient-level data (utils/figure1.py).
from utils import figure1                                 # noqa: E402
_strobe = dict(pl.read_csv(FINAL / "strobe_counts.csv").select("metric", "n").iter_rows())
_dc = pl.read_csv(FINAL / "definition_counts.csv").row(0, named=True)
_organ = {k: int(v) for k, v in pl.read_csv(FINAL / "organ_criteria.csv")
          .filter(pl.col("population") == figure1.ORGAN_POPULATION).row(0, named=True).items()
          if k in figure1.ORGAN}
_flow = figure1.site_flow(cohort, _strobe, _dc)
_f1_problems = figure1.all_problems(_flow, _organ)
if _f1_problems:
    raise SystemExit("Figure 1 not drawn; this run's counts disagree:\n  " + "\n  ".join(_f1_problems))
(pl.DataFrame({"metric": list(_flow), "n": list(_flow.values())})
   .with_columns(pl.lit(SITE).alias("site")).select("site", "metric", "n")
   .write_csv(FINAL / "figure1_counts.csv"))
_f1 = figure1.render(_flow, _organ, figure1.excluded_text(pl.read_csv(FINAL / "hospitals.csv")),
                     1, FINAL / "figure1_cohort")
print(f"\nFigure 1 -> {_f1[0].name}; counts -> figure1_counts.csv")

# ── provenance: one document, and a check that the artifacts agree ───────────
# Last site step, so every artifact exists by now. A mismatch between the
# STROBE counts, the CONSORT, definition_counts and the Table 2 denominators
# means one of them is wrong, and a wrong number that only appears in one place
# is exactly what nobody notices.
from utils.provenance import reconcile, render          
_problems = reconcile(FINAL)
_prov = render(FINAL, _problems)
print(f"\nwrote {_prov}")
if _problems:
    print("\nRECONCILIATION FAILED:")
    for _p in _problems:
        print(f"  - {_p}")
    raise SystemExit("The run's own counts disagree. See provenance.md.")
print("Reconciliation passed: strobe, Figure 1, definition_counts, Table 2 and the hospital counts agree")
print(f"\nwrote -> {FINAL}")
