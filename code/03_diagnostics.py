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

    CLIF_DONOR_SITE=ucmc python code/03_diagnostics.py
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import polars as pl                      # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from utils.audit import StageAudit        # noqa: E402
from utils.config import config           # noqa: E402

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
DEFCOL = {"CLIF-donor": "clif_eligible_donors", "CALC": "calc_flag",
          "Ventilated Patient": "ventilated_patient_no_age_limit"}

# Reporting fields that are not definition inputs but appear in Table 2.
REPORTING = ["gcs_total_value", "rass_value", "first_icu_los_days",
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
            .sort("n_patients", descending=True).head(15)
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
pl.DataFrame(hrows).write_csv(FINAL / "missingness_by_hospital.csv")
print(f"\n2b. missingness by hospital -> {len(hrows)} field-hospital rows")

# ── 2c. data availability, by hospital ───────────────────────────────────────
# How much of each source actually exists for these decedents, so a low
# eligibility count can be told apart from a thin extract. Diagnosis coverage is
# reported alongside the measurement fields because the CALC definition rests
# entirely on ICD codes: a hospital with sparse coding cannot
# qualify anyone, and that is a data property, not a clinical one.
import duckdb  # noqa: E402

TABLES = Path(config["tables_path"])
FT = config.get("file_type", "parquet")
con = duckdb.connect()
avail_rows = []
if "hospital_label" in cohort.columns and "hospitalization_id" in cohort.columns:
    ids = cohort.select(["hospitalization_id", "hospital_label"]).to_pandas()
    ids["hospitalization_id"] = ids["hospitalization_id"].astype(str)
    con.register("ids_df", ids)

    def coverage(table: str, label: str, extra_select: str = "") -> None:
        f = TABLES / f"clif_{table}.{FT}"
        if not f.exists():
            for h in sorted(set(ids.hospital_label.dropna())):
                avail_rows.append({"site": SITE, "hospital_label": h, "source": label,
                                   "n_decedents": None, "n_with_data": 0, "pct_with_data": 0.0,
                                   "mean_rows_per_decedent": None})
            return
        q = con.sql(f"""
            SELECT i.hospital_label,
                   count(DISTINCT i.hospitalization_id) AS n_decedents,
                   count(DISTINCT CASE WHEN t.hospitalization_id IS NOT NULL
                                       THEN i.hospitalization_id END) AS n_with_data,
                   COALESCE(count(t.hospitalization_id),0) AS n_rows
                   {extra_select}
            FROM ids_df i
            LEFT JOIN read_parquet('{f}') t
              ON CAST(t.hospitalization_id AS VARCHAR) = i.hospitalization_id
            GROUP BY 1""").df()
        for r in q.itertuples():
            if not r.hospital_label:
                continue
            avail_rows.append({"site": SITE, "hospital_label": r.hospital_label, "source": label,
                               "n_decedents": int(r.n_decedents),
                               "n_with_data": int(r.n_with_data),
                               "pct_with_data": round(100 * r.n_with_data / r.n_decedents, 1)
                               if r.n_decedents else 0.0,
                               "mean_rows_per_decedent": round(r.n_rows / r.n_decedents, 1)
                               if r.n_decedents else None})

    for _tbl, _lab in [("hospital_diagnosis", "ICD diagnosis codes"),
                       ("labs", "laboratory results"),
                       ("respiratory_support", "respiratory support"),
                       ("microbiology_culture", "cultures"),
                       ("patient_procedures", "procedures"),
                       ("medication_admin_intermittent", "medications (intermittent)"),
                       ("medication_admin_continuous", "medications (continuous)"),
                       ("patient_assessments", "GCS / RASS assessments"),
                       ("vitals", "vitals"),
                       ("position", "position"),
                       ("crrt_therapy", "CRRT")]:
        coverage(_tbl, _lab)

pl.DataFrame(avail_rows).write_csv(FINAL / "data_availability_by_hospital.csv")
print(f"\n2c. data availability -> {len(avail_rows)} source-hospital rows")

# ── 2d. element coverage: absent from the data, or absent from the cohort? ───
# A zero in Table 2 or Table 3 has two very different causes. Either the site
# never records that concept at all (a vocabulary gap — the row should be
# footnoted, not reported as 0%), or the site records it and no decedent had it
# (a real clinical finding). Those are indistinguishable in the tables, so each
# concept is probed twice: site-wide, and restricted to cohort decedents.
import re as _re                                                     # noqa: E402
import yaml as _yaml                                                 # noqa: E402

_CRIT = _yaml.safe_load((REPO / "config/donor_criteria.yaml").read_text())
elem_rows = []


def _probe(concept: str, group: str, table: str, col: str,
           values: list[str] | None = None, regexes: list[str] | None = None,
           extra: str = "") -> None:
    """Count hospitalizations matching a concept site-wide and in the cohort."""
    f = TABLES / f"clif_{table}.{FT}"
    if not f.exists():
        elem_rows.append({"site": SITE, "group": group, "concept": concept,
                          "source_table": table, "status": "table absent",
                          "n_hosp_site_wide": 0, "n_hosp_in_cohort": 0})
        return
    if values:
        vals = ", ".join("'" + str(v).lower().replace("'", "''") + "'" for v in values)
        cond = f"lower(trim(CAST({col} AS VARCHAR))) IN ({vals})"
    else:
        pats = " OR ".join("regexp_matches(lower(REGEXP_REPLACE(CAST("
                           f"{col} AS VARCHAR), '[^0-9A-Za-z]', '', 'g')), "
                           "'" + r.lower().replace("'", "''") + "')" for r in (regexes or []))
        cond = f"({pats})"
    if extra:
        cond = f"({cond}) AND ({extra})"
    q = con.sql(f"""
        SELECT count(DISTINCT CAST(t.hospitalization_id AS VARCHAR)) AS site_wide,
               count(DISTINCT CASE WHEN i.hospitalization_id IS NOT NULL
                                   THEN CAST(t.hospitalization_id AS VARCHAR) END) AS in_cohort
        FROM read_parquet('{f}') t
        LEFT JOIN ids_df i ON CAST(t.hospitalization_id AS VARCHAR) = i.hospitalization_id
        WHERE {cond}""").df().iloc[0]
    sw, ic = int(q.site_wide), int(q.in_cohort)
    elem_rows.append({
        "site": SITE, "group": group, "concept": concept, "source_table": table,
        "status": ("not recorded at this site" if sw == 0 else
                   "recorded, absent in decedents" if ic == 0 else "present"),
        "n_hosp_site_wide": sw, "n_hosp_in_cohort": ic})


if "hospitalization_id" in cohort.columns:
    # neuro procedures, by concept, across both vocabularies
    _np = pl.read_csv(REPO / "utils/codes/neuro_procedures.csv", comment_prefix="#")
    for concept in sorted(set(_np["concept"])):
        rows = _np.filter(pl.col("concept") == concept)
        ex = [str(v) for v in rows.filter(pl.col("match_type") == "exact")["pattern"]]
        rx = [str(v) for v in rows.filter(pl.col("match_type") != "exact")["pattern"]]
        if not (ex and rx):
            _probe(concept, "neurologic procedure", "patient_procedures",
                   "procedure_code", values=ex or None, regexes=rx or None)
            continue
        # exact codes and anchored PCS patterns are OR-ed, so probe together
        vals = ", ".join("'" + v.lower() + "'" for v in ex)
        pats = " OR ".join("regexp_matches(lower(REGEXP_REPLACE(CAST(procedure_code AS "
                           "VARCHAR), '[^0-9A-Za-z]', '', 'g')), '" + r.lower() + "')"
                           for r in rx)
        f = TABLES / f"clif_patient_procedures.{FT}"
        if not f.exists():
            continue
        q = con.sql(f"""
            SELECT count(DISTINCT CAST(t.hospitalization_id AS VARCHAR)) AS site_wide,
                   count(DISTINCT CASE WHEN i.hospitalization_id IS NOT NULL
                         THEN CAST(t.hospitalization_id AS VARCHAR) END) AS in_cohort
            FROM read_parquet('{f}') t
            LEFT JOIN ids_df i ON CAST(t.hospitalization_id AS VARCHAR) = i.hospitalization_id
            WHERE lower(REGEXP_REPLACE(CAST(procedure_code AS VARCHAR),
                  '[^0-9A-Za-z]', '', 'g')) IN ({vals}) OR {pats}""").df().iloc[0]
        sw, ic = int(q.site_wide), int(q.in_cohort)
        elem_rows.append({"site": SITE, "group": "neurologic procedure",
                          "concept": concept, "source_table": "patient_procedures",
                          "status": ("not recorded at this site" if sw == 0 else
                                     "recorded, absent in decedents" if ic == 0
                                     else "present"),
                          "n_hosp_site_wide": sw, "n_hosp_in_cohort": ic})

    # medications, per mCIDE category, in whichever table carries them
    for grp, meds in _CRIT["clinical_care"]["medications"].items():
        for route, tb in [("continuous", "medication_admin_continuous"),
                          ("intermittent", "medication_admin_intermittent")]:
            _probe(f"{grp} [{route}]", "medication", tb, "med_category", values=meds)

    # fungemia and the contraindication organisms
    _fg = _CRIT["clinical_care"]["fungemia"]
    _probe("fungemia organisms", "contraindication", "microbiology_culture", "organism_category",
           regexes=[_fg["organism_regex"].replace("|", "|")])
    for org in _CRIT["clinical_care"]["contraindications_not_in_clif"]:
        _probe(org, "contraindication", "microbiology_culture", "organism_category",
               regexes=[org.split("_")[0]])

    # prone positioning
    _probe("prone", "positioning", "position", "position_category", values=["prone"])

    # measurement vocabularies the tables depend on
    for lab in ["creatinine", "bilirubin_total", "ast", "alt", "platelet_count", "lactate"]:
        _probe(lab, "laboratory", "labs", "lab_category", values=[lab])
    for asm in ["gcs_total", "RASS"]:
        _probe(asm, "assessment", "patient_assessments", "assessment_category", values=[asm])

_ec = pl.DataFrame(elem_rows)
if _ec.height:
    _ec = _ec.sort(["group", "concept"])
    _ec.write_csv(FINAL / "element_coverage.csv")
    _bad = _ec.filter(pl.col("status") != "present")
    print(f"\n2d. element coverage -> {_ec.height} concepts; "
          f"{_bad.height} not present ("
          f"{_ec.filter(pl.col('status') == 'not recorded at this site').height} vocabulary gaps)")
    for r in _bad.iter_rows(named=True):
        print(f"     {r['status']:30s} {r['group']}: {r['concept']}")

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
        if len(on) > 1:
            pass
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
_dac_path = REPO / "utils/codes/donor_administrative_codes.csv"
if _dac_path.exists() and "hospitalization_id" in cohort.columns:
    _dac = pl.read_csv(_dac_path, comment_prefix="#", infer_schema_length=0)
    _nrm = lambda s: _re.sub(r"[^0-9A-Za-z]", "", str(s)).upper()

    con.register("coh_dac", cohort.select(
        ["patient_id", "hospitalization_id", "clif_eligible_donors"]).with_columns(
        pl.col("hospitalization_id").cast(pl.Utf8)).to_pandas())

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

# ── provenance: one document, and a check that the artifacts agree ───────────
# Last site step, so every artifact exists by now. A mismatch between the
# STROBE counts, the CONSORT, definition_counts and the Table 2 denominators
# means one of them is wrong, and a wrong number that only appears in one place
# is exactly what nobody notices.
from utils.provenance import reconcile, render          # noqa: E402
_problems = reconcile(FINAL)
_prov = render(FINAL, _problems)
print(f"\nwrote {_prov}")
if _problems:
    print("\nRECONCILIATION FAILED:")
    for _p in _problems:
        print(f"  - {_p}")
    raise SystemExit("The run's own counts disagree. See provenance.md.")
print("Reconciliation passed: STROBE, CONSORT, definition_counts and Table 2 agree")
print(f"\nwrote -> {FINAL}")
