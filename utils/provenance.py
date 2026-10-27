"""One provenance document per site, and a check that it hangs together.

The pipeline counts the same patients in four places: the STROBE dict in step 01,
the stage audit cards in 01-03, the per-definition CONSORT in 02, and the Table 2
denominators. Each was written for its own purpose and none of them checked the
others, so a mislabelled key or a filter applied in the wrong order could show up
in one artifact and nowhere else.

`reconcile()` asserts the identities that must hold between them and returns the
failures. `render()` writes `provenance.md`, a single PHI-free document a site can
read end to end and return with its results.

    from utils.provenance import reconcile, render
    problems = reconcile(final_dir)
    render(final_dir, problems)
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from utils.figure1 import flow_problems

# CONSORT definition name -> column in definition_counts.csv
DEF_COUNT_COL = {
    "CLIF-donor": "clif_donor",
    "CALC": "calc",
    "Ventilated Patient": "ventilated_no_age_limit",
}
# ... and the STROBE key that should agree with it
DEF_STROBE_KEY = {
    "CLIF-donor": "clif_eligible_donors",
    "CALC": "calc_qualified",
}


def _read(d: Path, name: str) -> pl.DataFrame | None:
    f = d / name
    if not f.is_file():
        return None
    try:
        return pl.read_csv(f, infer_schema_length=0)
    except Exception:
        return None


def _int(x) -> int | None:
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def reconcile(final_dir: Path) -> list[str]:
    """Return a list of mismatches. Empty means every artifact agrees."""
    d = Path(final_dir)
    strobe = _read(d, "strobe_counts.csv")
    counts = _read(d, "definition_counts.csv")
    stats = _read(d, "table_stats_raw.csv")
    problems: list[str] = []

    flow = _read(d, "figure1_counts.csv")
    missing = [n for n, v in [("strobe_counts.csv", strobe), ("figure1_counts.csv", flow),
                              ("definition_counts.csv", counts),
                              ("table_stats_raw.csv", stats)] if v is None]
    if missing:
        return [f"cannot reconcile: {', '.join(missing)} not found in {d}"]

    # strobe_counts.csv is long: site, order, metric, n
    S = {r["metric"]: _int(r["n"]) for r in strobe.iter_rows(named=True)}
    C = {c: _int(counts[c][0]) for c in counts.columns}
    F = {r["metric"]: _int(r["n"]) for r in flow.iter_rows(named=True)}
    stats = stats.with_columns(pl.col("n").cast(pl.Float64, strict=False),
                               pl.col("denom").cast(pl.Float64, strict=False))

    # 1. the cohort: the STROBE cascade, Figure 1 and definition_counts end on
    #    the same number, and Table 2 uses it as its denominator
    cohort_n = S.get("2_inpatient_decedents")
    for what, v in (("figure1_counts 'analytic'", F.get("analytic")),
                    ("definition_counts 'n_inpatient_deaths'", C.get("n_inpatient_deaths"))):
        if v != cohort_n:
            problems.append(f"cohort size: {what} is {v} but strobe_counts says {cohort_n}")

    # 2. each definition: Figure 1 == definition_counts == strobe == Table 2 "N patients"
    for defn, col, fkey, skey in (("CLIF-donor", "clif_donor", "clif_donor", "clif_eligible_donors"),
                                  ("CALC", "calc", "calc", "calc_qualified"),
                                  ("Ventilated Patient", "ventilated_no_age_limit", "imv", None)):
        n = C.get(col)
        if F.get(fkey) != n:
            problems.append(f"{defn}: figure1_counts '{fkey}' is {F.get(fkey)} but definition_counts '{col}' is {n}")
        if skey and S.get(skey) is not None and S[skey] != n:
            problems.append(f"{defn}: strobe_counts '{skey}' is {S[skey]:,} but definition_counts '{col}' is {n}")
        t2 = stats.filter((pl.col("variable") == "N patients") & (pl.col("definition") == defn))
        if t2.height:
            n_t2 = _int(t2["n"][0])
            if n_t2 != n:
                problems.append(f"{defn}: Table 2 'N patients' is {n_t2:,} but definition_counts '{col}' is {n}")
            denom = _int(t2["denom"][0])
            if denom != cohort_n:
                problems.append(f"{defn}: Table 2 denominator is {denom:,} but the cohort "
                                f"is {cohort_n:,}")

    # 3. Figure 1's own identities, re-checked on the file as written
    problems += [f"figure1_counts: {p}" for p in flow_problems(F)]

    # 4. the per-hospital counts cover the whole cohort, and the CALC readings
    #    they carry match the strobe rows they were counted from
    hosp = _read(d, "hospital_level_counts.csv")
    if hosp is not None and cohort_n is not None:
        n_h = sum(_int(x) or 0 for x in hosp["n_decedents"].to_list())
        if n_h != cohort_n:
            problems.append(f"hospital_level_counts covers {n_h:,} decedents but the cohort is {cohort_n:,}")
        for pos in ("any", "primary", "primary_poa"):
            col, key = f"CALC_{pos}", f"calc_qualified_{pos}"
            if col in hosp.columns and S.get(key) is not None:
                n_c = sum(_int(x) or 0 for x in hosp[col].to_list())
                if n_c != S[key]:
                    problems.append(f"hospital_level_counts '{col}' sums to {n_c:,} but strobe_counts '{key}' is {S[key]:,}")
    return problems


def render(final_dir: Path, problems: list[str] | None = None) -> Path:
    """Write provenance.md: cohort cascade, Figure 1 counts, stage audits, reconciliation."""
    d = Path(final_dir)
    site = d.name
    out = ["# Provenance — " + site, "",
           "Every count this run produced, and whether the artifacts agree.", ""]

    problems = reconcile(d) if problems is None else problems
    out += ["## Reconciliation", ""]
    if problems:
        out += ["**FAILED** — the artifacts disagree:", ""] + [f"- {p}" for p in problems]
    else:
        out += ["All artifacts agree: `figure1_counts.csv`, `definition_counts.csv`, "
                "`strobe_counts.csv`, the Table 2 denominators and the hospital-level counts are consistent."]
    out += [""]

    flags = _read(d, "data_quality_flags.csv")
    if flags is not None:
        out += ["## Data-quality flags", ""]
        if flags.height:
            out += ["These do not stop the run, but they change what a number means at this site.",
                    "", "| flag | severity | what it means |", "|---|---|---|"]
            out += [f"| `{r['flag']}` | {r['severity']} | {r['detail']} |"
                    for r in flags.iter_rows(named=True)]
        else:
            out += ["None raised by this run."]
        out += [""]

    strobe = _read(d, "strobe_counts.csv")
    if strobe is not None:
        out += ["## Cohort cascade (step 01)", "", "| count | n |", "|---|---:|"]
        for r in strobe.iter_rows(named=True):
            v = _int(r["n"])
            out.append(f"| `{r['metric']}` | {v:,} |" if v is not None
                       else f"| `{r['metric']}` | {r['n']} |")
        out += [""]

    flow = _read(d, "figure1_counts.csv")
    if flow is not None:
        out += ["## Figure 1 counts (step 03)", "",
                "Exclusions before analysis, CALC in sequence, CLIF-donor's six criteria each counted "
                "among all cohort decedents, and the organ split of the eligible.", "",
                "| count | n |", "|---|---:|"]
        out += [f"| `{r['metric']}` | {_int(r['n']):,} |" for r in flow.iter_rows(named=True)]
        out += [""]

    cards = sorted((d / "audit").glob("*_stage_counts.csv")) if (d / "audit").is_dir() else []
    if cards:
        out += ["## Stage audit", "", "| stage | in | out | dropped | rule |", "|---|---:|---:|---:|---|"]
        for f in cards:
            for r in pl.read_csv(f, infer_schema_length=0).iter_rows(named=True):
                # a note (audit.note) has no patients but carries its message in `rule`
                if _int(r["n_in"]) or (r.get("rule") or "").strip():
                    out.append(f"| `{r['stage']}` | {_int(r['n_in']):,} | {_int(r['n_out']):,} "
                               f"| {_int(r['n_dropped']):,} | {r['rule']} |")
        out += [""]

    p = d / "provenance.md"
    p.write_text("\n".join(out))
    return p
