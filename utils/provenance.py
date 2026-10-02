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
    consort = _read(d, "consort_counts.csv")
    counts = _read(d, "definition_counts.csv")
    stats = _read(d, "table_stats_raw.csv")
    problems: list[str] = []

    missing = [n for n, v in [("strobe_counts.csv", strobe), ("consort_counts.csv", consort),
                              ("definition_counts.csv", counts),
                              ("table_stats_raw.csv", stats)] if v is None]
    if missing:
        return [f"cannot reconcile: {', '.join(missing)} not found in {d}"]

    # strobe_counts.csv is long: site, order, metric, value, n
    S = {r["metric"]: _int(r["n"]) for r in strobe.iter_rows(named=True)}
    C = {c: _int(counts[c][0]) for c in counts.columns}
    consort = consort.with_columns(pl.col("n").cast(pl.Int64), pl.col("step").cast(pl.Int64))
    stats = stats.with_columns(pl.col("n").cast(pl.Float64, strict=False),
                               pl.col("denom").cast(pl.Float64, strict=False))

    # 1. the cohort: every CONSORT starts at the same place, and that is the
    #    number the STROBE cascade ends on and Table 2 uses as its denominator.
    cohort_n = S.get("2_inpatient_decedents")
    for defn in consort["definition"].unique().to_list():
        step0 = consort.filter((pl.col("definition") == defn) & (pl.col("step") == 0))["n"]
        if step0.len() and step0[0] != cohort_n:
            problems.append(f"cohort size: CONSORT '{defn}' starts at {step0[0]:,} but "
                            f"strobe_counts says {cohort_n:,}")
    if C.get("n_inpatient_deaths") != cohort_n:
        problems.append(f"cohort size: definition_counts says {C.get('n_inpatient_deaths'):,} "
                        f"but strobe_counts says {cohort_n:,}")

    # 2. each definition: CONSORT endpoint == definition_counts == Table 2 "N patients"
    for defn, col in DEF_COUNT_COL.items():
        rows = consort.filter(pl.col("definition") == defn).sort("step")
        if not rows.height:
            continue
        endpoint = rows["n"][-1]
        if C.get(col) != endpoint:
            problems.append(f"{defn}: CONSORT ends at {endpoint:,} but definition_counts "
                            f"'{col}' is {C.get(col):,}")
        t2 = stats.filter((pl.col("variable") == "N patients") & (pl.col("definition") == defn))
        if t2.height:
            n_t2 = _int(t2["n"][0])
            if n_t2 != endpoint:
                problems.append(f"{defn}: CONSORT ends at {endpoint:,} but Table 2 "
                                f"'N patients' is {n_t2:,}")
            denom = _int(t2["denom"][0])
            if denom != cohort_n:
                problems.append(f"{defn}: Table 2 denominator is {denom:,} but the cohort "
                                f"is {cohort_n:,}")
        key = DEF_STROBE_KEY.get(defn)
        if key and S.get(key) is not None and S[key] != endpoint:
            problems.append(f"{defn}: CONSORT ends at {endpoint:,} but strobe_counts "
                            f"'{key}' is {S[key]:,}")

    # 3. a CONSORT can only ever shrink
    for defn in consort["definition"].unique().to_list():
        rows = consort.filter(pl.col("definition") == defn).sort("step")
        ns = rows["n"].to_list()
        for a, b, lab in zip(ns, ns[1:], rows["label"].to_list()[1:]):
            if b > a:
                problems.append(f"{defn}: step '{lab.splitlines()[0]}' goes UP, {a:,} -> {b:,}")
    return problems


def render(final_dir: Path, problems: list[str] | None = None) -> Path:
    """Write provenance.md: cohort cascade, stage audits, CONSORT, reconciliation."""
    d = Path(final_dir)
    site = d.name
    out = ["# Provenance — " + site, "",
           "Every count this run produced, and whether the artifacts agree.", ""]

    problems = reconcile(d) if problems is None else problems
    out += ["## Reconciliation", ""]
    if problems:
        out += ["**FAILED** — the artifacts disagree:", ""] + [f"- {p}" for p in problems]
    else:
        out += ["All artifacts agree: the CONSORT endpoints, `definition_counts.csv`, "
                "`strobe_counts.csv` and the Table 2 denominators are consistent."]
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
                       else f"| `{r['metric']}` | {r['value']} |")
        out += [""]

    consort = _read(d, "consort_counts.csv")
    if consort is not None:
        out += ["## CONSORT by definition (step 02)", ""]
        for defn in consort["definition"].unique(maintain_order=True).to_list():
            out += [f"### {defn}", "", "| step | criterion | n remaining |", "|---:|---|---:|"]
            for r in consort.filter(pl.col("definition") == defn).sort(
                    pl.col("step").cast(pl.Int64)).iter_rows(named=True):
                out.append(f"| {r['step']} | {r['label'].replace(chr(10), ' ')} "
                           f"| {_int(r['n']):,} |")
            out += [""]

    cards = sorted((d / "audit").glob("*_stage_counts.csv")) if (d / "audit").is_dir() else []
    if cards:
        out += ["## Stage audit", "", "| stage | in | out | dropped | rule |", "|---|---:|---:|---:|---|"]
        for f in cards:
            for r in pl.read_csv(f, infer_schema_length=0).iter_rows(named=True):
                if _int(r["n_in"]):
                    out.append(f"| `{r['stage']}` | {_int(r['n_in']):,} | {_int(r['n_out']):,} "
                               f"| {_int(r['n_dropped']):,} | {r['rule']} |")
        out += [""]

    p = d / "provenance.md"
    p.write_text("\n".join(out))
    return p
