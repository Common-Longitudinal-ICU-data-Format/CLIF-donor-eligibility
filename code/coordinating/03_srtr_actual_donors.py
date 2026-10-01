#!/usr/bin/env python
"""Actual donor numbers from SRTR. Coordinating centre only — never run at a site.

Sites return `hospital_level_counts.csv` carrying `srtr_ccn_id`. This script
collects those CCNs across every returned site, tells you which hospitals
resolved in SRTR and which did not, and produces the "Actual donors" column for
manuscript Table 1.

The SRTR extract is under a DUA. Nothing here reads CLIF data and nothing here
runs in the per-site pipeline.

    python code/coordinating/03_srtr_actual_donors.py
    python code/coordinating/03_srtr_actual_donors.py --sites-dir manuscript_results \\
        --srtr-dir /path/to/00_SRTR_DATA --years 2020 2025

Writes to the sites directory:

    srtr_hospital_match.csv    one row per returned hospital: did its CCN
                               resolve, and how many donors did it have
    srtr_actual_donors.csv     the Table 1 "Actual donors" column
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from utils.srtr import donor_characteristics, match_hospitals   # noqa: E402

# The SRTR extract is under a DUA and lives outside the repository. Set
# CLIF_SRTR_DIR, or keep it in a 00_SRTR_DATA folder beside this repo.
DEFAULT_SRTR = os.environ.get("CLIF_SRTR_DIR", str(REPO.parent / "00_SRTR_DATA"))


def collect_hospitals(sites_dir: Path) -> pl.DataFrame:
    """Pool every returned site's hospital_level_counts.csv."""
    frames = []
    for f in sorted(glob.glob(str(sites_dir / "*/hospital_level_counts.csv"))):
        site = Path(f).parent.name
        d = pl.read_csv(f, infer_schema_length=0)
        if "site" not in d.columns:
            d = d.with_columns(pl.lit(site).alias("site"))
        frames.append(d)
    if not frames:
        raise SystemExit(f"no hospital_level_counts.csv under {sites_dir}\n"
                         "Collect the sites' returned <site>_upload_to_box/ folders first.")
    d = pl.concat(frames, how="diagonal")
    for c in ("n_decedents", "CLIF_donor", "CALC", "Ventilated_Patient"):
        if c in d.columns:
            d = d.with_columns(pl.col(c).cast(pl.Int64, strict=False))
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites-dir", default=str(REPO / "manuscript_results"))
    ap.add_argument("--srtr-dir", default=DEFAULT_SRTR)
    ap.add_argument("--years", nargs=2, type=int, default=[2020, 2025],
                    metavar=("MIN", "MAX"),
                    help="recovery-year window, inclusive (default 2020 2025)")
    a = ap.parse_args()

    sites_dir, srtr = Path(a.sites_dir), Path(a.srtr_dir)
    if not srtr.is_dir():
        raise SystemExit(f"SRTR directory not found: {srtr}")

    hosp = collect_hospitals(sites_dir)
    sites = sorted(set(hosp["site"]))
    print(f"sites returned: {', '.join(sites)}")
    print(f"hospitals returned: {hosp.height}")
    print(f"donor recovery window: {a.years[0]}–{a.years[1]}\n")

    matched, donor_ids = match_hospitals(srtr, hosp, a.years[0], a.years[1])

    # Which hospitals resolved in SRTR, and which did not. A hospital that does
    # not resolve should not be in the denominator at all — that is the rule the
    # per-site pipeline already applies, and this is the check on it.
    keep = [c for c in ("site", "hospital_label", "srtr_ccn_id", "matched_srtr",
                        "n_decedents", "CLIF_donor", "CALC", "srtr_donors")
            if c in matched.columns]
    out_match = matched.select(keep).sort(["site", "hospital_label"])
    out_match.write_csv(sites_dir / "srtr_hospital_match.csv")

    print("hospital match:")
    for s in sites:
        sub = out_match.filter(pl.col("site") == s)
        ok = int(sub["matched_srtr"].sum())
        print(f"  {s:6s} {ok}/{sub.height} hospitals resolved in SRTR, "
              f"{int(sub['srtr_donors'].sum()):,} donors")
    unmatched = out_match.filter(~pl.col("matched_srtr"))
    if unmatched.height:
        print(f"\n  {unmatched.height} hospital(s) did NOT resolve — these should not "
              "be in the denominator:")
        for r in unmatched.iter_rows(named=True):
            print(f"    {r['site']:6s} {r.get('hospital_label')} "
                  f"ccn={r.get('srtr_ccn_id')!r}")
    zero = out_match.filter(pl.col("matched_srtr") & (pl.col("srtr_donors") == 0))
    if zero.height:
        print(f"\n  {zero.height} hospital(s) resolved but had 0 donors in the window "
              "(real, not an error):")
        for r in zero.iter_rows(named=True):
            print(f"    {r['site']:6s} {r.get('hospital_label')} ccn={r.get('srtr_ccn_id')}")

    print(f"\ntotal distinct donors at these hospitals: {len(donor_ids):,}")

    col = donor_characteristics(srtr, donor_ids)
    if col is None:
        print("no donors matched — nothing to write for the Actual donors column")
        return 1
    col.write_csv(sites_dir / "srtr_actual_donors.csv")
    print(f"\nActual donors column -> {sites_dir / 'srtr_actual_donors.csv'}")
    with pl.Config(tbl_rows=40, fmt_str_lengths=44):
        print(col.select(["variable", "value"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
