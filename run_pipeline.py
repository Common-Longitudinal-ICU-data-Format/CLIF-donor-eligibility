#!/usr/bin/env python
"""Run the CLIF-donor pipeline at a site.

For each configured site, in order:
    1. copy config/config_<site>.json  ->  config/config.json
    2. run the pipeline (01, 02, 03)
    3. results land in <site>_upload_to_box/

That is where a site stops. <site>_upload_to_box/ holds the aggregate counts to return
to the coordinating centre. output/intermediate_phi/<site>/ holds the patient-level
cohort and the per-step logs, and stays local.

Each site needs its own config/config_<site>.json. A site with a single
config/config.json runs the three scripts directly instead (see README.md).

Pooling across sites is a coordinating-centre step and is deliberately NOT part
of this script: the coordinating centre collects the returned bundles into
manuscript_results/<site>/ and runs

    python code/coordinating/01_combined_report.py --sites-dir manuscript_results

    python run_pipeline.py
    python run_pipeline.py --sites ucmc nu
    python run_pipeline.py --steps 02 03      # re-run only the tables and diagnostics
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
CONFIG_DIR = REPO / "config"
LIVE_CONFIG = CONFIG_DIR / "config.json"

STEPS = ["01_cohort_and_definitions.py", "02_build_tables.py",
         "03_diagnostics.py"]
G, R, Y, B, X = "\033[32m", "\033[31m", "\033[33m", "\033[1m", "\033[0m"


def configured_sites() -> list[str]:
    out = []
    for p in sorted(CONFIG_DIR.glob("config_*.json")):
        if p.stem.startswith(("config_template", "_retired")):
            continue
        try:
            if Path(json.loads(p.read_text()).get("tables_path", "")).is_dir():
                out.append(p.stem.replace("config_", ""))
        except Exception:
            pass
    return out


def run_site(site: str, steps: list[str] | None = None) -> bool:
    src = CONFIG_DIR / f"config_{site}.json"
    print(f"\n{B}{site.upper()}{X}")
    if not src.is_file():
        print(f"  {R}{src.name} not found{X} — create it from config_template.json")
        return False

    # 1. make this site's config the live one
    shutil.copyfile(src, LIVE_CONFIG)
    print(f"  config    {src.name} -> config.json")

    # 2. run the pipeline
    env = {**os.environ, "CLIF_DONOR_SITE": site, "PYTHONUNBUFFERED": "1"}
    logdir = REPO / "output/intermediate_phi" / site / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    todo = [s for s in STEPS if steps is None or s[:2] in steps]
    for i, script in enumerate(todo, 1):
        log = logdir / f"{script.replace('.py', '')}.log"
        print(f"  [{i}/{len(todo)}]   {script:34s} … ", end="", flush=True)
        t = time.time()
        with open(log, "w") as fh:
            r = subprocess.run([sys.executable, str(REPO / "code" / script)],
                               stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=REPO)
        if r.returncode != 0:
            print(f"{R}FAILED{X}")
            print("".join(log.read_text().splitlines(keepends=True)[-20:]))
            return False
        print(f"{G}ok{X} ({time.time()-t:.0f}s)")

    # 3. output/ -> <site>_upload_to_box/
    # Nothing to copy: the scripts write straight to <site>_upload_to_box/.
    site_out = REPO / f"{site}_upload_to_box"
    n = len(list(site_out.glob("*"))) if site_out.exists() else 0
    print(f"  output    {site_out.name}/ ({n} files to return)")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", nargs="*")
    ap.add_argument("--steps", nargs="*", default=None,
                    help="run only these step prefixes, e.g. --steps 02 03")
    a = ap.parse_args()

    sites = a.sites or configured_sites()
    if not sites:
        print(f"{R}no usable site configs{X}")
        return 1
    print(f"{B}sites:{X} {', '.join(sites)}")

    backup = LIVE_CONFIG.read_bytes() if LIVE_CONFIG.exists() else None
    ok, failed = [], []
    try:
        for s in sites:
            (ok if run_site(s, a.steps) else failed).append(s)
    finally:
        if backup is not None:
            LIVE_CONFIG.write_bytes(backup)

    print(f"\n{B}sites complete{X} {', '.join(ok) or '—'}")
    if failed:
        print(f"{R}sites failed{X}   {', '.join(failed)}")
    for s in ok:
        print(f"  return  {s}_upload_to_box/")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
