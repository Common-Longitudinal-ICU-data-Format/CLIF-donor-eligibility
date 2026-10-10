#!/usr/bin/env python
"""Run the three steps of the CLIF-donor pipeline, in order, from config/config.json.

    python run_pipeline.py                 # 01, 02, 03
    python run_pipeline.py --steps 02 03   # only the tables and diagnostics

Each step's output is shown as it runs and also written to
output/intermediate_phi/<site>/logs/<step>.log; the run stops at the first step that fails. Results land in <site>_upload_to_box/,
the aggregate counts to return to the coordinating centre; output/intermediate_phi/
holds the patient-level cohort and stays at the site.

Pooling across sites is a coordinating-centre step and is deliberately not part
of this script.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
STEPS = ["01_cohort_and_definitions.py", "02_build_tables.py", "03_diagnostics.py"]
G, R, B, X = "\033[32m", "\033[31m", "\033[1m", "\033[0m"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", nargs="*", default=None,
                    help="run only these step prefixes, e.g. --steps 02 03")
    a = ap.parse_args()

    config = REPO / "config/config.json"
    if not config.is_file():
        print(f"{R}config/config.json not found{X}: copy config/config_template.json and fill it in")
        return 1
    site = str(json.loads(config.read_text())["site_name"]).lower()
    logdir = REPO / "output/intermediate_phi" / site / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}

    todo = [s for s in STEPS if a.steps is None or s[:2] in a.steps]
    print(f"{B}{site}{X}")
    for i, script in enumerate(todo, 1):
        log = logdir / f"{script.replace('.py', '')}.log"
        print(f"\n{B}[{i}/{len(todo)}] {script}{X}", flush=True)
        t = time.time()
        with open(log, "w") as fh, subprocess.Popen(
                [sys.executable, str(REPO / "code" / script)], stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, env=env, cwd=REPO, text=True) as proc:
            for line in proc.stdout:            # shown live, and kept in the log
                sys.stdout.write(line); sys.stdout.flush(); fh.write(line)
        if proc.returncode != 0:
            print(f"\n{R}{script} FAILED{X} after {time.time() - t:.0f}s; see {log}")
            return 1
        print(f"{G}{script} ok{X} ({time.time() - t:.0f}s)")
    print(f"  return  {site}_upload_to_box/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
