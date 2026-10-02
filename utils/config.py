"""Site config loader.

Which site runs is chosen by the CLIF_DONOR_SITE environment variable, so the
same code runs unmodified at every site:

    CLIF_DONOR_SITE=ucmc python code/01_cohort_and_definitions.py

Falls back to config/config.json when the variable is unset, preserving the
original single-site behaviour.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"


def load_config(site: str | None = None) -> dict:
    site = site or os.environ.get("CLIF_DONOR_SITE")
    path = CONFIG_DIR / (f"config_{site.lower()}.json" if site else "config.json")
    if not path.exists():
        avail = sorted(p.stem.replace("config_", "") for p in CONFIG_DIR.glob("config_*.json")
                       if not p.stem.startswith(("config_template", "_retired")))
        raise FileNotFoundError(
            f"{path.name} not found. Set CLIF_DONOR_SITE to one of: {avail}")
    cfg = json.load(open(path))

    # project_root is NOT overwritten here. It used to be, which silently made
    # the value in config.json meaningless and made 01's check of it unable to
    # ever fail. 01 validates that it points at this repo and stops if it does
    # not. Output paths below still derive from this file's location, so they
    # are correct even while that check is running.
    cfg.setdefault("project_root", str(PROJECT_ROOT))
    cfg.setdefault("site_name", site or "unknown")
    cfg["site_name"] = str(cfg["site_name"]).lower()
    # Two output roots, and the names say what each is for.
    #   output/intermediate_phi/<site>/  patient-level working data, NEVER leaves the site
    #   <site>_upload_to_box/            everything to return, aggregate counts only
    # intermediate_phi keeps the CLIF project-template name: the folder says what
    # it holds, and output/intermediate_phi/README.md spells out the rule.
    # The site subfolder under intermediate is what lets the coordinating centre
    # run several sites on one machine without one overwriting another.
    cfg.setdefault("output_intermediate",
                   str(PROJECT_ROOT / "output/intermediate_phi" / cfg["site_name"]))
    cfg.setdefault("output_final",
                   str(PROJECT_ROOT / f"{cfg['site_name']}_upload_to_box"))
    print(f"Loaded configuration from {path.name}")
    return cfg


config = load_config()
