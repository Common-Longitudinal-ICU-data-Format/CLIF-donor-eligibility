"""Site config loader: reads config/config.json."""
from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.json"


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"{CONFIG_PATH} not found. Copy config/config_template.json to config/config.json "
            f"and fill it in.")
    cfg = json.loads(CONFIG_PATH.read_text())
    cfg["site_name"] = str(cfg.get("site_name", "unknown")).lower()
    # Two output roots:
    #   output/intermediate_phi/<site>/  patient-level working data, never leaves the site
    #   <site>_upload_to_box/            aggregate results to return
    cfg["output_intermediate"] = str(PROJECT_ROOT / "output/intermediate_phi" / cfg["site_name"])
    cfg["output_final"] = str(PROJECT_ROOT / f"{cfg['site_name']}_upload_to_box")
    print(f"Loaded configuration from {CONFIG_PATH.name}")
    return cfg


config = load_config()
