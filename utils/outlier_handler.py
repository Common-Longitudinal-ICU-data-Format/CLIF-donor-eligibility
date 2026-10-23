"""Plausibility ranges from config/outlier_config.yaml, read inline by step 01."""
from __future__ import annotations

from pathlib import Path

import yaml

_CONFIG = Path(__file__).resolve().parent.parent / "config" / "outlier_config.yaml"


def outlier_range(table_name: str, column_name: str, category: str | None = None,
                  outlier_config_path: str | Path | None = None) -> tuple[float, float]:
    """(min, max) for one column, or for one category within it.

        >>> outlier_range('vitals', 'vital_value', 'weight_kg')
        (20.0, 1100.0)

    Raises rather than returning a default: a missing entry means a range was
    renamed or deleted, and silently applying no bound is how bad values reach
    the cohort.
    """
    config = yaml.safe_load(Path(outlier_config_path or _CONFIG).read_text())
    spec = config["tables"][table_name][column_name]
    if category is not None:
        spec = spec[category]
    return float(spec["min"]), float(spec["max"])
