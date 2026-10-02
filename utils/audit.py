"""Stage audit trail.

Every filtering step in this pipeline records what it received, what it emitted,
and what it dropped. The point is that a reviewer can answer "where did those
patients go?" from the audit cards alone, without rerunning anything.

Counts only. Sample rows were removed on 2026-09-28 so the cards can travel to
the coordinating centre with the rest of the upload bundle: an audit trail nobody can
read is not an audit trail, and the samples were the only patient-level content.

Usage
-----
    audit = StageAudit(site="ucmc", outdir=Path("output/intermediate_phi/ucmc"))
    audit.record("02_age", df_in, df_out, key="patient_id",
                 rule="age at death <= 75",
                 reason="older than 75 at death")
    audit.write()          # -> audit/stage_cards.md + stage_counts.csv
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl


def _n(df: Any, key: str | None) -> int:
    """Distinct count on `key`, else row count. Works for polars or pandas."""
    if df is None:
        return 0
    if key and key in getattr(df, "columns", []):
        col = df[key]
        return int(col.n_unique() if hasattr(col, "n_unique") else col.nunique())
    return int(len(df))


@dataclass
class Stage:
    name: str
    rule: str
    reason: str
    key: str | None
    n_in: int
    n_out: int
    sample: list[dict] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    @property
    def n_dropped(self) -> int:
        return self.n_in - self.n_out

    @property
    def pct_retained(self) -> float:
        return round(100.0 * self.n_out / self.n_in, 2) if self.n_in else float("nan")


class StageAudit:
    def __init__(self, site: str, outdir: Path, label: str = "cohort"):
        self.site, self.label = site, label
        self.outdir = Path(outdir)
        self.audit_dir = self.outdir / "audit"
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.stages: list[Stage] = []
        self.started = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def record(self, name, df_in, df_out, *, key=None, rule="", reason="",
               sample_cols=None, n_sample=5, **extra) -> None:
        # sample_cols / n_sample are accepted and ignored: callers still pass
        # them, and no patient rows are recorded.
        self.stages.append(Stage(name, rule, reason, key,
                                 _n(df_in, key), _n(df_out, key), [], extra))

    def note(self, name: str, message: str, **extra) -> None:
        """Record something that is not a filter (a diagnostic, a known gap)."""
        self.stages.append(Stage(name, message, "", None, 0, 0, [], extra))

    def write(self) -> Path:
        rows = [{
            "site": self.site, "label": self.label, "stage": s.name,
            "n_in": s.n_in, "n_out": s.n_out, "n_dropped": s.n_dropped,
            "pct_retained": s.pct_retained, "rule": s.rule, "reason": s.reason,
            **{k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in s.extra.items()},
        } for s in self.stages]
        pl.DataFrame(rows).write_csv(self.audit_dir / f"{self.label}_stage_counts.csv")

        # CSV only. A markdown twin of the same table had to be read by a human
        # per site and could not be concatenated; the counts are what the
        # coordinating centre aggregates, and provenance.md is the one narrative
        # document per site.
        return self.audit_dir / f"{self.label}_stage_counts.csv"

    def summary(self) -> str:
        return "\n".join(
            f"  {s.name:34s} {s.n_in:>9,} -> {s.n_out:>9,}  ({s.n_dropped:>8,} dropped)"
            for s in self.stages if s.n_in or s.n_out)
