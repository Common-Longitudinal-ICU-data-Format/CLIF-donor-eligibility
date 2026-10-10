"""Elapsed time and peak memory per section of a run, for run_log.txt.

Sites differ a hundredfold in table size, and "the run struggles" cannot be
acted on until the log says where. Each mark prints how long the section just
finished took and the process's peak memory so far. The lines ship in
run_log.txt and hold no patient data.
"""
from __future__ import annotations

import sys
import time


def peak_memory_gb() -> float | None:
    """Peak resident memory of this process so far, in GB; None where unknown."""
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / 1e9 if sys.platform == "darwin" else rss / 1e6     # bytes on macOS, KiB elsewhere
    except Exception:                                                     # Windows has no `resource`
        try:
            import psutil
            return psutil.Process().memory_info().peak_wset / 1e9
        except Exception:
            return None


def format_mark(section: str, seconds: float, total: float, peak_gb: float | None) -> str:
    memory = f", peak memory {peak_gb:.1f} GB" if peak_gb is not None else ""
    return f"  [{total:6.0f}s] {section}: {seconds:.1f}s{memory}"


class Progress:
    """`mark(section)` prints a line for the section that just finished."""

    def __init__(self) -> None:
        self._start = self._last = time.monotonic()

    def mark(self, section: str) -> str:
        now = time.monotonic()
        line = format_mark(section, now - self._last, now - self._start, peak_memory_gb())
        self._last = now
        print(line)
        return line
