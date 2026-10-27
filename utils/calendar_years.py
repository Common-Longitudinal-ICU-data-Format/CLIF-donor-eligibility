"""Which calendar years a hospital covers in full, from the months its deaths span.

SRTR counts a hospital's donors by calendar year. A hospital whose extract starts
or stops mid-year would set a part-year of decedents against a full year of
donors, so each hospital contributes only the years it covers in full: its first
death is in or before that year's January and its last death in or after its
December. Months are indexes, year * 12 + month - 1, on the site's wall clock.
"""
from __future__ import annotations

import polars as pl


def month_index(local: pl.Expr, first_year: int, last_year: int) -> pl.Expr:
    """Month index of a local-time datetime, clipped to the window's months: a
    date-only 1 January death after a 31 December discharge belongs to the
    window's last December, not to a new year."""
    return (local.dt.year() * 12 + local.dt.month() - 1).clip(first_year * 12, last_year * 12 + 11)


def in_partial_year(month: str, hospital: str) -> pl.Expr:
    """True where the row's calendar year is one its hospital does not cover in full."""
    jan = (pl.col(month) // 12) * 12
    return (pl.col(month).min().over(hospital) > jan) | (pl.col(month).max().over(hospital) < jan + 11)


def complete_years(first: int, last: int) -> tuple[int, int]:
    """First and last calendar year covered in full by the months first..last;
    last comes before first when there is none."""
    return -(-first // 12), (last + 1) // 12 - 1


def ym(month: int) -> str:
    return f"{month // 12}-{month % 12 + 1:02d}"
