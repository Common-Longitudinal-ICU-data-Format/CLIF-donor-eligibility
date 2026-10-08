"""The time of death used to anchor every "within N hours of death" window.

Sites record death differently. Some store a time of day, some store only the
date (as a timestamp at 00:00:00), and some leave it empty. 

    death_dttm has a time of day   -> death_dttm
    death_dttm is a date           -> discharge_dttm of the death hospitalization,
                                      kept inside that date: its last second if
                                      discharge is later, its first if earlier
                                      same logic if death_dttm time is at midnight.
    death_dttm is missing          -> discharge_dttm

A date-only death keeps its date because the recorded date is the death. For a
patient managed as a donor after death is declared, discharge is recorded at
organ recovery, a day or more later, and is not the time of death.

A death recorded long after discharge (a date linked from an outside registry,
or an error) is disregarded and discharge is used: the cohort is in-hospital
deaths, so the patient died no later than discharge.

The last recorded vital is not used. Units such as inpatient hospice chart no
vitals, and a vitals extract that ends before the hospitalization extract makes
the last vital the extract's cut-off, not the death.
"""
from __future__ import annotations

import polars as pl

# Where each patient's anchor came from. Reported per site so the mix is visible.
SOURCES = (
    "death_dttm",                        # recorded with a time of day
    "discharge_on_death_date",           # date-only death; discharge falls on that date
    "death_date_end",                    # date-only death; discharge is later (or absent)
    "death_date_start",                  # date-only death; discharge is shortly before the date
    "discharge_death_missing",           # no death_dttm
    "discharge_death_after_discharge",   # death recorded long after discharge; disregarded
    "none",                              # neither timestamp present
)


def _is_midnight(e: pl.Expr) -> pl.Expr:
    return (e.dt.hour() == 0) & (e.dt.minute() == 0) & (e.dt.second() == 0)


def _local(col: str, dtype: pl.Datetime, site_tz: str) -> pl.Expr:
    """Wall-clock reading at the hospital. A timezone-naive column is already local."""
    e = pl.col(col)
    return e.dt.convert_time_zone(site_tz) if dtype.time_zone else e


def has_time_of_day(col: str, dtype: pl.Datetime, site_tz: str) -> pl.Expr:
    """True where `col` is a recorded death time, not a date stored at local midnight."""
    return pl.col(col).is_not_null() & ~_is_midnight(_local(col, dtype, site_tz))


def comparable(col: str, dtype: pl.Datetime, target: pl.Datetime) -> pl.Expr:
    """`col` as the same instant in the dtype of `target`, so the two can be compared.

    Sites store the two columns with different precision, and sometimes in
    different (aware) zones; polars compares neither without this.
    """
    e = pl.col(col)
    if target.time_zone and dtype.time_zone != target.time_zone:
        e = e.dt.convert_time_zone(target.time_zone)
    return e.cast(target)


def add_death_anchor(df: pl.DataFrame, site_tz: str, max_hours_after_discharge: float,
                     death: str = "death_dttm", discharge: str = "discharge_dttm") -> pl.DataFrame:
    """Add `final_death_dttm` and `death_time_source` to `df`.

    `final_death_dttm` has the dtype of the death column. `site_tz` is the
    hospital's timezone: "date only" means exactly 00:00:00 on its wall clock.
    """
    d_ty, c_ty = df.schema[death], df.schema[discharge]
    if not isinstance(d_ty, pl.Datetime) or not isinstance(c_ty, pl.Datetime):
        raise TypeError(f"{death} and {discharge} must be datetimes, got {d_ty} and {c_ty}")
    if bool(d_ty.time_zone) != bool(c_ty.time_zone):
        raise ValueError(f"{death} is {d_ty} and {discharge} is {c_ty}: one has a timezone and "
                         f"the other does not, so they cannot be compared")

    dth = pl.col(death)
    disc = comparable(discharge, c_ty, d_ty)                  # same zone and time unit as the death column

    loc = _local(death, d_ty, site_tz)
    date_only = dth.is_not_null() & _is_midnight(loc)
    start = dth                                               # local midnight is the start of the death date
    # The last second of the death's calendar date. Worked out on the wall clock
    # and then placed back in the hospital's timezone, so a 23- or 25-hour day is
    # right, and so a timed death an hour after midnight never asks for a
    # clock time that occurs twice.
    wall = loc.dt.replace_time_zone(None) if d_ty.time_zone else loc
    end = wall.dt.truncate("1d").dt.offset_by("1d") - pl.duration(seconds=1)
    if d_ty.time_zone:
        end = (end.dt.replace_time_zone(site_tz, ambiguous="latest", non_existent="null")
                  .dt.convert_time_zone(d_ty.time_zone))
    end = end.cast(d_ty)

    recorded_long_after = (dth - disc).dt.total_seconds() > max_hours_after_discharge * 3600

    cases = [   # first match wins
        (dth.is_null() & disc.is_null(), pl.lit(None, d_ty), "none"),
        (dth.is_null(), disc, "discharge_death_missing"),
        (recorded_long_after, disc, "discharge_death_after_discharge"),
        (~date_only, dth, "death_dttm"),
        (disc.is_null() | (disc > end), end, "death_date_end"),
        (disc < start, start, "death_date_start"),
    ]
    anchor = pl.when(cases[0][0]).then(cases[0][1])
    source = pl.when(cases[0][0]).then(pl.lit(cases[0][2]))
    for cond, value, label in cases[1:]:
        anchor = anchor.when(cond).then(value)
        source = source.when(cond).then(pl.lit(label))
    return df.with_columns(
        anchor.otherwise(disc).alias("final_death_dttm"),
        source.otherwise(pl.lit("discharge_on_death_date")).alias("death_time_source"),
    )


def midnight_shares(df: pl.DataFrame, site_tz: str, death: str = "death_dttm") -> dict[str, float]:
    """Share of recorded death times at exactly local midnight, and at exactly UTC midnight.

    A real death lands on any given minute about 0.07% of the time, so a share
    well above that means dates stored as timestamps. Local midnight is handled
    by `add_death_anchor`. A large share at UTC midnight is a different encoding
    of the same thing that the rule does not handle, and should be flagged.
    """
    ty = df.schema[death]
    rec = df.filter(pl.col(death).is_not_null())
    if rec.height == 0:
        return {"local": 0.0, "utc": 0.0}
    local = _is_midnight(_local(death, ty, site_tz))
    utc = (_is_midnight(pl.col(death).dt.convert_time_zone("UTC")) & ~local) if ty.time_zone else pl.lit(False)
    out = rec.select(local.mean().alias("local"), utc.mean().alias("utc")).row(0, named=True)
    return {k: float(v) for k, v in out.items()}
