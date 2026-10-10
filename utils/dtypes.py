"""Canonical dtypes for anything this pipeline writes to parquet.

Sites store timestamps differently: microsecond or nanosecond precision, labelled
UTC (UCMC, NU), or unlabelled local time (RUSH). Every saved file is meant to be
readable on its own and to concatenate across sites, so every frame is
normalised on the way out:

  * datetimes  -> the same instant, labelled UTC, microsecond precision. A
                  labelled column is converted; an unlabelled one is read as the
                  site's local time (config timezone) and then converted.
  * dates      -> local midnight of that date, as above (NU stores birth_date as
                  a date, UCMC and RUSH as a timestamp).
  * integers   -> Int64, since width varies by site.
  * durations  -> microsecond precision.

Before 2026-10-07 the timezone label was dropped without converting, so UCMC and
NU files held UTC wall-clock times and RUSH files held Chicago wall-clock times,
with nothing in the file to tell them apart.
"""
from __future__ import annotations

import polars as pl

CANONICAL_TIME_UNIT = "us"
SAVED_DATETIME = pl.Datetime(CANONICAL_TIME_UNIT, "UTC")


def normalize_datetimes(df: pl.DataFrame, site_tz: str) -> pl.DataFrame:
    """Return `df` with temporal and integer columns at canonical types.

    `site_tz` is the hospital's timezone, used only for unlabelled columns.
    """
    casts = []
    for name, dt in df.schema.items():
        e = pl.col(name)
        if dt == pl.Date:
            e = e.cast(pl.Datetime(CANONICAL_TIME_UNIT))
            dt = pl.Datetime(CANONICAL_TIME_UNIT)
        if isinstance(dt, pl.Datetime):
            if dt.time_zone is None:
                e = e.dt.replace_time_zone(site_tz, ambiguous="earliest", non_existent="null")
            casts.append(e.dt.convert_time_zone("UTC").cast(SAVED_DATETIME).alias(name))
        elif dt in (pl.Int8, pl.Int16, pl.Int32, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
            casts.append(e.cast(pl.Int64).alias(name))
        elif isinstance(dt, pl.Duration) and dt.time_unit != CANONICAL_TIME_UNIT:
            casts.append(e.cast(pl.Duration(CANONICAL_TIME_UNIT)).alias(name))
    return df.with_columns(casts) if casts else df


def write_parquet(df: pl.DataFrame, path, site_tz: str) -> pl.DataFrame:
    """Normalise, then write. Use everywhere instead of df.write_parquet()."""
    out = normalize_datetimes(df, site_tz)
    out.write_parquet(str(path))
    return out


def align_time_zone(col: str, dtype, target) -> pl.Expr:
    """`col` relabelled to the timezone of `target`, keeping its wall-clock reading.

    For subtracting two timestamp columns a site stores under different timezone
    labels, most often a naive birth_date and an aware death_dttm, which polars
    refuses. A Date column, or one already in the same zone, is returned as is.
    """
    e = pl.col(col)
    if (isinstance(dtype, pl.Datetime) and isinstance(target, pl.Datetime)
            and dtype.time_zone != target.time_zone):
        e = e.dt.replace_time_zone(target.time_zone)
    return e


def hours_between(a: str, b: str, schema, site_tz: str) -> pl.Expr:
    """Elapsed hours from `b` to `a`, as time that passed, not as wall-clock arithmetic.

    Labelled columns are compared as instants, whatever their labels or time
    units. Unlabelled columns are read as the site's local time and placed in
    that zone first, so an interval across a daylight-saving change is right:
    01:30 to 03:30 on the spring-forward night is one hour, not two. A reading
    in the repeated hour at fall-back is taken as its first occurrence; one in
    the skipped hour at spring-forward does not exist and gives null.
    """
    def instant(name: str) -> pl.Expr:
        e = pl.col(name)
        if schema[name].time_zone is None:
            e = e.dt.replace_time_zone(site_tz, ambiguous="earliest", non_existent="null")
        return e.dt.convert_time_zone("UTC").cast(SAVED_DATETIME)
    return (instant(a) - instant(b)).dt.total_seconds() / 3600
