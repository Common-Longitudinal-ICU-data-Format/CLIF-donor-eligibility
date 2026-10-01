"""Canonical dtypes for anything this pipeline writes to parquet.

Sites store timestamps differently — microsecond vs nanosecond precision, some
timezone-aware in local time, some naive. Polars treats each variant as a
distinct type, so concatenating two sites' cohorts fails with

    SchemaError: type Datetime('us') is incompatible with Datetime('ns','UTC')

and any aggregation across sites breaks on the first mismatch. Rather than have
every consumer work around it, every frame is normalised on the way out:

  * dates      -> Datetime, because the same field arrives as Date at one site
                  and Datetime at another (NU stores birth_date as Date, UCMC and
                  RUSH as Datetime) and polars will not concatenate the two.
  * integers   -> Int64, since width varies by site (age_at_admission is Int32 at
                  RUSH, Int64 elsewhere).
  * datetimes  -> microsecond precision, timezone-naive **in the site's own
                  local time**. Local time is what the clinical logic already
                  uses (a 48-hour window before death is a wall-clock window),
                  and each site's config declares its timezone, so converting to
                  UTC here would silently shift the timestamps the cascade
                  already keyed on.
  * durations  -> microsecond precision, same reason.

This is a storage-layer normalisation only; it does not move any instant in time.
"""
from __future__ import annotations

import polars as pl

CANONICAL_TIME_UNIT = "us"


def normalize_datetimes(df: pl.DataFrame) -> pl.DataFrame:
    """Return `df` with temporal and integer columns at canonical types."""
    casts = []
    for name, dt in df.schema.items():
        if dt == pl.Date:
            casts.append(pl.col(name).cast(pl.Datetime(CANONICAL_TIME_UNIT)).alias(name))
        elif dt in (pl.Int8, pl.Int16, pl.Int32, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
            casts.append(pl.col(name).cast(pl.Int64).alias(name))
        elif isinstance(dt, pl.Datetime):
            expr = pl.col(name)
            if dt.time_zone is not None:
                # drop the tz label without shifting the wall-clock reading
                expr = expr.dt.replace_time_zone(None)
            casts.append(expr.cast(pl.Datetime(CANONICAL_TIME_UNIT)).alias(name))
        elif isinstance(dt, pl.Duration) and dt.time_unit != CANONICAL_TIME_UNIT:
            casts.append(pl.col(name).cast(pl.Duration(CANONICAL_TIME_UNIT)).alias(name))
    return df.with_columns(casts) if casts else df


def write_parquet(df: pl.DataFrame, path) -> pl.DataFrame:
    """Normalise, then write. Use everywhere instead of df.write_parquet()."""
    out = normalize_datetimes(df)
    out.write_parquet(str(path))
    return out
