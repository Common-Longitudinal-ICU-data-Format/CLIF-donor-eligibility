"""Tests for utils.dtypes.align_time_zone."""
from __future__ import annotations

from datetime import date, datetime, timezone

import polars as pl
import pytest

from utils.dtypes import align_time_zone, hours_between


def test_naive_birth_date_can_be_subtracted_from_an_aware_death_time():
    df = pl.DataFrame({"birth": [datetime(1950, 1, 1)],
                       "death": [datetime(2020, 1, 1, 12, tzinfo=timezone.utc)]})
    with pytest.raises(Exception):                      # what polars does unaided
        df.select(pl.col("death") - pl.col("birth"))
    birth = align_time_zone("birth", df.schema["birth"], df.schema["death"])
    assert df.select((pl.col("death") - birth).dt.total_days()).item() == 25567


def test_columns_already_in_the_same_zone_are_untouched():
    df = pl.DataFrame({"birth": [datetime(1950, 1, 1, tzinfo=timezone.utc)],
                       "death": [datetime(2020, 1, 1, 12, tzinfo=timezone.utc)]})
    birth = align_time_zone("birth", df.schema["birth"], df.schema["death"])
    assert df.select(birth).equals(df.select("birth"))


def test_a_date_column_is_left_as_it_is():
    df = pl.DataFrame({"birth": [date(1950, 1, 1)],
                       "death": [datetime(2020, 1, 1, 12, tzinfo=timezone.utc)]})
    birth = align_time_zone("birth", df.schema["birth"], df.schema["death"])
    assert df.select(birth).schema["birth"] == pl.Date


# ── normalize_datetimes: every saved timestamp is the same instant, labelled UTC ──
from utils.dtypes import normalize_datetimes

UTC_US = pl.Datetime("us", "UTC")


def _one(value, dtype, tz="US/Central"):
    return normalize_datetimes(pl.DataFrame({"t": [value]}, schema={"t": dtype}), tz)


def test_a_utc_column_keeps_its_instant_and_its_label():
    v = datetime(2024, 5, 3, 20, 0, tzinfo=timezone.utc)
    out = _one(v, pl.Datetime("ns", "UTC"))
    assert out.schema["t"] == UTC_US and out["t"][0] == v


def test_a_column_labelled_in_another_zone_becomes_the_same_instant_in_utc():
    out = normalize_datetimes(pl.DataFrame({"t": [datetime(2024, 5, 3, 20, 0, tzinfo=timezone.utc)]},
                                           schema={"t": pl.Datetime("us", "UTC")})
                              .with_columns(pl.col("t").dt.convert_time_zone("America/Chicago")), "US/Central")
    assert out.schema["t"] == UTC_US and out["t"][0] == datetime(2024, 5, 3, 20, 0, tzinfo=timezone.utc)


def test_an_unlabelled_column_is_read_as_the_sites_local_time():
    out = _one(datetime(2024, 5, 3, 15, 0), pl.Datetime("ns"))          # 15:00 in Chicago (CDT)
    assert out.schema["t"] == UTC_US and out["t"][0] == datetime(2024, 5, 3, 20, 0, tzinfo=timezone.utc)


def test_a_date_becomes_local_midnight():
    out = _one(date(2024, 1, 3), pl.Date)                                # CST, UTC-6
    assert out.schema["t"] == UTC_US and out["t"][0] == datetime(2024, 1, 3, 6, 0, tzinfo=timezone.utc)


def test_sites_storing_the_same_moment_differently_concatenate_and_agree():
    moment = datetime(2024, 5, 3, 20, 0, tzinfo=timezone.utc)
    ucmc = _one(moment, pl.Datetime("us", "UTC"))
    rush = _one(datetime(2024, 5, 3, 15, 0), pl.Datetime("ns"))
    both = pl.concat([ucmc, rush])
    assert both["t"].to_list() == [moment, moment]


def test_hours_between_counts_elapsed_time_across_daylight_saving():
    # 2024-03-10 is the spring-forward night in Chicago: 01:30 to 03:30 is one real hour.
    naive = pl.DataFrame({"a": [datetime(2024, 3, 10, 3, 30)], "b": [datetime(2024, 3, 10, 1, 30)]})
    assert naive.select(hours_between("a", "b", naive.schema, "America/Chicago")).item() == 1.0
    assert naive.select((pl.col("a") - pl.col("b")).dt.total_seconds() / 3600).item() == 2.0  # wall clock
    # Labelled columns are instants already: these two are two hours apart.
    aware = naive.with_columns(pl.col("a", "b").dt.replace_time_zone("UTC"))
    assert aware.select(hours_between("a", "b", aware.schema, "America/Chicago")).item() == 2.0


def test_hours_between_at_the_fall_back_and_skipped_hours():
    # 2024-11-03: 01:30 happens twice in Chicago; the first occurrence is used.
    df = pl.DataFrame({"a": [datetime(2024, 11, 3, 3, 0)], "b": [datetime(2024, 11, 3, 1, 30)]})
    assert df.select(hours_between("a", "b", df.schema, "America/Chicago")).item() == 2.5
    # 2024-03-10 02:30 does not exist in Chicago, so the interval is null.
    df = pl.DataFrame({"a": [datetime(2024, 3, 10, 2, 30)], "b": [datetime(2024, 3, 10, 1, 0)]})
    assert df.select(hours_between("a", "b", df.schema, "America/Chicago")).item() is None
