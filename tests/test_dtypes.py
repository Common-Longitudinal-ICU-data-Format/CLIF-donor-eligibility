"""Tests for utils.dtypes.align_time_zone."""
from __future__ import annotations

from datetime import date, datetime, timezone

import polars as pl
import pytest

from utils.dtypes import align_time_zone


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
