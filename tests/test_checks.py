"""Tests for utils.checks: setup checks that need no patient data."""
from __future__ import annotations

from utils.checks import mixed_tz_awareness


def test_all_aware_is_fine():
    assert mixed_tz_awareness({"patient.death_dttm": "TIMESTAMP WITH TIME ZONE",
                               "labs.lab_collect_dttm": "TIMESTAMP WITH TIME ZONE"}) == []


def test_all_naive_is_fine():
    assert mixed_tz_awareness({"patient.death_dttm": "TIMESTAMP",
                               "vitals.recorded_dttm": "TIMESTAMP_NS"}) == []


def test_mixed_names_the_minority_columns():
    problems = mixed_tz_awareness({
        "patient.death_dttm": "TIMESTAMP WITH TIME ZONE",
        "labs.lab_collect_dttm": "TIMESTAMP WITH TIME ZONE",
        "respiratory_support.recorded_dttm": "TIMESTAMP",
    })
    assert len(problems) == 1
    assert "respiratory_support.recorded_dttm" in problems[0]
    assert "patient.death_dttm" not in problems[0]


def test_non_timestamp_dttm_column_is_reported():
    problems = mixed_tz_awareness({"patient.death_dttm": "VARCHAR",
                                   "labs.lab_collect_dttm": "TIMESTAMP WITH TIME ZONE"})
    assert len(problems) == 1 and "patient.death_dttm" in problems[0] and "VARCHAR" in problems[0]
