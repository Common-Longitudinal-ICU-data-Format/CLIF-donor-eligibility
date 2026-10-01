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


# ── usable_number_sql: what may count as a "last value" ──────────────────────
import duckdb

from utils.checks import usable_number_sql


def _ids(con, col="x"):
    return [r[0] for r in con.execute(
        f"SELECT id FROM t WHERE {usable_number_sql(col)} ORDER BY id").fetchall()]


def test_usable_number_rejects_null_and_nan():
    con = duckdb.connect()
    con.execute("CREATE TABLE t AS SELECT * FROM (VALUES (1, 1.2::DOUBLE), (2, NULL), "
                "(3, 'NaN'::DOUBLE), (4, 0.0::DOUBLE)) v(id, x)")
    assert _ids(con) == [1, 4]


def test_usable_number_works_on_integer_and_text_columns():
    con = duckdb.connect()
    con.execute("CREATE TABLE t AS SELECT * FROM (VALUES (1, 5), (2, NULL)) v(id, x)")
    assert _ids(con) == [1]
    con.execute("DROP TABLE t")
    con.execute("CREATE TABLE t AS SELECT * FROM (VALUES (1, '1.5'), (2, 'HEMOLYZED'), (3, NULL)) v(id, x)")
    assert _ids(con) == [1]


def test_a_later_nan_does_not_win_the_last_value_ranking():
    """A NaN at the latest timestamp ranks first under DESC and reads back as null."""
    con = duckdb.connect()
    con.execute("CREATE TABLE t AS SELECT * FROM (VALUES (1, TIMESTAMP '2024-01-01 01:00', 1.2::DOUBLE), "
                "(1, TIMESTAMP '2024-01-01 05:00', 'NaN'::DOUBLE)) v(id, ts, x)")
    q = ("SELECT x FROM (SELECT x, ROW_NUMBER() OVER (PARTITION BY id ORDER BY ts DESC, x DESC) rn "
         "FROM t WHERE {w}) WHERE rn = 1")
    assert str(con.execute(q.format(w="x IS NOT NULL")).fetchone()[0]) == "nan"      # the defect
    assert con.execute(q.format(w=usable_number_sql("x"))).fetchone()[0] == 1.2
