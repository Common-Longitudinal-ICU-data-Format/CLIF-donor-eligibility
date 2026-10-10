"""Tests for utils.criteria: the ICD-10 range predicate."""
from __future__ import annotations

import re
from pathlib import Path

import duckdb
import pytest

from utils.criteria import icd_range_sql

REPO = Path(__file__).resolve().parents[1]

# The regexes step 01 used before the ranges were read from the YAML. The new
# predicate must select exactly the same codes.
OLD = {
    ("I20", "I25"): r"^i2[0-5]\w*$",
    ("I60", "I69"): r"^i6[0-9]\w*$",
    ("V01", "Y89"): r"^(v0[1-9]|v[1-9]\d|w\d{2}|x\d{2}|y[0-8]\d)\w*$",
}


def _codes() -> list[str]:
    out = ["", "i2", "123", "i1a", "i5a", "c7a", "i1a0", "y8", "v0"]
    for letter in "abcdefghijklmnopqrstuvwxyz":
        for n in range(100):
            base = f"{letter}{n:02d}"
            out += [base, base + "9", base + "x1a"]
    return out


def _matches(predicate: str) -> set[str]:
    con = duckdb.connect()
    con.execute("CREATE TABLE t(dx_norm VARCHAR)")
    con.executemany("INSERT INTO t VALUES (?)", [(c,) for c in _codes()])
    return {r[0] for r in con.execute(f"SELECT dx_norm FROM t WHERE {predicate}").fetchall()}


@pytest.mark.parametrize("bounds", list(OLD))
def test_icd_range_selects_same_codes_as_old_regex(bounds):
    lo, hi = bounds
    new = _matches(icd_range_sql("dx_norm", lo, hi))
    old = _matches(f"REGEXP_MATCHES(dx_norm, '{OLD[bounds]}')")
    assert new == old
    assert len(new) > 0


def test_external_cause_range_spans_w_and_x_blocks():
    got = _matches(icd_range_sql("dx_norm", "V01", "Y89"))
    for code in ("v01", "v899", "w19", "x42", "x95x1a", "y89"):
        assert code in got
    for code in ("v00", "y90", "i26", "i70", "u07"):
        assert code not in got


@pytest.mark.parametrize("lo,hi", [("I2", "I25"), ("I20", "I2X"), ("I25", "I20"), ("", "I25")])
def test_icd_range_rejects_malformed_or_reversed_bounds(lo, hi):
    with pytest.raises(SystemExit):
        icd_range_sql("dx_norm", lo, hi)
