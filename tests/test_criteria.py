"""Tests for utils.criteria: tracked access to donor_criteria.yaml and ICD ranges."""
from __future__ import annotations

import re
from pathlib import Path

import duckdb
import pytest

from utils.criteria import DECLARED_NOT_APPLIED, OPERATORS, Criteria, icd_range_sql

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


def test_get_returns_value_and_records_use():
    c = Criteria({"clif_donor": {"bmi_max": 50, "kidney": {"creatinine_max": 4.0}}})
    assert c.get("clif_donor.kidney.creatinine_max") == 4.0
    assert c.unused("clif_donor") == ["clif_donor.bmi_max"]
    assert c.get("clif_donor.bmi_max") == 50
    assert c.unused("clif_donor") == []


def test_get_of_a_section_marks_every_leaf_below_it():
    c = Criteria({"calc": {"cause": {"a": ["I20", "I25"], "b": ["I60", "I69"]}, "age": 75}})
    assert c.get("calc.cause") == {"a": ["I20", "I25"], "b": ["I60", "I69"]}
    assert c.unused("calc") == ["calc.age"]


def test_missing_key_stops_the_run_with_the_key_named():
    c = Criteria({"clif_donor": {"bmi_max": 50}})
    with pytest.raises(SystemExit) as e:
        c.get("clif_donor.kidney.creatinine_max")
    assert "clif_donor.kidney.creatinine_max" in str(e.value)


def test_operators_cover_the_two_inequalities_the_yaml_may_name():
    assert OPERATORS["<="](75, 75) and not OPERATORS["<"](75, 75)


def test_every_definition_key_in_the_real_yaml_is_referenced_by_step_01():
    """The YAML is only the source of truth if step 01 asks for each key.

    Static twin of the runtime check at the end of step 01: every leaf under the
    three definition sections must be named in the script (directly or through
    a parent section), or be listed as knowingly not applied.
    """
    crit = Criteria(REPO / "config/donor_criteria.yaml")
    src = (REPO / "code/01_cohort_and_definitions.py").read_text()
    referenced = set(re.findall(r"""(?:CRIT\.get|_num)\(\s*["']([a-z0-9_.]+)["']""", src))
    orphans = []
    for leaf in sorted(crit.leaves("clif_donor", "calc", "ventilated_patient")):
        parts = leaf.split(".")
        parents = {".".join(parts[:i]) for i in range(1, len(parts) + 1)}
        if not (parents & referenced) and leaf not in DECLARED_NOT_APPLIED:
            orphans.append(leaf)
    assert orphans == [], f"declared in donor_criteria.yaml but never read: {orphans}"
