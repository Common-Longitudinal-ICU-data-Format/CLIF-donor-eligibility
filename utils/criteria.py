"""Access to config/donor_criteria.yaml that records which keys were read.

The YAML is documented as the single source of truth for every threshold. That
is only true if the code reads each value. Every lookup therefore goes through
`Criteria.get()`, and `Criteria.unused()` lists the leaf keys nothing asked for.
Step 01 stops the run if a key under a definition section was declared but never
applied, unless it is listed in DECLARED_NOT_APPLIED below.
"""
from __future__ import annotations

import operator
import re
from pathlib import Path

import yaml

# Comparison operators the YAML may name (clif_donor.age_operator).
OPERATORS = {"<=": operator.le, "<": operator.lt}

# Keys that are in the YAML but that the code knowingly does NOT apply. Listed
# here, and printed on every run, so the gap is visible instead of silent.
# Removing an entry requires implementing it.
DECLARED_NOT_APPLIED = {
    "clif_donor.contraindications.use_poa":
        "contraindication codes count at any diagnosis position regardless of "
        "present-on-admission; whether POA should be required is an open PI decision",
    "clif_donor.sepsis_window_hours":
        "no time-windowed sepsis flag exists; icd10_sepsis is diagnosis-code based "
        "and is reporting only",
}

_ICD3 = re.compile(r"^[A-Za-z][0-9]{2}$")


class Criteria:
    """donor_criteria.yaml with dotted-path lookups that record what was read."""

    def __init__(self, source: Path | str | dict):
        self._d = source if isinstance(source, dict) else yaml.safe_load(Path(source).read_text())
        self._used: set[str] = set()

    def get(self, dotted: str):
        """Value at e.g. 'clif_donor.kidney.creatinine_max'. Missing key stops the run."""
        node = self._d
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                raise SystemExit(f"config/donor_criteria.yaml: '{dotted}' is missing")
            node = node[part]
        self._used |= self._leaves_under(dotted, node)
        return node

    @staticmethod
    def _leaves_under(prefix: str, node) -> set[str]:
        if not isinstance(node, dict):
            return {prefix}
        out: set[str] = set()
        for k, v in node.items():
            out |= Criteria._leaves_under(f"{prefix}.{k}", v)
        return out

    def leaves(self, *sections: str) -> set[str]:
        out: set[str] = set()
        for s in sections:
            if s in self._d:
                out |= self._leaves_under(s, self._d[s])
        return out

    def unused(self, *sections: str) -> list[str]:
        """Leaf keys under `sections` that no get() call has reached."""
        return sorted(self.leaves(*sections) - self._used)


def icd_range_sql(col: str, lo: str, hi: str) -> str:
    """SQL predicate: `col` falls in the ICD-10 category range lo..hi inclusive.

    `col` must already be lower-case with punctuation stripped. Bounds are
    three-character categories (a letter and two digits), e.g. 'I20', 'Y89'.
    Fixed-width letter+digits compare lexicographically in numeric order, so a
    range that spans letters (V01-Y89) covers the W and X blocks without
    enumerating them.
    """
    for v in (lo, hi):
        if not _ICD3.match(str(v)):
            raise SystemExit(
                f"config/donor_criteria.yaml: ICD-10 range bound '{v}' must be a letter "
                f"and two digits, e.g. I20")
    lo, hi = str(lo).lower(), str(hi).lower()
    if lo > hi:
        raise SystemExit(f"config/donor_criteria.yaml: ICD-10 range {lo}-{hi} is reversed")
    return (f"(REGEXP_MATCHES({col}, '^[a-z][0-9]{{2}}') "
            f"AND SUBSTR({col}, 1, 3) BETWEEN '{lo}' AND '{hi}')")
