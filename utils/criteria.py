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
# Keys in the YAML that the code knowingly does not apply, with the reason. Each
# is printed on every run. Empty since use_poa was settled on 2026-10-07.
DECLARED_NOT_APPLIED: dict[str, str] = {}

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


# ── contraindication list (utils/icd10_contraindications.csv) ────────────────
# One row per code range. A diagnosis code matches a row when its first
# len(code_start) characters fall in code_start..code_end, so 'C30'-'C39' covers
# C34.11 without listing it. Only rows with exclude = yes are applied; the
# 'no' rows record what was deliberately left out, and why, so a decision can
# be reversed by changing one cell.
CONTRA_COLUMNS = ["code_start", "code_end", "description", "category",
                  "exclude", "source", "note"]
_CODE = re.compile(r"^[a-z][0-9a-z]{2,6}$")


def norm_code(code) -> str:
    """Lower-case, punctuation and whitespace stripped: 'C34.11' -> 'c3411'."""
    return re.sub(r"[^0-9a-z]", "", str(code).lower())


def load_contraindications(path: Path | str) -> list[dict]:
    """Rows of the contraindication list, validated. Bad file stops the run."""
    import csv
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != CONTRA_COLUMNS:
            raise SystemExit(f"{path.name}: columns must be {CONTRA_COLUMNS}, "
                             f"got {reader.fieldnames}")
        rows = list(reader)
    for i, r in enumerate(rows, start=2):
        lo, hi = norm_code(r["code_start"]), norm_code(r["code_end"])
        if not (_CODE.match(lo) and _CODE.match(hi)) or len(lo) != len(hi) or lo > hi:
            raise SystemExit(f"{path.name}:{i}: bad code range "
                             f"{r['code_start']}-{r['code_end']}")
        if r["exclude"] not in ("yes", "no"):
            raise SystemExit(f"{path.name}:{i}: exclude must be yes or no")
        if not r["source"].strip():
            raise SystemExit(f"{path.name}:{i}: source is empty")
        r["lo"], r["hi"] = lo, hi
    return rows


def code_range_sql(col: str, lo: str, hi: str) -> str:
    """SQL predicate: the first len(lo) characters of `col` fall in lo..hi."""
    return f"(SUBSTR({col}, 1, {len(lo)}) BETWEEN '{lo}' AND '{hi}')"


def contraindication_sql(col: str, rows: list[dict]) -> str:
    """SQL predicate over a normalised code column: matches any exclude = yes row."""
    parts = [code_range_sql(col, r["lo"], r["hi"]) for r in rows if r["exclude"] == "yes"]
    return "(" + " OR ".join(parts) + ")" if parts else "FALSE"


def contraindication_row(code, rows: list[dict]) -> dict | None:
    """The exclude = yes row a code matches, or None. Python twin of the SQL."""
    c = norm_code(code)
    for r in rows:
        if r["exclude"] == "yes" and r["lo"] <= c[:len(r["lo"])] <= r["hi"]:
            return r
    return None
