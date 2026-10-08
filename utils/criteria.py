"""ICD-10 code helpers for the CALC cause ranges and the contraindication list."""
from __future__ import annotations

import re
from pathlib import Path

_ICD3 = re.compile(r"^[A-Za-z][0-9]{2}$")


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
# The ICD-10-CM codes that exclude a patient, one prefix per row: 'C34' covers
# C34.11 without listing it. `source` says why each is on the list: OPTN 1.2
# (OPTN eligible-death definition), CMS-3380-P (the 2019 proposed rule's list),
# or study decision. What is deliberately not excluded, and why, is in
# guides/contraindications.md.
_PREFIX = re.compile(r"^[a-z][0-9a-z]{2,6}$")


def norm_code(code) -> str:
    """Lower-case, punctuation and whitespace stripped: 'C34.11' -> 'c3411'."""
    return re.sub(r"[^0-9a-z]", "", str(code).lower())


def load_contraindications(path: Path | str) -> dict[str, str]:
    """Normalised code prefix -> description. A malformed file stops the run."""
    import csv
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["code_prefix", "description", "source"]:
            raise SystemExit(f"{path.name}: columns must be code_prefix, description, source; "
                             f"got {reader.fieldnames}")
        out: dict[str, str] = {}
        for i, r in enumerate(reader, start=2):
            p = norm_code(r["code_prefix"])
            if not _PREFIX.match(p) or p in out:
                raise SystemExit(f"{path.name}:{i}: bad or repeated code '{r['code_prefix']}'")
            if not (r["source"] or "").strip():
                raise SystemExit(f"{path.name}:{i}: source is empty")
            out[p] = r["description"]
    return out


def contraindication_sql(col: str, prefixes) -> str:
    """SQL predicate over a normalised code column: starts with any prefix."""
    return "(" + " OR ".join(f"{col} LIKE '{p}%'" for p in prefixes) + ")" if prefixes else "FALSE"


def contraindication_prefix(code, prefixes) -> str | None:
    """The prefix a code starts with, or None. Python twin of the SQL."""
    c = norm_code(code)
    return next((p for p in prefixes if c.startswith(p)), None)
