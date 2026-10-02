"""Setup checks and SQL predicates that need schemas only, never patient rows."""
from __future__ import annotations

_AWARE = "TIMESTAMP WITH TIME ZONE"


def mixed_tz_awareness(types: dict[str, str]) -> list[str]:
    """Problems with the timestamp columns the pipeline does arithmetic on.

    `types` maps 'table.column' to its DuckDB type. Every 48-hour window is a
    subtraction between two of these columns. DuckDB subtracts a timezone-aware
    from a naive timestamp without complaint, silently applying the session
    timezone offset, which moves patients across the window edge. So the columns
    must be all aware or all naive, and all actually timestamps.
    """
    problems: list[str] = []
    not_ts = {c: t for c, t in types.items() if not t.upper().startswith("TIMESTAMP")}
    for c, t in sorted(not_ts.items()):
        problems.append(f"{c} is {t}, not a timestamp")
    ts = {c: t.upper() for c, t in types.items() if c not in not_ts}
    aware = sorted(c for c, t in ts.items() if t == _AWARE)
    naive = sorted(c for c, t in ts.items() if t != _AWARE)
    if aware and naive:
        minority, kind = (naive, "timezone-naive") if len(naive) <= len(aware) else (aware, "timezone-aware")
        other = "timezone-aware" if kind == "timezone-naive" else "timezone-naive"
        problems.append(
            f"timestamp columns mix timezone-aware and naive: {', '.join(minority)} "
            f"{'is' if len(minority) == 1 else 'are'} {kind} while the other "
            f"{max(len(aware), len(naive))} {'is' if max(len(aware), len(naive)) == 1 else 'are'} {other}. "
            f"Every 48-hour window would shift by the UTC offset. Re-export so all "
            f"*_dttm columns share one convention (CLIF specifies UTC)")
    return problems


def usable_number_sql(col: str) -> str:
    """SQL predicate: `col` holds a real number.

    NULL and NaN both mean "no value". An extract written by polars, R or Spark
    can carry NaN where another carries NULL, and DuckDB sorts NaN above every
    number, so a NaN at the latest timestamp would win a "last value" ranking
    and read back as missing. TRY_CAST also drops text results in a column a
    site stores as text.
    """
    return (f"(TRY_CAST({col} AS DOUBLE) IS NOT NULL "
            f"AND NOT isnan(TRY_CAST({col} AS DOUBLE)))")
