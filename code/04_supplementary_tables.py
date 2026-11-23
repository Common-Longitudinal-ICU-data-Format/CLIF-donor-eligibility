#!/usr/bin/env python
"""Supplementary tables: manuscript rows steps 01-03 did not collect. One run per site, after them.

Reads output/intermediate_phi/<site>/ (as left by steps 01-02), the CLIF tables and the returned bundle.
Writes aggregate CSVs to <site>_upload_to_box/supplementary/, only if it reproduces that bundle exactly.
    uv run python code/04_supplementary_tables.py
"""
from __future__ import annotations

import hashlib
import re
import shutil
import sys
import traceback
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import duckdb
import polars as pl
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from utils.criteria import DEFS, icd_range_sql, norm_code   # noqa: E402
from utils.dtypes import hours_between, normalize_datetimes   # noqa: E402
from utils.outlier_handler import outlier_range   # noqa: E402
from utils.progress import Progress   # noqa: E402

ALL, NEW = "all_decedents", "supplementary"
CHAPTERS = ("A00-B99 C00-D49 D50-D89 E00-E89 F01-F99 G00-G99 H00-H59 H60-H95 I00-I99 J00-J99 K00-K95 L00-L99 M00-M99 "
            "N00-N99 O00-O99 P00-P96 Q00-Q99 R00-R99 S00-T88 U00-U85 V00-Y99 Z00-Z99").split()
SOFA_PARTS = ["sofa_cv_97", "sofa_coag", "sofa_liver", "sofa_resp", "sofa_cns", "sofa_renal"]
DEF_NAMES = [d.replace(" ", "_").replace("-", "_") for d, _ in DEFS]
# Labels, columns and bin widths of code/02_build_tables.py, so its rows can be reproduced and checked.
T2_VALUES = [("Age at death", "age_at_death", 1), ("Weight [kg]", "last_weight_kg", 1),
             ("Height [cm]", "last_height_cm", 1), ("BMI", "bmi", .5),
             ("Terminal creatinine [mg/dL]", "creatinine_value", .1),
             ("Terminal total bilirubin [mg/dL]", "bilirubin_total_value", .1),
             ("Terminal AST [U/L]", "ast_value", 10), ("Terminal ALT [U/L]", "alt_value", 10),
             ("Terminal BUN [mg/dL]", "bun_value", 1), ("Terminal sodium [mmol/L]", "sodium_value", 1),
             ("Terminal GCS", "gcs_total_value", 1), ("Terminal RASS", "rass_value", 1),
             ("Hospital LOS [days]", "hospital_length_of_stay_days", .5),
             ("First ICU LOS [days]", "first_icu_los_days", .5), ("Last PaO2/FiO2 [mmHg]", "pf_ratio", 10)]
T2_FLAGS = [("HCV infection", "icd10_hcv"), ("Hypertension", "icd10_htn"), ("Diabetes", "icd10_dm"),
            ("History of CVA", "icd10_cva"), ("Ischemic heart disease", "icd10_ischemic"),
            ("Cerebrovascular disease", "icd10_cerebro"), ("External causes", "icd10_external"),
            ("Z52.9 donor of organs", "dx_z529"), ("Cancer", "icd10_cancer_only"), ("Admitted to an ICU", "ever_icu"),
            ("Invasive mechanical ventilation", "imv_48hr_expire"), ("Brain death (G93.82)", "icd10_brain_death")]
DTTM = {"patient": ["death_dttm"], "hospitalization": ["admission_dttm", "discharge_dttm"], "adt": ["in_dttm", "out_dttm"],
        "medication_admin_continuous": ["admin_dttm"], "medication_admin_intermittent": ["admin_dttm"],
        "position": ["recorded_dttm"], "labs": ["lab_result_dttm"], "vitals": ["recorded_dttm"],
        "respiratory_support": ["recorded_dttm"], "patient_assessments": ["recorded_dttm"]}
SOFA_TABLES = ("labs", "vitals", "respiratory_support", "patient_assessments", "medication_admin_continuous")
# the dose units clifpy converts; any other (mL/hr, units/min, none) it would read as mcg/min
MASS_RATE = ("COALESCE(regexp_matches(regexp_replace(lower(med_dose_unit), '\\s+', '', 'g'), "
             "'^(mcg|ug|µg|μg|mg|ng|g|grams?)/(kg/|lb/)?(m|min|minute|h|hr|hour)$'), false)")
BUNDLE = ("definition_counts.csv", "table_stats_raw.csv", "hospital_level_counts.csv")   # checked against, and hashed


# ── Helpers (tests/test_supplementary_tables.py) ─────────────────────────────

def in_window_sql(death: str, ts: str, hours: float) -> str:
    """`ts` in the `hours` before `death`, ends included; rows after death do not count (as utils/windows.py)."""
    return f"(EXTRACT(EPOCH FROM ({death} - {ts})) / 3600 BETWEEN 0 AND {float(hours)})"


def utc_sql(col: str, duck_type: str, site_tz: str) -> str:
    """`col` as an instant; a timezone-naive column is the site's wall clock."""
    return col if duck_type.upper() == "TIMESTAMP WITH TIME ZONE" else f"timezone('{site_tz}', CAST({col} AS TIMESTAMP))"


def tz_mismatched(types: dict[str, str], reference: str, if_date: str | None = None) -> list[str]:
    """Columns whose timezone awareness differs from the reference's; all of them when it is unusable.
    A DATE reference has no time of day, so step 01 took the time from `if_date`, which then is the reference."""
    kind = {c: t.upper() == "TIMESTAMP WITH TIME ZONE" if t.upper().startswith("TIMESTAMP") else None
            for c, t in types.items()}
    ref = kind.pop(reference, None)
    if ref is None and types.get(reference, "").upper() == "DATE":
        ref = kind.get(if_date)
    return sorted(c for c, k in kind.items() if ref is None or k != ref)


def icd_family_sql(col: str, ranges=(), prefixes=(), regex=()) -> str:
    """Predicate on a normalised code column: in any ICD-10 range, starting with any prefix, or matching any regex."""
    parts = ([icd_range_sql(col, lo, hi) for lo, hi in ranges] + [f"{col} LIKE '{norm_code(p)}%'" for p in prefixes]
             + [f"regexp_matches({col}, '{x}')" for x in regex])
    return "(" + " OR ".join(parts) + ")" if parts else "FALSE"


def calc_variant_sql(variant: dict, reading: str) -> str:
    """`ranges` and `prefixes` follow the reading; `principal_only_ranges` are principal under both."""
    follows = icd_family_sql("dx_norm", variant.get("ranges", ()), variant.get("prefixes", ()))
    fixed = icd_family_sql("dx_norm", variant.get("principal_only_ranges", ()))
    return f"(({follows} AND {'is_primary' if reading == 'principal' else 'TRUE'}) OR ({fixed} AND is_primary))"


def diagnosis_group_sql(groups: dict) -> dict[str, str]:
    """'dxg|<group>|<reading>' -> predicate: the named groups under both readings, the ICD-10 chapters at principal."""
    fam = {g: icd_family_sql("dx_norm", **v) for g, v in groups.items()}
    return ({f"dxg|{g}|any": s for g, s in fam.items()} | {f"dxg|{g}|principal": f"({s} AND is_primary)" for g, s in fam.items()}
            | {f"dxg|chapter_{c}|principal": f"({icd_range_sql('dx_norm', *c.split('-'))} AND is_primary)" for c in CHAPTERS})


def diagnosis_group_counts(full: pl.DataFrame, clif: pl.Expr, calc: pl.Expr) -> pl.DataFrame:
    """Decedents with each 'dxg|...' flag per CLIF-donor/CALC overlap group; the four groups partition all decedents."""
    several = pl.sum_horizontal([pl.col(c) for c in full.columns if c.startswith("dxg|chapter_")]) > 1    # one site flags many principals
    full = full.with_columns(several.alias("dxg|n_with_more_than_one_principal_chapter|principal"))
    ov = {"clif_only": clif & ~calc, "both": clif & calc, "calc_only": ~clif & calc, "neither": ~clif & ~calc, ALL: pl.lit(True)}
    return pl.DataFrame([{"group": c.split("|")[1], "reading": c.split("|")[2], "overlap_group": g,
                          "n": full.filter(e & pl.col(c)).height, "denom": full.filter(e).height}
                         for c in full.columns if c.startswith("dxg|") for g, e in ov.items()])


def icu_stays(adt: pl.DataFrame, max_out_year: int | None = None) -> pl.DataFrame:
    """First and last ICU row per patient by step 01's rule (ties at in_dttm: latest out; out before in: unknown)."""
    bad = pl.col("out_dttm") < pl.col("in_dttm")
    if max_out_year:      # not in step 01, so the first-ICU check runs without it
        bad = bad | (pl.col("out_dttm").dt.year() >= max_out_year)
    icu = (adt.with_columns(pl.when(bad).then(None).otherwise(pl.col("out_dttm")).alias("out_dttm"))
           .filter(pl.col("location_category").str.to_lowercase() == "icu"))
    picks = [icu.join(icu.group_by("patient_id").agg(edge.alias("_edge")), on="patient_id")
             .filter(pl.col("in_dttm") == pl.col("_edge")).group_by("patient_id")
             .agg(pl.col("_edge").first().alias(f"{name}_icu_in_dttm"), pl.col("out_dttm").max().alias(f"{name}_icu_out_dttm"))
             for name, edge in (("first", pl.col("in_dttm").min()), ("last", pl.col("in_dttm").max()))]
    return picks[0].join(picks[1], on="patient_id").sort("patient_id")


def bin_counts(s: pl.Series, width: float) -> list[tuple[float, int]]:
    """(bin_lo, n) per fixed-width bin, as code/02_build_tables.py bins."""
    g = (s.drop_nulls().cast(pl.Float64).to_frame("v")
         .with_columns(((pl.col("v") / width + 1e-9).floor() * width).round(6).alias("bin_lo"))
         .group_by("bin_lo").len().sort("bin_lo"))
    return [(float(b), int(n)) for b, n in g.iter_rows()]


def top_code_age(bins: pl.DataFrame, top: float = 90.0) -> pl.DataFrame:
    """One 'Age at death' bin for `top` and over: single years above 89 identify."""
    lo = pl.when((pl.col("variable") == "Age at death") & (pl.col("bin_lo") > top)).then(top).otherwise(pl.col("bin_lo"))
    return bins.with_columns(lo.alias("bin_lo")).group_by([c for c in bins.columns if c != "n"], maintain_order=True).agg(pl.col("n").sum())


def resolve_min_cell(override: int, bins_shipped: bool, masks_seen: list[int], default_n: int) -> int:
    """Masking threshold from the returned bundle: override, else its largest '<k', else the study's if bins were withheld."""
    return int(override) or (max(masks_seen) if masks_seen else 0 if bins_shipped else int(default_n))


def mask_counts(df: pl.DataFrame, cols: list[str], k: int, denom: str | None = None) -> pl.DataFrame:
    """1..k-1 written '<k'; with `denom`, a count within 1..k-1 of it written '>denom-k' (no subtraction)."""
    out = []
    for c in cols if k > 1 else []:
        e = pl.when(pl.col(c).is_between(1, k - 1)).then(pl.lit(f"<{k}"))
        if denom:
            e = e.when((pl.col(denom) - pl.col(c)).is_between(1, k - 1)).then(pl.format(">{}", pl.col(denom) - k))
        out.append(e.otherwise(pl.col(c).cast(pl.Utf8)).alias(c))
    return df.with_columns(out)


def mask_partner(df: pl.DataFrame, col: str, by: list, k: int) -> pl.DataFrame:
    """A breakdown with exactly one cell of 1..k-1 gives it away as total minus the rest: blank the next smallest too."""
    small, big = pl.col(col).is_between(1, k - 1), pl.when(pl.col(col) >= k).then(pl.col(col))
    return df if k <= 1 else df.with_columns(
        pl.when((small.sum().over(by) == 1) & (pl.col(col) == big.min().over(by))).then(None).otherwise(pl.col(col)).alias(col))


def shown(n: int, k: int) -> str:
    """A count in the shipped log or provenance: 1..k-1 as '<k'."""
    return f"<{k}" if 0 < n < k else f"{n:,}"


def hospitals_to_ship(has_small_cell: dict[str, bool], n_decedents: dict[str, int]) -> list[str]:
    """Hospitals with no small cell; a single withheld one is site minus the rest, so the smallest other goes too."""
    ok = sorted(h for h, small in has_small_cell.items() if not small)
    if len(has_small_cell) - len(ok) == 1 and ok:
        ok.remove(min(ok, key=lambda h: (n_decedents[h], h)))
    return ok


def cell_diff(mine, shipped) -> float:
    """|mine - shipped| for a shipped CSV cell read as text; a shipped '<k' agrees with any count of 1..k-1."""
    shipped = None if shipped in (None, "") else str(shipped)
    if mine is None or shipped is None:
        return 0.0 if mine is None and shipped is None else float("inf")
    if shipped.startswith("<"):
        return 0.0 if 1 <= float(mine) < float(shipped[1:]) else float("inf")
    return abs(float(mine) - float(shipped))


def diagnostic_counts(full: pl.DataFrame, adt: pl.DataFrame, tz: str, y0: int, y1: int) -> pl.DataFrame:
    """Decedents per (check, stratum, year): the data questions put to sites, counted the same way everywhere."""
    lc = pl.col("location_category").str.to_lowercase()
    loc = adt.group_by("patient_id").agg(lc.eq("icu").any().alias("_icu"),
                                         pl.when(lc.n_unique() == 1).then(lc.first()).alias("_only"))
    d = full.join(loc, on="patient_id", how="left").with_columns(
        pl.col("final_death_dttm").dt.convert_time_zone(tz).dt.year().clip(y0, y1).alias("year"),
        pl.when(pl.col("_icu")).then(pl.lit("ever_icu")).when(pl.col("_only").is_in(["procedural", "ed", "ward"]))
        .then(pl.col("_only") + "_only").otherwise(pl.lit("other")).alias("_mix"),
        pl.col("n_principal").fill_null(0).clip(upper_bound=2).alias("_np"))
    checks = [   # (check, stratum, flag, also by year)
        *[("principal_dx_per_hospitalization", s, pl.col("_np") == i, False) for i, s in enumerate(("0", "1", "2+"))],
        ("no_rows", "no_diagnosis_rows", pl.col("n_dx").is_null(), True),
        ("no_rows", "no_adt_rows", pl.col("_icu").is_null(), True), ("no_rows", "no_icu_row", ~f("_icu"), True),
        *[("location_mix", s, pl.col("_mix") == s, False) for s in ("procedural_only", "ed_only", "ward_only", "ever_icu", "other")],
        ("external_cause_any", "V01-Y89 any position", f("ext_cause_any"), True),
        *[("death_time_quality", f"source={s}", pl.col("death_time_source") == s, True)
          for s in sorted(d["death_time_source"].drop_nulls().unique())]]
    groups = [("all", d)] + sorted((str(y), g) for (y,), g in d.group_by("year"))
    return pl.DataFrame([{"check": c, "stratum": s, "year": y, "n": g.filter(e).height, "denom": g.height}
                         for c, s, e, by_year in checks for y, g in groups[:None if by_year else 1]])


def last_location(adt: pl.DataFrame, site: str, fold: int = 11) -> pl.DataFrame:
    """Decedents per unit of their last ADT row (step 01's terminal location), so a category such as 'other' can be read.
    A unit with fewer than `fold` decedents is folded into one row per category."""
    last = adt.sort("in_dttm", descending=True, nulls_last=True).group_by("patient_id", maintain_order=True).agg(
        pl.col("location_category").first().str.to_lowercase(), pl.col("location_name").first().cast(pl.Utf8))
    n = last.group_by("location_category", "location_name").len("n")
    n = n.with_columns(pl.when(pl.col("n") >= fold).then("location_name").otherwise(pl.lit(f"(units under {fold})")))
    return (n.group_by("location_category", "location_name").agg(pl.col("n").sum())
            .sort(["n", "location_category", "location_name"], descending=[True, False, False], nulls_last=True)
            .select(pl.lit(site).alias("site"), "location_category", "location_name", "n"))


def write_sofa_shim(con: duckdb.DuckDBPyConnection, src, shim: Path, site_tz: str, log=print) -> list[str]:
    """Copy the rows clifpy's SOFA reads (stays and hours of `sofa_win` in `con`); return the unbounded inputs.
    Why: clifpy loads whole tables in pandas, matches categories by case, reads naive time two ways, bounds little."""
    from clifpy.utils.sofa_polars import (DEVICE_RANK_DICT, REQUIRED_ASSESSMENTS, REQUIRED_LABS,
                                          REQUIRED_MEDS, REQUIRED_VITALS)
    low = "lower(trim(CAST({} AS VARCHAR)))".format
    special = {"device_category": f"CASE {low('device_category')} " + " ".join(
                   f"WHEN '{k.lower()}' THEN '{k}'" for k in DEVICE_RANK_DICT) + " ELSE CAST(device_category AS VARCHAR) END",
               "fio2_set": "CASE WHEN fio2_set > 1 THEN fio2_set / 100 ELSE fio2_set END",   # a percentage, as step 01 reads it
               "categorical_value": "CAST(NULL AS VARCHAR)"}                                # folded into numerical_value below
    spec = {   # table: (timestamp, category column, categories kept, value column, {output column: type})
        "labs": ("lab_result_dttm", "lab_category", REQUIRED_LABS, "lab_value_numeric",
                 {"lab_category": "VARCHAR", "lab_value": "VARCHAR", "lab_value_numeric": "DOUBLE"}),
        "vitals": ("recorded_dttm", "vital_category", REQUIRED_VITALS, "vital_value",
                   {"vital_category": "VARCHAR", "vital_value": "DOUBLE"}),
        "patient_assessments": ("recorded_dttm", "assessment_category", REQUIRED_ASSESSMENTS, "numerical_value",
                                {"assessment_category": "VARCHAR", "numerical_value": "DOUBLE", "categorical_value": "VARCHAR"}),
        "respiratory_support": ("recorded_dttm", None, [], None,
                                {"device_category": "VARCHAR", "mode_category": "VARCHAR", "fio2_set": "DOUBLE",
                                 "lpm_set": "DOUBLE", "tidal_volume_set": "DOUBLE", "resp_rate_set": "DOUBLE"}),
        "medication_admin_continuous": ("admin_dttm", "med_category", REQUIRED_MEDS, "med_dose",
                                        {"med_category": "VARCHAR", "med_dose": "DOUBLE", "med_dose_unit": "VARCHAR"})}
    unbounded = []
    for table, (ts, cat, cats, val, cols) in spec.items():
        target = shim / f"clif_{table}.parquet"
        have = ({r[0]: r[1] for r in con.sql(f"DESCRIBE SELECT * FROM read_parquet('{src(table)}')").fetchall()}
                if Path(src(table)).is_file() else {})
        missing = [c for c in (ts, cat, val) if c and c not in have]
        if missing:    # an empty table of the right shape: its components go missing (clifpy stops with no SpO2 at all)
            log(f"  SOFA: clif_{table} " + ("is absent" if not have else "lacks " + ", ".join(missing))
                + ("; clifpy cannot score without it" if table == "vitals" else "; the components it feeds are missing"))
            con.sql(f"""COPY (SELECT CAST(NULL AS VARCHAR) AS hospitalization_id, CAST(NULL AS TIMESTAMPTZ) AS {ts},
                              {", ".join(f"CAST(NULL AS {ty}) AS {c}" for c, ty in cols.items())} LIMIT 0)
                        TO '{target}' (FORMAT parquet)""")
            continue
        v = f"TRY_CAST({val} AS DOUBLE)"
        if "categorical_value" in have:      # clifpy falls back on it; folded in so it is bounded too
            v = f"COALESCE({v}, TRY_CAST(categorical_value AS DOUBLE))"
        keep, bounds = ["TRUE"], {}
        for c in cats:
            try:
                bounds[c] = outlier_range(table, val, c)
            except KeyError:                 # no range in outlier_config.yaml: passed on, and reported
                unbounded.append(f"{c} ({table})")
        if cat:
            keep.append(f"{low(cat)} IN ({str(list(cats))[1:-1]})")
        if bounds:
            keep.append(f"CASE {low(cat)} " + " ".join(
                f"WHEN '{c}' THEN {v} BETWEEN {lo} AND {hi}" for c, (lo, hi) in bounds.items()) + " ELSE TRUE END")
        if val == "med_dose":                # a stop or pause row charted at dose 0 would score cardiovascular 3
            keep += [f"{v} > 0", MASS_RATE if "med_dose_unit" in have else "FALSE"]
        sel = {c: f"CAST(NULL AS {ty})" if c not in have else low(c) if c == cat else v if c == val
               else special.get(c) or (f"TRY_CAST({c} AS DOUBLE)" if ty == "DOUBLE" else f"CAST({c} AS VARCHAR)")
               for c, ty in cols.items()}
        instant, hid = utc_sql("t." + ts, have[ts], site_tz), "CAST(t.hospitalization_id AS VARCHAR)"
        always = f"{low(cat)} = 'weight_kg'" if table == "vitals" else "FALSE"   # the weight behind a dose can be older
        con.sql(f"""
            COPY (SELECT {hid} AS hospitalization_id, {instant} AS {ts}, {", ".join(f"{s} AS {c}" for c, s in sel.items())}
                  FROM read_parquet('{src(table)}') t
                  WHERE {" AND ".join(keep)} AND {hid} IN (SELECT hospitalization_id FROM sofa_win)
                    AND ({always} OR EXISTS (SELECT 1 FROM sofa_win w WHERE w.hospitalization_id = {hid}
                                             AND {instant} BETWEEN w.lo AND w.hi)))
            TO '{target}' (FORMAT parquet)""")
    return unbounded


def floor_unsupported_resp(s: pl.DataFrame, when: str) -> pl.DataFrame:
    """SOFA scores P/F < 200 without ventilatory support 2; clifpy 2.1.0 leaves it null. Flagged in <when>_resp_floor."""
    resp, hit = pl.col(f"{when}_sofa_resp"), (pl.col(f"{when}_sofa_resp").is_null() & (pl.col(f"{when}_p_f") < 200)).fill_null(False)
    return s.with_columns(hit.alias(f"{when}_resp_floor"), pl.when(hit).then(2.0).otherwise(resp).alias(f"{when}_sofa_resp"))


# ── Run state and self-check ─────────────────────────────────────────────────

def f(c: str) -> pl.Expr:
    return pl.col(c).fill_null(False)


def live(df: pl.DataFrame, c: str) -> bool:
    return c in df.columns and df[c].null_count() < df.height


def tbl(r, name: str) -> Path:
    return r.tables / f"clif_{name}.parquet"


def col_types(r, name: str) -> dict[str, str]:
    if not tbl(r, name).is_file():
        return {}
    return {x[0]: x[1] for x in r.con.sql(f"DESCRIBE SELECT * FROM read_parquet('{tbl(r, name)}')").fetchall()}


def shipped(r, name: str) -> pl.DataFrame:
    return pl.read_csv(r.final / name, infer_schema_length=0)      # as text, so a masked '<k' reads


def note(r, family: str, status: str, detail: str = "", diffs=()) -> None:
    r.checks.append({"site": r.site, "family": family, "status": status, "n_compared": len(diffs),
                     "n_different": sum(d > 1e-9 for d in diffs), "max_abs_diff": max(diffs, default=0.0), "detail": detail})


def check(r, family: str, diffs: list[float]) -> None:
    note(r, family, "fail" if any(d > 1e-9 for d in diffs) else "pass", diffs=diffs)
    r.log(f"  check {family:44s} {len(diffs):>6,} compared, max abs diff {max(diffs, default=0.0):g}")


def check_keyed(r, family: str, mine: pl.DataFrame, theirs: pl.DataFrame, keys: list[str], vals: list[str]) -> None:
    a = {tuple(str(x[k]) for k in keys): x for x in mine.iter_rows(named=True)}
    b = {tuple(str(x[k]) for k in keys): x for x in theirs.iter_rows(named=True)}
    check(r, family, [cell_diff(a[k][v], b[k][v]) for k in a.keys() & b.keys() for v in vals]
          + [float("inf")] * len(a.keys() ^ b.keys()))        # a key on one side only is a difference


def skip(r, output: str, why: str) -> None:
    note(r, output, "skipped", why)
    r.log(f"  SKIPPED {output}: {why}")


def local_error(r, where: str, e: Exception) -> str:
    """For the shipped log: type, place and the message with quoted text and digits masked, since a message can
    quote a data value. The full traceback stays in the local PHI folder."""
    fr, log = traceback.extract_tb(e.__traceback__)[-1], r.inter / "logs" / "04_sofa_error.txt"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(f"== {where}\n{''.join(traceback.format_exception(e))}\n")
    masked = re.sub(r"'[^']*'|\"[^\"]*\"|\d+", "#", " ".join(str(e).split()))[:300]
    return (f"{type(e).__name__} at {Path(fr.filename).name}:{fr.lineno} in {fr.name}: {masked} "
            f"[unmasked in output/intermediate_phi/{r.inter.name}/logs/{log.name}, local only]")


def unsure(r, *tables: str) -> str:
    """Why an output over these tables cannot be windowed against the death anchor; '' when it can."""
    hit = [c for c in r.off if c.split(".")[0] in tables]
    return "" if not hit else (", ".join(f"{c} is {r.dttm[c]}" for c in hit) + " while patient.death_dttm is "
                               + r.dttm.get("patient.death_dttm", "absent") + "; site time or UTC cannot be told apart")


# ── Inputs, and the bundle they must match ───────────────────────────────────

def load_inputs(r) -> None:
    rerun = "Re-run the pipeline (uv run python run_pipeline.py), return that bundle, then run this step."
    needed = [(r.inter / "cohort_with_definitions.parquet", "It is written by step 01. " + rerun),
              (r.inter / "encounter_block_members.parquet", "It is written by step 01. " + rerun),
              (r.inter / "cohort_with_variables.parquet",
               "It is written by step 02. Re-run step 02 (uv run python code/02_build_tables.py), then this step."),
              *[(r.final / name, "This step is checked against the bundle you returned: put that folder back at "
                 "the repository root and run again.") for name in BUNDLE],
              *[(tbl(r, t), "Point config.json tables_path at the CLIF tables steps 01-03 were run on.")
                for t in ("adt", "hospital_diagnosis")]]
    for path, fix in needed:
        if not path.is_file():
            raise SystemExit(f"{path.relative_to(REPO) if path.is_relative_to(REPO) else path.name} is missing. {fix}")
    cohort = pl.read_parquet(r.inter / "cohort_with_definitions.parquet")
    r.cv = pl.read_parquet(r.inter / "cohort_with_variables.parquet")
    r.members = (pl.read_parquet(r.inter / "encounter_block_members.parquet")
                 .with_columns(pl.col("hospitalization_id").cast(pl.Utf8))
                 .join(cohort.select("patient_id", "final_death_dttm"), on="patient_id", how="inner"))
    r.n = cohort.height
    flags = ["patient_id"] + [c for _, c in DEFS]
    mine = {"n_inpatient_deaths": r.n, "clif_donor": int(cohort["clif_eligible_donors"].sum()),
            "calc": int(cohort["calc_flag"].sum()),
            "ventilated_no_age_limit": int(cohort["ventilated_patient_no_age_limit"].sum())}
    theirs = shipped(r, "definition_counts.csv").row(0, named=True)
    bad = [k for k, v in mine.items() if cell_diff(v, theirs.get(k)) != 0]
    if cohort["patient_id"].n_unique() != r.n or bad or not r.cv.select(flags).equals(cohort.select(flags)):
        raise SystemExit(f"The local cohort does not match the bundle in {r.final.name} ("
                         + (", ".join(f"{k}: local {mine[k]:,} vs shipped {theirs.get(k)}" for k in bad)
                            or "cohort_with_variables.parquet is not from the same run") + "). " + rerun)
    r.bins_shipped = (r.final / "value_bins.csv").is_file()
    masks = [int(c[1:]) for name in ("table_stats_raw.csv", "value_bins.csv", "table2_characteristics.csv",
                                     "table3_clinical_care.csv") if (r.final / name).is_file()
             for col in shipped(r, name).iter_columns() for c in col.drop_nulls().to_list() if re.fullmatch(r"<\d+", c)]
    r.year_min = int(r.study["small_cell_suppression_n"])
    r.min_cell = resolve_min_cell(r.cfg["min_cell"], r.bins_shipped, masks, r.year_min)
    r.log(f"[{r.site}] cohort check passed: {r.n:,} decedents; " + ", ".join(f"{k}={shown(v, r.min_cell)}" for k, v in mine.items()))
    r.log("small cells: " + (f"counts below {r.min_cell} masked" if r.min_cell > 1 else "raw counts, as in the main bundle"))
    r.hashes = {name: hashlib.sha256((r.final / name).read_bytes()).hexdigest() for name in BUNDLE}
    for name, h in r.hashes.items():
        note(r, name, "sha256", h)
    r.con = duckdb.connect()
    r.con.sql("SET TimeZone='UTC'")
    if r.mem:      # config.json duckdb_memory_limit, e.g. "8GB"; DuckDB's own default is 80% of RAM
        r.con.sql(f"SET memory_limit='{r.mem}'")
    (r.inter / "duckdb_tmp").mkdir(exist_ok=True)
    r.con.sql(f"SET temp_directory='{r.inter / 'duckdb_tmp'}'")
    r.con.register("members", r.members.to_arrow())
    # Step 01 built the death anchor from patient.death_dttm. A column that is naive where that one is aware
    # (or the reverse) may hold site time or UTC, so outputs windowed on it are skipped, not guessed.
    r.dttm = {f"{t}.{c}": col_types(r, t)[c] for t, cs in DTTM.items() for c in cs if c in col_types(r, t)}
    r.off = tz_mismatched(r.dttm, "patient.death_dttm", if_date="hospitalization.discharge_dttm")
    r.log(f"timestamps: {len(r.dttm)} columns read, {len(r.off)} with a different timezone convention from death_dttm")


# ── Per-patient variables ────────────────────────────────────────────────────

def patient_variables(r):
    con, cv, tz = r.con, r.cv, r.tz
    add = cv.select("patient_id", "final_death_dttm")
    adt = con.sql(f"""
        SELECT m.patient_id, a.in_dttm, a.out_dttm, CAST(a.location_category AS VARCHAR) location_category,
               {"CAST(a.location_name AS VARCHAR)" if "location_name" in col_types(r, "adt") else "NULL::VARCHAR"} location_name
        FROM read_parquet('{tbl(r, "adt")}') a
        JOIN members m ON CAST(a.hospitalization_id AS VARCHAR) = m.hospitalization_id""").pl()
    # UCSF branch: morgue rows are dropped, as in step 01
    adt = adt.filter(~pl.col("location_name").str.to_lowercase().str.contains("morgue").fill_null(False))
    stays = icu_stays(adt).select("patient_id", "^first_.*$").join(      # last stay: an out_dttm from 2100 on is unknown too
        icu_stays(adt, max_out_year=2100).select("patient_id", "^last_.*$"), on="patient_id")
    stays = stays.with_columns(
        (hours_between("first_icu_out_dttm", "first_icu_in_dttm", stays.schema, tz) / 24).alias("_first_icu_los_days"),
        (hours_between("last_icu_out_dttm", "last_icu_in_dttm", stays.schema, tz) / 24).alias("last_icu_los_days"),
        (pl.col("first_icu_in_dttm") == pl.col("last_icu_in_dttm")).alias("single_icu_stay"))
    add = add.join(normalize_datetimes(stays, tz), on="patient_id", how="left")
    check(r, "first_icu_los_days (patient level)",
          [cell_diff(a, b) for a, b in zip(add["_first_icu_los_days"], cv["first_icu_los_days"])])
    if unsure(r, "adt"):
        skip(r, "Last ICU LOS closed at time of death", unsure(r, "adt"))
    else:      # a decedent who died in the ICU often has no out_dttm on that row
        add = add.with_columns(pl.coalesce("last_icu_out_dttm", pl.when(
            pl.col("final_death_dttm") >= pl.col("last_icu_in_dttm")).then("final_death_dttm")).alias("_last_end"))
        add = add.with_columns((hours_between("_last_end", "last_icu_in_dttm", add.schema, tz) / 24)
                               .alias("last_icu_los_closed_at_death_days"))
    r.mark("ICU stays")
    icd10 = "sys IN ('icd10','icd10cm')"
    variants = [f"{name}|{reading}" for name in r.cfg["calc_variants"] for reading in ("principal", "any")]
    dx_sql = {"hiv": icd_family_sql("dx_norm", prefixes=r.cfg["hiv_icd10_prefixes"]),
              "dx_z52x": icd_family_sql("dx_norm", prefixes=r.cfg["donor_family_icd10_prefixes"]),
              "ext_cause_any": icd_family_sql("dx_norm", ranges=[("V01", "Y89")]), **diagnosis_group_sql(r.cfg["diagnosis_groups"]),
              **{f"cause_{k}": calc_variant_sql(r.cfg["calc_variants"][k.split("|")[0]], k.split("|")[1]) for k in variants}}
    dx = con.sql(f"""
        WITH d AS (            -- normalised exactly as step 01 normalises it
            SELECT m.patient_id,
                   LOWER(REGEXP_REPLACE(CAST(diagnosis_code AS VARCHAR), '[^A-Za-z0-9]', '', 'g')) AS dx_norm,
                   REGEXP_REPLACE(LOWER(CAST(diagnosis_code_format AS VARCHAR)), '[^a-z0-9]', '', 'g') AS sys,
                   COALESCE(TRY_CAST(diagnosis_primary AS DOUBLE) = 1, false) AS is_primary,
                   COUNT(*) FILTER (WHERE TRY_CAST(diagnosis_primary AS DOUBLE) = 1)
                       OVER (PARTITION BY m.patient_id, hd.hospitalization_id) AS np      -- principal rows of one hospitalization
            FROM read_parquet('{tbl(r, "hospital_diagnosis")}') hd
            JOIN members m ON CAST(hd.hospitalization_id AS VARCHAR) = m.hospitalization_id)
        SELECT patient_id, {", ".join(f'BOOL_OR({icd10} AND {s}) AS "{k}"' for k, s in dx_sql.items())},
               COUNT(*) AS n_dx, MAX(np) AS n_principal
        FROM d GROUP BY 1""").pl()
    add = add.join(dx, on="patient_id", how="left").with_columns([f(k) for k in dx_sql])
    for reading, col in (("principal", "calc_cause_primary"), ("any", "calc_cause_any")):
        check(r, f"CALC cause (i) {reading} vs {col} (patient level)",
              [float(a != b) for a, b in zip(add[f"cause_i_current|{reading}"], cv[col])])
    r.mark("Diagnoses")
    exposures, groups = {}, r.crit["clinical_care"]["medications"]      # Table 3 column -> its twins
    not_given = ",".join(f"'{str(a).lower()}'" for a in r.cfg["mar_actions_not_given"])
    for table, suffix in (("medication_admin_continuous", "cont"), ("medication_admin_intermittent", "intermit"),
                          ("position", None)):
        ts = "admin_dttm" if suffix else "recorded_dttm"
        cols = {f"med_{g}_{suffix}": cats for g, cats in groups.items()} if suffix else {"prone_position": ["prone"]}
        have = col_types(r, table)
        if not have or not live(cv, next(iter(cols))):
            r.log(f"  {table}: not captured at this site; no supplementary rows for {', '.join(cols)}")
            continue
        # med_category is not trimmed: step 02 does not trim it, and the any-time check below would stop the run
        cat = "lower(CAST(med_category AS VARCHAR))" if suffix else "lower(trim(COALESCE(CAST(position_category AS VARCHAR),'')))"
        why = unsure(r, table) or ("" if ts in have else f"clif_{table} has no {ts}")
        if why:
            skip(r, f"48-h rows from clif_{table}", why)
        w = "FALSE" if why else in_window_sql("m.final_death_dttm", utc_sql("t." + ts, have[ts], tz), r.hours48)
        ng, given = "FALSE", "FALSE"       # "given": not a stop or not-given action, and the dose is not 0
        if suffix and "mar_action_category" in have:
            ng = f"lower(trim(CAST(mar_action_category AS VARCHAR))) IN ({not_given})"
            given = f"COALESCE(NOT ({ng}), TRUE)" + (
                " AND COALESCE(TRY_CAST(med_dose AS DOUBLE) <> 0, TRUE)" if "med_dose" in have else "")
        hit = con.sql(f"""
            SELECT m.patient_id, {cat} AS cat, COALESCE(BOOL_OR({w}), false) AS w, COALESCE(BOOL_OR({given}), false) AS g,
                   COALESCE(BOOL_OR(({given}) AND {w}), false) AS gw, COALESCE(BOOL_OR({ng}), false) AS ng
            FROM read_parquet('{tbl(r, table)}') t
            JOIN members m ON CAST(t.hospitalization_id AS VARCHAR) = m.hospitalization_id
            WHERE {cat} IN ({",".join(f"'{c}'" for cats in cols.values() for c in cats)}) GROUP BY 1, 2""").pl()
        if ng != "FALSE":      # 0 here means the action values are not the mCIDE ones, and _given equals any-time
            r.not_given[table] = hit.filter(pl.col("ng"))["patient_id"].n_unique()
        twins = {k: v for k, v in (("_48h", "w"), ("_given", "g"), ("_48h_given", "gw"))
                 if ("48h" not in k or not why) and ("given" not in k or given != "FALSE")}
        for col, cats in cols.items():
            g = (hit.filter(pl.col("cat").is_in(cats)).group_by("patient_id")
                 .agg(pl.lit(True).alias(f"_any_{col}"), *[pl.col(v).any().alias(col + k) for k, v in twins.items()]))
            add = add.join(g, on="patient_id", how="left").with_columns(f(f"_any_{col}"), *[f(col + k) for k in twins])
            assert not any(add.filter(pl.col(col + k) & ~pl.col(f"_any_{col}")).height for k in twins), f"{col}: twin outside any-time"
            exposures[col] = list(twins)
    check(r, "medication and proning any-time flags (patient level)",
          [float(a != b) for c in exposures for a, b in zip(add[f"_any_{c}"], cv[c].fill_null(False))])
    r.mark("Medications and proning")
    return add, adt, exposures, variants


# ── SOFA, scored by clifpy ───────────────────────────────────────────────────

def sofa(r, add: pl.DataFrame, adt: pl.DataFrame) -> pl.DataFrame:
    con, tz, span = r.con, r.tz, pl.duration(hours=r.sofa_hours)
    why = {"terminal": unsure(r, *SOFA_TABLES), "icu_admission": unsure(r, *SOFA_TABLES, "adt", "hospitalization")}
    win = {"terminal": add.select("patient_id", (pl.col("final_death_dttm") - span).alias("start_dttm"),
                                  pl.col("final_death_dttm").alias("end_dttm"))}
    if not why["icu_admission"]:
        # starts at the first ICU in_dttm not before the block's first admission (an earlier one is a
        # placeholder); a time of death before it gives no window
        adm = normalize_datetimes(con.sql(f"""
            SELECT m.patient_id, MIN(h.admission_dttm) AS adm FROM read_parquet('{tbl(r, "hospitalization")}') h
            JOIN members m ON CAST(h.hospitalization_id AS VARCHAR) = m.hospitalization_id GROUP BY 1""").pl(), tz)
        icu_in = (normalize_datetimes(adt, tz).filter(pl.col("location_category").str.to_lowercase() == "icu")
                  .join(adm, on="patient_id", how="left")
                  .filter(pl.col("adm").is_null() | (pl.col("in_dttm") >= pl.col("adm")))
                  .group_by("patient_id").agg(pl.col("in_dttm").min().alias("sofa_icu_in_dttm")))
        add = add.join(icu_in, on="patient_id", how="left")
        win["icu_admission"] = add.filter(pl.col("final_death_dttm") >= pl.col("sofa_icu_in_dttm")).select(
            "patient_id", pl.col("sofa_icu_in_dttm").alias("start_dttm"),
            pl.min_horizontal(pl.col("sofa_icu_in_dttm") + span, "final_death_dttm").alias("end_dttm"))
    for when, reason in why.items():
        if reason:
            skip(r, f"SOFA {when}", reason)
            win.pop(when, None)
    win = {k: w.drop_nulls().join(r.members.select("patient_id", "hospitalization_id"), on="patient_id") for k, w in win.items()}
    shim = r.inter / "supplementary_sofa_tmp"            # patient-level: stays in the local PHI folder
    try:
        if win:
            from clifpy.utils.sofa_polars import compute_sofa_polars
            shutil.rmtree(shim, ignore_errors=True)
            shim.mkdir(parents=True)
            # 24 h before each window as well: clifpy carries FiO2 and device forward from there
            con.register("sofa_win", pl.concat(list(win.values())).select(
                "hospitalization_id", (pl.col("start_dttm") - pl.duration(hours=24)).alias("lo"),
                pl.col("end_dttm").alias("hi")).to_arrow())
            r.unbounded = write_sofa_shim(con, lambda t: tbl(r, t), shim, tz, r.log)
            r.mark("SOFA: local copy of the scored rows")
        for when, w in win.items():
            ids = w["patient_id"].unique().sort()      # scored in even batches: clifpy's hourly grid grows with the cohort
            n_b = max(-(-ids.len() // r.sofa_batch), 1)
            size, scored, lost = -(-ids.len() // n_b), [], 0
            for i in range(n_b):
                part = w.filter(pl.col("patient_id").is_in(ids.slice(i * size, size).implode()))
                try:
                    scored.append(compute_sofa_polars(str(shim), part, filetype="parquet", id_name="patient_id",
                                                      fill_na_scores_with_zero=False, remove_outliers=True, timezone=tz))
                except Exception as e:      # one batch failing must not cost the window
                    lost, err = lost + part["patient_id"].n_unique(), e
            if not scored:
                raise err
            if lost:
                skip(r, f"SOFA {when} for {shown(lost, r.min_cell)} decedents",
                     "batch not scored: " + local_error(r, f"{when}: batch", err))
            s = pl.concat(scored, how="diagonal_relaxed")
            parts = [f"{when}_{c}" for c in SOFA_PARTS]
            s = s.select("patient_id", *[pl.col(c).cast(pl.Float64).alias(p) for c, p in zip(SOFA_PARTS, parts)],
                         pl.col("p_f").cast(pl.Float64).alias(f"{when}_p_f")).pipe(floor_unsupported_resp, when)
            add = (add.join(s, on="patient_id", how="left")
                   .join(w.select("patient_id", pl.lit(True).alias(f"{when}_window")).unique(), on="patient_id", how="left")
                   .with_columns(f(f"{when}_window"), f(f"{when}_resp_floor"),
                                 # a component with no data scores 0 (clifpy's default); no component at all is no score
                                 pl.when(pl.any_horizontal([pl.col(c).is_not_null() for c in parts]))
                                 .then(pl.sum_horizontal(parts)).alias(f"{when}_sofa_total")))
            r.sofa_done.append(when)
            r.mark(f"SOFA: {when}")
    except Exception as e:      # type and place only: a message can quote data values
        why = "not computed: " + local_error(r, "window", e)
        for when in [w for w in win if w not in r.sofa_done]:
            skip(r, f"SOFA {when}", why)
    finally:
        shutil.rmtree(shim, ignore_errors=True)
    return add


# ── Aggregates ───────────────────────────────────────────────────────────────

def aggregates(r, full: pl.DataFrame, exposures: dict, variants: list[str]) -> dict[str, pl.DataFrame]:
    cv, site = r.cv, r.site
    y0, y1 = int(r.study["window_start_year"]), int(r.study["window_end_year"])
    full = full.with_columns(pl.lit(site).alias("site"),      # year on the site clock, as srtr_ref/hospital_years.csv
                             pl.col("final_death_dttm").dt.convert_time_zone(r.tz).dt.year().clip(y0, y1).alias("year"))
    masks = {ALL: pl.Series([True] * r.n), **{d: full[c].fill_null(False) for d, c in DEFS}}
    shipped_defs = pl.col("definition").is_in([d for d, _ in DEFS])
    # the other CALC readings, so a change of the primary reading needs no site re-run
    masks |= {f"CALC[{v}]": (full[f"cause_{v}"] & full["age_75_less"]).fill_null(False)
              for v in variants if v != "i_current|principal"}
    has_icu = pl.col("last_icu_in_dttm").is_not_null()
    # (table, variable, flag, within): decedents with the flag, among those `within`, per definition
    counts = [("table2", "Male sex", pl.col("sex_category").str.to_lowercase() == "male", None),
              ("table2", "Positive blood culture <=48h", ~pl.col("no_positive_culture_48hrs"), None),
              ("table2", ">=1 relative contraindication", pl.col("contra_any"), None),
              (NEW, "HIV (B20 or Z21)", f("hiv"), None), (NEW, "Any ICU stay", has_icu, None),
              (NEW, "Last ICU ADT row has no usable out_dttm", pl.col("last_icu_out_dttm").is_null(), has_icu),
              (NEW, "Last ICU ADT row is the first ICU ADT row", f("single_icu_stay"), has_icu)]
    counts += [("table2", label, f(c), None) for label, c in T2_FLAGS if c in cv.columns]
    counts += [("table2", f"{cat.replace('_category', '').title()}: {v}", pl.col(cat) == v, None)
               for cat in ("race_category", "ethnicity_category") if cat in cv.columns
               for v in sorted(x for x in cv[cat].unique().to_list() if x)]
    if live(cv, "contra_fungemia"):
        counts += [("table2", "Fungemia", f("contra_fungemia"), None),
                   ("table2", "Blood culture obtained", f("tested_blood_culture"), None)]
    counts += [("table3", c, f(c), None) for c in [c for c in cv.columns if c.startswith(("proc_", "med_", "cpt_", "micro_"))]
               + ["on_crrt_48h_before_death", "prone_position"] if live(cv, c)]
    counts += [(NEW + "_48h", col + k, f(col + k), None) for col, twins in exposures.items() for k in ["", *twins]]
    # (table, variable, column, bin width)
    medians = [("table2", label, c, w) for label, c, w in T2_VALUES if c in cv.columns]
    medians += [(NEW, label, c, .5) for label, c in (
        ("Last ICU LOS, last ICU ADT row, strict out_dttm [days]", "last_icu_los_days"),
        ("Last ICU LOS, closed at time of death [days]", "last_icu_los_closed_at_death_days")) if c in full.columns]
    for when, label in (("icu_admission", "SOFA ICU admission"), ("terminal", "SOFA terminal")):
        if when not in r.sofa_done:
            continue
        scored, window = pl.col(f"{when}_sofa_total").is_not_null(), pl.col(f"{when}_window")
        counts += [(NEW + "_sofa", f"{label}: window defined", window, None),
                   (NEW + "_sofa", f"{label}: score available", scored, None),
                   (NEW + "_sofa", f"{label}: score missing", ~scored, None),
                   (NEW + "_sofa", f"{label}: P/F < 200 without ventilatory support, respiratory scored 2",
                    pl.col(f"{when}_resp_floor"), window)]
        if when == "icu_admission":      # an ICU row before the admission time is passed over: how often
            counts.append((NEW + "_sofa", f"{label}: window does not start at the first ICU in_dttm",
                           (pl.col("sofa_icu_in_dttm") != pl.col("first_icu_in_dttm")).fill_null(True), has_icu))
        counts += [(NEW + "_sofa", f"{label}: {c} missing", pl.col(f"{when}_{c}").is_null(), window) for c in SOFA_PARTS]
        medians += [(NEW + "_sofa", f"{label}: {c.replace('sofa_total', 'total')}", f"{when}_{c}", 1)
                    for c in ["sofa_total", *SOFA_PARTS]]
    key = ["site", "table", "variable", "definition", "stat"]
    blank = dict.fromkeys(["n", "denom", "median", "q25", "q75", "n_nonnull", "parent"])
    raw, bins = [], []
    for d, m in masks.items():
        raw.append(dict(zip(key, [site, "table2", "N patients", d, "count"])) | blank | {"n": int(m.sum()), "denom": r.n})
        for table, variable, flag, within in counts:
            base = m if within is None else m & within
            raw.append(dict(zip(key, [site, table, variable, d, "count"])) | blank
                       | {"n": full.filter(base & flag).height, "denom": full.filter(base).height,
                          "parent": None if within is None else int(m.sum())})
        for table, variable, col, width in medians:
            s = full.filter(m)[col].drop_nulls()
            q = {"median": s.median(), "q25": s.quantile(.25), "q75": s.quantile(.75)} if s.len() else {}
            raw.append(dict(zip(key, [site, table, variable, d, "median"])) | blank
                       | {k: float(v) for k, v in q.items()} | {"denom": int(m.sum()), "n_nonnull": s.len()})
            bins += [{"site": site, "table": table, "variable": variable, "definition": d, "bin_lo": b,
                      "width": float(width), "n": n} for b, n in bin_counts(s, width)]
    stats = pl.DataFrame(raw, infer_schema_length=None).sort("table", "variable", "definition", "stat")
    bins = pl.DataFrame(bins).sort("table", "variable", "definition", "bin_lo")
    assert stats.filter((pl.col("definition") == ALL) & (pl.col("variable") == "N patients"))["n"].item() == r.n
    old = stats.filter(pl.col("table").is_in(["table2", "table3"]) & shipped_defs)
    for (table, stat), theirs in shipped(r, "table_stats_raw.csv").group_by(["table", "stat"], maintain_order=True):
        check_keyed(r, f"table_stats_raw {table} {stat}s", old.filter((pl.col("table") == table) & (pl.col("stat") == stat)),
                    theirs, key[1:], ["n", "denom", "median", "q25", "q75", "n_nonnull"])
    bins_file = r.final / "value_bins.csv" if r.bins_shipped else r.inter / "value_bins.csv"
    as_key = pl.col("bin_lo", "width").cast(pl.Float64).round(6).cast(pl.Utf8)
    check_keyed(r, "value_bins", bins.filter((pl.col("table") == "table2") & shipped_defs).with_columns(as_key),
                pl.read_csv(bins_file, infer_schema_length=0).with_columns(as_key),
                ["table", "variable", "definition", "bin_lo", "width"], ["n"])
    bins = top_code_age(bins)      # after the check: step 02 shipped single years
    clif, age = f("clif_eligible_donors"), f("age_75_less")
    def_counts = [pl.len().alias("n_decedents"), *[f(c).sum().alias(n) for n, (_, c) in zip(DEF_NAMES, DEFS)]]
    no_hosp = [pl.lit(None, pl.Utf8).alias("hospital_label"), pl.lit(None, pl.Utf8).alias("srtr_ccn_id")]
    labelled = full.filter(pl.col("hospital_label").is_not_null()).with_columns(pl.col("srtr_ccn_id").cast(pl.Utf8))

    def variant_counts(df: pl.DataFrame, by: list[str], level: str) -> pl.DataFrame:
        return pl.concat([df.group_by(by, maintain_order=True).agg(
            pl.len().alias("n_decedents"), pl.col(f"cause_{k}").sum().alias("n_cause_any_age"),
            (pl.col(f"cause_{k}") & age).sum().alias("n_calc"), (pl.col(f"cause_{k}") & age & clif).sum().alias("n_calc_and_clif_donor"),
            clif.sum().alias("n_clif_donor")).with_columns(
            pl.lit(k.split("|")[0]).alias("cause_definition"), pl.lit(k.split("|")[1]).alias("position"),
            pl.lit(level).alias("level")) for k in variants])
    cvar = (pl.concat([variant_counts(full, ["site"], "site").with_columns(no_hosp),
                       variant_counts(labelled, ["site", "hospital_label", "srtr_ccn_id"], "hospital")], how="diagonal")
            .select("site", "level", "hospital_label", "srtr_ccn_id", "cause_definition", "position", "n_decedents",
                    "n_cause_any_age", "n_calc", "n_calc_and_clif_donor", "n_clif_donor")
            .sort("level", "hospital_label", "cause_definition", "position", descending=[True, False, False, False]))
    site_calc = {f"CALC[{d}|{p}]": n for d, p, n in cvar.filter(pl.col("level") == "site")
                 .select("cause_definition", "position", "n_calc").iter_rows()}
    assert all(int(m.sum()) == site_calc[d] for d, m in masks.items() if d in site_calc), "CALC reading columns"
    ship_h = shipped(r, "hospital_level_counts.csv")
    mine_h = labelled.group_by("hospital_label").agg(
        *def_counts, *[(f(c) & age).sum().alias(n) for n, c in (("CALC_any", "calc_cause_any"), ("CALC_primary", "calc_cause_primary"),
                                                               ("CALC_primary_poa", "calc_cause_primary_poa"))])
    check_keyed(r, "hospital_level_counts", mine_h, ship_h, ["hospital_label"], mine_h.columns[1:])
    for reading, col in (("principal", "CALC_primary"), ("any", "CALC_any")):
        check_keyed(r, f"CALC variant (i) {reading} vs shipped {col}, per hospital",
                    cvar.filter((pl.col("level") == "hospital") & (pl.col("cause_definition") == "i_current")
                                & (pl.col("position") == reading)).rename({"n_calc": col}), ship_h, ["hospital_label"], [col])
    markers = {k: c for k, c in (("G93.82 brain death", "icd10_brain_death"), ("Z52.9 donor of organs", "dx_z529"),
                                 ("Z52.x donor family", "dx_z52x"), ("CPT 01990 donor support", "cpt_01990")) if live(full, c)}
    groups = {ALL: pl.lit(True), **{d: f(c) for d, c in DEFS}, **{f"CALC[{k}]": pl.col(f"cause_{k}") & age for k in variants}}
    mk = pl.DataFrame([{"site": site, "marker": k, "definition": g, "n": full.filter(e & f(c)).height,
                        "denom": full.filter(e).height} for k, c in markers.items() for g, e in groups.items()])
    # the columns of calc_vs_clif.csv, plus age: CALC cannot fail it, a diagnosis code can
    fails = {"age_above_limit": ~age, "not_ventilated_48h": ~f("ventilated_patient_no_age_limit"),
             "contraindicating_cancer": f("icd10_contraindication"), "positive_culture_48h": ~f("no_positive_culture_48hrs"),
             "bmi_above_limit_or_missing": ~f("bmi_eligible"),
             "no_organ_criteria_met": ~(f("kidney_eligible") | f("liver_eligible"))}
    other = pl.any_horizontal([e for k, e in fails.items() if k != "no_organ_criteria_met"])

    def vs_clif(label: str, group: pl.Expr) -> pl.DataFrame:
        out = group & ~clif
        return full.select(pl.lit(site).alias("site"), pl.lit(label).alias("marker"), group.sum().alias("n_marker"),
                           (group & clif).sum().alias("n_clif_donor"), out.sum().alias("n_not_clif_donor"),
                           *[(out & e).sum().alias(k) for k, e in fails.items()],
                           (out & fails["no_organ_criteria_met"] & ~other).sum().alias("organ_only")).limit(1)
    mvc = pl.concat([vs_clif(k, f(c)) for k, c in markers.items()])
    if (r.final / "calc_vs_clif.csv").is_file():          # the same function on CALC must give the shipped row
        check_keyed(r, "calc_vs_clif (CALC row)", vs_clif("CALC", f("calc_flag")).rename({"n_marker": "n_calc", "marker": "calc_reading"}),
                    shipped(r, "calc_vs_clif.csv").filter(pl.col("calc_reading") == "CALC"), ["calc_reading"],
                    ["n_calc", "n_clif_donor", "n_not_clif_donor", *[k for k in fails if k != "age_above_limit"], "organ_only"])
    dxg = diagnosis_group_counts(full, clif, f("calc_flag")).select(pl.lit(site).alias("site"), pl.all())
    n, dc = dict(dxg.limit(5).select("overlap_group", "denom").iter_rows()), shipped(r, "definition_counts.csv").row(0, named=True)
    check(r, "diagnosis_groups overlap groups vs definition_counts",
          [abs(n["clif_only"] + n["both"] - int(dc["clif_donor"])), abs(n["both"] + n["calc_only"] - int(dc["calc"])),
           abs(sum(n.values()) - 2 * int(dc["n_inpatient_deaths"]))])      # the four groups and their total
    hy = labelled.group_by("site", "hospital_label", "srtr_ccn_id", "year").agg(def_counts).with_columns(pl.lit("hospital").alias("level"))
    if (r.final / "srtr_ref/hospital_years.csv").is_file():      # shipped per hospital_id; two ids can share a label
        check_keyed(r, "hospital_years n_decedents", hy, shipped(r, "srtr_ref/hospital_years.csv")
                    .group_by("hospital_label", "year").agg(pl.col("n_decedents").cast(pl.Int64).sum()),
                    ["hospital_label", "year"], ["n_decedents"])
    # hospital totals are already shipped, so one masked year could be had from the others: all of a hospital's years or none
    small = pl.any_horizontal([pl.col(c).is_between(1, r.year_min - 1) for c in ["n_decedents", *DEF_NAMES]])
    per_h = hy.group_by("hospital_label").agg(small.any().alias("s"), pl.col("n_decedents").sum().alias("n"))
    r.year_hospitals = (hospitals_to_ship(dict(zip(per_h["hospital_label"], per_h["s"])),
                                          dict(zip(per_h["hospital_label"], per_h["n"]))), per_h.height)
    r.log(f"by year: {len(r.year_hospitals[0])} of {per_h.height} hospitals ship by year")
    by_year = (pl.concat([full.group_by("site", "year").agg(def_counts).with_columns(pl.lit("site").alias("level"), *no_hosp),
                          hy.filter(pl.col("hospital_label").is_in(r.year_hospitals[0]))], how="diagonal")
               .select("site", "level", "hospital_label", "srtr_ccn_id", "year", "n_decedents", *DEF_NAMES)
               .sort("level", "hospital_label", "year", descending=[True, False, False]))
    timed = pl.concat([full.filter(e).select(pl.lit(site).alias("site"), pl.lit(k).alias("population"), *def_counts).limit(1)
                       for k, e in ((ALL, pl.lit(True)), ("timed_death_dttm", pl.col("death_time_source") == "death_dttm"))])
    r.mark("Aggregates")
    k = r.min_cell
    if k > 1:     # a median over fewer than k decedents describes them individually
        stats = stats.with_columns([pl.when(pl.col("n_nonnull") < k).then(None).otherwise(pl.col(c)).alias(c)
                                    for c in ("median", "q25", "q75")])
    cols, race = ["n_decedents", *DEF_NAMES], pl.col("variable").str.extract(r"^(Race|Ethnicity): ")
    stats = mask_partner(stats, "n", ["definition", race.fill_null(pl.col("table") + pl.col("variable") + pl.col("stat"))], k)
    stats = mask_counts(mask_counts(stats, ["n", "n_nonnull"], k, "denom"), ["denom"], k, "parent")
    if k > 1:     # a subset denominator near its parent gives the complement away, and its n would bound it
        stats = stats.with_columns(pl.when(pl.col("denom").str.starts_with(">")).then(None).otherwise(pl.col("n")).alias("n"))
    for c in cols:      # years of a shipped total; and timed against all, whose difference is the untimed
        by_year = mask_partner(by_year, c, ["level", "hospital_label"], k)
        timed = mask_counts(timed.with_columns(pl.lit(timed[c][0]).alias("_all")), [c], k, "_all")
    out = {"table_stats_raw": stats.drop("parent"),
           "calc_variants": mask_counts(mask_counts(cvar, ["n_calc_and_clif_donor"], k, "n_calc"),
                                        ["n_cause_any_age", "n_calc", "n_clif_donor"], k),
           "markers_by_definition": mask_counts(mk, ["n"], k, "denom"),
           "marker_vs_clif": mask_counts(mask_counts(mvc, ["n_clif_donor", "n_not_clif_donor"], k, "n_marker"),
                                         ["n_marker", *fails, "organ_only"], k),
           "by_year": mask_counts(by_year, cols, k), "timed_death": timed.drop("_all"),
           "diagnosis_groups": mask_counts(mask_partner(dxg, "n", ["group", "reading"], k), ["n"], k, "denom")}
    if r.bins_shipped:      # as step 02: bins ship only where the main bundle shipped them
        out["value_bins"] = mask_counts(mask_partner(bins, "n", ["table", "variable", "definition"], k), ["n"], k)
    return out


def diagnostics(r, full: pl.DataFrame, adt: pl.DataFrame) -> pl.DataFrame:
    y0, y1, k = int(r.study["window_start_year"]), int(r.study["window_end_year"]), r.min_cell
    out, hosp = diagnostic_counts(full, adt, r.tz, y0, y1), col_types(r, "hospitalization")
    more = [{"check": "mar_not_given", "stratum": t, "year": "all", "n": n, "denom": r.n} for t, n in r.not_given.items()]
    if {"age_at_admission", "admission_dttm"} <= hosp.keys():      # is the extract every adult admission, or ICU only
        n, denom = r.con.sql(f"""
            SELECT COUNT(*) FILTER (WHERE CAST(hospitalization_id AS VARCHAR) IN (
                       SELECT CAST(hospitalization_id AS VARCHAR) FROM read_parquet('{tbl(r, "adt")}')
                       WHERE lower(CAST(location_category AS VARCHAR)) = 'icu')), COUNT(*)
            FROM read_parquet('{tbl(r, "hospitalization")}')
            WHERE TRY_CAST(age_at_admission AS DOUBLE) >= 18
              AND year(TRY_CAST(admission_dttm AS TIMESTAMP)) BETWEEN {y0} AND {y1}""").fetchone()
        more.append({"check": "icu_share_all_adults", "stratum": "any_icu_adt_row", "year": "all", "n": n, "denom": denom})
    else:
        skip(r, "diagnostics icu_share_all_adults", "clif_hospitalization lacks age_at_admission or admission_dttm")
    split = pl.col("check").is_in(["principal_dx_per_hospitalization", "location_mix", "death_time_quality"])
    out = mask_partner(pl.concat([out, pl.DataFrame(more, schema=out.schema)]), "n",      # strata of one check, then years of one stratum
                       ["check", "year", pl.when(split).then(pl.lit("")).otherwise(pl.col("stratum"))], k)
    out = mask_partner(out, "n", ["check", "stratum", pl.col("year") == "all"], k)
    out = mask_counts(mask_counts(out, ["n"], k, "denom"), ["denom"], k).with_columns(pl.col("n", "denom").cast(pl.Utf8))
    fresh = []      # whole-table facts, not patient counts: has the extract changed since the bundle was made
    for t in [*DTTM, "hospital_diagnosis"]:
        if not tbl(r, t).is_file():
            continue
        ts = next((c for c in DTTM.get(t, []) if col_types(r, t).get(c, "").upper().startswith(("TIMESTAMP", "DATE"))), None)
        n, latest = r.con.sql(f"SELECT COUNT(*), {f'strftime(MAX({ts}), $$%Y-%m$$)' if ts else 'NULL'} "
                              f"FROM read_parquet('{tbl(r, t)}')").fetchone()
        fresh += [{"check": "table_freshness", "stratum": f"{t}|modified={date.fromtimestamp(tbl(r, t).stat().st_mtime)}",
                   "year": "all", "n": str(n), "denom": None},
                  {"check": "table_freshness", "stratum": f"{t}|latest={latest}", "year": "all", "n": None, "denom": None}]
    r.mark("Diagnostics")
    return pl.concat([out, pl.DataFrame(fresh, schema=out.schema)])


# ── Self-check gate, then write ──────────────────────────────────────────────

def write(r, tables: dict[str, pl.DataFrame]) -> None:
    out = r.final / NEW
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    checks = pl.DataFrame(r.checks)
    checks.write_csv(out / f"{NEW}_selfcheck.csv")
    failed, passed = checks.filter(pl.col("status") == "fail"), checks.filter(pl.col("status") == "pass")
    if failed.height:
        (out / f"{NEW}_run_log.txt").write_text("\n".join(r.lines) + "\nSELF-CHECK FAILED\n", encoding="utf-8")
        raise SystemExit(
            "SELF-CHECK FAILED: this step does not reproduce the returned bundle for: "
            + "; ".join(f"{c['family']} ({c['n_different']} of {c['n_compared']} differ)" for c in failed.iter_rows(named=True))
            + ". No result was written. The usual cause is CLIF tables re-extracted after steps 01-03 were run: if so, "
              "re-run the pipeline (uv run python run_pipeline.py), return that bundle, and run this step again. Otherwise "
              f"send {r.final.name}/{NEW}/{NEW}_selfcheck.csv and {NEW}_run_log.txt to the coordinating centre.")
    r.log(f"self-check passed: {int(passed['n_compared'].sum()):,} values equal the returned bundle")
    for name, df in tables.items():
        df.write_csv(out / f"{NEW}_{name}.csv")
        r.log(f"  {NEW}_{name + '.csv':30s} {df.height:>6,} rows")
    import clifpy
    k, nl = r.min_cell, "\n"
    skipped = [f"- NOT WRITTEN, {c['family']}: {c['detail']}" for c in r.checks if c["status"] == "skipped"]
    (out / f"{NEW}_provenance.md").write_text(f"""# Supplementary tables provenance — {r.site}

`code/04_supplementary_tables.py`, steps 01-03 not re-run. {r.n:,} decedents. clifpy {clifpy.__version__}, polars {pl.__version__}, duckdb {duckdb.__version__}.

## Bundle checked against
{nl.join(f"- `{n}` sha256 `{h}`" for n, h in r.hashes.items())}
- {passed.height} families, {int(passed['n_compared'].sum()):,} values recomputed, all equal (`{NEW}_selfcheck.csv`).
{nl.join(skipped) or "- Nothing was skipped."}

## Small cells
- {f"Counts of 1 to {k - 1} are written `<{k}`, a count within {k - 1} of its denominator `>denominator-{k}`, and medians over fewer than {k} decedents are blank. Where exactly one cell of a breakdown (bins, years, strata, race) is masked, the next smallest is blank." if k > 1 else "Raw counts, as in the returned bundle."}{"" if r.bins_shipped else " No value bins: the returned bundle has none."}
- By year: {len(r.year_hospitals[0])} of {r.year_hospitals[1]} hospitals ship (every hospital-year cell 0 or >= {r.year_min}; never exactly one withheld).

## Rules
- `*_48h`: at least one row in the {r.hours48:g} h before the time of death, ends included; any row counts, as in step 02.
- `*_given`: only rows whose mar_action_category is not one of {", ".join(r.cfg["mar_actions_not_given"])} and whose dose is not 0 (neither recorded counts as given).
- Last ICU LOS: latest ICU row in clif_adt; an out_dttm before its in_dttm or from 2100 on is unknown; "closed at time of death" ends an open row there.
- `CALC[<variant>|<reading>]` definition columns: Tables 1-2 under each other CALC reading, age 75 or less. Age bins: 90 is 90 and over.
- `{NEW}_diagnosis_groups.csv`: decedents with each named diagnosis group (principal and any position) and each ICD-10 chapter (principal) per overlap group of CLIF-donor and the bundle's CALC; a decedent with several principal rows counts in each chapter.
- CALC variants: `principal_only_ranges` (S00-T88) are read at the principal position under both readings.
- `{NEW}_last_location.csv`: decedents per unit (location_name) of their last ADT row; units with fewer than 11 decedents are folded into one row per location category.
- `{NEW}_diagnostics.csv`: decedents per data-quality stratum (diagnosis flags, missing rows, location mix, external-cause codes, death-time source), the ICU share of all adult admissions, and each table's file date, row count and latest month.
- Clock changes: naive timestamps are site time and windows are elapsed time; step 01 subtracts wall-clock readings, so a window spanning a clock change can differ by one hour at its edge.

## SOFA ({"clifpy compute_sofa_polars, worst value in the window: " + ", ".join(r.sofa_done) if r.sofa_done else "not computed"})
- Windows: {r.sofa_hours:g} h from the first ICU in_dttm not before the block's first admission, cut at the time of death (none without an ICU stay or if death precedes it); and the {r.sofa_hours:g} h before the time of death.
- A component with no data scores 0 in the total; no component at all is no score.
- Inputs bounded by `config/outlier_config.yaml`; pressor rows with dose 0 or missing, or a unit that is not a mass rate (mcg, mg, ng or g, per kg or lb or not, per min or h), dropped; FiO2 above 1 read as a percentage. No range defined, none applied: {", ".join(r.unbounded) or "n/a"}.
- Respiratory (`P/F < 200 without ventilatory support, respiratory scored 2`): clifpy scores 3-4 on IMV, NIPPV or CPAP and leaves other devices null under 200; set to 2 here, as SOFA defines.
- Not adjusted: a vasopressor dose with no /kg unit and no earlier weight under its hospitalization_id, and a decedent with no recorded MAP, are scored as clifpy 2.1.0 scores them (component missing, counted as 0).
- Labs by lab_result_dttm; renal is creatinine only; respiratory needs an arterial PaO2 with FiO2 in the prior 4 h.
""", encoding="utf-8")
    r.log(f"done: {len(tables) + 3} files -> {r.final.name}/{NEW}/")
    r.mark("Write")
    (out / f"{NEW}_run_log.txt").write_text("\n".join(r.lines) + "\n", encoding="utf-8")      # ships: counts and timings only


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    from utils.config import config
    crit = yaml.safe_load((REPO / "config/donor_criteria.yaml").read_text())
    progress, lines = Progress(), []

    def log(msg: str) -> None:
        print(msg)
        lines.append(msg)
    r = SimpleNamespace(site=config["site_name"], tz=config["timezone"], tables=Path(config["tables_path"]),
                        inter=Path(config["output_intermediate"]), final=Path(config["output_final"]), crit=crit,
                        study=crit["study"], cfg=crit[NEW], hours48=float(crit["clif_donor"]["imv_hours_before_death"]),
                        sofa_hours=float(crit[NEW]["sofa"]["window_hours"]), lines=lines, log=log,
                        mark=lambda section: lines.append(progress.mark(section)),
                        sofa_batch=int(crit[NEW]["sofa"].get("batch_size", 5000)), mem=config.get("duckdb_memory_limit"),
                        checks=[], sofa_done=[], unbounded=[], not_given={})
    load_inputs(r)
    add, adt, exposures, variants = patient_variables(r)
    add = sofa(r, add, adt)
    assert add["patient_id"].equals(r.cv["patient_id"]), "a join reordered or repeated decedents"
    full = pl.concat([r.cv, add.drop("patient_id", "final_death_dttm")], how="horizontal")
    write(r, aggregates(r, full, exposures, variants) | {"diagnostics": diagnostics(r, full, adt),
                                                             "last_location": mask_counts(last_location(adt, r.site), ["n"], r.min_cell)})


if __name__ == "__main__":
    main()
