#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script identifies potential organ donors based on the 
CALC criteria and the CLIF eligible criteria using the CLIF dataset. 
It performs the following steps:
1. Load the required datasets
2. Identify decedents
3. Stitch encounters
4. Apply outlier handling
5. Create consort diagram
"""
################################################################################
# Setup
################################################################################

import sys
import duckdb
import polars as pl
import pandas as pd
import re
import yaml
from utils.config import config
from utils.io import read_data
from clifpy.utils.stitching_encounters import stitch_encounters
from utils.outlier_handler import outlier_range
from utils.checks import mixed_tz_awareness, usable_number_sql
from utils.dtypes import align_time_zone, hours_between
from utils.death_time import (SOURCES as DEATH_TIME_SOURCES, add_death_anchor, comparable, has_time_of_day,
                              midnight_shares)
from utils.windows import imv_before_death_sql
from utils.progress import Progress
from utils.criteria import contraindication_sql, icd_range_sql, load_contraindications

# Timestamp arithmetic must not depend on the machine this runs on. DuckDB's
# session timezone defaults to the operating system's; pin it so a subtraction
# gives the same answer everywhere. The setup check below additionally refuses
# an extract that mixes timezone-aware and naive timestamps, which is the case
# where the session timezone would silently change a 48-hour window.
duckdb.sql("SET TimeZone='UTC'")
_progress = Progress()                 # one line per section in run_log.txt: seconds and peak memory

# Fix Windows encoding issue for Unicode characters
sys.stdout.reconfigure(encoding='utf-8')

# load config
site_name = config['site_name']
tables_path = config['tables_path']
file_type = "parquet"                  # CLIF tables are read as parquet only
project_root = config['project_root']
sys.path.insert(0, project_root)
print(f"Site Name: {site_name}")
print(f"Tables Path: {tables_path}")
from pathlib import Path
PROJECT_ROOT = Path(config['project_root'])
# SANITY CHECK: Checked here, before anything is read from it: every code list, criteria file
# and output path hangs off this, so a wrong value gives a FileNotFoundError
_REPO_HERE = Path(__file__).resolve().parent.parent
if PROJECT_ROOT.expanduser().resolve() != _REPO_HERE:
    raise SystemExit(
        f"config.json: project_root is '{PROJECT_ROOT}', which is not the folder "
        f"this code is in ({_REPO_HERE}). Set project_root to that path and rerun.")
# PHI split: output/intermediate_phi/<site>/ never leaves the site;
# <site>_upload_to_box/ holds aggregate, shareable output. Per-site subfolders so
# three sites can be run on one machine without clobbering each other.
# Every study criterion, read from config/donor_criteria.yaml and used directly.
CRIT = yaml.safe_load((PROJECT_ROOT / "config/donor_criteria.yaml").read_text())
STUDY, DONOR, CALC_CFG = CRIT["study"], CRIT["clif_donor"], CRIT["calc"]
COHORT_LOCATIONS = [str(x).lower() for x in STUDY["cohort_locations"]]
CONTRA_CODES = load_contraindications(PROJECT_ROOT / DONOR["contraindications"]["icd10_file"])
_RANGE_SQL = {k: icd_range_sql("dx_norm", lo, hi)
              for k, (lo, hi) in CALC_CFG["cause_icd10_prefixes"].items()}
_ISCHEMIC_SQL = _RANGE_SQL["ischemic_heart_disease"]
_CEREBRO_SQL = _RANGE_SQL["cerebrovascular_disease"]
_EXTERNAL_SQL = _RANGE_SQL["external_causes"]


################################################################################
# Setup checks
# Everything that can be verified before reading a single patient row.
################################################################################

REQ = yaml.safe_load((PROJECT_ROOT / "config/clif_data_requirements.yaml").read_text())
_problems, _warnings = [], []
_TIMESTAMPS_NAIVE = False

# 1. config
for _k in ("site_name", "tables_path", "timezone", "project_root", "hospital_ids"):
    if config.get(_k) in (None, "", []):
        _problems.append(f"config.json: '{_k}' is missing or empty")
DECLARED_HOSPITAL_IDS = [str(h).lower().strip() for h in config.get("hospital_ids", [])]
if not DECLARED_HOSPITAL_IDS:
    _problems.append("config.json: hospital_ids is empty — list every hospital_id in clif_adt")

# 2. tables and columns, from schemas only
_tp = Path(tables_path)
if not _tp.is_dir():
    _problems.append(f"config.json: tables_path does not exist: {_tp}")
else:
    _impact = REQ.get("optional_table_impact", {})
    _dttm_types: dict[str, str] = {}
    OPTIONAL_UNUSABLE: set[str] = set()      # optional tables absent or missing columns
    for _kind in ("required", "optional"):
        for _tbl, _cols in REQ["tables"][_kind].items():
            _f = _tp / f"clif_{_tbl}.{file_type}"
            _msg = None
            if not _f.is_file():
                _msg = f"clif_{_tbl} is absent"
            else:
                try:
                    _types = {r[0]: r[1] for r in duckdb.sql(
                        f"DESCRIBE SELECT * FROM read_parquet('{_f}')").fetchall()}
                    _have = set(_types)
                    _dttm_types.update({f"clif_{_tbl}.{c}": _types[c] for c in _cols
                                        if c.endswith("_dttm") and c in _types})
                    _missing = [c for c in _cols if c not in _have]
                    if _missing:
                        _msg = f"clif_{_tbl} is missing columns: {', '.join(_missing)}"
                except Exception as e:
                    _msg = f"clif_{_tbl} is unreadable — {type(e).__name__}: {e}"
            if _msg:
                if _kind == "optional":
                    OPTIONAL_UNUSABLE.add(_tbl)
                (_problems if _kind == "required" else _warnings).append(
                    _msg + (f" ({_impact[_tbl]})" if _tbl in _impact else ""))
    # 3. the timestamps step 01 subtracts from one another: the death-anchored
    #    windows and the death/discharge comparison. A mix of timezone-aware and
    #    naive among THESE silently shifts a window, so it stops the run. The
    #    other declared *_dttm columns are not used in arithmetic here; one that
    #    differs is a warning, not a reason to make a site re-export.
    _WINDOW_DTTM = {"clif_patient.death_dttm", "clif_hospitalization.discharge_dttm",
                    "clif_vitals.recorded_dttm", "clif_respiratory_support.recorded_dttm",
                    "clif_crrt_therapy.recorded_dttm", "clif_labs.lab_collect_dttm",
                    "clif_microbiology_culture.collect_dttm",
                    "clif_patient_assessments.recorded_dttm"}
    _window_types = {c: t for c, t in _dttm_types.items() if c in _WINDOW_DTTM}
    _problems += mixed_tz_awareness(_window_types)
    _aware = {t.upper() == "TIMESTAMP WITH TIME ZONE" for t in _window_types.values()}
    _TIMESTAMPS_NAIVE = _aware == {False}
    for _c, _t in sorted(_dttm_types.items()):
        if _c not in _WINDOW_DTTM and len(_aware) == 1 and (
                not _t.upper().startswith("TIMESTAMP")
                or (_t.upper() == "TIMESTAMP WITH TIME ZONE") not in _aware):
            _warnings.append(f"{_c} is {_t}, unlike the timestamps used for the death-anchored "
                             f"windows; variables built from it may be affected")

# 4. the hospital crosswalk must know this site and every hospital it declares.
#    A hospital without a crosswalk row cannot be linked to SRTR; its decedents
#    would leave every hospital-level output without any error.
_XWALK = yaml.safe_load(
    (PROJECT_ROOT / "config/hospital_crosswalk.yaml").read_text())["hospitals"]
_xw_ids = {str(r["hospital_id"]).lower().strip() for r in _XWALK if r["site"] == site_name}
if not _xw_ids:
    _problems.append(f"config/hospital_crosswalk.yaml has no rows for site '{site_name}'. "
                     f"Check site_name, or ask the coordinating centre to add your hospitals")
for _h in (DECLARED_HOSPITAL_IDS if _xw_ids else []):
    if _h not in _xw_ids:
        _problems.append(f"hospital_id '{_h}' is in config.json hospital_ids but has no row for "
                         f"site '{site_name}' in config/hospital_crosswalk.yaml. Ask the "
                         f"coordinating centre to add it")

for _w in _warnings:
    print(f"  warn  {_w}")
if _problems:
    print("\nSetup check FAILED:")
    for _pr in _problems:
        print(f"  - {_pr}")
    raise SystemExit(1)
print(f"Setup check passed: config valid, {len(REQ['tables']['required'])} required tables present"
      + (f", {len(_warnings)} warning(s)" if _warnings else ""))

SITE_TZ = config["timezone"]


def _year_local(df: pl.DataFrame, col: str) -> pl.Expr:
    """Calendar year of `col` on the site's wall clock.

    CLIF timestamps are stored in UTC. The study window and SRTR's recovery
    years are calendar years where the hospital is, so a death at 21:00 on
    31 December local time belongs to that year, not to the next one that UTC
    has already entered. A timezone-naive column is taken to be local already.
    """
    e = pl.col(col)
    if getattr(df.schema[col], "time_zone", None):
        e = e.dt.convert_time_zone(SITE_TZ)
    return e.dt.year()


def _require_matches(n: int, table: str, column: str, wanted: str) -> None:
    """Stop when a category filter matched nothing for the whole cohort.

    That is almost always a spelling difference in the site's extract, and the
    silent consequence is a plausible-looking wrong answer: no labs makes every
    patient organ-ineligible, no cultures makes nobody culture-positive. The
    values listed are vocabulary strings, not patient data.
    """
    if n > 0:
        return
    _seen = duckdb.sql(
        f"SELECT DISTINCT CAST({column} AS VARCHAR) FROM read_parquet("
        f"'{tables_path}/clif_{table}.{file_type}') LIMIT 30").fetchall()
    raise SystemExit(
        f"clif_{table}: no usable rows with {column} = '{wanted}' for this cohort. "
        f"Values present in {column}: {sorted(str(v[0]) for v in _seen)}. "
        f"The CLIF mCIDE spelling is expected; case and surrounding spaces are ignored.")


OUTLIER_CONFIG = PROJECT_ROOT / "config/outlier_config.yaml"
# Bounds are read from the config rather than written into the SQL below, so
# outlier_config.yaml is the only place any range is defined.
WEIGHT_MIN, WEIGHT_MAX = outlier_range("vitals", "vital_value", "weight_kg", OUTLIER_CONFIG)
HEIGHT_MIN, HEIGHT_MAX = outlier_range("vitals", "vital_value", "height_cm", OUTLIER_CONFIG)
HR_MIN, HR_MAX = outlier_range("vitals", "vital_value", "heart_rate", OUTLIER_CONFIG)
AGE_MIN, AGE_MAX = outlier_range("hospitalization", "age_at_admission", None, OUTLIER_CONFIG)
GCS_MIN, GCS_MAX = outlier_range("patient_assessments", "numerical_value", "gcs_total", OUTLIER_CONFIG)
RASS_MIN, RASS_MAX = outlier_range("patient_assessments", "numerical_value", "rass", OUTLIER_CONFIG)

OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_FINAL_DIR = Path(config["output_final"])
OUTPUT_INTERMEDIATE_DIR = Path(config["output_intermediate"])
OUTPUT_FINAL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
# DuckDB spills to disk instead of failing when a query outgrows memory, but
# only once it has a temp_directory; the in-memory default has none. Kept with
# the site's local output, never shipped.
(OUTPUT_INTERMEDIATE_DIR / "duckdb_tmp").mkdir(exist_ok=True)
duckdb.sql(f"SET temp_directory='{OUTPUT_INTERMEDIATE_DIR / 'duckdb_tmp'}'")

# ---- Run log -----------------------------------------------------------
# Tee stdout so every print() in this run also lands in <site>_upload_to_box/run_log.txt.
# That file ships, so nothing printed may identify a patient: counts only, never ids.
# Overwrites the previous run's log; sites only need the most recent.
import atexit, datetime as _dt

class _Tee:
    """Write to multiple streams (e.g., terminal + log file) simultaneously."""
    def __init__(self, *streams):
        self._streams = streams
    def write(self, msg):
        for s in self._streams:
            try:
                s.write(msg); s.flush()
            except Exception:
                pass
    def flush(self):
        for s in self._streams:
            try: s.flush()
            except Exception: pass

_RUN_LOG_PATH = OUTPUT_FINAL_DIR / "run_log.txt"
_run_log_handle = open(_RUN_LOG_PATH, "w", encoding="utf-8")
sys.stdout = _Tee(sys.__stdout__, _run_log_handle)
atexit.register(lambda: (_run_log_handle.flush(), _run_log_handle.close()))
print(f"=== Run started {_dt.datetime.now():%Y-%m-%d %H:%M:%S} ===")
print(f"Site: {site_name} | Tables: {tables_path}")
print(f"Logging stdout to: {_RUN_LOG_PATH}")
print("-" * 80)

strobe_counts = {}
# Conditions that do not stop the run but change what a number means at this
# site. Written to data_quality_flags.csv and shown in provenance.md.
dq_flags: list[dict] = []
if _TIMESTAMPS_NAIVE:
    dq_flags.append({
        "site": site_name, "flag": "timestamps_timezone_naive", "severity": "medium",
        "detail": (f"The timestamp columns carry no timezone. They are taken to be local "
                   f"wall-clock time ({SITE_TZ}) when assigning calendar years; if they are "
                   f"in fact UTC, admissions and deaths within a few hours of New Year are "
                   f"assigned to the wrong year.")})

_progress.mark("Setup checks")

################################################################################
# Load data
################################################################################

# read required tables
adt_filepath = f"{tables_path}/clif_adt.{file_type}"
hospitalization_filepath = f"{tables_path}/clif_hospitalization.{file_type}"
patient_filepath = f"{tables_path}/clif_patient.{file_type}"
adt_df = read_data(adt_filepath, file_type)
hospitalization_df = read_data(hospitalization_filepath, file_type)
patient_df = read_data(patient_filepath, file_type)

# Informational baseline (entire CLIF dataset at this site, all years)
total_patients = patient_df["patient_id"].n_unique()
strobe_counts["00_all_patients_in_clif"] = total_patients

_progress.mark("Load data")

################################################################################
# Apply cohort time-window filter (2020-01-01 to 2025-12-31)
# STRICT: a hospitalization is kept only if BOTH admission and discharge fall
# inside the window. This matches the SRTR donor cohort's strict calendar
# boundaries so the two are directly comparable.
################################################################################

WINDOW_START_YEAR = int(STUDY["window_start_year"])
WINDOW_END_YEAR = int(STUDY["window_end_year"])
STRICT_WINDOW = bool(STUDY.get("require_admission_and_discharge_in_window", True))

# Compare on the calendar year in the site's timezone (see _year_local).
_adm_year = _year_local(hospitalization_df, 'admission_dttm')
_dis_year = _year_local(hospitalization_df, 'discharge_dttm')
_admitted_in_window = (_adm_year >= WINDOW_START_YEAR) & (_adm_year <= WINDOW_END_YEAR)
_discharged_in_window = (_dis_year >= WINDOW_START_YEAR) & (_dis_year <= WINDOW_END_YEAR)
# STRICT keeps only hospitalizations wholly inside the window. Otherwise keep
# any that OVERLAP it, and require the DEATH itself to fall in the window at the
# decedent step below 
_overlaps_window = (_adm_year <= WINDOW_END_YEAR) & (_dis_year >= WINDOW_START_YEAR)
_n_before = hospitalization_df.height
hospitalization_df = hospitalization_df.filter(
    (_admitted_in_window & _discharged_in_window) if STRICT_WINDOW else _overlaps_window)
print(f"Window {WINDOW_START_YEAR}-{WINDOW_END_YEAR} "
      f"({'admission AND discharge inside' if STRICT_WINDOW else 'overlapping'}): "
      f"{_n_before:,} -> {hospitalization_df.height:,} hospitalizations")

# Restrict ADT to hospitalizations that survived the window filter
_hosp_ids_in_window = hospitalization_df['hospitalization_id'].unique().to_list()
adt_df = adt_df.filter(pl.col('hospitalization_id').is_in(_hosp_ids_in_window))

# SANITY CHECK - Ensure that out_dttm is not before in_dttm
# An out_dttm before its own in_dttm is a placeholder, not a time: 
_out_before_in = pl.col("out_dttm") < pl.col("in_dttm")
_n_out_before_in = adt_df.filter(_out_before_in).height
if _n_out_before_in:
    adt_df = adt_df.with_columns(pl.when(_out_before_in).then(None).otherwise(pl.col("out_dttm")).alias("out_dttm"))
    dq_flags.append({
        "site": site_name, "flag": "adt_out_dttm_before_in_dttm", "severity": "medium",
        "detail": (f"{_n_out_before_in:,} of {adt_df.height:,} ADT rows in the window end before they begin "
                   f"(a placeholder date). Their out_dttm is read as unknown: the first-ICU length of stay "
                   f"is missing where that stay has no other end, and the row is not taken as the last of its stay.")})
    print(f"  warn  {_n_out_before_in:,} ADT rows end before they begin; out_dttm read as unknown "
          f"(see data_quality_flags.csv)")

# STROBE step 1: Full population — patients with any hospitalization overlapping 2020-2025
full_population_n = hospitalization_df['patient_id'].n_unique()
strobe_counts["0_full_population_2020_2025"] = full_population_n
print(f"Full population (any hosp 2020-2025): {full_population_n:,}")

_progress.mark("Apply cohort time-window filter (2020-01-01 to 2025-12-31)")

################################################################################
# Identify decedents
################################################################################

_death_category = str(STUDY.get("death_discharge_category", "expired")).lower()
all_decedents_df = hospitalization_df.filter(
    pl.col('discharge_category').str.to_lowercase() == _death_category
)
if not STRICT_WINDOW:
    # The death itself defines inclusion, so a patient admitted before the
    # window who died inside it is kept, and one who died after it is not.
    _n_deaths = all_decedents_df.height
    all_decedents_df = all_decedents_df.filter(_discharged_in_window)
    print(f"Deaths with death date in {WINDOW_START_YEAR}-{WINDOW_END_YEAR}: "
          f"{_n_deaths:,} -> {all_decedents_df.height:,}")
strobe_counts["0c_decedents_in_window"] = all_decedents_df['patient_id'].n_unique()

all_decedent_patient_ids = all_decedents_df.select('patient_id').to_series().to_list()
all_decedent_hosp_ids = all_decedents_df.select('hospitalization_id').to_series().to_list()

_progress.mark("Identify decedents")

################################################################################
# Stitch Encounters
################################################################################

# Check if hospitalization_df has duplicate patient_id, hospitalization_id pairs
if hospitalization_df.shape[0] != hospitalization_df.unique(subset=["patient_id", "hospitalization_id"]).shape[0]:
    print("Warning: hospitalization_df contains duplicate (patient_id, hospitalization_id) rows.")

# Check if adt_df has duplicate patient_id, hospitalization_id, adt_event_id (or similar) triplets
# If adt_df has an event or unique identifier column, replace 'adt_event_id' with correct column
adt_unique_cols = [col for col in [ "hospitalization_id", "in_dttm"] if col in adt_df.columns]
if len(adt_unique_cols) >= 2 and adt_df.shape[0] != adt_df.unique(subset=adt_unique_cols).shape[0]:
    print("Warning: adt_df contains duplicate rows for identifier columns:", adt_unique_cols)

# Filter hospitalization_df and adt_df down to all_decedent_hosp_ids
hospitalization_df_subset = hospitalization_df.filter(
    pl.col("patient_id").is_in(all_decedent_patient_ids)
)
all_decedent_hosp_ids = hospitalization_df_subset.select('hospitalization_id').to_series().to_list()
adt_df_subset = adt_df.filter(
    pl.col("hospitalization_id").is_in(all_decedent_hosp_ids)
)

hosp_stitched, adt_stitched, encounter_mapping = stitch_encounters(
      hospitalization=hospitalization_df_subset.to_pandas(),
      adt=adt_df_subset.to_pandas(),
      time_interval=STUDY['encounter_stitch_hours']
  )

hosp_stitched = pl.from_pandas(hosp_stitched)
adt_stitched = pl.from_pandas(adt_stitched)

# Ensure encounter_block is int32 (if present)
hosp_stitched = hosp_stitched.with_columns(
    pl.col("encounter_block").cast(pl.Int32)
)
adt_stitched = adt_stitched.with_columns(
    pl.col("encounter_block").cast(pl.Int32)
)

# Only stays with an ADT row go on: without one there is no location and no hospital.
hosp_stitched = hosp_stitched.filter(
    pl.col("hospitalization_id").is_in(adt_stitched["hospitalization_id"].unique()))
encounter_mapping = pl.from_pandas(encounter_mapping)
encounter_mapping = encounter_mapping.with_columns(
    pl.col("encounter_block").cast(pl.Int32)
)


# Identify expired encounters
decedents_df = hosp_stitched.filter(
    pl.col('discharge_category').str.to_lowercase() == 'expired'
)

# Make hospitalization subset for expired
# Join patient_id and death_dttm from patient_df to final_df

final_df = (
    decedents_df
    .select([
        'patient_id',
        'hospitalization_id',
        'encounter_block',
        'admission_dttm',
        'discharge_dttm', # discharge datetime for the death hospitalization
        "age_at_admission", 
        "discharge_category",
        "admission_type_category"
    ])
    .with_columns([
        pl.col("discharge_category").str.to_lowercase(),
        pl.col("admission_type_category").str.to_lowercase()
    ])
    .unique()
)

# Now join patient_id and death_dttm from patient_df to final_df
demog_cols = ['patient_id', 'death_dttm', 'race_category', 'sex_category','ethnicity_category' ]
final_df = final_df.join(
    patient_df.select(demog_cols), on='patient_id', how='left'
)

# ----------------------------------------------------------------------
# Death hospitalization analysis + early per-patient dedup
# ----------------------------------------------------------------------
# Diagnostic FIRST — show raw counts of multi-death patients (rare data
# quirks worth surfacing), THEN unconditionally collapse to one row per
# patient (the latest death encounter). Doing the dedup early means every
# downstream operation operates on one-row-per-patient and the strobe
# counts therefore agree with Table 1 by construction.
print("\n" + "="*80)
print("DEATH HOSPITALIZATION ANALYSIS")
print("="*80)

death_encounters_per_patient = (
    final_df
    .group_by('patient_id')
    .agg([
        pl.count().alias('death_encounter_count'),
        pl.col('encounter_block').alias('encounter_blocks'),
        pl.col('discharge_dttm').alias('discharge_times'),
    ])
    .sort('death_encounter_count', descending=True)
)

total_death_encounters = len(final_df)
unique_patients = final_df['patient_id'].n_unique()
multi_death_patients = death_encounters_per_patient.filter(pl.col('death_encounter_count') > 1)
num_multi_death = len(multi_death_patients)

print(f"Total death hospitalizations found: {total_death_encounters:,}")
print(f"Unique patients who died:           {unique_patients:,}")
print(f"Patients with multiple deaths:      {num_multi_death:,}")

if num_multi_death > 0:
    # Count only. Everything printed is teed into run_log.txt, which ships, so
    # no patient_id, encounter key or timestamp may appear here. The ids go to
    # the local PHI folder for the site to inspect.
    multi_death_patients.select(["patient_id", "death_encounter_count"]).write_csv(
        OUTPUT_INTERMEDIATE_DIR / "multi_death_patients.csv")
    print(f"\nNOTE: {num_multi_death} patients have multiple death rows — likely a data quality issue.")
    print("  (ids written to the local intermediate folder, never shipped)")
    print("Will collapse each patient to their LATEST death encounter below.")
else:
    print("✓ No patients with multiple death hospitalizations — data is clean.")

# Always collapse to one row per patient = the latest death encounter.
# (For decedents the latest encounter_block IS the death encounter; before
# this point a patient could technically have multiple expired rows due to
# data quirks. From here on, final_df is strict patient-level.)
final_df = (
    final_df
    .sort('discharge_dttm', descending=True)
    .unique(subset='patient_id', keep='first')   # latest after desc sort
)
assert final_df['patient_id'].n_unique() == len(final_df), "Early dedup failed"
print(f"\n✓ After early dedup: {len(final_df):,} unique patients (one row each).")
print("="*80 + "\n")

decedents_df_n = final_df["patient_id"].n_unique()
strobe_counts["1_decedents_df_n"] = decedents_df_n
# Decedents with no ADT row for any death stay left at the stitching step above:
# with no location and no hospital they can go no further. Named, so the CONSORT
# states the drop instead of leaving it as the difference of the two rows, and
# asserted, so nothing else can drop a patient between them unnoticed.
_pts_with_adt = all_decedents_df.filter(
    pl.col("hospitalization_id").is_in(adt_df["hospitalization_id"].unique()))["patient_id"]
strobe_counts["1b_decedents_without_adt"] = int(
    all_decedents_df.filter(~pl.col("patient_id").is_in(_pts_with_adt))["patient_id"].n_unique())
assert strobe_counts["0c_decedents_in_window"] - strobe_counts["1b_decedents_without_adt"] \
    == decedents_df_n, "decedent flow does not reconcile: 0c - 1b != 1"

_progress.mark("Stitch Encounters")

################################################################################
# Time of death
################################################################################

# final_death_dttm anchors every "within N hours of death" window. The rule and
# its reasons are in utils/death_time.py. In short:
#   death_dttm with a time of day  -> death_dttm
#   death_dttm that is only a date -> discharge_dttm, kept inside that date
#   death_dttm missing             -> discharge_dttm
# and a death recorded long after discharge is disregarded in favour of
# discharge. The last recorded vital is not used; discharge_vs_death_timing.csv
# and vitals_timing.csv, written at the end of this step, are the evidence.
final_df = add_death_anchor(final_df, SITE_TZ, STUDY['death_time']['max_hours_after_discharge'])

# Dates stored as timestamps at UTC midnight are a different encoding of a
# date-only death, which the rule above does not recognise. A real death lands
# on a given minute about 0.07% of the time, so more than 1% means an encoding.
_midnight = midnight_shares(final_df, SITE_TZ)
if _midnight["utc"] > 0.01:
    dq_flags.append({
        "site": site_name, "flag": "death_dttm_date_only_at_utc_midnight", "severity": "high",
        "detail": (f"{100 * _midnight['utc']:.0f}% of recorded death times are exactly 00:00:00 UTC, "
                   f"which looks like dates stored at UTC midnight. They are treated as real times, "
                   f"so the windows before death end the evening before. Tell the coordinating centre.")})
    print("  WARNING  death_dttm looks date-only at UTC midnight (see data_quality_flags.csv)")

_progress.mark("Time of death")

################################################################################
# Inpatient decedents
# Identify inpatient encounters - the stay must have touched one of study.cohort_locations
################################################################################

# Check that all decedents are present in ADT table
decedent_hosp_in_adt = set(adt_df.select('hospitalization_id').to_series().to_list())
missing_in_adt = set(all_decedent_hosp_ids) - decedent_hosp_in_adt

if missing_in_adt:
    # Count only: run_log.txt ships, and these are encounter keys.
    pl.DataFrame({"hospitalization_id": sorted(str(h) for h in missing_in_adt)}).write_csv(
        OUTPUT_INTERMEDIATE_DIR / "decedent_hospitalizations_missing_in_adt.csv")
    print(f"Warning: {len(missing_in_adt)} hospitalization(s) missing in ADT table "
          f"(ids written to the local intermediate folder, never shipped)")
else:
    print(f"✓ All {len(all_decedent_hosp_ids)} decedent hospitalizations present in ADT table")

last_location_per_hosp = (
      adt_df
      .filter(pl.col('hospitalization_id').is_in(all_decedent_hosp_ids))
      .sort('out_dttm', descending=True, nulls_last=True)
      .group_by('hospitalization_id')
      .agg([
          pl.col('location_category').first().alias('last_location_category'),
          (pl.col('location_category').str.to_lowercase() == 'icu').any().alias('ever_icu'),
          (pl.col('location_category').str.to_lowercase() == 'hospice').any().alias('ever_hospice'),
          (pl.col('location_category').str.to_lowercase()
             .is_in(COHORT_LOCATIONS)).any().alias('in_cohort_location'),
      ])
  )

final_df = final_df.join(
    last_location_per_hosp,
    on='hospitalization_id',
    how='left'
)

# Decedent counts by ADT location, so a site difference like a hospice unit
# is a reported number rather than something absorbed into a filter.
# Grouped by the TERMINAL location, but inclusion depends on ANY location in
# the stay: a patient whose last record is 'procedural' is still in the cohort
# if they were in an ICU earlier. Reporting only the terminal location made the
# excluded count look ~20x larger than it is.
_by_loc = (
    final_df.group_by(pl.col('last_location_category').str.to_lowercase()
                        .fill_null('(no ADT record)').alias('terminal_location'))
    .agg(pl.col('patient_id').n_unique().alias('n_decedents'),
         pl.col('patient_id').filter(pl.col('in_cohort_location').fill_null(False))
           .n_unique().alias('n_in_cohort'))
    .with_columns((pl.col('n_decedents') - pl.col('n_in_cohort')).alias('n_excluded'))
    .sort('n_decedents', descending=True)
)
_by_loc.write_csv(OUTPUT_FINAL_DIR / "decedents_by_location.csv")
print(f"Decedents by terminal ADT location "
      f"(in cohort if the stay touched any of: {', '.join(COHORT_LOCATIONS)}):")
print(f"    {'terminal location':20}{'deaths':>9}{'in cohort':>11}{'excluded':>10}")
for r in _by_loc.iter_rows(named=True):
    print(f"    {str(r['terminal_location']):20}{r['n_decedents']:>9,}"
          f"{r['n_in_cohort']:>11,}{r['n_excluded']:>10,}")

_in_cohort = pl.col('in_cohort_location').fill_null(False)
strobe_counts["2a_deaths_in_cohort_locations"] = final_df.filter(_in_cohort)['patient_id'].n_unique()
strobe_counts["2b_deaths_excluded_by_location"] = final_df.filter(~_in_cohort)['patient_id'].n_unique()
strobe_counts["2c_deaths_with_hospice"] = (
    final_df.filter(pl.col('ever_hospice').fill_null(False))['patient_id'].n_unique())

final_cohort_df = final_df.filter(_in_cohort)

# Decedents who died at a hospital the study excludes leave the cohort here, so
# every output, site totals included, is over the same hospitals. A hospital is
# excluded when its crosswalk row has include_flag false or its CMS number does
# not resolve in SRTR (config/hospital_crosswalk.yaml); a decedent there can never
# be matched to a donor. The hospital is the one on the last ADT row. Ids missing
# from the crosswalk are kept here and stop the run in the hospital-identity step.
def _hospital_in_study(row: dict) -> bool:
    return bool(row.get("include_flag") and row.get("ccn_in_srtr") and row.get("srtr_ccn_id"))


_excluded_ids = {str(r["hospital_id"]).lower().strip() for r in _XWALK
                 if r["site"] == site_name and not _hospital_in_study(r)}
_terminal_hosp = (
    adt_df.filter(pl.col("hospitalization_id").is_in(final_cohort_df["hospitalization_id"].implode())
                  & pl.col("hospital_id").is_not_null())
    .sort(["out_dttm", "hospital_id"], descending=[True, False], nulls_last=True)
    .group_by("hospitalization_id", maintain_order=True).first()
    .select(pl.col("hospitalization_id").cast(pl.Utf8),
            pl.col("hospital_id").cast(pl.Utf8).str.to_lowercase().str.strip_chars().alias("hospital_id_key"),
            pl.col("hospital_type").cast(pl.Utf8).str.to_lowercase().str.strip_chars().alias("hospital_type")))
_drop = _terminal_hosp.filter(pl.col("hospital_id_key").is_in(list(_excluded_ids)))["hospitalization_id"]
strobe_counts["2g_deaths_at_excluded_hospitals"] = int(_drop.len())
# hospital_id_key and hospital_type (academic or community, a CLIF field) stay on
# the cohort: the hospital where the patient died.
final_cohort_df = (final_cohort_df.with_columns(pl.col("hospitalization_id").cast(pl.Utf8))
                   .filter(~pl.col("hospitalization_id").is_in(_drop.implode()))
                   .join(_terminal_hosp, on="hospitalization_id", how="left"))
print(f"Decedents at hospitals excluded from the study: {_drop.len():,}")

# Age at death, in years, from birth_date. Where the birth date is missing,
# age_at_admission stands in: it understates age by the length of stay, days for
# a decedent, so at the limit it can only over-include.  A decedent with neither has no age, and two
# of the three definitions need one, so they leave the cohort here and every
# definition keeps the same denominator. CLIF stores birth_date without a
# timezone while final_death_dttm has one; align_time_zone relabels it so the
# two can be subtracted.
_birth = align_time_zone('birth_date', patient_df.schema['birth_date'],
                         final_cohort_df.schema['final_death_dttm'])
final_cohort_df = (
    final_cohort_df
    .join(patient_df.select(['patient_id', 'birth_date']), on='patient_id', how='left')
    .with_columns(((pl.col('final_death_dttm') - _birth).dt.total_days() / 365.25).alias('_age_from_birth'))
    # An age outside the plausible range, from either source, is no age: some
    # extracts record an unknown birth date as 1900-01-01 and carry the patient at 120+.
    .with_columns(
        pl.when(pl.col('_age_from_birth').is_between(AGE_MIN, AGE_MAX)).then('_age_from_birth').alias('_age_from_birth'),
        pl.when(pl.col('age_at_admission').is_between(AGE_MIN, AGE_MAX)).then('age_at_admission')
          .cast(pl.Float64).alias('_age_at_admission'))
    .with_columns(
        pl.coalesce('_age_from_birth', '_age_at_admission').alias('age_at_death'),
        pl.when(pl.col('_age_from_birth').is_not_null()).then(pl.lit('birth_date'))
          .when(pl.col('_age_at_admission').is_not_null()).then(pl.lit('age_at_admission'))
          .alias('age_source'))
    .drop('_age_from_birth', '_age_at_admission'))
strobe_counts["2h_deaths_without_age"] = final_cohort_df.filter(pl.col('age_at_death').is_null()).height
strobe_counts["2i_age_from_age_at_admission"] = final_cohort_df.filter(
    pl.col('age_source') == 'age_at_admission').height
final_cohort_df = final_cohort_df.filter(pl.col('age_at_death').is_not_null())
print(f"Decedents with no age at all (dropped): {strobe_counts['2h_deaths_without_age']:,}; "
      f"age from age_at_admission: {strobe_counts['2i_age_from_age_at_admission']:,}")

# Where each cohort decedent's time of death came from (utils/death_time.py).
# Counted on the cohort, the same patients every definition is computed on.
_src_n = dict(final_cohort_df.group_by("death_time_source").len().iter_rows())
print("Time of death in the cohort, by where it came from:")
for _s in DEATH_TIME_SOURCES:
    strobe_counts[f"2d_death_time_{_s}"] = int(_src_n.get(_s, 0))
    print(f"    {_s:34s}{_src_n.get(_s, 0):>8,}")
# A time of death before admission is a recording error. Every window before it
# is empty, so the patient reads as ineligible. Flagged without a count, because
# the count is almost always a handful; the patients are the rows of the local
# cohort file with final_death_dttm < admission_dttm.
_adm = comparable("admission_dttm", final_cohort_df.schema["admission_dttm"],
                  final_cohort_df.schema["final_death_dttm"])
if final_cohort_df.filter(pl.col("final_death_dttm") < _adm).height:
    dq_flags.append({
        "site": site_name, "flag": "death_time_before_admission", "severity": "low",
        "detail": ("At least one decedent has a time of death before the admission time of the "
                   "death hospitalization. No window before death can contain anything for them, "
                   "so they read as not ventilated and ineligible. Check death_dttm for those rows.")})
print(f"Cohort locations {COHORT_LOCATIONS}: {final_df.height:,} -> "
      f"{final_cohort_df.height:,} hospitalizations")

all_decedent_inpatient_patient_ids = final_cohort_df.select('patient_id').to_series().to_list()
all_decedent_inpatient_hosp_ids = final_cohort_df.select('hospitalization_id').to_series().to_list()
strobe_counts["2_inpatient_decedents"] = len(all_decedent_inpatient_patient_ids)

# Every criterion with a clock on it, every lab, culture and diagnosis, is taken
# from the decedent's whole ENCOUNTER BLOCK: the death hospitalization plus any
# stays linked to it by clifpy's stitch_encounters (admitted within
# study.encounter_stitch_hours of the previous discharge). A patient moved between linked stays gets a new
# hospitalization_id at each move, for example from an ICU bed to an inpatient
# hospice unit, or from a feeder hospital to a hub. Looking at the death
# hospitalization alone missed the ventilation, labs and cultures recorded in
# the earlier stay.
block_members = (
    hosp_stitched.select(["hospitalization_id", "encounter_block"])
    .join(final_cohort_df.select(["encounter_block", "patient_id", "final_death_dttm"]),
          on="encounter_block", how="inner"))
block_members_df = block_members.to_pandas()
block_ids_df = pd.DataFrame(
    {"hospitalization_id": block_members["hospitalization_id"].cast(pl.Utf8).to_list()})
strobe_counts["2f_deaths_with_linked_earlier_stay"] = (
    block_members.group_by("patient_id").len().filter(pl.col("len") > 1).height)
print(f"Decedents whose death stay is linked to an earlier stay: "
      f"{strobe_counts['2f_deaths_with_linked_earlier_stay']:,}")
# Patient-level membership for steps 02 and 03, which look up procedures,
# medications and cultures over the same stays. Kept local.
block_members.select(["patient_id", "hospitalization_id"]).write_parquet(
    OUTPUT_INTERMEDIATE_DIR / "encounter_block_members.parquet")

_progress.mark("Inpatient decedents")

################################################################################
# Weight, height and BMI; the last recorded vital
################################################################################

# Streamed through DuckDB over the stays in each cohort decedent's encounter
# block, so a patient moved between linked stays keeps the measurements charted
# in the earlier one. Only the columns used leave the query: vitals is the
# largest CLIF table, and loading it whole has run sites out of memory.
# Plausibility bounds come from config/outlier_config.yaml and are applied
# before choosing the latest reading, so it is the latest PLAUSIBLE value: a
# zeroed bed scale at the latest timestamp must not null the weight.
vitals_filepath = f"{tables_path}/clif_vitals.{file_type}"


def _latest_plausible(category: str, lo: float, hi: float, pick: str = "vital_value") -> str:
    # arg_max ordered by (time, value): the most recent plausible reading, a tie
    # at the same timestamp going to the larger value, so the result is deterministic.
    return (f"arg_max({pick}, {{'t': recorded_dttm, 'v': vital_value}}) "
            f"FILTER (WHERE vital_category = '{category}' AND vital_value BETWEEN {lo} AND {hi})")


print("Processing vitals data with DuckDB...")
# One pass over the file and one small aggregate per block: memory does not grow
# with the size of the vitals table, which matters at the largest sites. The
# last vital of any kind and the last heart rate above zero feed the death-time
# diagnostics at the end of this step; neither is a criterion.
vitals_by_block = pl.from_pandas(duckdb.sql(f"""
SELECT b.encounter_block,
       MAX(recorded_dttm) AS last_recorded_vital_dttm,
       MAX(recorded_dttm) FILTER (WHERE vital_category = 'heart_rate'
                                  AND vital_value > 0 AND vital_value BETWEEN {HR_MIN} AND {HR_MAX})
           AS last_hr_above_zero_dttm,
       {_latest_plausible('weight_kg', WEIGHT_MIN, WEIGHT_MAX)} AS last_weight_kg,
       {_latest_plausible('height_cm', HEIGHT_MIN, HEIGHT_MAX)} AS last_height_cm,
       {_latest_plausible('weight_kg', WEIGHT_MIN, WEIGHT_MAX, 'recorded_dttm')} AS last_weight_dttm,
       {_latest_plausible('height_cm', HEIGHT_MIN, HEIGHT_MAX, 'recorded_dttm')} AS last_height_dttm
FROM (SELECT hospitalization_id, recorded_dttm, LOWER(TRIM(vital_category)) AS vital_category, vital_value
      FROM read_parquet('{vitals_filepath}')) v
JOIN block_members_df b ON v.hospitalization_id = b.hospitalization_id
GROUP BY b.encounter_block
""").df()).with_columns(
    pl.col('encounter_block').cast(pl.Int32),
    (pl.col('last_weight_kg') / ((pl.col('last_height_cm') / 100) ** 2)).alias('bmi'))
for _cat, _c in (("weight_kg", "last_weight_kg"), ("height_cm", "last_height_cm")):
    _require_matches(int(vitals_by_block[_c].is_not_null().sum()), "vitals", "vital_category", _cat)
print(f"✓ Processed vitals for {len(vitals_by_block)} encounter blocks")
final_cohort_df = final_cohort_df.join(vitals_by_block, on='encounter_block', how='left')

_progress.mark("Weight, height and BMI; the last recorded vital")

################################################################################
# ADT
################################################################################

# Hospital and first-ICU length of stay over the whole encounter block, in
# fractional days of elapsed time. The death stay alone is wrong wherever a
# transfer starts a new hospitalization_id. Hospital stay is first admission
# to last discharge from the hospitalization table; ICU stay comes from ADT.
def _days(later: str, earlier: str, df: pl.DataFrame) -> pl.Expr:
    return hours_between(later, earlier, df.schema, SITE_TZ) / 24


hospital_los = (
    hosp_stitched.filter(pl.col("hospitalization_id").is_in(block_members["hospitalization_id"].to_list()))
    .group_by("encounter_block")
    .agg(pl.col("admission_dttm").min().alias("first_admission_dttm"),
         pl.col("discharge_dttm").max().alias("last_discharge_dttm")))
hospital_los = hospital_los.with_columns(
    _days("last_discharge_dttm", "first_admission_dttm", hospital_los).alias("hospital_length_of_stay_days"))

adt_in_blocks = (
    adt_stitched
    .filter(pl.col("hospitalization_id").is_in(block_members["hospitalization_id"].to_list()))
    .with_columns(pl.col("location_category").str.to_lowercase()))

# The first ICU stay is the earliest ICU row of the block. Some sites log
# more than one ICU row at that in_dttm with different out_dttm; the latest out
# is taken, so the stay is the longest one starting at that moment.
_icu = adt_in_blocks.filter(pl.col("location_category") == "icu")
first_icu = (
    _icu.join(_icu.group_by("encounter_block").agg(pl.col("in_dttm").min().alias("first_icu_in_dttm")),
              on="encounter_block")
    .filter(pl.col("in_dttm") == pl.col("first_icu_in_dttm"))
    .group_by("encounter_block")
    .agg(pl.col("first_icu_in_dttm").first(), pl.col("out_dttm").max().alias("first_icu_out_dttm")))
first_icu = first_icu.with_columns(
    _days("first_icu_out_dttm", "first_icu_in_dttm", first_icu).alias("first_icu_los_days"))

final_cohort_df = (
    final_cohort_df
    .join(hospital_los.select("encounter_block", "hospital_length_of_stay_days"), on="encounter_block", how="left")
    .join(first_icu.select("encounter_block", "first_icu_los_days"), on="encounter_block", how="left"))

_progress.mark("ADT")

################################################################################
# Age
################################################################################

# 75 or under qualifies, inclusive. age_at_death was set when the cohort was formed.
final_cohort_df = final_cohort_df.with_columns(
    (pl.col('age_at_death') <= DONOR['age_at_death_max']).alias('age_75_less'))
strobe_counts["3_age_relevant_cohort_n"] = final_cohort_df.filter(pl.col('age_75_less'))['patient_id'].n_unique()
print(f"Age {chr(8804)} {DONOR['age_at_death_max']} at death: {strobe_counts['3_age_relevant_cohort_n']:,}")

_progress.mark("Age")

################################################################################
# ICD Codes
# The CALC criteria includes the following as cause:
# - I20–I25: ischemic heart disease
# - I60–I69: cerebrovascular disease
# - V01–Y89: external causes (e.g., blunt trauma, gunshot wounds, overdose, suicide, drowning, asphyxiation)
# [Reference](https://www.cms.gov/files/document/112020-opo-final-rule-cms-3380-f.pdf)
# We also flag contraindicating cancers using the ICD-10 code ranges in utils/codes/icd10_contraindications.csv
################################################################################

hospial_dx_filepath = f"{tables_path}/clif_hospital_diagnosis.{file_type}"

# Diagnosis coverage: decedents with at least one diagnosis row in any stay of
# their encounter block. Streamed through DuckDB; a polars read and join
# segfaulted on Windows at sites with large hospital_diagnosis tables.
_n_with_dx = duckdb.sql(f"""
    SELECT COUNT(DISTINCT b.patient_id)
    FROM read_parquet('{hospial_dx_filepath}') hd
    JOIN block_members_df b
      ON CAST(hd.hospitalization_id AS VARCHAR) = CAST(b.hospitalization_id AS VARCHAR)
""").fetchone()[0]
strobe_counts["5_decedents_with_any_diagnosis"] = _n_with_dx
print(f"Decedents with a diagnosis row in any stay of their encounter block: "
      f"{_n_with_dx:,} of {len(all_decedent_inpatient_patient_ids):,}")

# ---- 0) Load the contraindication list ----
# Code prefixes: C34 covers C34.11. The list follows the OPTN eligible-death
# cancer exclusions; see guides/contraindications.md. Sepsis is not on it and
# excludes no one (settled 2026-09-19).
_CONTRA_SQL = contraindication_sql("dx_norm", CONTRA_CODES)
print(f"Contraindication list: {len(CONTRA_CODES)} code prefixes")

# ---- 0b) Load comorbidity prefixes (HCV, HTN, DM, Hx CVA) from CSV ----
# These are PREFIX matches (3-4 char ICD blocks) — e.g. 'i10' matches any
# code starting with i10 (i10, i109, i1010, etc.). Lowercase + no periods.
comorbidities_df = pl.read_csv(PROJECT_ROOT / CRIT["reporting"]["comorbidities_file"])
comorbidity_prefixes: dict[str, list[str]] = {}
for row in comorbidities_df.iter_rows(named=True):
    prefix = str(row["code_prefix"]).strip().lower().replace(".", "")
    key = str(row["comorbidity"]).strip().lower()
    comorbidity_prefixes.setdefault(key, []).append(prefix)
print(f"Loaded comorbidity prefixes: " +
      ", ".join(f"{k}={len(v)}" for k, v in comorbidity_prefixes.items()))

# ---- 1) Compute ICD-10 cause + comorbidity flags via DuckDB SQL ----

# Build SQL clauses for each comorbidity (HCV/HTN/DM/CVA) — prefix LIKE chain
def _comorbidity_clause(key: str, prefixes: list[str]) -> str:
    likes = " OR ".join(f"dx_norm LIKE '{p}%'" for p in prefixes)
    return (
        f"CASE WHEN sys IN ('icd10','icd10cm') AND ({likes}) "
        f"THEN true ELSE false END AS icd10_{key}"
    )

comorbidity_select_clauses = ",\n        ".join(
    _comorbidity_clause(k, ps) for k, ps in comorbidity_prefixes.items()
)
comorbidity_bool_or_clauses = ",\n    ".join(
    f"BOOL_OR(icd10_{k}) AS icd10_{k}" for k in comorbidity_prefixes
)

# The three CALC cause ranges from 42 CFR 486.302, written once and reused for
# every diagnosis-position variant so the three cannot drift apart.
_CAUSE_SQL = "sys IN ('icd10','icd10cm') AND (" + " OR ".join(_RANGE_SQL.values()) + ")"

query = f"""
WITH hospital_dx_normalized AS (
    SELECT
        hospitalization_id,
        LOWER(REGEXP_REPLACE(CAST(diagnosis_code AS VARCHAR), '[^A-Za-z0-9]', '', 'g')) AS dx_norm,
        -- The FORMAT is normalised the same way as the code. Matching
        -- LOWER(diagnosis_code_format) alone against 'icd10cm' silently dropped
        -- every row at a site writing 'ICD-10-CM', taking CALC, the
        -- contraindications and the comorbidities to zero with no error.
        REGEXP_REPLACE(LOWER(CAST(diagnosis_code_format AS VARCHAR)), '[^a-z0-9]', '', 'g') AS sys,
        LOWER(CAST(diagnosis_primary AS VARCHAR)) IN ('1','true','t','y','yes') AS is_primary,
        LOWER(CAST(poa_present AS VARCHAR)) IN ('1','true','t','y','yes') AS is_poa
    FROM read_parquet('{hospial_dx_filepath}')
    WHERE CAST(hospitalization_id AS VARCHAR) IN (SELECT hospitalization_id FROM block_ids_df)
),
hospital_dx_flags AS (
    SELECT
        hospitalization_id,
        CASE WHEN sys IN ('icd10','icd10cm') AND {_ISCHEMIC_SQL} THEN true ELSE false END AS icd10_ischemic,
        CASE WHEN sys IN ('icd10','icd10cm') AND {_CEREBRO_SQL} THEN true ELSE false END AS icd10_cerebro,
        CASE WHEN sys IN ('icd10','icd10cm') AND {_EXTERNAL_SQL} THEN true ELSE false END AS icd10_external,
        -- Injury codes (S00-T88). Not a criterion: used only to detect an extract
        -- that carries injuries but has dropped the external-cause codes.
        CASE WHEN sys IN ('icd10','icd10cm') AND REGEXP_MATCHES(dx_norm, '^[st][0-9]{{2}}') THEN true ELSE false END AS icd10_injury,
        CASE WHEN sys IN ('icd10','icd10cm') AND {_CONTRA_SQL} THEN true ELSE false END AS icd10_contraindication,
        -- CALC cause of death, in three diagnosis-position variants. All three
        -- are computed every run; calc.diagnosis_position picks which becomes
        -- calc_flag. See config/donor_criteria.yaml and DECISIONS D-46.
        CASE WHEN {_CAUSE_SQL} THEN true ELSE false END AS calc_cause_any,
        CASE WHEN ({_CAUSE_SQL}) AND is_primary THEN true ELSE false END AS calc_cause_primary,
        CASE WHEN ({_CAUSE_SQL}) AND is_primary AND is_poa THEN true ELSE false END AS calc_cause_primary_poa,
        -- Brain death. Reporting only: deliberately NOT a CLIF-donor criterion,
        -- because it is also the face-validity check and using it both ways
        -- would be circular.
        CASE WHEN sys IN ('icd10','icd10cm') AND dx_norm = 'g9382' THEN true ELSE false END AS icd10_brain_death,
        {comorbidity_select_clauses}
    FROM hospital_dx_normalized
),
hospital_dx_with_patient AS (
    SELECT h.*, hosp.patient_id
    FROM hospital_dx_flags h
    LEFT JOIN hospitalization_df hosp ON h.hospitalization_id = hosp.hospitalization_id
)
SELECT
    patient_id,
    BOOL_OR(icd10_ischemic) AS icd10_ischemic,
    BOOL_OR(icd10_cerebro) AS icd10_cerebro,
    BOOL_OR(icd10_external) AS icd10_external,
    BOOL_OR(icd10_injury) AS icd10_injury,
    BOOL_OR(icd10_contraindication) AS icd10_contraindication,
    BOOL_OR(calc_cause_any) AS calc_cause_any,
    BOOL_OR(calc_cause_primary) AS calc_cause_primary,
    BOOL_OR(calc_cause_primary_poa) AS calc_cause_primary_poa,
    BOOL_OR(icd10_brain_death) AS icd10_brain_death,
    {comorbidity_bool_or_clauses}
FROM hospital_dx_with_patient
WHERE patient_id IS NOT NULL
GROUP BY patient_id
"""

print("Processing ICD flags with DuckDB...")
patient_cause_flags = pl.from_pandas(duckdb.sql(query).df())
print(f"✓ Processed {len(patient_cause_flags)} patients")
for col in ("icd10_ischemic", "icd10_cerebro", "icd10_external", "icd10_contraindication",
            "calc_cause_any", "calc_cause_primary", "calc_cause_primary_poa",
            "icd10_brain_death",
            *[f"icd10_{k}" for k in comorbidity_prefixes]):
    print(f"  {col}: {patient_cause_flags[col].sum()}")

# An extract can carry injury diagnoses and still hold no external-cause codes:
# V, W, X and Y codes often sit in a separate claims field that the ETL drops.
# CALC then loses its external-cause arm with no error, and the site's CALC
# count is not comparable with a site that kept those codes. The site cannot
# fix this in config, so the run continues and the flag ships with the results.
_n_external = int(patient_cause_flags["icd10_external"].sum())
_n_injury = int(patient_cause_flags["icd10_injury"].sum())
patient_cause_flags = patient_cause_flags.drop("icd10_injury")
if _n_external == 0 and _n_injury > 0:
    _lo, _hi = CALC_CFG['cause_icd10_prefixes']["external_causes"]
    dq_flags.append({
        "site": site_name, "flag": "calc_external_cause_codes_absent", "severity": "high",
        "detail": (f"No decedent carries an external-cause code ({_lo}-{_hi}) although injury "
                   f"codes (S00-T88) are present, so clif_hospital_diagnosis appears to omit "
                   f"external-cause codes. CALC at this site reflects ischemic heart disease "
                   f"and cerebrovascular disease only and is not comparable with sites that "
                   f"record external causes.")})
    print(f"  WARNING  no external-cause codes ({_lo}-{_hi}) although injury codes are present: "
          f"CALC lacks its external-cause arm at this site (see data_quality_flags.csv)")
_n_without_dx = len(all_decedent_inpatient_patient_ids) - _n_with_dx
strobe_counts["5b_decedents_without_any_diagnosis"] = _n_without_dx

# Join flags to final_df on patient_id; fill null flags to False ----
_comorbidity_fill = [pl.col(f"icd10_{k}").fill_null(False) for k in comorbidity_prefixes]
final_cohort_df = (
    final_cohort_df
    .join(patient_cause_flags, on="patient_id", how="left")
    .with_columns([
        pl.col("icd10_ischemic").fill_null(False),
        pl.col("icd10_cerebro").fill_null(False),
        pl.col("icd10_external").fill_null(False),
        pl.col("icd10_contraindication").fill_null(False),
        pl.col("calc_cause_any").fill_null(False),
        pl.col("calc_cause_primary").fill_null(False),
        pl.col("calc_cause_primary_poa").fill_null(False),
        pl.col("icd10_brain_death").fill_null(False),
        *_comorbidity_fill,
    ])
)

# CALC cause of death, all three diagnosis-position variants, before age is
# applied. Reported every run so the sensitivity analyses need no re-run.
CALC_POSITIONS = {"any": "calc_cause_any",
                  "primary": "calc_cause_primary",
                  "primary_poa": "calc_cause_primary_poa"}
for _name, _col in CALC_POSITIONS.items():
    strobe_counts[f"calc_cause_{_name}"] = (
        final_cohort_df.filter(pl.col(_col))["patient_id"].n_unique())
strobe_counts["calc_cause"] = strobe_counts["calc_cause_any"]   # legacy key

_progress.mark("ICD Codes")

################################################################################
# CALC Criteria
# CMS adopts the Cause, Age, and Location-consistent (CALC) method to define “death consistent with organ donation” for donor-potential calculations:
# - **Age**: deaths ≤75 years
# - **Location**: inpatient deaths (death occurs in the hospital)
# - **Cause** (ICD-10-CM, inclusion ranges):
# - I20–I25: ischemic heart disease
# - I60–I69: cerebrovascular disease
# - V01–Y89: external causes (e.g., blunt trauma, gunshot wounds, overdose, suicide, drowning, asphyxiation)
# [Reference](https://www.cms.gov/files/document/112020-opo-final-rule-cms-3380-f.pdf)
################################################################################

# Which diagnosis position defines the cause of death (config/donor_criteria.yaml).
_POSITION = str(CALC_CFG['diagnosis_position']).lower()

print(f"CALC diagnosis position: {_POSITION} -> {CALC_POSITIONS[_POSITION]}")
for _name, _col in CALC_POSITIONS.items():
    _n = final_cohort_df.filter(pl.col('age_75_less') & pl.col(_col))["patient_id"].n_unique()
    strobe_counts[f"calc_qualified_{_name}"] = _n
    print(f"  age <= {DONOR['age_at_death_max']:g} and cause ({_name:11}): {_n:>7,}"
          + ("   <- calc_flag" if _name == _POSITION else ""))

final_cohort_df = final_cohort_df.with_columns(
    (
        (pl.col('age_75_less')) &
        (pl.col(CALC_POSITIONS[_POSITION]))
    ).alias('calc_flag')
)

# Count for STROBE tracking
calc_qualified_n = final_cohort_df.filter(pl.col('calc_flag'))['patient_id'].n_unique()
strobe_counts["calc_qualified"] = calc_qualified_n

print(f"\nCALC flag qualified: {calc_qualified_n} patients")

_progress.mark("CALC Criteria")

################################################################################
# IMV — streamed via DuckDB on clif_respiratory_support.parquet
################################################################################

resp_filepath = f"{tables_path}/clif_respiratory_support.{file_type}"
print("Processing IMV data with DuckDB...")


imv_query = imv_before_death_sql(f"read_parquet('{resp_filepath}')", "block_members_df", DONOR['imv_hours_before_death'])

resp_expired_cohort = pl.from_pandas(duckdb.sql(imv_query).df())
_require_matches(resp_expired_cohort.height, "respiratory_support", "device_category", "imv")

# imv_48hr_expire: at least one IMV record in the window before death. It is
# the Ventilated Patient definition and the IMV criterion of CLIF-donor. The
# hours from the last such record to death feed ventilation_timing.csv only.
final_cohort_df = (
    final_cohort_df
    .join(resp_expired_cohort.select(["patient_id", pl.col("hr_2death_last_imv").alias("hours_last_imv_to_death")]),
          on="patient_id", how="left")
    .with_columns(pl.col("hours_last_imv_to_death").is_not_null().alias("imv_48hr_expire")))
# With the age limit: the second step of the CLIF-donor CONSORT.
strobe_counts["6_died_while_imv_age_le75"] = final_cohort_df.filter(
    pl.col("imv_48hr_expire") & pl.col("age_75_less"))["patient_id"].n_unique()
print(f"✓ IMV within {DONOR['imv_hours_before_death']:g} h of death and age <= {DONOR['age_at_death_max']:g}: "
      f"{strobe_counts['6_died_while_imv_age_le75']:,}")

_progress.mark("IMV — streamed via DuckDB on clif_respiratory_support.parquet")

################################################################################
# Organ quality check
# Pass the potential organ quality assessment check (independent assessment) using last recorded lab values, as defined by CMS
# * Kidney: recorded creatinine, cr  <4  AND not on CRRT
# * Liver: recorded TB, AST, ALT and Total bilirubin < 4, AST < 700, AND ALT< 700
# * BMI <=50
################################################################################

crrt_filepath = f"{tables_path}/clif_crrt_therapy.{file_type}"
labs_filepath = f"{tables_path}/clif_labs.{file_type}"

# ============================================
# CRRT within 48h of death — streamed via DuckDB
# ============================================
print("Processing CRRT data with DuckDB...")

crrt_query = f"""
WITH crrt_data AS (
    SELECT
        hospitalization_id,
        recorded_dttm
    FROM read_parquet('{crrt_filepath}')
    WHERE hospitalization_id IN (SELECT hospitalization_id FROM block_members_df)
),
crrt_with_death AS (
    SELECT
        f.patient_id,
        c.recorded_dttm,
        f.final_death_dttm,
        EXTRACT(EPOCH FROM (f.final_death_dttm - c.recorded_dttm)) / 3600 AS hrs_before_death
    FROM crrt_data c
    INNER JOIN block_members_df f ON c.hospitalization_id = f.hospitalization_id
    WHERE c.recorded_dttm <= f.final_death_dttm
)
SELECT DISTINCT patient_id
FROM crrt_with_death
WHERE hrs_before_death <= {DONOR['kidney']['exclude_if_crrt_within_hours']} AND hrs_before_death >= 0
"""

crrt_48h_result = pl.from_pandas(duckdb.sql(crrt_query).df())
# No CRRT row for anyone in the cohort reads as "nobody was on dialysis", which
# makes more patients kidney-eligible. That is correct at a hospital that does
# not provide CRRT and wrong where the table is empty or keyed differently, and
# the two cannot be told apart from here, so it is flagged rather than fatal.
_n_crrt_rows = duckdb.sql(f"""
    SELECT COUNT(*) FROM read_parquet('{crrt_filepath}')
    WHERE hospitalization_id IN (SELECT hospitalization_id FROM block_members_df)
""").fetchone()[0]
if _n_crrt_rows == 0:
    dq_flags.append({
        "site": site_name, "flag": "no_crrt_records_for_cohort", "severity": "high",
        "detail": ("No decedent has any row in clif_crrt_therapy, so every patient is read as "
                   "not on CRRT and kidney eligibility may be over-counted. Expected only at "
                   "a hospital that does not provide CRRT.")})
    print("  WARNING  no clif_crrt_therapy rows for any decedent: everyone is read as not on "
          "CRRT (see data_quality_flags.csv)")
on_crrt_flag = crrt_48h_result.with_columns(
    pl.lit(True).alias('on_crrt_48h_before_death')
)
final_cohort_df = final_cohort_df.join(
    on_crrt_flag, on='patient_id', how='left'
).with_columns(pl.col('on_crrt_48h_before_death').fill_null(False))

on_crrt_n = final_cohort_df.filter(pl.col('on_crrt_48h_before_death'))['patient_id'].n_unique()
print(f"✓ Patients on CRRT within {DONOR['kidney']['exclude_if_crrt_within_hours']:g}h before death: {on_crrt_n}")

# ============================================
# Organ-quality labs (creatinine, bili, AST, ALT) — streamed via DuckDB
# ============================================
print("Processing Labs data with DuckDB...")

_LAB_VALUE_USABLE = usable_number_sql("l.lab_value_numeric")
# Plausibility bounds from config/outlier_config.yaml, applied BEFORE the
# "last value" ranking, as for weight and height: the last PLAUSIBLE result
# counts, and a unit error or placeholder at the latest timestamp does not.
_LAB_BOUNDS = {c: outlier_range("labs", "lab_value_numeric", c, OUTLIER_CONFIG)
               for c in ("creatinine", "bilirubin_total", "ast", "alt", "bun", "sodium")}
_LAB_PLAUSIBLE = " OR ".join(f"(l.lab_category = '{c}' AND l.lab_value_numeric BETWEEN {lo} AND {hi})"
                             for c, (lo, hi) in _LAB_BOUNDS.items())
labs_query = f"""
WITH labs_data AS (
    SELECT
        hospitalization_id,
        lab_collect_dttm,
        LOWER(TRIM(lab_category)) AS lab_category,
        lab_value_numeric
    FROM read_parquet('{labs_filepath}')
    WHERE hospitalization_id IN (SELECT hospitalization_id FROM block_members_df)
),
labs_with_death AS (
    SELECT
        l.hospitalization_id,
        l.lab_collect_dttm,
        l.lab_category,
        l.lab_value_numeric,
        f.patient_id,
        f.final_death_dttm
    FROM labs_data l
    INNER JOIN block_members_df f ON l.hospitalization_id = f.hospitalization_id
    WHERE l.lab_collect_dttm <= f.final_death_dttm
      -- Only results that carry a number can be the "last value". A text-only
      -- result (haemolysed, see note) or a NaN at the latest timestamp otherwise
      -- won the ranking below and nulled the value, which reads as missing,
      -- which is ineligible.
      AND {_LAB_VALUE_USABLE}
      AND ({_LAB_PLAUSIBLE})
),
latest_creatinine AS (
    -- Every "last value before death" selection breaks ties deterministically.
    -- Without a tiebreaker two results at the same timestamp were picked
    -- arbitrarily and the same code gave different counts on consecutive runs
    SELECT
        patient_id,
        lab_value_numeric AS creatinine_value,
        lab_collect_dttm AS creatinine_dttm
    FROM (
        SELECT
            patient_id,
            lab_value_numeric,
            lab_collect_dttm,
            ROW_NUMBER() OVER (PARTITION BY patient_id
                               ORDER BY lab_collect_dttm DESC, lab_value_numeric DESC) AS rn
        FROM labs_with_death
        WHERE lab_category = 'creatinine'
    ) ranked
    WHERE rn = 1
),
latest_liver AS (
    SELECT
        patient_id,
        MAX(CASE WHEN lab_category = 'bilirubin_total' THEN lab_value_numeric END) AS bilirubin_total_value,
        MAX(CASE WHEN lab_category = 'bilirubin_total' THEN lab_collect_dttm END) AS bilirubin_total_dttm,
        MAX(CASE WHEN lab_category = 'ast' THEN lab_value_numeric END) AS ast_value,
        MAX(CASE WHEN lab_category = 'ast' THEN lab_collect_dttm END) AS ast_dttm,
        MAX(CASE WHEN lab_category = 'alt' THEN lab_value_numeric END) AS alt_value,
        MAX(CASE WHEN lab_category = 'alt' THEN lab_collect_dttm END) AS alt_dttm,
        -- BUN and sodium added for v6 Table 1; reporting only, no criterion
        -- depends on them.
        MAX(CASE WHEN lab_category = 'bun' THEN lab_value_numeric END) AS bun_value,
        MAX(CASE WHEN lab_category = 'bun' THEN lab_collect_dttm END) AS bun_dttm,
        MAX(CASE WHEN lab_category = 'sodium' THEN lab_value_numeric END) AS sodium_value,
        MAX(CASE WHEN lab_category = 'sodium' THEN lab_collect_dttm END) AS sodium_dttm
    FROM (
        SELECT
            patient_id,
            lab_category,
            lab_value_numeric,
            lab_collect_dttm,
            ROW_NUMBER() OVER (PARTITION BY patient_id, lab_category
                               ORDER BY lab_collect_dttm DESC, lab_value_numeric DESC) AS rn
        FROM labs_with_death
        WHERE lab_category IN ('bilirubin_total', 'ast', 'alt', 'bun', 'sodium')
    ) ranked
    WHERE rn = 1
    GROUP BY patient_id
)
SELECT
    f.patient_id,
    c.creatinine_value,
    c.creatinine_dttm,
    l.bilirubin_total_value,
    l.bilirubin_total_dttm,
    l.ast_value,
    l.ast_dttm,
    l.alt_value,
    l.alt_dttm,
    l.bun_value,
    l.bun_dttm,
    l.sodium_value,
    l.sodium_dttm
FROM (SELECT DISTINCT patient_id FROM block_members_df) f
LEFT JOIN latest_creatinine c ON f.patient_id = c.patient_id
LEFT JOIN latest_liver l ON f.patient_id = l.patient_id
"""

organ_labs = pl.from_pandas(duckdb.sql(labs_query).df())
print(f"✓ Organ labs loaded: {len(organ_labs)} patients")
for _cat in ("creatinine", "bilirubin_total", "ast", "alt"):
    _require_matches(int(organ_labs[f"{_cat}_value"].is_not_null().sum()),
                     "labs", "lab_category", _cat)
print(f"  Patients with creatinine: {organ_labs.filter(pl.col('creatinine_value').is_not_null())['patient_id'].n_unique()}")
print(f"  Patients with bilirubin: {organ_labs.filter(pl.col('bilirubin_total_value').is_not_null())['patient_id'].n_unique()}")
print(f"  Patients with AST: {organ_labs.filter(pl.col('ast_value').is_not_null())['patient_id'].n_unique()}")
print(f"  Patients with ALT: {organ_labs.filter(pl.col('alt_value').is_not_null())['patient_id'].n_unique()}")
print(f"  Patients with BUN: {organ_labs.filter(pl.col('bun_value').is_not_null())['patient_id'].n_unique()}")
print(f"  Patients with sodium: {organ_labs.filter(pl.col('sodium_value').is_not_null())['patient_id'].n_unique()}")

# Join organ_labs onto final_cohort_df by patient_id
final_cohort_df = final_cohort_df.join(
    organ_labs, on='patient_id', how='left', suffix='_organlab'
)
print(f"Final cohort with organ labs shape: {final_cohort_df.shape}")

# ============================================
# Create organ quality assessment flags
# ============================================
final_cohort_df = final_cohort_df.with_columns([
    # Kidney criteria: creatinine < 4 AND not on CRRT
    (
        (pl.col('creatinine_value').is_not_null()) &
        (pl.col('creatinine_value') < DONOR['kidney']['creatinine_max']) &
        (~pl.col('on_crrt_48h_before_death'))
    ).alias('kidney_eligible'),

    # Liver criteria: all three labs recorded AND values within limits
    (
        (pl.col('bilirubin_total_value').is_not_null()) &
        (pl.col('ast_value').is_not_null()) &
        (pl.col('alt_value').is_not_null()) &
        (pl.col('bilirubin_total_value') < DONOR['liver']['total_bilirubin_max']) &
        (pl.col('ast_value') < DONOR['liver']['ast_max']) &
        (pl.col('alt_value') < DONOR['liver']['alt_max'])
    ).alias('liver_eligible'),

    # BMI criteria: <= 50
    (
        (pl.col('bmi').is_not_null()) &
        (pl.col('bmi') <= DONOR['bmi_max'])
    ).alias('bmi_eligible'),
])

# Overall: (kidney OR liver) AND BMI - done in separate call
final_cohort_df = final_cohort_df.with_columns([
    (
        (
            pl.col('kidney_eligible') | pl.col('liver_eligible')
        ) &
        pl.col('bmi_eligible')
    ).alias('organ_check_pass')
])

# Count for STROBE tracking
kidney_eligible_n = final_cohort_df.filter(pl.col('kidney_eligible'))['patient_id'].n_unique()
liver_eligible_n = final_cohort_df.filter(pl.col('liver_eligible'))['patient_id'].n_unique()
bmi_eligible_n = final_cohort_df.filter(pl.col('bmi_eligible'))['patient_id'].n_unique()
organ_check_pass_n = final_cohort_df.filter(pl.col('organ_check_pass'))['patient_id'].n_unique()

strobe_counts["organ_kidney_eligible"] = kidney_eligible_n
strobe_counts["organ_liver_eligible"] = liver_eligible_n
strobe_counts["organ_bmi_eligible"] = bmi_eligible_n
strobe_counts["organ_check_pass"] = organ_check_pass_n

print(f"\nOrgan Quality Assessment:")
print(f"  Kidney eligible: {kidney_eligible_n} patients")
print(f"  Liver eligible: {liver_eligible_n} patients")
print(f"  BMI eligible: {bmi_eligible_n} patients")
print(f"  Overall organ check pass: {organ_check_pass_n} patients")

_progress.mark("Organ quality check")

################################################################################
# Microbiology
# Identify negative blood cultures and patients with no cultures in last 48h
################################################################################

# Microbiology — streamed via DuckDB on clif_microbiology_culture.parquet
print("Processing microbiology data with DuckDB...")

micro_query = f"""
WITH blood_cultures AS (
    SELECT
        hospitalization_id,
        collect_dttm,
        organism_category
    FROM read_parquet('{tables_path}/clif_microbiology_culture.{file_type}')
    WHERE LOWER(TRIM(fluid_category)) = 'blood_buffy'
        AND LOWER(TRIM(method_category)) = 'culture'
        AND hospitalization_id IN (SELECT hospitalization_id FROM block_members_df)
),
cultures_with_death AS (
    SELECT
        f.patient_id,
        b.collect_dttm,
        b.organism_category,
        f.final_death_dttm,
        EXTRACT(EPOCH FROM (f.final_death_dttm - b.collect_dttm)) / 3600 AS hrs_before_death
    FROM blood_cultures b
    INNER JOIN block_members_df f ON b.hospitalization_id = f.hospitalization_id
    WHERE b.collect_dttm IS NOT NULL
),
cultures_48h AS (
    SELECT
        *,
        CASE
            WHEN LOWER(organism_category) LIKE '%no_growth%'
                OR organism_category IS NULL
                OR LOWER(organism_category) = ''
            THEN true ELSE false
        END AS is_negative_culture
    FROM cultures_with_death
    WHERE hrs_before_death >= 0 AND hrs_before_death <= {DONOR['contraindications']['positive_blood_culture_hours']}
),
positive_cultures AS (
    SELECT DISTINCT patient_id
    FROM cultures_48h
    WHERE is_negative_culture = false
)
SELECT
    f.patient_id,
    CASE WHEN p.patient_id IS NULL THEN true ELSE false END AS no_positive_culture_48hrs
FROM (SELECT DISTINCT patient_id FROM block_members_df) f
LEFT JOIN positive_cultures p ON f.patient_id = p.patient_id
"""

# A cohort of in-hospital deaths with no blood culture at all means the filter
# matched nothing, and then every patient would pass as "no positive culture".
_n_blood_cx = duckdb.sql(f"""
    SELECT COUNT(*) FROM read_parquet('{tables_path}/clif_microbiology_culture.{file_type}')
    WHERE LOWER(TRIM(fluid_category)) = 'blood_buffy'
      AND LOWER(TRIM(method_category)) = 'culture'
      AND hospitalization_id IN (SELECT hospitalization_id FROM block_members_df)
""").fetchone()[0]
_require_matches(_n_blood_cx, "microbiology_culture", "fluid_category",
                 "blood_buffy' with method_category = 'culture")
no_positive_culture_flag = pl.from_pandas(duckdb.sql(micro_query).df())
final_cohort_df = final_cohort_df.join(
    no_positive_culture_flag, on='patient_id', how='left'
).with_columns(pl.col('no_positive_culture_48hrs').fill_null(False))

# STROBE tracking
no_positive_culture_n = final_cohort_df.filter(pl.col('no_positive_culture_48hrs'))['patient_id'].n_unique()
positive_culture_n = final_cohort_df.filter(~pl.col('no_positive_culture_48hrs'))['patient_id'].n_unique()
strobe_counts["no_positive_culture_48hrs"] = no_positive_culture_n
strobe_counts["positive_culture_48hrs"] = positive_culture_n
print(f"  Patients with no positive cultures in last {DONOR['contraindications']['positive_blood_culture_hours']:g}h: {no_positive_culture_n}")
print(f"  Patients with positive cultures in last {DONOR['contraindications']['positive_blood_culture_hours']:g}h: {positive_culture_n}")

_progress.mark("Microbiology")

################################################################################
# CLIF Eligible Donor
# Medically eligible potential deceased abdominal organ donor. Every threshold
# is read from config/donor_criteria.yaml; the numbers here are its values.
# * From the cohort: in-hospital deaths whose stay touched one of
#   study.cohort_locations (ED, ward, stepdown, ICU, hospice)
# * Age at death 75 or under (inclusive)
# * At least one IMV record in the 48 h before the time of death
# * No contraindicating cancer diagnosis on any stay of the death encounter:
#   every row of utils/codes/icd10_contraindications.csv
# * No positive blood culture collected in the 48 h before death
# * Organ quality, from the last recorded values before death:
#   kidney: creatinine < 4 and no CRRT in the 48 h before death
#   liver:  total bilirubin < 4, AST < 700 and ALT < 700, all three recorded
#   BMI 50 or under; and (kidney OR liver) AND BMI
################################################################################

# ============================================
# Create CLIF-eligible-donors flag
# ============================================

final_cohort_df = final_cohort_df.with_columns([
    # Overall CLIF-eligible-donors flag
    (
        # age 75 or under
        (pl.col('age_75_less')) &
        # IMV within 48 h of death
        (pl.col('imv_48hr_expire')) &
        # no contraindicating cancer (utils/codes/icd10_contraindications.csv)
        (~pl.col('icd10_contraindication')) &
        # no positive blood culture within 48 h
        (pl.col('no_positive_culture_48hrs')) &
        # (kidney OR liver) AND BMI
        (pl.col('organ_check_pass'))
    ).alias('clif_eligible_donors')
])

# Count for STROBE tracking
clif_eligible_n = final_cohort_df.filter(pl.col('clif_eligible_donors'))['patient_id'].n_unique()
strobe_counts["clif_eligible_donors"] = clif_eligible_n

_progress.mark("CLIF Eligible Donor")

################################################################################
# Patient assessments
################################################################################

# GCS total and RASS for Table 2: the last value recorded before death, with
# no time limit, the same rule as the labs. Neither is a criterion. They used
# to be the value closest to death on either side; at the sites checked no
# decedent has one only after death, so this changes which value is taken for
# the ~2% whose nearest record was after death, and reaches the same patients.
print("Processing patient assessments with DuckDB...")

assessments_query = f"""
WITH assessments_filtered AS (
    SELECT
        hospitalization_id,
        recorded_dttm,
        LOWER(TRIM(assessment_category)) AS assessment_category,
        numerical_value
    FROM read_parquet('{tables_path}/clif_patient_assessments.{file_type}')
    WHERE ((LOWER(TRIM(assessment_category)) = 'gcs_total' AND numerical_value BETWEEN {GCS_MIN} AND {GCS_MAX})
        OR (LOWER(TRIM(assessment_category)) = 'rass' AND numerical_value BETWEEN {RASS_MIN} AND {RASS_MAX}))
        AND hospitalization_id IN (SELECT hospitalization_id FROM block_members_df)
),
before_death AS (
    SELECT
        f.patient_id,
        a.assessment_category,
        a.numerical_value,
        a.recorded_dttm
    FROM assessments_filtered a
    INNER JOIN block_members_df f ON a.hospitalization_id = f.hospitalization_id
    WHERE a.recorded_dttm <= f.final_death_dttm
),
last_per_category AS (
    SELECT
        patient_id,
        assessment_category,
        numerical_value,
        recorded_dttm,
        ROW_NUMBER() OVER (
            PARTITION BY patient_id, assessment_category
            ORDER BY recorded_dttm DESC, numerical_value ASC
        ) AS rn
    FROM before_death
)
SELECT
    patient_id,
    MAX(CASE WHEN assessment_category = 'gcs_total' THEN numerical_value END) AS gcs_total_value,
    MAX(CASE WHEN assessment_category = 'rass'      THEN numerical_value END) AS rass_value,
    MAX(CASE WHEN assessment_category = 'gcs_total' THEN recorded_dttm END) AS gcs_total_dttm,
    MAX(CASE WHEN assessment_category = 'rass'      THEN recorded_dttm END) AS rass_dttm
FROM last_per_category
WHERE rn = 1
GROUP BY patient_id
"""

if "patient_assessments" in OPTIONAL_UNUSABLE:
    # Optional table. Without it GCS and RASS are null for everyone and their
    # Table 2 rows report as unavailable; no definition depends on them.
    patient_gcs_rass = pl.DataFrame(schema={
        "patient_id": final_cohort_df.schema["patient_id"],
        "gcs_total_value": pl.Float64, "rass_value": pl.Float64,
        "gcs_total_dttm": final_cohort_df.schema["final_death_dttm"],
        "rass_dttm": final_cohort_df.schema["final_death_dttm"]})
    print("  patient assessments SKIPPED (clif_patient_assessments absent or incomplete): "
          "GCS and RASS unavailable")
else:
    patient_gcs_rass = pl.from_pandas(duckdb.sql(assessments_query).df())
    print(f"✓ Processed assessments for {len(patient_gcs_rass)} patients")

final_cohort_df = final_cohort_df.join(
    patient_gcs_rass, on='patient_id', how='left'
)

# ================================================================================
# ENSURE PATIENT-LEVEL ANALYSIS
# ================================================================================
print("\n" + "="*80)
print("FINALIZING PATIENT-LEVEL COHORT")
print("="*80)

# Sanity check — early dedup upstream means we should already be 1 row/patient.
# If this assertion fails, a downstream join introduced duplicates (would need a fix).
n_patients_final = final_cohort_df['patient_id'].n_unique()
n_rows_final = len(final_cohort_df)
assert n_patients_final == n_rows_final, (
    f"CRITICAL: Expected one row per patient (early dedup invariant violated). "
    f"{n_rows_final} rows but {n_patients_final} unique patients."
)
print(f"✓ One-row-per-patient invariant holds: {n_patients_final:,} patients = {n_rows_final:,} rows")

print(f"\n✓ Final verification passed: {n_patients_final:,} unique patients")
print(f"Final cohort shape: {final_cohort_df.shape}")
print("="*80 + "\n")

# Canonical dtypes on write so cohorts from different sites concatenate.
from utils.dtypes import write_parquet as _write_parquet
_write_parquet(final_cohort_df, OUTPUT_INTERMEDIATE_DIR / "final_cohort_df.parquet", SITE_TZ)
# Long, not wide. One row per count, so three sites concatenate even when a
# site is missing an optional table and therefore a metric.
_strobe_long = pl.DataFrame({
    "site": [site_name] * len(strobe_counts),
    "order": list(range(1, len(strobe_counts) + 1)),
    "metric": list(strobe_counts.keys()),
    "n": [int(v) for v in strobe_counts.values()],
})
_strobe_long.write_csv(OUTPUT_FINAL_DIR / "strobe_counts.csv")

_progress.mark("Patient assessments")

################################################################################
# Definitions that need the finished cohort
################################################################################

from utils.audit import StageAudit                                 
_audit = StageAudit(site_name, OUTPUT_FINAL_DIR, label="definitions")

# ── Ventilated Patient ───────────────────────────────────────────────────────
# Both variants are computed: Table 1 of the manuscript says "No restrictions",
# so the no-age-limit flag is the reported one, but the age-capped variant is
# kept so the difference is a reported number rather than a re-run.
final_cohort_df = final_cohort_df.with_columns([
    pl.col("imv_48hr_expire").alias("ventilated_patient_no_age_limit"),
    (pl.col("imv_48hr_expire") & pl.col("age_75_less")).alias("ventilated_patient_age_le75"),
    pl.col("imv_48hr_expire").alias("ventilated_patient"),
])
_audit.record("20_ventilated_no_age_limit", final_cohort_df,
              final_cohort_df.filter(pl.col("ventilated_patient_no_age_limit")),
              key="patient_id",
              rule=f"IMV within {DONOR['imv_hours_before_death']:g}h of death, NO age restriction (Table 1 as written)")
_audit.record("21_ventilated_age_le75", final_cohort_df,
              final_cohort_df.filter(pl.col("ventilated_patient_age_le75")),
              key="patient_id",
              rule=f"IMV within {DONOR['imv_hours_before_death']:g}h of death AND age <= {DONOR['age_at_death_max']:g}")

# ── hospital identity ────────────────────────────────────────────────────────
# The hospital where the patient died is the one on the last ADT row, attached
# when the cohort was formed. The crosswalk gives the CMS number SRTR files its
# donors under; that number is the hospital for everything that leaves the site.
_rows = [r for r in _XWALK if r["site"] == site_name]
final_cohort_df = final_cohort_df.join(
    pl.DataFrame([{"hospital_id_key": str(r["hospital_id"]).lower().strip(),
                   "srtr_ccn_id": r.get("srtr_ccn_id")} for r in _rows]),
    on="hospital_id_key", how="left")

# Anonymous labels for anything that leaves the site, numbered by CCN ascending
# so the mapping is deterministic across runs. The real CCN never appears in
# shareable output; the site keeps the key locally.
_ccns = sorted({c for c in final_cohort_df["srtr_ccn_id"].drop_nulls().unique().to_list()})
_labels = {c: f"{site_name}_hospital_{i}" for i, c in enumerate(_ccns, 1)}
final_cohort_df = final_cohort_df.with_columns(
    pl.col("srtr_ccn_id").replace_strict(_labels, default=None).alias("hospital_label"))
pl.DataFrame([{"site": site_name, "srtr_ccn_id": c, "hospital_label": l} for c, l in _labels.items()]
             ).write_csv(OUTPUT_INTERMEDIATE_DIR / "hospital_label_key.csv")
print(f"  hospital labels: {len(_labels)} -> hospital_label_key.csv (kept local)")
# The hospital_ids contract, checked on the COHORT rather than on raw ADT: a
# hospital that appears in ADT but contributes no decedents is not a problem,
# whereas one contributing decedents that the site never declared is — it
# silently changes the denominator. Stops the run in both directions.
_observed = {h for h in final_cohort_df["hospital_id_key"].drop_nulls().unique().to_list()}
_undeclared = sorted(_observed - set(DECLARED_HOSPITAL_IDS))
_absent = sorted(set(DECLARED_HOSPITAL_IDS) - _observed)
_n_null_hosp = int(final_cohort_df.filter(pl.col("hospital_id_key").is_null()).height)

# A declared hospital with no decedents is NOT an error. Deaths are attributed to
# the hospital of the terminal ADT record, which is what SRTR does, so a feeder
# hospital that transfers dying patients out contributes zero by design: its
# deaths are attributed to the hospitals it transfers to. Warn, so a genuinely
# missing extract is still visible, but do not stop.
for _h in _absent:
    print(f"  warn  '{_h}' is declared in config but no decedent died there "
          f"(transfers out, or no deaths in the window)")
if _n_null_hosp:
    print(f"  warn  {_n_null_hosp:,} decedents have no hospital_id in clif_adt")

# No decedent attributable to any hospital means adt.hospital_id is empty. The
# declared-hospital check below would pass on that empty set and every
# hospital-level output would be written with no rows.
if not _observed:
    raise SystemExit(
        "No decedent has a hospital_id in clif_adt, so no death can be attributed to a "
        "hospital. Populate adt.hospital_id in the extract and rerun.")

# An UNDECLARED hospital contributing decedents is an error: it silently changes
# the denominator, and the site has not said it should be in the study.
if _undeclared:
    print("\nDeclared-hospital check FAILED:")
    for _h in _undeclared:
        _n = final_cohort_df.filter(pl.col("hospital_id_key") == _h).height
        print(f"  - '{_h}' contributes {_n:,} decedents but is not in config hospital_ids")
    raise SystemExit(
        "Add it to hospital_ids in config/config.json, or correct the hospital_id "
        "values in the ADT extract, then rerun. No definition outputs were written; "
        "the partial cohort files already in output/ are from this failed run and "
        "must not be used.")
print(f"Declared-hospital check passed: {len(_observed)} hospital_ids contribute "
      f"decedents, all declared"
      + (f"; {len(_absent)} declared id(s) contributed none" if _absent else ""))

_unmapped = sorted(final_cohort_df.filter(
    pl.col("srtr_ccn_id").is_null() & pl.col("hospital_id_key").is_not_null()
)["hospital_id_key"].unique().to_list())
if _unmapped:
    raise SystemExit(
        f"hospital_id value(s) {_unmapped} contribute decedents but have no row for site "
        f"'{site_name}' in config/hospital_crosswalk.yaml. Ask the coordinating centre to add them.")

_audit.record("10_hospital_identity", final_cohort_df, final_cohort_df, key="patient_id",
              rule="terminal ADT hospital_id -> srtr_ccn_id via crosswalk",
              n_unmapped=int(final_cohort_df.filter(pl.col("srtr_ccn_id").is_null()).height))

_write_parquet(final_cohort_df, OUTPUT_INTERMEDIATE_DIR / "cohort_with_definitions.parquet", SITE_TZ)

_counts = {
    "site": site_name,
    "n_inpatient_deaths": final_cohort_df["patient_id"].n_unique(),
    "clif_donor": int(final_cohort_df.filter(pl.col("clif_eligible_donors"))["patient_id"].n_unique()),
    "calc": int(final_cohort_df.filter(pl.col("calc_flag"))["patient_id"].n_unique()),
    "ventilated_no_age_limit": int(final_cohort_df.filter(
        pl.col("ventilated_patient_no_age_limit"))["patient_id"].n_unique()),
    "ventilated_age_le75": int(final_cohort_df.filter(
        pl.col("ventilated_patient_age_le75"))["patient_id"].n_unique()),
    "n_analytic_hospitals": int(final_cohort_df["srtr_ccn_id"].drop_nulls().n_unique()),
    # The arms of the CALC cause criterion (any age, any diagnosis position) and
    # the diagnosis coverage behind them, so a site whose extract lacks an arm
    # is visible in the pooled table rather than hidden inside its CALC count.
    "icd10_ischemic": int(final_cohort_df.filter(pl.col("icd10_ischemic"))["patient_id"].n_unique()),
    "icd10_cerebro": int(final_cohort_df.filter(pl.col("icd10_cerebro"))["patient_id"].n_unique()),
    "icd10_external": int(final_cohort_df.filter(pl.col("icd10_external"))["patient_id"].n_unique()),
    "n_decedents_without_dx": int(_n_without_dx),
}
pl.DataFrame([_counts]).write_csv(OUTPUT_FINAL_DIR / "definition_counts.csv")

# Ventilation timing, per hospital: among Ventilated Patient decedents, the
# share whose last IMV record falls within 1, 6, 24 and 48 h before death.
# Supports keeping the 48 h window.
_vent = final_cohort_df.filter(pl.col("ventilated_patient_no_age_limit").fill_null(False))
_h = pl.col("hours_last_imv_to_death")
(_vent.group_by("hospital_label")
 .agg(pl.len().alias("n_ventilated"),
      *[(100 * (_h <= _k).mean()).round(1).alias(f"pct_within_{_k}h") for _k in (1, 6, 24, 48)])
 .with_columns(pl.lit(site_name).alias("site"))
 .select(["site", "hospital_label", "n_ventilated", "pct_within_1h", "pct_within_6h",
          "pct_within_24h", "pct_within_48h"])
 .sort("hospital_label")
 .write_csv(OUTPUT_FINAL_DIR / "ventilation_timing.csv"))

# Organ-quality criteria, counted over every decedent and over the patients the
# organ step actually decides (ventilated, 75 or under, no contraindicating
# cancer, no positive culture). Failure reasons are not exclusive, a patient can
# fail on two labs, so each is counted. Site totals ship; the per-hospital rows
# stay local, since some cells are small.
_K, _L = DONOR['kidney'], DONOR['liver']
_ORGAN_COUNTS = [
    pl.len().alias("n_patients"),
    pl.col("kidney_eligible").sum().alias("kidney_pass"),
    pl.col("creatinine_value").is_null().sum().alias("creatinine_missing"),
    (pl.col("creatinine_value") >= _K['creatinine_max']).sum().alias("creatinine_at_or_above_limit"),
    pl.col("on_crrt_48h_before_death").sum().alias("on_crrt"),
    pl.col("liver_eligible").sum().alias("liver_pass"),
    (pl.col("bilirubin_total_value").is_null() | pl.col("ast_value").is_null()
     | pl.col("alt_value").is_null()).sum().alias("liver_lab_missing"),
    (pl.col("bilirubin_total_value") >= _L['total_bilirubin_max']).sum().alias("bilirubin_at_or_above_limit"),
    (pl.col("ast_value") >= _L['ast_max']).sum().alias("ast_at_or_above_limit"),
    (pl.col("alt_value") >= _L['alt_max']).sum().alias("alt_at_or_above_limit"),
    pl.col("bmi_eligible").sum().alias("bmi_pass"),
    pl.col("bmi").is_null().sum().alias("bmi_missing"),
    (pl.col("bmi") > DONOR['bmi_max']).sum().alias("bmi_above_limit"),
    (pl.col("kidney_eligible") & pl.col("liver_eligible")).sum().alias("kidney_and_liver_pass"),
    pl.col("organ_check_pass").sum().alias("organ_pass"),
]
_reached_organ_step = (pl.col("imv_48hr_expire") & pl.col("age_75_less")
                       & ~pl.col("icd10_contraindication") & pl.col("no_positive_culture_48hrs"))


def _organ_criteria(by: list[str]) -> pl.DataFrame:
    parts = []
    for population, sub in (("all_decedents", final_cohort_df),
                            ("reached_organ_step", final_cohort_df.filter(_reached_organ_step))):
        g = sub.group_by(by).agg(_ORGAN_COUNTS) if by else sub.select(_ORGAN_COUNTS)
        parts.append(g.with_columns(pl.lit(population).alias("population")))
    out = pl.concat(parts).with_columns(pl.lit(site_name).alias("site"))
    return out.select(["site", *by, "population", *[c.meta.output_name() for c in _ORGAN_COUNTS]]).sort(by + ["population"])


_organ_criteria([]).write_csv(OUTPUT_FINAL_DIR / "organ_criteria.csv")
_organ_criteria(["hospital_label"]).write_csv(OUTPUT_INTERMEDIATE_DIR / "organ_criteria_by_hospital.csv")

# Death-time diagnostics, per hospital. With ventilation_timing.csv they are the
# evidence for the time-of-death rule in utils/death_time.py. Every interval is
# elapsed time (utils/dtypes.hours_between), so a daylight-saving change inside
# it does not add or remove an hour.
_sch = final_cohort_df.schema
_pct = lambda cond, name: (100 * cond.fill_null(False).mean()).round(1).alias(name)  # noqa: E731

# discharge_vs_death_timing.csv: among decedents whose death_dttm carries a time
# of day, how far discharge falls from it. Discharge stands in for the death
# whenever the death is date-only or missing; this tests that substitution on
# the patients where both are known.
_g = hours_between("discharge_dttm", "death_dttm", _sch, SITE_TZ)
(final_cohort_df.filter(has_time_of_day("death_dttm", _sch["death_dttm"], SITE_TZ))
 .group_by("hospital_label")
 .agg(pl.len().alias("n_timed_death"),
      _pct(_g.abs() < 1 / 60, "pct_same_minute"),
      *[_pct(_g.is_between(0, _k), f"pct_discharge_within_{_k}h_after") for _k in (1, 6, 24)],
      _pct(_g > 24, "pct_discharge_over_24h_after"),
      _pct(_g < 0, "pct_discharge_before_death"))
 .with_columns(pl.lit(site_name).alias("site"))
 .select(["site", "hospital_label", "n_timed_death", "pct_same_minute",
          "pct_discharge_within_1h_after", "pct_discharge_within_6h_after",
          "pct_discharge_within_24h_after", "pct_discharge_over_24h_after",
          "pct_discharge_before_death"])
 .sort("hospital_label")
 .write_csv(OUTPUT_FINAL_DIR / "discharge_vs_death_timing.csv"))

# vitals_timing.csv: where the last recorded vital falls relative to the time of
# death. A last vital long before death is a unit that charts none (inpatient
# hospice) or a vitals extract that ends early; a heart rate above zero an hour
# after the recorded death is a death time recorded early. Either is why the
# last vital is not the anchor. Shares are of all decedents at the hospital.
_v = hours_between("final_death_dttm", "last_recorded_vital_dttm", _sch, SITE_TZ)
_hr = hours_between("last_hr_above_zero_dttm", "final_death_dttm", _sch, SITE_TZ)
(final_cohort_df.group_by("hospital_label")
 .agg(pl.len().alias("n_decedents"),
      _pct(pl.col("last_recorded_vital_dttm").is_null(), "pct_no_vitals"),
      _pct(_v < 0, "pct_last_vital_after_death"),
      *[_pct(_v.is_between(0, _k), f"pct_within_{_k}h") for _k in (1, 6, 24, 48)],
      _pct(_v > 48, "pct_over_48h_before"),
      _pct(_hr > 1, "pct_hr_above_zero_after_1h"))
 .with_columns(pl.lit(site_name).alias("site"))
 .select(["site", "hospital_label", "n_decedents", "pct_no_vitals", "pct_last_vital_after_death",
          "pct_within_1h", "pct_within_6h", "pct_within_24h", "pct_within_48h",
          "pct_over_48h_before", "pct_hr_above_zero_after_1h"])
 .sort("hospital_label")
 .write_csv(OUTPUT_FINAL_DIR / "vitals_timing.csv"))

# value_timing.csv: when each Table 2 value was taken, relative to the time of
# death, as the share of decedents whose value is within 1, 6, 24, 48 and 168 h
# before it. One row per measure, so the pooled report can lay the sites over
# one another and a site with stale values stands out. Every value here is the
# last one before death, so no hours are negative. Site totals ship; the
# per-hospital rows stay local.
_VALUE_TIMES = {"creatinine": "creatinine_dttm", "bilirubin_total": "bilirubin_total_dttm",
                "ast": "ast_dttm", "alt": "alt_dttm", "bun": "bun_dttm", "sodium": "sodium_dttm",
                "gcs_total": "gcs_total_dttm", "rass": "rass_dttm",
                "weight_kg": "last_weight_dttm", "height_cm": "last_height_dttm"}


def _value_timing(by: list[str]) -> pl.DataFrame:
    parts = []
    for measure, col in _VALUE_TIMES.items():
        _hrs = hours_between("final_death_dttm", col, _sch, SITE_TZ)
        aggs = [pl.len().alias("n_decedents"), pl.col(col).is_not_null().sum().alias("n_with_value"),
                *[_pct(_hrs.is_between(0, _k), f"pct_within_{_k}h") for _k in (1, 6, 24, 48, 168)],
                _pct(pl.col(col).is_null(), "pct_missing")]
        g = final_cohort_df.group_by(by).agg(aggs) if by else final_cohort_df.select(aggs)
        parts.append(g.with_columns(pl.lit(measure).alias("measure")))
    out = pl.concat(parts).with_columns(pl.lit(site_name).alias("site"))
    return out.select(["site", *by, "measure", "n_decedents", "n_with_value", "pct_within_1h", "pct_within_6h",
                       "pct_within_24h", "pct_within_48h", "pct_within_168h", "pct_missing"]).sort(by + ["measure"])


_value_timing([]).write_csv(OUTPUT_FINAL_DIR / "value_timing.csv")
_value_timing(["hospital_label"]).write_csv(OUTPUT_INTERMEDIATE_DIR / "value_timing_by_hospital.csv")
pl.DataFrame(dq_flags, schema={"site": pl.Utf8, "flag": pl.Utf8, "severity": pl.Utf8,
                               "detail": pl.Utf8}).write_csv(
    OUTPUT_FINAL_DIR / "data_quality_flags.csv")
_audit.write()
print("\n" + _audit.summary())
for _k, _v in _counts.items():
    print(f"  {_k:28s} {_v}")

# ── SRTR reference: hospital identity and year coverage ──────────────────────
# The coordinating centre has to align an SRTR donor count (a fixed recovery-year
# window at a CCN) against a denominator this site actually observed. Those two
# periods are NOT the same: a hospital that joined the system mid-study
# contributes decedents for only part of the window.
# Where each cohort decedent's time of death came from, per hospital, with how
# many of them are ventilated and CLIF-donor eligible. Kept local: the smaller
# categories are small counts. The site totals (the 2d_death_time_* rows of
# strobe_counts.csv) are over the same patients.
(final_cohort_df.group_by(["hospital_label", "death_time_source"])
 .agg(pl.len().alias("n_decedents"),
      pl.col("imv_48hr_expire").sum().alias("n_ventilated"),
      pl.col("clif_eligible_donors").sum().alias("n_clif_donor"))
 .sort(["hospital_label", "death_time_source"])
 .write_csv(OUTPUT_INTERMEDIATE_DIR / "death_time_source_by_hospital.csv"))

_SREF = OUTPUT_FINAL_DIR / "srtr_ref"
_SREF.mkdir(parents=True, exist_ok=True)
_cov = final_cohort_df.select(
    ["patient_id", "hospital_id_key", "hospital_label", "srtr_ccn_id", "hospital_type", "final_death_dttm"]
).drop_nulls("hospital_id_key")

# Year of death only. The earliest and latest death timestamps are dates tied to
# two individual patients, which may not leave the site; SRTR linkage needs the
# calendar years covered and nothing finer.
_cov = _cov.with_columns(_year_local(_cov, "final_death_dttm").alias("death_year"))

(_cov.rename({"death_year": "year"})
 .group_by(["hospital_id_key", "hospital_label", "srtr_ccn_id", "year"])
 .agg(pl.col("patient_id").n_unique().alias("n_decedents"))
 .with_columns(pl.lit(site_name).alias("site"))
 .select(["site", "hospital_id_key", "hospital_label", "srtr_ccn_id", "year", "n_decedents"])
 .sort(["hospital_id_key", "year"])
 .write_csv(_SREF / "hospital_years.csv"))

_hc = (_cov.group_by(["hospital_id_key", "hospital_label", "srtr_ccn_id", "hospital_type"])
       .agg(pl.col("patient_id").n_unique().alias("n_decedents"),
            pl.col("death_year").min().alias("first_year"),
            pl.col("death_year").max().alias("last_year"))
       .with_columns(pl.lit(site_name).alias("site"))
       .with_columns((pl.col("last_year") - pl.col("first_year") + 1).alias("n_years_spanned"))
       .sort("n_decedents", descending=True))
# Declared hospitals that contributed no decedents get a zero row, so the
# coordinating centre can tell a feeder hospital from a thin extract without
# reading the run log. No label: labels are given only to hospitals with decedents.
_zero = pl.DataFrame([{
    "hospital_id_key": h, "hospital_label": None,
    "srtr_ccn_id": r.get("srtr_ccn_id"), "hospital_type": None,
    "n_decedents": 0, "first_year": None, "last_year": None,
    "site": site_name, "n_years_spanned": None,
} for h in _absent for r in _rows if str(r["hospital_id"]).lower().strip() == h],
    schema=_hc.schema)
pl.concat([_hc, _zero]).write_csv(_SREF / "hospital_coverage.csv")

pl.DataFrame([{
    "site": site_name,
    "n_hospital_ids": _cov["hospital_id_key"].n_unique(),
    "n_distinct_ccn": _cov["srtr_ccn_id"].drop_nulls().n_unique(),
    "n_decedents": _cov["patient_id"].n_unique(),
    "first_year": _cov["death_year"].min(),
    "last_year": _cov["death_year"].max(),
}]).write_csv(_SREF / "site_coverage.csv")

print(f"\nSRTR reference -> {_SREF}")
for _r in _hc.iter_rows(named=True):
    _fd = _r["first_year"] if _r["first_year"] else "?"
    _ld = _r["last_year"] if _r["last_year"] else "?"
    _flag = "" if _r["first_year"] and _r["first_year"] <= WINDOW_START_YEAR else "   <-- PARTIAL COVERAGE"
    print(f"    {str(_r['hospital_id_key'])[:32]:32s} ccn={str(_r['srtr_ccn_id']):8s} "
          f"n={_r['n_decedents']:5,}  {_fd} -> {_ld}{_flag}")

_progress.mark("Definitions that need the finished cohort")
