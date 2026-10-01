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

import math
import sys
import duckdb
import polars as pl
import pandas as pd
import re
import yaml
from zoneinfo import ZoneInfo
from utils.config import config
from utils.io import read_data
from clifpy.utils.stitching_encounters import stitch_encounters
from utils.outlier_handler import outlier_range
from utils.checks import mixed_tz_awareness, usable_number_sql
from utils.dtypes import align_time_zone
from utils.criteria import DECLARED_NOT_APPLIED, OPERATORS, Criteria, icd_range_sql
import gc

# Timestamp arithmetic must not depend on the machine this runs on. DuckDB's
# session timezone defaults to the operating system's; pin it so a subtraction
# gives the same answer everywhere. The setup check below additionally refuses
# an extract that mixes timezone-aware and naive timestamps, which is the case
# where the session timezone would silently change a 48-hour window.
duckdb.sql("SET TimeZone='UTC'")

# Fix Windows encoding issue for Unicode characters
sys.stdout.reconfigure(encoding='utf-8')
site_name = config['site_name']
tables_path = config['tables_path']
file_type = config['file_type']
project_root = config['project_root']
sys.path.insert(0, project_root)
print(f"Site Name: {site_name}")
print(f"Tables Path: {tables_path}")
print(f"File Type: {file_type}")
from pathlib import Path
PROJECT_ROOT = Path(config['project_root'])
# Checked here, before anything is read from it: every code list, criteria file
# and output path hangs off this, so a wrong value gives a FileNotFoundError
# from somewhere deep in the script instead of a usable message.
_REPO_HERE = Path(__file__).resolve().parent.parent
if PROJECT_ROOT.expanduser().resolve() != _REPO_HERE:
    raise SystemExit(
        f"config.json: project_root is '{PROJECT_ROOT}', which is not the folder "
        f"this code is in ({_REPO_HERE}). Set project_root to that path and rerun.")
UTILS_DIR = PROJECT_ROOT / "utils"
# PHI split: output/intermediate_phi/<site>/ never leaves the site;
# <site>_upload_to_box/ holds aggregate, shareable output. Per-site subfolders so
# three sites can be run on one machine without clobbering each other.
# Every study criterion comes from this one file, loaded once.
CRIT = Criteria(PROJECT_ROOT / "config/donor_criteria.yaml")
STUDY = CRIT.get("study")

# ADT locations that put a death in the cohort. A death at any other location,
# or with no ADT record at all, is excluded and counted separately.
COHORT_LOCATIONS = [str(x).lower() for x in
                    STUDY.get("cohort_locations", ["ed", "ward", "stepdown", "icu"])]
if not COHORT_LOCATIONS:
    raise SystemExit("study.cohort_locations is empty — fix config/donor_criteria.yaml")


# ── Definition thresholds ────────────────────────────────────────────────────
# Read once, here, and used by name below. Nothing further down may write a
# clinical number. The check after this block stops the run, before any data is
# read, if a key under clif_donor, calc or ventilated_patient is declared in the
# YAML and not read here.
#
# Flag and column names such as age_75_less, imv_48hr_expire and creatinine_lt_4
# keep the default value in their name because steps 02 and 03 and the
# coordinating scripts read them. The value applied is the one in the YAML.
def _num(dotted: str) -> float:
    """A numeric criterion. These are formatted into SQL, so a non-number stops the run."""
    v = CRIT.get(dotted)
    if (isinstance(v, bool) or not isinstance(v, (int, float))
            or not math.isfinite(v) or v <= 0):
        raise SystemExit(f"config/donor_criteria.yaml: '{dotted}' must be a positive number, "
                         f"got {v!r}")
    return v


AGE_MAX = _num("clif_donor.age_at_death_max")
_AGE_OP_NAME = str(CRIT.get("clif_donor.age_operator"))
if _AGE_OP_NAME not in OPERATORS:
    raise SystemExit(f"clif_donor.age_operator is '{_AGE_OP_NAME}'; expected one of "
                     f"{sorted(OPERATORS)} — fix config/donor_criteria.yaml")
AGE_OK = OPERATORS[_AGE_OP_NAME]
IMV_HOURS = _num("clif_donor.imv_hours_before_death")
IMV_POST_DEATH_HOURS = _num("clif_donor.imv_post_death_tolerance_hours")
BMI_MAX = _num("clif_donor.bmi_max")
CREATININE_MAX = _num("clif_donor.kidney.creatinine_max")
CRRT_HOURS = _num("clif_donor.kidney.exclude_if_crrt_within_hours")
BILIRUBIN_MAX = _num("clif_donor.liver.total_bilirubin_max")
AST_MAX = _num("clif_donor.liver.ast_max")
ALT_MAX = _num("clif_donor.liver.alt_max")
CULTURE_HOURS = _num("clif_donor.contraindications.positive_blood_culture_hours")
CONTRA_FILE = PROJECT_ROOT / str(CRIT.get("clif_donor.contraindications.icd10_file"))
if not CONTRA_FILE.is_file():
    raise SystemExit(f"clif_donor.contraindications.icd10_file not found: {CONTRA_FILE}")

# One age flag and one IMV flag each serve two definitions, so the values the
# YAML gives those definitions have to agree. Different values would need
# separate flags; refuse, rather than silently apply one value to both.
if _num("calc.age_at_death_max") != AGE_MAX:
    raise SystemExit("calc.age_at_death_max differs from clif_donor.age_at_death_max. "
                     "This code computes one age flag for both definitions; make them equal.")
if _num("ventilated_patient.imv_hours_before_death") != IMV_HOURS:
    raise SystemExit("ventilated_patient.imv_hours_before_death differs from "
                     "clif_donor.imv_hours_before_death. This code computes one IMV flag "
                     "for both definitions; make them equal.")
if CRIT.get("clif_donor.missing_is_ineligible") is not True:
    raise SystemExit("clif_donor.missing_is_ineligible must be true: a missing organ-quality "
                     "value is always ineligible, and no other behaviour is implemented.")

# The CALC cause-of-death ranges, each as a SQL predicate on the normalised code.
CALC_CAUSE_RANGES = CRIT.get("calc.cause_icd10_prefixes")
for _k in ("ischemic_heart_disease", "cerebrovascular_disease", "external_causes"):
    if not (isinstance(CALC_CAUSE_RANGES.get(_k), list) and len(CALC_CAUSE_RANGES[_k]) == 2):
        raise SystemExit(f"calc.cause_icd10_prefixes.{_k} must be a [first, last] pair of "
                         f"ICD-10 categories, e.g. [I20, I25]")
_RANGE_SQL = {k: icd_range_sql("dx_norm", lo, hi) for k, (lo, hi) in CALC_CAUSE_RANGES.items()}
_ISCHEMIC_SQL = _RANGE_SQL["ischemic_heart_disease"]
_CEREBRO_SQL = _RANGE_SQL["cerebrovascular_disease"]
_EXTERNAL_SQL = _RANGE_SQL["external_causes"]

CALC_POSITION = str(CRIT.get("calc.diagnosis_position")).lower()
CALC_APPLY_CONTRAINDICATIONS = CRIT.get("calc.apply_contraindications")
VENT_APPLY_AGE_LIMIT = CRIT.get("ventilated_patient.apply_age_limit")
if VENT_APPLY_AGE_LIMIT:
    raise SystemExit(
        "ventilated_patient.apply_age_limit is true. Steps 02 and 03 and the pooled report "
        "use the Ventilated Patient definition with no age limit; the age-capped count is "
        "always reported beside it as ventilated_age_le75. An age-limited primary definition "
        "is not implemented, so the switch would change no reported number.")

_unread = [k for k in CRIT.unused("clif_donor", "calc", "ventilated_patient")
           if k not in DECLARED_NOT_APPLIED]
if _unread:
    raise SystemExit(
        "config/donor_criteria.yaml declares criteria that this code does not apply: "
        + ", ".join(_unread) + ". Implement them in code/01_cohort_and_definitions.py, "
        "or remove them from the YAML.")


################################################################################
# Setup checks
# Everything that can be verified before reading a single patient row. These
# used to live in a separate 00_setup_check.py; keeping them here means the run
# either validates and proceeds or stops in about a second, with no second pass
# over the data. Schemas and row counts only — no patient-level reads.
################################################################################

REQ = yaml.safe_load((PROJECT_ROOT / "config/clif_data_requirements.yaml").read_text())
_problems, _warnings = [], []
_TIMESTAMPS_NAIVE = False

# 1. config
for _k in ("site_name", "tables_path", "file_type", "timezone", "project_root",
           "clif_version", "hospital_ids"):
    if config.get(_k) in (None, "", []):
        _problems.append(f"config.json: '{_k}' is missing or empty")
if not re.fullmatch(r"[a-z0-9_]+", str(site_name)):
    _problems.append(f"config.json: site_name '{site_name}' — lowercase, digits, underscore only")
if file_type != "parquet":
    _problems.append(f"config.json: file_type '{file_type}' — only parquet is supported")
try:
    ZoneInfo(config["timezone"])
except Exception:
    _problems.append(f"config.json: timezone '{config.get('timezone')}' is not a valid IANA zone")
if str(config.get("timezone", "")).upper().replace("ETC/", "") in {
        "UTC", "GMT", "UCT", "Z", "ZULU", "UNIVERSAL", "GMT0", "GMT+0", "GMT-0"}:
    _problems.append("config.json: timezone is UTC. Give the zone your hospital is in, e.g. "
                     "US/Central. CLIF stores timestamps in UTC, but this setting decides "
                     "which calendar year an admission or a death falls in")
if str(config.get("clif_version")) != str(REQ["clif_version"]):
    _problems.append(f"config.json: clif_version '{config.get('clif_version')}' — "
                     f"this pipeline requires CLIF {REQ['clif_version']}")
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

OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_FINAL_DIR = Path(config["output_final"])
OUTPUT_INTERMEDIATE_DIR = Path(config["output_intermediate"])
OUTPUT_FINAL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

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
print(f"Site: {site_name} | Tables: {tables_path} | File type: {file_type}")
print(f"Logging stdout to: {_RUN_LOG_PATH}")
for _k, _why in DECLARED_NOT_APPLIED.items():
    if _k in CRIT.leaves("clif_donor", "calc", "ventilated_patient"):
        print(f"  note  {_k} is in donor_criteria.yaml but is NOT applied: {_why}")
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
# decedent step below — which is what the Methods describe: "patients ... who
# died during admission, 2020 - 2025".
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

# STROBE step 1: Full population — patients with any hospitalization overlapping 2020-2025
full_population_n = hospitalization_df['patient_id'].n_unique()
strobe_counts["0_full_population_2020_2025"] = full_population_n
print(f"Full population (any hosp 2020-2025): {full_population_n:,}")

# STROBE step 2: ICU population — subset with ≥1 ICU ADT stay
icu_population_n = (
    adt_df
    .filter(pl.col('location_category').str.to_lowercase() == 'icu')
    .join(
        hospitalization_df.select(['hospitalization_id', 'patient_id']),
        on='hospitalization_id',
        how='left',
    )
    .select('patient_id')
    .drop_nulls()
    .unique()
    .height
)
strobe_counts["0b_icu_population_2020_2025"] = icu_population_n
print(f"ICU population (any ICU stay): {icu_population_n:,}")

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
      time_interval=12
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

# Filter hosp_stitched to only hospitalizations that are present in the adt_stitched table
if "hospitalization_id" in hosp_stitched.columns and "hospitalization_id" in adt_stitched.columns:
    hosp_stitched = (
        hosp_stitched.filter(
            pl.col("hospitalization_id").is_in(adt_stitched["hospitalization_id"].unique())
        )
    )
encounter_mapping = pl.from_pandas(encounter_mapping)
encounter_mapping = encounter_mapping.with_columns(
    pl.col("encounter_block").cast(pl.Int32)
)

gc.collect()

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
strobe_counts

################################################################################
# Final outcome dttm
################################################################################

vitals_filepath = f"{tables_path}/clif_vitals.{file_type}"

# Stream clif_vitals via DuckDB instead of loading into polars. Sites with
# very large vitals tables (e.g., JHU at >2 GB) were getting OOM-killed
# inside the polars `with_columns` outlier-handler chain. We only consume
# four fields downstream (first/last vital timestamps + last weight_kg +
# last height_cm), so a streamed SQL query is both faster and bounded in RAM.
# Outlier ranges are applied in the WHERE of each ranked subquery, using the
# bounds read from config/outlier_config.yaml at the top of this file.
all_decedent_hosp_ids_df = pd.DataFrame(
    {"hospitalization_id": list(all_decedent_hosp_ids)}
)

vitals_query = f"""
WITH vitals_cohort AS (
    SELECT hospitalization_id, recorded_dttm,
           LOWER(TRIM(vital_category)) AS vital_category, vital_value
    FROM read_parquet('{vitals_filepath}')
    WHERE hospitalization_id IN (
        SELECT hospitalization_id FROM all_decedent_hosp_ids_df
    )
),
time_bounds AS (
    SELECT
        hospitalization_id,
        MIN(recorded_dttm) AS first_recorded_vital_dttm,
        MAX(recorded_dttm) AS last_recorded_vital_dttm
    FROM vitals_cohort
    GROUP BY hospitalization_id
),
weight_ranked AS (
    -- The plausibility range is applied BEFORE ranking, so rn = 1 is the most
    -- recent PLAUSIBLE value. Applied after ranking, a zeroed bed scale at the
    -- latest timestamp won rn = 1, nulled the value, and made the patient
    -- BMI-ineligible although a valid weight was charted an hour earlier.
    SELECT
        hospitalization_id,
        vital_value,
        ROW_NUMBER() OVER (
            PARTITION BY hospitalization_id ORDER BY recorded_dttm DESC, vital_value DESC
        ) AS rn
    FROM vitals_cohort
    WHERE vital_category = 'weight_kg'
      AND vital_value BETWEEN {WEIGHT_MIN} AND {WEIGHT_MAX}
),
height_ranked AS (
    SELECT
        hospitalization_id,
        vital_value,
        ROW_NUMBER() OVER (
            PARTITION BY hospitalization_id ORDER BY recorded_dttm DESC, vital_value DESC
        ) AS rn
    FROM vitals_cohort
    WHERE vital_category = 'height_cm'
      AND vital_value BETWEEN {HEIGHT_MIN} AND {HEIGHT_MAX}
)
SELECT
    t.hospitalization_id,
    t.first_recorded_vital_dttm,
    t.last_recorded_vital_dttm,
    w.vital_value AS last_weight_kg,
    h.vital_value AS last_height_cm
FROM time_bounds t
LEFT JOIN weight_ranked w
    ON t.hospitalization_id = w.hospitalization_id AND w.rn = 1
LEFT JOIN height_ranked h
    ON t.hospitalization_id = h.hospitalization_id AND h.rn = 1
"""

print("Processing vitals data with DuckDB...")
vitals_first_last = pl.from_pandas(duckdb.sql(vitals_query).df())
# DuckDB returns UTC; match the canonical IANA TZ that clifpy uses on the
# rest of final_df (config['timezone'] may be a legacy alias like "US/Central"
# that polars treats as a different dtype than "America/Chicago", so detect
# the actual TZ from an existing column).
_dt_dtype = next(
    (dt for col, dt in final_df.schema.items()
     if isinstance(dt, pl.Datetime) and dt.time_zone is not None),
    None,
)
if _dt_dtype is not None:
    _site_tz = _dt_dtype.time_zone
    vitals_first_last = vitals_first_last.with_columns([
        pl.col('first_recorded_vital_dttm').dt.convert_time_zone(_site_tz),
        pl.col('last_recorded_vital_dttm').dt.convert_time_zone(_site_tz),
    ])
print(f"✓ Processed vitals for {len(vitals_first_last)} hospitalizations")
for _cat, _c in (("weight_kg", "last_weight_kg"), ("height_cm", "last_height_cm")):
    _require_matches(int(vitals_first_last[_c].is_not_null().sum()),
                     "vitals", "vital_category", _cat)

# Calculate BMI
vitals_first_last = vitals_first_last.with_columns(
    (pl.col('last_weight_kg') / ((pl.col('last_height_cm') / 100) ** 2)).alias('bmi')
)

# Join with final_df
final_df = final_df.join(vitals_first_last, on='hospitalization_id', how='left')

# Define final_death_dttm using the actual death timestamp where present.
# We deliberately do NOT cap death_dttm at discharge_dttm — sites that pull
# death data from external registries (state vital records, SSA Death Master,
# etc.) can legitimately have death_dttm > discharge_dttm for patients who
# were discharged alive and died later. Those patients should NOT pass the
# downstream 48-h before-death donor filters, and trusting death_dttm
# directly makes that natural (their in-hospital labs/vitals will fall
# outside the 48-h window relative to their later death).
# Fall back to last_recorded_vital_dttm only when death_dttm is null.
final_df = final_df.with_columns(
    pl.when(pl.col("death_dttm").is_not_null())
      .then(pl.col("death_dttm"))
      .otherwise(pl.col("last_recorded_vital_dttm"))
      .alias("final_death_dttm")
)

# Diagnostic: how many decedents have death_dttm > discharge_dttm + 24h?
# This indicates sites with external death-registry data linked into CLIF;
# such patients were discharged alive and died later — they auto-fail the
# CLIF donor criteria but remain in the cohort for transparency.
_delayed_death = final_df.filter(
    pl.col("death_dttm").is_not_null()
    & pl.col("discharge_dttm").is_not_null()
    & ((pl.col("death_dttm") - pl.col("discharge_dttm")).dt.total_hours() > 24)
)["patient_id"].n_unique()
strobe_counts["1c_died_post_discharge_24h"] = _delayed_death
print(f"Decedents with death_dttm > discharge_dttm + 24h (likely external registry): {_delayed_death:,}")

################################################################################
# Inpatient decedents
# Identify inpatient encounters - location must be ed, ward, stepdown, icu at last_recorded_vital_dttm
################################################################################

eligible_locations = ['ed', 'ward', 'stepdown', 'icu']

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
      .sort('out_dttm', descending=True)
      .group_by('hospitalization_id')
      .agg([
          pl.col('location_category').first().alias('last_location_category'),
          # the unit name is informational and not every extract carries it
          *([pl.col('location_name').first().alias('last_location_name')]
            if 'location_name' in adt_df.columns else []),
          pl.col('out_dttm').first().alias('last_location_out_dttm'),
          (pl.col('location_category').str.to_lowercase() == 'icu').any().alias('ever_icu'),
          (pl.col('location_category').str.to_lowercase() == 'ward').any().alias('ever_ward'),
          (pl.col('location_category').str.to_lowercase() == 'ed').any().alias('ever_ed'),
          (pl.col('location_category').str.to_lowercase() == 'stepdown').any().alias('ever_stepdown'),
          (pl.col('location_category').str.to_lowercase() == 'hospice').any().alias('ever_hospice'),
          (pl.col('location_category').str.to_lowercase()
             .is_in(COHORT_LOCATIONS)).any().alias('in_cohort_location'),
          pl.col('location_category').unique().sort().alias('all_locations')
      ])
  )

final_df = final_df.join(
    last_location_per_hosp,
    on='hospitalization_id',
    how='left'
)

# Decedent counts by ADT location, so a site difference like RUSH's hospice unit
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
print(f"Cohort locations {COHORT_LOCATIONS}: {final_df.height:,} -> "
      f"{final_cohort_df.height:,} hospitalizations")

all_decedent_inpatient_patient_ids = final_cohort_df.select('patient_id').to_series().to_list()
all_decedent_inpatient_hosp_ids = final_cohort_df.select('hospitalization_id').to_series().to_list()
strobe_counts["2_inpatient_decedents"] = len(all_decedent_inpatient_patient_ids)
strobe_counts

adt_stitched.columns

################################################################################
# ADT
################################################################################

# Calculate hospital and ICU length of stay using approach similar to the provided reference (adapted for Polars)

# Filter adt_df to only the relevant hospitalizations
adt_in_cohort = adt_stitched.filter(pl.col("hospitalization_id").is_in(all_decedent_inpatient_hosp_ids))

# Lowercase location_category (just the column, not the whole DataFrame)
adt_in_cohort = adt_in_cohort.with_columns(
    pl.col("location_category").str.to_lowercase().alias("location_category")
)

# Hospital admission summary per encounter_block: first in and last out, first admission location
hosp_admission_summary = (
    adt_in_cohort
    .group_by("encounter_block")
    .agg([
        pl.col("in_dttm").min().alias("min_in_dttm"),
        pl.col("out_dttm").max().alias("max_out_dttm"),
        pl.col("location_category").first().alias("first_admission_location")
    ])
    .with_columns([
        ((pl.col("max_out_dttm") - pl.col("min_in_dttm")).dt.total_days()).alias("hospital_length_of_stay_days")
    ])
)

# Join first_admission_location and hospital_length_of_stay_days to final_cohort_df on encounter_block
final_cohort_df = final_cohort_df.join(
    hosp_admission_summary.select([
        "encounter_block", 
        "first_admission_location", 
        "hospital_length_of_stay_days"
    ]),
    on="encounter_block",
    how="left"
)

# Restrict to ICU stays only
icu_df = adt_in_cohort.filter(pl.col("location_category") == "icu")

# Find first ICU admission per encounter_block
first_icu_in = (
    icu_df
    .group_by("encounter_block")
    .agg(pl.col("in_dttm").min().alias("first_icu_in_dttm"))
)

# Join back to get corresponding out_dttm for the first ICU in_dttm.
# Some sites (e.g. NU) log >1 ADT-ICU row at the same first in_dttm with
# different out_dttm (overlapping unit transfers recorded simultaneously).
# Collapse to one row per encounter_block, keeping the latest out_dttm so
# first_icu_los_days reflects the longest stay starting at that moment.
icu_summary = (
    first_icu_in.join(
        icu_df.select(["encounter_block", "in_dttm", "out_dttm"]),
        left_on=["encounter_block", "first_icu_in_dttm"],
        right_on=["encounter_block", "in_dttm"],
        how="left"
    )
    .group_by("encounter_block")
    .agg([
        pl.col("first_icu_in_dttm").first(),
        pl.col("out_dttm").max().alias("first_icu_out_dttm"),
    ])
    .with_columns(
        ((pl.col("first_icu_out_dttm") - pl.col("first_icu_in_dttm")).dt.total_seconds() / (3600*24))
        .alias("first_icu_los_days")
    )
    .select([
        "encounter_block", "first_icu_in_dttm", "first_icu_out_dttm", "first_icu_los_days"
    ])
)

final_cohort_df = final_cohort_df.join(
    icu_summary.select([
        "encounter_block", 
        "first_icu_los_days"
    ]),
    on="encounter_block",
    how="left"
)

# Now, hosp_admission_summary contains hospital LOS and first_admission_location, and icu_summary contains first ICU LOS

################################################################################
# Age
################################################################################

# Age < 75
final_cohort_df = final_cohort_df.join(
    patient_df.select(['patient_id', 'birth_date']),
    on='patient_id',
    how='left'
)
# Only cast birth_date to datetime if not already a datetime type
# if final_cohort_df.schema["birth_date"] != pl.Datetime:
#     final_cohort_df = final_cohort_df.with_columns(
#         pl.col('birth_date').str.to_datetime().alias('birth_date')
#     )

# Age at death = (final_death_dttm - birth_date) in years.
#
# DE-IDENTIFIED SITES. Some sites ship extracts with birth_date fully redacted
# (RUSH: 0 of 138,070 populated). There, age_at_death is null for everyone, the
# age filter drops the entire cohort, and the site silently reports zero
# eligible donors. We therefore fall back to hospitalization.age_at_admission,
# which such extracts do populate.
#
# The fallback is age at ADMISSION, not at death, so it understates age by the
# length of stay. For decedents that is typically days, and it can only make a
# patient look younger -- i.e. it can only over-include at the <=75 boundary,
# never under-include. HIPAA caps age_at_admission at 89, which is well above
# the 75 threshold and so does not affect the flag. The substitution is
# recorded in strobe_counts and must be reported as a site-level deviation.
# Some sites store birth_date without a timezone and death_dttm with one; polars
# refuses to subtract the two, so relabel birth_date to match first.
_birth = (align_time_zone('birth_date', final_cohort_df.schema['birth_date'],
                          final_cohort_df.schema['final_death_dttm'])
          if 'birth_date' in final_cohort_df.columns else pl.col('birth_date'))
_age_from_birth = (
    (pl.col('final_death_dttm') - _birth).dt.total_days() / 365.25
)
_has_birth_date = (
    'birth_date' in final_cohort_df.columns
    and final_cohort_df['birth_date'].null_count() < final_cohort_df.height
)
if _has_birth_date:
    final_cohort_df = final_cohort_df.with_columns(_age_from_birth.alias('age_at_death'))
    _age_source = 'birth_date'
elif 'age_at_admission' in final_cohort_df.columns:
    final_cohort_df = final_cohort_df.with_columns(
        pl.col('age_at_admission').cast(pl.Float64).alias('age_at_death')
    )
    _age_source = 'age_at_admission_fallback'
    print("WARNING: birth_date is empty at this site; age_at_death falls back to "
          "age_at_admission. See strobe_counts['age_source'].")
else:
    raise RuntimeError(
        "Cannot determine age: birth_date is empty and age_at_admission is absent.")
strobe_counts['age_source'] = _age_source
print(f"Age source: {_age_source}")

# Create age_75_less flag per patient_id (age_at_death within clif_donor.age_at_death_max)
age_flag_df = (
    final_cohort_df
    .group_by('patient_id')
    .agg([
        (
            AGE_OK(pl.col('age_at_death'), AGE_MAX).any()
        ).alias('age_75_less')
    ])
)

# Join age_75_less flag onto final_df; fill nulls with False
final_cohort_df = (
    final_cohort_df
    .join(age_flag_df, on='patient_id', how='left')
    .with_columns(
        pl.col('age_75_less').fill_null(False)
    )
)

# Filter age < 75 using the flag, not the missing column
age_relevant_cohort = final_cohort_df.filter(
    pl.col('age_75_less') == True
)
age_relevant_cohort_n = age_relevant_cohort["patient_id"].n_unique()
strobe_counts["3_age_relevant_cohort_n"] = age_relevant_cohort_n
strobe_counts

################################################################################
# ICD Codes
# The CALC criteria includes the following as cause:
# - I20–I25: ischemic heart disease
# - I60–I69: cerebrovascular disease
# - V01–Y89: external causes (e.g., blunt trauma, gunshot wounds, overdose, suicide, drowning, asphyxiation)
# [Reference](https://www.cms.gov/files/document/112020-opo-final-rule-cms-3380-f.pdf)
# We also flag contraindications of sepsis and cancer using ICD10 codes. We use the ICD codes for these specified in utils/icd10_contraindications.csv
################################################################################

hospial_dx_filepath = f"{tables_path}/clif_hospital_diagnosis.{file_type}"

# Diagnostic counts via DuckDB (streamed; previously a polars read+join that
# segfaulted on Windows at sites with large hospital_diagnosis tables).
# DuckDB infers the column type of an EMPTY pandas frame as INTEGER, which then
# fails to compare against VARCHAR ids ("Cannot compare values of type VARCHAR
# and INTEGER"). Forcing str dtype keeps the comparison valid even when a site
# legitimately has zero rows at this stage.
all_ids_df = pd.DataFrame(
    {"hospitalization_id": pd.Series(list(all_decedent_inpatient_hosp_ids), dtype="str")})
age_relevant_ids_df = (
    age_relevant_cohort.select("patient_id").unique().to_pandas().astype({"patient_id": "str"}))

n_present = duckdb.sql(f"""
    SELECT COUNT(DISTINCT hospitalization_id)
    FROM read_parquet('{hospial_dx_filepath}')
    WHERE CAST(hospitalization_id AS VARCHAR) IN (SELECT hospitalization_id FROM all_ids_df)
""").fetchone()[0]
print(f"Hospitalization IDs present in hospital_dx: {n_present} out of "
      f"{len(all_decedent_inpatient_hosp_ids)}")
strobe_counts["5_present_inpatient_hospitalization_ids_in_hospital_dx"] = n_present

n_age_relevant = duckdb.sql(f"""
    SELECT COUNT(DISTINCT hosp.patient_id)
    FROM read_parquet('{hospial_dx_filepath}') hd
    JOIN hospitalization_df hosp ON hd.hospitalization_id = hosp.hospitalization_id
    WHERE CAST(hd.hospitalization_id AS VARCHAR) IN (SELECT hospitalization_id FROM all_ids_df)
      AND CAST(hosp.patient_id AS VARCHAR)     IN (SELECT patient_id      FROM age_relevant_ids_df)
""").fetchone()[0]
strobe_counts["5_age_relevant_in_hospital_dx"] = n_age_relevant

# ---- 0) Load contraindications list from CSV ----
contraindications_df = pl.read_csv(str(CONTRA_FILE))
_contra_norm = contraindications_df.with_columns([
    pl.col("ICD-10-CM").cast(pl.Utf8).str.to_lowercase()
      .str.replace_all(r"[.\s]", "").alias("code_norm")
])

# The list has three arms: cancer, other (also neoplasms) and sepsis.
#
# CLIF-donor excludes on the cancer and other arms only. The sepsis arm is NOT
# an exclusion for any definition — that was settled 2026-09-19 and the former
# clif_donor.apply_sepsis_exclusion switch removed. Sepsis codes are still read,
# but only to build the reporting flag icd10_sepsis (Table 1 row), never to
# exclude anyone.
_CLIF_ARMS = ["cancer", "other"]
contraindication_codes = _contra_norm["code_norm"].to_list()
_sepsis_codes = (
    _contra_norm.filter(pl.col("dx_broad") == "sepsis")["code_norm"].to_list()
)
clif_contraindication_codes = (
    _contra_norm.filter(pl.col("dx_broad").is_in(_CLIF_ARMS))["code_norm"].to_list()
)
print(f"Contraindication list: {len(contraindication_codes)} codes (full list, "
      f"reporting only)")
print(f"CLIF-donor exclusion arms {_CLIF_ARMS}: "
      f"{len(clif_contraindication_codes)} codes")
print(f"Sepsis: {len(_sepsis_codes)} codes, reporting flag only, never an exclusion")

print(f"Loaded {len(contraindication_codes)} contraindication ICD-10 codes")

# ---- 0b) Load comorbidity prefixes (HCV, HTN, DM, Hx CVA) from CSV ----
# These are PREFIX matches (3-4 char ICD blocks) — e.g. 'i10' matches any
# code starting with i10 (i10, i109, i1010, etc.). Lowercase + no periods.
comorbidities_df = pl.read_csv(str(UTILS_DIR / "icd10_comorbidities.csv"))
comorbidity_prefixes: dict[str, list[str]] = {}
for row in comorbidities_df.iter_rows(named=True):
    prefix = str(row["code_prefix"]).strip().lower().replace(".", "")
    key = str(row["comorbidity"]).strip().lower()
    comorbidity_prefixes.setdefault(key, []).append(prefix)
print(f"Loaded comorbidity prefixes: " +
      ", ".join(f"{k}={len(v)}" for k, v in comorbidity_prefixes.items()))

# ---- 1) Compute ICD-10 cause + comorbidity flags via DuckDB SQL ----
# (all_ids_df is already bound above for the diagnostic queries.)
contraindication_codes_df = pd.DataFrame({"code": contraindication_codes})
sepsis_codes_df = pd.DataFrame({"code": _sepsis_codes})
clif_contraindication_codes_df = pd.DataFrame({"code": clif_contraindication_codes})

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
    WHERE CAST(hospitalization_id AS VARCHAR) IN (SELECT hospitalization_id FROM all_ids_df)
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
        CASE WHEN sys IN ('icd10','icd10cm') AND dx_norm IN (SELECT code FROM contraindication_codes_df) THEN true ELSE false END AS icd10_contraindication,
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
        -- Sepsis kept as a reporting flag regardless of whether it is applied
        -- as an exclusion, so the row can still be shown in Table 1.
        CASE WHEN sys IN ('icd10','icd10cm') AND dx_norm IN (SELECT code FROM sepsis_codes_df) THEN true ELSE false END AS icd10_sepsis,
        -- The arms CLIF-donor excludes on: cancer and "other". The full list
        -- above additionally holds the sepsis arm, which is reporting only.
        CASE WHEN sys IN ('icd10','icd10cm') AND dx_norm IN (SELECT code FROM clif_contraindication_codes_df) THEN true ELSE false END AS icd10_contraindication_clif,
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
    BOOL_OR(icd10_sepsis) AS icd10_sepsis,
    BOOL_OR(icd10_contraindication_clif) AS icd10_contraindication_clif,
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
            "icd10_brain_death", "icd10_sepsis", "icd10_contraindication_clif",
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
    _lo, _hi = CALC_CAUSE_RANGES["external_causes"]
    dq_flags.append({
        "site": site_name, "flag": "calc_external_cause_codes_absent", "severity": "high",
        "detail": (f"No decedent carries an external-cause code ({_lo}-{_hi}) although injury "
                   f"codes (S00-T88) are present, so clif_hospital_diagnosis appears to omit "
                   f"external-cause codes. CALC at this site reflects ischemic heart disease "
                   f"and cerebrovascular disease only and is not comparable with sites that "
                   f"record external causes.")})
    print(f"  WARNING  no external-cause codes ({_lo}-{_hi}) although injury codes are present: "
          f"CALC lacks its external-cause arm at this site (see data_quality_flags.csv)")
_n_without_dx = len(all_decedent_inpatient_hosp_ids) - n_present
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
        pl.col("icd10_sepsis").fill_null(False),
        pl.col("icd10_contraindication_clif").fill_null(False),
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

# Which diagnosis position defines the cause of death. Set in
# config/donor_criteria.yaml; invalid values stop the run rather than silently
# falling back, because the wrong variant is a plausible-looking wrong answer.
_POSITION = CALC_POSITION
if _POSITION not in CALC_POSITIONS:
    raise SystemExit(
        f"calc.diagnosis_position is '{_POSITION}'; expected one of "
        f"{sorted(CALC_POSITIONS)} — fix config/donor_criteria.yaml")
if CALC_APPLY_CONTRAINDICATIONS:
    raise SystemExit(
        "calc.apply_contraindications is true. CALC applies no contraindication "
        "codes (42 CFR 486.302; CMS-3380-F pp. 45, 51). See DECISIONS D-43.")

print(f"CALC diagnosis position: {_POSITION} -> {CALC_POSITIONS[_POSITION]}")
for _name, _col in CALC_POSITIONS.items():
    _n = final_cohort_df.filter(pl.col('age_75_less') & pl.col(_col))["patient_id"].n_unique()
    strobe_counts[f"calc_qualified_{_name}"] = _n
    print(f"  age {_AGE_OP_NAME} {AGE_MAX:g} and cause ({_name:11}): {_n:>7,}"
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

strobe_counts

################################################################################
# IMV — streamed via DuckDB on clif_respiratory_support.parquet
################################################################################

resp_filepath = f"{tables_path}/clif_respiratory_support.{file_type}"
print("Processing IMV data with DuckDB...")

final_cohort_for_imv = final_cohort_df.select([
    "hospitalization_id", "patient_id", "encounter_block", "final_death_dttm"
]).to_pandas()

imv_query = f"""
WITH imv_data AS (
    SELECT
        hospitalization_id,
        recorded_dttm,
        device_category
    FROM read_parquet('{resp_filepath}')
    WHERE LOWER(TRIM(device_category)) = 'imv'
        AND hospitalization_id IN (SELECT hospitalization_id FROM final_cohort_for_imv)
),
imv_with_death AS (
    SELECT
        i.hospitalization_id,
        i.recorded_dttm,
        f.patient_id,
        f.encounter_block,
        f.final_death_dttm,
        EXTRACT(EPOCH FROM (f.final_death_dttm - i.recorded_dttm)) / 3600 AS hr_2death_last_imv
    FROM imv_data i
    INNER JOIN final_cohort_for_imv f ON i.hospitalization_id = f.hospitalization_id
),
-- Get latest IMV record per patient first, then apply the time window
latest_imv_per_patient AS (
    SELECT
        patient_id,
        hospitalization_id,
        encounter_block,
        final_death_dttm,
        recorded_dttm,
        hr_2death_last_imv,
        ROW_NUMBER() OVER (
            PARTITION BY patient_id
            ORDER BY recorded_dttm DESC, hospitalization_id ASC
        ) AS rn
    FROM imv_with_death
)
SELECT
    patient_id,
    hospitalization_id,
    encounter_block,
    final_death_dttm,
    recorded_dttm,
    hr_2death_last_imv
FROM latest_imv_per_patient
WHERE rn = 1
    AND hr_2death_last_imv <= {IMV_HOURS}
    AND hr_2death_last_imv >= -{IMV_POST_DEATH_HOURS}
"""

resp_expired_cohort = pl.from_pandas(duckdb.sql(imv_query).df())
_require_matches(resp_expired_cohort.height, "respiratory_support", "device_category", "imv")

# Add imv_48hr_expire flag to final_cohort_df first (so we can apply the age
# filter when counting). True if patient_id appears in resp_expired_cohort.
imv_48hr_expire_patients = resp_expired_cohort.select(["patient_id"]).unique().with_columns(
    pl.lit(True).alias("imv_48hr_expire")
)
final_cohort_df = final_cohort_df.join(imv_48hr_expire_patients, on="patient_id", how="left")
final_cohort_df = final_cohort_df.with_columns(pl.col("imv_48hr_expire").fill_null(False))

# Materialize the canonical "Died_While_IMV" cohort flag = age <=75 AND IMV
# within 48h before death. This is the same population shown as STROBE Stage 3
# and used as the Table 1 "Died_While_IMV" column and the Table 2 denominator,
# so all three stay consistent regardless of where the population gets read.
final_cohort_df = final_cohort_df.with_columns(
    (pl.col("imv_48hr_expire") & pl.col("age_75_less")).alias("died_while_imv")
)

imv_48hr_expire = final_cohort_df.filter(pl.col("died_while_imv"))["patient_id"].n_unique()
# Named for what it counts: IMV within 48 h AND age <= 75. The bare
# "imv_48hr_expire" flag, without the age cap, is the Ventilated Patient
# definition and is counted separately.
strobe_counts["6_died_while_imv_age_le75"] = imv_48hr_expire
print(f"✓ Died while receiving IMV (age {_AGE_OP_NAME}{AGE_MAX:g} + IMV <={IMV_HOURS:g}h): {imv_48hr_expire}")

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
final_cohort_for_crrt = final_cohort_df.select([
    "hospitalization_id", "final_death_dttm"
]).to_pandas()

crrt_query = f"""
WITH crrt_data AS (
    SELECT
        hospitalization_id,
        recorded_dttm
    FROM read_parquet('{crrt_filepath}')
    WHERE hospitalization_id IN (SELECT hospitalization_id FROM final_cohort_for_crrt)
),
crrt_with_death AS (
    SELECT
        c.hospitalization_id,
        c.recorded_dttm,
        f.final_death_dttm,
        EXTRACT(EPOCH FROM (f.final_death_dttm - c.recorded_dttm)) / 3600 AS hrs_before_death
    FROM crrt_data c
    INNER JOIN final_cohort_for_crrt f ON c.hospitalization_id = f.hospitalization_id
    WHERE c.recorded_dttm <= f.final_death_dttm
)
SELECT DISTINCT hospitalization_id
FROM crrt_with_death
WHERE hrs_before_death <= {CRRT_HOURS} AND hrs_before_death >= 0
"""

crrt_48h_result = pl.from_pandas(duckdb.sql(crrt_query).df())
# No CRRT row for anyone in the cohort reads as "nobody was on dialysis", which
# makes more patients kidney-eligible. That is correct at a hospital that does
# not provide CRRT and wrong where the table is empty or keyed differently, and
# the two cannot be told apart from here, so it is flagged rather than fatal.
_n_crrt_rows = duckdb.sql(f"""
    SELECT COUNT(*) FROM read_parquet('{crrt_filepath}')
    WHERE hospitalization_id IN (SELECT hospitalization_id FROM final_cohort_for_crrt)
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
    on_crrt_flag, on='hospitalization_id', how='left'
).with_columns(pl.col('on_crrt_48h_before_death').fill_null(False))

on_crrt_n = final_cohort_df.filter(pl.col('on_crrt_48h_before_death'))['patient_id'].n_unique()
print(f"✓ Patients on CRRT within {CRRT_HOURS:g}h before death: {on_crrt_n}")

# ============================================
# Organ-quality labs (creatinine, bili, AST, ALT) — streamed via DuckDB
# ============================================
print("Processing Labs data with DuckDB...")
final_cohort_for_labs = final_cohort_df.select([
    "patient_id", "hospitalization_id", "final_death_dttm",
]).to_pandas()

_LAB_VALUE_USABLE = usable_number_sql("l.lab_value_numeric")
labs_query = f"""
WITH labs_data AS (
    SELECT
        hospitalization_id,
        lab_collect_dttm,
        LOWER(TRIM(lab_category)) AS lab_category,
        lab_value_numeric
    FROM read_parquet('{labs_filepath}')
    WHERE hospitalization_id IN (SELECT hospitalization_id FROM final_cohort_for_labs)
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
    INNER JOIN final_cohort_for_labs f ON l.hospitalization_id = f.hospitalization_id
    WHERE l.lab_collect_dttm <= f.final_death_dttm
      -- Only results that carry a number can be the "last value". A text-only
      -- result (haemolysed, see note) or a NaN at the latest timestamp otherwise
      -- won the ranking below and nulled the value, which reads as missing,
      -- which is ineligible.
      AND {_LAB_VALUE_USABLE}
),
latest_creatinine AS (
    -- Every "last value before death" selection breaks ties deterministically.
    -- Without a tiebreaker two results at the same timestamp were picked
    -- arbitrarily and the same code gave different counts on consecutive runs
    -- (organ_kidney_eligible 2,906 vs 2,907 at UCMC, 2026-09-28).
    SELECT
        hospitalization_id,
        lab_value_numeric AS creatinine_value,
        lab_collect_dttm AS creatinine_dttm
    FROM (
        SELECT
            hospitalization_id,
            lab_value_numeric,
            lab_collect_dttm,
            ROW_NUMBER() OVER (PARTITION BY hospitalization_id
                               ORDER BY lab_collect_dttm DESC, lab_value_numeric DESC) AS rn
        FROM labs_with_death
        WHERE lab_category = 'creatinine'
    ) ranked
    WHERE rn = 1
),
latest_liver AS (
    SELECT
        hospitalization_id,
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
            hospitalization_id,
            lab_category,
            lab_value_numeric,
            lab_collect_dttm,
            ROW_NUMBER() OVER (PARTITION BY hospitalization_id, lab_category
                               ORDER BY lab_collect_dttm DESC, lab_value_numeric DESC) AS rn
        FROM labs_with_death
        WHERE lab_category IN ('bilirubin_total', 'ast', 'alt', 'bun', 'sodium')
    ) ranked
    WHERE rn = 1
    GROUP BY hospitalization_id
)
SELECT DISTINCT
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
FROM final_cohort_for_labs f
LEFT JOIN latest_creatinine c ON f.hospitalization_id = c.hospitalization_id
LEFT JOIN latest_liver l ON f.hospitalization_id = l.hospitalization_id
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
        (pl.col('creatinine_value') < CREATININE_MAX) &
        (~pl.col('on_crrt_48h_before_death'))
    ).alias('kidney_eligible'),

    # Liver criteria: all three labs recorded AND values within limits
    (
        (pl.col('bilirubin_total_value').is_not_null()) &
        (pl.col('ast_value').is_not_null()) &
        (pl.col('alt_value').is_not_null()) &
        (pl.col('bilirubin_total_value') < BILIRUBIN_MAX) &
        (pl.col('ast_value') < AST_MAX) &
        (pl.col('alt_value') < ALT_MAX)
    ).alias('liver_eligible'),

    # BMI criteria: <= 50
    (
        (pl.col('bmi').is_not_null()) &
        (pl.col('bmi') <= BMI_MAX)
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

################################################################################
# Microbiology
# Identify negative blood cultures and patients with no cultures in last 48h
################################################################################

# Microbiology — streamed via DuckDB on clif_microbiology_culture.parquet
print("Processing microbiology data with DuckDB...")
final_cohort_for_micro = final_cohort_df.select([
    'hospitalization_id', 'final_death_dttm'
]).to_pandas()

micro_query = f"""
WITH blood_cultures AS (
    SELECT
        hospitalization_id,
        collect_dttm,
        organism_category
    FROM read_parquet('{tables_path}/clif_microbiology_culture.{file_type}')
    WHERE LOWER(TRIM(fluid_category)) = 'blood_buffy'
        AND LOWER(TRIM(method_category)) = 'culture'
        AND hospitalization_id IN (SELECT hospitalization_id FROM final_cohort_for_micro)
),
cultures_with_death AS (
    SELECT
        b.hospitalization_id,
        b.collect_dttm,
        b.organism_category,
        f.final_death_dttm,
        EXTRACT(EPOCH FROM (f.final_death_dttm - b.collect_dttm)) / 3600 AS hrs_before_death
    FROM blood_cultures b
    INNER JOIN final_cohort_for_micro f ON b.hospitalization_id = f.hospitalization_id
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
    WHERE hrs_before_death >= 0 AND hrs_before_death <= {CULTURE_HOURS}
),
positive_cultures AS (
    SELECT DISTINCT hospitalization_id
    FROM cultures_48h
    WHERE is_negative_culture = false
)
SELECT
    f.hospitalization_id,
    CASE WHEN p.hospitalization_id IS NULL THEN true ELSE false END AS no_positive_culture_48hrs
FROM final_cohort_for_micro f
LEFT JOIN positive_cultures p ON f.hospitalization_id = p.hospitalization_id
"""

# A cohort of in-hospital deaths with no blood culture at all means the filter
# matched nothing, and then every patient would pass as "no positive culture".
_n_blood_cx = duckdb.sql(f"""
    SELECT COUNT(*) FROM read_parquet('{tables_path}/clif_microbiology_culture.{file_type}')
    WHERE LOWER(TRIM(fluid_category)) = 'blood_buffy'
      AND LOWER(TRIM(method_category)) = 'culture'
      AND hospitalization_id IN (SELECT hospitalization_id FROM final_cohort_for_micro)
""").fetchone()[0]
_require_matches(_n_blood_cx, "microbiology_culture", "fluid_category",
                 "blood_buffy' with method_category = 'culture")
no_positive_culture_flag = pl.from_pandas(duckdb.sql(micro_query).df())
final_cohort_df = final_cohort_df.join(
    no_positive_culture_flag, on='hospitalization_id', how='left'
).with_columns(pl.col('no_positive_culture_48hrs').fill_null(False))

# STROBE tracking
no_positive_culture_n = final_cohort_df.filter(pl.col('no_positive_culture_48hrs'))['patient_id'].n_unique()
positive_culture_n = final_cohort_df.filter(~pl.col('no_positive_culture_48hrs'))['patient_id'].n_unique()
strobe_counts["no_positive_culture_48hrs"] = no_positive_culture_n
strobe_counts["positive_culture_48hrs"] = positive_culture_n
print(f"  Patients with no positive cultures in last {CULTURE_HOURS:g}h: {no_positive_culture_n}")
print(f"  Patients with positive cultures in last {CULTURE_HOURS:g}h: {positive_culture_n}")

final_cohort_df.columns

################################################################################
# CLIF Eligible Donor
# Medically eligible potential deceased abdominal organ donor (CLIF-eligible-donors):
# * From ALL inpatient deaths (ensure death location = ED, ward, stepdown, ICU)
# * Age < 75
# * On invasive mechanical ventilation
# * IF death date/time available: within 48h of death
# * IF no death date/time available: at time of last recorded vital signs
# * No contraindications
# * CLIF Microbiology_culture:
# * No positive blood cultures within 2 days - 'no_positive_culture_48hrs'
# * Hospital diagnosis (ICD based) -- 'icd10_contraindication',
# * Cancer
# * Severe sepsis
# * Pass the potential organ quality assessment check (independent assessment) using last recorded lab values, as defined by CMS:- organ_check_pass
# * Kidney: recorded creatinine, Cr < 4 AND not on CRRT
# * Liver: recorded TB, AST, ALT and
# * Total bilirubin < 4
# * AST < 700
# * ALT < 700
# * BMI <= 50
################################################################################

# ============================================
# Create CLIF-eligible-donors flag
# ============================================

final_cohort_df = final_cohort_df.with_columns([
    # Overall CLIF-eligible-donors flag
    (
        # 2. Age < 75
        (pl.col('age_75_less')) &
        # 3. On invasive mechanical ventilation (within 48h of death)
        (pl.col('imv_48hr_expire')) &
        # 4. No contraindicating cancer diagnosis. Cancer and "other" arms of
        #    utils/icd10_contraindications.csv. Sepsis is never an exclusion.
        (~pl.col('icd10_contraindication_clif')) &
        # 5. No positive blood cultures within 48h
        (pl.col('no_positive_culture_48hrs')) &
        # 6. Pass organ quality assessment (kidney OR liver AND BMI)
        (pl.col('organ_check_pass'))
    ).alias('clif_eligible_donors')
])

# Count for STROBE tracking
clif_eligible_n = final_cohort_df.filter(pl.col('clif_eligible_donors'))['patient_id'].n_unique()
strobe_counts["clif_eligible_donors"] = clif_eligible_n

################################################################################
# CLIF Donor Organ Eligibility Statistics
################################################################################
# Individual lab threshold flags for Table One
final_cohort_df = final_cohort_df.with_columns([
    # Terminal creatinine < 4
    (
        (pl.col('creatinine_value').is_not_null()) &
        (pl.col('creatinine_value') < CREATININE_MAX)
    ).alias('creatinine_lt_4'),

    # Terminal bilirubin < 4
    (
        (pl.col('bilirubin_total_value').is_not_null()) &
        (pl.col('bilirubin_total_value') < BILIRUBIN_MAX)
    ).alias('bilirubin_lt_4'),

    # Terminal AST < 700
    (
        (pl.col('ast_value').is_not_null()) &
        (pl.col('ast_value') < AST_MAX)
    ).alias('ast_lt_700'),

    # Terminal ALT < 700
    (
        (pl.col('alt_value').is_not_null()) &
        (pl.col('alt_value') < ALT_MAX)
    ).alias('alt_lt_700'),
])

# BMI-filtered versions of terminal lab thresholds (for fair comparison)
final_cohort_df = final_cohort_df.with_columns([
    # Terminal creatinine < 4 (BMI ≤50 only)
    (
        (pl.col('bmi_eligible') == True) &
        (pl.col('creatinine_value').is_not_null()) &
        (pl.col('creatinine_value') < CREATININE_MAX)
    ).alias('creatinine_lt_4_bmi50'),

    # Terminal bilirubin < 4 (BMI ≤50 only)
    (
        (pl.col('bmi_eligible') == True) &
        (pl.col('bilirubin_total_value').is_not_null()) &
        (pl.col('bilirubin_total_value') < BILIRUBIN_MAX)
    ).alias('bilirubin_lt_4_bmi50'),

    # Terminal AST < 700 (BMI ≤50 only)
    (
        (pl.col('bmi_eligible') == True) &
        (pl.col('ast_value').is_not_null()) &
        (pl.col('ast_value') < AST_MAX)
    ).alias('ast_lt_700_bmi50'),

    # Terminal ALT < 700 (BMI ≤50 only)
    (
        (pl.col('bmi_eligible') == True) &
        (pl.col('alt_value').is_not_null()) &
        (pl.col('alt_value') < ALT_MAX)
    ).alias('alt_lt_700_bmi50'),
])

# Filter to CLIF donors only (883 patients)
clif_donors_df = final_cohort_df.filter(pl.col('clif_eligible_donors'))

# Count organ eligibility among CLIF donors
clif_kidney_eligible_n = clif_donors_df.filter(pl.col('kidney_eligible'))['patient_id'].n_unique()
clif_liver_eligible_n = clif_donors_df.filter(pl.col('liver_eligible'))['patient_id'].n_unique()
clif_both_eligible_n = clif_donors_df.filter(
    pl.col('kidney_eligible') & pl.col('liver_eligible')
)['patient_id'].n_unique()

# Calculate percentages
clif_kidney_pct = (clif_kidney_eligible_n / clif_eligible_n * 100) if clif_eligible_n > 0 else 0
clif_liver_pct = (clif_liver_eligible_n / clif_eligible_n * 100) if clif_eligible_n > 0 else 0
clif_both_pct = (clif_both_eligible_n / clif_eligible_n * 100) if clif_eligible_n > 0 else 0

# Add to strobe_counts for tracking
strobe_counts["clif_kidney_eligible"] = clif_kidney_eligible_n
strobe_counts["clif_liver_eligible"] = clif_liver_eligible_n
strobe_counts["clif_both_kidney_liver_eligible"] = clif_both_eligible_n

print(f"\nCLIF Donor Organ Eligibility (n={clif_eligible_n}):")
print(f"  Kidney eligible (Cr <{CREATININE_MAX:g} AND not on CRRT): {clif_kidney_eligible_n} ({clif_kidney_pct:.1f}%)")
print(f"  Liver eligible (Bili <{BILIRUBIN_MAX:g} AND AST <{AST_MAX:g} AND ALT <{ALT_MAX:g}): {clif_liver_eligible_n} ({clif_liver_pct:.1f}%)")
print(f"  Both kidney AND liver eligible: {clif_both_eligible_n} ({clif_both_pct:.1f}%)")

################################################################################
# Patient assessments
################################################################################

# Patient assessments — streamed via DuckDB on clif_patient_assessments.parquet
print("Processing patient assessments with DuckDB...")
final_cohort_for_assessments = final_cohort_df.select([
    "hospitalization_id", "final_death_dttm"
]).to_pandas()

assessments_query = f"""
WITH assessments_filtered AS (
    SELECT
        hospitalization_id,
        recorded_dttm,
        LOWER(TRIM(assessment_category)) AS assessment_category,
        numerical_value
    FROM read_parquet('{tables_path}/clif_patient_assessments.{file_type}')
    WHERE LOWER(TRIM(assessment_category)) IN ('gcs_total', 'rass')
        AND numerical_value IS NOT NULL
        AND hospitalization_id IN (SELECT hospitalization_id FROM final_cohort_for_assessments)
),
with_death_time AS (
    SELECT
        a.hospitalization_id,
        a.assessment_category,
        a.numerical_value,
        ABS(EXTRACT(EPOCH FROM (f.final_death_dttm - a.recorded_dttm))) AS abs_time_to_death
    FROM assessments_filtered a
    INNER JOIN final_cohort_for_assessments f ON a.hospitalization_id = f.hospitalization_id
),
closest_per_category AS (
    SELECT
        hospitalization_id,
        assessment_category,
        numerical_value,
        ROW_NUMBER() OVER (
            PARTITION BY hospitalization_id, assessment_category
            ORDER BY abs_time_to_death, numerical_value ASC
        ) AS rn
    FROM with_death_time
)
SELECT
    hospitalization_id,
    MAX(CASE WHEN assessment_category = 'gcs_total' THEN numerical_value END) AS gcs_total_value,
    MAX(CASE WHEN assessment_category = 'rass'      THEN numerical_value END) AS rass_value
FROM closest_per_category
WHERE rn = 1
GROUP BY hospitalization_id
"""

if "patient_assessments" in OPTIONAL_UNUSABLE:
    # Optional table. Without it GCS and RASS are null for everyone and their
    # Table 2 rows report as unavailable; no definition depends on them.
    patient_gcs_rass = pl.DataFrame(schema={
        "hospitalization_id": final_cohort_df.schema["hospitalization_id"],
        "gcs_total_value": pl.Float64, "rass_value": pl.Float64})
    print("  patient assessments SKIPPED (clif_patient_assessments absent or incomplete): "
          "GCS and RASS unavailable")
else:
    patient_gcs_rass = pl.from_pandas(duckdb.sql(assessments_query).df())
    print(f"✓ Processed assessments for {len(patient_gcs_rass)} hospitalizations")

final_cohort_df = final_cohort_df.join(
    patient_gcs_rass, on='hospitalization_id', how='left'
)

# ================================================================================
# ENSURE PATIENT-LEVEL ANALYSIS
# ================================================================================
print("\n" + "="*80)
print("FINALIZING PATIENT-LEVEL COHORT")
print("="*80)

# Step 1: First, remove encounter-level identifiers
print("Step 1: Renaming encounter-level identifiers to terminal_* ...")
# The cohort is already deduplicated to one row per patient (their LAST death
# encounter), so these identifiers describe the TERMINAL hospitalization. They
# are renamed rather than dropped because downstream steps need them to attach
# the hospital where the patient died (adt.hospital_id) and to join diagnoses.
# The frame stays patient-level: one row per patient_id either way.
for _src, _dst in (('hospitalization_id', 'terminal_hospitalization_id'),
                   ('encounter_block', 'terminal_encounter_block')):
    if _src in final_cohort_df.columns:
        final_cohort_df = final_cohort_df.rename({_src: _dst})
        print(f"  renamed {_src} -> {_dst}")
columns_to_drop = []

if columns_to_drop:
    final_cohort_df = final_cohort_df.drop(columns_to_drop)
    print(f"✓ Dropped: {', '.join(columns_to_drop)}")
else:
    print("✓ No encounter-level identifiers found to drop")

# Step 2: Sanity check — early dedup upstream means we should already be 1 row/patient.
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
_write_parquet(final_cohort_df, OUTPUT_INTERMEDIATE_DIR / "final_cohort_df.parquet")
# Long, not wide. One row per count, so three sites concatenate even when a
# site is missing an optional table and therefore a metric. `value` keeps the
# raw text (one entry, age_source, is not a number); `n` is the numeric form.
_strobe_long = pl.DataFrame({
    "site": [site_name] * len(strobe_counts),
    "order": list(range(1, len(strobe_counts) + 1)),
    "metric": list(strobe_counts.keys()),
    "value": [str(v) for v in strobe_counts.values()],
}).with_columns(pl.col("value").cast(pl.Int64, strict=False).alias("n"))
_strobe_long.write_csv(OUTPUT_FINAL_DIR / "strobe_counts.csv")

################################################################################
# Definitions that need the finished cohort
# Merged from the former 02_apply_definitions.py on 2026-09-25 (D-49), so that
# every definition is computed once, from one read and one normalisation of the
# diagnosis table. Possible Donor was removed from the study in the same change.
################################################################################

from utils.audit import StageAudit                                  # noqa: E402
_audit = StageAudit(site_name, OUTPUT_FINAL_DIR, label="definitions")

# ── Ventilated Patient ───────────────────────────────────────────────────────
# Both variants are computed: Table 1 of the manuscript says "No restrictions",
# so the no-age-limit flag is the reported one, but the age-capped variant is
# kept so the difference is a reported number rather than a re-run.
_vent_age = VENT_APPLY_AGE_LIMIT
final_cohort_df = final_cohort_df.with_columns([
    pl.col("imv_48hr_expire").alias("ventilated_patient_no_age_limit"),
    (pl.col("imv_48hr_expire") & pl.col("age_75_less")).alias("ventilated_patient_age_le75"),
])
final_cohort_df = final_cohort_df.with_columns(
    (pl.col("ventilated_patient_age_le75") if _vent_age
     else pl.col("ventilated_patient_no_age_limit")).alias("ventilated_patient"))
_audit.record("20_ventilated_no_age_limit", final_cohort_df,
              final_cohort_df.filter(pl.col("ventilated_patient_no_age_limit")),
              key="patient_id",
              rule=f"IMV within {IMV_HOURS:g}h of death, NO age restriction (Table 1 as written)")
_audit.record("21_ventilated_age_le75", final_cohort_df,
              final_cohort_df.filter(pl.col("ventilated_patient_age_le75")),
              key="patient_id",
              rule=f"IMV within {IMV_HOURS:g}h of death AND age {_AGE_OP_NAME} {AGE_MAX:g}")

# ── hospital identity ────────────────────────────────────────────────────────
# The TERMINAL ADT record is the hospital where the patient died, which is the
# unit CMS and SRTR attribute a donor to.
# _XWALK was loaded and checked in the setup checks at the top of this file.
_hid_col = ("terminal_hospitalization_id" if "terminal_hospitalization_id"
            in final_cohort_df.columns else "hospitalization_id")
_ids = pd.DataFrame({"hospitalization_id":
                     final_cohort_df[_hid_col].cast(pl.Utf8).unique().to_list()})
_con = duckdb.connect()
_con.register("cohort_ids", _ids)
_adt_h = _con.sql(f"""
    SELECT CAST(hospitalization_id AS VARCHAR) AS {_hid_col},
           lower(trim(CAST(hospital_id AS VARCHAR))) AS hospital_id_key,
           lower(trim(CAST(hospital_type AS VARCHAR))) AS hospital_type,
           ROW_NUMBER() OVER (PARTITION BY hospitalization_id
                              ORDER BY out_dttm DESC NULLS LAST,
                                       lower(trim(CAST(hospital_id AS VARCHAR))) ASC) rn
    FROM read_parquet('{tables_path}/clif_adt.{file_type}')
    WHERE CAST(hospitalization_id AS VARCHAR) IN (SELECT hospitalization_id FROM cohort_ids)
      AND hospital_id IS NOT NULL
    QUALIFY rn = 1
""").pl().drop("rn")

_rows = [r for r in _XWALK if r["site"] == site_name]
final_cohort_df = final_cohort_df.with_columns(pl.col(_hid_col).cast(pl.Utf8)).join(
    _adt_h, on=_hid_col, how="left")
if _rows:
    _xw = pl.DataFrame([{
        "hospital_id_key": str(r["hospital_id"]).lower().strip(),
        "analytic_hospital_id": r["analytic_hospital_id"],
        "srtr_ccn_id": r.get("srtr_ccn_id"),
        "ccn_facility_type": r.get("ccn_facility_type"),
        "hospital_include": bool(r.get("include_flag")),
    } for r in _rows])
    final_cohort_df = final_cohort_df.join(_xw, on="hospital_id_key", how="left")
else:
    final_cohort_df = final_cohort_df.with_columns(
        [pl.lit(None, pl.Utf8).alias(c) for c in
         ("analytic_hospital_id", "srtr_ccn_id", "ccn_facility_type")]
    ).with_columns(pl.lit(False).alias("hospital_include"))
final_cohort_df = final_cohort_df.with_columns(
    pl.col("hospital_include").fill_null(False))

# Anonymous labels for anything that leaves the site, numbered by CCN ascending
# so the mapping is deterministic across runs. The real CCN never appears in
# shareable output; the site keeps the key locally.
_ccns = sorted({c for c in final_cohort_df["srtr_ccn_id"].drop_nulls().unique().to_list()})
_labels = {c: f"{site_name}_hospital_{i}" for i, c in enumerate(_ccns, 1)}
final_cohort_df = final_cohort_df.with_columns(
    pl.col("srtr_ccn_id").replace_strict(_labels, default=None).alias("hospital_label"))
pl.DataFrame([{"site": site_name, "srtr_ccn_id": c, "hospital_label": l,
               "analytic_hospital_id": f"{site_name}_{c}"} for c, l in _labels.items()]
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
# hospital that transfers dying patients out contributes zero by design — all 64
# of NU Woodstock's deaths are attributed to McHenry and Huntley. Warn, so a
# genuinely missing extract is still visible, but do not stop.
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
    pl.col("analytic_hospital_id").is_null() & pl.col("hospital_id_key").is_not_null()
)["hospital_id_key"].unique().to_list())
if _unmapped:
    raise SystemExit(
        f"hospital_id value(s) {_unmapped} contribute decedents but have no row for site "
        f"'{site_name}' in config/hospital_crosswalk.yaml. Ask the coordinating centre to add them.")

_audit.record("10_hospital_identity", final_cohort_df, final_cohort_df, key="patient_id",
              rule="terminal ADT hospital_id -> analytic_hospital_id via crosswalk",
              n_unmapped=int(final_cohort_df.filter(
                  pl.col("analytic_hospital_id").is_null()).height))

# ── SRTR denominator restriction ─────────────────────────────────────────────
_analytic = final_cohort_df.filter(pl.col("hospital_include"))
_audit.record("30_srtr_linkable_denominator", final_cohort_df, _analytic, key="patient_id",
              rule="hospital retained only if include_flag AND srtr_ccn_id resolves in SRTR",
              reason="hospital has no SRTR-resolvable CCN; keeping it would inflate "
                     "the denominator")
final_cohort_df = final_cohort_df.with_columns(
    pl.col("hospital_include").alias("in_srtr_denominator"))

# Downstream steps join on `hospitalization_id`. The cohort frame carries it as
# `terminal_hospitalization_id` (renamed above to make the patient-level grain
# explicit); the former 02 renamed it back before writing, so keep that contract.
if _hid_col != "hospitalization_id":
    final_cohort_df = final_cohort_df.rename({_hid_col: "hospitalization_id"})
_write_parquet(final_cohort_df, OUTPUT_INTERMEDIATE_DIR / "cohort_with_definitions.parquet")

_counts = {
    "site": site_name,
    "n_inpatient_deaths": final_cohort_df["patient_id"].n_unique(),
    "n_srtr_linkable": _analytic["patient_id"].n_unique(),
    "clif_donor": int(final_cohort_df.filter(pl.col("clif_eligible_donors"))["patient_id"].n_unique()),
    "calc": int(final_cohort_df.filter(pl.col("calc_flag"))["patient_id"].n_unique()),
    "ventilated_no_age_limit": int(final_cohort_df.filter(
        pl.col("ventilated_patient_no_age_limit"))["patient_id"].n_unique()),
    "ventilated_age_le75": int(final_cohort_df.filter(
        pl.col("ventilated_patient_age_le75"))["patient_id"].n_unique()),
    "n_analytic_hospitals": int(_analytic["analytic_hospital_id"].n_unique()),
    # The arms of the CALC cause criterion (any age, any diagnosis position) and
    # the diagnosis coverage behind them, so a site whose extract lacks an arm
    # is visible in the pooled table rather than hidden inside its CALC count.
    "icd10_ischemic": int(final_cohort_df.filter(pl.col("icd10_ischemic"))["patient_id"].n_unique()),
    "icd10_cerebro": int(final_cohort_df.filter(pl.col("icd10_cerebro"))["patient_id"].n_unique()),
    "icd10_external": int(final_cohort_df.filter(pl.col("icd10_external"))["patient_id"].n_unique()),
    "n_decedents_without_dx": int(_n_without_dx),
}
pl.DataFrame([_counts]).write_csv(OUTPUT_FINAL_DIR / "definition_counts.csv")
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
_SREF = OUTPUT_FINAL_DIR / "srtr_ref"
_SREF.mkdir(parents=True, exist_ok=True)
_cov = final_cohort_df.select(
    ["patient_id", "hospital_id_key", "hospital_label", "srtr_ccn_id",
     "ccn_facility_type", "hospital_type", "in_srtr_denominator", "final_death_dttm"]
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

_hc = (_cov.group_by(["hospital_id_key", "hospital_label", "srtr_ccn_id",
                      "ccn_facility_type", "hospital_type", "in_srtr_denominator"])
       .agg(pl.col("patient_id").n_unique().alias("n_decedents"),
            pl.col("death_year").min().alias("first_year"),
            pl.col("death_year").max().alias("last_year"))
       .with_columns(pl.lit(site_name).alias("site"))
       .with_columns((pl.col("last_year") - pl.col("first_year") + 1).alias("n_years_spanned"))
       .sort("n_decedents", descending=True))
_hc.write_csv(_SREF / "hospital_coverage.csv")

pl.DataFrame([{
    "site": site_name,
    "n_hospital_ids": _cov["hospital_id_key"].n_unique(),
    "n_hospital_ids_in_srtr_denominator":
        _cov.filter(pl.col("in_srtr_denominator"))["hospital_id_key"].n_unique(),
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
