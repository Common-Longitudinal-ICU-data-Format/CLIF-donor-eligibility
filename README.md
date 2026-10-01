# CLIF Donor Identifier

Identifies medically eligible deceased organ donors among in-hospital decedents in
CLIF, and compares that definition against the two administrative definitions in
current use.

**CLIF version:** 2.1 · **Study window:** 2020-01-01 to 2025-12-31

## Definitions

All three are computed on the same cohort, in `01_cohort_and_definitions.py`.

| Criterion | CLIF-donor | CALC | Ventilated Patient |
|---|:--:|:--:|:--:|
| In-hospital death (`discharge_category = expired`) | ✓ | ✓ | ✓ |
| Age ≤ 75 at death | ✓ | ✓ | — |
| Cause of death consistent with donation (I20–I25, I60–I69, V01–Y89) | — | ✓ | — |
| IMV within 48 h of death | ✓ | — | ✓ |
| No contraindicating cancer diagnosis | ✓ | — | — |
| No positive blood culture within 48 h of death | ✓ | — | — |
| BMI ≤ 50, and kidney **or** liver function compatible with donation | ✓ | — | — |

Kidney = terminal creatinine < 4 mg/dL and no CRRT within 48 h.
Liver = terminal bilirubin < 4 mg/dL, AST < 700, ALT < 700.

Two points where the code deliberately departs from a naive reading:

- **CALC applies no contraindication codes.** 42 CFR § 486.302 defines it by
  inclusion codes only, and CMS explicitly tested adding exclusions and declined,
  finding "no additive value" (CMS-3380-F pp. 45, 51).
- **Sepsis is never an exclusion** for any definition. Sepsis codes are reported
  in Table 2 but exclude nobody.

Every criterion lives in `config/donor_criteria.yaml`. Nothing is hard-coded in
the scripts: step 01 reads each value and stops if the YAML declares a criterion
the code does not apply. Two keys are declared and knowingly not applied,
`use_poa` and `sepsis_window_hours`; every run prints them.

"IMV within 48 h of death" counts ventilator records from 48 h before to 24 h
after the recorded death time. The 24 h allowance is
`clif_donor.imv_post_death_tolerance_hours`; it exists because some sites record
death as a date, stored as midnight.

## Running it

```bash
# 1. Configure
cp config/config_template.json config/config.json
#    Fill in ALL of: site_name, tables_path, file_type, timezone,
#    project_root, clif_version, hospital_ids
#    hospital_ids must list every hospital_id in clif_adt that has decedents.
#    The run stops if the cohort contains an id you did not declare. A declared
#    id that contributes no decedents is reported as a warning.

# 2. Install (Python 3.10 or newer)
uv sync

# 3. Run, in order
uv run python code/01_cohort_and_definitions.py
uv run python code/02_build_tables.py
uv run python code/03_diagnostics.py
```

Or, if you keep your config as `config/config_<site>.json`, all three at once:
`uv run python run_pipeline.py --sites <site>`.

The run validates the config, the required tables and the required columns
before reading any patient data, and stops with a plain message if something is
wrong. `03_diagnostics.py` finishes by reconciling the STROBE counts, the
CONSORT, the definition counts and the Table 2 denominators against each other,
and fails the run if they disagree.

## What to return

Ship the whole of **`<site>_upload_to_box/`**. It contains aggregate counts
only — no patient-level data.

| file | contents |
|---|---|
| `provenance.md` | **start here.** Cohort cascade, CONSORT per definition, every audit stage, and the reconciliation result |
| `definition_counts.csv` | one row: decedents, each definition's count, the three CALC cause-arm counts, and decedents with no diagnosis rows |
| `data_quality_flags.csv` | conditions that do not stop the run but change what a number means at your site; also shown in `provenance.md` |
| `definition_counts_by_hospital_type.csv` | definition counts for academic and community hospitals |
| `consort_counts.csv` | per-definition cascade, step by step |
| `strobe_counts.csv` | every filter-stage count from step 01 |
| `table2_characteristics.csv`, `table3_clinical_care.csv` | manuscript tables |
| `table_stats_raw.csv` | the same numbers as raw n / denominator, so they pool exactly |
| `tableS2_missingness.csv` | per-variable missingness |
| `hospital_level_counts.csv` | per analytic hospital |
| `exclusion_codes_by_step.csv` | for each CONSORT step, how many patients it excluded and whether a diagnosis code decided it. The per-code breakdown stays in your local folder unless `study.ship_exclusion_code_detail` is true |
| `donor_administrative_code_availability.csv` | whether donor-related procedure codes exist in your extract |
| `decedents_by_location.csv` | decedents by terminal ADT location, and which are in the cohort |
| `missingness_*.csv`, `element_coverage.csv`, `data_availability_by_hospital.csv` | data-quality diagnostics |
| `definition_overlap_upset.{csv,png}` | which patients the definitions share |
| `srtr_ref/` | hospital ids and the calendar years covered, for linking to SRTR centrally |
| `audit/` | stage cards, counts only |
| `run_log.txt` | full stdout |

`output/intermediate_phi/<site>/` holds the patient-level cohort and **never leaves
the site**.

## Repository layout

```
code/01_cohort_and_definitions.py   cohort, then CLIF-donor / CALC / Ventilated
code/02_build_tables.py             clinical-care flags, Tables 2-3, CONSORT
code/03_diagnostics.py              missingness, coverage, overlap, reconciliation
code/coordinating/                  pooling and SRTR linkage — coordinating centre only
config/donor_criteria.yaml          every criterion
config/clif_data_requirements.yaml  required tables, columns and mCIDE values
utils/                              shared python modules
tests/                              unit tests, no patient data needed (uv run pytest)
utils/codes/, utils/*.csv           ICD-10 and procedure code lists
```

## Required CLIF tables

The authoritative list is `config/clif_data_requirements.yaml`, which the run
checks against your extract. Required: `patient`, `hospitalization`, `adt`,
`hospital_diagnosis`, `labs`, `respiratory_support`, `vitals`, `crrt_therapy`,
`microbiology_culture`. Optional, each with a named cost to the manuscript if
absent: `patient_procedures`, `patient_assessments`,
`medication_admin_continuous`, `medication_admin_intermittent`, `ecmo_mcs`,
`position`.

## Engine note

Large tables are queried with DuckDB streaming SQL over the parquet files rather
than loaded into memory, so the pipeline runs at any site scale. Polars handles
the cohort logic. Every "last value before death" selection breaks ties
deterministically, so repeated runs give identical counts.
