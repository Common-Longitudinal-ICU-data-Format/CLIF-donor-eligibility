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


Every criterion lives in `config/donor_criteria.yaml`. How the time of death, linked stays, last
recorded values, age and the hospital of death are handled, with the evidence for each:
[`guides/notes_for_manuscript.md`](guides/notes_for_manuscript.md).


## Running it

```bash
# 1. Configure
cp config/config_template.json config/config.json
#    Fill in ALL of: site_name, tables_path, timezone, project_root,
#    hospital_ids. timezone is your hospital's, e.g. US/Central (not UTC).
#    hospital_ids must list every hospital_id in clif_adt that has decedents.
#    The run stops if the cohort contains an id you did not declare. A declared
#    id that contributes no decedents is reported as a warning.

# 2. Install (Python 3.12; uv installs it if needed)
uv sync

# 3. Run all at once
uv run python run_pipeline.py

# or in in order
uv run python code/01_cohort_and_definitions.py
uv run python code/02_build_tables.py
uv run python code/03_diagnostics.py
```

The run validates the config, the required tables and the required columns
before reading any patient data, and stops with a plain message if something is
wrong. `03_diagnostics.py` finishes by reconciling the STROBE counts, the
CONSORT, the definition counts and the Table 2 denominators against each other,
and fails the run if they disagree.

## What to return

Ship the whole of **`<site>_upload_to_box/`**: aggregate counts only, no patient-level data.
What each file holds is described in [`output/README.md`](output/README.md).
`output/intermediate_phi/<site>/` holds the patient-level cohort and **never leaves the site**.

## Repository layout

```
code/01_cohort_and_definitions.py   cohort, then CLIF-donor / CALC / Ventilated
code/02_build_tables.py             clinical-care flags, Tables 2-3, CONSORT
code/03_diagnostics.py              missingness, coverage, overlap, reconciliation
code/coordinating/                  pooling and SRTR linkage — coordinating centre only
config/donor_criteria.yaml          every criterion
config/clif_data_requirements.yaml  required tables, columns and mCIDE values
utils/                              shared python modules
utils/codes/                        ICD-10 and procedure code lists
```

## Required CLIF tables

The run checks these against your extract before reading any patient data
(`config/clif_data_requirements.yaml`). A required table that is absent or lacks a
column stops the run. An optional table that is absent is reported as missing, with
the manuscript rows it costs, and the run continues.

| table | columns | category values the code looks for | used for |
|---|---|---|---|
| `patient` | `patient_id`, `birth_date`, `death_dttm`, `sex_category`, `race_category`, `ethnicity_category` | — | cohort, age at death, time of death, Table 2 |
| `hospitalization` | `patient_id`, `hospitalization_id`, `admission_dttm`, `discharge_dttm`, `discharge_category`, `age_at_admission`, `admission_type_category` | `discharge_category`: `expired` | in-hospital deaths, linked stays, time of death when `death_dttm` is date-only or missing, age when `birth_date` is missing, length of stay |
| `adt` | `hospitalization_id`, `hospital_id`, `hospital_type`, `in_dttm`, `out_dttm`, `location_category` | `location_category`: `ed`, `ward`, `stepdown`, `icu`, `hospice`; `hospital_type`: `academic`, `community` | cohort locations, hospital of death, ICU length of stay |
| `hospital_diagnosis` | `hospitalization_id`, `diagnosis_code`, `diagnosis_code_format`, `diagnosis_primary`, `poa_present` | `diagnosis_code_format`: `icd10cm` | CALC cause of death, cancer contraindication, Table 2 comorbidities |
| `labs` | `hospitalization_id`, `lab_category`, `lab_value_numeric`, `lab_collect_dttm` | `lab_category`: `creatinine`, `bilirubin_total`, `ast`, `alt`, `bun`, `sodium` | organ quality, Table 2 |
| `respiratory_support` | `hospitalization_id`, `recorded_dttm`, `device_category` | `device_category`: `imv` | IMV within 48 h of death |
| `vitals` | `hospitalization_id`, `recorded_dttm`, `vital_category`, `vital_value` | `vital_category`: `weight_kg`, `height_cm`, `heart_rate` | BMI, death-time diagnostics |
| `crrt_therapy` | `hospitalization_id`, `recorded_dttm` | any row | CRRT within 48 h of death |
| `microbiology_culture` | `hospitalization_id`, `collect_dttm`, `fluid_category`, `method_category`, `organism_category` | `fluid_category`: `blood_buffy`; `method_category`: `culture`; `organism_category`: `no_growth` against any organism | positive blood culture within 48 h of death, Table 3 organism rows |
| `patient_procedures` | `hospitalization_id`, `procedure_code`, `procedure_code_format`, `procedure_billed_dttm` | `procedure_code_format`: `cpt`, `icd10pcs`; the codes in `utils/codes/` | Table 3 neurologic procedures, donor administrative codes |
| `patient_assessments` | `hospitalization_id`, `recorded_dttm`, `assessment_category`, `numerical_value` | `assessment_category`: `gcs_total`, `rass` | Table 2 GCS and RASS |
| `medication_admin_continuous`, `medication_admin_intermittent` | `hospitalization_id`, `admin_dttm`, `med_category` | `med_category`: `dexamethasone`, `methylprednisolone`, `hydrocortisone`, `vasopressin`, `levothyroxine`, `propofol`, `midazolam`, `fentanyl`, `valproate` (`clinical_care.medications` in `donor_criteria.yaml`) | Table 3 medication rows |
| `position` | `hospitalization_id`, `recorded_dttm`, `position_category` | `position_category`: `prone` | Table 3 prone positioning |

Category values are matched ignoring case and surrounding spaces. A category filter that
matches nothing for the whole cohort stops the run and lists the values present, so a
spelling difference cannot silently empty a criterion.

