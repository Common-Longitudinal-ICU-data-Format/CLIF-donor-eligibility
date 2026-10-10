## Configuration

Site settings live in `config/config.json`. It is git-ignored and never leaves your site.

1. Copy the template: `cp config/config_template.json config/config.json`
2. Replace every value:

| field | what to put |
|---|---|
| `site_name` | Your site's short name: lowercase letters, digits and underscores, e.g. `ucmc`, `nu`, `rush`. It names your output folder and must match your site's rows in `hospital_crosswalk.yaml`. |
| `tables_path` | Absolute path to the folder holding your CLIF tables (`clif_patient.parquet`, `clif_adt.parquet`, …). |
| `timezone` | The timezone your hospital is in: `US/Eastern`, `US/Central`, `US/Mountain`, `US/Arizona`, `US/Pacific`, `US/Alaska` or `US/Hawaii`. It places admissions, discharges and deaths in the right calendar year, and it decides which death times count as date-only (exactly midnight at your hospital). Not `UTC`, even though CLIF stores timestamps in UTC. |
| `project_root` | Absolute path to this repository on your machine. |
| `hospital_ids` | Every `hospital_id` in `clif_adt` where your decedents died, written exactly as in your data. Check them against your site's rows in `hospital_crosswalk.yaml`; if one is missing or spelled differently, tell the coordinating centre so the crosswalk can be updated. |

The fields starting with `_` in the template are notes and are ignored.

### What the run checks before reading any patient data

`code/01_cohort_and_definitions.py` validates the setup in about a second and stops with a
plain message if something is wrong:

- every field above is filled in;
- every required table in `clif_data_requirements.yaml` exists and has the listed columns
  (a missing optional table is a warning that names the manuscript rows it costs);
- the timestamps used for the 48-hour windows are all timezone-aware or all timezone-naive,
  not a mix;
- `hospital_crosswalk.yaml` has rows for your `site_name` and for every id in `hospital_ids`;
- every row of `utils/codes/icd10_contraindications.csv` is well-formed (see
  [`../guides/contraindications.md`](../guides/contraindications.md)).

### The other files in this folder

| file | what it is | who edits it |
|---|---|---|
| `donor_criteria.yaml` | Every threshold and code range in the three definitions. The code reads all of them from here. | Coordinating centre. Do not change it at a site: results would no longer pool. |
| `clif_data_requirements.yaml` | The CLIF tables and columns the pipeline reads. | Coordinating centre. |
| `hospital_crosswalk.yaml` | Maps each `hospital_id` to its CMS Certification Number for linkage to SRTR. | Coordinating centre. If the setup check says your hospital has no row, ask them to add it. |
| `outlier_config.yaml` | Plausibility ranges for weight, height, the labs, GCS, RASS and age; a value outside its range is read as missing. | Coordinating centre. |
