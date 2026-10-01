## Configuration

Site settings live in `config/config.json`. It is git-ignored and never leaves your site.

1. Copy the template: `cp config/config_template.json config/config.json`
2. Replace every value:

| field | what to put |
|---|---|
| `site_name` | Your site's short name: lowercase letters, digits and underscores, e.g. `ucmc`, `nu`, `rush`. It names your output folder and must match your site's rows in `hospital_crosswalk.yaml`. |
| `tables_path` | Absolute path to the folder holding your CLIF tables (`clif_patient.parquet`, `clif_adt.parquet`, …). |
| `file_type` | `parquet`. No other type is supported. |
| `timezone` | The IANA zone your hospital is in, e.g. `US/Central`. It places admissions, discharges and deaths in the right calendar year. Not `UTC`, even though CLIF stores timestamps in UTC. |
| `project_root` | Absolute path to this repository on your machine. |
| `clif_version` | `2.1` |
| `hospital_ids` | Every `hospital_id` that appears in `clif_adt` for your decedents, written exactly as it appears there. |

The fields starting with `_` in the template are notes and are ignored.

### What the run checks before reading any patient data

`code/01_cohort_and_definitions.py` validates the setup in about a second and stops with a
plain message if something is wrong:

- every field above is present and well-formed;
- every required table in `clif_data_requirements.yaml` exists and has the listed columns
  (a missing optional table is a warning that names the manuscript rows it costs);
- the timestamps used for the 48-hour windows are all timezone-aware or all timezone-naive,
  not a mix;
- `hospital_crosswalk.yaml` has rows for your `site_name` and for every id in `hospital_ids`;
- every criterion in `donor_criteria.yaml` is one the code applies, apart from two keys that
  are listed as not applied and printed on every run (`use_poa`, `sepsis_window_hours`).

### The other files in this folder

| file | what it is | who edits it |
|---|---|---|
| `donor_criteria.yaml` | Every threshold and code range in the three definitions. The code reads all of them from here. | Coordinating centre. Do not change it at a site: results would no longer pool. |
| `clif_data_requirements.yaml` | The CLIF tables and columns the pipeline reads. | Coordinating centre. |
| `hospital_crosswalk.yaml` | Maps each `hospital_id` to its CMS Certification Number for linkage to SRTR. | Coordinating centre. If the setup check says your hospital has no row, ask them to add it. |
| `outlier_config.yaml` | Plausibility ranges, used for weight and height. | Coordinating centre. |

### Several sites on one machine

Keep one file per site as `config/config_<site>.json` and choose it with an environment
variable, or let the runner do it:

```bash
CLIF_DONOR_SITE=ucmc uv run python code/01_cohort_and_definitions.py
uv run python run_pipeline.py --sites ucmc nu
```

`run_pipeline.py` copies `config_<site>.json` over `config.json` for the run and restores it
afterwards. All `config_*.json` files except the template are git-ignored.
