## Code

Three scripts, run in order. Each reads `config/config.json` (see
[`../config/README.md`](../config/README.md)) and writes to two places:

- `<site>_upload_to_box/` at the repository root: aggregate results, the folder you return;
- `output/intermediate_phi/<site>/`: patient-level working files, which never leave your site.

| script | what it does | main outputs |
|---|---|---|
| `01_cohort_and_definitions.py` | Setup checks, then the cohort of in-hospital deaths and the three definitions (CLIF-donor, CALC, Ventilated Patient). | `strobe_counts.csv`, `definition_counts.csv`, `decedents_by_location.csv`, `data_quality_flags.csv`, `srtr_ref/`, `run_log.txt` |
| `02_build_tables.py` | Clinical-care variables from the optional tables, the manuscript tables, and a CONSORT per definition. | `table2_characteristics.csv`, `table3_clinical_care.csv`, `tableS2_missingness.csv`, `table_stats_raw.csv`, `consort_counts.csv`, `hospital_level_counts.csv`, `exclusion_codes_by_step.csv` |
| `03_diagnostics.py` | Missingness, data-element coverage and definition overlap, then a reconciliation of the counts above and `provenance.md`. | `missingness_*.csv`, `element_coverage.csv`, `definition_overlap_upset.*`, `provenance.md` |

```bash
uv sync
uv run python code/01_cohort_and_definitions.py
uv run python code/02_build_tables.py
uv run python code/03_diagnostics.py
```

Step 01 stops within a second if the config, the tables or the criteria are not usable, before
any patient data is read. Step 03 fails the run if the counts it reconciles disagree.

Every clinical threshold is read from [`../config/donor_criteria.yaml`](../config/donor_criteria.yaml).
Flag names such as `age_75_less` and `imv_48hr_expire` keep the default value in their name; the
value applied is the one in the YAML.

### `coordinating/`

Pooling across sites and linkage to SRTR. Run at the coordinating centre only, on the bundles
sites return. The SRTR extract is under a data use agreement and lives outside the repository;
point `CLIF_SRTR_DIR` at it.

### Tests

```bash
uv run pytest
```

The tests need no patient data. They cover the criteria loader, the ICD-10 range matching, the
timestamp check and the provenance document.
