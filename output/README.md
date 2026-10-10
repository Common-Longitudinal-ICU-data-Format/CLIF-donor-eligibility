# What a site returns

Ship the whole of **`<site>_upload_to_box/`**. It contains aggregate counts
only — no patient-level data.

| file | contents |
|---|---|
| `provenance.md` | **start here.** Cohort cascade, CONSORT per definition, every audit stage, and the reconciliation result |
| `ventilation_timing.csv` | per hospital, share of ventilated decedents whose last ventilator record is within 1, 6, 24 and 48 h of death |
| `discharge_vs_death_timing.csv` | per hospital, among decedents with a timed `death_dttm`: how far `discharge_dttm` falls from it. Evidence that discharge can stand in for a date-only or missing death time |
| `vitals_timing.csv` | per hospital, where the last recorded vital falls relative to the time of death, and how often a heart rate above zero is charted more than an hour after it. Evidence for not anchoring on the last vital |
| `organ_criteria.csv` | site totals for each organ-quality criterion: kidney, liver and BMI passes and every reason for failing, over all decedents and over the patients who reached the organ step. The same counts per hospital stay local in `output/intermediate_phi/<site>/organ_criteria_by_hospital.csv` |
| `value_timing.csv` | per measure (labs, GCS, RASS, weight, height): share of decedents whose last value before death is within 1, 6, 24, 48 and 168 h of it, and share missing. For laying sites over one another. Per-hospital rows stay local in `value_timing_by_hospital.csv` |
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
| `run_log.txt` | full stdout, with seconds and peak memory at the end of each section of step 01 |

`output/intermediate_phi/<site>/` holds the patient-level cohort and **never leaves
the site**.
