# Notes for the manuscript methods

Rules the code applies that the methods or supplement must state. Each names the file that
applies it and the shipped file that is the evidence for it.

## Time of death

Every "within N hours of death" window is measured back from one time per patient:

| `patient.death_dttm` | time of death used |
|---|---|
| has a time of day | `death_dttm` |
| is only a date (00:00:00 local) | `discharge_dttm` of the death hospitalization, kept inside that date |
| is missing | `discharge_dttm` |

A date-only death keeps its date because the recorded date is the death: for a patient managed
as a donor after death is declared, discharge is recorded at organ recovery, a day or more
later. A death recorded more than 24 h after discharge is disregarded and discharge is used.
The rule is in `utils/death_time.py`, the per-hospital comparison behind it is in
`code/exploration/death_dttm_definition.ipynb`, and `strobe_counts.csv` reports how many
patients fall under each case. `discharge_vs_death_timing.csv` and `vitals_timing.csv` are the
evidence: how far discharge falls from a timed death, and why the last vital is not the anchor.

"IMV within 48 h of death" means at least one ventilator record in the 48 hours before that
time. `ventilation_timing.csv` shows how close the last record falls.

## Which stays count

Every criterion, lab, culture and diagnosis is taken from the decedent's whole encounter: the
death hospitalization plus any hospitalization linked to it by clifpy's `stitch_encounters`
(admitted within 6 hours of the previous discharge; `study.encounter_stitch_hours`). A move
between units can open a new hospitalization, for example from an ICU bed to an inpatient
hospice unit, and the death hospitalization alone would miss what was recorded before the
move. `strobe_counts.csv` reports how many decedents have a linked earlier stay. Lengths of
stay are measured over the whole encounter, hospital stay from first admission to last
discharge, in elapsed days.

## Last recorded values

Every measured value is the **last one recorded before the time of death**, anywhere in the
encounter, with no time limit: creatinine, total bilirubin, AST, ALT, BUN, sodium, weight,
height, GCS total and RASS. Only plausible values count, using the ranges in
`config/outlier_config.yaml`, so an implausible result at the latest timestamp yields to the
previous plausible one. A missing value is ineligible for the organ criteria. `value_timing.csv`
reports, per measure, how long before death the chosen value was taken.

The criteria with a clock on them use a window: IMV, CRRT and blood cultures within 48 h
before death (`config/donor_criteria.yaml`). Any record in the window counts; records after
death do not. A blood culture is positive when any organism grew, contaminants included; a
decedent with no culture passes.

## Age

Age at death is computed from `birth_date`; where that is missing, `age_at_admission` stands
in (it understates age by the length of stay, so at the limit it can only over-include). An
age outside the plausible range from either source is no age, and a decedent with no age
leaves the cohort before it is counted (`strobe_counts.csv`, rows `2h` and `2i`). The limit,
75, is inclusive, as is BMI 50.

## Hospital of death

A decedent belongs to the hospital on their last ADT record, which is how SRTR attributes a
donor. Decedents at a hospital the study excludes (`config/hospital_crosswalk.yaml`) leave the
cohort before it is counted, so every output is over the same hospitals. Hospitals sharing a
CMS Certification Number are one hospital in every output.

## Diagnoses

A diagnosis counts at any position, present on admission or not, on any stay of the encounter
(`calc.diagnosis_position: any`). The three CALC readings, any position, principal, and
principal present on admission, are all reported in `strobe_counts.csv`. The cancer exclusion
is described in `guides/contraindications.md`.
