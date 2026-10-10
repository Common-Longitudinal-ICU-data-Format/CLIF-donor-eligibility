"""SQL for criteria of the form "within N hours before death".

Kept as functions, apart from the pipeline script, so each can be run against a
small fixture. See tests/test_windows.py.
"""
from __future__ import annotations


def imv_before_death_sql(source: str, cohort: str, hours: float) -> str:
    """One row per patient with at least one IMV record in the `hours` before death.

    `source`  table expression with hospitalization_id, recorded_dttm, device_category
    `cohort`  table with hospitalization_id, patient_id, encounter_block, final_death_dttm

    ANY IMV row in the window counts. The test used to take each patient's
    LATEST row and ask whether that one fell in the window, so a patient
    ventilated up to death whose ventilation then continued for more than a day
    (donor management after a declaration of death) was counted as not
    ventilated. The window is applied before the ranking. Rows after death do
    not count. The row returned is the latest one inside the window.
    """
    return f"""
WITH imv_data AS (
    SELECT hospitalization_id, recorded_dttm, device_category
    FROM {source}
    WHERE LOWER(TRIM(device_category)) = 'imv'
        AND hospitalization_id IN (SELECT hospitalization_id FROM {cohort})
),
imv_in_window AS (
    SELECT
        i.hospitalization_id,
        i.recorded_dttm,
        f.patient_id,
        f.encounter_block,
        f.final_death_dttm,
        EXTRACT(EPOCH FROM (f.final_death_dttm - i.recorded_dttm)) / 3600 AS hr_2death_last_imv
    FROM imv_data i
    INNER JOIN {cohort} f ON i.hospitalization_id = f.hospitalization_id
    WHERE EXTRACT(EPOCH FROM (f.final_death_dttm - i.recorded_dttm)) / 3600 BETWEEN 0 AND {hours}
),
latest_imv_per_patient AS (
    SELECT *,
        ROW_NUMBER() OVER (
            PARTITION BY patient_id
            ORDER BY recorded_dttm DESC, hospitalization_id ASC
        ) AS rn
    FROM imv_in_window
)
SELECT patient_id, hospitalization_id, encounter_block, final_death_dttm, recorded_dttm,
       hr_2death_last_imv
FROM latest_imv_per_patient
WHERE rn = 1
"""
