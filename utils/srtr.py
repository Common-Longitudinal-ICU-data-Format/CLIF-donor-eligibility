"""SRTR donor lookup. Coordinating centre only.

The SRTR extract is under a DUA and never leaves the coordinating centre, so
nothing in the per-site pipeline touches this module. Sites return
`hospital_level_counts.csv` carrying `srtr_ccn_id`; everything here starts from
those CCNs.

Two entry points:

    match_hospitals(...)  which returned hospitals resolve to an SRTR CCN, and
                          how many deceased donors each had in the study window
    donor_characteristics(...)  the Table 1 "Actual donors" column for a set of
                          donor ids

Both are used by `code/coordinating/03_srtr_actual_donors.py` and by the aggregate report, so
there is one implementation rather than two that can drift.
"""
from __future__ import annotations

import re
from pathlib import Path

import polars as pl

# SRTR's own race vocabulary, mapped onto the manuscript's row labels.
SRTR_RACE = {"WHITE": "Race: White", "BLACK": "Race: Black or African American",
             "ASIAN": "Race: Asian", "NATIVE": "Race: American Indian or Alaska Native",
             "PACIFIC": "Race: Native Hawaiian or Other Pacific Islander",
             "MULTI": "Race: Other"}

# Columns pulled from donor_deceased. Kept explicit so a schema change fails
# loudly rather than silently returning a blank column.
DONOR_COLS = ["DONOR_ID", "DON_AGE", "DON_GENDER", "DON_RACE_SRTR",
              "DON_ETHNICITY_SRTR", "DON_ANTI_HCV", "DON_HIST_HYPERTEN",
              "DON_HIST_DIAB", "DON_HGT_CM", "DON_WGT_KG",
              "DON_FINAL_SERUM_CREAT", "DON_TOT_BILI", "DON_SGOT", "DON_SGPT",
              "DON_BUN", "DON_SODIUM"]


def normalise_ccn(v) -> str | None:
    """CCNs arrive as text, as numbers with lost leading zeros, and with stray
    punctuation. Strip to alphanumerics and zero-pad a numeric CCN to six."""
    import pandas as pd
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = re.sub(r"[^0-9A-Za-z]", "", str(v).strip()).upper()
    if not s:
        return None
    return s.zfill(6) if s.isdigit() else s


def load_donor_hospital_links(srtr: Path, year_min: int, year_max: int):
    """One row per (donor, hospital CCN) for donors recovered in the window."""
    import pandas as pd
    import pyreadstat

    d2h = pd.read_sas(srtr / "deceasedtodonhosp2506.sas7bdat", encoding="latin-1")
    d2h["ccn"] = d2h.DON_HOSP_PROV_NUM.map(normalise_ccn)
    rec, _ = pyreadstat.read_sas7bdat(str(srtr / "donor_deceased.sas7bdat"),
                                      usecols=["DONOR_ID", "DON_RECOV_DT"])
    rec["yr"] = pd.to_datetime(rec.DON_RECOV_DT, errors="coerce").dt.year
    j = d2h.merge(rec[["DONOR_ID", "yr"]], on="DONOR_ID", how="left")
    return j[j.yr.between(year_min, year_max)].drop_duplicates(["DONOR_ID", "ccn"])


def match_hospitals(srtr: Path, hosp: pl.DataFrame,
                    year_min: int = 2020, year_max: int = 2025) -> tuple[pl.DataFrame, set]:
    """Attach an SRTR donor count to each returned hospital.

    `hosp` is the pooled `hospital_level_counts.csv` and must carry
    `srtr_ccn_id`. Returns the annotated frame and the set of donor ids at those
    CCNs. A hospital whose CCN does not resolve gets 0, not null, so it is
    visibly unmatched rather than quietly dropped.
    """
    links = load_donor_hospital_links(srtr, year_min, year_max)
    cohort_ccns = {normalise_ccn(c) for c in hosp["srtr_ccn_id"].drop_nulls().to_list()}
    cohort_ccns.discard(None)
    donor_ids = set(links[links.ccn.isin(cohort_ccns)].DONOR_ID)

    by = (links.groupby("ccn").DONOR_ID.nunique()
          .rename("srtr_donors").reset_index())
    out = (hosp.with_columns(
              pl.col("srtr_ccn_id").map_elements(normalise_ccn, return_dtype=pl.Utf8)
              .alias("_ccn"))
           .join(pl.from_pandas(by).rename({"ccn": "_ccn"}), on="_ccn", how="left")
           .with_columns(pl.col("srtr_donors").fill_null(0).cast(pl.Int64),
                         pl.col("_ccn").is_not_null().alias("matched_srtr"))
           .drop("_ccn"))
    return out, donor_ids


def donor_characteristics(srtr: Path, donor_ids) -> pl.DataFrame | None:
    """The "Actual donors" column: characteristics of donors actually recovered
    at the cohort's hospitals.

    Row labels match the manuscript's, so the column slots straight in. A field
    SRTR does not carry (history of CVA, GCS, RASS, length of stay, the culture
    rows) is simply absent rather than reported as zero.
    """
    import pandas as pd
    import pyreadstat

    d, _ = pyreadstat.read_sas7bdat(str(srtr / "donor_deceased.sas7bdat"),
                                    usecols=DONOR_COLS)
    d = d[d.DONOR_ID.isin(set(donor_ids))]
    n = len(d)
    if not n:
        return None

    rows: list[tuple] = [("N patients", "count", float(n), float(n))]

    def prop(label, mask, denom_mask=None):
        dm = pd.Series(True, index=d.index) if denom_mask is None else denom_mask
        den = int(dm.sum())
        if den:
            rows.append((label, "prop", float(int((mask & dm).sum())), float(den)))

    prop("Male sex", d.DON_GENDER.eq("M"), d.DON_GENDER.isin(["M", "F"]))
    # UNOS convention: 1 = no, 2-5 are duration bands (i.e. yes), 998 = unknown.
    yes = [2.0, 3.0, 4.0, 5.0]
    prop("Hypertension", d.DON_HIST_HYPERTEN.isin(yes),
         d.DON_HIST_HYPERTEN.isin([1.0] + yes))
    prop("Diabetes", d.DON_HIST_DIAB.isin(yes), d.DON_HIST_DIAB.isin([1.0] + yes))
    prop("HCV infection", d.DON_ANTI_HCV.eq("P"), d.DON_ANTI_HCV.isin(["P", "N"]))

    rd = d.DON_RACE_SRTR.replace("", None)
    for code, label in SRTR_RACE.items():
        prop(label, rd.eq(code), rd.notna())
    ed = d.DON_ETHNICITY_SRTR.replace("", None)
    prop("Ethnicity: Hispanic", ed.eq("LATINO"), ed.notna())
    prop("Ethnicity: Non-Hispanic", ed.eq("NLATIN"), ed.notna())

    for label, col in [("Age at death", "DON_AGE"), ("Weight [kg]", "DON_WGT_KG"),
                       ("Height [cm]", "DON_HGT_CM"),
                       ("Terminal creatinine [mg/dL]", "DON_FINAL_SERUM_CREAT"),
                       ("Terminal total bilirubin [mg/dL]", "DON_TOT_BILI"),
                       ("Terminal AST [U/L]", "DON_SGOT"),
                       ("Terminal ALT [U/L]", "DON_SGPT"),
                       # DON_SODIUM is documented as the last serum sodium prior
                       # to procurement, which is the closest SRTR analogue to
                       # our terminal value. DON_BUN carries no such note.
                       ("Terminal BUN [mg/dL]", "DON_BUN"),
                       ("Terminal sodium [mmol/L]", "DON_SODIUM")]:
        v = pd.to_numeric(d[col], errors="coerce").dropna()
        if len(v):
            rows.append((label, "median", float(v.median()), float(v.quantile(.25)),
                         float(v.quantile(.75)), float(len(v))))

    bmi = (pd.to_numeric(d.DON_WGT_KG, errors="coerce")
           / (pd.to_numeric(d.DON_HGT_CM, errors="coerce") / 100) ** 2)
    bmi = bmi.replace([float("inf"), float("-inf")], None).dropna()
    bmi = bmi[(bmi > 5) & (bmi < 120)]        # guard against unit errors
    if len(bmi):
        rows.append(("BMI", "median", float(bmi.median()), float(bmi.quantile(.25)),
                     float(bmi.quantile(.75)), float(len(bmi))))

    out = []
    for r in rows:
        if r[1] == "median":
            out.append({"variable": r[0], "stat": "median",
                        "value": f"{r[2]:.1f} [{r[3]:.1f}–{r[4]:.1f}]",
                        "n": r[2], "denom": r[5]})
        elif r[1] == "count":
            out.append({"variable": r[0], "stat": "count",
                        "value": f"{int(r[2]):,}", "n": r[2], "denom": r[3]})
        else:
            out.append({"variable": r[0], "stat": "prop",
                        "value": f"{int(r[2]):,} ({100 * r[2] / r[3]:.1f}%)",
                        "n": r[2], "denom": r[3]})
    return pl.DataFrame(out)
