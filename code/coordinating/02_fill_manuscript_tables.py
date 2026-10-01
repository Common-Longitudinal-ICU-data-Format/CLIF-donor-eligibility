#!/usr/bin/env python
"""Fill the v6 manuscript tables with pooled three-site results.

Writes a new .docx. **No prose, no row label and no table structure is
changed** — only empty result cells are populated. A cell we cannot populate is
left empty rather than guessed at, and every empty cell is listed in
docs/manuscript_build/MANUSCRIPT_GAPS.md.

    python code/coordinating/02_fill_manuscript_tables.py \
        --docx "manuscript/CLIF-donor manuscript_6.docx" \
        --sites-dir manuscript_results
"""
from __future__ import annotations

import argparse
import glob
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import polars as pl

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
REPO = Path(__file__).resolve().parents[2]

# Table 1 and Table 2 row label -> the variable that fills it. The label is
# matched on a normalised prefix so trailing footnote markers do not matter.
T1 = {
    "Decedents":                       ("count", "N patients"),
    "Age, years":                      ("median", "Age at death"),
    "Male sex":                        ("count", "Male sex"),
    "Height":                          ("median", "Height [cm]"),
    "Weight":                          ("median", "Weight [kg]"),
    "Body mass index":                 ("median", "BMI"),
    "Hypertension":                    ("count", "Hypertension"),
    "Diabetes":                        ("count", "Diabetes"),
    "Hx CVA":                          ("count", "History of CVA"),
    "Ischemic heart disease":          ("count", "Ischemic heart disease"),
    "Cerebrovascular disease":         ("count", "Cerebrovascular disease"),
    "External causes":                 ("count", "External causes"),
    "Z52.9":                           ("count", "Z52.9 donor of organs"),
    "G93.82":                          ("count", "Brain death (G93.82)"),
    "Cancer":                          ("count", "Cancer"),
    "Positive blood culture":          ("count", "Positive blood culture <=48h"),
    "Fungemia":                        ("count", "Fungemia"),
    "At least one contraindication":   ("count", ">=1 relative contraindication"),
    "Terminal RASS":                   ("median", "Terminal RASS"),
    "Terminal GCS":                    ("median", "Terminal GCS"),
    "Terminal creatinine":             ("median", "Terminal creatinine [mg/dL]"),
    "BUN":                             ("median", "Terminal BUN [mg/dL]"),
    "Sodium":                          ("median", "Terminal sodium [mmol/L]"),
    "Total bilirubin":                 ("median", "Terminal total bilirubin [mg/dL]"),
    "AST":                             ("median", "Terminal AST [U/L]"),
    "ALT":                             ("median", "Terminal ALT [U/L]"),
}
T2 = {
    "N patients":                      ("count", "N patients"),
    "Hospital LOS":                    ("median", "Hospital LOS [days]"),
    "Admitted to an ICU":              ("count", "Admitted to an ICU"),
    "Last ICU LOS":                    ("median", "First ICU LOS [days]"),
    "Sedatives":                       ("count", "med_sedative_cont"),
    "Analgesics":                      ("count", "med_analgesic_cont"),
    "Invasive mechanical ventilation": ("count", "Invasive mechanical ventilation"),
    "Prone positioning":               ("count", "prone_position"),
    "CRRT":                            ("count", "on_crrt_48h_before_death"),
    "Decompressive craniectomy":       ("count", "proc_decompressive_craniectomy"),
    "Craniotomy for hematoma":         ("count", "proc_craniotomy_hematoma"),
    "Endoventricular drain":           ("count", "proc_icp_evd"),
    "Cerebral angiography":            ("count", "proc_cerebral_angiography"),
    "Endovascular stroke":             ("count", "proc_endovascular_stroke"),
    "Continuous EEG":                  ("count", "proc_continuous_eeg"),
    "Antiepileptic medication":        ("count", "med_antiepileptic_intermit"),
    "Corticosteroids":                 ("count", "med_corticosteroid_bolus_intermit"),
    "Levothyroxine":                   ("count", "med_levothyroxine_cont"),
    "Vasopressin":                     ("count", "med_vasopressin_cont"),
    "Admitted to an ICU":              ("count", "Admitted to an ICU"),
    "CMV":                             ("count", "micro_cmv"),
    "EBV":                             ("count", "micro_ebv"),
    "Tuberculosis":                    ("count", "micro_tuberculosis"),
    "Aspergillus":                     ("count", "micro_aspergillus"),
    "Physiological support for harvest": ("count", "cpt_01990"),
    "Donor pneumonectomy":             ("count", "cpt_32850"),
    "Donor cardiectomy-pneumonectomy": ("count", "cpt_33930"),
    "Donor cardiectomy,":              ("count", "cpt_33940"),
    "Donor enterectomy":               ("count", "cpt_44132"),
    "Donor hepatectomy":               ("count", "cpt_47133"),
    "Donor pancreatectomy":            ("count", "cpt_48550"),
    "Donor nephrectomy":               ("count", "cpt_50300"),
}


def ptext(p) -> str:
    return "".join(t.text or "" for t in p.iter(W + "t")).strip()


def cell_text(c) -> str:
    return " ".join(ptext(p) for p in c.iter(W + "p") if ptext(p)).strip()


def set_cell(c, value: str) -> None:
    """Write into the first paragraph of a cell, preserving its run formatting."""
    paras = c.findall(W + "p")
    if not paras:
        return
    p = paras[0]
    runs = p.findall(W + "r")
    if runs:
        for extra in runs[1:]:
            p.remove(extra)
        ts = runs[0].findall(W + "t")
        if ts:
            for extra in ts[1:]:
                runs[0].remove(extra)
            ts[0].text = value
            ts[0].set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
            return
    r = ET.SubElement(p, W + "r")
    t = ET.SubElement(r, W + "t")
    t.text = value
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")


def load_stats(sites_dir: Path) -> pl.DataFrame:
    fr = [pl.read_csv(f, infer_schema_length=0)
          for f in sorted(glob.glob(str(sites_dir / "*/table_stats_raw.csv")))]
    if not fr:
        raise SystemExit(f"no table_stats_raw.csv under {sites_dir}")
    d = pl.concat(fr, how="diagonal")
    for c in ("n", "denom", "median", "q25", "q75", "n_nonnull"):
        if c in d.columns:
            d = d.with_columns(pl.col(c).cast(pl.Float64, strict=False))
    return d


COLMAP = {
    "Age at death": "age_at_death", "Weight [kg]": "last_weight_kg",
    "Height [cm]": "last_height_cm", "BMI": "bmi",
    "Terminal creatinine [mg/dL]": "creatinine_value",
    "Terminal total bilirubin [mg/dL]": "bilirubin_total_value",
    "Terminal AST [U/L]": "ast_value", "Terminal ALT [U/L]": "alt_value",
    "Terminal BUN [mg/dL]": "bun_value", "Terminal sodium [mmol/L]": "sodium_value",
    "Terminal GCS": "gcs_total_value", "Terminal RASS": "rass_value",
    "Hospital LOS [days]": "hospital_length_of_stay_days",
    "First ICU LOS [days]": "first_icu_los_days",
}
DEFCOL = {"CLIF-donor": "clif_eligible_donors", "CALC": "calc_flag"}


def load_srtr_column(sites_dir: Path) -> dict[str, str]:
    """The Actual donors column, produced by code/coordinating/03_srtr_actual_donors.py.
    Absent is fine — the column is then left blank rather than guessed at."""
    f = sites_dir / "srtr_actual_donors.csv"
    if not f.exists():
        return {}
    d = pl.read_csv(f, infer_schema_length=0)
    return dict(zip(d["variable"], d["value"]))


def load_patient_level() -> pl.DataFrame | None:
    """Union of the per-site cohorts, for exact median pooling. Coordinating
    centre only — a site that only returned aggregates will not have this, and
    medians are then left blank rather than approximated by a range."""
    need = list(COLMAP.values()) + list(DEFCOL.values())
    frames = []
    for p in sorted(glob.glob(str(REPO / "output/intermediate_phi/*/cohort_with_variables.parquet"))):
        df = pl.read_parquet(p)
        frames.append(df.select([c for c in need if c in df.columns]))
    return pl.concat(frames, how="diagonal") if frames else None


def pooled(stats: pl.DataFrame, plevel: pl.DataFrame | None,
           variable: str, definition: str, kind: str) -> str | None:
    """Counts pool additively. Medians are recomputed from the patient-level
    union, never averaged across sites."""
    if kind == "count":
        g = stats.filter((pl.col("variable") == variable)
                         & (pl.col("definition") == definition)
                         & (pl.col("stat") == "count"))
        if not g.height:
            return None
        n, dn = g["n"].sum(), g["denom"].sum()
        if not dn:
            return None
        if variable == "N patients":
            return f"{int(n):,}"
        return f"{int(n):,} ({100 * n / dn:.1f}%)"

    col, dcol = COLMAP.get(variable), DEFCOL.get(definition)
    if plevel is None or col is None or dcol is None:
        return None
    if col not in plevel.columns or dcol not in plevel.columns:
        return None
    s = plevel.filter(pl.col(dcol).fill_null(False))[col].drop_nulls()
    if not s.len():
        return None
    return f"{s.median():.1f} [{s.quantile(.25):.1f}\u2013{s.quantile(.75):.1f}]"


def norm_label(x: str) -> str:
    return " ".join(x.replace("\u2013", "-").split()).lower()


def fill_table(tbl, mapping: dict, stats, plevel, srtr: dict,
               gaps: list, tname: str) -> int:
    """Fill a table's value cells. Column 0 is the row label; columns 1..n are
    CLIF-donor, CALC, Actual donors in v6's order."""
    filled = 0
    rows = list(tbl.iter(W + "tr"))
    header = [cell_text(c) for c in rows[0].findall(W + "tc")]
    defs = []
    for h in header[1:]:
        hl = norm_label(h)
        defs.append("CLIF-donor" if "clif-donor" in hl
                    else "CALC" if hl.startswith("calc")
                    else "SRTR" if "donor" in hl else None)
    for r in rows[1:]:
        cells = r.findall(W + "tc")
        if len(cells) < 2:
            continue
        label = norm_label(cell_text(cells[0]))
        if not label:
            continue
        hit = next((v for k, v in mapping.items() if label.startswith(norm_label(k))), None)
        if hit is None:
            if any(not cell_text(c) for c in cells[1:]):
                gaps.append((tname, cell_text(cells[0]), "no mapping"))
            continue
        kind, var = hit
        for c, dname in zip(cells[1:], defs):
            if cell_text(c):          # never overwrite an existing value
                continue
            if dname == "SRTR":
                val = srtr.get(var)
                if val is None:
                    gaps.append((tname, cell_text(cells[0]),
                                 f"Actual donors: SRTR carries no '{var}'"))
                    continue
                set_cell(c, val)
                filled += 1
                continue
            if dname is None:
                gaps.append((tname, cell_text(cells[0]), "unrecognised column"))
                continue
            val = pooled(stats, plevel, var, dname, kind)
            if val is None:
                gaps.append((tname, cell_text(cells[0]), f"{dname}: no value for '{var}'"))
                continue
            set_cell(c, val)
            filled += 1
    return filled


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docx", required=True)
    ap.add_argument("--sites-dir", default=str(REPO / "manuscript_results"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    src, sites = Path(a.docx), Path(a.sites_dir)
    out = Path(a.out) if a.out else src.with_name(src.stem + "_with_results.docx")
    stats = load_stats(sites)
    plevel = load_patient_level()
    srtr = load_srtr_column(sites)
    print(f"SRTR column: {len(srtr)} rows"
          + ("" if srtr else "  (run code/coordinating/03_srtr_actual_donors.py first)"))
    print(f"stats rows {stats.height:,} | patient-level "
          f"{'yes, ' + format(plevel.height, ',') + ' rows' if plevel is not None else 'no'}")

    zin = zipfile.ZipFile(src)
    doc = ET.fromstring(zin.read("word/document.xml"))
    for k, v in {"w": W[1:-1]}.items():
        ET.register_namespace(k, v)
    body = doc.find(W + "body")
    tables = [b for b in body if b.tag == W + "tbl"]

    gaps: list = []
    total = 0
    for tbl in tables:
        rows = list(tbl.iter(W + "tr"))
        if not rows:
            continue
        cells0 = rows[0].findall(W + "tc")
        hdr = norm_label(" ".join(cell_text(c) for c in cells0))
        # Only the two main result tables. They are the ones whose stub column
        # reads "Medical eligibility definition ..." and which carry an Actual
        # donors column; the supplemental missingness tables share the words
        # "CLIF-donor" and "CALC" in their headers and must not be touched.
        if not hdr.startswith("medical eligibility definition") or len(cells0) < 4:
            continue
        labels = {norm_label(cell_text(r.findall(W + "tc")[0]))
                  for r in rows[1:] if r.findall(W + "tc")}
        is_t1 = any(l.startswith("terminal creatinine") for l in labels)
        mapping, tname = (T1, "Table 1") if is_t1 else (T2, "Table 2")
        n = fill_table(tbl, mapping, stats, plevel, srtr, gaps, tname)
        total += n
        print(f"  {tname}: filled {n} cells")

    xml = ET.tostring(doc, encoding="UTF-8", xml_declaration=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zo:
        for item in zin.infolist():
            zo.writestr(item, xml if item.filename == "word/document.xml"
                        else zin.read(item.filename))
    print(f"\nwrote {out}  ({total} cells filled)")

    gp = REPO / "docs/manuscript_build/manuscript_unfilled_cells.csv"
    gp.parent.mkdir(parents=True, exist_ok=True)
    seen, rows = set(), []
    for tname, label, why in gaps:
        if (tname, label, why) in seen:
            continue
        seen.add((tname, label, why))
        rows.append({"table": tname, "row": label, "reason": why})
    pl.DataFrame(rows).write_csv(gp)
    print(f"wrote {gp}  ({len(rows)} unfilled)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
