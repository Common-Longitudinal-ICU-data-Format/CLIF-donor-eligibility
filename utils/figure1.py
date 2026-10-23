"""Figure 1: cohort selection, and the CLIF-donor organ assessment.

Drawn once per site by code/03_diagnostics.py (figure1_cohort.png in the bundle,
with the counts behind it in figure1_counts.csv) and pooled over sites by
code/coordinating/04_figure1_cohort.py from those files. One drawing, two callers.

Panel A follows Will Parker's request of 2026-10-21. CALC is applied in sequence:
a qualifying cause of death, then age; location is the cohort entry, because an
ED death without admission is not an inpatient death and never enters the
cohort. CLIF-donor is applied in parallel: each of its six criteria is counted
among all cohort decedents, BMI as a criterion of its own, and eligibility is
the decedents meeting all six, split by the organ criteria they met. Exclusions
before analysis sit in one box, with excluded hospitals by reason from
hospitals.csv. Panel B is the organ assessment among the CLIF-donor eligible:
which organ qualified each decedent, and for those qualifying on one organ, why
the other did not; from the clif_donor_eligible row of organ_criteria.csv.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle       # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator       # noqa: E402
import polars as pl                                            # noqa: E402
import yaml                                                    # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CRIT = yaml.safe_load((REPO / "config/donor_criteria.yaml").read_text())
STUDY, DONOR, CALC_CFG = CRIT["study"], CRIT["clif_donor"], CRIT["calc"]
WINDOW = f"{STUDY['window_start_year']}–{STUDY['window_end_year']}"
CALC_POSITIONS = {"any": "calc_cause_any", "primary": "calc_cause_primary",
                  "primary_poa": "calc_cause_primary_poa"}
POSITION = str(CALC_CFG["diagnosis_position"]).lower()
POSITION_WORD = {"any": "any", "primary": "principal",
                 "primary_poa": "principal, present-on-admission"}[POSITION]
CODES = ", ".join(f"{a}–{b}" for a, b in CALC_CFG["cause_icd10_prefixes"].values())

# Okabe-Ito; the pair passes the dataviz palette validator (CVD ΔE 11.4).
INK, MUTED, RULE = "#222222", "#666666", "#D9D9D9"
CLIF, CLIF_TINT = "#009E73", "#DCF1EA"
CALC, CALC_TINT = "#E69F00", "#FCEFD6"
BOTH, BOTH_TINT = "#7A7A7A", "#EFEFEF"
NEITHER, REASON = "#BDBDBD", "#8C8C8C"

# Exclusions between the decedents in the study window and the analytic cohort,
# in the order step 01 applies them.
UPSTREAM = [("no_adt", "No ADT record"),
            ("ed_only", "Died in the ED without admission"),
            ("other_location", "Died outside the study units"),
            ("excluded_hospital", "Died at an excluded hospital"),
            ("no_age", "Age unknown")]
_AGE = DONOR["age_at_death_max"]
CRITERIA = [("no_cancer", "No contraindicating cancer diagnosis"),
            ("no_culture", f"No positive blood culture within "
                           f"{DONOR['contraindications']['positive_blood_culture_hours']} h"),
            ("age_le75", f"Age ≤ {_AGE} at death"),
            ("imv", f"Invasive mechanical ventilation within {DONOR['imv_hours_before_death']} h of death"),
            ("bmi", f"BMI ≤ {DONOR['bmi_max']} kg/m²"),
            ("organ", "Kidney or liver criteria met")]
# figure1_counts.csv metrics, in the order the figure reads them.
FLOW_KEYS = ["in_window", *(k for k, _ in UPSTREAM), "analytic", "hospitals", "calc_cause", "calc",
             *(k for k, _ in CRITERIA), "all_six", "clif_donor", "kidney_only", "liver_only", "both"]
# organ_criteria.csv: the population panel B draws, and the columns it reads.
ORGAN_POPULATION = "clif_donor_eligible"
ORGAN = ["n_patients", "kidney_pass", "liver_pass", "kidney_and_liver_pass", "bmi_pass", "organ_pass",
         "creatinine_missing", "creatinine_at_or_above_limit", "on_crrt",
         "liver_lab_missing", "bilirubin_at_or_above_limit", "ast_at_or_above_limit",
         "alt_at_or_above_limit", "bmi_missing", "bmi_above_limit"]
KIDNEY_WHY = ["on_crrt", "creatinine_at_or_above_limit", "creatinine_missing"]
LIVER_WHY = ["bilirubin_at_or_above_limit", "ast_at_or_above_limit", "alt_at_or_above_limit",
             "liver_lab_missing"]
BMI_WHY = ["bmi_missing", "bmi_above_limit"]
REASON_WORDS = {"long_term_care": "long-term acute care", "inpatient_rehabilitation": "inpatient rehabilitation",
                "not_a_system_facility": "not a system hospital", "no_cms_number": "no CMS number",
                "not_in_srtr": "not in SRTR"}


# ── counts ──────────────────────────────────────────────────────────────────

def site_flow(cohort: pl.DataFrame, strobe: dict[str, int], dc: dict) -> dict[str, int]:
    """Panel A's numbers for one site, from the patient-level cohort, the strobe
    rows above the cohort, and definition_counts.csv (for the hospital count)."""
    c = lambda name: pl.col(name).fill_null(False)                    # noqa: E731
    n = lambda e: int(cohort.select(e.cast(pl.Int64).sum()).item())   # noqa: E731
    crit = {"no_cancer": ~c("icd10_contraindication"), "no_culture": c("no_positive_culture_48hrs"),
            "age_le75": c("age_75_less"), "imv": c("ventilated_patient_no_age_limit"),
            "bmi": c("bmi_eligible"), "organ": c("kidney_eligible") | c("liver_eligible")}
    elig, kidney, liver = c("clif_eligible_donors"), c("kidney_eligible"), c("liver_eligible")
    ed_only = int(strobe["2b_ed_only_deaths"])
    f = {"in_window": int(strobe["0c_decedents_in_window"]),
         "no_adt": int(strobe["1b_decedents_without_adt"]),
         "ed_only": ed_only,
         "other_location": int(strobe["2b_deaths_excluded_by_location"]) - ed_only,
         "excluded_hospital": int(strobe["2g_deaths_at_excluded_hospitals"]),
         "no_age": int(strobe["2h_deaths_without_age"]),
         "analytic": cohort.height, "hospitals": int(dc["n_analytic_hospitals"]),
         "calc_cause": n(c(CALC_POSITIONS[POSITION])), "calc": n(c("calc_flag"))}
    f.update({k: n(e) for k, e in crit.items()})
    f["all_six"] = n(pl.all_horizontal(*crit.values()))
    f["clif_donor"] = n(elig)
    f["kidney_only"] = n(elig & kidney & ~liver)
    f["liver_only"] = n(elig & ~kidney & liver)
    f["both"] = n(elig & kidney & liver)
    return f


def flow_problems(f: dict[str, int]) -> list[str]:
    """Identities panel A's counts must satisfy before it is drawn."""
    p = []
    up = sum(f[k] for k, _ in UPSTREAM)
    if f["in_window"] - up != f["analytic"]:
        p.append(f"decedents in the window ({f['in_window']:,}) less the exclusions ({up:,}) "
                 f"are not the analytic cohort ({f['analytic']:,})")
    if f["all_six"] != f["clif_donor"]:
        p.append(f"the six criteria together ({f['all_six']:,}) are not CLIF-donor ({f['clif_donor']:,})")
    if f["kidney_only"] + f["liver_only"] + f["both"] != f["clif_donor"]:
        p.append("kidney only, liver only and both do not add to CLIF-donor")
    if not f["calc"] <= f["calc_cause"] <= f["analytic"]:
        p.append(f"CALC ({f['calc']:,}) exceeds its cause step ({f['calc_cause']:,}) or the cohort")
    if f["clif_donor"] > min(f[k] for k, _ in CRITERIA):
        p.append("CLIF-donor exceeds one of its own criteria")
    return p


def organ_problems(o: dict[str, int]) -> list[str]:
    """Identities organ_criteria.csv must satisfy for panel B to be drawn from it."""
    p = []
    n, k, l, kl = o["n_patients"], o["kidney_pass"], o["liver_pass"], o["kidney_and_liver_pass"]
    if min(kl, k - kl, l - kl, n - k - l + kl) < 0:
        p.append(f"kidney and liver passes do not partition: n {n}, kidney {k}, liver {l}, both {kl}")
    if o["bmi_pass"] + o["bmi_missing"] + o["bmi_above_limit"] != n:
        p.append("BMI met, missing and above the limit do not add to the organ step")
    for label, failed, why in (("kidney", n - k, KIDNEY_WHY), ("liver", n - l, LIVER_WHY)):
        reasons = [o[w] for w in why]
        if not max(reasons) <= failed <= sum(reasons):
            p.append(f"{label} failures ({failed}) are not covered by their reasons {reasons}")
    if o["organ_pass"] > min(k + l - kl, o["bmi_pass"]):
        p.append("organ passes exceed kidney-or-liver or BMI passes")
    return p


def all_problems(f: dict[str, int], o: dict[str, int]) -> list[str]:
    """Both panels' identities, and the ties between them."""
    p = flow_problems(f) + organ_problems(o)
    kl = o["kidney_and_liver_pass"]
    if o["n_patients"] != f["clif_donor"]:
        p.append(f"organ_criteria counts {o['n_patients']:,} eligible, the flow {f['clif_donor']:,}")
    if (kl, o["kidney_pass"] - kl, o["liver_pass"] - kl) != (f["both"], f["kidney_only"], f["liver_only"]):
        p.append("the organ split differs between organ_criteria and the flow")
    if o["organ_pass"] != o["n_patients"] or o["bmi_pass"] != o["n_patients"]:
        p.append("an eligible decedent did not pass the organ assessment or BMI")
    return p


def excluded_text(hospitals: pl.DataFrame | None) -> str:
    """'long-term acute care 60; inpatient rehabilitation 35' from hospitals.csv rows."""
    if hospitals is None or not hospitals.height:
        return ""
    ex = hospitals.filter(pl.col("status") == "excluded")
    if not ex.height:
        return ""
    by = (ex.group_by("reason").agg(pl.col("n_decedents").sum())
            .sort("n_decedents", descending=True))
    return "; ".join(f"{REASON_WORDS.get(r, r.replace('_', ' '))} {int(n):,}" for r, n in by.iter_rows())


# ── drawing ─────────────────────────────────────────────────────────────────

W, HA, HB = 7.2, 6.6, 2.6                # inches: width, panel A, panel B
H = HA + HB
LINE, PAD = 0.1215, 0.07                 # 7 pt text at 1.25 leading; box padding


def _style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.edgecolor": INK, "xtick.color": INK, "ytick.color": INK,
        "pdf.fonttype": 42, "svg.fonttype": "none"})


def _fig_axes(fig, x, y, w, h):
    """Axes at (x, y) inches from the bottom-left of the figure, w by h inches."""
    return fig.add_axes([x / W, y / H, w / W, h / H])


def _box(ax, cx, cy, w, lines, edge, fill, dashed=False, lw=0.7):
    """A rounded box centred at (cx, cy) inches from the top-left of panel A.

    lines: (text, bold) pairs, one per text line. Returns the box height."""
    h = len(lines) * LINE + 2 * PAD
    y = HA - cy
    ax.add_patch(FancyBboxPatch((cx - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0,rounding_size=0.05",
                                fc=fill, ec=edge, lw=lw, ls=(0, (3, 2)) if dashed else "-",
                                zorder=2))
    top = y + (len(lines) - 1) * LINE / 2
    for i, (text, bold) in enumerate(lines):
        ax.text(cx, top - i * LINE, text, ha="center", va="center", color=INK,
                fontweight="bold" if bold else "normal", zorder=3)
    return h


def _arrow(ax, x0, y0, x1, y1, dashed=False):
    ax.annotate("", xy=(x1, HA - y1), xytext=(x0, HA - y0),
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color=INK if not dashed else MUTED,
                                ls=(0, (3, 2)) if dashed else "-", mutation_scale=7,
                                shrinkA=0, shrinkB=0), zorder=1)


def _line(ax, xs, ys):
    ax.plot(xs, [HA - y for y in ys], color=INK, lw=0.7, zorder=1, solid_capstyle="butt")


def draw_flow(ax, f: dict[str, int], excluded_line: str, n_sites: int) -> None:
    ax.set_xlim(0, W)
    ax.set_ylim(0, HA)
    ax.axis("off")
    N = f["analytic"]
    pct = lambda n: f"{n:,} ({100 * n / N:.1f}%)"   # noqa: E731

    # legend, top left
    for i, (label, edge, fill) in enumerate((("CLIF-donor criterion", CLIF, CLIF_TINT),
                                             ("CALC criterion", CALC, CALC_TINT),
                                             ("Excluded before analysis", BOTH, BOTH_TINT))):
        y = HA - 0.30 - i * 0.2
        ax.add_patch(FancyBboxPatch((0.10, y - 0.06), 0.26, 0.12,
                                    boxstyle="round,pad=0,rounding_size=0.03",
                                    fc=fill, ec=edge, lw=0.7, ls=(0, (3, 2))))
        ax.text(0.44, y, label, va="center", color=INK)

    # trunk: deaths in the window, the exclusions, the analytic cohort
    x_trunk, w_trunk = 3.6, 2.6
    x_grey, w_grey = 5.85, 2.5
    excluded = sum(f[k] for k, _ in UPSTREAM)
    systems = f"{n_sites} health system{'s' if n_sites != 1 else ''}, {f['hospitals']} hospitals"
    h_top = _box(ax, x_trunk, 0.45, w_trunk, [(f"In-hospital deaths, {WINDOW}", False),
                                              (f"{f['in_window']:,} decedents", True), (systems, False)],
                 INK, "white")
    grey = [(f"Excluded before analysis: {excluded:,}", True)]
    grey += [(f"{label}: {f[k]:,}", False) for k, label in UPSTREAM]
    if excluded_line:
        grey.append((f"({excluded_line})", False))
    _box(ax, x_grey, 1.3, w_grey, grey, BOTH, BOTH_TINT, dashed=True)
    y_an = 2.2
    h_an = _box(ax, x_trunk, y_an, w_trunk, [("Analytic cohort: inpatient deaths", False),
                                             (f"{N:,} decedents", True)], INK, "white")
    _line(ax, [x_trunk, x_trunk], [0.45 + h_top / 2, y_an - h_an / 2 - 0.012])
    _arrow(ax, x_trunk, y_an - h_an / 2 - 0.02, x_trunk, y_an - h_an / 2)
    _arrow(ax, x_trunk, 1.3, x_grey - w_grey / 2, 1.3, dashed=True)

    # the split into the two definitions
    y_split, y_end = 2.62, 6.15
    x_calc, x_clif = 1.15, 3.72
    _line(ax, [x_trunk, x_trunk], [y_an + h_an / 2, y_split])
    _line(ax, [x_calc, x_clif], [y_split, y_split])
    ax.text(x_calc + 0.15, HA - 2.8, "CALC, applied in sequence", color=CALC, fontweight="bold",
            ha="left", va="center")
    ax.text(x_clif + 0.15, HA - 2.8, f"CLIF-donor, applied in parallel (each of {N:,})", color=CLIF,
            fontweight="bold", ha="left", va="center")

    # CALC: cause, then age; location is the cohort entry
    x_cm, w_cm = 1.15, 2.15
    x_ce, w_ce = 2.95, 1.25
    end_h = 3 * LINE + 2 * PAD
    _line(ax, [x_calc, x_calc], [y_split, y_end - end_h / 2 - 0.012])
    _box(ax, x_ce, 3.2, w_ce, [("No qualifying", False), ("cause-of-death code", False),
                               (f"{N - f['calc_cause']:,}", True)], CALC, CALC_TINT, dashed=True)
    _arrow(ax, x_calc, 3.2, x_ce - w_ce / 2, 3.2, dashed=True)
    y1 = 3.95
    h1 = _box(ax, x_cm, y1, w_cm, [("Cause consistent with donation", False),
                                   (f"{POSITION_WORD} diagnosis in", False), (CODES, False),
                                   (pct(f["calc_cause"]), True)], INK, "white")
    _arrow(ax, x_calc, y1 - h1 / 2 - 0.02, x_calc, y1 - h1 / 2)
    _box(ax, x_ce, 4.55, w_ce, [(f"Age > {_AGE} at death", False),
                                (f"{f['calc_cause'] - f['calc']:,}", True)], CALC, CALC_TINT, dashed=True)
    _arrow(ax, x_calc, 4.55, x_ce - w_ce / 2, 4.55, dashed=True)
    y2 = 5.08
    h2 = _box(ax, x_cm, y2, w_cm, [(f"Age ≤ {_AGE} at death", False), (pct(f["calc"]), True)], INK, "white")
    _arrow(ax, x_calc, y2 - h2 / 2 - 0.02, x_calc, y2 - h2 / 2)
    _arrow(ax, x_calc, y_end - end_h / 2 - 0.02, x_calc, y_end - end_h / 2)
    _box(ax, x_cm, y_end, w_cm, [("Met CALC criteria", True), (pct(f["calc"]), True),
                                 ("location: inpatient death, by cohort entry", False)],
         CALC, CALC_TINT, lw=1.2)

    # CLIF-donor: six criteria side by side, then the decedents meeting all six
    x_cb, w_cb = 5.5, 3.0
    _line(ax, [x_clif, x_clif], [y_split, y_end])
    for (k, label), y in zip(CRITERIA, (3.1 + i * 0.47 for i in range(len(CRITERIA)))):
        _box(ax, x_cb, y, w_cb, [(label, False), (f"met {pct(f[k])} · not met {N - f[k]:,}", False)],
             CLIF, CLIF_TINT, dashed=True)
        _arrow(ax, x_clif, y, x_cb - w_cb / 2, y)
    w_end = 2.7
    _arrow(ax, x_clif, y_end, x_cb - w_end / 2, y_end)
    _box(ax, x_cb, y_end, w_end, [("CLIF-donor eligible: all six criteria", True), (pct(f["clif_donor"]), True),
                                  (f"kidney only {f['kidney_only']:,} · liver only {f['liver_only']:,} · "
                                   f"both {f['both']:,}", False)],
         CLIF, CLIF_TINT, lw=1.2)


def draw_organ(fig, o: dict[str, int]) -> None:
    n, k, l, kl = o["n_patients"], o["kidney_pass"], o["liver_pass"], o["kidney_and_liver_pass"]
    pct = lambda v: f"{100 * v / n:.1f}%"   # noqa: E731
    lab = lambda v: f"{v:,} ({pct(v)})"     # noqa: E731
    x_left, x_right = 0.30, 4.15            # where the two blocks start

    fig.text(x_left / W, (HB - 0.08) / H,
             f"Organ function among the {n:,} CLIF-donor eligible decedents; percentages are of the eligible",
             ha="left", va="top", color=INK, fontsize=7.5, fontweight="bold")
    fig.text(x_left / W, (HB - 0.33) / H, "Eligible decedents by the kidney and liver criteria they met",
             ha="left", va="top", color=INK)
    fig.text(x_right / W, (HB - 0.33) / H,
             "Why the other organ did not qualify (reasons can overlap)",
             ha="left", va="top", color=INK)

    # UpSet of the kidney and liver criteria
    cols = sorted([("Kidney and liver", True, True, kl), ("Kidney only", True, False, k - kl),
                   ("Liver only", False, True, l - kl)], key=lambda c: -c[3])
    xs = list(range(len(cols)))
    counts = [c[3] for c in cols]
    ax_bar = _fig_axes(fig, 1.95, 0.72, 1.8, 1.35)
    ax_mat = _fig_axes(fig, 1.95, 0.22, 1.8, 0.40)
    ax_set = _fig_axes(fig, x_left, 0.22, 0.70, 0.40)
    ax_bar.bar(xs, counts, width=0.6, color=CLIF, edgecolor="none", zorder=2)
    top = max(counts)
    for x, v in zip(xs, counts):
        ax_bar.text(x, v + top * 0.03, f"{v:,}\n{pct(v)}", ha="center", va="bottom", color=INK,
                    fontsize=6.5, linespacing=1.1)
    ax_bar.set_xlim(-0.6, len(cols) - 0.4)
    ax_bar.set_ylim(0, top * 1.55)
    ax_bar.set_ylabel("Decedents", color=INK)
    ax_bar.spines[["top", "right", "bottom"]].set_visible(False)
    ax_bar.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_bar.yaxis.set_major_locator(MaxNLocator(3))
    ax_bar.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax_bar.yaxis.grid(True, color="#EBEBEB", lw=0.5)
    ax_bar.set_axisbelow(True)

    rows = [("Kidney criteria met", 1), ("Liver criteria met", 2)]
    for r, (_, i) in enumerate(rows):
        ax_mat.scatter(xs, [r] * len(cols), s=26, lw=0, zorder=3,
                       color=[INK if c[i] else RULE for c in cols])
    for x, c in zip(xs, cols):
        if c[1] and c[2]:
            ax_mat.plot([x, x], [0, 1], color=INK, lw=1.1, zorder=2)
    ax_mat.axhspan(-0.5, 0.5, color="#F6F6F6", zorder=0, lw=0)
    ax_mat.set_xlim(-0.6, len(cols) - 0.4)
    ax_mat.set_ylim(len(rows) - 0.5, -0.5)
    ax_mat.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax_mat.tick_params(axis="y", left=False, pad=4)
    ax_mat.tick_params(axis="x", bottom=False, labelbottom=False)
    for sp in ax_mat.spines.values():
        sp.set_visible(False)

    met = [k, l]
    ax_set.barh(range(len(rows)), met, height=0.55, color=MUTED, edgecolor="none")
    for r, v in enumerate(met):
        ax_set.text(v + max(met) * 0.05, r, lab(v), va="center", ha="right", color=INK, fontsize=6.5)
    ax_set.set_ylim(len(rows) - 0.5, -0.5)
    ax_set.set_xlim(max(met) * 2.25, 0)
    ax_set.tick_params(axis="y", left=False, labelleft=False)
    ax_set.tick_params(axis="x", bottom=False, labelbottom=False)
    for sp in ax_set.spines.values():
        sp.set_visible(False)

    # why each criterion was not met: a bold header per criterion, its reasons below
    k_, lv = DONOR["kidney"], DONOR["liver"]
    names = {
        "on_crrt": f"CRRT within {k_['exclude_if_crrt_within_hours']} h",
        "creatinine_at_or_above_limit": f"Creatinine ≥ {k_['creatinine_max']} mg/dL",
        "creatinine_missing": "Creatinine missing",
        "bilirubin_at_or_above_limit": f"Total bilirubin ≥ {lv['total_bilirubin_max']} mg/dL",
        "ast_at_or_above_limit": f"AST ≥ {lv['ast_max']} U/L",
        "alt_at_or_above_limit": f"ALT ≥ {lv['alt_max']} U/L",
        "liver_lab_missing": "A liver test missing",
        "bmi_missing": "BMI missing",
        "bmi_above_limit": f"BMI > {DONOR['bmi_max']} kg/m²"}
    labels, values = [], []
    for head, why in ((f"Kidney criteria not met: {lab(n - k)}", KIDNEY_WHY),
                      (f"Liver criteria not met: {lab(n - l)}", LIVER_WHY)):
        labels.append(head), values.append(None)
        for w in sorted(why, key=lambda w: -o[w]):
            labels.append(names[w]), values.append(o[w])
    x_axes = 5.50
    ax_why = _fig_axes(fig, x_axes, 0.42, 0.90, 1.55)
    ys = list(range(len(labels)))
    vmax = max(v for v in values if v is not None)
    for y, text, v in zip(ys, labels, values):
        if v is None:
            ax_why.annotate(text, xy=(0, y), xycoords=("axes fraction", "data"),
                            xytext=(-(x_axes - x_right) * 72, 0), textcoords="offset points",
                            ha="left", va="center", color=INK, fontweight="bold")
        else:
            ax_why.barh(y, v, height=0.62, color=REASON, edgecolor="none")
            ax_why.text(v + vmax * 0.04, y, lab(v), va="center", ha="left", color=INK, fontsize=6.5)
    ax_why.set_ylim(len(labels) - 0.5, -0.5)
    ax_why.set_xlim(0, vmax * 1.05)
    ax_why.set_yticks(ys, [t if v is not None else "" for t, v in zip(labels, values)])
    ax_why.tick_params(axis="y", left=False, pad=3)
    ax_why.tick_params(axis="x", bottom=False, labelbottom=False)
    for sp in ax_why.spines.values():
        sp.set_visible(False)


def render(flow: dict[str, int], organ: dict[str, int], excluded_line: str, n_sites: int,
           stem: Path, formats=("png",), dpi: int = 300) -> list[Path]:
    """Draw both panels and save <stem>.<format> for each format."""
    _style()
    fig = plt.figure(figsize=(W, H))
    draw_flow(fig.add_axes([0, HB / H, 1, HA / H]), flow, excluded_line, n_sites)
    draw_organ(fig, organ)
    for label, y in (("A", H - 0.05), ("B", HB + 0.02)):
        fig.text(0.04 / W, y / H, label, fontsize=12, fontweight="bold", va="top", color=INK)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [stem.with_suffix(f".{ext}") for ext in formats]
    for p in paths:
        fig.savefig(p, dpi=dpi if p.suffix == ".png" else None, facecolor="white")
    plt.close(fig)
    return paths


def caption(flow: dict[str, int], organ: dict[str, int], n_sites: int) -> str:
    k, lv, cx = DONOR["kidney"], DONOR["liver"], DONOR["contraindications"]
    systems = f"{n_sites} US health system{'s' if n_sites != 1 else ''}"
    return f"""**Figure 1. Cohort selection and the CLIF-donor organ assessment.**

(A) Decedents in {systems} ({flow['hospitals']} hospitals), {WINDOW}. Decedents with no ADT
record, deaths in the emergency department without admission, deaths outside the study units,
deaths at hospitals excluded from the study and decedents without a known age are excluded before
analysis; the analytic cohort of {flow['analytic']:,} decedents is the inpatient deaths, the
denominator CMS uses for CALC. CALC is applied in sequence: a {POSITION_WORD} diagnosis of
{CODES}, then age ≤{_AGE} at death; location is the cohort entry; no contraindications are
applied (42 CFR 486.302). CLIF-donor's six criteria are applied in parallel, each counted among
all cohort decedents: no contraindicating cancer diagnosis, no positive blood culture within
{cx['positive_blood_culture_hours']} h before death, age ≤{_AGE}, invasive mechanical ventilation
within {DONOR['imv_hours_before_death']} h before death, BMI ≤{DONOR['bmi_max']} kg/m², and the
kidney or liver criteria met; a decedent is CLIF-donor eligible when all six hold, shown split by
the organ criteria met. Percentages are of the analytic cohort. (B) The organ assessment among the
{organ['n_patients']:,} CLIF-donor eligible decedents: the kidney and liver criteria each met, and for
those qualifying on one organ, why the other criterion was not met; percentages are of the eligible.
Kidney criteria: last creatinine <{k['creatinine_max']} mg/dL and no continuous renal
replacement therapy within {k['exclude_if_crrt_within_hours']} h before death. Liver criteria: last
AST and ALT <{lv['ast_max']} U/L and total bilirubin <{lv['total_bilirubin_max']} mg/dL. A patient
passes with BMI ≤{DONOR['bmi_max']} kg/m² and the kidney or the liver criteria met; a missing value
counts as not met. Left: eligible decedents by the kidney and liver criteria they met (dots: met; side
bars: decedents meeting each). Right: reasons the other organ's criteria were not met; a decedent can
have several, so reasons are not additive.

ADT, admission-discharge-transfer; ALT, alanine aminotransferase; AST, aspartate aminotransferase;
BMI, body mass index; CALC, Cause, Age, and Location-Consistent; CLIF, Common Longitudinal ICU data
Format; CRRT, continuous renal replacement therapy; ED, emergency department; IMV, invasive
mechanical ventilation.
"""
