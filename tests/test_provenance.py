"""Tests for utils.provenance.render: the data-quality section."""
from __future__ import annotations

from utils.provenance import render

HEADER = "site,flag,severity,detail\n"


def test_flags_are_rendered_before_the_cohort_cascade(tmp_path):
    (tmp_path / "data_quality_flags.csv").write_text(
        HEADER + "x,calc_external_cause_codes_absent,high,CALC lacks its external-cause arm\n")
    (tmp_path / "strobe_counts.csv").write_text("site,order,metric,n\nx,1,2_inpatient_decedents,100\n")
    text = render(tmp_path, problems=[]).read_text()
    assert "## Data-quality flags" in text
    assert "calc_external_cause_codes_absent" in text and "CALC lacks its external-cause arm" in text
    assert text.index("## Data-quality flags") < text.index("## Cohort cascade")


def test_no_flags_is_stated_explicitly(tmp_path):
    (tmp_path / "data_quality_flags.csv").write_text(HEADER)
    text = render(tmp_path, problems=[]).read_text()
    assert "## Data-quality flags" in text and "None raised" in text


def test_bundles_without_the_file_render_as_before(tmp_path):
    text = render(tmp_path, problems=[]).read_text()
    assert "## Data-quality flags" not in text


def test_audit_notes_are_shown_even_with_no_patients(tmp_path):
    (tmp_path / "audit").mkdir()
    (tmp_path / "audit" / "tables_stage_counts.csv").write_text(
        "site,label,stage,n_in,n_out,n_dropped,pct_retained,rule,reason\n"
        "x,tables,var_med_prone,0,0,0,,UNAVAILABLE — BinderException: column med_category not found,\n")
    text = render(tmp_path, problems=[]).read_text()
    assert "var_med_prone" in text and "BinderException" in text
