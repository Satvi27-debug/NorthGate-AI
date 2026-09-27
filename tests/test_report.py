"""Tests for the written report.

The report is a graded deliverable, so these check the two ways it can be wrong
without anyone noticing: a pipe in a cell value silently splitting a row, and a
LIVE marker surviving into the published text.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FILL = ROOT / "scripts" / "fill_report.py"
TEMPLATE = ROOT / "reports" / "final_report.template.md"
REPORT = ROOT / "reports" / "final_report.md"
PDF_SCRIPT = ROOT / "scripts" / "export_report_pdf.py"


class TestTableCellEscaping:
    def test_pipes_in_cells_are_escaped(self):
        """A literal `|` in a cell must not split the row.

        This was a live bug. The acceptance row describing the label-leak guard
        contains the text `|r| > 0.99`, which a Markdown table parser reads as
        two cell separators. The row rendered as a "Measured" cell ending
        mid-sentence at "any input at" and a "Verdict" cell containing "r" - so
        the table appeared to report a verdict of "r" rather than showing a
        formatting fault, which is the worst kind of silent error.
        """
        sys.path.insert(0, str(FILL.parent))
        from fill_report import md_table

        out = md_table(["Criterion", "Measured"],
                       [["Guard", "any input at |r| > 0.99 fails the build"]])

        rows = [ln for ln in out.splitlines() if ln.startswith("|")]
        assert len(rows) == 3, f"expected a header, an align row and one data row, got {len(rows)}"
        # Two columns means three pipes per line, one of which is escaped.
        for line in rows:
            unescaped = re.sub(r"\\\|", "", line)
            assert unescaped.count("|") == 3, (
                f"row has {unescaped.count('|')} unescaped pipes, expected 3 for a "
                f"two-column table: {line!r}"
            )
        assert r"\|r\|" in out, "the pipe should be backslash-escaped, not removed"

    def test_generated_tables_have_consistent_column_counts(self):
        """No table in the rendered report may have a ragged row.

        A ragged row is always an unescaped pipe, and it always damages the
        table rather than failing loudly.
        """
        if not REPORT.exists():
            pytest.skip("final_report.md not built")
        text = REPORT.read_text(encoding="utf-8")
        lines = text.splitlines()

        checked = 0
        i = 0
        while i < len(lines):
            if re.match(r"^\s*\|.*\|\s*$", lines[i]) and i + 1 < len(lines) \
                    and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
                header = lines[i]
                ncols = len(re.sub(r"\\\|", "", header).split("|")) - 2
                j = i + 2
                while j < len(lines) and re.match(r"^\s*\|.*\|\s*$", lines[j]):
                    cells = len(re.sub(r"\\\|", "", lines[j]).split("|")) - 2
                    assert cells == ncols, (
                        f"ragged table row at line {j + 1}: header declares "
                        f"{ncols} columns, row has {cells}.\n"
                        f"  {lines[j][:150]}\n"
                        f"An unescaped '|' in a cell value is the usual cause."
                    )
                    j += 1
                checked += 1
                i = j
            else:
                i += 1
        assert checked >= 4, (
            f"only {checked} tables found in the report; the acceptance and "
            f"leaderboard tables should be among them"
        )


class TestReportIntegrity:
    def test_no_unrendered_live_markers(self):
        """An unfilled marker ships as visible junk in the deliverable."""
        if not REPORT.exists():
            pytest.skip("final_report.md not built")
        text = REPORT.read_text(encoding="utf-8")
        left = re.findall(r"<!--LIVE:([A-Z_]+)-->", text)
        assert not left, f"unrendered markers survived into the report: {left}"

    def test_report_states_the_negative_result(self):
        """Section 8.3 requires the failure be reported, not buried.

        Asserted on the published text rather than the template, because the
        template is the thing that could drift away from what was measured.
        """
        if not REPORT.exists():
            pytest.skip("final_report.md not built")
        text = REPORT.read_text(encoding="utf-8").lower()
        assert "neither of" in text and "acceptance bar" in text, (
            "the abstract must state plainly that the acceptance bars were not met"
        )
        # The random-walk comparison has to be present with real numbers.
        assert "random walk" in text
        assert "directional accuracy" in text

    def test_report_has_no_absolute_user_paths(self):
        """A committed report must not carry the build machine's username."""
        for p in (REPORT, TEMPLATE):
            if not p.exists():
                continue
            text = p.read_text(encoding="utf-8")
            assert not re.search(r"[A-Za-z]:\\\\?Users\\\\", text), (
                f"{p.name} contains an absolute user path, which leaks the build "
                f"machine's account name and is wrong for every other reader"
            )


class TestPdfExport:
    def test_pdf_export_exists_and_is_runnable(self):
        assert PDF_SCRIPT.exists(), (
            "scripts/export_report_pdf.py is the PDF deliverable and is missing"
        )
        src = PDF_SCRIPT.read_text(encoding="utf-8")
        assert "tables" in src, (
            "the Markdown `tables` extension is required; without it every "
            "comparison table renders as a paragraph of pipe characters"
        )
        assert "--print-to-pdf" in src, (
            "the PDF is produced by a headless browser; the flag must be present"
        )

    def test_pdf_exists_and_is_valid(self):
        pdf = ROOT / "NorthGate-AI-Report.pdf"
        if not pdf.exists():
            pytest.skip("PDF not generated; run scripts/export_report_pdf.py")
        raw = pdf.read_bytes()
        assert raw[:4] == b"%PDF", "the file is not a PDF"
        # A browser that rendered an empty page writes a few KB; a 20-page
        # report with embedded fonts is hundreds.
        assert len(raw) > 40_000, (
            f"the PDF is only {len(raw):,} bytes, which is too small for the "
            f"report - the render probably produced a near-empty document"
        )
        pages = len(re.findall(rb"/Type\s*/Page[^s]", raw))
        assert pages >= 10, f"only {pages} pages; the report should be longer"
