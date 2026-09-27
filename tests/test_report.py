"""Tests for the written report.

The report is a graded deliverable, so these check the two ways it can be wrong
without anyone noticing: a pipe in a cell value silently splitting a row, and a
LIVE marker surviving into the published text.
"""

from __future__ import annotations

import os
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
        """A committed report must not carry the build machine's username.

        Scans every generated artifact under reports/, not just the two prose
        files. The original version checked `final_report.md` and the template
        only, and so passed while `clean_env_check.json` was sitting in the
        repository holding the full temp path of the build account - because
        nothing was looking there.
        """
        reports_dir = ROOT / "reports"
        targets = [REPORT, TEMPLATE] + sorted(reports_dir.glob("*"))
        checked = 0
        for p in targets:
            if not p.exists() or not p.is_file() or p.suffix == ".ipynb":
                continue
            # The notebook is large and legitimately holds JSON; read as text is
            # fine, but skip binaries that are not text at all.
            try:
                text = p.read_text(encoding="utf-8")
            except (UnicodeDecodeError, ValueError):
                continue
            checked += 1
            hit = re.search(r"[A-Za-z]:\\+Users\\|/(?:home|Users)/[^\s\"']", text)
            assert not hit, (
                f"{p.relative_to(ROOT)} contains an absolute user path "
                f"({hit.group(0) if hit else ''}), which leaks the build "
                f"machine's account name and is wrong for every other reader"
            )
        assert checked >= 5, (
            f"only {checked} report artifacts were scanned; the confidentiality "
            f"check is not covering the directory it is meant to cover"
        )

    def test_every_committed_file_is_free_of_this_accounts_paths(self):
        """Repo-wide sweep for the BUILD ACCOUNT's own paths.

        Keyed to the account running the build rather than to a generic path
        shape. A shape-based rule flags legitimate documentation like
        `C:\\Users\\<name>\\Desktop\\...` and, worse, forces the tests that
        exercise the scrubber to stop using realistic path-shaped fixtures - so
        the redaction stops being tested against the thing it must handle. What
        must never ship is *this* machine's home directory and account name, and
        those are known at runtime without ever writing them into the source.
        """
        import getpass

        needles = set()
        try:
            home = str(Path.home())
        except (RuntimeError, OSError):
            home = ""
        if home and len(home) > 3:
            needles.add(home)
            needles.add(home.replace("\\", "/"))
        for var in ("USER", "USERNAME", "LOGNAME", "HOME", "USERPROFILE"):
            val = os.environ.get(var, "")
            if len(val) >= 4 and not val.startswith("/"):
                needles.add(val)
        try:
            user = getpass.getuser()
            if len(user) >= 4:
                needles.add(user)
        except Exception:  # noqa: BLE001
            pass
        needles = {n for n in needles if n and len(n) >= 4}
        assert needles, (
            "could not determine the build account, so the confidentiality "
            "sweep cannot run; refusing to report a pass it did not verify"
        )

        skip_dirs = {".git", "__pycache__", ".pytest_cache", "data", "models",
                     ".venv", "gdelt_cache", "node_modules"}
        skip_ext = {".parquet", ".pkl", ".keras", ".h5", ".pdf", ".png", ".jpg",
                    ".pyc", ".ipynb", ".zip"}
        offenders = []
        scanned = 0
        for p in ROOT.rglob("*"):
            if not p.is_file() or p.suffix.lower() in skip_ext:
                continue
            if any(part in skip_dirs for part in p.parts):
                continue
            try:
                text = p.read_text(encoding="utf-8")
            except (UnicodeDecodeError, ValueError):
                continue
            scanned += 1
            for needle in needles:
                if needle in text:
                    offenders.append(
                        f"{p.relative_to(ROOT)} (contains {needle!r})")
                    break
        assert scanned > 20, (
            f"only {scanned} files were scanned; the sweep is not covering the "
            f"repository it is meant to cover"
        )
        assert not offenders, (
            f"these committed files contain this build account's name or home "
            f"directory, which would be published on clone: {offenders}"
        )


class TestCleanEnvScrubber:
    """The clean-env report is generated, so its redaction needs its own test.

    Three separate bugs lived in this one regex and each was invisible until it
    was exercised: an over-escaped drive-letter pattern that matched nothing at
    all, a body that stopped at the first space and so leaked the remainder of a
    spaced username, and a scrubber only ever tested against POSIX input. A
    redaction that silently fails looks exactly like one that works.
    """

    @staticmethod
    def _scrubber():
        import importlib.util

        src = ROOT / "scripts" / "clean_env_check.py"
        spec = importlib.util.spec_from_file_location("cec_under_test", src)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_redacts_windows_paths_including_spaced_usernames(self):
        mod = self._scrubber()
        # Assembled at runtime from fragments. Writing this build account's real
        # home directory into the fixture would make the test file itself the
        # very leak it exists to prevent, and a placeholder is not path-shaped
        # enough to exercise the greedy body of the pattern.
        sep = "\\"
        home = sep.join(["C:", "Users", "some user", "AppData", "Local", "Temp"])
        cases = [
            f'Actual location:    "{home}{sep}venv"',
            f'  File "{home}{sep}proj{sep}src{sep}x.py", line 3',
            f"{home}{sep}venv{sep}Scripts{sep}python.exe -m pytest",
        ]
        for case in cases:
            out = " ".join(mod._clean(case, limit=5))
            assert "some user" not in out, f"username survived: {out!r}"
            assert "AppData" not in out, f"home directory survived: {out!r}"
            assert "Local" not in out, f"home tail survived: {out!r}"

    def test_redacts_posix_paths(self):
        mod = self._scrubber()
        for case in ("/home/runner/work/repo/venv/lib",
                     "/Users/someone/Desktop/NorthGate",
                     "/root/.cache/pip"):
            out = " ".join(mod._clean(case, limit=5))
            assert out == "<path>", f"{case!r} was not redacted: {out!r}"

    def test_leaves_ordinary_diagnostics_intact(self):
        """Redaction must not destroy the evidence the report exists to carry."""
        mod = self._scrubber()
        for case in ("159 passed in 82.36s",
                     "FINNHUB_API_KEY is not set",
                     "ModuleNotFoundError: No module named 'numpy.rec'"):
            out = " ".join(mod._clean(case, limit=5))
            assert case in out, (
                f"a non-path diagnostic was mangled: {case!r} -> {out!r}"
            )


class TestReadmeIsGenerated:
    """The README's numbers must come from artifacts, not from a paste.

    It carried hand-copied figures that drifted: an optimised Sharpe of 1.357
    was quoted while the artifact said 1.039, because the number was written
    once and never refreshed. A headline figure that disagrees with the artifact
    it describes is worse than no figure, so the two blocks are generated.
    """

    def test_no_live_marker_survives_into_the_readme(self):
        p = ROOT / "README.md"
        if not p.exists():
            pytest.skip("README.md missing")
        left = re.findall(r"<!--LIVE:[A-Z_]+-->", p.read_text(encoding="utf-8"))
        assert not left, (
            f"unfilled markers in README.md: {left}; run "
            f"`python scripts/fill_readme.py`"
        )

    def test_every_renderer_has_a_marker(self):
        """A renderer with no marker is dead code; a marker with no renderer
        ships a raw comment to the reader."""
        import importlib.util

        script = ROOT / "scripts" / "fill_readme.py"
        spec = importlib.util.spec_from_file_location("fill_readme_t", script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        tpl = ROOT / "reports" / "readme.template.md"
        if not tpl.exists():
            pytest.skip("readme.template.md missing")
        text = tpl.read_text(encoding="utf-8")
        for name in mod.SECTIONS:
            assert f"<!--LIVE:{name}-->" in text, (
                f"renderer {name!r} has no marker in the template, so it never "
                f"runs and is dead code"
            )
        for marker in re.findall(r"<!--LIVE:([A-Z_]+)-->", text):
            assert marker in mod.SECTIONS, (
                f"template has marker {marker!r} with no renderer behind it"
            )

    def test_quoted_figures_agree_with_the_artifacts(self):
        """The check that would have caught the stale Sharpe."""
        import json

        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        pm_path = ROOT / "data" / "processed" / "portfolio_metrics.json"
        if not pm_path.exists():
            pytest.skip("portfolio_metrics.json not built")
        pm = json.loads(pm_path.read_text(encoding="utf-8"))
        bt = pm.get("backtest", {})
        opt = (bt.get("optimised") or {}).get("sharpe")
        eq = (bt.get("equal_weight") or {}).get("sharpe")
        if opt is None or eq is None:
            pytest.skip("portfolio backtest not populated")
        assert f"{float(opt):.3f}" in readme, (
            f"the README does not quote the measured optimised Sharpe "
            f"{float(opt):.3f}; it is carrying a stale hand-copied figure"
        )
        assert f"{float(eq):.3f}" in readme, (
            f"the README does not quote the equal-weight Sharpe {float(eq):.3f}"
        )

        rb_path = ROOT / "data" / "processed" / "recommendation_backtest.json"
        if rb_path.exists():
            rb = json.loads(rb_path.read_text(encoding="utf-8"))
            bh = rb.get("buy_and_hold_hit_rate")
            g = rb.get("arms", {}).get("gated", {}).get("hit_rate")
            if bh is not None and g is not None:
                assert f"{bh * 100:.2f}%" in readme, (
                    "the README does not quote the measured buy-and-hold "
                    f"hit-rate {bh * 100:.2f}%"
                )

    def test_readme_does_not_assert_an_unmet_bar_is_met(self):
        """The headline must not claim a pass the acceptance table does not."""
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        if "| Criterion |" not in (ROOT / "reports" / "final_report.md").read_text(
                encoding="utf-8"):
            pytest.skip("acceptance table not rendered")
        rep = (ROOT / "reports" / "final_report.md").read_text(encoding="utf-8")
        block = rep[rep.rindex("| Criterion |"):]
        failing = []
        cur = None
        for ln in block.splitlines():
            if not ln.startswith("|"):
                if failing:
                    break
                continue
            cells = [c.strip().replace("**", "") for c in re.split(r"(?<!\\)\|", ln)]
            if len(cells) < 5:
                continue
            if cells[1] not in ("", "---"):
                cur = cells[1]
            v = cells[4].split("(")[0].split("—")[0].strip()
            if v in ("NOT MET", "NOT RUN") and cur:
                failing.append(cur)
        for crit in failing:
            # The README must name every unmet criterion, not quietly omit it.
            assert crit.split("(")[0].strip()[:30] in readme or crit in readme, (
                f"the acceptance table reports {crit!r} as unmet, but the README "
                f"does not mention it - the summary is not telling the whole truth"
            )


class TestAcceptanceTable:
    def test_no_criterion_appears_twice(self):
        """A duplicated criterion shows two verdicts for one requirement.

        The clean-environment row was emitted twice - once from the signed-off
        result of `clean_env_check.py` and once as a hardcoded "not yet run" -
        so the table asserted both MET and NOT MET for the same row. A reader
        checking a specific criterion could land on either.
        """
        if not REPORT.exists():
            pytest.skip("final_report.md not built")
        text = REPORT.read_text(encoding="utf-8")
        start = text.rindex("| Criterion |")
        criteria = []
        for line in text[start:].splitlines():
            if not line.startswith("|"):
                if criteria:
                    break
                continue
            cells = re.split(r"(?<!\\)\|", line.strip())
            if len(cells) > 2 and cells[1].strip() not in ("", "---"):
                c = cells[1].strip()
                if c != "Criterion":
                    criteria.append(c)
        dupes = {c for c in criteria if criteria.count(c) > 1}
        assert not dupes, (
            f"the acceptance table lists these criteria more than once, which "
            f"means contradictory verdicts for the same requirement: {sorted(dupes)}"
        )
        assert len(criteria) >= 20, f"only {len(criteria)} acceptance rows rendered"

    def test_verdicts_are_from_the_known_set(self):
        if not REPORT.exists():
            pytest.skip("final_report.md not built")
        text = REPORT.read_text(encoding="utf-8")
        start = text.rindex("| Criterion |")
        allowed = {"MET", "NOT MET", "NOT RUN"}
        bad = []
        for line in text[start:].splitlines():
            if not line.startswith("|"):
                continue
            cells = [c.strip() for c in re.split(r"(?<!\\)\|", line.strip())]
            if len(cells) < 5:
                continue
            verdict = cells[4].replace("**", "").strip()
            if verdict in ("", "---", "Verdict"):
                continue
            # Match on a prefix, not the first word: two of the three verdicts
            # are two words ("NOT MET", "NOT RUN"), so splitting on spaces
            # truncates them to "NOT" and then rejects a perfectly valid row.
            if not any(verdict.startswith(v) for v in allowed):
                bad.append(verdict[:60])
        assert not bad, (
            f"unrecognised verdicts in the acceptance table: {bad}; each must "
            f"start with one of {sorted(allowed)}"
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
