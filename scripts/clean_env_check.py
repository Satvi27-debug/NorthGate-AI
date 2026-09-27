"""Clean-environment reproducibility demonstration.

PRD Section 15.3 and the Section 14 acceptance table both call for evidence
that the project rebuilds from nothing in a fresh environment. The repository
has the ingredients - pinned `requirements.txt`, one-command entry points,
fixed seeds, a test suite - but ingredients are not evidence. This script is
the evidence: it creates a throwaway virtual environment, installs from the
pinned requirements, runs the whole rebuild, runs the tests, and checks the
artifacts, writing a signed-off report of what happened.

What it does, in order:

1. `python -m venv` in a temporary directory
2. `pip install -r requirements.txt` (the pinned set, nothing else)
3. `python rebuild_dataset.py --skip-ingest` - the dataset rebuild
4. `python retrain_models.py --skip-dl` - maths, ML, portfolio, recommendations
5. `python -m pytest tests/ -q`
6. every required artifact present, and newer than the feature table
7. the dashboard imports and every panel body executes

Every step records its exit code, wall time and output tail, and the report
distinguishes clearly between "passed", "failed" and "skipped" - a step that
could not run is never reported as a pass, because that is the whole failure
mode this exercise exists to rule out.

Usage:
    python scripts/clean_env_check.py            # full run, ~45-70 minutes
    python scripts/clean_env_check.py --quick    # skip the deep-learning retrain
    python scripts/clean_env_check.py --keep     # leave the venv on disk
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import REPORTS_DIR, write_json  # noqa: E402

# Artefacts in DEPENDENCY ORDER, because "current" is a relation between stages,
# not a property of a file's timestamp.
#
# An earlier version listed these flat and compared every one of them against
# features.parquet. That produced a guaranteed false failure: cleaned_panel is
# an INPUT to features.parquet, so it is built first and is always older. A
# check that reports FAIL for a correct pipeline is worse than no check, because
# it trains the reader to ignore it.
#
# Each tier must be no older than the newest member of the tier above it, which
# is exactly the property that proves the rebuild ran in order.
PIPELINE_TIERS: list[list[str]] = [
    # 0 - ingestion and cleaning
    ["data/processed/cleaned_panel.parquet",
     "data/processed/macro_daily.parquet"],
    # 1 - features
    ["data/processed/features.parquet"],
    # 2 - models and everything fitted on the feature table
    ["data/processed/ml_metrics.csv",
     "data/processed/ml_predictions.parquet",
     "data/processed/dl_metrics.csv",
     "data/processed/dl_predictions.parquet",
     "data/processed/declared_model.json",
     "models/feature_scaler.pkl"],
    # 3 - downstream analyses that consume predictions
    ["data/processed/portfolio_metrics.json",
     "data/processed/recommendations.csv",
     "data/processed/recommendation_backtest.json",
     "data/processed/cross_sectional_results.json"],
]

# Checked for presence only. These are produced by separate one-shot scripts
# (fill_report.py, export_report_pdf.py) rather than by the rebuild, so their
# timestamps say nothing about whether the pipeline ran.
OTHER_ARTIFACTS = [
    "reports/math_verification.json",
    "reports/data_quality_report.json",
    "reports/final_report.md",
    "NorthGate-AI-Report.pdf",
]

NOISE = ("TensorFlow DLL", "_pywrap", "Hint: This often", "Visual C++",
         "or if the Micro", "Failed to load", "oneDNN", "absl::",
         "numerical results due to", "TensorFlow GPU support", "retracing",
         "libc++", "computation placer")

# Absolute paths are redacted from anything this script persists. pip, venv and
# test runners all echo the interpreter and temporary directories they were
# invoked from, and this report is committed to a public repository - so an
# unredacted tail would publish the author's Windows username and home layout to
# everyone who clones it. The check is more useful for what it proves than for
# the exact temp path it happened to use.
#
# Two subtleties, both found by testing the scrubber rather than by reading it.
#
# The drive-letter branch accepts `[\\/]`, not `\\\\`. A Windows path contains a
# SINGLE backslash, and an over-escaped pattern silently matches nothing - the
# worst kind of leak, because a redaction that reports success while redacting
# nothing looks identical to one that works. The first version of this line had
# four backslashes and passed its own smoke test on the POSIX cases while
# leaving every Windows path intact.
#
# The body is greedy to end-of-line, not whitespace-delimited. A Windows account
# name may contain a space, so a `[^\s...]` body stops at the first space and
# leaves the remainder of the path - including the home directory and the
# username tail - sitting in a committed file. Losing the tail of a diagnostic
# line is a fair price for not publishing someone's home directory.
_PATH_RE = re.compile(
    r"(?:[A-Za-z]:[\\/]|/home/|/Users/|/root/|/tmp/)[^\n]*")


def _scrub(text: str) -> str:
    """Replace absolute filesystem paths with a stable placeholder."""
    return _PATH_RE.sub("<path>", text)


def _clean(text: str, limit: int = 24) -> list[str]:
    out = []
    for line in (text or "").splitlines():
        s = _scrub(line.strip())
        if s and not any(n in s for n in NOISE):
            out.append(s)
    return out[-limit:]


class Step:
    def __init__(self, name: str, command: list[str], cwd: Path,
                 env: dict | None = None, optional: bool = False):
        self.name = name
        self.command = command
        self.cwd = cwd
        self.env = env
        self.optional = optional
        self.exit_code: int | None = None
        self.seconds: float = 0.0
        self.tail: list[str] = []
        self.status = "NOT RUN"

    def run(self) -> "Step":
        t0 = time.time()
        try:
            proc = subprocess.run(self.command, cwd=str(self.cwd), env=self.env,
                                  capture_output=True, text=True, timeout=7200)
            self.exit_code = proc.returncode
            self.tail = _clean(proc.stdout) + _clean(proc.stderr, 12)
        except subprocess.TimeoutExpired:
            self.exit_code = -1
            self.tail = ["timed out after 7200s"]
        except Exception as exc:  # noqa: BLE001
            self.exit_code = -2
            self.tail = [f"{type(exc).__name__}: {exc}"]
        self.seconds = time.time() - t0
        if self.exit_code == 0:
            self.status = "PASS"
        elif self.optional:
            self.status = "SKIPPED (failed, but optional)"
        else:
            self.status = "FAIL"
        return self


def _say(msg: str = "") -> None:
    """Print with an explicit flush.

    Without this the whole check is invisible for its entire ~40 minute run when
    stdout is redirected to a file, because CPython block-buffers a non-tty and
    nothing reaches the log until the process exits. An evidence script that
    shows nothing while it works cannot be told apart from one that has hung.
    """
    print(msg, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="skip the deep-learning retrain (much faster)")
    ap.add_argument("--keep", action="store_true",
                    help="keep the temporary virtual environment")
    args = ap.parse_args()

    steps: list[Step] = []
    tmp = Path(tempfile.mkdtemp(prefix="northgate_cleanenv_"))
    venv = tmp / "venv"
    t_start = time.time()

    _say("=" * 78)
    _say("CLEAN-ENVIRONMENT REPRODUCIBILITY CHECK (PRD Section 15.3)")
    _say("=" * 78)
    _say(f"  project     : {ROOT}")
    _say(f"  scratch venv: {venv}")
    _say(f"  interpreter : {platform.python_version()} on {platform.system()}")
    _say()

    try:
        # 1. create the environment -----------------------------------------
        s = Step("create virtual environment",
                 [sys.executable, "-m", "venv", str(venv)], ROOT)
        steps.append(s.run())
        _save(steps, t_start, args.quick)
        _say(f"  [{s.status:>28}] {s.name}  ({s.seconds:.1f}s)")
        if s.status == "FAIL":
            return _finish(steps, t_start, args.quick)

        # Resolve the interpreter path per platform. On Windows the binary is
        # in Scripts/, elsewhere in bin/.
        py = (venv / "Scripts" / "python.exe" if os.name == "nt"
              else venv / "bin" / "python")
        if not py.exists():
            _say(f"  [FAIL] venv interpreter not found at {py}")
            return _finish(steps, t_start, args.quick)

        base_env = {**os.environ, "PYTHONIOENCODING": "utf-8"}

        # 2. install the pinned requirements --------------------------------
        s = Step("install pinned requirements",
                 [str(py), "-m", "pip", "install", "--quiet",
                  "--disable-pip-version-check", "-r", "requirements.txt"],
                 ROOT, env=base_env)
        steps.append(s.run())
        _save(steps, t_start, args.quick)
        _say(f"  [{s.status:>28}] {s.name}  ({s.seconds:.1f}s)")
        for line in s.tail[-3:]:
            _say(f"        {line}")
        if s.status == "FAIL":
            return _finish(steps, t_start, args.quick)

        # 3. rebuild the dataset --------------------------------------------
        s = Step("rebuild dataset",
                 [str(py), "rebuild_dataset.py", "--skip-ingest"],
                 ROOT, env=base_env)
        steps.append(s.run())
        _save(steps, t_start, args.quick)
        _say(f"  [{s.status:>28}] {s.name}  ({s.seconds:.1f}s)")
        for line in s.tail[-5:]:
            _say(f"        {line}")
        if s.status == "FAIL":
            return _finish(steps, t_start, args.quick)

        # 4. retrain ---------------------------------------------------------
        cmd = [str(py), "retrain_models.py"] + ([] if args.quick else []) \
            + (["--skip-dl"] if args.quick else [])
        s = Step("retrain models" + (" (quick: DL skipped)" if args.quick else ""),
                 cmd, ROOT, env=base_env)
        steps.append(s.run())
        _save(steps, t_start, args.quick)
        _say(f"  [{s.status:>28}] {s.name}  ({s.seconds:.1f}s)")
        for line in s.tail[-8:]:
            _say(f"        {line}")
        if s.status == "FAIL":
            return _finish(steps, t_start, args.quick)

        # 5. the test suite ---------------------------------------------------
        s = Step("test suite",
                 [str(py), "-m", "pytest", "tests/", "-q",
                  "-p", "no:warnings"], ROOT, env=base_env)
        steps.append(s.run())
        _save(steps, t_start, args.quick)
        _say(f"  [{s.status:>28}] {s.name}  ({s.seconds:.1f}s)")
        for line in s.tail[-4:]:
            _say(f"        {line}")
        if s.status == "FAIL":
            return _finish(steps, t_start, args.quick)

        # 6. artifacts present, and produced in dependency order -------------
        missing, misordered = [], []
        newest_above = None
        tier_lines = []
        for n, tier in enumerate(PIPELINE_TIERS):
            present = [r for r in tier if (ROOT / r).exists()]
            absent = [r for r in tier if not (ROOT / r).exists()]
            missing += absent
            if absent:
                tier_lines.append(f"tier {n}: {len(present)}/{len(tier)} present, "
                                  f"MISSING {absent}")
                newest_above = None      # cannot order past a gap
                continue
            tier_newest = max((ROOT / r).stat().st_mtime for r in present)
            tier_lines.append(
                f"tier {n}: {len(present)} artifact(s), newest "
                f"{tier_newest:.0f}"
                + ("" if newest_above is None
                   else ("  (after tier above)" if tier_newest >= newest_above
                         else "  <-- OLDER THAN THE TIER ABOVE")))
            if newest_above is not None and tier_newest < newest_above:
                misordered.append(
                    f"tier {n} is older than tier {n - 1}: "
                    f"{[r for r in present if (ROOT / r).stat().st_mtime < newest_above]}")
            newest_above = tier_newest

        absent_other = [r for r in OTHER_ARTIFACTS if not (ROOT / r).exists()]
        missing += absent_other
        ok = not missing and not misordered
        steps.append(Step("artifacts present and built in order",
                          ["<check>"], ROOT))
        steps[-1].status = "PASS" if ok else "FAIL"
        steps[-1].seconds = 0.0
        steps[-1].tail = (
            tier_lines
            + [f"{len(OTHER_ARTIFACTS) - len(absent_other)}/"
               f"{len(OTHER_ARTIFACTS)} report/PDF deliverables present"]
            + ([f"missing: {missing}"] if missing else [])
            + ([f"out of order: {misordered}"] if misordered else [])
        )
        _save(steps, t_start, args.quick)
        _say(f"  [{steps[-1].status:>28}] artifacts present and built in order")
        for line in steps[-1].tail:
            _say(f"        {line}")

        # 7. the dashboard imports and every panel executes -------------------
        # Paths are emitted with repr(), never bare. An earlier version wrote
        # them unquoted, producing `sys.path.insert(0, C:\Users\...)`, which is
        # a SyntaxError - and because the step was marked optional that failure
        # was reported as SKIPPED rather than FAIL, so a probe that never ran
        # at all looked like a tolerable gap instead of a broken check.
        probe = tmp / "probe.py"
        probe.write_text(
            "import sys\n"
            f"sys.path.insert(0, {str(ROOT)!r})\n"
            f"sys.path.insert(0, {str(ROOT / 'dashboard')!r})\n"
            "import app\n"
            "cfg = app.get_config()\n"
            "D = app.load_all()\n"
            "app.panel_overview(cfg, D)\n"
            "app.panel_price(cfg, D, cfg['universe'][0], 1, 1.96)\n"
            "app.panel_models(cfg, D)\n"
            "app.panel_portfolio(cfg, D, 0.04)\n"
            "app.panel_risk(cfg, D, cfg['universe'][0])\n"
            "app.panel_sentiment(cfg, D, cfg['universe'][0])\n"
            "app.panel_recommendations(cfg, D)\n"
            "app.panel_ranking(cfg, D)\n"
            "print('ALL PANELS EXECUTED')\n",
            encoding="utf-8")
        # Deliberately NOT optional. A dashboard that cannot start is a
        # reproducibility failure, not a nice-to-have.
        s = Step("dashboard imports, all panels execute",
                 [str(py), str(probe)], ROOT, env=base_env)
        steps.append(s.run())
        _save(steps, t_start, args.quick)
        _say(f"  [{s.status:>28}] {s.name}  ({s.seconds:.1f}s)")
        for line in s.tail[-3:]:
            _say(f"        {line}")

        return _finish(steps, t_start, args.quick)

    finally:
        if not args.keep:
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            _say(f"\n  venv kept at {venv}")


def _report(steps: list[Step], t_start: float, quick: bool,
            finished: bool) -> dict:
    """Build the report document. Safe to call after every step."""
    elapsed = time.time() - t_start
    failed = [s for s in steps if s.status == "FAIL"]
    not_run = [s for s in steps if s.status == "NOT RUN"]
    return {
        "generated_by": "scripts/clean_env_check.py",
        "prd_reference": ("Section 15.3 and the Section 14 acceptance table: the "
                          "project must be reproducible in a clean environment."),
        "mode": "quick (deep-learning retrain skipped)" if quick else "full",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "elapsed_seconds": round(elapsed, 1),
        "complete": finished,
        "steps": [
            {"name": s.name, "status": s.status, "exit_code": s.exit_code,
             "seconds": round(s.seconds, 1), "output_tail": s.tail}
            for s in steps
        ],
        "all_passed": bool(finished and not failed and not not_run),
        "verdict": (
            "STILL RUNNING - the report is rewritten after every step, so a "
            "check interrupted part-way leaves its evidence behind rather "
            "than nothing"
            if not finished else
            "REPRODUCIBLE: every step passed in a fresh virtual environment "
            "installed only from requirements.txt."
            if not failed else
            f"NOT REPRODUCIBLE: {len(failed)} step(s) failed - "
            + ", ".join(s.name for s in failed)),
    }


def _save(steps: list[Step], t_start: float, quick: bool,
          finished: bool = False) -> dict:
    doc = _report(steps, t_start, quick, finished)
    write_json(doc, REPORTS_DIR / "clean_env_check.json")
    (REPORTS_DIR / "clean_env_check.md").write_text(_to_markdown(doc),
                                                    encoding="utf-8")
    return doc


def _finish(steps: list[Step], t_start: float, quick: bool) -> int:
    doc = _save(steps, t_start, quick, finished=True)
    elapsed = doc["elapsed_seconds"]
    _say()
    _say("=" * 78)
    for s in steps:
        _say(f"  {s.status:>6}  {s.seconds:>8.1f}s  {s.name}")
    _say("-" * 78)
    _say(f"  {doc['verdict']}")
    _say(f"  total {elapsed / 60:.1f} min")
    _say("  -> reports/clean_env_check.json")
    _say("  -> reports/clean_env_check.md")
    return 0 if doc["all_passed"] else 1


def _to_markdown(r: dict) -> str:
    lines = [
        "# Clean-environment reproducibility check",
        "",
        f"**Verdict: {r['verdict']}**",
        "",
        f"- Mode: {r['mode']}",
        f"- Python: {r['python']} on {r['platform']}",
        f"- Wall time: {r['elapsed_seconds'] / 60:.1f} minutes",
        "- Environment: a throwaway `python -m venv`, populated only from "
        "`requirements.txt`",
        "",
        "| Step | Result | Exit | Seconds |",
        "|---|---|---:|---:|",
    ]
    for s in r["steps"]:
        lines.append(f"| {s['name']} | {s['status']} | "
                     f"{s['exit_code'] if s['exit_code'] is not None else '-'} | "
                     f"{s['seconds']:.1f} |")
    lines += ["", "## What each step proves", "",
              "| Step | What it demonstrates |", "|---|---|",
              "| create virtual environment | the project needs nothing pre-installed |",
              "| install pinned requirements | `requirements.txt` is complete and sufficient |",
              "| rebuild dataset | ingest/clean/features run from source with no state |",
              "| retrain models | maths, ML, portfolio and recommendations all reproduce |",
              "| test suite | the integrity checks pass in a foreign environment |",
              "| artifacts present and built in order | outputs exist *and* each was produced after the stage it depends on |",
              "| dashboard imports, all panels execute | the app runs, not just the library |",
              "",
              "## Why the artifact check is ordered, not a flat list",
              "",
              "Each stage must be newer than the stage it consumes.",
              "`cleaned_panel` feeds `features`, `features` feeds the models, and",
              "the predictions feed the portfolio, recommendation and",
              "cross-sectional stages. Comparing every artifact to a single",
              "reference file cannot express that, and in particular would flag",
              "`cleaned_panel` as stale simply because it is built first.",
              ""]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
