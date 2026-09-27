"""One-command dataset rebuild (PRD Section 3.4 reproducibility row).

Runs the pipeline stages in dependency order as **separate subprocesses**, so a
crash or a hard interpreter fault in one stage cannot take the others with it and
each stage's exit code is visible. A stage that fails is reported and the build
is marked FAILED rather than silently continuing.

Usage
-----
    python rebuild_dataset.py            # ingest -> clean -> features
    python rebuild_dataset.py --with-sentiment
    python rebuild_dataset.py --skip-ingest   # reuse the existing raw parquet
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.common import get_logger, load_config  # noqa: E402

LOG = get_logger("rebuild")

STAGES = [
    ("ingest", "src/ingest.py", "Download raw market and macro data (Section 3)"),
    ("clean", "src/clean.py", "Five-stage cleaning pipeline (Section 4)"),
    ("features", "src/features.py", "Feature engineering and target (Section 5)"),
]
OPTIONAL = [
    ("sentiment", "src/sentiment.py", "News sentiment features (Section 10)"),
]


def run_stage(name: str, script: str, description: str, allow_fail: bool = False) -> bool:
    path = ROOT / script
    if not path.exists():
        LOG.error("  script missing: %s", script)
        return False

    LOG.info("")
    LOG.info("-" * 78)
    LOG.info(">> %s - %s", name, description)
    LOG.info("-" * 78)
    t0 = time.time()
    proc = subprocess.run([sys.executable, script], cwd=str(ROOT),
                          capture_output=True, text=True)
    elapsed = time.time() - t0

    for line in (proc.stdout or "").splitlines():
        if line.strip() and "TensorFlow DLL" not in line and "_pywrap" not in line \
                and "Hint: This often" not in line and "Visual C++" not in line \
                and "or if the Micro" not in line and "Failed to load" not in line:
            LOG.info("  %s", line)

    if proc.returncode != 0:
        LOG.error("  FAILED (exit %d) after %.1fs", proc.returncode, elapsed)
        for line in (proc.stderr or "").splitlines()[-25:]:
            if line.strip():
                LOG.error("  | %s", line)
        if not allow_fail:
            LOG.error("  aborting the rebuild")
        return False

    LOG.info("  OK in %.1fs", elapsed)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild the dataset from scratch.")
    parser.add_argument("--with-sentiment", action="store_true",
                        help="also run the sentiment stage (needs FINNHUB_API_KEY)")
    parser.add_argument("--skip-ingest", action="store_true",
                        help="reuse the existing raw parquet instead of re-downloading")
    args = parser.parse_args()

    cfg = load_config()
    LOG.info("=" * 78)
    LOG.info("Northgate AI Stock Predictor - dataset rebuild")
    LOG.info("=" * 78)
    LOG.info("Universe    : %s", ", ".join(cfg["universe"]))
    LOG.info("Benchmark   : %s   Volatility index: %s", cfg["benchmark"], cfg["volatility_index"])
    LOG.info("FRED series : %s", ", ".join(cfg["fred_series"]))
    LOG.info("Date range  : %s -> %s", cfg["start_date"],
             cfg.get("end_date") or "today (resolved at runtime)")
    LOG.info("Target      : %d-session forward %s of %s",
             cfg["target"]["horizon"], cfg["target"]["kind"], cfg["price_column"])
    LOG.info("Seed        : %s", cfg["random_seed"])

    stages = list(STAGES)
    if args.skip_ingest:
        stages = [s for s in stages if s[0] != "ingest"]
        LOG.info("Skipping ingestion; reusing the existing raw data.")
    if args.with_sentiment:
        stages += OPTIONAL

    t0 = time.time()
    failed: list[str] = []
    for name, script, description in stages:
        allow_fail = name in ("sentiment",)
        if not run_stage(name, script, description, allow_fail=allow_fail):
            failed.append(name)

    total = time.time() - t0
    LOG.info("")
    LOG.info("=" * 78)
    if failed:
        hard = [f for f in failed if f != "sentiment"]
        LOG.info("REBUILD FAILED after %.1fs. Failed stages: %s", total, ", ".join(failed))
        if hard:
            LOG.info("A hard failure means downstream artifacts are stale. Fix the stage above.")
        return 1
    LOG.info("REBUILD COMPLETE in %.1fs", total)
    LOG.info("")
    LOG.info("Next: python retrain_models.py     # train + evaluate all eight models")
    LOG.info("      python src/portfolio.py      # efficient frontier and max-Sharpe weights")
    LOG.info("      python src/recommend.py      # BUY / HOLD / SELL and rebalancing")
    LOG.info("      streamlit run dashboard/app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
