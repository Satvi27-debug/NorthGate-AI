"""Snapshot every generated artifact so reruns can be diffed.

Writes reports/artifact_manifest.json: for each artifact, its size, mtime and a
row/column count. Two consecutive runs of the same pipeline should produce
matching shapes, which is the practical half of a reproducibility check
available without a fresh virtual environment.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import (  # noqa: E402
    DL_METRICS,
    FEATURES,
    MODEL_LEADERBOARD,
    ML_METRICS,
    PORTFOLIO_METRICS,
    PROCESSED_DIR,
    REPORTS_DIR,
)

MANIFEST = REPORTS_DIR / "artifact_manifest.json"

ARTIFACTS = [
    "data/raw/prices.parquet",
    "data/raw/macro.parquet",
    "data/processed/cleaned_panel.parquet",
    "data/processed/macro_daily.parquet",
    "data/processed/features.parquet",
    "data/processed/ml_metrics.csv",
    "data/processed/dl_metrics.csv",
    "data/processed/model_leaderboard.csv",
    "data/processed/optimal_weights.csv",
    "data/processed/min_variance_weights.csv",
    "data/processed/efficient_frontier.csv",
    "data/processed/portfolio_backtest.csv",
    "data/processed/portfolio_metrics.json",
    "data/processed/recommendations.csv",
    "data/processed/rebalance_plan.csv",
    "data/processed/ml_predictions.parquet",
    "data/processed/dl_predictions.parquet",
    "data/processed/dl_loss_curves.csv",
    "reports/data_quality_report.json",
    "reports/data_quality_report.md",
    "reports/math_verification.json",
    "reports/eda_findings.json",
    "reports/final_report.md",
    "notebooks/eda.ipynb",
]


def describe(rel: str) -> dict:
    p = ROOT / rel
    if not p.exists():
        return {"present": False}
    stat = p.stat()
    info = {
        "present": True,
        "bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
        info["rows"], info["cols"] = int(len(df)), int(df.shape[1])
    elif p.suffix == ".csv":
        df = pd.read_csv(p)
        info["rows"], info["cols"] = int(len(df)), int(df.shape[1])
    elif p.suffix == ".ipynb":
        nb = json.loads(p.read_text(encoding="utf-8"))
        info["cells"] = len(nb.get("cells", []))
    return info


def main() -> int:
    snap = {rel: describe(rel) for rel in ARTIFACTS}
    present = sum(1 for v in snap.values() if v["present"])
    print(f"{present}/{len(ARTIFACTS)} artifacts present\n")

    hdr = f"{'artifact':<48}{'bytes':>12}{'rows':>10}{'cols':>8}"
    print(hdr)
    print("-" * len(hdr))
    for rel, v in snap.items():
        if not v["present"]:
            print(f"{rel:<48}{'MISSING':>12}")
            continue
        rows = f"{v['rows']:,}" if "rows" in v else "-"
        cols = str(v.get("cols", "-")) if "rows" in v else "-"
        print(f"{rel:<48}{v['bytes']:>12,}{rows:>10}{cols:>8}")

    payload = {
        "generated_by": "scripts/snapshot_artifacts.py",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_count": len(ARTIFACTS),
        "present_count": present,
        "artifacts": snap,
    }
    MANIFEST.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nmanifest -> {MANIFEST}")
    return 0 if present == len(ARTIFACTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
