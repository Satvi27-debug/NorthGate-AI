"""Build the dashboard data snapshot.

Why this exists
---------------
The dashboard is deployed to Streamlit Cloud, which runs the app as a fresh
clone. The full `data/processed/*.parquet` set is gitignored on the stated rule
that data is regenerated rather than committed - a good rule, and one that made
the deployed instance show "not built" on every tile.

Committing the full artifacts would work but costs 23 MB, and 19.5 MB of that is
`features.parquet`: 142 columns of which the dashboard reads four. Paying 19.5
MB to ship 138 unused columns is a bad trade for a repository anyone has to
clone.

So this writes two small files:

* `features_dashboard.parquet` - the four columns the panels actually read.
* `feature_manifest.json` - the true feature count and column list.

The manifest is not a convenience. The Overview panel's "signals per stock" tile
counts columns in whatever table it loaded, so with a slim file it would report
"4 signals per stock" instead of 132. The honest number is a property of the
model, not of whichever subset of the table happens to be on disk, so it is
recorded explicitly and the tile reads it from there.

`load_all` prefers the full table when present and falls back to the snapshot,
so local development and the deployed instance read the same code path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROCESSED = ROOT / "data" / "processed"
FEATURES = PROCESSED / "features.parquet"
SNAPSHOT = PROCESSED / "features_dashboard.parquet"
MANIFEST = PROCESSED / "feature_manifest.json"

# Exactly the columns the dashboard reads. Verified against dashboard/app.py
# rather than assumed - the first version of this list omitted Vol_21d and
# VIX_Level, which only panel_risk touches, so the Risk panel raised KeyError
# against the snapshot while the other seven rendered. Every column below is
# traceable to a panel:
#   Date, Ticker            Overview counts and filters; Price series
#   Adjusted Close          Price & Prediction history
#   Return_1d               market-mood gauge breadth
#   Vol_21d                 Risk: realised volatility, rockiest/steadiest
#   VIX_Level               Risk: the volatility-index overlay
DASHBOARD_COLUMNS = [
    "Date", "Ticker", "Return_1d", "Adjusted Close", "Vol_21d", "VIX_Level",
]

# Columns in the feature table that are bookkeeping, not signals. Mirrors the
# exclusion list in the Overview panel so the two agree by construction.
NON_SIGNAL_COLUMNS = {
    "Date", "Ticker", "Target", "Target_Price", "Relative_Target",
    "Rank_Target", "Open", "High", "Low", "Close", "Adjusted Close",
    "Volume", "Outlier", "Outlier_IQR",
}


def build() -> dict:
    import pandas as pd

    if not FEATURES.exists():
        raise FileNotFoundError(
            f"{FEATURES} not found. Run `python rebuild_dataset.py` first."
        )

    df = pd.read_parquet(FEATURES)
    signal_cols = [c for c in df.columns if c not in NON_SIGNAL_COLUMNS]

    missing = [c for c in DASHBOARD_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(
            f"the feature table is missing {missing}, which the dashboard reads. "
            f"Regenerate it; the snapshot cannot be built from a table the app "
            f"cannot use."
        )

    slim = df[DASHBOARD_COLUMNS].copy()
    slim["Date"] = pd.to_datetime(slim["Date"])
    slim.to_parquet(SNAPSHOT, index=False, compression="zstd")

    manifest = {
        "generated_by": "scripts/build_dashboard_snapshot.py",
        "full_table": str(FEATURES.relative_to(ROOT)).replace("\\", "/"),
        "full_table_size_mb": round(FEATURES.stat().st_size / 1e6, 2),
        "snapshot": str(SNAPSHOT.relative_to(ROOT)).replace("\\", "/"),
        "snapshot_size_mb": round(SNAPSHOT.stat().st_size / 1e6, 3),
        "n_rows": int(len(slim)),
        "n_signal_features": len(signal_cols),
        "signal_columns": signal_cols,
        "dashboard_columns": DASHBOARD_COLUMNS,
        "note": (
            f"n_signal_features={len(signal_cols)} is the number the models were "
            f"fitted on and is what the Overview panel must report. The snapshot "
            f"carries only {len(DASHBOARD_COLUMNS)} of those columns, so counting "
            f"the snapshot's own columns would understate the system by "
            f"{len(signal_cols) - len(DASHBOARD_COLUMNS)}."
        ),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"  snapshot : {SNAPSHOT.name}  "
          f"{manifest['snapshot_size_mb']:.2f} MB  "
          f"({manifest['n_rows']:,} rows x {len(DASHBOARD_COLUMNS)} cols)")
    print(f"  manifest : {MANIFEST.name}  "
          f"n_signal_features={manifest['n_signal_features']}")
    print(f"  saved    : {manifest['full_table_size_mb']:.2f} MB -> "
          f"{manifest['snapshot_size_mb']:.2f} MB  "
          f"({100 * (1 - manifest['snapshot_size_mb'] / manifest['full_table_size_mb']):.1f}% smaller)")
    return manifest


if __name__ == "__main__":
    build()
