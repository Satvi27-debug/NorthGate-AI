"""Shared infrastructure: paths, configuration, determinism and logging.

Every module in the pipeline resolves its paths through this file so that a
script behaves identically whether it is launched from the repository root, from
``src/``, or as a subprocess of ``rebuild_dataset.py`` (PRD Section 15.3:
"config-driven so no magic numbers are buried in code").
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_PRICES = RAW_DIR / "prices.parquet"
RAW_MACRO = RAW_DIR / "macro.parquet"
CLEANED_PANEL = PROCESSED_DIR / "cleaned_panel.parquet"
FEATURES = PROCESSED_DIR / "features.parquet"
# The four columns the dashboard reads, committed so a fresh clone - which is
# what Streamlit Cloud runs - can still render. 0.4 MB against 19.5 MB for the
# full table. Built by scripts/build_dashboard_snapshot.py.
DASHBOARD_SNAPSHOT = PROCESSED_DIR / "features_dashboard.parquet"
# The true count of signal features. The Overview panel reads this rather than
# counting the loaded table's columns, which would report 4 for a system built
# on 135 whenever the snapshot is in use.
FEATURE_MANIFEST = PROCESSED_DIR / "feature_manifest.json"
SENTIMENT_FEATURES = PROCESSED_DIR / "sentiment_features.parquet"
MARKET_MOOD = PROCESSED_DIR / "market_mood.parquet"
DQ_REPORT_JSON = REPORTS_DIR / "data_quality_report.json"
DQ_REPORT_MD = REPORTS_DIR / "data_quality_report.md"

# Derived outputs consumed by the dashboard (PRD Section 13: the dashboard
# reads pre-computed artifacts and never retrains on load).
ML_METRICS = PROCESSED_DIR / "ml_metrics.csv"
DL_METRICS = PROCESSED_DIR / "dl_metrics.csv"
MODEL_LEADERBOARD = PROCESSED_DIR / "model_leaderboard.csv"
ML_PREDICTIONS = PROCESSED_DIR / "ml_predictions.parquet"
DL_PREDICTIONS = PROCESSED_DIR / "dl_predictions.parquet"
PORTFOLIO_WEIGHTS = PROCESSED_DIR / "optimal_weights.csv"
PORTFOLIO_MIN_VARIANCE = PROCESSED_DIR / "min_variance_weights.csv"
PORTFOLIO_FRONTIER = PROCESSED_DIR / "efficient_frontier.csv"
PORTFOLIO_BACKTEST = PROCESSED_DIR / "portfolio_backtest.csv"
PORTFOLIO_METRICS = PROCESSED_DIR / "portfolio_metrics.json"
MONTE_CLOUD_PATH = PROCESSED_DIR / "monte_carlo_cloud.npy"
RECOMMENDATIONS = PROCESSED_DIR / "recommendations.csv"
REBALANCE_PLAN = PROCESSED_DIR / "rebalance_plan.csv"
SENTIMENT_ABLATION = PROCESSED_DIR / "sentiment_ablation.json"
CROSS_SECTIONAL = PROCESSED_DIR / "cross_sectional_results.json"
DECLARED_MODEL = PROCESSED_DIR / "declared_model.json"

FEATURE_SCALER = MODELS_DIR / "feature_scaler.pkl"
TARGET_SCALER = MODELS_DIR / "target_scaler.pkl"

ALL_TICKERS_KEY = "__all__"


def ensure_dirs() -> None:
    """Create every directory the pipeline writes into."""
    for d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load ``config.yaml``. This is the only place configuration is read."""
    cfg_path = Path(path) if path else CONFIG_PATH
    with cfg_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_end_date(cfg: dict[str, Any]) -> str:
    """``end_date: null`` resolves to today (PRD Listing 3.1)."""
    import datetime as dt

    return cfg.get("end_date") or dt.date.today().isoformat()


def all_tickers(cfg: dict[str, Any]) -> list[str]:
    """Universe + benchmark + volatility index, de-duplicated, order preserved."""
    seen: list[str] = []
    for t in [*cfg["universe"], cfg["benchmark"], cfg["volatility_index"]]:
        if t not in seen:
            seen.append(t)
    return seen


# --------------------------------------------------------------------------
# Determinism (PRD Section 9.4 / 15.3)
# --------------------------------------------------------------------------
def set_seed(seed: int | None = None) -> int:
    """Seed every source of randomness the pipeline can reach.

    Returns the seed actually applied so callers can log it.

    TensorFlow is handled with care. It is seeded **only if it has already been
    imported** (``"tensorflow" in sys.modules``); this function never imports it.
    Two reasons:

    1. Importing TensorFlow here would make every module pay a multi-second
       import cost for a library most of them do not use.
    2. On this environment, importing pyarrow (which pandas loads for parquet)
       *before* TensorFlow aborts with a DLL initialisation failure inside
       ``_pywrap_tensorflow_internal``. Deferring the TensorFlow import to
       ``src/models_dl.py``, which imports TensorFlow first thing, keeps that
       ordering under our control.
    """
    if seed is None:
        seed = int(load_config().get("random_seed", 42))
    seed = int(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # TensorFlow reads these at import time, so set them regardless of whether
    # TensorFlow is loaded yet.
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    if "tensorflow" in sys.modules:
        tf = sys.modules["tensorflow"]
        try:
            tf.random.set_seed(seed)
            tf.keras.utils.set_random_seed(seed)
        except Exception:  # pragma: no cover
            pass
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:  # pragma: no cover
            pass

    return seed


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s"


def get_logger(name: str, logfile: str | Path | None = None) -> logging.Logger:
    """Console logger, optionally tee'd to a file.

    The cleaning stage writes a full audit trail so that "every dropped or
    altered row is recoverable from logs" (PRD Section 4.2).
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False

    fmt = logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S")

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    if logfile:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile, mode="w", encoding="utf-8")
        fh.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(fh)

    return logger


# --------------------------------------------------------------------------
# Small IO helpers
# --------------------------------------------------------------------------
def write_json(obj: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        if isinstance(o, (pd.Timestamp,)):  # noqa: F821
            return str(o)
        return str(o)

    path.write_text(json.dumps(obj, indent=2, default=default), encoding="utf-8")
    return path


def banner(logger: logging.Logger, title: str, width: int = 78) -> None:
    logger.info("=" * width)
    logger.info(title)
    logger.info("=" * width)


def fmt_pct(x: float, digits: int = 2) -> str:
    return f"{x * 100:.{digits}f}%"


def fmt_num(x: float, digits: int = 4) -> str:
    return f"{x:.{digits}f}"


import pandas as pd  # noqa: E402  (imported late so `default` above can reference it)
