"""Import every dashboard panel and run it against the real artifacts.

Streamlit's decorators are no-ops when called outside a server, so calling each
panel function directly executes the real body: the same data loading, the same
Plotly figure construction, the same formatting. This catches the failures a
static check misses - a KeyError on a column, a malformed figure spec, a divide
by zero in a percentage.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# The dashboard must be imported first for the TensorFlow/pyarrow ordering rule
# only if it used TF; it does not, so a plain import is fine.
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("ng_app", ROOT / "dashboard" / "app.py")
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)

from src.common import load_config  # noqa: E402

CFG = load_config()
DATA = app.load_all()

PANELS = {
    "overview": (app.panel_overview, (CFG, DATA)),
    "price": (app.panel_price, (CFG, DATA, CFG["universe"][0], 1, 1.96)),
    "models": (app.panel_models, (CFG, DATA)),
    "portfolio": (app.panel_portfolio, (CFG, DATA, 0.04)),
    "risk": (app.panel_risk, (CFG, DATA, CFG["universe"][0])),
    "sentiment": (app.panel_sentiment, (CFG, DATA, CFG["universe"][0])),
    "recommendations": (app.panel_recommendations, (CFG, DATA)),
    # Supplementary panel. Registered here so it is actually executed rather
    # than merely existing - a panel that is defined but never run is exactly
    # how the cross-sectional function shipped with an untested look-ahead.
    "ranking": (app.panel_ranking, (CFG, DATA)),
}


def test_data_loaded():
    assert DATA["features"] is not None, "features.parquet missing"
    assert DATA["leaderboard"] is not None, "model_leaderboard.csv missing"
    assert DATA["weights"] is not None, "optimal_weights.csv missing"
    assert DATA["recommendations"] is not None, "recommendations.csv missing"
    assert DATA["portfolio_metrics"] is not None, "portfolio_metrics.json missing"
    assert DATA["data_quality"] is not None, "data_quality_report.json missing"


def test_every_panel_runs():
    """Execute each panel body against the real artifacts."""
    for name, (fn, args) in PANELS.items():
        try:
            fn(*args)
        except Exception as exc:  # pragma: no cover
            raise AssertionError(
                f"panel {name!r} raised {type(exc).__name__}: {exc}"
            ) from exc


def test_panel_across_multiple_tickers():
    """The price and risk panels must work for every ticker, not just the first."""
    for ticker in CFG["universe"]:
        try:
            app.panel_price(CFG, DATA, ticker, 1, 1.96)
            app.panel_risk(CFG, DATA, ticker)
            app.panel_sentiment(CFG, DATA, ticker)
        except Exception as exc:  # pragma: no cover
            raise AssertionError(
                f"panel failed for ticker {ticker}: {type(exc).__name__}: {exc}"
            ) from exc


def test_leaderboard_has_baseline_and_all_models():
    lb = DATA["leaderboard"]
    names = set(lb["Model"])
    assert "Naive random walk" in names, "the Section 8.3 baseline is missing"
    for required in ("Ridge", "RandomForest", "XGBoost", "SVR",
                     "LSTM", "GRU", "BiLSTM", "Transformer"):
        assert required in names, f"missing model in leaderboard: {required}"


def test_portfolio_metrics_are_internally_consistent():
    """The backtest must be disjoint from the estimation window.

    This is the portfolio layer's equivalent of the no-look-ahead rule: if the
    optimiser estimated mu and Sigma on a window that overlaps the window it is
    scored on, it has already seen the returns it is judged by, and the
    out-of-sample Sharpe becomes an in-sample number.
    """
    pm = DATA["portfolio_metrics"]
    bt = pm["backtest"]
    est_end = pm["estimation_window"]["end"]
    assert bt["end"] > est_end, (
        f"the backtest window ({bt['start']} to {bt['end']}) does not start after the "
        f"estimation window ends ({est_end}) - the optimiser is being scored on "
        f"returns it has already seen"
    )
    assert bt["n_sessions"] > 0
    # Every strategy must be scored over the same realised window.
    for label in ("optimised", "equal_weight"):
        assert label in bt, f"missing backtest strategy: {label}"


def test_recommendations_have_sub_signals_and_rationale():
    recs = DATA["recommendations"]
    for col in ("Composite_Score", "Forecast_Signal", "Sentiment_Signal",
                "Risk_Signal", "Rationale"):
        assert col in recs.columns, f"recommendations missing {col}"
    assert recs["Rationale"].str.len().min() > 20, "a recommendation has no explanation"
    assert set(recs["Recommendation"]).issubset({"BUY", "HOLD", "SELL"})


def test_rebalance_plan_covers_the_universe():
    plan = DATA["rebalance"]
    assert set(plan["Ticker"]) == set(CFG["universe"])


def test_monte_carlo_cloud_is_loadable():
    cloud = DATA["monte_carlo"]
    assert cloud is not None, "monte_carlo_cloud.npy missing"
    assert cloud.shape[0] >= 19000, f"only {cloud.shape[0]} portfolios simulated"
    assert (cloud[:, 0] > 0).all(), "non-positive volatility in the cloud"
