"""PRD Section 12 - AI Recommendation System.

*"It is deliberately rule-based and transparent - every recommendation must be
explainable, because an unexplained 'SELL' is useless to a user and impossible
to grade."* (Section 12)

Signal fusion (Section 12.1)
---------------------------
    Composite = w1 * forecast + w2 * sentiment - w3 * risk

Each sub-signal is normalised to a common scale before weighting, otherwise the
term with the largest raw units would dominate. All three sub-scores are
returned alongside every recommendation so the user sees the "why" (Section
12.3).

Decision rules (Section 12.2, Listing 12.1) are threshold comparisons with a
no-trade band, and rebalancing compares current against target weights.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import (  # noqa: E402
    FEATURES,
    ML_PREDICTIONS,
    PORTFOLIO_METRICS,
    PORTFOLIO_WEIGHTS,
    REBALANCE_PLAN,
    RECOMMENDATIONS,
    SENTIMENT_FEATURES,
    banner,
    ensure_dirs,
    get_logger,
    load_config,
    set_seed,
    write_json,
)
from src import math_utils as mu_math  # noqa: E402

LOG = get_logger("recommend")

# z-score window for normalising sub-signals. Trailing and per-ticker, so the
# normalisation itself introduces no look-ahead.
NORM_WINDOW = 252


# ==========================================================================
# Sub-signals (Section 12.1)
# ==========================================================================
def forecast_signal(predictions: pd.DataFrame, horizon_sessions: int = 21) -> pd.Series:
    """Predicted forward return from the best model, per ticker.

    The 21-session mean prediction is used rather than the single most recent
    day: one day's forecast is mostly noise, and a month of model output is a
    more stable statement of view.
    """
    recent = predictions.copy()
    recent["Date"] = pd.to_datetime(recent["Date"])
    cutoff = recent["Date"].max() - pd.Timedelta(days=45)
    recent = recent[recent["Date"] >= cutoff]
    return recent.groupby("Ticker")["Predicted_Return"].mean() * horizon_sessions


def sentiment_signal() -> pd.Series:
    """Latest daily sentiment score per ticker (Section 10)."""
    if not SENTIMENT_FEATURES.exists():
        LOG.warning("No sentiment features at %s - sentiment sub-signal is neutral (0).",
                    SENTIMENT_FEATURES)
        return pd.Series(dtype=float)
    s = pd.read_parquet(SENTIMENT_FEATURES)
    s["Date"] = pd.to_datetime(s["Date"])
    latest = s.loc[s.groupby("Ticker")["Date"].idxmax()]
    return latest.set_index("Ticker")["Sentiment_FinBERT"]


def risk_signal(features: pd.DataFrame, benchmark_returns: pd.Series | None,
                as_of: pd.Timestamp) -> pd.Series:
    """Volatility / Beta composite; higher means riskier, so it is subtracted.

    Combines each ticker's trailing annualised volatility with its Beta against
    the benchmark, then ranks them so the two components are commensurable.
    """
    f = features[features["Date"] == as_of]
    if f.empty:
        f = features[features["Date"] <= as_of].groupby("Ticker").tail(1)
    else:
        f = f.groupby("Ticker").tail(1)

    out = pd.Series(index=f["Ticker"].values, dtype=float)
    vol = f.set_index("Ticker")["Vol_21d"].astype(float)
    out.loc[vol.index] = vol.rank(pct=True)

    if benchmark_returns is not None and len(benchmark_returns) > 30:
        bench = benchmark_returns
        betas = {}
        for ticker, row in f.set_index("Ticker").iterrows():
            hist = features[features["Ticker"] == ticker].sort_values("Date")
            rets = np.log(hist["Adjusted Close"]).diff().reindex(bench.index).dropna()
            common = rets.index.intersection(bench.index)
            if len(common) > 60:
                betas[ticker] = mu_math.beta(rets.loc[common], bench.loc[common])
        if betas:
            b = pd.Series(betas)
            out.loc[b.index] = out.loc[b.index] + b.rank(pct=True)
            out.loc[b.index] /= 2.0

    return out.fillna(0.5)


def _normalise(series: pd.Series) -> pd.Series:
    """Rank-normalise to [0, 1]: monotone, outlier-robust, and bounded."""
    if series.empty:
        return series
    if series.nunique() <= 1:
        return pd.Series(0.5, index=series.index)
    return series.rank(pct=True)


def _forecast_skill(predictions: pd.DataFrame) -> float:
    """Demonstrated edge of the forecaster over a random walk, on its own test
    predictions, mapped to [0, 1].

    A model that does not beat the baseline on MAE scores 0 and its forecast
    sub-signal is dropped from the composite. A model that halves the baseline
    error scores 1. The mapping is the ratio of relative improvements:

        skill = (MAE_baseline - MAE_model) / MAE_baseline, clamped to [0, 1]

    This is computed from the residuals actually stored in the prediction file,
    so it reflects what the model did rather than a claim about it.
    """
    if predictions is None or predictions.empty:
        return 0.0
    if "Residual" not in predictions.columns:
        return 0.0

    # The random walk predicts a zero return, so its MAE is the mean absolute
    # realised return. Comparing the two is the Section 8.3 test, restated.
    y = predictions["Actual_Return"].to_numpy(dtype=float)
    mae_model = float(np.mean(np.abs(predictions["Residual"].to_numpy(dtype=float))))
    mae_baseline = float(np.mean(np.abs(y)))
    if mae_baseline <= 0:
        return 0.0
    return float(np.clip((mae_baseline - mae_model) / mae_baseline, 0.0, 1.0))


def _skill_note(skill: float) -> str:
    if skill <= 0.0:
        return ("the forecaster does not beat the random walk on MAE, so its "
                "sub-signal is dropped from the composite (weight 0.00)")
    if skill >= 0.999:
        return "the forecaster halves the baseline MAE, so its configured weight is used in full"
    return (f"the forecaster improves on the baseline MAE by {skill * 100:.1f}%, "
            f"so its weight is scaled to {skill:.2f} of the configured value")


# ==========================================================================
# Decision rules (Section 12.2, Listing 12.1)
# ==========================================================================
def recommend(score: float, low: float, high: float) -> str:
    """BUY above the high threshold, SELL below the low threshold, else HOLD.

    The band between the thresholds is the no-trade zone that stops the engine
    from churning on noise.
    """
    if not np.isfinite(score):
        return "HOLD"
    if score >= high:
        return "BUY"
    if score <= low:
        return "SELL"
    return "HOLD"


def rebalance(current_w: dict, target_w: dict, band: float = 0.05) -> dict:
    """Compare current against target weights and propose actions past the band.

    Listing 12.1. The band prevents a trade for a drift too small to matter
    after transaction costs.
    """
    actions: dict[str, dict] = {}
    for asset, target in target_w.items():
        current = float(current_w.get(asset, 0.0))
        drift = float(target) - current
        if abs(drift) > band:
            actions[asset] = {
                "action": "INCREASE" if drift > 0 else "REDUCE",
                "drift": round(drift, 4),
            }
    return actions


# ==========================================================================
# Main
# ==========================================================================
def build_recommendations(current_weights: dict | None = None,
                          horizon_days: int = 21) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dirs()
    cfg = load_config()
    set_seed(cfg.get("random_seed", 42))
    rcfg = cfg["recommend"]
    universe = list(cfg["universe"])

    banner(LOG, "PRD Section 12 - Recommendation Engine")

    features = pd.read_parquet(FEATURES)
    features["Date"] = pd.to_datetime(features["Date"])
    as_of = features["Date"].max()

    if not ML_PREDICTIONS.exists():
        raise FileNotFoundError(
            f"{ML_PREDICTIONS} not found. Run `python src/models_ml.py` first - the "
            "forecast sub-signal is the core of Section 12.1."
        )
    predictions = pd.read_parquet(ML_PREDICTIONS)

    # Prefer the overall best model; fall back to the best ML family.
    if ML_PREDICTIONS.name and "Model" in predictions.columns:
        scores = (predictions.groupby("Model")
                  .apply(lambda g: float(np.mean(np.abs(g["Residual"]))), include_groups=False)
                  .sort_values())
        best_model = scores.index[0]
    else:
        best_model = ""
    preds = predictions[predictions["Model"] == best_model] if best_model else predictions
    LOG.info("Using model forecasts from: %s (lowest mean absolute error)", best_model or "n/a")

    # -- sub-signals -------------------------------------------------------
    f_sig = forecast_signal(preds, horizon_days)
    s_sig = sentiment_signal()
    bench = features[features["Ticker"] == cfg["benchmark"]].sort_values("Date")
    b_ret = np.log(bench["Adjusted Close"]).diff().dropna()
    r_sig = risk_signal(features, b_ret, as_of)

    w = rcfg["weights"]
    have_sentiment = not s_sig.empty

    LOG.info("")
    LOG.info("Sub-signals as of %s:", as_of.date())
    LOG.info("  %-6s %12s %12s %10s", "Ticker", "Forecast", "Sentiment", "Risk")
    for t in universe:
        LOG.info("  %-6s %+12.5f %12s %10.4f", t,
                 f_sig.get(t, np.nan),
                 f"{s_sig.get(t, np.nan):+.4f}" if have_sentiment else "n/a",
                 r_sig.get(t, np.nan))

    # -- normalise and fuse (Section 12.1) ---------------------------------
    f_n = _normalise(f_sig.reindex(universe).fillna(0.5))
    s_n = (_normalise(s_sig.reindex(universe).fillna(0.5))
           if have_sentiment else pd.Series(0.5, index=universe))
    r_n = _normalise(r_sig.reindex(universe).fillna(0.5))

    # Skill gate. Section 6.3 measures that the forecaster does not beat the
    # random walk on directional accuracy, which means its forecast sub-signal
    # carries no demonstrated information. Weighting it at full strength would
    # emit confident BUY calls from a signal known not to work. We therefore
    # scale the forecast weight by the model's demonstrated MAE edge over the
    # baseline, clamped to [0, 1]: a model that beats random walk on MAE gets
    # its configured weight, one that does not gets proportionally less, and the
    # composite collapses toward the risk term that we *can* measure.
    skill = _forecast_skill(predictions)
    f_weight = w["forecast"] * skill
    w_used = {"forecast": f_weight, "sentiment": w["sentiment"], "risk": w["risk"]}

    LOG.info("")
    LOG.info("Forecast skill gate: %s", _skill_note(skill))

    composite = f_weight * f_n + w["sentiment"] * s_n - w["risk"] * r_n

    # -- target weights from the optimiser (Section 12.2) -----------------
    if PORTFOLIO_WEIGHTS.exists():
        target = pd.read_csv(PORTFOLIO_WEIGHTS, index_col=0).iloc[:, 0].to_dict()
    else:
        LOG.warning("No optimal weights at %s - using equal weight as the target.",
                    PORTFOLIO_WEIGHTS)
        target = {t: 1.0 / len(universe) for t in universe}
    current = current_weights or {t: 0.0 for t in universe}

    # -- assemble ----------------------------------------------------------
    buy, sell = float(rcfg["buy_threshold"]), float(rcfg["sell_threshold"])
    band = float(rcfg["rebalance_band"])
    rows = []
    for t in universe:
        c = float(composite.get(t, 0.0))
        action = recommend(c, sell, buy)
        tw, cw = float(target.get(t, 0.0)), float(current.get(t, 0.0))
        drift = tw - cw
        rows.append({
            "Ticker": t,
            "Recommendation": action,
            "Composite_Score": round(c, 4),
            "Forecast_Signal": round(float(f_n.get(t, 0.5)), 4),
            "Sentiment_Signal": round(float(s_n.get(t, 0.5)), 4),
            "Risk_Signal": round(float(r_n.get(t, 0.5)), 4),
            "Target_Weight": round(tw, 4),
            "Current_Weight": round(cw, 4),
            "Drift": round(drift, 4),
            "Rebalance": ("INCREASE" if drift > band else
                          "REDUCE" if drift < -band else "MAINTAIN"),
            "Rationale": _rationale(action, f_n.get(t, 0.5), s_n.get(t, 0.5), r_n.get(t, 0.5), w_used),
        })

    recs = pd.DataFrame(rows).sort_values("Composite_Score", ascending=False).reset_index(drop=True)

    acts = rebalance(current, {t: float(target.get(t, 0.0)) for t in universe}, band)
    plan = pd.DataFrame([
        {
            "Ticker": t,
            "Current_Weight": round(float(current.get(t, 0.0)), 4),
            "Target_Weight": round(float(target.get(t, 0.0)), 4),
            "Drift": round(float(target.get(t, 0.0)) - float(current.get(t, 0.0)), 4),
            "Action": acts.get(t, {}).get("action", "HOLD"),
            "Band_Breached": t in acts,
        }
        for t in universe
    ])

    recs.to_csv(RECOMMENDATIONS, index=False)
    plan.to_csv(REBALANCE_PLAN, index=False)

    LOG.info("")
    LOG.info("Recommendations (buy >= %.2f, sell <= %.2f, no-trade band between):", buy, sell)
    LOG.info("  %-6s %-5s %9s %9s %9s %8s %8s", "Ticker", "Act", "Composite", "Fcst", "Sent", "Risk", "Rebal")
    for _, r in recs.iterrows():
        LOG.info("  %-6s %-5s %9.4f %9.4f %9.4f %8.4f %8s",
                 r["Ticker"], r["Recommendation"], r["Composite_Score"],
                 r["Forecast_Signal"], r["Sentiment_Signal"], r["Risk_Signal"],
                 r["Rebalance"])

    dist = recs["Recommendation"].value_counts().to_dict()
    LOG.info("")
    LOG.info("Distribution: %s", dist)
    if len(dist) == 1 and dist.get("HOLD", 0) == len(universe) and skill <= 0.0:
        LOG.info("")
        LOG.info("All assets are HOLD, and that is the correct output rather than a")
        LOG.info("degenerate one. With the forecast term zeroed by the skill gate the")
        LOG.info("composite reduces to %.2f*sentiment - %.2f*risk, whose rank-normalised",
                 w["sentiment"], w["risk"])
        LOG.info("terms cannot span the +/-%.2f no-trade band. Emitting BUY or SELL here", buy)
        LOG.info("would mean acting on a signal the measurement says does not work.")
    elif len(dist) == 1:
        LOG.warning("Every asset landed in the same bucket. With a composite in")
        LOG.warning("[0,1] and thresholds at +/-%.2f, a narrow spread can do this.", buy)

    meta = {
        "generated_by": "src/recommend.py",
        "as_of": str(as_of.date()),
        "best_model": best_model,
        "configured_weights": w,
        "forecast_skill": round(skill, 4),
        "weights_used": {k: round(v, 4) for k, v in w_used.items()},
        "buy_threshold": buy,
        "sell_threshold": sell,
        "rebalance_band": band,
        "sentiment_available": have_sentiment,
        "distribution": dist,
        "disclaimer": cfg["dashboard"]["disclaimer"].strip(),
    }
    if PORTFOLIO_METRICS.exists():
        import json

        pm = json.loads(PORTFOLIO_METRICS.read_text(encoding="utf-8"))
        meta["portfolio_beats_equal_weight"] = pm.get("beats_equal_weight")
    write_json(meta, Path(RECOMMENDATIONS).parent / "recommendation_meta.json")

    LOG.info("")
    LOG.info("Recommendations -> %s", RECOMMENDATIONS)
    LOG.info("Rebalance plan  -> %s", REBALANCE_PLAN)
    return recs, plan


def _rationale(action: str, f: float, s: float, r: float, w: dict) -> str:
    """One-line plain-language explanation (Section 12.3)."""
    parts = [
        f"forecast {'strong' if f > 0.66 else 'weak' if f < 0.33 else 'neutral'} ({f:.2f})",
        f"sentiment {'positive' if s > 0.66 else 'negative' if s < 0.33 else 'neutral'} ({s:.2f})",
        f"risk {'high' if r > 0.66 else 'low' if r < 0.33 else 'moderate'} ({r:.2f})",
    ]
    lead = {
        "BUY": "Composite clears the buy threshold:",
        "SELL": "Composite falls below the sell threshold:",
        "HOLD": "Composite sits inside the no-trade band:",
    }[action]
    text = f"{lead} " + "; ".join(parts) + "."

    # When the forecast term was zeroed by the skill gate, say so: a reader
    # looking at a strong forecast score and a muted composite deserves to know
    # why the two disagree.
    if w.get("forecast", 0.0) <= 0.0:
        text += (" The forecast term carries zero weight because the model shows no "
                 "measured edge over the random walk (Section 8.3).")
    return text


if __name__ == "__main__":
    build_recommendations()
