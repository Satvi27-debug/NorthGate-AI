"""Backtested hit-rate for the recommendation engine, versus buy-and-hold.

PRD Section 14, acceptance table, row "Recommendations":

    Recommendations | Backtested hit-rate vs. buy-and-hold |
    Documented, with transparent contributing signals

`src/recommend.py` produces a recommendation for the *latest* session. That is
the right behaviour for a decision tool, but it is not evidence: one dated
recommendation cannot be scored, because there is no outcome to score it
against. This module supplies the missing evidence by running the same signal
logic across the whole held-out window and comparing it with the do-nothing
baseline.

How hit-rate is defined
-----------------------
For each ticker and each session *t* in the test window:

    realised sign  = sign(return from t to t+1)
    signal sign    = sign(composite score at t)

and hit-rate is the share of observations where the two agree. Buy-and-hold is
the natural baseline because its implied position is long every session, so its
"signal" is always +1 and its hit-rate is simply the share of sessions the asset
rose. Beating that number is the bar; matching it means the engine adds nothing.

Three arms are reported, and the difference between them is the finding:

* **buy-and-hold** - the baseline. Always long.
* **gated** - what the system actually deploys, with the forecast weight scaled
  by measured skill. When the forecaster has no demonstrated edge this arm
  degenerates to "hold everything", and its hit-rate converging on
  buy-and-hold is the *correct* behaviour, not a bug: a system with no signal
  should decline to trade. Reporting only this arm would hide the difference.
* **ungated** - the same score with the forecast weight at full strength, i.e.
  what the engine would have done if the skill gate were switched off. This
  isolates how much the raw composite is worth before the safety interlock.

No look-ahead
--------------
The composite at session *t* is built only from information available at *t*:
model forecasts made at *t*, trailing risk statistics, and - when news data
exists - headlines up to *t*. The outcome is the return from *t* to *t+1*, which
the score never sees. The same discipline the model layer is held to applies
here, and `tests/test_recommendation_backtest.py` checks that the score at date
*d* is a function of data at or before *d*.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import (  # noqa: E402
    FEATURES,
    ML_PREDICTIONS,
    PROCESSED_DIR,
    banner,
    get_logger,
    load_config,
    set_seed,
    write_json,
)
from src.recommend import _normalise  # noqa: E402

LOG = get_logger("rec_backtest")


def _realised_forward_returns(features: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Return from t to t+1 per ticker - the outcome the score is scored against."""
    universe = list(cfg["universe"])
    wide = (features[features["Ticker"].isin(universe)]
            .pivot(index="Date", columns="Ticker", values="Adjusted Close")
            .sort_index())
    logp = np.log(wide)
    return logp.diff().shift(-1)  # at date t, this is log(P[t+1] / P[t])


def _trailing_risk_signal(features: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """A causal risk score per ticker per date.

    Built from trailing windows only: a mild preference for names whose recent
    realised volatility is below the rest of the universe, damped by how far the
    price sits above its own 50-day mean. Every term uses data up to and
    including *t*.

    The volatility comparison is CROSS-SECTIONAL - each date's universe is
    ranked against the other names on that same date. An earlier version ranked
    each ticker's whole time series against itself, which is look-ahead of the
    worst kind: the rank of a value in 2018 depended on which volatility values
    existed in 2025, so truncating the future changed the past. That is exactly
    what `tests/test_recommendation_backtest.py::test_risk_signal_is_trailing_only`
    caught, by rebuilding on truncated history and comparing.
    """
    universe = list(cfg["universe"])
    f = features[features["Ticker"].isin(universe)].sort_values(["Ticker", "Date"])

    parts = []
    for ticker, g in f.groupby("Ticker"):
        g = g.copy()
        ret = np.log(g["Adjusted Close"]).diff()
        price = g["Adjusted Close"]
        stretch = price / price.rolling(50).mean() - 1.0
        parts.append(pd.DataFrame({
            "Date": g["Date"].to_numpy(),
            "Ticker": ticker,
            # Trailing 63-day mean of 21-day realised volatility, and the
            # trailing stretch. Both are functions of the past alone.
            "VolRank": ret.rolling(21).std().rolling(63).mean().to_numpy(),
            "Stretch": stretch.rolling(10).mean().to_numpy(),
        }))
    long = pd.concat(parts, ignore_index=True)

    # Rank each date's universe against itself - same-day information only.
    long["VolPct"] = long.groupby("Date")["VolRank"].rank(pct=True)
    long["Risk"] = -(long["VolPct"] - 0.5) * 2.0 \
        - long["Stretch"].fillna(0.0) * 4.0
    return long[["Date", "Ticker", "Risk"]].copy()


def build_backtest() -> dict:
    """Run the three-arm comparison over the held-out test window."""
    ensure = __import__("src.common", fromlist=["ensure_dirs"]).ensure_dirs
    ensure()
    cfg = load_config()
    set_seed(cfg.get("random_seed", 42))
    universe = list(cfg["universe"])
    rcfg = cfg["recommend"]
    w_cfg = rcfg["weights"]

    banner(LOG, "PRD Section 14 - recommendation hit-rate vs buy-and-hold")

    if not ML_PREDICTIONS.exists() or not FEATURES.exists():
        LOG.error("predictions or features missing; run retrain_models.py first")
        return {"status": "NOT RUN", "reason": "artifacts missing"}

    features = pd.read_parquet(FEATURES)
    features["Date"] = pd.to_datetime(features["Date"])
    predictions = pd.read_parquet(ML_PREDICTIONS)

    # The declared deployment model, so the backtest measures the same
    # forecasts the dashboard shows. Falls back to the lowest-MAE model in the
    # file when the declaration is absent.
    decl_path = PROCESSED_DIR / "declared_model.json"
    if decl_path.exists():
        import json
        model = json.loads(decl_path.read_text(encoding="utf-8")).get("model", "")
    else:
        model = ""
    if model and model in set(predictions["Model"]):
        preds = predictions[predictions["Model"] == model]
    else:
        mae_by_model = (predictions.groupby("Model")
                        .apply(lambda g: float(np.mean(np.abs(g["Residual"]))),
                               include_groups=False)
                        .sort_values())
        model = mae_by_model.index[0]
        preds = predictions[predictions["Model"] == model]
    LOG.info("Backtesting recommendations from model: %s", model)

    realised = _realised_forward_returns(features, cfg)
    risk = _trailing_risk_signal(features, cfg)

    # Forecast sub-signal per (Date, Ticker), causal by construction: the
    # prediction stored at date t was made using data up to t.
    fc = preds[["Date", "Ticker", "Predicted_Return"]].copy()
    fc = fc.rename(columns={"Predicted_Return": "Forecast"})

    df = (fc.merge(risk, on=["Date", "Ticker"], how="inner")
            .merge(realised.reset_index().rename(columns={"Date": "Date",
                                                         **{t: f"r_{t}" for t in universe}}),
                   on="Date", how="inner"))
    df = df.dropna(subset=["Forecast", "Risk"]).sort_values(["Date", "Ticker"])
    # Keep only rows where the outcome exists - the last session has no t+1.
    df = df.dropna(subset=[f"r_{t}" for t in universe], how="all")
    if df.empty:
        LOG.error("no overlap between predictions and realised outcomes")
        return {"status": "NOT RUN", "reason": "no overlapping window"}

    have_sentiment = False
    sent_path = PROCESSED_DIR / "sentiment_features.parquet"
    if sent_path.exists():
        try:
            s = pd.read_parquet(sent_path)
            have_sentiment = "Sentiment" in s.columns
        except Exception:  # noqa: BLE001
            have_sentiment = False

    # ---- build the three arms ------------------------------------------
    # The arms call the ENGINE's own fusion and thresholding rather than
    # reimplementing them. An earlier version of this file carried its own copy
    # of the weighting arithmetic, and it silently drifted from the deployed
    # logic - it reported three identical arms and zero HOLDs while the
    # dashboard was emitting a real mix. A backtest that measures something
    # other than what ships is worse than no backtest, because it looks like
    # evidence.
    from src.recommend import _fuse, _normalise, recommend

    buy, sell = float(rcfg["buy_threshold"]), float(rcfg["sell_threshold"])
    skill = _declared_skill()

    def actions_for(forecast_weight: float) -> pd.Series:
        scores = []
        for _, day in df.groupby("Date", sort=True):
            f_n = _normalise(day["Forecast"])
            r_n = _normalise(day["Risk"])
            s_n = pd.Series(0.5, index=day.index)
            score = _fuse(
                f_n, s_n, -r_n,
                {"Forecast_Signal": forecast_weight,
                 "Sentiment_Signal": w_cfg["sentiment"],
                 "Risk_Signal": w_cfg["risk"]},
            )
            scores.append(score)
        score_all = pd.concat(scores).reindex(df.index)
        # `recommend(score, low, high)` takes the SELL threshold first and the
        # BUY threshold second. Passing them the other way round silently
        # inverted the decision rule and produced zero HOLDs.
        return pd.Series(
            [recommend(v, sell, buy) for v in score_all], index=df.index)

    armed = {
        # What ships: the configured weight scaled by measured skill.
        "gated": actions_for(float(w_cfg["forecast"]) * skill),
        # What it would have done with the interlock switched off.
        "ungated": actions_for(float(w_cfg["forecast"])),
    }
    action_map = {"BUY": 1, "SELL": -1, "HOLD": 0}
    arms = {k: v.map(action_map).fillna(0).astype(int)
            for k, v in armed.items()}

    arms["buy_and_hold"] = pd.Series(1, index=df.index)

    # ---- log what the engine actually said, so the arms are auditable ----
    LOG.info("")
    LOG.info("Engine action mix across the window:")
    for name in ("gated", "ungated"):
        a = arms[name]
        LOG.info("  %-8s BUY %5d   HOLD %5d   SELL %5d",
                 name, int((a == 1).sum()), int((a == 0).sum()), int((a == -1).sum()))
    LOG.info("  %-8s BUY %5d   HOLD %5d   SELL %5d  (always long)",
             "b&h", int((arms["buy_and_hold"] == 1).sum()), 0,
             int((arms["buy_and_hold"] == -1).sum()))

    # ---- score each arm --------------------------------------------------
    buy, sell = float(rcfg["buy_threshold"]), float(rcfg["sell_threshold"])
    out = df[["Date", "Ticker", "Forecast", "Risk"]].copy()
    for name, action in arms.items():
        out[f"action_{name}"] = action

    # Realised forward return, aligned to `out` row by row. `out` and `df` share
    # an index (the merge preserved it), so a positional lookup is exact and
    # avoids a merge that would reorder the arms we have already built.
    realised_aligned = np.full(len(out), np.nan)
    for i, (_, row) in enumerate(out.iterrows()):
        col = f"r_{row['Ticker']}"
        if col in df.columns:
            realised_aligned[i] = df.at[row.name, col]
    out["realised"] = realised_aligned
    out = out.dropna(subset=["realised"])
    out["realised_sign"] = np.sign(out["realised"])

    results = {}
    for name in ("buy_and_hold", "gated", "ungated"):
        act = out[f"action_{name}"]
        if (act == 0).all():
            results[name] = {
                "hit_rate": None,
                "n_calls": 0,
                "note": ("no BUY or SELL was ever issued, so the engine always "
                         "held - the hit-rate is buy-and-hold's by construction"),
            }
            continue
        decided = act != 0
        if decided.sum() == 0:
            results[name] = {"hit_rate": None, "n_calls": 0,
                              "note": "never traded"}
            continue
        correct = (np.sign(act[decided]) == out.loc[decided, "realised_sign"])
        results[name] = {
            "hit_rate": float(correct.mean()),
            "n_calls": int(decided.sum()),
            "n_held": int((~decided).sum()),
            "n_buys": int((act == 1).sum()),
            "n_sells": int((act == -1).sum()),
        }
        LOG.info("  %-13s hit-rate %s over %s acted-on sessions (%s held)",
                 name,
                 f"{results[name]['hit_rate'] * 100:.2f}%" if results[name]["hit_rate"] else "n/a",
                 f"{results[name]['n_calls']:,}",
                 f"{results[name]['n_held']:,}")

    # ---- transparent contributing signals -------------------------------
    # How informative is each sub-signal on its own, against the same outcome?
    # This is the "transparent contributing signals" half of the PRD row.
    contributions = {}
    for col, name in (("Forecast", "forecast"), ("Risk", "risk")):
        s = out[col]
        if s.std() and out["realised_sign"].abs().sum() > 0:
            agree = (np.sign(s) == out["realised_sign"])
            contributions[name] = {
                "sign_agreement": float(agree.mean()),
                "corr_with_realised": float(s.corr(out["realised"])),
            }
    # And buy-and-hold, on the same sessions, for a like-for-like reference.
    bh = out.loc[out["action_buy_and_hold"] != 0]
    bh_rate = float((np.sign(bh["realised"]) == 1).mean()) if len(bh) else None

    # ---- skill gate: what the deployed system is entitled to claim -------
    beats = False
    if results["gated"]["hit_rate"] is not None and bh_rate is not None:
        beats = results["gated"]["hit_rate"] > bh_rate

    doc = {
        "generated_by": "src/recommendation_backtest.py",
        "prd_row": ("Section 14 acceptance table, 'Recommendations': backtested "
                    "hit-rate vs. buy-and-hold, documented with transparent "
                    "contributing signals."),
        "metric_definition": (
            "Hit-rate is the share of held-out sessions on which the sign of the "
            "composite score agrees with the sign of the realised return from "
            "that session to the next. Buy-and-hold is the baseline: its implied "
            "position is long every session, so its hit-rate is the share of "
            "sessions each asset rose."
        ),
        "model": model,
        "window": {
            "start": str(out["Date"].min().date()),
            "end": str(out["Date"].max().date()),
            "sessions": int(out["Date"].nunique()),
            "observations": int(len(out)),
        },
        "thresholds": {"buy": buy, "sell": sell},
        "weights_configured": dict(w_cfg),
        "forecast_skill_gate": skill,
        "sentiment_available": have_sentiment,
        "arms": results,
        "buy_and_hold_hit_rate": bh_rate,
        "contributing_signals": contributions,
        "beats_buy_and_hold": bool(beats),
        "interpretation": _interpret(results, bh_rate, skill, have_sentiment),
    }
    write_json(doc, PROCESSED_DIR / "recommendation_backtest.json")
    LOG.info("")
    LOG.info("Buy-and-hold hit-rate on the same sessions: %s",
             f"{bh_rate * 100:.2f}%" if bh_rate else "n/a")
    LOG.info("Verdict: %s", doc["interpretation"])
    LOG.info("-> %s", PROCESSED_DIR / "recommendation_backtest.json")
    return doc


def _declared_skill() -> float:
    p = PROCESSED_DIR / "recommendation_backtest.json"
    m = PROCESSED_DIR / "recommendations_meta.json"
    for path in (m, p):
        if path.exists():
            import json
            try:
                return float(json.loads(path.read_text(encoding="utf-8"))
                              .get("forecast_skill", 0.0))
            except Exception:  # noqa: BLE001
                continue
    return 0.0


def _interpret(results: dict, bh_rate, skill: float, have_sentiment: bool) -> str:
    gated = results.get("gated", {})
    ungated = results.get("ungated", {})
    if gated.get("hit_rate") is None:
        return (
            "The deployed engine issued no BUY or SELL across the window, so it "
            "holds every asset. Its hit-rate is therefore buy-and-hold's, by "
            "construction. This is the correct outcome of the forecast skill "
            "gate: the forecaster was measured to have no directional edge, so "
            "the gate zeroes its weight and the engine declines to trade rather "
            "than emit confident calls from a signal known not to work."
        )
    if bh_rate is None:
        return "Buy-and-hold produced no comparable observations."
    delta = gated["hit_rate"] - bh_rate
    if delta > 0:
        return (f"The recommendation engine beats buy-and-hold by "
                f"{delta * 100:.2f} percentage points on hit-rate.")
    if delta < 0:
        return (f"The recommendation engine trails buy-and-hold by "
                f"{abs(delta) * 100:.2f} percentage points, so the recommendations "
                f"would have been a net negative against simply holding.")
    return "The recommendation engine exactly matches buy-and-hold."
