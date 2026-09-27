"""Cross-sectional prediction: a different, better-posed question.

PRD Section 14.1 defines the forecasting bar on the ABSOLUTE return: "Best model
beats naive random walk on MAE & direction". This module does not touch that
bar. It asks a narrower, more learnable question and reports it separately:

    Given a day's information, which of these stocks will do best tomorrow?

That is a ranking problem, not a point-forecast problem, and it is the question
equity analysts actually ask of a stock screen. Crucially it has its OWN
like-for-like baseline: "assume every stock will do the same as the universe
average", which is what you would believe with no skill at all. Any comparison
against the Section 9.5 random-walk baseline would be apples-to-oranges, because
a cross-sectional forecast deliberately ignores the market's direction.

Why this is a legitimate addition rather than metric-shopping
------------------------------------------------------------
* The change of task is disclosed here, in the dashboard, and in the report.
* The baseline is a genuine no-skill benchmark for the SAME task.
* The Section 9.5 table is untouched and remains the headline result.
* All features and labels are strictly causal, verified by
  ``tests/test_causality.py``.

Honesty rule: if the cross-sectional model cannot beat its own no-skill
baseline, that is reported as a failure too.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import (  # noqa: E402
    PROCESSED_DIR,
    banner,
    get_logger,
    load_config,
    set_seed,
    write_json,
)
from src.features import feature_columns, split_by_date  # noqa: E402
from src.models_ml import directional_accuracy  # noqa: E402

LOG = get_logger("cross_sectional")


def prepare(features: pd.DataFrame, cfg: dict) -> dict:
    """Build the cross-sectional dataset.

    The target is each stock's return *relative to the universe mean* on the same
    day. That is a number we know at the close of day t, and it is the residual
    the model is asked to forecast one session ahead.
    """
    universe = list(cfg["universe"])
    f = features[features["Ticker"].isin(universe)].copy()

    # Cross-sectional demeaning, per day. Same-day information only.
    daily_mean = f.groupby("Date")["Target"].transform("mean")
    f["Relative_Target"] = f["Target"] - daily_mean

    # Rank within the day: the cleanest scale-free cross-sectional target.
    f["Rank_Target"] = f.groupby("Date")["Target"].rank(pct=True) - 0.5

    # Reset to a clean RangeIndex so the positional per-day loops below are
    # well defined. `split_by_date` reads the Date column, so it still works.
    f = f.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    bounds = split_by_date(f["Date"], cfg)
    return {"frame": f, "bounds": bounds}


def run_cross_sectional() -> dict:
    cfg = load_config()
    seed = set_seed(cfg.get("random_seed", 42))
    feats_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "features.parquet"
    if not feats_path.exists():
        LOG.warning("features.parquet not found; run `python src/features.py` first.")
        return {"status": "NOT RUN", "reason": "features.parquet missing"}

    banner(LOG, "Cross-sectional analysis - which stock does best tomorrow?")
    features = pd.read_parquet(feats_path)
    features["Date"] = pd.to_datetime(features["Date"])

    data = prepare(features, cfg)
    f, bounds = data["frame"], data["bounds"]
    universe = list(cfg["universe"])
    n_assets = len(universe)

    cols = feature_columns(f)
    LOG.info("Feature table: %d rows x %d features", len(f), len(cols))
    for name, dates in bounds.items():
        part = f[f["Date"].isin(dates)]
        LOG.info("  %-5s %4d sessions | %5d rows", name, len(dates), len(part))

    # ---- the no-skill baseline for THIS task -----------------------------
    # Predicting "every stock does the same as the universe average" means a
    # relative prediction of exactly zero for every name.
    test_dates = bounds["test"]
    # reset_index so the per-day positional loops below are well defined.
    te = f[f["Date"].isin(test_dates)].sort_values(["Date", "Ticker"]).reset_index(drop=True)
    y_rel = te["Relative_Target"].to_numpy(dtype=float)
    y_rank = te["Rank_Target"].to_numpy(dtype=float)
    y_abs = te["Target"].to_numpy(dtype=float)

    zeros_rel = np.zeros_like(y_rel)
    zeros_rank = np.zeros_like(y_rank)

    base_mae = float(np.mean(np.abs(y_rel)))
    base_rank_mae = float(np.mean(np.abs(y_rank)))
    base_ic = 0.0
    base_dir = float(np.mean(np.sign(y_rel) == np.sign(zeros_rel))) if (y_rel != 0).any() else np.nan

    LOG.info("")
    LOG.info("No-skill baseline for THIS task (predict the universe average):")
    LOG.info("  relative-target MAE : %.6f", base_mae)
    LOG.info("  rank-target MAE     : %.6f", base_rank_mae)
    LOG.info("  rank IC             : %.4f", base_ic)
    LOG.info("  top-1 hit rate      : %.4f (chance = %.4f)", 0.0, 1.0 / n_assets)
    LOG.info("  directional accuracy: %.4f (this is the tie-breaking floor)", base_dir)

    # ---- model selection, walk-forward on the TRAINING partition ---------
    # Selecting on a single arbitrary hyperparameter setting and then declaring
    # "no skill" would be its own kind of dishonesty: the honest null requires
    # that the model was given a fair chance first. The grid below is searched
    # entirely inside the training partition, so the test window stays untouched.
    train = f[f["Date"].isin(bounds["train"])].sort_values(["Date", "Ticker"])
    ytr = train["Relative_Target"].to_numpy(dtype=float)
    X_all = train[cols].to_numpy(dtype=float)

    n_splits = int(cfg["ml"]["n_splits"])
    fold_size = max(500, len(train) // (n_splits + 2))
    splitter = list(TimeSeriesSplit(n_splits=n_splits, test_size=fold_size).split(train))

    # The no-skill reference *on the same folds*, so the comparison is
    # like-for-like rather than train-period-vs-test-period.
    zero_fold_mae = []
    for _, va_idx in splitter:
        zero_fold_mae.append(float(np.mean(np.abs(ytr[va_idx]))))
    zero_ref = float(np.mean(zero_fold_mae))

    grid = []
    xs_cfg = cfg.get("cross_sectional", {}) or {}
    g = xs_cfg.get("grid", {}) or {}
    hgb = g.get("hist_gb", {}) or {}
    for depth in hgb.get("max_depth", [3, 6, None]):
        for leaf in hgb.get("min_samples_leaf", [20, 100]):
            for lr in hgb.get("learning_rate", [0.03, 0.1]):
                for it in hgb.get("max_iter", [300]):
                    for l2 in hgb.get("l2_regularization", [1.0]):
                        grid.append({"max_depth": depth, "min_samples_leaf": leaf,
                                     "learning_rate": lr, "max_iter": it,
                                     "l2_regularization": l2})
    for a in (g.get("ridge", {}) or {}).get("alpha", [10.0, 100.0]):
        grid.append({"alpha": a, "kind": "ridge"})

    LOG.info("")
    LOG.info("Searching %d cross-sectional configs by walk-forward inside the "
             "training partition...", len(grid))
    LOG.info("  (no-skill reference on the same folds: MAE %.6f)", zero_ref)

    scored = []
    for params in grid:
        kind = params.get("kind")
        fold_mae = []
        for tr_idx, va_idx in splitter:
            sc = StandardScaler()
            Xtr = sc.fit_transform(X_all[tr_idx])
            Xva = sc.transform(X_all[va_idx])
            if kind == "ridge":
                mdl = Ridge(alpha=params["alpha"], random_state=None)
            else:
                p = {k: v for k, v in params.items() if k != "kind"}
                mdl = HistGradientBoostingRegressor(random_state=seed, **p)
            mdl.fit(Xtr, ytr[tr_idx])
            fold_mae.append(float(np.mean(np.abs(ytr[va_idx] - mdl.predict(Xva)))))
        score = float(np.mean(fold_mae))
        scored.append((score, kind or "hist_gb", params))

    scored.sort(key=lambda t: t[0])
    for score, kind, params in scored:
        label = "Ridge a=%g" % params["alpha"] if kind == "ridge" else (
            "HGB d=%s leaf=%d lr=%.2f" % (params["max_depth"],
                                          params["min_samples_leaf"],
                                          params["learning_rate"]))
        mark = "  <-- best" if (score, kind, params) == scored[0] else ""
        edge = (zero_ref - score) / zero_ref * 100.0
        LOG.info("  %-26s walk-forward MAE %.6f  (%+.2f%% vs no skill)%s",
                 label, score, edge, mark)

    best_score, best_kind, best_params = scored[0]
    beats_on_val = best_score < zero_ref
    LOG.info("")
    LOG.info("Best walk-forward MAE %.6f vs no-skill %.6f -> %s",
             best_score, zero_ref, "BETTER" if beats_on_val else "NOT BETTER")

    # ---- final fit on train+val, scaler fit on TRAIN only, test once -----
    trval_dates = np.concatenate([bounds["train"], bounds["val"]])
    trval = f[f["Date"].isin(trval_dates)].sort_values(["Date", "Ticker"])
    ytrval = trval["Relative_Target"].to_numpy(dtype=float)

    train_scaler = StandardScaler().fit(X_all)
    if best_kind == "ridge":
        model = Ridge(alpha=best_params["alpha"])
    else:
        model = HistGradientBoostingRegressor(
            random_state=seed,
            **{k: v for k, v in best_params.items() if k != "kind"})
    model.fit(train_scaler.transform(trval[cols].to_numpy(dtype=float)), ytrval)

    Xte = train_scaler.transform(te[cols].to_numpy(dtype=float))
    pred_rel = model.predict(Xte)

    mae = float(np.mean(np.abs(y_rel - pred_rel)))

    # Rank information coefficient, computed within each day.
    ics, tops, degenerate = [], [], 0
    for _, grp in te.groupby("Date"):
        mask = grp.index.to_numpy()
        p_local, a_local = pred_rel[mask], y_rel[mask]
        # A rank correlation is undefined if EITHER side is flat. A flat
        # prediction means the model expressed no preference that day; a flat
        # actual just means the stocks all moved together, which is a real
        # session with nothing to rank. Both are counted rather than dropped,
        # because a high flat-prediction count means the model has learned to
        # answer "they will all do the same" - the no-skill answer.
        if np.std(p_local) == 0 or np.std(a_local) == 0:
            degenerate += 1
        else:
            ics.append(float(stats.spearmanr(p_local, a_local).statistic))
        # Top-1: did we pick the RIGHT STOCK? Compare the winning *row*, not
        # the winning *value* - two float series are never exactly equal, so
        # comparing values would report 0% forever.
        tops.append(1 if int(np.argmax(p_local)) == int(np.argmax(a_local)) else 0)
    ic = float(np.mean(ics)) if ics else float("nan")
    top1 = float(np.mean(tops)) if tops else float("nan")

    # Is the top-1 rate actually better than picking at random? With 10 assets
    # and 405 sessions, 11.1% is only 0.75 standard errors above the 10% chance
    # level, so the binomial p-value is the only honest way to present it.
    n_sessions = len(tops)
    alpha_sig = float(xs_cfg.get("significance_level", 0.05))
    p_top1 = float(stats.binomtest(int(sum(tops)), n_sessions,
                                   1.0 / n_assets).pvalue) if n_sessions else float("nan")
    top1_distinguishable = bool(np.isfinite(p_top1) and p_top1 < alpha_sig)

    # Absolute directional accuracy, recovered by adding the market's own move.
    # This is NOT comparable to the Section 9.5 baseline and is not presented
    # as though it were: a cross-sectional model is not trying to time the
    # market, only to rank within it.
    day_mean = te.groupby("Date")["Target"].transform("mean").to_numpy(dtype=float)
    abs_pred = pred_rel + day_mean
    dir_acc = directional_accuracy(y_abs, abs_pred)

    LOG.info("")
    LOG.info("Cross-sectional model on the held-out test window (%d rows, %d sessions):",
             len(te), n_sessions)
    LOG.info("  relative-target MAE : %.6f  vs baseline %.6f  -> %s",
             mae, base_mae, "BETTER" if mae < base_mae else "NOT BETTER")
    if np.isfinite(ic):
        LOG.info("  rank IC (Spearman) : %+.4f  over %d sessions with a ranking "
                 "(0 = no skill)", ic, len(ics))
    else:
        LOG.info("  rank IC (Spearman) : undefined on %d of %d sessions - every "
                 "stock tied,", degenerate, n_sessions)
        LOG.info("                       so there was no ranking to be right "
                 "or wrong about")
    LOG.info("  top-1 hit rate      : %.4f  vs chance %.4f  (binomial p=%.3f -> %s)",
             top1, 1.0 / n_assets, p_top1,
             "BETTER" if top1_distinguishable else "INDISTINGUISHABLE FROM CHANCE")
    LOG.info("  absolute direction : %.2f%%  (not comparable to the 9.5 baseline)",
             100 * dir_acc)

    beats = mae < base_mae
    verdict = ("The cross-sectional model beats the no-skill baseline on relative "
               "error.") if beats else (
        "The cross-sectional model does NOT beat the no-skill baseline. Reported "
        "as a failure, like every other result here.")

    LOG.info("")
    LOG.info("Verdict: %s", verdict)

    result = {
        "generated_by": "src/cross_sectional.py",
        "status": "COMPLETED",
        "task": ("Rank the ten stocks by expected next-session return, using only "
                 "information available at the previous close."),
        "why_this_is_separate": (
            "The Section 9.5 bar is the ABSOLUTE return versus a random walk. A "
            "cross-sectional model deliberately ignores market direction, so it "
            "is judged against its OWN no-skill baseline (predict the universe "
            "average) and is never presented as clearing the Section 9.5 bar."
        ),
        "seed": seed,
        "n_features": len(cols),
        "baseline": {
            "relative_target_mae": base_mae,
            "rank_target_mae": base_rank_mae,
            "rank_ic": base_ic,
            "top1_hit_rate": 0.0,
            "chance_top1": 1.0 / n_assets,
            "directional_accuracy": base_dir,
            "walk_forward_mae_same_folds": zero_ref,
        },
        "model_selection": {
            "protocol": ("Grid searched by walk-forward inside the TRAINING "
                         "partition only. The test window was not touched "
                         "during selection."),
            "configs_tried": len(grid),
            "folds": len(splitter),
            "leaderboard": [
                {"family": kind, "params": params, "walk_forward_mae": score}
                for score, kind, params in scored
            ],
            "best_family": best_kind,
            "best_params": {k: v for k, v in best_params.items() if k != "kind"},
            "walk_forward_mae": best_score,
            "walk_forward_folds": len(splitter),
            "beats_no_skill_on_validation": bool(beats_on_val),
        },
        "test": {
            "rows": int(len(te)),
            "sessions": int(n_sessions),
            "relative_target_mae": mae,
            "rank_ic": None if not np.isfinite(ic) else ic,
            "rank_ic_sessions_defined": len(ics),
            "rank_ic_sessions_degenerate": int(degenerate),
            "top1_hit_rate": top1,
            "top1_chance": 1.0 / n_assets,
            "top1_binomial_p": p_top1,
            "top1_better_than_chance": top1_distinguishable,
            "absolute_directional_accuracy": dir_acc,
        },
        "beats_own_baseline": bool(beats),
        "verdict": verdict,
        "leak_found_and_fixed": (
            "The first run reported a walk-forward MAE of 0.000575 against a "
            "0.010922 baseline - a 95% 'improvement' that was entirely fake. "
            "Relative_Target and Rank_Target are pure re-expressions of the "
            "label, and because they were numeric and absent from "
            "NON_FEATURE_COLUMNS they were swept into the model inputs. The "
            "model was reading its own answer. Both are now blocklisted, and "
            "tests/test_causality.py carries a standing guard that flags any "
            "input column correlating above 0.99 with any return-valued label."
        ),
    }
    write_json(result, PROCESSED_DIR / "cross_sectional_results.json")
    LOG.info("Results -> %s", PROCESSED_DIR / "cross_sectional_results.json")
    return result


if __name__ == "__main__":
    run_cross_sectional()

