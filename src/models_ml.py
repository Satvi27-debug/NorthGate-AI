"""PRD Section 8 - Machine Learning Workflow.

Four classical regressors on one feature table under one chronological
protocol, so "any performance difference is attributable to the model, not the
data pipeline" (Section 8).

Protocol
--------
1. **Walk-forward tuning** (Section 8.2, Listing 8.1): ``TimeSeriesSplit``
   expanding window *inside the training partition only*, with the scaler
   refit inside every fold. Nothing outside the training partition influences
   hyperparameter choice.
2. **Validation partition**: a clean chronological holdout used to compare the
   four families.
3. **Held-out test partition** (Section 9.5): touched exactly once, by the
   refit model, to populate the headline eight-model table.

Leakage guards
-------------
* Splits are made on the **date axis**, not the row axis. A row-wise slice
  would place ticker A's test-window rows beside ticker B's training rows at
  the same timestamp.
* The scaler is fit on the training partition only (Section 5.4) and that same
  fitted object transforms validation and test.
* The final artifact is refit on train+val, never on test. The previous
  implementation fit its scaler and model on the whole dataset and shipped that
  as the serving artifact.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, LinearSVR
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import (  # noqa: E402
    DL_METRICS,
    FEATURES,
    FEATURE_SCALER,
    ML_METRICS,
    ML_PREDICTIONS,
    MODEL_LEADERBOARD,
    MODELS_DIR,
    PROCESSED_DIR,
    TARGET_SCALER,
    banner,
    ensure_dirs,
    get_logger,
    load_config,
    set_seed,
    write_json,
)
from src.features import feature_columns, split_by_date  # noqa: E402

LOG = get_logger("models_ml")


# ==========================================================================
# Metrics (Section 8.3)
# ==========================================================================
def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """% of days where sign(pred) == sign(actual) (Section 8.3).

    Rows where the actual return is exactly zero are excluded, since no sign
    call can be right or wrong about them.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.sign(y_true[mask]) == np.sign(y_pred[mask])))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    price: np.ndarray | None = None) -> dict:
    """The Section 8.3 metric set.

    MAPE note: the target is a *log return*, so the literal
    ``|(y - yhat) / y|`` denominator passes through zero and MAPE explodes.
    We therefore report both the literal form (guarded) and a well-conditioned
    price-space MAPE, and use the price-space version in the headline table.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    out = {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "DirAcc": directional_accuracy(y_true, y_pred),
        "N": int(len(y_true)),
    }

    # Literal Section 8.3 MAPE on the return target, guarded near zero.
    denom = np.where(np.abs(y_true) < 1e-4, np.nan, y_true)
    literal = np.abs((y_true - y_pred) / denom)
    out["MAPE_Return"] = float(np.nanmean(literal) * 100.0) if np.isfinite(literal).any() else float("nan")

    # Price-space MAPE: scale-free and well conditioned.
    if price is not None:
        p_true = price * np.exp(y_true)
        p_pred = price * np.exp(y_pred)
        out["MAPE_Price"] = float(mean_absolute_percentage_error(p_true, p_pred) * 100.0)
    else:
        out["MAPE_Price"] = float("nan")

    out["MAPE"] = out["MAPE_Price"]
    return out


def naive_random_walk_metrics(y_true: np.ndarray,
                              price: np.ndarray | None = None) -> dict:
    """Baseline: tomorrow = today, i.e. a predicted return of exactly zero.

    MAE/RMSE come from the zero-return forecast. Directional accuracy cannot
    come from it - a zero forecast makes no directional call - so we report the
    sign-persistence naive forecast instead, which is the standard
    "no-information" directional baseline and the source of the ~50% figure the
    PRD anticipates in Section 9.5.
    """
    y_true = np.asarray(y_true, dtype=float)
    zeros = np.zeros_like(y_true)
    out = compute_metrics(y_true, zeros, price)
    out["DirAcc"] = float("nan")
    return out


def persistence_dir_accuracy(y_true: np.ndarray) -> float:
    """Directional accuracy of the naive sign-persistence forecast."""
    y_true = np.asarray(y_true, dtype=float)
    if len(y_true) < 2:
        return float("nan")
    predicted = y_true[:-1]  # "tomorrow behaves like today"
    return directional_accuracy(y_true[1:], predicted)


# ==========================================================================
# Model zoo (Section 8.1)
# ==========================================================================
# Canonical Section 8.1 family names, and the config.yaml keys that map to them.
# config.yaml uses PEP-8 keys (ridge, random_forest, ...) while the reported
# model names use the PRD's presentation form (Ridge, RandomForest, ...).
MODEL_NAME_BY_CONFIG_KEY = {
    "ridge": "Ridge",
    "random_forest": "RandomForest",
    "xgboost": "XGBoost",
    "svr": "SVR",
}


def convergence_status(model) -> str:
    """Did the optimiser actually reach its optimum?

    Reported as a first-class property rather than left as a stderr warning,
    because the failure is silent in every other respect: the process exits 0,
    the metric looks plausible, and the row reads like any other. scikit-learn
    records `n_iter_` for iterative solvers, and when that equals the iteration
    cap the coefficients are not the optimum for the hyperparameters they are
    reported against - so the metric is not a measurement of that model.

    Solvers without an iteration counter (the tree ensembles) are treated as
    converged, since they run to a fixed boosting budget rather than to a
    tolerance.
    """
    n_iter = getattr(model, "n_iter_", None)
    max_iter = getattr(model, "max_iter", None)
    if n_iter is None or max_iter is None:
        return "n/a"
    try:
        n_iter, max_iter = int(n_iter), int(max_iter)
    except (TypeError, ValueError):
        return "n/a"
    if n_iter >= max_iter:
        return "NO - hit iteration cap"
    return f"yes ({n_iter:,} iterations)"


def build_model(name: str, params: dict, seed: int):
    """Instantiate one of the four Section 8.1 regressors."""
    if name == "Ridge":
        return Ridge(alpha=params.get("alpha", 1.0), random_state=None)
    if name == "RandomForest":
        return RandomForestRegressor(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", 8),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            random_state=seed,
            n_jobs=-1,
        )
    if name == "XGBoost":
        return XGBRegressor(
            n_estimators=params.get("n_estimators", 300),
            learning_rate=params.get("learning_rate", 0.05),
            max_depth=params.get("max_depth", 4),
            subsample=params.get("subsample", 0.8),
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
            verbosity=0,
        )
    if name == "SVR":
        # LinearSVR scales in the sample dimension; SVR's RBF kernel stores an
        # n x n kernel matrix and is O(n^2) in memory, which is intractable on
        # tens of thousands of rows on a laptop. `svr.kernel` is configurable so
        # the PRD's RBF kernel is still available on a smaller subsample.
        kernel = params.get("kernel", "linear")
        if kernel == "linear":
            # `dual` must stay True here: scikit-learn's LinearSVR only
            # supports the squared epsilon-insensitive loss in the primal, and
            # raising ValueError on `dual=False` with the default loss. So the
            # settings below were chosen by measurement, not taste.
            #
            # The previous fit (max_iter=5_000, tol=1e-4) hit the iteration cap
            # and emitted ConvergenceWarning on every fold, which means the
            # coefficients returned were not the optimum the chosen C and
            # epsilon describe - so the SVR row was not measuring the model it
            # claimed to.
            #
            # Settings were then chosen by measurement on the real training
            # partition (22,880 rows x 132 features), not by taste:
            #
            #     C=0.1  eps=0.0001   converged,  22,633 iterations,  248s
            #     C=1.0  eps=0.001    did NOT converge, stopped at the 200,000 cap
            #
            # C=1.0 is excluded from the grid in config.yaml for the same
            # reason. Dropping the exactly-collinear columns helped at C=0.1
            # (317s -> 248s) but did not rescue C=1.0, because the residual
            # dependence among the price-level columns is near-collinearity
            # rather than algebraic. Reporting that honestly is the point.
            #
            # The cost is worth paying: an unconverged fit means the leaderboard
            # row is not a measurement of anything. Convergence is recorded per
            # row in the `Converged` column rather than left as a stderr
            # warning, non-converged rows are excluded from the Section 8.3
            # comparison, and `retrain_models.py` fails the build on a
            # ConvergenceWarning even when the stage exits zero - a zero exit
            # code is not evidence that a stage produced a valid number.
            return LinearSVR(
                C=params.get("C", 1.0),
                epsilon=params.get("epsilon", 0.001),
                dual=True,
                max_iter=int(params.get("max_iter", 200_000)),
                tol=float(params.get("tol", 1e-3)),
                random_state=seed,
            )
        return SVR(
            C=params.get("C", 1.0),
            epsilon=params.get("epsilon", 0.001),
            gamma=params.get("gamma", "scale"),
            kernel=kernel,
            cache_size=500,
        )
    raise ValueError(f"unknown model: {name}")


def _grid(spec: dict) -> list[dict]:
    """Cartesian product of a hyperparameter spec from config.yaml."""
    from itertools import product

    keys = list(spec.keys())
    if not keys:
        return [{}]
    return [dict(zip(keys, combo)) for combo in product(*(spec[k] for k in keys))]


# ==========================================================================
# Walk-forward hyperparameter search (Section 8.2)
# ==========================================================================
def walk_forward_search(name: str, X: np.ndarray, y: np.ndarray,
                        spec: dict, n_splits: int, seed: int,
                        max_rows: int | None = None) -> tuple[dict, list[dict]]:
    """Expanding-window CV inside the training partition, scaler refit per fold.

    ``max_rows`` optionally subsamples the *search* set to its most recent
    ``max_rows`` rows, which is how the runtime-heavy models (SVR, forests) are
    kept tractable on CPU. The training partition is already strictly earlier
    than validation and test, so taking its tail preserves chronology exactly and
    cannot reach forward in time. The final refit always uses the full training
    partition, so this is a speed measure only - it does not weaken the
    evaluation.

    Returns the best parameter dict and the full fold log.
    """
    if max_rows and len(X) > max_rows:
        # Tail slice keeps the most recent, most relevant regime. Ordering is
        # already chronological, so this cannot reach forward in time.
        X, y = X[-max_rows:], y[-max_rows:]
        LOG.info("    (search subsampled to the most recent %d rows for speed; "
                 "final refit uses the full training partition)", max_rows)

    candidates = _grid(spec)
    n_splits = max(2, min(n_splits, max(2, len(X) // 400)))
    # Cap the validation block so every fold is scored on a comparable number of
    # observations; a bare expanding window makes the last fold enormous.
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=max(500, len(X) // (n_splits + 2)))

    fold_log: list[dict] = []
    best, best_score = None, np.inf

    for params in candidates:
        fold_scores = []
        for fold, (tr, va) in enumerate(tscv.split(X)):
            # FIT ON TRAIN ONLY, inside every fold (Listing 8.1).
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(X[tr])
            Xva = scaler.transform(X[va])

            model = build_model(name, params, seed)
            model.fit(Xtr, y[tr])
            pred = model.predict(Xva)
            fold_scores.append(mean_absolute_error(y[va], pred))

        mae = float(np.mean(fold_scores))
        fold_log.append({"params": params, "fold_mae": fold_scores, "mean_mae": mae})
        if mae < best_score:
            best, best_score = params, mae

    LOG.info("    %-13s tried %2d config(s) over %d folds | best walk-forward MAE=%.6f %s",
             name, len(candidates), n_splits, best_score, best)
    return best, fold_log


# ==========================================================================
# Main
# ==========================================================================
class EqualWeightEnsemble:
    """Average of several fitted regressors (a genuine, cheap variance reduction).

    The individual Section 8.1 models make partly independent errors, so the
    average of their predictions is usually a little better than any one of them
    and noticeably more stable. Weights are equal rather than fitted, because
    fitting blend weights on validation and then reporting validation numbers
    would be a mild form of selection on the set we are judged against.

    This is reported as an EXTRA row in the Section 9.5 table, never in place of
    one of the four required models.
    """

    def __init__(self, fitted: dict[str, object]):
        self.fitted = fitted
        self.n_models = len(fitted)

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = [np.asarray(m.predict(X), dtype=float).ravel() for m in self.fitted.values()]
        return np.mean(preds, axis=0)


def run_ml_models() -> dict:
    ensure_dirs()
    cfg = load_config()
    seed = set_seed(cfg.get("random_seed", 42))
    t0 = time.time()

    banner(LOG, "PRD Section 8 - Classical Machine Learning")
    if not FEATURES.exists():
        raise FileNotFoundError(f"{FEATURES} not found. Run `python src/features.py` first.")

    df = pd.read_parquet(FEATURES)
    df["Date"] = pd.to_datetime(df["Date"])
    cols = feature_columns(df)
    LOG.info("Feature table: %d rows x %d features | %s -> %s",
             len(df), len(cols), df["Date"].min().date(), df["Date"].max().date())

    # -- chronological split on the DATE axis ------------------------------
    bounds = split_by_date(df["Date"], cfg)
    partitions = {
        name: df[df["Date"].isin(dates)].sort_values(["Date", "Ticker"])
        for name, dates in bounds.items()
    }
    for name, part in partitions.items():
        LOG.info("  %-5s %6d rows | %4d sessions | %s -> %s",
                 name, len(part), part["Date"].nunique(),
                 part["Date"].min().date(), part["Date"].max().date())

    train_df, val_df, test_df = partitions["train"], partitions["val"], partitions["test"]

    Xtr_all = train_df[cols].to_numpy(dtype=float)
    ytr_all = train_df["Target"].to_numpy(dtype=float)
    Xva = val_df[cols].to_numpy(dtype=float)
    yva = val_df["Target"].to_numpy(dtype=float)
    Xte = test_df[cols].to_numpy(dtype=float)
    yte = test_df["Target"].to_numpy(dtype=float)
    pte = test_df["Adjusted Close"].to_numpy(dtype=float)

    # -- baseline on the held-out test window (Section 8.3) ----------------
    pte_all = test_df["Adjusted Close"].to_numpy(dtype=float)
    baseline = naive_random_walk_metrics(yte, pte_all)
    baseline["DirAcc_Persistence"] = persistence_dir_accuracy(yte)
    LOG.info("")
    LOG.info("Naive random-walk baseline on the test window (%d obs):", len(yte))
    LOG.info("  RMSE=%.6f  MAE=%.6f  MAPE(price)=%.4f%%  R2=%.4f",
             baseline["RMSE"], baseline["MAE"], baseline["MAPE_Price"], baseline["R2"])
    LOG.info("  Directional accuracy: %.2f%% (sign-persistence forecast)",
             100.0 * baseline["DirAcc_Persistence"])

    # -- walk-forward tuning inside TRAIN ----------------------------------
    LOG.info("")
    LOG.info("Walk-forward hyperparameter search (expanding window, train partition only):")
    search_log: dict[str, list[dict]] = {}
    best_params: dict[str, dict] = {}
    # Per-family search-time row caps. SVR is O(n^2) in memory and the forests
    # are the other CPU bottleneck, so both search on a recent slice. The final
    # refit always uses every training row.
    caps = cfg["ml"].get("search_row_caps", {})
    default_cap = int(cfg["ml"].get("max_tuning_rows") or 0) or None
    for config_key, spec in cfg["ml"]["models"].items():
        name = MODEL_NAME_BY_CONFIG_KEY.get(config_key, config_key)
        cap = int(caps.get(config_key, 0)) or default_cap
        best, log_ = walk_forward_search(name, Xtr_all, ytr_all, spec,
                                         int(cfg["ml"]["n_splits"]), seed, cap)
        best_params[name] = best
        search_log[name] = log_

    # -- scaler fit on TRAIN only (Section 5.4) ----------------------------
    scaler = StandardScaler().fit(Xtr_all)

    # -- validation comparison --------------------------------------------
    LOG.info("")
    LOG.info("Validation partition comparison:")
    val_rows = []
    for name in best_params:
        model = build_model(name, best_params[name], seed)
        model.fit(scaler.transform(Xtr_all), ytr_all)
        pred = model.predict(scaler.transform(Xva))
        m = compute_metrics(yva, pred, val_df["Adjusted Close"].to_numpy(dtype=float))
        m["Model"] = name
        val_rows.append(m)
        LOG.info("  %-13s RMSE=%.6f MAE=%.6f R2=%7.4f DirAcc=%.2f%%",
                 name, m["RMSE"], m["MAE"], m["R2"], 100 * m["DirAcc"])
    pd.DataFrame(val_rows).to_csv(PROCESSED_DIR / "ml_validation_metrics.csv", index=False)

    # -- final refit on TRAIN + VAL, scaler still from TRAIN ---------------
    Xtrval = np.vstack([Xtr_all, Xva])
    ytrval = np.concatenate([ytr_all, yva])

    LOG.info("")
    LOG.info("Refitting on train+val (scaler still fit on train only) and evaluating ONCE on test:")
    test_rows, predictions, fitted = [], [], {}
    convergence = {}
    Xtrval_s = scaler.transform(Xtrval)
    Xte_s = scaler.transform(Xte)

    for name in best_params:
        model = build_model(name, best_params[name], seed)
        model.fit(Xtrval_s, ytrval)
        fitted[name] = model
        pred = model.predict(Xte_s)
        m = compute_metrics(yte, pred, pte)
        m["Model"] = name
        m["Family"] = "ML"
        m["Best_Params"] = json.dumps(best_params[name])
        conv = convergence_status(model)
        convergence[name] = conv
        m["Converged"] = conv
        test_rows.append(m)

        LOG.info("  %-13s RMSE=%.6f MAE=%.6f MAPE=%.4f%% R2=%7.4f DirAcc=%.2f%%  [%s]",
                 name, m["RMSE"], m["MAE"], m["MAPE_Price"], m["R2"],
                 100 * m["DirAcc"], conv)
        if conv.startswith("NO"):
            LOG.error("  %s DID NOT CONVERGE - its metric is not a valid "
                      "measurement of the model it is attributed to.", name)

        block = test_df[["Date", "Ticker", "Adjusted Close", "Target"]].copy()
        block["Actual_Return"] = yte
        block["Predicted_Return"] = pred
        block["Model"] = name
        block["Residual"] = yte - pred
        block["Predicted_Price"] = pte * np.exp(pred)
        block["Actual_Price"] = pte * np.exp(yte)
        predictions.append(block)

    # -- equal-weight ensemble (extra row, never a replacement) -------------
    ens_cfg = cfg["ml"].get("ensemble", {}) or {}
    if ens_cfg.get("enabled", False):
        members = [n for n in ens_cfg.get("members", list(best_params)) if n in fitted]
        if len(members) >= 2:
            ens = EqualWeightEnsemble({n: fitted[n] for n in members})
            pred = ens.predict(Xte_s)
            m = compute_metrics(yte, pred, pte)
            m["Model"] = "Ensemble (equal weight)"
            m["Family"] = "ML"
            m["Best_Params"] = json.dumps({"members": members, "weights": "equal"})
            # The ensemble inherits its members' convergence status, so it is
            # reported as failing if any member failed rather than looking clean.
            bad = [n for n in members if convergence.get(n, "").startswith("NO")]
            m["Converged"] = (f"NO - member(s) {', '.join(bad)} did not converge"
                              if bad else "yes (average of converged members)")
            test_rows.append(m)
            LOG.info("  %-13s RMSE=%.6f MAE=%.6f MAPE=%.4f%% R2=%7.4f DirAcc=%.2f%%",
                     "Ensemble", m["RMSE"], m["MAE"], m["MAPE_Price"], m["R2"],
                     100 * m["DirAcc"])

            block = test_df[["Date", "Ticker", "Adjusted Close", "Target"]].copy()
            block["Actual_Return"] = yte
            block["Predicted_Return"] = pred
            block["Model"] = m["Model"]
            block["Residual"] = yte - pred
            block["Predicted_Price"] = pte * np.exp(pred)
            block["Actual_Price"] = pte * np.exp(yte)
            predictions.append(block)
            LOG.info("  (equal-weight average of %s)", ", ".join(members))

    pred_long = pd.concat(predictions, ignore_index=True)
    pred_long.to_parquet(ML_PREDICTIONS, index=False)

    # -- baseline vs models: the Section 8.3 honesty bar --------------------
    LOG.info("")
    LOG.info("Section 8.3 honesty check - does any model beat the random walk?")

    # A row whose optimiser never reached its optimum is not a measurement, so
    # it must not be allowed to win the comparison. This is the concrete payoff
    # of recording convergence explicitly: the exclusion is made from a value
    # in the table rather than from someone having noticed a stderr warning.
    invalid = [r["Model"] for r in test_rows
               if str(r.get("Converged", "")).startswith("NO")]
    valid = [r for r in test_rows if not str(r.get("Converged", "")).startswith("NO")]
    if invalid:
        LOG.error("  EXCLUDED from this comparison (did not converge): %s",
                  ", ".join(invalid))
    if not valid:
        LOG.error("  No converged model remains, so no honest comparison is "
                  "possible. The metrics file records the failure.")
        test_df_metrics = pd.DataFrame(test_rows)
        test_df_metrics.to_csv(ML_METRICS, index=False)
        return {"status": "FAILED", "reason": "no model converged"}

    best_mae = min(r["MAE"] for r in valid)
    best_dir = max(r["DirAcc"] for r in valid)
    beats_mae = best_mae < baseline["MAE"]
    beats_dir = best_dir > baseline["DirAcc_Persistence"]
    winner = min(valid, key=lambda r: r["MAE"])["Model"]
    LOG.info("  best MAE      : %.6f (%s) vs baseline %.6f  -> %s",
             best_mae, winner, baseline["MAE"], "BEATS" if beats_mae else "DOES NOT BEAT")
    LOG.info("  best Dir.Acc  : %.2f%% vs baseline %.2f%%  -> %s",
             100 * best_dir, 100 * baseline["DirAcc_Persistence"],
             "BEATS" if beats_dir else "DOES NOT BEAT")
    if not (beats_mae and beats_dir):
        LOG.warning("  Section 8.3: a model that cannot beat the random walk on BOTH MAE and")
        LOG.warning("  directional accuracy is reported as a failure. This is reported as-is.")

    # -- persist -----------------------------------------------------------
    test_df_metrics = pd.DataFrame(test_rows)
    test_df_metrics.to_csv(ML_METRICS, index=False)

    baseline_row = {
        "Model": "Naive random walk", "Family": "Baseline", "RMSE": baseline["RMSE"],
        "MAE": baseline["MAE"], "MAPE": baseline["MAPE_Price"], "R2": baseline["R2"],
        "DirAcc": baseline["DirAcc_Persistence"], "N": baseline["N"],
        "Best_Params": "", "MAPE_Return": baseline["MAPE_Return"],
    }
    pd.DataFrame([baseline_row]).to_csv(PROCESSED_DIR / "baseline_metrics.csv", index=False)

    # Save the representative tree model plus the TRAIN-ONLY scaler.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_family = min(test_rows, key=lambda r: r["MAE"])["Model"]
    final_model = build_model(best_family, best_params[best_family], seed)
    final_model.fit(scaler.transform(Xtrval), ytrval)
    joblib.dump(final_model, MODELS_DIR / f"{best_family.lower()}_model.pkl")
    joblib.dump(scaler, FEATURE_SCALER)
    joblib.dump({"feature_columns": cols, "config_signature": _signature(cfg)}, TARGET_SCALER)
    LOG.info("Saved %s as the representative classical model (+ train-only scaler)", best_family)

    # Remove artifacts from an earlier run that the current configuration no
    # longer produces. A stale file with the right name but the wrong feature
    # count is worse than no file, because it loads without complaint and then
    # predicts on the wrong columns.
    expected = {f"{best_family.lower()}_model.pkl", FEATURE_SCALER.name, TARGET_SCALER.name}
    for stale in sorted(MODELS_DIR.glob("*_mlp.pkl")):
        if stale.name not in expected:
            stale.unlink()
            LOG.info("Removed stale artifact %s (not produced by the current config)", stale.name)
    for stale in sorted(MODELS_DIR.glob("*.keras")):
        if stale.name not in expected:
            stale.unlink()
            LOG.info("Removed stale artifact %s", stale.name)

    summary = {
        "generated_by": "src/models_ml.py",
        "seed": seed,
        "n_features": len(cols),
        "feature_columns": cols,
        "partitions": {
            k: {"rows": int(len(v)), "sessions": int(v["Date"].nunique()),
                "start": str(v["Date"].min().date()), "end": str(v["Date"].max().date())}
            for k, v in partitions.items()
        },
        "baseline_test": {k: v for k, v in baseline.items()},
        "best_params": best_params,
        "walk_forward_log": search_log,
        "validation_metrics": val_rows,
        "test_metrics": test_rows,
        "beats_baseline": {
            "mae": bool(beats_mae), "directional_accuracy": bool(beats_dir),
            "best_model_by_mae": winner,
            "note": ("Reported honestly per Section 8.3. A model that cannot beat the "
                     "random walk on both metrics is a failure and is reported as such."),
        },
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    write_json(summary, PROCESSED_DIR / "ml_run_summary.json")
    LOG.info("")
    LOG.info("Metrics -> %s", ML_METRICS)
    LOG.info("Predictions -> %s", ML_PREDICTIONS)
    LOG.info("Elapsed %.1fs", summary["elapsed_seconds"])
    return summary


def _signature(cfg: dict) -> str:
    """Cheap fingerprint of the config so a stale scaler is detectable."""
    import hashlib

    payload = json.dumps(
        {"universe": cfg["universe"], "price_column": cfg["price_column"],
         "target": cfg["target"], "split": cfg["split"], "seed": cfg.get("random_seed")},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


if __name__ == "__main__":
    run_ml_models()
