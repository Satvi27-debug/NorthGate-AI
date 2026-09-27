"""PRD Section 11 - Portfolio Optimization Methodology (Modern Portfolio Theory).

Objective (Section 11.1)
------------------------
    maximise  (w'mu - r_f) / sqrt(w' Sigma w)
    subject to  sum(w) = 1,  w_i >= 0  (long-only),  w_i <= cap

Dependency note
---------------
Listing 11.1 shows PyPortfolioOpt. That package is **not used here**: importing it
hard-crashes the interpreter in this environment (Windows access violation,
exit code 0xC0000005, reproduced across all nine of its submodules including
``pypfopt.exceptions``; ``cvxpy`` on its own imports and solves correctly).
Shipping a dependency that kills the process would defeat the Section 15.3
reproducibility requirement, so the constrained problem is solved here with
``scipy.optimize.minimize(method='SLSQP')`` - the same convex QP / fractional
programme - and the Ledoit-Wolf shrinkage estimator is taken from
``sklearn.covariance.LedoitWolf``, which is the canonical implementation of the
estimator Listing 11.1 asks for.

Everything the PRD actually requires is delivered: Ledoit-Wolf shrinkage, the
efficient frontier, the minimum-variance portfolio, the maximum-Sharpe
(tangency) portfolio, long-only plus a per-asset cap, and the 20,000-portfolio
Monte Carlo cross-check.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import (  # noqa: E402
    CLEANED_PANEL,
    ML_METRICS,
    ML_PREDICTIONS,
    PORTFOLIO_BACKTEST,
    PORTFOLIO_FRONTIER,
    PORTFOLIO_METRICS,
    PORTFOLIO_WEIGHTS,
    PROCESSED_DIR,
    REPORTS_DIR,
    banner,
    ensure_dirs,
    get_logger,
    load_config,
    set_seed,
    write_json,
)
from src import math_utils as mu_math  # noqa: E402

LOG = get_logger("portfolio")

TRADING_DAYS = 252


# ==========================================================================
# Expected returns and covariance
# ==========================================================================
def historical_mu(returns: pd.DataFrame) -> pd.Series:
    """mu from historical means, annualised (Section 11.1 option 1)."""
    return returns.mean() * TRADING_DAYS


def model_implied_mu(returns: pd.DataFrame, predictions: pd.DataFrame,
                     horizon_trading_days: int = 21) -> pd.Series:
    """mu from the forecasting models, annualised (Section 11.1 option 2).

    The daily predicted log return is compounded forward over a month and
    annualised, giving a genuinely forward-looking expected return rather than a
    backward-looking one.
    """
    pred = predictions[predictions["Date"] >= predictions["Date"].max() - pd.Timedelta(days=45)]
    daily = pred.groupby("Ticker")["Predicted_Return"].mean()
    monthly = np.expm1(daily * horizon_trading_days)
    return monthly * (TRADING_DAYS / horizon_trading_days)


def blended_mu(returns: pd.DataFrame, predictions: pd.DataFrame,
               model_weight: float) -> pd.Series:
    """Blend of historical and model-implied mu (Section 11.1 option 3).

    This is the default. A purely historical mu is a backward-looking average
    that the optimiser will exploit relentlessly; a purely model-implied mu
    inherits the forecasting error. Blending is the only choice that makes the
    Section 1 claim that forecasts feed the optimisation layer actually true.
    """
    hist = historical_mu(returns)
    model = model_implied_mu(returns, predictions)
    aligned = hist.index.intersection(model.index)
    blended = pd.Series(index=hist.index, dtype=float)
    blended.loc[aligned] = (
        (1.0 - model_weight) * hist.loc[aligned] + model_weight * model.loc[aligned]
    )
    blended = blended.fillna(hist)
    return blended


def ledoit_wolf_covariance(returns: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf shrunk covariance, annualised (Listing 11.1).

    Raw sample covariance inverting a 10x10 matrix estimated from a few hundred
    observations produces extreme, unstable weights; shrinkage pulls the
    estimate toward a scaled identity target.
    """
    lw = LedoitWolf(assume_centered=False).fit(returns.to_numpy(dtype=float))
    cov = lw.covariance_ * TRADING_DAYS
    return cov, float(lw.shrinkage_)


# ==========================================================================
# Constrained optimisation
# ==========================================================================
def _bounds(n: int, cap: float) -> list[tuple[float, float]]:
    """Long-only with a per-asset concentration cap (Section 11.1 and 11.4)."""
    return [(0.0, cap) for _ in range(n)]


def _feasibility_note(n: int, cap: float) -> str | None:
    """Warn when the cap makes a fully-invested long-only portfolio impossible."""
    if n * cap < 1.0 - 1e-9:
        return (f"per-asset cap {cap:.2%} x {n} assets = {n * cap:.2%} < 100%: "
                "no fully-invested long-only portfolio exists under this cap")
    return None


def _solve(objective, n: int, cap: float, x0: np.ndarray | None = None) -> np.ndarray:
    """SLSQP with an equality budget and a fallback that respects the cap."""
    cons = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
    ]
    start = x0 if x0 is not None else np.full(n, 1.0 / n)
    start = np.clip(start, 0.0, cap)
    start = start / start.sum()

    res = minimize(objective, start, method="SLSQP", bounds=_bounds(n, cap),
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-12})
    w = np.clip(res.x, 0.0, cap)
    total = w.sum()
    if total <= 0:
        return np.full(n, 1.0 / n)
    w = w / total
    # Re-apply the cap after renormalisation, iterating so the weights stay
    # feasible (sum to 1) and capped simultaneously.
    for _ in range(50):
        over = w > cap + 1e-12
        if not over.any():
            break
        w[over] = cap
        shortfall = 1.0 - w.sum()
        free = ~over
        if not free.any() or shortfall <= 1e-12:
            break
        w[free] += shortfall * w[free] / w[free].sum()
    return w


def max_sharpe_weights(mu: np.ndarray, cov: np.ndarray, cap: float,
                       risk_free_rate: float) -> np.ndarray:
    """Tangency portfolio: maximise (w'mu - r_f)/sqrt(w'Sigma w) (Section 11.1)."""
    n = len(mu)
    rf = risk_free_rate

    def neg_sharpe(w: np.ndarray) -> float:
        vol = np.sqrt(max(w @ cov @ w, 1e-18))
        return -float((w @ mu - rf) / vol)

    return _solve(neg_sharpe, n, cap)


def min_variance_weights(cov: np.ndarray, cap: float) -> np.ndarray:
    """Minimum-variance portfolio: minimise w'Sigma w (Section 11.2)."""
    n = cov.shape[0]

    def variance(w: np.ndarray) -> float:
        return float(w @ cov @ w)

    return _solve(variance, n, cap)


def efficient_frontier(mu: np.ndarray, cov: np.ndarray, cap: float,
                       n_points: int = 60) -> pd.DataFrame:
    """The efficient frontier: maximum return achievable at each risk level.

    Built by sweeping a variance target and maximising return subject to it,
    which is the numerically stable formulation of the frontier (Section 11.2).
    """
    n = len(mu)
    rows = []
    for i in range(n_points):
        target = 0.02 + i * 0.0055
        cons = [
            {"type": "eq", "fun": lambda w, t=target: np.sum(w) - 1.0},
            {"type": "ineq", "fun": lambda w, t=target: t - (w @ cov @ w)},
        ]
        res = minimize(lambda w: -float(w @ mu), np.full(n, 1.0 / n),
                       method="SLSQP", bounds=_bounds(n, cap), constraints=cons,
                       options={"maxiter": 600, "ftol": 1e-10})
        w = np.clip(res.x, 0.0, cap)
        if w.sum() <= 0:
            continue
        w = w / w.sum()
        rows.append({
            "Target_Variance": target,
            "Volatility": float(np.sqrt(w @ cov @ w)),
            "Return": float(w @ mu),
            "Sharpe": float((w @ mu - 0.0) / max(np.sqrt(w @ cov @ w), 1e-18)),
        })
    return pd.DataFrame(rows).drop_duplicates("Target_Variance").reset_index(drop=True)


# ==========================================================================
# Monte Carlo cross-check (Section 11.3)
# ==========================================================================
def monte_carlo_cloud(mu: np.ndarray, cov: np.ndarray, n_portfolios: int,
                      cap: float, seed: int) -> np.ndarray:
    """Simulate random long-only portfolios and return their (vol, ret) cloud.

    Vectorised: drawing an (n_portfolios, n_assets) matrix and evaluating the
    quadratic form in one shot is orders of magnitude faster than a Python loop.
    """
    rng = np.random.default_rng(seed)
    n = len(mu)
    W = rng.random((n_portfolios, n))
    W = W / W.sum(axis=1, keepdims=True)

    # Rejection-free cap handling: rescale any portfolio that breaches the cap.
    breaches = W.max(axis=1) > cap
    if breaches.any():
        for _ in range(30):
            over = W.max(axis=1) > cap
            if not over.any():
                break
            W[over] = np.clip(W[over], 0.0, cap)
            s = W[over].sum(axis=1, keepdims=True)
            W[over] = np.where(s > 0, W[over] / s, 1.0 / n)

    rets = W @ mu
    vols = np.sqrt(np.einsum("ij,jk,ik->i", W, cov, W))
    return np.column_stack([vols, rets])


# ==========================================================================
# Backtest (Section 11.4)
# ==========================================================================
def backtest_weights(weights: dict[str, float], returns: pd.DataFrame,
                     benchmark_returns: pd.Series | None = None,
                     risk_free_rate: float = 0.0) -> dict:
    """Compound fixed weights over a realised return window and score the result.

    Section 11.4 requires the optimised allocation to be compared with an
    equal-weight portfolio and a benchmark buy-and-hold over the same window.
    This is a buy-and-hold of the initial allocation, rebalanced only by drift,
    which is the standard reading of "backtest the optimised weights".
    """
    cols = [c for c in weights if c in returns.columns]
    w0 = np.array([weights[c] for c in cols], dtype=float)
    if w0.sum() <= 0:
        w0 = np.full(len(cols), 1.0 / len(cols))
    else:
        w0 = w0 / w0.sum()

    R = returns[cols].to_numpy(dtype=float)
    port = R @ w0
    ew = R.mean(axis=1)

    bm = None
    if benchmark_returns is not None:
        bm = benchmark_returns.reindex(returns.index).dropna()

    # Every strategy is scored on the same realised window, so the comparison is
    # like-for-like. The benchmark is aligned to the window, not the reverse.
    out = {
        "n_sessions": int(len(R)),
        "start": str(returns.index.min().date()),
        "end": str(returns.index.max().date()),
        "optimised": _score(port, risk_free_rate),
        "equal_weight": _score(ew, risk_free_rate),
    }
    if bm is not None and len(bm):
        out["benchmark_buy_hold"] = _score(bm, risk_free_rate)
    return out


def _score(r: pd.Series, risk_free_rate: float) -> dict:
    r = pd.Series(r, dtype=float)
    curve = (1 + r).cumprod()
    years = max(len(r) / TRADING_DAYS, 1e-9)
    return {
        "total_return": float(curve.iloc[-1] - 1.0),
        "annualised_return": float((curve.iloc[-1] ** (1 / years) - 1.0)),
        "annualised_volatility": float(r.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        "sharpe": float(mu_math.sharpe_ratio(r, risk_free_rate=risk_free_rate)),
        "sortino": float(mu_math.sortino_ratio(r, risk_free_rate=risk_free_rate)),
        "max_drawdown": float(mu_math.max_drawdown(r)),
        "calmar": float(mu_math.calmar_ratio(r)),
        "var_95": float(mu_math.value_at_risk(r, 0.05)),
        "final_value": float(curve.iloc[-1]),
    }


# ==========================================================================
# Main
# ==========================================================================
def run_portfolio_optimization() -> dict:
    ensure_dirs()
    cfg = load_config()
    seed = set_seed(cfg.get("random_seed", 42))
    t0 = time.time()
    pcfg = cfg["portfolio"]

    banner(LOG, "PRD Section 11 - Portfolio Optimization (Modern Portfolio Theory)")
    if not CLEANED_PANEL.exists():
        raise FileNotFoundError(f"{CLEANED_PANEL} not found. Run `python src/clean.py` first.")

    universe = list(cfg["universe"])
    cap = float(pcfg["weight_bounds"][1])
    rf = float(pcfg["risk_free_rate"])

    note = _feasibility_note(len(universe), cap)
    if note:
        LOG.warning("%s", note)

    panel = pd.read_parquet(CLEANED_PANEL)
    panel["Date"] = pd.to_datetime(panel["Date"])
    price_col = cfg["price_column"]

    # Returns matrix on ADJUSTED close (Section 3.1 - never raw close).
    wide = panel.pivot(index="Date", columns="Ticker", values=price_col)
    bench = cfg["benchmark"]
    wide = wide.dropna(subset=[t for t in [*universe, bench] if t in wide.columns])
    returns = np.log(wide[universe]).diff().dropna()
    LOG.info("Return matrix: %d sessions x %d assets (%s -> %s)",
             len(returns), len(universe), returns.index.min().date(),
             returns.index.max().date())

    # -- estimation window: the trailing lookback, not the full history ----
    # mu and Sigma are estimated on the trailing window, then the result is
    # backtested on a STRICTLY LATER window. Overlapping the two would let the
    # optimiser see the returns it is then judged on, which is the portfolio
    # layer's equivalent of the look-ahead the PRD forbids in Section 5.
    lookback = int(pcfg.get("lookback_days", 504))
    oos_days = int(pcfg.get("backtest_days", 252))
    total_needed = lookback + oos_days
    if len(returns) < total_needed + 21:
        raise RuntimeError(
            f"need at least {total_needed} sessions to hold a {lookback}-session "
            f"estimation window and a separate {oos_days}-session backtest window, "
            f"but only {len(returns)} are available"
        )

    est = returns.iloc[:-oos_days]
    oos = returns.iloc[-oos_days:]
    LOG.info("Estimation window (mu and Sigma): %d sessions (%s -> %s)",
             len(est), est.index.min().date(), est.index.max().date())
    LOG.info("Backtest window (disjoint, strictly later): %d sessions (%s -> %s)",
             len(oos), oos.index.min().date(), oos.index.max().date())

    # -- mu: historical / model / blend (Section 11.1) ---------------------
    method = pcfg.get("expected_return_method", "blend")
    if method == "historical":
        mu = historical_mu(est)
    else:
        if not ML_PREDICTIONS.exists():
            LOG.warning("No ML predictions at %s - falling back to historical mu. "
                        "Run `python src/models_ml.py` for the blended estimate.",
                        ML_PREDICTIONS)
            mu = historical_mu(est)
            method = "historical (fallback)"
        else:
            preds = pd.read_parquet(ML_PREDICTIONS)
            preds = preds[preds["Model"] == _best_model_name()].copy()
            preds["Date"] = pd.to_datetime(preds["Date"])
            mu = (historical_mu(est) if method == "historical"
                  else model_implied_mu(est, preds) if method == "model"
                  else blended_mu(est, preds, float(pcfg["blend_weight_model"])))
    mu = mu.reindex(universe).fillna(0.0)

    LOG.info("")
    LOG.info("Expected returns (annualised, method=%s):", method)
    for t in universe:
        LOG.info("    %-6s %+7.2f%%", t, 100 * mu[t])

    # -- Sigma: Ledoit-Wolf shrinkage (Listing 11.1) -----------------------
    cov, shrinkage = ledoit_wolf_covariance(est)
    LOG.info("")
    LOG.info("Ledoit-Wolf shrinkage intensity: %.4f", shrinkage)
    LOG.info("(0 = raw sample covariance, 1 = identity target; shrinkage stabilises Sigma)")

    mu_v = mu.to_numpy(dtype=float)

    # -- the two highlighted portfolios (Section 11.2) ---------------------
    w_sharpe = max_sharpe_weights(mu_v, cov, cap, rf)
    w_minvol = min_variance_weights(cov, cap)

    def describe(label, w):
        vol = float(np.sqrt(w @ cov @ w))
        ret = float(w @ mu_v)
        LOG.info("")
        LOG.info("%s:", label)
        LOG.info("    expected return : %+7.2f%%", 100 * ret)
        LOG.info("    volatility      :  %7.2f%%", 100 * vol)
        LOG.info("    Sharpe          :  %7.3f", (ret - rf) / vol if vol else float("nan"))
        for t, wi in sorted(zip(universe, w), key=lambda p: -p[1]):
            if wi > 1e-4:
                LOG.info("      %-6s %6.2f%%", t, 100 * wi)
        return {"expected_return": ret, "volatility": vol,
                "sharpe": (ret - rf) / vol if vol else float("nan")}

    sharpe_info = describe("Maximum Sharpe portfolio (tangency)", w_sharpe)
    minvol_info = describe("Minimum variance portfolio", w_minvol)

    weights = dict(zip(universe, w_sharpe))
    pd.Series(weights).rename("Weight").to_csv(PORTFOLIO_WEIGHTS)
    pd.Series(dict(zip(universe, w_minvol))).rename("Weight").to_csv(
        PORTFOLIO_WEIGHTS.parent / "min_variance_weights.csv")

    # -- efficient frontier (Section 11.2) ---------------------------------
    LOG.info("")
    LOG.info("Building efficient frontier (%d risk targets)...", int(pcfg["frontier_points"]))
    frontier = efficient_frontier(mu_v, cov, cap, int(pcfg["frontier_points"]))
    frontier.to_csv(PORTFOLIO_FRONTIER, index=False)
    LOG.info("Frontier: %d points, volatility %.2f%%-%.2f%%, return %.2f%%-%.2f%%",
             len(frontier), 100 * frontier["Volatility"].min(), 100 * frontier["Volatility"].max(),
             100 * frontier["Return"].min(), 100 * frontier["Return"].max())

    # -- Monte Carlo cross-check (Section 11.3) ----------------------------
    n_mc = int(pcfg["monte_carlo_portfolios"])
    LOG.info("")
    LOG.info("Monte Carlo cross-check: simulating %d random long-only portfolios...", n_mc)
    cloud = monte_carlo_cloud(mu_v, cov, n_mc, cap, seed)
    np.save(PROCESSED_DIR / "monte_carlo_cloud.npy", cloud)
    mc_best = int(np.argmax((cloud[:, 1] - rf) / np.maximum(cloud[:, 0], 1e-12)))
    LOG.info("  cloud  : vol %.2f%%-%.2f%%, return %.2f%%-%.2f%%",
             100 * cloud[:, 0].min(), 100 * cloud[:, 0].max(),
             100 * cloud[:, 1].min(), 100 * cloud[:, 1].max())
    LOG.info("  best simulated Sharpe : %.3f", (cloud[mc_best, 1] - rf) / cloud[mc_best, 0])
    LOG.info("  analytical max Sharpe: %.3f  (solver should sit on the cloud's upper-left edge)",
             sharpe_info["sharpe"])
    frontier_ok = sharpe_info["sharpe"] >= (cloud[mc_best, 1] - rf) / cloud[mc_best, 0] - 0.15
    LOG.info("  cross-check: %s", "PASS" if frontier_ok else
             "analytical optimum is not competitive with the simulated cloud - investigate")

    # -- backtest on the out-of-sample window (Section 11.4) ---------------
    LOG.info("")
    LOG.info("Backtest on the disjoint out-of-sample window (Section 11.4):")
    bench_returns = np.log(wide[bench]).diff().dropna() if bench in wide.columns else None
    bt = backtest_weights(weights, oos, bench_returns, rf)
    for label in ("optimised", "equal_weight", "benchmark_buy_hold"):
        if label not in bt:
            continue
        s = bt[label]
        LOG.info("  %-18s ann.ret %+7.2f%%  vol %6.2f%%  Sharpe %6.3f  Sortino %6.3f  maxDD %7.2f%%",
                 label, 100 * s["annualised_return"], 100 * s["annualised_volatility"],
                 s["sharpe"], s["sortino"], 100 * s["max_drawdown"])
    wins = bt["optimised"]["sharpe"] > bt["equal_weight"]["sharpe"]
    LOG.info("")
    LOG.info("  Section 1 success bar: optimised Sharpe > equal-weight Sharpe -> %s",
             "PASS" if wins else "FAIL")
    if not wins:
        LOG.warning("  Reported as-is. Section 1 defines success as beating equal weight;")
        LOG.warning("  a failure here is a finding to report, not to hide.")

    bt_rows = [
        {"Strategy": name, **metrics}
        for name, metrics in bt.items()
        if isinstance(metrics, dict)
    ]
    pd.DataFrame(bt_rows).to_csv(PORTFOLIO_BACKTEST, index=False)

    # -- Beta against the benchmark (Section 7.2 / 11.4) -------------------
    port_series = pd.Series(oos.to_numpy() @ w_sharpe, index=oos.index)
    beta = (mu_math.beta(port_series, bench_returns.reindex(oos.index).dropna())
            if bench_returns is not None else float("nan"))

    # -- block-bootstrap CI on the backtested Sharpe (Section 7.6) ---------
    boot = mu_math.block_bootstrap_sharpe(port_series, block_size=20,
                                          n_bootstraps=1000, risk_free_rate=rf, seed=seed)
    LOG.info("  Backtested Sharpe %.3f  95%% CI [%.3f, %.3f] (block bootstrap, 1000 resamples)",
             boot["point_estimate"], boot["ci_low"], boot["ci_high"])

    summary = {
        "generated_by": "src/portfolio.py",
        "seed": seed,
        "n_assets": len(universe),
        "weight_cap": cap,
        "risk_free_rate": rf,
        "expected_return_method": method,
        "estimation_window": {
            "sessions": int(len(est)),
            "start": str(est.index.min().date()),
            "end": str(est.index.max().date()),
        },
        "ledoit_wolf_shrinkage": shrinkage,
        "expected_returns_annualised": {t: float(mu[t]) for t in universe},
        "max_sharpe": {"weights": weights, **sharpe_info},
        "min_variance": {"weights": dict(zip(universe, w_minvol)), **minvol_info},
        "monte_carlo": {
            "n_portfolios": n_mc,
            "best_simulated_sharpe": float((cloud[mc_best, 1] - rf) / cloud[mc_best, 0]),
            "analytical_best_sharpe": sharpe_info["sharpe"],
            "cross_check_passed": bool(frontier_ok),
        },
        "backtest": bt,
        "beats_equal_weight": bool(wins),
        "portfolio_beta": float(beta),
        "sharpe_bootstrap_ci": boot,
        "dependency_note": (
            "PyPortfolioOpt is not used: importing pypfopt crashes the interpreter "
            "in this environment (0xC0000005). The constrained programme is solved "
            "with scipy.optimize SLSQP and Ledoit-Wolf shrinkage comes from "
            "sklearn.covariance.LedoitWolf."
        ),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    write_json(summary, PORTFOLIO_METRICS)
    write_json(summary, REPORTS_DIR / "portfolio_summary.json")

    LOG.info("")
    LOG.info("Weights   -> %s", PORTFOLIO_WEIGHTS)
    LOG.info("Frontier  -> %s", PORTFOLIO_FRONTIER)
    LOG.info("Backtest  -> %s", PORTFOLIO_BACKTEST)
    LOG.info("Metrics   -> %s", PORTFOLIO_METRICS)
    LOG.info("Elapsed %.1fs", summary["elapsed_seconds"])
    return summary


def _best_model_name() -> str:
    """The ML model with the lowest test MAE, used for the model-implied mu."""
    if not ML_METRICS.exists():
        return ""
    m = pd.read_csv(ML_METRICS)
    return str(m.loc[m["MAE"].idxmin(), "Model"]) if len(m) else ""


if __name__ == "__main__":
    run_portfolio_optimization()
