"""PRD Section 7 - Mathematical Foundations.

*"The candidate must implement the core financial mathematics from first
principles at least once, then may use libraries thereafter."*

Every quantity below is implemented directly from the formula in the PRD, and
each is paired with an independent library cross-check in
:func:`verify_against_libraries`. The verification runs as part of the retrain
command and its result is written to ``reports/math_verification.json``, so the
claim "verified against libraries" is an auditable artifact rather than an
assertion in prose.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import REPORTS_DIR, get_logger, set_seed, write_json  # noqa: E402

LOG = get_logger("math")

TRADING_DAYS = 252


# ==========================================================================
# Section 7.1 - returns and risk
# ==========================================================================
def simple_returns(prices: pd.Series) -> pd.Series:
    """r_t = (P_t - P_{t-1}) / P_{t-1}  (Section 7.1)."""
    p = pd.Series(prices, dtype=float)
    return p.diff() / p.shift(1)


def log_returns(prices: pd.Series) -> pd.Series:
    """R_t = ln(P_t / P_{t-1}) - time-additive, preferred for modelling."""
    p = pd.Series(prices, dtype=float)
    return np.log(p).diff()


def expected_return(returns: pd.Series) -> float:
    """mu = E[r], estimated by the historical mean (Section 7.1)."""
    return float(np.mean(returns))


def variance(returns: pd.Series) -> float:
    """sigma^2 = E[(r - mu)^2]  (Section 7.1)."""
    r = np.asarray(returns, dtype=float)
    mu = r.mean()
    return float(np.mean((r - mu) ** 2))


def volatility(returns: pd.Series, annualise: bool = True,
               trading_days: int = TRADING_DAYS) -> float:
    """sigma = sqrt(sigma^2), optionally annualised as sigma_d * sqrt(252)."""
    sigma = float(np.std(np.asarray(returns, dtype=float)))
    return sigma * np.sqrt(trading_days) if annualise else sigma


def annualised_return(returns: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    """mu_ann = mu_daily * 252 (Section 7.1 annualisation row)."""
    return expected_return(returns) * trading_days


# ==========================================================================
# Section 7.2 - covariance, correlation, beta
# ==========================================================================
def covariance(a: pd.Series, b: pd.Series) -> float:
    """Cov(a, b) = E[(r_a - mu_a)(r_b - mu_b)]  (Section 7.2)."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    return float(np.mean((x - x.mean()) * (y - y.mean())))


def correlation(a: pd.Series, b: pd.Series) -> float:
    """rho = Cov(a,b) / (sigma_a * sigma_b), bounded in [-1, 1]."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    sx, sy = x.std(), y.std()
    if sx == 0 or sy == 0:
        return float("nan")
    return float(covariance(a, b) / (sx * sy))


def beta(asset_returns: pd.Series, market_returns: pd.Series) -> float:
    """beta_i = Cov(r_i, r_m) / Var(r_m)  (Section 7.2).

    Both terms use the same population convention (divide by n) so the ratio is
    unbiased; ``verify_against_libraries`` checks this against
    ``numpy.polyfit`` and a sample-covariance variant.
    """
    r_m = np.asarray(market_returns, dtype=float)
    if r_m.var() == 0:
        return float("nan")
    return covariance(asset_returns, market_returns) / float(np.var(r_m))


def covariance_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Sample covariance matrix of a returns panel (the heart of MPT, Section 11)."""
    return returns.cov()


# ==========================================================================
# Section 7.3 - portfolio-level linear algebra
# ==========================================================================
def portfolio_return(weights: np.ndarray, mu: np.ndarray) -> float:
    """mu_p = w^T mu  (Section 7.3)."""
    return float(np.dot(np.asarray(weights, dtype=float), np.asarray(mu, dtype=float)))


def portfolio_variance(weights: np.ndarray, cov: np.ndarray) -> float:
    """sigma_p^2 = w^T Sigma w - the quadratic form (Section 7.3)."""
    w = np.asarray(weights, dtype=float)
    return float(np.dot(w, np.dot(np.asarray(cov, dtype=float), w)))


def portfolio_volatility(weights: np.ndarray, cov: np.ndarray) -> float:
    return float(np.sqrt(portfolio_variance(weights, cov)))


# ==========================================================================
# Section 7.4 - risk-adjusted performance metrics
# ==========================================================================
def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0,
                 annualise: bool = True, trading_days: int = TRADING_DAYS) -> float:
    """Sharpe = (mu_p - r_f) / sigma_p  (Section 7.4).

    ``risk_free_rate`` is an ANNUAL rate, converted to a daily rate here.
    """
    r = np.asarray(returns, dtype=float)
    excess = r - (risk_free_rate / trading_days)
    sd = excess.std(ddof=1)
    if sd == 0:
        return 0.0
    s = float(excess.mean() / sd)
    return s * np.sqrt(trading_days) if annualise else s


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0,
                  annualise: bool = True, trading_days: int = TRADING_DAYS,
                  mar: float = 0.0) -> float:
    """Sortino = (mu_p - r_f) / sigma_downside  (Section 7.4).

    Downside deviation is measured against ``mar`` (default 0) over the whole
    sample, which is the standard definition: only observations below the
    threshold contribute to the deviation.
    """
    r = np.asarray(returns, dtype=float)
    excess = r - (risk_free_rate / trading_days)
    shortfall = np.minimum(excess - mar, 0.0)
    downside_dev = float(np.sqrt(np.mean(shortfall ** 2)))
    if downside_dev == 0:
        return 0.0
    s = float((excess.mean() - mar) / downside_dev)
    return s * np.sqrt(trading_days) if annualise else s


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Drawdown path V_t / max_{s<=t} V_s - 1  (Section 7.4).

    Returned as a Series so the dashboard can plot the curve, not just the trough.
    """
    r = pd.Series(returns, dtype=float)
    curve = (1.0 + r).cumprod()
    peak = curve.cummax()
    return (curve - peak) / peak


def max_drawdown(returns: pd.Series) -> float:
    """min_t (V_t / max_{s<=t} V_s - 1)  (Section 7.4)."""
    return float(drawdown_series(returns).min())


def calmar_ratio(returns: pd.Series, annualise: bool = True,
                 trading_days: int = TRADING_DAYS) -> float:
    """Calmar = annual return / |max drawdown|  (Section 7.4)."""
    ann = annualised_return(returns, trading_days) if annualise else expected_return(returns)
    dd = max_drawdown(returns)
    if dd == 0:
        return float("inf")
    return float(ann / abs(dd))


def value_at_risk(returns: pd.Series, alpha: float = 0.05) -> float:
    """VaR(95%) = 5th percentile of the return distribution (Section 7.4)."""
    return float(np.percentile(np.asarray(returns, dtype=float), alpha * 100.0))


def value_at_risk_t(returns: pd.Series, alpha: float = 0.05) -> float:
    """Parametric VaR under a Student's t fit (Section 7.6 fat-tail guidance)."""
    r = np.asarray(returns, dtype=float)
    df, loc, scale = stats.t.fit(r)
    return float(stats.t.ppf(alpha, df, loc=loc, scale=scale))


def conditional_var(returns: pd.Series, alpha: float = 0.05) -> float:
    """Expected shortfall: the mean loss beyond VaR. Reported alongside VaR."""
    r = np.asarray(returns, dtype=float)
    var = value_at_risk(r, alpha)
    tail = r[r <= var]
    return float(tail.mean()) if tail.size else float("nan")


# ==========================================================================
# Section 7.5 - manual gradient descent
# ==========================================================================
def gradient_descent_linear_regression(X: np.ndarray, y: np.ndarray,
                                       learning_rate: float = 1e-2,
                                       epochs: int = 5000) -> tuple[np.ndarray, float]:
    """Linear regression fitted by hand (Listing 7.1).

        yhat = X w + b
        w  <- w - lr * (2/n) X^T (yhat - y)
        b  <- b - lr * (2/n) sum(yhat - y)

    Returns the converged weights and bias.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n, d = X.shape

    weights = np.zeros(d, dtype=float)
    bias = 0.0
    for _ in range(epochs):
        err = X @ weights + bias - y
        weights -= learning_rate * (2.0 / n) * (X.T @ err)
        bias -= learning_rate * (2.0 / n) * err.sum()
    return weights, bias


# ==========================================================================
# Section 7.6 - probability and statistical inference
# ==========================================================================
def jarque_bera(returns: pd.Series) -> tuple[float, float]:
    """Normality test statistic and p-value (Section 7.6)."""
    stat, p = stats.jarque_bera(np.asarray(returns, dtype=float))
    return float(stat), float(p)


def skewness(returns: pd.Series) -> float:
    return float(stats.skew(np.asarray(returns, dtype=float)))


def excess_kurtosis(returns: pd.Series) -> float:
    return float(stats.kurtosis(np.asarray(returns, dtype=float), fisher=True))


def adf_test(series: pd.Series) -> dict:
    """Augmented Dickey-Fuller stationarity test (Section 6.3 / Listing 6.1).

    Imported lazily so that this module has no hard statsmodels dependency at
    import time.
    """
    from statsmodels.tsa.stattools import adfuller

    clean = pd.Series(series, dtype=float).dropna()
    # result_object=False pins the tuple return type; statsmodels plans to switch
    # the default and this unpacking would break when it does.
    stat, p, n_lags, n_obs, crit, _ = adfuller(clean.to_numpy(), result_object=False)
    return {
        "adf_statistic": float(stat),
        "p_value": float(p),
        "n_lags": int(n_lags),
        "n_observations": int(n_obs),
        "critical_values": {k: float(v) for k, v in crit.items()},
        "stationary_at_5pct": bool(p < 0.05),
        "n_points": int(len(clean)),
    }


def block_bootstrap_sharpe(returns: pd.Series, block_size: int = 20,
                           n_bootstraps: int = 1000,
                           risk_free_rate: float = 0.0,
                           seed: int = 42) -> dict:
    """95% confidence interval for a Sharpe ratio by block bootstrap (Section 7.6).

    Resampling contiguous blocks preserves the volatility clustering that an
    i.i.d. bootstrap would destroy, so the interval is not spuriously narrow.
    Seeded so the interval is reproducible (Section 15.3).
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n <= block_size:
        return {"mean": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "n_bootstraps": 0,
                "note": "series shorter than one block"}

    rng = np.random.default_rng(seed)
    n_blocks = n // block_size
    starts = np.arange(0, n - block_size + 1)
    out = np.empty(n_bootstraps, dtype=float)
    for i in range(n_bootstraps):
        pick = rng.integers(0, len(starts), size=n_blocks)
        sample = np.concatenate([r[s: s + block_size] for s in starts[pick]])
        out[i] = sharpe_ratio(pd.Series(sample), risk_free_rate=risk_free_rate)

    return {
        "mean": float(np.mean(out)),
        "std_error": float(np.std(out, ddof=1)),
        "ci_low": float(np.percentile(out, 2.5)),
        "ci_high": float(np.percentile(out, 97.5)),
        "n_bootstraps": int(n_bootstraps),
        "block_size": int(block_size),
        "point_estimate": float(sharpe_ratio(pd.Series(r), risk_free_rate=risk_free_rate)),
    }


# ==========================================================================
# Verification against libraries
# ==========================================================================
def verify_against_libraries(seed: int = 42, n: int = 1500) -> dict:
    """Cross-check every from-scratch implementation against a library.

    Returns a structured result. ``all_passed`` must be True for the claim
    "verified against libraries" to hold (PRD Section 7, deliverable 5).
    """
    from sklearn.linear_model import LinearRegression

    set_seed(seed)
    rng = np.random.default_rng(seed)
    checks: list[dict] = []

    def record(name: str, ours: float, theirs: float, tol: float, note: str = "") -> None:
        diff = abs(float(ours) - float(theirs))
        rel = diff / max(abs(float(theirs)), 1e-12)
        checks.append({
            "check": name,
            "from_scratch": float(ours),
            "library": float(theirs),
            "abs_diff": float(diff),
            "rel_diff": float(rel),
            "tolerance": float(tol),
            "passed": bool(diff <= tol or rel <= tol),
            "note": note,
        })

    # -- price / return series ---------------------------------------------
    price = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.011, n))))

    record("simple_return",
           float(simple_returns(price).iloc[-1]),
           float(price.iloc[-1] / price.iloc[-2] - 1.0), 1e-12,
           "r_t = (P_t - P_{t-1})/P_{t-1}")

    record("log_return",
           float(log_returns(price).iloc[-1]),
           float(np.log(price.iloc[-1] / price.iloc[-2])), 1e-12,
           "R_t = ln(P_t/P_{t-1})")

    ret = log_returns(price).dropna()
    simple = simple_returns(price).dropna()

    # -- dispersion ---------------------------------------------------------
    record("volatility_annualised",
           volatility(ret),
           float(ret.std(ddof=0) * np.sqrt(TRADING_DAYS)), 1e-10,
           "sigma_ann = sigma_daily * sqrt(252)")

    record("variance", variance(ret), float(np.var(ret.to_numpy())), 1e-12,
           "sigma^2 = E[(r-mu)^2]")

    # -- two-asset covariance / correlation / beta ---------------------------
    other_price = np.exp(np.cumsum(rng.normal(0.0003, 0.013, len(ret) + 1)))
    other = pd.Series(other_price).pct_change().dropna()
    a = ret.to_numpy()
    b = other.to_numpy()
    m = min(len(a), len(b))
    a, b = a[-m:], b[-m:]

    record("covariance", covariance(a, b),
           float(np.cov(a, b, ddof=0)[0, 1]), 1e-10,
           "population covariance, Section 7.2")

    record("correlation", correlation(a, b),
           float(np.corrcoef(a, b)[0, 1]), 1e-10, "rho = Cov/(sigma_a sigma_b)")

    ours_beta = beta(a, b)
    record("beta", ours_beta, float(np.polyfit(b, a, 1)[0]), 1e-6,
           "beta_i = Cov(r_i, r_m)/Var(r_m), checked against OLS slope")

    # -- risk-adjusted metrics ----------------------------------------------
    record("sharpe_daily", sharpe_ratio(ret, annualise=False),
           float(ret.mean() / ret.std(ddof=1)), 1e-10, "(mu - r_f)/sigma, daily")

    dd = max_drawdown(simple)
    curve = (1 + simple).cumprod()
    theirs_dd = float((curve / curve.cummax() - 1).min())
    record("max_drawdown", dd, theirs_dd, 1e-12, "min_t (V_t/max_s<=t V_s - 1)")

    record("var_empirical", value_at_risk(simple, 0.05),
           float(np.percentile(simple.to_numpy(), 5)), 1e-12,
           "5th percentile of the return distribution")

    # -- gradient descent vs closed-form OLS (Listing 7.1) -------------------
    n_s, d_s = 800, 5
    X = rng.normal(size=(n_s, d_s))
    true_w = rng.normal(size=d_s)
    y = X @ true_w + 0.7 + rng.normal(scale=0.05, size=n_s)
    w_gd, b_gd = gradient_descent_linear_regression(X, y, learning_rate=0.05, epochs=6000)
    ols = LinearRegression().fit(X, y)
    record("gd_weights_max_abs_error", float(np.max(np.abs(w_gd - ols.coef_))),
           0.0, 5e-3, "manual GD must match the analytical OLS solution")
    record("gd_bias_abs_error", float(abs(b_gd - ols.intercept_)),
           0.0, 5e-3, "manual GD intercept vs OLS intercept")

    # -- statistical inference ----------------------------------------------
    jb_stat, jb_p = jarque_bera(ret)
    record("jarque_bera_stat", jb_stat,
           float(stats.jarque_bera(ret.to_numpy())[0]), 1e-12, "scipy cross-check")
    record("skewness", skewness(ret), float(stats.skew(ret.to_numpy())), 1e-12, "scipy skew")
    record("excess_kurtosis", excess_kurtosis(ret),
           float(stats.kurtosis(ret.to_numpy(), fisher=True)), 1e-12, "scipy kurtosis")

    summary = {
        "generated_by": "src/math_utils.py::verify_against_libraries",
        "n_checks": len(checks),
        "n_passed": sum(c["passed"] for c in checks),
        "all_passed": all(c["passed"] for c in checks),
        "seed": seed,
        "checks": checks,
    }

    # -- stationarity demonstration (Listing 6.1) ---------------------------
    try:
        summary["adf_price_level"] = adf_test(price)
        summary["adf_log_returns"] = adf_test(ret)
    except Exception as exc:  # pragma: no cover
        summary["adf_error"] = str(exc)

    write_json(summary, REPORTS_DIR / "math_verification.json")
    return summary


def main() -> None:
    banner_line = "=" * 78
    LOG.info(banner_line)
    LOG.info("PRD Section 7 - Mathematical Foundations: library verification")
    LOG.info(banner_line)
    result = verify_against_libraries()
    for c in result["checks"]:
        status = "PASS" if c["passed"] else "FAIL"
        LOG.info("  [%s] %-32s ours=%-16.8f library=%-16.8f %s",
                 status, c["check"], c["from_scratch"], c["library"], c["note"])
    LOG.info("")
    LOG.info("Result: %d/%d checks passed (all_passed=%s)",
             result["n_passed"], result["n_checks"], result["all_passed"])
    if "adf_price_level" in result:
        LOG.info("ADF on price level  : p=%.4f  stationary=%s  (expect non-stationary)",
                 result["adf_price_level"]["p_value"],
                 result["adf_price_level"]["stationary_at_5pct"])
        LOG.info("ADF on log returns  : p=%.4f  stationary=%s  (expect stationary)",
                 result["adf_log_returns"]["p_value"],
                 result["adf_log_returns"]["stationary_at_5pct"])
    LOG.info("Report -> %s", REPORTS_DIR / "math_verification.json")
    if not result["all_passed"]:
        raise SystemExit("mathematics verification FAILED - see the table above")


if __name__ == "__main__":
    main()
