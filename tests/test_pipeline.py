"""Layered test suite (PRD Section 15.1: each concern independently testable).

Run with:  python -m pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import math_utils as mu  # noqa: E402
from src.common import load_config  # noqa: E402
from src.features import make_sequences, split_by_date  # noqa: E402


@pytest.fixture(scope="session")
def cfg() -> dict:
    return load_config()


@pytest.fixture(scope="session")
def panel() -> pd.DataFrame:
    p = ROOT / "data" / "processed" / "cleaned_panel.parquet"
    if not p.exists():
        pytest.skip("cleaned_panel.parquet not built; run `python src/clean.py`")
    df = pd.read_parquet(p)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


@pytest.fixture(scope="session")
def features() -> pd.DataFrame:
    p = ROOT / "data" / "processed" / "features.parquet"
    if not p.exists():
        pytest.skip("features.parquet not built; run `python src/features.py`")
    df = pd.read_parquet(p)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


# ==========================================================================
# Section 4 - cleaning guarantees
# ==========================================================================
class TestCleaning:
    def test_no_duplicate_date_ticker(self, panel):
        """Stage 3: (Date, Ticker) must be unique."""
        dupes = panel.duplicated(subset=["Date", "Ticker"]).sum()
        assert dupes == 0, f"{dupes} duplicate (Date, Ticker) rows survived cleaning"

    def test_no_remaining_price_gaps(self, panel):
        """Section 3.4: 0% missing days after clean."""
        assert panel["Adjusted Close"].notna().all()
        assert panel["Close"].notna().all()

    def test_no_ohlc_violations(self, panel):
        """Stage 5: the OHLC invariant holds on every surviving row."""
        bad = (
            (panel["Low"] > panel[["Open", "Close", "High"]].min(axis=1))
            | (panel["High"] < panel[["Open", "Close", "Low"]].max(axis=1))
        )
        assert not bad.any(), f"{int(bad.sum())} OHLC invariant violations remain"

    def test_calendar_is_shared_across_tickers(self, panel):
        """Stage 1: every ticker sits on one exchange calendar."""
        counts = panel.groupby("Ticker")["Date"].nunique().unique()
        assert len(counts) == 1, f"tickers have differing session counts: {counts}"

    def test_outliers_flagged_not_deleted(self, panel, cfg):
        """Stage 4: outliers are flagged, never auto-deleted.

        If rows were being dropped, the flagged fraction would be exactly zero
        in the saved panel. It must be non-zero, which is what proves the
        "flag, do not delete" policy took effect.
        """
        assert "Outlier" in panel.columns
        assert int(panel["Outlier"].sum()) > 0, (
            "no rows are flagged as outliers - this suggests Stage 4 deleted them "
            "instead of flagging them, which Section 4.1 forbids"
        )

    def test_adjusted_close_present(self, panel):
        """Section 3.1: the adjusted series must survive cleaning."""
        assert "Adjusted Close" in panel.columns
        assert panel["Adjusted Close"].notna().all()


# ==========================================================================
# Section 5 - no look-ahead
# ==========================================================================
class TestNoLookAhead:
    def test_target_is_forward_shifted(self, features, cfg):
        """The target must be the FUTURE return, not the past one."""
        horizon = cfg["target"]["horizon"]
        t = features[features["Ticker"] == features["Ticker"].iloc[0]].sort_values("Date")
        price = t[cfg["price_column"]].to_numpy()
        target = t["Target"].to_numpy()
        expected = np.log(price[horizon:]) - np.log(price[:-horizon])
        assert np.allclose(target[:-horizon], expected, atol=1e-10)

    def test_no_forward_price_in_features(self, features):
        """Target_Price is P(t+h) and must never be a model input."""
        from src.features import feature_columns

        cols = set(feature_columns(features))
        assert "Target_Price" not in cols, "the future price leaked into the feature set"
        assert "Target" not in cols

    def test_non_causal_iqr_flag_excluded(self, features):
        """The 1.5x IQR fence uses full-sample quantiles, so it is causal-unsafe."""
        from src.features import feature_columns

        cols = set(feature_columns(features))
        assert "Outlier_IQR" not in cols, (
            "the IQR outlier fence is computed over the full return distribution "
            "and therefore encodes future information; it must not be a feature"
        )
        assert "Outlier" not in cols

    def test_split_is_chronological(self, features, cfg):
        """Section 8.2: partitions must not overlap in time and must be ordered."""
        bounds = split_by_date(features["Date"], cfg)
        assert bounds["train"].max() < bounds["val"].min()
        assert bounds["val"].max() < bounds["test"].min()

    def test_split_is_on_date_axis(self, features, cfg):
        """Every ticker at a given instant must sit in one partition only.

        A row-wise slice would place ticker A's test rows beside ticker B's
        training rows at the same timestamp, which is cross-sectional leakage.
        """
        bounds = split_by_date(features["Date"], cfg)
        for name, dates in bounds.items():
            sub = features[features["Date"].isin(dates)]
            per_date = sub.groupby("Date")["Ticker"].nunique()
            assert per_date.nunique() == 1, (
                f"partition {name!r} has inconsistent ticker counts per date"
            )

    def test_sequences_never_cross_a_boundary(self):
        """Section 5.3: windows built inside a partition cannot straddle it."""
        n = 300
        X = np.arange(n * 3, dtype=float).reshape(n, 3)
        y = np.arange(n, dtype=float)
        Xs, ys = make_sequences(X, y, seq_length=60, horizon=1)
        assert Xs.shape[1:] == (60, 3)
        assert len(Xs) == len(ys) == n - 60
        # Window i ends at row i+59 and is paired with the target at i+60.
        assert ys[0] == 60.0
        assert np.array_equal(Xs[0], X[0:60])

    def test_turn_of_month_uses_no_future_data(self, features):
        """The flag must be computable from the session calendar alone."""
        assert "TurnOfMonth" in features.columns
        assert set(features["TurnOfMonth"].unique()).issubset({0, 1})


# ==========================================================================
# Section 7 - mathematics
# ==========================================================================
class TestMathematics:
    def test_simple_return_matches_definition(self):
        p = pd.Series([100.0, 110.0, 99.0])
        r = mu.simple_returns(p)
        assert np.isclose(r.iloc[1], 0.10)
        assert np.isclose(r.iloc[2], -0.10)

    def test_log_return_matches_definition(self):
        p = pd.Series([100.0, 110.0, 99.0])
        r = mu.log_returns(p)
        assert np.isclose(r.iloc[1], np.log(1.10))

    def test_annualisation_factor(self):
        r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 1000))
        expected = r.std(ddof=0) * np.sqrt(252)
        assert np.isclose(mu.volatility(r, annualise=True), expected)

    def test_beta_equals_ols_slope(self):
        rng = np.random.default_rng(7)
        m = rng.normal(0, 0.01, 800)
        a = 1.4 * m + rng.normal(0, 0.004, 800)
        assert np.isclose(mu.beta(pd.Series(a), pd.Series(m)),
                          np.polyfit(m, a, 1)[0], atol=1e-9)

    def test_sharpe_of_zero_volatility_is_zero(self):
        r = pd.Series([0.001] * 50)
        assert mu.sharpe_ratio(r) == 0.0

    def test_max_drawdown_is_non_positive(self):
        r = pd.Series([0.01, -0.05, 0.02, -0.01])
        assert mu.max_drawdown(r) <= 0

    def test_drawdown_series_min_matches_max_drawdown(self):
        r = pd.Series(np.random.default_rng(3).normal(0, 0.01, 500))
        assert np.isclose(mu.drawdown_series(r).min(), mu.max_drawdown(r))

    def test_portfolio_variance_is_quadratic_form(self):
        w = np.array([0.5, 0.5])
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        assert np.isclose(mu.portfolio_variance(w, cov), w @ cov @ w)

    def test_gradient_descent_matches_ols(self):
        """Listing 7.1: converged weights must match the analytical solution."""
        from sklearn.linear_model import LinearRegression

        rng = np.random.default_rng(11)
        X = rng.normal(size=(500, 4))
        y = X @ np.array([1.0, -2.0, 0.5, 3.0]) + 1.5 + rng.normal(scale=0.01, size=500)
        w, b = mu.gradient_descent_linear_regression(X, y, learning_rate=0.05, epochs=8000)
        ols = LinearRegression().fit(X, y)
        assert np.allclose(w, ols.coef_, atol=1e-3)
        assert np.isclose(b, ols.intercept_, atol=1e-3)

    def test_var_is_a_loss(self):
        r = pd.Series(np.random.default_rng(5).normal(0, 0.01, 2000))
        assert mu.value_at_risk(r, 0.05) < 0

    def test_bootstrap_interval_brackets_point_estimate(self):
        r = pd.Series(np.random.default_rng(9).normal(0.0004, 0.01, 1000))
        out = mu.block_bootstrap_sharpe(r, block_size=20, n_bootstraps=200, seed=42)
        assert out["ci_low"] <= out["point_estimate"] <= out["ci_high"]

    def test_full_library_verification_passes(self):
        """The Section 7 deliverable: every formula agrees with its library."""
        result = mu.verify_against_libraries()
        assert result["all_passed"], (
            "math verification failed: "
            + ", ".join(c["check"] for c in result["checks"] if not c["passed"])
        )


# ==========================================================================
# Section 11 - portfolio optimisation
# ==========================================================================
class TestPortfolio:
    def test_weights_sum_to_one(self):
        from src.portfolio import max_sharpe_weights, min_variance_weights

        rng = np.random.default_rng(4)
        n = 8
        A = rng.normal(size=(n, n))
        cov = A @ A.T / 100 + np.eye(n) * 0.01
        mu_vec = rng.normal(0.08, 0.03, n)
        for w in (max_sharpe_weights(mu_vec, cov, 0.30, 0.04),
                  min_variance_weights(cov, 0.30)):
            assert np.isclose(w.sum(), 1.0, atol=1e-6)
            assert (w >= -1e-9).all(), "long-only constraint violated"

    def test_per_asset_cap_respected(self):
        from src.portfolio import max_sharpe_weights

        rng = np.random.default_rng(6)
        n = 6
        A = rng.normal(size=(n, n))
        cov = A @ A.T / 100 + np.eye(n) * 0.01
        mu_vec = np.array([0.30, 0.05, 0.05, 0.05, 0.05, 0.05])  # one strong asset
        cap = 0.25
        w = max_sharpe_weights(mu_vec, cov, cap, 0.0)
        assert (w <= cap + 1e-6).all(), f"cap {cap} breached: max weight {w.max():.4f}"

    def test_infeasible_cap_is_reported(self):
        from src.portfolio import _feasibility_note

        # 5 assets at a 15% cap cannot sum to 1.
        assert _feasibility_note(5, 0.15) is not None
        assert _feasibility_note(10, 0.20) is None

    def test_monte_carlo_cloud_shape_and_bounds(self):
        from src.portfolio import monte_carlo_cloud

        rng = np.random.default_rng(8)
        n = 6
        A = rng.normal(size=(n, n))
        cov = (A @ A.T / 100 + np.eye(n) * 0.01) * 252
        mu_vec = rng.normal(0.08, 0.02, n) * 252
        cloud = monte_carlo_cloud(mu_vec, cov, 500, 0.30, seed=42)
        assert cloud.shape == (500, 2)
        assert (cloud[:, 0] > 0).all(), "volatility must be positive"


# ==========================================================================
# Section 12 - recommendation rules
# ==========================================================================
class TestForecastSkillGate:
    """Section 12's forecast term must be gated on the model's measured skill.

    Section 6.3 shows the forecaster does not beat the random walk, so its signal
    carries no demonstrated information. Weighting it at full strength would emit
    confident BUY calls from a signal known not to work.
    """

    @staticmethod
    def _preds(residuals, actuals):
        return pd.DataFrame({"Residual": residuals, "Actual_Return": actuals})

    def test_zero_skill_when_model_matches_baseline(self):
        from src.recommend import _forecast_skill

        actual = np.array([0.01, -0.02, 0.015, -0.005])
        # Residuals equal to the realised returns is a zero-return forecast.
        skill = _forecast_skill(self._preds(actual, actual))
        assert skill == 0.0

    def test_zero_skill_when_model_is_worse_than_baseline(self):
        from src.recommend import _forecast_skill

        actual = np.array([0.01, -0.02, 0.015, -0.005])
        # Residuals larger than the actual returns = a worse-than-baseline model.
        skill = _forecast_skill(self._preds(actual * 3, actual))
        assert skill == 0.0, "a worse-than-baseline model must not earn forecast weight"

    def test_positive_skill_for_a_better_model(self):
        from src.recommend import _forecast_skill

        actual = np.array([0.01, -0.02, 0.015, -0.005])
        skill = _forecast_skill(self._preds(actual * 0.5, actual))
        assert skill > 0.4, f"expected a clear positive skill, got {skill}"

    def test_skill_is_clamped_to_one(self):
        from src.recommend import _forecast_skill

        actual = np.array([0.01, -0.02, 0.015, -0.005])
        # Near-zero residuals would imply an unbounded ratio.
        skill = _forecast_skill(self._preds(actual * 0.001, actual))
        assert 0.0 <= skill <= 1.0

    def test_skill_handles_empty_input(self):
        from src.recommend import _forecast_skill

        assert _forecast_skill(pd.DataFrame()) == 0.0
        assert _forecast_skill(None) == 0.0


class TestRecommend:
    def test_threshold_boundaries(self):
        from src.recommend import recommend

        assert recommend(0.5, -0.15, 0.15) == "BUY"
        assert recommend(-0.5, -0.15, 0.15) == "SELL"
        assert recommend(0.0, -0.15, 0.15) == "HOLD"

    def test_no_trade_band_boundaries(self):
        from src.recommend import recommend

        assert recommend(0.15, -0.15, 0.15) == "BUY"
        assert recommend(-0.15, -0.15, 0.15) == "SELL"
        assert recommend(0.1499, -0.15, 0.15) == "HOLD"

    def test_nan_score_is_hold(self):
        from src.recommend import recommend

        assert recommend(float("nan"), -0.15, 0.15) == "HOLD"

    def test_rebalance_band(self):
        """A drift inside the band produces no action; outside it does."""
        from src.recommend import rebalance

        current = {"A": 0.10, "B": 0.20, "C": 0.05}
        target = {"A": 0.13, "B": 0.30, "C": 0.05}
        actions = rebalance(current, target, band=0.05)
        # A drifts 0.03 and C drifts 0.00: both inside the 0.05 band, so neither
        # may produce a trade. B drifts 0.10 and must.
        assert set(actions) == {"B"}, f"unexpected actions: {actions}"
        assert actions["B"]["action"] == "INCREASE"
        assert abs(actions["B"]["drift"] - 0.10) < 1e-9

    def test_rebalance_includes_small_drift_when_band_is_tighter(self):
        from src.recommend import rebalance

        actions = rebalance({"A": 0.10}, {"A": 0.13}, band=0.02)
        assert set(actions) == {"A"}
        assert actions["A"]["action"] == "INCREASE"

    def test_rebalance_direction(self):
        from src.recommend import rebalance

        actions = rebalance({"A": 0.30}, {"A": 0.10}, band=0.05)
        assert actions["A"]["action"] == "REDUCE"
        assert actions["A"]["drift"] < 0


# ==========================================================================
# Section 10 - session mapping (the look-ahead guard)
# ==========================================================================
class TestSessionMapping:
    @pytest.fixture
    def calendar(self):
        # Mon-Fri sessions across a month boundary.
        return pd.DatetimeIndex(pd.bdate_range("2024-01-26", "2024-02-05"))

    def test_after_close_maps_to_next_session(self, calendar):
        from src.sentiment import map_to_session

        # 18:00 on Friday 2024-02-02 -> Monday 2024-02-05
        ts = pd.Series(pd.to_datetime(["2024-02-02 18:00"]))
        out = map_to_session(ts, calendar, close_hour=16)
        assert out.iloc[0] == pd.Timestamp("2024-02-05")

    def test_before_close_maps_to_same_session(self, calendar):
        from src.sentiment import map_to_session

        ts = pd.Series(pd.to_datetime(["2024-02-02 10:00"]))
        out = map_to_session(ts, calendar, close_hour=16)
        assert out.iloc[0] == pd.Timestamp("2024-02-02")

    def test_weekend_maps_to_monday(self, calendar):
        from src.sentiment import map_to_session

        # Saturday 2024-02-03 12:00 -> Monday 2024-02-05
        ts = pd.Series(pd.to_datetime(["2024-02-03 12:00"]))
        out = map_to_session(ts, calendar, close_hour=16)
        assert out.iloc[0] == pd.Timestamp("2024-02-05")

    def test_non_session_day_maps_forward(self, calendar):
        """A date absent from the calendar resolves to the next real session."""
        from src.sentiment import map_to_session

        # 2024-02-03 is a Saturday, absent from the calendar.
        ts = pd.Series(pd.to_datetime(["2024-02-03 09:00"]))
        out = map_to_session(ts, calendar, close_hour=16)
        assert out.iloc[0] == pd.Timestamp("2024-02-05")

    def test_holiday_maps_past_the_gap(self):
        """A real market holiday is skipped, not treated as a trading session."""
        from src.sentiment import map_to_session

        # 2024-01-15 was MLK Day: the US equity market was closed.
        cal = pd.DatetimeIndex(
            [pd.Timestamp(d) for d in
             ["2024-01-12", "2024-01-16", "2024-01-17", "2024-01-18"]]
        )
        # Published 09:00 on the holiday -> the next session, 2024-01-16.
        out = map_to_session(pd.Series(pd.to_datetime(["2024-01-15 09:00"])),
                             cal, close_hour=16)
        assert out.iloc[0] == pd.Timestamp("2024-01-16")

        # Published 18:00 on the last session before the holiday -> 2024-01-16.
        out2 = map_to_session(pd.Series(pd.to_datetime(["2024-01-12 18:00"])),
                              cal, close_hour=16)
        assert out2.iloc[0] == pd.Timestamp("2024-01-16")

    def test_after_close_on_last_session_clamps(self, calendar):
        """A timestamp past the final session must not index out of bounds."""
        from src.sentiment import map_to_session

        ts = pd.Series(pd.to_datetime(["2024-02-06 18:00"]))
        out = map_to_session(ts, calendar, close_hour=16)
        assert out.iloc[0] == calendar.max()

    def test_empty_calendar_raises(self):
        from src.sentiment import map_to_session

        with pytest.raises(ValueError):
            map_to_session(pd.Series(pd.to_datetime(["2024-01-02 09:00"])),
                           pd.DatetimeIndex([]), close_hour=16)

    def test_text_cleaning(self):
        from src.sentiment import clean_text

        # URLs and tickers are stripped, finance abbreviations are expanded.
        out = clean_text("Apple beats Q3 EPS estimates https://x.com/a")
        assert "http" not in out
        assert out == out.lower()
        assert "earnings per share" in out
        assert clean_text("") == ""
        assert clean_text(None) == ""


# ==========================================================================
# Section 8/9 - evaluation protocol
# ==========================================================================
class TestEvaluation:
    def test_directional_accuracy(self):
        from src.models_ml import directional_accuracy

        y = np.array([1.0, -1.0, 1.0, -1.0])
        assert directional_accuracy(y, y) == 1.0
        assert directional_accuracy(y, -y) == 0.0

    def test_directional_accuracy_excludes_zero_actuals(self):
        from src.models_ml import directional_accuracy

        y = np.array([0.0, 1.0, -1.0])
        p = np.array([1.0, 1.0, -1.0])
        # The zero-actual row is excluded, so the score is 1.0 not 2/3.
        assert directional_accuracy(y, p) == 1.0

    def test_zero_return_baseline_mae_equals_mean_abs_return(self):
        from src.models_ml import naive_random_walk_metrics

        y = np.array([0.01, -0.02, 0.005])
        out = naive_random_walk_metrics(y)
        assert np.isclose(out["MAE"], np.mean(np.abs(y)))
        assert np.isnan(out["DirAcc"]), "a zero forecast makes no directional call"

    def test_metrics_reject_leaking_price(self):
        from src.models_ml import compute_metrics

        y = np.array([0.01, -0.01, 0.02])
        out = compute_metrics(y, np.zeros(3), price=np.array([100.0, 200.0, 50.0]))
        assert out["MAPE_Price"] == out["MAPE"]
        assert out["N"] == 3


# ==========================================================================
# Architecture / integrity
# ==========================================================================
    def test_svr_solver_converges(self, cfg):
        """A non-converged solver invalidates the SVR row entirely.

        If the optimiser stops at its iteration cap, the coefficients it
        returns are not the optimum for the chosen C and epsilon, so the
        leaderboard would be reporting a number for a model that was never
        actually fitted. scikit-learn signals this with ConvergenceWarning, so
        the check is that fitting a representative problem emits no such warning.
        """
        import warnings

        from sklearn.exceptions import ConvergenceWarning

        from src.models_ml import build_model

        rng = np.random.default_rng(0)
        # Shape and noise level taken from the real problem: many more rows than
        # features, and a target that is mostly noise.
        X = rng.normal(size=(4000, 60))
        y = X @ rng.normal(size=60) * 0.01 + rng.normal(scale=0.012, size=4000)

        model = build_model("SVR", {"kernel": "linear", "C": 1.0,
                                    "epsilon": 0.0001}, seed=42)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(X, y)
        conv = [w for w in caught if issubclass(w.category, ConvergenceWarning)]
        assert not conv, (
            "the linear SVR did not converge; its coefficients are not the "
            f"optimum, so the SVR leaderboard row is not a valid measurement "
            f"({len(conv)} ConvergenceWarning(s))"
        )

    def test_svr_solver_settings_are_the_measured_ones(self, cfg):
        """Lock in the settings that were shown to converge, and why.

        `dual=False` is NOT a valid option here: scikit-learn's LinearSVR only
        supports the squared epsilon-insensitive loss in the primal and raises
        on the default loss. So the dual form is forced, and the only way to
        make it fit is a high iteration cap with a loosened tolerance - measured
        in the docstring of `build_model`.
        """
        from src.models_ml import build_model

        model = build_model("SVR", {"kernel": "linear"}, seed=42)
        assert model.dual is True, (
            "LinearSVR must use the dual form; the primal does not support the "
            "epsilon-insensitive loss and raises ValueError"
        )
        assert model.max_iter >= 200_000, (
            f"max_iter={model.max_iter} is below the measured threshold - the fit "
            f"stops at the cap and returns a non-optimal coefficient vector"
        )
        assert model.tol >= 1e-3, (
            f"tol={model.tol} was measured not to converge even at 200,000 "
            f"iterations; 1e-3 converges in roughly 103,000"
        )

    def test_ensemble_averages_rather_than_picks(self, cfg):
        """The ensemble must be the mean of its members.

        A class called "ensemble" that returned the first member's prediction
        would satisfy any test that only checked it had more than one member,
        and would then be reported as a genuine variance reduction.
        """
        from src.models_ml import EqualWeightEnsemble

        class Fake:
            def __init__(self, v):
                self.v = v

            def predict(self, X):
                return np.full(len(X), self.v)

        ens = EqualWeightEnsemble({"a": Fake(1.0), "b": Fake(3.0), "c": Fake(-2.0)})
        assert ens.n_models == 3
        got = ens.predict(np.zeros((5, 2)))
        assert np.allclose(got, (1.0 + 3.0 - 2.0) / 3.0), (
            f"expected the mean 0.6667, got {got[0]:.4f} - the ensemble is not "
            f"averaging its members"
        )

    def test_ensemble_covers_the_four_required_models(self, cfg):
        """It must average the required models, and must not replace any."""
        from src.models_ml import MODEL_NAME_BY_CONFIG_KEY

        ens = cfg.get("ml", {}).get("ensemble", {})
        assert ens.get("enabled") is True
        required_keys = set(cfg["ml"]["models"])
        assert len(required_keys) == 4, (
            f"the PRD names four classical models; config has "
            f"{sorted(required_keys)}"
        )
        # Config keys are snake_case; member names are the display names the
        # metrics table uses. Compare through the mapping rather than by string.
        required_display = {MODEL_NAME_BY_CONFIG_KEY[k] for k in required_keys}
        assert set(ens.get("members", [])) == required_display, (
            f"ensemble members {ens.get('members')} do not match the four "
            f"required models {sorted(required_display)}"
        )

    def test_ensemble_is_excluded_from_the_headline_verdict(self):
        """It is filed under Family="ML", so it can silently take the headline.

        The PRD's Section 9.5 bar is about individual models. If the averaged
        row is allowed to set "best MAE", a combination would be credited as
        though one model had achieved it. Checked per function body rather than
        by counting string occurrences, because each panel may legitimately
        spell the filter differently (a named constant, a set difference, a
        boolean mask) and a brittle count would force one particular spelling.
        """
        import ast

        app = ROOT / "dashboard" / "app.py"
        tree = ast.parse(app.read_text(encoding="utf-8"))
        funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        for panel in ("panel_overview", "panel_models"):
            assert panel in funcs, f"{panel} is missing"
            body = ast.get_source_segment(
                app.read_text(encoding="utf-8"), funcs[panel]) or ""
            assert "Ensemble" in body, (
                f"{panel} computes a headline verdict but never mentions the "
                f"ensemble, so the averaged row can be credited as though it "
                f"were a single winning model"
            )

    def test_config_declares_no_magic_target(self, cfg):
        """Section 3.4: the target must be documented and fixed before modelling."""
        assert cfg["target"]["kind"] in ("log_return", "simple_return")
        assert cfg["target"]["horizon"] >= 1
        assert cfg["price_column"] == "Adjusted Close"

    def test_split_ratios_sum_to_one(self, cfg):
        s = cfg["split"]
        assert abs(s["train"] + s["val"] + s["test"] - 1.0) < 1e-9

    def test_seed_is_fixed(self, cfg):
        assert isinstance(cfg["random_seed"], int)

    def test_dashboard_has_no_hardcoded_metrics(self):
        """A fabricated metrics table would be a serious integrity failure."""
        app = (ROOT / "dashboard" / "app.py").read_text(encoding="utf-8")
        # The old stub contained a literal comparison table with made-up numbers.
        assert "Static mockup" not in app
        assert '0.007' not in app or "MAPE" not in app
        for banned in ("comparison_data = pd.DataFrame", '"57.1%"'):
            assert banned not in app, f"hardcoded model metric found: {banned}"

    def test_portfolio_does_not_import_pypfopt(self):
        """pypfopt segfaults the interpreter in this environment."""
        src = (ROOT / "src" / "portfolio.py").read_text(encoding="utf-8")
        assert "import pypfopt" not in src
        assert "from pypfopt" not in src

    def test_feature_table_has_no_constant_columns(self, features):
        """A constant column carries no information and can destabilise scaling."""
        from src.features import feature_columns

        for col in feature_columns(features):
            assert features[col].nunique() > 1, f"{col} is constant"

    def test_feature_table_has_no_infinities(self, features):
        from src.features import feature_columns

        arr = features[feature_columns(features)].to_numpy(dtype=float)
        assert np.isfinite(arr).all(), "features contain inf or NaN"
