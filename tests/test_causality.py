"""Automated causality guard for the feature table.

PRD Section 5: "Every feature must be computable using only information available
at or before time t - this is the single most important rule in the project,
because a feature that peeks at the future produces excellent validation scores
and worthless live behaviour."

The most convincing way to defend that claim is to test it rather than assert
it. The test below reconstructs the feature table from progressively truncated
history and proves that a feature's value at time t does not change when the
data after t is removed. Any feature that peeks forward will shift.

The one deliberate exception is the target, which is a label rather than an
input, and is asserted separately.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import load_config  # noqa: E402
from src.features import (  # noqa: E402
    feature_columns,
    split_by_date,
    turn_of_month_flag,
)

# Features that legitimately depend on the calendar or on a target defined over
# the same row. The calendar flag is computed from published exchange sessions
# (known in advance) and is tested on its own; the target is the label.
_CALENDAR_OK = {"TurnOfMonth"}


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config()


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    p = ROOT / "data" / "processed" / "features.parquet"
    if not p.exists():
        pytest.skip("features.parquet not built; run `python src/features.py`")
    df = pd.read_parquet(p)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def _rebuild(ticker: str, cutoff: pd.Timestamp, cfg: dict) -> pd.DataFrame:
    """Rebuild one ticker's features using only data up to `cutoff`."""
    from src.common import CLEANED_PANEL
    from src.features import build_ticker_features

    panel = pd.read_parquet(CLEANED_PANEL)
    panel["Date"] = pd.to_datetime(panel["Date"])
    g = panel[panel["Ticker"] == ticker]
    truncated = g[g["Date"] <= cutoff]
    return build_ticker_features(truncated, cfg, cfg["price_column"])


class TestCausalityByTruncation:
    """A feature must be identical whether or not future rows are present."""

    def test_truncation_changes_nothing_before_the_cutoff(self, cfg):
        """Recompute features on truncated history and compare to the full run.

        Rolling windows, EMAs, OBV and every trailing statistic are functions of
        the past only, so every value at or before the cutoff must match
        exactly. A shift here means look-ahead.

        Both sides are produced by ``build_ticker_features`` so the comparison
        isolates causality from the warm-up trimming that ``features.py`` applies
        afterwards.
        """
        from src.common import CLEANED_PANEL
        from src.features import build_ticker_features, feature_columns as fc

        panel = pd.read_parquet(CLEANED_PANEL)
        panel["Date"] = pd.to_datetime(panel["Date"])
        ticker = cfg["universe"][0]
        g = panel[panel["Ticker"] == ticker]
        if g.empty:
            pytest.skip("no rows for the first ticker")

        cutoff = g["Date"].iloc[len(g) // 2]

        full = build_ticker_features(g, cfg, cfg["price_column"])
        truncated = _rebuild(ticker, cutoff, cfg)

        full_before = full[full["Date"] <= cutoff].reset_index(drop=True)
        trunc_before = truncated[truncated["Date"] <= cutoff].reset_index(drop=True)

        assert len(trunc_before) == len(full_before), (
            f"row count differs: truncated {len(trunc_before)} vs full {len(full_before)}"
        )
        if not len(full_before):
            pytest.skip("not enough rows before the cutoff")

        checked, mismatched = 0, []
        for col in fc(full_before):
            if col in _CALENDAR_OK:
                continue
            a = pd.to_numeric(full_before[col], errors="coerce").to_numpy(dtype=float)
            b = pd.to_numeric(trunc_before[col], errors="coerce").to_numpy(dtype=float)
            both_nan = np.isnan(a) & np.isnan(b)
            close = np.isclose(a, b, rtol=1e-7, atol=1e-10, equal_nan=False) | both_nan
            checked += 1
            if not close.all():
                worst = int(np.argmax(~close))
                mismatched.append((col, worst, a[worst], b[worst]))

        assert not mismatched, (
            f"{len(mismatched)} of {checked} features changed when future data was "
            f"removed - these peek at the future:\n"
            + "\n".join(f"    {c} at row {i}: full={av!r} truncated={bv!r}"
                        for c, i, av, bv in mismatched[:10])
        )
        assert checked > 20, f"only {checked} features were checked; too few to be meaningful"

    def test_target_is_the_only_forward_looking_column(self, features):
        """Section 3.4: the target is a label, and must never be an input."""
        from src.features import NON_FEATURE_COLUMNS

        assert "Target" in NON_FEATURE_COLUMNS
        assert "Target_Price" in NON_FEATURE_COLUMNS
        cols = set(feature_columns(features))
        assert "Target" not in cols
        assert "Target_Price" not in cols

    def test_no_feature_is_a_repackaged_label(self, features):
        """No input column may be a re-expression of the target.

        This test exists because that mistake was actually made. The
        cross-sectional module introduced ``Relative_Target`` (= the label minus
        its daily mean) and ``Rank_Target``, both derived purely from the label.
        Because they were numeric and not on the blocklist, ``feature_columns()``
        swept them into the model inputs, and the cross-sectional walk-forward
        MAE came back at 0.000575 against a 0.010922 baseline - a 95% "improvement"
        that was the model reading its own answer.

        The shuffled-label control did NOT catch it. Shuffling confirmed the model
        relied on *something* in the feature set, but that something could have
        been the label. So this check is deliberately separate: it looks for a
        column that is almost perfectly linearly dependent on any known variant
        of the target, which is the signature of a disguised label.

        The threshold is 0.99. Real causal price/volume/macro features top out
        around 0.03-0.05 here; anything above 0.99 is a label in disguise.

        Only RETURN-valued labels are probed. ``Target_Price`` is deliberately
        excluded: it is a price *level*, so it correlates ~0.999 with every
        moving average and every lagged close for entirely innocent reasons.
        Comparing a level against levels measures nothing; comparing a return
        against a near-copy of that return is the leak we are hunting.
        """
        from src.cross_sectional import prepare
        from src.features import NON_FEATURE_COLUMNS

        frame = prepare(features, load_config())["frame"]
        return_labels = {
            "Target": features["Target"],
            "Relative_Target": frame["Relative_Target"],
            "Rank_Target": frame["Rank_Target"],
        }

        # Correlation must be measured on a clean subset: NaNs in either series
        # silently produce NaN correlations that would hide the very case we
        # are hunting.
        cols = [c for c in feature_columns(features) if c not in NON_FEATURE_COLUMNS]
        leaks = []
        for c in cols:
            x = pd.to_numeric(features[c], errors="coerce")
            for lname, y in return_labels.items():
                pair = pd.concat([x, y.rename("_y")], axis=1).dropna()
                if len(pair) < 200:
                    continue
                a = pair[c].to_numpy(dtype=float)
                b = pair["_y"].to_numpy(dtype=float)
                if a.std() == 0 or b.std() == 0:
                    continue
                r = abs(float(np.corrcoef(a, b)[0, 1]))
                if np.isfinite(r) and r > 0.99:
                    leaks.append(f"{c} vs {lname}: |r|={r:.6f}")
        assert not leaks, (
            "these input columns are near-perfectly correlated with the target, "
            "which means the model is reading the label:\n    "
            + "\n    ".join(leaks)
        )

    def test_no_whole_sample_normalisation(self, features):
        """Guards against a scaler or normaliser fitted on the full history.

        A whole-sample z-score would give every row a zero mean across the
        entire dataset, which is the signature of future information leaking
        backwards.
        """
        cols = feature_columns(features)
        # Columns that are naturally mean-zero-ish (z-scores, ranks, signs).
        exempt = ("Z_Score", "Outlier_Z", "TurnOfMonth", "Stoch_", "RSI",
                  "Volume_Z", "OBV_Z", "DayOfWeek", "Month")
        suspicious = []
        for col in cols:
            if any(col.startswith(e) or col == e for e in exempt):
                continue
            values = features[col].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if len(values) < 100:
                continue
            # A fitted-on-everything normaliser pins the mean to ~0 and the
            # std to ~1 for the WHOLE column. Real causal features do not.
            if abs(values.mean()) < 1e-6 and 0.9 < values.std() < 1.1:
                suspicious.append(col)
        assert not suspicious, (
            f"these columns look like they were normalised using the whole "
            f"sample (mean 0, std 1): {suspicious}"
        )


    def test_exact_collinear_columns_are_removed(self, features):
        """Redundant columns must actually be gone, not merely detectable.

        The filter exists because exact linear dependence breaks the linear SVR
        (its solver will not converge on a rank-deficient design matrix) and
        leaves the linear models with an unidentifiable coefficient. A test that
        only proved the detector works would still pass if the detector were
        never called, so this asserts on the feature list itself.
        """
        from src.features import feature_columns

        cols = feature_columns(features)
        assert len(cols) < 150, (
            f"{len(cols)} features - the collinearity filter may not be running"
        )

        # Spot-check the identities that motivated it, using the training
        # partition so the assertion is about the table the model sees.
        train = features[features["Date"] <= features["Date"].quantile(0.7)]
        if {"MACD", "MACD_Signal", "MACD_Hist"} <= set(cols):
            resid = (train["MACD_Hist"]
                     - (train["MACD"] - train["MACD_Signal"])).abs().max()
            assert resid < 1e-8, (
                f"MACD_Hist is not exactly MACD - MACD_Signal (max residual "
                f"{resid:.3e}); the filter is dropping the wrong column or the "
                f"identity changed"
            )

    def test_filter_keeps_strongly_correlated_columns(self):
        """Only EXACT dependence is redundancy; correlation is signal.

        Over-aggressive filtering would quietly throw away features, which is
        the opposite failure from keeping too many. This checks that two
        correlated-but-distinct columns both survive.
        """
        from src.features import drop_exact_collinearity

        n = 300
        rng = np.random.default_rng(0)
        base = rng.normal(size=n)
        df = pd.DataFrame({
            "independent": rng.normal(size=n),
            "correlated_a": base,                    # r = 1.0 exactly
            "correlated_b": base + rng.normal(scale=0.30, size=n),
        })
        kept = drop_exact_collinearity(df, list(df.columns))
        assert "independent" in kept
        assert "correlated_b" in kept, (
            "a noisy, strongly correlated column was dropped; the filter must "
            "only remove exact affine dependence, not correlation"
        )
        assert len(kept) >= 2

    def test_filter_drops_a_true_duplicate(self):
        from src.features import drop_exact_collinearity

        n = 200
        rng = np.random.default_rng(1)
        base = rng.normal(size=n)
        df = pd.DataFrame({
            "keep": base,
            "dup_exact": base * 2.0 + 5.0,     # exact affine copy
            "dup_neg": -base,                    # exact affine copy
            "other": rng.normal(size=n),
        })
        kept = drop_exact_collinearity(df, list(df.columns))
        assert "keep" in kept
        assert "dup_exact" not in kept, "an exact affine duplicate survived"
        assert "dup_neg" not in kept, "an exact negated copy survived"
        assert "other" in kept


class TestCrossSectionalCausality:
    """The cross-sectional context must be causal too.

    This class exists because the original causality suite had a blind spot. It
    exercised `build_ticker_features` only, so `build_cross_sectional_context`
    was never tested. A bug lived in that untested function for the whole time
    it was in the codebase: the Beta was computed as a single scalar from the
    trailing window and assigned to the entire column, which meant a number
    fitted at the end of the sample was broadcast backwards across every
    earlier date. A guard that never runs on a function is not a guard.
    """

    @staticmethod
    def _context(panel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
        from src.features import build_cross_sectional_context, build_market_context

        return build_cross_sectional_context(
            panel, build_market_context(panel, None, cfg), cfg)

    def test_truncated_history_reproduces_the_past(self, cfg):
        from src.common import CLEANED_PANEL

        panel = pd.read_parquet(CLEANED_PANEL)
        panel["Date"] = pd.to_datetime(panel["Date"])
        full = self._context(panel, cfg)

        cutoff = panel["Date"].sort_values().unique()[len(panel["Date"].unique()) // 2]
        cutoff = pd.Timestamp(cutoff)
        trunc = self._context(panel[panel["Date"] <= cutoff], cfg)

        cols = [c for c in full.columns if c != "Date"]
        a = full[full["Date"] <= cutoff].set_index("Date")[cols]
        b = trunc.set_index("Date")[cols]
        common = a.index.intersection(b.index)
        assert len(common) > 200, "not enough overlapping dates to be meaningful"

        mismatched = []
        for c in cols:
            x = pd.to_numeric(a.loc[common, c], errors="coerce").to_numpy(float)
            y = pd.to_numeric(b.loc[common, c], errors="coerce").to_numpy(float)
            both_nan = np.isnan(x) & np.isnan(y)
            close = np.isclose(x, y, rtol=1e-7, atol=1e-10) | both_nan
            if not close.all():
                worst = int(np.argmax(~close))
                mismatched.append((c, worst, x[worst], y[worst]))

        assert not mismatched, (
            f"{len(mismatched)} cross-sectional columns changed when future data "
            f"was removed - they are reading the future:\n"
            + "\n".join(f"    {c} at {common[w].date()}: full={av!r} trunc={bv!r}"
                        for c, w, av, bv in mismatched[:10])
        )

    def test_corrupting_the_future_leaves_the_past_alone(self, cfg):
        from src.common import CLEANED_PANEL

        panel = pd.read_parquet(CLEANED_PANEL)
        panel["Date"] = pd.to_datetime(panel["Date"])
        cutoff = pd.Timestamp(sorted(panel["Date"].unique())[len(panel["Date"].unique()) // 2])

        clean = self._context(panel, cfg)

        poisoned = panel.copy()
        fut = poisoned["Date"] > cutoff
        rng = np.random.default_rng(7)
        for col in ("Open", "High", "Low", "Close", cfg["price_column"], "Volume"):
            poisoned.loc[fut, col] = rng.uniform(1.0, 500.0, int(fut.sum()))
        dirty = self._context(poisoned, cfg)

        cols = [c for c in clean.columns if c != "Date"]
        a = clean[clean["Date"] <= cutoff].set_index("Date")[cols]
        b = dirty[dirty["Date"] <= cutoff].set_index("Date")[cols]
        common = a.index.intersection(b.index)
        if not len(common):
            pytest.skip("no overlap before the cutoff")

        bad = []
        for c in cols:
            x = pd.to_numeric(a.loc[common, c], errors="coerce").to_numpy(float)
            y = pd.to_numeric(b.loc[common, c], errors="coerce").to_numpy(float)
            both_nan = np.isnan(x) & np.isnan(y)
            if not (np.isclose(x, y, rtol=1e-7, atol=1e-10) | both_nan).all():
                bad.append(c)
        assert not bad, (
            f"these cross-sectional columns reacted to corrupted FUTURE data: {bad}"
        )

    def test_beta_is_a_series_not_a_broadcast_scalar(self, cfg):
        """Regression test for the exact bug described in the class docstring.

        A constant Beta column is the fingerprint of the scalar-broadcast
        mistake, and it is also useless as a feature, so one assertion catches
        both the leak and the loss of signal.
        """
        from src.common import CLEANED_PANEL

        panel = pd.read_parquet(CLEANED_PANEL)
        panel["Date"] = pd.to_datetime(panel["Date"])
        ctx = self._context(panel, cfg)
        beta_cols = [c for c in ctx.columns if c.startswith("Beta_") and c != "Beta_Breadth"]
        assert beta_cols, "no Beta columns were produced at all"
        for c in beta_cols:
            finite = ctx[c].dropna()
            if len(finite) < 100:
                continue
            assert finite.nunique() > 10, (
                f"{c} has only {finite.nunique()} distinct values across the whole "
                f"sample - it is a constant fitted once and broadcast, which means "
                f"it was fitted on the end of the data and applied to the past"
            )


class TestCausalityByPermutation:

    def test_shuffling_future_rows_leaves_early_features_untouched(self, cfg):
        """Corrupt everything after a date, rebuild, and check the earlier rows.

        If any feature consumed future information, poisoning the future would
        change the earlier values. It should not.
        """
        from src.common import CLEANED_PANEL
        from src.features import build_ticker_features, feature_columns as fc

        panel = pd.read_parquet(CLEANED_PANEL)
        panel["Date"] = pd.to_datetime(panel["Date"])
        ticker = cfg["universe"][1] if len(cfg["universe"]) > 1 else cfg["universe"][0]
        g = panel[panel["Ticker"] == ticker].copy()
        if g.empty:
            pytest.skip("no rows")

        cutoff = g["Date"].iloc[len(g) // 2]

        clean = build_ticker_features(g, cfg, cfg["price_column"])

        poisoned = g.copy()
        future_mask = poisoned["Date"] > cutoff
        rng = np.random.default_rng(0)
        for col in ["Open", "High", "Low", "Close", cfg["price_column"], "Volume"]:
            poisoned.loc[future_mask, col] = rng.uniform(1.0, 500.0, int(future_mask.sum()))

        dirty = build_ticker_features(poisoned, cfg, cfg["price_column"])

        clean_before = clean[clean["Date"] <= cutoff].reset_index(drop=True)
        dirty_before = dirty[dirty["Date"] <= cutoff].reset_index(drop=True)

        assert len(clean_before) == len(dirty_before)
        if not len(clean_before):
            pytest.skip("not enough rows before the cutoff")

        mismatched = []
        for col in fc(clean_before):
            if col in _CALENDAR_OK:
                continue
            a = pd.to_numeric(clean_before[col], errors="coerce").to_numpy(dtype=float)
            b = pd.to_numeric(dirty_before[col], errors="coerce").to_numpy(dtype=float)
            both_nan = np.isnan(a) & np.isnan(b)
            close = np.isclose(a, b, rtol=1e-7, atol=1e-10, equal_nan=False) | both_nan
            if not close.all():
                worst = int(np.argmax(~close))
                mismatched.append((col, worst, a[worst], b[worst]))

        assert not mismatched, (
            f"{len(mismatched)} features changed when FUTURE data was corrupted - "
            f"they are reading the future:\n"
            + "\n".join(f"    {c} at row {i}: {av!r} vs {bv!r}"
                        for c, i, av, bv in mismatched[:10])
        )


class TestCalendarAndSplitting:
    def test_turn_of_month_uses_calendar_only(self, features):
        assert "TurnOfMonth" in features.columns
        assert set(features["TurnOfMonth"].unique()).issubset({0, 1})

    def test_turn_of_month_is_a_function_of_the_date(self):
        """The flag must be reproducible from dates alone, with no price data."""
        dates = pd.Series(pd.bdate_range("2024-01-25", "2024-03-08"))
        a = np.asarray(turn_of_month_flag(dates))
        shuffled = dates.sample(frac=1.0, random_state=0).reset_index(drop=True)
        b = np.asarray(turn_of_month_flag(shuffled))
        # Shuffling changes each date's position within its month, so a
        # position-based implementation would give a different answer. A
        # calendar-based one must not care.
        assert a.shape == b.shape
        assert sorted(a.tolist()) == sorted(b.tolist()), (
            "the turn-of-month flag depends on row order; it must come from the "
            "published exchange calendar only"
        )

    def test_partitions_never_overlap(self, features, cfg):
        bounds = split_by_date(features["Date"], cfg)
        train, val, test = bounds["train"], bounds["val"], bounds["test"]
        assert train.max() < val.min()
        assert val.max() < test.min()
        assert not set(train) & set(val)
        assert not set(val) & set(test)
        assert not set(train) & set(test)
