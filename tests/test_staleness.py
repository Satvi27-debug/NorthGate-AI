"""Staleness guards: every trained artifact must match the current inputs.

Two separate mistakes during this build came from the same root cause, and both
produced artifacts that looked perfectly valid while being fitted to the wrong
thing:

1. The feature table grew from 64 to 135 columns, and the saved ``.keras``
   models were quietly deleted as stale but ``dl_metrics.csv`` still held
   numbers from the 64-column run. Downstream stages then reported portfolio
   results computed from those stale diagnostics.

2. ``Beta_*`` was briefly a constant column - a scalar fitted at the end of the
   sample and broadcast backwards - which is look-ahead. The DL run in flight
   at that moment was training on the contaminated table.

The failure mode in both cases is the same: a *timestamp* disagreement between
an input and a derived artifact, which nothing was checking. Feature counts
were the only structural signal, and they were only checked in one place.

These tests make staleness loud. They are deliberately strict: a build that
cannot prove its artifacts are current should fail rather than quietly report
numbers from a superseded run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import (  # noqa: E402
    CROSS_SECTIONAL,
    DL_METRICS,
    DL_PREDICTIONS,
    FEATURES,
    ML_METRICS,
    ML_PREDICTIONS,
    MODEL_LEADERBOARD,
    MODELS_DIR,
    PORTFOLIO_BACKTEST,
    PORTFOLIO_METRICS,
    REBALANCE_PLAN,
    RECOMMENDATIONS,
)

PROCESSED = ROOT / "data" / "processed"


def _newer(derived: Path, source: Path) -> bool:
    return derived.exists() and source.exists() and \
        derived.stat().st_mtime >= source.stat().st_mtime


@pytest.fixture(scope="module")
def n_features() -> int:
    if not FEATURES.exists():
        pytest.skip("features.parquet not built")
    from src.features import feature_columns

    return len(feature_columns(pd.read_parquet(FEATURES)))


class TestNoStaleDerivedArtifacts:
    """Nothing downstream may predate the table it was fitted on."""

    DERIVED = [
        ML_METRICS, ML_PREDICTIONS, DL_METRICS, DL_PREDICTIONS,
        MODEL_LEADERBOARD, PORTFOLIO_METRICS, PORTFOLIO_BACKTEST,
        RECOMMENDATIONS, REBALANCE_PLAN,
    ]

    def test_derived_artifacts_are_not_older_than_the_feature_table(self):
        if not FEATURES.exists():
            pytest.skip("features.parquet not built")
        feat_mtime = FEATURES.stat().st_mtime
        stale = []
        for p in self.DERIVED:
            if p.exists() and p.stat().st_mtime < feat_mtime:
                stale.append(
                    f"{p.name} ({p.stat().st_mtime:.0f}) predates "
                    f"features.parquet ({feat_mtime:.0f})")
        assert not stale, (
            "these artifacts were fitted on an OLDER feature table than the one "
            "on disk, so their numbers describe a superseded run. Retrain:\n    "
            + "\n    ".join(stale)
        )

    def test_saved_scaler_matches_the_current_feature_count(self, n_features):
        """A shape mismatch here means the saved model cannot be used at all.

        This is the cheap structural check that catches a feature-table change
        even when timestamps happen to line up (a fresh checkout, a copied
        directory, a restored backup).
        """
        import joblib

        scaler_path = MODELS_DIR / "feature_scaler.pkl"
        if not scaler_path.exists():
            pytest.skip("feature_scaler.pkl not built")
        # Written with joblib.dump in models_ml, so it must be read the same
        # way. Plain pickle.load raises on the joblib container.
        scaler = joblib.load(scaler_path)
        saved = getattr(scaler, "n_features_in_", None)
        if saved is None:
            pytest.skip("saved object is not a fitted scaler")
        assert saved == n_features, (
            f"feature_scaler.pkl was fitted on {saved} features but the feature "
            f"table now has {n_features}. The saved model and the current data "
            f"disagree; retrain with `python retrain_models.py`."
        )

    def test_saved_column_list_matches_the_current_features(self):
        """The strongest available check: compare names, not just the count.

        `target_scaler.pkl` records the exact feature column list used at fit
        time. A count can match while the columns differ - one column dropped
        and another added leaves the total unchanged and the model silently
        mis-scored. Comparing the list catches that.
        """
        import joblib

        from src.features import feature_columns

        path = MODELS_DIR / "target_scaler.pkl"
        if not path.exists() or not FEATURES.exists():
            pytest.skip("artifacts not built")
        blob = joblib.load(path)
        saved_cols = blob.get("feature_columns") if isinstance(blob, dict) else None
        if not saved_cols:
            pytest.skip("saved artifact does not record a feature column list")

        current = feature_columns(pd.read_parquet(FEATURES))
        missing = [c for c in current if c not in set(saved_cols)]
        extra = [c for c in saved_cols if c not in set(current)]
        assert not missing and not extra, (
            "the saved model was fitted on a different feature table:\n"
            f"    present now but absent at fit time: {missing}\n"
            f"    present at fit time but gone now  : {extra}\n"
            f"A matching total would not reveal this; the column names do."
        )

    def test_saved_model_artifact_matches_the_scaler(self, n_features):
        """The representative model must agree with the saved feature scaler.

        A model and a scaler fitted in different runs is a runtime failure at
        best, and a silent misprediction at worst if the shapes happen to match
        while the columns do not.
        """
        import joblib

        model_path = MODELS_DIR / "randomforest_model.pkl"
        scaler_path = MODELS_DIR / "feature_scaler.pkl"
        if not (model_path.exists() and scaler_path.exists()):
            pytest.skip("model or scaler not built")
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        m_n = getattr(model, "n_features_in_", None)
        s_n = getattr(scaler, "n_features_in_", None)
        if m_n is None or s_n is None:
            pytest.skip("one of the artifacts does not record a feature count")
        assert m_n == s_n == n_features, (
            f"model expects {m_n} features, scaler expects {s_n}, feature table "
            f"has {n_features}"
        )


class TestRecordedFeatureCounts:
    """Metrics files record the feature count they were produced with.

    A mismatch is not cosmetic: it means the numbers in the leaderboard were
    produced against a different input than the one on disk.
    """

    def test_dl_metrics_record_the_current_feature_count(self, n_features):
        if not DL_METRICS.exists():
            pytest.skip("dl_metrics.csv not built")
        df = pd.read_csv(DL_METRICS)
        if "Best_Params" not in df.columns:
            pytest.skip("no params column")
        import json as _json

        for _, row in df.iterrows():
            try:
                params = _json.loads(row["Best_Params"])
            except (ValueError, TypeError):
                continue
            if "n_features" in params:
                assert params["n_features"] == n_features, (
                    f"{row.get('Model', '?')} was trained on "
                    f"{params['n_features']} features but the table now has "
                    f"{n_features}"
                )

    def test_dl_run_summary_records_the_current_feature_count(self, n_features):
        p = PROCESSED / "dl_run_summary.json"
        if not p.exists():
            pytest.skip("dl_run_summary.json not built")
        blob = json.loads(p.read_text(encoding="utf-8"))
        n = blob.get("n_features")
        if n is not None:
            assert n == n_features, (
                f"dl_run_summary.json says {n} features, table has {n_features}"
            )

    def test_cross_sectional_result_matches_the_feature_count(self, n_features):
        if not CROSS_SECTIONAL.exists():
            pytest.skip("cross_sectional_results.json not built")
        blob = json.loads(CROSS_SECTIONAL.read_text(encoding="utf-8"))
        n = blob.get("n_features")
        if n is not None:
            assert n == n_features, (
                f"cross-sectional analysis used {n} features, table has "
                f"{n_features}. Re-run `python src/cross_sectional.py`."
            )


class TestLeakGuardIsActuallyWiredIn:
    """The guard that caught the label leak must be part of the default run.

    A test that is skipped by default, or that lives outside the test
    directory, is documentation rather than protection. This asserts the
    essential property is asserted *somewhere* in the suite.
    """

    def test_causality_suite_guards_against_label_like_features(self):
        p = ROOT / "tests" / "test_causality.py"
        assert p.exists(), "tests/test_causality.py is missing"
        src = p.read_text(encoding="utf-8")
        assert "test_no_feature_is_a_repackaged_label" in src, (
            "the label-leak regression test has been removed from the causality "
            "suite - that test is the only thing standing between this project "
            "and repeating the cross-sectional leak"
        )
        assert "0.99" in src, "the label-correlation threshold should remain 0.99"

    def test_cross_sectional_context_is_covered_by_the_causality_suite(self):
        """The blind spot that hid the Beta bug must stay closed.

        The original suite only exercised `build_ticker_features`, so
        `build_cross_sectional_context` shipped a look-ahead bug undetected for
        as long as it existed there.
        """
        src = (ROOT / "tests" / "test_causality.py").read_text(encoding="utf-8")
        assert "build_cross_sectional_context" in src, (
            "the causality suite must exercise build_cross_sectional_context - "
            "omitting it previously allowed a look-ahead Beta to pass unnoticed"
        )
        assert "test_beta_is_a_series_not_a_broadcast_scalar" in src
