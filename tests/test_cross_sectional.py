"""Tests for the supplementary cross-sectional analysis.

The failure this module actually suffered is the reason most of these tests
exist. Its first run reported a 95% error reduction against the baseline, which
was produced by the model reading two columns derived from its own label. The
corrected result is a clean null.

So the tests here are not mainly about "does it compute a number" — they are
about whether the number can be trusted, and whether a favourable result would
be reported as favourable without an independent reason to believe it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import load_config  # noqa: E402
from src.features import NON_FEATURE_COLUMNS, feature_columns  # noqa: E402

RESULTS = ROOT / "data" / "processed" / "cross_sectional_results.json"
FEATS = ROOT / "data" / "processed" / "features.parquet"


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config()


@pytest.fixture(scope="module")
def results() -> dict:
    if not RESULTS.exists():
        pytest.skip("cross_sectional_results.json not built; "
                    "run `python src/cross_sectional.py`")
    return json.loads(RESULTS.read_text(encoding="utf-8"))


class TestLabelIsolation:
    """The leak that was made, and the guard that now prevents it."""

    def test_label_variants_are_blocklisted(self):
        """Both derived labels must be explicitly excluded from model inputs.

        They are not caught by the Target rule because their names differ, and
        `feature_columns()` accepts any numeric column that is not on the
        blocklist. This is precisely how they got in.
        """
        assert "Relative_Target" in NON_FEATURE_COLUMNS
        assert "Rank_Target" in NON_FEATURE_COLUMNS

    def test_derived_labels_do_not_reach_the_feature_list(self, cfg):
        if not FEATS.exists():
            pytest.skip("features.parquet not built")
        feats = pd.read_parquet(FEATS)
        cols = set(feature_columns(feats))
        assert "Target" not in cols
        assert "Target_Price" not in cols
        assert "Relative_Target" not in cols
        assert "Rank_Target" not in cols

    def test_no_input_column_mirrors_a_label(self, cfg):
        """The standing structural guard, asserted here as well as in the suite.

        Catches the general case, not just the two columns that were actually
        used: anything built from the label and given a new name is still a
        leak, and a blocklist alone will not stop the next one.
        """
        if not FEATS.exists():
            pytest.skip("features.parquet not built")
        from src.cross_sectional import prepare

        feats = pd.read_parquet(FEATS)
        feats["Date"] = pd.to_datetime(feats["Date"])
        frame = prepare(feats, cfg)["frame"]
        labels = {
            "Target": frame["Target"],
            "Relative_Target": frame["Relative_Target"],
            "Rank_Target": frame["Rank_Target"],
        }
        offenders = []
        for c in feature_columns(frame):
            x = pd.to_numeric(frame[c], errors="coerce")
            for lname, y in labels.items():
                pair = pd.concat([x, y.rename("_y")], axis=1).dropna()
                if len(pair) < 200:
                    continue
                a = pair[c].to_numpy(float)
                b = pair["_y"].to_numpy(float)
                if a.std() == 0 or b.std() == 0:
                    continue
                r = abs(float(np.corrcoef(a, b)[0, 1]))
                if np.isfinite(r) and r > 0.99:
                    offenders.append(f"{c} ~ {lname} (r={r:.4f})")
        assert not offenders, f"label-like inputs present: {offenders}"


class TestReportedNumbers:
    """Whatever the result, the recorded numbers must be self-consistent."""

    def test_baseline_is_the_no_skill_answer(self, results):
        """The baseline must be exactly 'predict the universe average' = 0.

        If this drifts, every comparison below becomes meaningless, because
        the whole point of this module is that it is judged against a
        like-for-like no-skill reference rather than the Section 9.5 baseline.
        """
        assert results["baseline"]["rank_ic"] == 0.0
        assert results["baseline"]["top1_hit_rate"] == 0.0
        assert results["baseline"]["relative_target_mae"] > 0

    def test_validation_and_test_are_separate(self, results):
        """Model selection must not have seen the held-out window.

        The selection block records its own no-skill reference on the same
        folds. If that number were borrowed from the test window, selection
        would be contaminated and the comparison would not be like-for-like.
        """
        sel = results["model_selection"]
        assert sel["folds"] >= 3
        assert "walk_forward_mae_same_folds" in results["baseline"]
        wref = results["baseline"]["walk_forward_mae_same_folds"]
        assert np.isfinite(wref) and wref > 0
        # A genuine like-for-like reference should be in the same order of
        # magnitude as the test-set baseline, not orders away from it.
        assert 0.2 < wref / results["baseline"]["relative_target_mae"] < 5.0

    def test_a_grid_was_actually_searched(self, results):
        """A null from one arbitrary config is not evidence of anything."""
        sel = results["model_selection"]
        assert sel["configs_tried"] >= 5
        assert len(sel["leaderboard"]) == sel["configs_tried"]
        assert sel["best_family"] in {"ridge", "hist_gb"}

    def test_verdict_matches_the_measurement(self, results):
        """The headline sentence must follow from the number, not from hope."""
        base = results["baseline"]["relative_target_mae"]
        test = results["test"]["relative_target_mae"]
        actual_beats = test < base
        assert results["beats_own_baseline"] == actual_beats
        if actual_beats:
            assert "beats" in results["verdict"].lower()
        else:
            assert "does NOT beat" in results["verdict"]
            assert "failure" in results["verdict"].lower()

    def test_top1_is_judged_against_chance_not_by_eyeball(self, results):
        """The skill flag must come from the binomial test, never from the sign.

        The observed hit rate here is 11.1% against a 10% chance level. It is
        *above* chance, but the binomial p-value is 0.46, so it is entirely
        consistent with random selection. Reporting "11.1% > 10%, so the model
        wins" is exactly the eyeball error this test exists to prevent, and the
        contract is that the boolean is derived from the p-value alone.
        """
        test = results["test"]
        p = test["top1_binomial_p"]
        assert np.isfinite(p)
        assert test["top1_better_than_chance"] == (p < 0.05), (
            f"skill flag disagrees with the binomial test: "
            f"top1_better_than_chance={test['top1_better_than_chance']} but p={p:.3f}"
        )
        # And the specific trap this guards: above chance, not better than chance.
        if test["top1_hit_rate"] > test["top1_chance"] and p >= 0.05:
            assert not test["top1_better_than_chance"]


class TestHonestFraming:
    """This module must never be usable to imply the Section 9.5 bar passed."""

    def test_never_claims_the_forecast_bar(self, results):
        blob = json.dumps(results).lower()
        for forbidden in ("beats the random walk", "clears section 8.3",
                          "satisfies section 9.5", "passes the prd"):
            assert forbidden not in blob, (
                f"cross-sectional results must not contain '{forbidden}' — this "
                f"task has a different baseline and cannot satisfy the forecast bar"
            )

    def test_separate_baseline_is_explained(self, results):
        assert "baseline" in results["why_this_is_separate"].lower()
        assert "never" in results["why_this_is_separate"].lower()

    def test_the_leak_is_recorded(self, results):
        """The caught leak belongs in the artifact, not only in a commit message."""
        note = results.get("leak_found_and_fixed", "")
        assert note, "the cross-sectional leak and its fix must be recorded"
        assert "Relative_Target" in note
        assert "NON_FEATURE_COLUMNS" in note
