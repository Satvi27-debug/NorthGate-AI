"""The dashboard must survive having no data.

Why this file exists
--------------------
The app was deployed to Streamlit Cloud and died on the Portfolio panel with
`TypeError: object of type 'NoneType' has no len()`. The Monte-Carlo scatter is
absent when the artifacts have not been built - and they are not in the repo,
because `data/processed/*.parquet` is gitignored on the stated rule that data
is regenerated rather than committed. So *any* fresh clone, and the deployed
instance in particular, runs the app with the parquet-backed artifacts missing.

The chart above the failing line was correctly guarded. The caption below it
called `len(cloud)` unguarded, and took the whole page down. Every existing
panel test ran against a populated `data/`, so none of them ever saw the state
this app actually ships in.

The test is therefore: point every path the dashboard reads at an empty
directory, execute every panel, and require that it renders a message rather
than raising. A panel that has nothing to show must say so; it must not crash.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dashboard"))


# Suffixes excluded by .gitignore: the artifacts a fresh clone does NOT have.
#
# Modelling "no data at all" is not the same as modelling a fresh clone, and
# here the difference is the whole point. The deployed instance HAD the small
# committed evidence (portfolio_metrics.json, model_leaderboard.csv,
# recommendations.csv) and did NOT have the binaries.
# `panel_portfolio` returns early when portfolio_metrics is missing, so an
# all-empty fixture never reaches the crashing caption at all - the first
# version of this test passed with the bug still present, which is worse than
# no test because it reported safety it had not checked.
GITIGNORED_SUFFIXES = (".parquet", ".npy", ".pkl", ".keras", ".h5", ".joblib")


@pytest.fixture(scope="module")
def empty_app(tmp_path_factory):
    """Redirect ONLY the gitignored binaries, exactly as a fresh clone would.

    Small committed evidence stays readable, because that is the real deployed
    state: a panel can be holding real metrics while the scatter it draws them
    against is missing.

    The paths are DISCOVERED from the module rather than hand-listed, so a newly
    added artifact cannot silently escape the test.
    """
    import app

    empty = tmp_path_factory.mktemp("no_binaries")
    (empty / "processed").mkdir(parents=True, exist_ok=True)
    (empty / "models").mkdir(parents=True, exist_ok=True)

    patched: dict[str, Path] = {}
    kept: list[str] = []
    for attr, val in list(vars(app).items()):
        if not isinstance(val, Path):
            continue
        try:
            val.relative_to(ROOT)
        except ValueError:
            continue
        if "data" not in val.parts and "models" not in val.parts:
            continue
        if val.suffix.lower() not in GITIGNORED_SUFFIXES:
            kept.append(attr)
            continue
        sub = "models" if "models" in val.parts else "processed"
        setattr(app, attr, empty / sub / val.name)
        patched[attr] = val

    for cache_name in ("load_all",):
        fn = getattr(app, cache_name, None)
        try:
            fn.clear()
        except AttributeError:
            pass

    assert patched, (
        "no gitignored artifact paths were discovered on the app module, so "
        "this fixture would test the populated state and pass for the wrong "
        "reason"
    )
    yield app, empty, patched, kept
    for attr, orig in patched.items():
        setattr(app, attr, orig)
    try:
        app.load_all.clear()
    except AttributeError:
        pass


PANELS = [
    ("Overview", "panel_overview"),
    ("Price & Prediction", "panel_price"),
    ("Models", "panel_models"),
    ("Portfolio", "panel_portfolio"),
    ("Risk", "panel_risk"),
    ("Sentiment", "panel_sentiment"),
    ("Recommendations", "panel_recommendations"),
    ("Ranking", "panel_ranking"),
]


class TestNoArtifacts:
    @pytest.fixture(autouse=True)
    def _cfg(self, empty_app):
        app, _, self.patched, self.kept = empty_app
        self.app = app
        self.cfg = app.get_config()

    def test_redirect_and_real_evidence_both_present(self, empty_app):
        """Guard against the fixture testing the populated state.

        Worth asserting on its own: the first version of this fixture missed two
        constants, loaded the real artifacts, and every test in the class passed
        while proving nothing about the state the app ships in.
        """
        app, empty, patched, kept = empty_app
        assert len(patched) >= 5, (
            f"only {len(patched)} gitignored paths were redirected; expected the "
            f"parquet/npy/pkl set a fresh clone lacks"
        )
        assert len(kept) >= 5, (
            f"only {len(kept)} committed evidence paths were left in place. The "
            f"deployed app HAS these, and they are what let panel_portfolio get "
            f"past its early return and reach the crashing caption."
        )
        for attr in ("FEATURES", "ML_PREDICTIONS", "DL_PREDICTIONS",
                     "MONTE_CLOUD_PATH", "CLEANED_PANEL"):
            if hasattr(app, attr):
                assert str(getattr(app, attr)).startswith(str(empty)), (
                    f"{attr} still points at the real tree: {getattr(app, attr)}"
                )

    def test_reproduces_the_deployed_state(self, empty_app):
        """The precise combination that killed the deployed instance.

        Small evidence present, Monte-Carlo cloud absent. This is the state the
        crash needs, and it is the state Streamlit Cloud ran in.
        """
        D = self.app.load_all()
        assert D["portfolio_metrics"] is not None, (
            "portfolio_metrics.json is committed, so it must still load - if it "
            "does not, this fixture is not the deployed state and the panel "
            "tests below are not testing the crash path"
        )
        assert D["monte_carlo"] is None, (
            "the .npy is gitignored, so it must be absent - this is what made "
            "len(cloud) raise in the deployed app"
        )

    def test_load_all_succeeds_and_is_fully_keyed(self, empty_app):
        """The loader must return a fully-populated dict of empties, not raise.

        Every key is asserted, not sampled. A key that silently disappears from
        `load_all` is not a crash - it is a panel quietly receiving `KeyError`
        later, or an `or` default that makes a missing artifact look like an
        empty one.
        """
        D = self.app.load_all()
        assert isinstance(D, dict) and D, "load_all must return a dict even with no data"
        expected = {
            "backtest", "cleaned", "cross_sectional", "data_quality", "declared",
            "dl_metrics", "dl_predictions", "features", "frontier", "leaderboard",
            "ml_metrics", "ml_predictions", "monte_carlo", "mood",
            "portfolio_metrics", "rebalance", "rec_backtest", "recommendations",
            "sentiment", "sentiment_ablation", "weights",
        }
        missing = expected - set(D)
        assert not missing, (
            f"load_all omitted {sorted(missing)}; every key must exist so a "
            f"panel can distinguish 'no data' from 'no such artifact'"
        )

    @pytest.mark.parametrize("label,fn_name", PANELS,
                             ids=[p[0] for p in PANELS])
    def test_panel_renders_instead_of_raising(self, label, fn_name):
        """The exact failure from the deployed instance.

        `panel_portfolio` called `len(cloud)` in a caption while the chart above
        it was properly guarded, so a fresh clone - and the deployed app, which
        is a fresh clone - raised TypeError. Every panel is now executed in that
        state.
        """
        import contextlib
        import io

        fn = getattr(self.app, fn_name)
        D = self.app.load_all()
        t = self.cfg["universe"][0]
        args = {
            "panel_overview": (),
            "panel_price": (t, 1, 1.96),
            "panel_models": (),
            "panel_portfolio": (0.04,),
            "panel_risk": (t,),
            "panel_sentiment": (t,),
            "panel_recommendations": (),
            "panel_ranking": (),
        }[fn_name]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            fn(self.cfg, D, *args)


# A note on what deliberately is NOT here.
#
# The first version of this file also carried an AST check that flagged every
# `len(x)` where `x` is a name the loader can return as None. It reported 14
# hits, and every one of them was a false positive: `if cloud is not None and
# len(cloud)` is the *correct* pattern, and the AST cannot see the short-circuit
# that already protects it.
#
# It was removed deliberately. A check that fires on correct code trains its
# reader to ignore it, and the next real instance of this bug would sit behind
# a test everyone has learned to skip. The behavioural test above cannot produce
# a false positive - it runs the panels against an empty tree, and either they
# render or they raise. That is the whole argument, and it needs no AST.
