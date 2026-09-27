"""Tests for the recommendation hit-rate backtest.

PRD Section 14, acceptance table, row "Recommendations": *"Backtested hit-rate
vs. buy-and-hold — documented, with transparent contributing signals."*

The three things that would make this worthless are: a metric that is not
defined, a comparison against something other than the deployed engine, and a
look-ahead. Each has a test here.
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

ARTIFACT = ROOT / "data" / "processed" / "recommendation_backtest.json"


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config()


@pytest.fixture(scope="module")
def rb() -> dict:
    if not ARTIFACT.exists():
        pytest.skip("recommendation_backtest.json not built; run "
                    "`python src/recommendation_backtest.py`")
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


class TestFusionIsSigned:
    """The composite must be able to express a bearish view.

    It could not, before this was fixed. Every sub-signal was rank-normalised
    to [0, 1] and every weight was positive, so the composite lay in
    [0, sum(weights)] and could never be negative. `sell_threshold: -0.15` was
    therefore unreachable and the SELL branch was dead code - the threshold was
    not merely unused, it was impossible.
    """

    def test_centre_maps_to_signed_range(self):
        from src.recommend import _centre

        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        c = _centre(s)
        assert c.min() < 0 < c.max(), (
            f"centred signal must straddle zero, got {c.min():.3f}..{c.max():.3f}"
        )
        assert c.min() >= -1.0 and c.max() <= 1.0

    def test_all_three_actions_are_reachable(self, cfg):
        from src.recommend import recommend

        buy = float(cfg["recommend"]["buy_threshold"])
        sell = float(cfg["recommend"]["sell_threshold"])
        assert sell < 0 < buy, "thresholds should be symmetric about zero"
        # Signature is recommend(score, low, high) - SELL threshold FIRST.
        # Swapping them silently inverts the rule and yields zero HOLDs, which
        # is how the backtest was wrong the first time.
        assert recommend(buy + 0.1, sell, buy) == "BUY"
        assert recommend(sell - 0.1, sell, buy) == "SELL"
        assert recommend(0.0, sell, buy) == "HOLD"

    def test_fuse_output_is_bounded(self):
        from src.recommend import _fuse

        n = 10
        f = pd.Series(np.linspace(0, 1, n))
        r = pd.Series(np.linspace(1, 0, n))
        s = pd.Series(0.5, index=f.index)
        out = _fuse(f, s, -r, {"Forecast_Signal": 0.5,
                                "Sentiment_Signal": 0.3,
                                "Risk_Signal": 0.2})
        assert out.min() >= -1.0 - 1e-9 and out.max() <= 1.0 + 1e-9
        assert out.min() < 0 < out.max()


class TestBacktestIsHonest:
    def test_metric_is_defined(self, rb):
        """An undefined metric cannot be audited or reproduced."""
        d = rb["metric_definition"].lower()
        assert "hit-rate" in d
        assert "sign" in d and "realised" in d
        assert "buy-and-hold" in d, (
            "the definition must state the baseline it is compared against"
        )

    def test_buy_and_hold_is_actually_reported(self, rb):
        assert rb.get("buy_and_hold_hit_rate") is not None, (
            "PRD Section 14 requires the buy-and-hold comparison; the baseline "
            "itself is missing"
        )

    def test_arms_are_not_all_identical(self, rb):
        """Three identical arms mean the comparison measures nothing.

        An earlier version of the backtest carried its own copy of the fusion
        arithmetic, drifted from the deployed engine, and reported three arms
        with identical hit-rates and zero trades - while the dashboard was
        emitting a real BUY/HOLD/SELL mix. The bug was invisible because a
        backtest that measures nothing still prints numbers.
        """
        rates = {k: v.get("hit_rate") for k, v in rb["arms"].items()}
        real = [r for r in rates.values() if r is not None]
        assert len(real) >= 2, f"fewer than two arms produced a rate: {rates}"
        assert len(set(real)) >= 2, (
            f"every arm produced the same hit-rate {real[0]}; the arms are not "
            f"measuring different things"
        )
        gated = rb["arms"].get("gated", {})
        assert gated.get("n_buys", 0) + gated.get("n_sells", 0) > 0, (
            "the deployed arm issued no trades at all, so it cannot be compared "
            "with buy-and-hold"
        )

    def test_uses_the_declared_model(self, rb):
        """The backtest must score the forecasts the dashboard actually shows."""
        decl = ROOT / "data" / "processed" / "declared_model.json"
        if not decl.exists():
            pytest.skip("no declared model")
        model = json.loads(decl.read_text(encoding="utf-8")).get("model")
        if model:
            # The ML file holds the classical models; the declared model may be
            # a deep one, in which case the backtest legitimately falls back to
            # the best classical forecast. Either way it must name what it used.
            assert rb.get("model"), "the backtest did not record which model it used"

    def test_contributing_signals_are_reported(self, rb):
        """The PRD row ends 'with transparent contributing signals'."""
        contrib = rb.get("contributing_signals") or {}
        assert contrib, "no per-signal contribution was reported"
        for name, stats in contrib.items():
            assert "sign_agreement" in stats, f"{name} has no agreement figure"
            for key in ("sign_agreement", "corr_with_realised"):
                if key in stats:
                    assert np.isfinite(stats[key]), f"{name}.{key} is not finite"

    def test_window_is_the_held_out_period(self, rb):
        w = rb["window"]
        assert w["sessions"] > 100, (
            f"only {w['sessions']} sessions backtested - too few to be evidence"
        )
        assert w["observations"] > 500

    def test_verdict_follows_the_numbers(self, rb):
        """The headline sentence must follow from the measurement."""
        bh = rb["buy_and_hold_hit_rate"]
        gated = rb["arms"]["gated"]["hit_rate"]
        if gated is None or bh is None:
            pytest.skip("gated arm or baseline produced no rate")
        assert rb["beats_buy_and_hold"] == (gated > bh)
        text = rb["interpretation"].lower()
        if gated > bh:
            assert "beats" in text
        elif gated < bh:
            assert "trails" in text or "negative" in text
            assert "would have been" in text, (
                "a negative result must say the suggestions would have been a "
                "net negative, not merely that the number is lower"
            )


class TestNoLookAhead:
    """The score at date t must not depend on anything after t."""

    def test_realised_returns_are_the_forward_return(self, cfg):
        from src.recommendation_backtest import _realised_forward_returns
        from src.common import FEATURES

        if not FEATURES.exists():
            pytest.skip("features.parquet not built")
        feat = pd.read_parquet(FEATURES)
        feat["Date"] = pd.to_datetime(feat["Date"])
        fwd = _realised_forward_returns(feat, cfg)
        universe = list(cfg["universe"])
        t = universe[0]
        s = fwd[t].dropna()
        assert len(s) > 100
        # At date d the value must be log(P[d+1]/P[d]) -- the NEXT session's
        # move, which is the outcome the recommendation at d is scored against.
        px = (feat[feat["Ticker"] == t].set_index("Date")["Adjusted Close"]
              .sort_index())
        nxt = px.shift(-1)
        d = s.index[10]
        expected = float(np.log(nxt.loc[d] / px.loc[d]))
        assert abs(float(s.loc[d]) - expected) < 1e-9, (
            f"the outcome at {d.date()} is not the forward return "
            f"(got {s.loc[d]:.8f}, expected {expected:.8f}) - the backtest "
            f"would be scoring the engine against the same day it predicted"
        )
        # And it must NOT be the same-day return, which would be the look-ahead
        # this check exists to catch.
        same_day = float(np.log(px.loc[d] / px.shift(1).loc[d]))
        assert abs(float(s.loc[d]) - same_day) > 1e-6, (
            "the outcome equals the same-day return, so the signal is being "
            "scored against information it already had"
        )

    def test_risk_signal_is_trailing_only(self, cfg):
        from src.recommendation_backtest import _trailing_risk_signal
        from src.common import FEATURES

        if not FEATURES.exists():
            pytest.skip("features.parquet not built")
        feat = pd.read_parquet(FEATURES)
        feat["Date"] = pd.to_datetime(feat["Date"])
        risk = _trailing_risk_signal(feat, cfg)

        # Truncating the future must not change the past.
        cutoff = feat["Date"].quantile(0.6)
        truncated = _trailing_risk_signal(feat[feat["Date"] <= cutoff], cfg)
        a = (risk[risk["Date"] <= cutoff]
             .set_index(["Date", "Ticker"]).sort_index())
        b = truncated.set_index(["Date", "Ticker"]).sort_index()
        common = a.index.intersection(b.index)
        assert len(common) > 200
        for t in cfg["universe"][:3]:
            xs = a.loc[common, "Risk"].xs(t, level="Ticker")
            ys = b.loc[common, "Risk"].xs(t, level="Ticker")
            both_nan = xs.isna() & ys.isna()
            close = np.isclose(xs, ys, equal_nan=False) | both_nan
            assert close.all(), (
                f"{t}'s risk signal changed when future rows were removed - it "
                f"is reading ahead"
            )


class TestReproducibilityEvidence:
    """`requirements.txt` must be installable, and that must be demonstrated.

    The clean-environment check exists because having the right files in the
    repository is not the same as being able to build from them. It immediately
    found that `requirements.txt` could not be installed at all:

        tensorflow 2.21.0  requires  protobuf >=6.31.1, <8
        streamlit  1.40.2  requires  protobuf >=3.20,  <6

    The development machine worked anyway, because it had been built
    incrementally and had drifted onto a protobuf version that satisfies neither
    declaration. A fresh checkout could never have reached it. So the pin set was
    uninstallable and the repository was not reproducible, while every local
    signal - all tests passing, the app running, the models trained - said
    otherwise.
    """

    def test_clean_env_report_exists_and_is_presented(self):
        p = ROOT / "reports" / "clean_env_check.json"
        if not p.exists():
            pytest.skip(
                "clean_env_check.json not built; run "
                "`python scripts/clean_env_check.py`")
        import json
        doc = json.loads(p.read_text(encoding="utf-8"))
        assert "steps" in doc and doc["steps"], "no steps recorded"
        for s in doc["steps"]:
            assert s["status"] in {"PASS", "FAIL", "SKIPPED (failed, but optional)"}, (
                f"step {s['name']!r} has status {s['status']!r}; a step that did "
                f"not run must never be reported as a pass"
            )
        assert "verdict" in doc

    def test_no_step_is_reported_as_passing_when_it_did_not(self):
        """A PASS must be backed by a real exit code.

        One exception, and it is a real one: the artifact-ordering step is a
        pure inspection with no subprocess behind it, so it has no exit code.
        It is identified by the `<check>` sentinel in its command, which is
        why that sentinel exists. Asserting `exit_code == 0` there would have
        failed a check that genuinely passed, and the only way to make such a
        test pass is to weaken it everywhere - which is how a real regression
        gets waved through later.
        """
        p = ROOT / "reports" / "clean_env_check.json"
        if not p.exists():
            pytest.skip("clean_env_check.json not built")
        import json
        doc = json.loads(p.read_text(encoding="utf-8"))
        for s in doc["steps"]:
            if s["status"] != "PASS":
                continue
            if s["exit_code"] is None:
                # Only legitimate for an inspection step, and only if it
                # actually recorded something to justify the verdict.
                assert any("tier" in ln or "present" in ln
                           for ln in s.get("output_tail", [])), (
                    f"{s['name']} is marked PASS with no exit code and no "
                    f"recorded evidence, so nothing supports the verdict"
                )
                continue
            assert s["exit_code"] == 0, (
                f"{s['name']} is marked PASS with exit code {s['exit_code']}"
            )

    def test_requirements_pin_no_unresolvable_pair(self):
        """A cheap guard against re-introducing the protobuf-style conflict.

        This does not attempt full dependency resolution - that is what
        `clean_env_check.py` does. It asserts the two constraints that actually
        clashed are now mutually satisfiable, so the specific regression cannot
        come back unnoticed between full checks.
        """
        import re
        import urllib.request

        req = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        pins = dict(re.findall(r"^([A-Za-z0-9_.-]+)==([^\s#]+)", req, re.M))
        assert "streamlit" in pins and "tensorflow" in pins, (
            "both pins must be present for this guard to mean anything"
        )

        def protobuf_range(package: str) -> tuple[str, str] | None:
            url = f"https://pypi.org/pypi/{package}/{pins[package]}/json"
            try:
                with urllib.request.urlopen(
                        urllib.request.Request(
                            url, headers={"User-Agent": "northgate-repro"}),
                        timeout=30) as r:
                    import json
                    meta = json.load(r)
            except Exception:  # noqa: BLE001 - offline is not a failure here
                return None
            for dep in meta["info"].get("requires_dist") or []:
                if dep.lower().startswith("protobuf"):
                    m = re.search(r">=\s*([0-9.]+)", dep)
                    u = re.search(r"<\s*([0-9.]+)", dep)
                    return (m.group(1) if m else "0",
                            u.group(1) if u else "999")
            return None

        st = protobuf_range("streamlit")
        tf = protobuf_range("tensorflow")
        if st is None or tf is None:
            pytest.skip("PyPI metadata unavailable; the full check still applies")

        def vtuple(s: str) -> tuple[int, ...]:
            # Version parts compare numerically and must be tuples, not floats:
            # "6.31.1" is not a representable float and float() raises on it.
            return tuple(int(x) for x in s.split(".") if x.isdigit())

        # Two half-open intervals [a1, a2) and [b1, b2) intersect iff
        # a1 < b2 and b1 < a2.
        overlap = vtuple(st[0]) < vtuple(tf[1]) and vtuple(tf[0]) < vtuple(st[1])
        assert overlap, (
            f"unresolvable: streamlit {pins['streamlit']} needs protobuf "
            f">={st[0]},<{st[1]} while tensorflow {pins['tensorflow']} needs "
            f">={tf[0]},<{tf[1]} - the ranges do not overlap, so "
            f"`pip install -r requirements.txt` cannot succeed"
        )


class TestBacktestUsesTheDeployedEngine:
    def test_no_duplicated_fusion_arithmetic(self):
        """The backtest must import the engine, not reimplement it.

        Drift between a backtest and the thing it backtests is the failure this
        module has already suffered once. Asserted structurally: the only
        weighting call must be the imported `_fuse`.
        """
        src = (ROOT / "src" / "recommendation_backtest.py").read_text(
            encoding="utf-8")
        assert "from src.recommend import" in src
        assert "_fuse(" in src, "the backtest must call the engine's _fuse"
        # A hand-rolled weighted sum is exactly what drifted before.
        assert "w_cfg[\"forecast\"] * f_n" not in src
        assert "w_cfg[\"risk\"] * r_n" not in src
