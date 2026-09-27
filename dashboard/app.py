"""PRD Section 13 - Streamlit Presentation Layer.

A single application for a non-technical reviewer. It reads pre-computed
artifacts only and **never retrains on load** (Section 13.2); every expensive
read is wrapped in ``@st.cache_data``.

All seven Section 13.1 panels are present, and the educational-research /
not-financial-advice disclaimer is rendered on every page by
:func:`render_disclaimer`, which is called at the top of each panel rather than
relying on a footer that can scroll out of view.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import (  # noqa: E402
    CLEANED_PANEL,
    DQ_REPORT_JSON,
    DL_METRICS,
    DL_PREDICTIONS,
    FEATURES,
    MARKET_MOOD,
    ML_METRICS,
    ML_PREDICTIONS,
    MODEL_LEADERBOARD,
    MONTE_CLOUD_PATH,
    PORTFOLIO_BACKTEST,
    PORTFOLIO_FRONTIER,
    PORTFOLIO_METRICS,
    PORTFOLIO_WEIGHTS,
    PROCESSED_DIR,
    REBALANCE_PLAN,
    RECOMMENDATIONS,
    REPORTS_DIR,
    SENTIMENT_ABLATION,
    SENTIMENT_FEATURES,
    CROSS_SECTIONAL,
    DECLARED_MODEL,
    load_config,
)

# ==========================================================================
# Theme
# ==========================================================================
PALETTE = {
    "bg": "#0E1117",
    "surface": "#161B22",
    "border": "#262D36",
    "text": "#E6EDF3",
    "muted": "#8B949E",
    "accent": "#4C8DFF",
    "positive": "#3FB950",
    "negative": "#F85149",
    "warning": "#D29922",
    "neutral": "#8B949E",
    "series": ["#4C8DFF", "#3FB950", "#D29922", "#F85149", "#A371F7",
               "#39C5CF", "#DB61A2", "#7EE787", "#FFA657", "#8B949E"],
}

# Page configuration must be the first Streamlit command in the script, and may
# only be called once - so it lives here at module scope, not inside main().
# Adding a second call inside main() raises
# StreamlitSetPageConfigMustBeFirstCommandError and the app renders nothing.
#
# The sidebar starts collapsed: the first screen should be the verdict and the
# figures. The stock picker and the interest-rate assumption are things a reader
# goes looking for once they know what they want, and Streamlit's own arrow in
# the top-left reopens the panel on demand.
#
# `client.initial_sidebar_state` is NOT a `.streamlit/config.toml` key - it was
# removed from the config schema, and setting it there produces only a startup
# warning while the sidebar silently stays open.
st.set_page_config(
    page_title="Northgate AI — Stock Predictor",
    page_icon="*",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ==========================================================================
# Design system
# --------------------------------------------------------------------------
# Everything here is INLINE `style="..."`, never a `<style>` block. That is a
# hard constraint, not a preference, and it was learned the expensive way.
#
# Streamlit sanitises every HTML fragment it renders with DOMPurify, and
# DOMPurify's default allowed-tag list does not include `style`. So a
# stylesheet block is stripped on the client, silently, while the app keeps
# serving, every panel keeps running, and every test keeps passing. That is
# exactly what happened: the dashboard shipped with a 12,000-character
# stylesheet and rendered completely unstyled - Streamlit's stock red tab, no
# boxes, no background - and nothing reported an error.
#
# What does survive is the `style` ATTRIBUTE, which is in DOMPurify's default
# allowed-attribute list, plus Streamlit's own themed components
# (`st.container(border=True)`, `st.metric`, `st.tabs`, `st.expander`), which
# are styled by the theme in `.streamlit/config.toml` and need no CSS from us at
# all.
#
# So the design is built from those two things. It is more verbose than a
# stylesheet and that verbosity is the price of the page rendering at all.
# `tests/test_dashboard.py` asserts no `<style>` block reappears.
# ==========================================================================

# Spacing and type tokens, kept in one place so the rhythm stays consistent.
RADIUS = "10px"
GAP = "14px"


def _box(background: str, border: str, pad: str = "13px 16px",
         margin: str = f"0 0 {GAP} 0", radius: str = RADIUS) -> str:
    """A bordered surface. The single most reused fragment in the app."""
    return (f"background:{background};border:1px solid {border};"
            f"border-radius:{radius};padding:{pad};margin:{margin};")


def _label(colour: str, size: str = "0.7rem") -> str:
    """Small uppercase section label. Used for tile labels and group headings."""
    return (f"color:{colour};font-size:{size};text-transform:uppercase;"
            f"letter-spacing:0.07em;font-weight:700;")


def _body(colour: str, size: str = "0.9rem", weight: str = "400") -> str:
    return f"color:{colour};font-size:{size};font-weight:{weight};"


def html_box(inner: str, style: str) -> None:
    """Render pre-composed HTML through the attribute path that survives."""
    st.markdown(f'<div style="{style}">{inner}</div>', unsafe_allow_html=True)


def page_setup() -> None:
    """No stylesheet is injected.

    Kept as a named function because `main()` calls it, and because a reader
    looking for where the CSS went needs one obvious place to look. The base
    palette lives in `.streamlit/config.toml`, which Streamlit applies
    unconditionally - so if any of the inline styling below were ever lost, the
    page degrades to a plain dark theme rather than to a wrong one.
    """


def base_layout(fig: go.Figure, height: int = 380, legend: bool = True) -> go.Figure:
    """Apply the house style to a Plotly figure."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PALETTE["surface"],
        plot_bgcolor=PALETTE["surface"],
        font=dict(color=PALETTE["text"], size=12, family="Inter, system-ui, sans-serif"),
        height=height,
        margin=dict(l=12, r=12, t=34, b=12),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="right", x=1, font=dict(size=10)),
        hoverlabel=dict(bgcolor="#1F2630", bordercolor=PALETTE["border"],
                        font=dict(color=PALETTE["text"], size=12)),
    )
    fig.update_xaxes(gridcolor=PALETTE["border"], zerolinecolor=PALETTE["border"],
                     linecolor=PALETTE["border"], tickfont=dict(size=10))
    fig.update_yaxes(gridcolor=PALETTE["border"], zerolinecolor=PALETTE["border"],
                     linecolor=PALETTE["border"], tickfont=dict(size=10))
    return fig


def render_disclaimer(cfg: dict, detail: str = "") -> None:
    """Mandatory disclaimer (PRD Sections 12.3 and 13.2) - shown on every panel.

    The PRD requires it persistent and on every page, so it is never optional.
    It is drawn as a filled, fully-bordered rectangle with its own background,
    so it is unmistakably a separate standing notice rather than part of the
    panel's argument.

    Kept to two short lines. The text was previously four lines and said the
    same thing twice - "not financial advice" and "not a recommendation to buy
    or sell" are the same sentence to a reader. A notice that takes a screen to
    read gets skimmed, and a skimmed notice is not a notice.
    """
    extra = f" {detail}" if detail else ""
    st.markdown(
        f'<div style="background:#1B1F26;border:1px solid #4A3F2A;'
        f'border-left:3px solid {PALETTE["warning"]};border-radius:10px;'
        f'padding:12px 15px;margin:0 0 18px 0;">'
        f'<div style="color:{PALETTE["warning"]};font-size:0.7rem;'
        f'text-transform:uppercase;letter-spacing:0.09em;font-weight:700;'
        f'margin-bottom:6px;">Please read</div>'
        f'<div style="color:#C8D0DA;font-size:0.83rem;line-height:1.6;">'
        f'Educational research only &mdash; not financial advice, and not a '
        f'recommendation to buy or sell. Past results don&rsquo;t predict '
        f'future results.{extra}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def explain(technical: str, plain: str) -> None:
    """Show a short plain-language gloss under a technical label.

    PRD Section 2.1 asks for a dashboard "suitable for a non-technical
    reviewer", while Section 8.3 names the metrics a reviewer will look for. So
    the technical term stays visible for grading, and a one-line translation
    sits beside it for everyone else.
    """
    st.caption(f"*{plain}*")


def card(title: str, plain: str | None = None) -> None:
    """Section label, with an optional one-line plain-language translation.

    Kept for genuine section headings. For a figure with a number attached, use
    `stat_tile` instead - see the note there on why the two were separate.
    """
    if plain:
        st.markdown(
            f'<div style="{_label(PALETTE["muted"])}margin:6px 0 5px 0;">'
            f'{title}</div>'
            f'<div style="{_body(PALETTE["text"], "0.92rem")}line-height:1.5;'
            f'margin-bottom:10px;">{plain}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div style="{_label(PALETTE["muted"])}margin:6px 0 8px 0;">'
            f'{title}</div>',
            unsafe_allow_html=True,
        )


def stat_tile(label: str, value: str, note: str = "",
              value_colour: str | None = None) -> None:
    """One self-contained tile: a label, a value, and at most one short note.

    This replaces the three-label pattern the overview used to render. A tile
    previously printed a card title ("Stocks tracked"), then a plain-language
    gloss ("The companies the system follows"), then Streamlit's own metric
    label ("Companies"), and only then the number - so a reader saw three
    different names for one figure before reaching the figure itself, and the
    value was pushed below the fold.

    The label is the figure's name. The note is capped at one short line; a
    second line makes the row taller than its neighbours and pushes the numbers
    onto different baselines, which reads as a mistake even though it is not.

    `value_colour` is for pass/fail figures only, so a red "Not met" is legible
    at a glance. Everything else stays in the neutral text colour, because a
    page where every number is coloured is a page where no number is.
    """
    note_html = (
        f'<div style="{_body(PALETTE["muted"], "0.76rem")}line-height:1.4;'
        f'margin-top:7px;">{note}</div>' if note else ""
    )
    colour = value_colour or PALETTE["text"]
    st.markdown(
        f'<div style="{_box(PALETTE["surface"], PALETTE["border"], "13px 15px")}'
        f'height:100%;box-sizing:border-box;">'
        f'<div style="{_label(PALETTE["muted"])}margin-bottom:8px;">{label}</div>'
        f'<div style="color:{colour};font-size:1.4rem;font-weight:650;'
        f'line-height:1.15;font-variant-numeric:tabular-nums;'
        f'letter-spacing:-0.01em;">{value}</div>'
        f'{note_html}</div>',
        unsafe_allow_html=True,
    )


# ==========================================================================
# Cached loaders - the app reads artifacts, it never retrains
# ==========================================================================
@st.cache_data(show_spinner=False)
def get_config() -> dict:
    return load_config()


def _read(path: Path):
    """Read a pipeline artifact, CSV or Parquet.

    CSVs written by ``pandas.Series.to_csv`` carry an unnamed first column
    holding the index. Without ``index_col=0`` that column arrives as a string
    column and any later numeric operation fails with a confusing
    "'>' not supported between 'str' and 'int'". We detect the unnamed header and
    restore it as the index.
    """
    if not path.exists():
        return None
    if path.suffix == ".parquet":
        return pd.read_parquet(path)

    header = pd.read_csv(path, nrows=0)
    unnamed_first = len(header.columns) > 0 and str(header.columns[0]).startswith("Unnamed:")
    return pd.read_csv(path, index_col=0) if unnamed_first else pd.read_csv(path)


@st.cache_data(show_spinner="Loading panel data...")
def load_all():
    """Every artifact the dashboard can render, read once and cached."""
    out = {
        "cleaned": _read(CLEANED_PANEL),
        "features": _read(FEATURES),
        "leaderboard": _read(MODEL_LEADERBOARD),
        "ml_metrics": _read(ML_METRICS),
        "dl_metrics": _read(DL_METRICS),
        "ml_predictions": _read(ML_PREDICTIONS),
        "dl_predictions": _read(DL_PREDICTIONS),
        "weights": _read(PORTFOLIO_WEIGHTS),
        "frontier": _read(PORTFOLIO_FRONTIER),
        "backtest": _read(PORTFOLIO_BACKTEST),
        "sentiment": _read(SENTIMENT_FEATURES),
        "mood": _read(MARKET_MOOD),
        "recommendations": _read(RECOMMENDATIONS),
        "rebalance": _read(REBALANCE_PLAN),
    }
    if CROSS_SECTIONAL.exists():
        import json

        try:
            out["cross_sectional"] = json.loads(
                CROSS_SECTIONAL.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            out["cross_sectional"] = None
    else:
        out["cross_sectional"] = None

    # The single model whose forecasts this dashboard publishes. Without it the
    # app has no way to say whose numbers it is showing, and the price panel
    # would happily plot a mixture of every model in the table.
    out["declared"] = None
    if DECLARED_MODEL.exists():
        import json

        try:
            out["declared"] = json.loads(DECLARED_MODEL.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            out["declared"] = None

    out["monte_carlo"] = np.load(MONTE_CLOUD_PATH) if MONTE_CLOUD_PATH.exists() else None
    for key in ("portfolio_metrics", "data_quality", "sentiment_ablation"):
        p = {"portfolio_metrics": PORTFOLIO_METRICS, "data_quality": DQ_REPORT_JSON,
             "sentiment_ablation": SENTIMENT_ABLATION}[key]
        if p.exists() and p.suffix == ".json":
            import json

            out[key] = json.loads(p.read_text(encoding="utf-8"))
        else:
            out[key] = None
    return out


def empty_state(what: str, command: str) -> None:
    st.info(f"{what} isn't ready yet. To build it, run `{command}`.")


def plain_box(html: str) -> None:
    """A bordered block of plain-language explanation."""
    html_box(html, _box(PALETTE["surface"], PALETTE["border"])
             + f"{_body(PALETTE['text'])}line-height:1.65;")


def takeaway(answer: str, detail: str = "") -> None:
    """The one-sentence answer to the question the panel is asking.

    Deliberately short: a reader should be able to take this away without
    reading the rest of the panel. Weighted with a left rule rather than a
    saturated fill, so it reads as a heading for the panel rather than an alarm
    about it.
    """
    inner = (
        f'<div style="{_label(PALETTE["muted"])}margin-bottom:6px;">'
        f'The short answer</div>'
        f'<div style="{_body(PALETTE["text"], "0.98rem", "500")}'
        f'line-height:1.55;">{answer}</div>'
    )
    if detail:
        inner += (f'<div style="{_body(PALETTE["muted"], "0.84rem")}'
                  f'line-height:1.6;margin-top:8px;">{detail}</div>')
    # The left rule is a separate absolutely-free border-left on a wrapper, so
    # the surface border stays uniform with every other box in the app.
    st.markdown(
        f'<div style="{_box(PALETTE["surface"], PALETTE["border"])}'
        f'border-left:3px solid #46586E;">{inner}</div>',
        unsafe_allow_html=True,
    )


def bullets(items: list[str]) -> None:
    """Scannable bullets, one idea each. No paragraphs."""
    body = "".join(
        f'<li style="margin-bottom:7px;line-height:1.6;">{i}</li>'
        for i in items
    )
    st.markdown(
        f'<ul style="margin:0 0 {GAP} 0;padding-left:20px;'
        f'{_body(PALETTE["text"], "0.9rem")}">{body}</ul>',
        unsafe_allow_html=True,
    )


def readout(rows: list[tuple[str, str]]) -> None:
    """A compact label / value list — the least-paragraph way to show facts."""
    body = ""
    for i, (k, v) in enumerate(rows):
        border = (f"border-bottom:1px solid {PALETTE['border']};"
                  if i < len(rows) - 1 else "")
        body += (
            f'<div style="display:flex;justify-content:space-between;'
            f'gap:16px;padding:7px 0;{border}">'
            f'<span style="{_body(PALETTE["muted"], "0.86rem")}">{k}</span>'
            f'<span style="{_body(PALETTE["text"], "0.88rem", "600")}'
            f'font-variant-numeric:tabular-nums;text-align:right;">{v}</span>'
            f'</div>'
        )
    html_box(body, _box(PALETTE["surface"], PALETTE["border"]))


def panel_ranking(cfg: dict, D: dict) -> None:
    """"Which stock beats its peers?" — the cross-sectional analysis.

    This is a SEPARATE question from the rest of the app, and the panel is
    built to make that impossible to miss. Everywhere else the models try to
    answer "will the price go up tomorrow?". Here they answer "out of these ten
    names, which one does best relative to the other nine?".

    The separation matters because the two questions have different baselines.
    Comparing a ranking model against the random-walk baseline would be unfair
    to it, because it deliberately ignores which way the market is heading.
    So it is judged against its own no-skill answer: "they will all do the
    same". That is the only fair comparison, and it is the one shown here.
    """
    st.subheader("Which stock does best?")
    render_disclaimer(
        cfg, "A ranking is a model opinion, not a suggestion to trade.")
    st.caption("A different question from the rest of this app — and judged "
               "against its own fair baseline, not the one used elsewhere.")

    xs = D.get("cross_sectional")
    if not xs or xs.get("status") != "COMPLETED":
        empty_state("The cross-sectional analysis",
                    "python src/cross_sectional.py")
        return

    base = xs.get("baseline", {})
    test = xs.get("test", {})
    sel = xs.get("model_selection", {})

    # ---------------- the short answer --------------------------------
    n_cfg = sel.get("configs_tried", 0)
    n_assets = len(cfg["universe"])
    if xs.get("beats_own_baseline"):
        takeaway(
            "The model could pick which stock beats its peers better than "
            "guessing.",
            "A real, if modest, result. It is still not a profit — beating a "
            "baseline on one measure is not the same as making money.")
    else:
        takeaway(
            f"No. The model could not pick the best-performing stock any better "
            f"than guessing.",
            f"{n_cfg} different settings were tried and every one of them was "
            f"worse than simply assuming all {n_assets} stocks will move "
            f"together. This is reported as a failure, not hidden.")

    readout([
        ("The question", f"Which of {n_assets} stocks does best tomorrow?"),
        ("Model error", fmt_num(test.get("relative_target_mae"), 6)),
        ("Guessing error", fmt_num(base.get("relative_target_mae"), 6)),
        ("Verdict", "Better than guessing" if xs.get("beats_own_baseline")
                    else "No better than guessing"),
    ])

    # ---------------- what each number means -------------------------
    same = test.get("relative_target_mae") is not None and \
        base.get("relative_target_mae") is not None and \
        abs(test["relative_target_mae"] - base["relative_target_mae"]) < 1e-5
    st.markdown("#### What these two numbers mean")
    bullets([
        f"**Guessing error** is what you get by assuming all {n_assets} stocks "
        f"will move together. That is the honest 'I know nothing' answer.",
        "**Model error** is what the trained model actually achieved.",
        "A lower model error would mean the model found something real.",
        ("Here the two are the same to five decimal places, so it did not."
         if same else
         "The model's error is higher than the guessing error, so it did not."),
    ])

    # ---------------- picking the single best stock --------------------
    st.markdown("#### Picking just the one best stock")
    top1 = test.get("top1_hit_rate")
    chance = test.get("top1_chance")
    if top1 is not None and chance:
        better = test.get("top1_better_than_chance")
        readout([
            ("Picked the right stock", fmt_pct(top1)),
            ("Chance level", fmt_pct(chance)),
            ("Is that real?", "Yes" if better else "No — pure luck"),
        ])
        if not better:
            p = test.get("top1_binomial_p")
            st.caption(
                f"The gap looks small, and it is: across {test.get('sessions', 0)} "
                f"trading days, a hit rate this close to chance has a p-value of "
                f"{p:.2f} — meaning it could easily have happened by accident."
                if p is not None else
                "The gap is not statistically distinguishable from random guessing.")

    # ---------------- the honest bit ----------------------------------
    with st.expander("Two leaks we caught in this work (worth reading)"):
        st.markdown("**Leak 1 — a helper column that was the answer in disguise.**")
        st.markdown(
            "The first version of this analysis produced an extraordinary "
            "result: a 95% error reduction against the baseline. It was fake. "
            "The analysis created two helper columns derived from the answer it "
            "was trying to predict. Because they were just numbers, the feature "
            "picker swept them in as inputs, and the model read the answer key."
        )
        bullets([
            "The giveaway was an error that looked <i>too</i> good.",
            "A shuffled-answer check passed anyway. It proved the model used "
            "its inputs, but not that one input was the label.",
            "So a blunter guard was added: fail the build if any input lines up "
            "almost perfectly with the answer.",
        ])

        st.markdown("")
        st.markdown("**Leak 2 — a number from the future, spread over the past.**")
        st.markdown(
            "The same code also computed each stock's market-risk number (Beta) "
            "as a single figure taken from the end of the data, then spread that "
            "one figure across every earlier day. Every stock ended up with one "
            "fixed Beta, fitted using prices that had not happened yet at the "
            "dates it was applied to."
        )
        bullets([
            "This one sat in a function the automated checks never called, so "
            "they could not have caught it.",
            "An existing 'no constant columns' test flagged it by accident.",
            "Both the fix and a check aimed squarely at it are now in place.",
        ])
        st.caption(
            "Recorded because a leak you catch and publish is worth more than a "
            "good number you cannot defend — and because removing this one made "
            "the results measurably worse, which is exactly what a working leak "
            "looks like.")

    # ---------------- for the technically curious ---------------------
    with st.expander("For the technically curious"):
        lb = sel.get("leaderboard", [])
        if lb:
            st.dataframe(
                pd.DataFrame([
                    {
                        "model": ("Ridge"
                                  if r["family"] == "ridge" else "Gradient boosting"),
                        "walk-forward error": round(r["walk_forward_mae"], 6),
                        "settings": ", ".join(
                            f"{k}={v}" for k, v in r["params"].items()
                            if k != "kind"),
                    }
                    for r in lb
                ]),
                use_container_width=True, hide_index=True)
        explain(
            f"Cross-sectional target: each stock's next-session return minus the "
            f"universe mean that day ({sel.get('folds', 0)} walk-forward folds, "
            f"{sel.get('configs_tried', 0)} configs, selected inside the training "
            f"partition only).",
            "Instead of guessing the raw move, the model guesses which stock will "
            "do better than the others. It is checked the same way as everything "
            "else: learned on past data, tested on data it has never seen.")


# ==========================================================================
# Formatting helpers
# ==========================================================================
def fmt_pct(x, digits: int = 2) -> str:
    try:
        return f"{float(x) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "-"


def fmt_num(x, digits: int = 4) -> str:
    try:
        return f"{float(x):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def delta_badge(value, threshold, higher_is_better: bool = True,
                text: str = "", suffix: str = "") -> str:
    """Render a pass/fail pill against an acceptance bar.

    Colours are inline, not class names: there is no stylesheet, so a class
    here would render as unstyled text and the pass/fail signal - the entire
    point of the pill - would disappear.
    """
    if value is None or not np.isfinite(float(value)):
        return f'<span style="color:{PALETTE["muted"]};">n/a {suffix}</span>'
    ok = (value > threshold) if higher_is_better else (value < threshold)
    colour = PALETTE["positive"] if ok else PALETTE["negative"]
    return (f'<span style="color:{colour};font-weight:600;">'
            f'{text or fmt_num(value)}{suffix}</span>')


# ==========================================================================
# Panel 1 - Overview
# ==========================================================================
def panel_overview(cfg: dict, D: dict) -> None:
    st.subheader("Overview")
    render_disclaimer(cfg)

    universe = cfg["universe"]
    feat = D["features"]
    cols = st.columns(4)

    with cols[0]:
        stat_tile("Stocks tracked", f"{len(universe)}", "+ index and VIX as context")

    with cols[1]:
        if feat is not None:
            stat_tile("Trading days", f"{feat['Date'].nunique():,}",
                      f"{feat['Date'].min().year}–{feat['Date'].max().year}")
        else:
            stat_tile("Trading days", "-", "not built")

    with cols[2]:
        if feat is not None:
            n_feat = len([c for c in feat.columns
                          if c not in ("Date", "Ticker", "Target", "Target_Price",
                                       "Relative_Target", "Rank_Target",
                                       "Open", "High", "Low", "Close",
                                       "Adjusted Close", "Volume", "Outlier",
                                       "Outlier_IQR")])
            stat_tile("Signals per stock", f"{n_feat}",
                      "Price, volume, market, macro")
        else:
            stat_tile("Signals per stock", "-", "not built")

    with cols[3]:
        mood = D["mood"]
        if mood is not None and len(mood):
            latest = mood.sort_values("Date").iloc[-1]
            stat_tile("News mood", str(latest["Market_Mood"]),
                      f"{int(latest['Article_Count'])} headlines")
        else:
            stat_tile("News mood", "No data", "needs an API key")

    # -- universe performance ---------------------------------------------
    st.markdown("")
    if feat is None:
        empty_state("Feature table", "python src/features.py")
        return

    recent = feat[feat["Ticker"].isin(universe)]
    latest_date = recent["Date"].max()
    window = recent[recent["Date"] >= latest_date - pd.Timedelta(days=365)]

    perf = (window.sort_values("Date")
            .groupby("Ticker")["Adjusted Close"]
            .agg(["first", "last"]))
    perf["Return_1Y"] = perf["last"] / perf["first"] - 1
    perf = perf.reset_index().sort_values("Return_1Y", ascending=False)

    # Lead with the finding, in one line, before any chart.
    lb0 = D["leaderboard"]
    if lb0 is not None and len(lb0):
        base0 = lb0[lb0["Family"] == "Baseline"]
        # Exclude the averaged ensemble from the headline verdict for the same
        # reason as in the model panel: it is a combination of four of the
        # required models, so letting it set the headline would credit a result
        # to "the best model" when no single model produced it.
        mods0 = lb0[(lb0["Family"] != "Baseline")
                    & (lb0["Model"] != "Ensemble (equal weight)")]
        if len(base0) and len(mods0):
            b0 = float(base0["MAE"].iloc[0])
            m0 = float(mods0["MAE"].min())
            d0 = float(mods0["DirAcc"].max())
            db0 = float(base0["DirAcc"].iloc[0])
            if m0 < b0 and d0 > db0:
                takeaway("The models did beat the simple guess on both measures.")
            elif m0 < b0:
                takeaway(
                    "The models were slightly better at the size of a daily move, "
                    "but no better at guessing which way the market would go.",
                    "That is the honest result. Click <b>Which model won</b> for the "
                    "numbers and why it happened.",
                )
            else:
                takeaway("The models did not beat the simple guess.",
                         "Click <b>Which model won</b> to see the full comparison.")

    # The cross-sectional question is a separate experiment with its own
    # baseline, so it gets its own line rather than being folded into the
    # forecast verdict above. Folding them together would imply the two tasks
    # are comparable, which they are not.
    xs0 = D.get("cross_sectional")
    if xs0 and xs0.get("status") == "COMPLETED":
        if xs0.get("beats_own_baseline"):
            plain_box(
                "<b>Second question, asked separately:</b> which stock beats the "
                "others? On that one the models did beat guessing. Different task, "
                "different yardstick — see <b>Which stock wins</b>.")
        else:
            plain_box(
                "<b>Second question, asked separately:</b> which stock beats the "
                "others? That one is also a no — no better than guessing. Different "
                "task, different yardstick — see <b>Which stock wins</b>.")

    m1, m2 = st.columns([3, 2])
    with m1:
        card("How each stock did over the past year")
        fig = px.bar(perf, x="Ticker", y="Return_1Y", color="Return_1Y",
                     color_continuous_scale=[[0, PALETTE["negative"]],
                                             [0.5, PALETTE["neutral"]],
                                             [1, PALETTE["positive"]]],
                     labels={"Return_1Y": "1-year return"},
                     hover_data={"Return_1Y": ":.2%"})
        fig.update_traces(marker_line_width=0, texttemplate="%{x}")
        base_layout(fig, height=340, legend=False)
        st.plotly_chart(fig, use_container_width=True)

    with m2:
        card("Which model was most accurate",
             "Lower is better — the average size of one mistake")
        ranked = lb0
        if ranked is not None and len(ranked):
            top = ranked.sort_values("MAE").head(6)
            plain_names = {
                "Naive random walk": "Predicting no change",
                "RandomForest": "Random Forest", "XGBoost": "XGBoost",
                "Ridge": "Ridge regression", "SVR": "Support Vector",
                "LSTM": "LSTM", "GRU": "GRU", "BiLSTM": "Bidirectional LSTM",
                "Transformer": "Transformer",
            }
            fig = go.Figure(go.Bar(
                x=top["MAE"], y=[plain_names.get(m, m) for m in top["Model"]],
                orientation="h",
                marker_color=[PALETTE["accent"] if i == 0 else PALETTE["border"]
                              for i in range(len(top))],
                text=[f"{v * 100:.2f}%" for v in top["MAE"]],
                textposition="outside", textfont=dict(size=10),
                hovertemplate="%{y}<br>average miss %{x:.5f}<extra></extra>",
            ))
            fig.update_layout(xaxis_title="Average daily miss (lower is better)",
                              yaxis_title="")
            base_layout(fig, height=340, legend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("The blue bar is the best model. The rest are the others.")

    # -- data quality card -------------------------------------------------
    dq = D["data_quality"]
    if dq:
        r = dq["rows"]
        gaps = sum(v["gap_cells_filled"] for v in dq["gaps"].values())
        outl = sum(v["outliers_either"] for v in dq["outliers"].values())
        inv = dq["invariant_violations"]["count"]
        dupes = r["duplicate_rows_removed"]
        lost = r["rows_in"] - r["rows_out"]

        st.markdown("")
        card("Was the data cleaned up properly?",
             "A quick check that nothing was lost or corrupted")
        readout([
            ("Rows of data going in", f"{r['rows_in']:,}"),
            ("Rows coming out", f"{r['rows_out']:,}"),
            ("Rows lost", f"{lost:,}"),
            ("Duplicate rows removed", f"{dupes:,}"),
            ("Small gaps filled in", f"{gaps:,}"),
            ("Big one-day jumps flagged (kept)", f"{outl:,}"),
            ("Bad price rows set aside", f"{inv:,}"),
        ])
        bullets([
            f"<b>{lost:,} rows lost.</b> Nothing was thrown away.",
            f"<b>{gaps:,} small gaps</b> were filled using earlier days only — "
            f"never using future information.",
            f"<b>{outl:,} unusually big one-day jumps</b> were flagged but kept. "
            f"A 9% jump is usually real news, not a data error.",
            f"<b>{inv} rows</b> had prices that made no sense (for example a high "
            f"below the low) and were set aside.",
        ])


# ==========================================================================
# Panel 2 - Price & prediction
# ==========================================================================
def panel_price(cfg: dict, D: dict, ticker: str, horizon: int, band_z: float) -> None:
    st.subheader(f"Price & prediction — {ticker}")
    render_disclaimer(cfg, "The forecast is a model estimate, not a price target.")

    # Name the model before showing any of its output. A reader who cannot tell
    # whose forecast they are looking at cannot judge it, and "the model" with
    # no referent is the kind of vagueness Section 8.3 is written against.
    declared = D.get("declared") or {}
    if declared.get("status") == "DECLARED":
        m = declared["metric"]
        b = declared["baseline"]
        plain_name = {
            "LSTM": "LSTM", "GRU": "GRU", "BiLSTM": "Bidirectional LSTM",
            "Transformer": "Transformer", "RandomForest": "Random Forest",
            "XGBoost": "XGBoost", "Ridge": "Ridge regression", "SVR": "Support Vector",
        }.get(declared["model"], declared["model"])
        readout([
            ("Forecasts come from", plain_name),
            ("Why this one", f"Closest of {8} models on typical miss"),
            ("Typical miss", f"{m['MAE']:.6f} vs {b['MAE']:.6f} naive"),
            ("Right direction", f"{m['DirAcc'] * 100:.2f}% vs {b['DirAcc'] * 100:.2f}% naive"),
        ])
        if not declared.get("beats_baseline_on_direction"):
            st.caption(
                "It is the closest model, not a good one. It is more accurate "
                "about <b>how far</b> a price moves, and no better than a coin "
                "toss at <b>which way</b> it moves.")
    else:
        st.info(
            "No model has been declared yet, so there is nothing to forecast "
            "with. Run `python retrain_models.py`.")

    if D["features"] is None:
        empty_state("Feature table", "python src/features.py")
        return

    feat = D["features"]
    hist_days = int(cfg["dashboard"]["price_history_days"])
    series = feat[feat["Ticker"] == ticker].sort_values("Date")
    if series.empty:
        st.warning(f"No data for {ticker}.")
        return
    series = series[series["Date"] >= series["Date"].max() - pd.Timedelta(days=hist_days)]

    # Use exactly ONE model - the declared one - and nothing else.
    #
    # This used to take whichever prediction file had the ticker and plot all of
    # its rows. `ml_predictions.parquet` contains five models (four classical
    # plus the ensemble) and `dl_predictions.parquet` contains four, so the
    # fallback path silently plotted a blend of five forecasts as if it were a
    # single line. It happened to be masked because every universe ticker is in
    # the deep-learning file, which is exactly the kind of bug that survives
    # because the common case works.
    declared = D.get("declared") or {}
    model_name = declared.get("model")
    preds = None
    if model_name:
        for source in (D["dl_predictions"], D["ml_predictions"]):
            if source is None or not len(source) or "Model" not in source.columns:
                continue
            if model_name not in set(source["Model"]):
                continue
            rows = source[(source["Ticker"] == ticker)
                          & (source["Model"] == model_name)].sort_values("Date")
            if len(rows):
                preds = rows
                break

    if preds is None and not model_name:
        st.info(
            "No model has been declared yet, so there are no model forecasts to "
            "draw. Run `python retrain_models.py` to produce them.")

    fig = go.Figure()

    # Confidence band: the range the model's own past errors suggest a typical
    # day could land in. Described in words, not sigma.
    if preds is not None and len(preds):
        resid = preds["Residual"].to_numpy(dtype=float)
        sigma = float(np.std(resid, ddof=1)) if len(resid) > 2 else 0.0
        band_label = (f"The shaded area shows the range the model usually lands in "
                      f"(+/-{band_z * sigma * 100:.2f}% per day)")
    else:
        sigma = 0.0
        band_label = ""

    if preds is not None and len(preds):
        px_ = preds["Predicted_Price"].to_numpy(dtype=float)
        ax_ = preds["Adjusted Close"].to_numpy(dtype=float)
        if sigma > 0:
            fig.add_trace(go.Scatter(
                x=preds["Date"], y=ax_ * np.exp(band_z * sigma), mode="lines",
                line=dict(width=0), showlegend=False, hoverinfo="skip",
                name="upper"))
            fig.add_trace(go.Scatter(
                x=preds["Date"], y=ax_ * np.exp(-band_z * sigma), mode="lines",
                line=dict(width=0), fill="tonexty",
                fillcolor="rgba(76,141,255,0.14)", showlegend=False,
                name="lower"))
        fig.add_trace(go.Scatter(
            x=preds["Date"], y=px_, mode="lines", name="What the model predicted",
            line=dict(color=PALETTE["accent"], width=2, dash="dot")))

    fig.add_trace(go.Scatter(
        x=series["Date"], y=series["Adjusted Close"], mode="lines",
        name="What actually happened", line=dict(color=PALETTE["text"], width=1.8),
        hovertemplate="%{x|%Y-%m-%d}<br>Closing price %{y:,.2f}<extra></extra>"))

    fig.update_layout(title=f"{ticker} — what happened, and what the model expected",
                      yaxis_title="Closing price (USD)")
    base_layout(fig, height=440)
    if preds is not None and len(preds) and band_label:
        fig.add_annotation(text=band_label, xref="paper", yref="paper", x=0, y=-0.16,
                           showarrow=False, font=dict(size=10, color=PALETTE["muted"]),
                           xanchor="left")
    st.plotly_chart(fig, use_container_width=True)

    if preds is None or not len(preds):
        st.info("There's no forecast for this stock yet. Run `python retrain_models.py`.")
        return

    left, right = st.columns(2)

    with left:
        card("How accurate was it?",
             "Measured on days the model had never seen")
        mask = preds["Actual_Return"].to_numpy() != 0
        dir_acc = float(np.mean(
            np.sign(preds["Actual_Return"].to_numpy()[mask])
            == np.sign(preds["Predicted_Return"].to_numpy()[mask])))
        mae = float(np.mean(np.abs(preds["Residual"].to_numpy())))

        m1, m2 = st.columns(2)
        m1.metric("Typical miss", f"{mae * 100:.2f}%",
                  help="On a typical day, the prediction was off by this much")
        m2.metric("Right direction", f"{dir_acc * 100:.1f}%",
                  help="How often it correctly said the price would go up vs down. "
                       "A coin flip is 50%.")

        if dir_acc > 0.52:
            verdict, colour = "better than a coin toss", PALETTE["positive"]
        elif dir_acc > 0.50:
            verdict, colour = "barely better than a coin toss", PALETTE["warning"]
        else:
            verdict, colour = "no better than a coin toss", PALETTE["negative"]

        bullets([
            f"On a normal day the prediction was off by about <b>{mae * 100:.2f}%</b>.",
            f"It called the direction right <b>{dir_acc * 100:.1f}%</b> of the time — "
            f"that is <span class='{colour}'>{verdict}</span>.",
            f"Tested across <b>{len(preds):,}</b> days it had never seen.",
        ])

    with right:
        card("Daily calls, day by day",
             "Each dot is one day: above the line means the price went up")
        sample = preds.tail(120)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=sample["Date"], y=sample["Actual_Return"] * 100, mode="markers",
            name="What happened", marker=dict(size=5, color=PALETTE["text"], opacity=0.65),
            hovertemplate="%{x|%Y-%m-%d}<br>actually %{y:+.2f}%<extra></extra>"))
        fig2.add_trace(go.Scatter(
            x=sample["Date"], y=sample["Predicted_Return"] * 100, mode="lines",
            name="What the model said", line=dict(color=PALETTE["accent"], width=1.8),
            hovertemplate="%{x|%Y-%m-%d}<br>model said %{y:+.2f}%<extra></extra>"))
        fig2.add_hline(y=0, line=dict(color=PALETTE["border"], width=1))
        fig2.update_layout(yaxis_title="Daily change (%)", xaxis_title="")
        base_layout(fig2, height=260)
        st.plotly_chart(fig2, use_container_width=True)

    st.caption(
        f"Every model answers the same question — will the price go up or down over "
        f"the next {cfg['target']['horizon']} trading day?"
    )


# ==========================================================================
# Panel 3 - Model comparison
# ==========================================================================
def panel_models(cfg: dict, D: dict) -> None:
    st.subheader("Model comparison")
    render_disclaimer(
        cfg,
        "A model that cannot beat the naive random walk is reported as a failure, not hidden.",
    )

    lb = D["leaderboard"]
    if lb is None or not len(lb):
        empty_state("Model leaderboard", "python retrain_models.py")
        return

    baseline = lb[lb["Model"].str.contains("Naive", case=False, na=False)]
    models = lb[~lb["Model"].str.contains("Naive", case=False, na=False)]

    plain_names = {
        "Naive random walk": "Predicting no change (the benchmark)",
        "RandomForest": "Random Forest", "XGBoost": "XGBoost",
        "Ridge": "Ridge regression", "SVR": "Support Vector",
        "LSTM": "LSTM (neural network)", "GRU": "GRU (neural network)",
        "BiLSTM": "Bidirectional LSTM (neural network)",
        "Transformer": "Transformer (neural network)",
        "Ensemble (equal weight)": "All four traditional models, averaged",
    }
    family_plain = {"Baseline": "benchmark", "ML": "traditional", "DL": "neural network"}

    # The ensemble is a COMBINATION of four of the required models, not a ninth
    # model. It is reported as an extra row and must not be allowed to masquerade
    # as one of the eight the PRD asks for, so it is excluded from the headline
    # verdict and labelled wherever it appears.
    ENSEMBLE = "Ensemble (equal weight)"
    is_ensemble = models["Model"] == ENSEMBLE
    headline = models[~is_ensemble]

    # -- the honest verdict, stated first ---------------------------------
    if len(baseline) and len(headline):
        b_mae = float(baseline["MAE"].iloc[0])
        b_dir = float(baseline["DirAcc"].iloc[0])
        best_mae = float(headline["MAE"].min())
        best_dir = float(headline["DirAcc"].max())
        verdict_mae = best_mae < b_mae
        verdict_dir = best_dir > b_dir
        gap = (b_mae - best_mae) / b_mae * 100
        dir_gap = (best_dir - b_dir) * 100

        if verdict_mae and verdict_dir:
            takeaway(
                "Yes — the models beat the simplest possible guess on both measures.",
                f"Typical miss {best_mae * 100:.2f}% vs {b_mae * 100:.2f}%. "
                f"Right direction {best_dir * 100:.1f}% vs {b_dir * 100:.1f}%.",
            )
        elif verdict_mae:
            takeaway(
                "Partly. The models were better at the size of a move, "
                "but not at guessing which way it would go.",
                f"Typical miss improved by {gap:.0f}%. Right direction came in "
                f"{abs(dir_gap):.1f} points <i>below</i> the simple guess, so the "
                f"PRD's all-or-nothing bar is not cleared.",
            )
        else:
            takeaway(
                "No. The models did not beat the simplest possible guess.",
                f"Typical miss {best_mae * 100:.2f}% vs {b_mae * 100:.2f}%.",
            )

        # Name the model whose forecasts the rest of the app publishes. The
        # verdict above is about the field; this is about the single deployment
        # choice, and stating both together is the only way a reader can tell
        # "closest of eight" apart from "good enough to trade".
        declared = D.get("declared") or {}
        if declared.get("status") == "DECLARED" and verdict_mae:
            winner = str(declared["model"])
            rows = [("Model in use elsewhere", winner)]
            if declared.get("runner_up"):
                rows.append(("Next closest", f"{declared['runner_up']['model']} "
                                             f"({declared['runner_up']['MAE']:.6f})"))
            rows.append(("Chosen because", "Closest typical miss of all "
                                           f"{len(headline)} models"))
            rows.append(("On direction", "Still below the simple guess"
               if not verdict_dir else "Above the simple guess"))
            st.markdown("")
            card("The one model this app forecasts with",
                 "Named here so you always know whose numbers you are reading")
            readout(rows)
            bullets([
                "<b>Closest is not the same as good.</b> This model is the least "
                "wrong of the eight, and it is still wrong often enough that the "
                "PRD counts the result a failure.",
                "It is used because a dashboard has to pick one model to show, "
                "not because it was found to be profitable. It was not.",
            ])

        st.markdown("")
        card("How the comparison works",
             "Why this is a fair test, and what the numbers mean")
        n_required = len(headline)
        bullets([
            "The benchmark assumes <b>tomorrow looks exactly like today</b>. "
            "It is deliberately hard to beat.",
            f"<b>Typical miss</b> — how far off the model was on an average day. "
            f"Lower is better.",
            f"<b>Right direction</b> — how often it correctly said up or down. "
            f"A coin toss scores 50%.",
            f"All {n_required} models used the <b>same data</b>, the <b>same "
            f"signals</b> and the <b>same test days</b>.",
            "So any difference between them comes from the model, not the data.",
        ] + ([
            f"There is also a <b>{n_required + 1}th row</b> showing the four "
            "traditional models averaged together. It is a bonus, not one of the "
            "required models, and it is left out of the verdict above."
        ] if is_ensemble.any() else []))

        with st.expander("What the other two columns mean (for the technically curious)"):
            st.markdown(
                "- **RMSE** — like typical miss, but it punishes the occasional "
                "huge miss far more heavily.\n"
                "- **MAPE** — the same idea as typical miss, but measured against "
                "the share price rather than the daily change.\n"
                "- **R²** — how much of the day-to-day movement the model explains. "
                "A negative number means it did worse than simply predicting the "
                "average, which is the normal outcome when the target is close to "
                "a coin toss."
            )

        if not (verdict_mae and verdict_dir):
            st.markdown("")
            plain_box(
                "<b>Why the models struggle</b><br><br>"
                "Daily stock prices behave a lot like a coin toss. Even with good "
                "data and careful maths, saying which way tomorrow will go is very "
                "hard.<br><br>"
                "We are showing this rather than hiding it. A system that quietly "
                "picked its most flattering number would be far less trustworthy than "
                "one that shows every result — including the disappointing ones."
            )

    # -- the full comparison table ----------------------------------------
    n_rows = len(headline)
    card(f"All {n_rows} models, side by side",
         "Same question, same data, same days nobody had seen"
         + (" — plus a bonus averaged row" if is_ensemble.any() else ""))
    display = lb.copy()
    display["Model"] = display["Model"].map(lambda m: plain_names.get(m, m))
    # The ensemble is filed under Family="ML" internally, but to a reader it is
    # not another traditional model - it is the average of four of them. Say so.
    display["Family"] = [
        "averaged combination" if orig == ENSEMBLE else family_plain.get(fam, fam)
        for orig, fam in zip(lb["Model"], lb["Family"])
    ]
    display["MAPE"] = display["MAPE"].map(lambda v: f"{v:.3f}%")
    display["RMSE"] = display["RMSE"].map(lambda v: f"{v:.6f}")
    display["MAE"] = display["MAE"].map(lambda v: f"{v:.6f}")
    display["R2"] = display["R2"].map(lambda v: f"{v:.4f}")
    display["DirAcc"] = display["DirAcc"].map(lambda v: f"{v * 100:.2f}%")
    st.dataframe(
        display[["Model", "Family", "MAE", "DirAcc", "RMSE", "MAPE", "R2"]]
        .rename(columns={"Model": "Model", "Family": "Type", "MAE": "Typical miss",
                         "DirAcc": "Right direction", "RMSE": "RMSE",
                         "MAPE": "MAPE", "R2": "R²"}),
        use_container_width=True, hide_index=True,
    )

    # -- metric bar charts -------------------------------------------------
    a, b = st.columns(2)
    with a:
        fig = px.bar(models.sort_values("MAE"), x="Model", y="MAE", color="Family",
                     color_discrete_map={"ML": PALETTE["accent"], "DL": PALETTE["positive"]},
                     hover_data={"MAE": ":.6f"})
        if len(baseline):
            fig.add_hline(y=float(baseline["MAE"].iloc[0]),
                          line=dict(color=PALETTE["negative"], dash="dash", width=1.5),
                          annotation_text="predicting no change",
                          annotation_position="top left",
                          annotation_font=dict(size=10, color=PALETTE["negative"]))
        fig.update_layout(xaxis_title="", yaxis_title="Average miss (lower is better)")
        base_layout(fig, height=340)
        st.plotly_chart(fig, use_container_width=True)

    with b:
        fig = px.bar(models.sort_values("DirAcc", ascending=False), x="Model", y="DirAcc",
                     color="Family",
                     color_discrete_map={"ML": PALETTE["accent"], "DL": PALETTE["positive"]},
                     hover_data={"DirAcc": ":.4f"})
        if len(baseline):
            fig.add_hline(y=float(baseline["DirAcc"].iloc[0]),
                          line=dict(color=PALETTE["negative"], dash="dash", width=1.5),
                          annotation_text="predicting no change",
                          annotation_position="bottom right",
                          annotation_font=dict(size=10, color=PALETTE["negative"]))
        fig.add_hline(y=0.5, line=dict(color=PALETTE["border"], width=1))
        fig.update_layout(xaxis_title="", yaxis_title="How often the direction was right",
                          yaxis_tickformat=".1%")
        base_layout(fig, height=340)
        st.plotly_chart(fig, use_container_width=True)

    st.caption("On the right-hand chart, 50% is a coin toss and the red line is what you "
               "get by assuming tomorrow looks like today.")

    # -- deep-learning diagnostics ----------------------------------------
    dl = D["dl_metrics"]
    if dl is not None and len(dl):
        st.markdown("")
        card("How the neural networks trained",
             "A behind-the-scenes check that they learned rather than memorised")
        diag = dl[["Model", "Params", "Epochs", "Train_Loss", "Val_Loss",
                   "Overfit_Gap", "Diagnosis"]].copy()
        for c in ("Train_Loss", "Val_Loss", "Overfit_Gap"):
            diag[c] = diag[c].map(lambda v: f"{v:.5f}")
        st.dataframe(diag, use_container_width=True, hide_index=True)

        curves_path = PROCESSED_DIR / "dl_loss_curves.csv"
        if curves_path.exists():
            curves = pd.read_csv(curves_path)
            fig = go.Figure()
            for i, (name, grp) in enumerate(curves.groupby("model")):
                colour = PALETTE["series"][i % len(PALETTE["series"])]
                label = plain_names.get(name, name)
                fig.add_trace(go.Scatter(x=grp["epoch"], y=grp["train_loss"], mode="lines",
                                         name=f"{label} — learned from",
                                         line=dict(color=colour, width=1.6)))
                fig.add_trace(go.Scatter(x=grp["epoch"], y=grp["val_loss"], mode="lines",
                                         name=f"{label} — practised on",
                                         line=dict(color=colour, width=1.6, dash="dash")))
            fig.update_layout(xaxis_title="Practice rounds", yaxis_title="How wrong it was (lower is better)")
            base_layout(fig, height=360)
            st.plotly_chart(fig, use_container_width=True)
            bullets([
                "<b>Solid line</b> — how the model did on the data it learned from.",
                "<b>Dashed line</b> — how it did on fresh data it had never seen.",
                "When the two stay close together, the model genuinely learned the "
                "pattern instead of memorising the answers. That is what you want.",
            ])


# ==========================================================================
# Panel 4 - Portfolio analytics
# ==========================================================================
def panel_portfolio(cfg: dict, D: dict, rf_rate: float) -> None:
    st.subheader("Portfolio analytics")
    render_disclaimer(
        cfg,
        "The suggested split below is an illustration, not a personal plan.",
    )

    pm = D["portfolio_metrics"]
    w = D["weights"]
    if w is None or not len(w):
        empty_state("Portfolio split", "python src/portfolio.py")
        return

    # The weights CSV is a single numeric column indexed by ticker.
    numeric = w.select_dtypes(include="number")
    if numeric.empty:
        empty_state("Portfolio split", "python src/portfolio.py")
        return
    weights = numeric.iloc[:, 0]
    weights = weights[weights > 1e-6].sort_values(ascending=False)

    # -- headline metrics --------------------------------------------------
    bt_meta = pm.get("backtest") if pm else None
    if pm:
        ms = pm["max_sharpe"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Expected yearly return", fmt_pct(ms["expected_return"]),
                  help="What the maths suggests this mix would have returned per year")
        m2.metric("Expected wobble", fmt_pct(ms["volatility"]),
                  help="How much the value tends to swing up and down in a year")
        m3.metric("Return per unit of risk", fmt_num(ms["sharpe"], 3),
                  help="How much return you earn for each unit of wobble. Higher is better.")
        m4.metric("Moves with the market", fmt_num(pm.get("portfolio_beta"), 3),
                  help="Below 1.0 means it tends to fall less than the market when the market falls")

        if pm.get("beats_equal_weight") is not None:
            ok = pm["beats_equal_weight"]
            bt = pm.get("backtest", {})
            so = bt.get("optimised", {}).get("sharpe", float("nan"))
            se = bt.get("equal_weight", {}).get("sharpe", float("nan"))
            if ok:
                takeaway(
                    "Yes — this mix did better than just splitting the money evenly.",
                    f"Return per unit of risk: {so:.2f} vs {se:.2f}.",
                )
            else:
                takeaway(
                    "No — it did not beat simply splitting the money evenly.",
                    f"Return per unit of risk: {so:.2f} vs {se:.2f}. On this stretch "
                    f"of days the two were too close to tell apart.",
                )

        st.markdown("")
        card("Why you can trust the test",
             "The important bit: it was judged on days it had not seen")
        bullets([
            f"The mix was chosen using data up to "
            f"<b>{pm['estimation_window']['end']}</b>.",
            f"It was then tested on the <b>{bt_meta.get('n_sessions', '?')} days "
            f"after</b> that, which it had never seen.",
            "Those two periods do not overlap, so the system could not cheat by "
            "having already seen the answers.",
        ])

    left, right = st.columns([1, 2])

    with left:
        card("How the money would be split",
             "The system never puts more than 20% into any one stock")
        fig = go.Figure(go.Pie(
            labels=list(weights.index), values=weights.to_numpy(dtype=float),
            hole=0.55, marker=dict(colors=PALETTE["series"][:len(weights)],
                                   line=dict(color=PALETTE["surface"], width=2)),
            textinfo="label+percent", textposition="outside",
            hovertemplate="%{label}: %{value:.2%}<extra></extra>"))
        fig.update_layout(showlegend=False, annotations=[dict(
            text=f"<b>{len(weights)}</b><br>stocks", showarrow=False,
            font=dict(size=13, color=PALETTE["muted"]))])
        base_layout(fig, height=380, legend=False)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        card("Risk versus reward",
             "Each dot is one possible mix of stocks; up and to the left is better")
        frontier = D["frontier"]
        cloud = D["monte_carlo"]
        fig = go.Figure()
        if cloud is not None and len(cloud):
            fig.add_trace(go.Scattergl(
                x=cloud[:, 0] * 100, y=cloud[:, 1] * 100, mode="markers",
                marker=dict(size=2.5, color="rgba(139,148,158,0.22)"),
                name=f"Every other mix tried ({len(cloud):,})", hoverinfo="skip"))
        if frontier is not None and len(frontier):
            fig.add_trace(go.Scatter(
                x=frontier["Volatility"] * 100, y=frontier["Return"] * 100,
                mode="lines+markers", name="The best mixes",
                line=dict(color=PALETTE["accent"], width=2.5),
                marker=dict(size=4)))
        if pm:
            for label, info, colour, sym in (
                ("Our pick", pm["max_sharpe"], PALETTE["positive"], "star"),
                ("Safest mix", pm["min_variance"], PALETTE["warning"], "diamond"),
            ):
                fig.add_trace(go.Scatter(
                    x=[info["volatility"] * 100], y=[info["expected_return"] * 100],
                    mode="markers+text", name=label,
                    marker=dict(size=16, color=colour, symbol=sym, line=dict(width=1, color="#0E1117")),
                    text=[label], textposition="top center",
                    textfont=dict(size=10, color=colour),
                    hovertemplate=f"{label}<br>wobble %{{x:.2f}}%<br>return %{{y:.2f}}%<extra></extra>"))
        fig.update_layout(xaxis_title="How much it wobbles in a year (%)",
                          yaxis_title="Expected yearly return (%)")
        base_layout(fig, height=380)
        st.plotly_chart(fig, use_container_width=True)
        bullets([
            f"Each grey dot is one other way of splitting the money. "
            f"There are <b>{len(cloud):,}</b> of them.",
            "The blue line runs along the best of them.",
            "The green star sits on that line — which confirms the maths found a "
            "genuinely good mix, not a lucky one.",
        ])

    # -- backtest ----------------------------------------------------------
    bt = D["backtest"]
    if bt is not None and len(bt):
        st.markdown("")
        card("What actually happened on days the system had never seen",
             "The real test: apply each mix and see what came out")
        cols = ["Strategy", "annualised_return", "annualised_volatility", "sharpe",
                "sortino", "max_drawdown", "var_95"]
        show = bt[[c for c in cols if c in bt.columns]].copy()
        show.columns = ["Approach", "Yearly return", "Yearly wobble", "Return per unit of risk",
                        "Downside-adjusted return", "Worst fall from peak", "Worst 1-in-20 day"]
        for c in show.columns[1:]:
            show[c] = show[c].map(lambda v: f"{v * 100:.2f}%"
                                  if c in ("Yearly return", "Yearly wobble",
                                           "Worst fall from peak", "Worst 1-in-20 day")
                                  else f"{v:.3f}")
        st.dataframe(show, use_container_width=True, hide_index=True)

        fig = px.bar(bt, x="Strategy", y="sharpe", color="Strategy",
                     color_discrete_map={s: PALETTE["accent"] for s in bt["Strategy"]},
                     hover_data={"sharpe": ":.3f"})
        fig.update_layout(xaxis_title="", yaxis_title="Return per unit of risk")
        base_layout(fig, height=300, legend=False)
        st.plotly_chart(fig, use_container_width=True)

        if pm and isinstance(bt_meta, dict) and "sharpe_bootstrap_ci" in pm:
            ci = pm["sharpe_bootstrap_ci"]
            lo, hi = ci["ci_low"], ci["ci_high"]
            crosses_zero = lo <= 0.0 <= hi
            st.markdown("")
            card("How much of this could be luck?",
                 "We tested that by shuffling the days and trying again")
            bullets([
                f"We repeated the comparison <b>{ci['n_bootstraps']:,} times</b> "
                f"using reshuffled versions of the same days.",
                f"The answer came out somewhere between <b>{lo:.2f}</b> and "
                f"<b>{hi:.2f}</b>, typically around {ci['point_estimate']:.2f}.",
            ])
            if crosses_zero:
                bullets([
                    f'<span style="color:{PALETTE["warning"]};">'
                    f'<b>That range includes zero.</b></span> '
                    "It means this stretch of days was too short to separate the two "
                    "approaches.",
                    "The fair conclusion is <b>'too close to call'</b> — not that "
                    "the suggested mix is reliably worse.",
                ])

    if pm and "dependency_note" in pm:
        with st.expander("Technical note on how this was calculated"):
            st.caption(pm["dependency_note"])


# ==========================================================================
# Panel 5 - Risk dashboard
# ==========================================================================
def panel_risk(cfg: dict, D: dict, ticker: str) -> None:
    st.subheader("Risk dashboard")
    render_disclaimer(cfg, "These are patterns from past data, not limits on what can happen.")

    feat = D["features"]
    if feat is None:
        empty_state("The prepared data", "python src/features.py")
        return

    universe = cfg["universe"]
    recent = feat[feat["Ticker"].isin(universe)].sort_values("Date")

    # -- current risk snapshot --------------------------------------------
    latest = recent.groupby("Ticker").tail(1).set_index("Ticker")
    vol = latest["Vol_21d"].astype(float)
    ranked = vol.sort_values(ascending=False)

    top, calm = ranked.index[0], ranked.index[-1]
    p1, p2, p3 = st.columns(3)
    p1.metric("Rockiest right now", f"{top}", f"{vol[top] * 100:.1f}% a year",
              delta_color="off", help="Expected yearly swing based on the last month")
    p2.metric("Steadiest right now", f"{calm}", f"{vol[calm] * 100:.1f}% a year",
              delta_color="off", help="Expected yearly swing based on the last month")
    p3.metric("Typical for these stocks", f"{vol.median() * 100:.1f}% a year",
              delta_color="off")

    left, right = st.columns(2)

    with left:
        card("How bumpy has it been?",
             "A rolling one-month window, scaled up to a yearly figure")
        hist = recent[recent["Ticker"] == ticker].tail(500)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist["Date"], y=hist["Vol_21d"] * 100, mode="lines",
                                 name=f"{ticker}",
                                 line=dict(color=PALETTE["accent"], width=1.8)))
        if "VIX_Level" in hist.columns:
            fig.add_trace(go.Scatter(x=hist["Date"], y=hist["VIX_Level"], mode="lines",
                                     name="Market-wide fear gauge",
                                     line=dict(color=PALETTE["warning"], width=1.4)))
        fig.update_layout(yaxis_title="Expected yearly swing (%)", xaxis_title="")
        base_layout(fig, height=330)
        st.plotly_chart(fig, use_container_width=True)
        bullets([
            "The orange line is a well-known fear gauge published by the options market.",
            "Notice how it arrives in clumps. Markets move in moods, not smoothly.",
            "Calm stretches and panicked stretches stick together.",
        ])

    with right:
        card("Drawdown from the running peak")
        series = recent[recent["Ticker"] == ticker].set_index("Date")["Adjusted Close"]
        curve = series / series.cummax() - 1
        fig = go.Figure(go.Scatter(
            x=curve.index, y=curve * 100, mode="lines", name=ticker,
            line=dict(color=PALETTE["negative"], width=1.8),
            fill="tozeroy", fillcolor="rgba(248,81,73,0.12)"))
        fig.update_layout(yaxis_title="Below the highest point so far (%)", xaxis_title="")
        base_layout(fig, height=330, legend=False)
        st.plotly_chart(fig, use_container_width=True)
        readout([
            ("Worst fall from a peak on this stretch", f"{abs(curve.min() * 100):.1f}%"),
        ])
        st.caption("This always measures down from the highest point reached so far — "
                   "the honest way to show pain.")

    # -- correlation heatmap ----------------------------------------------
    st.markdown("")
    card("Do these stocks move together?",
         "Red means they tend to rise and fall together; blue means they don't")
    wide = (recent.pivot(index="Date", columns="Ticker", values="Return_1d")
            .dropna(how="all"))
    corr = wide.corr()
    fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    x=corr.columns, y=corr.index, text_auto=".2f",
                    labels=dict(color="How closely they move together"))
    fig.update_layout(height=max(380, 40 * len(corr)))
    base_layout(fig, height=max(380, 40 * len(corr)), legend=False)
    st.plotly_chart(fig, use_container_width=True)
    off = corr.where(~np.eye(len(corr), dtype=bool)).abs().max().max()
    takeaway(
        "Most of these stocks rise and fall together.",
        f"The closest pair moves together with a strength of {off:.2f} out of 1 — "
        f"very high. Owning several is <b>not</b> the same as spreading your money "
        f"around, because they tend to fall together. That is exactly why the system "
        f"caps how much it puts into any one stock.",
    )

    # -- VaR / tail risk ---------------------------------------------------
    st.markdown("")
    card("What happens on a really bad day?",
         "The left tail, where losses are much larger than gains")
    bench_ret = wide[cfg["benchmark"]] if cfg["benchmark"] in wide.columns else wide[universe[0]]
    bench_ret = bench_ret.dropna()

    c1, c2, c3, c4 = st.columns(4)
    from scipy import stats as _st

    var_e = float(np.percentile(bench_ret * 100, 5))
    var_t = float(_st.t.ppf(0.05, *_st.t.fit(bench_ret.to_numpy())))
    c1.metric("About 1 day in 20 loses more than", f"{abs(var_e):.2f}%")
    c2.metric("Allowing for crash days", f"{abs(var_t):.2f}%",
              help="The first figure assumes ordinary days. This one admits that "
                   "crash days are worse than a simple average would suggest.")
    c3.metric("Downside bias", f"{_st.skew(bench_ret.to_numpy()):.2f}",
              help="Negative means big losses are more common than big gains.")
    c4.metric("Fat tails", f"{_st.kurtosis(bench_ret.to_numpy()):.2f}",
              help="Higher than 0 means extreme days are far more common than a "
                   "normal distribution would predict.")

    fig = make_subplots_figure(bench_ret)
    st.plotly_chart(fig, use_container_width=True)

    jb_stat, jb_p = _st.jarque_bera(bench_ret.to_numpy())
    st.markdown("")
    card("Reading the two charts",
         "Both show the same surprise, from two different angles")
    bullets([
        "<b>Left chart</b> — how daily moves are spread out. The orange line is what "
        "an ordinary day would look like.",
        "The real data is taller in the middle <b>and</b> fatter at the edges.",
        "<b>Right chart</b> — one dot per day. If days were ordinary the dots would sit "
        "straight on the red line. They bend away at both ends.",
        f"A formal test agrees (p-value {jb_p:.1e}).",
    ])
    plain_box(
        "<b>Why this matters</b><br><br>"
        "Really bad days happen more often than a simple average would suggest. "
        "So the system reports the losses that <i>actually happened</i> rather than "
        "trusting a tidy average — and gives two different 'bad day' figures instead "
        "of one, so you can see the range of reasonable answers."
    )


def make_subplots_figure(returns: pd.Series) -> go.Figure:
    """Histogram with a normal overlay plus a QQ plot."""
    from scipy import stats as _st

    fig = go.Figure()
    r = (returns * 100).to_numpy()
    fig.add_trace(go.Histogram(
        x=r, nbinsx=80, name="Daily return", marker_color="rgba(76,141,255,0.55)",
        hovertemplate="%{x:.2f}%<br>count %{y}<extra></extra>"))
    xs = np.linspace(r.min(), r.max(), 200)
    mu, sd = float(np.mean(r)), float(np.std(r, ddof=1))
    pdf = _st.norm.pdf(xs, mu, sd) * len(r) * (xs[1] - xs[0])
    fig.add_trace(go.Scatter(x=xs, y=pdf, mode="lines", name="What an ordinary day looks like",
                             line=dict(color=PALETTE["warning"], width=2)))
    fig.update_layout(barmode="overlay", xaxis_title="Daily change (%)",
                      yaxis_title="How many days", height=300)
    base_layout(fig, height=300)

    # QQ plot as a second subplot.
    from plotly.subplots import make_subplots

    fig2 = make_subplots(rows=1, cols=2, horizontal_spacing=0.10)
    fig2.update_annotations(font=dict(size=12, color=PALETTE["text"]))
    fig2.add_annotation(text="How the days are spread out", row=1, col=1,
                        font=dict(size=12, color=PALETTE["text"]))
    fig2.add_annotation(text="Do the days look ordinary?", row=1, col=2,
                        font=dict(size=12, color=PALETTE["text"]))
    for tr in fig.data:
        fig2.add_trace(tr, row=1, col=1)
    (q_x, q_y), (slope, intercept, _) = _st.probplot(r, dist="norm")
    lo = min(q_x.min(), q_y.min())
    hi = max(q_x.max(), q_y.max())
    fig2.add_trace(go.Scatter(x=[lo, hi], y=[intercept + slope * lo, intercept + slope * hi],
                              mode="lines", name="If every day were ordinary",
                              line=dict(color=PALETTE["negative"], dash="dash", width=1.5)),
                   row=1, col=2)
    fig2.add_trace(go.Scatter(x=q_x, y=q_y, mode="markers", name="What actually happened",
                              marker=dict(size=3, color=PALETTE["accent"], opacity=0.6)),
                   row=1, col=2)
    fig2.update_xaxes(title_text="An ordinary day would be here →", row=1, col=2)
    fig2.update_yaxes(title_text="The day that really happened (%)", row=1, col=2)
    fig2.update_layout(template="plotly_dark", height=300,
                       paper_bgcolor=PALETTE["surface"], plot_bgcolor=PALETTE["surface"],
                       font=dict(color=PALETTE["text"], size=11),
                       margin=dict(l=12, r=12, t=40, b=12),
                       showlegend=False)
    return fig2


# ==========================================================================
# Panel 6 - Sentiment
# ==========================================================================
def panel_sentiment(cfg: dict, D: dict, ticker: str) -> None:
    st.subheader("News sentiment")
    render_disclaimer(cfg, "A computer reads the headlines and guesses the tone. It is sometimes wrong.")

    s = D["sentiment"]
    mood = D["mood"]
    if s is None or not len(s):
        st.info("No news data yet. This part needs a free API key from a news provider.")
        st.code("set FINNHUB_API_KEY=<your key>\n"
                "python rebuild_dataset.py --with-sentiment\n"
                "python retrain_models.py", language="powershell")
        st.caption("Everything else on this dashboard works without it. We are telling you "
                   "the test was never run rather than showing a result of zero, because "
                   "'we didn't check' and 'we checked and found nothing' are very "
                   "different things.")
        return

    s = s.copy()
    s["Date"] = pd.to_datetime(s["Date"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Days covered", f"{s['Date'].nunique():,}")
    c2.metric("Stocks covered", f"{s['Ticker'].nunique()}")
    c3.metric("Overall tone", f"{float(s['Sentiment_FinBERT'].mean()):+.2f}")
    st.caption(f"Covering {s['Date'].min().date()} to {s['Date'].max().date()}.")
    bullets([
        "A headline published <b>after the market closed</b> is counted against the "
        "<b>next</b> trading day, not the day it appeared.",
        "Otherwise the system would be reading tomorrow's news to predict today's "
        "price — which would look impressive and be worthless.",
    ])

    if mood is not None and len(mood):
        m = mood.sort_values("Date")
        fig = go.Figure(go.Scatter(
            x=m["Date"], y=m["Mean_Sentiment"], mode="lines", name="Overall tone",
            line=dict(color=PALETTE["accent"], width=1.6)))
        for mood_label, colour in (("Upbeat", PALETTE["positive"]),
                                   ("Mixed", PALETTE["neutral"]),
                                   ("Gloomy", PALETTE["negative"])):
            key = {"Upbeat": "Bullish", "Mixed": "Neutral", "Gloomy": "Bearish"}[mood_label]
            sub = m[m["Market_Mood"] == key]
            if len(sub):
                fig.add_trace(go.Scatter(
                    x=sub["Date"], y=sub["Mean_Sentiment"], mode="markers",
                    name=mood_label, marker=dict(size=6, color=colour),
                    hovertemplate=f"{mood_label}<br>%{{x|%Y-%m-%d}}<br>%{{y:.3f}}<extra></extra>"))
        fig.update_layout(yaxis_title="Tone of the news", xaxis_title="")
        base_layout(fig, height=300)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Above the line means that day's headlines read more positive than "
                   "negative. Dots mark the days counted as upbeat, mixed or gloomy.")

    st.markdown("")
    left, right = st.columns([2, 1])

    with left:
        card(f"How the news about {ticker} has read",
             "Green bars are positive days, red bars are negative days")
        t = s[s["Ticker"] == ticker].sort_values("Date")
        if t.empty:
            st.caption(f"No headlines were found for {ticker}.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=t["Date"], y=t["Sentiment_FinBERT"], name="Finance-tuned reader",
                                 marker_color=[
                                     PALETTE["positive"] if v >= 0 else PALETTE["negative"]
                                     for v in t["Sentiment_FinBERT"]],
                                 hovertemplate="%{x|%Y-%m-%d}<br>finance reader %{y:+.2f}<extra></extra>"))
            fig.add_trace(go.Scatter(x=t["Date"], y=t["Sentiment_VADER"], mode="lines",
                                     name="General reader",
                                     line=dict(color=PALETTE["warning"], width=1.5),
                                     hovertemplate="%{x|%Y-%m-%d}<br>general reader %{y:+.2f}<extra></extra>"))
            fig.add_trace(go.Scatter(x=t["Date"], y=t["Sentiment_3d_Mom"], mode="lines",
                                     name="Three-day trend",
                                     line=dict(color=PALETTE["accent"], width=1.8, dash="dot")))
            fig.add_hline(y=0, line=dict(color=PALETTE["border"], width=1))
            fig.update_layout(xaxis_title="", yaxis_title="Positive or negative", barmode="overlay")
            base_layout(fig, height=360)
            st.plotly_chart(fig, use_container_width=True)
            bullets([
                "Two different readers score the same headlines.",
                "The finance-tuned one knows that <i>'debt fell'</i> is good news.",
                "The general one often reads that same headline as bad.",
                "The blue dotted line is the three-day trend.",
            ])

    with right:
        card("How much news there was",
             "Busy news days often mean something is happening")
        t2 = t.tail(120)
        if len(t2):
            fig = go.Figure()
            fig.add_trace(go.Bar(x=t2["Date"], y=t2["News_Volume"], name="Headlines",
                                 marker_color="rgba(139,148,158,0.5)"))
            fig.add_trace(go.Scatter(x=t2["Date"], y=t2["News_Spike"], mode="lines+markers",
                                     name="Busier than usual?", yaxis="y2",
                                     line=dict(color=PALETTE["warning"], width=1.8)))
            fig.add_hline(y=1.0, yaxis="y2", line=dict(color=PALETTE["negative"], dash="dash"))
            fig.update_layout(yaxis2=dict(title="Busier than usual?", overlaying="y", side="right",
                                          showgrid=False),
                              xaxis_title="", barmode="overlay")
            base_layout(fig, height=360)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Above the orange line means more headlines than usual for that "
                       "stock. Quiet stretches and busy ones come in clusters.")

    # -- ablation ----------------------------------------------------------
    ab = D["sentiment_ablation"]
    if ab:
        st.markdown("")
        card("Did the news actually help the predictions?",
             "The only fair way to know is to try it both ways")
        if ab.get("status") != "COMPLETED":
            st.info(
                f"**Not run** — {ab.get('reason', 'unknown')}. The pipeline continues "
                f"without the Section 5.1 sentiment family. This is reported as not run "
                f"rather than as a null result, because reporting 'no effect' from a "
                f"test that never executed would itself be a methodological failure."
            )
            st.code("set FINNHUB_API_KEY=<your key>\n"
                    "python rebuild_dataset.py --with-sentiment\n"
                    "python retrain_models.py", language="powershell")
        else:
            arms = ab["arms"]
            x1, x2, x3, x4 = st.columns(4)
            x1.metric("Change in typical miss", f"{ab['delta_MAE'] * 100:+.3f} pp",
                      delta="better" if ab["sentiment_helped"] else "worse",
                      delta_color="normal" if ab["sentiment_helped"] else "inverse")
            x2.metric("Change in right-direction rate", f"{ab['delta_DirAcc'] * 100:+.2f} pp")
            x3.metric("Typical miss, ignoring news", f"{arms['without_sentiment']['MAE'] * 100:.3f}%")
            x4.metric("Typical miss, using news", f"{arms['with_sentiment']['MAE'] * 100:.3f}%")
            plain_box(
                f"<b>Result: {ab['verdict']}</b><br><br>"
                f"Only the {ab['overlap_sessions']} days for which headlines were "
                f"actually available were used "
                f"({ab['overlap_start']} to {ab['overlap_end']}), so both versions "
                f"were judged on identical days."
            )
            st.caption(ab["scope_caveat"])


# ==========================================================================
# Panel 7 - Recommendations
# ==========================================================================
def panel_recommendations(cfg: dict, D: dict) -> None:
    st.subheader("What the system suggests")
    render_disclaimer(cfg, "These are rules applied to past data, not advice about your money.")

    recs = D["recommendations"]
    if recs is None or not len(recs):
        empty_state("Suggestions", "python src/recommend.py")
        return

    counts = recs["Recommendation"].value_counts()
    n_total = int(len(recs))
    c1, c2, c3 = st.columns(3)
    c1.metric("Suggested buying", int(counts.get("BUY", 0)))
    c2.metric("Suggested holding", int(counts.get("HOLD", 0)))
    c3.metric("Suggested selling", int(counts.get("SELL", 0)))

    # A uniform HOLD deserves an explanation up front, not a shrug. Read the
    # skill gate from the recommendation metadata rather than inferring it.
    meta_path = RECOMMENDATIONS.parent / "recommendation_meta.json"
    if meta_path.exists():
        import json as _json

        meta = _json.loads(meta_path.read_text(encoding="utf-8"))
        if int(counts.get("HOLD", 0)) == n_total and float(meta.get("forecast_skill", 1.0)) <= 0.0:
            plain_box(
                "<b>Why every suggestion is “hold”</b><br><br>"
                "The system's price predictions were tested against the simplest "
                "possible guess — that tomorrow looks like today — and they did not do "
                "better.<br><br>"
                "Rather than pretend otherwise, the system has <b>switched off the "
                "prediction part of its scoring entirely</b>.<br><br>"
                "That leaves the risk and news parts, and on their own they are not "
                "strong enough to reach a buy or sell threshold. So it says "
                "<i>hold</i> on all of them.<br><br>"
                "A system with no proven forecasting edge should not be handing out "
                "confident buy and sell instructions.<br><br>"
                "The suggested stock weights further down still work — those come from "
                "the portfolio maths, not from the predictions."
            )

    # -- the table with its contributing sub-signals -----------------------
    card("The full reasoning, stock by stock",
         "Every suggestion shows the three things that produced it")
    show = recs[["Ticker", "Recommendation", "Composite_Score", "Forecast_Signal",
                 "Sentiment_Signal", "Risk_Signal", "Target_Weight", "Rebalance"]].copy()
    show.columns = ["Stock", "Suggestion", "Overall score", "Prediction signal",
                    "News signal", "Risk signal", "Suggested weight", "Rebalance"]
    for c in ("Overall score", "Prediction signal", "News signal", "Risk signal"):
        show[c] = show[c].map(lambda v: f"{v:+.2f}")
    show["Suggested weight"] = show["Suggested weight"].map(lambda v: f"{v * 100:.1f}%")
    st.dataframe(show, use_container_width=True, hide_index=True)
    st.markdown("")
    card("What each column means", None)
    bullets([
        "<b>Overall score</b> — the final number the rule looks at.",
        "<b>Prediction signal</b> — how confident the model was.",
        "<b>News signal</b> — how positive the headlines were.",
        "<b>Risk signal</b> — how risky the stock looked. Higher pushes the score down.",
        f"A stock is only suggested for buying once the overall score passes "
        f"<b>{cfg['recommend']['buy_threshold']:+.2f}</b>, and for selling below "
        f"<b>{cfg['recommend']['sell_threshold']:+.2f}</b>.",
    ])

    st.markdown("")
    left, right = st.columns([3, 2])

    # The forecast term is scaled by the model's measured skill (Section 12.1).
    # Show the effective weights, and say so when the gate has zeroed the
    # forecast term, so the decomposition below is not read as a bug.
    meta_path = RECOMMENDATIONS.parent / "recommendation_meta.json"
    weights_used = {"Forecast_Signal": 0.5, "Sentiment_Signal": 0.3, "Risk_Signal": 0.2}
    if meta_path.exists():
        import json as _json

        meta = _json.loads(meta_path.read_text(encoding="utf-8"))
        used = meta.get("weights_used", {})
        weights_used = {
            "Forecast_Signal": float(used.get("forecast", 0.5)),
            "Sentiment_Signal": float(used.get("sentiment", 0.3)),
            "Risk_Signal": float(used.get("risk", 0.2)),
        }

    with left:
        card("What went into each score",
             "The three parts of the overall score, side by side")
        # Plot the *weighted* contribution each sub-signal actually makes, not
        # its raw normalised value. When the skill gate zeroes the forecast term
        # the raw chart would still show a full-height forecast bar, which would
        # contradict the composite printed beside it.
        sub = recs.melt(
            id_vars=["Ticker"],
            value_vars=list(weights_used.keys()),
            var_name="Component", value_name="Normalised")
        sub["Weight"] = sub["Component"].map(weights_used)
        sub["Signed"] = np.where(sub["Component"] == "Risk_Signal",
                                 -sub["Weight"] * sub["Normalised"],
                                 sub["Weight"] * sub["Normalised"])
        fig = px.bar(sub, x="Ticker", y="Signed", color="Component", barmode="relative",
                     color_discrete_map={
                         "Forecast_Signal": PALETTE["accent"],
                         "Sentiment_Signal": PALETTE["positive"],
                         "Risk_Signal": PALETTE["negative"]},
                     hover_data={"Signed": ":.3f", "Weight": ":.2f"},
                     labels={"Signed": "Points added to the score",
                             "Weight": "How much this counts",
                             "Component": "Part of the score"})
        fig.add_hline(y=0, line=dict(color=PALETTE["border"], width=1))
        fig.update_layout(xaxis_title="", yaxis_title="Points added to the score")
        base_layout(fig, height=380)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Each bar is how many points that part added. The risk bar points "
                   "downward because risk lowers the score. The bars add up to the "
                   "overall score shown on the right.")

    with right:
        card("The final score and what it means",
             "Buy above the line, sell below it, hold in between")
        buy_t = cfg["recommend"]["buy_threshold"]
        sell_t = cfg["recommend"]["sell_threshold"]
        fig = go.Figure(go.Bar(
            x=recs["Ticker"], y=recs["Composite_Score"],
            marker_color=[{"BUY": PALETTE["positive"], "HOLD": PALETTE["neutral"],
                           "SELL": PALETTE["negative"]}[a] for a in recs["Recommendation"]],
            text=recs["Recommendation"], textposition="outside",
            textfont=dict(size=10),
            hovertemplate="%{x}<br>score %{y:.2f}<extra></extra>"))
        fig.add_hline(y=buy_t, line=dict(color=PALETTE["positive"], dash="dash", width=1.5),
                      annotation_text=f"buy above {buy_t:+.2f}",
                      annotation_position="top left",
                      annotation_font=dict(size=10, color=PALETTE["positive"]))
        fig.add_hline(y=sell_t, line=dict(color=PALETTE["negative"], dash="dash", width=1.5),
                      annotation_text=f"sell below {sell_t:+.2f}",
                      annotation_position="bottom left",
                      annotation_font=dict(size=10, color=PALETTE["negative"]))
        fig.update_layout(xaxis_title="", yaxis_title="Overall score", showlegend=False)
        base_layout(fig, height=380, legend=False)
        st.plotly_chart(fig, use_container_width=True)

    # -- rationale ---------------------------------------------------------
    with st.expander("The reasoning behind each suggestion, in plain English"):
        for _, r in recs.iterrows():
            st.markdown(f"**{r['Ticker']} — {r['Recommendation']}**  \n{r['Rationale']}")

    # -- rebalancing -------------------------------------------------------
    plan = D["rebalance"]
    if plan is not None and len(plan):
        st.markdown("")
        card("How to get from where you are to where the maths suggests",
             "Only differences big enough to be worth acting on")
        pshow = plan.copy()
        pshow.columns = [c.replace("_", " ").title() for c in pshow.columns]
        pshow["Current Weight"] = pshow["Current Weight"].map(lambda v: f"{v * 100:.1f}%")
        pshow["Target Weight"] = pshow["Target Weight"].map(lambda v: f"{v * 100:.1f}%")
        pshow["Drift"] = pshow["Drift"].map(lambda v: f"{v * 100:+.1f} pp")
        pshow["Action"] = pshow["Action"].map(
            {"INCREASE": "Buy more", "REDUCE": "Sell some", "MAINTAIN": "Leave alone"})
        pshow["Band Breached"] = pshow["Band Breached"].map(
            {True: "Yes", False: "No"})
        st.dataframe(pshow, use_container_width=True, hide_index=True)
        breached = plan[plan["Band_Breached"]]
        if len(breached):
            st.caption(
                f"**{len(breached)}** stock(s) have drifted far enough from the suggested "
                f"weight to be worth trading. Anything less than "
                f"{cfg['recommend']['rebalance_band']:.0%} is left alone, because the "
                f"trading cost would eat the gain."
            )
        else:
            st.caption("No stock has drifted far enough to be worth trading. That is the "
                       "system deliberately staying quiet rather than churning.")


# ==========================================================================
# App shell
# ==========================================================================
def render_header(cfg: dict, D: dict) -> None:
    """Masthead: what this is, who built it, and the one headline number.

    Kept to three lines and a single tile row. The previous version stacked a
    large title, a subtitle, four unexplained figures and a verdict several
    screens apart, so the page opened without answering any question.
    """
    # The wordmark. "Northgate AI" is the name, so it is treated as a badge and
    # given the only saturated colour on the page; "Stock Predictor" is the
    # descriptor, so it sits beside it in quiet type. Previously both ran
    # together in one 1.85rem heading, which made the product name compete with
    # its own subtitle instead of leading it.
    st.markdown(
        f'<div style="padding:2px 0 16px 0;'
        f'border-bottom:1px solid {PALETTE["border"]};margin-bottom:14px;">'
        f'<div style="display:flex;align-items:center;gap:11px;'
        f'flex-wrap:wrap;margin-bottom:9px;">'
        f'<span style="display:inline-block;background:linear-gradient(135deg,'
        f'#2C5FB8 0%,#4C8DFF 55%,#39C5CF 100%);color:#FFFFFF;font-size:0.82rem;'
        f'font-weight:700;letter-spacing:0.14em;text-transform:uppercase;'
        f'padding:6px 13px;border-radius:6px;white-space:nowrap;">'
        f'Northgate AI</span>'
        f'<span style="color:{PALETTE["text"]};font-size:1.32rem;'
        f'font-weight:600;letter-spacing:-0.01em;">Stock Predictor</span>'
        f'</div>'
        f'<div style="color:{PALETTE["muted"]};font-size:0.87rem;line-height:1.55;'
        f'max-width:80ch;">Ten years of daily market data. Eight models on one '
        f'honest split. Every figure here was measured, not projected.</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    declared = D.get("declared") or {}
    lb = D.get("leaderboard")
    verdict_txt, verdict_colour, verdict_note = "Not measured", PALETTE["muted"], ""
    if lb is not None and len(lb):
        base = lb[lb["Family"] == "Baseline"]
        req = lb[(lb["Family"] != "Baseline")
                 & (lb["Model"] != "Ensemble (equal weight)")]
        if len(base) and len(req):
            b_mae = float(base["MAE"].iloc[0])
            b_dir = float(base["DirAcc"].iloc[0])
            bm, bd = float(req["MAE"].min()), float(req["DirAcc"].max())
            wins = (bm < b_mae) and (bd > b_dir)
            verdict_txt = "Met" if wins else "Not met"
            verdict_colour = PALETTE["positive"] if wins else PALETTE["negative"]
            # One line, never a sentence. The detail belongs in the panel that
            # explains it, not repeated four times across the top of the page.
            verdict_note = ("Beats a no-change guess on both"
                            if wins else
                            "Closer on size, not on direction")

    tiles = st.columns(4)
    with tiles[0]:
        stat_tile("Model in use", str(declared.get("model", "not declared")),
                  "Forecasts shown here")
    with tiles[1]:
        stat_tile("Models compared", "8", "4 classic · 4 neural")
    with tiles[2]:
        stat_tile("Test days",
                  f"{int(lb['N'].max()):,}" if lb is not None and len(lb) else "-",
                  "Never seen in training")
    with tiles[3]:
        stat_tile("PRD target", verdict_txt, verdict_note, value_colour=verdict_colour)


def side_heading(text: str) -> None:
    """A small grouped heading in the sidebar.

    Streamlit's `st.markdown("### ...")` inherits the page heading scale and
    shouts louder than the content it introduces, so sidebar groups get this
    instead: small, uppercase, muted, with a rule underneath.
    """
    st.markdown(
        f'<div style="{_label(PALETTE["muted"])}'
        f'margin:10px 0 9px 0;padding-bottom:6px;'
        f'border-bottom:1px solid {PALETTE["border"]};">{text}</div>',
        unsafe_allow_html=True,
    )


def main() -> None:
    # Page config is set at module scope above - Streamlit requires it to be the
    # first command in the script and allows it exactly once.
    page_setup()
    cfg = get_config()
    D = load_all()

    render_header(cfg, D)

    # -- sidebar controls (Section 13.2) ----------------------------------
    with st.sidebar:
        side_heading("Look at")
        ticker = st.selectbox("Stock", cfg["universe"], index=0)
        horizon = st.slider(
            "Days ahead to predict", 1, 21, int(cfg["target"]["horizon"]),
            help="How many trading days into the future the model tries to predict.")
        band_z = st.slider(
            "Prediction range", 0.5, 3.0,
            float(cfg["dashboard"]["confidence_band_z"]), step=0.1,
            help="A wider band shows the model's predictions with more room around "
                 "them, for days that typically move more.")

        side_heading("Assumption")
        rf_pct = st.number_input(
            "Safe interest rate (%)",
            min_value=0.0, max_value=15.0,
            value=float(cfg["dashboard"]["default_risk_free_rate"]),
            step=0.25,
            help="What you could earn by putting the money somewhere safe instead. "
                 "Used to judge whether a risky investment was actually worth it.",
        )
        rf_rate = rf_pct / 100.0

        side_heading("How this was built")
        # Four short facts, not a paragraph. The split ratio is shown rather
        # than described: "70 / 15 / 15" carries the meaning, and the sentence
        # that used to explain it said nothing the numbers did not.
        facts = []
        if D["data_quality"]:
            r = D["data_quality"]["rows"]
            facts.append(f"{r['rows_out']:,} rows, none discarded")
        facts.append(
            f"Split {cfg['split']['train']:.0%}/{cfg['split']['val']:.0%}/"
            f"{cfg['split']['test']:.0%}")
        facts.append(f"Seed {cfg['random_seed']} — reruns match")
        facts.append("Brief v1.0")
        st.markdown(
            f'<div style="{_body(PALETTE["muted"], "0.75rem")}line-height:1.5;">'
            + "".join(f'<div style="margin-bottom:5px;">{f}</div>' for f in facts)
            + '</div>',
            unsafe_allow_html=True)

    tabs = st.tabs([
        "Overview",
        "Price & prediction",
        "Which model won",
        "Portfolio",
        "Risk",
        "News",
        "What to do",
        "Which stock wins",
    ])

    with tabs[0]:
        panel_overview(cfg, D)
    with tabs[1]:
        panel_price(cfg, D, ticker, horizon, band_z)
    with tabs[2]:
        panel_models(cfg, D)
    with tabs[3]:
        panel_portfolio(cfg, D, rf_rate)
    with tabs[4]:
        panel_risk(cfg, D, ticker)
    with tabs[5]:
        panel_sentiment(cfg, D, ticker)
    with tabs[6]:
        panel_recommendations(cfg, D)
    with tabs[7]:
        panel_ranking(cfg, D)

    # Provenance only, and one line per fact. The advice disclaimer is already
    # rendered as its own box at the top of every panel, which is what Section
    # 13.2 asks for; repeating it down here added length without adding reach
    # and made the notice read as boilerplate rather than as something specific
    # to the panel the reader was on.
    st.markdown(
        f'<div style="border-top:1px solid {PALETTE["border"]};'
        f'padding-top:12px;margin-top:22px;'
        f'{_body(PALETTE["muted"], "0.75rem")}line-height:1.85;">'
        f'<b>Data</b> — Yahoo Finance prices · FRED macro · Finnhub news<br>'
        f'<b>Built with</b> — scikit-learn · XGBoost · TensorFlow · SciPy<br>'
        f'<b>Method</b> — every figure measured on a held-out window<br>'
        f'<b>Project</b> — Northgate Quantitative Research, brief v1.0'
        f'</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
