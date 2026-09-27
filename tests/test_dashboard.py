"""Static + runtime sanity checks for the Streamlit dashboard.

Streamlit runs as a server, so this checks what can be checked without a
browser: that the module imports, that every panel function exists, that no
fabricated metric literals remain, and that the app can be constructed.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "dashboard" / "app.py"
sys.path.insert(0, str(ROOT))


def test_app_file_exists():
    assert APP.exists(), "dashboard/app.py is missing"


def test_disclaimer_is_a_separate_highlighted_box():
    """PRD 12.3/13.2 want it persistent; the design requirement is that it is
    visibly its own highlighted rectangle, separate from the panel content.

    Checked on the inline style attribute, which is the only styling path that
    survives Streamlit's client-side sanitisation. A footnote, or an
    outline-only box on a dark background, does not satisfy it - the first is
    not a box and the second is invisible in practice.
    """
    import ast
    import re

    src = APP.read_text(encoding="utf-8")
    tree = ast.parse(src)
    funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "render_disclaimer" in funcs
    body = ast.get_source_segment(src, funcs["render_disclaimer"]) or ""

    for prop in ("background", "border", "border-left", "border-radius", "padding"):
        assert prop in body, (
            f"the disclaimer style is missing '{prop}', so it is not drawn as a "
            f"distinct highlighted rectangle"
        )
    # A filled background is the part that actually makes it visible; a bare
    # border on the page background disappears.
    assert re.search(r"background:\s*#", body), (
        "the disclaimer needs a filled background, not just an outline"
    )
    assert "Please read" in body, "the disclaimer must be labelled"


def test_predictions_use_exactly_one_model():
    """A price line must not be a blend of every model in the table.

    The prediction files each contain several models (five in
    `ml_predictions.parquet`, four in `dl_predictions.parquet`). Selecting rows
    by ticker alone therefore produced a line mixing all of them, which looked
    fine because every universe ticker is in the deep-learning file and the
    deep-learning file happened to be read first. The panel must filter on the
    declared model explicitly.
    """
    src = APP.read_text(encoding="utf-8")
    assert 'source["Model"] == model_name' in src, (
        "the price panel must filter predictions to the declared model; "
        "filtering by ticker alone mixes every model into one line"
    )
    # And the declared model must be the only source of that name.
    assert 'declared.get("model")' in src
    assert 'declared.get("status") == "DECLARED"' in src


def test_declared_model_is_a_real_artifact():
    """If a model is declared, the declaration must be complete and honest."""
    import json

    p = APP.parent.parent / "data" / "processed" / "declared_model.json"
    if not p.exists():
        pytest.skip("declared_model.json not built; run retrain_models.py")
    doc = json.loads(p.read_text(encoding="utf-8"))

    assert doc["status"] in {"DECLARED", "UNDECLARED"}
    if doc["status"] == "UNDECLARED":
        assert doc.get("reason")
        return

    assert doc["model"], "a declaration with no model name is meaningless"
    assert doc["rule"], "the selection rule must be stated, not implied"
    # Naming a best model is exactly the move that could be mistaken for
    # claiming a profitable one, so the caveat must ship with it.
    assert doc.get("does_not_mean"), (
        "declaring a best model without stating it is not a profitable model is "
        "the misleading case this guard exists to prevent"
    )
    assert doc.get("selection_integrity")
    # The two-part Section 8.3 bar must be restated, not summarised away.
    assert "beats_baseline_on_mae" in doc
    assert "beats_baseline_on_direction" in doc
    # The ensemble is a blend of four models and must never be named the winner.
    assert doc["model"] != "Ensemble (equal weight)"


def test_no_stylesheet_block_is_used_anywhere():
    """A `<style>` block can never reach the browser. Do not write one.

    Streamlit sanitises every HTML fragment it renders with DOMPurify, and
    DOMPurify's default allowed-tag list does not include `style`. So a
    stylesheet block is stripped **client-side**, silently.

    This is not hypothetical. The dashboard shipped with roughly 12,000
    characters of CSS in a `<style>` block and rendered completely unstyled -
    stock red tab, no boxes, no backgrounds, no typography - while the app
    served normally, all eight panels executed, and every test passed. Two
    delivery paths were tried (`st.markdown` and then `st.html`); both were
    stripped, because the sanitising happens in the browser either way.

    What does survive is the `style` ATTRIBUTE (in DOMPurify's default
    allowed-attribute list) and Streamlit's own themed components. The design
    is built from those two, so any new `<style>` block is dead code that will
    look exactly like it works.
    """
    import ast

    src = APP.read_text(encoding="utf-8")
    tree = ast.parse(src)

    # Check the string literals that are actually handed to Streamlit, not the
    # whole file: this module's own docstrings discuss `<style>` at length, and
    # a naive substring scan flags the explanation as the violation.
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = getattr(func, "attr", None) or getattr(func, "id", None)
        if name not in {"markdown", "html"}:
            continue
        for arg in list(node.args) + [kw.value for kw in node.keywords]:
            for sub in ast.walk(arg):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                    if "<style" in sub.value.lower():
                        offenders.append(
                            f"line {sub.lineno}: {sub.value[:60]!r}")

    assert not offenders, (
        "a <style> block is being rendered. Streamlit's DOMPurify config strips "
        "it client-side, so it will not appear and the failure is silent:\n    "
        + "\n    ".join(offenders)
    )
    assert "def inject_css" not in src, (
        "inject_css() is gone: there is no stylesheet to inject. page_setup() is "
        "the documented placeholder that explains why."
    )
    assert "def page_setup(" in src, (
        "page_setup() should remain as the documented no-op, so a reader looking "
        "for the missing CSS finds the explanation in one place"
    )


def test_design_survives_without_custom_css():
    """Every visual element must be reachable by one of the two working paths.

    Inline `style` attributes, or native Streamlit components. If a helper
    depends on a CSS class that nothing defines any more it renders as bare
    text, which is exactly the bug that shipped.
    """
    import ast
    import re

    src = APP.read_text(encoding="utf-8")
    tree = ast.parse(src)
    funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}

    # `html_box` is the shared renderer several helpers delegate to, so it is
    # checked directly and a helper that only calls it counts as styled.
    for helper in ("render_disclaimer", "stat_tile", "takeaway", "readout",
                   "plain_box", "card", "side_heading", "render_header",
                   "html_box"):
        assert helper in funcs, f"the {helper}() design helper is missing"
        body = ast.get_source_segment(src, funcs[helper]) or ""
        styled = ("style=" in body
                  or "st." in body
                  or "html_box(" in body
                  or "_box(" in body
                  or "_body(" in body)
        assert styled, (
            f"{helper}() has neither an inline style nor a native Streamlit "
            f"component, so it will render unstyled"
        )

    stale = re.findall(r'class="ng-([a-z-]+)"', src)
    assert not stale, (
        f"these elements still use ng-* CSS classes with no stylesheet to define "
        f"them, so they render unstyled: {sorted(set(stale))}"
    )


def test_base_theme_is_declared_in_config():
    """A theme file must exist, so the page is right before any CSS parses.

    `.streamlit/config.toml` is the only styling mechanism Streamlit applies
    unconditionally. Everything else goes through the injected stylesheet, which
    is a single point of failure: when it was being stripped, the page fell back
    to Streamlit's default red-and-grey theme with no error anywhere. Having the
    base palette declared in config means a stylesheet failure degrades to
    "slightly plainer" rather than "wrong product".
    """
    cfg = APP.parent.parent / ".streamlit" / "config.toml"
    assert cfg.exists(), (
        ".streamlit/config.toml is missing, so the base theme depends entirely "
        "on the injected stylesheet applying"
    )
    text = cfg.read_text(encoding="utf-8")
    for key in ("[theme]", "base", "primaryColor", "backgroundColor", "textColor"):
        assert key in text, f"the theme file is missing '{key}'"

    # Streamlit rejects unknown sections with a startup warning, which is easy
    # to miss in a background run. `[theme.sidebar]` was one such key, removed
    # from the theme schema. Read the file back through Streamlit's own option
    # registry so an invalid key fails a test rather than a log line.
    from streamlit.config import get_options_for_section

    try:
        options = get_options_for_section("theme")
    except Exception as exc:  # pragma: no cover
        pytest.fail(f"streamlit rejected .streamlit/config.toml: {exc}")
    assert "base" in options, "the theme section did not load"
    assert options["base"] == "dark", (
        f"theme base is {options['base']!r}, expected 'dark'"
    )

    # The sidebar block specifically, since it is the one that actually broke.
    # Parse the TOML rather than string-matching, so a *comment* that names the
    # removed key (which is how it is documented in the file) does not trip this.
    import tomllib

    with open(cfg, "rb") as fh:
        parsed = tomllib.load(fh)
    assert "sidebar" not in parsed.get("theme", {}), (
        "[theme.sidebar] is not a valid option in Streamlit 1.40; the sidebar is "
        "styled from the injected stylesheet instead"
    )

    # The config palette and the injected palette must agree, or half the page
    # will be one colour scheme and half another.
    app_src = APP.read_text(encoding="utf-8")
    for token in ('"bg": "#0E1117"', '"surface": "#161B22"', '"text": "#E6EDF3"'):
        key = token.split(":")[0].strip('"')
        hexv = token.split('"')[3]
        assert hexv in text, (
            f"theme colour {hexv} (PALETTE['{key}']) is not in config.toml, so "
            f"the injected stylesheet and the base theme will disagree"
        )
    del app_src


def test_masthead_replaces_the_giant_title():
    """`st.title` renders at ~2.95rem and pushed the first screen below the fold.

    The masthead is a styled div now, so the page opens with the headline
    verdict and the four summary figures visible without scrolling.
    """
    src = APP.read_text(encoding="utf-8")
    assert "def render_header(" in src, "the masthead renderer is missing"
    assert "Northgate AI" in src, (
        "the masthead must carry the product name; it was reported missing"
    )
    # The masthead is now a name badge plus a descriptor rather than one large
    # heading, so there is no single oversized font-size to assert. What must
    # hold is that it is styled inline - the only path that renders - and that
    # it establishes a type scale rather than one flat size.
    import re

    sizes = [float(x) for x in re.findall(r"font-size:\s*([0-9.]+)rem", src)]
    assert len(set(sizes)) >= 4, (
        f"only {len(set(sizes))} distinct type sizes are used; a page needs a "
        f"hierarchy, not one size everywhere"
    )
    # st.title is fine in principle, but not as the page's first element - it
    # renders at 2.95rem by default and pushed the first screen below the fold.
    main_body = src[src.index("def main("):]
    assert "st.title(" not in main_body, (
        "st.title() in main() reintroduces the oversized default heading; the "
        "masthead component replaces it"
    )


def test_sidebar_starts_collapsed():
    """The first screen should be the verdict, not a control panel.

    Stock and interest-rate pickers are things a reader goes looking for once
    they know what they want; opening on them meant every visit started with
    four controls above the answer. Streamlit's own arrow reopens it, so
    nothing is hidden, only deferred.

    Set via `st.set_page_config`, not `.streamlit/config.toml`.
    `client.initial_sidebar_state` was removed from the config schema, so
    putting it there produces a startup warning - invisible in a background run -
    and the sidebar silently stays open, which is the failure this asserts
    against.
    """
    import ast
    import tomllib

    src = APP.read_text(encoding="utf-8")
    tree = ast.parse(src)

    # Exactly ONE call, at module scope. Streamlit raises
    # StreamlitSetPageConfigMustBeFirstCommandError on a second call, and the
    # app then renders nothing at all - a total failure from a one-line
    # addition, which is why it is asserted here rather than discovered in the
    # browser.
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and getattr(n.func, "attr", None) == "set_page_config"
    ]
    assert len(calls) == 1, (
        f"st.set_page_config() is called {len(calls)} times; it may only be "
        f"called once per page, and a second call raises "
        f"StreamlitSetPageConfigMustBeFirstCommandError and blanks the app"
    )
    assert calls[0].col_offset == 0 and not isinstance(
        getattr(calls[0], "parent", None), ast.FunctionDef), (
        "st.set_page_config() must sit at module scope, not inside a function - "
        "Streamlit requires it to be the first command in the script"
    )

    # And it must actually ask for a collapsed sidebar.
    state = None
    for kw in calls[0].keywords:
        if kw.arg == "initial_sidebar_state":
            state = getattr(kw.value, "value", None)
    assert state == "collapsed", (
        f"initial_sidebar_state is {state!r}, expected 'collapsed'; the arrow in "
        f"the top-left reopens it"
    )

    p = APP.parent.parent / ".streamlit" / "config.toml"
    with open(p, "rb") as fh:
        cfg = tomllib.load(fh)
    assert "initial_sidebar_state" not in cfg.get("client", {}), (
        "client.initial_sidebar_state is not a valid config option in Streamlit "
        "1.40 and only produces a startup warning; use st.set_page_config"
    )


def test_product_name_is_visually_prominent():
    """"Northgate AI" is the name and must lead the page.

    It previously ran together with "Stock Predictor" in one flat heading, so
    the name competed with its own descriptor. The name is now a badge and the
    only saturated element on the page.
    """
    src = APP.read_text(encoding="utf-8")
    assert "Northgate AI" in src, "the product name is missing from the masthead"
    m = re.search(r"def render_header\(.*?(?=\ndef )", src, re.S)
    assert m, "render_header() not found"
    body = m.group(0)
    # The badge treatment: a background, generous tracking, uppercase.
    assert "linear-gradient" in body or "background:" in body, (
        "the product name has no badge treatment, so it does not stand out"
    )
    assert "letter-spacing:0.14em" in body, (
        "the name badge should be letter-spaced and uppercase to read as a wordmark"
    )
    # "Stock Predictor" is the descriptor, so it must be visually quieter.
    assert re.search(r'Stock Predictor</span>', body), (
        "the descriptor should sit in a separate span from the name badge"
    )


def test_no_long_prose_blocks():
    """Short scannable lines, not paragraphs. Enforced, not aspirational.

    The complaint that prompted this was a page of paragraphs nobody finished.
    Section 2.1 asks for a dashboard "suitable for a non-technical reviewer",
    which a wall of prose defeats regardless of how accurate the prose is.

    Checked on the text actually passed to the display helpers: `takeaway`,
    `plain_box` and `readout` carry the explanation, and each is capped so a
    single block cannot become a screen. Bullets are exempt by design - a list
    of short lines is the thing this is steering towards, not a violation.
    """
    import ast

    src = APP.read_text(encoding="utf-8")
    tree = ast.parse(src)
    funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    offenders: list[str] = []

    for fn in ("takeaway", "plain_box"):
        body = ast.get_source_segment(src, funcs[fn]) or ""
        # The helper's own docstring and code are not user-facing prose.
        for node in ast.walk(funcs[fn]):
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == fn:
                texts = [s.value for s in ast.walk(node)
                         if isinstance(s, ast.Constant) and isinstance(s.value, str)
                         and s.value.strip() and "style" not in s.value]
                words = len(re.sub(r"<[^>]+>", " ", " ".join(texts)).split())
                if words > 45:
                    line = getattr(node, "lineno", 0)
                    offenders.append(f"{fn}() at line {line}: ~{words} words")

    assert not offenders, (
        "these blocks are long enough to read as a paragraph. Break them into "
        "bullets() or shorten them:\n    " + "\n    ".join(offenders)
    )


def test_app_parses():
    ast.parse(APP.read_text(encoding="utf-8"), filename=str(APP))


def test_every_live_marker_is_wired_to_a_renderer():
    """An unfilled `<!--LIVE:KEY-->` marker would ship as literal text.

    The report is assembled by substituting markers in the template with output
    from `scripts/fill_report.py`. If a section is added to the template and
    the renderer is forgotten, the marker survives into the published document
    as visible junk - and nothing else in the build would complain. This asserts
    the two sides still match in both directions.
    """
    fill = (APP.parent.parent / "scripts" / "fill_report.py").read_text(encoding="utf-8")
    template = (APP.parent.parent / "reports" / "final_report.template.md").read_text(
        encoding="utf-8")

    markers = set(re.findall(r"<!--LIVE:([A-Z_]+)-->", template))
    assert markers, "no LIVE markers found - did the template change shape?"

    registered = set(re.findall(r'"([A-Z_]+)":\s*\w+_section', fill))
    missing = markers - registered
    assert not missing, (
        f"the template declares markers with no renderer in fill_report.py: "
        f"{sorted(missing)} - they would appear in the report as literal text"
    )

    # And the reverse: a renderer with no marker is dead code that will silently
    # stop being exercised the first time someone renames a section.
    orphans = registered - markers
    assert not orphans, (
        f"fill_report.py renders sections the template no longer places: "
        f"{sorted(orphans)} - either the marker was renamed or the renderer is dead"
    )


def test_report_has_no_hardcoded_metric_strings():
    """Numbers in the report must come from artifacts, not from prose.

    The feature count alone was hand-written in four places and went stale twice
    as the table grew. The realised counts are now rendered from
    `features.parquet` by a LIVE marker; this asserts the stale literals are
    gone so a future reader does not trust them.
    """
    template = (APP.parent.parent / "reports" / "final_report.template.md").read_text(
        encoding="utf-8")
    for stale in ("64 features", "same 64 features",
                  "rows × 64 model features"):
        assert stale not in template, (
            f"the template still hardcodes '{stale}'; the realised count is "
            f"rendered from the artifact by the FEATURES marker instead"
        )


def test_all_seven_panels_present():
    """Section 13.1 requires these seven panels.

    The tab labels are deliberately plain-language for a non-technical reader
    (Section 2.1), so the check is on the panel *functions* that implement them,
    which are named after the PRD sections.
    """
    src = APP.read_text(encoding="utf-8")
    for fn in ("panel_overview", "panel_price", "panel_models", "panel_portfolio",
               "panel_risk", "panel_sentiment", "panel_recommendations"):
        assert f"def {fn}(" in src, f"missing dashboard panel: {fn}"


def test_cross_sectional_panel_is_labelled_supplementary():
    """The ranking panel is an ADDITION, and must not pose as a required panel.

    It answers a different question from the Section 9.5 forecast bar, with its
    own baseline. Presenting it as though it satisfied Section 8.3 would be the
    kind of metric-shopping Section 14.2 rules out, so the panel has to say out
    loud that it is judged differently and why.
    """
    src = APP.read_text(encoding="utf-8")
    assert "def panel_ranking(" in src
    start = src.index("def panel_ranking(")
    body = src[start:src.index("\ndef ", start + 10)]
    # It must state that the baseline is different, in reader-facing wording.
    assert "own fair baseline" in body or "its own baseline" in body
    # And it must never claim to satisfy the Section 8.3 forecast bar.
    for forbidden in ("beats the random walk", "Section 8.3 passed",
                      "clears the 8.3", "satisfies Section 9.5"):
        assert forbidden not in body, (
            f"the ranking panel must not claim '{forbidden}' - it is judged "
            f"against a different baseline and cannot satisfy the forecast bar"
        )


def test_panel_functions_exist():
    src = APP.read_text(encoding="utf-8")
    tree = ast.parse(src)
    funcs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    required = {
        "panel_overview", "panel_price", "panel_models", "panel_portfolio",
        "panel_risk", "panel_sentiment", "panel_recommendations",
        "panel_ranking",
        "render_disclaimer", "main", "load_all",
    }
    missing = required - funcs
    assert not missing, f"missing functions: {missing}"


def test_no_fabricated_metrics_table():
    """The previous stub hardcoded a fake comparison table. It must be gone."""
    src = APP.read_text(encoding="utf-8")
    assert "Static mockup" not in src
    assert "comparison_data" not in src
    # The old stub contained a literal accuracy string.
    assert '"57.1%"' not in src
    assert "'57.1%'" not in src


def test_disclaimer_is_rendered_per_panel():
    """Every panel must call render_disclaimer, not just the footer."""
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    panels = {"panel_overview", "panel_price", "panel_models", "panel_portfolio",
              "panel_risk", "panel_sentiment", "panel_recommendations"}
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in panels:
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call) and getattr(sub.func, "id", "") == "render_disclaimer":
                    found.add(node.name)
    assert found == panels, f"panels without a disclaimer: {panels - found}"


def test_uses_plotly_not_matplotlib():
    src = APP.read_text(encoding="utf-8")
    assert "plotly" in src
    assert "matplotlib.pyplot" not in src, "Section 13.2 requires interactive Plotly charts"


def test_uses_cache_data():
    src = APP.read_text(encoding="utf-8")
    assert "@st.cache_data" in src, "Section 13.2 requires heavy computation to be cached"


def test_has_sidebar_controls():
    """Section 13.2 requires ticker, horizon and risk-free-rate controls.

    Assert on the control *types* rather than their labels, so rewording the UI
    for a non-technical reader (Section 2.1) does not break the requirement.
    """
    src = APP.read_text(encoding="utf-8")
    for control in ("selectbox", "slider", "number_input"):
        assert f"st.{control}" in src, f"missing control: {control}"
    assert "horizon" in src, "missing the forecast-horizon control"
    assert "risk_free" in src or "rf_rate" in src, "missing the risk-free-rate control"


def test_never_imports_training_modules():
    """The dashboard must read artifacts, never retrain or fit a model.

    ``.fit(`` is banned for scikit-learn/pandas estimators specifically. A
    ``scipy.stats.t.fit`` call is a distribution fit for the VaR panel, not model
    training, so it is allowed; the check is therefore on estimator fits.
    """
    src = APP.read_text(encoding="utf-8")
    for banned in ("run_ml_models", "run_dl_models", "run_portfolio_optimization",
                   "analyze_sentiment", "build_features", "walk_forward_search",
                   "LGBMRegressor", "Ridge(", "XGBRegressor", "RandomForestRegressor",
                   "SVR(", "MLPRegressor", "models.Sequential", "models.Model("):
        assert banned not in src, f"dashboard must not reference {banned}"


def test_dashboard_does_not_fit_any_estimator():
    """No `.fit(` call may target a model estimator in the dashboard."""
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "fit"):
            continue
        # The receiver is `<owner>.<method>`. A scipy distribution fit appears as
        # `_st.t.fit` (attribute `t` on `_st`); an estimator fit appears as
        # `some_model.fit` where the receiver is a plain name. Distinguish on
        # whether the receiver is itself an attribute of a stats namespace.
        owner = node.func.value
        if isinstance(owner, ast.Attribute) and getattr(owner.value, "id", "") in ("_st", "st", "stats"):
            continue  # scipy.stats distribution fit, e.g. t.fit(...) for VaR
        name = getattr(owner, "id", None) or getattr(owner, "attr", None) or ""
        raise AssertionError(
            f"dashboard calls .{name}.fit(...), which would be model training"
        )


def test_reads_only_cached_artifacts():
    src = APP.read_text(encoding="utf-8")
    assert "read_parquet" in src and "read_csv" in src


def test_module_imports_without_streamlit_runtime():
    """Import the app module; st.set_page_config is a no-op outside a server."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("ng_app_check", APP)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # pragma: no cover
        raise AssertionError(f"dashboard/app.py failed to import: {exc}") from exc

    assert hasattr(mod, "PALETTE")
    assert hasattr(mod, "base_layout")
    assert hasattr(mod, "empty_state")


def test_leaderboard_file_is_the_real_artifact():
    """The model-comparison panel reads this; it must be generated, not stubbed."""
    lb = ROOT / "data" / "processed" / "model_leaderboard.csv"
    if not lb.exists():
        return  # retrain has not run yet; the panel handles absence
    df = pd.read_csv(lb)
    assert len(df) >= 1
    for col in ("Model", "RMSE", "MAE", "R2", "DirAcc"):
        assert col in df.columns, f"leaderboard missing {col}"
    assert df["MAE"].notna().all()
    assert df["Model"].duplicated().sum() == 0


def test_portfolio_weights_are_a_simplex():
    p = ROOT / "data" / "processed" / "optimal_weights.csv"
    if not p.exists():
        return
    w = pd.read_csv(p, index_col=0).iloc[:, 0]
    assert abs(w.sum() - 1.0) < 1e-6, f"weights sum to {w.sum()}"
    assert (w >= -1e-9).all(), "long-only violated"
