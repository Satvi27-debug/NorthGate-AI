"""Inject live artifact numbers into README.md.

The README carried hand-copied metrics that went stale: it quoted an optimised
Sharpe of 1.357 while the artifact said 1.039, because the figures were pasted in
once and never refreshed. A headline number that silently disagrees with the
artifact it describes is worse than no number, so the summary block is now
generated from the same files the report reads.

Markers are `<!--LIVE:NAME-->`, matching the convention `fill_report.py` already
uses, and the README is rendered from `reports/readme.template.md` so re-running
is idempotent rather than cumulative.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import PROCESSED_DIR, REPORTS_DIR  # noqa: E402

TEMPLATE = REPORTS_DIR / "readme.template.md"
README = ROOT / "README.md"


def _j(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _get(d: dict, *names, default=None):
    """First present key, searching nested one level deep."""
    for src in (d, d.get("backtest", {}) if isinstance(d.get("backtest"), dict) else {}):
        for n in names:
            if isinstance(src, dict) and n in src:
                return src[n]
    return default


def _leaderboard():
    import pandas as pd

    lb = PROCESSED_DIR / "model_leaderboard.csv"
    if not lb.exists():
        return None, None, None
    df = pd.read_csv(lb)
    col = {c.lower(): c for c in df.columns}
    mcol = col.get("model", "Model")
    maecol = col.get("mae", "MAE")
    dcol = next((c for k, c in col.items() if "dir" in k), None)

    def find(pat):
        hit = df[df[mcol].astype(str).str.lower().str.contains(pat, na=False)]
        return hit.iloc[0] if len(hit) else None

    base = find("random walk|naive")
    # Only real candidates go in the "best model" column. Leaving the baseline
    # in makes the table read as though the model merely matched it, which is
    # both confusing and wrong - the baseline is the thing being beaten.
    cands = df[~df[mcol].astype(str).str.lower().str.contains(
        "random walk|naive|baseline", na=False)]
    if not len(cands):
        return base, None, dcol
    return base, cands, dcol


def headline() -> str:
    import pandas as pd  # noqa: F401  (import kept local; leaderboard is optional)

    base, cands, dcol = _leaderboard()
    if cands is None:
        return "_Model leaderboard not built. Run `python retrain_models.py`._"

    col = {c.lower(): c for c in cands.columns}
    maecol = col.get("mae", "MAE")
    out = ["", "**Forecasting (§8.3)** — the bar needs *both* halves.", "",
           "| Bar | Random walk | Best model | Verdict |", "|---|---|---|---|"]

    def f(r, c, pct=False):
        if r is None or c not in cands.columns:
            return "n/a"
        v = float(r[c])
        return f"{v * 100:.2f}%" if pct else f"{v:.6f}"

    bm = cands.loc[cands[maecol].idxmin()]
    met = float(bm[maecol]) < float(base[maecol]) if base is not None else False
    out.append(f"| MAE (lower is better) | {f(base, maecol)} | "
               f"{f(bm, maecol)} | {'**MET**' if met else '**NOT MET**'} |")
    if dcol:
        bd = cands.loc[cands[dcol].idxmax()]
        metd = float(bd[dcol]) > float(base[dcol]) if base is not None else False
        out.append(f"| Directional accuracy | {f(base, dcol, pct=True)} | "
                   f"{f(bd, dcol, pct=True)} | {'**MET**' if metd else '**NOT MET**'} |")
    out.append("")
    # Only forecasting and the portfolio belong in this summary. The
    # recommendation hit-rate has its own section further down, and printing it
    # here as well put the same two numbers on one screen twice.
    out.append(f"**Portfolio (§11.4)** — {portfolio_line()}")
    return "\n".join(out)


def portfolio_line() -> str:
    pm = _j(PROCESSED_DIR / "portfolio_metrics.json")
    if not pm:
        return "portfolio backtest not built."
    # The real shape is backtest.optimised.sharpe / backtest.equal_weight.sharpe,
    # with the interval in a sibling `sharpe_bootstrap_ci` dict. Guessing flat
    # key names is what made this render "figures unavailable" while the
    # artifact was complete.
    bt = pm.get("backtest", {}) if isinstance(pm.get("backtest"), dict) else {}
    opt = (bt.get("optimised") or {}).get("sharpe") if isinstance(
        bt.get("optimised"), dict) else None
    eq = (bt.get("equal_weight") or {}).get("sharpe") if isinstance(
        bt.get("equal_weight"), dict) else None
    ci = pm.get("sharpe_bootstrap_ci") if isinstance(
        pm.get("sharpe_bootstrap_ci"), dict) else {}
    lo, hi = ci.get("ci_low"), ci.get("ci_high")
    if opt is None or eq is None:
        return "Sharpe figures unavailable in `portfolio_metrics.json`."
    txt = f"optimised Sharpe **{float(opt):.3f}** vs equal weight **{float(eq):.3f}**"
    if lo is not None and hi is not None:
        txt += f", 95% CI [{float(lo):.3f}, {float(hi):.3f}]"
        if float(lo) < 0 < float(hi):
            txt += " — spans zero, so the gap is not statistically meaningful"
    txt += ". **NOT MET** — equal weight wins."
    return txt


def hit_rate_line() -> str:
    rb = _j(PROCESSED_DIR / "recommendation_backtest.json")
    if not rb:
        return "backtest not run."
    bh = rb.get("buy_and_hold_hit_rate")
    g = rb.get("arms", {}).get("gated", {}).get("hit_rate")
    if bh is None or g is None:
        return "the engine issued no trades, so hit-rate equals buy-and-hold by construction."
    w = rb.get("window", {})
    gap = (g - bh) * 100
    rel = "beats" if gap > 0 else f"trails by {abs(gap):.2f} points"
    return (f"hit-rate **{g * 100:.2f}%** vs buy-and-hold **{bh * 100:.2f}%** "
            f"over {w.get('observations', 0):,} held-out ticker-sessions — the "
            f"engine {rel}. Documented as required; the result is unfavourable.")


def acceptance_summary() -> str:
    """Counts only. A 28-row table in a README is noise; the report owns it."""
    src = REPORTS_DIR / "final_report.md"
    if not src.exists():
        return "_Acceptance table not rendered yet; run `python scripts/fill_report.py`._"
    text = src.read_text(encoding="utf-8")
    if "| Criterion |" not in text:
        return "_Acceptance table not found._"
    block = text[text.rindex("| Criterion |"):]
    verdicts = []
    for ln in block.splitlines():
        if not ln.startswith("|"):
            if verdicts:
                break
            continue
        cells = [c.strip().replace("**", "") for c in re.split(r"(?<!\\)\|", ln)]
        if len(cells) < 5:
            continue
        v = cells[4].split("(")[0].split("—")[0].strip()
        if v in ("MET", "NOT MET", "NOT RUN"):
            verdicts.append(v)
    if not verdicts:
        return "_Acceptance table is malformed._"
    met = verdicts.count("MET")
    nmet = verdicts.count("NOT MET")
    nrun = verdicts.count("NOT RUN")
    fails = []
    cur = None
    for ln in block.splitlines():
        if not ln.startswith("|"):
            if fails:
                break
            continue
        cells = [c.strip().replace("**", "") for c in re.split(r"(?<!\\)\|", ln)]
        if len(cells) < 5:
            continue
        if cells[1] not in ("", "---"):
            cur = cells[1]
        v = cells[4].split("(")[0].split("—")[0].strip()
        if v in ("NOT MET", "NOT RUN") and cur:
            fails.append(f"`{cur}` — {v}")
    body = (f"**{met} of {len(verdicts)}** PRD criteria are met. "
            f"{nmet} not met, {nrun} not run.")
    if fails:
        body += "\n\n" + "\n".join(f"- {f}" for f in fails)
    body += ("\n\nFull table with measured values and the reason for each "
             "verdict: `reports/final_report.md`, Appendix A.")
    return body


def model_table() -> str:
    """The full eight-model table, straight from the leaderboard."""
    import pandas as pd

    lb = PROCESSED_DIR / "model_leaderboard.csv"
    if not lb.exists():
        return "_Model leaderboard not built._"
    df = pd.read_csv(lb, index_col=0)
    rows = ["| Model | Family | MAE | RMSE | R² | Dir. Acc. | N |",
            "|---|---|---:|---:|---:|---:|---:|"]
    for name, r in df.sort_values("MAE").iterrows():
        fam = str(r.get("Family", ""))
        rows.append(f"| {name} | {fam} | {r['MAE']:.6f} | {r['RMSE']:.6f} | "
                    f"{r['R2']:+.4f} | {r['DirAcc'] * 100:.2f}% | {int(r['N']):,} |")
    return "\n".join(rows)


def reproducibility() -> str:
    ce = _j(REPORTS_DIR / "clean_env_check.json")
    if not ce or not ce.get("steps"):
        return "⚠️ Not yet verified"
    done = [s for s in ce["steps"] if s["status"] in ("PASS", "FAIL", "NOT RUN")]
    n_pass = sum(1 for s in ce["steps"] if s["status"] == "PASS")
    if not ce.get("complete"):
        return f"🔄 In progress — {n_pass}/{len(done)} steps passed"
    if ce.get("all_passed"):
        return (f"✅ **{n_pass}/{len(ce['steps'])} steps** in a throwaway venv "
                f"({ce.get('elapsed_seconds', 0) / 60:.0f} min)")
    bad = [s["name"] for s in ce["steps"] if s["status"] == "FAIL"]
    return f"⚠️ **{n_pass}/{len(ce['steps'])} steps** — failed: {', '.join(bad)}"


SECTIONS = {
    "MODELTABLE": model_table,
    "HEADLINE": headline,
    "HITRATE": lambda: (
        f"Hit-rate **{_hit()[1] * 100:.2f}%** against buy-and-hold "
        f"**{_hit()[0] * 100:.2f}%** over "
        f"{_j(PROCESSED_DIR / 'recommendation_backtest.json').get('window', {}).get('observations', 0):,} "
        f"held-out ticker-sessions — the engine "
        f"{'beats' if _hit()[1] > _hit()[0] else 'trails'} doing nothing. "
        f"Documented as the PRD requires; the result is unfavourable."
    ) if all(x is not None for x in _hit()) else hit_rate_line(),
    "PORTFOLIO": lambda: f"### 📊 Current Sharpe\n\n{portfolio_line()}",
    "REPRO": reproducibility,
    "ACCEPTANCE": acceptance_summary,
}


def _hit():
    rb = _j(PROCESSED_DIR / "recommendation_backtest.json")
    if not rb:
        return None, None
    return (rb.get("buy_and_hold_hit_rate"),
            rb.get("arms", {}).get("gated", {}).get("hit_rate"))


def main() -> int:
    if not TEMPLATE.exists():
        print(f"template not found at {TEMPLATE}", file=sys.stderr)
        return 1
    text = TEMPLATE.read_text(encoding="utf-8")
    for name, fn in SECTIONS.items():
        marker = f"<!--LIVE:{name}-->"
        if marker not in text:
            print(f"  {name}: MARKER MISSING from template", file=sys.stderr)
            return 1
        body = fn()
        text = text.replace(marker, body)
        print(f"  {name}: injected ({len(body)} chars)")
    README.write_text(text, encoding="utf-8")
    print(f"wrote {README}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
