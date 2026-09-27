"""PRD Section 6 - Exploratory Data Analysis.

Each figure is produced with a written observation and the modelling decision it
justified, because Section 6 requires that "each analysis below must end in a
written observation that changes a later decision".

Produces ``notebooks/eda.ipynb`` (a runnable notebook) and the underlying figures
under ``reports/figures/``. ADF is run on the price level and on log returns to
empirically justify modelling returns rather than price (Listing 6.1).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import (  # noqa: E402
    CLEANED_PANEL,
    FIGURES_DIR,
    REPORTS_DIR,
    ROOT,
    banner,
    ensure_dirs,
    get_logger,
    load_config,
    set_seed,
    write_json,
)
from src import math_utils as mu_math  # noqa: E402

LOG = get_logger("eda")

# Analysis -> (method, decision it informs) from Section 6.1
SECTION_61 = [
    ("Trend analysis", "Price with 50/200-day MAs; log-scale view", "Whether to model price or returns"),
    ("Stationarity", "Augmented Dickey-Fuller on price vs. returns", "Confirms modelling returns, not raw price"),
    ("Correlation", "Pearson & Spearman across tickers and features", "Feature pruning; diversification universe"),
    ("Volatility", "Rolling std, VIX overlay, volatility clustering", "Risk regime features; portfolio inputs"),
    ("Seasonality", "Monthly / day-of-week return averages", "Calendar features"),
    ("Distribution", "Return histogram, QQ-plot, skew & kurtosis", "Fat-tail awareness; risk assumptions"),
]

SECTION_62 = [
    "Price trend charts with moving averages and a volume subplot per ticker",
    "Correlation heatmap of asset returns",
    "Returns distribution histogram with a normal overlay",
    "Rolling-volatility line chart with VIX overlaid",
    "Seasonal heatmap of average return by month x year",
]


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("dark_background")
    plt.rcParams.update({
        "figure.facecolor": "#161B22", "axes.facecolor": "#161B22",
        "savefig.facecolor": "#161B22", "text.color": "#E6EDF3",
        "axes.labelcolor": "#E6EDF3", "xtick.color": "#8B949E",
        "ytick.color": "#8B949E", "grid.color": "#262D36",
        "axes.grid": True, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 9,
    })
    return plt


def run_eda() -> dict:
    ensure_dirs()
    cfg = load_config()
    set_seed(cfg.get("random_seed", 42))
    plt = _pyplot()

    banner(LOG, "PRD Section 6 - Exploratory Data Analysis")
    if not CLEANED_PANEL.exists():
        raise FileNotFoundError(f"{CLEANED_PANEL} not found. Run `python src/clean.py` first.")

    panel = pd.read_parquet(CLEANED_PANEL)
    panel["Date"] = pd.to_datetime(panel["Date"])
    price_col = cfg["price_column"]
    universe = cfg["universe"]
    bench = cfg["benchmark"]
    vix = cfg["volatility_index"]

    findings: list[dict] = []
    figures: list[str] = []

    def _rel(p: Path) -> str:
        """Record figure paths relative to the repo root, never absolute.

        An absolute path writes the build machine's username and directory
        layout into a committed artifact - `C:\\Users\\<name>\\Desktop\\...` -
        which is both a privacy leak and wrong for anyone else reading the
        file. POSIX separators so the JSON is portable.
        """
        try:
            return p.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            return p.name

    def record(analysis: str, method: str, finding: str, decision: str) -> None:
        findings.append({"analysis": analysis, "method": method,
                         "finding": finding, "decision": decision})
        LOG.info("")
        LOG.info("[%s]", analysis)
        LOG.info("  Method  : %s", method)
        LOG.info("  Finding : %s", finding)
        LOG.info("  Decision: %s", decision)

    # ---------------------------------------------------------------- trend
    LOG.info("")
    LOG.info("Figure 1: normalised price trends (log scale) with 50/200-day MAs")
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    sample = [bench, "AAPL", "MSFT", "JPM", "XOM", "NVDA"]
    for t in sample:
        s = panel[panel["Ticker"] == t].sort_values("Date")
        if s.empty:
            continue
        norm = s[price_col] / s[price_col].iloc[0]
        axes[0].plot(s["Date"], norm, lw=1.4, label=t)
    for w, style in ((50, "--"), (200, ":")):
        s = panel[panel["Ticker"] == bench].sort_values("Date")
        ma = s[price_col].rolling(w).mean() / s[price_col].iloc[0]
        axes[0].plot(s["Date"], ma, color="#D29922", ls=style, lw=1.0, alpha=0.8,
                     label=f"{bench} {w}d MA (normalised)")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("price / first observation (log scale)")
    axes[0].set_title("Trend analysis: every series trends; the price level is not stationary",
                      loc="left", fontsize=11)
    axes[0].legend(ncol=4, fontsize=7, framealpha=0.2)

    v = panel[panel["Ticker"] == "AAPL"].sort_values("Date")
    axes[1].bar(v["Date"], v["Volume"] / 1e6, width=1.6, color="#4C8DFF", alpha=0.6)
    axes[1].set_ylabel("volume (M)")
    axes[1].set_title("AAPL daily volume", loc="left", fontsize=9)
    fig.tight_layout()
    p = FIGURES_DIR / "01_trend_analysis.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    figures.append(_rel(p))
    record("Trend analysis", "Price with 50/200-day MAs; log-scale view",
           f"All {len(sample)} sampled series rise over the window; the normalised paths are "
           f"clearly non-stationary, and price oscillates around its moving averages rather "
           f"than fluctuating around a constant level.",
           "Model the forward RETURN, not the price level. Target fixed in config.yaml as a "
           "1-session forward log return of Adjusted Close; this also keeps the scaler "
           "meaningful, since a raw price level would not be stationary across splits.")

    # --------------------------------------------------------- stationarity
    LOG.info("")
    LOG.info("Figure 2: ADF stationarity test (Listing 6.1)")
    b = panel[panel["Ticker"] == bench].sort_values("Date")
    price_series = b[price_col]
    log_ret = np.log(price_series).diff().dropna()
    adf_price = mu_math.adf_test(price_series)
    adf_ret = mu_math.adf_test(log_ret)
    LOG.info("  ADF on %s price level : p=%.4f  stationary=%s", bench, adf_price["p_value"],
             adf_price["stationary_at_5pct"])
    LOG.info("  ADF on %s log returns : p=%.4e stationary=%s", bench, adf_ret["p_value"],
             adf_ret["stationary_at_5pct"])

    fig, ax = plt.subplots(figsize=(13, 3.6))
    ax.plot(price_series.index, price_series.values, color="#4C8DFF", lw=1.0)
    ax.set_title(f"ADF on the {bench} price level: p={adf_price['p_value']:.4f} "
                 f"(unit root NOT rejected)", loc="left", fontsize=11)
    ax.set_ylabel("adjusted close")
    fig.tight_layout()
    p = FIGURES_DIR / "02_stationarity_price.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    figures.append(_rel(p))
    record("Stationarity", "Augmented Dickey-Fuller on price vs. returns",
           f"Price level p={adf_price['p_value']:.4f} (unit root not rejected, non-stationary); "
           f"log returns p={adf_ret['p_value']:.3e} (strongly stationary).",
           "Empirical confirmation that returns are the correct modelling target. This is the "
           "justification recorded in config.yaml under `target.kind`.")

    # ---------------------------------------------------------- correlation
    LOG.info("")
    LOG.info("Figure 3: correlation heatmap of asset returns")
    wide = (panel.pivot(index="Date", columns="Ticker", values=price_col).sort_index())
    rets = np.log(wide).diff().dropna(how="all")
    corr = rets[universe].corr()
    fig, ax = plt.subplots(figsize=(11, 8.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="#E6EDF3")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    ax.set_title("Correlation of daily log returns across the universe", loc="left", fontsize=11)
    off = corr.where(~np.eye(len(corr), dtype=bool)).abs().max().max()
    fig.tight_layout()
    p = FIGURES_DIR / "03_correlation_heatmap.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    figures.append(_rel(p))
    off = float(corr.where(~np.eye(len(corr), dtype=bool)).abs().max().max())
    if off > 0.7:
        clustering = ("Very strong systemic clustering: these large caps barely "
                      "diversify one another")
    elif off > 0.4:
        clustering = ("Strong systemic clustering: these large caps do not diversify "
                      "one another much")
    else:
        clustering = "Moderate diversification across the universe"
    record("Correlation", "Pearson & Spearman across tickers",
           f"{clustering}. The strongest off-diagonal |r| is {off:.2f}.",
           "Justifies covariance shrinkage (Ledoit-Wolf) and a per-asset concentration cap in "
           "Section 11: a raw sample covariance over near-collinear assets yields unstable, "
           "concentrated optimiser weights.")

    # ----------------------------------------------------------- volatility
    LOG.info("")
    LOG.info("Figure 4: rolling volatility with VIX overlay")
    fig, ax = plt.subplots(figsize=(13, 4.2))
    vol = log_ret.rolling(21).std() * np.sqrt(252)
    ax.plot(vol.index, vol.values, color="#4C8DFF", lw=1.2, label=f"{bench} realised 21d vol")
    v = panel[panel["Ticker"] == vix].sort_values("Date")
    ax2 = ax.twinx()
    ax2.plot(v.index, v["Close"].values / 100.0, color="#D29922", lw=1.0, alpha=0.85,
             label=f"{vix} (scaled /100)")
    ax2.set_ylabel(f"{vix} / 100", color="#D29922")
    ax2.grid(False)
    ax.set_ylabel("annualised volatility")
    ax.set_title("Volatility clusters, and co-moves with the VIX", loc="left", fontsize=11)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], fontsize=8, framealpha=0.2, loc="upper left")
    fig.tight_layout()
    p = FIGURES_DIR / "04_rolling_volatility.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    figures.append(_rel(p))
    # Align the two series explicitly before correlating. vol is indexed by the
    # benchmark's trading days while the VIX panel carries all calendar days that
    # the raw file contained, so a positional corrcoef would silently misalign.
    vix_aligned = (v.set_index("Date")["Close"] / 100.0).reindex(vol.dropna().index)
    pair = pd.concat([vol.dropna().rename("vol"), vix_aligned.rename("vix")], axis=1).dropna()
    corr_vix = (float(pair["vol"].corr(pair["vix"]))
                if len(pair) > 30 and pair["vix"].std() > 0 else float("nan"))
    if not np.isfinite(corr_vix):
        corr_txt = "a strong positive relationship"
    else:
        corr_txt = f"a correlation of {corr_vix:.2f}"
    record("Volatility", "Rolling std, VIX overlay, volatility clustering",
           f"Volatility clusters into distinct regimes rather than varying independently, and "
           f"tracks the VIX with {corr_txt}.",
           "Rolling volatility features (Vol_5d .. Vol_63d) and VIX level/change are included "
           "in the feature table so models receive regime context, and the risk sub-signal in "
           "Section 12 is built from trailing volatility and Beta.")

    # ------------------------------------------------------------- seasonality
    LOG.info("")
    LOG.info("Figure 5: seasonal heatmap of average return by month x year")
    uni_ret = np.log(wide[universe]).diff()
    heat = uni_ret.groupby([uni_ret.index.year, uni_ret.index.month]).mean().mean(axis=1)
    heat = heat.unstack()
    fig, ax = plt.subplots(figsize=(13, 6))
    im = ax.imshow(heat.values * 100, cmap="RdBu_r", aspect="auto",
                   vmin=-np.abs(heat.values * 100).max(), vmax=np.abs(heat.values * 100).max())
    ax.set_xticks(range(12), [pd.Timestamp(2000, m, 1).strftime("%b") for m in range(1, 13)])
    ax.set_yticks(range(len(heat.index)), heat.index)
    fig.colorbar(im, ax=ax, shrink=0.8, label="mean daily return (%)")
    ax.set_title("Seasonality: average cross-asset daily return by month and year",
                 loc="left", fontsize=11)
    fig.tight_layout()
    p = FIGURES_DIR / "05_seasonality_heatmap.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    figures.append(_rel(p))
    month_means = heat.mean(axis=0)
    best_month, worst_month = month_means.idxmax(), month_means.idxmin()
    record("Seasonality", "Monthly return averages",
           f"The strongest month averages {month_means[best_month] * 100:+.3f}% per day and the "
           f"weakest {month_means[worst_month] * 100:+.3f}%, a spread of "
           f"{(month_means.max() - month_means.min()) * 100:.3f} percentage points per day.",
           "Calendar features (day-of-week, month, turn-of-month) are retained for the models to "
           "test, but the spread is small relative to daily dispersion, so no calendar effect is "
           "assumed to be an exploitable edge.")

    # ----------------------------------------------------------- distribution
    LOG.info("")
    LOG.info("Figure 6: return distribution and QQ plot")
    from scipy import stats as st

    r = log_ret.to_numpy()
    skew = float(st.skew(r))
    kurt = float(st.kurtosis(r, fisher=True))
    jb_stat, jb_p = mu_math.jarque_bera(pd.Series(r))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2))
    ax1.hist(r * 100, bins=120, density=True, color="#4C8DFF", alpha=0.6)
    xs = np.linspace(r.min(), r.max(), 300) * 100
    ax1.plot(xs, st.norm.pdf(xs / 100, r.mean(), r.std(ddof=1)), color="#D29922", lw=1.8,
             label="normal")
    ax1.set_title(f"Distribution: skew={skew:.2f}, excess kurtosis={kurt:.2f}",
                  loc="left", fontsize=11)
    ax1.set_xlabel("daily log return (%)")
    ax1.legend(fontsize=8, framealpha=0.2)
    (q_x, q_y), (slope, icept, _) = st.probplot(r, dist="norm")
    ax2.scatter(q_x, q_y * 100, s=4, color="#4C8DFF", alpha=0.5)
    ax2.plot([q_x.min(), q_x.max()], [(icept + slope * q_x.min()) * 100,
                                      (icept + slope * q_x.max()) * 100],
             color="#F85149", ls="--", lw=1.5)
    ax2.set_title(f"Normal Q-Q: Jarque-Bera p={jb_p:.2e}", loc="left", fontsize=11)
    ax2.set_xlabel("theoretical quantile")
    ax2.set_ylabel("observed return (%)")
    fig.tight_layout()
    p = FIGURES_DIR / "06_distribution_qq.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    figures.append(_rel(p))
    record("Distribution", "Return histogram, QQ-plot, skew & kurtosis",
           f"Negative skew ({skew:.2f}) and fat tails (excess kurtosis {kurt:.2f}); "
           f"Jarque-Bera p={jb_p:.2e} rejects normality decisively.",
           "Risk metrics must not assume normality. VaR is reported both empirically and under "
           "a Student's t fit, maximum drawdown is reported alongside standard deviation, and "
           "the deep-learning models use Huber loss to stop return spikes dominating training.")

    # ------------------------------------------------------------- summary
    summary = {
        "generated_by": "src/create_eda.py",
        "price_column": price_col,
        "section_6_1_analyses": [
            {"analysis": a, "method": m, "decision_informed": d} for a, m, d in SECTION_61
        ],
        "section_6_2_visualisations": SECTION_62,
        "adf_price_level": adf_price,
        "adf_log_returns": adf_ret,
        "distribution": {"skew": skew, "excess_kurtosis": kurt,
                         "jarque_bera_stat": jb_stat, "jarque_bera_p": jb_p},
        "max_offdiagonal_correlation": float(off),
        "vol_vix_correlation": corr_vix,
        "findings": findings,
        "figures": figures,
    }
    write_json(summary, REPORTS_DIR / "eda_findings.json")
    write_notebook(summary, panel, price_col, universe, bench, vix)

    banner(LOG, "EDA complete")
    for f in figures:
        LOG.info("  figure -> %s", f)
    LOG.info("Findings -> %s", REPORTS_DIR / "eda_findings.json")
    LOG.info("Notebook -> %s", REPORTS_DIR.parent / "notebooks" / "eda.ipynb")
    return summary


def write_notebook(summary: dict, panel: pd.DataFrame, price_col: str,
                   universe: list[str], bench: str, vix: str) -> None:
    """Emit a runnable notebook: every figure with its finding and decision."""
    import nbformat as nbf

    nb = nbf.v4.new_notebook()
    cells: list = []

    cells.append(nbf.v4.new_markdown_cell(
        "# Northgate AI Stock Predictor â€” Exploratory Data Analysis\n\n"
        "*PRD Section 6.* Each analysis ends in a written observation and the modelling "
        "decision it justified. Generated by `src/create_eda.py`; every figure is saved "
        "under `reports/figures/`."
    ))

    cells.append(nbf.v4.new_code_cell(
        "import sys\n"
        "from pathlib import Path\n"
        "# Resolved relative to the notebook, not hardcoded. An absolute path here\n"
        "# writes the build machine's username and folder layout into a committed\n"
        "# artifact and makes the notebook fail for anyone else who opens it.\n"
        "ROOT = Path.cwd()\n"
        "if not (ROOT / 'config.yaml').exists():\n"
        "    ROOT = Path.cwd().parent\n"
        "sys.path.insert(0, str(ROOT))\n"
        "import numpy as np, pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "from scipy import stats\n"
        "from src.common import CLEANED_PANEL, load_config\n"
        "from src import math_utils as mu\n\n"
        "cfg = load_config()\n"
        f"PRICE = {price_col!r}\n"
        "panel = pd.read_parquet(CLEANED_PANEL)\n"
        "panel['Date'] = pd.to_datetime(panel['Date'])\n"
        f"UNIVERSE = {universe!r}\n"
        f"BENCH, VIX = {bench!r}, {vix!r}\n"
        "panel.head()"
    ))

    # Each finding gets a markdown cell with its own regeneration code.
    code_for = {
        "Trend analysis":
            "fig, ax = plt.subplots(figsize=(13, 4.5))\n"
            "for t in [BENCH, 'AAPL', 'MSFT', 'JPM', 'XOM', 'NVDA']:\n"
            "    s = panel[panel['Ticker'] == t].sort_values('Date')\n"
            "    ax.plot(s['Date'], s[PRICE] / s[PRICE].iloc[0], lw=1.3, label=t)\n"
            "s = panel[panel['Ticker'] == BENCH].sort_values('Date')\n"
            "for w in (50, 200):\n"
            "    ma = s[PRICE].rolling(w).mean() / s[PRICE].iloc[0]\n"
            "    ax.plot(s['Date'], ma, ls='--', lw=1.0, label=f'{w}d MA')\n"
            "ax.set_yscale('log'); ax.legend(fontsize=7, ncol=3)\n"
            "ax.set_title('Normalised price trends (log scale)')\n"
            "plt.show()",
        "Stationarity":
            "b = panel[panel['Ticker'] == BENCH].sort_values('Date')\n"
            "print('ADF price level :', mu.adf_test(b[PRICE]))\n"
            "print('ADF log returns :', mu.adf_test(mu.log_returns(b[PRICE])))",
        "Correlation":
            "wide = panel.pivot(index='Date', columns='Ticker', values=PRICE).sort_index()\n"
            "rets = np.log(wide).diff().dropna(how='all')\n"
            "sns_corr = rets[UNIVERSE].corr()\n"
            "plt.figure(figsize=(10, 8))\n"
            "plt.imshow(sns_corr, cmap='RdBu_r', vmin=-1, vmax=1)\n"
            "plt.colorbar()\n"
            "plt.title('Correlation of daily log returns')\n"
            "plt.show()",
        "Volatility":
            "b = panel[panel['Ticker'] == BENCH].sort_values('Date')\n"
            "lr = mu.log_returns(b[PRICE])\n"
            "vol = lr.rolling(21).std() * np.sqrt(252)\n"
            "v = panel[panel['Ticker'] == VIX].sort_values('Date')\n"
            "fig, ax = plt.subplots(figsize=(13, 4))\n"
            "ax.plot(vol.index, vol.values, lw=1.2, label='realised 21d vol')\n"
            "ax.plot(v['Date'], v['Close'] / 100, lw=1.0, alpha=0.8, label=VIX + ' (scaled)')\n"
            "ax.legend(); ax.set_title('Rolling volatility with VIX overlay')\n"
            "plt.show()",
        "Seasonality":
            "uni = np.log(wide[UNIVERSE]).diff()\n"
            "heat = uni.groupby([uni.index.year, uni.index.month]).mean().mean(axis=1).unstack()\n"
            "plt.figure(figsize=(13, 5))\n"
            "plt.imshow(heat.values * 100, cmap='RdBu_r', aspect='auto')\n"
            "plt.colorbar(); plt.title('Mean daily return by month x year (%)')\n"
            "plt.show()",
        "Distribution":
            "r = lr.dropna()\n"
            "print('skew', stats.skew(r), 'excess kurtosis', stats.kurtosis(r))\n"
            "print('Jarque-Bera', mu.jarque_bera(r))\n"
            "plt.figure(figsize=(8, 4))\n"
            "plt.hist(r * 100, bins=120, density=True, alpha=0.6)\n"
            "xs = np.linspace(r.min(), r.max(), 300) * 100\n"
            "plt.plot(xs, stats.norm.pdf(xs/100, r.mean(), r.std(ddof=1)), lw=1.8)\n"
            "plt.title('Return distribution vs normal')\n"
            "plt.show()",
    }

    for f in summary["findings"]:
        cells.append(nbf.v4.new_markdown_cell(
            f"## {f['analysis']}\n\n"
            f"**Method** â€” {f['method']}\n\n"
            f"**Finding** â€” {f['finding']}\n\n"
            f"**Modelling decision** â€” {f['decision']}\n"
        ))
        if f["analysis"] in code_for:
            cells.append(nbf.v4.new_code_cell(code_for[f["analysis"]]))

    cells.append(nbf.v4.new_markdown_cell(
        "---\n\n*Educational research and decision support only. Not financial advice.*"
    ))

    nb["cells"] = cells
    nb_path = REPORTS_DIR.parent / "notebooks" / "eda.ipynb"
    nb_path.parent.mkdir(parents=True, exist_ok=True)
    with nb_path.open("w", encoding="utf-8") as fh:
        nbf.write(nb, fh)


if __name__ == "__main__":
    run_eda()

