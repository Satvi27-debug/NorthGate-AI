"""PRD Section 4 - Data Cleaning Pipeline.

Raw multi-source data in, one tidy analysis panel out. The five stages run in a
fixed, auditable order (PRD Section 4.1):

    Stage 1  Structural alignment      reindex every series onto the exchange
                                        trading calendar; forward-fill lower
                                        frequency macro onto the daily grid
    Stage 2  Missing-value handling    forward-fill price gaps of at most
                                        ``clean.max_price_ffill`` sessions.
                                        NEVER back-filled.
    Stage 3  Duplicate removal         drop exact (date, ticker) duplicates,
                                        log disagreements
    Stage 4  Outlier detection         rolling z-score AND 1.5x IQR fence on
                                        daily log returns. FLAGGED, never
                                        auto-deleted
    Stage 5  Consistency validation    OHLC invariant, quarantine violations

Emits the audit report required by Section 4.2 to
``reports/data_quality_report.{json,md}``.

Design note on Stage 4
----------------------
The PRD is explicit that a large move is usually signal rather than error:
"a 20% jump on an earnings date is signal, not error, and must be kept", and
Listing 4.1 annotates the flag line with "do NOT auto-delete". Confirming which
flagged rows are genuine vendor errors requires a corporate-actions calendar
that this project does not have, so we do **not** winsorise. We flag, we log
every flagged (date, ticker), we count them in the quality report, and we
carry the flags forward as columns so the modelling stage can decide. This is
the conservative reading of Section 4.1 Stage 4.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import (  # noqa: E402
    CLEANED_PANEL,
    DQ_REPORT_JSON,
    DQ_REPORT_MD,
    PROCESSED_DIR,
    RAW_MACRO,
    RAW_PRICES,
    banner,
    ensure_dirs,
    get_logger,
    load_config,
    set_seed,
    write_json,
)

LOG = get_logger("clean")

PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Adjusted Close"]
OHLCV_COLUMNS = PRICE_COLUMNS + ["Volume"]


# ==========================================================================
# Stage 3 (run first in practice) - duplicate removal
# ==========================================================================
def remove_duplicates(prices: pd.DataFrame) -> tuple[pd.DataFrame, int, list[str]]:
    """Drop duplicate (Date, Ticker) rows, logging every disagreement.

    When duplicates disagree we keep the vendor-adjusted record, i.e. the row
    carrying a non-null ``Adjusted Close`` (PRD Section 4.1 Stage 3).
    """
    before = len(prices)
    key = ["Date", "Ticker"]

    dup_mask = prices.duplicated(subset=key, keep=False)
    n_dupes = int(dup_mask.sum())
    conflicts: list[str] = []

    if n_dupes:
        # Prefer rows that carry an adjusted close; break remaining ties with the
        # row that has the fewest nulls, then keep the last for determinism.
        prices = prices.copy()
        prices["_has_adj"] = prices["Adjusted Close"].notna().astype(int)
        prices["_nulls"] = prices[OHLCV_COLUMNS].isna().sum(axis=1)
        prices = prices.sort_values(key + ["_has_adj", "_nulls"], ascending=[True, True, False, False])

        dup_rows = prices[dup_mask]
        for (date, ticker), grp in dup_rows.groupby(key, sort=True):
            if len(grp) == 1:
                continue
            differing = [c for c in OHLCV_COLUMNS if grp[c].nunique(dropna=False) > 1]
            if differing:
                conflicts.append(
                    f"{pd.Timestamp(date).date()} {ticker}: {len(grp)} rows disagree on {differing}"
                )

        prices = prices.drop_duplicates(subset=key, keep="first")
        prices = prices.drop(columns=["_has_adj", "_nulls"])

    dropped = before - len(prices)
    LOG.info("Stage 3  duplicates: %d duplicate rows, %d removed", n_dupes, dropped)
    for c in conflicts[:20]:
        LOG.info("         conflict -> %s", c)
    if len(conflicts) > 20:
        LOG.info("         ... and %d more conflicts (see JSON report)", len(conflicts) - 20)

    return prices.reset_index(drop=True), dropped, conflicts


# ==========================================================================
# Stage 1 - structural alignment
# ==========================================================================
def build_trading_calendar(prices: pd.DataFrame, calendar_source: str) -> pd.DatetimeIndex:
    """The exchange trading calendar, taken from the benchmark index.

    Holidays and weekends are structural gaps and are therefore absent from the
    calendar by construction (PRD Section 3.4, Section 4.1 Stage 1).
    """
    if calendar_source not in set(prices["Ticker"]):
        raise RuntimeError(
            f"calendar_source {calendar_source!r} is not present in the raw data; "
            "cannot establish a trading calendar."
        )
    days = pd.DatetimeIndex(
        prices.loc[prices["Ticker"] == calendar_source, "Date"].sort_values().unique()
    )
    LOG.info("Stage 1  trading calendar: %d sessions, %s -> %s",
             len(days), days.min().date(), days.max().date())
    return days


def align_prices(prices: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Reindex every ticker onto the single trading calendar (wide, per column)."""
    wide = {
        col: prices.pivot(index="Date", columns="Ticker", values=col).reindex(calendar)
        for col in OHLCV_COLUMNS
    }
    frame = pd.DataFrame(index=calendar)
    for col, mat in wide.items():
        mat.columns.name = None
        frame = frame.join(mat.rename(columns=lambda t: f"{col}|{t}"))

    LOG.info("Stage 1  aligned panel: %d sessions x %d tickers x %d price fields",
             len(frame), prices["Ticker"].nunique(), len(OHLCV_COLUMNS))
    return frame


def align_macro(macro: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Forward-fill lower-frequency macro onto the daily trading grid.

    "a macro figure remains the last known value until the next release"
    (PRD Section 4.1 Stage 1). Back-filling is not used anywhere: a value is
    only ever carried forward from its own release date.
    """
    if macro is None or macro.empty:
        LOG.warning("Stage 1  macro frame is empty - no macro context will be available")
        return pd.DataFrame(index=calendar)

    macro = macro.copy()
    macro["Date"] = pd.to_datetime(macro["Date"])
    macro = macro.sort_values("Date").set_index("Date")

    # Restrict to the trading window, then carry values forward onto the grid.
    macro = macro.reindex(macro.index.union(calendar)).ffill().reindex(calendar)

    LOG.info("Stage 1  macro aligned: %d series x %d sessions", macro.shape[1], len(macro))
    for col in macro.columns:
        n_filled = int(macro[col].isna().sum())
        if n_filled:
            LOG.info("         %-10s %d leading NaN (before first release)", col, n_filled)
    return macro


# ==========================================================================
# Stage 2 - missing-value handling
# ==========================================================================
def impute_price_gaps(frame: pd.DataFrame, max_ffill: int) -> tuple[pd.DataFrame, dict]:
    """Forward-fill price gaps of at most ``max_ffill`` sessions. Never back-fill.

    Returns the imputed frame and a per-ticker accounting of what was filled and
    what genuine gaps remain.
    """
    out = frame.copy()
    accounting: dict[str, dict] = {}

    for ticker in sorted({c.split("|", 1)[1] for c in out.columns}):
        cols = [f"{c}|{ticker}" for c in OHLCV_COLUMNS]
        block = out[cols]

        was_missing = block["Close|{}".format(ticker)].isna()

        # Prices carry forward; volume is a count, so a carried price row means
        # "no reported volume" -> 0 (PRD Listing 4.1).
        block = block.ffill(limit=max_ffill)
        block["Volume|{}".format(ticker)] = block["Volume|{}".format(ticker)].fillna(0)

        out[cols] = block

        still_missing = int(block["Close|{}".format(ticker)].isna().sum())
        filled = int(was_missing.sum()) - still_missing
        leading = int(was_missing.iloc[: max_ffill + 1].sum())

        accounting[ticker] = {
            "sessions_missing_before": int(was_missing.sum()),
            "gap_cells_filled": filled,
            "remaining_gaps": still_missing,
            "remaining_are_leading_history": leading,
        }
        if filled:
            LOG.info("Stage 2  %-6s filled %5d gap cells (<=%d sessions)", ticker, filled, max_ffill)
        if still_missing:
            LOG.warning("Stage 2  %-6s %d sessions still missing after ffill "
                        "(> %d sessions, or before first observation) - NOT back-filled",
                        ticker, still_missing, max_ffill)

    total_remaining = sum(v["remaining_gaps"] for v in accounting.values())
    LOG.info("Stage 2  total remaining gaps: %d (policy: never back-fill)", total_remaining)
    return out, accounting


# ==========================================================================
# Stage 4 - outlier detection (flag only)
# ==========================================================================
def flag_return_outliers(frame: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Flag extreme daily log returns with BOTH a rolling z-score and a 1.5x IQR fence.

    Rows are never dropped (PRD Section 4.1 Stage 4, Listing 4.1).
    """
    params = cfg["clean"]["outlier"]
    win = int(params["rolling_window"])
    z_thresh = float(params["z_threshold"])
    iqr_mult = float(params["iqr_multiplier"])
    use_log = bool(params.get("log_return", True))

    out = frame.copy()
    summary: dict[str, dict] = {}

    price_cols = [c for c in out.columns if c.startswith("Adjusted Close|")]
    for col in price_cols:
        ticker = col.split("|", 1)[1]
        price = out[col]

        ret = np.log(price).diff() if use_log else price.pct_change()

        # Rolling z-score: local, so a volatility regime shift does not make
        # every subsequent day look "extreme".
        mu = ret.rolling(win, min_periods=max(20, win // 3)).mean()
        sd = ret.rolling(win, min_periods=max(20, win // 3)).std()
        z = (ret - mu) / sd.replace(0.0, np.nan)

        # 1.5 x IQR fence over the full return distribution.
        q1, q3 = ret.quantile(0.25), ret.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - iqr_mult * iqr, q3 + iqr_mult * iqr

        flag_z = (z.abs() > z_thresh).fillna(False)
        flag_iqr = ((ret < lo) | (ret > hi)).fillna(False)

        out[f"Return|{ticker}"] = ret
        out[f"Z_Score|{ticker}"] = z
        out[f"Outlier_Z|{ticker}"] = flag_z
        out[f"Outlier_IQR|{ticker}"] = flag_iqr
        out[f"Outlier|{ticker}"] = flag_z | flag_iqr

        n_z, n_iqr, n_any = int(flag_z.sum()), int(flag_iqr.sum()), int((flag_z | flag_iqr).sum())
        summary[ticker] = {
            "outliers_rolling_zscore": n_z,
            "outliers_iqr_fence": n_iqr,
            "outliers_either": n_any,
            "rows_retained": len(out),
        }
        LOG.info("Stage 4  %-6s rolling-z(>%g)=%4d  IQR(%gx)=%4d  union=%4d  [rows RETAINED]",
                 ticker, z_thresh, n_z, iqr_mult, n_iqr, n_any)

    LOG.info("Stage 4  total flagged rows: %d - flagged only, NOT deleted (PRD 4.1 Stage 4)",
             sum(v["outliers_either"] for v in summary.values()))
    return out, summary


# ==========================================================================
# Stage 5 - consistency validation
# ==========================================================================
def enforce_ohlc_invariant(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Assert the OHLC invariant on every row and quarantine violations.

    Invariant (PRD Section 4.1 Stage 5): ``Low <= min(Open, Close, High)`` and
    ``High >= max(Open, Close, Low)``. This is the exact form used in Listing
    4.1 and it subsumes Low<=Open, Close<=High and Low<=High.
    """
    out = frame.copy()
    tickers = sorted({c.split("|", 1)[1] for c in out.columns if c.startswith("Close|")})
    violations: list[dict] = []

    for ticker in tickers:
        o = out[f"Open|{ticker}"]
        h = out[f"High|{ticker}"]
        low = out[f"Low|{ticker}"]
        c = out[f"Close|{ticker}"]

        # Low <= min(Open, Close, High)  AND  High >= max(Open, Close, Low)
        min_och = pd.concat([o, c, h], axis=1).min(axis=1)
        max_ocl = pd.concat([o, c, low], axis=1).max(axis=1)
        bad = ((low > min_och) | (h < max_ocl)).fillna(False) & c.notna()

        n_bad = int(bad.sum())
        if n_bad:
            for ts in out.index[bad]:
                violations.append({
                    "date": str(pd.Timestamp(ts).date()),
                    "ticker": ticker,
                    "open": float(o.loc[ts]),
                    "high": float(h.loc[ts]),
                    "low": float(low.loc[ts]),
                    "close": float(c.loc[ts]),
                })
            for col in OHLCV_COLUMNS:
                out.loc[bad, f"{col}|{ticker}"] = np.nan
            LOG.warning("Stage 5  %-6s %d OHLC invariant violations quarantined", ticker, n_bad)

    LOG.info("Stage 5  total quarantined rows: %d", len(violations))
    return out, violations


# ==========================================================================
# Assemble the long-form analysis panel
# ==========================================================================
FLAG_FIELDS = ("Outlier", "Outlier_Z", "Outlier_IQR", "Z_Score", "Return")


def to_long_form(frame: pd.DataFrame) -> pd.DataFrame:
    """Melt the wide aligned frame into the tidy long panel the rest of the
    pipeline consumes: one row per (Date, Ticker).

    The aligned frame stores columns as ``"<Field>|<Ticker>"`` strings. We build
    one small frame per ticker so that ``Ticker`` appears exactly once per row,
    then concatenate.
    """
    tickers = sorted({c.partition("|")[2] for c in frame.columns if c.partition("|")[2]})
    if not tickers:
        raise RuntimeError("no '<Field>|<Ticker>' columns found in the aligned frame")

    price_parts, flag_parts = [], []
    for ticker in tickers:
        pcols = {f: frame[f"{f}|{ticker}"] for f in OHLCV_COLUMNS if f"{f}|{ticker}" in frame}
        if not pcols:
            continue
        block = pd.DataFrame(pcols)
        block.index.name = "Date"
        block["Ticker"] = ticker
        price_parts.append(block.reset_index())

        fcols = {f: frame[f"{f}|{ticker}"] for f in FLAG_FIELDS if f"{f}|{ticker}" in frame}
        if fcols:
            fblock = pd.DataFrame(fcols)
            fblock.index.name = "Date"
            fblock["Ticker"] = ticker
            flag_parts.append(fblock.reset_index())

    if not price_parts:
        raise RuntimeError("aligned frame contained no usable price columns")

    panel = pd.concat(price_parts, ignore_index=True)
    if flag_parts:
        panel = panel.merge(pd.concat(flag_parts, ignore_index=True),
                            on=["Date", "Ticker"], how="left")

    for col in ("Outlier", "Outlier_Z", "Outlier_IQR"):
        if col in panel.columns:
            panel[col] = panel[col].astype("object").where(panel[col].notna(), False).astype(bool)

    panel["Date"] = pd.to_datetime(panel["Date"])
    panel = panel.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    if panel.empty:
        raise RuntimeError("long-form panel is empty - check the aligned frame")

    assert not panel.columns.duplicated().any(), (
        f"duplicate columns after reshape: "
        f"{panel.columns[panel.columns.duplicated()].tolist()}"
    )
    return panel


# ==========================================================================
# Orchestration
# ==========================================================================
def clean() -> pd.DataFrame:
    """Run the ordered, idempotent cleaning pipeline."""
    ensure_dirs()
    cfg = load_config()
    set_seed(cfg.get("random_seed", 42))

    LOG.info("Cleaning log -> %s", (DQ_REPORT_JSON.parent / "cleaning.log"))
    banner(LOG, "PRD Section 4 - Data Cleaning Pipeline")

    # ---- load -------------------------------------------------------------
    if not RAW_PRICES.exists():
        raise FileNotFoundError(f"{RAW_PRICES} not found. Run `python src/ingest.py` first.")
    prices = pd.read_parquet(RAW_PRICES)
    prices["Date"] = pd.to_datetime(prices["Date"]).dt.tz_localize(None)
    rows_in = len(prices)
    LOG.info("Loaded raw prices: %d rows, %d tickers, %s -> %s",
             rows_in, prices["Ticker"].nunique(),
             prices["Date"].min().date(), prices["Date"].max().date())

    macro = None
    if RAW_MACRO.exists():
        macro = pd.read_parquet(RAW_MACRO)
        LOG.info("Loaded raw macro: %d rows, %d series", len(macro), max(0, macro.shape[1] - 1))
    else:
        LOG.warning("No raw macro file at %s", RAW_MACRO)

    # ---- Stage 3 (dedup first: alignment requires unique keys) ------------
    prices, dupes_dropped, conflicts = remove_duplicates(prices)

    # ---- Stage 1 ----------------------------------------------------------
    calendar = build_trading_calendar(prices, cfg["clean"]["calendar_source"])
    wide = align_prices(prices, calendar)
    macro_daily = align_macro(macro, calendar)

    # ---- Stage 2 ----------------------------------------------------------
    wide, gap_accounting = impute_price_gaps(wide, int(cfg["clean"]["max_price_ffill"]))

    # ---- Stage 4 ----------------------------------------------------------
    wide, outlier_summary = flag_return_outliers(wide, cfg)

    # ---- Stage 5 ----------------------------------------------------------
    wide, violations = enforce_ohlc_invariant(wide)

    # ---- assemble ---------------------------------------------------------
    panel = to_long_form(wide)

    # Quarantined / genuinely-missing rows cannot be modelled on.
    before_drop = len(panel)
    panel = panel[panel["Adjusted Close"].notna() & panel["Close"].notna()].copy()
    dropped_unusable = before_drop - len(panel)

    panel = panel.replace([np.inf, -np.inf], np.nan)
    panel["Date"] = pd.to_datetime(panel["Date"])
    panel = panel.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    panel.to_parquet(CLEANED_PANEL, index=False)
    if not macro_daily.empty:
        # Name the index explicitly: an unnamed DatetimeIndex round-trips through
        # parquet as __index_level_0__, which downstream code cannot identify.
        macro_daily.index.name = "Date"
        macro_daily.to_parquet(PROCESSED_DIR / "macro_daily.parquet")

    # ---- audit report (PRD Section 4.2) -----------------------------------
    report = {
        "generated_by": "src/clean.py",
        "config": {
            "calendar_source": cfg["clean"]["calendar_source"],
            "max_price_ffill": cfg["clean"]["max_price_ffill"],
            "outlier": cfg["clean"]["outlier"],
            "backfill_used": False,
        },
        "rows": {
            "rows_in": rows_in,
            "duplicate_rows_removed": dupes_dropped,
            "rows_out": int(len(panel)),
            "rows_dropped_unusable": int(dropped_unusable),
            "net_change_pct": round(100.0 * (len(panel) - rows_in) / rows_in, 4),
        },
        "calendar": {
            "sessions": int(len(calendar)),
            "start": str(calendar.min().date()),
            "end": str(calendar.max().date()),
            "tickers": int(panel["Ticker"].nunique()),
        },
        "gaps": gap_accounting,
        "outliers": outlier_summary,
        "outlier_policy": (
            "FLAGGED ONLY - not deleted, not winsorised. Section 4.1 Stage 4 requires that "
            "corporate-action moves be kept; confirming which flags are genuine vendor errors "
            "requires a corporate-actions calendar this project does not have."
        ),
        "invariant_violations": {
            "count": len(violations),
            "rule": "Low <= min(Open,Close,High) AND High >= max(Open,Close,Low)",
            "rows": violations[:200],
        },
        "duplicate_conflicts": conflicts,
    }
    write_json(report, DQ_REPORT_JSON)
    _write_markdown_report(report)

    banner(LOG, "Data Quality Summary")
    LOG.info("Rows in            : %d", rows_in)
    LOG.info("Duplicates removed : %d", dupes_dropped)
    LOG.info("Rows out           : %d", len(panel))
    LOG.info("Gap cells filled   : %d", sum(v["gap_cells_filled"] for v in gap_accounting.values()))
    LOG.info("Gaps remaining     : %d (never back-filled)",
             sum(v["remaining_gaps"] for v in gap_accounting.values()))
    LOG.info("Outliers flagged   : %d (retained)",
             sum(v["outliers_either"] for v in outlier_summary.values()))
    LOG.info("OHLC violations    : %d (quarantined)", len(violations))
    LOG.info("Report             : %s", DQ_REPORT_MD)
    return panel


def _write_markdown_report(rep: dict) -> None:
    """One-page data-quality report that ships with every dataset build."""
    r = rep["rows"]
    lines = [
        "# Data Quality Report",
        "",
        "Generated by `src/clean.py` (PRD Section 4.2). Ships with every dataset build.",
        "",
        "## Row accounting",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Rows in | {r['rows_in']:,} |",
        f"| Duplicate rows removed (Stage 3) | {r['duplicate_rows_removed']:,} |",
        f"| Rows dropped as unusable (quarantined / no price) | {r['rows_dropped_unusable']:,} |",
        f"| **Rows out** | **{r['rows_out']:,}** |",
        f"| Net change | {r['net_change_pct']:+.2f}% |",
        "",
        "## Calendar (Stage 1)",
        "",
        f"- Trading sessions: **{rep['calendar']['sessions']:,}** "
        f"({rep['calendar']['start']} to {rep['calendar']['end']})",
        f"- Tickers: **{rep['calendar']['tickers']}** (universe + benchmark + volatility index)",
        f"- Calendar source: `{rep['config']['calendar_source']}`",
        "",
        "## Missing-value handling (Stage 2)",
        "",
        f"- Forward-fill limit: **{rep['config']['max_price_ffill']} sessions**",
        "- **Back-filling: never used.** Back-filling leaks future information into the past "
        "(PRD Section 4.1 Stage 2).",
        "",
        "| Ticker | Missing before | Cells filled | Remaining | Of which leading history |",
        "|---|---|---|---|---|",
    ]
    for t, g in rep["gaps"].items():
        lines.append(
            f"| {t} | {g['sessions_missing_before']:,} | {g['gap_cells_filled']:,} | "
            f"{g['remaining_gaps']:,} | {g['remaining_are_leading_history']:,} |"
        )

    tot_filled = sum(v["gap_cells_filled"] for v in rep["gaps"].values())
    tot_rem = sum(v["remaining_gaps"] for v in rep["gaps"].values())
    lead = sum(v["remaining_are_leading_history"] for v in rep["gaps"].values())
    lines += [
        "",
        f"Totals: **{tot_filled:,}** cells filled, **{tot_rem:,}** remaining "
        f"({lead:,} of which precede each ticker's first observation).",
        "",
        "## Outlier detection (Stage 4)",
        "",
        f"- Rolling window: {rep['config']['outlier']['rolling_window']} sessions, "
        f"z-score fence > {rep['config']['outlier']['z_threshold']}",
        f"- IQR fence: {rep['config']['outlier']['iqr_multiplier']}x IQR on daily log returns",
        "",
        "> **Policy: flagged only.** Rows are never auto-deleted and never winsorised. "
        "Section 4.1 Stage 4 requires that a large move on an earnings date be kept as "
        "signal; distinguishing that from a vendor error needs a corporate-actions calendar "
        "this project does not have, so we preserve the data and expose the flags.",
        "",
        "| Ticker | Rolling z-score | 1.5x IQR | Union | Rows retained |",
        "|---|---|---|---|---|",
    ]
    for t, o in rep["outliers"].items():
        lines.append(
            f"| {t} | {o['outliers_rolling_zscore']:,} | {o['outliers_iqr_fence']:,} | "
            f"**{o['outliers_either']:,}** | {o['rows_retained']:,} |"
        )

    iv = rep["invariant_violations"]
    lines += [
        "",
        f"Total flagged rows: **{sum(o['outliers_either'] for o in rep['outliers'].values()):,}** "
        "(all retained)",
        "",
        "## Consistency validation (Stage 5)",
        "",
        f"- Invariant: `{iv['rule']}`",
        f"- **Violations quarantined: {iv['count']:,}**",
        "",
    ]
    if iv["rows"]:
        lines += ["| Date | Ticker | Open | High | Low | Close |", "|---|---|---|---|---|---|"]
        for row in iv["rows"][:25]:
            lines.append(
                f"| {row['date']} | {row['ticker']} | {row['open']} | {row['high']} | "
                f"{row['low']} | {row['close']} |"
            )
        if len(iv["rows"]) > 25:
            lines.append(f"| ... | *{len(iv['rows']) - 25} more in JSON report* | | | | |")

    lines += [
        "",
        "## Duplicate conflicts (Stage 3)",
        "",
        (f"- {len(rep['duplicate_conflicts'])} conflicting (date, ticker) groups resolved, "
         "vendor-adjusted record kept" if rep["duplicate_conflicts"] else "- None"),
        "",
        "---",
        "",
        "*Educational research and decision support only. Not financial advice.*",
        "",
    ]
    DQ_REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    clean()
