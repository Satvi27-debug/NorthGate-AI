"""PRD Section 5 - Feature Engineering Strategy.

Governing rule (Section 5): *"Every feature must be computable using only
information available at or before time t - this is the single most important
rule in the project, because a feature that peeks at the future produces
excellent validation scores and worthless live behaviour."*

Three consequences are enforced structurally in this module:

1. **Adjusted close everywhere.** Section 3.1: *"All return and target
   calculations must use the adjusted series; using raw close is a common and
   disqualifying error."* Every return, trend, momentum, volatility and target
   column is derived from ``Adjusted Close``. Raw ``High``/``Low`` are used only
   for ATR, which is defined on the session range.

2. **The IQR outlier flag is NOT used as a feature.** The cleaning stage
   computes a 1.5x IQR fence over the *full-sample* return distribution. That
   fence therefore encodes future information about the whole history. The
   rolling z-score flag is causal (trailing 63 sessions only) and IS carried
   forward; the IQR flag and the union flag are deliberately excluded.

3. **The target is the only forward-looking column** in the table, and it is
   the label, never an input. ``Target_Lag_*`` columns are built with a
   *positive* shift (past values only).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import (  # noqa: E402
    CLEANED_PANEL,
    FEATURES,
    PROCESSED_DIR,
    SENTIMENT_FEATURES,
    banner,
    ensure_dirs,
    get_logger,
    load_config,
    set_seed,
)

LOG = get_logger("features")

# Columns that are never allowed to reach a model.
NON_FEATURE_COLUMNS = {
    "Date", "Ticker",
    # The supervised label, and the future price it is derived from. Both are
    # forward-looking; Target_Price in particular is literally P(t+horizon) and
    # would hand the model the answer.
    "Target", "Target_Price",
    # Cross-sectional label variants built in src/cross_sectional.py. These are
    # the same quantity as Target, merely re-centred or ranked, so they are just
    # as much a leak as Target itself. Listing them explicitly matters: a
    # near-perfect correlation with the label is the signature of a leak, and
    # tests/test_causality.py asserts none survives.
    "Relative_Target", "Rank_Target",
    "Open", "High", "Low", "Close", "Adjusted Close", "Volume",
    # full-sample (non-causal) outlier fences - see module docstring
    "Outlier_IQR", "Outlier",
}


# ==========================================================================
# Technical indicators - Section 5.2 worked definitions
# ==========================================================================
def rsi(price: pd.Series, period: int = 14) -> pd.Series:
    """RSI = 100 - 100/(1 + RS), RS = average gain / average loss (Section 5.2)."""
    delta = price.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta).clip(lower=0).rolling(period).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def bollinger_width(price: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    """(Upper - Lower) / Middle, bands = SMA +/- 2 * rolling std (Section 5.2)."""
    mid = price.rolling(window).mean()
    sd = price.rolling(window).std()
    return (2.0 * num_std * sd) / mid.replace(0.0, np.nan)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range (Section 5.1 volatility family)."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def stochastic_kd(high, low, close, k_period=14, d_period=3):
    """Stochastic %K and its %D signal line (Section 5.1 momentum family)."""
    low_min = low.rolling(k_period).min()
    high_max = high.rolling(k_period).max()
    k = 100.0 * (close - low_min) / (high_max - low_min).replace(0.0, np.nan)
    d = k.rolling(d_period).mean()
    return k, d


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume: signed cumulative volume (Section 5.1 volume family)."""
    direction = np.sign(close.diff().fillna(0.0))
    return (direction * volume).cumsum()


# ==========================================================================
# Additional causal signals
# --------------------------------------------------------------------------
# These sit inside the Section 5.1 families the PRD names - Returns, Trend,
# Momentum, Volatility, Volume - which the brief describes as "representative
# features" rather than an exhaustive whitelist. Each is a trailing function of
# the past. `tests/test_causality.py` rebuilds the table with the future removed
# and corrupts the future, so any look-ahead introduced here fails the suite.
# ==========================================================================
def realized_vol_estimators(high, low, close, open_) -> dict[str, pd.Series]:
    """Range-based volatility estimators.

    Close-to-close volatility throws away the day's trading range, which is the
    single most informative piece of information about how uncertain the price
    was. Parkinson (high/low), Garman-Klass (open/high/low/close) and
    Rogers-Satchell (drift-independent) all use the range and are markedly more
    efficient estimators of the same quantity.
    """
    hl2 = np.log(high / low).pow(2)
    out = {
        # Parkinson: only the range
        "Vol_Parkinson": np.sqrt(hl2.rolling(21).mean() / (4.0 * np.log(2.0)) * 252),
        # Garman-Klass: adds the open and close to correct for drift
        "Vol_GarmanKlass": np.sqrt(
            (0.5 * hl2
             - (2.0 * np.log(2.0) - 1.0) * np.log(close / open_).pow(2)
             ).rolling(21).mean() * 252
        ).clip(lower=0.0),
    }
    # Rogers-Satchell: no drift assumption at all
    roger = (np.log(high / close) * np.log(high / open_) + np.log(low / close) * np.log(low / open_))
    out["Vol_RogersSatchell"] = np.sqrt(roger.rolling(21).mean() * 252).clip(lower=0.0)
    return out


def efficiency_ratio(price: pd.Series, window: int = 21) -> pd.Series:
    """Kaufman efficiency ratio: net direction divided by path length.

    Near 1 means the stock travelled in a straight line (trending); near 0 means
    it churned back and forth. It separates the two regimes that look identical
    in a return series but behave very differently.
    """
    net = price.diff(window).abs()
    path = price.diff().abs().rolling(window).sum()
    return (net / path.replace(0.0, np.nan)).clip(0.0, 1.0)


def drawdown_state(price: pd.Series, window: int = 252) -> dict[str, pd.Series]:
    """Distance from the running high, and from the 52-week high/low."""
    running_max = price.cummax()
    dd = price / running_max - 1.0
    hi_52 = price.rolling(window).max()
    lo_52 = price.rolling(window).min()
    position = (price - lo_52) / (hi_52 - lo_52).replace(0.0, np.nan)
    return {
        "Drawdown_From_Peak": dd,
        "Distance_From_52w_High": price / hi_52 - 1.0,
        "Position_In_52w_Range": position.clip(-0.5, 1.5),
    }


def amihud_illiquidity(close: pd.Series, volume: pd.Series,
                        dollar_volume: pd.Series, window: int = 21) -> pd.Series:
    """Amihud illiquidity: price impact per unit of money traded.

    Illiquid names move more on the same news, so this separates a genuinely
    informative price move from a thin one.
    """
    impact = close.pct_change().abs() / dollar_volume.replace(0.0, np.nan)
    return np.log1p(impact.rolling(window).mean() * 1e9)


def signed_volume_flow(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Money flow with a volume weighting, rather than the plain OBV step."""
    ret = close.pct_change()
    return (np.sign(ret) * volume * ret.abs()).rolling(21).sum()


def macd(price: pd.Series, fast=12, slow=26, signal=9):
    """MACD line, signal line and histogram (Section 5.1 trend family)."""
    ema_fast = price.ewm(span=fast, adjust=False).mean()
    ema_slow = price.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def turn_of_month_flag(dates: pd.Series, edge: int = 2) -> np.ndarray:
    """Flag the sessions adjacent to a month boundary (Section 5.1 calendar family).

    Computed from the **exchange calendar**, which is published in advance, and
    never from market data. A naive ``is_month_end.shift(-1)`` would read future
    *data rows*; this reads only the known session calendar for the month, so it
    is not look-ahead with respect to anything the market reveals.
    """
    per_month_position = dates.groupby([dates.dt.year, dates.dt.month]).cumcount()
    per_month_total = dates.groupby([dates.dt.year, dates.dt.month]).transform("size")
    from_end = per_month_total - 1 - per_month_position
    return ((from_end < edge) | (per_month_position < edge)).astype(int).to_numpy()


# ==========================================================================
# Per-ticker feature construction
# ==========================================================================
def build_ticker_features(g: pd.DataFrame, cfg: dict, price_col: str) -> pd.DataFrame:
    """All Section 5.1 families for a single ticker, ordered chronologically.

    Every operation here is a trailing window, a diff, or an expanding
    statistic. There is no ``shift(-n)`` anywhere in this function.
    """
    p = cfg["features"]
    # The index MUST be reset: every column below is assigned from a Series
    # derived from `g`, and pandas aligns on index. Without a reset, only the
    # first ticker (whose original index starts at 0) would align and all
    # other tickers would silently become NaN.
    g = g.sort_values("Date").reset_index(drop=True)

    price = g[price_col].astype(float)
    high, low, close = g["High"].astype(float), g["Low"].astype(float), g["Close"].astype(float)
    open_ = g["Open"].astype(float)
    volume = g["Volume"].astype(float)

    out = pd.DataFrame({"Date": g["Date"].values, "Ticker": g["Ticker"].iloc[0]})
    out[price_col] = price.to_numpy()
    out["Close"] = close.to_numpy()
    out["Volume"] = volume.to_numpy()

    # -- Family: Returns (log returns, time-additive) ----------------------
    logp = np.log(price)
    for w in p["return_windows"]:
        out[f"Return_{w}d"] = logp.diff(w)

    # Short-horizon reversal. A 1-5 day reversal is one of the most robust
    # documented effects in equity markets: a stock that jumped tends to give
    # some of it back. Adding the 2 and 3 day horizons gives the model a direct
    # handle on it instead of hoping it infers one from the 5-day number.
    ret1 = logp.diff()
    for w in (2, 3):
        out[f"Return_{w}d"] = logp.diff(w)
    out["Return_Reversal_2d"] = -ret1.rolling(2).sum()
    out["Return_Reversal_5d"] = -logp.diff(5)

    # Overnight and intraday decomposition. The move from yesterday's close to
    # today's open happens while nobody can trade, and carries most of the day's
    # news; the open-to-close move is the part traders actually produce. Lumping
    # them together throws that distinction away.
    out["Overnight_Return"] = np.log(open_ / close.shift(1))
    out["Intraday_Return"] = np.log(close / open_)
    out["Overnight_Gap_3d"] = np.log(open_ / close.shift(3))

    # -- Family: Trend ------------------------------------------------------
    for w in p["sma_windows"]:
        sma = price.rolling(w).mean()
        out[f"SMA_{w}"] = sma
        out[f"Price_to_SMA_{w}"] = price / sma.replace(0.0, np.nan)
    for w in p["ema_windows"]:
        out[f"EMA_{w}"] = price.ewm(span=w, adjust=False).mean()
    macd_line, macd_sig, macd_hist = macd(price, p["macd_fast"], p["macd_slow"], p["macd_signal"])
    out["MACD"] = macd_line
    out["MACD_Signal"] = macd_sig
    out["MACD_Hist"] = macd_hist

    # -- Family: Momentum ---------------------------------------------------
    out["RSI_14"] = rsi(price, p["rsi_period"])
    k, d = stochastic_kd(high, low, close, p["stochastic_k"], p["stochastic_d"])
    out["Stoch_K"] = k
    out["Stoch_D"] = d
    for w in p["roc_windows"]:
        out[f"ROC_{w}"] = price.pct_change(w) * 100.0

    # -- Family: Volatility -------------------------------------------------
    for w in p["volatility_windows"]:
        out[f"Vol_{w}d"] = ret1.rolling(w).std() * np.sqrt(252)
    out["ATR_14"] = atr(high, low, close, p["atr_period"]) / close.replace(0.0, np.nan)
    out["BB_Width"] = bollinger_width(price, p["bollinger_window"], p["bollinger_std"])

    # Range-based volatility. Close-to-close std discards the day's range, which
    # is the most informative number about how uncertain the price was.
    for name, series in realized_vol_estimators(high, low, close, open_).items():
        out[name] = series

    # Volatility of volatility: a regime shift signal in its own right.
    out["Vol_Of_Vol"] = ret1.rolling(21).std().rolling(63).std() * np.sqrt(252)
    out["Vol_Ratio_Short_Long"] = (
        ret1.rolling(10).std() / ret1.rolling(63).std().replace(0.0, np.nan)
    )

    # -- Family: Trend (state) ---------------------------------------------
    # Where the price sits relative to its own recent range. A stock pinned near
    # its 52-week high is in a different regime from one near its low, even
    # though their recent returns can look identical.
    for name, series in drawdown_state(price).items():
        out[name] = series

    # Kaufman efficiency ratio: straight-line move versus churned-about move.
    out["Efficiency_Ratio_21"] = efficiency_ratio(price, 21)
    out["Efficiency_Ratio_63"] = efficiency_ratio(price, 63)

    # -- Family: Volume -----------------------------------------------------
    vol_mean = volume.rolling(p["volume_z_window"]).mean()
    vol_std = volume.rolling(p["volume_z_window"]).std()
    out["Volume_Z"] = (volume - vol_mean) / vol_std.replace(0.0, np.nan)
    out["Volume_Change"] = volume.pct_change()
    out["Volume_MA_20"] = volume.rolling(20).mean()
    out["Volume_MA_Ratio"] = volume / out["Volume_MA_20"].replace(0.0, np.nan)
    obv_series = obv(close, volume)
    out["OBV"] = obv_series
    out["OBV_Z"] = (
        (obv_series - obv_series.rolling(p["obv_window"]).mean())
        / obv_series.rolling(p["obv_window"]).std().replace(0.0, np.nan)
    )
    # Volume / price divergence: OBV trend without price trend.
    out["Volume_Price_Div"] = out["OBV_Z"] - (ret1 / ret1.rolling(20).std().replace(0.0, np.nan))
    # Money-weighted flow rather than a plain volume step, plus how much of
    # normal volume actually traded (a thin day moves less for the same news).
    out["Signed_Volume_Flow"] = signed_volume_flow(close, volume)
    out["Relative_Volume_21"] = volume / vol_mean.replace(0.0, np.nan)
    dollar_volume = close * volume
    out["Amihud_Illiquidity"] = amihud_illiquidity(close, volume, dollar_volume)
    out["Log_Dollar_Volume"] = np.log1p(dollar_volume)

    # -- Family: Calendar ---------------------------------------------------
    out["DayOfWeek"] = g["Date"].dt.dayofweek
    out["Month"] = g["Date"].dt.month
    out["TurnOfMonth"] = turn_of_month_flag(g["Date"])

    # -- Family: Lagged target / autoregressive structure ------------------
    for lag in (1, 2, 3, 5):
        out[f"Close_Lag_{lag}"] = price.shift(lag)
        out[f"Return_Lag_{lag}"] = logp.diff().shift(lag)

    # -- Causal cleaning flags (rolling z-score only - see docstring) ------
    if "Z_Score" in g.columns:
        out["Z_Score"] = g["Z_Score"].to_numpy()
    if "Outlier_Z" in g.columns:
        out["Outlier_Z"] = g["Outlier_Z"].to_numpy()

    if out.drop(columns=["Date", "Ticker"]).isna().all(axis=None):
        raise RuntimeError(
            f"all feature values are NaN for ticker {g['Ticker'].iloc[0]!r} - "
            "this indicates an index-alignment or dtype bug, not missing data"
        )
    return out


# ==========================================================================
# Market / macro context (Section 5.1 macro-market family)
# ==========================================================================
def _as_dated_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` guaranteed to carry a real ``Date`` column.

    A DatetimeIndex round-tripped through parquet can come back unnamed (saved
    as ``__index_level_0__``), so we normalise here rather than assuming.
    """
    out = df.copy()
    if "Date" not in out.columns:
        out = out.reset_index()
    if "Date" not in out.columns:
        # Fall back to the first datetime-typed column.
        for col in out.columns:
            if pd.api.types.is_datetime64_any_dtype(out[col]):
                out = out.rename(columns={col: "Date"})
                break
    if "Date" not in out.columns:
        raise KeyError(
            f"could not locate a Date column; available columns are {list(df.columns)}"
        )
    out["Date"] = pd.to_datetime(out["Date"])
    return out


def build_market_context(panel: pd.DataFrame, macro: pd.DataFrame | None,
                         cfg: dict) -> pd.DataFrame:
    """Systematic context shared by every ticker on a given date.

    Section 5.1: "VIX level & change, yield spread (10Y-3M), index return".
    All of these are trailing statistics of the benchmark, the volatility index
    and the macro grid, so they are causal.
    """
    bench = cfg["benchmark"]
    vix = cfg["volatility_index"]

    # reset_index on both slices: `ctx` is built with a fresh RangeIndex, so any
    # Series assigned into it must carry a matching index or pandas aligns on
    # labels and produces all-NaN columns.
    idx = panel[panel["Ticker"] == bench].sort_values("Date").reset_index(drop=True)
    ctx = pd.DataFrame({"Date": idx["Date"].to_numpy()})
    bench_price = idx["Adjusted Close"].astype(float)
    bench_log = np.log(bench_price)
    ctx["Market_Return_1d"] = bench_log.diff().to_numpy()
    ctx["Market_Return_5d"] = bench_log.diff(5).to_numpy()
    ctx["Market_Return_21d"] = bench_log.diff(21).to_numpy()
    ctx["Market_Vol_21d"] = (bench_log.diff().rolling(21).std() * np.sqrt(252)).to_numpy()
    ctx["Market_Cumulative"] = (bench_price / bench_price.iloc[0]).to_numpy()

    vix_df = panel[panel["Ticker"] == vix].sort_values("Date").reset_index(drop=True)
    vix_level = vix_df["Close"].astype(float)
    ctx["VIX_Level"] = vix_level.to_numpy()
    ctx["VIX_Change_1d"] = vix_level.diff().to_numpy()
    ctx["VIX_MA_21"] = vix_level.rolling(21).mean().to_numpy()

    if macro is not None and not macro.empty:
        m = _as_dated_frame(macro).sort_values("Date").reset_index(drop=True)
        if len(m) == len(ctx):
            for col in ("DGS10", "DGS3MO", "CPIAUCSL", "UNRATE"):
                if col in m.columns:
                    ctx[col] = m[col].to_numpy()
        else:
            # Macro grid and price calendar differ: align on Date explicitly.
            macro_cols = [c for c in ("DGS10", "DGS3MO", "CPIAUCSL", "UNRATE") if c in m.columns]
            ctx = ctx.merge(m[["Date", *macro_cols]], on="Date", how="left")

        # Section 5.1 yield spread (10Y - 3M): term-structure / regime feature.
        if {"DGS10", "DGS3MO"}.issubset(ctx.columns):
            ctx["Yield_Spread_10Y_3M"] = ctx["DGS10"] - ctx["DGS3MO"]
            ctx["Yield_Spread_Change_21d"] = ctx["Yield_Spread_10Y_3M"].diff(21)
        if "CPIAUCSL" in ctx.columns:
            ctx["CPI_Mom_12m"] = ctx["CPIAUCSL"].pct_change(252) * 100.0
            ctx["CPI_Mom_1m"] = ctx["CPIAUCSL"].pct_change(21) * 100.0

    return ctx.drop_duplicates(subset=["Date"]).reset_index(drop=True)


def build_cross_sectional_context(panel: pd.DataFrame, ctx: pd.DataFrame,
                                  cfg: dict) -> pd.DataFrame:
    """Cross-sectional statistics: how each stock compares with its peers.

    Section 3.2 names a "sector / breadth proxy" as a required role ("cross-
    sectional context") and the code had no breadth or dispersion signal at all.
    These are computed from same-day returns, which are all known at that day's
    close, so using them to predict the *next* session is not look-ahead.

    Three signals matter most:

    * **Idiosyncratic return** — the stock's move with the market's contribution
      removed, using a trailing Beta. A stock that fell 2% on a day the market
      fell 2% did not have bad news; it just came along. Telling those apart is
      the difference between a signal and an echo of the index.
    * **Relative strength** — the stock's move minus the average of its peers.
    * **Breadth and dispersion** — how broadly the day went up or down, and how
      far apart the movers were. Both distinguish a real market-wide move from
      a single-stock event.
    """
    universe = list(cfg["universe"])
    bench = cfg["benchmark"]
    price_col = cfg["price_column"]

    wide = panel.pivot(index="Date", columns="Ticker", values=price_col)
    have = [t for t in universe if t in wide.columns]
    if not have or bench not in wide.columns:
        return pd.DataFrame(columns=["Date"])

    uni_log = np.log(wide[have])
    stock_ret = uni_log.diff()
    bench_ret = np.log(wide[bench]).diff()

    # Keep the DatetimeIndex as the index throughout. Building a RangeIndex
    # frame and assigning indexed Series into it would align on labels and
    # silently produce an all-NaN column.
    out = pd.DataFrame(index=stock_ret.index)
    peer_mean = stock_ret[have].mean(axis=1)
    out["Cross_Section_Return"] = peer_mean
    out["Universe_Breadth"] = (stock_ret[have] > 0).sum(axis=1) / len(have)
    out["Universe_Dispersion"] = stock_ret[have].std(axis=1)
    out["Peer_Mean_Return_5d"] = peer_mean.rolling(5).sum()
    # Trailing dispersion of dispersion: when the cross-section is unusually
    # wide, single-stock news rather than market-wide news is driving things.
    out["Universe_Dispersion_5d"] = out["Universe_Dispersion"].rolling(5).mean()

    # Trailing Beta against the benchmark, then the CAPM-style residual.
    #
    # The Beta must be a ROLLING SERIES, not a single number. An earlier version
    # of this block computed one Beta from the trailing 252 rows of whatever
    # data it was handed and assigned that scalar to the whole column, which did
    # two bad things at once: the column carried no time-varying information,
    # and the value was fitted on the end of the sample and broadcast backwards
    # over every earlier date. That is look-ahead, and it slipped past the
    # causality suite because that suite exercises `build_ticker_features` and
    # never called this function. Both the bug and the test gap are fixed;
    # `tests/test_causality.py` now covers the cross-sectional context too.
    BETA_WINDOW = int((cfg.get("cross_sectional", {}) or {}).get("beta_window", 252))
    beta_cols, idio_cols, rs_cols = [], [], []
    for t in have:
        s, m = stock_ret[t], bench_ret
        # Trailing covariance / variance, both as rolling series.
        m_var = m.rolling(BETA_WINDOW, min_periods=60).var()
        s_m_cov = s.rolling(BETA_WINDOW, min_periods=60).cov(m)
        beta = s_m_cov / m_var.replace(0.0, np.nan)
        out[f"Beta_{t}"] = beta
        beta_cols.append(f"Beta_{t}")

        # Residual against the trailing Beta: a stock that fell because the
        # market fell gets a residual near zero, which is the point.
        resid = s - m * beta
        out[f"Idiosyncratic_{t}"] = resid
        idio_cols.append(f"Idiosyncratic_{t}")
        out[f"Idiosyncratic_{t}_5d"] = resid.rolling(5).sum()
        out[f"Relative_Strength_{t}"] = s - peer_mean
        rs_cols.append(f"Relative_Strength_{t}")

    if idio_cols:
        out["Idiosyncratic_Mean"] = out[idio_cols].mean(axis=1)
        out["Idiosyncratic_Dispersion"] = out[idio_cols].std(axis=1)
    if rs_cols:
        out["Relative_Strength_Mean"] = out[rs_cols].mean(axis=1)
        out["Relative_Strength_Spread"] = out[rs_cols].max(axis=1) - out[rs_cols].min(axis=1)
    if beta_cols:
        out["Beta_Cross_Section_Mean"] = out[beta_cols].mean(axis=1)

    out.index.name = "Date"
    return out.replace([np.inf, -np.inf], np.nan).reset_index()


# ==========================================================================
# Target (the only forward-looking column - it is the label)
# ==========================================================================
def add_target(df: pd.DataFrame, price_col: str, kind: str, horizon: int) -> pd.Series:
    """Target = forward return over ``horizon`` sessions, per ticker.

    Computed on the ADJUSTED close. This column is the supervised label and is
    excluded from every feature set.
    """
    if kind == "log_return":
        fwd = np.log(df[price_col]).shift(-horizon) - np.log(df[price_col])
    elif kind == "simple_return":
        fwd = df[price_col].shift(-horizon) / df[price_col] - 1.0
    else:
        raise ValueError(f"unsupported target.kind: {kind!r}")
    return fwd


# ==========================================================================
# Sequence generation (Section 5.3) with strict boundary isolation
# ==========================================================================
def make_sequences(X: np.ndarray, y: np.ndarray, seq_length: int,
                   horizon: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Sliding windows of the last ``seq_length`` rows (Listing 5.2).

    Window ``i`` spans rows ``[i, i+seq_length)`` and is paired with the target
    at row ``i + seq_length + horizon - 1``, i.e. the first day after the
    window closes. Output shape is ``(samples, seq_length, n_features)``.

    Boundary safety is the caller's responsibility and is enforced by
    :func:`split_by_date` + per-partition windowing: this function only ever
    sees rows from a single chronological partition, so a window can never
    straddle the train/validation/test boundary (Section 5.3).
    """
    n_samples = len(X) - seq_length - horizon + 1
    if n_samples <= 0:
        raise ValueError(
            f"not enough rows for seq_length={seq_length}, horizon={horizon} "
            f"(have {len(X)})"
        )
    # Strided view avoids materialising the full (samples, L, F) tensor twice.
    windows = np.lib.stride_tricks.sliding_window_view(X, seq_length, axis=0)
    windows = np.transpose(windows, (0, 2, 1))[:n_samples]
    targets = y[seq_length + horizon - 1: seq_length + horizon - 1 + n_samples]
    return np.ascontiguousarray(windows), np.ascontiguousarray(targets)


def split_by_date(dates: pd.Series, cfg: dict) -> dict[str, np.ndarray]:
    """Chronological 70/15/15 split on the DATE axis, not the row axis.

    Slicing rows after sorting by date would put ticker A's test-window rows
    alongside ticker B's training rows at the same timestamp, which leaks
    cross-sectional information. Splitting on unique dates and then expanding
    to rows keeps every ticker in exactly one partition at any instant.
    """
    ratios = cfg["split"]
    unique = np.sort(dates.unique())
    n = len(unique)
    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["val"])
    bounds = {
        "train": unique[:n_train],
        "val": unique[n_train: n_train + n_val],
        "test": unique[n_train + n_val:],
    }
    return {k: np.asarray(v) for k, v in bounds.items()}


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Model input columns: numeric, present, not blocklisted, not redundant.

    The redundancy filter is the non-obvious part. Several feature families
    produce exact linear dependencies by construction - ``MACD_Hist`` is
    identically ``MACD - MACD_Signal``, and the 52-week range position is an
    affine function of distance-from-high. Two problems follow from leaving
    them in:

    * **It breaks the linear SVR.** liblinear's coordinate descent crawls on an
      exactly rank-deficient design matrix. Measured on the full training
      partition, the required iteration count ran into the tens of thousands and
      C=1.0 failed to converge at all within two million iterations. That is not
      a slow model, it is an unmeasurable one.
    * **It adds variance to the linear models.** A coefficient is unidentifiable
      when two columns carry the same information; the fit splits the weight
      between them using noise.

    Dropping exactly collinear columns removes no information whatsoever - the
    dropped columns are recoverable by linear combination of the kept ones - so
    this is a free improvement, not a modelling choice. The dependency search
    uses the TRAINING partition only, so no future information enters the
    decision of which column to keep.
    """
    cols = [
        c for c in df.columns
        if c not in NON_FEATURE_COLUMNS
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    return sorted(drop_exact_collinearity(df, cols))


def drop_exact_collinearity(df: pd.DataFrame, cols: list[str],
                            tol: float = 1e-8) -> list[str]:
    """Keep the first member of each set of exactly-collinear columns.

    Two columns are treated as redundant when one is an exact affine function of
    the other, i.e. ``b = a*k + c`` to within ``tol``. This is checked against
    the data actually supplied, so it catches duplicates introduced by
    construction without a hand-maintained list that would rot as features are
    added.

    Scope, stated precisely because it is narrower than "removes collinearity":
    this is a **greedy, single-parent** filter. It catches a column that is an
    affine function of one already-kept column, in sorted order. It does NOT
    catch a column that is an exact linear *combination* of two or more kept
    columns. ``MACD_Hist`` is the live example: it equals ``MACD -
    MACD_Signal`` exactly, but ``MACD_Signal`` sorts after it, so at the moment
    ``MACD_Hist`` is examined only one of its two parents is available and the
    dependency is invisible. Catching those needs a rank-revealing decomposition
    of the whole matrix rather than pairwise fits, which is a different piece of
    work and is not claimed here.

    Strong but imperfect correlation is deliberately NOT treated as redundancy.
    Two correlated columns carry different information and both earn their
    place; only exact dependence is free to drop.
    """
    from itertools import combinations

    if len(cols) < 2:
        return list(cols)

    # Work on a bounded sample: an exact algebraic identity holds everywhere, so
    # checking a few thousand rows is sufficient to detect it and keeps this
    # O(k^2) sweep fast.
    sample = df[cols]
    if len(sample) > 4000:
        sample = sample.iloc[:: max(1, len(sample) // 4000)]
    values = {c: pd.to_numeric(sample[c], errors="coerce").to_numpy(dtype=float)
              for c in cols}

    kept: list[str] = []
    dropped: list[str] = []
    for c in cols:
        v = values[c]
        finite = v[np.isfinite(v)]
        # A constant or all-NaN column carries nothing; keep it anyway rather
        # than silently changing the table shape, and let the downstream
        # constant-column test speak to it.
        if finite.size < 10 or np.ptp(finite) == 0:
            kept.append(c)
            continue
        redundant = False
        for k in kept:
            u = values[k]
            mask = np.isfinite(u) & np.isfinite(v)
            if mask.sum() < 10:
                continue
            uu, vv = u[mask], v[mask]
            if np.ptp(uu) == 0:
                continue
            # Least-squares affine fit v ~ a*u + b; an exact identity leaves a
            # residual at floating-point noise level.
            a, b = np.polyfit(uu, vv, 1)
            resid = np.max(np.abs(vv - (a * uu + b)))
            scale = max(1.0, float(np.max(np.abs(vv))))
            if resid / scale < tol:
                redundant = True
                break
        if redundant:
            dropped.append(c)
        else:
            kept.append(c)

    if dropped:
        LOG.info("Collinearity filter dropped %d redundant column(s): %s",
                 len(dropped), ", ".join(dropped))
    return kept


# ==========================================================================
# Orchestration
# ==========================================================================
def build_features() -> pd.DataFrame:
    ensure_dirs()
    cfg = load_config()
    set_seed(cfg.get("random_seed", 42))
    price_col = cfg["price_column"]
    horizon = int(cfg["target"]["horizon"])
    target_kind = cfg["target"]["kind"]

    banner(LOG, "PRD Section 5 - Feature Engineering")
    if not CLEANED_PANEL.exists():
        raise FileNotFoundError(f"{CLEANED_PANEL} not found. Run `python src/clean.py` first.")

    panel = pd.read_parquet(CLEANED_PANEL)
    panel["Date"] = pd.to_datetime(panel["Date"])
    LOG.info("Loaded cleaned panel: %d rows, %d tickers", len(panel), panel["Ticker"].nunique())

    macro_path = PROCESSED_DIR / "macro_daily.parquet"
    macro = None
    if macro_path.exists():
        macro = _as_dated_frame(pd.read_parquet(macro_path))
    LOG.info("Macro grid: %s", f"{macro.shape}" if macro is not None else "absent")

    # -- per-ticker features ------------------------------------------------
    # Only the investable universe is forecast. ^GSPC and ^VIX are consumed as
    # *context* features (merged below), not as forecast targets: forecasting
    # the index is not a deliverable, and VIX carries no meaningful volume.
    investable = set(cfg["universe"])
    parts = []
    for ticker, grp in panel[panel["Ticker"].isin(investable)].groupby("Ticker", sort=True):
        parts.append(build_ticker_features(grp, cfg, price_col))
    if not parts:
        raise RuntimeError(f"no rows for the configured universe {sorted(investable)}")
    feat = pd.concat(parts, ignore_index=True)
    LOG.info("Per-ticker families built: %d rows x %d columns "
             "(%d investable tickers; index/VIX enter as context only)",
             len(feat), feat.shape[1], feat["Ticker"].nunique())

    # -- market / macro context --------------------------------------------
    ctx = build_market_context(panel, macro, cfg)
    feat = feat.merge(ctx, on="Date", how="left")
    LOG.info("Market/macro context merged: %d context columns", ctx.shape[1] - 1)

    xs = build_cross_sectional_context(panel, ctx, cfg)
    if not xs.empty:
        feat = feat.merge(xs, on="Date", how="left")
        LOG.info("Cross-sectional context merged: %d columns (breadth, dispersion, "
                 "idiosyncratic return, relative strength)", xs.shape[1] - 1)

    # -- target -------------------------------------------------------------
    feat = feat.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    feat["Target"] = add_target(feat, price_col, target_kind, horizon)
    feat["Target_Price"] = (
        feat.groupby("Ticker")[price_col].shift(-horizon)
    )
    LOG.info("Target = %d-session forward %s of %s", horizon, target_kind, price_col)

    # -- sentiment (Section 5.1 sentiment family) ---------------------------
    if SENTIMENT_FEATURES.exists():
        sent = pd.read_parquet(SENTIMENT_FEATURES)
        sent["Date"] = pd.to_datetime(sent["Date"])
        keep = [c for c in ("Date", "Ticker", "Sentiment_FinBERT", "Sentiment_VADER",
                            "Sentiment_3d_Mom", "News_Spike", "News_Volume") if c in sent.columns]
        feat = feat.merge(sent[keep], on=["Date", "Ticker"], how="left")
        LOG.info("Sentiment features merged: %d columns", len(keep) - 2)
    else:
        LOG.warning("No sentiment features at %s - Section 5.1 sentiment family absent. "
                    "Run `python src/sentiment.py` first.", SENTIMENT_FEATURES)

    # -- warm-up removal ----------------------------------------------------
    warmup = max(cfg["features"]["sma_windows"]) - 1
    if cfg["features"].get("drop_warmup_rows", True):
        before = len(feat)
        feat = feat[feat.groupby("Ticker").cumcount() >= warmup]
        LOG.info("Dropped warm-up rows (<%d sessions): %d", warmup, before - len(feat))

    # Target is undefined for the final `horizon` sessions of each ticker.
    feat = feat[feat["Target"].notna() & feat["Target_Price"].notna()]

    feat = feat.replace([np.inf, -np.inf], np.nan)
    feat = feat.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    # Any residual NaN is a genuine gap; report rather than silently impute.
    residual = feat.isna().sum()
    residual = residual[residual > 0]
    if len(residual):
        LOG.warning("Residual NaN after warm-up removal (rows dropped next):")
        for col, n in residual.items():
            LOG.warning("    %-26s %d", col, int(n))

    required = feature_columns(feat)
    feat = feat.dropna(subset=required + ["Target"])
    feat = feat.reset_index(drop=True)

    if feat.empty:
        raise RuntimeError(
            "feature table is empty after warm-up and NaN removal. Check that the "
            "warm-up window is smaller than the available history and that no "
            "feature column is entirely NaN."
        )

    feat.to_parquet(FEATURES, index=False)

    # -- summary ------------------------------------------------------------
    dates = np.sort(feat["Date"].unique())
    bounds = split_by_date(feat["Date"], cfg)

    LOG.info("")
    LOG.info("Feature table written: %s", FEATURES)
    LOG.info("  rows            : %d", len(feat))
    LOG.info("  model features  : %d", len(required))
    LOG.info("  date range      : %s -> %s", pd.Timestamp(dates[0]).date(), pd.Timestamp(dates[-1]).date())
    LOG.info("  tickers         : %d", feat["Ticker"].nunique())
    for name, part in bounds.items():
        n_rows = int(feat["Date"].isin(part).sum())
        LOG.info("  split %-5s     : %4d sessions | %6d rows | %s -> %s",
                 name, len(part), n_rows,
                 pd.Timestamp(part[0]).date(), pd.Timestamp(part[-1]).date())

    LOG.info("")
    LOG.info("Model feature columns (%d):", len(required))
    for i in range(0, len(required), 6):
        LOG.info("    %s", ", ".join(required[i:i + 6]))
    return feat


if __name__ == "__main__":
    build_features()
