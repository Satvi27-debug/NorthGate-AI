"""PRD Section 3 deterministic market and macroeconomic data ingestion."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pandas_datareader.data as web
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import (  # noqa: E402
    RAW_DIR,
    all_tickers,
    load_config,
    resolve_end_date,
)

REQUIRED_PRICE_COLUMNS = ["Date", "Ticker", "Open", "High", "Low", "Close", "Adjusted Close", "Volume"]


def _normalise_prices(download: pd.DataFrame) -> pd.DataFrame:
    """Convert yfinance's multi-ticker result to the PRD long-form contract."""
    if download.empty or not isinstance(download.columns, pd.MultiIndex):
        raise RuntimeError("yfinance did not return a multi-ticker market-data result.")
    ticker_level = download.columns.names.index("Ticker") if "Ticker" in download.columns.names else 1
    prices = download.stack(level=ticker_level, future_stack=True).reset_index()
    prices = prices.rename(columns={"Adj Close": "Adjusted Close"})
    if "Date" not in prices.columns:
        prices = prices.rename(columns={prices.columns[0]: "Date"})
    missing = set(REQUIRED_PRICE_COLUMNS).difference(prices.columns)
    if missing:
        raise RuntimeError(f"yfinance response is missing required fields: {sorted(missing)}")
    return prices[REQUIRED_PRICE_COLUMNS].sort_values(["Ticker", "Date"]).reset_index(drop=True)


def ingest_data() -> tuple[Path, Path]:
    """Download PRD-required sources and atomically replace raw artifacts."""
    config = load_config()
    # Keep yfinance's SQLite cache inside the writable repository.
    yf.set_tz_cache_location(str(ROOT / ".yfinance-cache"))
    start = config["start_date"]
    end = resolve_end_date(config)
    tickers = all_tickers(config)
    print(f"Downloading {len(tickers)} series from {start} to {end} ...")
    prices = _normalise_prices(
        yf.download(tickers, start=start, end=end, auto_adjust=False,
                    group_by="column", progress=False)
    )
    print(f"  prices: {len(prices):,} rows, {prices['Ticker'].nunique()} tickers, "
          f"{prices['Date'].min().date()} -> {prices['Date'].max().date()}")

    print(f"Downloading FRED series {config['fred_series']} ...")
    macro = web.DataReader(config["fred_series"], "fred", start, end).reset_index()
    macro = macro.rename(columns={macro.columns[0]: "Date"}).sort_values("Date").reset_index(drop=True)
    print(f"  macro: {len(macro):,} rows, {macro.shape[1] - 1} series")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    price_tmp, macro_tmp = RAW_DIR / "prices.tmp.parquet", RAW_DIR / "macro.tmp.parquet"
    prices.to_parquet(price_tmp, index=False)
    macro.to_parquet(macro_tmp, index=False)
    price_tmp.replace(RAW_DIR / "prices.parquet")
    macro_tmp.replace(RAW_DIR / "macro.parquet")
    return RAW_DIR / "prices.parquet", RAW_DIR / "macro.parquet"


if __name__ == "__main__":
    written = ingest_data()
    print("")
    for p in written:
        print(f"Wrote {p}  ({p.stat().st_size / 1e6:.2f} MB)")
