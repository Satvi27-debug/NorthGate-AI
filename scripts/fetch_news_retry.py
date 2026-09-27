"""Drive the GDELT fetch to completion, surviving rate limiting.

GDELT returns HTTP 429 under sustained querying, and a single fetch of 60
requests reliably draws some. The fetcher caches per (ticker, chunk) so a run
that gets throttled keeps whatever it already retrieved. This driver simply
re-invokes the fetch until every chunk is cached or a patience limit is hit,
which turns a throttling problem into a few extra minutes of waiting.

Prints a coverage summary at the end so the caller knows what it actually got,
rather than assuming a full fetch.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Derived from this file's own location rather than hard-coded. An absolute user
# path in a committed script publishes the author's directory layout to everyone
# who clones the repository, and it breaks the moment the repo moves.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import PROCESSED_DIR, load_config, resolve_end_date  # noqa: E402
from src.sentiment import fetch_news  # noqa: E402

MAX_PASSES = 8
COOLDOWN = 90


def cached_chunks() -> int:
    d = PROCESSED_DIR / "gdelt_cache"
    return len(list(d.glob("*.json"))) if d.exists() else 0


def main() -> int:
    cfg = load_config()
    universe = list(cfg["universe"])
    gcfg = cfg["sentiment"]["gdelt"]
    want = len(universe) * len(range(0, 600, int(gcfg["chunk_days"])))

    news = None
    for p in range(1, MAX_PASSES + 1):
        have = cached_chunks()
        print(f"\n===== PASS {p}/{MAX_PASSES}  ({have} chunks cached) =====",
              flush=True)
        news = fetch_news(universe, cfg["start_date"], resolve_end_date(cfg), cfg)
        have = cached_chunks()
        print(f"  -> {have}/{want} chunks cached, {len(news)} articles",
              flush=True)
        if have >= want:
            print("  COMPLETE", flush=True)
            break
        if p < MAX_PASSES:
            print(f"  cooling down {COOLDOWN}s before retrying", flush=True)
            time.sleep(COOLDOWN)

    if news is None or news.empty:
        print("FETCH PRODUCED NO ARTICLES", flush=True)
        return 1

    news.to_parquet(PROCESSED_DIR / "news_raw.parquet", index=False)
    print(f"\nWROTE {len(news)} articles -> news_raw.parquet", flush=True)

    import pandas as pd
    n = news.copy()
    n["Date"] = pd.to_datetime(n["Timestamp"]).dt.normalize()
    print(f"  span      : {n['Date'].min().date()} .. {n['Date'].max().date()}")
    print(f"  news days : {n['Date'].nunique()} distinct calendar days")
    per = n.groupby("Ticker")["Date"].nunique()
    print(f"  per ticker: min {per.min()}, median {int(per.median())}, "
          f"max {per.max()} days")
    missing = sorted(set(universe) - set(per.index))
    if missing:
        print(f"  NO NEWS AT ALL for: {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
