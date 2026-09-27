"""GDELT news provider for the sentiment pipeline.

Why this exists
---------------
The PRD's Section 10 sentiment work needs timestamped financial news, and
`src/sentiment.py` is wired to Finnhub, which needs an API key. This module
exists so the with-vs-without ablation can actually be *measured* in an
environment that has no key, rather than being reported NOT RUN forever.

GDELT's DOC 2.0 API is free, needs no registration, and has a historical
archive. That makes it the only no-key source capable of supporting a
chronological train/test split - a source like yfinance's `.news`, which was
checked and rejected, returns roughly ten items per ticker covering the last
one or two days and nothing else, so it cannot support a split at all.

This is a **disclosed deviation**, in the same category as the PyPortfolioOpt
substitution: the PRD implies Finnhub, and Finnhub remains the preferred
provider whenever `FINNHUB_API_KEY` is set. `fetch_news` in `sentiment.py`
prefers Finnhub and only falls through to here when no key is present. The
report states which source produced the numbers.

GDELT's quirks, all found by testing rather than by reading documentation
---------------------------------------------------------------------
`sourcetype:financial` is **not** a valid DOC 2.0 operator. Sending it makes
GDELT reject the whole query with

    One or more of your keywords were too short, too long or too common

and, because the rejection is returned as HTTP 200 with a plain-text body, a
client that assumes JSON either crashes or — worse — reads the error string as
"zero articles" and concludes there is no news. Malformed queries also appear to
draw heavier throttling, so this one mistake caused a cascade of misleading
429s during development. The only content filters used here are
`sourcelang:english` and, optionally, a `domain:` allow-list, both of which are
valid.

English filtering is not optional. A bare `"Apple Inc"` query returns a large
fraction of non-English articles — the development probe came back largely in
Chinese and Japanese. VADER is an English-lexicon scorer, so those articles score
as neutral noise rather than as garbage to be discarded, which would quietly
dilute every sentiment aggregate with unmeasurable text.

Rate limiting is aggressive: roughly one request every 15-20 seconds is safe,
and bursts earn a 429. Each request therefore sleeps between calls, and a 429
triggers exponential backoff. The fetch is slow by design; that is the price of
the only keyless archive available.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import timedelta

import pandas as pd

from src.common import PROCESSED_DIR, get_logger

LOG = get_logger("gdelt")

ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"
_UA = {"User-Agent": "Mozilla/5.0 (northgate-repro; research use)"}

# GDELT DOC 2.0 keeps a rolling archive; anything older is simply absent, and
# asking for it returns an error rather than an empty result.
ARCHIVE_START = "20170309000000"


def _http_get_json(url: str, throttle: float, max_tries: int = 5) -> dict | None:
    """GET with throttle + 429 backoff. Returns None if GDELT refuses us.

    A plain-text body is treated as an error, never as an empty result. That
    distinction matters: GDELT signals a rejected query with HTTP 200 and prose,
    so a client that only checks the status code will happily record "no news
    found" and then report a meaningless sentiment ablation.

    The backoff is exponential but CAPPED. An uncapped `throttle * 2**attempt`
    reaches roughly 19 minutes on the fifth attempt, so one request that keeps
    drawing 429s can consume the entire budget of a 60-request fetch and still
    return nothing. Capping at 120s keeps a run bounded: it finishes, and the
    chunks that were refused are reported as refused rather than silently
    becoming "no news in that period".
    """
    for attempt in range(1, max_tries + 1):
        if throttle:
            time.sleep(throttle)
        try:
            with urllib.request.urlopen(
                    urllib.request.Request(url, headers=_UA), timeout=60) as resp:
                body = resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 503):
                wait = min(throttle * (2 ** attempt), 120.0)
                LOG.info("    throttled (HTTP %s), backing off %.0fs [try %d/%d]",
                         exc.code, wait, attempt, max_tries)
                time.sleep(wait)
                continue
            LOG.warning("    HTTP %s from GDELT", exc.code)
            return None
        except Exception as exc:  # noqa: BLE001
            LOG.warning("    %s: %s", type(exc).__name__, exc)
            return None

        stripped = body.lstrip()
        if not stripped.startswith("{"):
            # GDELT answers a malformed or too-broad query with prose, not JSON.
            LOG.warning("    GDELT returned prose, not JSON: %s",
                        stripped[:120].replace("\n", " "))
            return None
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            LOG.warning("    GDELT returned unparseable JSON")
            return None
    LOG.warning("    exhausted %d attempts against GDELT", max_tries)
    return None


def _chunks(start: pd.Timestamp, end: pd.Timestamp, days: int):
    cur = start
    while cur <= end:
        stop = min(cur + timedelta(days=days - 1), end)
        yield cur, stop
        cur = stop + timedelta(days=1)


def fetch_news_gdelt(universe: list[str], start: str, end: str,
                     cfg: dict) -> pd.DataFrame:
    """Fetch English-language company news per ticker from GDELT.

    Returns the same schema as the Finnhub provider so the rest of the
    pipeline cannot tell which source it came from: ``Ticker``, ``Timestamp``,
    ``Headline``, ``Summary``.

    The window is deliberately clamped to `gdelt.lookback_days`. GDELT is not
    instantaneous, so fetching the project's full 2016-2026 history would mean
    ~31 chunks per ticker, roughly 300 rate-limited requests and well over an
    hour. The ablation is a *held-out window* measurement, so history older
    than the test window is fetched, paid for, and then discarded. The clamped
    window is returned in the log rather than left implicit, because a reader
    comparing the sentiment coverage against the price coverage needs to know
    they differ.
    """
    gcfg = cfg.get("sentiment", {}).get("gdelt", {}) or {}
    queries = gcfg.get("queries", {}) or {}
    chunk_days = int(gcfg.get("chunk_days", 120))
    throttle = float(gcfg.get("throttle_seconds", 18))
    max_records = int(gcfg.get("max_records", 250))
    lookback = int(gcfg.get("lookback_days", 0) or 0)

    s = pd.Timestamp(start).floor("D")
    e = pd.Timestamp(end).floor("D")
    requested_start = s
    if lookback > 0:
        clamped = e - pd.Timedelta(days=lookback)
        if clamped > s:
            LOG.info("Clamping fetch window to the last %d days: %s -> %s",
                     lookback, clamped.date(), e.date())
            LOG.info("(the project starts %s; older history is not fetched)",
                     requested_start.date())
            s = clamped

    windows = list(_chunks(s, e, chunk_days))
    LOG.info("GDELT: %d tickers x %d window(s) of ~%d days = %d requests, "
             "%ds apart (about %.0f min)",
             len(universe), len(windows), chunk_days,
             len(universe) * len(windows), throttle,
             len(universe) * len(windows) * throttle / 60)

    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    total_requests = 0
    refused = 0
    from_cache = 0
    t_fetch = time.time()
    deadline = t_fetch + float(gcfg.get("max_total_seconds", 0) or 0)

    # Chunk-level cache. GDELT's throttling is aggressive enough that a fetch
    # of this size will draw 429s partway through, and without a cache every
    # retry restarts from zero and may never finish: one run spent four minutes
    # backing off and produced nothing, and a full retry cycle here returned
    # 0 of 60 chunks because the service had stopped answering this IP at all.
    # With a cache, each attempt keeps whatever it managed to fetch, so repeated
    # attempts accumulate coverage instead of replacing it.
    cache_dir = PROCESSED_DIR / "gdelt_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    for ticker in universe:
        term = queries.get(ticker, f'"{ticker}"')
        got = 0
        for w0, w1 in windows:
            cache = cache_dir / f"{ticker}_{w0.strftime('%Y%m%d')}.json"
            if cache.exists():
                try:
                    cached = json.loads(cache.read_text(encoding="utf-8"))
                    arts = cached.get("articles", [])
                    from_cache += 1
                except json.JSONDecodeError:
                    arts, cached = None, None
            else:
                arts = None

            if arts is None:
                # A wall-clock deadline, not a request count. When GDELT stops
                # answering entirely, a per-request retry budget still means
                # waiting out every backoff of every chunk; a deadline stops the
                # run and reports what it has.
                if deadline and time.time() > deadline:
                    LOG.warning("")
                    LOG.warning("GDELT fetch deadline of %.0fs reached after %d "
                                "requests. Stopping with %d articles from %d "
                                "cached chunks. Re-running resumes from the "
                                "cache.",
                                gcfg.get("max_total_seconds"), total_requests,
                                len(rows), from_cache)
                    break
                q = f'{term} sourcelang:english'
                url = (f"{ENDPOINT}?query={urllib.parse.quote(q)}"
                       f"&mode=artlist&maxrecords={max_records}&format=json"
                       f"&startdatetime={w0.strftime('%Y%m%d000000')}"
                       f"&enddatetime={w1.strftime('%Y%m%d235959')}")
                payload = _http_get_json(url, throttle)
                total_requests += 1
                if payload is None:
                    refused += 1
                    continue
                arts = payload.get("articles", []) or []
                cache.write_text(json.dumps(
                    {"ticker": ticker, "start": str(w0.date()),
                     "end": str(w1.date()), "articles": arts},
                    ensure_ascii=False), encoding="utf-8")

            for art in arts:
                title = (art.get("title") or "").strip()
                if not title:
                    continue
                seen_key = (ticker, art.get("url") or title)
                if seen_key in seen:
                    continue
                seen.add(seen_key)
                seen_date = str(art.get("seendate") or "")
                ts = pd.to_datetime(seen_date, format="%Y%m%dT%H%M%SZ",
                                    errors="coerce")
                if pd.isna(ts):
                    continue
                rows.append({
                    "Ticker": ticker,
                    "Timestamp": ts,
                    "Headline": title,
                    "Summary": "",
                })
                got += 1
        LOG.info("  %-6s TOTAL %4d articles  (%.0f%% through the fetch)",
                 ticker, got,
                 100.0 * (universe.index(ticker) + 1) / len(universe))

    frame = pd.DataFrame(rows, columns=["Ticker", "Timestamp", "Headline", "Summary"])
    if not frame.empty:
        frame = frame.drop_duplicates(subset=["Ticker", "Timestamp", "Headline"])
        frame = frame.sort_values(["Ticker", "Timestamp"]).reset_index(drop=True)
    LOG.info("GDELT: %d articles from %d live requests, %d cached chunks, "
             "%d refused, in %.1f min",
             len(frame), total_requests, from_cache, refused,
             (time.time() - t_fetch) / 60)
    if not frame.empty:
        span = frame["Timestamp"].agg(["min", "max"])
        per_ticker = frame.groupby("Ticker")["Timestamp"].nunique()
        LOG.info("GDELT coverage: %s .. %s", span["min"].date(), span["max"].date())
        LOG.info("GDELT distinct news-days per ticker (min %d, median %d, max %d)",
                 int(per_ticker.min()), int(per_ticker.median()),
                 int(per_ticker.max()))
    return frame
