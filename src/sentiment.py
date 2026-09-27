"""PRD Section 10 - Sentiment Analysis Module.

Pipeline (Section 10.1)
-----------------------
1. Ingest timestamped headlines/summaries per ticker and for the market.
2. Clean text: lowercase, strip URLs and tickers, expand finance abbreviations.
3. Score with FinBERT (finance-tuned) and, as a comparison, VADER (lexicon).
4. Aggregate to a daily score per asset; **map after-close news to the next
   trading session** so an 18:00 headline cannot inform a 16:00 forecast.
5. Derive daily sentiment, 3-session momentum, and a news-volume spike flag.

Look-ahead guard (Section 10.3)
-------------------------------
"News published at 18:00 cannot inform a forecast for that same 16:00 close.
The timestamp-to-session mapping is the correctness crux of this entire module."
The mapping therefore uses the actual exchange calendar, not "tomorrow's date" -
a Saturday 18:00 headline must land on Monday, and a pre-close headline must stay
on its own session.

Data availability caveat (documented, not hidden)
-------------------------------------------------
The configured free news tier does not provide a decade of history, while the
price panel spans 2015-2026. Sentiment features are therefore only populated
over the window the provider actually covers, and the with/without-sentiment
ablation in :func:`run_ablation` is evaluated **on that overlapping sub-period
only**. The overlap is reported explicitly rather than the gap being papered
over.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import (  # noqa: E402
    CLEANED_PANEL,
    FEATURES,
    MARKET_MOOD,
    SENTIMENT_ABLATION,
    SENTIMENT_FEATURES,
    banner,
    ensure_dirs,
    get_logger,
    load_config,
    resolve_end_date,
    set_seed,
    write_json,
)
from src.features import split_by_date  # noqa: E402

LOG = get_logger("sentiment")

URL_RE = re.compile(r"https?://\S+|www\.\S+")
TICKER_RE = re.compile(r"[$]([A-Z]{1,5})\b")
HTML_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")

# Common finance abbreviations expanded before scoring (Section 10.1).
ABBREVIATIONS = {
    r"\bup\b": "up", r"\bdown\b": "down",
    r"\bvs\.?\b": "versus", r"\bvs\b": "versus",
    r"\betc\.?\b": "et cetera", r"\bapprox\.?\b": "approximately",
    r"\bco\.?\b": "company", r"\binc\.?\b": "incorporated",
    r"\bcorp\.?\b": "corporation", r"\bdept\.?\b": "department",
    r"\bearnings y/y\b": "earnings year over year",
    r"\bq/q\b": "quarter over quarter", r"\by/y\b": "year over year",
    r"\bq[1-4]\b": "quarter", r"\bfy\d{2,4}\b": "fiscal year",
    r"\bebitda\b": "earnings before interest taxes depreciation amortization",
    r"\bpe\b": "price earnings", r"\beps\b": "earnings per share",
    r"\bguidance\b": "outlook guidance", r"\bbeat\b": "exceeded expectations",
    r"\bmiss(?:ed)?\b": "fell short of expectations",
    r"\bbuyback\b": "share repurchase", r"\bdividend\b": "dividend",
}


# ==========================================================================
# Text cleaning (Section 10.1)
# ==========================================================================
def clean_text(text: str) -> str:
    """Lowercase, strip URLs/HTML/tickers, expand abbreviations, collapse space."""
    if not isinstance(text, str) or not text.strip():
        return ""
    out = HTML_RE.sub(" ", text)
    out = URL_RE.sub(" ", out)
    out = TICKER_RE.sub(r"\1", out)
    out = out.lower()
    for pattern, replacement in ABBREVIATIONS.items():
        out = re.sub(pattern, replacement, out)
    return WS_RE.sub(" ", out).strip()


# ==========================================================================
# Session mapping (Section 10.3) - the correctness crux
# ==========================================================================
def load_trading_calendar() -> pd.DatetimeIndex:
    """The exchange calendar, taken from the cleaned panel."""
    cfg = load_config()
    panel = pd.read_parquet(CLEANED_PANEL)
    panel["Date"] = pd.to_datetime(panel["Date"])
    days = panel.loc[panel["Ticker"] == cfg["benchmark"], "Date"]
    return pd.DatetimeIndex(sorted(days.unique()))


def map_to_session(timestamps: pd.Series, calendar: pd.DatetimeIndex,
                   close_hour: int) -> pd.Series:
    """Assign each article the trading session it can legitimately inform.

    * Published **at or after** the close -> the *next* trading session.
    * Published before the close -> the *same day's* session, provided that day
      is a trading day; otherwise the next one.
    * A weekend/holiday publication -> the next trading session.

    Implemented with ``searchsorted`` against the real calendar so a Saturday
    headline lands on Monday rather than on a non-existent Saturday session.
    """
    cal = pd.DatetimeIndex(calendar).normalize()
    ts = pd.to_datetime(timestamps)
    dates = ts.dt.normalize()
    after_close = ts.dt.hour >= close_hour

    if len(cal) == 0:
        raise ValueError("trading calendar is empty; cannot map timestamps to sessions")

    # `searchsorted(side="left")` returns the index at which the date would be
    # inserted, so for a date that IS a session, pos is that session's own index,
    # and for a date that is NOT a session, pos is the next session's index.
    pos = cal.searchsorted(dates, side="left")
    is_session = (pos < len(cal)) & (cal[np.clip(pos, 0, len(cal) - 1)] == dates)

    # Step forward by one session only when the article landed on a real trading
    # day AND was published at or after the close:
    #   after close on a session      -> the following session
    #   after close on a non-session  -> the next session (pos already points there)
    #   before close on a session     -> its own session
    #   before close on a non-session -> the next session
    step = (after_close & is_session).astype(int).to_numpy()
    nxt = np.clip(pos + step, 0, len(cal) - 1)

    return pd.Series(cal[nxt], index=ts.index)


# ==========================================================================
# News ingestion
# ==========================================================================
def fetch_news_finnhub(universe: list[str], start: str, end: str,
                       api_key: str) -> pd.DataFrame:
    """Company news per ticker from the Finnhub free tier."""
    import finnhub

    client = finnhub.Client(api_key=api_key)
    rows = []
    s = pd.Timestamp(start).strftime("%Y-%m-%d")
    e = pd.Timestamp(end).strftime("%Y-%m-%d")

    for ticker in universe:
        try:
            res = client.company_news(ticker, _from=s, to=e)
        except Exception as exc:
            LOG.warning("  %-6s fetch failed: %s", ticker, exc)
            continue
        n = 0
        for art in res or []:
            headline = (art.get("headline") or "").strip()
            summary = (art.get("summary") or "").strip()
            if not headline and not summary:
                continue
            rows.append({
                "Ticker": ticker,
                "Timestamp": pd.to_datetime(art.get("datetime"), unit="s"),
                "Headline": headline,
                "Summary": summary,
            })
            n += 1
        LOG.info("  %-6s %4d articles", ticker, n)
    return pd.DataFrame(rows)


def fetch_news(universe: list[str], start: str, end: str, cfg: dict) -> pd.DataFrame:
    """Dispatch to the configured provider (Section 10.1 step 1).

    With `provider: auto`, Finnhub is preferred whenever an API key is present,
    because it is the provider the PRD implies. The GDELT fallback exists so
    the with/without ablation in `run_ablation` can be *measured* rather than
    reported NOT RUN forever: GDELT needs no key and has a real archive, which
    yfinance's news feed does not. Whichever source is used is recorded in the
    sentiment output so the report can name it - an ablation run on a different
    corpus than the report implies would be a quiet misrepresentation.
    """
    scfg = cfg["sentiment"]
    provider = scfg.get("provider", "auto")
    api_key = os.environ.get(scfg.get("api_key_env", "FINNHUB_API_KEY"), "").strip()

    if api_key and provider in ("finnhub", "auto"):
        LOG.info("Provider: finnhub (API key present)")
        return fetch_news_finnhub(universe, start, end, api_key)

    if provider == "finnhub":
        # Explicitly pinned to Finnhub and no key: fail loudly rather than
        # silently substituting a different corpus.
        LOG.error("")
        LOG.error("provider is pinned to `finnhub` but %s is not set.",
                  scfg.get("api_key_env"))
        return pd.DataFrame(columns=["Ticker", "Timestamp", "Headline", "Summary"])

    if provider in ("gdelt", "auto"):
        from src.sentiment_gdelt import fetch_news_gdelt

        if not api_key:
            LOG.warning("")
            LOG.warning("No %s set - falling back to GDELT, which needs no key.",
                        scfg.get("api_key_env"))
            LOG.warning("This is a DEVIATION from the PRD's implied news source.")
            LOG.warning("GDELT is a global wire-to-web index, not a curated")
            LOG.warning("financial newswire, so coverage and relevance are lower.")
        LOG.info("Provider: gdelt (no API key required)")
        return fetch_news_gdelt(universe, start, end, cfg)

    raise ValueError(f"unsupported sentiment provider: {provider!r}")


# ==========================================================================
# Scoring (Section 10.2)
# ==========================================================================
def score_vader(texts: list[str]) -> np.ndarray:
    """Lexicon scorer; returns compound polarity in [-1, 1]."""
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
    sia = SentimentIntensityAnalyzer()
    return np.array([sia.polarity_scores(t)["compound"] if t else 0.0 for t in texts])


def score_finbert(texts: list[str], model_name: str, batch_size: int) -> np.ndarray:
    """Finance-tuned transformer; polarity in [-1, 1].

    Section 10.2: a generic sentiment model mislabels finance text
    ("debt falls" is positive), which is why FinBERT is specified.
    """
    from transformers import pipeline

    classifier = pipeline("sentiment-analysis", model=model_name,
                         truncation=True, batch_size=batch_size)
    weights = {"positive": 1.0, "neutral": 0.0, "negative": -1.0, "LABEL_0": -1.0, "LABEL_1": 0.0, "LABEL_2": 1.0}
    out = np.zeros(len(texts), dtype=float)
    for i in range(0, len(texts), batch_size):
        chunk = texts[i: i + batch_size]
        try:
            results = classifier(chunk)
        except Exception as exc:
            LOG.warning("  FinBERT batch %d-%d failed: %s", i, i + len(chunk), exc)
            continue
        for j, r in enumerate(results):
            label = r.get("label", "")
            out[i + j] = weights.get(label, 0.0) * float(r.get("score", 0.0))
    return out


# ==========================================================================
# Aggregation and derived features (Section 10.1 steps 4-5)
# ==========================================================================
def aggregate_daily(news: pd.DataFrame, calendar: pd.DatetimeIndex,
                    cfg: dict) -> pd.DataFrame:
    """Daily sentiment per ticker, plus momentum and news-volume spike."""
    scfg = cfg["sentiment"]
    close_hour = int(scfg.get("market_close_hour", 16))

    news = news.copy()
    news["Date"] = map_to_session(news["Timestamp"], calendar, close_hour)
    LOG.info("Mapped %d articles onto %d trading sessions (%s -> %s)",
             len(news), news["Date"].nunique(),
             news["Date"].min().date(), news["Date"].max().date())

    daily = (
        news.groupby(["Date", "Ticker"])
        .agg(
            Sentiment_FinBERT=("Sentiment_FinBERT", "mean"),
            Sentiment_VADER=("Sentiment_VADER", "mean"),
            News_Volume=("Text", "count"),
        )
        .reset_index()
        .sort_values(["Ticker", "Date"])
    )

    # 3-session sentiment momentum (Section 10.1 step 5). The rolling window is
    # over the article rows per ticker, which are already session-aligned.
    mom_w = int(scfg.get("momentum_window", 3))
    vol_w = int(scfg.get("news_volume_window", 7))
    daily["Sentiment_3d_Mom"] = (
        daily.groupby("Ticker")["Sentiment_FinBERT"]
        .transform(lambda s: s.rolling(mom_w, min_periods=1).mean())
    )
    daily["Vol_Avg"] = (
        daily.groupby("Ticker")["News_Volume"]
        .transform(lambda s: s.rolling(vol_w, min_periods=1).mean())
    )
    daily["News_Spike"] = np.where(
        daily["Vol_Avg"] > 0, daily["News_Volume"] / daily["Vol_Avg"].replace(0, np.nan), 1.0
    )
    daily["News_Spike"] = daily["News_Spike"].fillna(1.0)

    return daily.reset_index(drop=True)


def build_market_mood(daily: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Universe-wide bullish / neutral / bearish gauge (Section 10.3)."""
    scfg = cfg["sentiment"]
    bull = float(scfg.get("mood_bullish_threshold", 0.15))
    bear = float(scfg.get("mood_bearish_threshold", -0.15))

    mood = (
        daily.groupby("Date")
        .agg(
            Mean_Sentiment=("Sentiment_FinBERT", "mean"),
            Article_Count=("News_Volume", "sum"),
            Tickers_Covered=("Ticker", "nunique"),
        )
        .reset_index()
    )

    def label(v: float) -> str:
        if v > bull:
            return "Bullish"
        if v < bear:
            return "Bearish"
        return "Neutral"

    mood["Market_Mood"] = mood["Mean_Sentiment"].map(label)
    return mood


# ==========================================================================
# Ablation (Section 10.3 integration test)
# ==========================================================================
def run_ablation_selftest() -> dict:
    """Prove the ablation harness works, using a synthetic sentiment block.

    Why this exists. The ablation cannot run for real without a news API key,
    and an untested experiment that is merely *written* is not evidence that it
    will work. If the harness has a bug in its split, its feature selection, or
    its delta arithmetic, that bug would surface only on the day a key is
    supplied - and the number it produced would then be published.

    So the harness is exercised here on a block of **synthetic** sentiment that
    is built to be genuinely predictive of the target. A working harness must
    detect that the with-sentiment arm is better, because it is. If it does not,
    the harness is broken and this function says so.

    What this is NOT: a sentiment result. The synthetic block is generated, not
    fetched from news, so it says nothing about whether real headlines help. The
    return value says so in a field the report and dashboard both read, and the
    real ablation remains NOT RUN until a key exists.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error

    from src.features import feature_columns

    if not FEATURES.exists():
        return {"status": "NOT RUN", "reason": "features.parquet missing"}

    df = pd.read_parquet(FEATURES)
    df["Date"] = pd.to_datetime(df["Date"])
    base_cols = feature_columns(df)

    LOG.info("")
    LOG.info("")
    LOG.info("Ablation HARNESS SELF-TEST (synthetic sentiment - not a result)")
    LOG.info("-" * 74)

    # Restrict to the test window so the synthetic block is large enough to
    # train on, and build a sentiment feature that carries real signal: a
    # noisy function of the target, i.e. something a competent model should be
    # able to exploit. A harness that cannot find this is broken.
    bounds = split_by_date(df["Date"], load_config())
    dates = np.sort(bounds["test"])
    if len(dates) < 60:
        return {"status": "NOT RUN",
                "reason": f"only {len(dates)} test sessions available"}
    work = df[df["Date"].isin(dates)].sort_values(["Date", "Ticker"]).copy()

    rng = np.random.default_rng(load_config().get("random_seed", 42))
    y = work["Target"].to_numpy(dtype=float)
    # 70% signal, 30% noise. Correlated with the target, not a copy of it.
    work["Sentiment_Synthetic"] = 0.7 * y + 0.3 * rng.normal(scale=y.std(), size=len(y))
    sent_cols = ["Sentiment_Synthetic"]

    n = work["Date"].nunique()
    cut = int(n * 0.70)
    tr = work[work["Date"].isin(dates[:cut])]
    te = work[work["Date"].isin(dates[cut:])]

    arms = {}
    for arm, use in (("without_sentiment", base_cols),
                     ("with_sentiment", base_cols + sent_cols)):
        used = [c for c in use if c in work.columns]
        Xtr = tr[used].to_numpy(dtype=float)
        Xte = te[used].to_numpy(dtype=float)
        ytr = tr["Target"].to_numpy(dtype=float)
        yte = te["Target"].to_numpy(dtype=float)
        model = HistGradientBoostingRegressor(max_iter=300, random_state=42)
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        mae = float(mean_absolute_error(yte, pred))
        mask = yte != 0
        dir_acc = float(np.mean(np.sign(yte[mask]) == np.sign(pred[mask])))
        arms[arm] = {"MAE": mae, "DirAcc": dir_acc, "n_features": len(used),
                     "n_test": int(len(yte))}
        LOG.info("  %-20s MAE=%.6f  DirAcc=%.2f%%  (%d features)",
                 arm, mae, 100 * dir_acc, len(used))

    d_mae = arms["with_sentiment"]["MAE"] - arms["without_sentiment"]["MAE"]
    d_dir = arms["with_sentiment"]["DirAcc"] - arms["without_sentiment"]["DirAcc"]
    # The harness works if it recovers the injected signal.
    detected = d_mae < 0 and d_dir > 0
    LOG.info("  delta MAE     : %+.6f (%s)", d_mae, "better" if d_mae < 0 else "worse")
    LOG.info("  delta Dir.Acc : %+.2f pp", 100 * d_dir)
    LOG.info("  HARNESS %s", "OK - it recovers an injected signal"
             if detected else "BROKEN - it failed to recover an injected signal")
    LOG.info("-" * 74)

    return {
        "status": "SELFTEST PASSED" if detected else "SELFTEST FAILED",
        "is_a_sentiment_result": False,
        "what_this_is": (
            "A validation of the ablation harness, not a measurement of "
            "sentiment. The sentiment column here is SYNTHETIC - generated from "
            "the target with added noise - and carries no information about news. "
            "It exists so the with/without comparison, the chronological split "
            "and the delta arithmetic are proven to work before anyone relies on "
            "a real number."
        ),
        "overlap_sessions": int(n),
        "train_rows": int(len(tr)),
        "test_rows": int(len(te)),
        "arms": arms,
        "delta_MAE": float(d_mae),
        "delta_DirAcc": float(d_dir),
        "harness_detected_injected_signal": bool(detected),
        "real_ablation_status": "NOT RUN - no news API key",
    }


def run_ablation(daily: pd.DataFrame) -> dict:
    """Retrain the best forecaster with and without sentiment; report the delta.

    Section 10.3: "The candidate must state honestly whether sentiment helped on
    this dataset." The comparison is restricted to the sessions for which
    sentiment actually exists, so the two arms see identical rows.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error

    from src.features import feature_columns

    if daily is None or daily.empty:
        return {"status": "NOT RUN", "reason": "no sentiment features available"}

    df = pd.read_parquet(FEATURES)
    df["Date"] = pd.to_datetime(df["Date"])
    cols = feature_columns(df)
    sent_cols = [c for c in ("Sentiment_FinBERT", "Sentiment_VADER",
                             "Sentiment_3d_Mom", "News_Spike") if c in df.columns]
    if not sent_cols:
        return {"status": "NOT RUN", "reason": "sentiment columns absent from features.parquet"}

    overlap = df[df["Date"].isin(daily["Date"])].sort_values(["Date", "Ticker"])
    if overlap["Date"].nunique() < 60:
        return {"status": "NOT RUN",
                "reason": f"only {overlap['Date'].nunique()} overlapping sessions "
                          f"(minimum 60 required for a chronological split)"}

    dates = np.sort(overlap["Date"].unique())
    n = len(dates)
    cut = int(n * 0.70)
    train_dates, test_dates = set(dates[:cut]), set(dates[cut:])

    tr = overlap[overlap["Date"].isin(train_dates)]
    te = overlap[overlap["Date"].isin(test_dates)]
    LOG.info("")
    LOG.info("Sentiment ablation on the overlapping window:")
    LOG.info("  %d sessions (%s -> %s): %d train / %d test rows",
             n, pd.Timestamp(dates[0]).date(), pd.Timestamp(dates[-1]).date(),
             len(tr), len(te))

    results = {}
    for arm, use_cols in (("without_sentiment", cols), ("with_sentiment", cols + sent_cols)):
        cols_used = [c for c in use_cols if c in overlap.columns]
        Xtr = tr[cols_used].to_numpy(dtype=float)
        ytr = tr["Target"].to_numpy(dtype=float)
        Xte = te[cols_used].to_numpy(dtype=float)
        yte = te["Target"].to_numpy(dtype=float)

        model = HistGradientBoostingRegressor(max_iter=300, random_state=42)
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        mae = float(mean_absolute_error(yte, pred))
        mask = yte != 0
        dir_acc = float(np.mean(np.sign(yte[mask]) == np.sign(pred[mask])))
        results[arm] = {"MAE": mae, "DirAcc": dir_acc, "n_features": len(cols_used),
                        "n_test": int(len(yte))}
        LOG.info("  %-20s MAE=%.6f  DirAcc=%.2f%%  (%d features)",
                 arm, mae, 100 * dir_acc, len(cols_used))

    d_mae = results["with_sentiment"]["MAE"] - results["without_sentiment"]["MAE"]
    d_dir = results["with_sentiment"]["DirAcc"] - results["without_sentiment"]["DirAcc"]
    helped = d_mae < 0
    verdict = (
        "Sentiment improved MAE and directional accuracy on this window."
        if helped and d_dir > 0 else
        "Sentiment improved MAE but did not improve directional accuracy."
        if helped else
        "Sentiment did NOT improve MAE on this window."
    )
    LOG.info("  delta MAE     : %+.6f (%s)", d_mae, "better" if d_mae < 0 else "worse")
    LOG.info("  delta Dir.Acc : %+.2f pp", 100 * d_dir)
    LOG.info("  Verdict: %s", verdict)

    return {
        "status": "COMPLETED",
        "overlap_sessions": int(n),
        "overlap_start": str(pd.Timestamp(dates[0]).date()),
        "overlap_end": str(pd.Timestamp(dates[-1]).date()),
        "train_rows": int(len(tr)),
        "test_rows": int(len(te)),
        "sentiment_columns": sent_cols,
        "arms": results,
        "delta_MAE": float(d_mae),
        "delta_DirAcc": float(d_dir),
        "sentiment_helped": bool(helped),
        "verdict": verdict,
        "scope_caveat": (
            "Evaluated only on the sessions for which the news provider returned "
            "coverage, not the full 2015-2026 price history. The free news tier does "
            "not supply a decade of headlines."
        ),
    }


# ==========================================================================
# Main
# ==========================================================================
def analyze_sentiment() -> dict:
    ensure_dirs()
    cfg = load_config()
    set_seed(cfg.get("random_seed", 42))
    scfg = cfg["sentiment"]

    banner(LOG, "PRD Section 10 - Sentiment Analysis")

    universe = list(cfg["universe"])
    calendar = load_trading_calendar()
    LOG.info("Trading calendar: %d sessions", len(calendar))
    LOG.info("Provider: %s (key from $%s)", scfg.get("provider"), scfg.get("api_key_env"))

    news = fetch_news(universe, cfg["start_date"], resolve_end_date(cfg), cfg)

    if news.empty:
        LOG.warning("")
        LOG.warning("No news retrieved. Writing no sentiment artifacts.")
        LOG.warning("The rest of the pipeline runs without the Section 5.1 sentiment")
        LOG.warning("family; run this again with an API key to populate it.")
        ablation = run_ablation(pd.DataFrame())
        write_json(ablation, SENTIMENT_ABLATION)
        return {"status": "NO_NEWS", "ablation": ablation}

    LOG.info("")
    LOG.info("Retrieved %d articles", len(news))

    # -- clean and score ---------------------------------------------------
    news["Text"] = (news["Headline"].fillna("") + ". " + news["Summary"].fillna("")).str.strip()
    news["Text"] = news["Text"].map(clean_text)
    usable = news["Text"].str.len() > 0
    LOG.info("Usable after cleaning: %d of %d", int(usable.sum()), len(news))
    news = news[usable].copy()

    LOG.info("Scoring with VADER (lexicon)...")
    news["Sentiment_VADER"] = score_vader(news["Text"].tolist())
    LOG.info("  mean compound polarity: %+.4f", float(news["Sentiment_VADER"].mean()))

    LOG.info("Scoring with FinBERT (%s)...", scfg.get("finbert_model"))
    try:
        news["Sentiment_FinBERT"] = score_finbert(
            news["Text"].tolist(), scfg["finbert_model"], int(scfg.get("finbert_batch_size", 32))
        )
        LOG.info("  mean polarity: %+.4f", float(news["Sentiment_FinBERT"].mean()))
    except Exception as exc:
        LOG.error("FinBERT unavailable (%s). Falling back to VADER for the primary score.", exc)
        news["Sentiment_FinBERT"] = news["Sentiment_VADER"]

    # -- aggregate ---------------------------------------------------------
    daily = aggregate_daily(news, calendar, cfg)
    mood = build_market_mood(daily, cfg)

    daily.to_parquet(SENTIMENT_FEATURES, index=False)
    mood.to_parquet(MARKET_MOOD, index=False)

    LOG.info("")
    LOG.info("Sentiment features -> %s", SENTIMENT_FEATURES)
    LOG.info("Market mood        -> %s", MARKET_MOOD)
    LOG.info("Coverage: %d sessions (%s -> %s), %d tickers",
             daily["Date"].nunique(), daily["Date"].min().date(),
             daily["Date"].max().date(), daily["Ticker"].nunique())
    LOG.info("Mood distribution: %s",
             dict(mood["Market_Mood"].value_counts()))

    # -- integration test --------------------------------------------------
    ablation = run_ablation(daily)
    # Record which corpus produced the measurement. The ablation is only
    # interpretable alongside its source: a GDELT web-index result and a
    # Finnhub newswire result answer the same question about different data, and
    # a reader who cannot see which was used will over-read whichever number
    # they find. Stated here so it travels with the artifact.
    provider = ("finnhub" if os.environ.get(
        scfg.get("api_key_env", "FINNHUB_API_KEY"), "").strip()
        else "gdelt")
    ablation["news_provider"] = provider
    ablation["news_provider_note"] = (
        "GDELT DOC 2.0, a keyless global wire-to-web index. It is a weaker "
        "proxy for financial news than the curated newswire the PRD implies, "
        "so a null result here is partly a statement about the corpus rather "
        "than only about the model. Set FINNHUB_API_KEY to re-run against the "
        "intended source."
        if provider == "gdelt" else
        "Finnhub company news, the provider the PRD implies.")
    write_json(ablation, SENTIMENT_ABLATION)
    LOG.info("")
    LOG.info("Ablation -> %s", SENTIMENT_ABLATION)

    return {
        "status": "OK",
        "n_articles": int(len(news)),
        "n_sessions": int(daily["Date"].nunique()),
        "coverage_start": str(daily["Date"].min().date()),
        "coverage_end": str(daily["Date"].max().date()),
        "ablation": ablation,
    }


if __name__ == "__main__":
    analyze_sentiment()
