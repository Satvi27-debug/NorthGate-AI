# Northgate AI Stock Predictor — TODO

Status legend: `[x]` implemented and verified · `[~]` implemented, needs an API
key or external data to exercise · `[ ]` not yet done

## Current status

All thirteen PRD sections are implemented and the pipeline runs end to end.
**77 tests pass.**

Both of the PRD's empirical acceptance bars are **not met**, and both are reported
as measured rather than engineered around:

- The four deep models beat the naive random walk on MAE (0.011682 vs 0.012360)
  but not on directional accuracy (53.37% best vs 53.93% baseline).
- The optimised portfolio returns a Sharpe of 1.357 against equal weight's 1.648
  on a disjoint out-of-sample window, with a bootstrap interval of
  [−0.536, 2.970] spanning zero.

The engineering requirements are met in full. `reports/final_report.md` contains
the full analysis.

## Governing requirements

- [x] PRD v1.0 treated as the source of truth
- [x] One-way data flow: `raw → cleaned panel → features → models → portfolio & recs → dashboard`
- [x] Python 3.11+ (developed and verified on 3.13)
- [x] All randomness seeded; randomness flows from `config.yaml: random_seed`
- [x] No magic numbers in code — every tunable lives in `config.yaml`
- [x] Strict chronological handling; no shuffling, no look-ahead
- [x] Educational research framing; not financial advice

## Phase 1 — Data foundation and EDA

### Section 3 Ingestion
- [x] `src/ingest.py` implemented
- [x] Documented universe: AAPL, MSFT, JPM, XOM, JNJ, PG, NVDA, KO, CAT, HD
- [x] Benchmark `^GSPC` and volatility index `^VIX`
- [x] Daily OHLCV + adjusted close via yfinance
- [x] DGS10, DGS3MO, CPIAUCSL, UNRATE via FRED
- [x] Atomic writes to `data/raw/prices.parquet` and `macro.parquet`
- [x] Deterministic and rerunnable; `end_date: null` resolves to today

### Section 4 Cleaning
- [x] Single ordered, idempotent pipeline
- [x] Stage 1 — all series aligned to one exchange trading calendar
- [x] Stage 1 — macro forward-filled onto the daily grid
- [x] Stage 2 — price gaps ≤3 sessions forward-filled; **never back-filled**
- [x] Stage 3 — duplicate `(Date, Ticker)` removed, conflicts logged
- [x] Stage 4 — rolling z-score **and** 1.5×IQR fence on daily log returns
- [x] Stage 4 — outliers **flagged, not deleted**; earnings-day moves retained
- [x] Stage 5 — OHLC invariant asserted; violations quarantined
- [x] Data-quality report shipped with every build (`reports/data_quality_report.md`)

### Section 5 Features
- [x] All features computed from data at or before time t
- [x] Adjusted close used for every return and the target
- [x] All nine Section 5.1 families implemented
- [x] RSI(14) and Bollinger band width to the documented definitions
- [x] Sequence windows of length 60, built inside each partition so no window
      crosses a train/test boundary
- [x] Scalers fit on the training split only; target scaler retained for inversion
- [x] Non-causal columns excluded from the feature set (`Outlier_IQR`, `Outlier`,
      `Target`, `Target_Price`, `Relative_Target`, `Rank_Target`)
- [x] Feature set more than doubled with range-based volatility, overnight/intraday
      decomposition, explicit reversal horizons, and cross-sectional context
      (breadth, dispersion, trailing-Beta idiosyncratic return, relative strength)
- [x] Exactly-collinear columns dropped automatically (no information lost, better
      conditioning): the realised count is rendered from the artifact, not typed
- [x] Causality proven by test, not asserted: truncation and future-corruption
      probes cover both the per-ticker builder and the cross-sectional context
- [x] Standing guard against a renamed label entering the feature set
      (any input at |r| > 0.99 with a return-valued label fails the build)

### Section 6 EDA
- [x] EDA notebook generated (`notebooks/eda.ipynb`) plus six figures
- [x] Trend, stationarity, correlation, volatility, seasonality, distribution
- [x] ADF on price vs returns, skewness, excess kurtosis
- [x] Every figure captioned with its finding and the decision it changed

## Phase 2 — Mathematics and classical ML

### Section 7 Mathematics
- [x] Returns, volatility, covariance, correlation, Beta, Sharpe, Sortino,
      max drawdown, Calmar, VaR, conditional VaR
- [x] Manual gradient-descent linear regression, **verified against analytical OLS**
- [x] Empirical VaR, t-distribution VaR, Jarque-Bera, QQ analysis
- [x] Block-bootstrap Sharpe confidence intervals, seeded
- [x] Verification artifact written to `reports/math_verification.json`

### Section 8 Classical ML
- [x] Ridge, Random Forest, XGBoost, SVR
- [x] Identical feature table and chronological protocol for all four
- [x] Expanding-window `TimeSeriesSplit`, scaler refit inside every fold
- [x] Naive random-walk baseline established
- [x] RMSE, MAE, MAPE, R², directional accuracy reported
- [x] Result compared honestly against the baseline, pass or fail

## Phase 3 — Deep learning and sentiment

### Section 9 Deep learning
- [x] LSTM, GRU, BiLSTM, Transformer as genuine sequence architectures
- [x] Chronological 70/15/15 split on the date axis
- [x] Early stopping with `restore_best_weights`, dropout, seeds fixed, `shuffle=False`
- [x] Huber loss
- [x] Train vs validation loss curves exported and diagnosed
- [x] Eight-model comparison table on one held-out test window

### Section 10 Sentiment
- [x] Provider selected: **Finnhub** (PRD permitted NewsAPI / Finnhub / GNews)
- [x] Timestamped headline and summary ingestion
- [x] Text cleaning: lowercase, URL/ticker stripping, abbreviation expansion
- [x] FinBERT scoring with VADER comparison
- [x] After-close news mapped to the **next trading session** on the real
      exchange calendar (weekends and holidays handled)
- [x] Daily sentiment, 3-session momentum, news-volume spike
- [x] Bullish / neutral / bearish market-mood index
- [x] With/without-sentiment ablation, scoped honestly to the overlapping window
- [~] **Needs `FINNHUB_API_KEY`** to retrieve articles. The module, the session
      mapping, the ablation harness and the dashboard panel are implemented and
      unit-tested; without a key the pipeline runs without the Section 5.1
      sentiment family and the ablation is reported as NOT RUN, not as a null result.

## Phase 4 — Optimisation, recommendations, dashboard, report

### Section 11 Portfolio optimisation
- [x] Modern Portfolio Theory implemented
- [x] Ledoit-Wolf covariance shrinkage
- [x] Efficient frontier, minimum-variance and maximum-Sharpe portfolios
- [x] Long-only plus a per-asset concentration cap, and an explicit
      feasibility check on the cap
- [x] 20,000 random long-only portfolios as an independent cross-check
- [x] Backtest against equal-weight and benchmark buy-and-hold
- [x] Sharpe, Sortino, annualised volatility, Beta, maximum drawdown reported
- [x] μ source documented and justified (blend of historical and model-implied)
- [x] Solved with SciPy SLSQP rather than PyPortfolioOpt — see README for why

### Section 12 Recommendations
- [x] Transparent composite of forecast, sentiment and risk
- [x] Rule-based BUY / HOLD / SELL with a no-trade band
- [x] Rebalancing actions from current versus target weights
- [x] Contributing sub-signals shown with every recommendation
- [x] Educational-research and not-financial-advice disclaimer

### Section 13 Streamlit dashboard
- [x] Single application reading cached artifacts; never retrains on load
- [x] Plotly interactive charts with hover detail
- [x] Ticker, horizon and risk-free-rate controls
- [x] All seven panels: Overview, Price & prediction, Model comparison, Portfolio
      analytics, Risk dashboard, Sentiment, Recommendations
- [x] Disclaimer rendered on every panel
- [x] Honest-baseline verdict shown first on the model-comparison panel
- [x] Missing artifacts reported as unavailable rather than as zeros or stubs

### Section 14 Completion
- [x] One command rebuilds the dataset (`rebuild_dataset.py`)
- [x] One command retrains and re-evaluates all models (`retrain_models.py`)
- [x] Dashboard launches from cached outputs
- [x] Written report in `reports/final_report.md`
- [x] Layered test suite, 77 tests across three files:
      `tests/test_pipeline.py` (cleaning invariants, no-look-ahead, mathematical
      verification, portfolio constraints, recommendation thresholds, session
      mapping),
      `tests/test_dashboard.py` (panel presence, disclaimer per panel, no
      fabricated metrics, no training calls),
      `tests/test_panels.py` (executes every panel body against the real
      artifacts, asserts the backtest window is disjoint from estimation)
- [x] Dashboard verified to serve headlessly (`scripts/verify_dashboard.py`)
- [ ] Reproducibility check across a clean environment — needs a fresh venv run

## Known limitations

1. **Neither PRD acceptance bar is met.** Forecasting beats random walk on MAE
   but not direction; the portfolio does not beat equal weight. Both reported as
   measured, with the analysis in `reports/final_report.md` §6.3 and §8.5.
2. **Sentiment has no decade of history.** The free news tier covers roughly a
   year, so the ablation runs only on the overlapping window. Disclosed in the
   dashboard, the report and the ablation JSON. Without an API key it reports
   NOT RUN, not a null result.
3. **PyPortfolioOpt is unusable in this environment** (interpreter crash on
   import). Section 11 is implemented on SciPy SLSQP with the same mathematics.
4. **TensorFlow must be imported before pyarrow** or the interpreter aborts on a
   DLL conflict. `src/models_dl.py` imports it first by design.
5. **SVR defaults to a linear kernel.** The RBF variant is O(n²) in memory and
   does not scale to ~23k rows on a laptop. `kernel: rbf` remains configurable.
6. **Deep learning trains on CPU** and is correspondingly sized; early stopping
   halts training between epoch 9 and 12 in every run.
7. `R²` is negative for most models. That is the correct signal that a
   near-random-walk target is not linearly predictable from these features, not
   a bug.
8. **The cross-sectional question is a null result.** 14 configurations, none
   better than assuming all ten stocks move together. Reported as a failure.
   It is a separate task with its own baseline and is never counted toward the
   Section 8.3 bar.

## Leaks found and fixed during model improvement

Each of these was made while genuinely trying to improve the models, and each
would have survived a superficial review. Each is now a test, not a note.

1. **A renamed label is still a label.** `Relative_Target` and `Rank_Target`
   are pure re-expressions of the target. The generic numeric-column selector
   admitted them as inputs, and the model read its own answer — a spurious 95%
   error reduction. The shuffled-label control **passed** and did not catch it.
   Guard: `tests/test_causality.py::test_no_feature_is_a_repackaged_label`.

2. **A negative control is necessary but not sufficient.** Shuffling the labels
   proves the model uses its feature set. It cannot prove the feature set is
   legitimate. Structural invariants are required alongside behavioural ones.

3. **An untested function is a blind spot.** `build_cross_sectional_context` was
   never called by the causality suite, and it contained a look-ahead: `Beta`
   was a scalar fitted at the end of the sample, broadcast backwards over every
   earlier date. Guard: `TestCrossSectionalCausality`, including a regression
   test that Beta varies over time.

4. **A suspiciously good result is evidence of a bug.** A 19x error reduction
   should have stopped the run immediately. The suspicion is now written into
   the procedure in advance so it is not skipped next time.

5. **A stale artifact is not inert.** After the feature table changed, saved
   models were removed as stale but `dl_metrics.csv` was not, and downstream
   stages consumed the old numbers. Guard: `tests/test_staleness.py`.

6. **A zero exit code is not a valid measurement.** The linear SVR hit its
   iteration cap on every fold. `scikit-learn` warned; the process exited 0;
   the leaderboard carried a coefficient vector that was not the optimum for the
   C and epsilon it was reported against. The settings were then chosen by
   measurement (tol 1e-3, max_iter 200,000) at a cost of about eleven minutes,
   which is the right trade. Guards: a convergence test, and `retrain_models.py`
   now fails the build on a `ConvergenceWarning` even when the stage exits zero.

The SVR convergence cost, measured on the real training partition: at C=0.1 the fit
converges in ~23,000 iterations and about four minutes; at C=1.0 it does not
converge at all within a 200,000-iteration cap, so C=1.0 is excluded from the
grid. Removing the exactly-collinear columns helped (317s → 248s at C=0.1) but
could not rescue C=1.0 — the residual dependence among the price-level columns is
near-collinearity, not algebra. Convergence is now recorded per row in the
`Converged` column, non-converged rows are excluded from the Section 8.3
comparison, and `retrain_models.py` fails the build on a `ConvergenceWarning`
even when the stage exits zero.

## Deliberate deferrals

Per the PRD: intraday trading, brokerage execution, options and derivatives
pricing, reinforcement-learning agents, and guaranteed-profit claims are out of
scope. Future enhancements (real-time ingestion, Docker, CI, MLflow, model
registry, drift monitoring, Black-Litterman, CVaR optimisation,
transaction-cost modelling, Temporal Fusion Transformer) are documented in the
PRD as roadmap items, not baseline deliverables.
