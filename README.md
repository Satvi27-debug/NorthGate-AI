<<<<<<< HEAD
Northgate Quantitative Research
AI-Based Stock Market Prediction & Portfolio Optimization System

Python 3.11+ · educational research and decision support · not financial advice

## What this is

A reproducible, leak-free research pipeline that moves from raw market data to
risk-aware portfolio allocation and transparent buy/hold/sell guidance.

    raw sources -> cleaned panel -> features -> models -> portfolio & recs -> dashboard

Data flows one way. No layer reaches back into an earlier one at runtime.

## Quick start

    pip install -r requirements.txt

    python rebuild_dataset.py          # ingest -> clean -> features
    python retrain_models.py           # math verify -> ML -> DL -> portfolio -> recs
    streamlit run dashboard/app.py     # interactive dashboard

    python -m pytest tests/ -v         # layer + integrity tests
    python src/create_eda.py           # EDA notebook + figures
    python scripts/verify_dashboard.py # launch headless, assert it serves
    python scripts/snapshot_artifacts.py

Optional sentiment (needs a free API key):

    set FINNHUB_API_KEY=<your key>      # PowerShell
    python rebuild_dataset.py --with-sentiment
    python retrain_models.py

The pipeline runs without sentiment; the dashboard reports it as not run rather
than inventing a result.

## Results, stated up front

**Neither of the PRD's two acceptance bars is met, and both are reported as
measured.** The best model beats the naive random walk on MAE (0.011682 vs
0.012360) but not on directional accuracy (53.37% vs 53.93%). The optimised
portfolio returns a Sharpe of 1.357 against equal weight's 1.648 on a disjoint
out-of-sample window, with a block-bootstrap 95% interval of [−0.536, 2.970]
spanning zero.

The dashboard leads with this verdict rather than burying it, and
`reports/final_report.md` explains why in §6.3 and §8.5: daily equity returns are
close to a martingale process, so the binding constraint is the problem, not the
model. The PRD states that "a modest, correct, leakage-free result with honest
analysis scores higher than an impressive-looking result built on a subtle
look-ahead bug."

## Layout

    config.yaml              every tunable; no magic numbers in code
    rebuild_dataset.py       one command to rebuild the dataset
    retrain_models.py        one command to retrain and evaluate all models
    requirements.txt         pinned dependencies
    data/raw/                downloaded, immutable
    data/processed/          cleaned panel, features, metrics, predictions
    models/                  saved artifacts (.keras, .pkl)
    notebooks/eda.ipynb      EDA with captioned, decision-linked findings
    dashboard/app.py         Streamlit, eight panels
    reports/                 data-quality report, figures, written report
    tests/                   layer tests
    src/
      common.py              paths, config, seeding, logging
      ingest.py              Section 3
      clean.py               Section 4
      features.py            Section 5
      create_eda.py          Section 6
      math_utils.py          Section 7
      models_ml.py           Section 8
      models_dl.py           Section 9
      sentiment.py           Section 10
      portfolio.py           Section 11
      recommend.py           Section 12
      cross_sectional.py     supplementary ranking analysis (own baseline)
      dashboard app          Section 13
    scripts/
      fill_report.py         inject live numbers into the written report
      verify_dashboard.py    headless launch check + artifact presence
      improvement_rules.py   what counts as a legitimate improvement, written
                             down before attempting one
      snapshot_artifacts.py  artifact manifest for reproducibility diffing
    tests/
      test_pipeline.py       cleaning, no-look-ahead, maths, portfolio, recs
      test_causality.py      proves features are causal; blocks label leaks
      test_dashboard.py      panel presence, disclaimers, no fabricated metrics
      test_panels.py         executes every panel against real artifacts
      test_cross_sectional.py supplementary analysis cannot pose as the 9.5 bar
      test_staleness.py      every artifact matches the table it was fitted on

## Methodology commitments

**Adjusted close everywhere.** Section 3.1 calls raw-close returns a
disqualifying error. Every return, trend, momentum, volatility and target column
is derived from `Adjusted Close`. Raw High/Low appear only in ATR, which is
defined on the session range. The target is fixed in `config.yaml` as a
1-session forward log return, justified empirically by the ADF test in EDA.

**Splits on the date axis.** Partitions are cut on unique dates and then expanded
to rows, so every ticker at a given instant sits in exactly one partition. A
row-wise slice would put ticker A's test rows beside ticker B's training rows at
the same timestamp.

**Scaler fit on train only.** Fitted on the training partition, used to transform
validation and test. Refit inside every walk-forward fold during tuning.

**Redundant columns removed, correlated ones kept.** Columns that are exact
linear functions of other columns are dropped automatically — `MACD_Hist` is
identically `MACD - MACD_Signal`, `Return_Reversal_5d` is `−Return_5d`. That
discards no information (each is recoverable by linear combination) but removes
an unidentifiable coefficient from the linear models and conditions the design
matrix for the support-vector model. Strong *correlation* is left alone: two
correlated columns carry different information.

**Test touched once.** Hyperparameters come from walk-forward CV inside the
training partition. Models are refit on train+validation and evaluated once on
the held-out test window that populates the eight-model table.

**Outliers flagged, never deleted.** Section 4.1 requires that an earnings-day
move be kept as signal. Rows beyond the rolling z-score or 1.5x IQR fence are
flagged and logged; nothing is winsorised, because confirming which flags are
genuine vendor errors needs a corporate-actions calendar this project does not
have.

**Non-causal columns excluded.** The IQR fence uses full-sample quantiles, so
`Outlier_IQR` and the union flag `Outlier` are never model inputs. The rolling
z-score is trailing-only and is retained. The target and the future price it is
derived from are blocklisted, as are the two label variants the cross-sectional
module constructs (`Relative_Target`, `Rank_Target`) — a blocklist naming
`Target` alone does not stop a re-expression of it under a different name.

**Causality proven, not asserted.** `tests/test_causality.py` rebuilds the
feature table from history truncated at a mid-sample cutoff and asserts every
value at or before the cutoff is bit-identical to the full-history build; it
then corrupts every row *after* the cutoff and asserts the earlier rows are
still identical. Both probes cover the per-ticker builder **and** the
cross-sectional context builder. A feature that consumed any future information
would shift under both.

**Reported honestly.** Section 8.3 requires that a model unable to beat the naive
random walk be reported as a failure. The dashboard shows that verdict first, not
a cherry-picked ranking.

## Leaks found and fixed

Five defects were found and corrected during the build. Each would have survived a
superficial review, and each is now a test rather than a note.

| # | What was wrong | How it surfaced | Guard |
|---|---|---|---|
| 1 | A hardcoded metrics table in the dashboard | manual review | static assertion in `test_dashboard.py` |
| 2 | Estimation and backtest windows overlapped, inflating Sharpe to 2.998 | window audit | disjointness assertion in `test_panels.py` |
| 3 | Two columns derived from the target entered the feature set, giving a fake 95% error reduction | result looked *too* good | any input above \|r\| 0.99 with a return-valued label fails the build |
| 4 | `Beta` was one scalar fitted at the end of the sample, broadcast over every earlier date | an existing "no constant columns" test | causality probes extended to the cross-sectional builder |
| 5 | The linear SVR hit its iteration cap, so its coefficients were not the optimum for the hyperparameters reported | solver warning; process still exited 0 | convergence test, plus `retrain_models.py` fails the build on `ConvergenceWarning` |

Two lessons from these are worth more than the individual fixes. **A renamed
label is still a label** — defect 3 was caught by a structural correlation guard,
because the shuffled-label negative control had already passed and proved only
that the model used its inputs, not that the inputs were legitimate. And **an
untested function is a blind spot, not a clean bill of health** — defect 4 lived
in a function the causality suite never called.

Defect 4 is also the clearest evidence that a leak does not merely fail to help:
removing it made validation *worse* (XGBoost directional accuracy fell from
54.59% to 50.14%), because the models had been leaning on a feature whose value
at any past date was partly set by returns that had not happened yet. The
reported results are the worse, trustworthy ones.

## Environment notes

`PyPortfolioOpt` is deliberately not used: importing `pypfopt` hard-crashes the
interpreter in this environment (Windows access violation, exit code 0xC0000005),
reproduced across all nine of its submodules. `cvxpy` alone imports and solves
correctly. Section 11 is therefore solved with `scipy.optimize.minimize(SLSQP)`
on the same constrained programme, with Ledoit-Wolf shrinkage from
`sklearn.covariance.LedoitWolf`. Everything Section 11 requires is delivered.

**TensorFlow must be imported before pandas or pyarrow.** In the reverse order
the interpreter aborts with a DLL initialisation failure inside
`_pywrap_tensorflow_internal`; bisecting ten import permutations isolated pyarrow
as the trigger. `src/models_dl.py` imports TensorFlow first by design, and
`retrain_models.py` runs Section 9 as a separate subprocess so a native-library
conflict cannot take the verified ML results down with it.

**Disjoint estimation and backtest windows.** μ and Σ are estimated on the
trailing 504 sessions and the portfolio is scored on the following 252, never
overlapping. An earlier version overlapped them, which inflated the reported
Sharpe by letting the optimiser see the returns it was judged on.
`tests/test_panels.py` asserts the disjointness so it cannot regress.

Sequence windows for the deep models are built per ticker
(`dl.per_ticker_windows: true`) so a window is 60 consecutive sessions of one
asset rather than an interleaving of ten unrelated price paths.

`SVR` defaults to a linear kernel. Its RBF variant materialises an *n* × *n*
kernel matrix and is O(*n*²) in memory, which does not scale to ~23,000 training
rows on a laptop; `kernel: rbf` remains configurable for a smaller dataset. The
linear solver is run to convergence at `tol=1e-3, max_iter=200,000`, and the
settings were chosen by measurement rather than taste. `C=1.0` is excluded from
the grid because it does not converge on this feature table at all — the columns
are dominated by strongly co-moving price-level terms, and liblinear's
coordinate descent needs more than 200,000 iterations to reach a tolerance of
1e-3 at that regularisation. A configuration that cannot be fitted to
optimality cannot be scored honestly, so it is not searched.

Deep learning trains on CPU. The hardware note in the PRD applies: a free Colab
or Kaggle GPU session only speeds up training, it changes nothing about the
results.

## A separate, supplementary analysis

`src/cross_sectional.py` asks a different question from the rest of the system:
not *will the price rise?* but *which of these ten stocks beats the other nine?*
It is reported on its own panel and in its own report section, judged against its
own no-skill baseline ("they will all move together"), and **never** counted
toward the Section 8.3 forecast bar. A cross-sectional model deliberately ignores
which way the market is heading, so comparing it to the random-walk baseline
would be unfair to it and flattering to the headline.

The result is a null: none of the 14 configurations beat the no-skill baseline.
That is informative rather than disappointing. If ranking the universe were
easy, it would have told us the features carry signal that the absolute-return
formulation is failing to extract.

## Outputs

| Artifact | Contents |
|---|---|
| `reports/data_quality_report.md` | rows in/out, gaps, outliers, invariants |
| `reports/math_verification.json` | every formula vs. its library |
| `reports/eda_findings.json` | findings and the decisions they drove |
| `reports/artifact_manifest.json` | every artifact's size and shape |
| `data/processed/model_leaderboard.csv` | Section 9.5 eight-model table |
| `data/processed/*_predictions.parquet` | per-observation predictions |
| `data/processed/portfolio_metrics.json` | weights, frontier, backtest, Beta |
| `data/processed/recommendations.csv` | decisions with sub-signals |
| `data/processed/cross_sectional_results.json` | ranking analysis, own baseline |
| `reports/improvement_rules.json` | what counts as a legitimate improvement |
| `reports/final_report.md` | written report |

## Scope

Out of scope by design, per the PRD: intraday trading, brokerage execution,
options and derivatives pricing, reinforcement-learning agents, and any claim of
guaranteed profitability.

---

*This system is an educational research and decision-support tool. Its outputs do
not constitute financial advice.*
=======
# NorthGate-AI
>>>>>>> origin/main
