<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/AI%2FML-Research-8A2BE2?style=for-the-badge">
  <img src="https://img.shields.io/badge/TensorFlow-Deep%20Learning-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white">
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white">
</p>

<p align="center">
  <b>Stock Market Forecasting • Portfolio Optimization • Market Intelligence</b>
  <br>
  <sub>Predict 📊 · Analyze 🔍 · Optimize 💼 · Test 🧪</sub>
</p>

---

## 🌟 About NorthGate

**NorthGate AI** is an AI/ML stock-market research platform bringing forecasting,
sentiment analysis, portfolio optimisation, risk analysis and out-of-sample
evaluation into one system.

Instead of only asking:

> 🤔 **"Can we predict the market?"**

NorthGate goes one step further:

> 🎯 **"Do the predictions actually provide useful information on unseen data?"**

The project focuses on **testing models honestly**, comparing them against
simple baselines, and studying what happens when predictions drive portfolio
decisions.

---

## 🚀 What's Inside?

| 🔥 | Module | What it does |
|---|---|---|
| 📈 | **Forecasting** | Predict market movements across 8 models + a baseline |
| 🤖 | **Machine Learning** | Ridge, Random Forest, XGBoost & Linear SVR |
| 🧠 | **Deep Learning** | LSTM, GRU, BiLSTM & Transformer |
| 📰 | **Sentiment** | Score financial news with FinBERT & VADER |
| 💼 | **Portfolio Optimisation** | Efficient frontier, min-variance & max-Sharpe |
| ⚠️ | **Risk Analysis** | Volatility, drawdown, VaR, Sortino, Calmar & Beta |
| 🧪 | **Backtesting** | Disjoint out-of-sample windows, never in-sample |
| 🎯 | **Recommendations** | BUY / HOLD / SELL with a backtested hit-rate |
| 🖥️ | **Dashboard** | 8 Streamlit panels, disclaimer on every one |

---

## 🧩 How NorthGate Works

```text
                 📥 DATA
                   │
       ┌───────────┼───────────┐
       ▼           ▼           ▼
    📈 Prices   📉 Macro    📰 News
       │        (FRED)         │
       └───────────┼───────────┘
                   ▼
          ⚙️ FEATURE ENGINEERING
              132 signals
                   │
                   ▼
             🤖 FORECASTING
                   │
       ┌───────────┼───────────┐
       ▼           ▼           ▼
      📊 ML      🧠 Deep     🎲 Random
       │        Learning      Walk
       └───────────┼───────────┘
                   ▼
             🔍 EVALUATION
       MAE · RMSE · MAPE · R² · DirAcc
                   │
                   ▼
          💼 PORTFOLIO ENGINE
                   │
          ┌────────┴────────┐
          ▼                 ▼
   ⚠️ RISK ANALYSIS   🎯 RECOMMENDATIONS
          │                 │
          └────────┬────────┘
                   ▼
             🖥️ DASHBOARD
```

Data flows one way. No layer reaches back into an earlier one at runtime.

---

## 📈 Forecasting Models

NorthGate compares several model families on **one** held-out window rather than
relying on a single algorithm.

| 🤖 Machine Learning | 🧠 Deep Learning | 🎲 Baseline |
|---|---|---|
| Ridge Regression | LSTM | Random Walk |
| Random Forest | GRU | |
| XGBoost | BiLSTM | |
| Linear SVR | Transformer | |

An **equal-weight ensemble** is also reported, but is **ineligible** to be named
the winner — a blend is not a finding.

The Random Walk is the bar. A complicated model still has to prove its extra
complexity buys information.

**Declared for deployment: LSTM** — lowest MAE among models whose optimiser
converged. Ensembles and the baseline are excluded by rule.

---

## 📊 Results

### 🎯 Forecasting

<!--LIVE:MODELTABLE-->

<!--LIVE:HEADLINE-->

### 🧾 PRD acceptance, in one line

<!--LIVE:ACCEPTANCE-->

### 🧪 Out-of-sample honesty

The test window is touched **once**. Hyperparameters come from walk-forward CV
*inside* the training partition; models are refit on train+validation and scored
once on the held-out window that fills the table above.

### 🔐 The causality rule

> **If the model sees the future, the result doesn't count.**

This is proven, not asserted. `tests/test_causality.py` rebuilds the feature
table from history truncated at a mid-sample cutoff and requires every value at
or before the cutoff to be bit-identical; it then corrupts every row *after* the
cutoff and requires the earlier rows to be unchanged. Both probes cover the
per-ticker builder **and** the cross-sectional builder.

### 🎯 Recommendations

<!--LIVE:HITRATE-->

### 📰 Sentiment

The with/without-sentiment ablation is **NOT RUN**. The provider is built and
reachable, but the news API rate-limited this machine to **0 of 60 chunks**
retrieved. No unsupported sentiment numbers are reported.

The harness is independently verified on a *synthetic* signal (directional
accuracy 47.25% → 81.67%), which proves the split, feature selection and Δ
arithmetic work — and is explicitly **not** a sentiment result.

---

## 🛡️ Evaluation

Financial ML can easily produce impressive-looking results for the wrong
reasons. NorthGate therefore includes safeguards for:

| ✅ Safeguard | How it is enforced |
|---|---|
| Chronological train/test splits | partitions cut on unique dates, then expanded to rows |
| Train-only scaling | refit inside every walk-forward fold |
| Leakage checks | any input above \|r\| 0.99 with a return-valued label fails the build |
| Causality checks | truncation + future-corruption probes, asserted bit-identical |
| Fixed random seeds | `config.yaml: random_seed` drives NumPy, sklearn and TF |
| Artifact validation | every output is checked against the table it was fitted on |
| Out-of-sample evaluation | estimation and backtest windows asserted disjoint |
| Baseline comparisons | random walk and equal weight on every comparison |
| Reproducibility checks | full rebuild in a throwaway venv |
| Optimiser convergence | the build **fails** on `ConvergenceWarning` |

**Outliers are flagged, never deleted** — an earnings-day move is signal, and
deciding which flags are vendor errors needs a corporate-actions calendar this
project does not have.

**Redundant columns removed, correlated ones kept.** Exact linear duplicates
(`MACD_Hist` ≡ `MACD − MACD_Signal`) are dropped; strong correlation is left
alone, because two correlated columns carry different information.

---

## 💼 Portfolio Optimisation

### ⚙️ Methods

- 📉 Minimum Variance
- 📈 Maximum Sharpe
- 📊 Efficient Frontier
- ⚖️ Equal-Weight benchmark
- 🔒 Long-only constraints
- 📏 Position caps
- 🧮 Ledoit-Wolf covariance shrinkage
- 🧪 Disjoint out-of-sample backtest
- 🎲 20,000-portfolio Monte-Carlo cross-check

<!--LIVE:PORTFOLIO-->

---

## 🗂️ Market Universe

NorthGate evaluates **10 equities**:

| Ticker | Company | Ticker | Company |
|---|---|---|---|
| 🍎 **AAPL** | Apple | 🟢 **NVDA** | NVIDIA |
| 🪟 **MSFT** | Microsoft | 🥤 **KO** | Coca-Cola |
| 🏦 **JPM** | JPMorgan Chase | 🏗️ **CAT** | Caterpillar |
| 🛢️ **XOM** | Exxon Mobil | 🏠 **HD** | Home Depot |
| 💊 **JNJ** | Johnson & Johnson | | |
| 🛒 **PG** | Procter & Gamble | | |

Plus market and macroeconomic context: 📊 S&P 500, 📉 VIX, and FRED macro
series. Price history spans **2015-01-02 to 2026-09-17 (11.7 years)**.

---

## 🖥️ Dashboard

Eight interactive panels. The disclaimer appears on **every** one.

| Panel | What it shows |
|---|---|
| 🏠 **Overview** | Universe snapshot, market-mood gauge, headline metrics |
| 📈 **Price & Prediction** | Historical price, model forecast, **calibrated confidence band** |
| 🤖 **Model Comparison** | All 8 models + baseline on one window, and which is declared |
| 💼 **Portfolio Analytics** | Frontier, weights, backtest vs. baselines |
| ⚠️ **Risk Dashboard** | Volatility, drawdown, VaR, Sortino, Calmar, Beta |
| 📰 **Sentiment** | News sentiment, market mood, the ablation status |
| 🎯 **Recommendations** | BUY/HOLD/SELL with sub-signals and the backtested hit-rate |
| 🔀 **Cross-Sectional Ranking** | Which stock beats the other nine — own baseline, reported separately |

Two panels are worth calling out because they were rebuilt after a defect:

- **The confidence band** is centred on the *forecast*, not the actual close, and
  is built from the empirical quantiles of the model's own out-of-sample errors.
  Measured coverage lands exactly on its stated level (80% → 80.0%, 90% → 90.0%,
  95% → 95.0%), and the figure is printed under the chart.
- **The market-mood gauge** states which source it is reading. With no news API
  it falls back to market breadth and says so, rather than passing breadth off as
  sentiment.

---

## 🧰 Tech Stack

| Area | Tools |
|---|---|
| 🐍 Core | `Python` · `Pandas` · `NumPy` · `PyArrow` |
| 🤖 Machine Learning | `Scikit-learn` · `XGBoost` |
| 🧠 Deep Learning | `TensorFlow` · `Keras` |
| 📰 NLP | `FinBERT` · `VADER` |
| 📊 Data | `yfinance` · `pandas-datareader` (FRED) · `Finnhub` · `GDELT` |
| 💼 Optimisation | `SciPy` (SLSQP) · `Ledoit-Wolf` |
| 🖥️ Dashboard | `Streamlit` · `Plotly` |
| 📈 Charts | `Matplotlib` |
| 🧪 Testing | `Pytest` |
| 🔧 Development | `Git` · `GitHub` |

> **Disclosed deviations.** `PyPortfolioOpt` is not used — importing `pypfopt`
> hard-crashes the interpreter (`0xC0000005`), reproduced across all nine of its
> submodules — so the same constrained programme is solved with
> `scipy.optimize.minimize(SLSQP)`, with shrinkage from
> `sklearn.covariance.LedoitWolf`. News defaults to Finnhub and falls back to
> GDELT, a weaker corpus, only when no key is present.

---

## 📁 Project Structure

```text
NorthGate-AI/
│
├── 📂 src/                        flat modules, one per PRD section
│   ├── common.py                  paths, config, seeding, logging
│   ├── ingest.py                  Section 3
│   ├── clean.py                   Section 4
│   ├── features.py                Section 5  (132 features, no look-ahead)
│   ├── create_eda.py              Section 6
│   ├── math_utils.py              Section 7  (15/15 vs libraries)
│   ├── models_ml.py               Section 8
│   ├── models_dl.py               Section 9  (LSTM/GRU/BiLSTM/Transformer)
│   ├── sentiment.py               Section 10 (Finnhub + ablation harness)
│   ├── sentiment_gdelt.py         keyless news fallback
│   ├── portfolio.py               Section 11
│   ├── recommend.py               Section 12
│   ├── cross_sectional.py         supplementary ranking analysis
│   └── recommendation_backtest.py Section 14 (hit-rate vs buy-and-hold)
│
├── 📂 dashboard/
│   └── app.py                     Section 13, 8 panels
│
├── 📂 scripts/                    flat utilities
│   ├── fill_report.py             inject live numbers into the report
│   ├── fill_readme.py             inject live numbers into this file
│   ├── export_report_pdf.py       report → NorthGate-AI-Report.pdf
│   ├── clean_env_check.py         rebuild in a throwaway venv
│   ├── fetch_news_retry.py        survive news-API rate limiting
│   ├── verify_dashboard.py        headless launch check
│   ├── improvement_rules.py       what counts as a real improvement
│   └── snapshot_artifacts.py      artifact manifest
│
├── 📂 tests/                      167 tests
│
├── 📂 reports/
│   ├── final_report.md            the research report
│   ├── final_report.template.md
│   ├── readme.template.md
│   ├── clean_env_check.md         fresh-venv rebuild evidence
│   ├── data_quality_report.md
│   ├── math_verification.json
│   └── figures/
│
├── 📂 notebooks/eda.ipynb
├── 📂 data/{raw,processed}/
├── 📂 models/                     .keras, .pkl
├── 📄 config.yaml                 every tunable
├── 📄 requirements.txt            pinned
├── 📄 rebuild_dataset.py          one command
├── 📄 retrain_models.py           one command
├── 📄 NorthGate-AI-Report.pdf
└── 📄 README.md
```

---

## 🔍 What Did We Learn?

NorthGate is not designed to make every experiment look successful.

### ✅ What worked

- Lower MAE than the Random Walk baseline
- 8 models + baseline compared on one held-out window
- Complete portfolio optimisation workflow, Monte-Carlo cross-check passed
- Disjoint out-of-sample testing throughout
- Leakage, causality and reproducibility safeguards
- A backtested recommendation engine with transparent sub-signals

### ⚠️ What did not beat its benchmark

- Directional accuracy did not beat the Random Walk
- Optimised portfolio Sharpe did not beat equal weight
- Recommendation hit-rate did not beat buy-and-hold
- Cross-sectional ranking did not beat its own no-skill baseline
- The sentiment ablation could not be measured — external rate limiting

> 💡 **A result doesn't have to be positive to be useful.**

The most instructive finding is negative. A `Beta` feature had been fitted as a
single scalar at the *end* of the sample and broadcast over every earlier date.
Removing that leak made validation **worse** — XGBoost directional accuracy fell
from 54.59% to 50.14% — because the models had been leaning on a feature whose
value at any past date was partly set by returns that had not yet happened. The
reported numbers are the worse, trustworthy ones.

---

## 📌 Current Project Status

| Component | Status |
|---|---|
| 📊 Market Data | ✅ |
| ⚙️ Feature Engineering | ✅ |
| 🤖 ML Forecasting | ✅ |
| 🧠 Deep Learning | ✅ |
| 🎲 Baseline Comparison | ✅ |
| 💼 Portfolio Optimisation | ✅ |
| ⚠️ Risk Analysis | ✅ |
| 📰 Sentiment Pipeline | ✅ |
| 🧪 Sentiment Ablation | ⚠️ Not Measured — provider rate-limited |
| 🔍 Leakage & Causality Checks | ✅ |
| 🎯 Recommendation Hit-Rate | ✅ Measured, unfavourable |
| 🔄 Reproducibility | <!--LIVE:REPRO--></p>

---

## ▶️ Getting Started

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Satvi27-debug/NorthGate-AI.git
cd NorthGate-AI
```

### 2️⃣ Create and activate a virtual environment

```bash
python -m venv .venv
```

**Windows**

```powershell
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

> The pins resolve together. An earlier set did not — `tensorflow 2.21` needs
> `protobuf >=6.31.1,<8` while `streamlit 1.40.2` needed `protobuf <6` — and a
> test now fails the build if they ever stop overlapping.

### 4️⃣ Build and train

```bash
python rebuild_dataset.py     # ingest → clean → 132 features
python retrain_models.py      # maths → ML → DL → portfolio → recommendations
```

### 5️⃣ Launch the dashboard

```bash
streamlit run dashboard/app.py
```

### 6️⃣ Generate the report

```bash
python scripts/fill_report.py
python scripts/export_report_pdf.py
```

### 7️⃣ Verify everything

```bash
python -m pytest tests/ -q      # 167 tests
python scripts/clean_env_check.py
```

---

## 📚 Documentation

- 📄 [`reports/final_report.md`](reports/final_report.md) — full research report, ~12k words
- 📕 [`NorthGate-AI-Report.pdf`](NorthGate-AI-Report.pdf) — the same, rendered
- 🧪 [`tests/`](tests/) — automated tests
- 📊 [`reports/`](reports/) — generated artifacts

---

## ⚠️ Disclaimer

NorthGate AI is an **educational and research project**.

It is not financial advice and should not be used as a standalone system for
making real investment decisions. Past performance and model predictions do not
guarantee future results.

This disclaimer appears on every panel of the dashboard and is asserted by test.

---

<p align="center">

## 🚀 NorthGate AI

### Predict less. Test more.

🐍 Python · 🤖 AI/ML · 📊 Data Science · 💼 Quantitative Research

</p>
