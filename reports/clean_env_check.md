# Clean-environment reproducibility check

**Verdict: REPRODUCIBLE: every step passed in a fresh virtual environment installed only from requirements.txt.**

- Mode: full
- Python: 3.13.5 on Windows-11-10.0.26200-SP0
- Wall time: 38.0 minutes
- Environment: a throwaway `python -m venv`, populated only from `requirements.txt`

| Step | Result | Exit | Seconds |
|---|---|---:|---:|
| create virtual environment | PASS | 0 | 10.5 |
| install pinned requirements | PASS | 0 | 368.2 |
| rebuild dataset | PASS | 0 | 12.8 |
| retrain models | PASS | 0 | 1837.5 |
| test suite | PASS | 0 | 48.4 |
| artifacts present and built in order | PASS | - | 0.0 |
| dashboard imports, all panels execute | PASS | 0 | 3.1 |

## What each step proves

| Step | What it demonstrates |
|---|---|
| create virtual environment | the project needs nothing pre-installed |
| install pinned requirements | `requirements.txt` is complete and sufficient |
| rebuild dataset | ingest/clean/features run from source with no state |
| retrain models | maths, ML, portfolio and recommendations all reproduce |
| test suite | the integrity checks pass in a foreign environment |
| artifacts present and built in order | outputs exist *and* each was produced after the stage it depends on |
| dashboard imports, all panels execute | the app runs, not just the library |

## Why the artifact check is ordered, not a flat list

Each stage must be newer than the stage it consumes.
`cleaned_panel` feeds `features`, `features` feeds the models, and
the predictions feed the portfolio, recommendation and
cross-sectional stages. Comparing every artifact to a single
reference file cannot express that, and in particular would flag
`cleaned_panel` as stale simply because it is built first.
