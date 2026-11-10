# AI6124 Group 18 — DBS Stock Price Prediction with DENFIS

Neuro-fuzzy hybrid system for next-day DBS Group Holdings (`D05.SI`) return prediction:
ECM online clustering → first-order TSK fuzzy rules → weighted RLS consequents.

## Team

- LIN Jingzhao
- HU Junyu

## Documentation

- [Unified protocol](PROTOCOL.md) — data, features, folds, model interface, evaluation, artifacts

Internal workplans and the course proposal are kept out of this repository.

## Status

- [x] Proposal submitted (2026-10-14)
- [x] Data pipeline and cached data (D05.SI 4,219 rows; ^STI 4,210 rows)
- [x] 13-feature, 9-fold walk-forward, metrics and backtest modules; method contracts included
- [x] XGBoost and LSTM baselines on all 9 folds
- [x] Buy-and-hold comparison and leakage audit (see `docs/leakage_audit.md`)
- [x] DM tests, bootstrap Sharpe CIs, model-agnostic result tables
- [x] ^STI transfer alignment helper for the method workstream
- [x] DENFIS implementation and nine-fold predictions (frozen + online)
- [x] Method report, integration notebook and five-slide draft (`writing/`)
- [ ] Joint review, consultation decisions, scheduled freeze and final holdout
- [ ] Final notebook and slides (due 2026-11-18 23:59)

Current pooled result (2017–2025, 10 bp one-way cost):

| model | RMSE | direction acc. | Sharpe | max drawdown | exposure |
|---|---|---|---|---|---|
| XGBoost | 0.0120 | 0.508 | 0.238 | 32.2% | 53.6% |
| LSTM | 0.0120 | 0.485 | 0.407 | 27.5% | 30.7% |
| DENFIS frozen | 0.0121 | 0.501 | 0.654 | 20.3% | 34.4% |
| buy & hold | — | — | 1.113 | 38.6% | 100% |

All main models have pooled R² below zero and directional accuracy near 50%. They reduce
drawdown but give up most of the upside versus buy & hold. This is the honest
comparison to report without overclaiming.

The online extension is reported separately: pooled RMSE 0.012185 and Sharpe
0.260, below frozen DENFIS on this run. DM comparisons are not significant;
see `docs/statistical_tests.md`. Method-line handoff: [docs/handoff_to_B.md](docs/handoff_to_B.md).

DENFIS 2026 holdout remains reserved until the scheduled freeze. Baseline
holdout artifacts predate this handoff and must not guide further tuning.

## Quick start

Use Python 3.11 and install `requirements.txt` in a virtual environment.
DENFIS does not import baseline dependencies unless a baseline is requested.
For notebook/slide generation, also install `requirements-deliverables.txt`.

```bash
python -m src.data                                  # download and cache data
python scripts/eda.py                               # EDA figures
python scripts/check_pipeline.py                    # feature/fold sanity check
python -m pytest -q                                 # contract tests
python scripts/run_baseline.py --model xgboost --folds all
python scripts/run_baseline.py --model lstm --folds all
python scripts/run_baseline.py --model xgboost --folds r1f1   # one fold
python scripts/make_results_table.py                # docs/results_table.md
python scripts/plot_results.py                      # equity and DA figures
python scripts/statistical_tests.py                 # docs/statistical_tests.md
python scripts/leakage_audit.py                     # docs/leakage_audit.md
```

The implemented `src/models/denfis.py` is exposed through
`--model denfis_frozen` and `--model denfis_online` (lazy import; both search
the protocol's `Dthr × λ` grid).

Data are downloaded from the Yahoo Finance chart endpoint (the same daily
OHLCV and adjusted-close fields returned by `yfinance`); the cache is what the
final notebook reads. No network access is needed after `python -m src.data`.

## Repository layout

```text
config.py                 # single source of truth
data/raw/                 # cached D05.SI and ^STI CSV files
src/data.py               # loading, cleaning
src/features.py           # 13 causal features
src/transfer.py           # ^STI ↔ DBS alignment for transfer experiments
src/splits.py             # 9 walk-forward folds + 2026 holdout + purge
src/models/denfis.py      # DENFIS implementation
src/models/baselines.py   # XGBoost, LSTM
src/evaluation/           # metrics, backtest, CV runner
results/                  # predictions, metrics, figures, manifest
tests/                    # contract and leakage tests
docs/                     # method notes, references, consultation notes
writing/                  # report drafts before notebook integration
```

## Key dates

| Date | Milestone |
|---|---|
| 2026-11-08 | Protocol frozen; data cached |
| 2026-11-13 | All models run on all 9 folds (hard gate) |
| 2026-11-15 22:00 | Results frozen |
| 2026-11-16 | Dedicated writing day |
| 2026-11-18 12:00 | Submit (official deadline 23:59) |

## Reproducibility

The final notebook must run top-to-bottom on a fresh kernel using only cached data in
`data/raw/`. All randomness is seeded; all fitted preprocessing uses training-fold
statistics only.
