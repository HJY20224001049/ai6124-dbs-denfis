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
- [x] Consultation and implementation checks: target label, synthetic signal, no-prediction scope (2026-11-11)
- [x] Original-target pilots: simplified models and ex-dividend features (2026-11-11)
- [x] B2 frozen and run once on the 2017–2025 test years and the 2026 holdout (2026-11-15)
- [ ] Joint review and method-line sections
- [ ] Final notebook and slides (due 2026-11-18 23:59)

Current pooled result — frozen B2 plus the protocol models (2017–2025, 10 bp one-way cost):

| model | RMSE | RMSE skill vs zero | direction acc. | Sharpe | max drawdown | exposure |
|---|---:|---:|---:|---:|---:|---:|
| no prediction (ŷ = 0) | 0.011975 | 0 | — | — | — | 0% |
| **DENFIS B2 (frozen)** | **0.011904** | **+0.59%** | 0.515 | 0.537 | 27.8% | 41.5% |
| XGBoost | 0.012009 | −0.28% | 0.508 | 0.238 | 32.2% | 53.6% |
| LSTM | 0.012043 | −0.57% | 0.485 | 0.407 | 27.5% | 30.7% |
| DENFIS frozen (full 13-feature) | 0.012066 | −0.76% | 0.501 | 0.654 | 20.3% | 34.4% |
| buy & hold | — | — | — | 1.113 | 38.6% | 100% |

The frozen configuration (B2, three one-rule components on `ma5`, `boll_b` and
`ret_0` plus ex-dividend dummies) beats the zero forecast by +0.59% pooled RMSE
skill, with 8 of 9 test years positive, and the difference is significant by
Diebold–Mariano against both zero (p = 0.0024) and XGBoost (p = 0.0006); at
α = 0.10 it is the only model left in the range-based model confidence set. The
three earlier models remain worse than the zero forecast — DENFIS with the full
13-feature rule base included. Trading still does not beat buy & hold on Sharpe
(0.537 versus 1.113), and the 2026 holdout is only +0.14% RMSE skill, so the
honest summary is: a small, statistically detectable forecast edge, not a
profitable strategy. See `docs/freeze.md` and `docs/results_table.md`.

The online extension is reported separately: pooled RMSE 0.012185 and Sharpe
0.260, below frozen DENFIS on this run. DM comparisons are not significant;
see `docs/statistical_tests.md`. Method-line handoff: [docs/handoff_to_B.md](docs/handoff_to_B.md).

### Original-target pilots (simplified models)

The full 13-feature models overfit a weak signal and all lose to the zero
forecast. The diagnostics then simplified each model while keeping the same
next-day close-to-close log-return target:

| model | RMSE skill vs zero | folds > 0 | configuration |
|---|---:|---:|---|
| DENFIS, one rule + ex-dividend dummies | **+0.510%** | 7/9 | `ma5`, `boll_b`, `ret_0`, `is_exdiv`, `is_post_exdiv_1`; `max_rules=1`, `Dthr=3.0`, `λ=1.0` |
| LSTM, small network | +0.260% | 6/9 | 8 hidden, dropout 0.3, weight decay 1e-2 |
| XGBoost, shallow | +0.212% | 6/9 | depth 1, learning rate 0.01, 100 trees |
| no prediction (ŷ = 0) | 0 | — | benchmark |

`RMSE skill vs zero = 1 − RMSE_model / RMSE_zero`, pooled over the 2017–2025
test years; the zero forecast has RMSE ≈ 0.011975. The point-estimate ordering
DENFIS > LSTM > XGBoost > zero holds for the original target.

Caveats: the ex-dividend pair helps DENFIS only (XGBoost and LSTM are
unchanged-to-worse); on validation years LSTM (+0.520%) is still above DENFIS
with the ex-dividend pair (+0.432%), so the test ordering is not
validation-guaranteed; the intervals are wide and the test years were already
inspected during the diagnostics. See `docs/return_pilot.md` and
`docs/event_feature_pilot.md`. The 2026 holdout is the clean confirmation.

The frozen B2 holdout was run once after the freeze: n = 209 (2026-01 to
2026-10), RMSE 0.010140 versus 0.010154 for the zero forecast (+0.14% skill),
directional accuracy 0.522, Sharpe 1.71 at 12% exposure. The skill is small and
the Sharpe comes from a short, low-exposure window; neither guided any choice.
Baseline holdout artifacts predate this handoff.

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
python scripts/return_pilot.py                      # simplified models, original target
python scripts/event_feature_pilot.py               # ex-dividend feature pilot
python scripts/run_b2.py                            # B2 validation score (no test years)
python scripts/fair_validation.py                   # baselines with inner early stopping
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
| 2026-11-15 | B2 frozen; test and holdout run once |
| 2026-11-16 | Method-line sections due; writing day |
| 2026-11-18 12:00 | Submit (official deadline 23:59) |

## Reproducibility

The final notebook must run top-to-bottom on a fresh kernel using only cached data in
`data/raw/`. All randomness is seeded; all fitted preprocessing uses training-fold
statistics only.
