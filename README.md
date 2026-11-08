# AI6124 Group 18 — DBS Stock Price Prediction with DENFIS

Neuro-fuzzy hybrid system for next-day DBS Group Holdings (`D05.SI`) return prediction:
ECM online clustering → first-order TSK fuzzy rules → weighted RLS consequents.

## Team

- LIN Jingzhao
- HU Junyu

## Project documents

- [Unified protocol](PROTOCOL.md) — data, features, folds, model interface, evaluation, artifacts
- [Workplan A — method](WORKPLAN_A.md)
- [Workplan B — data & evaluation](WORKPLAN_B.md)
- [Project plan](project_plan.html)

## Status

- [x] Proposal submitted (2026-10-14)
- [ ] Data pipeline and cached data
- [ ] DENFIS implementation (frozen + online)
- [ ] XGBoost / LSTM baselines
- [ ] 3×3 walk-forward evaluation
- [ ] Final notebook and slides (due 2026-11-18 23:59)

## Repository layout

```text
config.py                 # single source of truth (to be added on D0)
data/raw/                 # cached D05.SI and ^STI CSV files
src/data.py               # loading, cleaning
src/features.py           # 13 causal features
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
