# Result freeze — B2 frozen 2026-11-15

Status: **FROZEN** (2026-11-15, Asia/Singapore). Recorded by the evaluation line;
the method line receives this record with the handoff specs in `writing/`.
From this point the model, features, hyperparameters, folds, cost and threshold
protocol are fixed. Only documentation may change.

Code state at freeze: commit `e0484f3` (B2 runner `9b00b5d`, audit `6dea769`,
statistics `e0484f3`). This note is committed immediately after that state; the
freeze refers to the code, not to this file's hash.

## Frozen configuration

| Item | Value |
|---|---|
| Model | B2: equal-weight average of three one-rule DENFIS models |
| Components | one per feature: `ma5`, `boll_b`, `ret_0` |
| Component inputs | the feature plus `is_exdiv` and `is_post_exdiv_1` |
| DENFIS settings | `dthr = 3.0`, `lam = 1.0`, `max_rules = 1`, `merge_factor = 0.5`, `prediction_clip = 0.2` |
| Target | `y_t = ln(AdjClose_{t+1} / AdjClose_t)` |
| Preprocessing | per-column scaler fitted on the training window only |
| Selection | validation year Y−1 only; labels through Y−2 for selection training |
| Final fit | train + validation (labels through Y−1) |
| Folds | nine regular folds, test years 2017–2025; 2026 holdout separate |
| Threshold | `tau` chosen on the validation year (Sharpe), grid from `config.TAU_GRID` |
| Costs | 10 bp one-way, long/flat |
| Seeds | `config.fold_seed(rep, fold)`; holdout uses `HOLDOUT_SEED` |
| Protocol | version 1.1; B2 is added as a row, existing rows are unchanged |

## Known results at freeze (validation only)

| metric | value |
|---|---:|
| mean validation RMSE skill vs zero | +0.497% |
| median | +0.616% |
| folds > 0 | 8/9 |
| pooled skill | +0.480% |
| pooled 95% CI (fold bootstrap, 5000 draws) | [+0.161%, +0.780%] |

Fair-validation context (`docs/fair_validation.md`): with inner early stopping,
XGBoost scores +0.021% and the LSTM −1.237% mean validation skill, so the
validation ordering is B2 > XGBoost > LSTM > zero. B2 is itself the best of
sixteen pre-registered variants selected on validation years; the 2026 holdout
is the only clean confirmation.

## One-shot test and holdout

The 2017–2025 test years and the 2026 holdout have **not** been run for B2 at the
time of freezing. They are run exactly once each, now, and reported whatever they
show; no configuration changes follow from either result.

```bash
python scripts/run_b2.py --scope test --after-freeze --folds all
python scripts/run_b2.py --scope holdout --after-freeze
```

Artifacts written by those commands:

- `results/preds/denfis_b2/r*f*.csv` and `results/preds/denfis_b2/holdout.csv`
- `results/metrics/denfis_b2/*.json`
- `results/summary.csv` and `results/manifests/denfis_b2.json`

## Change control

- No feature, hyperparameter, fold, cost or threshold change after this point.
- Results are reported as computed; a negative test or holdout result is not a
  reason to revisit the configuration.
- Any correction must be a documentation fix that does not alter a number, or a
  documented protocol change with a new freeze.
