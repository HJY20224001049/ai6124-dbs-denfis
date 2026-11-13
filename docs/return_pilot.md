# Return-target pilot: simplified models on the one-day log return

Run by `scripts/return_pilot.py`. The target is the one-day log return, a
price difference, so the assignment stays within "price or price difference".
All choices are made on validation years; test years are evaluated, never used
for selection; the 2026 holdout is untouched.

- DENFIS: one rule, features `ma5`, `boll_b`, `ret_0`, `lam = 1.0`.
- XGBoost: depth 1, learning rate 0.01, 30 minimum child weight, lambda 5,
  100 trees.
- LSTM: configuration 1 ({'hidden_size': 8, 'dropout': 0.3, 'weight_decay': 0.01, 'epochs': 50}), selected
  from 16 candidates by mean validation RMSE; early stopping picks the epochs
  per fold.

## Test years (2017-2025)

| model | pooled RMSE skill | fold-bootstrap 95% CI | folds > 0 |
|---|---:|---|---:|
| denfis | +0.264% | [-0.142%, +0.605%] | 6/9 |
| xgboost | +0.212% | [-0.028%, +0.432%] | 6/9 |
| lstm | +0.260% | [-0.028%, +0.491%] | 6/9 |

## Per fold

| fold | year | DENFIS | XGBoost | LSTM |
|---|---:|---:|---:|---:|
| r1f1 | 2017 | +0.027% | +0.511% | -0.190% |
| r1f2 | 2018 | -0.049% | -0.299% | -0.244% |
| r1f3 | 2019 | +0.937% | +0.530% | +0.222% |
| r2f1 | 2020 | +0.390% | +0.314% | +0.553% |
| r2f2 | 2021 | -0.953% | -0.276% | -0.206% |
| r2f3 | 2022 | +0.180% | +0.034% | +0.129% |
| r3f1 | 2023 | -0.476% | -0.036% | +0.303% |
| r3f2 | 2024 | +1.091% | +0.879% | +0.503% |
| r3f3 | 2025 | +0.984% | +0.320% | +0.814% |

## Reading

- On point estimates, the ordering holds: DENFIS +0.264%
  is above LSTM +0.260%
  and XGBoost +0.212%,
  and all three are above the zero forecast.
- The margins are tiny. DENFIS leads LSTM by
  +0.004%
  and XGBoost by
  +0.052%.
  Every interval includes zero; the ordering is not statistically established.
- The pass depends on simplifying all three models. The full 13-feature
  DENFIS, XGBoost and LSTM do not beat zero; the one-rule DENFIS removes the
  evolving structure, the XGBoost is shallow, and the LSTM is small and
  heavily regularised. The report must disclose this.
- The test years have been inspected during the diagnostics. The clean
  confirmation is the 2026 holdout, evaluated once after the protocol is
  frozen.

## Recommendation

Treat this as a pre-registration candidate, not as a finished positive result.
If the instructor accepts the simplified models, freeze the protocol and
evaluate once on the 2026 holdout. Do not claim statistical superiority; the
defensible claim is "all three beat zero on point estimates, with DENFIS
nominally highest and a margin well inside the noise".

## Reproduction

```bash
python scripts/return_pilot.py
```
