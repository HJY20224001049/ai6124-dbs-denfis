# Baseline results

## Pooled, 2017–2025

| model | n | rmse | mae | r2 | ic | da | da5 | sharpe | ann_ret | max_dd | turnover | exposure |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lstm | 2258 | 0.0120 | 0.0085 | -0.0156 | 0.0061 | 0.4845 | 0.4949 | 0.4066 | 0.0385 | 0.2752 | 0.1382 | 0.3074 |
| xgboost | 2258 | 0.0120 | 0.0085 | -0.0098 | -0.0094 | 0.5084 | 0.5191 | 0.2376 | 0.0237 | 0.3219 | 0.1926 | 0.5359 |

## Holdout, 2026 Jan–Oct

| model | n | rmse | mae | r2 | ic | da | da5 | sharpe | ann_ret | max_dd | turnover | exposure |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lstm | 209 | 0.0100 | 0.0072 | 0.0073 | 0.0962 | 0.5885 | 0.5911 | 2.1474 | 0.3247 | 0.0885 | 0.1914 | 0.7225 |
| xgboost | 209 | 0.0101 | 0.0073 | -0.0040 | 0.0561 | 0.5359 | 0.5419 | 2.1433 | 0.3861 | 0.0895 | 0.0909 | 0.9426 |

## Trading performance vs buy & hold

### Pooled, 2017–2025

| model | n | sharpe | ann_ret | max_dd | turnover | exposure |
|---|---|---|---|---|---|---|
| lstm | 2258 | 0.4066 | 0.0385 | 0.2752 | 0.1382 | 0.3074 |
| xgboost | 2258 | 0.2376 | 0.0237 | 0.3219 | 0.1926 | 0.5359 |
| buy_hold | 2258 | 1.1125 | 0.2132 | 0.3860 | 0.0000 | 1.0000 |

### Holdout, 2026 Jan–Oct

| model | n | sharpe | ann_ret | max_dd | turnover | exposure |
|---|---|---|---|---|---|---|
| lstm | 209 | 2.1474 | 0.3247 | 0.0885 | 0.1914 | 0.7225 |
| xgboost | 209 | 2.1433 | 0.3861 | 0.0895 | 0.0909 | 0.9426 |
| buy_hold | 209 | 2.3372 | 0.4344 | 0.0917 | 0.0000 | 1.0000 |

## Reading notes

- Pooled R² is below zero for every model (lstm -0.0156, xgboost -0.0098): next-day log returns are close to unpredictable with this feature set.
- Pooled Sharpe ranking: lstm 0.41 (exposure 31%), xgboost 0.24 (exposure 54%).
- Directional accuracy: lstm 0.484, xgboost 0.508 (0.5 is a coin flip).
- Pooled maximum drawdown: lstm 27.5%, xgboost 32.2%, versus 38.6% for buy & hold. Lower drawdown only counts as an improvement if the Sharpe is not sacrificed.
- The 2026 holdout is a single period reported separately; it is not used for model selection and should not be treated as a second validation set.
