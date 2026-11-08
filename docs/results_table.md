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

- Both models have pooled R² below zero and directional accuracy near 50%: next-day log returns are close to unpredictable at this feature set.
- LSTM has the higher pooled Sharpe (0.41 vs 0.24) because it trades less (31% exposure vs 54%) and takes a lower turnover; XGBoost has the higher directional accuracy (0.508 vs 0.485).
- Both long/flat strategies cut maximum drawdown relative to buy & hold (27.5% and 32.2% vs 38.6%) but give up most of the upside: buy & hold has a pooled Sharpe of 1.11 and 465% cumulative return over 2017–2025.
- The 2026 holdout is a single favourable period for both models; it is reported separately and is not used for model selection.
