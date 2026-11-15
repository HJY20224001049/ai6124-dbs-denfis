# Results

## Pooled, 2017–2025

| model | n | rmse | mae | r2 | ic | da | da5 | sharpe | ann_ret | max_dd | turnover | exposure |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| denfis_b2 | 2258 | 0.0119 | 0.0084 | 0.0077 | 0.0949 | 0.5151 | 0.5257 | 0.5375 | 0.0622 | 0.2777 | 0.1940 | 0.4154 |
| denfis_frozen | 2258 | 0.0121 | 0.0086 | -0.0194 | 0.0357 | 0.5009 | 0.5117 | 0.6541 | 0.0701 | 0.2026 | 0.2095 | 0.3437 |
| lstm | 2258 | 0.0120 | 0.0085 | -0.0156 | 0.0061 | 0.4845 | 0.4949 | 0.4066 | 0.0385 | 0.2752 | 0.1382 | 0.3074 |
| xgboost | 2258 | 0.0120 | 0.0085 | -0.0098 | -0.0094 | 0.5084 | 0.5191 | 0.2376 | 0.0237 | 0.3219 | 0.1926 | 0.5359 |

## Forecast accuracy vs no prediction (pooled, 2017–2025)

No prediction is the zero-return forecast (`y_pred = 0`): tomorrow's adjusted close equals today's. Positive skill means the model beats that random-walk forecast; negative means it is worse.

| model | n | rmse | rmse_zero | rmse_skill | mae | mae_zero | mae_skill |
|---|---|---|---|---|---|---|---|
| no_prediction | 2258 | 0.011975 | 0.011975 | +0.00% | 0.008430 | 0.008430 | +0.00% |
| denfis_b2 | 2258 | 0.011904 | 0.011975 | +0.59% | 0.008398 | 0.008430 | +0.38% |
| denfis_frozen | 2258 | 0.012066 | 0.011975 | -0.76% | 0.008564 | 0.008430 | -1.58% |
| xgboost | 2258 | 0.012009 | 0.011975 | -0.28% | 0.008487 | 0.008430 | -0.66% |
| lstm | 2258 | 0.012043 | 0.011975 | -0.57% | 0.008526 | 0.008430 | -1.13% |

## Extension: test-time online adaptation (reported separately)

Predict first, update only after the label is realised. These models are not ranked in the frozen main comparison.

| model | n | rmse | mae | r2 | ic | da | da5 | sharpe | ann_ret | max_dd | turnover | exposure |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| denfis_online | 2258 | 0.0122 | 0.0086 | -0.0397 | 0.0012 | 0.5018 | 0.5121 | 0.2595 | 0.0235 | 0.2461 | 0.2356 | 0.3578 |

## Holdout, 2026 Jan–Oct

The frozen model (B2) was run once on this period after the recorded freeze and is reported whatever it shows; baseline holdout predictions predate the method line and are kept for reference. None of these numbers guided selection.

| model | n | rmse | mae | r2 | ic | da | da5 | sharpe | ann_ret | max_dd | turnover | exposure |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| denfis_b2 | 209 | 0.0101 | 0.0074 | -0.0175 | -0.0163 | 0.5215 | 0.5271 | 1.7123 | 0.2836 | 0.0882 | 0.1196 | 0.8995 |
| lstm | 209 | 0.0100 | 0.0072 | 0.0073 | 0.0962 | 0.5885 | 0.5911 | 2.1474 | 0.3247 | 0.0885 | 0.1914 | 0.7225 |
| xgboost | 209 | 0.0101 | 0.0073 | -0.0040 | 0.0561 | 0.5359 | 0.5419 | 2.1433 | 0.3861 | 0.0895 | 0.0909 | 0.9426 |

## Trading performance vs buy & hold

### Pooled, 2017–2025

| model | n | sharpe | ann_ret | max_dd | turnover | exposure |
|---|---|---|---|---|---|---|
| denfis_b2 | 2258 | 0.5375 | 0.0622 | 0.2777 | 0.1940 | 0.4154 |
| denfis_frozen | 2258 | 0.6541 | 0.0701 | 0.2026 | 0.2095 | 0.3437 |
| lstm | 2258 | 0.4066 | 0.0385 | 0.2752 | 0.1382 | 0.3074 |
| xgboost | 2258 | 0.2376 | 0.0237 | 0.3219 | 0.1926 | 0.5359 |
| buy_hold | 2258 | 1.1125 | 0.2132 | 0.3860 | 0.0000 | 1.0000 |

### Holdout, 2026 Jan–Oct

| model | n | sharpe | ann_ret | max_dd | turnover | exposure |
|---|---|---|---|---|---|---|
| denfis_b2 | 209 | 1.7123 | 0.2836 | 0.0882 | 0.1196 | 0.8995 |
| lstm | 209 | 2.1474 | 0.3247 | 0.0885 | 0.1914 | 0.7225 |
| xgboost | 209 | 2.1433 | 0.3861 | 0.0895 | 0.0909 | 0.9426 |
| buy_hold | 209 | 2.3372 | 0.4344 | 0.0917 | 0.0000 | 1.0000 |

## Reading notes

- Pooled R² is positive for denfis_b2 and negative for the others (denfis_b2 0.0077, denfis_frozen -0.0194, lstm -0.0156, xgboost -0.0098); the test years are not uniformly favourable.
- Pooled Sharpe ranking: denfis_frozen 0.65 (exposure 34%), denfis_b2 0.54 (exposure 42%), lstm 0.41 (exposure 31%), xgboost 0.24 (exposure 54%).
- Directional accuracy: denfis_b2 0.515, denfis_frozen 0.501, lstm 0.484, xgboost 0.508 (0.5 is a coin flip).
- Pooled maximum drawdown: denfis_b2 27.8%, denfis_frozen 20.3%, lstm 27.5%, xgboost 32.2%, versus 38.6% for buy & hold. Lower drawdown only counts as an improvement if the Sharpe is not sacrificed.
- RMSE skill vs no prediction: denfis_b2 +0.59%, denfis_frozen -0.76%, lstm -0.57%, xgboost -0.28%. A non-positive value means the model does not beat assuming no price change.
- The 2026 holdout is a single period reported separately; it is not used for model selection and should not be treated as a second validation set.
