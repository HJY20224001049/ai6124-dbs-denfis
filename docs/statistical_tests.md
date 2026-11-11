# Statistical tests

Pooled out-of-sample results, 2017–2025 (nine walk-forward folds; the 2026 holdout is excluded).

## Pooled performance and Sharpe confidence intervals

| model | n | rmse | rmse_zero | rmse_skill | mae | r2 | ic | da | sharpe | sharpe_95ci |
|---|---|---|---|---|---|---|---|---|---|---|
| denfis_frozen | 2258 | 0.0121 | 0.0120 | -0.76% | 0.0086 | -0.0194 | 0.0357 | 0.5009 | 0.6541 | [-0.041, 1.309] |
| denfis_online | 2258 | 0.0122 | 0.0120 | -1.76% | 0.0086 | -0.0397 | 0.0012 | 0.5018 | 0.2595 | [-0.413, 0.889] |
| lstm | 2258 | 0.0120 | 0.0120 | -0.57% | 0.0085 | -0.0156 | 0.0061 | 0.4845 | 0.4066 | [-0.285, 1.114] |
| no_prediction | 2258 | 0.0120 | 0.0120 | +0.00% | 0.0084 | -0.0041 | 0.0000 | 0.0204 | 0.0000 | [0.000, 0.000] |
| xgboost | 2258 | 0.0120 | 0.0120 | -0.28% | 0.0085 | -0.0098 | -0.0094 | 0.5084 | 0.2376 | [-0.474, 0.941] |

Sharpe CI: stationary block bootstrap, average block 20 days, 2000 resamples, seed 123 (protocol section 6.5). The `no_prediction` row is the zero-return forecast; its trading column is flat (cash), not buy-and-hold. Use the buy-and-hold figures in the results table for the trading comparison.

## Diebold–Mariano test (squared-error loss, Newey–West)

Positive DM statistic = model A has the larger squared errors (worse forecast). Lag uses the standard rule `floor(4 * (n/100)^(2/9))`; n here is the aligned pooled sample.

| model_a | model_b | mean_loss_diff | dm_stat | p_value | lag | n | outcome |
|---|---|---|---|---|---|---|---|
| denfis_frozen | no_prediction | +2.18e-06 | 1.7039 | 0.08841 | 7 | 2258 | not significant |
| denfis_online | no_prediction | +5.08e-06 | 1.7373 | 0.08234 | 7 | 2258 | not significant |
| xgboost | no_prediction | +8.12e-07 | 0.9899 | 0.3222 | 7 | 2258 | not significant |
| lstm | no_prediction | +1.64e-06 | 1.5250 | 0.1273 | 7 | 2258 | not significant |
| denfis_frozen | xgboost | +1.37e-06 | 1.2306 | 0.2185 | 7 | 2258 | not significant |
| denfis_online | xgboost | +4.27e-06 | 1.5092 | 0.1312 | 7 | 2258 | not significant |
| denfis_online | denfis_frozen | +2.90e-06 | 1.0932 | 0.2743 | 7 | 2258 | not significant |

## Reading the table

- `mean_loss_diff` is mean squared error of A minus that of B; a positive value means A is worse.
- A significant DM result is evidence of a mean loss difference under the test assumptions. Comparisons are exploratory and not multiplicity-adjusted; the reference baseline is selected by pooled RMSE.
- The Sharpe interval resamples contiguous blocks of days, preserving some autocorrelation; an interval spanning zero means the trading result is not distinguishable from no edge.
