# Statistical tests

Pooled out-of-sample results, 2017–2025 (nine walk-forward folds; the 2026 holdout is excluded).

## Pooled performance and Sharpe confidence intervals

| model | n | rmse | rmse_zero | rmse_skill | mae | r2 | ic | da | sharpe | sharpe_95ci |
|---|---|---|---|---|---|---|---|---|---|---|
| denfis_b2 | 2258 | 0.0119 | 0.0120 | +0.59% | 0.0084 | 0.0077 | 0.0949 | 0.5151 | 0.5375 | [-0.166, 1.207] |
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
| denfis_b2 | no_prediction | -1.69e-06 | -3.0419 | 0.002351 | 7 | 2258 | reject equal accuracy at 5% |
| denfis_frozen | no_prediction | +2.18e-06 | 1.7039 | 0.08841 | 7 | 2258 | not significant |
| denfis_online | no_prediction | +5.08e-06 | 1.7373 | 0.08234 | 7 | 2258 | not significant |
| xgboost | no_prediction | +8.12e-07 | 0.9899 | 0.3222 | 7 | 2258 | not significant |
| lstm | no_prediction | +1.64e-06 | 1.5250 | 0.1273 | 7 | 2258 | not significant |
| denfis_b2 | xgboost | -2.50e-06 | -3.4468 | 0.0005674 | 7 | 2258 | reject equal accuracy at 5% |
| denfis_frozen | xgboost | +1.37e-06 | 1.2306 | 0.2185 | 7 | 2258 | not significant |
| denfis_online | xgboost | +4.27e-06 | 1.5092 | 0.1312 | 7 | 2258 | not significant |
| denfis_online | denfis_frozen | +2.90e-06 | 1.0932 | 0.2743 | 7 | 2258 | not significant |

## Clark–West test against the zero forecast (one-sided)

The zero forecast is the restricted model nested in any linear forecast; for the tree and LSTM baselines the nesting is approximate, so read their rows as a diagnostic rather than a formal nested test. The null is that every coefficient of the model is zero, and `p_value` is one-sided. The adjusted loss differential adds `(pred_zero − pred_model)²`, which does **not** make this an MSE-improvement test: the adjustment credits the forecast's own variance, so a positive statistic alongside a worse raw RMSE is the signature of an over-dispersed forecast (right sign, too large). Use the DM rows and the confidence set for accuracy ordering; use this table only for whether the forecast carries information.

| model | mean_loss_diff | cw_stat | p_value | n | outcome |
|---|---|---|---|---|---|
| denfis_b2 | +2.66e-06 | 4.5114 | 3.22e-06 | 2258 | rejects no-information null at 5% |
| denfis_frozen | +2.33e-06 | 1.7262 | 0.04215 | 2258 | rejects no-information null at 5% |
| denfis_online | +6.84e-07 | 0.3067 | 0.3796 | 2258 | positive, not significant |
| lstm | +4.14e-07 | 0.3958 | 0.3461 | 2258 | positive, not significant |
| xgboost | +4.18e-07 | 0.5203 | 0.3014 | 2258 | positive, not significant |

## Model confidence set (range-based, α = 0.10)

Hansen, Lunde and Nason (2011): models are eliminated one at a time, largest standardised excess loss first, until the range statistic falls below its stationary-bootstrap critical value. The surviving set contains every model that cannot be separated from the best at this level. Elimination order is data-driven; a wide set is the honest outcome for daily returns.

Included: `denfis_b2`

| step | model | tr | critical |
|---|---|---|---|
| 1 | xgboost | 3.298 | 2.505 |
| 2 | no_prediction | 3.094 | 2.383 |
| 3 | denfis_frozen | 2.900 | 2.220 |
| 4 | lstm | 2.838 | 1.927 |
| 5 | denfis_online | 2.197 | 1.554 |

## Minimum detectable effect (power check)

80%-power minimum detectable RMSE skill at the pooled sample size, from `scripts/detectable_skill.py`. An observed skill below the MDE cannot be reliably distinguished from zero in this sample.

| comparison | observed_rmse_skill | mde_rmse_skill | power_at_observed | n |
|---|---|---|---|---|
| ma5 | +0.39% | 0.50% | 0.59 | 2258 |
| ret_3 | +0.15% | 0.23% | 0.42 | 2258 |
| constant mean | +0.15% | 0.23% | 0.45 | 2258 |
| denfis_frozen | -0.76% | 1.26% | 0.40 | 2258 |
| xgboost | -0.28% | 0.80% | 0.17 | 2258 |
| lstm | -0.57% | 1.05% | 0.33 | 2258 |

## Reading the table

- `mean_loss_diff` is mean squared error of A minus that of B; a positive value means A is worse.
- A significant DM result is evidence of a mean loss difference under the test assumptions. Comparisons are exploratory and not multiplicity-adjusted; the reference baseline is selected by pooled RMSE.
- The Clark–West statistic is one-sided: positive rejects the null that all model coefficients are zero. It is not an MSE-improvement test — see the note above the table.
- The model confidence set is the multiplicity-controlled summary: a model outside it is significantly worse than the best one, a model inside it is not distinguishable from the best at α = 0.10.
- The Sharpe interval resamples contiguous blocks of days, preserving some autocorrelation; an interval spanning zero means the trading result is not distinguishable from no edge.
