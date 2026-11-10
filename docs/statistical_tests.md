# Statistical tests

Pooled out-of-sample results, 2017–2025 (nine walk-forward folds; the 2026 holdout is excluded).

## Pooled performance and Sharpe confidence intervals

| model | n | rmse | mae | r2 | ic | da | sharpe | sharpe_95ci |
|---|---|---|---|---|---|---|---|---|
| lstm | 2258 | 0.0120 | 0.0085 | -0.0156 | 0.0061 | 0.4845 | 0.4066 | [-0.285, 1.114] |
| xgboost | 2258 | 0.0120 | 0.0085 | -0.0098 | -0.0094 | 0.5084 | 0.2376 | [-0.474, 0.941] |

Sharpe CI: stationary block bootstrap, average block 20 days, 2000 resamples, seed 123 (protocol section 6.5).

## Diebold–Mariano test (squared-error loss, Newey–West)

Positive DM statistic = model A has the larger squared errors (worse forecast). Lag uses the standard rule `floor(4 * (n/100)^(2/9))`; n here is the aligned pooled sample.

| model_a | model_b | mean_loss_diff | dm_stat | p_value | lag | n | outcome |
|---|---|---|---|---|---|---|---|
| xgboost | lstm | -8.25e-07 | -0.7470 | 0.4551 | 7 | 2258 | not significant |

## Reading the table

- `mean_loss_diff` is mean squared error of A minus that of B; a positive value means A is worse.
- A significant positive DM statistic means the better model is genuinely more accurate, not just luckier on these folds.
- The Sharpe interval resamples contiguous blocks of days, preserving some autocorrelation; an interval spanning zero means the trading result is not distinguishable from no edge.

> **Placeholder:** DENFIS predictions are not on disk yet, so the DM row compares the two best baselines. Re-run this script after the DENFIS folds land; the pair selection will switch automatically to DENFIS vs the best baseline.
