# AI6124 consultation handout — 11 November 2026

**Team:** HU Junyu / LIN Jingzhao. **Task:** next-day DBS log-return forecasting.

## Architecture and evaluation

13 causal features -> training-only 5/95 winsorisation and z-score -> ECM
clusters -> Gaussian activations -> first-order TSK rules -> weighted RLS.

For test year Y: selection training through Y-2 -> validation in Y-1 -> final
refit through Y-1 -> test in Y. Nine tests: 2017–2025. The 2026 holdout is
reserved for the scheduled final freeze; baseline holdout artifacts already
existed before method integration and are not used for our model selection.

## Preliminary evidence (not frozen)

Frozen DENFIS has completed all nine folds: pooled RMSE 0.012066, DA 0.5009,
Sharpe 0.6541, maximum drawdown 20.3%, exposure 34.4%. Historical baseline
Sharpes: XGBoost 0.2376, LSTM 0.4066, buy-and-hold 1.1125. Thus a larger
strategy Sharpe than the fitted baselines is not evidence of lower forecast
error or superiority to buying and holding. Statistical uncertainty is reported
in docs/statistical_tests.md after both model runs finish.

## Decisions requested from instructor

1. Is frozen as the main comparison and predict-then-update online adaptation
   as a separate extension acceptable?
2. How should a self-written established method and the cap/merge/online
   extensions be described for the bonus criterion?
3. Are long/flat and one-way 10 bp costs appropriate, and how should forecast
   error versus strategy risk/return be discussed?

Answers must be recorded after the actual discussion in docs/consultation.md.
