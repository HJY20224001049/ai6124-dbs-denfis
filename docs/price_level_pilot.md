# Price level or price difference? A validation-only pilot

**Question.** The assignment slides say "predicting the stock prices" (slide 27) but do not define whether the target is the price level or the price change. Our approved proposal specifies next-day log-return. This pilot asks what happens if we take "price" literally, before changing the protocol.

**Design.** Nine walk-forward folds; for each fold, fit on the selection-training window and score on the validation year only (2016–2024). No test index is used. The no-prediction forecast is the last observed adjusted close, `P_t`. Direct price models receive the 13 stationary features plus an unclipped price anchor (raw `P_t` is the more favourable case); reconstructed models fit log-return or price difference on the 13 features and rebuild the price.

## Results (pooled validation years)

| Forecast | Price RMSE | Price skill | Return skill | Folds won |
|---|---:|---:|---:|---:|
| No prediction (`P_t`) | 0.2207 | 0.00% | 0.00% | — |
| Constant drift | 0.2203 | +0.20% | +0.15% | 6/9 |
| DENFIS, direct price | 0.2206 | +0.07% | −0.20% | 5/9 |
| DENFIS, price difference + `P_t` | 0.2204 | +0.16% | −0.05% | 5/9 |
| DENFIS, log-return → `P_t·exp(·)` | 0.2205 | +0.10% | −0.18% | 3/9 |
| XGBoost, direct price | 3.1020 | −1305% | −873% | 0/9 |
| XGBoost, price difference + `P_t` | 0.2213 | −0.24% | −0.25% | 4/9 |
| XGBoost, log-return → `P_t·exp(·)` | 0.2232 | −1.10% | −1.45% | 2/9 |
| Ridge, direct price | 2.6489 | −1100% | −961% | 0/9 |

## Findings

1. **Direct price forecasting is dominated by the random walk.** Even the best direct model (DENFIS with a raw price anchor) essentially learns `P̂ ≈ P_t`; it improves price RMSE by only 0.07%, and the same forecast is worse than no prediction in return space (−0.20%). A univariate OLS on `P_t` learns a coefficient of 1.0001 and ties the random walk (skill ≈ 0).

2. **XGBoost cannot extrapolate a price level.** In 5 of 9 validation years, 66–90% of validation prices lie above the training maximum (a sixth year has 9%). In 2024 it predicts a mean of 25.27 against an actual 31.57 (RMSE 7.66 versus 0.35 for `P_t`). This is structural, not a tuning issue.

3. **Price RMSE is a price-weighted return metric.** A positive price skill can coexist with a negative return skill: the gain is concentrated in high-price years, not in daily forecasting. Reporting only price RMSE would overstate the result.

4. **The only consistently positive forecast is a constant drift** (+0.20% price, +0.15% return), which is not a learned daily model.

## Recommendation

- Keep **next-day log-return** as the main target, as in the approved proposal.
- If the deliverable must show a price forecast, reconstruct it from the return model: `P̂_{t+1} = P_t · exp(ŷ_t)`, and report **both** price RMSE and return RMSE. Do not evaluate XGBoost on a raw price level.
- If the target itself must be the price level, the protocol needs an explicit amendment and a model class that can extrapolate. The current XGBoost baseline cannot.

## Reproduction

```bash
python scripts/price_level_pilot.py
```

Per-fold detail: `results/analysis/price_level_pilot.csv`.
