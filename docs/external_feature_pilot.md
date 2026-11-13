# Overnight external-feature pilot (validation only)

Run by `scripts/external_feature_pilot.py`; data cached by
`scripts/download_external.py`. For each Singapore date, the last US/FRED
observation strictly before it is used, so the features are known at the
Singapore open. Validation years only; the test years and 2026 holdout are
untouched.

## DENFIS (one rule, `ma5`, `boll_b`, `ret_0`)

| feature set | mean validation skill | folds > 0 | delta vs base |
|---|---:|---:|---:|
| base | +0.212% | 6/9 | +0.000% |
| S1_xlf | +0.156% | 6/9 | -0.056% |
| S2_bac | +0.198% | 6/9 | -0.014% |
| S3_bac_fx | +0.174% | 6/9 | -0.038% |
| S4_bac_fx_vix | +0.159% | 5/9 | -0.052% |
| S5_vix | +0.272% | 6/9 | +0.060% |
| S6_jpm | +0.173% | 6/9 | -0.039% |

## XGBoost (shallow, 13 features)

| feature set | mean validation skill | folds > 0 | delta vs base |
|---|---:|---:|---:|
| base | +0.233% | 6/9 | +0.000% |
| S1_xlf | +0.219% | 6/9 | -0.014% |
| S2_bac | +0.225% | 6/9 | -0.008% |
| S3_bac_fx | +0.217% | 6/9 | -0.016% |
| S4_bac_fx_vix | +0.224% | 6/9 | -0.009% |
| S5_vix | +0.234% | 6/9 | +0.001% |
| S6_jpm | +0.233% | 7/9 | -0.000% |

## LSTM (small network, 13 features)

| feature set | mean validation skill | folds > 0 | delta vs base |
|---|---:|---:|---:|
| base | +0.520% | 8/9 | +0.000% |
| S5_vix | -0.001% | 7/9 | -0.521% |

## Reading

- No feature set improves all three models. `vix_chg` is the only set that
  improves DENFIS (+0.060 pp) and XGBoost (+0.001 pp), but it destroys LSTM
  (+0.520% to -0.001%).
- The univariate validation signals were positive (`bac_ret` +0.235%, `vix_chg`
  +0.297%), but adding them to the models does not translate into a joint
  improvement. This is the same overfitting pattern as the technical features.
- FOMC historical dates were not available in the accessible public CSV (only
  future 2026-2027 meetings); earnings dates were approximated by the earnings
  month in `docs/event_feature_pilot.md`, where they diluted the signal.
- Per the rule "if all three models do not improve on validation, stop", this
  direction is closed. Do not evaluate it on the test years or the 2026
  holdout.

## Reproduction

```bash
python scripts/download_external.py
python scripts/external_feature_pilot.py
```
