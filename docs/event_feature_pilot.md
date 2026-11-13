# Event/calendar feature pilot (validation only, plus a descriptive test)

Run by `scripts/event_feature_pilot.py`. Fit on labels through Y-2, score the
validation year Y-1. The test years and the 2026 holdout are not used for any
selection.

Ex-dividend dates are inferred from positive `Adj Close / Close` jumps above
0.5% (48 dates). `is_post_exdiv_1` marks the first trading day after an
ex-date. Feature set A is the two ex-dividend dummies; set B adds earnings
month, month-end and Monday.

## DENFIS (one rule, `ma5`, `boll_b`, `ret_0`)

| feature set | mean validation skill | folds > 0 | delta vs base |
|---|---:|---:|---:|
| base | +0.212% | 6/9 | +0.000% |
| A_exdiv | +0.432% | 7/9 | +0.220% |
| B_exdiv_calendar | +0.259% | 5/9 | +0.048% |

## XGBoost (shallow, 13 features)

| feature set | mean validation skill | folds > 0 | delta vs base |
|---|---:|---:|---:|
| base | +0.233% | 6/9 | +0.000% |
| A_exdiv | +0.220% | 6/9 | -0.013% |
| B_exdiv_calendar | +0.198% | 6/9 | -0.035% |

## LSTM (small network, 13 features)

| feature set | mean validation skill | folds > 0 | delta vs base |
|---|---:|---:|---:|
| base | +0.520% | 8/9 | +0.000% |
| A_exdiv | +0.465% | 7/9 | -0.055% |
| B_exdiv_calendar | +0.516% | 8/9 | -0.004% |

## Descriptive test of the DENFIS feature sets

The same one-rule configuration was fitted on train + validation and scored on
the test years (2017-2025). The test years were already inspected during the
earlier diagnostics, so this is a descriptive check, not a clean confirmation.

| feature set | test RMSE skill | fold-bootstrap 95% CI | folds > 0 |
|---|---:|---|---:|
| base | +0.264% | [-0.142%, +0.605%] | 6/9 |
| A_exdiv | +0.510% | [+0.074%, +0.982%] | 7/9 |
| B_exdiv_calendar | +0.399% | [-0.060%, +0.999%] | 6/9 |

## Reading

- The ex-dividend features help **DENFIS only**: +0.212% to +0.432% mean
  validation skill, 7/9 folds positive. XGBoost is unchanged-to-slightly worse
  and LSTM is slightly worse.
- The univariate `is_post_exdiv_1` rule was +0.324% with 9/9 validation folds,
  but the model gain is smaller; the event sample is only 48 dates, so the
  estimate is fragile.
- Adding calendar features (set B) dilutes DENFIS. The ex-dividend pair is the
  only promising set.
- The descriptive test gives +0.510% (7/9) for the ex-dividend pair versus
  +0.264% (6/9) for the base features. The fold-bootstrap interval for the
  ex-dividend pair is wide and close to zero, and the test years were already
  seen during the diagnostics, so this is not a clean confirmation.
- LSTM is still higher on validation (+0.520% base) than DENFIS with the
  ex-dividend pair (+0.432%), so the point-estimate test ordering is not
  guaranteed by the validation data.
- If adopted, the ex-dividend features must be pre-registered and evaluated
  once on the 2026 holdout. Do not select them on 2017-2025.

## Reproduction

```bash
python scripts/event_feature_pilot.py
python scripts/event_feature_pilot.py --test-only
```
