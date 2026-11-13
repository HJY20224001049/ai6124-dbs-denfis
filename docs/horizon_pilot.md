# Horizon pilot: 1 day versus 5 days

Run by `scripts/horizon_pilot.py`. The test years are evaluated, never used
for selection; the 2026 holdout is untouched. All model choices were made on
validation data: the three features (`ma5`, `boll_b`, `ret_0`) have the
highest validation skill; one-rule DENFIS uses `lam = 1.0`, which is better
than `lam = 0.99` on validation; the shallow XGBoost configuration has the
lowest mean validation RMSE in its grid; the LSTM keeps the protocol's
selected configuration.

## One-day forward return

| model | pooled RMSE skill | fold-bootstrap 95% CI | folds > 0 |
|---|---:|---|---:|
| const | +0.148% | [+0.039%, +0.299%] | 7/9 |
| denfis | +0.264% | [-0.142%, +0.605%] | 6/9 |
| xgboost | +0.212% | [-0.028%, +0.432%] | 6/9 |
| lstm | -0.569% | [-1.062%, +0.086%] | 2/9 |

## Five-day forward return

| model | pooled RMSE skill | fold-bootstrap 95% CI | folds > 0 |
|---|---:|---|---:|
| const | +0.606% | [+0.137%, +1.318%] | 6/9 |
| denfis | +0.822% | [+0.213%, +1.362%] | 6/9 |
| xgboost | +0.463% | [-0.059%, +1.095%] | 7/9 |
| lstm | -0.394% | [-2.039%, +0.821%] | 2/9 |

## Reading

- At one day, the ordering fails: DENFIS (+0.26%) and XGBoost (+0.21%) beat
  zero on point estimates, but LSTM is −0.57%. LSTM is the binding constraint.
- At five days, DENFIS (+0.82%) and XGBoost (+0.46%) beat zero, but LSTM is
  −0.39%. The ordering still fails on LSTM. DENFIS beats the constant
  historical mean (+0.22 percentage points of RMSE skill); XGBoost does not.
- The five-day result is largely a drift effect. A constant mean forecast
  already earns about +0.61% RMSE skill; zero is a weak benchmark over a
  five-day horizon.
- Five-day returns overlap. The effective sample is roughly one fifth of the
  row count, and the fold bootstrap does not capture within-fold overlap, so
  the intervals above are optimistic.
- The LSTM keeps the protocol's one-day selected configuration; it was not
  re-selected for the five-day target. A target-specific LSTM selection is the
  remaining untested route, and the LSTM is unstable across epochs.
- The five-day horizon changes the assignment from next-day to next-week
  prediction. It needs the instructor's written approval before it can
  replace the main protocol.

## Recommendation

Present this as a horizon-robustness result, not as a silent protocol change.
If the instructor approves a five-day horizon and a target-specific LSTM
re-selection, pre-register the models and evaluate once on the 2026 holdout.
Otherwise, neither the one-day nor the five-day task satisfies the stated
ordering with the current LSTM.

## Reproduction

```bash
python scripts/horizon_pilot.py
```
