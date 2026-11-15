# Fair validation scores (no selection on the scored year)

Both baselines early-stop on the outer validation year in the protocol runner, so their reported validation numbers are optimistic relative to B2 (fixed parameters). Here the selection-training window is split in time: its last year (Y-2) selects the configuration and the number of trees/epochs, and the outer validation year (Y-1) is scored once.

| model | mean skill | median | folds > 0 | pooled skill |
|---|---:|---:|---:|---:|
| lstm | -1.237% | -0.621% | 3/9 | -0.887% |
| xgboost | +0.021% | -0.036% | 4/9 | -0.029% |

## Reference point

B2 (three one-rule DENFIS components, parameters fixed): mean +0.497%, 8/9 folds, pooled +0.480% (`docs/validation_candidates.md`, `scripts/run_b2.py`).

Ordering on mean validation skill: **B2 (+0.497%) > XGBoost (+0.021%) > LSTM (−1.237%) > zero**. The same ordering holds on the pooled validation years, although the XGBoost pooled value is slightly below zero (−0.029%).

## Reading

- The protocol runner early-stops both baselines on the year it then scores. That is why the validation numbers reported earlier (for example +0.520% for the LSTM) were higher: the epoch or tree count was chosen on the scored year.
- This procedure is deliberately conservative for the baselines: the inner window is one year shorter, and two folds select a single tree, which is a symptom of a short inner window rather than of a useless model. Treat these numbers as lower bounds, not as the best the baselines can do.
- B2 is not free of selection either: its features, ex-dividend dummies and hyperparameters were chosen on validation years, and +0.497% is the best of sixteen pre-registered variants. The comparison is validation-against-validation; the 2026 holdout is the only clean confirmation.
- No test-year or holdout number appears here; none was computed.

## Reproduction

```bash
python scripts/fair_validation.py
```
