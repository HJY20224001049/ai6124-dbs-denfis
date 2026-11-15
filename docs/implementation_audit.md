# Implementation audit — is the negative result a bug?

Prepared 2026-11-15. This note answers the instructor's 2026-11-11 comment that the
result "does not qualify" and that he suspected an implementation error. It lists
what has been checked, with the command or test that checks it, and what the
checks do and do not rule out.

## 1. The target is a price difference, not a price level

`src/features.py:87` defines

```text
y_t = ln(AdjClose_{t+1} / AdjClose_t)
```

which is the next-day log return. The price-level pilot (`docs/price_level_pilot.md`)
was run only to show why a level target is the wrong task: a direct price forecast
is dominated by the random walk and the same forecast is worse than no prediction
in return space. The main experiment never trains on the level.

## 2. A single rule is a closed-form linear fit, and we can prove which one

With `max_rules=1` the normalised activation is identically 1, so the model reduces
to a linear fit on `[1, x]`. The recursive update is not plain OLS: each rule
starts with covariance `100·I` (`src/models/denfis.py:_new_rule`), which is a prior
precision of 0.01, and with `λ = 1` the recursion terminates at

```text
a = (ΦᵀΦ + 0.01·I)⁻¹ Φᵀy .
```

Checked numerically, not asserted:

| Check | Result | Where |
|---|---|---|
| Coefficients equal the closed form | max abs difference 8e-17 | `tests/test_denfis_contract.py::test_one_rule_rls_matches_regularised_least_squares` |
| Predictions equal the closed form | max abs difference 9e-16 | `…::test_one_rule_predictions_match_closed_form_ridge` |
| Distance from plain OLS on real folds | ≤ 1.4e-6 × sd(y) | `…::test_one_rule_is_indistinguishable_from_ols_at_daily_return_scale` |
| Distance from plain OLS on r1f1–r1f3 | 1.27e-6, 1.36e-6, 1.36e-6 × sd(y) | `docs/implementation_audit.md` (this run) |

So the one-rule component is not an opaque learner that could be broken in an
unknown way: it is a linear fit whose coefficients can be computed in closed form,
and they agree to machine precision. The prior is numerically negligible at this
sample size (a 0.01 addition to a Gram matrix of order 10³) but it is not zero, and
the method chapter states it as such.

## 3. The pipeline learns when a signal exists

`scripts/synthetic_signal_check.py` replaces the target with
`y = 0.25·ret_0 + N(0, 0.012)`, whose oracle R² is about 5%, and runs the same
folds, scaler and runner. RMSE skill versus the zero forecast:

| model | skill on the synthetic target |
|---|---:|
| DENFIS | +2.45% |
| XGBoost | +2.10% |
| LSTM | +1.48% |

All three learn the planted signal. The negative result on the real target is
therefore not a failure of the training or evaluation loop.

## 4. Data and folds are clean

`python scripts/leakage_audit.py` reports 151 checks, 0 failures: label dates
follow feature dates, folds are disjoint after purging, and no training label
reaches into a validation or test period. `scripts/data_integrity_check.py`
(`docs/data_integrity_check.md`) checks checksums, OHLCV sanity, corporate-action
adjustments, label alignment, feature prefix-invariance and fold disjointness; the
only anomaly is three missing DBS trading days relative to STI, one inside a test
period, and dropping the affected row moves pooled skills by at most 0.04 pp.

## What this does and does not establish

- It rules out an estimator bug of the kind that would make a correct model fail:
  the simplest configuration is exactly solvable and matches.
- It rules out a target-definition error: the label is the price difference.
- It does **not** rule out that the daily return signal is too weak for this sample.
  `docs/detectable_skill.md` reports the 80%-power minimum detectable effect at
  n = 2258: 0.50% RMSE skill for `ma5`, 0.80% for XGBoost against zero and 1.10%
  for the DENFIS–XGBoost gap. The observed effects are at or below those bounds.

## Reproduction

```bash
python -m pytest tests/test_denfis_contract.py -q
python scripts/synthetic_signal_check.py --models denfis,xgboost,lstm
python scripts/leakage_audit.py
python scripts/data_integrity_check.py
```
