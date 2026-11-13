# Is the requirement statistically resolvable?

Diagnostic run by `scripts/detectable_skill.py`. The zero forecast is the
benchmark; `mse_zero = 1.4340e-04` and `rmse_zero = 0.011975`
over the nine test years (n = 2258). Test years are used for
evaluation only; nothing here selects a model, feature or fold, and the 2026
holdout is untouched.

## What "minimum detectable effect" means

For each candidate we form the daily squared-error difference against the zero
forecast, estimate its Newey-West standard error (lag 7), and compute the
RMSE improvement that would be detected with 80% power at a two-sided 5% test.
`power at observed` is the probability of a significant result if the observed
effect were the true effect. Effective n below the nominal 2,258 indicates how
much autocorrelation inflates the standard error (values above 2,258 indicate
slightly negative autocorrelation).

## Results

| rule | observed skill | NW SE (MSE) | DM p | MDE at 80% power | power at observed | effective n |
|---|---:|---:|---:|---:|---:|---:|
| ma5 | +0.393% | 5.12e-07 | 0.028 | +0.501% | 0.59 | 2394 |
| ret_3 | +0.145% | 2.37e-07 | 0.079 | +0.232% | 0.42 | 1746 |
| constant mean | +0.148% | 2.31e-07 | 0.066 | +0.226% | 0.45 | 1773 |
| denfis_frozen | -0.759% | 1.28e-06 | 0.088 | +1.260% | 0.40 | 2310 |
| xgboost | -0.283% | 8.20e-07 | 0.322 | +0.804% | 0.17 | 2503 |
| lstm | -0.569% | 1.07e-06 | 0.127 | +1.054% | 0.33 | 3715 |

## Paired comparison: DENFIS against XGBoost

- Observed RMSE gap: **+0.47%** (positive means DENFIS is worse).
- Mean squared-error difference: +1.37e-06, Newey-West SE 1.12e-06.
- DM statistic +1.23, p = 0.218.
- MDE for the gap at 80% power: **+1.10%**.

## Reading

- **The best observed effect is below the detection threshold.** The strongest
  rule (`ma5`) improves RMSE by +0.39%,
  while the study needs about
  +0.50% to reach 80% power.
  Its power at the observed effect is only
  0.59.
- **DENFIS versus XGBoost is not resolvable either.** The observed gap is
  +0.47% but the MDE is +1.10%; a true gap of this
  size would be detected only about 23% of the time.
- **The baseline clause is a demand on the baselines, not on DENFIS.** To make
  `xgboost < zero` detectable at 80% power needs an improvement of about
  +0.80%; LSTM needs about
  +1.05%. Both currently have the
  wrong sign.
- **Sample size is the binding constraint.** Detecting a true 0.4% RMSE edge at
  80% power would need roughly 3,500–3,700 trading days (about 15 years), or a
  larger effect. The nine test years are enough to rule out a large edge, not
  to establish a small one.

## Caveat

The MDE uses daily squared-error differences and a Newey-West correction. It
does not model cross-fold dependence explicitly; the fold-level bootstrap in
`docs/feature_information_check.md` gives wider intervals, so these MDEs are
optimistic. The conclusion is therefore conservative: the sample is, if
anything, less able to resolve the requirement than the table suggests.

## Implication for the consultation

When the brief says "better", it matters whether it means a point-estimate
ordering or a statistically established improvement. For effects below about
1% RMSE, this sample cannot reliably distinguish the two. That is the question
to put to the instructor, together with the no-prediction comparison.

## Reproduction

```bash
python scripts/detectable_skill.py
```
