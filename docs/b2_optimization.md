# B2 incremental optimization (validation only)

Run by `scripts/b2_optimization.py`. B2 is the equal-weight average of three
one-rule DENFIS models on `ma5`, `boll_b`, `ret_0`, each with the two
ex-dividend dummies. The target is unchanged:
`y_t = ln(AdjClose_{t+1} / AdjClose_{t})`. Each candidate is fitted on the
selection-training window (labels through Y-2) and scored on the validation
year (Y-1); the test years and the 2026 holdout are never indexed.

Adoption rule, fixed before the run: a variant replaces B2 only if mean
validation skill improves by at least 0.05 pp and the positive-fold count does
not fall. With nine validation years and this many candidates, any gain is
still optimistic; the 2026 holdout remains the clean test.

| candidate | mean validation skill | delta vs B2 | median | folds > 0 | pooled skill | pooled 95% CI |
|---|---:|---:|---:|---:|---:|---|
| B2_shrink125 | +0.511% | +0.015% | +0.650% | 7/9 | +0.507% | [+0.093%, +0.879%] |
| B2_invvar | +0.497% | +0.000% | +0.616% | 8/9 | +0.480% | [+0.161%, +0.780%] |
| B2_base | +0.497% | +0.000% | +0.616% | 8/9 | +0.480% | [+0.161%, +0.780%] |
| B2_k5 | +0.495% | -0.002% | +0.488% | 8/9 | +0.464% | [+0.221%, +0.733%] |
| B2_winsor | +0.492% | -0.004% | +0.500% | 8/9 | +0.463% | [+0.210%, +0.738%] |
| B2_median | +0.489% | -0.008% | +0.522% | 8/9 | +0.475% | [+0.146%, +0.783%] |
| B2_k7 | +0.482% | -0.014% | +0.371% | 8/9 | +0.452% | [+0.220%, +0.712%] |
| B2_k2 | +0.482% | -0.015% | +0.569% | 7/9 | +0.487% | [+0.150%, +0.776%] |
| B2_plus_ref | +0.475% | -0.021% | +0.546% | 7/9 | +0.451% | [+0.105%, +0.793%] |
| B2_gap_intraday | +0.465% | -0.031% | +0.462% | 8/9 | +0.442% | [+0.235%, +0.680%] |
| B2_shrink075 | +0.438% | -0.058% | +0.489% | 8/9 | +0.416% | [+0.187%, +0.648%] |
| B2_post_only | +0.436% | -0.061% | +0.235% | 8/9 | +0.423% | [+0.084%, +0.731%] |
| B2_roll5 | +0.371% | -0.125% | +0.474% | 6/9 | +0.415% | [-0.071%, +0.776%] |
| B2_no_exdiv | +0.269% | -0.227% | +0.234% | 6/9 | +0.311% | [-0.028%, +0.578%] |
| B2_roll3 | +0.221% | -0.276% | -0.112% | 4/9 | +0.184% | [-0.345%, +0.762%] |
| B2_calib | -0.673% | -1.170% | -0.104% | 4/9 | -0.473% | [-2.212%, +0.626%] |

## Adoptable under the rule

None. B2 stays the candidate.

## Caveat

These are validation-year results, and the validation years have already been
used to choose the features and the ex-dividend encoding. The table is an
optimization guide, not a confirmation. Do not switch the frozen model on the
basis of any 2017-2025 test result.

## Reproduction

```bash
python scripts/b2_optimization.py
```
