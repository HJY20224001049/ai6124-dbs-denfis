# Validation-only candidates (original target)

Run by `scripts/validation_candidates.py`. The target is unchanged:
`y_t = ln(AdjClose_{t+1} / AdjClose_{t})`. Each model is fitted on the
selection-training window (labels through Y-2) and scored on the validation
year (Y-1). The test years and the 2026 holdout are never indexed.

- B1: one-rule DENFIS on `ma5` plus `is_exdiv`, `is_post_exdiv_1`.
- B2: equal-weight average of three one-rule DENFIS models, each on one of
  `ma5`, `boll_b`, `ret_0`, plus the ex-dividend dummies.
- Reference rows: `ma5` alone, the three-feature subset with/without dummies.

| candidate | mean validation skill | median | folds > 0 | pooled skill | pooled 95% CI |
|---|---:|---:|---:|---:|---|
| B1_ma5_only | +0.246% | +0.545% | 5/9 | +0.317% | [-0.076%, +0.604%] |
| B1_ma5_exdiv | +0.475% | +0.560% | 6/9 | +0.486% | [+0.101%, +0.807%] |
| B2_equal3_base | +0.269% | +0.234% | 6/9 | +0.311% | [-0.028%, +0.578%] |
| B2_equal3_exdiv | +0.497% | +0.616% | 8/9 | +0.480% | [+0.161%, +0.780%] |
| ref_3feat_base | +0.212% | +0.180% | 6/9 | +0.235% | [-0.163%, +0.569%] |
| ref_3feat_exdiv | +0.432% | +0.456% | 7/9 | +0.399% | [+0.024%, +0.784%] |

## Reading

The validation set is the only selection window here. The comparison with the
LSTM is against its validation-selected configuration (+0.520% mean validation
skill, 8/9 folds), which is itself a maximum over 16 candidates and therefore
optimistic. A candidate is only worth freezing if it improves on the current
three-feature + ex-dividend variant without relying on the test years.

## Reproduction

```bash
python scripts/validation_candidates.py
```
