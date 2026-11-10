# Result freeze register — NOT FROZEN

Prepared 2026-11-10. Scheduled decision: 2026-11-15 22:00 Asia/Singapore.
This file is a candidate artifact checklist, not evidence that the future freeze
or an instructor consultation has occurred. No DENFIS holdout has been run.

## Candidate artifacts

- Main: results/preds/denfis_frozen/r*f*.csv and corresponding metrics; docs/results_table.md.
- Online extension: results/preds/denfis_online/r*f*.csv, separately labelled.
- Statistics: docs/statistical_tests.md, results/statistical_tests.json.
- Transfer: results/transfer/{preds,metrics}/{scratch,zero_shot,fine_tune}/.
- Ablations: results/analysis/ablations.csv and variant predictions/metrics.
- Sensitivity: results/analysis/validation_grid.csv (validation profiles only).
- Interpretation: results/figures/denfis_*.png and results/analysis/r3f3_rules.npz.

## Before formal freeze

- Record actual consultation outcome and any jointly agreed protocol changes.
- Complete cross-review of methodology, implementation differences, and artifacts.
- Verify every result against saved predictions, with tests and leakage audit.
- Record the final commit and final table/figure list here.
- Only then run the final DENFIS holdout once under the agreed protocol.
- Integrate and reproduce the final combined notebook; rehearse; each student submits.
