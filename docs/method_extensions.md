# Method extensions — preliminary results

These exploratory experiments are separate from the nine-fold frozen main comparison. No DENFIS holdout was evaluated.

## STI to DBS transfer

Aligned label dates and separate market training scalers. Strict zero-shot uses STI validation labels for both hyperparameters and the trading threshold; scratch/fine-tune use DBS validation. Every variant is frozen during target test evaluation.

| protocol | fold_id | n | rmse | da | sharpe | n_rules |
|---|---|---|---|---|---|---|
| fine_tune | r1f1 | 250 | 0.0106 | 0.4840 | 0.7298 | 15 |
| fine_tune | r2f2 | 253 | 0.0113 | 0.4704 | 0.0474 | 26 |
| fine_tune | r3f3 | 252 | 0.0120 | 0.4960 | 1.9845 | 200 |
| scratch | r1f1 | 250 | 0.0106 | 0.5040 | 1.6395 | 11 |
| scratch | r2f2 | 253 | 0.0113 | 0.4506 | -0.8560 | 19 |
| scratch | r3f3 | 252 | 0.0120 | 0.5357 | 1.2484 | 200 |
| zero_shot | r1f1 | 250 | 0.0106 | 0.4800 | 1.1704 | 12 |
| zero_shot | r2f2 | 253 | 0.0112 | 0.4387 | -0.1288 | 22 |
| zero_shot | r3f3 | 252 | 0.0121 | 0.4762 | 1.2315 | 200 |

Fine-tuning has lower RMSE than aligned scratch in all three sampled folds, but these are only three exploratory folds. Trading improvements are not uniform. Do not compare these aligned-sample scores as if their rows were identical to the full DBS nine-fold run.

## Ablations

All ablations inherit each frozen model's selected parameters. no_rls uses weighted constants, changing consequent complexity as well as the update. r2f1 is added because it actually exercises merging.

| fold_id | variant | rmse | da | sharpe | n_rules | merges |
|---|---|---|---|---|---|---|
| r1f1 | no_rls | 0.0105 | 0.5240 | 2.4324 | 11 | 0 |
| r1f1 | no_merge | 0.0106 | 0.5000 | 0.2796 | 11 | 0 |
| r1f1 | denfis_frozen | 0.0106 | 0.5000 | 0.2796 | 11 | 0 |
| r1f1 | denfis_online | 0.0106 | 0.4880 | 0.2605 | 12 | 0 |
| r2f1 | no_rls | 0.0181 | 0.4861 | 0.2233 | 66 | 1 |
| r2f1 | no_merge | 0.0182 | 0.4980 | 1.0596 | 67 | 0 |
| r2f1 | denfis_frozen | 0.0182 | 0.4980 | 1.0980 | 66 | 1 |
| r2f1 | denfis_online | 0.0188 | 0.4861 | 0.1270 | 84 | 1 |
| r2f2 | no_rls | 0.0111 | 0.5296 | 1.7217 | 19 | 0 |
| r2f2 | no_merge | 0.0113 | 0.4664 | -0.8854 | 19 | 0 |
| r2f2 | denfis_frozen | 0.0113 | 0.4664 | -0.8854 | 19 | 0 |
| r2f2 | denfis_online | 0.0113 | 0.4545 | -1.1969 | 19 | 0 |
| r3f3 | no_rls | 0.0121 | 0.5595 | 2.5669 | 200 | 0 |
| r3f3 | no_merge | 0.0120 | 0.5397 | 1.2484 | 200 | 0 |
| r3f3 | denfis_frozen | 0.0120 | 0.5397 | 1.2484 | 200 | 0 |
| r3f3 | denfis_online | 0.0121 | 0.5397 | 1.1528 | 200 | 0 |

The simpler constant variant sometimes has lower error. Disabling merge is identical in the planned three representative fits because no merge occurred; the merge-active r2f1 changes 66 rules to 67 with only a small error change.

## Figures and interpretation

- `results/figures/denfis_sensitivity.png`: validation profile minima over the other grid parameter, not test-driven tuning.
- `results/figures/denfis_rule_evolution.png`: online rule counts after realised labels.
- `results/figures/denfis_ablations.png`: inherited-parameter test scores.
- `results/figures/denfis_memberships.png`: marginal coordinate memberships of the three most active r3f3 rules.
- `results/figures/denfis_case_contributions.png`: post-hoc high-volatility example; no causal economic interpretation.

Candidate outputs remain subject to consultation and the scheduled joint freeze. See docs/freeze.md.
