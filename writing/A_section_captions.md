## Figure captions

### `denfis_ablations.png`

**Figure 8.2.** Test RMSE on four representative folds (r1f1, r2f1, r2f2, r3f3) compares frozen DENFIS with three variants that inherit its validation-selected parameters: `no_merge` disables merging, `no_rls` uses firing-weighted local means, and `denfis_online` updates after realised labels. The y-axis is next-day log-return RMSE; merging changes RMSE by less than 1×10⁻⁵ on these folds, removing RLS lowers error on three folds and raises it on r3f3, and online updating raises error on all four folds. This figure comes from the full 13-feature DENFIS configuration used for the method diagnostics; the final configuration (B2) is a three-component one-rule average reported in Section 7.

### `denfis_memberships.png`

**Figure 8.3a.** Marginal Gaussian memberships are shown for the three rules with the highest mean firing strength in fold r3f3. The horizontal axis is the winsorised z-score of `ret_0` on a −3 to 3 grid, the vertical axis is coordinate membership, and the legend gives mean firing strength over the test year; these marginal curves do not show the complete multidimensional rule activation. This figure comes from the full 13-feature DENFIS configuration used for the method diagnostics; the final configuration (B2) is a three-component one-rule average reported in Section 7.

### `denfis_rule_evolution.png`

**Figure 8.3b.** Online rule counts are plotted against the number of realised test labels processed for all nine folds, starting from each final trained rule base. The vertical axis is rule count: r1f2, r2f3 and r3f3 remain at the 200-rule cap, r2f1 grows from 66 to 84, and the other five folds change by at most one rule; each label is used only after its forecast has been recorded. This figure comes from the full 13-feature DENFIS configuration used for the method diagnostics; the final configuration (B2) is a three-component one-rule average reported in Section 7.

### `denfis_case_contributions.png`

**Figure 8.3c.** Contributions of the eight strongest rules are shown for 2025-04-07, the largest absolute realised-return day in fold r3f3, with realised log return −8.35% and predicted log return −0.27%. The horizontal axis identifies rules and the vertical axis gives firing strength multiplied by local output in log-return units; all 200 contributions sum to the forecast, and this retrospectively selected case explains a numerical output without establishing economic causality. This figure comes from the full 13-feature DENFIS configuration used for the method diagnostics; the final configuration (B2) is a three-component one-rule average reported in Section 7.

### `denfis_sensitivity.png`

**Figure 8.2b.** Validation RMSE is plotted as a profile minimum for folds r1f1, r2f2 and r3f3: at each value of the displayed parameter, the other parameter takes its error-minimising grid value. The vertical axes measure next-day log-return RMSE; the left horizontal axis varies `Dthr` from 0.25 to 1.5 and the right varies λ from 0.95 to 1.00, so the curves describe validation-grid profiles rather than holding the other parameter constant. This figure comes from the full 13-feature DENFIS configuration used for the method diagnostics; the final configuration (B2) is a three-component one-rule average reported in Section 7.
