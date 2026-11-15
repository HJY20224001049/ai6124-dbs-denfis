# 图注模板（任务 4）— 给 LIN Jingzhao 执行

这份文件只规定图注文字。图不重画、数值不改、不重跑实验。五张图都在
`results/figures/`，底层数据在 `results/analysis/`，图注会贴进 notebook 的
8.2 / 8.3 和 slides 的结果页。

## 三条硬规则

1. 图注用英文，固定三句结构：**画的是什么 → 轴/单位/样本 → 限制**。
2. 每张图注结尾必须带这一句（因为最终配置是 B2，不是这些图对应的完整版）：

   > This figure comes from the full 13-feature DENFIS configuration used for the
   > method diagnostics; the final configuration (B2) is a three-component
   > one-rule average reported in Section 7.

3. 数字只能取自图本身或 `results/analysis/*.csv`，不许估算、不许四舍五入到
   图上看不出的精度。拿不准就把数字写成定性描述。

## 五条图注（可直接粘贴）

### 1. `denfis_ablations.png`

> **Figure 8.2.** Test RMSE on four representative folds (r1f1, r2f1, r2f2,
> r3f3) for the frozen model and three ablations that inherit its
> validation-selected parameters. `no_merge` disables support-based rule
> merging, `no_rls` replaces the weighted-RLS local linear consequents with
> firing-weighted local means, and `denfis_online` updates the model after each
> realised label. Merging changes RMSE by less than 1×10⁻⁵ on every fold;
> disabling RLS lowers RMSE on three of the four folds (by 0.9–1.6×10⁻⁴) and
> raises it slightly on r3f3; online updating raises RMSE on all four folds.
> The y-axis is RMSE of next-day log returns. This figure comes from the full
> 13-feature DENFIS configuration used for the method diagnostics; the final
> configuration (B2) is a three-component one-rule average reported in Section 7.

### 2. `denfis_memberships.png`

> **Figure 8.3a.** Marginal Gaussian membership functions of the three rules
> with the highest mean firing strength in fold r3f3, drawn over the winsorised
> z-score of the feature `ret_0` on a −3 to 3 grid. Each curve is the marginal
> implied by one rule's centre and radius in that coordinate; the legend reports
> each rule's mean firing strength over the test year. The three rules peak near
> zero and differ mainly in width. This figure comes from the full 13-feature
> DENFIS configuration used for the method diagnostics; the final configuration
> (B2) is a three-component one-rule average reported in Section 7.

### 3. `denfis_rule_evolution.png`

> **Figure 8.3b.** Rule count of the online model as realised test labels are
> processed one at a time, for all nine folds; the count starts from the final
> trained rule base. Three folds (r1f2, r2f3, r3f3) stay at the 200-rule cap for
> the whole test year, fold r2f1 grows from 66 to 84 rules, and the remaining
> five folds change by at most one rule. No label is used before its prediction
> has been recorded. This figure comes from the full 13-feature DENFIS
> configuration used for the method diagnostics; the final configuration (B2) is
> a three-component one-rule average reported in Section 7.

### 4. `denfis_case_contributions.png`

> **Figure 8.3c.** Contributions of the eight strongest rules to the prediction
> on 2025-04-07, the day with the largest absolute realised return in fold r3f3
> (realised −8.35%, predicted −0.27%). A contribution is a rule's firing
> strength multiplied by its local linear output, in log-return units, so the
> contributions of all 200 rules sum to the predicted return. The sign is right
> and the magnitude is roughly thirty times too small, which is what a locally
> linear model produces on an outlier day. The case was selected after the fact
> to explain one numerical output; it is not an economic explanation and not
> evidence of causality. This figure comes from the full 13-feature DENFIS
> configuration used for the method diagnostics; the final configuration (B2) is
> a three-component one-rule average reported in Section 7.

### 5. `denfis_sensitivity.png`

> **Figure 8.2b.** Validation RMSE of the full DENFIS configuration as one
> parameter is varied while the other is set to its error-minimising value at
> that point (a profile minimum), for folds r1f1, r2f2 and r3f3. Left: the
> centre-distance threshold `Dthr` from 0.25 to 1.5 with λ fixed. Right: the RLS
> forgetting factor λ from 0.95 to 1.00 with `Dthr` fixed. Each curve is flat or
> nearly flat over its range, while levels differ across folds by up to a factor
> of three, so the validation surface has no sharp optimum: the fold, not
> the parameter, sets the error level. This figure comes from the full 13-feature
> DENFIS configuration used for the method diagnostics; the final configuration
> (B2) is a three-component one-rule average reported in Section 7.

## 交给我之前的自检

- [ ] 五条都用了三句结构，最后一句是上面那句原文
- [ ] 每条都写了轴和单位（RMSE 是次日对数收益率、贡献是对数收益率）
- [ ] 图上的折号与文字一致（消融/敏感性不是全部九折）
- [ ] 引用的数字能在图上或 `results/analysis/` 里找到
- [ ] 全文没有 "proves" / "significant" / "outperforms" 这类词
- [ ] 图注编号（8.2、8.2b、8.3a–c）在集成时由我统一，你按上面的写即可
