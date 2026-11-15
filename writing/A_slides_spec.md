# Slides 页面清单（任务 5）— 给 LIN Jingzhao 执行

总 deck 定为 10 页（英文，16:9）。你负责其中 **4 页**：第 3、4、5、10 页。
你交付一个只有这 4 页的新文件 `writing/A_method_pages.pptx`，我 11/17 下午
把它并进总 deck，页码由我统一。

## 硬规则

1. 只交这 4 页，顺序就是下面 3 → 4 → 5 → 10。不增页、不删页、不合并。
2. 文案按下面给的英文原文，只允许为了排版缩短句子，不允许新增论点、数字或文献。
3. 每页最多 6 个 bullet，每个 bullet 不超过 12 个词，标题不超过 8 个词。
4. 不许写 speaker notes（不需要讲稿）。
5. 旧文件 `writing/A_method_slides.pptx` 里的这些内容一律不要带过来：
   - 所有旧数字（RMSE 0.012066、DA 50.1%、Sharpe 0.654、max DD 20.3%、exposure 34.4%）
   - "preliminary, not frozen"
   - "rehearsal"、"hand over"、"method section" 之类的讲稿用语
   - "Fuzzy-LSTM"（旧名，全 deck 只用 DENFIS）
6. 每页页脚统一写：`AI6124 Group 18 — method, literature, future work`。页码我来加。

## 第 3 页（rubric 2）

**Title:** Method: an evolving Takagi–Sugeno system built from scratch

- ECM creates and moves rule centres online; `Dthr` decides when a new rule is needed.
- Each rule carries a Gaussian membership per input coordinate and a first-order TSK linear consequent.
- Consequents are fitted by weighted recursive least squares; a two-pass fit sets antecedents first.
- The rule base is capped at 200 rules, with optional support-based merging.
- The online mode predicts first and updates only after the label is realised.
- Differences from the original DENFIS paper are listed in `docs/denfis_notes.md`; the code is ours.

允许保留旧文件第 2 页的那张架构图（Picture 3），放在右侧或下方；不要新增图。

## 第 4 页（rubric 2、6）

**Title:** Final configuration: three one-rule components, chosen on validation only

- B2 was selected on validation years only; the test years had been inspected earlier in the diagnostic phase, and the 2026 holdout is unopened.
- B2 averages three one-rule DENFIS models: `ma5`, Bollinger %B (`boll_b`) and the last return (`ret_0`), each with two ex-dividend dummies.
- With one rule the activations are identically 1, so each component is a least-squares linear fit with a weak 𝓁2 prior; the fuzzy layer supplies normalisation, not extra capacity.
- Validation: mean RMSE skill +0.497% over zero prediction, 8/9 folds positive, pooled +0.480%, 95% CI [+0.161%, +0.780%].
- Sixteen one-factor variants were rejected under a pre-registered adoption rule.
- Test years and the 2026 holdout are reported once, after the freeze.

第 4 条的四个数字必须原样保留，不许改写成百分比以外的形式。

## 第 5 页（rubric 3）

**Title:** From ANFIS to evolving and deep neuro-fuzzy systems

表格（5 列，行名照抄）：

| Year | Method | Rule creation | Online | Interpretability | Cost |
|---|---|---|---|---|---|
| 1993 | ANFIS | fixed grid | no | high | high |
| 2002 | DENFIS | ECM | yes | high | moderate |
| 2004 | eTS | eClustering | yes | high | low |
| 2006 | SAFIS | sequential | yes | high | moderate |
| 2014 | PANFIS | pandimensional | yes | medium | high |
| 2023 | DCNFIS | learned | partly | low | high |

表格下方最多两条 bullet：

- Compared on four axes: how rules are created, whether the model updates online, interpretability and computational cost.
- To our knowledge no published study reports an evolving fuzzy system beating a tuned gradient-boosted model on daily equity returns.

## 第 10 页（rubric 7）

**Title:** Where this line goes next

- More independent test data: a panel of SGX banks or similar markets; nine single-stock years cannot resolve the observed Sharpe differences.
- A target the model can fit: realised volatility or multi-day horizons, where local rules have a structural reason to help.
- Principled complexity control: price a rule in validation-error units instead of a fixed 200-rule cap.
- Longer online streams: adaptation needs more than one test year per fold to pay for its variance.

## 总 deck 长什么样（供你定位，不用你做）

| 页 | 内容 | 谁做 |
|---|---|---|
| 1 | 标题、组号、姓名学号、一句话结论 | 我 |
| 2 | 数据与目标：D05.SI、2010–2026、13 个因果特征、次日对数收益率、9 个滚动测试年 + 2026 holdout | 我 |
| **3** | **方法：DENFIS 结构** | **你** |
| **4** | **最终配置 B2 与验证集数字** | **你** |
| **5** | **文献综述与 taxonomy** | **你** |
| 6 | Benchmark 与评估协议：XGBoost、LSTM、3×3 滚动、purge、10 bp 成本 | 我 |
| 7 | 预测精度：主表、对零预测的 RMSE skill、DM / MCS | 我 |
| 8 | 交易结果：Sharpe、回撤、暴露度 vs 买入持有 | 我 |
| 9 | 分析与限制：消融、敏感性、可解释性、STI 迁移、局限 | 我 |
| **10** | **未来工作** | **你** |

## 交给我之前的自检

- [ ] 文件叫 `writing/A_method_pages.pptx`，正好 4 页，顺序 3→4→5→10
- [ ] 没有旧数字、没有 "preliminary"、没有 "Fuzzy-LSTM"、没有 speaker notes
- [ ] 第 4 页四个验证集数字与 `docs/validation_candidates.md` 一致
- [ ] 第 5 页表格 6 行 5 列，年份顺序正确
- [ ] 每页页脚文字一致
