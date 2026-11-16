# 任务 6：第 7 节审阅 checklist（给 LIN Jingzhao 填写）

只填这张表，不要写评语、不要提修改建议。判断标准是"能不能在文件里核对到"，
不是"你觉得好不好"。不通过的项，在"证据"列写清楚：第几行/哪张表，看到的数值，
以及它在哪个文件里是多少。

## 要审的文件

- notebook：`writing/AI6124_Project_Group18_HuJunyu_LinJingzhao.ipynb`（11/17 版）
- 对照文件：`docs/results_table.md`、`docs/statistical_tests.md`、`results/summary.csv`

## 表

| # | 位置 | 检查什么 | 判断标准 | 通过/不通过 | 证据（行号或表名） |
|---|---|---|---|---|---|
| 1 | 7.1 主表 | 三个模型 + 买入持有的 RMSE/MAE/R²/IC/DA/Sharpe/回撤/暴露度 | 每个数与 `docs/results_table.md` 和 `results/summary.csv` 完全一致 | ❌ | Notebook cell 40（cell 从 1 计）7.1 主表只有 denfis_b2、denfis_frozen、xgboost、lstm；cell 42 仅对临时 DataFrame.index.name 赋值，没有保存的 buy & hold 表输出；docs/results_table.md:52 有 buy_hold Sharpe 1.1125、回撤 0.3860、exposure 1.0000。 |
| 2 | 7.1 主表 | 小数位 | 收益 4 位、比率 3 位、Sharpe 2 位，全表统一 | ❌ | Notebook cell 39 的格式约定及 cell 40 输出：RMSE/MAE 为 6 位（B2 0.011904/0.008398），R²/IC/DA 为 4 位，Sharpe 为 3 位（0.537）；模板要求分别为 4 位、3 位、2 位。 |
| 3 | 7.1b 对零预测表 | 每个模型是否有 `no-pred RMSE` 与 `RMSE skill` 两列 | 四个模型（含 no_prediction 自身）都在；skill 的符号与数值方向一致 | ✅ | Notebook cell 44 的 7.1b 保存表包含 no_prediction、denfis_b2、denfis_frozen、xgboost、lstm，均有 no-pred RMSE/RMSE skill；依次为 +0.00%、+0.59%、−0.76%、−0.28%、−0.57%，对应 docs/results_table.md:18–22。 |
| 4 | 7.2 分折表 | 折号是否 r1f1…r3f3 共 9 个 | 三张表（RMSE/Sharpe/DA）都齐全，没有缺折 | ✅ | Notebook cell 46 保存了 RMSE by fold、Sharpe by fold、Directional accuracy by fold 三张表，每张均含 r1f1、r1f2、r1f3、r2f1、r2f2、r2f3、r3f1、r3f2、r3f3 九列。 |
| 5 | 7.2 分折表 | 分折数与 pooled 是否来自同一份结果 | 分折均值与 `summary.csv` 的 pooled 行不矛盾（允许加权差异，但不能方向相反） | ✅ | Notebook cell 46 的三张表逐项对应 results/summary.csv 的 scope=fold 行；例如 B2 r2f1 RMSE 0.018021、Sharpe 1.271、DA 0.5378；cell 40 pooled 0.011904/0.537/0.5151 对应 scope=pooled 行的显示精度。 |
| 6 | 7.3 权益曲线 | 轴、单位、图例、买入持有对比 | 有日期轴、有累计收益刻度、图例含 buy & hold | ✅ | Notebook cell 48 第一张保存图 baseline_equity.png：横轴 Date，纵轴 Growth of 1 SGD，图例含 buy & hold，虚线为同期买入持有；cell 47 文字说明相同。 |
| 7 | 7.4 统计检验 | DM 的符号解释 | 正文说明"正值 = 模型 A 更差"，且与表格符号一致 | ✅ | Notebook cell 49 说明 positive mean loss difference means the first model is worse；cell 51 的 B2−zero 为 −1.69e−06、DM −3.0419，full DENFIS−zero 为 +2.18e−06、DM +1.7039；与 docs/statistical_tests.md:20、24–25 一致。 |
| 8 | 7.4 统计检验 | "显著"一词的使用 | 只有 p<0.05 的才写 significant；p≥0.05 一律写 not significant | ✅ | Notebook cell 51：B2 vs zero p=0.0024、vs XGBoost p=0.0006 标为 reject equal accuracy；其他七项 p≥0.05 均为 not significant；cell 55 的准确性结论与该表一致。 |
| 9 | 7.4 统计检验 | Sharpe 置信区间 | 每个 Sharpe 都有区间；区间跨 0 的不能写成"更高" | ❌ | Notebook cell 50 的 CI 表只有六个预测模型，无 buy & hold，也无 7.6 的 holdout Sharpe CI；cell 55 引用 buy & hold Sharpe 1.113，cell 58 报告 B2 holdout Sharpe 1.712，均未配区间；docs/statistical_tests.md:9–14 仅列 pooled 预测模型 CI。 |
| 10 | 7.5 复现 | 现场重跑一折的断言 | 输出显示误差量级（应为 1e-13 左右），没有跳过或注释掉 | ✅ | Notebook cell 56 保存了五个模型的复现输出：max abs(Δ prediction) 为 5.0e−13、5.0e−13、4.9e−13、5.0e−13、4.7e−13；源码中的 max_diff/b2_diff < 1e−9 断言未注释。这里只核对已有输出，未运行该 cell。 |
| 11 | 7.6 holdout | 出现次数与用途 | 只出现一次；未跑时明确写 reserved；跑过之后不能出现"据此调整"之类表述 | ❌ | Holdout 除 7.6（cells 57–59）外，还见 cell 1、cell 35（5.7）、cell 74（8.5）；cell 59 写 B2 exposure 12%，docs/results_table.md:38、58 和 results/summary.csv 的 B2 holdout 行为 0.8995/0.8995215311004785。 |
| 12 | 7.7 合成信号 | 三个模型的 skill | 三个都为正，且正文没有把它当成真实数据的证据 | ✅ | Notebook cell 61 的合成信号保存表：DENFIS +2.45%、LSTM +1.48%、XGBoost +2.10%；cell 60 明确使用 y_syn = 0.25·ret_0 + N(0,0.012)，将其限定为 pipeline 学习已知信号的实现检查。 |
| 13 | 全文 | 结论句与表格是否一致 | 出现 "better/worse/best" 的句子，都能在同一页的表格里找到对应数字 | ❌ | Notebook cell 59 将 B2 holdout 称为 at 12% exposure / low exposure；results/summary.csv 的 denfis_b2,holdout 行 exposure=0.8995215311004785、turnover=0.11961722488038277；docs/results_table.md:38、58 同样是 exposure 0.8995、turnover 0.1196。 |
| 14 | 全文 | 模型名 | 只出现 DENFIS / XGBoost / LSTM / no prediction / buy & hold，没有 Fuzzy-LSTM 等旧名 | ✅ | Notebook 第 7 节表格使用 denfis_b2、denfis_frozen、denfis_online、xgboost、lstm、no_prediction 和 buy & hold；全文 source 中 Fuzzy-LSTM 出现次数为 0。 |

## 填完的格式

- 在表里打 ✅ 或 ❌；❌ 必须带证据，例如：
  `7.2 RMSE 表 r2f1 行 0.0182，summary.csv 为 0.018200`。
- 不要改 notebook，不要提修改方案。
- 交回方式：填好后发回给 HU Junyu。
