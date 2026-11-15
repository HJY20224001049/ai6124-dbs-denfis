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
| 1 | 7.1 主表 | 三个模型 + 买入持有的 RMSE/MAE/R²/IC/DA/Sharpe/回撤/暴露度 | 每个数与 `docs/results_table.md` 和 `results/summary.csv` 完全一致 | | |
| 2 | 7.1 主表 | 小数位 | 收益 4 位、比率 3 位、Sharpe 2 位，全表统一 | | |
| 3 | 7.1b 对零预测表 | 每个模型是否有 `no-pred RMSE` 与 `RMSE skill` 两列 | 四个模型（含 no_prediction 自身）都在；skill 的符号与数值方向一致 | | |
| 4 | 7.2 分折表 | 折号是否 r1f1…r3f3 共 9 个 | 三张表（RMSE/Sharpe/DA）都齐全，没有缺折 | | |
| 5 | 7.2 分折表 | 分折数与 pooled 是否来自同一份结果 | 分折均值与 `summary.csv` 的 pooled 行不矛盾（允许加权差异，但不能方向相反） | | |
| 6 | 7.3 权益曲线 | 轴、单位、图例、买入持有对比 | 有日期轴、有累计收益刻度、图例含 buy & hold | | |
| 7 | 7.4 统计检验 | DM 的符号解释 | 正文说明"正值 = 模型 A 更差"，且与表格符号一致 | | |
| 8 | 7.4 统计检验 | "显著"一词的使用 | 只有 p<0.05 的才写 significant；p≥0.05 一律写 not significant | | |
| 9 | 7.4 统计检验 | Sharpe 置信区间 | 每个 Sharpe 都有区间；区间跨 0 的不能写成"更高" | | |
| 10 | 7.5 复现 | 现场重跑一折的断言 | 输出显示误差量级（应为 1e-13 左右），没有跳过或注释掉 | | |
| 11 | 7.6 holdout | 出现次数与用途 | 只出现一次；未跑时明确写 reserved；跑过之后不能出现"据此调整"之类表述 | | |
| 12 | 7.7 合成信号 | 三个模型的 skill | 三个都为正，且正文没有把它当成真实数据的证据 | | |
| 13 | 全文 | 结论句与表格是否一致 | 出现 "better/worse/best" 的句子，都能在同一页的表格里找到对应数字 | | |
| 14 | 全文 | 模型名 | 只出现 DENFIS / XGBoost / LSTM / no prediction / buy & hold，没有 Fuzzy-LSTM 等旧名 | | |

## 填完的格式

- 在表里打 ✅ 或 ❌；❌ 必须带证据，例如：
  `7.2 RMSE 表 r2f1 行 0.0182，summary.csv 为 0.018200`。
- 不要改 notebook，不要提修改方案。
- 交回时间：11/17 18:00。
