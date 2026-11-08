# 胡俊宇 · 数据与评估线详细工作计划

**角色**：数据、特征、切分、评估骨架、baseline、交易分析、Notebook 集成
**主责文件**：`config.py`、`src/data.py`、`src/features.py`、`src/splits.py`、`src/evaluation/*`、`src/models/baselines.py`、`requirements.txt`、`README.md`
**写作分工**：数据章、benchmark/评估章、结果与交易分析、局限性
**总投入**：约 60–65 小时（11 天）

---

## 0. 关键日期与交接点

| 时间 | 必须发生的事 | 对方需要拿到什么 |
|---|---|---|
| 11/08 20:00 | 仓库、数据缓存、`config.py`、`Fold` 定义 | A 可以直接 import 数据与折 |
| 11/09 18:00 | 特征/切分/指标/回测稳定，契约测试通过 | A 可以在真实数据上跑首折 |
| 11/10 20:00 | CV runner 端到端可用，XGB 首折、LSTM 骨架 | 双方各有首折结果 |
| 11/13 18:00 | **硬门槛**：四个模型全部 9 折、summary、泄漏审计 | 主结果表可生成 |
| 11/15 22:00 | 结果冻结，统计检验与最终图完成 | 写作日只读结果 |
| 11/16 18:00 | 数据/benchmark/结果/局限性初稿 | A 交叉审阅 |
| 11/17 18:00 | Notebook 集成、slides 结果半场、彩排 | 可提交版本 |
| 11/18 12:00 | 新 kernel 复现并提交 | 提交确认 |

---

## 1. 逐日计划

### 11/08 周日 · D0：仓库、数据、config（约 5–6h）

**任务**

1. 建仓库与目录：
   - `git init`，写 `.gitignore`（排除 `.ipynb_checkpoints`、虚拟环境、临时文件）；
   - 建 `src/models`、`src/evaluation`、`data/raw`、`results/{preds,metrics,figures}`、`docs`、`tests`；
   - 写 `requirements.txt` 初版，安装并固定 `yfinance` 版本。
2. 下载与缓存：
   - `D05.SI`、`^STI`，2010-01-01 至 2026-10-31；
   - 压平 MultiIndex 列；统一列名；日期索引无时区；
   - 删除缺失 OHLCV 与 `Volume <= 0` 的行；不 forward fill；
   - 存 `data/raw/D05_SI.csv`、`data/raw/STI.csv`；
   - 计算 SHA256 写 `data/checksums.txt`。
3. EDA（保存 4 张图到 `docs/figs/`）：
   - 收盘价与成交量；
   - 日对数收益分布与累计收益；
   - 缺失/停牌检查；
   - 单日 |return| > 50% 的异常点核对。
4. 写 `config.py`：
   - `SYMBOLS`、`START`、`END`、`FEATURES`（13 个，顺序固定）；
   - `FOLDS`（9 折 + holdout 的起止日期）；
   - `SEEDS`、`COST_BPS=10`、`TAU_GRID`、`TARGET_COL`；
   - `PROTOCOL_VERSION="1.0"`；
   - 所有路径用 `pathlib.Path`，相对项目根目录。
5. 给 A 发消息：数据行数、日期范围、列名、`Fold` 字段、`X` 列顺序。

**交付物**
- 仓库骨架
- `data/raw/*.csv` + `checksums.txt`
- `config.py`
- 4 张 EDA 图
- `requirements.txt`

**完成标准**
- D05.SI 行数约 4,218（±50 可接受，记录实际值）；
- 无缺失日期重复、无未来日期；
- A 能 `from config import FEATURES` 并读到数据；
- 最终 Notebook 可以完全离线读缓存。

**风险/退路**
- yfinance 限流或列结构异常：改用 Stooq 日线 CSV，字段映射后同样清洗，记录来源变更；
- `Adj Close` 缺失：改用 `auto_adjust=True` 的 `Close`，在 README 写明口径。

---

### 11/09 周一 · D1：特征、切分、指标、回测（约 7h）

**任务**

1. `src/features.py`：
   - 严格实现协议里的 13 维特征；
   - EMA `adjust=False`、RSI Wilder、Bollinger `ddof=0`、`rv20`/`vol_z` `ddof=1`；
   - warmup 60 天，NaN 行删除；
   - 写因果性测试：
     - 任取 `t`，只用 `≤t` 数据重算特征，与全量计算结果一致；
     - 把 `t` 之后的价格随机扰动，`t` 时刻特征不变。
2. `src/splits.py`：
   - `@dataclass Fold(fold_id, rep, fold, train_idx, val_idx, test_idx, train_end, val_start, ...)`；
   - `make_folds(dates)` 生成 9 折 + holdout；
   - 实现 purge：标签日期 ≥ 测试段第一天的训练/验证行剔除；
   - 写测试：每折训练集最大标签日期 < 测试段第一天。
3. `src/evaluation/metrics.py`：
   - `rmse`、`mae`、`r2`、`ic`、`da`、`da5`；
   - 用手算的玩具例子做单测。
4. `src/evaluation/backtest.py`：
   - `make_signal(pred, tau)`；
   - `run_backtest(signal, returns, cost_bps)`，严格按 `s_t * r_{t+1} - c*|Δs|`；
   - `sharpe`、`ann_ret`、`max_dd`、`turnover`、`exposure`；
   - 单测：全 1 信号应等于买入持有减成本；全 0 信号收益为 0。
5. 18:00 前把可用接口发给 A。

**交付物**
- `src/features.py`、`src/splits.py`
- `src/evaluation/metrics.py`、`backtest.py`
- `tests/` 下对应单测

**完成标准**
- 因果性测试通过；
- 10 个折的索引不重叠、不越界；
- 指标与手算玩具例子一致；
- A 能直接调用 `build_features` 与 `make_folds`。

**风险/退路**
- 特征因果性测试失败：优先检查 `rolling` 是否用了 `center=True`、是否 `shift(-1)` 误用；
- 折边界与协议不一致：以 `config.FOLDS` 为唯一真源，测试里直接断言日期。

---

### 11/10 周二 · D2：CV runner + XGBoost 首折 + LSTM 骨架（约 7h）

**任务**

1. `src/evaluation/cv.py`：
   - 输入 `model_factory`、`fold`、`X, y`、超参网格；
   - 流程：选参训练 → 验证段选超参（RMSE）→ 选阈值（验证段 Sharpe）→ 最终训练 → 测试预测；
   - 支持 `denfis_online`：测试循环里 `predict → 记录 → partial_fit`；
   - 落盘 `preds/{model}/{fold}.csv`、`metrics/{model}/{fold}.json`；
   - 写 `manifest.json`（config hash、git commit、包版本、数据 SHA256）。
2. 用 dummy 模型（恒预测 0）跑通全流程，确认文件格式与字段完全符合协议。
3. `src/models/baselines.py`：
   - `XGBModel`：封装 `xgboost.XGBRegressor`，实现 `fit/predict`；
   - 跑 `r1f1`，落盘。
4. LSTM 骨架：
   - 序列构造器（窗口 20，13 维）；
   - `LSTMModel`：1 层 LSTM，hidden 32，dropout 0.1，Adam 1e-3，MSE，early stopping；
   - 先只验证能 import、能跑一个 batch。

**交付物**
- `src/evaluation/cv.py`
- dummy 模型的端到端产物（证明格式正确）
- `results/preds/xgboost/r1f1.csv`
- LSTM 骨架

**完成标准**
- dummy 模型的 `preds` 与 `metrics` 字段与协议一致；
- XGB 首折无 NaN，指标可算；
- CV runner 不依赖任何模型内部实现，只依赖 `fit/predict`。

**风险/退路**
- CV runner 设计过复杂：先支持 frozen，online 用 `hasattr(model, "partial_fit")` 分支；
- XGB 版本 API 差异：用 sklearn 接口，固定 `tree_method="hist"`。

---

### 11/11 周三 · D3：课程 + consultation + LSTM 首折（约 4–5h）

**任务**

1. 与 A 一起参加第 13 周课和 project consultation，重点问：
   - 评估协议是否接受“验证段选阈值、测试段固定”；
   - 交易成本 10bp、long/flat 是否符合预期；
   - 组合收益按 Adj Close（总回报）是否合适。
2. LSTM 首折：
   - 序列构造、标准化（用折内训练段）、early stopping；
   - 记录单折运行时间；目标 <10 分钟；
   - 落盘 `results/preds/lstm/r1f1.csv`。
3. 如果 CPU 太慢：
   - hidden 32 → 16，序列 20 → 15，epoch 上限 100 → 60；
   - 仍不达标则记录“LSTM 使用 Colab GPU 一次性训练 + 结果回填”，并在 README 写明。

**交付物**
- LSTM `r1f1` 预测
- 运行时间记录
- consultation 笔记（与 A 合并）

**完成标准**
- LSTM 首折无 NaN、预测分布合理；
- 单折 <10 分钟；
- 评估口径与讲师回答一致，若不一致当天改协议并通知 A。

---

### 11/12 周四 · D4：两个 baseline 全折（约 7h）

**任务**

1. XGBoost：
   - 每折验证段跑 9 组网格（depth × lr），按 RMSE 选；
   - 用最佳参数在最终训练段重训；
   - 9 折全部跑完并缓存。
2. LSTM：
   - 每折验证段跑 2 组（hidden 16/32），按 RMSE 选；
   - 用最佳配置与最佳 epoch 数在最终训练段重训；
   - 9 折全部跑完并缓存。
3. 生成 `results/summary.csv` 的 baseline 行。
4. 检查：
   - 每折预测日期与测试年完全一致；
   - 没有跨折复用模型；
   - 同 seed 重跑结果一致。

**交付物**
- `results/preds/xgboost/r1f1…r3f3.csv`
- `results/preds/lstm/r1f1…r3f3.csv`
- 对应 metrics
- baseline 部分的 `summary.csv`

**完成标准**
- 18 个预测文件齐全；
- 每个折的最佳参数记录在 metrics JSON；
- 重跑一致性抽查 1 折通过。

**风险/退路**
- LSTM 全折太慢：减少配置数到 1 组、降低 epoch 上限、缓存预测；
- XGB 过拟合验证段：加 `min_child_weight`、`reg_lambda`，或减少网格。

---

### 11/13 周五 · D5：全模型汇总 + 泄漏审计（硬门槛）（约 7h）

**任务**

1. 把 A 的 DENFIS 预测接入 `summary.csv`：
   - 四个模型 × 9 折；
   - 生成 pooled 指标（按日期拼接测试段）。
2. 主结果表：
   - 行：模型；列：RMSE、MAE、R²、IC、DA、DA5、Sharpe、年化收益、最大回撤、换手、暴露；
   - 同时生成 9 折 mean ± std 附表。
3. 与 A 一起做泄漏审计（逐项打勾并签字）：
   - 特征只用到 `t`；
   - scaler 只在训练段拟合；
   - purge 生效；
   - 阈值只在验证段选；
   - 在线模型严格 `predict → 记录 → partial_fit`；
   - 测试段没有任何调参。
4. 运行时间审计：记录每个模型每折耗时，确认总预算可控。
5. 如果 DENFIS 未按时全折：先出 XGB/LSTM 结果，A 的缺口用“待补”标记，但当天必须给出明确补交时间。

**交付物**
- 主结果表（CSV + markdown）
- 分折附表
- 泄漏审计清单（双方签字）
- `results/manifest.json`

**完成标准**
- 四个模型全部 9/9；
- 主表与附表数字能从 `metrics/*.json` 逐行复算；
- 审计五项全部通过。

---

### 11/14 周六 · D6：GDELT + 交易对比（约 6h）

**任务**

1. GDELT（限时 4 小时）：
   - 用 GDELT DOC 2.0 API 取 `"DBS" OR "Singapore banks"` 的每日语调（tone）；
   - 覆盖范围优先 2017–2026（DOC API 的可靠区间）；2010–2016 若拿不到，就只在 2017 年之后的折做消融，并写明覆盖限制；
   - 聚合为日频，构造 5 日/20 日滚动均值与变化量；
   - 合并到特征表，跑 `denfis_gdelt` 的 3 个代表折；
   - 严格因果：`t` 日语调只用于预测 `t+1`。
2. 交易对比：
   - 有预测 vs 买入持有；
   - 成本敏感性：0 / 10 / 20 bp；
   - 阈值敏感性：0 / 5 / 10 / 20 bp；
   - 统计：胜率、换手、最大回撤。
3. 若 GDELT 失败：
   - 写 `docs/gdelt_notes.md`：接口、覆盖、失败原因、是否值得继续；
   - 在报告中如实写“尝试过但未纳入主结果”，不影响主线评分。

**交付物**
- `results/preds/denfis_gdelt/*` 或 `docs/gdelt_notes.md`
- 成本/阈值敏感性表
- 有/无预测对比图

**完成标准**
- GDELT 无论成败都有书面结论；
- 敏感性表覆盖 3 个成本 × 4 个阈值；
- 交易对比图能直接进报告。

---

### 11/15 周日 · D7：统计检验、最终图、结果冻结（约 5h）

**任务**

1. 统计检验：
   - Diebold–Mariano：DENFIS vs 最好的 baseline，平方误差损失，Newey–West；
   - Sharpe 的 95% bootstrap CI（block=20，2000 次，seed=123）。
2. 最终图（全部从 `results/` 读，不重训）：
   - 四个模型的累计收益曲线；
   - 回撤曲线；
   - 分折 RMSE / DA 柱状图；
   - 预测 vs 实际散点；
   - 模型对比条形图（Sharpe、年化收益）。
3. 更新 `summary.csv`、`manifest.json`；
4. **22:00 结果冻结**：写 `docs/freeze.md`，列出报告要用的图/表编号；之后不再改实验。

**交付物**
- DM 检验与 bootstrap 结果
- 5–6 张最终图
- `docs/freeze.md`
- 完整 `summary.csv`

**完成标准**
- 每张图都能由当前 `results/` 复现；
- manifest 的 `config_hash` 与当前 `config.py` 一致；
- 冻结后不再跑新实验。

---

### 11/16 周一 · D8：论文/报告写作日（全天，不跑实验）

**写作分工（英文，目标 3,000–3,800 词）**

| 章节 | 内容 | 目标词数 |
|---|---|---|
| 2 数据 | 来源、期间、字段、清洗、EDA、数据限制 | 500–700 |
| 4 切分与 benchmark | 折设计、purge、XGB/LSTM 配置、为什么这样比 | 500–700 |
| 7 结果 | 主表、分折表、回测、统计检验 | 1,000–1,200 |
| 8 分析（交易侧） | 成本/阈值敏感性、有/无预测、与 SOTA 对比 | 600–800 |
| 9 局限性与加分 | 数据限制、单标的、成本假设、GDELT 结论 | 300–500 |

**写作规则**
- 只引用 11/15 冻结的图表；
- 每个数字必须能追溯到 `summary.csv` 或 `metrics/*.json`；
- 不把“方向准确率略高于 50%”写成“稳定盈利”；
- 明确写出回测假设（long/flat、10bp、无做空、按 Adj Close）；
- 与 A 统一术语：log return、directional accuracy、Sharpe、drawdown。

**时间安排**
- 09:00–12:00 写结果章（先写主表与回测）
- 13:00–15:00 写数据与切分/benchmark
- 15:00–17:00 写交易分析与局限性
- 17:00–18:00 与 A 交换稿件：你审 A 的方法章，A 审你的结果章
- 19:00–21:00 按意见修改，冻结初稿

**交付物**
- `writing/B_results.md`（或直接写进 Notebook 的 markdown cells）
- 五节初稿

**完成标准**
- 五节都有完整段落，没有 “TBD”；
- 结果章每个表格都有解读，不是只贴数字；
- 交易结论有成本敏感性支撑。

---

### 11/17 周二 · D9：Notebook 集成、slides、彩排（约 6h）

**任务**

1. Notebook 集成（你主责）：
   - 合并 A 的方法/文献/分析/未来工作；
   - 按协议第 10 节的章节结构排好；
   - 加目录、运行说明、环境说明；
   - 从头到尾跑一遍，修格式与路径问题；
   - 确认核心 DENFIS 代码内联、关键行有注释。
2. 按 rubric 逐条检查：
   - 1 数据、2 方法、3 文献、4 benchmark、5 实现、6 分析、7 未来、8 slides、9 bonus；
   - 缺哪项补哪项，但不新增实验。
3. Slides 结果半场（约 5 分钟）：
   - 数据与折设计（1 页）
   - 主结果表（2 页）
   - 交易与成本敏感性（1 页）
   - 局限性（1 页）
4. 与 A 完整彩排，计时。

**交付物**
- 可运行的完整 Notebook
- Slides 结果部分
- rubric 自查表
- 彩排记录

**完成标准**
- 新 kernel 从头到尾无报错；
- 9 项 rubric 都有对应章节；
- slides 总时长 ≤10 分钟；
- 两人对主结果口径一致。

---

### 11/18 周三 · D10：最终复现与提交（约 3h）

**任务**

1. 新 kernel 冷启动运行，确认：
   - 只读 `data/raw/`，不联网；
   - 无绝对路径；
   - `manifest.json` 的哈希与当前文件一致；
   - 所有图正常显示。
2. 检查提交物：
   - Notebook 文件名符合要求；
   - Slides 导出 PDF/PPTX；
   - `results/`、`data/raw/`、`requirements.txt`、`README.md` 齐全。
3. 12:00 前两人都在 NTULearn 提交；
4. 本地保存 zip：Notebook + Slides + results + data + manifest。

**完成标准**
- 提交确认；
- zip 可在另一台机器解压并读到数据（至少路径正确）；
- 没有遗漏的 `TBD` 或空单元格。

---

## 2. 与 A 的接口清单

| 时点 | B 向 A 提供 | B 需要 A 提供 |
|---|---|---|
| 11/08 | `config.py`、数据、`Fold` 字段、特征列顺序 | DENFIS 公式笔记、接口确认 |
| 11/09 | `features.py`、`splits.py`、`metrics.py`、`backtest.py` | `fit/predict` 契约测试通过 |
| 11/10 | `cv.py`、dummy 端到端产物、XGB 首折 | `denfis_frozen/r1f1.csv` |
| 11/13 | 全折 runner、summary、审计清单 | 9 折 frozen + online 预测 |
| 11/15 | pooled 指标、统计检验、最终图 | 方法侧消融/可解释性图 |
| 11/16 | 结果章初稿 | 方法/文献/分析/未来工作初稿 |
| 11/17 | Notebook 集成、slides 结果半场 | slides 方法半场、方法章审阅 |

---

## 3. 评估线风险与退路

| 风险 | 触发信号 | 退路 |
|---|---|---|
| yfinance 下载失败 | 11/08 无法拿到完整数据 | 改用 Stooq CSV；或只保留已有缓存并在 README 注明来源 |
| 特征因果性测试失败 | 扰动未来价格后特征变化 | 逐列检查 `rolling/center/shift`，以 `t` 为界重算 |
| 折边界泄漏 | 训练集最后标签日期 ≥ 测试首日 | 用 purge 测试兜底；以 `config.FOLDS` 为唯一真源 |
| LSTM CPU 超时 | 单折 >10 分钟 | 减 hidden/序列/epoch；只跑 1 组配置；必要时 Colab 一次性训练 |
| GDELT 不可用 | 覆盖不足或接口失败 | 限时 4h，写失败说明，放弃加分项 |
| Notebook 集成冲突 | 两人同时改同一 cell | 开发期用 `writing/*.md`；11/17 由你统一合并 |
| 写作日写不完 | 11/16 18:00 仍有空节 | 优先保结果章与数据章；局限性与 GDELT 压到 11/17 上午 |

---

## 4. 每日自查（30 秒）

- [ ] 今天的数据/特征/指标是否已落盘并有测试？
- [ ] 所有拟合是否只在训练段？测试段是否只读？
- [ ] 新增的图是否能从 `results/` 复现？
- [ ] `summary.csv` 是否已包含今天的模型/折？
- [ ] 明天需要 A 提供什么？是否已经发了消息？
