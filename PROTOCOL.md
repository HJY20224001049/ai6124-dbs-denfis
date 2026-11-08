# AI6124 Group 18 · 项目统一协议

**版本**：v1.1（v1.0 已按建议冻结；v1.1 为 XGBoost 正则化修订，待 A 确认）
**状态**：冻结后任何修改需双方同意、升版本，并重跑受影响的结果
**适用范围**：全部代码、实验、Notebook、Slides
**项目**：DBS Group Holdings (D05.SI) 次日收益预测 — DENFIS vs XGBoost / LSTM

---

## 0. 五条原则

1. **单一真源**：常量、特征列、折区间、指标定义只写在 `config.py` 和本协议；Notebook 不另写一套。
2. **预测先落盘**：训练与画图解耦，所有分析只读 `results/`。
3. **测试段只读**：超参数、阈值、特征筛选只在训练/验证段决定，测试段只在最终评估时打开。
4. **可复现**：固定种子、数据本地缓存、离线可跑、Notebook 从头到尾无报错。
5. **主结果与扩展结果分开**：主表只放 frozen DENFIS + XGBoost + LSTM；在线演化、迁移学习、GDELT 单列，不混在一起讲故事。

---

## 1. 数据协议

| 项目 | 规定 |
|---|---|
| 标的 | `D05.SI`（DBS Group Holdings，SGX 银行股） |
| 期间 | 2010-01-01 至 2026-10-31（含） |
| 频率 | 日线，按 SGX 实际交易日 |
| 来源 | Yahoo Finance chart API（字段与 `yfinance` 一致） |
| 字段 | Open, High, Low, Close, Adj Close, Volume |
| 价格口径 | `auto_adjust=False`；**收益与全部价格类指标一律用 `Adj Close`**，`Close` 只用于展示 |
| 收益口径 | 总回报（含股息再投资），按 Adj Close 计算 |
| 时区/索引 | Asia/Singapore；索引为无时区的日期（`YYYY-MM-DD`），升序 |
| 清洗 | 去重复日期；删除 OHLCV 缺失或 `Volume <= 0` 的行；`^STI` 为指数，豁免 Volume 过滤；**不 forward fill** |
| 异常检查 | 单日 `|r_t| > 0.5` 必须人工核对，确认不是拆股/分红导致的错误 |
| 缓存 | `data/raw/D05_SI.csv`、`data/raw/STI.csv`；提交进仓库 |
| 校验 | `data/checksums.txt` 记录两个文件的 SHA256；最终 Notebook 启动时校验 |
| 迁移学习数据 | `^STI`，同期间、同清洗规则，只用于迁移学习实验 |

**数据下载只做一次**：由 B 在 11/08 完成并提交缓存。之后所有人只读缓存，不在 Notebook 里实时下载。

> 实现说明：当前 Python 3.9 环境装不上新版 `yfinance`（它要求 `curl_cffi>=0.17`，没有 3.9 wheel），所以 `src/data.py` 直接调用 Yahoo chart endpoint，返回同样的 OHLCV 与 `Adj Close`。下载结果已缓存为 CSV，最终 Notebook 不需要网络。

> yfinance 注意：单标的下载可能返回 MultiIndex 列，落地前压平；`Adj Close` 必须存在，否则改用 `auto_adjust=True` 并把 `Close` 当调整后价格，同时在 README 记录这次口径变更。

---

## 2. 目标与特征协议

### 2.1 预测目标

对每个交易日 `t`：

```text
y_t = ln(AdjClose_{t+1} / AdjClose_t)      # 次日对数收益
```

- 特征只用 `t` 日收盘及之前的信息；预测在 `t` 日收盘后做出，对应 `t+1` 日收益。
- 最后一行没有 `y`，从训练/评估中删除。
- 模型输出单位必须是**对数收益**，不是价格、不是方向、不是排名。

### 2.2 特征集（主实验固定 13 维）

记 `C_t = AdjClose_t`，`V_t = Volume_t`，`r_t = ln(C_t / C_{t-1})`。

| # | 名称 | 定义 | 窗口 |
|---|---|---|---|
| 1 | `ret_0` | `r_t` | 1 |
| 2 | `ret_1` | `r_{t-1}` | 1 |
| 3 | `ret_2` | `r_{t-2}` | 1 |
| 4 | `ret_3` | `r_{t-3}` | 1 |
| 5 | `ret_4` | `r_{t-4}` | 1 |
| 6 | `ma5` | `C_t / SMA5(C)_t - 1` | 5 |
| 7 | `ma20` | `C_t / SMA20(C)_t - 1` | 20 |
| 8 | `ma60` | `C_t / SMA60(C)_t - 1` | 60 |
| 9 | `rsi14` | `(RSI14_t - 50) / 50` | 14 |
| 10 | `macd` | `(EMA12_t - EMA26_t - EMA9(EMA12-EMA26)_t) / C_t` | 12/26/9 |
| 11 | `boll_b` | `(C_t - L_t) / (U_t - L_t)`，`U/L = SMA20 ± 2σ20` | 20 |
| 12 | `rv20` | `std(r, 20) * sqrt(252)` | 20 |
| 13 | `vol_z` | `(V_t - mean20(V)) / std20(V)` | 20 |

实现口径（写进代码注释）：

- EMA 用 `adjust=False`；RSI 用 Wilder 平滑。
- Bollinger 的 `σ` 用 `ddof=0`；`rv20` 与 `vol_z` 用 `ddof=1`。
- 分母为 0 时：`boll_b = 0.5`；`vol_z = 0`。
- 前 60 个交易日为 warmup，特征齐备后才保留；NaN 行直接删除，不插值。
- **主实验固定这 13 维**。增删特征只能作为消融实验，不能替换主表。

### 2.3 预处理（每个外层折独立）

1. 只在该折的**训练段**上计算 0.5% / 99.5% 分位数，对全部特征做 winsorize。
2. 只在该折的**训练段**上计算均值/标准差，做 z-score。
3. 用同一组参数变换验证段与测试段。
4. **测试段不更新缩放参数**，在线 DENFIS 也不例外。

> 目的：让 DENFIS 的 ECM 在标准化空间里工作，同时保证与 XGBoost/LSTM 使用的是同一份输入。

---

## 3. 切分协议

### 3.1 总原则

- **Expanding walk-forward**，不做随机切分，不 shuffle。
- 初始训练 7 年：2010–2016。
- 9 个外层折，分 3 组，覆盖 2017–2025 全部年份。
- 2026 年 1–10 月作为**额外盲测**，不进 3×3 主表，只在最终报告单独一段。

### 3.2 折定义

对测试年 `Y`：

| 用途 | 区间 |
|---|---|
| 选参训练 | 2010-01-01 至 `Y-2` 年 12-31 |
| 验证 | `Y-1` 年全年 |
| 最终训练 | 2010-01-01 至 `Y-1` 年 12-31 |
| 测试 | `Y` 年全年 |

这样每个折都是：先用“前一年”做验证选超参，再把验证年并入训练，最后预测测试年。

| 组 | 折 ID | 测试年 | 选参训练截止 | 验证年 | 最终训练截止 |
|---|---|---|---|---|---|
| 1 | `r1f1` | 2017 | 2015-12-31 | 2016 | 2016-12-31 |
| 1 | `r1f2` | 2018 | 2016-12-31 | 2017 | 2017-12-31 |
| 1 | `r1f3` | 2019 | 2017-12-31 | 2018 | 2018-12-31 |
| 2 | `r2f1` | 2020 | 2018-12-31 | 2019 | 2019-12-31 |
| 2 | `r2f2` | 2021 | 2019-12-31 | 2020 | 2020-12-31 |
| 2 | `r2f3` | 2022 | 2020-12-31 | 2021 | 2021-12-31 |
| 3 | `r3f1` | 2023 | 2021-12-31 | 2022 | 2022-12-31 |
| 3 | `r3f2` | 2024 | 2022-12-31 | 2023 | 2023-12-31 |
| 3 | `r3f3` | 2025 | 2023-12-31 | 2024 | 2024-12-31 |
| 盲测 | `holdout` | 2026-01-01 至 2026-10-31 | 2024-12-31 | 2025 | 2025-12-31 |

### 3.3 边界清理（purge）

训练/验证段的每一行 `d`，只有当它的**标签日期**（下一个交易日）严格早于测试段第一天时，才允许使用。

效果：测试年开始前最后一个交易日的“跨年收益”不会进入训练，避免 1 个测试样本泄漏。实现时用日期 shift 判断，不要手写索引。

### 3.4 盲测规则

- 2026 盲测在 11/15 结果冻结后才跑，只跑一次。
- 看到结果后**不允许**回头改超参、特征、阈值。若表现差，如实报告。
- 盲测不参与模型选择，也不参与主表排名。

---

## 4. 模型接口协议

所有模型实现同一个接口，CV runner 只依赖这个接口。

```python
from typing import Protocol
import numpy as np

class Predictor(Protocol):
    name: str

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """重置并从零训练。X: (n, 13) float64；y: (n,) float64。"""

    def predict(self, X: np.ndarray) -> np.ndarray:
        """返回 (n,) float64，单位与 y 相同（对数收益）。不得修改模型状态。"""

class OnlinePredictor(Predictor, Protocol):
    def partial_fit(self, x: np.ndarray, y: float) -> None:
        """只在该样本的真实标签已经实现之后调用。x: (13,)；y: float。"""
```

硬性规则：

1. `X` 的列顺序必须等于 `config.FEATURES`，dtype `float64`，不允许 NaN/Inf。
2. `predict` 返回有限值，形状 `(n,)`；不做方向二值化，不反标准化。
3. `fit` 必须可重复调用并重置状态；`predict` 不得改变状态。
4. 模型内部**不得读文件、不得联网、不得访问日期或原始价格**，一切信息只通过 `X, y` 进入。
5. 给定 seed 必须确定性输出；LSTM 固定权重初始化种子与 `torch.use_deterministic_algorithms(True)`。
6. 模型可暴露 `n_rules_`、`get_params()` 等只读信息，供分析和记录使用。
7. 每个模型必须通过契约测试：合成数据上 `fit → predict` 形状正确、无 NaN、同 seed 两次结果一致。

模型注册名固定为：`denfis_frozen`、`denfis_online`、`xgboost`、`lstm`。

---

## 5. 训练与调参协议

### 5.1 搜索空间（只在选参训练段上拟合，验证段上选择）

**DENFIS**

| 参数 | 搜索范围 | 默认/说明 |
|---|---|---|
| ECM 半径 `Dthr` | {0.25, 0.5, 0.75, 1.0, 1.5} | 标准化空间 |
| RLS 遗忘因子 `λ` | {0.95, 0.98, 0.99, 1.0} | 1.0 表示不遗忘 |
| 隶属函数 | 高斯 | 宽度由聚类半径导出，具体公式写进方法章 |
| 规则合并阈值 | `0.5 * Dthr` | 固定，不做网格 |
| 最大规则数 | 200 | 运行时保护，不做网格 |

共 20 组。按验证段 **RMSE** 选最优；同分时选规则数更少、`Dthr` 更大的组合。

**XGBoost**

| 参数 | 搜索范围 |
|---|---|
| `max_depth` | {2, 3} |
| `learning_rate` | {0.01, 0.03} |
| `min_child_weight` | {5, 30} |
| `reg_lambda` | {1.0, 5.0} |
| `n_estimators` | 最多 1000，验证段 early stopping 50 |
| `subsample` / `colsample_bytree` | 0.8 / 0.8 |

共 16 组，按验证段 RMSE 选择。

> v1.1 修订原因：原网格（depth 3–5、lr 0.03–0.1、固定 `min_child_weight=5`、`reg_lambda=1`）在 2016 验证段上第 1 棵树后误差立即上升，早停退化为 1 棵树、预测近似常数。加强正则化后，验证误差在 100+ 棵树处才见底，baseline 才有比较意义。

**LSTM**

| 参数 | 规定 |
|---|---|
| 输入 | 同一 13 维特征，序列长度 20 |
| 结构 | 1 层 LSTM，hidden ∈ {16, 32}，dropout 0.1 |
| 优化 | Adam 1e-3，batch 64，MSE loss |
| 训练 | 最多 100 epoch，验证段 early stopping 15 |
| 设备 | CPU；单折目标 ≤10 分钟 |

2 组配置，按验证段 RMSE 选择。

> 输入差异是刻意保留的：DENFIS 与 XGBoost 用 `t` 时刻的 13 维向量；LSTM 用同一 13 维的 20 天序列。这是模型归纳偏置的差异，报告里要写明，不要假装三者输入完全一致。

### 5.2 最终拟合

选定超参后：

- DENFIS / XGBoost：在“最终训练段”（训练 + 验证）上重新拟合。
- LSTM：用验证段得到的最佳 epoch 数，在最终训练段上重训同样轮数。
- 阈值 `τ`：用**选参阶段**的验证段预测值选择（见 6.3），再应用到测试段。

### 5.3 在线 DENFIS 的更新顺序

```text
for t in test_days:
    pred_t = model.predict(x_t)        # 先预测
    记录 pred_t
    model.partial_fit(x_t, y_t)        # 收盘后真实 y_t 已知，再更新
```

- 在线模式在测试段的更新只使用**已经实现**的标签，属于因果在线学习。
- 主表用 `denfis_frozen`；`denfis_online` 单列，标题写清“测试期在线适应”，不和 frozen 混在一起排名。
- Notebook 必须用一段 markdown 解释这个区别，避免被质疑数据泄漏。

### 5.4 种子

```text
global_seed = 42
fold_seed   = 42 + 100 * (rep - 1) + (fold - 1)     # r1f1=42, r1f2=43, ..., r3f3=48
holdout_seed = 99
bootstrap_seed = 123
```

所有模型、所有折使用 `fold_seed`；同一折内不同模型也用同一 seed，减少随机性差异。

---

## 6. 评估协议

### 6.1 回归与方向指标

```text
RMSE = sqrt(mean((y - ŷ)^2))
MAE  = mean(|y - ŷ|)
R²   = 1 - Σ(y - ŷ)² / Σ(y - ȳ)²          # 测试段均值 ȳ，允许为负
IC   = Pearson corr(ŷ, y)                  # 信息系数
DA   = mean(1[sign(ŷ) == sign(y)])         # sign(0)=0
DA5  = mean over {|y| >= 5e-4} of 1[sign(ŷ) == sign(y)]   # 5bp 死区
```

### 6.2 信号与回测

```text
s_t = 1[pred_t > τ]                        # long/flat，不做空
r_strat_{t+1} = s_t * r_{t+1} - c * |s_t - s_{t-1}|
```

- `s_{-1} = 0`，每个测试折独立从空仓开始。
- 成本 `c = 0.001`（单边 10bp）。分析章另报 0bp 与 20bp 的敏感性。
- 买入持有基准：同一测试区间，`r_bench_{t+1} = r_{t+1}`，不计成本。
- 收益按 Adj Close 计算，是总回报口径。

指标：

```text
Sharpe   = mean(r_strat) / std(r_strat) * sqrt(252)      # rf = 0, ddof=1
AnnRet   = (∏(1 + r_strat))^(252 / n) - 1
MaxDD    = max peak-to-trough drawdown of cumulative equity
Turnover = mean(|s_t - s_{t-1}|)
Exposure = mean(s_t)
```

### 6.3 阈值选择

- 候选 `τ ∈ {0, 5e-4, 1e-3, 2e-3}`（0 / 5 / 10 / 20 bp）。
- 在**验证段预测值**上最大化 Sharpe（`c = 10bp`）选择；平手取较小 `τ`。
- 若所有候选的 Sharpe 都为负，取 `τ = 0`。
- 选定后固定，应用于测试段；测试段不再调阈值。

### 6.4 聚合与报告

- 每个折单独算一套指标；同时把 9 个折的测试日按日期拼接，算**pooled** 指标。
- 主表报 pooled；附表报 9 折的 mean ± std。
- 不做“只报最好的折”。所有折、所有模型、所有指标都进结果文件。
- 负结果（跑输买入持有、R² 为负）如实报告，并在分析里解释。

### 6.5 统计检验（分析章，可选但推荐）

- Diebold–Mariano：平方误差损失，NeWey–West 滞后按标准公式；对比 DENFIS 与表现最好的 baseline。
- Sharpe 的 95% 置信区间：stationary block bootstrap，平均块长 20 天，2000 次，`seed=123`。

---

## 7. 产物与命名协议

### 7.1 目录

```text
project/
  config.py
  data/
    raw/D05_SI.csv
    raw/STI.csv
    checksums.txt
  src/
    data.py
    features.py
    splits.py
    models/{base.py, denfis.py, baselines.py}
    evaluation/{metrics.py, backtest.py, cv.py}
  results/
    preds/{model}/{fold_id}.csv
    metrics/{model}/{fold_id}.json
    summary.csv
    manifest.json
    figures/
  AI6124_Project_Group18_HuJunyu_LinJingzhao.ipynb
```

### 7.2 预测文件

`results/preds/{model}/{fold_id}.csv`，UTF-8，逗号分隔，按日期升序：

| 列 | 类型 | 说明 |
|---|---|---|
| `date` | `YYYY-MM-DD` | 测试日 |
| `y_true` | float | 次日对数收益 |
| `y_pred` | float | 模型预测，对数收益 |
| `signal` | 0/1 | `1[pred > τ]` |
| `position` | 0/1 | 实际持仓（与 signal 相同，保留以便扩展） |
| `strat_ret` | float | 扣成本后的策略日收益 |
| `bench_ret` | float | 买入持有日收益 |

不四舍五入存储；展示时再格式化。

### 7.3 指标文件

`results/metrics/{model}/{fold_id}.json`：

```json
{
  "model": "denfis_frozen",
  "fold_id": "r1f1",
  "n_test": 250,
  "rmse": 0.0, "mae": 0.0, "r2": 0.0, "ic": 0.0,
  "da": 0.0, "da5": 0.0,
  "sharpe": 0.0, "ann_ret": 0.0, "max_dd": 0.0,
  "turnover": 0.0, "exposure": 0.0,
  "tau": 0.0,
  "params": {},
  "seed": 42
}
```

`results/summary.csv` 列：
`model, scope, rep, fold, n, rmse, mae, r2, ic, da, da5, sharpe, ann_ret, max_dd, turnover, exposure, tau`
`scope` 取 `fold` 或 `pooled`。

### 7.4 Manifest

`results/manifest.json` 每次完整运行覆盖写入：

```json
{
  "protocol_version": "1.0",
  "config_hash": "...",
  "git_commit": "...",
  "python": "3.9.0",
  "packages": {"numpy": "1.24.4", "pandas": "2.3.3", "...": "..."},
  "data_sha256": {"D05_SI.csv": "...", "STI.csv": "..."},
  "generated_at": "2026-11-18T10:00:00+08:00"
}
```

### 7.5 命名

- 模型：`denfis_frozen` / `denfis_online` / `xgboost` / `lstm`
- 折：`r1f1` … `r3f3`，盲测 `holdout`
- 图：`results/figures/{section}_{name}.png`，150 dpi，标题、坐标轴、单位齐全
- Notebook：`AI6124_Project_Group18_HuJunyu_LinJingzhao.ipynb`

---

## 8. 可复现协议

1. `requirements.txt` 固定版本；当前环境：Python 3.9.0、numpy 1.24.4、pandas 2.3.3、scikit-learn 1.6.1、xgboost 2.1.4、torch 2.8.0+cpu、matplotlib 3.9.4、scipy 1.13.1；**yfinance 需安装后固定版本**。
2. 数据只从 `data/raw/` 读取；`RUN_MODE=download` 才联网，默认 `cached`。
3. 所有随机性走 `config` 的种子；LSTM 开确定性算法，`num_threads=1`。
4. 最终 Notebook 在**新 kernel** 从头到尾运行，不依赖隐藏状态、不使用绝对路径。
5. 每次完整运行更新 `manifest.json`；提交前核对 `config_hash` 与 `git_commit`。
6. 任何一次结果如果无法用当前 commit 复现，视为无效结果。

---

## 9. 协作协议

### 9.1 仓库与分支

- 仓库：`ai6124-dbs-denfis`（私有）。
- `main` 保护；功能分支 `feat/denfis`、`feat/eval`、`feat/features`、`feat/baselines`。
- 每个 PR 至少 1 人审阅；合并前必须跑通契约测试。
- `data/raw/`、`results/`、`config.py` 进版本控制；不要提交 `.ipynb_checkpoints`、大文件、虚拟环境。

### 9.2 文件归属

| 文件/目录 | 主责 | 审阅 |
|---|---|---|
| `config.py`、`src/splits.py`、`src/evaluation/*` | B | A |
| `src/data.py`、`src/features.py` | B | A |
| `src/models/denfis.py` | A | B |
| `src/models/baselines.py` | B | A |
| `results/` 的生成与汇总 | B | A |
| Notebook 集成 | B | A |
| Slides | 各写各的一半 | 互相审 |

### 9.3 Definition of Done

一个工作包算完成，必须同时满足：

1. 代码合并到 `main`；
2. 契约测试通过；
3. 对应的 `preds/*.csv` 与 `metrics/*.json` 已生成；
4. `summary.csv` 有该行；
5. PR 描述里有一句话结论（例如“r1f1 DENFIS RMSE 0.0123，低于 XGB 0.0128”）。

### 9.4 同步节奏

- 每天 15 分钟：昨天完成、今天计划、当前阻塞。
- 阻塞超过 2 小时就换方案或求助，不硬扛。
- 11/13 硬门槛检查：三个模型是否全折跑通；未通过则砍扩展功能。

### 9.5 变更控制

- 协议冻结后，改折、改指标、改特征、改成本假设都算协议变更。
- 变更流程：提出 → 双方同意 → 升版本 → 重跑受影响结果 → 在 `manifest.json` 记录新版本。
- 11/15 结果冻结后，不再接受影响主表的协议变更。

---

## 10. Notebook 与提交协议

### 10.1 章节结构（与 rubric 一一对应）

| 章节 | 内容 | 对应 rubric |
|---|---|---|
| 0 | 标题、姓名、学号、分工、运行说明 | — |
| 1 | Setup：导入、种子、版本、数据校验 | 5 |
| 2 | 数据：来源、期间、EDA、清洗与异常检查 | 1 |
| 3 | 特征：定义表、因果性检查、标准化 | 2 |
| 4 | 切分：折表、时间线图、purge 说明 | 4 |
| 5 | 方法：DENFIS 理论、ECM/TSK/RLS 推导、实现讲解、复杂度 | 2、3 |
| 6 | Baselines：XGBoost、LSTM 配置与训练 | 4 |
| 7 | 结果：主表、分折表、回测曲线、统计检验 | 4、6 |
| 8 | 分析：参数敏感性、消融、可解释性、优缺点、与 SOTA 对比 | 6 |
| 9 | 未来工作 | 7 |
| 10 | 参考文献 | 3 |
| 附录 | 完整实现代码（若开发期放在 `src/`） | 5 |

### 10.2 写作规定

- **交付语言：英文**（课程、rubric、slides 均为英文）；内部协议与沟通用中文。
- 每个代码块前有 markdown 说明“为什么这么做”，不是只写“加载数据”。
- 核心 DENFIS 实现必须内联展示，不能只写“见 `src/models/denfis.py`”。
- 每张图有标题、坐标轴、单位；表格小数位统一（收益 4 位、比率 3 位、Sharpe 2 位）。
- 不夸大：R² 为负就写负；跑输买入持有就解释原因。

### 10.3 提交清单

- [ ] 新 kernel 从上到下无报错，只读本地缓存
- [ ] 9 项 rubric 各有对应章节
- [ ] 首页有姓名、学号、分工
- [ ] 核心代码内联、关键行有注释
- [ ] `results/`、`data/raw/`、`manifest.json` 完整
- [ ] Slides ≤10 分钟，覆盖 8 项 rubric，方法名统一为 DENFIS
- [ ] 两人都在 NTULearn 提交，目标 11/18 12:00 前完成
- [ ] 11/23 收到互评材料，11/27 前提交互评分数

---

## 11. 待确认（回复“按建议”或指出要改的项）

| # | 决策 | 建议 | 备选 | 影响 |
|---|---|---|---|---|
| 1 | 折方案 | 三块不重叠：2017–19 / 2020–22 / 2023–25 + 2026 盲测 | 真 rolling origin（步长 1 年，测试年重叠） | 样本量与年份覆盖 |
| 2 | 在线 DENFIS 定位 | 主表用 frozen；online 单列为扩展 | online 进主表 | 与 baseline 的公平性 |
| 3 | 成本与仓位 | 单边 10bp、long/flat、τ ∈ {0,5,10,20}bp | 0/20bp 敏感性、long/short | 组合指标与结论 |
| 4 | 特征集 | 固定 13 维 | 增加更多技术指标 | ECM 维度、过拟合风险 |
| 5 | 交付语言 | Notebook/Slides 英文，内部中文 | 全中文 | 评分可读性 |

---

## 12. 变更记录

| 版本 | 日期 | 变更 | 确认 |
|---|---|---|---|
| v1.0 | 2026-11-08 | 初稿，按建议冻结 | 双方已确认 |
| v1.1 | 2026-11-08 | B 提议：XGBoost 改为强正则搜索空间（depth 2/3、lr 0.01/0.03、`min_child_weight` 5/30、`reg_lambda` 1/5），避免早停退化为 1 棵树 | 待 A 确认；确认前代码按 v1.1 执行 |
