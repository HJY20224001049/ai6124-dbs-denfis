# 交付说明：评估线 → 方法线

**日期**：2026-11-10
**从**：HU Junyu（数据 / 评估 / baseline）
**给**：LIN Jingzhao（DENFIS 方法线）
**仓库**：`ai6124-dbs-denfis`，分支 `main`

---

## 一句话

评估、数据、回测、落盘、汇总、统计检验和作图已经全部就绪并推到 `main`。
你只需要交付 `src/models/denfis.py`；跑折、保存预测、生成 `summary.csv` 和
`manifest.json`、DM 检验、结果表、作图都会自动完成，不需要你另写一套。

---

## 1. 现在可以直接用的东西

| 能力 | 位置 |
|---|---|
| 13 维特征（严格因果） | `src/features.py::compute_features` |
| 9 折 walk-forward + 2026 holdout + purge | `src/splits.py::make_folds_from_frame` |
| 每折标准化（只在训练段拟合） | `src/evaluation/cv.py`（自动完成） |
| 选超参 → 验证段选阈值 → 最终训练 → 测试预测 | `src/evaluation/cv.py::run_fold` |
| 预测/指标落盘、`summary.csv`、`manifest.json` | `src/evaluation/cv.py::run_model_on_folds` |
| 指标与回测 | `src/evaluation/metrics.py`、`src/evaluation/backtest.py` |
| DM 检验 + Sharpe bootstrap CI | `src/evaluation/stats.py` |
| ^STI ↔ DBS 特征对齐 | `src/transfer.py` |

运行入口已经预埋好 `denfis_frozen` 和 `denfis_online`（延迟 import，
现在运行会提示缺 `src/models/denfis.py`）：

```bash
python scripts/run_baseline.py --model denfis_frozen --folds r1f1
python scripts/run_baseline.py --model denfis_frozen --folds all
python scripts/run_baseline.py --model denfis_online --folds all
```

---

## 2. DENFIS 接口契约（唯一需要严格遵守的东西）

`scripts/run_baseline.py` 会按下面的签名构造模型：

```python
DENFIS(
    dthr=dthr,                  # config.DENFIS_DTHR_GRID 中的一个值
    lam=lam,                    # config.DENFIS_LAMBDA_GRID 中的一个值
    max_rules=200,              # config.DENFIS_MAX_RULES
    merge_factor=0.5,           # config.DENFIS_MERGE_FACTOR
    seed=seed,                  # global/fold/holdout seed
    evolve_rules=False,         # frozen=False, online=True
)
```

需要实现的方法：

```python
def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...
def predict(self, X: np.ndarray) -> np.ndarray: ...
def partial_fit(self, x: np.ndarray, y: float) -> None: ...   # 在线模式需要
```

关键约定：

1. **输入已经标准化。** CV runner 用**当折训练段**的 5%/95% winsorisation +
   z-score 处理过 `X`，列顺序固定为 `config.FEATURES` 的 13 列。模型内部
   不要再拟合 scaler，也不要改变列顺序。
2. **输出单位是对数收益**，形状 `(n,)`，与输入行一一对应，不得返回 NaN/Inf。
3. `fit` 必须重置内部状态；同一 `seed` 两次结果一致。
4. **在线模式的顺序由 runner 保证**：逐日 `predict` → 记录预测 →
   `partial_fit(x_t, y_t)`。你只需保证 `predict` 不修改状态、
   `partial_fit` 只在标签已知后更新；`evolve_rules` 控制测试期 ECM 是否继续
   增长规则。
5. 超参网格只用 `dthr × lam`（20 组）；`max_rules`、`merge_factor` 由
   `config.py` 固定，不要另设一套默认值。
6. 如果构造器参数名必须不同，先告诉我，我改 runner 的 factory，**不要同时改
   runner 和 class**，避免接口漂移。

参考实现文件位置：

- `src/models/denfis.py`（你负责）
- `tests/test_denfis_contract.py`（你负责）
- `docs/denfis_notes.md`、`docs/refs.md`（仍欠，11/08 应交）

---

## 3. 落盘格式（全部自动，不要手写）

跑完一折后，runner 会自动生成：

```text
results/preds/denfis_frozen/r1f1.csv
results/preds/denfis_online/r1f1.csv
results/metrics/denfis_frozen/r1f1.json
results/metrics/denfis_online/r1f1.json
results/summary.csv          # 自动重建，包含 pooled 行
results/manifest.json        # 自动重建
```

预测 CSV 的列固定为：

```text
date,y_true,y_pred,signal,position,strat_ret,bench_ret
```

不要自己写这些文件，也不要在测试段外另算指标；否则 `summary.csv` 和报告会
对不上。

---

## 4. 交付节点

| 时间 | 你要交付 | 验收方式 |
|---|---|---|
| **11/10 今晚** | `src/models/denfis.py`（frozen 可跑）、契约测试、`r1f1` 预测落盘 | `pytest tests/test_denfis_contract.py` + 单折命令成功 |
| **11/13 18:00** | `denfis_frozen` + `denfis_online` 各 9 折 | 18 个预测文件 + 指标齐全，`summary.csv` 有对应行 |
| 11/14 | 迁移学习 3 折 × 3 协议 | 用 `src/transfer.py`，结果放 `results/preds/denfis_transfer/` |
| 11/15 22:00 | 方法侧消融/可解释性图 + 结果冻结 | 写进 `docs/freeze.md` |

单折目标 <5 分钟；若规则数爆炸（>200）先调 `Dthr` 下限或开合并，不要改协议。

---

## 5. 统计检验和结果表

DENFIS 跑完后执行：

```bash
python scripts/statistical_tests.py
python scripts/make_results_table.py
python scripts/plot_results.py
```

- `scripts/statistical_tests.py` 会自动发现 `denfis_*`，把 DM 对比切换成
  「每个 DENFIS 变体 vs 表现最好的 baseline」，并额外做
  `denfis_online vs denfis_frozen`。输出 `docs/statistical_tests.md` 和
  `results/statistical_tests.json`。
- `scripts/make_results_table.py` 会自动把标题从 “Baseline results” 改成
  “Results”，buy & hold 也不再依赖某个具体模型目录。

当前 baseline 的统计结论（写报告时可直接引用）：

- XGBoost vs LSTM，DM = −0.747，p = 0.455，差异不显著；
- Sharpe 95% CI：LSTM `[−0.285, 1.114]`，XGBoost `[−0.474, 0.941]`，都跨 0。

---

## 6. ^STI 迁移实验怎么接

```python
from src.transfer import load_transfer_frames, feature_matrix

frames = load_transfer_frames()      # {"DBS": ..., "STI": ...}
dbs, sti = frames["DBS"], frames["STI"]
# 两个 frame 已按 label_date 逐行对齐，长度相同（当前为 4,146 行）。
X_sti = feature_matrix(sti)
```

约束：

- 两个市场**各自**用本市场训练段拟合 scaler，规则在标准化空间迁移；报告里要
  写明这个假设。
- STI 的训练截止日必须与 DBS 对应折的 `final_train_end` 一致，禁止用未来数据。
- ^STI 的 `vol_z` 因指数成交量大量为 0 而退化，不要把它解释成有效信号。

---

## 7. 两个待你确认的事项

1. **协议 v1.1**：`config.py` 和 `PROTOCOL.md` 标注 “pending A sign-off”。
   现有 baseline 结果就是按 v1.1 的 XGBoost 正则化网格跑的；你确认后我把
   pending 标记去掉。
2. **2026 holdout**：baseline 的 holdout 预测已经存在（`results/preds/*/holdout.csv`），
   但按协议它只用于最终冻结检查。DENFIS 的 holdout 请在 9 折全部稳定后再跑一次，
   中间不要用它调参。

---

## 8. 不要做的事

- 不要另写 CV 循环或 scaler；直接复用 `run_model_on_folds`。
- 不要修改 `config.FEATURES`、`FOLD_SPECS`、`TAU_GRID`；如需改，先升协议版本并
  通知我重跑受影响结果。
- 不要把 `denfis_online` 和 `denfis_frozen` 混在同一张主表里排名。
- 不要手工编辑 `results/summary.csv`、`results/metrics/*.json` 或 `manifest.json`。

---

## 9. 你的最小验收清单

- [ ] `pytest tests/test_denfis_contract.py` 全绿
- [ ] `python scripts/run_baseline.py --model denfis_frozen --folds r1f1` 成功
- [ ] `results/preds/denfis_frozen/r1f1.csv` 无 NaN，预测均值接近 0、标准差与收益同量级
- [ ] `results/metrics/denfis_frozen/r1f1.json` 中规则数在 2–200
- [ ] `python scripts/run_baseline.py --model denfis_frozen --folds all` 在 90 分钟内跑完
- [ ] `docs/denfis_notes.md` 写清实现与论文的差异、数值保护、失败尝试
