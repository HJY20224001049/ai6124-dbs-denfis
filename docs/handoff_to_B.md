# 方法线交接：LIN Jingzhao → HU Junyu

更新：2026-11-10（Asia/Singapore）。分支：`feat/denfis-method-line`。
**这是可集成的阶段性交付；不是 11/15 的正式结果冻结。**

## 可直接使用

- `src/models/denfis.py`：ECM + Gaussian + 一阶 TSK + 加权 RLS；frozen/online 接口与原 factory 一致。
- `tests/test_denfis_contract.py`：非线性学习、重置、纯预测、ECM、上限、合并、RLS 闭式解与加速等价测试。
- `results/preds/denfis_{frozen,online}/`：各九折；指标目录含 20 组验证记录、选中参数和规则诊断。
- `docs/results_table.md`、`docs/statistical_tests.md`：主表与 online 扩展分开，2026 baseline 结果保留并说明历史状态。
- `results/transfer/`：scratch / strict zero_shot / fine_tune，三个对齐折。独立目录，避免三折结果混入主表。
- `results/analysis/`：消融、验证网格、规则快照与高波动日贡献；`results/figures/denfis_*.png` 可直接用于报告。
- `writing/A_method.md`：英文文献、方法、分析和未来工作初稿，约 3,400 词；`docs/refs.md` 有 12 条文献来源。
- `writing/A_method.ipynb`：方法集成草稿，核心实现已内联；不替代你负责的最终完整 Notebook。
- `writing/A_method_slides.pptx`：五页方法 slides 草稿；`writing/A_slides_speaker_notes.md` 是讲稿。

## 当前结果口径

| 九折 pooled | RMSE | DA | Sharpe |
|---|---:|---:|---:|
| frozen | 0.012066 | 0.5009 | 0.6541 |
| online（扩展） | 0.012185 | 0.5018 | 0.2595 |

- frozen 的预测误差未优于 baseline；策略 Sharpe 点估计高于两个 baseline，仍低于 buy-and-hold。
- frozen Sharpe 95% CI 为约 [-0.041, 1.309]；vs XGB 的 DM p=0.2185。不得写成显著提升。
- online vs frozen 的 DM p=0.2743，没有已确立的在线优势。
- 多数代表折的合并未触发；另加实际发生合并的 r2f1 做诊断。不要把“有合并代码”写成“合并已证明有效”。
- no-RLS 消融同时改变局部模型复杂度；不能解释成仅对更新公式的纯因果消融。

## 评估线需要审阅的修改

1. 修复 `--folds all` 实际包含 holdout 的问题，默认只运行九折；未知折名报错。
2. RMSE 完全同分时按协议选少规则、更大 Dthr；新增搜索和规则诊断。
3. baseline 按需 import，因此只跑 DENFIS 不要求安装 torch/xgboost。
4. `run_fold(..., selection_frame=...)` 允许严格 source-only 验证。
   zero-shot 的拟合、选参和选交易阈值均不用 DBS 标签；scratch/fine-tune 用 DBS 验证。
5. 转移目录采用 `results/transfer/preds/{protocol}/{fold}.csv`，区别于旧 handoff 的单目录设计，防止三个协议互相覆盖或污染九折统计。
6. online 的合并扫描做了等价预筛选；参考实现状态对照与九折选中模型 replay 验证其一致性。
7. 历史 baseline manifest 保存在 `docs/baseline_manifest.json`。旧公共 writer 使用 10 位有效数字保存预测；本轮审计按相应浮点容差比较。

## 接入与验证

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -r requirements-deliverables.txt  # 仅 Notebook/slides 生成需要
python -m pytest -q
python scripts/verify_method_artifacts.py
python scripts/leakage_audit.py
python scripts/build_method_deliverables.py
```

完整重跑入口：

```bash
python scripts/run_baseline.py --model denfis_frozen --folds all
python scripts/run_baseline.py --model denfis_online --folds all
python scripts/run_transfer.py
python scripts/analyze_denfis.py
python scripts/statistical_tests.py
python scripts/make_results_table.py
python scripts/plot_results.py
```

## 仍待发生的事

- 11/11 真实课程咨询：使用 `docs/consultation_handout.md`，把实际答复写入 `docs/consultation.md`。
- v1.1 正则化网格与当前 baseline 配置一致，本轮未改变实验网格；原待确认标记未冒充本人签名移除。
- 11/15 联合复核和正式冻结：`docs/freeze.md` 当前明确标为 NOT FROZEN。
- DENFIS holdout 尚未跑，不能为调参提前运行；baseline holdout 在接管前已有。
- 将方法稿、内联代码和图表合入最终 Notebook；核对文献编号、学号、英文表述；真正彩排并由两人各自提交。

## 远程同步状态

本地分支和分阶段 commits 已保存。当前 Git HTTPS 无凭据、SSH 无可用公钥权限；
GitHub 连接器虽能读取仓库，但创建 tree/ref 均返回 403 Resource not accessible by integration。
因此**尚未推送 GitHub**，不能假设远程 main 已包含本交付。
权限恢复后执行 `git push -u origin feat/denfis-method-line`，再交叉审阅合并。
离线交付包将另行提供，用于在有权限的机器上导入该分支。

离线导入：在已有仓库中执行 `git fetch /path/to/denfis-method-line.bundle feat/denfis-method-line:feat/denfis-method-line`，再 checkout 该分支。ZIP 是完整已提交文件快照，不含虚拟环境。
