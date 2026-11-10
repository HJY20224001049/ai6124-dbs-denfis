# Repository Guidelines

## 项目结构与模块组织

本项目使用 Python 预测星展银行（DBS）股票的次日收益率，并将 DENFIS 与 XGBoost、LSTM 基线进行比较。修改实验行为前先阅读 `PROTOCOL.md`；特征、数据折、随机种子、路径和交易假设统一定义在 `config.py`。

- `src/`：数据加载、因果特征、时间划分与迁移数据对齐。
- `src/models/`：模型接口、DENFIS 与基线实现。
- `src/evaluation/`：滚动前推交叉验证、指标、回测与统计检验。
- `scripts/`：实验运行、审计、绘图与报告生成脚本。
- `tests/`：使用 pytest 编写的接口契约与数据泄漏测试。
- `data/raw/`：缓存行情 CSV；`results/`：预测、指标、图表与运行清单；`docs/` 和 `writing/`：方法说明与报告草稿。

## 环境配置、测试与开发命令

所有命令均在仓库根目录执行；项目没有单独的构建步骤。

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
python -m src.data                           # 下载并缓存行情
python scripts/check_pipeline.py             # 检查特征和数据折，生成汇总
python -m pytest -q                          # 运行测试
python scripts/run_baseline.py --model xgboost --folds r1f1
python scripts/leakage_audit.py               # 审计缓存数据与实验产物
```

使用 `--folds all` 运行九个常规数据折。通过 `python scripts/make_results_table.py` 更新结果表，通过 `python scripts/plot_results.py` 更新图表。

## 代码风格与命名约定

遵循现有 Python 风格：使用四空格缩进，函数和模块采用 `snake_case`，类采用 `PascalCase`，常量采用 `UPPER_SNAKE_CASE`。公共接口应添加类型注解和简洁的文档字符串。共享配置从 `config.py` 导入，避免重复硬编码。当前未配置代码格式化或静态检查工具。

## 测试规范

测试文件命名为 `tests/test_<module>.py`，测试函数命名为 `test_<behavior>`。优先使用固定种子的合成数据，验证特征因果性、标签日期边界清理、预测形状、输出数值有限性与结果可复现性。开发时运行相关测试文件，提交评审前运行完整测试集。当前没有覆盖率百分比要求；数据泄漏审计失败时，不得将相关结果纳入主要实验结论。

## 实验完整性与可复现性

预处理参数只能使用训练折数据拟合。保持特征顺序与配置中的随机种子一致。结果冻结前，不得使用 2026 年留出集调参或评估。缓存数据必须支持离线复现实验。

## 提交与合并请求规范

现有提交使用 `feat:`、`chore:` 等前缀，主题简洁并直接描述所做修改；新增提交应沿用该风格。合并请求应说明修改内容、涉及的实验协议假设，以及验证命令和结果。有相关 issue 时附上链接；若报告结果发生变化，应同时更新对应表格或图表。
