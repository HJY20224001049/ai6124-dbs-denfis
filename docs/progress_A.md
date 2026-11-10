# 方法线进度与交接

更新日期：2026-11-10（新加坡）。负责人：LIN Jingzhao；编码助手按授权执行。
分支：`feat/denfis-method-line`。详细交接见 [handoff_to_B.md](handoff_to_B.md)。

## 已完成并可交接

| 计划节点 | 当前产物 |
|---|---|
| D0–D2：方法与首折 | DENFIS、契约测试、公式/偏差笔记、12 条文献、真实数据首折 |
| D4–D5：扩展与全折 | frozen/online 各九折，共 18 份预测与指标；规则与参数诊断 |
| D6：迁移 | 三折 × scratch/严格 zero-shot/fine-tune，独立结果目录 |
| D7：分析 | 两类消融各四折、验证敏感性、规则演化、隶属度和高波动日解释图 |
| D8：英文初稿 | writing/A_method.md，约 3,400 词；结果解释明确不声称显著提升 |
| D9：方法交付 | 已执行的方法 Notebook、五页 PPTX、讲稿；完整 Notebook 等 B 集成 |

## 验证结果

- Python 3.11 虚拟环境；数值依赖按 requirements.txt 锁定。baseline 历史产物保留，未重跑。
- 全套测试：48 passed。
- 产物验收：35 个预测文件、8,793 行；日期/目标、阈值、成本、规则上限、STI 截止日全部通过。
- 当前代码重放 18 个已选 frozen/online 模型，与保存预测在浮点保存精度内一致。
- 泄漏审计：133 checks, 0 failures；详见 docs/leakage_audit.md。
- 方法 Notebook 在新 kernel 全部执行通过；PPTX 五页可解析，所有元素位于画布内。尚未进行真人计时彩排。

## 已知结果

- frozen pooled：RMSE 0.012066，DA 50.09%，Sharpe 0.654。
- online pooled：RMSE 0.012185，DA 50.18%，Sharpe 0.260。
- 预测误差未优于 baseline；DM 比较均不显著，Sharpe CI 跨零。
- frozen Sharpe 点估计高于两个 baseline，低于买入持有；不能写成已确立的交易优势。

## 尚待实际发生

- 11/11 咨询与真实答复记录；不能代替本人参会。
- v1.1 原待确认标记仍保留；本轮没有改变其特征、网格、折和成本设定。
- 11/15 22:00 正式冻结与之后的最终 DENFIS holdout，当前未提前执行。
- B 集成最终 Notebook、双方交叉审阅、11/17 彩排和 11/18 双人各自提交。

## 同步状态

代码和产物已提交本地分支；**尚未推送远程**。
Git HTTPS 无凭据、SSH 未获公钥授权；GitHub 连接器写入 tree/ref 均返回 403。
权限恢复后执行 `git push -u origin feat/denfis-method-line`。
另提供 `delivery/denfis-method-line.bundle` 和 ZIP，供有权限的机器导入与交接。
