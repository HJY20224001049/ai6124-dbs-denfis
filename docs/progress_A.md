# 方法线进度与交接

更新日期：2026-11-10（新加坡）。负责人：LIN Jingzhao；由编码助手执行。
分支：feat/denfis-method-line。结果未冻结。

## 已完成

- DENFIS 的 ECM、Gaussian/TSK、加权 RLS、规则上限、合并、在线更新已实现。
- 5 项模型契约测试通过；host 环境首折 smoke test 成功。
- 正式运行环境已建立：Python 3.11，数值依赖按 requirements.txt 锁定。
- runner 按需导入 baseline 依赖；补充同分规则、搜索记录和规则诊断。
- 方法公式、实现偏差与咨询问题已落盘。

## 正在进行

- 锁定环境的全套测试、frozen/online 九折实验。
- 文献核实、迁移实验、消融与可解释性产物。

## 需要人工完成／确认

- 11/11 咨询：请 LIN Jingzhao 参加并补充 docs/consultation.md。
- v1.1 baseline 网格沿用当前实现；仍保留 pending A sign-off，未冒充本人签字。
- baseline holdout 已在接管前生成；本轮不运行或调阅 DENFIS holdout。
- 11/15 正式冻结、11/17 彩排与 11/18 NTULearn 提交尚未发生。

## 胡俊宇可接入的接口

保持 DENFIS(dthr, lam, max_rules, merge_factor, seed, evolve_rules)。
fit/predict/partial_fit 签名不变，预测不会更新模型。不要再实现另一套评估。
模型额外暴露 firing_strengths、centers_、radii_、coef_ 与诊断计数器，供写作解释。
