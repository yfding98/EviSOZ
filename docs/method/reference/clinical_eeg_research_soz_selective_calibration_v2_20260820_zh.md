# 长程 EEG 研究性头皮 SOZ 选择性校准 v2

## 目的

本模块不再把“是否输出候选”和“证据有多强”混为一件事。对每份具有有效 EEG-only 事件排序的记录始终输出 Top-k 头皮电极候选；开发集校准只决定报告使用“证据相对较强”“证据有限”或“证据较弱/模式不一致”哪一档措辞。任何档位都不能表述为皮层 SOZ、致痫区、治疗靶点或个体概率。

实现位于 `src/clinical_eeg_long_recording/research_soz_selective_calibration_v2.py`。

## 输入边界

预测侧每条记录只接受四个字段：患者不透明 ID、记录不透明 ID、冻结后的 `research_soz_prediction` artifact 和与之哈希绑定的 `research_soz_evidence` strength artifact。schema 会拒绝 EDF annotation、Excel 字段、医生标签、临床自由文本和任意额外字段。模块从两个已验证 artifact 抽取 Top-k 与以下 EEG-only 描述量：事件数、Top-1/Top-3 跨事件支持率、JS 一致性、模式簇、多模式标志、归一化熵、首位 margin 和原描述性证据层。

这些量经预先固定、无标签参与的公式得到 `[0,1]` 排序分数。该数仅用于风险-覆盖排序，不是置信概率，也不得进入概率措辞。

## 标签边界

DeepSOZ 所用 TUSZ 标签被明确建模为患者/记录级临床头皮电极弱标签，不是逐事件皮层 SOZ 真值。一个样本可有多个 hard-positive 电极；Top-1、Hit@3 和 MRR 均以命中其中任意一个为正确。soft spread 不属于此 TUSZ hard GT，标签 schema 不提供该字段，传入即失败。

## 患者隔离与冻结顺序

1. 先分别冻结 `source_dev` 与 `source_eval` 的 EEG-only 预测 cohort，并生成内容哈希。
2. source-dev 标签只能绑定完全相同的 source-dev prediction hash。
3. 拟合函数显式接收尚未读取标签的 source-eval prediction cohort，冻结其内容哈希和患者 membership token。
4. source-dev 与 source-eval 患者 token 必须零交集，否则拟合失败。
5. 只在 source-dev 上按患者宏平均 Top-1 风险选择“满足风险上限时覆盖最大”的两条阈值。
6. source-eval 评价只接受预先锁定的 prediction hash，禁止重拟合、模型选择或阈值选择。

完整资格化采用三层患者隔离：`source_train` 可训练和选择跨事件排序器；`source_dev` 只冻结连续检测 operating point 与报告措辞/risk 阈值；`source_eval` 在所有模型和阈值冻结后只打开一次。当前代码只实现了措辞选择性校准契约，尚未训练能纠正 P8/T8/T7 集中偏倚的新跨事件 ranker。旧 DeepSOZ OOF 约 47% 的结果和 123 份私有研究 sidecar 均不得改写成 v2 成绩。

本 Python 模块是无状态的，能锁定输入并记录一次性发布要求，但不能独立阻止进程外重复调用。因此正式 source-eval 必须由外部只增不改的访问台账保证只释放一次；评价 receipt 不虚构本模块具有持久化防重放能力。

## 输出与评价

全覆盖结果包含记录微平均与患者宏平均的 Top-1 accuracy、Hit@3、MRR。MRR 在已输出 Top-k 内计算，未召回 hard positive 记为 0，并在 schema 中显式声明截断语义。

选择性结果以 Top-1 集合错误率为 risk，按相同分数全部纳入的 tie-safe 阈值生成 risk-coverage 曲线，并报告 AURC 及患者宏风险对记录覆盖率积分。阈值只控制研究性措辞，不删除、改写或重排 Top-k。

所有 prediction cohort、label cohort、calibrator、projection 和 evaluation artifact 均使用 canonical JSON 内容哈希；验证器同时检查严格键集合、嵌套语义、患者泄漏、哈希绑定、覆盖率计数和 AURC 复算。

## 尚未完成的资格化工作

本实现是协议和代码基础，不代表阈值已在真实 DeepSOZ source-dev 上冻结，也不代表 source-eval 已打开。正式使用前仍需在患者隔离的真实 source-dev 上拟合一次，在外部访问台账控制下对 source-eval 评价一次。将 TUSZ 阈值迁移到私有数据时属于外部数据分布迁移，必须单独报告校准漂移，不能沿用 source-dev 风险保证。

## 集成入口

`scripts/run_research_soz_selective_calibration_v2.py` 提供 `freeze-predictions`、`materialize-source-dev-labels`、`fit`、`project` 和 `evaluate-source-eval` 五个分离命令。预测冻结和投影命令没有标签参数；开发标签命令拒绝 source-eval；正式评估命令必须提供外部一次性访问台账 receipt，且授权验证通过前不会打开 source-eval 标签。请求结构由 `schemas/clinical_eeg_research_soz_selective_calibration_v2.schema.json` 约束。

报告语言质量仍走独立的 `scripts/evaluate_clinical_eeg_report_language_quality_v1.py`。仅同记录、去标识化、完整医生 EEG 报告可用于 corpus BLEU-1～4、ROUGE-L，以及可选本地 METEOR/BERTScore。Excel“起始”只用于报告冻结后的 SOZ 事实一致性，不能成为 BLEU/ROUGE 等语言指标的参考文本。
