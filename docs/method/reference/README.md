# EviSOZ-LM 方法参考资料

本目录是 EviSOZ-LM clean worktree 中的**受控方法参考层**。它保存当前在线主链和
Stage-0 数据合同直接依赖的设计、协议和审计文档；不把父工作区 `research/02_method`
下的全部历史实验记录复制进来。

## 核心入口

- [`evisoz_lm_repository_aligned_design_v1_20260830_zh.md`](evisoz_lm_repository_aligned_design_v1_20260830_zh.md)：仓库对齐后的完整方法与训练边界。
- [`evisoz_evidence_json_runtime_usage_v1_20260901_zh.md`](evisoz_evidence_json_runtime_usage_v1_20260901_zh.md)：Evidence JSON 拆分后的真实 schema、生产/消费入口和实例路径。
- [`soz_target_definition.md`](soz_target_definition.md)：SOZ 目标、候选集合和评估口径。
- [`labram_portable_equal_ensemble_protocol_v29_20260815_zh.md`](labram_portable_equal_ensemble_protocol_v29_20260815_zh.md)：canonical v29 H/D 冻结协议。

## 已迁移的配套协议

LaBraM v16/v17/v28 恢复与辅助协议、post-open 固定审计扩展，以及 Findings、montage/reference、
多事件报告、claim factuality、selective calibration 和知识库约束报告协议均位于本目录。
它们是 lineage 和审计参考，不会自动打开训练权限。

## 目录边界

- canonical v29 的模型定义、基础 checkpoint 和 H/D 状态位于仓库内的
  [`../../../models/canonical_v29_h_d`](../../../models/canonical_v29_h_d) 与受控 `outputs/`；完整哈希见其
  [`artifact_manifest.json`](../../../models/canonical_v29_h_d/artifact_manifest.json)。
- 知识库不在本目录，而在 [`../../../knowledge/eeg`](../../../knowledge/eeg)；它只负责术语、规则、
  不确定性和报告边界，不能创造患者事实。
- raw EDF、患者映射、私有报告正文、CerebraGloss/ELM 模型和 private prediction cache 不属于
  clean worktree 的迁移内容。
- Stage-0 当前仍为 `NO_GO`：迁移和哈希校验不等于训练授权，也不等于独立外部验证。

## 来源与可复现性

文件来自父工作区对应的冻结协议或已验证实现；迁移后代码默认只引用本仓库的
`third_party/labram/modeling_finetune.py` 和 `models/canonical_v29_h_d/labram-base.pth`。
如需复现实验，应先运行资源 registry、schema registry 和 Stage-0 gate 的只读校验，再根据 gate
状态决定后续动作。
