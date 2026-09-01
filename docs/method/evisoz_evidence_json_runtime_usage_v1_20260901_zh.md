# EviSOZ Evidence JSON：运行时使用地图与实际结构

> **目标 clean worktree 路径约定（2026-09-01）：** 本附录位于
> `/mnt/hd1/dyf/workspace/laptop/EviSOZ/docs/method/`。文中的
> `outputs/...` 均指外部受控 artifact 根
> `/mnt/hd1/dyf/workspace/laptop/EEG_Seizure/outputs/`；迁移文档不会复制或授权
> 这些生成物。当前状态必须以目标仓库重新生成的 registry、gate 和 receipt 为准。

**适用版本：** `evisoz_training_example_v1`、`evisoz_bound_evidence_example_v1`、`evisoz_private_physician_report_release_v1`（Stage‑0 r50/r53 工件）  
**状态：** 真实数据 shadow/evaluator-only；不代表训练授权或临床验证。

**最新 Stage‑0 重放：** `outputs/evisoz_stage0_gate_v1_20260901_r60/gate.json`
（`NO_GO`），执行计划为 `outputs/evisoz_execution_plan_v1_20260901_r29/plan.json`，
补件包为 `outputs/evisoz_stage0_remediation_packet_v1_20260901_r24/`。r60 已将
已有 CerebraGloss candidate-only 缓存纳入 lineage；ELM、fold-local calibration、
治理/报告 release 和公开身份审计仍未闭合。

## 1. 先给结论

方案中的“Evidence JSON”不是一个直接喂给 LaBraM 或 Qwen 的大 JSON。仓库将它拆成职责隔离的内容寻址对象：

```text
原始/派生信号
  -> event identity + split + montage receipt
  -> evisoz_field_release_v1             （字段来源/状态/权限）
  -> evisoz_training_example_v1          （单事件 envelope）
  -> event Findings projection            （直接测量/候选证据分层）
  -> reference/signal claim graph         （事件/患者级主张）
  -> knowledge selection + canonical report（术语/边界和确定性表达）
  -> evisoz_bound_evidence_example_v1     （只读绑定边界）
  -> loader / shadow inference / evaluator
```

因此，`generated_text`、知识卡片、教师候选和医生标签不会在同一个无类型字段中互相升级。每个引用都必须通过 `ArtifactRef` 的内容哈希和对应 validator 重放。

## 2. 真实 schema 和代码位置

| 层 | 实际 schema | 生产入口 | 消费入口 | 当前状态 |
|---|---|---|---|---|
| 事件身份 | `evisoz_event_identity_v1` | `src/evisoz/data/event_identity.py` | 所有 materializer/loader | 真实 88 事件 |
| 患者划分 | `evisoz_split_roster_v1`、linkage group | `src/evisoz/data/split_ledger.py` | envelope、loader、gate | 患者级绑定 |
| 双 montage | `evisoz_montage_derivation_receipt_v1`、`evisoz_dual_montage_cache_materialization_receipt_v1` | `src/evisoz/data/tcp22_views.py`、`stage0_dual_montage_cache.py` | training envelope、Findings projection、loader | 88 事件 replay 通过 |
| 字段释放 | `evisoz_field_release_v1` | `src/evisoz/data/dataset_policy.py`、`src/evisoz/forge/private_stage0_examples.py` | envelope、Findings、claim graph | evaluator-only |
| 医生报告 release | `evisoz_private_physician_report_release_v1` | `src/evisoz/data/private_physician_report_release.py`、`scripts/materialize_private_evisoz_physician_report_release_v1.py` | bound 的可选 `physician_authored_text` lane、未来 Qwen 文本 lane | 合同与绑定入口已实现，当前未发布 |
| 单事件 envelope | `evisoz_training_example_v1` | `src/evisoz/forge/training_example.py` | bound-evidence materializer/loader | 已落盘，当前无 loss |
| 教师候选 | `evisoz_teacher_candidate_cache_v1` | `src/evisoz/forge/teacher_candidates.py` | Findings/claim projection | CG 少量未校准，ELM 缺失 |
| 确定性候选 | `evisoz_deterministic_signal_candidate_cache_v1` | `src/evisoz/forge/deterministic_signal_candidates.py` | Findings/患者 signal graph | candidate-only |
| 事件 Findings | `evisoz_event_findings_projection_v1` | `src/evisoz/forge/findings_claims_reports.py` | claim/report projection | shadow/evaluator-only |
| 患者级主张 | `evisoz_reference_claim_graph_v1`、`evisoz_signal_candidate_claim_graph_v1` | 同上 | patient shadow/Qwen input | 不得产生临床 SOZ |
| 知识选择 | `evisoz_knowledge_selection_receipt_v1` | `findings_claims_reports.py` + `knowledge/eeg` validator | bound shadow report/Qwen packet | 只约束术语和边界 |
| 报告 | `evisoz_canonical_report_v1` | deterministic report builder | shadow evaluator/Qwen packet | canonical shadow |
| 绑定边界 | `evisoz_bound_evidence_example_v1` | `src/evisoz/forge/evidence_binding.py` | `src/evisoz/data/bound_evidence_loader.py` | 88 条可 replay |

关键点：`evisoz_training_example_v1` 只保存引用、状态计数和 loss 权限；它不复制 Findings 的测量数组，也不把报告正文嵌入 envelope。

## 3. 实际单事件 envelope

当前真实文件示例：

`outputs/evisoz_stage0_private_real_examples_v1_20260831/events/PRIV-E0001/training_example.json`

其实际顶层键为：

```json
{
  "schema_version": "evisoz_training_example_v1",
  "example_id": "EVISOZ-EX-...",
  "sample_id": "PRIV-E0001",
  "event_id": "PRIV-E0001",
  "dataset_id": "private",
  "linkage_group_id": "EVISOZ-PAT-...",
  "anchor": {
    "condition": "known_seizure_segment",
    "quality": "exact",
    "t0_seconds": 0.0,
    "analysis_interval_seconds": [-12.0, 48.0]
  },
  "split_assignment": {
    "evisoz_role": "development_cv",
    "outer_holdout_fold": 0,
    "locked": false
  },
  "report_scope": "full_soz",
  "artifact_refs": {
    "event_identity": {"...": "evisoz_artifact_ref_v1"},
    "split_roster": {"...": "evisoz_artifact_ref_v1"},
    "montage_derivation": {"...": "evisoz_artifact_ref_v1"},
    "field_release": {"...": "evisoz_artifact_ref_v1"}
  },
  "field_state_counts": {
    "provided": 4,
    "not_provided": 6,
    "not_evaluable": 0,
    "technical_failure": 0
  },
  "unavailable_field_ids": ["..."],
  "enabled_loss_ports": [],
  "safety_contract": {
    "generated_text_can_supervise_localization": false,
    "knowledge_can_create_patient_facts": false,
    "teacher_runtime_required_at_deployment": false,
    "node_and_edge_coordinates_interchangeable": false
  },
  "receipt_sha256": "..."
}
```

`enabled_loss_ports=[]` 是当前真实私有数据的关键事实：字段值可供 evaluator 复核，但没有治理授权，不能构造训练 loss。完整闭合定义见 [`schemas/evisoz_training_example_v1.schema.json`](../../schemas/evisoz_training_example_v1.schema.json) 与 [`src/evisoz/forge/training_example.py`](../../src/evisoz/forge/training_example.py)。

## 4. 字段内容在哪里

字段值在 `evisoz_field_release_v1`，而不是 envelope 中。一个字段行的结构是：

```json
{
  "field_id": "PRIVATE-ONSET-NODES",
  "field_path": "clinical_labels.onset_candidate_channels",
  "state": "provided",
  "authority": "physician",
  "quality_tier": "gold_lite",
  "semantic_role": "node_label",
  "value_ref": {"...": "field_value ArtifactRef"},
  "value_payload": {
    "values": ["T8"],
    "semantics": "incomplete_positive"
  },
  "claim_permission": "direct",
  "loss_permissions": {
    "typed_slot_loss": false,
    "node_localization_loss": false,
    "report_text_loss": false
  }
}
```

四个概念必须分开：

- `clinical_labels`：医生或数据集原始标签；当前私有 release 只允许 evaluation-only。
- `observed/direct_measurements`：通过 Findings 证据链可从波形重放的测量；不能用候选规则冒充。
- `teacher_candidates/derived_candidates`：CerebraGloss、ELM 或确定性规则提出的软候选；当前只能 `candidate_only/soft_auxiliary`。
- `generated_text`：语言表达，不是新的患者标签。真实 DOCX 另走 `physician_authored_text` lane，不能标为 generated text；但当前人工 release 尚未开放。

`evisoz_field_release_v1` 的 authority、claim 和 loss validator 在 [`src/evisoz/data/dataset_policy.py`](../../src/evisoz/data/dataset_policy.py)；它会拒绝 teacher/derived 对 node localization 的直接监督，也会拒绝 knowledge 创建患者事实。

医生报告不能通过修改 `evisoz_private_physician_report_deidentification_candidates_v1` 自行晋升。独立 release 必须由 [`private_physician_report_release.py`](../../src/evisoz/data/private_physician_report_release.py) 验证：候选文本必须来自已固定的 `text_ref`，患者关联必须是 high-confidence，开发文本只能用于 Qwen 文本训练，锁定测试文本只能用于语言评价；每一行还必须带外部 authorization/ref 和人工审核 receipt。该 release 不复制正文，也永远不开放 `report_text_can_supervise_localization`。

## 5. bound-evidence 才是后续组件的输入边界

实际文件示例：

`outputs/evisoz_stage0_bound_evidence_v1_20260901_r50/events/PRIV-E0100/bound_evidence.json`

它固定包含：

```json
{
  "schema_version": "evisoz_bound_evidence_example_v1",
  "bound_example_id": "...",
  "event_id": "PRIV-E0100",
  "linkage_group_id": "EVISOZ-PAT-...",
  "evisoz_role": "development_cv",
  "outer_holdout_fold": 0,
  "status": "...",
  "source_refs": {
    "training_example": {"...": "..."},
    "event_findings": {"...": "..."},
    "reference_claim_graph": {"...": "..."},
    "field_release": {"...": "..."},
    "montage_derivation": {"...": "..."},
    "dual_montage_cache": {"...": "..."},
    "patient_signal_graph": {"...": "..."},
    "knowledge_selection": {"...": "..."},
    "canonical_report": {"...": "..."},
    "physician_report_release": null
  },
  "lanes": {
    "clinical_labels": {"source": "field_release", "state": "evaluator_only"},
    "direct_measurements": {"source": "event_findings", "state": "not_released"},
    "teacher_candidates": {"source": "candidate_exposure_ledger", "state": "absent"},
    "derived_candidates": {"source": "event_findings", "state": "soft_auxiliary_uncalibrated"},
    "physician_authored_text": {"source": "private_report_release", "state": "not_released"},
    "generated_text": {"source": "canonical_report", "state": "shadow_only"}
  },
  "permissions": {
    "training_allowed": false,
    "node_localization_supervision_allowed": false,
    "report_text_loss_allowed": false,
    "prompt_or_rag_allowed": false,
    "knowledge_can_create_patient_fact": false,
    "generated_text_can_create_patient_fact": false
  },
  "receipt_sha256": "..."
}
```

后续 loader 只能返回经过 source replay 的 bound object：

```text
materialize_evisoz_bound_evidence_v1.py
  -> src/evisoz/data/bound_evidence_loader.py
  -> structural shadow inference / patient aggregation / evaluator
```

因此训练器、报告生成器和 Qwen adapter 不应直接读取 `knowledge/eeg`、DOCX 或未绑定 candidate cache。

## 6. `knowledge/eeg` 的确切使用点

`knowledge/eeg` 不是 Evidence JSON 的患者事实来源，也不参与 v29 `z0_node` 计算。它只在 Findings/claim/report 投影阶段被查询，用来：

1. 根据已存在的证据字段选择规范术语和适用规则；
2. 固定 `report_scope`、允许推理、禁止推理和不确定性措辞；
3. 生成 `evisoz_knowledge_selection_receipt_v1`，把 card ID、版本和选择依据内容寻址保存；
4. 让 deterministic report、shadow report 和未来 Qwen 输入共享同一安全边界。

当前实现入口是 [`src/evisoz/forge/findings_claims_reports.py`](../../src/evisoz/forge/findings_claims_reports.py) 中的 knowledge validator/selection 路径，知识源是 [`knowledge/eeg`](../../knowledge/eeg/README.md)。`bound_shadow_report.py` 和 Qwen structured/patient input 只接受已验证 receipt，不允许调用方注入一张未绑定卡片。

知识库可以规范化“节律性 theta”“候选起始”“提示/可能”等表达，但不能创造 `F7`、左侧、某个时间点、MRI 结果或真实 SOZ。`knowledge_can_create_patient_facts=false` 是 envelope、claim graph、Qwen packet 的共同硬约束。

## 7. 真实状态与此前 `NO_GO`

截至当前最新工件：

| 检查 | 状态 | 含义 |
|---|---|---|
| `private_real_dual_montage` | `QUALIFIED_GO` | 88/88 事件、1,936 条 TCP22 edge 可 replay；仍保留 opaque reference 和历史 v29 exposure 限制 |
| `private_field_envelopes` | `EVALUATOR_ONLY_GO` | 88 个 envelope 已落盘，但训练权限为 0 |
| `findings_claim_graph_and_reports` | `QUALIFIED_GO` | 结构化 shadow 可 replay，不是临床报告 release |
| `private_report_text_release` | `NO_GO` | 人工去标识审核/开发与锁定语言 release 尚未完成 |
| `offline_teacher_and_derived_candidates` | `PARTIAL` | ELM artifact 和 fold-local calibration 缺失；候选未校准 |
| `public_auxiliary_exposure_projection` | `PARTIAL` | near/partial overlap 与 TUEV eval identity 未闭合 |
| `clean_freeze_audit` | `NO_GO` | worktree 有既有未提交修改，不能冒充 clean snapshot |
| `Stage0_overall` | `NO_GO` | 任何正式训练、Qwen SFT、alignment 和非零 residual 均必须保持关闭 |

所以此前的 `Stage0_real_dual_montage_data = NO_GO` 不是说 88 条真实 dual-montage 回放失败；更准确的当前语义是“数据合同 `QUALIFIED_GO`，但整个 Stage‑0 因安全/治理/暴露审计仍 `NO_GO`”。

## 8. 推荐复核命令

这些命令只做 schema/replay/shadow 检查，不会启动训练：

```bash
python3 -m pytest -q tests/test_evisoz_stage0_contracts.py tests/test_evisoz_evidence_binding.py
python3 -m pytest -q tests/test_evisoz_findings_claim_reports.py tests/test_evisoz_bound_evidence_loader.py
python3 scripts/replay_evisoz_bound_evidence_loader_v1.py
python3 scripts/run_evisoz_shadow_inference_smoke_v1.py
python3 scripts/materialize_evisoz_qwen_patient_shadow_v1.py
```

若 gate 仍为 `NO_GO`，训练入口必须在构造模型、优化器和 loader 之前由 `src/evisoz/training/stage0_guard.py` fail closed。报告 release CLI 在缺少 `inputs/private_report_release_authorization.json` 或人工审核矩阵时也会直接失败；只有独立治理授权、人工报告 release、ELM/calibration artifact、公开 exposure 凭据和 clean freeze 全部重新验证后，`enabled_loss_ports` 才可能从空数组变为有权限的端口。

## 9. 训练与评价桥接的最新实现

当前新增的 `src/evisoz/training/targets.py` 将**已授权**的
`evisoz_field_release_v1` 字段转换为 typed targets：

- `node_label + node_localization_loss` 生成 `[19]` target/mask；
  `incomplete_positive` 只把明确阳性作为可评估位置，不把未列通道当作阴性；
- categorical `quality/morphology/evolution/localizability` 只接受冻结词表和
  足够确定性的标签；低/未知 certainty 返回不可评估 mask；
- spread channel-set 不会被错误转换为 TCP22 edge 类别，区域/侧别也不会被
  冒充为不存在的 decoder head；
- `report_text_loss` 永远不返回给 EEG objective。

`src/evisoz/training/evidence_trainer.py` 和
`scripts/run_evisoz_stage1_evidence_training_v1.py` 已接入真实 adapter、
Clinical Evidence Pipeline 和 typed loss。入口在 Stage‑0 `NO_GO` 时先写出
`blocked_before_model_or_loader_construction` receipt；当前 r60 receipt 为
`outputs/evisoz_stage1_evidence_training_v1_20260901_r4/receipt.json`，其中
`model_constructed=false`、`optimizer_constructed=false`、
`training_loader_opened=false`。因此该入口已经可复现地证明“先 gate、后 loader”，
但没有把空 loss port 误当作训练完成。

`src/evisoz/evaluation/clinical_localization.py` 负责 evaluator-only 的
direct-label 定位指标：candidate-set Top-1/Top-3、MRR、ECE、risk-coverage、
侧别和区域命中；teacher/derived candidate、医生正文和 TCP22 endpoint 都被
明确排除。只有在未来 field release 和 aggregate gate 同时允许时，才可将其
用于私有/锁定验证集。

## 10. r60 真实闭合与 Stage‑2 接口

当前真实闭合产物已经更新为：

```text
freeze:    outputs/evisoz_clean_freeze_audit_v1_20260901_r10/audit.json
gate:      outputs/evisoz_stage0_gate_v1_20260901_r60/gate.json
plan:      outputs/evisoz_execution_plan_v1_20260901_r29/plan.json
remediate: outputs/evisoz_stage0_remediation_packet_v1_20260901_r24/
loader:    outputs/evisoz_stage0_bound_evidence_loader_replay_v1_20260901_r53/receipt.json
```

loader replay 实际重放 88 个事件、31 个 opaque linkage groups；它只验证
bound-evidence、dual-montage cache、Findings/claim/report 和 knowledge
selection 的引用及 hash，`physician_report_text_opened=false`、
`teacher_runtime_opened=false`、`training_allowed=false`。这解释了为什么真实
数据链已经可 replay，而总体 Stage‑0 仍是 `NO_GO`。

私有真实医生报告 inventory 也已完成只读审计：43/43 为有效 DOCX，全部保持
`physician_authored`；40 条通过唯一身份规则高置信关联（开发 33、锁定语言评价
候选 7），3 条仍需外部权威 mapping。43 条自动 PHI 扫描候选均通过，但人工审核
通过数和 Qwen release 数仍为 0。因此报告的价值是为 V2 文本侧提供临床措辞、
起始/扩散叙述和边界样本；当前只能作为待审核 artifact，不能进入 Qwen 训练、语言
评价或任何 SOZ 定位监督。原始 DOCX 不会复制到 EviSOZ 输出目录。

Stage‑1 的 `stage1_objectives.py` 是证据表征 objective port；Stage‑2 的
`training/residual_trainer.py` 是显式的定位 residual 边界：

* `residual_node_localization_loss` 只接受 Standard19 `[B,19]` target/mask；
  `incomplete_positive` 未列出的节点不会变成负样本；
* `baseline_preservation_kl_loss` 在 frozen canonical v29 H/D 已正确的事件上
  约束新分布，避免辅助证据破坏基线；
* `run_authorized_residual_epoch` 先执行 aggregate Stage‑0 guard，再构造
  loader/model/optimizer；`alpha>0`、Query/Residual formal training 与
  non-zero residual 在当前 r60 均会被 fail-closed。

新增 `tests/test_evisoz_residual_trainer.py` 覆盖 mask、baseline-preservation、
稀疏可选 target 与 guard-before-loader。它们与 Stage‑1/Clinical Evidence
测试共 27 项通过；这只是实现和安全边界验证，不是训练或性能结果。

`src/evisoz/training/grounding_feedback.py` 提供 V4 的一次性反馈边界。它先
检查每条报告 claim 的 evidence IDs、节点/形态/区域集合和 certainty 是否能在
trusted structured evidence 中回放；未通过的 claim 不产生 eligibility。即使
claim gate 通过，`apply_one_shot_grounded_feedback` 仍需同时通过 aggregate
Stage‑0 guard，并只将非负 delta 施加到已有 candidate mask，不能新增节点、改变
报告事实或把 TCP22 edge 展开成 node label。当前 r60 下该函数始终在更新前
抛出 `Stage0TrainingBlocked`；对应测试为
`tests/test_evisoz_grounding_feedback.py`。

## 11. 当前 r63 运行时回执

为保持本附录与最新状态同步，当前不可变回执为：

```text
gate:      outputs/evisoz_stage0_gate_v1_20260901_r63/gate.json
plan:      outputs/evisoz_execution_plan_v1_20260901_r30/plan.json
remediate: outputs/evisoz_stage0_remediation_packet_v1_20260901_r27/
loader:    outputs/evisoz_stage0_bound_evidence_loader_replay_v1_20260901_r54/receipt.json
smoke:     outputs/evisoz_stage0_shadow_inference_smoke_v1_20260901_r13/
```

r63 的聚合门仍为 `NO_GO`。`private_real_dual_montage` 为
`QUALIFIED_GO`，表示 88 个真实事件的 CAR19 与 signed TCP22 视图和引用可重放，
并不表示训练授权或定位性能；`bound_evidence`、知识选择和结构 shadow 也只具
有 evaluator/shadow 权限。当前 remediation packet 仍等待 clean-freeze、ELM
及 fold-local calibration、私有治理授权、3 条报告关联、43 条报告人工去标识/
release，以及公开 overlap/TUEV identity 凭据。任何缺件到达前，loader 和训练
入口都必须保持 fail-closed。

## 12. r66 后的报告关联状态

3 条无法权威关联的医生报告已由独立
`outputs/private_public_mapping_split_deid_v1_20260901_r4/private_reports/exclusion_manifest.json`
以 `operational_quarantine` 方式逐条排除。该 receipt 绑定原始文档字节哈希，且
明确禁止 linkage、split、预处理、Qwen 训练和语言评价；它不是报告 release，
也不是私有训练授权。因此 r66 中 `private_report_linkage=GO` 只表示“未解析报告
不会进入数据图”，并不表示 43 条报告均可用于训练。43 条报告的人工去标识和
development/evaluation release 仍为 0，`private_report_text_release=NO_GO`。

当前聚合 gate/plan/remediation 分别为：

```text
freeze:    outputs/evisoz_clean_freeze_audit_v1_20260901_r10
outputs/evisoz_stage0_gate_v1_20260901_r66/gate.json
outputs/evisoz_execution_plan_v1_20260901_r31/plan.json
outputs/evisoz_stage0_remediation_packet_v1_20260901_r28/
```
