# EviSOZ-LM：仓库对齐的方法架构与分阶段实施合同 v1

> **目标 clean worktree 路径约定（2026-09-01）：** 本文迁移到
> `/mnt/hd1/dyf/workspace/laptop/EviSOZ/docs/method/`。文中的
> `outputs/...` 均指外部受控 artifact 根
> `/mnt/hd1/dyf/workspace/laptop/EEG_Seizure/outputs/`，不是本文档所在仓库的
> 默认输入；历史 receipt 不能替代当前 Stage-0 授权。Evidence JSON 的逐字段
> 运行时地图见同目录的
> [`evisoz_evidence_json_runtime_usage_v1_20260901_zh.md`](evisoz_evidence_json_runtime_usage_v1_20260901_zh.md)。

**日期：** 2026-08-30  
**双 montage 修订：** 2026-08-31；canonical v29 H/D 保留为 **Standard19-CAR 冻结 SOZ 定位参考 `z0_node`**，TCP22 不替换它，而作为独立的有符号 bipolar-edge 证据视图，用于相位反转、局部场、起始演变和传播建模。两路必须由同一原始记录派生并绑定同一时间轴；任何一路不可由另一路无依据补造。common17、SOZPreNet 和 TCP22→19 LaBraM integration 只作为历史/辅助 comparator，不定义正式 `z0_node`。  
**状态：** 方法设计与实施合同；尚未训练新的 Evidence Query Decoder、EEG-to-Qwen connector 或 Qwen3.8 LoRA，亦不代表新增临床验证结果。
**任务边界：** 输入为已经给定发作锚点的一次或多次头皮 EEG 片段；不包含连续 EEG 发作检测。自动输出是最早头皮可见发作起始候选、早期演变/募集证据、跨事件头皮定位一致性和需要医师复核的研究性报告；只有结合获授权的独立多模态证据与临床复核后，才可升级为非侵入性 SOZ 假设。它不等同于皮层 SOZ、致痫区、切除靶点或临床签署结论。

**Qwen 本地工件核验（2026-09-01）：** `models/Qwen3.8-27B-FP8` 的 README 将 base model 声明为 `Qwen/Qwen3.8-27B`，权重索引引用的分片均存在；配置解析得到 `Qwen3_5ForConditionalGeneration`、语言隐藏维度 5120、64 层和 262,144 上下文，符合当前 Qwen connector 的 32×5120 接口。可重放 receipt 为 `outputs/evisoz_qwen38_runtime_probe_v1_20260901/receipt.json`，入口为 `scripts/validate_evisoz_qwen38_runtime_v1.py`。本次仅完成文件/配置及 vLLM `ModelConfig` 解析，没有加载权重进行生成；当前机器没有可用 NVIDIA 驱动，因此 GPU 端到端生成仍为未验证。该 receipt 明确不授予 Stage‑0、Qwen SFT 或 EEG-to-Qwen alignment 权限。

**继续执行更新（2026-09-01）：** 在加入上述 Qwen probe、修复 teacher discovery 对生成式 `outputs/` 文件的假阳性，并用新物化的 Findings/claim/report 与 bound-evidence 工件重放后，当前 gate 为 `outputs/evisoz_stage0_gate_v1_20260901_live_r6/gate.json`，执行计划为 `outputs/evisoz_execution_plan_live_r6_20260901/plan.json`；两者均明确为 `NO_GO`/`STAGE0_NO_GO`。新增 probe、discovery 修复和 shadow replay 不属于 Stage‑0 训练授权条件，也没有改变六项阻断。当前可继续执行的动作仍限于外部凭据包准备、只读审计和实现测试；不能以“配置可解析”替代 GPU 生成、ELM checkpoint、fold-local calibration、报告人工 release、治理授权或公开重叠权威凭据。

**前一版真实回放（2026-09-01）：** 使用 r27 bound-evidence、r3 Findings/claim/report 和真实私有 dual-montage cohort 生成了 r47 gate/r16 plan；结果为 `Stage0_overall=NO_GO`，并保留 `edf_reference_token_unobservable` 与 `locked_test_has_prior_frozen_v29_exposure` 的安全边界。该历史回放没有打开医生 DOCX、没有加载 ELM、没有运行 Qwen 或任何正式训练。

**当前真实回放（2026-09-01）：** physician-authored release lane 接入后，使用 r3 Findings/claim/report 重新生成了 `outputs/evisoz_stage0_bound_evidence_v1_20260901_r50`（88 个事件，报告 release 计数 0）。绑定层的 source refs 现在额外保留可选 `physician_report_release=null`，历史无 release 工件仍可重放；最新定向 loader replay 继续保持 `physician_report_text_opened=false`。随后将已验证的 CerebraGloss development-only candidate materialization（2 events、29 candidates，仍未校准且不能监督 SOZ）传入 gate，生成 `outputs/evisoz_stage0_gate_v1_20260901_r51/gate.json` 与 `outputs/evisoz_execution_plan_v1_20260901_r20/plan.json`；整体仍为 `NO_GO`，但已移除其“candidate artifact missing”子阻断，剩余 ELM/calibration 与治理/审计阻断不变。该回放只验证绑定、mask、source-ref 和权限合同，不代表模型性能、报告文本 release 或训练授权。

**历史可复现状态（2026-09-01，live_r3）：** Stage-0 gate `live_r3` 为 `NO_GO`；真实双 montage 检查为 `QUALIFIED_GO`，不是回退到 common17。当前仍禁止正式 Query Decoder/residual、Qwen SFT、EEG-to-Qwen alignment、私有标签训练和医生报告语言评价。该历史 gate、执行计划和 remediation packet 分别为：

```text
outputs/evisoz_stage0_gate_v1_20260901_live_r3/gate.json
outputs/evisoz_execution_plan_live_r3_20260901/plan.json
outputs/evisoz_stage0_remediation_packet_v1_20260901_r11/
```

本轮的 `evisoz_teacher_artifact_discovery_v1` 仍是只读 inventory：对 ELM 的实际扫描结果为 `missing`，对 CerebraGloss 的完整数据目录扫描为 `found_unvalidated`；CerebraGloss 的 2-event/29-candidate materialization 已由独立 candidate validator 验证并传入最新 gate，但仍不生成 calibration receipt 或训练授权。公共近/部分重叠审计现在有可选的 `evisoz_public_overlap_audit_receipt_v1` 导入合同，只有数据集权威签发的三项完整 receipt 才能让 gate 移除对应 exposure blockers，receipt 本身仍不授予训练权限。

**Evidence JSON 的实际落点：** 方案中的“大而全 Evidence JSON”不作为第二套事实本体；它在仓库中实现为 `evisoz_training_example_v1` envelope 加多个内容寻址引用。患者/事件/split/montage/字段权限由 envelope 和 `evisoz_field_release_v1` 承载；直接测量与派生候选由 Event Findings projection 承载；教师候选由 candidate cache 承载；患者级综合由 reference/signal claim graph 承载；知识选择和确定性报告分别由 selection receipt/canonical report 承载。`bound_evidence` loader 只在这些对象通过验证后重新绑定它们，不复制或升级事实。当前 88 个真实事件已经物化为 evaluator-only/shadow 对象，但 `training_authorized_event_count=0`，所以不能称为正式训练数据。

> Evidence JSON 的逐字段运行时地图、真实实例路径和可复核命令见 [`research/02_method/evisoz_evidence_json_runtime_usage_v1_20260901_zh.md`](evisoz_evidence_json_runtime_usage_v1_20260901_zh.md)。该附录以当前 r50/r50 工件为准，不把设计态富结构示例误写成已落盘或已授权结构。

当前已加入 implementation-only 的 `src/evisoz/models/clinical_evidence.py`：Clinical Motif Adapter、稀疏时空证据编码器、六查询 Evidence Decoder、非负 gated residual 和患者级聚合。其输入合同固定为 `[B,19,T,128]` 的 Standard19 node tokens 与 `[B,22,T,128]` 的有符号 TCP22 edge tokens，并强制显式 mask；全通道缺失时使用零 sentinel 仅保证数值稳定，不创建证据。稀疏编码器现在会将 Top-k relation mask 转换为带 mask 的节点邻居上下文，并在空间聚合后再次应用 node mask，避免缺失节点被“复活”。`src/evisoz/inference/patient.py` 将 loader-backed event packet 按 opaque linkage group 聚合，并保留 abstention/non-localizing 状态。`configs/evisoz_structured_evidence_pipeline_v1.json` 及对应 schema/registry binding 已落盘；在 Stage 0 `NO_GO` 时 smoke path 严格保持 `z1=z0`，不开放正式训练。可用 `rtk python3 scripts/smoke_evisoz_structured_evidence_pipeline_v1.py` 复现合成 smoke，但其结果不代表真实数据性能。

所有未来 trainer 必须先调用 `src/evisoz/training/stage0_guard.py` 的 `require_stage0_training_authorized`，同时验证聚合 gate、阻断检查和 pipeline config；当前 `live_r3` `NO_GO` 会抛出 `Stage0TrainingBlocked`，因此不会意外创建正式 Query/Residual、teacher 或 Qwen 数据加载器。`src/evisoz/training/loader_entrypoint.py` 将这一顺序固化为 guard-before-loader；这个 guard 是执行安全边界，不等同于治理授权，也不会把 evaluator-only 字段提升为训练标签。

`src/evisoz/training/typed_loss.py` 进一步把 `typed_slot_loss`、`node_localization_loss` 和 `report_text_loss` 固化为显式 loss ports：只有 bound training example 声明的端口才可请求，且先经过 Stage‑0 authorization 再构造目标；locked/external 样本不能开启 loss，report text loss 明确拒绝进入 EEG 路径。当前 `live_r3` gate 下，任何非空 loss port 都会在 objective 计算前 fail closed；空端口只返回零梯度占位，不代表训练已授权。

Qwen 连接层已以无模型依赖的数值组件落盘于 `src/evisoz/models/qwen_connector.py`：`EvidenceTokenResampler` 将变长证据压缩为 32 个 5120 维 token，`clause_mil_alignment_loss` 使用多正样本而非单 patch 对齐，`evidence_guided_mask` 只遮挡已获证据支持的 token。`src/evisoz/models/predicted_evidence.py` 将 decoder 输出序列化为节点/边分离的 `evisoz_predicted_evidence_v1` packet，`src/evisoz/reporting/predicted_report_plan.py` 再生成 `model_candidate_shadow` 计划；`src/evisoz/reporting/bound_shadow_report.py` 强制报告计划只能复放 bound record 的 knowledge-selection card ID。两者均保留 mask、置信度和禁止升级的权限字段。它们目前仅做合成 shape/finite smoke；没有加载 Qwen 权重，也没有在 Stage 0 `NO_GO` 下执行任何 LoRA 或文本训练。

本轮新增 `src/evisoz/reporting/qwen_structured_input.py` 作为二者之间的显式合同：它同时验证 predicted report plan 与 `knowledge/eeg` 的 selection receipt，只把 card ID、knowledge bundle/version 和 candidate-only 计划交给后续 Qwen adapter；卡片正文、原始 EEG、医生正文和任何未绑定患者事实均不进入 packet。packet 固定为 `shadow_input_no_generation`，`qwen_generation_allowed=false`，并声明 32×5120 的 evidence-token runtime contract。`src/evisoz/reporting/qwen_patient_input.py` 又将患者级 signal-derived candidate claim graph、deterministic canonical shadow report 与同一 selection receipt 绑定，作为多发作聚合后的 Qwen 输入边界；它不会把事件候选升级为患者 SOZ。真实 loader-backed shadow smoke 已覆盖 88 个事件和 31 个患者级 packet，输出 `outputs/evisoz_stage0_shadow_inference_smoke_v1_20260901_r9`，仍只报告结构一致性而非性能。`src/evisoz/models/qwen_connector.py::assemble_qwen_embedding_inputs` 只负责在运行时把 `[B,K,5120]` EEG evidence embeddings 插入文本 `[B,L,5120]`，返回 attention mask 与 EEG modality mask；该函数不分词、不调用 Qwen，也不改变任何定位或 certainty 字段。

患者级 packet 现在另有独立的 `evisoz_qwen_patient_shadow_materialization_v1` manifest，由 `scripts/materialize_evisoz_qwen_patient_shadow_v1.py` 从 loader receipt、结构 shadow evaluation 和 31 个 packet 重新构建。manifest 为每个 opaque linkage group 记录事件 roster、split role/fold、packet ArtifactRef，并可从 `outputs/evisoz_stage0_qwen_patient_shadow_v1_20260901_r4` 逐文件回放；同目录的 `patient_qwen_evaluation.json` 还逐患者重放 signal graph、canonical report 和 knowledge selection 到 Qwen packet 的绑定，四项 replay rate 均为 1.0。该 bundle 仍是 `real_loader_patient_shadow_materialized`、`training_allowed=false`、`qwen_generation_allowed=false`，不会改变 Stage‑0 总体 `NO_GO`。

评价入口已补齐于 `src/evisoz/evaluation/metrics.py`，并新增 `src/evisoz/evaluation/bound_evidence_eval.py` 的 loader-backed structural shadow evaluator，覆盖 candidate-set Top-k、MRR、ECE、multiclass Brier、risk–coverage、onset-before-spread 顺序、unsupported-claim rate 及 Correction/Corruption Rate。shadow evaluator 只检查 bound loader 的事件/mask/packet/report-plan 绑定，不打开 evaluator-only 标签；真实 private/locked 评价仍受独立 release 和 split gate 控制。

**实施进度（2026-08-31 至 2026-09-01）：** P0 的 schema/channel/montage 子合同、P1 public frozen-baseline 切片、私有真实双 montage/字段 envelope、88 个真实事件的确定性信号候选缓存，以及 Findings/claim/report 的只读 shadow 投影已经落地，但 P0/Stage 0 尚未整体完成。candidate exposure ledger 已将 88 个事件、4,030 个 deterministic proposal 及 development/locked-test 暴露关系固化为不可训练的 lineage receipt；另为 3 条 unresolved 医师报告物化了不含姓名、路径或正文的 authoritative mapping intake，并在 Stage-0 gate 中绑定当前 inventory/split，防止 intake 陈旧或请求集合漂移。历史 gate/registry 产物保留为不可变审计记录；当前 schema registry 为 `EVISOZ-SCHEMAS-9357beb7fed44bc6a5b589af`（仓库 registry 文件当前 SHA-256 `12978e0b85b5d5bd561cf1f992950de141e8da6ae78c21d20df06a7c85ec1978`），已新增绑定 `evisoz_qwen_structured_input_v1` 与 `evisoz_qwen_patient_input_v1`，并更新 shadow evaluator 对 Qwen packet 的 source replay 检查。最新聚合门位于 `outputs/evisoz_stage0_gate_v1_20260901_r30`，状态仍为 `NO_GO`；这不是双 montage 回退，`private_real_dual_montage` 仍为 `QUALIFIED_GO`。新增的 `outputs/evisoz_public_v29_tusz_crosswalk_v1_20260831/crosswalk.json` 已证明 102/102 public-v29 患者到 TUSZ identity-v2 的唯一映射（553 条记录）；新增的 `outputs/evisoz_public_auxiliary_field_release_v1_20260831/field_release.json` 仅发布 TUSZ 字段能力目录（579 患者、5 字段、无值），两者均只用于 leakage/permission audit，不能授权训练。

正式 public held-fold cache 仍位于 `outputs/evisoz_v29_public_held_fold_cache_v2_20260831`：`cache_id=EVISOZ-V29-98cb2658519d1cbc4016f808`，`p0_C18=[102,18]`，tensor SHA-256 为 `6aae67212390f4037be896d6d020468a33d7210289a0d22a590819f740a470ea`，materialization receipt SHA-256 为 `096f5ea18f72e6465f3fb731b0f351f4689eb498565d9abd86614b1ca452cdf5`。从磁盘独立重开、`alpha=0` 硬旁路和 clone-only checkout mutation-isolation 均通过；`target_tensor_values_deserialized=false` 且 `targets_or_target_mask_get_tensor_calls=0`。历史 manifest 指标回放为 exact `54/102`、N4 `78/102`、Hit@3 `79/102`、Hit@5 `90/102`、MRR `0.6669437885284424`；这些是冻结 manifest receipt 回放，不是读取 targets 后的独立重算。public route 的运行时内容安全审查为 `GO`；旧 v1 目录仅保留为 quarantine 审计工件。event-mean route 仍为消费端 `NO_GO`，即使 `alpha=0` 也不可绕过。

synthetic Stage 0 dual-montage carrier/materializer 已闭合单事件派生合同：由同一个 CPU contiguous `torch.float32 [21,12000]`、200 Hz、`Standard19+A1+A2` common-reference parent，独立派生 authoritative Standard19-only CAR19 v29 view、有符号 TCP22 context view，以及直接从 parent `[2000:4000]` 派生的 source-isolated onset view；缺失行必须全零并由 bool mask 标记，A1/A2 永不进入 CAR，插值不能复活正式 v29 view。

私有真实路线随后完成 120 个唯一 EDF 的 header/reference 审计：120/120 都具备唯一 `Standard19+A1+A2`，120/120 可算术派生全部 22 条有符号 TCP22 edge；但 EDF label 不显式暴露公共参考，因此正式类型为 `exact_derived_from_protocol_authorized_opaque_common_reference`，不是伪称的 header-proven `exact_derived_from_common_reference`。94 个时间支持事件中 88 个成功物化，6 个 EDF+D 因 discontinuous event clock 排除；另有 29 个事件在物化前因时间支持不足排除。88 个真实 cache（development 65、locked test 23）及 1,936 条 TCP22 edge 均通过逐字节重开验证。患者级 split 为 34 development / 9 locked；由于 frozen v29 可能已经处理过这些患者，locked 仅表示不参与新的 EviSOZ fitting，不是 pristine external validation。

真实医生报告目录审计为 43/43 个有效 DOCX；39 个由唯一文件名实名子串、1 个由报告内明确“姓名”字段高置信绑定到冻结 split，3 个保持 unresolved。隐私安全 inventory 不保存姓名、源文件名或报告正文。43 个发作期/印象到报告医师签名前的自动去标识候选均通过已知姓名、日期、联系方式和长编号扫描，但 `manual_review_pass_count=0`，所以 Qwen 训练和语言评价 release 仍均为 0。88 个真实事件已经生成 `evisoz_field_release_v1` 与 `evisoz_training_example_v1`；现有私有医生标签的外部治理状态仍是 evaluation-only，因而所有 private training loss 都为关闭，代码生成 schema receipt 不得自行创造伦理/数据治理授权。

公共辅助 exposure registry 已进一步投影为隐私安全的 EviSOZ patient split：TUSZ source-train 共 579 名患者，五折为 115/116/116/116/116；其中 70 名与 DeepSOZ source-train overlay 重合，9 名对 TUEV train-visible。投影不保存原始 TUH patient ID，并冻结“若以后获授权，自监督/辅助训练必须按 outer fold 排除”的约束。新增的 public-v29↔TUSZ identity-v2 crosswalk 已在独立 receipt 中完成 102/102 唯一患者映射（553 条记录）；新增的 field-release catalog 已发布 TUSZ 的 5 个能力字段但不含任何患者值。TUEV eval opaque identity 与 near/partial content overlap 仍未闭合，投影和 catalog 均不自行授权训练。

确定性信号候选已由同一批 88 个已验证 cache 全量物化到 `outputs/evisoz_stage0_deterministic_signal_candidates_v1_20260831`：每个事件固定产生 19 个 CAR electrode 和 22 个 TCP bipolar-edge 特征行，以及左右同步性/空间熵全局行；共生成 4,030 个未校准 proposal（attenuation 182、LVFA 75、rhythmic theta 107、rhythmic delta 1,760、frequency evolution 1,468、near-synchronous bilateral change 88、shared-electrode phase-reversal 350）。该数目是规则触发次数，不是阳性真值或性能结果。manifest 为 `EVISOZ-SIGCANDS-522192cc928aaf620bf5c643`（receipt `5e443e9f645d0ee49b748ce65e7042c79f59fd681df9b193952ac2d0d9add289`），88/88 已从源 waveform 重新计算并逐值回放。所有行强制为 `authority=signal_derived`、`status=derived_candidate`、`calibration_state=uncalibrated`、`soft_auxiliary_only=true`，并显式禁止创建 clinical label、冒充 measured fact 或直接监督 node localization。对应的 `outputs/evisoz_candidate_exposure_ledger_v1_20260831/ledger.json` 为 `EVISOZ-EXPOSURE-83e21b47348617d70553be60`（receipt `38f33ebc57789609815be9ed077efdcc95d165baee8dc6d733003c60a6022397`），其中 development 65 / locked-test 23，locked-test 的 outer fold 保持 null；CerebraGloss、ELM 和 fold-local calibration 仍明确标记为缺失，training authorization 为 false。

当前状态为：`Stage0_public_v29_reference=GO`、`Stage0_synthetic_dual_montage_contract=GO`、`Stage0_private_real_dual_montage_data=QUALIFIED_GO`、`Stage0_private_field_envelopes=EVALUATOR_ONLY_GO`、`Stage0_private_report_linkage=PARTIAL`、`Stage0_private_report_text_release=NO_GO`、`Stage0_public_auxiliary_exposure_projection=PARTIAL`、`Stage0_offline_teacher_and_derived_candidates=PARTIAL`、`Stage0_findings_claim_graph_and_reports=QUALIFIED_GO`、`Stage0_overall=NO_GO`。r9 gate 已移除 `public_v29_to_tusz_crosswalk_not_materialized` 与 `auxiliary_field_releases_not_materialized`，并将已完成的 deterministic candidate、Findings/report 和 knowledge receipt 从 authorized-next-action 列表中移除；但 near/partial overlap、TUEV eval opaque identity、私有训练授权、3 个 unresolved 报告映射、人工去标识审核、CerebraGloss/ELM 小规模校准候选、fold-local calibration 和 clean freeze audit 仍未闭合，因此不启动 Query Decoder/residual 正式训练、Qwen SFT 或大规模 teacher inference。

上段中的 r17/r27/r28/r29/r30 标识是历史摘要；本文件后续以 r31 的不可变 gate、registry 和 bound-evidence 产物为准。

绑定层之后的最新不可变 gate 为 `outputs/evisoz_stage0_gate_v1_20260901_r31/gate.json`（`NO_GO`）；新增的 `bound_evidence_materialization` 检查为 `GO`，其余阻断项保持不变。对应 registry 已纳入 loader replay、结构化 shadow evaluator、event-level 和 patient-level candidate-only Qwen input、患者级 shadow materialization manifest、患者级 graph/report/selection evaluator，以及可复现实验计划的 schema，并绑定其实现字节。

补充更新（2026-09-01）：新增患者级 shadow materialization、患者级 graph/report/selection evaluator 与 `evisoz_execution_plan_v1` schema 后，当前 registry 为 `EVISOZ-SCHEMAS-7f69fcd1620adeca1b47cc56`（registry SHA-256 `6b1a7846a42392ec3a662808880e09d15653a519092efc5ce862ee3d3efb1fff`，45 entries），最新 gate 为 `outputs/evisoz_stage0_gate_v1_20260901_r31/gate.json`；上文 r27/r28/r29/r30 registry/gate 仅作为不可变历史审计记录。r31 由真实 cohort、reference inventory、loader replay 和患者级 Qwen shadow 重新计算，结果仍为 `NO_GO`，并明确保留 5 类阻断：CerebraGloss/ELM 与 fold-local calibration 缺失、私有训练治理授权缺失、3 条报告映射未权威解决、人工去标识及 release 未完成、公开 near/partial overlap 与 TUEV eval identity 审计未闭合。`outputs/evisoz_stage0_private_real_dual_montage_validation_v1_20260901_r1` 的 88-event 全量回放和 61 个定向回归测试通过；这些结果只证明数据/合同可复现，不授权训练或临床部署。

执行计划状态修订（2026-09-01）：`execution_plan.py` 现将 Stage‑0 `QUALIFIED_GO` 视为通过的合同状态（而非缺失的 `GO`），与聚合 gate 的 blocking 规则保持一致；因此真实双 montage 与 Findings/report shadow 不再被错误列入 `missing_or_non_go_checks`。这只是状态语义修正，不改变当前 gate 的 5 个实际阻断项。最新重算产物为 `outputs/evisoz_stage0_gate_v1_20260901_r32/gate.json`（仍为 `NO_GO`）和 `outputs/evisoz_execution_plan_v1_20260901_r4/plan.json`（`STAGE0_NO_GO`）；schema registry 因实现字节变化同步重算为 `EVISOZ-SCHEMAS-2d6d958b2627780979b00d80`，registry SHA-256 `4bcfe901780c596881fde483188a00d0034921206cbafbd8944e34f15de9680f`。该修订不授权任何训练、Qwen 生成或非零 residual。

离线教师合同修订（2026-09-01）：新增 `src/evisoz/forge/teacher_candidates.py` 以及 `evisoz_teacher_candidate_cache_v1`/`evisoz_teacher_candidate_materialization_v1`。CerebraGloss 和 ELM 的结果现在必须以 development-only、uncalibrated、candidate-only cache 导入；每条候选绑定事件 identity、双 montage receipt、教师模型 ArtifactRef 和输入视图。CerebraGloss 允许 TCP22/CAR19 局部候选，ELM 的 crop 语义候选不得声明通道坐标；所有教师候选均禁止创建 clinical label、测量事实、node localization supervision 或 edge endpoint 展开。`scripts/materialize_evisoz_teacher_candidates_v1.py` 只导入已产生的外部候选，不运行教师模型、不读取 EEG/医生报告，也不解除校准和 Stage‑0 训练门。当前 CerebraGloss 已有 2 个 development 事件、29 条候选并纳入最新 gate；ELM artifact 与 fold-local calibration 仍缺失。

最新 gate/plan 物化为 `outputs/evisoz_stage0_gate_v1_20260901_r35/gate.json` 和 `outputs/evisoz_execution_plan_v1_20260901_r7/plan.json`；registry 已同步为 `EVISOZ-SCHEMAS-b4b173a7273f70000e669c81`（SHA-256 `596c981c6ad8851ffe314d38fd1f7729b29c37b42b0901fa79cc3728f8249cce`）。

Stage‑0 的可复现实验入口为 `scripts/materialize_evisoz_execution_plan_v1.py`。它读取已验证的 gate 和 pipeline config，生成 `evisoz_execution_plan_v1`，列出真实数据闭合状态、A–J 消融、报告语义对照、后续授权动作和禁止动作；在当前 r35 `NO_GO` 下输出 `STAGE0_NO_GO`、`training_authorized=false`，不会构造模型、优化器或训练 loader。推荐顺序如下：

```text
materialize_evisoz_stage0_gate_v1.py
→ validate_private_evisoz_real_stage0_cohort_v1.py
→ replay_evisoz_bound_evidence_loader_v1.py
→ run_evisoz_shadow_inference_smoke_v1.py
→ materialize_evisoz_qwen_patient_shadow_v1.py
→ materialize_evisoz_execution_plan_v1.py
```

执行计划当前落盘于 `outputs/evisoz_execution_plan_v1_20260901_r7/plan.json`；它是状态/实验清单，不是 Stage‑0 授权票据。只有 gate 重新计算为 `GO` 且各阶段 required checks 满足后，后续 trainer 才可调用 `require_stage0_training_authorized`。

## 1. 设计结论

EviSOZ-LM 采用“一座离线数据工厂 + 一条在线主链”，但在本仓库中必须实现为已有三条研究 lineage 之上的**隔离适配层**，不能直接拼接旧模块：

1. `src/soz` 提供 canonical v29 H/D、Standard19-CAR LaBraM 定位参考、患者级 positive-set 学习和 typed evidence 基础设施；
2. `code/soz_pre` 与 `code/models/manifest_dataset.py` 提供 TCP22 有符号双极衍生、边顺序和输入/标签 mask 的历史实现，EviSOZ 只复用经新 channel-registry 校验后的信号构造逻辑；
3. `src/clinical_eeg_long_recording` 提供 event Findings、跨事件 claim graph、事实锁定 renderer 和确定性 fallback；
4. EviSOZ 新层只负责训练样本 envelope、TCP22 variable-support/montage-aware token 载波、临床证据解码、相对 canonical v29 `z0_node` 的零初始残差定位和三条 lineage 之间的无损投影；
5. 连续检测器不进入本任务；医生标签、教师输出、知识库和生成文本必须分别经过权限门，不能共用一个无类型字段。

推荐的主链如下：

```text
离线 SOZ-Forge
patient/split ledger
  -> canonical signal + channel registry
  -> frozen v29 reference cache + signed TCP22 waveform sibling caches
     (Stage 1 encoder 通过后才产生 edge-token caches)
  -> local-OOF and frozen-external teacher/programmatic candidate caches
  -> typed training envelope + field-level permissions
  -> validated canonical claim graph
  -> knowledge/eeg deterministic selector + frozen selection-wrapper receipt
  -> deterministic canonical report
  -> optional constrained Qwen text pairs using the same frozen receipt

在线 EviSOZ-LM
known seizure segment [-12,+48) s
  -> acquisition montage/reference/observability receipt
  -> one common-reference electrode-measurement master when recoverable
       |-> Standard19 CAR view -> frozen canonical v29 H/D -> z0_node over C18
       `-> signed TCP22 edge view + edge observability mask
  -> mask-aware, montage-tokenized TCP22 evidence carrier
       |-> source-isolated [-2,+8) onset edge carrier
       `-> longer context edge carrier for morphology/evolution/spread
  -> montage/status tokens + [view,unit,time,200] typed carriers
  -> clinical motif/features + sparse temporal/spatial adapter
  -> six masked evidence queries
  -> typed event slots [E,6,Dq]
       |-> per-event Findings/report evidence
       `-> mode-preserving patient evidence aggregation [P,Mmax,6,Dq] + mode_mask
             -> unique-qualified-mode selector / conflict hard abstention
             -> typed edge evidence [P,22] + qualified edge-to-node residual over z0_node [P,19]
                (PZ/noncandidate/unobserved entries masked; no unconditional endpoint expansion)
  -> validated multievent claim graph
  -> knowledge/eeg deterministic selector + frozen wrapper receipt
  -> deterministic report / optional constrained Qwen lexicalization
  -> physician-review-required draft
```

## 2. 必须先修正的口径

### 2.1 正式定位参考采用 canonical v29 H/D

本方案不再把较新的 common17 event-set 当作正式 `z0_node`。根据公开与私有回放的综合表现，冻结定位参考采用 canonical v29 H/D：

| 模型/队列 | Exact | N4 relaxed | Hit@3 | 角色 |
|---|---:|---:|---:|---|
| canonical v29，public patient OOF，n=102 | 54/102 = 52.94% | 78/102 = 76.47% | 79/102 = 77.45% | 正式 `z0_node` 的公开 OOF 参考 |
| canonical v29，private target-blind 后开标，51 events/23 patients | 25/51 = 49.02% | 38/51 = 74.51% | 29/51 = 56.86% | post-open 跨域历史 guardrail |
| common17 verified，public patient OOF，n=102 | 60/102 = 58.82% | 79/102 = 77.45% | 83/102 = 81.37% | 历史公开重放对照 |
| common17，private strict replay，51 records/23 patients | 14/51 = 27.45% | 24/51 = 47.06% | 30/51 = 58.82% | 历史退化对照 |

canonical v29 的冻结协议和主工件为：

```text
research/02_method/labram_portable_equal_ensemble_protocol_v29_20260815_zh.md
outputs/labram_portable_equal_ensemble_public_oof_v29_20260815/manifest.json
outputs/labram_portable_equal_private_target_blind_v29_20260815/
outputs/labram_portable_equal_private_evaluation_v29_20260815/result.json
```

这里的 private 结果已经开标，只能用于说明为何选择 v29 作为开发基线和历史 guardrail，不能再充当 fresh external confirmation，也不能用于继续调 H/D 权重。论文、配置和验收均须同时写明 exact、N4、Hit@3、分母和 roster；不得把 N4 写成严格准确率。common17 和历史 v88 仍进入附录敏感性比较，但均不再进入 EviSOZ 在线主链。

canonical v29 **不支持 TCP22 输入**；保留它的含义是保留一个 Standard19-CAR node-space reference，而不是强迫 TCP22 经过 v29。仓库中 SOZPreNet/EEGNet 的 TCP22 edge 模型可作为独立 comparator，但现有 public 工件在 dev 上选 epoch，private LOPO 工件也使用 held-out patient 选 best epoch，且部分 profile 额外读取真实 SPHL/SPHR；因此这些数值只能标为 historical development，不替换 `z0_node`，也不能与 v29 的 C18 exact/N4 指标直接排序。若论文需要 TCP22-only localization comparator，应在相同 TCP22-only、inner-val/outer-test 协议下重新训练并单独报告 edge-space 指标。

### 2.2 canonical v29 的精确输入与 TCP22 证据视图

canonical v29 的精确基线信号合同是：

```text
19 个 standard19 物理电极作为内部载波
standard19 monopolar CAR
200 Hz
[-12,+48) s
[19,12000]
PZ 参与内部上下文，但在冻结 C18 候选空间中不可评价
```

其张量链为：

```text
[19,12000]
 -> 15 x [19,4,200]
 -> frozen LaBraM blocks 0--9
 -> 15 x [CLS + 19x4, 200] = [15,77,200]
 -> H carrier + five-component direct-token carrier
 -> fold-local H/D reasoners
 -> candidate-masked pH/pD
 -> p0_node = 0.5*pH + 0.5*pD
```

因此 V1 必须保留 `X_v29=[-12,+48)`、原 preprocessing、H/D reasoners 和固定概率融合，任何 `X_core` 都不能替换 baseline 输入。也不能从已经双向 contextualized 的 v29 60 秒 token 序列事后切 `[-2,+8)` 作为 onset-positive residual carrier。formal residual 改由下述同源 TCP22 分支提供，并在原始 edge waveform 上独立物化 `X_tcp_onset_iso=[-2,+8)`；它不改变 canonical v29 的 `p0_node`。若 TCP22 adapter 无法在任何跨时 attention、归一化或图混合前完成源端隔离，V1 residual 必须 `NO_GO/alpha=0`。

这里不要求后续证据模块也只读 CAR19。对同一记录另外物化有符号 TCP22：

```text
V_ref[e,t]                           # 同一可验证共同参考下的电极测量，不是绝对电位
X_v29[n,t] = CAR19(V_ref)[n,t]       # 冻结 v29 路径，严禁因 TCP 分支改预处理
X_tcp22[k,t] = V_ref[a_k,t] - V_ref[b_k,t]
edge_observed[k] = observed[a_k] & observed[b_k]
```

CAR 均值严格只按 v29 冻结协议中的 19 个电极计算，不能为了 TCP22 把 A1/A2 加入平均参考。TCP 边优先从 CAR 前、共享同一可验证参考的 `V_ref` 相减；对同属 Standard19 的两个端点，先 CAR 再相减在代数上等价，但 A1/A2 边仍要求原始耳电极信号和参考 provenance。若只保留了 CAR19，最多只能 exact 派生不涉及 A1/A2 的 20 条边。

TCP22 固定采用 `code/soz_pre/constants.py` 的有向顺序：

```text
FP1-F7, F7-T3, T3-T5, T5-O1,
FP2-F8, F8-T4, T4-T6, T6-O2,
A1-T3, T3-C3, C3-CZ, CZ-C4, C4-T4, T4-A2,
FP1-F3, F3-C3, C3-P3, P3-O1,
FP2-F4, F4-C4, C4-P4, P4-O2
```

每条边必须保存原始端点、规范端点、`positive_minus_negative` 公式和相对仓库其他 TCP 排列的 permutation receipt。不能只按集合相等认定顺序一致，也不能在预处理时对波形取绝对值或任意翻转极性。若源文件把一条导联写成反向顺序，必须对波形乘 `-1` 后再进入 canonical TCP22，并保存 `orientation_flip=true`；否则相位反转监督无意义。

两路的职责严格分开：

| 路径 | 输入 | 主要作用 | 禁止 |
|---|---|---|---|
| `v29_reference` | 完整、直接可观察的 Standard19-CAR `[-12,+48)` | 冻结 C18 node ranking、`z0_node`、保护损失与 non-inferiority 参考 | 用 TCP22 或插值输入冒充 exact replay |
| `tcp22_evidence` | 同时轴有符号 bipolar edges；core 与 context 分开物化 | 相位反转候选、局部场、形态、演变、募集和传播 | 把单条阳性边拆成两个阳性电极，或把相位反转直接写成皮层 SOZ |

TCP22 的 onset-positive carrier 必须从原始 edge waveform 源端裁出 `[-2,+8)`；较长 context 可以覆盖 `[-12,+48)`。二者可复用同一 patch encoder 权重，但不得先让全窗双向 attention 看完 60 秒后再裁 core。

### 2.3 canonical v29 不是一个单 checkpoint 或单 logit head

canonical v29 包含五折 H reasoner、五折 D reasoner和冻结的等概率融合。公共 OOF 评价时，每名患者只能使用其 held-out fold 的 `pH/pD`；新患者部署时才按冻结协议平均五个 fold-specific `p_equal`。EviSOZ 的 `FrozenCanonicalV29Ensemble` 必须封装这一差异，不能在 OOF 训练中使用看过该患者的其他 fold。

由于 v29 在概率空间融合，先在 native C18 上定义：

\[
p^{(0)}_{C18}=\tfrac12p_H+\tfrac12p_D,\qquad
z^{(0)}_{C18,c}=
\begin{cases}
\log p^{(0)}_{C18,c}, & p^{(0)}_{C18,c}>0,\\
-\infty, & p^{(0)}_{C18,c}=0,
\end{cases}
\quad c\in C18.
\]

随后只通过冻结 registry 把 C18 展开为 typed `z0_node[P,19]`，PZ 和其他不具备候选资格的单元使用候选 mask，不给有限 logit。`p0_C18` artifact 是 baseline 概率真值；`alpha=0` 的硬旁路直接返回该 artifact，不经过 log→softmax 往返。对 `z0_node` 有效 C18 项的 masked softmax 只作冻结容差内的诊断重算，不能用 `log(p+epsilon)` 后仍宣称逐位 exact。任何 mask/montage 扩展若改变 `p0_C18`、rank 或 hard-bypass prediction hash，不能再称 canonical v29 exact route。

### 2.4 四种输入可用性 profile

双视图不是要求每条记录都伪造两套完整 montage。路由固定如下：

本节“完整 Standard19/exact”专指：19 个电极均为 direct electrode measurements，名称在 suffix/重复导消歧后唯一，PZ 实测存在，所有通道共同参考兼容，且未使用插值、零填、bipolar inverse 或 GT 辅助映射；还必须满足 v29 原冻结 preprocessing receipt。任一条件不成立都降级为 transport/shadow。

| profile | 实际可观察输入 | v29 reference | TCP22 evidence | 处理 |
|---|---|---|---|---|
| `dual_native` | exact Standard19，且 A1/A2 与其共享可验证参考 | exact | 22/22 exact-derived | 主分析 profile |
| `standard19_native` | exact Standard19，无 A1/A2 | exact | 20/22；`A1-T3`、`T4-A2` 硬 mask | 保留 v29，TCP 证据按实际覆盖报告 |
| `tcp22_native` | 原生 TCP22，无法恢复完整物理电极记录 | unavailable | 22/22 native | 只走 evidence/shadow 定位；不得声称 v29 parity |
| `partial_or_mixed` | 缺导、混合参考或部分边 | 仅 transport/shadow，或 unavailable | 逐边 mask | 不与 exact v29 主结果合并 |

原生 TCP22 即使图上可恢复一组相对电位，也缺少 FZ/PZ，且逆变换依赖 gauge、边一致性和参考假设；因此不能据此伪造 canonical v29 所需的完整直接记录 CAR19。反过来，Standard19 通常不含 A1/A2，所以也不能凭空补全两条耳电极边。

缺失值默认使用显式 mask 和训练期 channel/edge dropout。球面样条插值只允许作为单独的 `interpolated_transport_shadow`：

- 保存 `observed_mask` 与 `model_input_mask`，插值后前者仍为 `false`；
- 不进入 canonical v29 exact route、直接 measurement、相位反转 claim 或临床 gold；
- 不插值 A1/A2、SP1/SP2 等非标准头皮位置；
- 与 mask-only arm 分开评价并报告 calibration/transport drift；
- 任何使用插值输入的结果不得与 2.1 节 exact v29 数字合并。

### 2.5 发作锚点只是任务条件，不是模型证据

`t0` 可以来自数据集给定的发作起点或人工选定片段，因为本任务明确是 seizure-segment-conditioned；但模型输入只得到裁剪后的相对时间坐标和 `anchor_quality`，不能读取 annotation 文本、医生通道、发作类型或其他标签。评价必须始终标记 `oracle_onset_conditional=true`，也不能把这条链描述成连续 EEG 检测到报告的端到端系统。

## 3. 数据合同：新增 envelope，不复制现有证据本体

不新增一个比现有 Findings/claim graph 更宽松的平行 Evidence JSON。新增对象应命名为 `evisoz_training_example_v1`，只作为训练/数据工厂 envelope，并通过内容寻址引用现有对象。所有 `ArtifactRef` 必须包含 `{schema_version, artifact_id, sha256}`；只给文件名、ID 或自报布尔值均不算闭合引用：

```json
{
  "schema_version": "evisoz_training_example_v1",
  "example_id": "P001-SZ02",
  "dataset_id": "deepsoz_public_v1",
  "patient_binding": {
    "dataset_namespace": "tuh",
    "local_pseudonym": "P001",
    "global_linkage_group_ref": {
      "schema_version": "evisoz_patient_linkage_group_v1",
      "artifact_id": "PLG-...",
      "sha256": "..."
    }
  },
  "split_binding": {
    "official_source_role": "source_train",
    "evisoz_model_role": "train",
    "outer_fold": 0,
    "locked_test": false,
    "roster_ref": {
      "schema_version": "evisoz_split_roster_v1",
      "artifact_id": "SPLIT-...",
      "sha256": "..."
    }
  },
  "signal_binding": {
    "canonical_signal_ref": {
      "schema_version": "canonical_eeg_signal_v1",
      "artifact_id": "SIG-...",
      "sha256": "..."
    },
    "channel_registry_ref": {
      "schema_version": "evisoz_channel_registry_v1",
      "artifact_id": "CHAN-...",
      "sha256": "..."
    },
    "montage_derivation_receipt_ref": {
      "schema_version": "evisoz_montage_derivation_receipt_v1",
      "artifact_id": "MONT-...",
      "sha256": "..."
    },
    "v29_car19_view_ref": {
      "schema_version": "canonical_eeg_signal_v1",
      "artifact_id": "SIG-CAR19-...",
      "sha256": "..."
    },
    "tcp22_signed_view_ref": {
      "schema_version": "canonical_eeg_signal_v1",
      "artifact_id": "SIG-TCP22-...",
      "sha256": "..."
    },
    "event_findings_ref": {
      "schema_version": "clinical_eeg_event_findings_v3",
      "artifact_id": "FIND-...",
      "sha256": "..."
    }
  },
  "supervision": {
    "release_ref": {
      "schema_version": "evisoz_field_release_v1",
      "artifact_id": "REL-...",
      "sha256": "..."
    },
    "field_labels": []
  },
  "teacher_candidate_refs": [],
  "derived_candidate_refs": [],
  "permissions": {
    "quality_tier": "silver",
    "report_scope": "soz_localization",
    "claim_allowlist": [],
    "loss_mask": {}
  },
  "text_targets": {
    "reference_claim_graph_ref": {
      "schema_version": "evisoz_reference_claim_graph_v1",
      "artifact_id": "RCG-...",
      "sha256": "..."
    },
    "canonical_report_ref": {
      "schema_version": "evisoz_canonical_report_v1",
      "artifact_id": "RPT-...",
      "sha256": "..."
    },
    "generated_variant_refs": []
  },
  "provenance": {
    "preprocess_receipt_refs": [],
    "producer_receipt_refs": [],
    "freeze_audit_ref": {
      "schema_version": "evisoz_holdout_freeze_audit_v1",
      "artifact_id": "FREEZE-...",
      "sha256": "..."
    }
  }
}
```

上例展示 `dual_native`。其他 profile 不能用伪 ArtifactRef 填满：每个 view slot 必须是有效内容寻址引用，或结构化的 `{state: unavailable, reason_code: ...}`；缺字段、空字符串和全零张量都不能代表 unavailable。`montage_derivation_receipt_ref` 负责证明两个 view 的共同父 signal、sample clock、滤波/单位、端点公式、orientation/permutation 和各自 observability。

`ArtifactRef.sha256` 的计算域也必须冻结，不能由各 producer 自选。结构化 JSON/YAML artifact 先投影为 schema 规定的 JSON payload，移除递归的顶层 integrity 字段，再按 RFC 8785 JCS 序列化为 UTF-8 bytes；二进制 artifact 则散列未经改写的原始 bytes。artifact 自身的 integrity receipt 保存 `hash_domain=canonical_json_without_integrity_fields | raw_bytes`、`byte_length` 和 SHA-256；validator 必须按该域重算。路径、mtime、展示缩进和压缩容器名不进入 canonical-JSON hash，除非 schema 明确把它们定义为 payload。

`patient_binding` 使用 dataset namespace 防止同名 ID 碰撞；跨 DeepSOZ/TUSZ/TUEV 或两套私有数据的同人关系只能来自去标识化 linkage-group receipt。official/source split、EviSOZ 训练角色、outer fold 和 locked-test 身份分别保存。所有派生窗、teacher cache、claim graph 和文本只引用父对象的 split binding，不能自行声明另一个 split。

`official_source_role`、`evisoz_model_role`、`outer_fold` 和 `locked_test` 是 `roster_ref` 的缓存投影，不是四个可独立编辑的真值。validator 必须从内容寻址 roster 重算并要求逐字段相等；缺 roster、成员不存在或任一投影不一致时 fail closed。

### 3.1 五类事实必须物理分栏

| 分栏 | 含义 | 是否能作为硬监督 | 是否能进入报告事实 |
|---|---|---:|---:|
| `clinical_labels` / `supervision.field_labels` | 获授权的原始数据集或医生标签 | 仅该字段的 release/粒度允许时 | 仅相应 scope |
| `direct_measurements` | `signal_binding.event_findings_ref` 中可重放的 measurement/evidence ID | 可训练数值/状态头 | 通过术语资格后 |
| `teacher_candidate_refs` | CerebraGloss/ELM 等模型预测 artifact | 仅 soft target | 只能写 candidate/possible |
| `derived_candidate_refs` | 确定性规则或信号算法推导 artifact | 仅 soft/auxiliary | 只能写 candidate/possible |
| `generated_text` | `text_targets.generated_variant_refs` 指向的 claim graph 语言序列化 | 否 | 不是新事实来源 |

教师预测不能命名为 `observed_evidence` 或 `measured`。文本也不能通过回解析反向变成新标签。

envelope 不复制 direct measurement 的值；它引用已验证 Findings artifact，再以 evidence ID + row SHA-256 选择训练对象。这样既保留“五类分栏”，又不建立第二套 measurement 本体。

### 3.2 每个字段都需要认识状态和权限

样本级 A/B/C/D 与 `report_scope` 只用于初筛，不能代替字段级合同。每个标签或 evidence row 至少包含：

```text
state: present | absent_with_opportunity | uncertain |
       not_provided | not_assessable
assertion_level: direct_measurement | dataset_label |
                 model_candidate | derived_candidate
source/producer receipt
temporal support + coordinate system
spatial support + node/edge semantics
confidence semantics + calibration receipt or uncalibrated
positive-set exhaustiveness: exhaustive | incomplete_positive | unknown
training permissions
report permissions
evidence/claim IDs
```

`unknown`、`not_provided`、`not_assessable` 和“在完整机会下未见”必须分开，不能都塞进 `unknown_fields` 字符串数组。

与现有 fact ledger 的无损 crosswalk 固定为：

| 现有状态 | 唯一 EviSOZ 映射 | 附加条件 |
|---|---|---|
| `present` | `present` | 保留 `source_state=present`、原 assertion/provenance |
| `absent` | `absent_with_opportunity` | 仅当内容寻址 opportunity receipt 证明目标时空范围完整可评 |
| `absent` | `not_assessable` | 缺少上述证明时唯一映射为 `reason_code=opportunity_unverified`，保留 `source_state=absent` |
| `not_recorded` | `not_provided` | 保存 `source_state=not_recorded, reason_code=source_not_recorded`；不表示信号中不存在 |
| `not_assessable` | `not_assessable` | 保留原 `source_state` 与 reason code |
| `uncertain` | `uncertain` | 仅表示源事实自身模糊/冲突；保留原 provenance，不得提升 certainty |

`left_censored/right_censored/interval_censored` 是独立 temporal-support 字段，不是认识状态。现有 `absent` 若缺少完整机会证明，迁移时必须按上表唯一 fail closed 为 `not_assessable(reason_code=opportunity_unverified)`，不能任选 `uncertain`，也不能生成“未见”事实。

训练时的 `reference_claim_graph` 与推理产生的 `predicted_claim_graph` 必须使用不同类型和哈希域。前者只进入 loss/文本 target builder，绝不能作为模型 forward、患者聚合或部署 renderer 的输入；后者只能引用模型实际输出和 EEG-derived evidence。这样可防止“用 gold 生成报告，再把报告当模型能力”的闭环泄漏。

### 3.3 私有标签必须先取得新的 release

仓库现有 private doctor-label bundle 明确是 `evaluation_only=true`、`eligible_for_llm=false`。在新的人工审核、伦理和数据治理 release 生效前：

- 私有 significant electrode、spread、laterality 和医生文本不能用于训练、prompt 调整、RAG 查询或合成报告；
- 可以保留为冻结后 evaluator 的只读 reference；
- `full_soz` 训练样本数在此之前为零，而不是默认沿用表格内容。

当前实现遵守该停止门：88 个真实事件虽然已经落盘字段值和 split-bound envelope，但 `enabled_loss_ports=[]`，显著电极、扩散、侧别与区域仅作为 evaluator-side reference；自动去标识报告候选也全部保持 `manual_review_status=pending`。这里的 `evisoz_field_release_v1` 是字段状态/来源/权限的机器可读容器，不是伦理授权本身。未来若开放训练，必须额外输入独立、内容寻址、明确列出可用字段/患者角色/用途/到期条件的治理 authorization receipt；不得修改现有 materializer 常量或把用户运行一次脚本解释为授权。

真实医生报告的独立 release 合同现已落在 `evisoz_private_physician_report_release_v1`（`src/evisoz/data/private_physician_report_release.py`）。它只引用已去标识的候选文本和外部人工审核/治理 receipt，不把 DOCX 正文写入 Evidence envelope；开发文本仅可进入 Qwen 文本侧训练，锁定文本仅可进入语言评价，任何 physician-authored text 均禁止直接监督 SOZ 定位。当前没有可验证的人工 release 输入，因此该 lane 仍为 `not_released`，不会改变 Stage‑0 `NO_GO`。

此外，significant electrode 应保存为 `incomplete_positive`；集合外通道不自动成为医学阴性。spread 是独立 soft-positive 字段，不能并入 onset positive set。

新的 release 也不能是一次性全库授权。至少拆成 patient-roster-bound 的 `train/dev` release 与 `locked-test evaluator-only` release，并对每个字段分别声明 `allowed_roles`、`loss_allowed`、`report_target_allowed`、`prompt_or_RAG_allowed` 和 roster SHA-256。validator 在 release artifact 缺失、角色不匹配或测试 roster 被请求用于训练时必须 fail closed。

### 3.4 Stage-0 bound evidence：统一数据结构的实际落点

`evisoz_training_example_v1` 是单事件的基础 envelope；在它与 Findings/claim/report 产物真正交给后续 builder 之前，再经过一个只读的绑定层 `evisoz_bound_evidence_example_v1`。该层不是新的事实本体，而是把同一事件的可信引用和权限状态放在一个可重放边界内：

```text
private_real_stage0_examples/events/<event>/training_example.json
  + events/<event>/field_release.json
  + real_dual_montage/events/<event>/dual_montage/sidecars/{event_identity,montage_receipt}.json
  + findings_claim_reports/events/<event>/{findings,reference_claim_graph}.json
  + findings_claim_reports/patients/<linkage_group>/{signal_candidate_claim_graph,knowledge_selection,canonical_report}.json
  -> bound_evidence/events/<event>/bound_evidence.json
```

每个 bound example 的 `source_refs` 固定包含：`training_example`、`event_findings`、`reference_claim_graph`、`field_release`、`montage_derivation`、`dual_montage_cache`，以及可选的患者级 `patient_signal_graph`、`knowledge_selection`、`canonical_report` 和独立的 `physician_report_release`。当且仅当外部治理授权、逐行人工审核和去标识候选均通过时，`physician_report_release` 才能绑定到匹配的 linkage group/role；bound 层只保存 release manifest 的内容寻址引用，绝不复制报告正文。其 lane 状态为 `released_for_qwen_text_training` 或 `released_for_language_evaluation`，但全局 `training_allowed` 和 `report_text_loss_allowed` 仍由 Stage‑0 aggregate gate 控制。物化时会重新加载并验证 event identity、patient-level split roster、montage receipt 和 field release，而不是只相信 envelope 内缓存的 hash。`lanes` 明确区分 `clinical_labels`（evaluator-only）、`direct_measurements`、`teacher_candidates`、`derived_candidates`、`physician_authored_text` 和 `generated_text`；当前 r50 真实物化中没有 physician release，88 个事件的该 lane 仍为 `not_released`，CerebraGloss/ELM 未授权候选仍不能进入训练或定位监督。

物化器与语义验证器分别位于 `src/evisoz/forge/evidence_binding.py` 和 `scripts/materialize_evisoz_bound_evidence_v1.py`，真实 loader 位于 `src/evisoz/data/bound_evidence_loader.py`，schema 由 `evisoz_schema_registry_v1` 绑定。当前真实结果为 `outputs/evisoz_stage0_bound_evidence_v1_20260901_r50`，88 条事件（development 65、locked-test 23）；loader replay 的最新定向 receipt 为 `outputs/evisoz_stage0_bound_evidence_loader_replay_v1_20260901_r51/receipt.json`，此前完整 88-event replay 仍由 r49 receipt 保留。`src/evisoz/reporting/bound_shadow_report.py` 只从 loader 已验证的 `knowledge_selection` receipt 取得 card ID，再交给 candidate-only report plan；它不直接打开 `knowledge/eeg`，也不允许调用方注入未绑定卡片。后续训练 envelope、结构化报告 builder 和 evaluator 只能消费 loader 返回的已验证记录，不能直接从 `knowledge/eeg`、医生 DOCX 或未校准 candidate cache 读取患者事实。loader replay 仍是非授权的 shadow replay，不改变 Stage‑0 的总体 `NO_GO`。

## 4. 通道与 montage 合同

必须新增统一 channel registry，不能直接选用仓库中互相冲突的旧映射。每个输入单元保留：

```yaml
original_name: T3
normalized_name: T7
unit_type: physical_electrode
source_signal_index: 7
label_source: edf_signal_header
position_id: T7_10_20
position_source_ref:
  schema_version: evisoz_position_registry_v1
  artifact_id: POS-...
  sha256: ...
observability_receipt_ref:
  schema_version: evisoz_observability_receipt_v1
  artifact_id: OBS-...
  sha256: ...
recorded_or_observable: true
measurement_status: observed
signal_view_id: v29_car19
sample_clock_ref: CLK-...
```

EDF header 只证明名称来自哪里，不能独自证明空间位置或可观察性。`recorded_or_observable` 只由绑定 signal index、montage/reference 和重复导消歧后的 observability receipt 推导；空间位置只由内容寻址的 position registry 推导。二者都遵守 3 节的 hash-domain 合同。

窗口内信号质量不得塞回 `available` 或 `recorded_or_observable`。采样支持、finite 检查和伪迹/QC 分别产生 `node_cell_qc_mask[E,19,60]` 与 `edge_cell_qc_mask[E,22,60]`；因此“记录中存在但本窗受伪迹影响”与“物理缺导/不可由该 montage 观察”在 schema、模型 mask 和报告分母中始终不同。`node_cell_qc_mask` 绝不修改、遮挡或重归一化 frozen v29 waveform/H/D；它只进入可评价性、技术限制和 abstention ledger，否则就破坏 exact reference。

双极导联保存为边：

```yaml
original_name: F7-T3
normalized_name: F7-T7
unit_type: bipolar_derivation
original_positive_electrode: F7
original_negative_electrode: T3
positive_electrode: F7
negative_electrode: T7
derivation_formula: positive_minus_negative
source_signal_index: 3
orientation_flip: false
signal_view_id: tcp22_signed
sample_clock_ref: CLK-...
observability_receipt_ref:
  schema_version: evisoz_observability_receipt_v1
  artifact_id: OBS-...
  sha256: ...
recorded_or_observable: true
measurement_status: observed
```

硬约束如下：

- T3/T4/T5/T6 原名保留，规范名分别为 T7/T8/P7/P8；
- SP1/SP2 保留为 extra electrodes；没有已验证 position ID 时不得映射进 canonical v29 Standard19 或 TCP22；
- A1/A2 只有在来源语义明确时才可映射 M1/M2；
- 缺导默认使用显式 mask；允许的球面插值只产生 `interpolated_transport_shadow`，不得改写 `observed_mask` 或进入 direct evidence；
- derivation 必须保留源文件的端点顺序与差分公式；不从名称猜 anode/cathode 生理极性；
- bipolar lead 不能拆成两个 onset-positive 端点；
- 合成文本只能引用本记录实际存在且该 claim 有权使用的单元。

每个 TCP edge token 不是普通“通道编号”，而是以下 typed montage token 与有符号波形 patch 的组合：

```text
edge_token = waveform_patch
           + positive_electrode_embedding
           + negative_electrode_embedding
           + derivation_orientation_embedding
           + montage_id_embedding
           + observability/status_embedding
```

mask 不能只在输入处把缺失波形乘零；零 token 仍可能通过 attention bias、归一化或 pooling 影响结果。`edge_cell_mask` 必须进入 patch statistics、attention key/query mask、line-graph message passing、normalization/pooling 分母和所有 loss/metric denominator。训练期 edge dropout 使用同一闭合路径，并分别记录 natural-missing 与 augmented-dropout 状态。

相位反转候选只在**两条或以上实际可观察、共享一个物理电极、方向已规范且时间重叠**的边之间计算。使用插值边、未知极性边或仅单条边时，该字段为 `not_assessable`。相位反转只是局部场/共享电极候选证据，不能自动生成 onset-positive electrode 或 SOZ 标签。

V1 formal baseline 仍要求完整、直接记录的 Standard19-CAR，并精确重放 canonical v29。TCP22 evidence 按 2.4 节独立路由；缺失 TCP edge 只减少 edge evidence coverage，不会触发对 v29 输入的修改。只有 `dual_native`/`standard19_native` 可进入“相对 v29 的 residual”主分析；`tcp22_native` 进入 standalone evidence/shadow arm。

## 5. 主模型

### 5.1 精确 v29 reference 与隔离的 TCP22 evidence carrier

baseline cache 与 evidence cache 是同一事件的两个兄弟 artifact，不是前后转换：

```text
X_v29_car19       [E,19,12000]       # 200 Hz, [-12,+48) s
  -> FrozenCanonicalV29Ensemble
  |-> historical public OOF: held-fold pH_C18, pD_C18, p0_C18 [P,18]
  `-> unseen events: pH/pD/pEqual_fold [E,5,18]
                    -> mean(pEqual_fold, fold) = p0_C18 [E,18]
  -> typed Standard19 expansion
     z0_node                          [P,19] or [E,19]  # PZ=-inf/masked
     candidate_mask_node             [19]             # PZ=false

tcp22_parent_ref                       # signed-edge raw-parent artifact + sample clock
X_tcp_context = materialize(tcp22_parent_ref, samples=0:12000)
                  [E,22,12000]       # 同时轴 signed edges
edge_cell_mask    [E,22,60]
  -> LaBraM waveform patch embedding + BipolarDerivationAdapter
  -> H_edge_context                  [E,22,60,D]

X_tcp_onset_iso = materialize_core(
    tcp22_parent_ref, samples=2000:4000,
    preprocessing=core_local_frozen_receipt
)
  -> 独立 source-isolated forward
  -> H_edge_onset                    [E,22,10,D]
```

public route 的单位是 patient，直接重放历史 held-fold OOF；不在历史 public roster 中的患者以 event 为原生单位，且必须先在每折内计算 `pEqual_fold=0.5*pH_fold+0.5*pD_fold`，再对五折求均值。后续 event→patient 聚合是 EviSOZ 另行冻结的下游操作，不得冒充 canonical public patient pooling，也不得用“先分别对 H/D 求折均值再融合”替代原浮点运算顺序后声称 bitwise parity。

LaBraM 的现成物理电极 position embedding 不能直接冒充 bipolar derivation ID。TCP22 分支只复用经 parity smoke 验证的 waveform patch/token encoder，再显式加入正端点、负端点、方向和 montage/status embedding；后续 temporal/spatial adapter 在 edge graph 上运行。若当前 LaBraM 实现无法拆出可靠的 channel-agnostic waveform patch interface，则先使用独立但轻量的共享 1 秒 patch projector，TCP22 分支保持 shadow，不能声称“原生 LaBraM TCP22”。

两个 cache branch 都直接绑定同一 raw-parent signal、事件锚点和 sample clock，但 onset artifact 不得引用、切片或继承 context-filter/normalize/token artifact；两者只能共享 parent raw bytes、明确允许的确定性常量和 encoder 权重。它们各自绑定滤波/单位配置、channel registry 与 preprocessing SHA-256，并使用原子联合发布；不得通过重采样后的近似时间戳事后对齐。`materialize_core` 的滤波、padding 与归一化必须是 core-local，实际 receptive field 全部落在获授权样本范围内。`H_edge_onset` 还绑定 exact sample indices、global cells 10:20、edge mask、orientation receipt 和 visibility receipt。实现需要固定长度时只能使用显式 masked padding，不能把 cells 0:9 或 20:59 当上下文。`Qonset` 及任何可改变定位的 residual 只能读取 `H_edge_onset`；其他 query 可按权限读取 `H_edge_context`。

`p0_C18/z0_node` 仍由 v29 独立产生；Standard19 expansion 只能按冻结 candidate registry 插入 PZ mask，不能重排或平滑 C18 概率。TCP22 edge logits/evidence 默认保持为 derivation/local-field claim。正式 node residual 只允许由 **released node-level supervision** 训练并独立资格化的 typed projection 产生；region/laterality supervision 只能开启各自的 region/laterality head，不能间接监督 node projection。禁止使用“某边阳性，所以两个端点都阳性”的固定展开规则。

### 5.2 Clinical Motif Adapter

第一版不复制 TFM 离散码本。对每个 TCP22 edge patch token 融合：

```text
raw waveform token（仅在 5.1 parity 通过时称 LaBraM patch token）
+ gated lightweight spectral feature
+ gated baseline-relative feature
```

V1 输入合同固定为：

```text
H_edge_context       [E,22,60,D]    one token per signed-edge-second
edge_spectral        [E,22,60,Fs]   computed on the same one-second edge cells
edge_baseline        [E,22,60,Fb]   relative to [-12,0) s cells 0:12
edge_onset_spectral  [E,22,10,Fs]   recomputed from exact isolated core cells only
edge_observed_mask   [E,22]         native or exact-derived edge state
edge_orientation_ok  [E,22]         polarity convention is closed
time_support_mask    [E,60]         recorded temporal support only
edge_cell_qc_mask    [E,22,60]      per-edge/per-second finite and QC usability
edge_cell_mask       [E,22,60]      observed & oriented & time support & cell QC
baseline_assessable  [E,22]         complete, non-overlapping, QC-qualified baseline
```

`edge_cell_mask = edge_observed_mask[:,:,None] & edge_orientation_ok[:,:,None] & time_support_mask[:,None,:] & edge_cell_qc_mask`，并在进入模型前再次与 token finite mask 相交。`Fs`、`Fb`、频带、窗函数、单位、归一化和 baseline robust summary 必须由配置/receipt 冻结。spectral 与 baseline feature 只能由同一 TCP edge signal 和相同秒格生成。`baseline_assessable=false` 时 baseline branch 数值填零且 gate 硬置零，相关 claim 输出 `not_assessable`；不得用全事件均值、其他患者或 onset 后 token 补基线。短片段没有完整 60 秒支持时进入独立 masked profile；只要 v29 自身 60 秒合同不满足，就不能进入 formal residual 主分析。

onset-positive 与 `Qlocalizability` early-positive 分支是硬编码例外：只融合 `H_edge_onset` 与从 exact core 逐 cell 重算、使用冻结训练集归一化常数的 `edge_onset_spectral`；其 full-context `edge_spectral`、`edge_baseline`、baseline summary 和任何 sample-level 全窗统计 gate 全部固定为零且无计算图。形态 query 仍可按自身权限使用 baseline branch，但不得把该支路的输出共享回 onset-positive carrier。

形态输出使用**多标签 sigmoid + evaluability mask**，不能用互斥 softmax：attenuation、LVFA、rhythmic theta、rhythmic delta、alpha/beta rhythm、sharp-like 和 evolving rhythm 可共存。artifact 属于独立 quality head；`uncertain` 是认识状态，不是 morphology 类别。

现有 `src/soz/models/standard19_motif_filter.py` 可复用 sparse TimeFilter/CBraMod 的接口和零初始 residual 思路，但其 Standard19 token 和权重不能直接冒充 TCP22 edge adapter；相同算子迁移到边图后必须重新完成 mask、orientation 和 native-profile qualification。

### 5.3 稀疏时空证据编码

顺序固定为：

```text
per-edge temporal evolution
 -> derivation-line-graph spatial mixing
 -> a small number of learned extra edges
 -> quality/reference/lead-lag gated correction
```

Top-k 表示“物理图之外的额外关系”，不是把物理邻边也竞争掉的总 Top-k。wPLI 可对称；Granger/transfer entropy 保留方向；所有 connectivity bias 零初始化并带逐 cell 的 `edge_cell_mask`。晚期全导同步和 spread 信息只允许描述 course 或降低定位特异性，不能提高早期 onset channel logit。

TCP22 分支使用 derivation line graph：两条 edge token 仅在共享物理电极时具有固定邻接；邻接属性包含共享端点、两条边在该端点的符号、几何距离和是否构成可评价的相位反转对。`phase_reversal_score` 必须保留瞬时有符号信息，不能由纯功率/绝对振幅特征替代。它只进入 `Qmorphology/Qonset` 的候选证据和 grounding，不直接写入 node logit。

mask 必须从读取端一直闭合到特征生产端。供 `Qonset`/residual 以及 `Qlocalizability` early-positive port 使用的 carrier 必须是 5.1 节从 TCP edge signal 源端裁出的 `H_edge_onset`，不能从已经 contextualized 的 `H_edge_context` 事后 gather。该分支在任何跨时 mixing、归一化或 connectivity summary 前就排除 global cells 0:9 和 20:59；core 内 temporal adapter 采用 causal/visibility mask，spatial mixing 也只在同一隔离窗内进行。直接在 TCP context signal 替换任一被禁 cell 时，`H_edge_onset`、`Qonset`、`Delta_z_node`、`z1_node` 和 rank 必须逐位不变，否则仅在 query 末端加 mask 不算满足权限合同。

其余 context query/penalty ranges 各自在进入新增 temporal/spatial adapter 前先取得由权限 mask 裁出的 carrier copy，并禁止全窗 sample statistics 或跨 mask connectivity。任一 future boundary 改动都必须重新披露 patch encoder 与 adapter 的实际 receptive field；若任一底层调用跨越权限边界，必须像 onset 一样物化独立 source-isolated branch，不能只改末端 attention mask。

### 5.4 六查询 Evidence Decoder

六个查询共享 cross-attention block，但拥有不同的时间和信息权限：

| Query | 可读 token | 输出 |
|---|---|---|
| `Qquality` | 全 context | artifact/quality/evaluability |
| `Qmorphology` | baseline + early context | 多标签 morphology candidates |
| `Qonset` | onset-causal early TCP edge prefix | derivation/local-field candidate；node/region/laterality 分别按各自 supervision 与资格头输出 |
| `Qspread` | onset 后、截断的 early course | early-spread set 与 recruitment order |
| `Qevolution` | context | frequency/amplitude/spatial evolution |
| `Qlocalizability` | early positive port + 独立 penalty ports | focal/diffuse/nonlocalizing 与最细安全粒度 |

V1 将六个 query 的正向读取权限编译为不可学习的 `query_positive_mask[E,6,22,60]`：

```text
Qquality          cells 0:60   [-12,+48) s
Qmorphology       cells 0:24   [-12,+12) s
Qonset             cells 10:20  [-2,+8) s
Qspread            cells 12:32  [0,+20) s
Qevolution         cells 12:60  [0,+48) s
Qlocalizability    cells 10:20  early positive focal support only
```

质量、广泛化和晚期信息不伪装成第七个 query，也不复用 `Qlocalizability` 的同一次 attention；它们通过三个独立的只降级 penalty ports 读取：

```text
penalty_port_mask   [E,3,22,60]
Pquality            cells 0:60   -> b_event[:,0] in [0,1]
Pgeneralized        cells 12:60  -> b_event[:,1] in [0,1]
Plate               cells 20:60  -> b_event[:,2] in [0,1]
```

这些是 initial V1 arm 的冻结截止点；改变任一边界都产生新实验 arm。每个 `query_positive_mask` 和 `penalty_port_mask` 必须逐 cell 与 `edge_cell_mask[E,22,60]` 相交；依赖基线的 edge 还要与 `baseline_assessable` 相交。普通 shared cross-attention 不足以保证权限，因此 onset residual 只能读取 `Qonset` 的早期 positive port。三个 penalty port 只产生 monotone abstention/localizability burden；计算图与 `Delta_z_node`、node eligibility、node logits 和 rank 完全断开。于是六个 query 仍只输出 `E_event[E,6,Dq]`，三个 penalty scalars 是有类型的附属量，而不是第七个 dense evidence token。

查询输出必须再经过 typed heads，产生明确的 slot、mask、时间/空间支持和 evidence IDs。六个 dense vectors 本身不是可报告证据。

### 5.5 先聚合事件证据，再修正患者级基线

canonical v29 的两条 route 必须区分：历史 public OOF 直接给出患者级 `z0_node[P,19]`（C18 可评价候选 + PZ masked）；未见患者的五折推理只给出 event-level `z0_event[E,19]`。后者没有与历史 public OOF 等价的 canonical patient pooling，不能为了套用下式而将多次发作临时平均成“冻结患者基线”。在独立的 event→patient baseline aggregation 规则及其 parity/calibration receipt 通过前，未见患者只保留逐 event 基线与 mode-preserving 综合报告，患者级 residual 硬关闭。

对已具备合法患者级 reference 的 arm，新链固定为：

```text
E_event          [Ne,6,Dq]
event_patient_id [Ne]
 -> mode-preserving evidence aggregation
E_patient_mode   [P,Mmax,6,Dq]
mode_mask        [P,Mmax] + per-event/per-mode ledger
 -> unique-qualified-focal-mode selector
E_patient_onset  [P,Dq] + residual_mode_eligible [P,1]
 -> typed TCP22 edge evidence [P,22]
 -> qualified patient-level node residual over authorized z0_node [P,19]
```

`Mmax` 是配置冻结的 padding 上限，真实 mode 数由 `mode_mask` 给出，绝不能在张量化时压掉 mode 轴。逐事件 edge slots 仍独立进入 event Findings/report；只有唯一通过资格的 focal mode 才可形成 `E_patient_onset`。edge evidence 保持 edge 语义；只有使用 released node-level supervision 训练且通过独立 qualification receipt 的 typed edge-to-node projection 才进入 node residual。`z0_node` 保留 canonical v29 的 H/D fold 路由和融合，不能被未经识别的 `q_loc*q_quality*(1-u)` 权重替换。V1 新 evidence 使用透明、封顶、去重的事件聚合，并执行 leave-one-event-out 稳定性检查：

- 只有冻结的 early `edge_cell_mask`/最低可评机会可以在进入 `Qonset` 前硬排除不可用 cell/event；学习到的 `Qquality` 分数及 late quality 不能重加权 residual 的 focal spatial pooling，只进入 abstention/technical-limitation ledger；
- diffuse/nonlocalizing 事件仍保留在 phenotype/mode ledger 中；
- 两个可靠且冲突的 spatial mode 分别保存在 `E_patient_mode` 并分别报告，不能平均成单一向量或中线高置信分布；
- V1 若存在多于一个互斥的可靠 focal mode，或没有唯一合格 focal mode，则 `residual_mode_eligible=false`，对应患者在计算 `R/g` 前硬返回 `z0_node`；冲突只影响 localizability/报告，不允许模型任选一个 mode；
- 若真实 mode 数 `M_p>Mmax`，不得截断、合并或按置信度挑前 `Mmax` 个；完整 mode 留在 variable-length ledger，设置 `mode_overflow=true`、`mode_mask[p,:]=false`、`residual_mode_eligible=false` 和 hard abstention。实现可拒绝该 patient bag，但不能把不完整 tensor 当完整聚合；
- 未经患者隔离校准的值只能叫 score/weight，不能叫 probability 或 uncertainty probability。

互斥 mode 的 laterality/region/onset-support 判据、可靠性门和 `Mmax` 必须在 outer evaluation 前冻结。学习式 mode-aware MIL 在 event-to-mode 与 onset-field gold 足够后才有资格进入 fresh-development 晋升评估。

### 5.6 精确保护基线的患者级残差

定义：

```text
z0_node                    [P,19]  frozen fold-specific v29 logits; PZ masked
r_raw_node = R(E_patient_onset) [P,19]  qualified edge-to-node path only
Delta_z_node = softplus(r_raw_node) [P,19]  nonnegative early-evidence boost
node_eligibility_mask      [P,19]  bool; unavailable/noncandidate entries are zero
g_positive                 [P,1]   sigmoid, in [0,1]
residual_mode_eligible     [P,1]   bool; exactly one qualified focal mode
b_event                    [Ne,3]  event quality/generalized/late burdens in [0,1]
penalty_valid              [Ne,3]  evaluability/permission mask
b_patient=(b_q,b_g,b_l)    [P,3]   masked event maximum per patient
u0                         [P,1]   early-positive abstention score in [0,1]
a_abstain = 1-exp(-a_q*b_q-a_g*b_g-a_l*b_l), all a >= 0
u1 = 1-(1-u0)*(1-a_abstain)
alpha                      scalar in [0,alpha_max], fold/inner-dev bound
```

新模块产生：

\[
z_{1,node}=z_{0,node}+\alpha\,m_{mode}\,g_{positive}\,
\bigl(\Delta z_{node}\odot m_{node}\bigr).
\]

对 patient `p` 与 burden coordinate `j`，冻结聚合为

\[
b_{patient,pj}=\max_{e:\,patient(e)=p,\;valid_{ej}=1}b_{event,ej}.
\]

该 masked maximum 对每个 event burden 单调非减；不得用可正可负 attention 混合。三个 coordinate 在 V1 都是精细定位的必需 evaluability ports：任一 coordinate 无有效 event 时标记 `burden_assessable=false`，不实例化缺失的 `b_patient`、不填零，并直接令 `u1=1`。patient bag 全空、没有唯一合格 focal mode（零个或冲突多个）或 mode overflow 时同样令 `u1=1` 并拒绝精细定位。只有三坐标均可评且恰有一个合格 focal mode 的病例使用上式 `u1`，因此对 `b_q/b_g/b_l` 与所有非负 `a` 都有 \(\partial u_1/\partial b_j\geq0\)。`u0/u1` 未通过患者隔离校准前只能称 abstention score，不能称概率。

`Delta_z_node` 和 `g_positive` 的计算图都只接 `Qonset` 早期 TCP edge positive port。`m_node` 首先与 v29 C18 candidate mask、Standard19 observability 和字段权限相交；它不能由“edge 的任一端点”规则生成。无论单条 edge、共享端点多边场还是方向闭合的相位反转候选，都只提供 derivation/local-field evidence，不能单独训练或开启 node residual。相应 node residual 必须另有 released node-level supervision、patient-disjoint fitting 和 node-specific qualification receipt；region supervision 仅进入 region head。非负参数化保证 residual 只能提升有早期资格的候选，不能用负 residual 间接抬高另一个通道。质量、广泛化和晚期 ports 与整个 localization-logit 图断开，只通过上述完整链单调提高 abstention/降低最细可报告粒度；它们既不缩放 residual，也不改变 `m_node`，所以单独扰动 penalty burden 时 `z1_node` 和 node rank 必须逐位不变。该约束牺牲了 V1 的负向 logit 修正能力，换取可证明的非增益安全性；更灵活的 monotone projection 只能作为后续独立 arm。所有输入、`r_raw_node`、`Delta_z_node`、burden、gate 和 abstention 在组合前必须通过 finite/range 校验。

实现必须有真正的硬旁路：当 exact v29 reference 可用而 `residual_enabled=false`、`alpha==0` 或某患者 `residual_mode_eligible=false` 时，在计算该路径的 `R/g` 之前直接返回对应 `z0_node`；当 v29 不可用时则返回 typed `baseline_unavailable` 并转入 TCP22 standalone evidence/shadow route，不能制造零 logits。这避免 `0*NaN` 破坏 identity，并要求可用病例上 `torch.equal(z1_node,z0_node)`、逐患者 logit hash 和 prediction receipt 全部一致。`alpha_max`、gate 粒度与 inner-dev 选择规则在 fold 训练前冻结。`softplus` 分支不要求自身在参数初始化时数值为零；正式零初始语义由先行硬旁路的 `alpha=0` 精确实现。

训练顺序：

1. evidence pretraining 阶段启用硬旁路，正式输出恒为 `z0_node`；
2. residual 在独立 shadow objective 中训练并完成 finite/eligibility 审计；
3. decoder 结构化任务和 nested validation 通过后，才在相应 fold 的 shadow arm 中允许非零 `alpha`；
4. 每次 shadow 比较同时报告 Correction Rate 与 Corruption Rate；正式晋升另受 fresh-cohort 和预注册统计门约束。

`L_preserve` 不能依赖模糊的“77% 高置信”。基线分数未校准时使用 fold-local margin/rank 语义，并把 exact、N4、Hit@3、far/contralateral-far error 一起纳入预注册 non-inferiority gate。

### 5.7 报告生成与 SOZ 定位的非循环协同

二者首先形成单向、可审计的正向链路：EEG 经 TCP22 evidence carrier 和冻结的 canonical v29 H/D 参考分支产生通道、区域、形态、传播及可定位性预测，再写入 typed evidence slots 与 canonical claim graph；报告模块只能据此选择 `knowledge/eeg` 规则，并首先由确定性 renderer 输出。只有 V2 门通过后，才允许 Qwen 基于同一冻结 knowledge-selection receipt 做受限 lexicalization；两者均不能新增患者事实。因此，报告是定位证据的结构化呈现与一致性检查层，不是新的定位标签来源。

安全的反向协同仅在 V3 训练阶段发生，且唯一监督目标是知识检索和 Qwen 改写之前形成、经过审核的 canonical claim graph。Clause–EEG MIL 将起始、传播和形态主张对齐到允许的通道—时间候选集合；evidence-guided masking 根据其中的通道、时间窗和角色遮挡 TCP22 tokens，要求模型重建 latent、motif、招募顺序及 SOZ 分布。`knowledge/eeg` 卡片、RAG 结果、Qwen 文本及其 embedding 全部 stop-gradient，canonical v29 分支始终冻结。

V4 的一次性 grounded feedback 仅作为消融实验；必须同时报告正确、患者间打乱、左右交换及 onset/spread 交换对照，以及 Correction/Corruption Rate。模型生成的报告不得反过来充当其自身定位正确性的证明，以避免循环自证。

截至当前 Stage 0，只完成信号派生与缓存合同；typed query slots、claim graph 投影、确定性报告、MIL/masking 和 grounded feedback 均尚未训练或运行，不能把本节设计态协同描述成已有实验结果。

## 6. SOZ-Forge

### 6.1 先冻结全项目 split ledger

统一 ledger 至少覆盖：

```text
DeepSOZ / TUSZ / TUEV 的 TUH identity overlap
CHB-MIT / Siena
两套私有数据的重复患者
teacher pretraining exposure
teacher cache lineage
canonical report 与所有语言变体
```

同一患者、事件派生窗、teacher output、hard negative、claim graph 和文本变体必须继承同一 fold。teacher threshold 只在训练/验证患者上校准；锁定测试患者不能参与 prompt、RAG rule、schema 或失败修复策略选择。

ledger 对每名患者同时记录 dataset-official role、EviSOZ model role、outer/inner fold、locked-test flag 和去标识化 global-linkage group；role/fold/locked-test 只能从内容寻址 roster 推导并由 validator 重算。不能靠裸 `patient_id` 字符串相等执行跨库合并。

跨库 identity=`unknown` 必须 fail closed，而不只是禁止写“已去重”：先把所有已知同人边和未解析的可能同人边构成 leakage-conflict graph。对任何声称 patient-disjoint 的 estimand，整个 conflict component 必须作为不可拆分单位共享 dataset/model role、outer fold，并在进入某个 outer-train 后共享相应 inner-fold assignment；不得跨 train、inner-dev、calibration、outer-held-out OOF、source dev/eval、external evaluation 或 locked-test 中的任一训练--评价边界。冻结前只能选择隐私保护 linkage 解析、把整个 component/整个来源固定到同一 role/fold、或从该主张排除；validator 一旦发现未知身份可能横跨任一 held-out boundary 便拒绝 roster。不能在看过结果后把 excluded/held-out/locked cohort 改作同版本开发集。

### 6.2 数据集权限不是统一标签表

| 数据源 | 允许监督 | 禁止 |
|---|---|---|
| DeepSOZ | patient-level documented positive set、laterality/region projection | 把 patient label 复制成逐事件真值；把集合外通道当完整阴性 |
| TUEV | 原子 event/quality/motif task | SOZ 通道监督 |
| TUSZ | seizure-visible/self-supervised/temporal auxiliary | 将标注通道或最早 annotation 当 SOZ |
| CHB-MIT/Siena | seizure representation/domain task | 全零 `y_soz` 作为真实阴性；SOZ 报告 |
| private | 仅有新的 reviewed release 后按字段使用 | 默认把历史 evaluation-only bundle 变成训练数据 |
| CerebraGloss/ELM | 本地 probe 的 OOF 或带 exposure 状态的 frozen-external soft candidate | 把固定外部 teacher 冒充 OOF；direct observation、hard SOZ label、无通道结果时补通道 |

采样先选 task/dataset，再按患者等权选样本。`P(d) proportional sqrt(Nd)` 可以是起点，但必须对单患者多发作和单事件多文本做去重/封顶。

### 6.3 teacher cache

每条 teacher row 保存：

```text
target_split_ref: ArtifactRef
input view and exact montage
components:
  encoder/checkpoint:
    component_role: frozen_external | locally_fitted | deterministic | absent
    artifact_ref: ArtifactRef | explicit_not_applicable
    fit_roster_ref: ArtifactRef | explicit_not_applicable
    exposure_receipt_ref: ArtifactRef
    target_relation: proven_oof | in_fit_roster |
                     known_external_unexposed | known_external_exposed | unknown
  probe/head: same component contract
  calibrator/threshold: same component contract
row_independence_status: all_components_proven_independent |
                         mixed_external_plus_local_oof |
                         exposed_or_unknown | training_only
concept and permitted semantic role
temporal support
spatial support or explicit none
raw score + calibration receipt
availability/failure reason
soft-target weight
```

上面的 `same component contract` 只是文档压缩写法；实际 JSON 必须为 encoder、probe 和 calibrator 分别展开全部字段，不能继承或省略。

CerebraGloss/ELM 的公开固定 encoder 无法事后变成患者 OOF；OOF 是每个本地组件相对该 target row 的关系，不是整条 teacher row 的单一标签。典型的“frozen external ELM encoder + fold-local probe/calibrator”必须分别记录为 external encoder 与 `proven_oof` local components，并将 row 标为 `mixed_external_plus_local_oof`；外部 encoder 的 exposure=`unknown` 时，不能因为 probe 是 OOF 就把整条结果称为 OOF/未暴露。只有当 validator 能逐组件证明 target patient 不在所有 local fit roster，且所有 causal external components 都有排除证明时，row 才能标 `all_components_proven_independent`。`known_external_unexposed` 只能由内容寻址 exposure receipt 对明确列举的预训练来源/患者域给出，不能由 README 或人工备注自报。

权限动作是机器可执行的：任一 causal component 为 `known_external_exposed`、`unknown`、`in_fit_roster` 或缺证明，均自动令 `independence_claim_allowed=false`，禁止“未暴露教师”及独立外部性能主张；若该 component 与目标确认性来源相同或 overlap 无法排除，则 primary confirmatory route fail closed，整条相关 arm 只能排除该 component/pipeline 或降级为明确的 exploratory sensitivity arm。缺任何必需 roster/exposure ref、hash 不闭合或 component relation 与 roster 不一致时，该 teacher row 不得进入 confirmatory residual、calibration 或报告事实。

CerebraGloss 不能从不可恢复的 bipolar-only 输入伪造 referential electrode。ELM crop-level语义没有空间支持时，输出必须保持 `spatial_scope=none`。teacher disagreement 产生 uncertain/soft target，而不是让 Qwen仲裁事实。

## 7. 训练阶段与晋升门

### Stage 0：数据和基线闭合

交付：

- clean/tracked source snapshot；
- master patient/split ledger；
- content-addressed ArtifactRef/schema registry；
- channel registry、Standard19-CAR/TCP22 sibling-view derivation receipt、TCP22 orientation/permutation receipt；
- `evisoz_training_example_v1` schema/validator；
- frozen canonical v29 H/D wrapper、Standard19-CAR baseline cache、signed TCP22 context cache 与 source-isolated TCP22 onset cache；
- exact baseline metric replay。

截至 2026-08-31，**public canonical v29 held-fold wrapper/cache、逐患者 route receipt、immutable checkout、磁盘 materialization receipt v2 和 manifest metric replay 已闭合**。新患者 event-mean cache 仅允许数值审计，生产消费门仍关闭。synthetic 单事件 dual-montage carrier/materializer 也已验证 Standard19-only CAR、A1/A2 排除、有符号 TCP22、缺失 mask、direct-parent onset、外窗不变性、内容寻址 receipt、磁盘重开、递归不可变 receipt、checkout mutation isolation，以及“结果构造失败不得发布/commit 后 close 错误不得反转成功”的进程内发布语义。

私有真实数据不再是 synthetic-only：120 个唯一 EDF 已完成 label/reference 审计；全部具备 `Standard19+A1+A2`，且可以从同一 opaque common-reference parent 派生 CAR19 与 signed TCP22。私有协议 authority 允许重放历史 `unlabeled_common_car19`，但不会把 suffix-free EDF 伪写为 label-proven reference。94 个时间支持事件中 88 个成功物化，6 个 EDF+D 因 discontinuous event clock 排除；88 个 parent/CAR19/TCP22 context/TCP22 onset cache 和 1,936 条 edge 已通过逐字节重开。对应冻结 split 为 34/9 patients、65/23 materialized events；locked 队列可能存在历史 v29 exposure。

88 个事件的医生直接字段已经以 `evisoz_field_release_v1` 落盘，并由 `evisoz_training_example_v1` 绑定 event identity、split、montage 和字段状态。空 positive set 保持 `not_provided`，未见 `DIFFUSE` token 不升级为“明确无弥散”，低置信字段可见但不产生 claim/loss。更重要的是，现有 private doctor-label bundle 的外部授权仍为 `evaluation_only`；当前所有 private envelope 的 `enabled_loss_ports=[]`，只有未来独立的人工/伦理/数据治理 authorization receipt 才能开启训练。43 个医生报告已生成隐私安全 inventory 和自动去标识候选，但仍有 3 个 unresolved association，且人工审核 release 为 0。

确定性 signal-derived candidate cache 也已闭合：88 个真实事件产生 3,608 个 unit feature rows、176 个 global feature rows 和 4,030 个 uncalibrated soft proposals，全部绑定源 dual-montage receipt 并通过 waveform 重算回放；其 `node_localization_supervision_candidate_count=0`。该完成项只把 `offline_teacher_and_derived_candidates` 从 `NO_GO` 推进到 `PARTIAL`，不代表 4,030 个规则触发已经临床校准，也不替代 CerebraGloss/ELM 的独立 candidate cache。

随后生成的 Findings/claim/report 投影是一个有权限边界的投影：88 个 event Findings、88 个 evaluator-only reference claim graph、31 个有时间支持患者的 signal-derived shadow claim graph、31 个 knowledge-selection wrapper 和 31 个 deterministic canonical shadow report。当前 r3 还把已验证的 CerebraGloss 29 条候选写入独立 `teacher_candidates` lane，并在患者级 signal claim graph 中保留 `teacher_candidate` assertion；该投影不复制第二套 measurement 本体，teacher/derived 候选均仅引用候选行及其 SHA-256，且禁止 node-localization supervision。`clinical_labels` 只保留 evaluator-only field release，`direct_measurements/physician_authored_text/generated_text` 仍为空。`knowledge/eeg` 只用于选择术语/安全边界卡片，并在 receipt 中固定 `patient_fact_creation_allowed=false`；它不能补写通道、形态、左右侧或 SOZ。shadow report 明确声明尚未形成可校准定位结论，不是临床发布，也不能作为 Qwen 训练目标。

因此当前状态是 `Stage0_public_v29_reference=GO`、`Stage0_synthetic_dual_montage_contract=GO`、`Stage0_private_real_dual_montage_data=QUALIFIED_GO`、`Stage0_private_field_envelopes=EVALUATOR_ONLY_GO`、`Stage0_private_report_linkage=PARTIAL`、`Stage0_private_report_text_release=NO_GO`、`Stage0_public_auxiliary_exposure_projection=PARTIAL`、`Stage0_offline_teacher_and_derived_candidates=PARTIAL`、`Stage0_findings_claim_graph_and_reports=QUALIFIED_GO`、`Stage0_overall=NO_GO`。deterministic candidate exposure ledger 已闭合其 lineage receipt，报告 mapping intake 已生成但仍等待外部权威填写；剩余阻断项包括公共/辅助数据 near/partial overlap、TUEV eval opaque identity、私有训练 authorization、3 个 unresolved 报告映射、人工去标识审核、CerebraGloss/ELM 小规模校准 candidate cache、fold-local calibration 和 clean freeze audit。不得提前训练 Query Decoder、调用 Qwen、运行大规模 teacher inference 或开放非零 residual。

当前目录发布只保证进程内 no-replace/precommit-replay 语义，不保证突然断电后的目录项持久性；这是可再生研究 cache 的已知非阻断限制。Stage 0 目前还复用了 v29 materializer 的 underscore-private 原子 I/O helpers，后续应提取为共享的公开模块并单独测试，但该技术债不改变本轮 synthetic 合同结论。

在此阶段不训练 Query Decoder，不调用 Qwen。硬门不只比较三个汇总数：patient roster/order、fold assignment、逐患者 `pH_C18/pD_C18/p0_C18`、typed `z0_node` expansion、rank、Top-k prediction、Hit@5、MRR、输入/模型/prediction receipt 与 SHA-256 都必须闭合；可重算浮点量使用事先冻结的逐元素容差。若任何对象漂移，或不能重放 canonical v29 的 52.94% exact、76.47% N4 和 77.45% Hit@3，停止后续实验。TCP22 的端点顺序、符号、edge mask、direct-parent onset 和外窗不变性已在 synthetic contract 上通过；真实数据物化后必须逐事件重新验证，并额外验证与 CAR19 的 sample-clock 对齐及可信 observability binding，不能继承 synthetic `GO`。

### Stage 1：shadow evidence representation

冻结 canonical v29 的 LaBraM/H/D 全链和 `alpha=0`，只训练 TCP22 patch/derivation adapter、motif adapter、sparse edge mixer、query decoder 与 typed heads。每个 loss 由字段级 permission mask 控制：

```text
field-authorized dataset/reviewed labels（不跨任务作全局质量排序）
> calibrated teacher soft targets
> derived candidates
> masked/self-supervised objectives
```

每个 concept 先产生独立 qualification receipt，至少冻结 gold/opportunity 定义、patient-disjoint train/calibration/evaluation roster、metric、最低患者/阳性机会数、校准方法、数值阈值、允许接入的 residual/report port 和失败动作。上述字段必须在读取 evaluation 结果前为非空具体值；未产生合格 receipt 的 concept 默认 `NO_GO`，不能只凭内部 consistency 接 residual 或 report claim。跨 montage/channel-drop consistency 是 guardrail，不代替 native clinical-task validity。

任何在本地数据上拟合过的 adapter/query head/probe/threshold/calibrator 都遵循 outer-fold isolation。对 fold `k`，组件 `A_k` 只能读取该 fold train 患者，并仅用嵌套 inner-dev 做超参、early stop 和 alpha/gate 选择；与 baseline head `B_k` 一起只在 outer-held-out 患者产生 OOF。只有完全冻结的外部 teacher 或不读取任何 cohort label/threshold 的确定性变换可以跨 fold 共享，且仍需 exposure receipt。

### Stage 2：residual localization

只在同时具备 exact v29 reference、可评价 TCP edge evidence、patient-disjoint split，以及 released node-level label 的训练患者上拟合 node residual。公共 OOF 每折只使用对应 held-out v29 H/D；private 只能在新的 release 后进入。edge-level motif/field loss、node documented-positive-set mass、region/laterality、ranking/localizability loss 分别按自己的 typed permission 计算；region/laterality loss 对 node projection stop-gradient，禁止把 edge endpoint 展开结果或 region label 当 node gold。

每个 outer fold 都拥有独立的 `R_k`、gate、alpha selector 和 calibrator；它们只在该 fold train/inner-dev 患者拟合。严禁先收集全部 102 人的 OOF rows，再在这些 rows 上拟合一个 global residual/stacker，并把同一批 rows 继续称为 OOF 评价。

但当前 102 人 public 队列已被反复用于架构开发，private 也已开标；patient OOF 不能消除跨版本模型选择偏倚。因此二者上的新 residual 结果只能叫 exploratory/shadow，不能据此把 `alpha` 晋升为正式部署非零值。正式模型选择需要新的 label-fresh development cohort；确认性主张还需要随后独立锁定的 fresh patient/site cohort。没有这些数据时，发布主链保持 `alpha=0`，结构化 evidence/report 可作为 shadow 输出。

fresh-development 晋升要求：

- paired patient-level exact Top-1 primary non-inferiority；
- N4、Hit@3、MRR 和 far-error 通过预注册 guardrail；
- Corruption Rate 通过预注册上限；
- localizability 的 patient-level macro-F1、balanced accuracy、nonlocalizing sensitivity/coverage 通过预注册下限，且 diffuse/nonlocalizing 中 false-focalization 与 focal 中 false-nonlocalizing 通过预注册上限；任一退化即联合 `NO_GO`，不得用定位 Top-1 的改善抵消；
- patient/fold/bootstrap 结果而非 event 伪重复；
- correct-semantic arm 相对 shuffled/left-right/onset-spread controls 通过预注册交互/差值门。

执行前必须发布 `evisoz_residual_statistical_plan_v1`，逐项写明 estimand、数值 non-inferiority/superiority margin、配对单侧 CI 或检验、置信水平、bootstrap 单位/次数、multiplicity 与联合 GO/NO-GO 规则；不允许 `null` 或“结果出来后确定”。“对照不显著”不等同于无收益，non-inferiority 也不能由“差异不显著”推出。

fresh-development 只负责选择一个候选版本，不能兼作最终确认。打开 fresh locked patient/site cohort 前，必须发布并验证 `evisoz_confirmation_freeze_receipt_v1`，以完整 `ArtifactRef` 绑定最终 source/code snapshot、全部 fold checkpoint、residual/gate、alpha selector、calibrator、环境与配置、eligibility/exclusion roster、确认 cohort roster、统计计划、评价代码、deterministic renderer、knowledge manifest/policy/EviSOZ wrapper receipt、prompt/template、tokenizer 和 decoding 参数。任一引用缺失、hash 漂移或 roster 投影不一致都禁止开盲。

确认性计划以独立 `evisoz_confirmation_statistical_plan_v1` 与 development plan 分开冻结，至少给出唯一 primary estimand、数值 non-inferiority/superiority margin、localizability/safety 数值 guardrail、multiplicity、缺失/不可评价处理，以及一票式联合 `GO/NO_GO`：全部门通过才 `GO`，任何一门失败即 `NO_GO`。fresh locked cohort 对该冻结版本只运行一次；失败后仍保持 evaluator-only，不得转为同一模型版本的开发、prompt 修复、阈值重选或知识规则调整数据。任何修改都产生新版本，并需要新的 label-fresh development 与新的 locked confirmation cohort。

Exact 定义下：

\[
\mathrm{CorrectionRate}=
\frac{\#\{p:\;base\ wrong,\ new\ correct\}}
{\#\{p:\;base\ wrong\}},
\]

\[
\mathrm{CorruptionRate}=
\frac{\#\{p:\;base\ correct,\ new\ wrong\ or\ abstain\}}
{\#\{p:\;base\ correct\}}.
\]

分母为全部可评价患者；new-model abstention 在替换式定位中计为 corruption，并另报 coverage。rank tie 使用冻结的 deterministic tie-break，不得按有利方向处理。N4 可另报同结构的次终点版本，但不能替代 exact 主定义。

### Stage 3：确定性结构化报告

把 typed event slots 通过无损 crosswalk 投影到新的、版本化的 EviSOZ event-Findings/report graph，并由配套 validator 与确定性 renderer 输出报告。该 route 可复用现有 graph 的语义与纯函数，但不得绕开其 public/synthetic/private/Qwen 冻结边界。V1 报告的硬门：

```text
unsupported derivation/node entity = 0
unsupported edge-to-endpoint upgrade = 0
unsupported morphology = 0
laterality reversal = 0
onset/spread reversal = 0
scope violation = 0
all claims carry resolvable evidence IDs
LinkageClosure = 1 on the complete report denominator
all invalid graphs fail closed
```

`LinkageClosure=1` 要求每个 ID 可解析到内容哈希一致的源对象，且 claim 的实体、时间、关系、否定和 certainty 均在授权源中闭合；“字符串里存在一个 evidence ID”不算通过。预注册的单变量 mutation suite 对 side/channel/time/onset-spread/negation/certainty/scope/owner/hash 的检出率必须为 100%，并报告完整 mutation 分母。

只有这一步通过，V1 才完成。

### V2：Qwen3.8 文本侧适配

Qwen 输入仅包含 validated claim graph、claim-level permissions 和白名单知识规则；输出优先是 sentence plan 或 claim-locked wording。任何新增实体、数值、时序、确定性或结论都拒绝并回退确定性 renderer。

在模型权重和 Stage-0 训练授权出现前，仓库只生成 `evisoz_qwen_structured_input_v1` 的 no-generation shadow packet：其 source plan、knowledge selection receipt、card roster、bundle hash 和 32×5120 token contract 均可独立回放。`knowledge/eeg` 的实际消费点是该 packet 所引用的已验证 selection receipt；Qwen adapter 只能按 card ID 读取白名单规则，不能把知识卡内容转成患者事实。只有 Stage-0 aggregate gate 为 `GO`、报告文本 release 已授权且 Qwen one-step/显存/保存恢复 smoke 通过后，才可将该 packet 投影为真实 `inputs_embeds` 并开放 lexicalization；在此之前 `status=shadow_input_no_generation` 必须保持不变。

本地 `models/Qwen3.8-27B-FP8` 的 config/README 确认为 Apache-2.0 的 dense native VLM、`model_type=qwen3_5`、text hidden size 5120、64 layers、native context 262144；但：

- 现有 local Qwen loader 只适配 qwen2.5-VL 和 qwen3.5/3.6 MoE GPTQ，不能直接加载该 dense release；
- FP8 inference artifact 不等于 NF4 4-bit QLoRA base；
- 仓库当前没有经过验证的 bitsandbytes/PEFT/TRL Qwen3.8 训练栈。

48 GB 上的 QLoRA 只是待验证假设。必须在独立环境依次通过：

```text
K=0, rank=16, seq=2048 one-step forward/backward/save/resume
 -> K=32 projector smoke
 -> rank/sequence length scaling
 -> peak VRAM receipt
```

不能从“权重约 4 bit”直接推出训练一定能放入 48 GB。

### V3：EEG--文本对齐和 evidence-guided masking

先训练 TCP22 edge token 到**结构化 slot/claim embedding** 的 MIL 对齐；唯一 target 是知识选择之前形成的、reviewed/pre-knowledge canonical patient evidence graph。知识卡、检索结果、Qwen 文本及其 embedding 全部 stop-gradient，不能选择 EEG mask、构成 reconstruction/residual target 或进入任何定位 loss；模型刚生成的自然语言也不是新真值。mask 只施加在 TCP22 evidence carrier；canonical v29 reference waveform、H/D reasoner 和 `z0_node` 始终冻结且不被遮挡。相位反转 claim 必须联合遮挡/重建构成该 claim 的相邻有向边、共享端点关系和 role，不得只遮一条边后把共享电极当标签。

EEG continuous embeddings 若以后送入 Qwen，必须单独验证：`inputs_embeds`、position IDs/mRoPE、special-token boundary、无 EEG 时 logits parity，以及 latent token 不能绕过 claim permissions。未通过前 EEG resampler 只用于辅助 slot reconstruction，不是报告事实来源。

### V4：一次性 grounded feedback（仅消融）

generated-report feedback 不属于主部署链。只有以下全部成立才可做预注册的一次性消融：

- 自动 claim 与独立事实 evaluator 的 precision/recall 通过数值门；
- correct feedback 相对 shuffled、cross-patient、left-right 和 onset/spread swap 通过配对交互门，且 controls 相对 no-feedback 通过预设等效性门；
- generated-report feedback 与 canonical-claim feedback 通过配对等效性门；
- paired Correction Rate 与 Corruption Rate 同时通过数值门；
- fresh patient/site 复核可用。

任何 claim 未过 grounding 时 gate 必须为零；反馈不得覆盖或重写原始 clinical label。

在生成任何 V4 结果前必须冻结 `evisoz_grounded_feedback_statistical_plan_v1`。V4 的初始硬阈值固定为：claim micro-precision 的 patient-bootstrap 95% 下界 `>=0.99`、micro-recall 下界 `>=0.95`，且 side/channel/onset-spread/scope 的 unsupported/reversal 计数为零；generated-vs-canonical 的 patient-level exact-correctness 配对风险差 90% CI 完全落在 `[-0.02,+0.02]`，且 generated-minus-canonical Corruption Rate 的单侧 95% 上界 `<=0.01`；Correction Rate 点估计 `>=0.05` 且单侧 95% 下界 `>0`，Corruption Rate 单侧 95% 上界 `<=0.01`。

边界事件不能使用会退化成 `[0,0]` 或 `[1,1]` 的普通 bootstrap 伪造完美区间。V4 最低有效分母固定为：`N_evaluable_patients>=399`、`N_base_correct>=299`、`N_base_wrong>=100`；每个 side/channel/onset-spread/scope safety family 至少有 299 个可评价 claims、来自至少 100 名患者，recall 的每个预注册 salient family 至少有 100 个 reference claims。a-priori 80% power 计算只能提高这些最低数，不能降低。分母不足、某一类别无 opportunity 或有效 paired rows 不足均直接 `NO_GO`，不能记为通过或从分母删除。

非边界配对差使用至少 10,000 次 patient-cluster BCa bootstrap；零错误/全正确等边界率同时计算 patient-level “任一错误”指标与 claim-level 指标的一侧 Clopper--Pearson bound，并取更不利的界。Corruption 的 exact bound 在 `base-correct` 分母上计算，Correction 的 bound 在 `base-wrong` 分母上计算。若 bootstrap 退化、BCa 不可定义或 exact/cluster 两种界结论冲突，按更保守结果判定；不得用退化 bootstrap 区间替代非退化边界方法。

语义特异性对每个 control 检验配对交互

\[
\bigl(\Delta exact_{correct}-\Delta exact_{control}\bigr)>0.02,
\]

并要求其单侧 95% 下界大于 `0.02`；同时 control-vs-no-feedback 的 90% 等效 CI 必须完全位于 `[-0.01,+0.01]`。四个 control 的交互与等效性检验分别使用 Holm 校正；总决策按 claim fidelity、canonical equivalence、semantic specificity、Correction/Corruption 的固定层级顺序执行，任一级失败即 `NO_GO`。这里“不显著”不能解释为“无收益”“等效”或“安全”。计划还必须冻结 estimand、患者级 resampling、tie/abstention、缺失值和完整 multiplicity 规则；结果出来后不得改 margin。

### 跨阶段正式消融矩阵

所有 arm 使用相同 patient split、token cache、seed roster、训练/调参预算、early-stop rule 和评价代码；不能把多个新增模块捆成一个 arm 后归因：

| Arm | 冻结定义 |
|---|---|
| A | frozen canonical v29 H/D Standard19-CAR reference |
| B | A + raw TCP22 edge shadow query decoder（`alpha=0`） |
| C | B + spectral/baseline Clinical Motif Adapter |
| D | C + sparse temporal--spatial mixer |
| E1/E2/E3 | D + CerebraGloss / ELM / deterministic candidate，分别加入 |
| F | D + 固定 teacher/candidate fusion |
| G | F + released residual gate |
| H | G + deterministic claim-locked report |
| I1 | H + knowledge rules |
| I2 | H + constrained Qwen without knowledge |
| I3 | H + knowledge rules + constrained Qwen |
| J1 | I3 + clause--EEG MIL |
| J2 | I3 + evidence-guided masking |
| J3 | I3 + clause--EEG MIL + evidence-guided masking |
| K | J3 + one-shot grounded feedback ablation |

J1--J3 从 I3 继承报告能力只用于保持系统配置可比；其 EEG-side target 始终回溯到 pre-knowledge canonical patient evidence graph，knowledge/Qwen 路径冻结且 stop-gradient。否则 J arm 不再是安全的 representation-feedback 实验。

语义对照至少包括 correct claim、cross-patient shuffle、左右交换、onset/spread 交换、Top-1-only、无知识库，以及 canonical-evidence feedback 与 generated-report feedback。每个“语义有效”结论必须检验 correct arm 相对相应 control 的配对差值，而不是只看各自是否显著。

主配对比较在训练前冻结为：`B-A`、`C-B`、`D-C`；`E1-D/E2-D/E3-D` 与 `F-D`；`G-F`；`H-G`；`I1-H`、`I2-H`、`I3-I1`、`I3-I2`；`J1-I3`、`J2-I3`、`J3-I3`、`J3-J1`、`J3-J2`；以及 `K-J3`。同一数据/模块 family 内采用 Holm 校正，跨 family 按 baseline/evidence、teacher、residual/report、language、representation-feedback、grounded-feedback 的预注册层级门依次检验；上游 family 未过门时，下游只作 exploratory。不得从未预注册的 arm 间最优差值回填主假设，也不得把 `J3` 的收益归因给单个组成模块。

每组比较的主 estimand、方向、配对单位和门固定如下，具体数值 margin 写入相应 family 的内容寻址 statistical plan 后才能运行：

| Family / comparison | 主 estimand 与方向 | 配对/分母与决策 |
|---|---|---|
| `A↔B` | frozen localization logits 必须 exact parity；B 的 typed evidence 通过绝对 qualification 门 | 定位按 patient 配对；A 无 evidence 输出，因此不得伪造 evidence 差值 |
| `C-B`, `D-C`, `E1/2/3-D`, `F-D` | 预注册 concept-specific typed-slot metric 提高，同时 calibration/consistency 不退化 | evidence opportunity 为观测、patient 为 cluster；每个 concept 单独方向/margin，未资格 concept 不参与汇总 |
| `G-F` | patient exact Top-1 non-inferiority 为主，预设 superiority 为可选次序；Correction 提高且 Corruption/localizability 过门 | patient-level paired estimand，完全受 residual/confirmation plans 约束 |
| `H-G` | 首次 report 的 factual safety、LinkageClosure 和 mutation detection 通过绝对门；localization logits exact parity | G 没有 report，故禁止计算虚假的 report paired delta；H 先过绝对资格门 |
| `I1-H`, `I2-H`, `I3-I1`, `I3-I2` | claim factuality/scope non-inferiority先过，之后才检验盲法语言质量或知识规则命中改善 | 同一 claim graph/report 的 patient/report 配对；安全失败覆盖任何语言偏好 |
| `J1-I3`, `J2-I3`, `J3-I3/J1/J2` | fresh-development patient exact 定位、Correction/Corruption 与 correct-vs-semantic-control 交互 | patient 配对；组合 arm 只声明联合/交互，不把效果归给单模块 |
| `K-J3` | V4 claim fidelity、generated-canonical equivalence、semantic specificity、Correction/Corruption 联合门 | patient 配对；仅按 `evisoz_grounded_feedback_statistical_plan_v1` 一票式判定 |

上述 family plan 必须填入数值 margin、CI/test、方向、完整分母和 multiplicity；任何字段为空时 validator 将对应比较标为 `NOT_RUN/NO_GO`，不能在结果后补写。

## 8. 报告链

正式报告数据流固定为：

```text
validated event slots
 -> event Findings
 -> multievent hypothesis/claim graph
 -> claim-level report_scope gate
 -> deterministic knowledge/eeg selection
 -> evisoz_knowledge_selection_receipt_v1 wrapper
 -> deterministic sentence plan
 -> optional constrained Qwen lexicalization using the same receipt
 -> independent validator
 -> deterministic fallback
```

### 8.1 `knowledge/eeg` 的准确使用位置

活动入口是 `knowledge/eeg/manifest.json`，不是历史设计稿。2026-08-31 的结构校验结果为：bundle `2.0.0-draft`，`active_bundle_sha256=cb593b0a6813aeea50bc0f80ad7da4b7e6d4dce5c6d68bde25d0acf6325532c2`，12 张 knowledge cards、25 条 source passages、0 张完成临床审核的卡片、18 个待补 source locators，且 `clinical_deployment_allowed=false`。因此它可用于研究性规则约束和草稿报告，不能被写成已获临床部署批准的知识服务。

它只在 **validated patient claim graph 已形成之后**参与选择、grounding 和措辞：

| 路径 | 在 EviSOZ 中的作用 | 明确禁止 |
|---|---|---|
| `terminology/electrode_aliases.json`、`legacy_terms.json` | 对已冻结实体做 T3↔T7 等显示/检索别名和高风险措辞检查 | 改写 waveform channel registry、补导或改变端点 |
| `ontology/semantic_layers.json`、`region_hierarchy.json`、`relations.json` | 限定 OBS→PAT→LOC→CLIN 层级、区域粒度和关系类型 | 从一般区域知识生成患者 laterality/channel |
| `reasoning/evidence_roles.json` | 判定 direct measurement、candidate、limitation 等证据角色 | 提升 teacher/derived candidate 为 gold |
| `reasoning/grounding_rules.json` | 检查 earliest、field、phase reversal、跨 montage 和 multi-event claim 的必要证据 | 用规则本身替代 patient evidence |
| `reasoning/inference_rules.json` | 执行允许升级、禁止跳跃和 fail-closed 降级 | 将最早头皮改变、IED 或相位反转升级为 cortical SOZ/EZ |
| `reporting/claim_policy.json` | 执行 `report_scope`、认识状态、必需局限性和允许措辞 | 自动生成未经授权的 `noninvasive_soz_hypothesis` 临床结论 |
| `profiles/*.json` | 仅在记录元数据明确匹配 routine/critical-care/pediatric/neonatal 时选择适用规则 | 根据模型预测猜 profile |
| `cards/core_safety_cards.jsonl`、`knowledge_base.jsonl` | 为已选择的一般规则提供 card/source provenance | 作为患者标签、SOZ 打分或缺失事实补全 |

TCP22 相位反转 claim 直接复用 `reasoning/grounding_rules.json` 的现有合同：至少两条实际 montage 中的相邻双极导联、共享电极和已知 polarity convention，且每条证据必须解析到同一记录的精确时间范围。满足这些条件只允许生成“共享电极附近的局部场/相位反转候选”；`reasoning/inference_rules.json` 中的 `phase_reversal_to_cortical_source` 禁止跳跃仍然生效。

每次选择先生成符合 `knowledge/eeg/schemas/knowledge_selection_receipt.schema.json` 的 base receipt。该 schema 已绑定 bundle SHA 与 per-card SHA，但只保存 `selected_source_ids`，没有 manifest ArtifactRef 或逐 source-passage row SHA；所以确认链还必须生成新的 `evisoz_knowledge_selection_receipt_v1` wrapper，至少绑定：

```text
knowledge_manifest_ref                  # manifest bytes/ArtifactRef SHA-256
knowledge_version + active_bundle_sha256
retrieval_policy_ref
base_selection_receipt_ref
profile_id
query_fact_ids                       # 只引用冻结的患者事实 ID
selected card IDs + per-card SHA-256
selected source passage IDs + per-row SHA-256 + source-registry refs
selection reasons
rejected high-risk cards
patient_fact_creation_allowed=false
```

患者 claim 同时闭合两条互不替代的链：

```text
patient claim -> patient evidence IDs
patient claim -> knowledge card IDs -> source IDs
```

Stage 3 的 deterministic sentence plan 与 V2 Qwen 必须读取**同一个冻结 wrapper receipt**；Qwen 不允许重新检索或扩大规则集合。`no-knowledge` 只作为预注册消融，不能通过改 prompt 或 test patient query 做补偿。

当前仓库的真实实现边界也必须披露：`src/soz/constrained_llm_reporting.py` 及现有 materialize/audit 脚本主要读取兼容层 `knowledge/eeg/knowledge_base.jsonl`；`configs/eeg_knowledge/retrieval_policy_v2.json` 明确是 `declaration_only_not_implemented`。因此在新增中立 v2 selector、receipt validator 和 qualification test 之前，不能声称 ontology/reasoning/reporting 全 bundle 已被在线 RAG 消费。V1 应先实现确定性 selector；V2 才将其冻结输出交给 Qwen。

### 8.2 报告与验证边界

知识库只能提供术语、一般判据、允许/禁止推理和局限性，不提供患者事实，也不进入 v29 定位、TCP22 波形特征、teacher target、loss label 或 calibration target。Qwen 不是 EEG measurement tool、SOZ predictor 或事实裁判。使用同一个 LLM 做文本生成和文本回解析不能构成独立质检；最终 validator 必须直接对 source claim graph 检查实体、关系、时序、否定、确定性、scope 和 evidence closure。

按当前 `claim_policy.json`，自动链最高默认停在 `LOC/scalp_localization`。若缺少独立多模态证据和 authorized reviewer，`CLIN.noninvasive_soz_hypothesis` 必须保存为 `not_authorized/pending_review`，正文写“头皮 EEG 支持的最早可见定位候选/跨事件定位一致性”，不能仅因多次 EEG 一致就自动升级为临床 SOZ 假设。

自动草稿固定使用“头皮 EEG 支持的最早可见发作起始候选”；只有完成上述临床授权升级后才可使用“非侵入性 SOZ 假设”，并始终明确：

- 仅分析给定发作段；
- 未提供或不可评价的字段不补写；
- scalp-visible onset 不等于 cortical SOZ/EZ；
- 未结合 iEEG、MRI、PET、手术范围或术后结局；
- 需要临床神经电生理医师复核。

## 9. 评价合同

### 9.1 定位主次终点

主终点：

```text
patient-level exact node documented-positive-set Top-1
（仅 exact v29 reference + released node label 可评价患者）
```

关键次终点：

```text
N4 relaxed Top-1
Hit@3 / Hit@5
MRR / documented-positive probability mass
laterality / region
far and contralateral-far errors
patient bootstrap confidence intervals
Correction Rate / Corruption Rate
```

N4 是评价容忍图，不是训练用物理连接图。DeepSOZ reference 不穷尽时，Macro-AP/AP/PR-AUC、Brier/ECE 和 prediction-set coverage 不能作为无条件医学正确性指标；只有对应 target 定义、穷尽性和 patient-disjoint calibration 成立时才报告。若保留 historical Macro-AP，只能明确标为 documented-positive reference-agreement diagnostic，不得把未标通道解释为真阴性。

TCP22 输出另设 edge-space/evidence 终点，不能与 C18 node 指标混算：

```text
edge Top-1/Top-k/MRR                    # 仅有原生 bipolar-edge gold 时
phase-reversal pair precision/recall   # 两条有向相邻边 + shared electrode
shared-electrode candidate accuracy    # 仅经独立人工审核机会集
edge coverage by input profile
orientation-flip equivariance error
unsupported endpoint-expansion rate = 0
```

node residual 的增益只在同一患者上与其 frozen v29 `z0_node` 配对比较；`tcp22_native` 且 v29 unavailable 的 standalone 输出单列，不进入 Correction/Corruption 分母。SOZPreNet 等 TCP22-only comparator 也只报告自己的 edge target，不用其 edge Top-1 与 v29 exact/N4 作数值排名。

### 9.2 onset/spread 与 localizability

只有拥有相应 gold/opportunity 的病例进入分母：

```text
pairwise recruitment-order accuracy
interval/censor-aware recruitment-time error
onset-spread reversal rate
early-spread positive-set recall/F1（仅 exhaustive reference）
focal/diffuse/nonlocalizing sensitivity
patient-level macro-F1 / balanced accuracy
false-focalization rate among diffuse/nonlocalizing cases
false-nonlocalizing rate among focal cases
abstention/coverage and risk-coverage（仅校准后）
```

localizability threshold、最小患者/类别支持数、不可评价处理和完整分母必须在 locked evaluation 前冻结；同时报告 sensitivity 与退化 guardrail，禁止通过“全报 diffuse”或“全报 focal”获得表面单项优势。

### 9.3 报告

报告主指标是 claim factuality，不是 BLEU/ROUGE：

```text
entity/relation/temporal/negation precision and recall
LinkageClosure
ChainPrecision + SalientChainRecall
unsupported entity/morphology/time rate
uncertainty preservation
scope/boundary compliance
physician accept/edit/reject and edit distance by claim type
```

模板 V1 要求安全类错误为零。语言质量在相同 claim graph 下做 template-vs-Qwen 盲法比较。

## 10. 仓库落点

### 10.1 可直接复用的模型/基础设施

```text
src/soz/models/labram.py
src/soz/v29_long_recording_inference.py
src/soz/formal_reasoner_pipeline.py
src/soz/evidence.py
src/soz/evidence_schema.py
src/soz/aggregation.py
src/soz/models/standard19_motif_filter.py        # 复用设计/局部算子
code/soz_pre/constants.py                        # TCP22 唯一顺序；不复用旧标签升级策略
code/soz_pre/preprocess_unified_soz.py           # 仅复用经审计的有符号波形衍生局部函数
knowledge/eeg/manifest.json
knowledge/eeg/reasoning/grounding_rules.json
knowledge/eeg/reasoning/inference_rules.json
knowledge/eeg/reporting/claim_policy.json
```

报告组件只能按现有授权边界复用：`clinical_eeg_multievent_soz_report_graph_v2` 当前是 public/synthetic shadow，private route 与 Qwen production route 都是断开的；`multievent_soz_claim_validation.py` 和 `multievent_report_render.py` 当前验证/渲染的是 graph v1。EviSOZ 可以复用其 claim/evidence closure、mutation validation 和确定性 lexicalization **思想与局部纯函数**，但不能把现有冻结常量改开、把 graph v2 直接送入 v1 renderer，或声称已经具备 private/Qwen 正式路由。

### 10.2 建议新增且隔离

所有强制 `ArtifactRef.schema_version` 必须先登记在内容寻址的 `evisoz_schema_registry_v1`，其条目绑定 `schema_version -> schema ArtifactRef -> validator implementation/version`；registry 自身的 trust-root SHA 写入 Stage 0/freeze receipt。validator 必须先验证 registry、再验证目标 artifact 的 schema 与内容 hash，未知版本一律拒绝。若引用仓库已有外部 schema（例如 canonical signal/Findings），也必须在 registry 中绑定其冻结 schema SHA 与 adapter，不能只凭名称假定兼容。

```text
schemas/evisoz_artifact_ref_v1.schema.json
schemas/evisoz_schema_registry_v1.schema.json
schemas/evisoz_patient_linkage_group_v1.schema.json
schemas/evisoz_split_roster_v1.schema.json
schemas/evisoz_field_release_v1.schema.json
schemas/evisoz_holdout_freeze_audit_v1.schema.json
schemas/evisoz_channel_registry_v1.schema.json
schemas/evisoz_montage_derivation_receipt_v1.schema.json
schemas/evisoz_dual_montage_cache_materialization_receipt_v1.schema.json
schemas/evisoz_position_registry_v1.schema.json
schemas/evisoz_observability_receipt_v1.schema.json
schemas/evisoz_teacher_cache_row_v1.schema.json
schemas/evisoz_training_example_v1.schema.json
schemas/evisoz_reference_claim_graph_v1.schema.json
schemas/evisoz_canonical_report_v1.schema.json
schemas/evisoz_event_findings_projection_v1.schema.json
schemas/evisoz_multievent_report_v1.schema.json
schemas/evisoz_knowledge_selection_receipt_v1.schema.json
schemas/evisoz_confirmation_freeze_receipt_v1.schema.json
schemas/evisoz_*_statistical_plan_v1.schema.json
src/evisoz/data/artifact_ref.py
src/evisoz/data/schema_registry.py
src/evisoz/data/channel_registry.py
src/evisoz/data/tcp22_views.py
src/evisoz/data/stage0_dual_montage_cache.py
src/evisoz/data/observability.py
src/evisoz/data/dataset_policy.py
src/evisoz/data/split_ledger.py
src/evisoz/forge/training_example.py
src/evisoz/forge/teacher_cache.py
src/evisoz/forge/canonical_claims.py
src/evisoz/models/evisoz_v1.py
src/evisoz/models/bipolar_derivation_adapter.py
src/evisoz/reporting/claim_graph_adapter.py
src/evisoz/reporting/knowledge_select.py
src/evisoz/reporting/validate.py
src/evisoz/reporting/render.py
experiments/evisoz/materialize_evisoz_token_cache_v1.py
experiments/evisoz/train_evisoz_v1.py
configs/evisoz_v1.json
configs/evisoz_schema_registry_v1.json
tests/test_evisoz_stage0_dual_montage_cache.py
tests/test_evisoz_*.py
```

### 10.3 V1 最小测试矩阵

```text
channel aliases preserve original + normalized
node/edge never interchange
ArtifactRef canonical-JSON/raw-byte hash domains replay exactly
schema registry rejects unknown/unbound schema versions and validator drift
Standard19 node observability [E,19,60] and TCP22 edge observability [E,22,60] never collapse
TCP22 endpoint order/permutation/sign round-trip exactly; reversed source edges are sign-corrected with receipts
dual-native CAR19/TCP22 sibling views share an exact sample clock and parent signal receipt
source-isolated onset artifact depends on raw parent + core-local preprocessing only, never on context artifact
missing/extra electrodes fail or mask by profile
changing arbitrary values under a false edge mask cannot change any token, pooled state, claim or loss
interpolated transport never becomes observed/direct evidence or phase-reversal support
deterministic fact-state/opportunity crosswalk fails closed
split/teacher/text lineage and unresolved identity components cannot cross any role/outer/inner/held-out boundary
component-level encoder/probe/calibrator fit/exposure rosters enforce mixed-pipeline OOF/independence permissions
unknown fields and unauthorized claims fail closed
baseline tensor/logit/metric exact replay
residual hard-bypass exact identity, NaN/Inf attacks and later gradient flow
event-to-patient [P,Mmax,6,Dq] shape/ownership, conflict/overflow hard-abstention and fold-specific stacking isolation
query-positive [E,6,22,60], penalty-port [E,3,22,60] and edge-cell-mask permission attacks
TCP-context replacement outside exact core followed by full adapter-feature recomputation leaves H_edge_onset/Qonset/Delta_z_node/logits/ranks bitwise invariant
onset-positive path has zero baseline/full-context feature gradients
no edge/local-field/phase-reversal evidence can activate a node residual without released node-level supervision; phase-reversal claims require two observed oriented adjacent edges
holding early cells fixed, quality/generalized/late penalty-port perturbations leave logits/ranks invariant and only non-decrease abstention
masked-max event-to-patient burdens are coordinate-monotone; empty bag, any missing burden coordinate, zero/multiple focal modes force abstention
patient-bag and fold isolation
claim mutation: side/channel/time/onset-spread/certainty/scope
EviSOZ knowledge wrapper receipt binds manifest/policy/base-receipt/bundle/card/source-row hashes and cannot create patient facts
v2 knowledge selector absent or invalid -> deterministic no-upgrade/fallback, never legacy free retrieval
Qwen failure -> deterministic fallback
confirmation-freeze receipt rejects any code/checkpoint/roster/statistical-plan/render/prompt drift
```

## 11. 实施顺序与停止规则

```text
P0  tracked clean snapshot + split/channel/schema/dual-view contracts
 -> P1 canonical v29 wrapper + CAR19/TCP22 sibling caches + exact replay/orientation audit
 -> P2 TCP22 shadow query decoder + typed edge/event slots
 -> P3 zero-initialized residual shadow + fresh-cohort paired non-inferiority gate
 -> P4 claim-graph projection + deterministic report
 -> V1 release
 -> Qwen3.8 isolated training/runtime smoke
 -> V2 constrained lexicalization
 -> V3 structured MIL alignment + evidence-guided masking
 -> V4 grounded-feedback ablation only
```

任一阶段失败均保留上一个通过版本，不让未资格化模块进入定位、报告事实或临床措辞。尤其在 `Stage0_overall=GO` 前，不得启动 Query Decoder/residual 正式训练、非零 `alpha`、大规模 teacher 推理、Qwen SFT 或新的私有标签训练；synthetic dual-montage contract 的 `GO` 不解除这些停止门。

## 12. 论文级概括

EviSOZ-LM 的核心不是把多个公开模型串成一条不可审计流水线，而是建立一个带权限和 provenance 的双 montage 证据瓶颈：冻结 canonical v29 H/D 在完整 Standard19-CAR 上保持可重放的 node ranking；同源、有符号 TCP22 edge carrier 提供相位反转、局部场、形态和传播证据；轻量 query decoder 将其解码为可评价、可缺失、可追溯的 typed event slots。edge 证据只有经过授权 projection 才能形成相对 `z0_node` 的残差，且残差只有在 label-fresh development 上通过预注册患者级 non-inferiority、并在 fresh locked cohort 确认后才能改变正式定位；多发作聚合保留冲突 mode；知识库和 Qwen 只约束已经授权的 claim；文本语义仅通过 canonical evidence-guided masking 或经严格消融的低权重路径帮助 EEG 表征。

最终可支持的主张是：

> 在已知发作片段条件下，对最早头皮可见发作起始候选、早期演变和多事件一致性进行证据约束建模，并生成事实可追溯、可失败回退、需医师复核的头皮定位研究草稿；只有在独立多模态证据与授权复核闭合后才升级为非侵入性 SOZ 假设。

## 13. Stage‑0 阻断补件（2026-09-01 r38）

针对五类独立阻断，新增了隐私安全的补件包：
`outputs/evisoz_stage0_remediation_packet_v1_20260901_r6/`。该目录只保存
哈希、计数、候选 ID 和空白审查/签发字段，不复制原始 EDF、报告正文或患者身份。

当前 r38 gate 仍为 `NO_GO`。补件包确认：CerebraGloss 的本地
`model.pkl` 虽存在，但其已登记的外部审计状态为
`no_go_for_current_tuev_bipolar_ce6_morphology_producer`；目前仅有
development-only 的 2 个事件/29 条 candidate cache，仍不能替代完整
development roster；仓库内没有 ELM checkpoint/manifest；fold-local calibration receipt 仍为 0。43 条报告的
自动 PHI 扫描为 43/43，但人工审核和 development/evaluator release 仍为
0。三条 unresolved 报告的精确源字节已在受控源目录中找到（未复制进仓库），
但文件名/正文没有给出冻结患者 roster 的唯一匹配，因此仍只保留待权威
crosswalk 或显式排除的请求。公开审计请求仍要求
TUSZ/TUEV decoded near/partial overlap、TUEV eval session→patient authority
和 TUEV fold-scoped label/exposure receipt。

私有授权模板已规范为
`claimed_authorized_pending_controller_confirmation`：`dyf`、`suat` 和
授权方 `suat` 被记录为用户声明，不被解释为机构签名或训练授权。只有
控制方签发的审批引用/签名和上述外部审计输入到位后，才能重新物化依赖
manifest 并重新计算 Stage‑0；任何修改状态字段而不重放内容寻址校验都属于
审计违规。

## 14. CerebraGloss development-only 小规模真实物化（2026-09-01 r37）

已新增 `scripts/run_evisoz_cerebragloss_stage0_inference_v1.py`。该入口只
读取已通过 replay 的真实双 montage cache，不读取 EDF、医生报告或
`knowledge/eeg`，并将 canonical v29 CAR19 的 `[t0-2s,t0+8s]` 切片重排为
CerebraGloss 训练时使用的 19 导联顺序。旧 TUH 名称 `T3/T4/T5/T6` 到
canonical `T7/T8/P7/P8` 的运输映射写入 teacher model manifest；Evidence
JSON 中仍保留 canonical 节点名，不发生 endpoint 或 SOZ 标签提升。

本轮受限运行了 2 个 `development_cv` events，产生 29 条
`candidate_only/uncalibrated/soft_auxiliary` 候选，并由
`scripts/materialize_evisoz_teacher_candidates_v1.py` 物化为
`outputs/evisoz_stage0_cerebragloss_candidates_v1_20260901_r2/`。候选只绑定
event identity、dual-montage receipt、teacher model manifest 和输入视图；
`training_authorized=false`、`calibration_authorized=false`，不能作为
clinical label、measured fact 或 node-localization supervision。

最新 gate 为
`outputs/evisoz_stage0_gate_v1_20260901_r37/gate.json`，整体仍为 `NO_GO`。
CerebraGloss 缺失项已移除；剩余 teacher-side 阻断为 ELM candidate、
fold-local calibration，以及私有治理、医生报告 release、公开 overlap/
TUEV eval identity 和 clean-freeze 审计。对应执行计划为
`outputs/evisoz_execution_plan_v1_20260901_r8/plan.json`；它仍禁止
Query/Residual 正式训练、非零 residual、Qwen SFT、EEG-to-Qwen 对齐和大规模
teacher inference。

## 15. clean-freeze 审计与 Stage-0 重放（2026-09-01 r39）

新增 `scripts/materialize_evisoz_clean_freeze_audit_v1.py`，把 P0 的
“tracked clean snapshot + contract presence”变成只读、内容寻址的审计入口。
审计固定检查 schema registry、结构化证据配置、训练 envelope schema、
`knowledge/eeg` manifest/规则/报告边界等 8 个合同文件；同时记录 Git HEAD、
分支、状态计数和不含文件名的状态摘要。该审计本身**永远不授予训练权限**，
且审计输出路径从自身状态摘要中排除，避免生成 receipt 反过来污染快照。

本轮结果为
`outputs/evisoz_clean_freeze_audit_v1_20260901_r2/audit.json`：8/8 合同存在，
但当前工作树有 34 个已修改条目和 119,074 个未跟踪条目，故
`status=NO_GO`、`training_authorized=false`。该结果反映的是发布快照未冻结，
不是 EEG 双 montage 物化失败。

Stage‑0 gate 已新增 `clean_freeze_audit` 检查，并在当前代码/registry 上重放为
`outputs/evisoz_stage0_gate_v1_20260901_r45/gate.json`；当前整体仍为 `NO_GO`。
对应执行计划为
`outputs/evisoz_execution_plan_v1_20260901_r14/plan.json`，新增的阻断项是
`clean_freeze_audit`，其余阻断仍为 ELM candidate、fold-local calibration、
私有治理授权、医生报告映射/release、公开 overlap/TUEV identity 审计。
真实 88-event cohort validation 和 bound-evidence loader replay 仍成功，
structured smoke 保持 residual zero。新的 remediation packet 位于
`outputs/evisoz_stage0_remediation_packet_v1_20260901_r7/`。

## 16. Findings 中的离线 teacher candidate lane（2026-09-01 r40）

为避免已产生的 CerebraGloss 候选只停留在独立 cache、无法进入统一
Evidence/claim graph，`build_event_findings_projection` 现在可接收经验证的
`teacher_candidate_materialization`。它把每条候选复制为带
`source_teacher_candidate_cache_ref` 和 `row_sha256` 的独立 `teacher_candidates`
lane；候选的 `teacher_id`、支持视图、支持区间、置信度和概率语义均保留，
但 `authority=offline_teacher`、`status=candidate_only`、
`calibration_state=uncalibrated` 和 `permitted_uses=[soft_auxiliary]` 是硬约束。

患者级 signal claim graph 可同时承载 `derived_candidate` 与
`teacher_candidate` 两类 shadow claim，但两者都禁止 clinical label、measured
fact、node localization supervision；teacher edge 还禁止 endpoint expansion。
bound evidence 只记录该 lane 的状态，不复制 teacher 数值，也不改变任何 loss
port。`scripts/materialize_evisoz_findings_claim_reports_v1.py --teacher-candidates`
是可复现入口。

本轮真实重放结果为：88 个事件、31 个患者级图、4,059 条候选 claim（其中
4,030 条 deterministic、29 条 CerebraGloss），所有 canonical report 仍为
`research_shadow_not_clinical`，`generated_text_fact_count=0`。结果位于
`outputs/evisoz_stage0_findings_claim_reports_v1_20260901_r3/` 和
`outputs/evisoz_stage0_bound_evidence_v1_20260901_r27/`；loader replay 覆盖
88/88 events 与 31/31 patients。该接入只推进 lineage 可见性，不解除 ELM、
fold-local calibration、治理授权、报告 release、公开 overlap/TUEV identity
或 clean-freeze 等 Stage‑0 阻断。

## 17. 真实双 montage shadow 推理闭环（2026-09-01）

在不越过 Stage‑0 训练门的前提下，新增
`scripts/run_evisoz_real_shadow_inference_v1.py`。该入口从 r50 bound-evidence
重放 88 个真实事件，使用官方 LaBraM `labram-base.pth` 编码同源的
Standard19/CAR 节点视图和有符号 TCP22 边视图，随后运行当前
implementation-only Clinical Motif/时空/query decoder，输出 event-level
`predicted_evidence`、candidate-only report plan、31 个患者级聚合和
no-generation Qwen input。所有结果都保持 `stage0_status=NO_GO`、残差恒等
（`z1=z0`），TCP22 边没有展开成端点标签，也没有打开医生 DOCX、教师运行时、
知识卡正文或 Qwen 生成。

全量 receipt 位于
`outputs/evisoz_real_labram_shadow_v1_20260901_r6/receipt.json`（仅 token
shape/identity shadow）和
`outputs/evisoz_real_shadow_inference_v1_20260901_r1/receipt.json`（证据、
报告计划和患者级结构闭环）。结构评估为 88/88 event packet、88/88 report
plan、31/31 patient packet，mask consistency、report linkage 和
claim-support validity 均为 1.0；这些是接口/结构指标，不是定位性能。
最新 Stage‑0 gate 为
`outputs/evisoz_stage0_gate_v1_20260901_r54/gate.json`，执行计划为
`outputs/evisoz_execution_plan_v1_20260901_r25/plan.json`，总体仍为
`NO_GO`。因此 :codex-annotation{index="1"} 中的
`Stage0_real_dual_montage_data = NO_GO` / `Stage0_overall = NO_GO` 仍应理解为
治理和外部审计闭合尚未完成，而不是本次真实双 montage shadow 运行失败；
正式训练、Qwen SFT、alignment、非零 residual 和临床语言评价继续保持关闭。

## 18. Stage‑0 最新重放（r55）

为登记真实 shadow receipt 的 schema/validator，schema registry 已重新物化
（当前包含 52 个条目），并新增
`evisoz_real_shadow_inference_receipt_v1`。最新 clean-freeze 审计为
`outputs/evisoz_clean_freeze_audit_v1_20260901_r7/audit.json`，由于工作树仍
包含用户已有修改和大量未跟踪文件，仍为 `NO_GO`；这不是对真实 EEG 的删除或
覆盖操作。以新 registry 重放后的 gate 为
`outputs/evisoz_stage0_gate_v1_20260901_r55/gate.json`，执行计划和补件包分别为
`outputs/evisoz_execution_plan_v1_20260901_r26/plan.json` 与
`outputs/evisoz_stage0_remediation_packet_v1_20260901_r21/`，六项阻断保持不变：
clean-freeze、ELM/校准、私有治理授权、报告映射与人工 release、公开 overlap/
TUEV identity 审计。因而不能把 registry 更新误读为 Stage‑0 授权；正式训练门、
Qwen 生成门和非零 residual 仍关闭。

## 19. Stage‑0 真实数据闭合继续执行（r57）

本轮以 r55 gate 为输入，重新生成了不可授权的 clean-freeze 审计、聚合门、
执行计划和补件包：

* `outputs/evisoz_clean_freeze_audit_v1_20260901_r8/audit.json`：记录当前 Git
  快照和 8 个冻结合同；由于工作区含用户已有修改/未跟踪文件，状态仍为
  `NO_GO`，该审计不具备训练授权。
* `outputs/evisoz_stage0_gate_v1_20260901_r57/gate.json`：将已有的
  CerebraGloss development-only candidate cache（2 个事件、29 条候选）正确
  传入重放，消除了旧的 CerebraGloss artifact-missing 子阻断；未校准状态仍
  保留 `fold_local_calibration_receipts_missing`，ELM 仍缺少经过验证的
  checkpoint/预处理/暴露 manifest。整体仍为 `NO_GO`，真实 dual-montage 仍为
  `QUALIFIED_GO`，并非回退到 common17。
* `outputs/evisoz_execution_plan_v1_20260901_r27/plan.json`：状态为
  `STAGE0_NO_GO`，所有正式训练、Qwen SFT、alignment、报告语言评价和非零
  residual 均为 blocked。
* `outputs/evisoz_stage0_remediation_packet_v1_20260901_r22/`：包含 ELM/校准、
  三条未解决报告关联、43 条人工去标识审核、私有治理授权和公开 overlap/TUEV
  identity 的隐私安全补件表及 dashboard；不复制报告正文或患者标识。

本轮还落盘了实现级 Stage‑1 objective ports：
`src/evisoz/training/stage1_objectives.py` 提供 masked latent reconstruction、
motif soft-target/BCE、motif teacher KL、双 montage 一致性以及 channel/edge
dropout 一致性。所有 objective 都使用显式 mask、对 teacher stop-gradient，
并禁止将 TCP22 edge 展开为 Standard19 node label；
`compute_authorized_stage1_objective` 在计算前调用 Stage‑0 guard。它们已通过
`tests/test_evisoz_stage1_objectives.py` 的 6 个单元测试，但在 r57 `NO_GO` 下
不会启动模型、optimizer、loader 或任何正式训练。

## 20. Stage‑0 真实数据闭合继续执行（r60/r29/r24）

本轮以当前 schema registry（54 entries，registry id
`EVISOZ-SCHEMAS-28a0c1d56b0e7ee3b0296009`，registry SHA-256
`7e6f075da970eef98e09c1a151743d654824cc292011f9bdb56e36ed00a8b007`）重放
Stage‑0，并物化：

* `outputs/evisoz_clean_freeze_audit_v1_20260901_r10/audit.json`；
* `outputs/evisoz_stage0_gate_v1_20260901_r60/gate.json`；
* `outputs/evisoz_execution_plan_v1_20260901_r29/plan.json`；
* `outputs/evisoz_stage0_remediation_packet_v1_20260901_r24/` 及其中的
  `evisoz_stage0_remediation_dashboard.html`。

最新聚合门仍为 `NO_GO`（gate id `EVISOZ-STAGE0-404ad41e8d14396c8dad787a`）。真实双 montage 合同仍为
`QUALIFIED_GO`（不是 common17 回退）；当前阻断为：worktree clean-freeze、
ELM checkpoint/预处理与 fold-local calibration、私有训练治理授权、3 条报告
关联的权威确认、43 条医生报告的人工去标识和开发/锁定 release、公开 near/
partial overlap 与 TUEV eval identity 审计。补件包只生成请求和摘要，不伪造
任何外部授权、人工审核或身份凭据。

使用 r50 bound evidence、r3 Findings/claim/report 和真实 dual-montage cache
完成了全量 88-event/31-patient loader replay：
`outputs/evisoz_stage0_bound_evidence_loader_replay_v1_20260901_r53/receipt.json`。
该 receipt 明确 `real_bound_evidence_replay_only`，不打开医生 DOCX、teacher
runtime 或 Qwen，也不授予训练权限。

为使后续 Stage‑2 可复现，新增
`src/evisoz/training/residual_trainer.py`：显式 Standard19 node mask 的
residual BCE、冻结 v29 H/D 的 baseline-preservation KL，以及 guard-before-
loader 的 residual epoch wiring。`alpha>0` 只有在聚合 Stage‑0 为 `GO` 且
pipeline config 已切换为 training-enabled 时才允许；当前 r60 会在模型、
optimizer 和 loader 构造前抛出 `Stage0TrainingBlocked`。r60 下重新执行的
Stage‑1 blocked receipt 为
`outputs/evisoz_stage1_evidence_training_v1_20260901_r4/receipt.json`。
对应 residual/Stage‑1/Clinical Evidence 测试合计 27 项通过。

同时新增 `src/evisoz/training/grounding_feedback.py`：它先对报告 claim 的
evidence IDs、通道/形态/区域、certainty 做结构化 grounding 检查；只有通过
检查且 Stage‑0 为 `GO` 时，才允许一次性、非负、候选 mask 内的 report residual。
当前 gate 下该接口只会 fail-closed，不会读取医生正文或执行 Qwen 生成；对应
`tests/test_evisoz_grounding_feedback.py` 验证了错误 onset/spread 和 guard 行为。

私有报告资产的当前只读价值也已纳入运行图：`EEG_Reports` inventory 共 43 条
有效 DOCX，全部保持 `physician_authored`；40 条高置信关联、3 条等待权威映射，
自动 PHI 扫描 43/43 通过但人工 release 仍为 0。它们可以在完成人工去标识和独立
授权后用于 Qwen 文本侧适配/语言评价，不能监督 SOZ；在 release 之前只作为
不可训练的候选 artifact 保存引用。

## 21. Stage‑0 当前重放（r63/r30/r27，2026-09-01）

在上一轮实现和测试完成后，使用未修改的真实输入工件重新生成了以下不可变
回执（不覆盖历史版本）：

* `outputs/evisoz_stage0_gate_v1_20260901_r63/gate.json`：聚合状态仍为
  `NO_GO`，gate id 为 `EVISOZ-STAGE0-6fa3dad13c46ae14f6a3e48b`。
* `outputs/evisoz_execution_plan_v1_20260901_r30/plan.json`：状态为
  `STAGE0_NO_GO`，正式训练/Qwen SFT/alignment/非零 residual 仍关闭。
* `outputs/evisoz_stage0_remediation_packet_v1_20260901_r27/`：补件包与
  r63 的六个 blocker 同步，未生成任何授权或报告 release。
* `outputs/evisoz_stage0_bound_evidence_loader_replay_v1_20260901_r54/receipt.json`：
  88 个事件、31 个 linkage group 全量 bound-evidence replay。
* `outputs/evisoz_stage0_shadow_inference_smoke_v1_20260901_r13/`：3 个事件的
  structural shadow smoke；mask、claim-support 和 report-plan linkage 均为 1.0，
  仅为接口一致性指标，不是定位性能。

本次 r63 的状态分解为：`public_v29_reference=GO`、
`private_real_dual_montage=QUALIFIED_GO`、`findings_claim_graph_and_reports=QUALIFIED_GO`、
`bound_evidence_materialization=GO`、`knowledge_authority=GO`；仍阻断
`clean_freeze_audit`、ELM/校准、私有治理授权、3 条报告关联、43 条报告人工
去标识/release，以及公开 near/partial overlap 与 TUEV eval identity 审计。
因此，真实双 montage 数据合同不是 `NO_GO`，但整个 Stage‑0 仍是 `NO_GO`；
该状态不会授权 Query/Residual、Qwen、教师运行时或任何私有 label training。

## 22. Stage‑0 当前重放（r66/r31/r28，2026-09-01）

schema registry 漂移已通过既有物化脚本修复并重新绑定。随后使用当前 registry、
真实双 montage、r50 bound evidence、r3 Findings/claim/report、CerebraGloss
development-only cache，以及 3 条 unresolved 报告的显式
`operational_quarantine` exclusion receipt 重放，得到：

当前 clean-freeze 输入文件为
`outputs/evisoz_clean_freeze_audit_v1_20260901_r10`（该文件记录的状态仍为
`NO_GO`；它是审计回执而非授权票据）。

* `outputs/evisoz_stage0_gate_v1_20260901_r66/gate.json`：`NO_GO`，其中
  `private_report_linkage=GO`（3 条报告不再进入任何数据/文本 lane，但也没有被
  猜测关联）；
* `outputs/evisoz_execution_plan_v1_20260901_r31/plan.json`：
  `STAGE0_NO_GO`；
* `outputs/evisoz_stage0_remediation_packet_v1_20260901_r28/`：只包含当前
  五项阻断的补件请求，未产生训练授权；
* 最新 registry：`EVISOZ-SCHEMAS-4fddba2ef88e126cc0593bc9`。

剩余阻断为 clean-freeze、ELM 与 fold-local calibration、私有训练治理授权、
43 条报告人工去标识/release，以及公开 near/partial overlap 与 TUEV eval
identity 审计。即使报告 linkage 通过 exclusion closure，`private_field_envelopes`
仍是 evaluator-only，`private_report_text_release` 仍为 `NO_GO`，所以所有
正式 Query/Residual、Qwen SFT、alignment、非零 residual 和私有 label training
继续保持 fail-closed。

## 23. clean-worktree 当前重放（r83，2026-09-01）

本节是对前述历史回执的追加，不覆盖任何历史结论。所有命令均从
`/mnt/hd1/dyf/workspace/laptop/EviSOZ` 执行；受控的 `EEG_Seizure` 路径只作为
显式只读输入，ELM checkpoint/config 位于仓库外的
`EviSOZ_artifacts/elm_public_artifacts_v1_20260901_r1/`。

当前不可变回执为：

```text
freeze:    outputs/clean_freeze_audit_target_v1_20260901_r17.json
gate:      outputs/evisoz_stage0_gate_v1_20260901_r83/gate.json
plan:      outputs/evisoz_execution_plan_v1_20260901_r41/plan.json
remediate: outputs/evisoz_stage0_remediation_packet_v1_20260901_r10/
elm:       outputs/evisoz_teacher_artifact_discovery_v1_20260901_evisoz_artifacts_r2.json
```

`private_report_text_release` 已通过外部 SUAT authorization ref、40 条人工审核
行和 candidate/text hash 绑定（development Qwen 33 条、locked language evaluation
7 条）；对应 gate 输入必须显式使用 `--private-report-release`，不能通过修改
de-identification candidate manifest 冒充 release。该 release 只开放
physician-authored text 的相应文本用途，不能监督 SOZ 定位，也不会解除 aggregate
Stage-0 训练门。

当前 aggregate `Stage0_overall=NO_GO`，剩余阻断为私有训练治理 authority、ELM
candidate/preprocessing/exposure 与 fold-local calibration、public near/partial
overlap 和 TUEV eval identity，以及双 montage reference observability/locked-test
exposure 限制。ELM discovery r2 只表示 4 个外部文件已完整哈希并仍为
`found_unvalidated`；在获得 audited candidate cache 和校准 receipt 前，不得运行
大规模 ELM 推理、正式 DataLoader、optimizer、Qwen SFT、EEG-to-Qwen alignment 或
非零 residual。
