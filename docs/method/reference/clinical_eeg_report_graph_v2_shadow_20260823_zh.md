# clinical EEG report graph v2：无损 EEG-only shadow 合同

**状态：** 公共/合成 shadow schema、materializer、validator、五角色 factuality bridge，以及 mode-aware candidate replay→确定性中文 claim-locked 预览与反事实测试已实现；未接私有数据、受信/校准后的正式 mode-aware MIL、真实 waveform PNG、Qwen lexical overlay、生产或临床使用路径。

## 1. 为什么不能继续复用 legacy v1

`clinical_eeg_multievent_soz_report_v1` 仍作为冻结兼容基线保留，但它不能无损表达：

- `present / absent_with_opportunity / uncertain / not_evaluable` 四态；
- `measured / model_candidate / report_eligible_automated` assertion level；
- 每条证据的 raw-sample dependency 与 future-free 时间权限；
- event v3 的 occurrence、burden、competing hypotheses 和 event outcome；
- ictal pattern qualification、onset time、onset topography、course/spread、counterevidence 五类互不替代的权限；
- event→mode→record 的来源路径；
- 由 causal onset leaf 构造的空间分辨率证明；
- claim 与 waveform panel 的双向闭合。

现有 v3 downstream sidecar 适合兼容展示，但仍是摘要投影。因此 report graph v2 把“精确重放”设为第一原则，而不是继续向 legacy predicate 中压缩新语义。

## 2. 结构

```text
ordered event_eeg_findings_v3 sources（逐个重新验证）
                    │
                    ├── exact source embedding + source roster hash
                    ├── four-state Finding nodes + exact raw dependencies
                    ├── five named permission-role edges
                    ├── competing hypotheses + event outcome snapshots
                    ├── constructive spatial-resolution receipts
                    ├── event → structural mode → record DAG
                    ├── typed claims
                    └── claim/evidence-double-closed waveform panels
```

整个 payload 可由嵌入的、独立重新验证后的有序 v3 roster 确定性重建。validator 不信任嵌入 source 的自报 hash；它重新运行 v3 validator、重新计算所有规范化节点、边、receipt、DAG、claim、panel 和总图 seal，并要求 canonical JSON 完全一致。host 还可传入原始冻结 v3 roster，要求逐对象、逐顺序精确重放。

### EEG-only 入口防火墙

graph 不能把上游迁移对象中的 `unknown` 当作“未使用”。每个 source 必须对九项 `provenance.inference_exclusions`（EDF annotations、Excel、医生标签、临床文本、患者资料、视频、ECG/EMG/EOG、睡眠分期、诱发信息）逐项提供字面量 `false`；任一字段为 `unknown`、`true`、缺失或多余，materializer 均拒绝，不能生成 Finding、claim 或 EEG-only 声明。零事件记录也不例外：`record_context.source_inference_exclusions` 必须提供同一组显式 `false` receipt，不能因为没有事件而绕过入口防火墙。

## 3. 五类 evidence permission

report graph 不把一个笼统的 `supports_soz` 边拆成五个默认授权。每条权限边必须包含 source JSON Pointer、source object hash 和命名 derivation rule：

| role | 唯一允许的 v2 shadow 来源 |
|---|---|
| ictal pattern qualification | v3 中显式 qualified `event_qualification` 及其 supporting evidence |
| onset time support | `present + onset_eligible` Finding，且所有 raw dependencies 均为 future-free `onset_causal` |
| onset topography support | onset-time 条件通过，且候选空间分辨率有 constructive receipt |
| course or spread support | 显式 `early_context` 或 `later_involvement` 的 present Finding；不得反向授权 onset |
| counterevidence | v3 中显式 `contradicts` relation 或 competing hypothesis 的 contradictory evidence |

若 v3 没有表达某一角色，roster 写 `not_expressed_by_source`；这不是阴性证据，更不是默认授权。尤其不能因为 offline context 看起来像一个完整发作，就自动创建 retrospective ictal-membership edge。

## 4. 空间分辨率

事件级正空间 claim 必须引用 constructive spatial receipt。目前 shadow 只允许保守规则：

- 同类型、同实体 identity；
- 单个标准电极向其较粗 region/laterality 回退；
- 同侧双极边可支持 laterality，但不能拆成任一端点电极，也不能自动支持一个 cerebral region；
- 带侧别的 region 可以向 laterality 回退。

若源 relation 声称的分辨率不能从 future-free onset leaves 构造，report graph 不生成对应 receipt、topography permission edge 或空间 claim。源 hypothesis 仍原样保留用于审计，但不能被 lexicalizer 当成已授权空间结论。

## 5. event→mode→record

当前 exact-source signature grouping 只是 structural partition，schema 中固定标记为：

```text
exact_eeg_source_signature_partition_not_calibrated_clinical_mode_inference
```

event node 原样保存 source scalp-onset hypothesis、event outcome、权限边和空间 receipt。由于正式 sealed hierarchical mode-aware MIL receipt 尚未接入，mode 与 record node 必须输出：

```text
conclusion_authorization = not_authorized_missing_mode_aware_mil_receipt
phenotype = null
selected_resolution = null
ranked_candidates = []
authorization_receipt_id = null
```

相应 mode/record claim 可沿 DAG 回放到全部 exact v3 leaves 和 named roles，但不得被渲染成记录级 SOZ impression。后续接入正式 MIL 时，应新增一个显式 receipt adapter，并保留当前 fail-closed branch 作为拒绝路径；不能直接覆盖 structural node 的语义。

另有 `mode_aware_claim_locked_report_shadow_v1` 在不改写上述正式授权面的前提下，提供独立的公共/合成审计预览：它重放未受信的 mode-aware MIL candidate，将每个 event Finding 无损投影为原子 claim，并按 mode 输出完整的 channel、region、laterality 与 phenotype probability-like ranking。多 mode 时明确禁止计算一个跨 mode 平均 SOZ 排序；record summary 只能陈述模式并存、各模式候选及其来源。该预览固定标记 `formal_clinical_claim_authorized=false`，不能反向填充 report graph 或 MIL bridge 中仍为空的 `authorized_claim_overlay`、`renderer_projection` 和 `qwen_lexicalization_slots`，也不能作为校准或临床正确性的证据。

## 6. 四态与文字边界

- `present + measured` 可形成测量事实 claim；
- `present + model_candidate` 只可形成带候选语气的研究 claim；
- `absent_with_opportunity` 只有在原始 Finding 为 `report_eligible_automated`，且完整 qualified term-decision、capability、sensitivity 与 opportunity 合同已由 v3 validator 验证后，才可形成显式阴性文字；
- 较低 assertion level 的 `absent_with_opportunity` 只保留为 structured evidence，不生成正文本阴性；
- `uncertain` 与 `not_evaluable` 不得折叠为 absent。

Qwen 的目标权限仍只允许 closed claim-graph lexicalization，不接触原始 EEG 或外部资料，不得改变否定、不确定性或空间分辨率；当前 shadow 仅物化不可变 lexical slot 与越权拒绝合同，尚未执行 Qwen overlay。确定性中文预览是 canonical artifact；未来 Qwen 超时、失败或越权时也必须保留同一确定性输出，生成失败不能取消报告 artifact。

## 7. waveform panel 双闭合

每个 panel 同时绑定：

- frozen claim ID；
- finding evidence ID；
- waveform evidence ID；
- event ID、物理区间、unit IDs、view role 与 raw dependency ID；
- source JSON Pointer 与对象 hash。

`claim_evidence_bindings` 显式保存 claim→Finding 配对，不能依赖两个并行数组的偶然顺序。panel 不得创建新的 Finding 或空间结论。

当前已闭合的是 waveform **panel metadata index**：可从句子/claim 追溯到 event、时间、unit、raw dependency 和 source hash。尚未生成或验证 PNG 像素、显示滤波、gain、通道顺序与 image SHA，因此不能声称“claim-bound 波形图已经完成”。

## 8. 对事实一致性评价的贡献

report graph v2 支持把“报告是否事实一致”分为至少三层，而不是只看 SOZ Top-1：

1. **字段级重放：** 四态、assertion、term、时间、raw dependency、competing hypothesis 与 outcome 是否逐字段等于冻结 v3；
2. **引用级闭合：** claim、permission edge、空间 receipt 与 panel 的 source pointer/hash、event ownership、negation、uncertainty 和 temporal permission 是否闭合；
3. **跨事件/跨层级链条：** mode/record claim 是否沿 DAG 回到 exact event leaves 与命名角色，是否出现 late-spread→onset、lead→endpoint electrode、无 receipt 升分辨率或跨 mode 拼接证据。

语言 lexicalization 仍应另算 supported-claim precision、salient-claim recall、relation/negation/uncertainty fidelity 与 major-error-free rate，不能用 BLEU 代替证据链正确性。

## 9. 实现与验证

- schema：`schemas/clinical_eeg_multievent_soz_report_graph_v2.schema.json`
- materializer/validator：`src/clinical_eeg_long_recording/multievent_soz_report_graph_v2.py`
- 五角色 claim factuality bridge：`src/clinical_eeg_long_recording/report_graph_v2_claim_factuality_bridge.py`
- mode-aware claim-locked 中文预览：`src/clinical_eeg_long_recording/mode_aware_claim_locked_report_shadow_v1.py`
- tests：`tests/test_clinical_eeg_multievent_soz_report_graph_v2.py`
- bridge tests：`tests/test_clinical_eeg_report_graph_v2_claim_factuality_bridge.py`
- preview tests：`tests/test_clinical_eeg_mode_aware_claim_locked_report_shadow_v1.py`

定向测试覆盖 exact ordered source replay、全图 canonical byte round-trip、embedded v3 重验证、四态/absence surface gate、五权限 source binding、constructive spatial receipt、claim/panel 双闭合、typed DAG、zero-event typed outcome、EEG-only firewall 和生产路由关闭。

2026-08-23 新增 bridge 将五类 permission 原样保留到 atomic claim case，不使用有损四角色映射；空间起始结论必须同时具备 future-free onset-time 与 constructive onset-topography facet。ictal-only、course/spread、counterevidence、新旧角色混用和跨事件同名 evidence 均失败关闭，`report_eligible_automated` 仅投影为 `model_candidate`。随后新增的 preview 已闭合公共/合成数据上的 atomic claim→sentence owner→确定性中文 text、event→mode→record provenance 和 waveform metadata index；双事件/双 mode、相反侧别、数值/时间/空间/四态完整复制及 Qwen/跨事件篡改反事实均已覆盖。它仍是未校准、非临床的 candidate preview，不等于正式 renderer、真实图像、Qwen overlay 或生产授权。
