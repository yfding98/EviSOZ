# EEG-only 原子 Claim 事实一致性评价器 v1

日期：2026-08-22  
状态：方法、合成 evaluator 与 source-bound case materializer 已实现；后者从冻结 EvidenceGraph、record graph、deterministic render 和完整 event roster 内生构造 claims/derivations/evidence flow/固定权重并重放源绑定。2026-08-23 已补充 conclusion strict support、空间/时间蕴含、所有正 onset/SOZ 结论的 causal-leaf 审计，以及 EEG-ClaimGround 的 canonical evidence-ID exact-binding 门。该 exact-binding 当前只作 `LinkageClosure` shadow；尚未接入私有生产路由，也尚未获得双 EEG 专家独立参考评价结果。

## 1. 审计结论

仓库原有三类能力回答的是不同问题，不能互相替代：

1. `postfreeze_evaluation.py` 在报告冻结后比较 Excel 起始语义和医生通道参考，主要评价 laterality/region/unclear 一致性与 Top-k 排序；它不评价生成文本中的每个 EEG claim。
2. `multievent_soz_claim_validation.py` 验证 source firewall、实体闭合、evidence/relation/hypothesis/sentence-plan 闭合和认识状态上限；它证明一个图**内部合法**，不证明图中的信号判断对独立 EEG EvidenceGraph 是真的。
3. `multievent_report_render.py` 保证确定性 renderer 的 sentence ledger 与 claim plan 一致；它不评价漏掉了多少检测前就应纳入的显著 EEG 证据。

因此 SOZ Top-1、结构 validator 通过、claim-plan byte closure 和临床事实一致性必须作为不同终点报告。

事实一致性进一步固定为两条不可循环替代的轴：

1. `serialization fidelity`：renderer/Qwen 是否忠实表达同一 frozen claim graph；
2. `evidence correctness`：raw EEG→measurement/Finding→record hypothesis 是否被信号重放和独立专家/reference 支持。

临床语言质量是正交的第三类评价维度，不是第三条“事实轴”。100% 的 serialization fidelity 不能证明上游 EEG 判断正确；独立参考下的证据正确也不保证文字清晰。

## 2. 为什么 SOZ Top-1 不够

Top-1 仅考察一个排序终点，无法发现以下错误：

- 找对了 T7，却把晚期扩散写成最早起始；
- 侧别正确，但发作事件、mode 或相对时间绑定错误；
- 事实方向颠倒，例如将 P7 随后募集写成 P7 早于 T7；
- 把 `model_candidate` 提升成 `clinically_qualified` 或把未校准研究假设写成风险受控结论；
- 同一参考事实被重复表述三次，表面 recall 增高；
- 漏掉其他显著事件或关键反证；
- 检测器漏掉整次发作后，只在成功候选上报告很高的条件事实精度；
- 偶然命中正确通道，但 premise 使用 spread/context-only 证据，推理链无效。

极端情况下，系统可以 Top-1 正确而报告包含多个严重事实错误；也可以所有 claim 均忠实于当前证据但由于头皮信号可观测性而 Top-1 不同于医生参考。二者不能合成一个分数。

## 3. 独立输入合同

实现位于：

`src/clinical_eeg_long_recording/claim_factuality_evaluation.py`

生产接线前使用更严格的桥接层：

`src/clinical_eeg_long_recording/source_bound_factuality_case_materializer.py`

它不接受调用者自报 predicted/reference claims、stage booleans、severity 或 salience；而是从 host-validated source artifacts 重新派生，并要求每个句子及语义原子分句具有唯一 claim ownership。验证时再次输入四个带外冻结源、逐字重放 deterministic renderer，并把 EvidenceGraph observation 的 predicate/object/measurement/time/polarity/epistemic/evidence IDs 绑定到 event hash。因而同时修改文字、ledger 与自含 hash，或保留 evidence ID 却改 onset 时间，均会 fail closed。该桥接层仍只证明 serialization/provenance closure，不证明上游 Finding 的临床正确性。

评价器本身不做文件 I/O，只接受去标识化结构化输入：

```text
case
├── predicted_claims       # 待评价的结构化实现
├── reference_claims       # 冻结 EEG EvidenceGraph 原子投影
├── derivations            # conclusion <- premises + rule
├── evidence_flow          # 从检测到报告的冻结阶段 ledger
└── claim_boundary         # private/Excel/annotation/doctor/source-eval 均 false
```

每条原子 claim 明确包含：

- subject、predicate、object entity/code/measurement；
- event ID、mode ID；
- recording interval 或 delay interval 与 censoring；
- polarity、negation scope、epistemic status；
- `present/absent_with_opportunity/uncertain/not_evaluable` assertion status；
- canonical evidence IDs 与 legacy `onset_support/spread_support/contradiction/context_only` 角色；report graph v2 路线则原样保留 `ictal_pattern_qualification/onset_time_support/onset_topography_support/course_or_spread_support/counterevidence` 五角色，不做有损改名；
- 可选但对正 onset/SOZ strict support 必需的逐 evidence 时间闭包：`intrinsic_evidence_role`、`view_role`、`future_sample_access`、`onset_evidence_authorized` 与 `onset_support_eligible`；
- salience weight、severity weight 和 critical 标志；
- 对关系 claim，显式 source/target claim IDs。

当前实现没有读取私有数据、Excel、EDF annotation、医生标签、临床文本或 source-eval。合成参考 claim 只验证算法合同，不能被表述为临床 factuality 结果。

旧 v1 claim 若缺逐 evidence 时间闭包仍可解析、做语义对齐和 role-only compatibility；这只用于迁移，不等于完成 future/view provenance 审计。凡 predicate 属于正 onset/SOZ，predicted claim 必须 `authorization_complete=true` 才可获得 strict support。reference legacy gold 暂不强迫该字段，以免在 gold 迁移前阻断评价，但结果会显式保留其不完整状态。

## 4. 原子 Claim 一对一匹配

对于非关系 claim，先计算以下兼容度：

```text
predicate + claim kind
+ subject semantics
+ object/entity/measurement/evidence-role semantics
+ event/mode attribution
+ physical-time compatibility
+ polarity
+ negation scope
+ epistemic status
```

本地 `finding/claim/hypothesis` ID 只是图内主键，不作为语义；canonical evidence ID、electrode、region、laterality、event、mode 等证据或物理层级实体必须相同。数值采用冻结 absolute/relative tolerance，时间采用冻结 tolerance、interval IoU 和 censoring。所有 policy 都写入结果并保存 SHA-256。

随后使用 O(n³) Hungarian 最大权匹配，得到一对一集合 `A*`。没有采用 greedy matching；重复预测不能再次消费已经匹配的参考事实。关系 claim 在基础 endpoint 映射后第二阶段匹配，必须保持 predicate、方向、source/target 和时间相容，且两个 endpoint 均需严格支持。

主要 claim 指标为：

\[
P_{supported}=
\frac{\sum_{p\in Pred}w_p I[p\text{ strict-supported}]}{\sum_{p\in Pred}w_p}
\]

\[
R_{salient}=
\frac{\sum_{g\in Gold_{salient}}s_g I[g\text{ strict-recalled}]}{\sum_{g\in Gold_{salient}}s_g}
\]

同时输出：

- unweighted strict precision/recall/F1；
- hallucination rate 与 salient omission rate；
- relation precision/recall/F1；
- event/mode attribution、channel/region/laterality、time、polarity、negation、assertion status、epistemic 分项准确率；
- canonical evidence-ID binding 与 evidence-role grounding accuracy；
- interval IoU 与上下界误差；
- epistemic overstatement count/rate；
- unsupported/overstated critical claim count。

每个 aligned claim 另输出 `EEG-ClaimGround`：

\[
G=I_{canonical\ evidence\mbox{-}ID\ exact\ binding}\,
I_{event}\,I_{mode}\,IoU_{recording-relative\ time}\,
Jaccard_{channel/region}\,I_{authorized\ temporal\ role}.
\]

canonical evidence-ID 按非空集合做顺序无关的精确比较，并显式输出预测/参考 ID、binding availability、`canonical_evidence_id_binding_exact` 与绑定准确率。这个 exact-ID facet 只在 `renderer output ↔ 同一 frozen claim plan` 内称为 `LinkageClosure`：它证明序列化/来源闭合，不证明 raw EEG 临床事实正确。当前 evaluator 将它作为 strict-grounded 的必需工程门，因而只是 source-bound serialization shadow；独立专家可能选择语义等价但 ID 不同的 evidence span，正式 evidence-correctness 评价必须另用 interval、unit/region、value、polarity、temporal role 与 ontology relation 做一对一匹配，不能要求 exact evidence-ID。legacy claim 仍可在缺少逐 evidence 时间闭包时执行，但 legacy role-only 兼容不能豁免当前 shadow 的 canonical ID 绑定。

通道与脑区分别保留 Jaccard，并报告可用空间项的平均值；非 recording-relative claim 不伪装成 recording-time IoU。正 onset/SOZ 的最后一项只有在每个 onset-support evidence 均来自 onset-causal、future-free、onset-authorized、onset-support-eligible 路径时为 1。`context_offline`、`context_only`、`later_involvement/spread_support` 可支持演变/后续累及/终止或反证，不能支持正起始定位。

支持精度和显著召回不能互相替代：只生成一句保守正确的话可获得高 precision，但会有严重 omission；把所有可能内容都写出可能提高 recall，却降低 precision。

## 5. ChainValidity

每个推导显式保存：

```json
{
  "derivation_id": "D-REC-01",
  "conclusion_claim_id": "C-REC-PRIMARY",
  "premise_claim_ids": ["C-MODE-A"],
  "rule_id": "mode_hypothesis_to_record_hypothesis_v1",
  "weight": 2.0
}
```

v1 使用闭合 rule registry，当前覆盖：

- observation → event hypothesis；
- event hypothesis → mode hypothesis；
- mode hypothesis → record hypothesis；
- bilateral synchrony → generalized record；
- limitation → nonlocalizable/technical record；
- at least two distinct modes → multiple-mode record。

该 registry 条目只说明 evaluator 能审计一个已存在的 multiple-mode derivation，不授权上游自动生成它。若没有患者隔离、穷尽的 event→mode assignment 与逐事件 onset-field gold，producer 必须停留在 latent event-heterogeneity/discordance，不能调用该规则晋升正式 multiple-mode claim。

一个 derivation 只有同时满足以下条件才有效：

1. rule 在闭合 registry 中，且 conclusion/premise kind 与 predicate 符合 rule；
2. conclusion 自身及所有直接 premise 都在独立参考中 strict-supported；不能因前提正确而让一个错误左右侧、脑区或认识状态的结论通过；
3. conclusion 显式声称的 laterality/region/electrode 与 premise/递归叶的可比较空间语义不矛盾，显式时间必须能由物理时间叶、必要时直接 premise 继承；
4. event rule 不跨事件，mode rule 不跨 mode，多模式 rule 至少有两个不同 mode；
5. 推导图无环且每个 conclusion 只有一个显式 derivation；
6. 所有正 onset/SOZ conclusion（包括 generalized-synchronous 与 multiple-mode）递归追溯到的每个叶节点均有完整且授权的 onset-causal per-evidence binding；legacy 路线只有 `onset_support` 字符串但缺 future/view 闭包仍记为未授权，不能以 `spread_support/context_only` 创建结论；v2 路线必须有 onset-time，空间定位还必须有 onset-topography，ictal-only、course/spread、counterevidence 或新旧角色混用均不能创建正向结论；
7. generalized、nonlocalizable 等特殊结论具有规则要求的正证据叶节点。

当前实现输出兼容字段：

\[
ChainValidity=
\frac{\sum_d w_d I(d\text{ premises supported and rule allowed})}
     {\sum_d w_d}.
\]

每条无效推导保留 reason codes，如 `cross_event_mixing`、`rule_not_in_closed_registry`、`premises_not_strictly_supported`、`localizing_chain_uses_non_onset_leaf`。该分数的分母只含**预测链**，论文语义应固定为 `ChainPrecision`；`ChainValidity` 仅保留为代码/工件兼容字段。它不能发现系统通过少报推理链、关键反证或整次事件来刷高 precision。

因此还必须以独立专家/reference 预先定义的显著推理链为分母计算 `SalientChainRecall`。零输出、漏掉关键反证或漏掉整次事件在 recall 中记 0。`ChainPrecision` 与 `SalientChainRecall` 必须并列；当前合成 evaluator 尚未物化独立显著链 reference，所以不能声称双向推理事实性已经实现或验证。两项均不与 Top-1 合并；偶然命中正确通道但链无效仍是推理错误。

## 6. 端到端证据流失率

仅在“成功检测并进入 Findings 的候选事件”上计算 claim precision 会产生条件选择偏倚。v1 因此单列从完整冻结 roster 开始的显著证据 ledger：

```text
detector_recovered
→ adaptive_window_retained
→ finding_emitted
→ record_claim_retained
→ rendered_claim_retained
```

每个 evidence item 有独立 salience weight，阶段状态必须单调非增；后级为 true 而前级为 false 时 fail closed。输出每阶段 weighted recall、相邻阶段 conditional loss 以及：

\[
EvidenceFlowLoss=1-
\frac{\sum_e s_e I(e\text{ retained in supported rendered claim})}
     {\sum_e s_e}.
\]

当前实现将 stage ledger 标记为 `frozen_external_stage_presence_not_inferred_from_prose`。真实实验必须由 detector roster、event bundle、Finding graph、record graph 和 render ledger 的哈希闭包自动生成；在该连接完成前，不能把人工填入的 stage flags 当成端到端性能证据。

## 7. 患者级聚合合同

所有 record 先在患者内合并 sufficient statistics，再计算一名患者的 precision/recall/relation/time/epistemic/chain/flow 指标；随后对患者等权 macro：

```text
records/events → within-patient statistics → one patient metric
                                        ↓
                        equal-weight patient macro
```

95% percentile CI 只对 patient ID 有放回重采样；不会独立重采样 record、event 或 claim。结果显式记录：

- `sampling_unit = patient`；
- `records_combined_within_patient_before_macro = true`；
- `events_never_treated_as_independent_samples = true`；
- missing metric 不作为 0；
- micro pooling 不是主估计量。

这避免“发作次数多的患者”通过贡献更多 event/claim 获得更高隐式权重。

聚合入口现会先验证每份 case-evaluation artifact 的闭合顶层 schema、状态、病例/患者/记录 ID、canonical policy 与 policy SHA-256、固定 sufficient-statistics 字段，以及去除 `artifact_sha256` 后的内容寻址哈希；修改统计量却保留旧哈希会被拒绝。这是聚合前的完整性门，不能替代从带外冻结 source case 重新执行 evaluator；能够同时改写内容与自含哈希的攻击仍应由 source-bound replay 防御。

## 8. 推荐论文 Dashboard

本模块只覆盖 L5 和端到端证据流的一部分，最终论文仍应并列报告：

| 层 | 必须单列的主要指标 |
|---|---|
| L0 detection | patient-macro event sensitivity、FA/h、onset/offset error、merge/split |
| L1 firewall | forbidden input count、label perturbation invariance、receipt closure |
| L2 measurement | 波形重放误差、单位/时间/hash mismatch、参考稳定性 |
| L3 Finding | per-term precision/coverage、qualification overreach、not-evaluable calibration |
| L4 SOZ | laterality/region/phenotype 分层；穷尽 gold 下的 set/PR/calibration；incomplete-positive 参考下仅 annotated-positive rank、Hit/recall@k、MRR、positive mass 与 PU/缺失标签敏感性；onset-spread confusion |
| L5 report factuality | `LinkageClosure`、独立参考下的 supported precision/salient recall、relation/time/epistemic、`ChainPrecision` 与 `SalientChainRecall`、critical errors |
| 跨层 | evidence-flow recall/loss，从完整 roster 开始 |
| L6 post-hoc descriptive / fresh confirmatory | 已历史开标私有 141 仅报 locked descriptive Excel/channel agreement；fresh patient/site 或从未打开 holdout 才报 confirmatory 双专家 major/minor error、修改时间 |

Top-1、BLEU、ROUGE、BERTScore 均为单独的次要终点，不能代替该 dashboard，也不应被平均成一个“综合可信度分数”。

### 8.1 指标所需参考来源必须分开

| 指标族 | 仅靠机器证据闭环可评价 | 必须新增什么参考 | 可作出的主张 |
|---|---|---|---|
| source/evidence/hash closure、forbidden-input count、imputed/QC-fail evidence count | 是 | 无 | 工程合同忠实性，不是临床正确性 |
| 数值/时间重放、单位、滤波/参考/通道绑定 | 是 | canonical EEG + 冻结数值 kernel | signal-grounded measurement fidelity |
| claim-plan→确定性文本 exact coverage、Qwen 输出→claim ledger closure | 是 | 冻结 claim graph；若评价 Qwen surface，需独立结构化解析或人工核验 | graph-to-text faithfulness |
| relation direction、event/mode attribution、polarity/negation/epistemic scope | 是，前提是参考 claim graph 已冻结 | EEG EvidenceGraph 原子投影 | 结构化事实忠实性 |
| predicted-chain `ChainValidity` compatibility field（论文称 `ChainPrecision`）、onset-vs-spread misuse、cross-event/mode mixing | 是，前提是 rule registry 与 evidence role 已冻结 | source-development 冻结规则/角色合同；完整性另需独立显著链 reference | 预测链 precision；没有 `SalientChainRecall` 时不能声称推理链完整性 |
| detector→window→Finding→record→render evidence-flow loss | 是，前提是完整 recording/event roster 与各阶段 hash ledger | 不能只用成功候选；需全 roster | 端到端证据保留率 |
| uncertainty/risk calibration、ECE/AURC/coverage-risk | 否 | patient-disjoint、endpoint-matched 冻结参考 | 选择性风险/校准 |
| Finding 临床术语正确性与 salience/severity 权重 | 否 | 双 EEG 专家独立原子标注和 adjudication | 临床 Finding factuality |
| SOZ phenotype/laterality/region/channel、onset-spread error | 否 | 穷尽双专家 gold，或明确标为 incomplete-positive 的医生/数据集参考；onset 与 spread 分开 | 只在穷尽 gold 下支持完整 set/PR/calibration；不完整阳性只支持 annotated-positive ranking/agreement |
| major/minor clinical error、可用性、修改时间 | 否 | 盲法双专家 reader study | 临床审阅安全性/效用 |
| BLEU/ROUGE/METEOR/BERTScore | 否 | 同一 EEG 的完整、去标识、配对医生报告 | 文字相似度，不能证明事实正确 |
| 临床风格、流畅度、可读性 | 自动指标只能探索 | 盲法医生评分或冻结语言 rubric | language quality，不是 SOZ accuracy |

Excel 起始字段只能在全部模型、阈值、prompt、模板和评价映射冻结后评价 laterality/region/unclear 一致性；它不是完整报告，不能提供 claim recall、BLEU 或逐句临床 factuality。

### 8.2 专家参考抽样、mode gold 与队列身份

上游 Findings/evidence-correctness 的专家参考采用双采样框：candidate-blind 的 `patient→full recording→time×channel opportunity` 随机核心（包含 detector-negative 区间）、患者级完整记录/事件 roster 复核子集，以及纳入全部模型阳性的主动富集补充集。三框保存 selection probability；主动集单独报告或使用预注册 design-based 权重，不能与随机核心直接混池估计总体性能。

若没有患者隔离、穷尽的 event→mode assignment 与逐事件 onset-field 双专家 gold，记录级 mode 只能评价重采样、参考扰动、事件复制/顺序和 leave-one-event-out stability；不得报告 mode purity/accuracy、校准的 multiple-mode probability，也不得把 latent cluster 自动晋升为 `multiple_scalp_onset_modes`。

现有私有 141、Excel 起始摘要和医生显著/扩散通道已在既往迭代中历史开标，因此无论本轮模型、阈值、prompt 和模板是否锁定，其身份都只能是 `post-hoc locked descriptive audit`，不得称 independent external/confirmatory validation。正式确认性终点必须来自 fresh patient/site，或从未被设计迭代、调参、模型选择和错误分析打开的预注册患者级 holdout。医生显著通道是 incomplete-positive primary set，扩散通道是独立且也可能不完整的 graded relevance；现有参考只允许 annotated-positive rank、Hit/recall@k、MRR、positive probability mass 与 PU/缺失标签敏感性分析。PR-AUC、set Jaccard/F1、mAP、Brier/ECE 和 prediction-set coverage 只在相应层级 reference 穷尽且正负/不确定状态可定义时计算。

## 9. 已实现的合成验收与剩余缺口

合成测试覆盖：

- local opaque ID 不同但物理语义一致时可匹配；
- duplicate claim 只能一对一匹配一次；
- relation 方向反转独立计错；
- 时间不相容和认识状态过度提升独立计错；
- strict-matched 的 late-spread premise 仍不能产生 localizing chain；
- 错误左右侧/脑区 conclusion、无法从叶证据继承的 conclusion 时间、generalized 结论使用 offline/context 叶，以及 multiple-mode 重复同一 mode 均被 ChainValidity 拒绝；
- 未知推理规则 fail closed；
- evidence-flow 后级复活被拒绝；
- Excel/private/annotation/doctor/source-eval boundary 不能改为 true；
- 两名患者、三份记录时先患者内合并，再做患者等权 macro 和 patient bootstrap；
- 混合 policy hash、重复 patient-record 被拒绝。
- EEG-ClaimGround 的 event/mode、时间 IoU、channel/region Jaccard 分项；
- context-offline、future-dependent、late-spread、onset-ineligible raw dependency 对正 onset/SOZ strict support 逐项归零；
- 同一 offline evidence 对非 onset 的 spread/evolution claim 仍可合法使用；
- legacy role-only onset claim 可解析/对齐，但 strict-supported 与 localizing ChainValidity 均失败关闭。

source-bound bridge 另有 `10` 项专项测试，覆盖四源内生物化、全句/原子分句唯一 ownership、完整 roster 五阶段 flow、固定权重、正 onset 时间权限、同步 text+ledger+self-hash 篡改，以及保留 evidence ID 后改变原子观察对象/时间的攻击；其与 claim evaluator、multievent graph 和 renderer 的交叉回归为 `53 passed`。截至 2026-08-22，完整 `tests/test_clinical_eeg_*.py` 为 `810 passed, 2 skipped`。这些仍是软件合同测试，不是临床事实性结果。

尚未完成、不得声称完成的部分：

1. 真实 report surface 的独立 claim extraction/双专家 adjudication；当前确定性 renderer ledger 能证明 claim coverage，不能证明临床正确性。
2. detector→Finding→record graph→render 的生产哈希自动生成 evidence-flow ledger。
3. 双专家冻结 salience/severity 权重与术语、数值、时间容差；当前权重/容差仅为 v1 方法默认值。
4. 患者级 paired clinical reference、major/minor error 和修改时间评价。
5. 跨 montage/reference 的临床等价实体映射，以及 10–20 邻接图距离的 graded spatial error。
6. 真实 patient-disjoint source-development 校准和完全冻结后的 source-eval/private 评价。
7. 当前 `clinical_eeg_multievent_soz_report_v1` 虽有 observation→core claim 的 support relation closure，但没有单独持久化 event→mode→record 的 derivation receipt；生产接入时必须由 aggregator 输出本评价器所需的 conclusion/premise/rule DAG，不能从最终文字反向猜测。
8. 旧 block-level Qwen surface validator 主要核对 fact IDs、通道、数字和有限区域词，不能阻止“频率增快→减慢”或“尖慢波复合→棘波”这类保留 fact ID 的谓词/形态改写。主路线必须收缩为 typed sentence plan + 预授权 frame，或使用确定性 lexicalizer；每个非版式原子分句都要纳入 claim ledger。
9. 持久化 multievent render 的自含 hash 只能发现未同步篡改；若攻击者同时改文字、ledger 和自含 hash，它不会重放冻结 source claim graph。生产 validator 必须接收带外冻结 manifest/source graph，逐句重新 lexicalize 并验证 byte/slot closure。
10. source-bound factuality case materializer 已能从冻结 EvidenceGraph projection、record hypothesis graph、deterministic sentence ledger 和 complete event roster 自动物化 case，并内置固定 severity/salience/flow policy；剩余缺口是让私有 v2 production route 直接发布这四类带外源 artifact 与内容哈希。当前合成/fixture 物化结果仍不能自动代表真实报告或临床正确性。
11. Excel `SZ` 起始字段是 recording-level 事后总结，不能复制绑定到每个 detector event；冻结后评价必须统一为 recording conclusion/ranking，再以 patient 为 bootstrap 单位。

当前 141 份 legacy 私有报告进一步说明该缺口：已登记 claim grounding 为 `500/500`，但 claim surface 只覆盖正文字符均值 `12.257%`、中位数 `13.428%`、范围 `0--28.272%`；同一 EEG 的完整配对医生报告为 0，空间可比较字段也为 0。因此这些数字只能支持“少量已登记引用闭合”，不能支持全文无幻觉、SOZ 事实一致或临床语言质量已验证。

因此当前结论仅为“评价合同与合成实现可执行”，不是“生成报告已通过临床事实一致性验证”。
