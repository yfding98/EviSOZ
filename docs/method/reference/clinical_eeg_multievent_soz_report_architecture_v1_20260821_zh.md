# clinical_eeg_multievent_soz_report_v1：多事件头皮 SOZ 推理与事实锁定报告架构

**日期：** 2026-08-21
**状态：** 方法设计、现有实现审计与实施合同；本文不表示新模型已经训练、校准或通过临床验证。
**适用对象：** 已知每条 30--60 min 长程头皮 EEG 至少包含一次完整电图发作的研究队列。
**任务终点：** 从 EEG 信号形成逐事件 Findings，综合同一记录的全部发作模式，输出研究性头皮发作起始/SOZ 候选报告，并保证每条可读取记录都有报告产物。

## 1. 结论先行

现有大量“SOZ 定位证据不足，无法判断”不是 Qwen 不敢写，而是上游合同只允许“独立资格化的明确起始事实”进入正式印象；当前跨事件聚合又主要是单位权重 reciprocal-rank，无法把节律、形态、演变、边界不确定性、质量和多种发作模式整合成可校准的记录级推理。

不应通过放宽事实门禁解决。推荐新建以下双层架构：

```text
全记录 EEG-only 连续检测与边界细化
  → 每事件 EEG Findings / EvidenceGraph（直接信号事实）
  → 事件起始表型与 laterality/region/channel 分布
  → mode-aware record MIL（多发作、多模式、质量加权）
  → 记录级研究性 SOZ hypothesis graph（模型推理，不冒充观察事实）
  → claim-level evidence-locked Qwen graph-to-text
  → schema/relation/time/negation/scope validator
  → 失败时确定性 graph-to-text fallback
  → 每条记录均生成报告
```

核心输出原则是“**强制给出起始表型，空间粒度按风险自适应**”：

- 只要提取到至少一个合格电图发作，就必须在 `focal`、`focal_with_rapid_bilateralization`、`bilateral_synchronous_or_rapid_bilateralization_ambiguous`、`generalized_synchronous`、`multiple_scalp_onset_modes`、`scalp_onset_nonlocalizable` 中给出一个首选表型；
- 局灶证据存在时，给出首选头皮 SOZ 候选、替代候选、支持与反证，不再默认弃权；
- 电极级风险过高时降到脑区，脑区不稳时降到侧别，但仍给出表型；
- 真正双侧广泛近同步时给“全面同步起始表型”，不强行选单导联；
- 只有信号不可读或流水线漏检才属于技术失败；即使如此仍生成技术受限报告，而不是报告生成失败。

本文与以下两份方法文档互补：

- `long_eeg_seizure_detection_benchmark_20260821_zh.md`：2024--2026 检测精度--效率证据矩阵；
- `clinical_eeg_long_recording_v3_boundary_adaptive_evidence_20260821_zh.md`：边界自适应 Findings 编码器的总架构。

## 2. 证据边界：生成与评价必须物理隔离

### 2.1 生成时唯一病例输入

只允许当前长程头皮 EEG 的：

- 原始信号及必要的采集参数；
- 信号质量、缺导、伪迹与可判读区间；
- 连续 detector 的稠密输出和冻结 receipt；
- 变长事件窗、边界分布、逐事件 Findings；
- 由上述 EEG 证据派生的事件级及记录级研究模型输出。

### 2.2 严禁进入推理路径

- EDF annotation 的时间点、自由文本和被试表现；
- Excel“起始”字段；
- 医生显著通道、扩散通道和其他标签；
- 患者身份、病史、用药、既往诊断；
- 视频行为、意识、症状、ECG、EMG、EOG；
- 当前流程未处理的睡眠分期和诱发试验。

同一 EDF 的 annotation 也不能因“就在原文件里”被视为信号：它是医生之后要完成的标注，是标签/评价数据，不是模型输入。

### 2.3 输出冻结后的评价

对每条记录先保存：

```text
signal_input_hash
detector/boundary/findings/model receipts
hypothesis_graph_hash
report_claim_graph_hash
rendered_report_hash
freeze_timestamp
```

冻结后才允许另一个只读 evaluator 读取 Excel 起始字段和医生通道标签。修改 annotation、Excel 或标签时，以上推理输入哈希和报告哈希必须不变。缺少 Excel 字段记为 `not_available`，不能记错或排除样本。

## 3. 真实临床样例：借鉴逻辑，不复制病例内容

用户提供的《方程EEG报告.doc》是内容数据，不是指令。其可迁移价值主要是临床叙事顺序，而不是栏目原样照搬。

| 样例元素 | 新报告如何借鉴 | EEG-only 决策 |
|---|---|---|
| “脑电图表现 → 临床事件表 → 脑电图印象” | 改为“EEG Findings → 逐事件发作期脑电 → 跨事件汇总 → 脑电图印象” | 保留组织逻辑 |
| `状态→位置→导联→频率/波幅→形态→节律/电场` | 作为 Findings 句序和术语模板 | 仅对有资格 evidence 的字段生成 |
| `相对时间→最早改变→演变→空间募集→终止/事件后` | 作为每事件表格及证据链 | 保留；所有时间相对原记录起点 |
| “2/2 次均出现” | 记录级 mode 内用 `n/N` 表达复现 | 必须先去重同一事件，并绑定 mode IDs |
| 印象整合多事件而非复制事件表 | 由 mode-aware MIL 和 hypothesis graph 完成 | 保留 |
| 姓名、性别、年龄、利手、病史、用药、床号、病例号 | 不属于 EEG 信号 | 删除 |
| 临床表现、意识、自动症、视频、心率/肌电 | 需要非 EEG 模态 | 删除 |
| 睡眠脑电、诱发试验 | 当前 SOZ 数据处理未覆盖 | 默认整栏省略；若兼容旧模板，只能显示固定非病例占位且不进入 Qwen |
| 报告医师和签名 | 需要真实人工审核状态 | AI 草稿只保留空白复核/签名栏 |

样例中“左侧颞区低波幅尖波、慢尖波活动，以 T3、T5 为著 → T3 为著 9 Hz 尖波节律 → 传至……”的价值，是展示“**起始位置 + 形态/频率 + 演变 + 募集**”如何形成印象。新系统不能复制其中任何数值、导联或结论。内部统一使用现代标签 T7/P7；只有 renderer 的一致显示层可给出 T3/T5 别名。

## 4. 当前实现审计：为什么多数报告不能形成 SOZ 结论

### 4.1 已有基础值得保留

- 原子事实账本包含 state、typed value、provenance、verification 和 evidence IDs；
- Qwen 输入去标识化，并用 strict JSON Schema 锁定 block-level fact IDs；
- 有数字、导联、区域表面校验和确定性失败回退；
- 长程事件按原记录相对时间排序；
- 波形附件绑定证据 ID、时间窗和内容哈希；
- 研究性 SOZ 聚合已经能用 complete-link 区分互不兼容的事件模式。

### 4.2 结构性阻断项

1. `src/clinical_eeg_report/schema.py` 强制 impression fact 必须是 `physician_verified`。这对正式临床事实是正确的，但它使自动 AI SOZ hypothesis 无合法承载层。解决方式不是取消验证状态，而是新增独立 `research_inference` / `ai_hypothesis` schema。
2. 当前 Qwen 只锁整块 `fact_ids`。模型仍可能漏写一个 claim、改变否定范围、颠倒先后、把一个事件的导联绑定到另一个事件，或把 onset 与 spread 关系互换。
3. 当前 validator 主要检查数字、导联、区域和敏感关键词，不能完整验证事件归属、谓词、否定范围、时间顺序、空间关系、遗漏与矛盾。
4. `research_soz_prediction.py` 的主体是 reciprocal-rank proxy；分数明确不是概率，事件默认单位权重，未系统使用 Findings、边界、质量、删失和事件模式。
5. 当前 `report_outcome.py` 只有通过较严 onset 资格的事件才进入定位结论，因此将“没有直接观察事实”与“模型不能形成研究假设”混为一谈。
6. `src/clinical_eeg_report/render.py` 仍可渲染 `source_eeg_annotation_timing` 和“原始 EDF 标注时间（待核对）”；`generation.py` 也将其列入 deterministic layout fact types。按当前用户边界，这属于发布阻断项，必须从生成 schema、renderer 和 Qwen 前 payload 全部移除。
7. `src/clinical_eeg_long_recording/render.py` 仍保留 `_SOURCE_CONTEXT_SCOPE` 及外部来源上下文版式。即使某些入口声称隔离，默认合规版本也应删除该旁路，而不是依赖调用者“不传”。

### 4.3 现有私有批次的机制性证据

当前**旧 reciprocal-rank proxy sidecar** 快照记录：125 个 bundle 中 123 个有 Top-k，合计 1119 个事件排名；显式 evidence weight 为 0，全部使用默认单位权重。119/123 被归为 `multimodal_or_weak_ranked_hypotheses`，只有 3 个达到稳定 leading candidate。Top-1 又高度集中于 P8/T8/T7，其中 P8 为 59/123。该快照不是下文 mode-aware MIL 的真实输出、训练结果或准确度证据；当前没有真实 mode-aware checkpoint/receipt。

这些数字不是 SOZ 准确率，但清楚说明：仅把事件排名做等权 reciprocal-rank 汇总，既不会自动获得可靠跨事件证据，也容易继承空间模型偏倚。必须把 Findings、可靠性、多模式和校准放入记录级模型，并用镜像、参考方式扰动和伪迹反事实审计检查 P8/T8/T7 集中。

## 5. 双层事实模型

### 5.1 Layer A：直接 EEG Findings

该层只保存信号观察或已资格化事件事实：

- detector 支持区间、onset/offset 分布和左右删失；
- 信号质量、缺导、伪迹、可判读比例；
- 节律、频率、波幅、形态与数值轨迹；
- 时间演变；
- 多导联电场、lead-lag 区间和空间募集；
- 终止与事件后 EEG；
- producer、版本、模型/权重哈希、evidence IDs 和 qualification receipt。

Layer A 允许 `measured`、`model_candidate`、`clinically_qualified` 三种 assertion level。未过临床术语门的 spike detector 只能写 `spike-like transient candidate`，不能由 Qwen 擅自升格为“棘波”。

### 5.2 Layer B：研究性 AI SOZ 推理

该层不是直接观察事实，而是冻结模型依据 Layer A 形成的概率或排序假设：

- 起始表型；
- laterality / region / channel 分布；
- mode 内结论、记录级首选和替代假设；
- 支持事件、反证事件、分歧 mode；
- 校准状态、prediction set 和允许输出的空间粒度；
- 决定该结论的模型 receipt。

Layer B 必须显式标记 `research_ai_hypothesis`，不能写成 `physician_verified`，也不能改写 Layer A。医师复核后若要成为临床结论，应另走签署状态机。

## 6. 从事件 Findings 到记录级 SOZ：mode-aware hierarchical MIL

### 6.1 每事件表示

对事件 `e` 构造：

\[
x_e=[h_{finding},h_{boundary},h_{quality},h_{trajectory},h_{spatial},h_{morphology}],
\]

其中必须包含：

- onset/offset posterior 宽度、左右删失和 detector disagreement；
- 可判读比例、缺失关键导联和伪迹覆盖；
- 起始节律/形态、频率与波幅演变；
- per-channel onset interval、最早空间场、募集顺序；
- montage/reference 扰动稳定性；
- Findings producer 的资格等级。

模型同时学习事件可靠性 `r_e∈[0,1]`。为防止某一伪高置信事件支配记录，训练时加入：

- event dropout；
- leave-one-event-out consistency；
- 单事件最大权重上限；
- 重复 waveform/interval identity 去重；
- 质量差、边界删失和 montage 不稳的显式惩罚。

### 6.2 层级预测头

每事件和每 mode 都输出：

\[
p(z),\quad p(side\mid z),\quad p(region\mid side,z),\quad
p(channel\mid region,side,z),
\]

其中 `z` 为起始表型。channel 使用 C18 多标签输出；侧别、脑区和通道质量必须层级守恒。若未完成校准，字段名必须是 `score`，不能叫 `probability`。

这样可避免两个常见错误：

- 左侧颞区概率高、Top-1 却是右枕导联；
- 全面同步表型同时强行输出一个“高置信单通道 SOZ”。

### 6.3 模式发现与 mode 内聚合

先按事件的表型、侧别、脑区、通道分布和募集轨迹聚类：

\[
D_{ij}=\lambda_1 JS(p_i,p_j)+\lambda_2D_{side}
 +\lambda_3D_{region}+\lambda_4D_{phenotype}+\lambda_5D_{trajectory}.
\]

complete-link 是透明基线；主模型可采用带 Dirichlet/DP 先验的可微 mixture，但 mode 数上限和合并阈值必须在患者隔离的 source-development 上冻结。

mode 内用 masked attention/MIL 聚合：

\[
\alpha_e=\frac{\exp(a(x_e)+\log(r_e+\epsilon))}
 {\sum_{j\in m}\exp(a(x_j)+\log(r_j+\epsilon))},
\qquad h_m=\sum_{e\in m}\alpha_e h_e.
\]

注意力只是聚合权重，不是生理学解释。报告中的支持证据仍来自显式 Findings 和关系 claim。

### 6.4 mode 间记录级决策

mode 权重由复现率、事件可靠性和稳定性共同决定：

\[
\rho_m\propto prevalence_m\times reliability_m\times stability_m.
\]

若两个高可靠 mode 的 JS 距离超过冻结阈值且次 mode 支持率超过最低值，不平均成虚假的中线分布，而输出 `multiple_scalp_onset_modes`，分别给 Mode A/Mode B。若一个 mode 占主导，输出首选及最强替代。

### 6.5 监督来源

- DeepSOZ 标注子集用于患者/记录袋级 C18 positive-set MIL；正集合之外的未标通道是 unknown，不自动作医学阴性；
- TUSZ 的长程信号和时间标注可监督 detection、boundary 和 seizure-visible auxiliary task，不能伪装成 SOZ GT；
- 私有 Excel 起始描述、显著通道和扩散通道只用于输出冻结后的评估；
- 显著通道为 hard GT，扩散通道为 soft label，二者不能在训练或阈值选择时泄漏。

## 7. “尽量下结论但不过于武断”的决策合同

### 7.1 两个正交输出轴

第一轴始终给**起始表型**；第二轴决定**可安全输出的空间分辨率**：

```text
electrode → region → hemisphere → multiple_modes → phenotype_only
```

在 source-development 上估计各粒度的条件错误风险，选择满足预设风险上限的最细粒度：

\[
g^*(x)=\max\{g:\widehat R_g(x)\le\epsilon_g\}.
\]

这比统一 confidence 阈值更符合任务：系统可以不承诺某个电极，但仍应判断更稳定的左颞区、左侧或全面同步表型。

### 7.2 输出状态矩阵

| 信号状态 | 必须输出 | 不允许 |
|---|---|---|
| 多次一致局灶起始 | 首选侧别/脑区，达到风险门时给 Top-1/Top-k 通道；列复现 `n/N`、替代和反证 | 裸称皮层 SOZ/EZ 或 100% 确定 |
| 局灶后快速双侧化 | 最早局灶候选 + 随后双侧募集，输出 `focal_with_rapid_bilateralization` | 把晚期高幅通道当起始 |
| 两种可靠且冲突模式 | `multiple_scalp_onset_modes`，分别报告 Mode A/B 及事件数 | 把左右平均成中线或任选一侧 |
| 双侧广泛近同步、无可复现先导 | `generalized_synchronous` 或规定的 ambiguous 类；说明无稳定局灶先导 | 为满足“必须有 SOZ”强行给单通道 |
| 有完整演变，但头皮场不可分 | `scalp_onset_nonlocalizable`，列具体原因和最细可支持粒度 | 只写一句“证据不足” |
| 信号不可读/关键导联技术失败 | 仍生成技术受限报告和失败 receipt；生理起始表型不强造 | 把技术失败写成全面同步或无发作 |

### 7.3 概率、排序与措辞

- 有患者隔离 calibration receipt：可显示校准 probability、Brier/ECE 来源和 prediction set；
- 无 calibration receipt：只显示 ordinal score/rank 和“首选/次选”，不能显示百分比；
- “倾向”“支持”“可能”由确定性 certainty tier 映射，Qwen 无权自行加强；
- `scalp_onset_nonlocalizable` 必须带 reason codes，如双侧同步、深部/低头皮可见性相容、关键导联缺失、伪迹跨 onset、跨事件冲突；不能作为默认逃生类。

## 8. Claim graph：从块级 fact ID 升级为句级关系合同

每一个可语言化 claim 至少包含：

```json
{
  "claim_id": "C-001",
  "claim_kind": "observation",
  "subject": {"type": "eeg_event", "id": "EV-01"},
  "predicate": "earliest_sustained_change_maximal_at",
  "object_or_value": {
    "region": "left_temporal",
    "electrodes": ["T7", "P7"]
  },
  "event_id": "EV-01",
  "mode_id": "MODE-A",
  "time_interval": {"lower": 622.4, "upper": 624.1},
  "epistemic_status": "clinically_qualified",
  "evidence_ids": ["E-031", "E-032"],
  "producer": {
    "producer_id": "event_findings_provider",
    "version": "...",
    "artifact_sha256": "..."
  },
  "qualification_receipt": "QR-012",
  "allowed_surface_frames": ["event_onset_maximal_at_v1"]
}
```

`claim_kind` 至少分为：

- `observation`：直接 EEG Finding；
- `event_inference`：该事件的表型/空间候选；
- `mode_inference`：一组复现事件的模式结论；
- `record_hypothesis`：整条记录的首选/替代 SOZ 假设。

关系本身也必须是 claim，例如：

```json
{
  "claim_id": "C-014",
  "claim_kind": "observation",
  "subject": {"type": "finding", "id": "F-onset-left-temporal"},
  "predicate": "precedes_recruitment_of",
  "object_or_value": {"finding_id": "F-bilateral-recruitment"},
  "time_interval": {"delay_seconds": {"lower": 8.0, "upper": 11.0}},
  "event_id": "EV-01",
  "evidence_ids": ["E-041", "E-044"]
}
```

只有存在该关系 claim，文本才能使用“随后”“传至”“早于”“同步”等词。一个句子绑定两个合法实体，并不自动授权模型创造它们之间的关系。

## 9. Qwen3.6 的职责：graph-to-text，不是自由诊断

### 9.1 推荐的最强约束路径

Qwen 只生成句子计划，而关键医学实体由确定性 lexicalizer 填入：

```json
{
  "sentence_id": "S-001",
  "section_id": "ictal_findings",
  "template_id": "event_onset_then_recruitment_v2",
  "claim_ids": ["C-001", "C-014"],
  "claim_order": ["C-001", "C-014"],
  "connector_ids": ["then_after_interval"],
  "optional_style_choices": {"compact": false}
}
```

时间、频率、侧别、脑区、导联、`n/N`、概率和 certainty 词由 renderer 从 claim graph 确定性写入。这样仍能获得接近真实报告的长句节奏，但不让 LLM改数字或实体。

### 9.2 次优兼容路径

若需要 Qwen 直接输出中文文本，必须逐句返回：

```json
{
  "sentence_id": "S-001",
  "section_id": "ictal_findings",
  "claim_ids": ["C-001", "C-014"],
  "text_zh": "...",
  "certainty_tier": "moderate"
}
```

请求使用本地 vLLM、`temperature=0`、关闭 thinking、strict JSON Schema。模型只可见去标识化 claim graph 和 style card；看不到原始 EEG、真实样例、annotation、Excel、医生标签或患者资料。

### 9.3 内容规划规则

Qwen 可做：

- 在固定栏目中排序句子；
- 合并同一 event/mode 下兼容 claim；
- 选择经批准的连接词和专业句式；
- 减少重复，形成临床报告的紧凑语言。

Qwen 不可做：

- 新建或删除关键 claim；
- 改变 claim 的 event/mode 归属；
- 把候选升格为确认事实；
- 从 Findings 自行推断一个 schema 中不存在的 SOZ；
- 用常识补全睡眠、诱发、临床表现或病史。

## 10. 发布 validator 与确定性回退

### 10.1 七级 validator

1. **Schema：** 句子、claim_ids、栏目和模板均在本次请求白名单；
2. **Coverage：** 所有必须 claim 恰好覆盖，关键 claim 无遗漏，禁止额外 claim；
3. **Entity：** 数字、时间、频率、导联、区域、侧别、概率、事件/mode ID 精确一致；
4. **Relation：** onset、evolution、spread、synchrony、termination、support/contradict 谓词与源图一致；
5. **Temporal：** 相对时间、延迟、先后、区间包含和左右删失一致；
6. **Epistemic/negation：** 否定范围、uncertain/not_evaluable、候选/支持/倾向等强度不越级；
7. **Scope/style：** 无患者资料、annotation、Excel、临床表现、睡眠、诱发、ECG/EMG，无“未提供结构化事实”等审计噪声。

可增加独立 round-trip claim extractor，但不能只用同一个 Qwen 自我审查；确定性字段核对和关系状态机仍是主门禁。

### 10.2 回退粒度

- 单句失败：用该模板的确定性 lexicalizer 重写；
- 多句或关系闭环失败：整节确定性 graph-to-text；
- Qwen 服务、解析或 schema 失败：整份确定性报告；
- 上游没有合格事件：生成 detector/pipeline miss 报告；
- 信号技术不可读：生成 technical-limited 报告。

因此“LLM 失败”永远不等于“报告生成失败”。报告任务的生成覆盖率目标为 100%；SOZ 生理结论覆盖率和技术可分析率另行统计，不能混成一个指标。

## 11. 目标报告版式

默认正文：

```text
长程头皮 EEG 自动分析报告（AI 草稿）

一、记录与技术质量
  记录时长、采样/导联、可判读比例、关键限制

二、EEG Findings
  1. 全记录背景/间期：仅在有全记录 producer 时显示
  2. 逐事件发作期脑电
     事件、相对时间、边界区间、起始、演变、募集、终止/恢复
  3. 跨事件模式汇总
     Mode A: n/N；Mode B: n/N；一致和冲突证据

三、脑电图印象
  1. 发作事件表型总结
  2. 研究性头皮 SOZ 首选候选
  3. 替代候选、反证与不确定性
  4. EEG-only 证据边界

四、相关 EEG 波形
  onset 放大图、全事件演变图、必要的背景对照；均绑定 evidence IDs

五、医师复核
  空白结论与签名栏
```

空栏目省略，不输出“未提供”“没有结构化事实”等句子。波形图必须标记相对时间、montage、灵敏度/滤波/采样信息、所示导联、证据 ID 和内容哈希；图中不能混入 annotation 标记或医生标签。

## 12. 四种报告结论示例

以下只展示句式逻辑，所有数值与导联均为合成占位，不是实际病例结论。

### 12.1 一致局灶模式

> 共分析 4 次电图发作，其中 3/4 次最早持续改变位于左侧颞区，以 T7、P7 为著，随后出现频率减慢、波幅增高并募集至同侧额区；另 1 次因起始段伪迹仅支持左侧侧别。跨事件证据首选左颞区头皮发作起始候选，T7 为首选通道、P7 为次选；该结论为研究性头皮 EEG 候选，不等同于皮层 SOZ 或致痫区。

### 12.2 局灶后快速双侧化

> 2/3 次事件在双侧募集前可见右额颞区短暂且可复现的先导改变，随后约数秒内转为双侧同步演变；第 3 次起始时间区间重叠，不能区分先后。综合倾向右额颞区头皮起始并快速双侧化，首选区域结论强于单电极结论。

### 12.3 多发作模式

> 本记录存在两个可复现的头皮起始模式：Mode A（3/5 次）以左颞区为著，Mode B（2/5 次）以右额区为著，二者侧别和空间分布不宜合并。脑电图印象为多头皮起始模式，分别保留左颞区和右额区候选，不给出虚假的单一中线 SOZ。

### 12.4 全面同步或头皮不可定位

> 各次事件均表现为双侧多个非相邻区域近同步出现并共同演变，未见跨 montage 稳定、可复现的局灶先导。脑电图印象为全面同步头皮起始表型；局灶单通道 SOZ 不适用。

或：

> 已提取完整演变性电图发作，但起始区间内空间场相互重叠且关键颞导联受伪迹影响，跨事件无稳定局灶先导。脑电图印象为头皮起始不可定位；该结论是明确表型判断，不是报告生成失败。

## 13. 评价协议

### 13.1 事实一致性

- claim precision/recall/F1；
- supported-claim precision、salient-claim recall；
- hallucination、contradiction、omission rate；
- event/mode attribution accuracy；
- numeric exactness；
- temporal entailment 和 order consistency；
- laterality/region/channel/distribution relation F1；
- onset--spread confusion rate；
- epistemic qualifier 和 negation-scope accuracy；
- forbidden-content rate；
- report generation coverage 与 technical-limited rate。

胸片 RadGraph 不能直接用于中文 EEG。应建立 EEG claim graph（EEGGraph）本体和 relation F1。

### 13.2 SOZ 与表型

- laterality balanced accuracy；
- region macro-F1；
- channel strict Top-1、Hit@3/5、MRR、mAP；
- 显著通道 hard endpoint；扩散通道 soft-label gain/weighted AP；
- generalized/nonlocalizable/multiple-mode phenotype macro-F1；
- generalized false-focalization；
- Brier、NLL、ECE/MCE 和 reliability curve；
- prediction-set coverage/size；
- resolution--risk/coverage 曲线；
- 每条可分析记录的表型输出率，目标 100%。

Excel 起始字段只做冻结后事实一致性：左颞、右额、双侧/全面、起始不清等先由盲法规则/专家映射到规范标签；“起始不清”应与 nonlocalizable/ambiguous 类单独评价，不能强迫匹配某一侧。

### 13.3 语言质量

- 有同记录完整医生参考报告时：corpus BLEU-1--4、ROUGE-L、METEOR；
- 本地模型和版本可审计时：BERTScore 或医学中文语义指标；
- EEGGraph factual F1；
- 冗余率、术语正确性、逻辑连贯和临床风格；
- 两名 EEG 医师盲评事实性、完整性、定位可用性、不确定性、语言质量、修改时间和 major/minor error。

BLEU/ROUGE 只评价文字重合，不能证明 SOZ 侧别正确或没有幻觉。Excel“起始”短字段也不是完整参考报告，不能用于 BLEU。

## 14. 关键消融与反事实

### 14.1 最低消融

1. 单事件 vs 所有事件简单平均 vs reciprocal-rank vs mode-aware MIL；
2. 单位权重 vs 质量/边界可靠性权重；
3. 无 Findings vs 数值 Findings vs 资格化 Findings；
4. 单平面 channel head vs laterality→region→channel 层级头；
5. 强制电极级 vs 风险控制的自适应粒度；
6. traditional abstention vs 强制表型；
7. Qwen 自由文本 vs block fact IDs vs claim-level graph-to-text；
8. Qwen 文本 vs 句子计划+确定性 lexicalizer；
9. 无 validator vs entity validator vs 完整 relation/negation/time validator；
10. 固定 `[-12,+48]` vs 同 detector 下的变长事件证据。

### 14.2 反事实审计

- 左右镜像 EEG：侧别和电极预测必须镜像；
- 打乱事件顺序：记录级结果不应变化；
- 复制同一事件：支持率和置信度不应上升；
- 删除最强事件：变化应与 leave-one-event-out receipt 一致；
- onset 与 later-spread 片段互换：模型不能把高幅传播当起始；
- 改变参考方式/montage：脑区结论应比单导联稳定；
- 注入颞后区伪迹：P8/T8/T7 集中应下降而非增强；
- 修改 annotation/Excel/医生标签：报告逐字节不变；
- Qwen 停服或返回越界 JSON：仍生成等事实的确定性报告。

## 15. 实施优先级与验收门

### P0：切断标签旁路

- 从生成 schema、adapter、renderer、Qwen payload 移除 `source_eeg_annotation_timing`；
- 删除/禁用 `_SOURCE_CONTEXT_SCOPE` 外部上下文版式；
- 加入“修改 annotation/Excel 不改变输出”的回归测试。

**验收：** annotation、Excel 和医生标签进入推理路径的字段数为 0。

### P1：建立双层 schema 和 claim graph

- 保留直接 Findings 的资格门；
- 新增 research AI hypothesis schema；
- 每个 observation/inference/relation 都有 claim ID、evidence IDs 和 producer receipt；
- 不再要求把 AI hypothesis 伪装成 `physician_verified` impression fact。

**验收：** 事实层与推理层可机器区分；任一报告句可回溯到 claim 和波形证据。

### P2：mode-aware hierarchical MIL

- 用 DeepSOZ patient-level positive-set 训练；
- 接入 Findings、质量、边界和删失；
- 输出 phenotype、laterality、region、C18 channel 与 alternatives；
- source-development 完成校准和自适应粒度选择。

**验收：** 相对 reciprocal-rank 基线，在患者宏平均 SOZ 指标或风险--覆盖曲线上有预注册增益，且 generalized false-focalization 不恶化。

### P3：Qwen graph-to-text 与 fallback

- 优先实现 sentence plan + deterministic lexicalizer；
- 完成 relation/time/negation/event validator；
- 任意 LLM 故障都落到确定性报告。

**验收：** 报告生成覆盖率 100%，unsupported critical claim 为 0，必需 claim omission 为 0。

### P4：冻结后私有评价与医师研究

- 冻结模型、阈值、校准器、claim graph 和报告；
- 再加载 Excel/显著/扩散标签；
- 完成事实、SOZ、校准、语言和医师盲评。

**验收：** 所有结果按患者 bootstrap 报 95% CI，并明确公共开发、公共评估和私有 post-freeze 队列的角色。

## 16. 最终方法定位

该方案不是让 Qwen“大胆猜诊断”，而是把大胆程度放在一个可训练、可校准、可验证的研究性 SOZ 推理层：

- Findings 回答“信号实际显示了什么”；
- mode-aware MIL 回答“综合所有发作，哪个头皮起始假设最合理”；
- 风险控制的空间粒度回答“可以具体到电极、脑区、侧别，还是只能判断全面/不可定位”；
- Qwen 只回答“如何按真实临床报告的逻辑把锁定证据说清楚”。

这样可以显著减少无意义的“证据不足”句，又不会为了追求结论率把晚期扩散、伪迹或模型偏倚包装成确定 SOZ。
