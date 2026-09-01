# SOZ target 定义、标签来源与传播处理合同

**状态：** 2026-08-15 v2.9 可信候选、selective evaluation、双roster、跨层与距离错误修订版
**适用范围：** DeepSOZ--TUSZ overlay、TUSZ原生标注和两份私有医生表  
**输出空间：** 标准19个头皮物理电极  
**核心变化：** 最终目标不再是“最早头皮可见电极”。最早变化、发作受累和时间演变均为证据；最终目标是医生综合判断形成的临床 SOZ 假设在标准19导空间中的投影。

> **v2.5论文目标覆盖：** 不再承诺private strict 80%或neighborhood-4 85%。主张改为可信
> scalp-electrode SOZ candidate ranking、阴性concept qualification、低置信弃权和typed-fact
> 可追溯报告。所有full-coverage strict结果继续强制报告；selective relaxed必须同时给coverage和
> 分子/分母。DeepSOZ论文约0.744是含条件式邻域的patient accuracy，不是strict电极accuracy。
> 当前v21.1 public/private selective relaxed点估计为75.31%/76.74%，但private已历史开标、单位不同且
> 区间不支持显著优越。详见
> [`trustworthy_soz_candidate_v21_method_and_result_20260815_zh.md`](trustworthy_soz_candidate_v21_method_and_result_20260815_zh.md)。

> **v2.6性能口径：** 希望在public与private分别取得full-coverage
> `official-neighborhood-4 > 0.70`的点估计，但这不是strict Top-1承诺或临床保证。DeepSOZ论文
> 实际中心值为`0.744`，因此“超过70%”与“超过DeepSOZ”不是同一句话。完整验收和claim边界见
> [`paper_objective_and_endpoints_v22_20260815_zh.md`](paper_objective_and_endpoints_v22_20260815_zh.md)。

> **v2.7 roster口径：** 不读target的完整signal replay得到114 patients/1,364 eligible events；
> 当前C18定位主队列是identity-v16的102 patients/1,145 events。旧102 patients/988 events是另一套
> legacy event-evidence/report core，两者只交集101 patients。patient 258因O2 reference不完整只能
> 保留legacy event reports并输出`localization_unavailable`；patient 10489可有patient-level候选但
> 不得补写event timing/morphology facts。历史`102/988 closed roster`只约束旧证据工件，不再代表
> 当前定位分母。详见
> [`v22_4_deepsoz_124_to_102_patient_flow_and_roster_audit_20260815_zh.md`](v22_4_deepsoz_124_to_102_patient_flow_and_roster_audit_20260815_zh.md)。

> **v2.8 evidence/target边界实证：** 438个可比较public事件中，target-free首批双极边侧别与冻结
> patient-level候选同侧211、跨侧张力227；68名有可比较事件且显示候选的患者中40名两种关系并存。
> 这不是target accuracy，也不能判定哪一层正确；它实证支持E1--E4不得重命名为R1/R2、不得作为
> H-only候选的因果解释或新路由gate。详见
> [`v22_5_cross_layer_evidence_candidate_concordance_audit_20260815_zh.md`](v22_5_cross_layer_evidence_candidate_concordance_audit_20260815_zh.md)。

> **v2.9终点分解：** public strict/neighbor-only/far=`47/27/28`，private=`21/16/14`；far中跨侧
> 分别为22与9，private另有2次known-spread Top-1。两个队列Hit@5均80.39%只表示5候选清单命中，
> 不是SOZ Top-1 80%。SOZ target评价必须保留exact、neighbor sensitivity、far/contralateral-far和
> spread exclusion的不同医学语义。详见
> [`v22_6_full_coverage_ranking_and_distance_error_audit_20260815_zh.md`](v22_6_full_coverage_ranking_and_distance_error_audit_20260815_zh.md)。

## 1. 最终冻结的研究目标

本项目统一研究构念为：

```text
clinician_integrated_scalp_electrode_SOZ_reference
```

中文定义：给定一次或多次已确认的头皮 EEG 发作，模型依据预先定义的 EEG 证据，对标准19个物理电极提供 SOZ 支持度排序，使结果尽可能符合独立临床参考。

该输出表示：

> 哪些标准19头皮电极最支持临床 SOZ 假设。

它不表示：

- 该电极正下方的皮层必然是生物学 SOZ；
- epileptogenic zone（EZ）；
- 必须切除或消融的组织；
- SEEG/ECoG 最早 contact；
- 手术靶点或术后无发作结局。

头皮电极是体积传导后电场的传感器。最早可见、最大幅度、最典型波形、临床综合定位和真实皮层起始源之间可以相关，但不是同一个变量。[1--3]

全文将 DeepSOZ 和 private 标签称为 `clinical reference` 或“操作性参考标准”。实验代码中可以使用 `gold target` 作为字段名，但论文不能无修饰地写成 biological ground truth；两者都没有 SEEG、切除范围或术后结局确认。

## 2. 两个 source-specific reference

### 2.1 DeepSOZ：训练参考与反复使用的公共开发评价

```text
patient_clinical_note_derived_standard19_SOZ_reference
```

属性：

- 粒度：`patient × physical electrode`；
- 来源：TUH clinical notes；
- 信号：TUSZ/TUH seizure recordings；
- 角色：SOZ reasoner 的主要公开监督、开发和患者隔离的探索性评价；
- 独立单位：患者，不是record或seizure；
- 可支持的结论：对 clinical-note-derived scalp-electrode SOZ reference 的预测；
- 不支持的结论：逐次发作最早电极、侵入性SOZ或独立外部验证。

DeepSOZ 是 TUSZ/TUH 信号上的标签 overlay，不是新的 EEG 数据集。[7]

本地映射成功的102人已经用于多轮患者隔离开发、LaBraM恢复实验和DeepSOZ官方权重迁移。
它只能称`repeatedly_used_public_development_benchmark`，不能再称label-unseen、untouched或
confirmatory test。v21.1在明确降级为post-open exploratory revision后，又仅用这102人的OOF
预测选择了一次margin弃权工作点；这不是确认性测试，也不提供风险控制保证。自此不得继续在该队列
扫描新结构、loss、seed、窗口、graph、fusion或拒答阈值。患者隔离OOF只防止单次fit直接读到该
患者target，不能消除反复方法开发造成的选择偏倚。

### 2.2 私有数据：历史复用的回顾性描述性迁移参考

```text
event_clinician_integrated_EEG_semiology_SOZ_reference
```

属性：

- 粒度：`event × physical electrode`；
- 来源：医生结合整次发作 EEG 和患者外在临床表现后的综合定位；
- positive：医生表中的 `显著电极`，在医生确认本合同语义后视为临床 SOZ reference positive；
- spread：`早期扩散`独立保存；
- 当前正式流水线角色：模型、阈值和报告规则冻结后的描述性迁移评价；
- 当前正式流水线禁止角色：foundation continuation、concept训练、reasoner训练、微调、校准、阈值选择、架构选择或报告模板选择；
- 证据等级限制：历史private LOPO、区域、校准和报告实验已影响方案形成，故同一队列不是独立、untouched或confirmatory验证。

私有参考包含模型没有输入的 semiology，因此它是一个具有 privileged clinical information 的较强参考。模型与其不一致可能来自模型错误，也可能来自仅凭 EEG 无法恢复的临床信息；这应作为性能上限和失败分析的一部分，而不是把 semiology 偷偷加入模型输入。

还存在明确的粒度差异：DeepSOZ训练只约束患者级聚合输出，private现有标签是事件级。因此private event结果是`cross-domain, cross-granularity secondary transfer`，不是与public主任务完全同粒度的确认性验证。若需要matched patient-level private endpoint，必须由医生基于该患者全部发作独立形成patient consensus；不得用event union/intersection/majority自动生成。

由于当前私有队列已经影响过历史方法讨论和实验，它应称为：

```text
historically reused retrospective descriptive transfer cohort
```

不能追溯性称为 untouched external/confirmatory test。确认性结论需要未来时间段或独立中心的新患者。

## 3. 证据与目标的层级边界

| 层级 | 变量 | 原生粒度 | 数据来源 | 在本方法中的角色 |
|---|---|---|---|---|
| E0 | 全局 seizure start/end | event × time | TUSZ、私有事件锚点 | 输入对齐，不是空间SOZ |
| E1 | ictal involvement | event × time × bipolar derivation | TUSZ `.csv`/`.rec`/`.lab` | ictal-pattern concept监督 |
| E2 | earliest scalp-visible change | event × time × signal coordinate | 可由TUSZ弱边界或额外专家标注获得 | onset evidence，不是最终答案 |
| E3 | epileptiform morphology | time × bipolar derivation | TUEV原生CE6 | morphology concept监督 |
| E4 | temporal evolution | event × time × electrode/edge | 波形描述量和自监督顺序 | evolution evidence；不称传播真值 |
| R1 | DeepSOZ clinical SOZ reference | patient × physical electrode | clinical notes | reasoner训练/公共评价 |
| R2 | private integrated clinical SOZ reference | event × physical electrode | whole-event EEG + semiology | 历史开标后的描述性迁移 |
| R3 | invasive SOZ | patient/event × iEEG contact/region | 当前无 | 未来更强验证 |
| R4 | treatment coverage/outcome | patient | 当前无 | 未来临床有效性验证 |

E1--E4 是 SOZ 推理证据，不能直接重命名为 SOZ 标签。R1 与 R2 的粒度不同，不能逐事件混在同一个 row-level loss 中。

## 4. DeepSOZ 标签的医学来源和限制

DeepSOZ论文说明，作者根据每位患者的 clinician notes 创建 SOZ 标签，并将临床定位投影到标准19个EEG电极的子集。论文模型对recording产生输出，再在患者内聚合。[7]

因此准确表述是：

> clinical-note-derived patient-level standard-19 scalp-electrode SOZ reference。

公开manifest没有逐患者说明clinician note中的定位究竟来自头皮EEG、semiology、影像、
侵入性检查或其组合。因此该reference可能包含模型不可见的临床信息，也可能与被研究EEG
存在criterion dependence。它适合定义可复现的operational benchmark，不足以作为独立、
模态纯净的生物学SOZ真值；论文必须把这一点作为标签来源限制而非普通随机噪声处理。

不应写成：

- 癫痫专家逐次盲法重标了600多次发作；
- 607个独立SOZ患者标签；
- TUSZ原生SOZ channel annotation；
- SEEG或手术结局验证的cortical gold truth。

### 4.0 Positive set 的粒度合同

官方manifest的652条记录中，阳性集合大小为：1导59条、2导237条、3导261条、4导85条、
5导6条、7导1条、8导3条。大量`Comments`只是`left temporal`、`right frontal`或
`central parietal posterior temporal`等粗区域文本，随后被展开为多个并列阳性电极。

因此 DeepSOZ target 是**无集合内优先级的 patient-level positive set reference**。本项目的
`exact-electrode positive-set membership Top-1`要求argmax落入该集合，是严格且可复现的
benchmark终点；但不得进一步解释为临床文本已经给出了集合内哪个电极是唯一第一名的gold。
当前102人中93人只有粗区域描述，只有9人含明确电极、`maximal`或`centered`等措辞。

### 4.1 公开清单与本地映射审计

- 固定上游版本：commit `913c921f8a08fa4df76ca0708126f565860f1068`；
- 公开manifest：652 records、124 patients；
- 论文报告：642 records、120 patients，二者存在未解释版本差异；
- 本地TUSZ v2.0.3保守映射：607 unique、14 ambiguous、31 unmapped；
- 607 records对应1,566个seizure events，但独立空间标签仍最多124名患者；
- 全source records上113名患者标签稳定，11名患者有冲突向量并进入quarantine；
- 607个EDF中580个完整包含标准19导，27个缺`FZ/PZ`，涉及6名患者；
- 历史header/time-bound筛查：106 patients、464 records、1,049 events；患者split为69/16/21；
- 历史warm-up-only规划估计：103 patients、450 records、998 events；患者split为66/16/21；
- target-independent identity replay从652 records/124 patients上的1,812个candidate events出发，
  得到114 patients/1,364 eligible events；其中10名患者没有任何合格信号；
- 与stable target相交后有103名患者，其中102名具完整C18 reference并构成当前1,145-event定位主队列，
  patient 258缺O2 reference；
- 历史closed replay从旧join流程内1,330个candidate events得到另一组102 patients/988 events，
  split为65/16/21；它仅是legacy event-evidence/report core，不是当前定位分母。

前两项是历史中间availability估计，不能作为正式reasoner roster。当前定位严格绑定identity-v16
102/1,145；legacy逐事件报告严格绑定另一组102/988，联合可用交集为101人。若PZ mask或outside-head
政策使某患者在可评价19导内没有任何positive，该患者不能被当作“全19导阴性”进入
channel-localization主终点，必须作为`positive_outside_or_masked_head`单独流转。

### 4.2 DeepSOZ label-state政策

公开manifest存在重复`PZ`字段、空值、out-of-head字段和患者内冲突。主分析采用以下状态：

| 源值/情况 | 主分析状态 | 允许用途 |
|---|---|---|
| 明确`1` | `reference_positive` | SOZ正例 |
| 明确`0` | `dataset_complement_negative` | 仅按DeepSOZ公开任务语义训练；不称临床确认阴性 |
| 空值/未解析 | `unknown` | loss mask=0 |
| 两个PZ字段冲突 | `schema_conflict` | 主分析mask；first/second/OR仅敏感性分析 |
| 患者内向量冲突 | `patient_target_conflict` | 整名患者quarantine |
| `OZ/A1/A2` positive | `positive_outside_head` | 流程报告；不能映射到邻近19导 |

主channel-localization cohort必须至少有一个未被mask的in-head reference positive。只有outside/PZ positive的患者不生成全零SOZ target。

若采用公开DeepSOZ二值benchmark语义，论文必须明确：`0`是dataset complement，并不证明逐电极临床排除。另做positive-only敏感性分析；positive-only视图适合Hit@K、MRR和positive recall，不足以单独支持普通19维BCE。

### 4.3 官方 localization 评价政策

官方代码不是统一的纯 exact Top-1：`szloc_all.ipynb`在阳性数不超过2时把任一阳性电极的
一跳邻域命中也计为正确；`final_eval_all.ipynb`把该阈值改为4。故正式评价必须并列报告：

1. `exact_set_membership_top1`：argmax必须属于positive set，严格主终点；
2. `official_neighborhood4_top1`：绑定官方final notebook的可比次终点；
3. `official_neighborhood2_top1`：绑定另一官方notebook的实现敏感性；
4. hemisphere、A/P quadrant及预注册clinical region的一致性；
5. AP、MRR、Hit@K、far error与patient-bootstrap置信区间。

Relaxed endpoint不能替换或改名为strict。当前102人上identity-v16/v17分别为：exact
`51/102`与`51/102`，neighborhood-2 `68/102`与`69/102`，neighborhood-4 `77/102`与
`78/102`。DeepSOZ论文约0.744的patient accuracy与50% exact不矛盾；前者包含邻域规则，
且两项实验并非同roster、同PZ政策或直接复现。完整证据见
[`deepsoz_official_endpoint_audit_20260813_zh.md`](deepsoz_official_endpoint_audit_20260813_zh.md)。

2026-08-14又使用官方15个fold中患者实际所属test fold的发表权重，在同一102人/C18/PZ-mask
终点完成信号版本迁移：strict `48/102=47.06%`、neighborhood-2 `64/102=62.75%`、
neighborhood-4 `76/102=74.51%`。它没有超过LaBraM-v17；配对bootstrap区间均跨0。
该结果是`published-weight signal-version transfer`，不是原始TUSZ快照的逐位复现，也不授权
在102人上搜索两模型融合。详见
[`deepsoz_official_signal_version_transfer_result_20260814_zh.md`](deepsoz_official_signal_version_transfer_result_20260814_zh.md)。

### 4.3 已生成的 target artifact v2 与 closed signal join

文件 [`outputs/deepsoz_tusz_patient_splits_v1/patient_targets.csv`](../../../outputs/deepsoz_tusz_patient_splits_v1/patient_targets.csv) 仍是保守的 positive-only 审计产物：所有非正例均被 mask，且 `ordinary_bce_ready=0`。它可以继续用于 crosswalk、稳定性和 positive-only 指标，但**不能直接供本方案的 BCE 或正负排序损失使用**。

2026-08-08 实时审计确认已生成独立的
[`outputs/deepsoz_target_v2`](../../../outputs/deepsoz_target_v2) bundle：

- schema: `deepsoz-patient-target-artifact-v2.0.0`；
- policy: `deepsoz-benchmark-target-v2.0.0`；
- source: 124 patients，target/header-policy eligible 106；
- `patient_targets_v2.csv` SHA-256:
  `5c01591c20328fb60817099cac669032bd743e36f47df77ac390842e9a2c67ed`；
- private labels are absent by construction；
- strict verified loader replays the frozen source and split inputs and rejects
  unknown fields, byte changes, patient changes, PZ policy drift, and label
  replacement.

方法已冻结采用 DeepSOZ 官方二值 benchmark 作为主训练视图。已生成的 patient-level `target artifact v2`必须继续同时保存两个互不覆盖的视图：

```text
benchmark_binary_view
  1                 → reference_positive, value=1, mask=1
  explicit 0        → dataset_complement_negative, value=0, mask=1
  blank/unparsed    → unknown, mask=0
  canonical PZ      → schema_conflict, mask=0 in primary

medical_positive_only_view
  1                 → observed_reference_positive
  all other states  → unknown/not-evaluable as negative
```

v2 保持一患者一行，继承 frozen split、冲突患者 quarantine、outside-head flow 和 source hashes。训练 loader 必须验证 artifact 的 schema/version/hash；不得在读取 v1 后于内存中把 mask 暗改为1。

Target gate、current localization join与legacy event-evidence join现均已通过，但必须分开命名。
下列正式产物
[`outputs/deepsoz_signal_preflight_v1_20260808/deepsoz_signal_preflight.json`](../../../outputs/deepsoz_signal_preflight_v1_20260808/deepsoz_signal_preflight.json)：

- schema：`soz_deepsoz_signal_preflight_artifact_v1`；
- candidate events：1,330（join流程候选数，不是原始DeepSOZ总量）；
- eligible：legacy event-evidence roster的102 patients、988 events，split为65/16/21；
- exclusions：`ambiguous_standard19=161`、`insufficient_warmup=125`、
  `insufficient_post=46`、`signal_qc=10`；
- 106名target/header-eligible患者中4名没有任何eligible event：3名患者的
  events全部缺少因果warm-up，1名患者的events全部触发signal QC；
- artifact SHA-256：
  `6d8808e3540f3ad9e2fb2e2f3ebca3b34e65c69c4519b2b6b11fc5827e198b73`；
- receipt SHA-256：
  `98b3c445a02bf4b07f4c4ac516f476637cbc212f921330d33f09b20eff787858`；
- 两遍strict replay得到相同闭环产物。

该产物逐事件重放并绑定crosswalk、EDF hash、TUSZ global event timeline、
精确`t0`、`[-12,+48)`窗口、冻结预处理配置、processed-window hash和signal
receipt，而不是仅做两个CSV的患者ID交集。`signal_qc`在滤波和CAR前检查每个
direct standard-19 trace：任一导联出现持续flatline、持续ADC极值/削顶或窗口内
记录gap/discontinuity即排除。临床原因是坏导联本身会伪造局灶证据，且CAR可将
单导污染传播到全部19导，因此这10个事件不能进入SOZ定位证据。

当前定位的完整患者流另由
[`outputs/deepsoz_124_to_102_patient_flow_v22_4_20260815/result.json`](../../../outputs/deepsoz_124_to_102_patient_flow_v22_4_20260815/result.json)
冻结：signal universe为114 patients/1,364 events，current localization primary为102 patients/
1,145 events，legacy event-evidence core为另一组102 patients/988 events，交集101 patients。
因此上述preflight的SHA继续证明legacy内容可重放，但不能证明两个102人roster同一，也不能把988
写成当前定位器的完整事件输入。

Reasoner仍是实施层`NO-GO`，但原因已不再是DeepSOZ signal join；剩余门槛是
M/I/V producer provenance、完整OOF evidence和正式reasoner artifacts。

## 5. 私有显著电极和 spread 的定义

正式打开 private labels 前，医生需完成并签署
[`private_annotation_adjudication_manual.md`](private_annotation_adjudication_manual.md)，
并逐项裁决重复事件与 significant/spread overlap。以下规则是拟采用的研究合同，
不能用代码默认值替代临床确认。

### 5.1 Significant electrode

在医生确认本合同后，私有表中`显著电极`定义为：

```text
positive_clinician_integrated_SOZ_reference
```

其含义是医生在审阅整次发作 EEG 并结合临床外在表现后，认为最支持该次发作 SOZ 假设的电极集合。它不要求等于：

- 第一个出现波形变化的电极；
- 最大幅度电极；
- TUSZ中第一个阳性双极derivation；
- 模型的ictal-emergence concept；
- 真实皮层SOZ的直接观测。

原始Excel字段名本身不能证明上述语义。正式研究需要保存医生确认的annotation manual或签字版字段定义，并记录标签是event-level还是patient-level。

由于模型只读取EEG、并不读取semiology，正式private reference还必须逐事件保存
`localization_basis = EEG_primary | EEG_and_semiology_concordant |
semiology_resolved_ambiguous_EEG | unknown`。后两类属于目标包含模型不可见信息的
临床综合终点，应分层报告，不能把差异简单解释为EEG模型定位错误，也不能把semiology
事后喂给模型或解释生成器。

若医生没有明确声明“逐一审阅标准19导且显著集合是穷尽性positive set”，未列出的电极仍为`not_marked/unknown`，不能自动作为临床确认阴性。此时主评价使用Hit@K、positive recall和ranking；exact F1、Jaccard、AUROC和校准只在具有完整complement合同的病例上计算。

### 5.2 Spread electrode

`早期扩散`保存为：

```text
known_clinical_spread_reference
```

规则：

1. 永不并入SOZ positive；
2. 不自动作为绝对生物学non-SOZ；
3. 可用于验证`score(significant) > score(known spread)`；
4. 没有原生recruitment time时不能声称学习了传播顺序；
5. `弥漫性`是event descriptor，不生成全19导spread或SOZ标签；
6. significant与spread重叠的事件必须人工裁决，不能由程序择一。

### 5.3 Private当前可评价分母

现有legacy matched manifest的只读审计显示：

- 45名患者、139个唯一patient-event keys；
- 现有匹配层为43名患者、123个事件；
- 86个事件、34名患者有非空significant内容；
- 55/86事件含`SPHL/SPHR`；只有31个事件的完整source-positive集合全部落在标准19导内；
- 标准19导内共有242个positive event-electrode incidences；
- spread包括91个电极列表、28个diffuse-only、3个diffuse+electrode和1个“无”；
- 1个事件的significant和spread出现同一电极，需裁决。

因此“45名患者”不是每个endpoint的固定分母。channel、region、spread-ranking和报告盲评必须分别给出eligible患者数与事件数。

### 5.4 原始工作簿优先级与冲突合同

2026-08-08直接重放`EEG-fMRI颞叶癫痫(1).xls`和`头皮扩散.xlsx`得到45名患者、
142个event row和139个唯一patient--event key。两份表有相同的两行表头和四个
`SZ`块，但3个跨表重复key的onset、significant和spread均冲突，必须由医生裁决；
不能依文件顺序静默保留第一条。

原始表中102/142行有可解析significant，其中63行同时含`SPHL/SPHR`；40行为显式
`无`。141/142行有spread，37行含弥漫/全导语义，88行含`SPHL/SPHR`。`覆盖全导`
字段为36个`是`/5个`否`和98个`是`/3个`无`，只表示后续全导受累，不是SOZ target。

历史`发作起始通道汇总.csv`相对原始表少1个唯一key，且共同key中significant有
10处、spread有13处不同；onset文本全部经过改变。历史builder还会把
`覆盖全导=是`改写为弥漫spread，并在某些分支把spread-only当作弱SOZ positive。
因此正式reference的证据优先级固定为：

```text
原始workbook cell + 医生裁决 + exact EDF/event crosswalk
    > 可重建且有receipt的派生表
    > 历史summary/legacy manifest（禁止作为formal gold）
```

正式loader必须拒绝任何含spread→SOZ、coverage→SOZ、SPH→邻近头皮点扩张或未裁决
duplicate的artifact。

## 6. 标准19导和TCP坐标合同

### 6.1 标准19物理输入

内部固定顺序：

```text
FP1 FP2 F7 F3 FZ F4 F8 T7 C3 CZ C4 T8 P7 P3 PZ P4 P8 O1 O2
```

只应用身份别名：

```text
T3→T7, T4→T8, T5→P7, T6→P8
```

`A1/A2`是耳/乳突位置，`SPHL/SPHR`是额外蝶骨电极，`OZ`也不在部署19导中。它们不得投影为邻近标准19导positive。

### 6.2 TUSZ信号与标注不是同一坐标

对已映射的`01_tcp_ar`记录：

```text
raw EDF signal: EEG FP1-REF, EEG F7-REF, ... EEG PZ-REF
TUSZ spatial annotation: FP1-F7, F7-T3, ... (TCP bipolar edge)
DeepSOZ target: FP1, F7, T3, ... (physical electrode)
```

DeepSOZ官方`readEDF`直接读取19个`*-REF`物理通道；没有TCP到单极的伪逆，也没有显式average-reference步骤。官方对缺失通道补零只能用于published-baseline复现，不能作为本项目主分析。

主信号流程为：

```text
raw physical EDF channels
  → exact-name and unit audit
  → read [t0-42 s, t0+48 s) finite segment with zero IIR state
  → causal 0.5--45 Hz IIR and causal resampling to 200 Hz
  → compensate the fixed FIR-resampling delay
  → complete-standard19 gate
  → fixed standard19 average reference
  → crop [t0-12 s, t0+48 s)
```

这一路径只有30秒真实warm-up，不是从EDF起点传播滤波状态的
`continuous-record filtering`。版本化receipt保存零状态初始化、实际read-start、warm-up
样本数以及17点频率依赖IIR group delay；IIR不存在一个可对所有频率统一校正的标量延迟，
因此报告不得把边界偏移解释为生理或检测延迟。正式预处理臂仍须由source-train-only
五臂parity门禁选择并冻结；在此之前生成的旧token只能作为candidate。

若19个通道共享公共参考，固定CAR为：

\[
\tilde x_c(t)=x_c(t)-\frac{1}{19}\sum_{k=1}^{19}x_k(t).
\]

DeepSOZ官方`REF`处理保留为复现baseline。主分析不得零填充缺失`FZ/PZ`；不完整记录单独进入robustness analysis。

### 6.3 为什么不能把TCP22重构称为标准19

若真的只有双极信号，关系为：

\[
\mathbf b(t)=A\mathbf v(t).
\]

只有目标电极图连通且`rank(A)=C-1`时，才可恢复到共同加性常数，即该节点集合的average-reference解。官方TCP22的活跃节点是17个标准头皮节点加`A1/A2`，缺`FZ/PZ`。所以它即便可伪逆，也不是standard-19 reconstruction。

信号可重构也不意味着标签可重构。`F7-T7`阳性不能推出F7、T7或二者为SOZ。

## 7. Source-aware target schema

每个target必须保存：

```text
patient_id_pseudo
event_id_or_patient_target_id
target_source
target_granularity              # patient | event
label_basis                     # clinical_note | full_event_EEG_plus_semiology
localization_basis_detail       # EEG_primary | concordant | semiology_resolved | unknown
reference_grade                 # high | moderate | low | unresolved
reference_components            # clinical_note | iEEG | focal_MRI | resection | outcome
reference_grade_source          # authoritative source; never inferred from model score
intended_role                   # train | dev | label_informed_source_eval | historical_private_descriptive
electrode
label_state                     # positive | dataset_negative | confirmed_negative | unknown | outside | conflict
target_view                     # benchmark_binary | medical_positive_only
raw_value
source_file / sheet / row / column
reader_ids / adjudicator
annotation_manual_version
channel_mapping_version
source_hash
```

Earliest-change和spread信息另存于evidence表，不进入SOZ target字段。

`reference_grade`评价的是标签证据链，不是模型置信度。DeepSOZ公开manifest没有逐患者完整iEEG、MRI、
resection和outcome ledger，因此现阶段统一保持`unresolved`，不得根据模型是否预测正确事后赋权。Private
已有标注同样不能在一次性结果打开后用错误模式倒推grade。未来新队列只有在医生/原始资料给出可审计
证据链时才能预注册分级。

## 8. 训练和评价合同

### 8.1 DeepSOZ

模型可产生event score `s[p,e,c]`，但必须先按患者聚合：

\[
s_{p,c}=\frac{1}{|E_p|}\sum_{e\in E_p}s_{p,e,c}.
\]

随后每名患者计算一次masked multilabel loss：

\[
\mathcal L_{SOZ}=\mathcal L_{masked\ BCE}
+\lambda_r\mathcal L_{pairwise\ rank}.
\]

loss、采样权重和bootstrap均以患者为单位。不能把607 records或1,566 events当成独立空间标签。

普通 BCE/ranking 只允许读取通过版本校验的 `benchmark_binary_view`。`medical_positive_only_view` 仅用于 Hit@K、MRR、positive recall 和标签政策敏感性；它不能被普通 BCE 隐式当作完整负例集合。

DeepSOZ主政策将canonical PZ对所有患者mask。因此模型可保留standard-19物理carrier，
但公共主任务只有18个可评价channel；PZ不得进入主top-k、阈值、校准或region maximum，
只能标为`not_benchmark_evaluable`并进入预注册PZ列政策敏感性分析。

当前patient-balanced BCE在每名患者内对positive set和dataset-complement set等权，raw
sigmoid首先是定位分数而非自然患病率概率。只有reasoner冻结后在source-dev上拟合的
全局校准器可产生DeepSOZ患者级benchmark-label probability；仍须同时报告校准前后
unweighted masked NLL/Brier。

患者内等权event平均是患者级target下的固定聚合假设，不提供event-level SOZ真值。
正式评价必须报告event-count分层、one-event-per-patient抽样、固定事件数截断和
patient内event-logit dispersion。

当前102人已经被用于多轮结构和恢复实验，只能称`repeatedly_used_public_development_benchmark`。
不得再用它选择auxiliary weight、L2、block、seed、pooling、graph、region loss、PZ政策、邻接表
或abstention阈值。新结构只能在新的、patient-disjoint、同endpoint临床标注开发队列上选择。

#### 8.1.1 未来四态 ledger 的 set-valued loss

当前 target artifact v2和已执行loss保持不变。未来新的S1-D/A5队列必须逐电极区分
`candidate_positive`、`reviewed_not_candidate`、`unknown_not_reviewed`和
`unavailable_signal_or_reference`，spread另存。令`P`为候选正集合、`N`为医生已复核非候选、
`R=P union N`；集合softmax只在`R`内归一化：

\[
L_{set}=-\log\frac{\sum_{c\in P}\exp(s_c/\tau)}
{\sum_{c\in R}\exp(s_c/\tau)}.
\]

Masked BCE和positive-vs-negative pairwise loss也只能读取`P/N`。unknown/unavailable不进入分母、
负类或伪标签。多个阳性是可接受集合，不允许用PiCO+/SoLar式单真类disambiguation强制选择唯一
“真电极”；无可信class prior和SCAR/SAR选择机制时也不采用PU学习。BUNDL只处理native seizure
interval噪声，不修改C18 SOZ target。完整边界见
[`v22_10_partial_noisy_label_set_target_and_uncertainty_protocol_20260815_zh.md`](v22_10_partial_noisy_label_set_target_and_uncertainty_protocol_20260815_zh.md)。

### 8.2 Private

当前正式模型只在锁定后对私有数据前向推理。由于该队列历史上已被反复用于实验，以下均为
描述性迁移分析而非独立验证：

- event-level Hit@1/Hit@3和positive recall；
- patient-macro average precision，仅限complement语义完整病例；
- significant-vs-known-spread pairwise ranking；
- laterality和临床region一致性；
- patient-cluster bootstrap 95% CI；
- 模型不确定性、abstention和outside-head coverage；
- 两名盲法癫痫EEG专家对解释事实正确性、证据一致性、专业性、过度断言和临床可用性的评价。

报告评价者应尽量不同于生成private reference的医生。报告不能获得reference label，也不能添加模型没有观察到的semiology事实。

上述event指标均属于secondary cross-granularity transfer。只有新增独立patient-level clinician consensus后，才能把patient-aggregated private结果称为与DeepSOZ同粒度的主要临床验证。

source-dev校准器拟合的是患者内多事件平均logit，不能把private单事件输出称为已校准
event probability。Private主结果使用ranking/coverage；若显示该单调变换，必须命名为
`patient_calibrator_transformed_score`。Private PZ-positive、partial/outside-only及冲突
事件分别报告，不静默并入complete-standard19分母。

任何未来reference-grade分析都只能作为预注册分层/敏感性，不得把low-grade病例从主flow中事后删除。
模型的`Q_obs`、fold/seed disagreement、医学reference grade与未来S1-C calibrated candidate-set risk
是四个独立字段；当前margin不等于其中任一概率。

## 9. 与概念分支的非循环关系

预注册的三个 concept family 原本回答的是：

1. 哪种可见EEG形态存在？
2. 哪些双极信号在何时表现出ictal involvement/onset-pattern evidence？
3. 各物理电极的显式信号描述量如何随时间变化？

它们不回答“哪个电极是SOZ”。因此最终reasoner学习的是：

```text
independently supervised EEG evidence
  → patient/event clinical SOZ reference
```

而不是把earliest score重新排序，也不是把TUSZ edge label展开为DeepSOZ node target。

本次 formal-v5 已在未打开 I-gate 前失败，因此第2项只作为冻结的阴性 concept 实验报告，
不进入当前 SOZ reasoner。随后 M 也在 source-train OOF 局灶 precision 门失败，未打开 TUSZ
dense audit。因此 I、M 均必须保持全 mask；失败的 involvement/morphology score 不能因为
与目标“看起来相关”而在下游复活。剩余 `V → SOZ reference` 只可作为不经过 foundation
encoder 的 engineered baseline，不能称为完整 foundation-model 主方法，也尚未获得正式
event issuer/cache 授权。

## 10. 允许和禁止的论文表述

### 可以声称

- 使用DeepSOZ clinical-note-derived患者级参考训练standard-19坐标carrier中的
  18-evaluable-channel SOZ candidate ranker；PZ仅作未验证/敏感性坐标；
- 使用私有医生综合EEG+semiology参考进行历史复用队列上的冻结描述性迁移评价；
- 研究独立形态和显式时间演变证据是否足以支持临床 SOZ 假设，并把未过门的 ictal
  involvement 分支报告为阴性实验；
- 输出可审计的channel/region排序、证据贡献和不确定性；
- 在患者隔离和pretraining-exposed条件下评价公共TUSZ/DeepSOZ泛化。

### 不能声称

- TUSZ提供原生SOZ channel ground truth；
- DeepSOZ是独立于TUSZ的新数据集；
- 607条记录是607个独立SOZ标签；
- private significant是SEEG或手术结局确认的绝对cortical SOZ；
- earliest scalp-visible channel等于SOZ；
- spread electrode必然是SOZ positive或绝对non-SOZ；
- TCP双极阳性可以拆成端点SOZ；
- `A1/A2`或`SPHL/SPHR`可以替代颞叶标准19导；
- 当前私有队列是未触碰的外部确认性test；
- 模型可直接决定侵入性监测或手术范围。

## 11. 冻结决策清单

- [x] 主目标改为临床综合SOZ参考的标准19导预测，不再以earliest为最终target。
- [x] DeepSOZ定义为TUSZ上的patient-level clinical-note overlay。
- [x] 607仅为唯一映射record数，训练/loss/bootstrap单位为patient。
- [x] 方法中已冻结private significant的预期语义为event-level clinical reference positive。
- [x] private退出当前正式流水线的训练、DAPT、校准、阈值和模型选择；历史复用影响单独披露。
- [x] spread独立保存并单独评价。
- [x] TUSZ标注保持TCP bipolar edge-time坐标。
- [x] raw TUSZ EDF直接读取标准19物理referential channels。
- [x] 不从TCP22反演standard19，不把edge positive展开到endpoint。
- [x] `A1/A2/OZ/SPHL/SPHR`不静默映射进19导head。
- [x] 11名DeepSOZ冲突患者quarantine；canonical PZ主分析mask。
- [x] 审计官方positive-set标签粒度、neighborhood-2/4实现分歧和重复PZ列。
- [x] exact membership、official-neighborhood-4、official-neighborhood-2与粗粒度区域终点分开报告。
- [x] 当前102人冻结为反复使用的public development benchmark，不再承担新结构选择。
- [x] patient split、patient-level loss和patient-cluster bootstrap。
- [x] 方法层冻结 DeepSOZ official-benchmark complement 与 medical-positive-only 双视图。
- [x] 生成并审计 patient-level target artifact v2；现有 positive-only v1 仍不得进入 BCE/ranking。
- [x] 保留102人/988-event legacy信号preflight、crosswalk、EDF hash和timeline闭环；两遍strict replay一致。
- [x] 完成124→114 signal universe→102 current localization primary患者流，并确认current 102/1,145与legacy 102/988仅交集101人。
- [x] 双roster报告fail closed：258不显示定位，10489不伪造event facts。
- [ ] 医生签署private significant字段语义与穷尽性合同。
- [ ] 对private工作簿冲突、significant/spread重叠和outside-head positives完成冻结裁决。
- [ ] 新时间段/外部患者用于未来确认性验证。

## 12. 证据与参考文献

### 本地和上游证据

- [DeepSOZ公开manifest](https://github.com/deeksha-ms/DeepSOZ/blob/913c921f8a08fa4df76ca0708126f565860f1068/data/TUH_manifest_final.csv)
- [DeepSOZ--TUSZ患者级split审计](../../../outputs/deepsoz_tusz_patient_splits_v1/README.md)
- [DeepSOZ closed signal-preflight artifact](../../../outputs/deepsoz_signal_preflight_v1_20260808/deepsoz_signal_preflight.json)
- [DeepSOZ官方19导读取实现](../../../../DeepSOZ/code/preprocess/utils_preprocess.py)
- [私有标签原始审计schema](../../../reports/schemas/eeg_soz_gold_annotation.schema.json)

### 医学和数据集文献

1. Rosenow F, Lüders HO. Presurgical evaluation of epilepsy. *Brain*.
   2001;124:1683--1700. doi:10.1093/brain/124.9.1683.
2. Jayakar P, et al. Diagnostic utility of invasive EEG for epilepsy surgery.
   *Epilepsia*. 2016;57:1735--1747. doi:10.1111/epi.13515.
3. Casale MJ, et al. The sensitivity of scalp EEG at detecting seizures--a
   simultaneous scalp and stereo EEG study. *J Clin Neurophysiol*.
   2022;39:78--84. doi:10.1097/WNP.0000000000000739.
4. Kane N, et al. A revised glossary of terms most commonly used by clinical
   electroencephalographers. *Clin Neurophysiol Pract*. 2017;2:170--185.
5. Beniczky S, et al. SCORE--Second version. *Clinical Neurophysiology*.
   2017;128:2334--2346.
6. Shah V, et al. The Temple University Hospital seizure detection corpus.
   *Front Neuroinform*. 2018;12:83. doi:10.3389/fninf.2018.00083.
7. Shama DM, Jing J, Venkataraman A. DeepSOZ. *MICCAI*. 2023:184--194.
   doi:10.1007/978-3-031-43993-3_18.
8. Seeck M, et al. The standardized EEG electrode array of the IFCN.
   *Clinical Neurophysiology*. 2017;128:2070--2077.
9. Barba C, et al. Grading system for assessing the confidence in the
   epileptogenic zone reported in published studies: A Delphi consensus study.
   *Epilepsia*. 2024. doi:10.1111/epi.17928.
10. Brookshire G, et al. Data leakage in deep learning studies of
    translational EEG. *Front Neurosci*. 2024;18:1373515.
    doi:10.3389/fnins.2024.1373515.
