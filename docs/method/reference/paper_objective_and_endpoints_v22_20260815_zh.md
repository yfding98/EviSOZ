# v22.10 论文目标与评价终点合同

**日期：** 2026-08-15  
**性质：** 已开标后的论文定位修订，不是前瞻性预注册  
**适用模型：** `NEB-LaBraM-v21.1` 已执行系统及其后续冻结复核  
**核心任务：** 标准19导头皮空间中的可信 SOZ 候选推理

## 1. 论文主目标

论文不再以 private strict Top-1 `80%` 或 neighborhood-4 `85%` 作为成功承诺。主目标冻结为：

> 从预训练 EEG foundation representation 出发，对临床 SOZ reference 生成可信的头皮电极候选；
> 对未通过原生任务验证的 concept 作阴性 qualification；对低置信样本明确弃权；并使报告中的每个
> 可核查陈述都能回指版本化 typed fact、模型收据和 claim boundary。

这一定义包含四个不可相互替代的贡献轴：

1. **SOZ candidate reasoning：** 在 C18 可评价电极上输出排序和有限候选集；
2. **negative concept qualification：** 未通过冻结 gate 的 M/I family 结构性缺席，不以零张量或未经验证的分数伪装为有效 concept；
3. **selective abstention：** benchmark 保留 full-coverage 排名，临床显示层在冻结 confidence gate 失败时不显示候选；
4. **traceable reporting：** 每个报告句子都有 fact ID、来源、阈值/版本和允许措辞，LLM 不参与 SOZ 打分。

目标不是 cortical SOZ、EZ、SEEG onset contact、切除区或手术靶点预测。

## 2. 数据单位与评价队列

| 队列 | 实际规模 | SOZ reference 粒度 | 论文角色 |
|---|---:|---|---|
| DeepSOZ source manifest | 652 records / 124 patients | patient × standard-19 electrode | 标签来源清单；record 不是独立病例 |
| 本地唯一映射 | 607 records / 1,566 seizure events | 仍共享 patient-level reference | 信号可追溯量；不得作为独立 SOZ 样本量 |
| target-independent signal universe | 114 patients / 1,364 eligible events | 尚未与稳定、完整C18 target相交 | 输入可用性与失败流审计；不是SOZ评价队列 |
| current localization primary（identity-v16） | 102 patients / 1,145 eligible events | patient × C18 positive set | 当前H-only定位、72.55%与risk--coverage的分母 |
| legacy event-evidence core | 另一组102 patients / 988 events | patient target + event typed facts | 旧逐事件证据与v22 event report；不是当前定位分母 |
| private primary | 51 evaluable events / 23 patients | event × clinician-integrated significant electrode set | 历史开标的描述性迁移队列 |
| private report roster | 88 target-blind signal events | 报告路由不读 target | 候选/弃权与报告完整性审计 |

论文不得写“600例 TUSZ 患者”或“988个独立 SOZ 样本”。DeepSOZ 是 TUSZ/TUH 信号上的
annotation overlay，不是另一个 EEG 数据集。

### 2.1 “600例 TUSZ 子集”的正式表述

“600例”不能出现在摘要、主结果表或贡献声明中，因为本地可追溯的量是 `607 records`，不是
607名独立患者，也不是607个彼此独立的SOZ标签。论文统一写作：

> DeepSOZ--TUSZ public-development cohort：公开清单含124名患者的652条记录；其中607条记录可唯一
> 映射至本地TUSZ版本。target-independent重放得到114名患者、1,364次合格事件；与稳定且完整的C18
> reference相交后，当前定位主队列为102名患者、1,145次事件。旧逐事件typed-fact/report core也含
> 102名患者、988次事件，但这是另一套roster；两者仅交集101名患者。SOZ监督、主评价和bootstrap
> 单位均为患者。

数据流程图可以分别报告607条映射记录、1,364次target-independent合格事件、1,145次当前定位输入
和988次legacy报告证据，但所有当前定位置信区间、split、bootstrap和主性能分母必须以identity-v16
的102名患者为单位。不得按相同患者数将两个102人roster逐行拼接。patient `258`仅有legacy event
reports、因O2 reference不完整而定位不可用；patient `10489`仅有当前患者级定位、无legacy event
facts，因而不得伪造逐事件时间/形态事实。

该患者流与双roster结论由
[`v22_4_deepsoz_124_to_102_patient_flow_and_roster_audit_20260815_zh.md`](v22_4_deepsoz_124_to_102_patient_flow_and_roster_audit_20260815_zh.md)
覆盖此前把`102/988`直接称作当前定位队列的历史表述。

## 3. 性能目标：期望，不是承诺

论文采用分层终点，避免把可信性贡献与一个事后准确率门槛混成同一件事：

| 层级 | 终点 | 论文地位 |
|---|---|---|
| 主要科学目标 | 候选推理、阴性qualification、弃权、逐句追溯四轴闭环 | 论文主张；分别验收，不由单一accuracy替代 |
| 主要定位报告 | full-coverage strict positive-set Top-1，并列AP/MRR/Hit@K与远端错误 | 必报；不设80%/85%承诺 |
| DeepSOZ对齐终点 | full-coverage official-neighborhood-4 | 次级benchmark终点；`>0.70`仅为描述性实用性目标 |
| 竞争性比较假设 | public patient-level full-coverage official-neighborhood-4 `>0.744` | 探索性期望而非承诺；当前portable主臂未达到，需新队列确认 |
| selective终点 | risk--coverage曲线及冻结工作点 | 支持弃权主张；不得替换full coverage |
| private迁移 | event-level strict/relaxed、patient-cluster区间 | 次级跨域、跨粒度描述；不是确认性外部验证 |

### 3.1 描述性实用性目标

希望冻结模型在 public 和 private 上分别达到：

```text
full-coverage official-neighborhood-4 point estimate > 0.70
```

这是一个**描述性实用性目标**，不是临床安全阈值、统计保证或模型选择承诺。它必须与以下结果同时报告：

- strict positive-set Top-1；
- 分子/分母和评价单位；
- coverage 与 abstention rate；
- patient-macro 或 patient-cluster sensitivity analysis；
- 95% confidence interval；
- AP、MRR、Hit@K、far/contralateral-far error；
- 标签粒度、PZ/unknown/conflict 和邻接图版本。

当前 portable H-only 的 full-coverage neighborhood-4 为 public `74/102=72.55%`、private
`37/51=72.55%`，因此两个**点估计**均超过 70%。但 public patient bootstrap 95% 区间为
`63.73%--80.39%`，private patient-cluster 95% 区间为 `60.87%--82.61%`；不能声称真实性能
下界超过 70%。对应 strict 仅为 `47/102=46.08%` 和 `21/51=41.18%`，必须置于同一主结果表。

### 3.2 与 DeepSOZ 的对照

DeepSOZ 论文报告的是 patient localization accuracy `0.744±0.058`。其官方 final notebook
实现为：预测落入 positive set，或在 positive count `<=4` 时落入任一 positive 的一跳邻域。
它不是 strict electrode Top-1。

因此：

- “超过 70%”只表示超过本次论文目标合同写明的 `0.70` 描述性点估计下限；
- “超过 DeepSOZ”至少应面对论文中心值 `0.744`，不能把它简写成 70%；
- 只有 public patient-level、绑定相同 neighborhood-4 实现的结果可作近似数值对照；
- roster、PZ 政策、预处理和交叉验证仍不同，所以即使点估计高于 `0.744` 也不能直接写成复现或统计优越。

历史 v21.1 selective 工作点是在已反复使用的 public OOF 结果上，以 `>0.744` 作为开发选择条件得到：
public `61/81=75.31%`（coverage `79.41%`），同一绝对阈值迁移到 private 为
`33/43=76.74%`（coverage `84.31%`）。这两个点估计可以描述为“高于论文中心值”，但：

1. public 数值参与了阈值选择，不能作为确认性 superiority test；
2. private 是 event-level 且已历史开标，不能与 DeepSOZ patient accuracy 直接比较；
3. private selective patient-cluster 95% 区间 `62.32%--88.41%` 包含 `0.744` 以下值；
4. selective accuracy 不能替代 full coverage 或隐藏弃权病例。

v22.3又在**不选择新阈值**的条件下完成每个coverage点的risk--coverage审计。冻结工作点的
`abstained risk - accepted risk`点估计为：public strict/relaxed `+0.1005/+0.1340`，private
`+0.1919/+0.2674`；但四个patient/patient-cluster bootstrap 95%区间均跨0。因此当前只能主张
“弃权错误富集趋势”，不能主张margin显著降低错误、已经校准或提供conformal/clinical risk guarantee。
完整定义和AURC见
[`v22_3_selective_risk_and_traceable_report_evaluation_20260815_zh.md`](v22_3_selective_risk_and_traceable_report_evaluation_20260815_zh.md)。

v22.5又在不读target、不改报告的条件下审计event facts与患者级候选的描述性侧别关系：438个可比较
事件中同侧211、跨侧张力227；68名有可比较事件且显示候选的患者中，40名同时存在同侧与跨侧事件。
这不是定位正确率或临床事实性，但明确否定了将旁路event facts包装成H-only分数因果解释的做法。
跨层状态只可用于透明性和未来reader-study分层，不得用于当前候选路由、弃权或调参。详见
[`v22_5_cross_layer_evidence_candidate_concordance_audit_20260815_zh.md`](v22_5_cross_layer_evidence_candidate_concordance_audit_20260815_zh.md)。

v22.6将full-coverage neighborhood-4拆成互斥错误层级：public为47 exact、27 neighbor-only、28 far，
其中22个far为跨侧；private为21 exact、16 neighbor-only、14 far，其中9个跨侧，另有2次Top-1落在
known-spread电极。两个队列Hit@5均为80.39%，但要达到该覆盖需显示5个C18候选，不能解释为Top-1
80%。因此论文主表必须加入neighbor-only、far、contralateral-far、known-spread Top-1和候选负担。
详见
[`v22_6_full_coverage_ranking_and_distance_error_audit_20260815_zh.md`](v22_6_full_coverage_ranking_and_distance_error_audit_20260815_zh.md)。

v22.7进一步显示冻结margin不是稳定的危险错误筛分器。Public accepted/abstained contralateral-far为
17/81与5/21，rate gap仅+2.82%且区间跨0；abstained Hit@5反而更高。Private为5/43与4/8，探索性
gap +38.37%，但只有8个abstentions、指标post-hoc且public未复制。两次private known-spread Top-1
均通过margin门。因此弃权只能称“可审计的保守显示策略和总体far-error富集趋势”，不能称临床安全门。
详见
[`v22_7_selective_distance_error_and_candidate_burden_audit_20260815_zh.md`](v22_7_selective_distance_error_and_candidate_burden_audit_20260815_zh.md)。

若把“超过DeepSOZ”定义为同规则下超过其论文中心值，则当前portable H-only full-coverage public结果
`0.7255`**尚未达到**；private的`0.7255`因评价单位为event也不能用于这一比较。public selective
`0.7531`和private selective `0.7674`只说明在明确coverage下的候选一致性，不能转写成full-coverage
superiority。历史public v17的`0.7647`及本地published-weight transfer的`0.7451`可放在探索性对照表，
但因队列反复用于开发且实现/roster并非论文原始设置，仍不能支持统计优越主张。

### 3.3 冻结后的双层性能合同

“结果不错”必须拆成两个不能混写的层级：

```text
最低描述性实用性：public/private各自full-coverage neighborhood-4 > 0.70
竞争性比较假设：仅public patient-level同规则full-coverage点估计 > 0.744
```

第一层当前为`0.7255/0.7255`，已达到点估计要求；第二层当前portable public为`0.7255`，尚未达到。
任何选择性结果都必须同时写coverage，不能用`0.7531/0.7674`替代第二层。private因reference粒度为
event、队列已历史开标，只能检验跨域描述性一致性，不能判定是否超过DeepSOZ。论文不以未达到第二层
否定可信性主贡献，也不把第一层包装成SOTA胜出。

### 3.4 标签与不确定性合同

当前 DeepSOZ 必须同时保留 `benchmark_binary_view` 与 `medical_positive_only_view`：显式 `0` 只在
前者中作为 `dataset-complement negative`，不是医生确认的 non-SOZ；blank/PZ conflict 始终 mask。
多个阳性电极是可接受集合，不使用单真类 partial-label disambiguation。未来新队列采用
`candidate_positive / reviewed_not_candidate / unknown_not_reviewed /
unavailable_signal_or_reference` 四态 ledger，spread 单列；unknown/unavailable 不进入负类或集合
softmax分母。当前102人和private均不得用于选择新的PU、robust loss或set loss。

不确定性分别报告为 model epistemic disagreement、scalp observability、reference/reader uncertainty
和 calibrated candidate-set risk。当前只实现未校准margin显示/弃权；它不能合成或替代上述四项，
也不是正确概率。完整方法边界见
[`v22_10_partial_noisy_label_set_target_and_uncertainty_protocol_20260815_zh.md`](v22_10_partial_noisy_label_set_target_and_uncertainty_protocol_20260815_zh.md)。

## 4. 四个主张轴的验收方式

| 主张轴 | 必须给出的证据 | 当前状态 |
|---|---|---|
| Candidate utility | full coverage + risk--coverage；strict / neighbor-only / far / contralateral-far与Hit@K并列 | relaxed点估计>70%，但strict<50%；public/private far均27.45%，跨侧far为21.57%/17.65% |
| Negative qualification | native-task gate、比较器、CI、失败后结构性移除 | M/I 已失败并移除；阴性结果成立 |
| Abstention | target-blind confidence、冻结阈值、完整risk--coverage/AURC、距离错误、reason code、弃权不显示排名 | 已物化；总体far富集点估计有利，但public跨侧筛分近零、known-spread错误仍通过，不是clinical safety/calibrated risk guarantee |
| Traceability | sentence-to-fact 100% 映射、越界措辞审计、独立医生逐句事实性 | 机器层已闭环；跨层同侧仅211/438且患者内常混合；医生事实性未完成；typed facts 不构成 H-only 因果解释 |

“concept qualification 阴性”不是“没有做成方法”。它证明系统会拒绝不满足证据门的分支；但论文必须
明确，当前 SOZ 分数来自 H-only latent representation，不能将旁路 typed facts 描述成分数的因果解释。

## 5. 论文可写与不可写的结论

### 可写

- 两个队列的 full-coverage neighborhood-4 点估计均为 72.55%，达到本次已开标后论文修订所声明的 `>70%` 描述性目标；
- public的72.55%分解为47 exact + 27 neighbor-only，private分解为21 + 16；两个队列Hit@5均80.39%，但需显示5个候选；
- public-only 冻结的 selective gate 在报告 coverage 后取得 75.31% public 和 76.74% private relaxed concordance；
- 系统对未资格化 concept fail closed，并在低 margin 时显式弃权；两个队列均观察到弃权错误富集点估计，但尚未统计确认；
- margin门不被描述为稳定筛除跨侧或known-spread错误；public accepted仍有17/81跨侧far，private accepted有5/43跨侧far和2/43 known-spread Top-1；
- 报告在机器字段层逐句可追溯，且 LLM 不作为 SOZ predictor；临床事实性仍待独立医生评价。
- event facts与患者候选的同侧/跨侧张力被完整保留，跨层比较不参与候选路由或准确率计算。

### 不可写

- “在600例/988例独立患者上验证”；
- “strict SOZ localization 超过70%”；
- “显著优于 DeepSOZ”或“外部验证证明超过 DeepSOZ”；
- “selective 76% 等同于100% coverage准确率”；
- “弃权显著降低临床错误”“margin是正确概率”或“当前系统具有conformal风险保证”；
- “头皮电极候选等同皮层 SOZ/EZ 或手术靶点”；
- “M/I/V 三个 concept 均成功并因果解释最终分数”。

## 6. 顶刊/顶会最小结果表

主文必须至少包含：

| Cohort/arm | Unit | N | Coverage | Strict Top-1 | Neighbor-only | Far/contralateral-far | Hit@3/5 | 95% CI | Abstain |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|

并配套：

1. full-coverage 主表；
2. 完整 risk--coverage 曲线，而非只报一个好看的 selective 点；
3. published DeepSOZ weights 的同 roster 本地迁移；
4. patient-cluster bootstrap；
5. M/I 阴性 qualification 表；
6. report traceability 与 prohibited-claim audit；
7. fresh same-endpoint cohort 尚缺失的明确限制。

若未来要把“>70%”升级为确认性结论，或检验`>0.744`竞争性比较假设，必须在新的label-fresh、
patient-level、同C18 reference和同neighborhood-4实现队列上预先冻结模型、阈值、邻接图、
PZ/unknown政策和统计检验；现有102人及已开标private只能提供开发性/描述性证据。
