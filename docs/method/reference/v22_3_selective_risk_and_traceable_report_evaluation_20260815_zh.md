# v22.3 选择性风险与可追溯报告评价协议及冻结结果

**日期：** 2026-08-15  
**适用系统：** `NEB-LaBraM-v22` frozen H-only candidate/abstention/reporting pipeline  
**性质：** post-open描述性审计；不训练、不改分数、不选新阈值、不建立临床风险保证

> **v22.4--v22.5覆盖：** 102份public patient reports绑定current localization primary
> （102 patients/1,145 events），988份public event reports绑定另一套legacy event-evidence roster；
> 两者患者交集101人，不能按同样的102人数假设逐行同一。target-free跨层审计进一步显示，在438个
> “单侧事件证据+显示单侧候选”的可比较事件中，同侧211、跨侧张力227；患者层面40/68同时出现
> 两种状态。这不是SOZ accuracy或临床事实性，而是支持event facts不得因果解释H-only score的阴性
> evidence。详见
> [`v22_4_deepsoz_124_to_102_patient_flow_and_roster_audit_20260815_zh.md`](v22_4_deepsoz_124_to_102_patient_flow_and_roster_audit_20260815_zh.md)
> 与
> [`v22_5_cross_layer_evidence_candidate_concordance_audit_20260815_zh.md`](v22_5_cross_layer_evidence_candidate_concordance_audit_20260815_zh.md)。

> **v22.6错误距离覆盖：** full-coverage public/private far error均为27.45%，其中contralateral far为
> 22/102与9/51；private另有2/51 known-spread Top-1。Hit@5均80.39%但需要5个候选。选择性风险必须
> 与这些全覆盖错误并列，不能用accepted relaxed accuracy隐藏远错或把Hit@5写成Top-1。

> **v22.7选择性距离覆盖：** public accepted/abstained contralateral-far为17/81与5/21，富集仅
> +2.82%且区间跨0；abstained Hit@5反而更高。Private为5/43与4/8，但只有8个弃权事件、分析post-hoc
> 且public未复制；2次known-spread Top-1全部位于accepted。Margin不是稳定危险错误筛分器。详见
> [`v22_7_selective_distance_error_and_candidate_burden_audit_20260815_zh.md`](v22_7_selective_distance_error_and_candidate_burden_audit_20260815_zh.md)。

## 1. 决定性结论

当前系统已经实现了真正的显示层弃权：低于冻结margin阈值时不暴露隐藏候选，并给出reason code；
报告也已由typed facts逐句重建。但这两项工程不变量本身不能证明：

1. confidence能稳定区分正确与错误定位；
2. accepted病例的临床错误率受控；
3. typed-fact句子在原始EEG上具有临床事实性；
4. 旁路event facts因果解释H-only SOZ分数。

本轮在不改变冻结模型和阈值的条件下，补做了完整risk--coverage/AURC审计。结果显示弃权组在点估计上
错误更多，但public与private患者bootstrap区间均跨0。因此冻结结论为：

> **可审计弃权已实现；错误富集呈有利点估计，但尚未统计确认，更不构成calibrated/conformal clinical-risk control。**

## 2. 为什么不能只报告selective accuracy

只报告accepted病例的`75.31%/76.74%`会产生三个问题：

- 任意系统都可通过多拒绝病例提高conditional accuracy；
- public阈值是在已经反复使用的102人OOF结果上选择，不是独立calibration；
- private已经历史开标，且评价单位是event，不能成为阈值确认或DeepSOZ patient-level优越性证据。

SelectiveNet说明reject option必须与明确coverage约束共同评价；Ovadia等表明常用confidence在真实dataset
shift下可失效；医学AI文献进一步要求把不确定/转诊输出放回实际工作流，而不是只展示一个高分子集。
因此本项目把full coverage、risk--coverage、accepted/abstained病例流和报告事实性拆成四项独立证据。

## 3. 冻结风险审计定义

### 3.1 Confidence与排序

对每个样本，在固定C18 mask内计算：

```text
c_i = softmax_top1_i - softmax_top2_i
```

`c_i`只用于排序和冻结阈值路由。它不是`P(correct)`，也没有单位为错误概率的校准含义。

按`c_i`从高到低排序。对保留前`k`个样本：

```text
coverage(k) = k / n
risk(k)     = 1 - mean(correct_1 ... correct_k)
AURC        = mean_k risk(k)
eAURC       = AURC - oracle_AURC(same labels)
```

strict和official-neighborhood-4分别计算。AURC/eAURC越低越好，但不能跨不同base error、不同评价单位
或不同endpoint直接比较。曲线包含每个可观察coverage点，不再从曲线上另选工作点。

### 3.2 冻结工作点

唯一工作点继续使用v21.1 public-only历史选择的绝对阈值：

```text
top1_top2_margin >= 0.03397908806800842
```

审计同时报告accepted risk、abstained risk及：

```text
risk enrichment = abstained risk - accepted risk
```

正值表示点估计方向有利。public以患者bootstrap；private以患者簇bootstrap并保留同一患者全部可评价
事件。5000次bootstrap只估计不确定性，不参与选择模型、阈值或措辞。

## 4. 正式冻结结果

### 4.1 全曲线

| 队列/单位 | Endpoint | Full accuracy | AURC | patient/bootstrap 95% CI | eAURC |
|---|---|---:|---:|---:|---:|
| Public，102 patients | strict | 46.08% | 0.3831 | 0.2673--0.5125 | 0.1983 |
| Public，102 patients | neighborhood-4 | 72.55% | 0.1829 | 0.1035--0.2806 | 0.1399 |
| Private，51 events/23 patients | strict | 41.18% | 0.4808 | 0.2885--0.7072 | 0.2522 |
| Private，51 events/23 patients | neighborhood-4 | 72.55% | 0.1799 | 0.0873--0.3003 | 0.1356 |

AURC低于各自full-coverage risk的点估计，说明margin排序存在有利趋势；eAURC仍明显大于0，说明距离同标签
oracle排序很远。当前样本不能把这种趋势升级为稳定风险排序能力。

### 4.2 冻结工作点的错误富集

| 队列 | Endpoint | Coverage | Accepted accuracy | Abstained accuracy | Risk enrichment | 95% CI |
|---|---|---:|---:|---:|---:|---:|
| Public | strict | 79.41%（81/102） | 48.15% | 38.10% | +0.1005 | -0.1395--0.3361 |
| Public | neighborhood-4 | 79.41%（81/102） | 75.31% | 61.90% | +0.1340 | -0.0941--0.3793 |
| Private | strict | 84.31%（43/51） | 44.19% | 25.00% | +0.1919 | -0.1825--0.5107 |
| Private | neighborhood-4 | 84.31%（43/51） | 76.74% | 50.00% | +0.2674 | -0.1541--0.6787 |

四个点估计方向均有利，但区间全部跨0。故论文只能写“冻结margin在两个队列均观察到弃权错误富集趋势”，
不能写“显著降低错误”“风险受控”或“安全弃权已验证”。Private尤其只有8个可评价abstained events，
区间很宽。

错误距离分解进一步表明，public contralateral-far在accepted/abstained为17/81与5/21，rate gap
`+0.0282`、95%区间`[-0.1695,0.2448]`；margin几乎没有筛分跨侧错误。Private对应5/43与4/8，
gap`+0.3837`、post-hoc patient-cluster bootstrap区间`[0.0071,0.7569]`，但不得称确认性显著：private
已开标、仅8个abstentions、未作前瞻性multiplicity控制且public未复制。Private两次known-spread
Top-1均在accepted组，也证明margin不是spread-aware safety gate。

## 5. 论文必须报告的选择性病例流

每个队列同时报告以下分母，禁止删除低质量或弃权病例后重新定义数据集：

```text
enrolled/source
  -> target available / unavailable / outside-head
  -> signal eligible / preprocessing unavailable
  -> full-coverage ranking evaluable
  -> display candidate / low-margin abstain / signal unavailable
  -> strict correct / neighborhood-only correct / far error
```

最低结果集：

1. full-coverage strict、AP/MRR/Hit@K和neighborhood-4；
2. 每个coverage点的strict/relaxed risk；
3. AURC、eAURC及患者级区间；
4. 冻结工作点accepted/abstained分子分母；
5. SOZ区域、event数量、参考方式稳定性、signal quality和label/reference状态分层的coverage；
6. far-error在accepted与abstained组的分布；
7. 每个abstention reason code的数量。

亚组样本不足时只显示计数和区间，不做多重事后显著性筛选，不由亚组结果修改阈值。

## 6. 为什么现在不能套用conformal风险保证

Distribution-free risk-controlling prediction sets、Learn-then-Test和conformal risk control要求预先固定loss、
候选规则、校准/测试关系及相应交换性或分布条件。当前不满足：

- 102名public患者已用于模型、表示、恢复策略和margin门开发；
- private已开标且存在跨域、跨粒度差异；
- margin没有在独立同终点calibration patients上校准；
- full-coverage strict error仍高，不能从relaxed selective点估计反推临床风险。

只有未来lineage-new C18队列才能按以下顺序使用风险控制：

```text
development patients: freeze model/confidence/reason codes
independent calibration patients: choose threshold or prediction set under prespecified loss
locked test patients: report achieved risk + coverage + finite-sample bound
external site: test bound degradation; never silently recalibrate
```

若交换性或site shift不成立，只报告经验risk--coverage，不保留distribution-free措辞。

## 7. 可追溯报告必须分成两个验证层

### 7.1 Layer A：机器可追溯性

这层验证“文本是否忠实于系统内部已有字段”，当前已经闭环：

| 指标 | 当前结果 |
|---|---:|
| public patient clauses与fact paths一一对应 | 102/102 reports |
| public event clauses与fact paths一一对应 | 988/988 reports |
| private clauses与fact paths一一对应 | 88/88 reports |
| abstain/unavailable隐藏候选 | 全部通过 |
| forbidden phrase hits | 0 |
| LLM参与SOZ预测/物化 | false |

必须继续报告：`clause_path_coverage`、源字段类型/版本、确定性重建一致性、unsupported slot率、禁用措辞、
候选隐藏和identity/roster join失败数。

这层不能证明句子在原始EEG上正确，也不能证明typed facts解释H-only score。

v22.5对这种“非因果”边界进行了target-free量化：988份event reports中，193份因margin弃权、4份
证据与定位均不可用；其余显示候选的事件中，240份首批双极边为双侧、113份无持续双极变化，只有438
份可进行单侧描述性比较。同侧仅211/438，跨侧张力227/438。事件重复patient-level candidate，故不得
当成独立准确率；在患者层面，15人仅同侧、13人仅跨侧、40人两者混合、12人无可比较单侧事件、22人
弃权/不可用。跨层张力不能自动证明候选错误，同侧也不能证明event facts临床正确。

### 7.2 Layer B：临床事实性和可用性

当前**未完成**，不能用Layer A代替。未来由至少两名癫痫EEG医生在不知道DeepSOZ/private target、模型
correctness和另一名reader意见时独立评价：

| Clause family | Reader可用证据 | 标签 |
|---|---|---|
| acquisition/window | EDF header、event anchor、预处理receipt | supported / contradicted / indeterminate |
| bipolar edge/time | 原始EEG、多蒙太奇、完整前后文 | supported / contradicted / indeterminate |
| rhythm/frequency | 原始EEG、冻结测量定义 | supported / contradicted / indeterminate |
| later-visible | 原始EEG；不显示“propagation” | supported / contradicted / indeterminate |
| artifact | 原始EEG；分type与burden | supported / contradicted / indeterminate |
| candidate/region | 单独按clinical SOZ reference评价 | exact / neighbor-only / far / unknown |
| limitation | 报告文本与target provenance | complete / incomplete / misleading |

每个错误再分`major`（可能改变候选/临床解释）与`minor`。必报逐类precision、contradiction率、indeterminate率、
reader agreement、裁决前后结果、报告审阅时间和医生是否要求额外查看原始EEG。候选准确率与语言事实性
分别统计，不能平均成一个“报告分数”。

没有专家可用时，Layer B保持`not qualified`；LLM不能代替reader生成gold或作为自身报告的事实性裁判。
GREEN等生成式报告评价可作为文献对照，但不适合作为本项目主要事实性终点：领域本体不同，而且自动
生成/自动评估会重新引入循环验证。Patterns 2023对胸片报告评价的审计也说明常规文本相似度不能替代
临床错误评价。

## 8. 阴性concept qualification如何进入结果

“阴性qualification”不是把失败concept输出写成0，而是显式改变信息流和报告schema：

| Family | 当前native证据 | Reasoner | Report |
|---|---|---|---|
| M morphology | focal precision门失败 | structurally absent | 不出现spike/sharp等结论 |
| I ictal involvement | 相对comparator增益门失败 | structurally absent | 不称SOZ onset evidence |
| V temporal | 仅规则性头皮可见描述，未reader-qualified | 不提供origin/propagation | 只允许保守候选措辞或缺席 |
| artifact | 0个已资格化事实 | absent | 明写类型/严重度未形成结论 |

论文必须把每个family的native task、数据、分母、point estimate、95% CI、promotion threshold和失败后动作
放在同一表中。失败family仍保留为阴性结果，不通过换名、LLM补写或连续latent旁路重新进入reasoner。

## 9. 当前允许和禁止的论文主张

允许：

- 实现了target-blind、隐藏候选的低margin弃权；
- 完整risk--coverage曲线的点估计支持confidence ordering有利趋势；
- 两个队列在冻结工作点均出现abstained错误富集点估计；
- 阴性concept qualification会结构性删除分支和报告槽；
- 所有当前报告子句都可回指typed facts/protocol路径。

禁止：

- 弃权显著降低SOZ错误或提供临床安全保证；
- margin是定位正确概率；
- 当前门是conformal/risk-controlling predictor；
- selective 75.31%/76.74%替代full-coverage结果；
- typed facts临床事实性已经由医生验证；
- 报告事实因果解释H-only SOZ score；
- 自动报告可独立用于皮层SOZ/EZ或手术决策。

## 10. 实现和证据产物

- 协议：`configs/trustworthy_soz_risk_coverage_audit_v22_3.json`
- 只读审计：`scripts/audit_trustworthy_soz_risk_coverage_v22_3.py`
- 结果：`outputs/trustworthy_soz_risk_coverage_v22_3_20260815/result.json`
- 测试：`tests/test_trustworthy_soz_risk_coverage_v22_3.py`
- 回归：`4 passed`

审计读取冻结public/private prediction和既有评价行；没有读取raw EEG、训练新模型、选择新阈值或改写报告。

## 11. 方法依据

1. Geifman Y, El-Yaniv R. SelectiveNet: A Deep Neural Network with an Integrated Reject Option. ICML 2019. <https://proceedings.mlr.press/v97/geifman19a.html>.
2. Ovadia Y, et al. Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift. NeurIPS 2019. <https://proceedings.neurips.cc/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html>.
3. Ding Y, et al. Revisiting the Evaluation of Uncertainty Estimation and Its Application to Explore Model Complexity-Uncertainty Trade-Off. CVPR Workshops 2020. <https://doi.org/10.1109/CVPRW50498.2020.00010>.
4. Bates S, et al. Distribution-free, Risk-controlling Prediction Sets. *JACM*. 2021. <https://doi.org/10.1145/3478535>.
5. Angelopoulos AN, et al. Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control. *Annals of Applied Statistics*. 2025. <https://doi.org/10.1214/24-AOAS1998>.
6. Kompa B, et al. Second opinion needed: communicating uncertainty in medical machine learning. *npj Digital Medicine*. 2021. <https://doi.org/10.1038/s41746-020-00367-3>.
7. Collins GS, et al. TRIPOD+AI statement. *BMJ*. 2024. <https://doi.org/10.1136/bmj-2023-078378>.
8. Moons KGM, et al. PROBAST+AI. *BMJ*. 2025. <https://doi.org/10.1136/bmj-2024-082505>.
9. Vasey B, et al. DECIDE-AI. *Nature Medicine*. 2022. <https://doi.org/10.1038/s41591-022-01772-9>.
10. Sounderajah V, et al. STARD-AI. *Nature Medicine*. 2025. <https://doi.org/10.1038/s41591-025-03953-8>.
11. Lekadir K, et al. FUTURE-AI. *BMJ*. 2025. <https://doi.org/10.1136/bmj-2024-081554>.
12. Yu F, et al. Evaluating progress in automatic chest X-ray radiology report generation. *Patterns*. 2023. <https://doi.org/10.1016/j.patter.2023.100802>.
13. Ostmeier S, et al. GREEN: Generative Radiology Report Evaluation and Error Notation. Findings of EMNLP 2024. <https://doi.org/10.18653/v1/2024.findings-emnlp.21>.
