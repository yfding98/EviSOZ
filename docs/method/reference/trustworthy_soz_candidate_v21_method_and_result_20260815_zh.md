# v21：可信 SOZ 候选推理、阴性 concept qualification、弃权与可追溯报告

**日期：** 2026-08-15  
**论文目标状态：** 已替代 80% strict / 85% relaxed 承诺  
**主干：** official pretrained LaBraM-Base，冻结；不从头训练 foundation model  
**主张边界：** scalp-electrode clinical-reference candidate，不是 cortical SOZ/EZ 或手术靶点

> **v22论文目标覆盖：** v21.1仍是已执行方法；论文贡献改按“候选推理—阴性qualification—
> 弃权—逐句追溯”组织。`>0.70`只作为两个队列full-coverage neighborhood-4的描述性点估计目标，
> 不作为strict、临床保证或DeepSOZ superiority claim。详见
> [`paper_objective_and_endpoints_v22_20260815_zh.md`](paper_objective_and_endpoints_v22_20260815_zh.md)。

> **v22.4 roster覆盖：** 当前H-only定位、full-coverage 72.55%、selective与risk--coverage绑定
> identity-v16的102 patients/1,145 events。旧102 patients/988 events是另一套legacy event-evidence/
> report core，两者交集101 patients。patient 258只保留legacy event reports且定位不可用；patient
> 10489只保留patient-level定位且无逐事件facts。历史`signal-closed core=102/988`措辞不得再解释为
> 当前定位分母。详见
> [`v22_4_deepsoz_124_to_102_patient_flow_and_roster_audit_20260815_zh.md`](v22_4_deepsoz_124_to_102_patient_flow_and_roster_audit_20260815_zh.md)。

> **v22.5跨层阴性结果：** target-free event facts与冻结患者级candidate在438个可比较事件中同侧
> 211、跨侧张力227；68名有可比较事件的显示候选患者中40名两种关系并存。该统计不是accuracy，
> 但进一步证明旁路event facts不能解释H-only score，且不得作为新的融合、路由或弃权依据。详见
> [`v22_5_cross_layer_evidence_candidate_concordance_audit_20260815_zh.md`](v22_5_cross_layer_evidence_candidate_concordance_audit_20260815_zh.md)。

> **v22.6 full-coverage分解：** public 72.55%=47 exact+27 neighbor-only，另有28 far（22跨侧）；
> private 72.55%=21+16，另有14 far（9跨侧、2次known-spread Top-1）。两个队列Hit@5均80.39%，
> 但需显示5个候选，不能替代strict Top-1。详见
> [`v22_6_full_coverage_ranking_and_distance_error_audit_20260815_zh.md`](v22_6_full_coverage_ranking_and_distance_error_audit_20260815_zh.md)。

> **v22.7 margin边界：** public accepted/abstained跨侧far=`17/81`与`5/21`，rate gap仅+2.82%且
> 区间跨0；private=`5/43`与`4/8`虽呈富集，但只有8个弃权事件、post-hoc且public未复制。Private
> 两次known-spread Top-1均通过margin。因此该门是保守显示策略，不是validated clinical safety
> detector。详见
> [`v22_7_selective_distance_error_and_candidate_burden_audit_20260815_zh.md`](v22_7_selective_distance_error_and_candidate_burden_audit_20260815_zh.md)。

## 1. 新论文问题

论文不再把“达到 private strict 80% 或 neighborhood-4 85%”作为成败定义。新的主要问题是：

> 在低密度头皮 EEG、弱/粗粒度临床参考和跨中心域偏移下，能否建立一个会拒绝不合格 evidence family、会对低置信病例弃权、并能逐句追溯证据的 SOZ 候选排序系统？

四个并列贡献为：

1. **可信候选排序：** 输出 standard-19/C18 电极支持度和有限候选集；
2. **阴性 concept qualification：** M/I 或其他 evidence 未过原生/迁移门时结构性缺席；阴性结果是方法结果；
3. **弃权：** benchmark 仍报告 full coverage；临床层只在冻结的 selective gate 通过时显示定位候选；
4. **可追溯报告：** 每个时段、边、频谱描述、候选、reason code 和限制都来自 typed facts，LLM 不预测 SOZ。

## 2. “600例”和 DeepSOZ 70%的口径纠正

这些数字不能直接作为独立样本数或 strict accuracy：

| 名称 | 真实单位与角色 |
|---|---|
| DeepSOZ source manifest | 652 records / 124 patients；同一患者多 records 共享 patient-level reference |
| 唯一映射 records | 607；仍不是607个独立SOZ标签 |
| target-independent signal universe | 114 patients / 1,364 events；不是SOZ评价分母 |
| current localization primary | identity-v16 102 patients / 1,145 events；SOZ loss和主评价单位是患者 |
| legacy event-evidence core | 另一组102 patients / 988 events；只提供旧typed facts与event reports；与定位roster交集101人 |
| DeepSOZ论文 `0.744±0.058` | patient localization accuracy；官方最终notebook含positive set≤4时的一跳邻近命中 |
| strict set-membership Top-1 | argmax必须在reference-positive set；与论文0.744不是同一计分 |

所以论文不能写“在988例上达到X%”，也不能把 relaxed 结果写成 strict channel accuracy。

## 3. v21实际模型

当前可跨域部署的最小模型为：

```text
standard-19 EEG [-12,+48) s
  -> causal preprocessing, 200 Hz, CAR19
  -> frozen official LaBraM block-9 carrier H
  -> five patient-excluded low-capacity H-only reasoners
  -> equal probability ensemble over C18
  -> full ranking for benchmark
  -> margin-based selective gate
       pass: candidate ranking + uncertainty reason codes
       fail: clinical localization abstains
  -> deterministic typed-fact report
```

它不声称已经使用三个成功的概念分支：

- M morphology producer未过局灶precision gate；
- I involvement producer在formal-v5和fresh official-dev transport均失败；
- V只允许做observable evolution描述，不能表示传播或起源；
- fine temporal descriptors在public可作为开发性辅助，但private显示明显标准化偏移，故不进入portable v21主臂。

因此，当前候选分数是H-only latent foundation-feature prediction。typed facts能证明报告每句话
来自哪个事件、边、时段或模型收据，但不能被描述为这些concept对H-only分数的因果解释；真正的
concept-only sufficiency/faithfulness仍是未来实验。

曾尝试用source-patient q99统计做fine-family无标签域门。该门被少数source极端患者撑宽，错误放行private，因而被保留为阴性gate-design结果，**不作为模型路由器**。这避免根据private H-only成绩事后收紧OOD阈值。

## 4. 冻结弃权规则

portable H-only对每个样本产生C18概率。置信指标固定为：

\[
m(x)=p_{(1)}(x)-p_{(2)}(x).
\]

只在102名public patient-OOF预测上考察预定义margin分位点
`{0,10,20,30,40,50}%`。选择规则是在至少保留50名患者的前提下，找到
`official-neighborhood-4 > 0.744`的**最高覆盖率**工作点；不得根据private更改。

得到：

```text
absolute margin threshold = 0.0339790881
public retained           = 81/102 = 79.41%
public abstained          = 21/102 = 20.59%
```

private使用同一个绝对阈值。阈值应用到88个target-blind signal events后才读取既有评价rows；71/88信号事件通过，primary分母中43/51通过。没有按private标签选择coverage。

该阈值是developmental selective operating point，不是Conformal Risk Control保证。102人被反复开发，public patient-level与private event-level也不exchangeable，故不能把margin解释为“错误概率”。

### 4.1 Target-blind候选/弃权报告

冻结阈值现已独立物化为报告协议。报告器只读取H-only score、candidate mask、public patient roster、
private target-blind event roster和绝对阈值；它不加载SOZ target tensor、private target ledger或
evaluation rows。结果为public `81/102`显示候选、`21/102`弃权，private全部88个信号事件中
`71/88`显示候选、`17/88`弃权。弃权记录的`displayed_candidates=[]`，reason code固定为
`top1_top2_margin_below_frozen_threshold`，并明确“弃权不表示无SOZ”。每个临床句子保存
`sentence_fact_map`。这一层只完成候选/弃权决策条款；既有event evidence正文仍由独立typed-fact
artifact提供，不能据此声称concept对H-only分数具有因果贡献。

## 5. 结果

### 5.1 Full coverage必须保留

| Arm / cohort | Unit | Coverage | Strict Top-1 | Neighborhood-4 |
|---|---|---:|---:|---:|
| published DeepSOZ weights，本地102人 | patient | 100% | 48/102 = 47.06% | 76/102 = 74.51% |
| LaBraM-v17 full，public | patient | 100% | 51/102 = 50.00% | 78/102 = 76.47% |
| portable H-only，public | patient | 100% | 47/102 = 46.08% | 74/102 = 72.55% |
| v18 H+fine，private | event | 100% | 21/51 = 41.18% | 34/51 = 66.67% |
| portable H-only，private探索性 | event | 100% | 21/51 = 41.18% | 37/51 = 72.55% |

public v17点估计高于DeepSOZ论文0.744和本地官方权重0.7451，但配对区间不证明显著优越。portable H-only在public/private full coverage均超过70%，但public未超过论文0.744点估计。

### 5.2 Selective candidate工作点

| Cohort | Retained | Coverage | Strict | Neighborhood-4 |
|---|---:|---:|---:|---:|
| public patient-OOF | 81/102 | 79.41% | 39/81 = 48.15% | 61/81 = 75.31% |
| private primary event | 43/51 | 84.31% | 19/43 = 44.19% | 33/43 = 76.74% |

private patient-macro为strict `42.03%`、neighborhood-4 `76.09%`。private relaxed的patient-cluster bootstrap 95%区间为`62.32%--88.41%`；event-micro Wilson 95%区间为`62.26%--86.85%`。因此点估计同时超过70%及论文0.744，但区间不能支持统计优越。

Selective accuracy不能替代full-coverage结果，也不能隐藏21个public/8个private abstentions。论文必须同时列出coverage、分子/分母、strict、relaxed和区间。

## 6. 顶刊顶会可支持与不可支持的claim

可以写：

- 在反复使用的public-development benchmark上，v17 relaxed点估计为76.47%；
- 一个只在public OOF上冻结的H-anchor margin gate，在79.41% public coverage和84.31% private event coverage下得到75.31%/76.74% relaxed concordance；
- M/I分支和一个过宽的fine OOD gate均产生可审计阴性结果，系统按规则回退/弃权；
- 所有报告句子可回指typed facts，且明确缺少侵入式确认。

不能写：

- “在600/988名独立患者上超过DeepSOZ”；
- “显著优于DeepSOZ”或“外部验证达到76.74%”；
- relaxed邻域命中等于strict SOZ channel accuracy；
- private用于选择H-only仍属于zero-shot confirmation；
- 最早头皮变化、later-visible或TUSZ involvement证明传播或皮层起源。

推荐标题方向：

> **Fail-Closed Evidence Qualification and Selective Scalp-EEG SOZ Candidate Reasoning with Auditable Clinical Reports**

这比“新网络把SOZ准确率提升到某个固定数字”更符合当前真正创新与证据强度。

## 7. 机器证据

- v21 H-only/full-coverage与fine q99阴性门：`outputs/trustworthy_soz_candidate_v21_20260815/result.json`
- v21.1 selective结果：`outputs/trustworthy_soz_selective_v21_1_20260815/result.json`
- v21.1冻结协议：`configs/trustworthy_soz_selective_v21_1.json`
- v21.1审计器：`scripts/audit_trustworthy_soz_selective_v21_1.py`
- v21.1 target-blind候选/弃权报告：`outputs/trustworthy_soz_selective_reports_v21_1_20260815/`
- v21.1报告协议/物化器：`configs/trustworthy_soz_selective_reporting_v21_1.json`、`scripts/materialize_trustworthy_soz_selective_reports_v21_1.py`
- DeepSOZ官方终点审计：`deepsoz_official_endpoint_audit_20260813_zh.md`
- private full-coverage正式结果：`labram_private_zero_adaptation_result_v18_20260814_zh.md`
