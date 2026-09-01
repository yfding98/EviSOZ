# v22.10 部分/噪声标签、集合目标与分解不确定性协议

**日期：** 2026-08-15  
**适用系统：** 冻结的 `NEB-LaBraM-v21.1/v22`  
**论文目标：** 可信头皮电极 SOZ 候选推理、阴性 concept qualification、低置信弃权和逐句可追溯报告  
**性质：** 已开标后的标签/方法审计及未来队列协议；不是当前 public/private 的新预注册或重训练授权

## 1. 决定性结论

1. **当前 DeepSOZ overlay 必须保留两个并行视图。** 官方二值视图中的显式 `0` 可以复现
   benchmark loss，但只能称 `dataset-complement negative`；它不是医生逐导确认的生理阴性。医学
   positive-only 视图只确认显式 `1`，其他状态均不支持负类语义。
2. **多个 reference-positive 电极构成可接受集合，不是假设其中隐藏唯一真类。** PiCO+、SoLar 的
   单真类 partial-label 候选集设定不匹配；Large Loss Matters 虽处理弱监督多标签漏标，也没有给出
   DeepSOZ 临床记录选择机制。二者都不能用于人为恢复 SOZ gold。
3. **普通 PU 学习也不能自动修复 DeepSOZ complement。** 当前没有可信的逐电极正类先验，也没有
   SCAR/SAR 式标注选择机制；nnPU/PU 风险估计的关键可识别条件不成立。
4. **BUNDL 处理的是发作时间标签噪声，不是 SOZ 空间 reference 噪声。** 它可在新的非 DeepSOZ
   患者上资格化 global seizure-state 或 I producer，但不能修改 C18 target、补造 SOZ 阳性或证明
   SOZ 排名改善。
5. **未来新队列必须使用四态逐电极 ledger 和集合损失。** unknown/unavailable 始终 mask；只有
   `reviewed_not_candidate` 才能作为临床已复核负类。所有 loss、权重和阈值只能在新的 S1-D/S1-C
   上冻结，不能在已反复使用的 public 102 人或已开标 private 上选择。
6. **“不确定性”拆成四个不同对象。** 模型认知分歧、头皮可观测性、reference/reader 不确定性和
   calibrated candidate-set risk 不得压成一个无定义的 confidence。当前 margin 只负责可审计显示/
   弃权，不是正确概率或临床安全保证。

本修订不改变已执行信息流：

```text
standard-19 EEG
  -> frozen official LaBraM block-9 H
  -> five patient-excluded H-only reasoners
  -> C18 full-coverage ranking
  -> frozen margin display/abstain
  -> deterministic typed-fact report
```

M/I 仍因原生资格化失败而结构性缺席；V 只描述 scalp-visible temporal change。Private 仍不得训练、
调参、校准、选模或修改报告措辞。

## 2. 三套标签语义不能混写

### 2.1 当前执行：DeepSOZ benchmark binary view

| 原始状态 | 计算状态 | 是否进入 benchmark loss | 医学解释上限 |
|---|---|---:|---|
| 显式 `1` | `reference_positive` | 是，正类 | 临床记录支持的头皮电极 SOZ reference positive |
| 显式 `0` | `dataset_complement_negative` | 是，负类 | 官方 benchmark 中未列为阳性；不是逐导排除 |
| blank/unparsed | `unknown` | 否 | 未知，不得补成 0 |
| canonical PZ conflict | `schema_conflict` | 否 | 主分析不可评价 |

该视图是当前已执行模型和 published-benchmark sensitivity 的唯一普通 BCE/ranking 入口。论文必须
同时披露：这个计算负类语义来自数据集构造，不是 18 个导联均经医生逐一判为 non-SOZ。

### 2.2 当前敏感性：medical positive-only view

```text
显式 1       -> observed_reference_positive
所有其他状态 -> unknown for negative-sensitive inference
```

该视图允许 patient-level positive recall、MRR、Hit@K 和 positive-set Top-1；不允许 ordinary BCE、
specificity、F1、校准负类概率或“confirmed non-SOZ”结论。它也不能作为 PU 方法的自动入口，因为
unlabeled pool 的阳性比例和标注选择机制均未知。

### 2.3 未来 S1-D/A5：四态临床 ledger

每个 C18 电极必须且只能取一个主状态：

```text
candidate_positive
reviewed_not_candidate
unknown_not_reviewed
unavailable_signal_or_reference
```

并单独保存：

```text
spread_electrode
reference_level
localization_basis
reader_ids / adjudication_status
source_file / row / hash
```

`spread_electrode` 不并入 SOZ positive，也不自动等于 reviewed negative。只有医生明确完成逐导复核后，
`reviewed_not_candidate` 才可进入负类 loss。未列出、未看过、信号缺失或 reference 冲突的电极必须
保持 unknown/unavailable。

## 3. 为什么现成弱标签算法不能直接移植

### 3.1 单真类 partial-label disambiguation 不成立

PiCO+ 和 SoLar 面向的典型 partial-label 设定是：每个样本有一个未观测的真实类别，候选集合包含
该类别，算法再在候选内消歧。DeepSOZ 的 `{F7,T7,P7}` 可能是多个同时可接受的临床 reference
positive，并没有证据说明其中恰有一个“真正标签”。将集合收缩为一个电极会：

- 创造原始标注中不存在的监督；
- 惩罚其余临床可接受正例；
- 把模型自身的 prevalence/拓扑偏好回写为 gold；
- 人为抬高唯一 Top-1 叙事而降低医学有效性。

因此这些方法可作为不匹配假设的文献对照，不能作为当前 target 恢复器。Large Loss Matters 处理的
则是弱监督多标签图像分类中的 partial annotations，并利用大损失寻找潜在漏标；它提醒我们不能把
所有未标注项当真阴性，但没有估计 DeepSOZ 临床记录的逐电极漏提机制，也没有提供 EEG/SOZ reference
验证。将其直接用于当前 C18 会把模型大损失再解释为新阳性，仍有循环确认风险。

### 3.2 PU learning 缺少可识别条件

经典 nnPU 风险估计至少需要可信的正类先验，并对 labeled-positive 的选择过程作 SCAR 或明确的
SAR/selection-bias 建模。当前 clinical-note-derived C18 overlay 的提及概率可能同时受脑区、导联、
montage、报告措辞、病史和多模态证据影响；没有逐电极 class prior，也没有可验证的标注选择概率。

把全部 complement 改成 unlabeled 后，模型可以无代价地抬高所有通道；再用当前模型挑伪负类或估 prior
会形成循环自训练。故当前 public 102 人上不得用 PU 方法重定义标签或选择新模型。

### 3.3 Robust/noisy-label loss 不能决定遗漏方向

大损失样本可能是错标，也可能是少见 extra-temporal、posterior、深部不可见、montage-sensitive 或
真正困难的病例。按 loss 降权/删除会优先丢弃最需要保护的亚群。除非新队列含独立复核的标签噪声
子集，否则不能把 small-loss、co-teaching 或 generalized cross-entropy 的收益解释为 gold 恢复。

## 4. BUNDL 的严格作用边界

BUNDL（PLOS ONE 2026，DOI `10.1371/journal.pone.0352191`）在 120 名 TUH focal-onset 患者、
19 导、200 Hz、1 s 窗上，用 MC-dropout uncertainty 和 Bayesian label transition 修正
seizure/non-seizure **时间区间标签**。论文报告 DeepSOZ patient-level localization 由 CEL 的
`0.591±0.144` 变为 BUNDL 的 `0.620±0.111`，机制是先改变 detection posterior，再改变后续时间
加权；它没有清洗患者级 SOZ 电极标签。

还需保留三个限制：

1. 它没有增加 lineage-new 患者，仍属于 TUH/DeepSOZ 数据谱系；
2. 正文描述 repeated nested 10-fold × 5，但未充分证明所有环节严格 patient-disjoint；
3. 公开训练脚本存在未定义变量/loader及缩进等问题，不能写作 turnkey reproduction。

合法角色只有：

```text
new non-DeepSOZ patients with explicit temporal labels
  -> global seizure-state / boundary or observed-cell I qualification
  -> native AP/AUROC/Brier/boundary/false-alarm evaluation
```

禁止角色为：修正 DeepSOZ C18 complement、生成 onset/origin channel、补充 spread label、将
`0.620` 当成本项目预期准确率，或在当前 102 人/private 上选择新 SOZ 模型。

## 5. 未来新队列的集合目标

对患者 `i` 定义：

```text
P_i = candidate_positive
N_i = reviewed_not_candidate
U_i = unknown_not_reviewed
A_i = unavailable_signal_or_reference
R_i = P_i union N_i
```

每位患者的多次发作先等权聚合为 `s_i,c`。训练 softmax 只在 `R_i` 内归一化；`U_i/A_i` 不进入
分母或任何负类项：

\[
\mathcal L_{set}(i)=-\log
\frac{\sum_{c\in P_i}\exp(s_{i,c}/\tau)}
{\sum_{c\in R_i}\exp(s_{i,c}/\tau)}.
\]

如新 ledger 真正提供 reviewed negatives，可预声明两个辅助目标：

\[
\mathcal L_{masked\ BCE}(i)=
\operatorname{mean}_{c\in P_i\cup N_i}\operatorname{BCE}(s_{i,c},y_{i,c}),
\]

\[
\mathcal L_{pair}(i)=
\operatorname{mean}_{p\in P_i,n\in N_i}
\operatorname{softplus}(m-s_{i,p}+s_{i,n}).
\]

约束如下：

- `L_set` 对多个正例只要求正集合获得概率质量，不强制集合内唯一真类；
- `L_masked_BCE/L_pair` 只能使用明确的 `reviewed_not_candidate`；
- unknown/unavailable 永远 mask，不得采伪负例；
- 每患者只贡献一次 loss，records/events 不能放大患者权重；
- `tau`、`m`、辅助 loss 权重、event aggregation 和 early stopping 在 S1-D 预先选择并冻结；
- S1-C 只校准候选集/弃权，不再改 reasoner；A5 只做一次标签隐藏确认；
- 这些 loss 不得在当前 public 102 人或 private 上重新比较、选择或追溯性声称更优。

当前已执行 benchmark 模型保持原训练合同，不因本节追溯性换 loss。

## 6. 评价合同：集合正确不等于模糊放宽

### 6.1 Full coverage 必报

- patient-equal strict positive-set Top-1；
- AP、MRR、Hit@3/Hit@5 和 positive recall；
- exact / neighbor-only / far / contralateral-far 互斥分解；
- known-spread Top-1；
- prediction-set miss-all、集合大小和医生复核负担；
- patient bootstrap 95% CI、分子/分母和完整患者流。

Strict positive-set Top-1 命中 `P_i` 中任一临床可接受电极，不等于把邻居加入 gold。Neighborhood-4
仍是单独的 DeepSOZ-style secondary endpoint，必须用冻结图，并与 strict 和 far error 同表。

### 6.2 当前性能解释

| Cohort | 单位 | Strict Top-1 | Full-coverage neighborhood-4 | 解释上限 |
|---|---|---:|---:|---|
| public | 102 patients | `47/102=46.08%` | `74/102=72.55%` | 反复使用的开发 benchmark |
| private | 51 events / 23 patients | `21/51=41.18%` | `37/51=72.55%` | 已开标、跨粒度描述迁移 |

两个 relaxed 点估计超过 `0.70`，只支持“达到已开标后冻结的描述性实用性目标”。DeepSOZ 论文实际
patient localization 中心值是 `0.744±0.058`，不是 70%；当前 public full-coverage `0.7255` 没有
超过该中心值，private event-level 也不能用于直接比较。任何 future “超过 DeepSOZ”结论都要求：
同患者、同信号、同 target/邻接/PZ规则的配对比较，或独立同终点队列上的预冻结统计检验。

## 7. 四类不确定性必须分别输出

| 不确定性 | 来源 | 合法计算/字段 | 允许动作 | 当前状态 |
|---|---|---|---|---|
| model epistemic disagreement | 五折/ensemble 对同一 C18 排名分歧 | rank/logit dispersion、Top-1 agreement | 降低显示支持或弃权 | 可描述，未校准为错误概率 |
| scalp observability `Q_obs` | SNR、缺导、montage稳定性、持续时间可见性 | typed quality/availability facts | 只降权、限句或弃权 | producer/calibration未资格化 |
| reference/reader uncertainty | positive-only、reader分歧、reference basis | label state、reader vote、adjudication、grade | mask、分层、限制claim | DeepSOZ多为 unresolved；reader study待完成 |
| calibrated candidate-set risk | 新校准队列上候选集合漏掉reference的风险 | coverage--risk、prediction set | 冻结集合大小/弃权策略 | 当前无独立S1-C，不能声称 |

不得将四项相加成一个“confidence”。尤其：

- 高 epistemic agreement 可能只是所有fold共享同一 shortcut；
- 高 scalp visibility 只表示头皮信号可读，不证明通道是SOZ；
- reference uncertainty 不是模型不确定性；
- 当前 Top1--Top2 margin 不等于 `P(correct)`，也没有 conformal 风险保证。

正式报告应分别给出 `model_disagreement_status`、`observability_status`、
`reference_qualification`、`candidate_set_risk_status` 和独立 reason code。任一项 unavailable 时明确写
`not_qualified/unavailable`，不得补值。

## 8. 对报告链的影响

报告只能使用四类句槽：

```text
signal fact          <- qualified typed EEG producer
candidate statement  <- frozen H-only C18 ranking or abstention
uncertainty statement<- the four axes above, named separately
scope limitation     <- scalp-electrode candidate != cortical SOZ/EZ/surgical target
```

逐句必须保存 `fact_id/source/version/threshold/label_view/allowed_wording`。LLM 不作为 SOZ predictor、
gold、缺失事实补全器或 reader-study 自评器。即便未来允许语言润色，也不得改变候选、电极、时间、
qualification、弃权、否定语义或不确定性类别。

## 9. 审稿人视角的主要漏洞与防线

| 可能批评 | 若不处理的后果 | 冻结防线 |
|---|---|---|
| 把 DeepSOZ 0 称作 confirmed non-SOZ | target validity 致命 | binary/positive-only双视图，明确 dataset complement |
| 用 partial-label 消歧多阳性 | 创造伪 gold | set-valued target，不选集合内唯一类 |
| 用 PU/robust loss“找回”漏标 | 不可识别且循环自训练 | 无 prior/SCAR/SAR 时 NO-GO |
| 借 BUNDL 声称修正 SOZ 标签 | temporal/spatial endpoint 错配 | BUNDL 仅限 native temporal/I qualification |
| unknown 进入 softmax/BCE 负类 | 系统性假阴性 | 未来 masked normalization over `P union N` |
| 把 margin 写成安全置信度 | selective overclaim | 四类不确定性拆分，full coverage 与 risk--coverage并报 |
| private 参与选择后仍称外部验证 | 数据泄漏 | 永久 post-open 描述性队列，不反馈 |
| 72.55%写作优于DeepSOZ 70% | 错误比较器 | 明列 DeepSOZ `0.744`，当前竞争假设未达到 |

## 10. 当前与未来的最终判决

```text
CURRENT PUBLIC/PRIVATE
  keep frozen H-only model and all reported outcomes
  keep benchmark_binary + medical_positive_only sensitivity
  no new loss/backbone/threshold selection
  no PU/PLL/BUNDL correction of C18 labels
  no private feedback

FUTURE S1-D
  four-state physical-electrode ledger
  patient-equal masked set-valued losses
  predeclare one low-capacity reasoner family

FUTURE S1-C
  calibration only
  separate uncertainty axes
  freeze prediction-set/abstention policy

FUTURE A5
  seal predictions and reports before labels open
  one-shot full-coverage + selective evaluation
```

这套协议使“可信候选、阴性 qualification、弃权、可追溯报告”形成逻辑闭环，但不会把当前 strict
低于 50%、far/跨侧错误、已开标数据和缺少 label-fresh C18 队列的问题包装消失。顶刊/顶会最关键的
下一证据仍是新的同终点患者，而不是在旧队列上继续扫描算法。

## 11. 核验文献

1. Wang H, et al. PiCO+: Contrastive Label Disambiguation for Robust Partial Label Learning.
   *IEEE TPAMI*. 2024. <https://doi.org/10.1109/TPAMI.2023.3342650>.
2. Wang H, et al. SoLar: Sinkhorn Label Refinery for Imbalanced Partial-Label Learning.
   *NeurIPS*. 2022. <https://doi.org/10.52202/068431-0588>.
3. Kim Y, Kim JM, Akata Z, Lee J. Large Loss Matters in Weakly Supervised Multi-Label Classification.
   *CVPR*. 2022. <https://doi.org/10.1109/CVPR52688.2022.01376>.
4. Kiryo R, Niu G, du Plessis MC, Sugiyama M. Positive-Unlabeled Learning with
   Non-Negative Risk Estimator. *NeurIPS*. 2017.
   <https://proceedings.neurips.cc/paper/2017/hash/7cce53cf90577442771720a370c3c723-Abstract.html>.
5. Bekker J, Davis J. Learning from Positive and Unlabeled Data: A Survey.
   *Machine Learning*. 2020. <https://doi.org/10.1007/s10994-020-05877-5>.
6. Shama DM, Venkataraman A. Bayesian Uncertainty-aware Deep Learning with noisy labels:
   Tackling annotation ambiguity in EEG seizure detection. *PLOS ONE*. 2026.
   <https://doi.org/10.1371/journal.pone.0352191>.
7. Shama DM, Jing J, Venkataraman A. DeepSOZ: A Robust Deep Model for Joint Temporal and
   Spatial Seizure Onset Localization from Multichannel EEG Data. *MICCAI*. 2023.
   <https://doi.org/10.1007/978-3-031-43993-3_18>.
