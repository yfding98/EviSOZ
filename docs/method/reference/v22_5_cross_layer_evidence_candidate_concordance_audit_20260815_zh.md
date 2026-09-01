# v22.5 事件证据—SOZ候选跨层一致性审计

**日期：** 2026-08-15  
**性质：** target-free透明性审计；不训练、不读取SOZ target、不访问private、不改候选/阈值/报告  
**适用系统：** frozen H-only candidate + legacy public typed-event facts  
**机器结果：** `outputs/trustworthy_soz_cross_layer_concordance_v22_5_20260815/result.json`

## 1. 为什么需要这项审计

当前定位分数来自冻结LaBraM H-only reasoner；逐事件时间、双极边和频谱字段来自另一个target-free
producer。后者没有参与定位打分，也未通过独立EEG reader事实性资格化。虽然v22报告已把两层分开，
审稿人仍会追问：二者在实际报告中有多少同侧、多少相互矛盾？

本审计回答的是：

> 固定规则标记的首批头皮可见双极边侧别，与冻结患者级Top-1头皮电极候选侧别，在报告层呈现何种
> 描述性关系？

它不回答哪一层“正确”，也不是SOZ accuracy、faithfulness、clinical factuality或新的弃权规则。

## 2. 输入和不访问项

只读取两个既有冻结工件：

1. legacy target-free event facts：102 patients / 988 events；
2. v22 qualified event reports：同一988-event report roster，其中定位来自current localization roster。

双roster接口保持v22.4规则：101名患者同时有current localization与legacy event facts；patient 258的
4个事件没有完整定位，保持`localization_unavailable`。current-only patient 10489没有legacy event
facts，不进入事件审计。

审计未加载raw EEG、DeepSOZ target、private EEG/target、模型权重或LLM，也没有训练、选模、改阈值、
隐藏候选或改写报告。

## 3. 冻结侧别归约

标准19导按现有laterality ontology归为left/right/midline。对每个首批双极边：

- 仅左侧非中线endpoint：`left`；
- 仅右侧非中线endpoint：`right`；
- 同时出现左右endpoint：`bilateral`，不强行裁决；
- `CZ--C4`一类中线—右侧边保留为`right`，中线endpoint不抹去已观察到的侧别；
- 只有中线endpoint：`midline_only`；
- `first_visible_derivations=null`且原producer明确给出`no_sustained_bipolar_change`：保留为无持续变化，
  不是阴性SOZ；
- typed facts缺失：`evidence_unavailable`。

只有`left/right evidence + displayed left/right candidate`进入同侧/跨侧比较。bilateral、midline、无持续
变化、证据不可用、定位弃权/不可用全部保留为不可比较状态。

## 4. 冻结结果

### 4.1 事件层

| 状态 | 事件数 |
|---|---:|
| same-side descriptive concordance | 211 |
| contralateral descriptive tension | 227 |
| displayed candidate + bilateral event evidence | 240 |
| displayed candidate + no sustained bipolar change | 113 |
| localization abstain | 193 |
| event evidence + localization unavailable | 4 |
| **总计** | **988** |

可比较的438个事件中：

```text
same side       = 211 / 438 = 48.17%
contralateral   = 227 / 438 = 51.83%
```

事件不是独立统计单位：患者级候选在同一患者的多次发作报告中重复，因此这两个百分比不能写成模型准确率、
置信区间分母或“超过50%”的假设检验。

原始事件证据侧别流为：left 284、right 266、bilateral 291、no sustained change 143、unavailable 4。
其中部分事件因患者级margin弃权而不进入同侧/跨侧比较。

### 4.2 患者层

| 患者状态 | 患者数 |
|---|---:|
| 所有可比较事件均同侧 | 15 |
| 所有可比较事件均跨侧张力 | 13 |
| 同一患者同时出现同侧与跨侧事件 | 40 |
| 显示候选但没有可比较单侧事件 | 12 |
| 定位弃权或不可用 | 22 |
| **总计** | **102** |

在有至少一个可比较事件且显示候选的68名患者中，`40/68=58.82%`同时包含同侧与跨侧事件。这说明事件
检测侧别具有显著的患者内异质性；不能通过挑选与候选一致的一次发作来生成看似忠实的解释。

## 5. 对方法架构的决定性影响

### 5.1 不能把事件事实写成H-only分数的原因

当前跨层关系接近均分且患者内经常改变。可能原因包括体积传导、快速双侧受累、参考方式影响、
target-free change detector误检、不同发作表型，以及patient-level clinical-note reference与event-level
信号事实的粒度差异。现有数据不能区分这些原因。

因此以下做法均不成立：

- “模型因为F7--T7首先变化，所以预测T7”；
- 用同侧事件挑选作为SOZ候选正确性的证明；
- 把跨侧张力自动解释成模型错误或传播；
- 用该一致性重新校准margin、修改候选或决定是否弃权；
- 在已反复使用的public队列上据此重新融合V/event evidence。

### 5.2 当前正确的信息流

```text
frozen H-only patient representation → candidate / abstention

target-free event facts ─────────────→ separate descriptive report layer

cross-layer audit ───────────────────→ transparency flag / future reader-study stratum
                                      never candidate routing or accuracy
```

M/I继续结构性缺席；V/event facts继续只能描述头皮可见变化。该阴性结果使“negative concept
qualification”不再只是一个训练gate失败，而是得到下游跨层不一致性的独立支持。

## 6. 对诊断报告的影响

现有报告的分层写法必须保留：事件事实、H-only候选和临床边界不能压缩成一个因果句。尤其不能自动生成：

> “F7--T7首先出现异常，因此SOZ位于左颞并传播到左额。”

允许的写法是：

> “固定规则在该事件标记到左侧首批头皮可见变化；冻结患者级定位器的候选为T8，两层结果存在跨侧
> 张力。该事件顺序不是传播或皮层起源真值，候选不等同侵入式SOZ或手术靶点，需医生复核。”

这里的“张力”仍只是机器内部字段关系，不是医生确认的临床矛盾。private当前没有资格化event facts，
不得把public的跨层统计迁移成private事件叙述。

## 7. 顶刊/顶会报告方式

主文可把这项审计放入trustworthiness/faithfulness结果，而不是性能主表：

- 必报patient-level状态，event-level只作描述并注明重复；
- 与M/I qualification失败、V不进入reasoner、报告分层放在同一图；
- 明确没有临床reader事实性，因此不能称“解释正确率”；
- future reader study按`concordance-only / tension-only / mixed / indeterminate`分层抽样，防止只展示好看病例；
- 不根据该结果修改当前模型、阈值或报告措辞后再在同一队列宣称改进。

这一结果会降低“完整concept reasoner已经成功”的叙事强度，但显著提高论文对negative result、
faithfulness和failure transparency的可信度。

## 8. 实现与验证

- 审计器：`scripts/audit_trustworthy_soz_cross_layer_concordance_v22_5.py`
- 结果：`outputs/trustworthy_soz_cross_layer_concordance_v22_5_20260815/result.json`
- 测试：`tests/test_trustworthy_soz_cross_layer_concordance_v22_5.py`
- 单项回归：`3 passed`

审计只做确定性laterality归约和roster连接；没有使用标签计算性能，没有进行SHA实验。
