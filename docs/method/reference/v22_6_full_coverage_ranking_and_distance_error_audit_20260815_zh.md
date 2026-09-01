# v22.6 全覆盖排序质量与临床距离错误审计

**日期：** 2026-08-15  
**性质：** 冻结预测的只读评价；不训练、不选模、不改阈值、不改候选或报告  
**模型：** portable frozen LaBraM H-only v16  
**机器结果：** `outputs/trustworthy_soz_ranking_distance_v22_6_20260815/result.json`

## 1. 审计问题

只报告full-coverage neighborhood-4 `72.55%`无法回答三个关键问题：

1. 有多少是真正命中reference-positive电极，有多少只靠一跳邻域得分？
2. 未命中邻域的错误是否仍在同侧，还是已经跨侧？
3. 即使Top-1不够好，完整排序是否仍能提供有限候选清单价值？

本审计对已冻结public patient-level OOF预测和private event-level H-only预测重新计算同一套排名/距离
分解。Public与private保持不同统计单位，绝不合并。

## 2. 冻结定义

每个样本在C18（standard-19去PZ）内分为互斥三类：

```text
exact         = Top-1属于reference positive set
neighbor-only = 不exact，但符合DeepSOZ final notebook的positive-count≤4一跳邻域规则
far           = 既不exact也不属于上述可接受邻域
```

Private中医生标注的known-spread electrode从neighbor acceptance中删除，不能因靠近significant reference
而计为成功。

Far error进一步分为：

- `contralateral_far`：reference只有一个非中线侧别，而Top-1落在对侧；
- `ipsilateral_far`：同侧但在reference及其可接受邻域之外；
- `midline_far_against_unilateral`：单侧reference但Top-1在中线；
- `nonunilateral_reference_far`：reference本身双侧/仅中线，不能定义对侧。

这仍是头皮电极图距离和侧别，不是皮层解剖距离或侵入式SOZ误差。

## 3. Full-coverage主结果

| Cohort | Unit | N | Exact | Neighbor-only | Far | Contralateral far |
|---|---|---:|---:|---:|---:|---:|
| Public H-only | patient | 102 | 47（46.08%） | 27（26.47%） | 28（27.45%） | 22（21.57%） |
| Private H-only | event，23 patient clusters | 51 | 21（41.18%） | 16（31.37%） | 14（27.45%） | 9（17.65%） |

两队列的relaxed `72.55%`相同是数值巧合，内部构成不同。Neighborhood-only占全部relaxed successes：

```text
public  = 27 / 74 = 36.49%
private = 16 / 37 = 43.24%
```

因此“72.55%”不能被称为72.55%的精确SOZ通道准确率。

### 3.1 Far error组成

Public的28个far errors中：

- contralateral far：22/28 = 78.57%；
- ipsilateral far：6/28 = 21.43%；
- midline/nonunilateral far：0。

Private的14个far errors中：

- contralateral far：9/14 = 64.29%；
- ipsilateral far：2/14 = 14.29%；
- nonunilateral-reference far：3/14 = 21.43%；
- known-spread Top-1：2/51 = 3.92%，是far error的一个临床重要子集。

5000次patient/patient-cluster bootstrap给出：public far rate 95%区间`18.63%--36.27%`、contralateral
far `13.73%--29.41%`；private far `19.05%--36.00%`、contralateral far `9.26%--26.53%`。这些区间
只描述当前已开标队列，不是未来临床风险保证。

## 4. 排序质量与候选负担

| Cohort | Macro AP | MRR | Hit@3 | Hit@5 | Positive recall@3 | Positive recall@5 |
|---|---:|---:|---:|---:|---:|---:|
| Public | 0.5062 | 0.6137 | 72/102 = 70.59% | 82/102 = 80.39% | 42.55% | 54.57% |
| Private | 0.4420 | 0.5691 | 34/51 = 66.67% | 41/51 = 80.39% | 34.87% | 46.21% |

第一枚reference-positive电极的rank分布：

| Cohort | Rank 1 | Rank 2 | Rank 3 | Rank 4 | Rank 5 | >5 | Median rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| Public | 47 | 14 | 11 | 6 | 4 | 20 | 2 |
| Private | 21 | 6 | 7 | 3 | 4 | 10 | 2 |

达到至少80% Hit所需候选清单均为5个电极；达到至少90% Hit，public需7个、private需9个。C18中列出5个
相当于暴露27.8%的候选空间。因此Hit@5支持“医生复核候选清单”的研究价值，但不能包装成高精度单通道
定位，也不能用Top-5掩盖Top-1 strict低于50%。

## 5. 对论文目标的影响

### 支持的结论

- H-only排序含有中等候选排序信号：两个队列的Hit@5均为80.39%，MRR约0.57--0.61；
- neighborhood-4为有临床容忍度的secondary sensitivity，不是精确电极终点；
- strict、neighbor-only、far、contralateral-far和known-spread Top-1必须并列报告；
- 当前最合适的产品形态是“有限候选清单或弃权”，而不是自动单电极定论。

### 不支持的结论

- “SOZ Top-1超过70%”；
- “72.55%证明模型精确定位良好”；
- “Hit@5 80.39%达到原80% Top-1目标”；
- “邻近命中与reference-positive命中等价”；
- “跨侧far error已由margin弃权稳定消除”；
- “private验证了DeepSOZ patient-level superiority”。

## 6. NeurIPS/MICCAI审稿视角

若只报72.55%，审稿人很容易指出评价规则宽松和label set粗糙。本分解应进入主文而不是补充材料：

1. **主表：** strict / neighbor-only / far三分解；
2. **安全错误：** contralateral far和private known-spread Top-1；
3. **排序效用：** AP、MRR、Hit@3/5、positive recall和所需候选数；
4. **选择性结果：** 另报coverage和risk--coverage，不能替换full coverage；
5. **标签限制：** DeepSOZ粗区域positive set与private event reference粒度不同。

这会削弱单一高准确率叙事，但能显著强化可信候选、错误透明性和临床工作流定位。当前结果更适合
“trustworthy candidate ranking under weak clinical reference”，不适合“SOTA exact SOZ localization”。

## 7. 实现与验证

- 审计器：`scripts/audit_trustworthy_soz_ranking_distance_v22_6.py`
- 结果：`outputs/trustworthy_soz_ranking_distance_v22_6_20260815/result.json`
- 测试：`tests/test_trustworthy_soz_ranking_distance_v22_6.py`
- 单项回归：`3 passed`

审计读取已冻结prediction与既有评价reference，没有读取raw EEG、加载模型权重、训练、选模、改阈值或
进行SHA实验。Private结果明确是post-open描述性分析。
