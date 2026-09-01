# DeepSOZ 官方标签与定位终点审计

**审计日期：** 2026-08-13  
**官方代码版本：** `913c921f8a08fa4df76ca0708126f565860f1068`  
**本地只读稀疏副本：** `/mnt/hd1/dyf/dataset/DeepSOZ_official_sparse`  
**适用模型：** 当前冻结的 LaBraM identity-v16 anchor 与 masked-variable auxiliary v17  
**核心结论：** DeepSOZ 的公开 channel vector 是可复现的 patient-level benchmark reference，
但多数标签由临床区域文本展开为**无集合内优先级的多电极 positive set**；官方论文的 patient
localization accuracy 还包含条件式邻域命中，不能与本项目的 exact positive-set membership
Top-1 直接比较。

## 1. 官方数据到底标注了什么

官方 `data/TUH_manifest_final.csv` 有 652 条 record、124 名患者。每行包括 19 导二值列、
`hemi`、前后象限 `region` 和 `Comments`。典型 `Comments` 不是逐电极盲法判读，而是：

```text
right temporal
left anterior temporal
centered around c4
central parietal posterior temporal
left temporal parietal
```

因此至少存在两类 reference：

1. **粗区域展开标签：** 由 `left temporal`、`right frontal` 等区域描述映射为多个电极；
2. **电极指向较强的标签：** 文本含明确电极或 `maximal`、`centered` 等措辞。

公开 CSV 没有提供 positive set 内哪个电极更接近真实起始源的顺序，也没有逐电极置信度、
unknown mask、读者间一致性、SEEG contact、切除范围或术后结局。故一个四阳性标签只能监督
“预测落入该集合”，不能监督“四个电极中的 exact 第一名”。

## 2. 标签基数证明它主要是 set endpoint

652 条官方记录的阳性电极数分布为：

| Positive count | Records |
|---:|---:|
| 1 | 59 |
| 2 | 237 |
| 3 | 261 |
| 4 | 85 |
| 5 | 6 |
| 7 | 1 |
| 8 | 3 |

当前 C18 stable public-development 队列的 102 名患者为：

| Positive count | Patients |
|---:|---:|
| 1 | 10 |
| 2 | 48 |
| 3 | 29 |
| 4 | 13 |
| 5 | 1 |
| 7 | 1 |

其中 93/102 仅有粗区域描述，9/102 含明确电极、`maximal` 或 `centered` 等文本。identity-v16
的 strict Top-1 在两组分别为 47/93（50.54%）和 4/9（44.44%）。后组太小，不能证明标签
粒度是唯一性能瓶颈；但这些数据明确反对把 positive set 解释为集合内有序的 exact-electrode
gold。

## 3. 官方代码并非统一 exact Top-1

两个官方 notebook 都先执行：

```text
argmax prediction 属于 positive set -> correct
```

随后又加入条件式邻域命中：

- `code/test/szloc_all.ipynb`：若 positive count `<=2`，预测位于任一 positive 的邻域也算正确；
- `code/test/final_eval_all.ipynb`：阈值改为 positive count `<=4`。

两个 notebook 都把这条规则累计进 `corr_pt`，即论文语境中的 patient localization accuracy
并非纯 exact positive-set membership。阈值 `2` 与 `4` 的不一致意味着“官方 accuracy”仍需
绑定到具体 notebook 和 commit，不能只写一个没有实现版本的 accuracy。

本项目以后固定同时报告：

| 名称 | 定义 | 角色 |
|---|---|---|
| Exact set membership Top-1 | argmax 必须属于 reference-positive set | 严格 benchmark 主终点 |
| Official-neighborhood-4 | exact，或 positive count <=4 时命中官方一跳邻域 | 与 `final_eval_all.ipynb` 可比的次终点 |
| Official-neighborhood-2 | exact，或 positive count <=2 时命中官方一跳邻域 | 实现敏感性分析 |
| Hemisphere concordance | 预测侧别与 reference 侧别一致 | 独立临床粗粒度终点 |
| A/P quadrant concordance | 预测前后象限与 reference 一致 | 独立临床粗粒度终点 |

任何 relaxed 指标都不能替换 exact，也不能改名为 exact channel accuracy。

## 4. 当前 LaBraM OOF 在这些终点上的位置

在固定 C18 mask 和同一 102 人 OOF 上重算：

| Endpoint | identity-v16 | auxiliary-v17 |
|---|---:|---:|
| Exact set membership | 51/102 = 50.00% | 51/102 = 50.00% |
| Official neighborhood，positive count <=2 | 68/102 = 66.67% | 69/102 = 67.65% |
| Official neighborhood，positive count <=4 | 77/102 = 75.49% | 78/102 = 76.47% |
| Hemisphere concordance | 76/102 = 74.51% | 77/102 = 75.49% |
| A/P quadrant concordance | 70/102 = 68.63% | 71/102 = 69.61% |

DeepSOZ 论文报告的 patient localization accuracy `0.744 +/- 0.058` 与本项目的约 75%--76%
neighborhood-4 点估计处在同一量级，但这**不是**论文的直接复现：患者 roster、PZ 政策、
预处理、模型、交叉验证和可能使用的 notebook 状态均不同。正确结论只是：论文约 74% 与
本项目 50% exact 并不矛盾，因为两者的计分定义不同。

## 5. PZ 双列是 schema conflict，不是建模机会

官方 CSV header 中 `pz` 出现两次，分别位于第 24 和第 33 个字段位置：

- 第一列有 43 条 positive；
- 第二列有 2 条 positive；
- 两列在 45 条记录上不一致。

常见 Python `csv.DictReader` 会让后一个同名字段覆盖前一个，所以官方 loader 的实际行为
取决于读取实现，而不是一个明确的医学裁决。不能：

- 默认为两列取 OR；
- 选择让模型成绩更高的一列；
- 把覆盖行为解释为 PZ 的医学真值；
- 在看过当前 102 人结果后重新训练并挑选 PZ 政策。

当前主分析继续对所有患者 mask PZ，形成 C18 benchmark。未来新队列必须把 PZ 作为独立字段
由临床标注者显式裁决，并预注册 first/second/OR 仅为 DeepSOZ schema 敏感性分析。

## 6. 对 v17 恢复实验的解释

v17 将 9 名 patient-variable 患者、182 次发作作为 masked auxiliary supervision 加回；冲突、
缺失和 PZ 均不进 loss，每名患者只贡献一次 patient-level loss。结果为：

```text
strict Top-1: 51/102 -> 51/102
macro AP:      0.52993 -> 0.54127
Hit@3:         75.49% -> 78.43%
far errors:    25 -> 24
```

这是合理而非反常的机制结果：partial positive-set loss 能把多个已知阳性整体向前排序，所以 AP
提高；它没有新的集合内电极优先级，且独立样本只有 9 名患者，因而不足以稳定改变 argmax。
该结果不能支持“换 backbone 即可解决 exact 定位”，也不能支持继续在相同 102 人上扫描 loss、
权重、graph、pooling 或随机种子。

## 7. 冻结后的论文合同

1. 把当前 102 人称为 repeatedly-used public development benchmark，而非确认性 test。
2. strict exact membership 保留为最严格 benchmark 主终点，但明确它不等于 clinical text
   提供了 positive set 内的 exact-electrode 排序真值。
3. neighborhood-4 用于与 DeepSOZ 官方最终 notebook 对照；neighborhood-2 单独作为实现敏感性。
4. 同时报告 AP、MRR、Hit@K、侧别、A/P 象限、far/contralateral-far error 和 patient-bootstrap CI。
5. 新的 exact-electrode 方法选择只能在新的同终点临床标注队列上进行。
6. private 在模型、abstention、邻接规则和报告槽位冻结后只做一次 zero-adaptation 描述性迁移。
7. 头皮电极 ranking 只能称 SOZ candidate/reference concordance；不得写成 cortical SOZ、SEEG
   contact 或手术靶点预测。

## 8. 可追溯证据

- 官方 manifest：`/mnt/hd1/dyf/dataset/DeepSOZ_official_sparse/data/TUH_manifest_final.csv`
- 官方 neighborhood-2：`code/test/szloc_all.ipynb`，`final_loc`
- 官方 neighborhood-4：`code/test/final_eval_all.ipynb`，`final_loc`
- v17 机器结果：`outputs/labram_masked_variable_auxiliary_oof_v17_20260812/manifest.json`
- v17 正式结论：`labram_masked_variable_auxiliary_recovery_result_v17_20260813_zh.md`
- DeepSOZ: Shama DM, Jing J, Venkataraman A. MICCAI 2023.
  <https://doi.org/10.1007/978-3-031-43993-3_18>
