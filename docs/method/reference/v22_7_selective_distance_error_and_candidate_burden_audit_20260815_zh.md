# v22.7 冻结弃权的距离错误与候选负担审计

**日期：** 2026-08-15  
**性质：** 已冻结margin门的post-open只读审计；不选新阈值、不训练、不改变报告  
**工作点：** `top1−top2 margin >= 0.03397908806800842`  
**机器结果：** `outputs/trustworthy_soz_selective_distance_v22_7_20260815/result.json`

## 1. 审计问题

v22.3已经证明弃权组的far risk点估计更高，但总体区间跨0。v22.6又发现大量far error是跨侧错误。
本轮进一步回答：冻结margin门是否真的筛出了临床上更危险的错误？

预先固定分析项：

- exact / neighbor-only / far；
- contralateral far；
- private known-spread Top-1；
- first-positive rank >5及Hit@5；
- accepted与abstained的rate difference及patient/patient-cluster bootstrap。

本轮没有从曲线选择新工作点。Abstained隐藏排名只在评价器内读取，不重新显示到临床报告。

## 2. Public patient-level结果

| Public | N | Exact | Neighbor-only | Far | Contralateral far | Hit@5 |
|---|---:|---:|---:|---:|---:|---:|
| Accepted/display | 81 | 39（48.15%） | 22（27.16%） | 20（24.69%） | 17（20.99%） | 64（79.01%） |
| Abstained/hidden audit | 21 | 8（38.10%） | 5（23.81%） | 8（38.10%） | 5（23.81%） | 18（85.71%） |

风险富集：

| Public metric | Abstained − accepted | 95% patient bootstrap CI |
|---|---:|---:|
| Strict error | +10.05% | -13.93%--33.96% |
| Far error | +13.40% | -9.16%--37.93% |
| Contralateral far | **+2.82%** | -16.95%--24.48% |
| First positive rank >5 | **-6.70%** | -23.22%--12.93% |

决定性解释：public margin对总体far error有有利点估计，但几乎没有筛分contralateral far。更重要的是，
abstained组Hit@5反而高于accepted，rank>5反而更少。Margin衡量Top-1与Top-2分离度，不是reference-positive
在完整排序中的深度，也不是“危险跨侧错误概率”。

## 3. Private post-open事件结果

| Private | N | Exact | Neighbor-only | Far | Contralateral far | Hit@5 |
|---|---:|---:|---:|---:|---:|---:|
| Accepted/display | 43 | 19（44.19%） | 14（32.56%） | 10（23.26%） | 5（11.63%） | 36（83.72%） |
| Abstained/hidden audit | 8 | 2（25.00%） | 2（25.00%） | 4（50.00%） | 4（50.00%） | 5（62.50%） |

风险富集：

| Private metric | Abstained − accepted | 95% patient-cluster bootstrap CI |
|---|---:|---:|
| Strict error | +19.19% | -17.89%--49.36% |
| Far error | +26.74% | -15.62%--68.31% |
| Contralateral far | +38.37% | 0.71%--75.69% |
| First positive rank >5 | +21.22% | -9.52%--55.56% |

Private中contralateral far确实集中在8个abstained events：4/8，相比accepted 5/43。其bootstrap区间
刚好不跨0，但**不能称确认性显著**，因为：

1. private已历史开标；
2. 只有8个abstained events；
3. error subtype是v22.6之后的post-hoc分析；
4. 同时查看多个指标且未作前瞻性multiplicity控制；
5. public同一指标没有复制该分离。

这只能作为未来新队列预注册`contralateral-far enrichment`的效应量依据。

## 4. Known-spread失败模式

Private全部2次known-spread Top-1都位于accepted组：

```text
accepted  = 2/43 = 4.65%
abstained = 0/8  = 0%
```

因此当前margin门没有识别这一临床重要错误。不能增加一个根据private spread结果设计的新gate，因为那会
直接在已开标private上选择路由规则。当前只能：

- 在主结果表单列known-spread Top-1；
- 报告中继续不把邻近/受累电极自动称SOZ；
- 未来新队列预先定义spread-aware safety endpoint；
- 如有新calibration cohort，再评估candidate set或structured reject rule。

## 5. 对“可信弃权”主张的修订

### 当前可支持

- 系统确实执行target-blind显示层弃权，并隐藏候选；
- public/private总体far risk都呈abstained更高的有利点估计；
- private跨侧far富集是值得未来验证的探索性信号；
- 完整risk--coverage、错误距离和coverage均可审计。

### 当前不可支持

- margin稳定筛除跨侧SOZ错误；
- abstention保证accepted候选排序更深或Hit@K更好；
- current gate是临床安全门、校准错误概率或conformal risk control；
- private contralateral结果已被独立复制或统计确认；
- known-spread错误被弃权机制控制。

最准确的定位是：

> 当前margin gate是一个可审计的保守显示策略，具有总体far-error富集趋势；它不是经验证的临床危险
> 错误筛分器。

## 6. NeurIPS/MICCAI报告建议

主文将选择性结果拆成三层：

1. full-coverage exact/neighbor/far主表；
2. accepted/abstained的far与contralateral-far分解；
3. risk--coverage曲线及所有coverage/CI。

不要只报`75.31%/76.74%`。特别应披露：public accepted仍有17/81 contralateral far，private accepted
仍有5/43 contralateral far和2/43 known-spread Top-1。这比使用“safe abstention”措辞更能经受审稿。

未来confirmatory protocol应在新患者上预先冻结：confidence定义、contralateral-far/spread loss、
calibration/test分割、coverage下限和multiplicity策略。当前public/private均不得再用于选择替代confidence。

## 7. 实现与验证

- 审计器：`scripts/audit_trustworthy_soz_selective_distance_v22_7.py`
- 结果：`outputs/trustworthy_soz_selective_distance_v22_7_20260815/result.json`
- 测试：`tests/test_trustworthy_soz_selective_distance_v22_7.py`
- 单项回归：`3 passed`

审计没有读取raw EEG、加载模型权重、训练、重选阈值、改候选或改写报告，也没有进行SHA实验。
