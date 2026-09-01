# LaBraM 不替换恢复：冲突通道掩码辅助监督协议 v17

**冻结日期：** 2026-08-12  
**性质：** 反复使用的 DeepSOZ--TUSZ 公共开发队列上的一次性、探索性数据恢复实验  
**主干：** official pretrained LaBraM-Base；block 0--9 完全冻结  
**private：** 不读取、不训练、不校准、不选择阈值  

## 1. 唯一研究问题

当前 target-v2 策略只要同一患者任一头皮通道在不同 DeepSOZ source rows 中出现 `0/1`
冲突，就隔离整名患者。正式 target join 显示，11 名此类患者中有 9 名仍至少包含一个跨记录一致的
头皮阳性通道，并在新 signal universe 中对应 182 个合格发作。整例删除可能丢失
少量但有用的 patient-level endpoint supervision。

本实验只回答：

> 保留所有一致通道、仅屏蔽患者内冲突通道，并把这些患者作为 train-only auxiliary，能否在
> 不改变 LaBraM、主 reasoner、候选空间和评价标签的情况下改善稳定标签患者的 SOZ-reference
> 排名？

它不检验 foundation model replacement，不把冲突或缺失通道猜成阳性，也不把 TUSZ
involvement、最早头皮变化或传播描述当作 SOZ 标签。

## 2. 训练标签规则

对每名 auxiliary patient、每个 standard-19 通道，先在该患者全部 DeepSOZ source rows 上审计
原始字段；PZ 始终沿用当前 primary endpoint 的 `mask=0`：

| source-row 状态 | auxiliary target | loss mask | 医学含义边界 |
|---|---:|---:|---|
| 所有有限合法值均为 `1` | `1` | `1` | benchmark reference positive |
| 所有有限合法值均为 `0` | `0` | `1` | dataset complement；不是医生确认的 biological negative |
| 同时出现 `0` 与 `1` | 任意占位 | `0` | patient-variable / unknown |
| 全缺失或非法 | 任意占位 | `0` | unknown |
| PZ / 头外电极 | 任意占位 | `0` | 不进入当前 C18 endpoint |

患者仅在至少有一个 `mask=1,target=1` 且有可用信号事件时进入 auxiliary training。不得使用：

- 多数投票；
- positive union；
- 将未列通道假定为遗漏阳性；
- one-hop label dilation；
- 按当前 102 人的预测错误选择患者或通道；
- private significant/spread labels。

这不是 PU learning。所有冲突值直接退出 loss；没有假设 SCAR/SAR、class prior 或 missing-positive
机制。

## 3. 信号与表示合同

正式运行前必须先发布完整 652-record / 124-patient DeepSOZ identity-overlay signal universe：

```text
complete identity-recovery audit
  -> local TUSZ TERM,seiz timeline
  -> frozen causal C-CAR19 preprocessing, 200 Hz, [-12,+48) s
  -> signal eligibility only
```

该 producer 不得接收 target-v2、target-derived split/quarantine、C18 mask、历史 predictor 或
private 数据。它仍然是 **DeepSOZ identity-overlay-conditioned**，不是全 TUSZ population；只是没有
按 SOZ target 值或完整性筛选。

辅助事件随后通过显式 target join 选择，并重新物化：

```text
X [event,19,12000]
  -> 15 independent 4-s calls [event,15,19,4,200]
  -> official LaBraM blocks 0--9, frozen
  -> prefix [event,15,77,200]
  -> H phase contrast [event,19,600]

same X
  -> frozen fine temporal descriptors [event,19,20]
```

位置 ID、legacy/modern electrode crosswalk、单位、滤波、参考和 phase extraction 必须与 v16
anchor 相同。辅助缓存只允许追加新事件；不能重算或覆盖既有 1149-event cache。

## 4. 一次性训练设计

评价总体保持 identity-v16 的 102 名 C18-complete stable patients / 1145 events 和冻结五折。
9 名 masked-variable patients 先按 eligible-event burden 与固定 salted identity hash 分配到 5 个
auxiliary outer folds；fold assignment 不读取标签值。它们仅作为额外训练患者，永不进入主 held
metric、bootstrap、阈值或 promotion denominator。

每个 outer fold：

1. held set 仍是该折 stable patients；
2. training set 为其余 stable patients，加上 `aux_outer_fold != current_outer_fold` 的 auxiliary
   patients；同折 auxiliary patient 完全不参与该 fold；
3. feature scaler/PCA 仍只在 stable outer-train patients 拟合，再原样变换 auxiliary features；
4. 不再运行新的 inner model selection；逐折直接复用 pinned identity-v16 full arm 的 L2：
   `[0.01,0.20,0.20,0.01,0.20]`；
5. Jeffreys channel prior 仍只由 stable outer-train target 计算；只有 36 参数 shared reasoner 的
   loss 读取 combined outer-training target/mask；
6. combined loss 中每名 stable/auxiliary 患者等权；冲突通道因 `mask=0` 不进入分子或分母；
7. 每个 outer fold 从零拟合，禁止从 102 人 final checkpoint 初始化；
8. foundation optimizer parameter count 必须为 `0`。

主模型仍是相同的 shared 36-parameter channel reasoner：

```text
block-9 H16 + fine20 -> shared linear score -> fixed C18 ranking
```

在首次构造 auxiliary loss 前必须完成 Phase-0 parity：使用相同 stable outer-train、stable-only
transform/prior 和上述逐折 L2，从零重拟合 no-aux control，并在固定容差 `1e-6` 内复现 pinned
identity-v16 full-arm OOF logits。parity 失败则停止，不得用 auxiliary 结果掩盖实现漂移。

不得同时新增 LoRA、attention、side/lobe loss、graph prior、M/I/V score、confidence weight、
event-consistency weight、新的 pooling 规则或新的超参数搜索。transform 与 prior 也不吸收
auxiliary 分布；这样候选与 v16 anchor 的唯一可解释差异是
masked-variable auxiliary supervision。

## 5. 评价与停止门

主比较器固定为 identity-v16 full anchor：strict `51/102`、one-hop `77/102`、macro AP
`0.5299`、far errors `25`。正式脚本需重新核对精确 artifact 数值，不得以本文四舍五入值替代。

只在 102 名 stable held patients 上报告：

- strict Top-1；
- one-hop relaxed Top-1，并报告 neighbor-eligible 分母；
- macro AP、MRR、Hit@3、Hit@5；
- far-error 与 contralateral-far count；
- 五折 strict；
- paired patient bootstrap 95% CI；
- candidate/anchor 的 win/loss/tie 与 Top-1 agreement。

只有同时满足以下条件，候选才可保留为下一阶段 engineering candidate：

1. strict 至少净增 `5/102`，且 paired 95% CI 下界 `>0`；
2. macro AP 不下降，paired 95% CI 下界 `>=0`；
3. one-hop 点估计不下降；
4. far errors 不多于 `25`；
5. 至少 `4/5` folds 的 strict 不低于 anchor；
6. auxiliary target、signal、cache、mask、patient firewall 和 foundation-freeze gates 全部通过。

任一条件失败即：

```text
MASKED_VARIABLE_AUXILIARY_STOP_ON_CURRENT_PUBLIC_COHORT
```

失败后不得在同一 102 人上改成 majority/union、扫描 auxiliary weight、最低 mask 数、L2、block、
seed、pooling 或通道先验。即使通过，因为 102 人已经被反复用于开发，也只能称 exploratory
development support，不能称 fresh test、external validation 或 clinical 80%/85% 达标。

## 6. 三轴谱系收据

每个产物分别记录：

| 产物 | direct target values | upstream target-conditioned roster | target-supervised model |
|---|---:|---:|---:|
| identity-overlay signal universe | no | no target-value filtering；但 identity-overlay-conditioned | no |
| auxiliary target join | yes | yes | no |
| auxiliary prefix/fine cache | no direct values | yes，继承 join event roster | frozen LaBraM only |
| v17 OOF reasoner | yes | yes | yes |

不得再用单个 `target_free=true/false` 概括完整 lineage。

## 7. 临床与论文边界

- M 仍是 morphology；I 仍是 scalp-visible ictal involvement；V 仍是 observable evolution。
  本实验不会让三个概念自动获得 SOZ 语义。
- DeepSOZ positive 仍是 clinical-note-derived scalp-electrode reference，不是 SEEG cortical SOZ、
  resection target 或手术金标准。
- private 必须继续封存；本实验不能用 private 选择是否通过。
- 若候选失败，结论是“整例隔离并非当前性能瓶颈的充分解释”，不是“LaBraM 无效”。
- 若候选通过，仍需新的同终点 S1 cohort 或预先冻结的 zero-adaptation private validation 才能支持
  泛化结论。

## 8. 启动条件

只有以下条件全部通过才允许启动 GPU cache extraction 和 OOF fit：

1. target-independent signal-universe builder 与测试通过；
2. 652 records / 124 patients / 1812 candidate events 闭合；
3. masked-variable join 固定为 11 名候选、9 名有稳定阳性的 admitted patients、182 个合格事件；
4. auxiliary cache 与已有 stable cache 的 event ID 不重叠；
5. GPU 无其他 compute process，磁盘空间满足临时文件和原子发布；
6. private access count 为 0。
