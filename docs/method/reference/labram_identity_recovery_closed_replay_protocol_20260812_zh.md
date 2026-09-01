# LaBraM 身份恢复闭环重放协议

**日期：** 2026-08-12  
**状态：** 在恢复信号缓存与本轮模型重放之前冻结；124 名患者的 target-v2 标签此前均已被开发流程访问  
**主干：** 官方预训练 LaBraM-Base block-9，不替换、不从头训练、不解冻  
**性质：** 已反复开发的 DeepSOZ--TUSZ public-development cohort 上的数据覆盖审计；不是外部验证、独立测试或确认性实验

## 1. 唯一问题

本轮只回答：旧 crosswalk 将 45 条 DeepSOZ 记录标记为 ambiguous/unmapped，是否使既有 LaBraM v11.1 结果受到可恢复信号缺失的影响。由于 124 名患者的 target-v2 标签均已被历史开发访问，本协议是固定配置的覆盖恢复审计，而不是 target-blind 预注册确认。

607 条旧 unique 记录先建立固定的 124↔124 DeepSOZ 数字患者与本地 TUSZ 匿名患者双射。随后仅使用患者身份、official split、session/year、trial、montage 和 EDF sample count 恢复记录身份。恢复过程禁止读取 SOZ target、private 数据或历史模型输出。

身份恢复本身不是新增 SOZ 数据集，也不能把已经访问过标签的患者重新定义为测试集。

## 2. 队列与固定 split

闭环 signal replay 预期产生：

- target-free signal union：103 patients、1149 events；
- 固定 C18 reference-complete primary：102 patients、1145 events；
- 患者 `258` 仍因 O2 reference 未解析而完全排除于 primary fit、inner selection 和评价；
- `10489` 是唯一新进入 signal-eligible model cohort 的患者，共 27 个 eligible events。

正式运行必须由实际产物复核这些数字；任何不一致均 fail closed。

旧 102 名患者的 outer-fold assignment 必须逐一保持不变。旧 988 个事件的 ID、顺序、信号 receipt 和已缓存 tensor 必须逐元素保持不变。161 个新恢复 eligible events 只能追加，不能替换或重排旧 core。新患者 `10489` 只能依照不读取 target 的固定规则分 fold，并记录规则及 fold burden；不得根据 SOZ 分布或性能选择 fold。

## 3. 冻结模型与输入

沿用 v11.1 的完整配置：

```text
EEG per event: [19,12000], 200 Hz, [-12,+48) s
LaBraM calls: 15 × [19,4,200]
official frozen blocks 0--9
block-9 prefix per event: [15,77,200]
phase-contrast carrier: [19,600]
fine evidence per event: [19,20]
fixed output candidates: C18 (PZ 仅作输入上下文)
```

旧 988 个事件直接复用冻结 cache，并以 event ID、processed-window receipt 和逐事件 tensor SHA 做一致性验证；只对新增 161 个事件计算 LaBraM prefix 和 fine evidence。

模型、聚合与训练均不得改变：

- 完整患者 event bag 的 target-free reliability robust pooling；
- fold-local scaler 与 PCA16；
- 无 per-channel trainable bias 的 shared scorer；
- Jeffreys reference-membership prior；
- positive-set mass loss；
- L2 候选 `{0.01, 0.05, 0.20}` 及原 inner-selection 规则；
- 五折 patient-level nested OOF、patient bootstrap 和固定 C18 mask。

禁止在这次重放后按结果扫描 layer、head、loss、seed、窗口、graph weight、融合权重或 threshold。

## 4. 评价与配对审计

Primary 在 102 名 C18-complete 患者上报告：

- strict Top-1、one-hop relaxed Top-1；
- macro AP、MRR、Hit@3、Hit@5；
- tie-aware expected far errors；
- patient-cluster bootstrap 95% CI。

还必须报告两个拆分审计：

1. 旧 101 名 C18-complete 患者交集上，新 replay 与冻结 v11.1 OOF 的逐患者 paired difference；
2. 患者 `10489` 单独列出事件数、reference positive set、Top-1 和 one-hop 结果，明确标注为 identity-recovery extension，而不是 fresh test。

strict 80% 与 one-hop 85% 只是预先给定的临床目标门，不参与模型选择。只新增 1 名患者，因此不得把事件数从 984 增至 1145 错写为有效样本量增加 161；统计独立单位仍是患者。

## 5. 声明边界与停止规则

本轮可以声称：固定 LaBraM v11.1 在更完整、经身份验证的 public-development signal join 上的内部 OOF 表现。

不得声称：

- external/independent/zero-shot validation；
- TUSZ 发作起点等于 SOZ 起点；
- scalp electrode reference 等于 cortical 或 SEEG SOZ；
- private 结果已被使用；
- 达到临床部署标准，除非冻结 private zero-adaptation protocol 后实际满足预注册门槛。

若新队列、旧 core cache parity、固定 fold、C18 完整性或 target/private access boundary 任一失败，则停止，不以放宽断言或删除困难病例完成训练。
