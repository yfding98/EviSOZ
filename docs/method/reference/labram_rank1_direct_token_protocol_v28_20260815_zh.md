# LaBraM rank-1 direct-token positive-set MIL v28 冻结协议

**冻结日期：** 2026-08-15，正式 102 人 OOF 结果打开前  
**目的：** 在不替换、不微调 LaBraM，且不读取 private EEG/标签的条件下，检验早期 65 人机制实验中表现最强的 low-capacity direct-token head 能否迁移到 identity-v16/v17 的 102 人正式队列。  
**性质：** public-development exploratory recovery；102 人已被反复使用，任何增益都不是 fresh confirmation。

## 1. 选择该候选的依据

现有 patient-OOF 证据如下：

- 8 s context、QKV-LoRA/PEFT、onset contrast 与复杂 temporal gate 均未超过对应冻结 anchor；
- v14 的 `low_capacity_direct_token_equal_positive_set_control` 在独立的早期 65 人 patient-OOF 中取得 strict `40/65=61.54%`、official-neighborhood-4 `57/65=87.69%`；
- robust pooling 与 equal pooling 结果相同，因此本协议选择更简单、跨域假设更少的 equal complete-bag pooling；
- v17 表明 masked-variable 患者主要改善 AP/Hit@K，因此保留 9 名 train-only partial-label auxiliary，但其未知位置不进入 loss。

这不是根据 private 结果选择模型；private 路径在 runner 中不可达。

## 2. 冻结结构

```text
official pretrained LaBraM-Base, frozen through block 9
  event prefix [15,77,200]
  -> remove CLS and restore physical slots [19,15,4,200]
  -> mean four slots inside each independent 4-s call [19,15,200]
  -> pre mean tiles 0:3                         [19,200]
  -> early mean tiles 3:6                       [19,200]
  -> late mean tiles 6:15                       [19,200]
  -> early-pre and late-early                   [19,2,200]
  -> five phase components                      [19,5,200]
  -> shared rank-1 scorer w[200] x a[5] + bias  206 trainable parameters
  -> event electrode logits [19]
  -> arithmetic mean over every seizure of one patient
  -> stable-train-only Jeffreys electrode prior
  -> C18 positive-set probability-mass loss
```

所有事件都保留，不选择“最好发作”；患者而不是事件等权。时间分段是相对已确认临床事件锚点的信号描述，不称 SOZ onset 或 propagation。

## 3. 数据与 fold 防火墙

- stable evaluation：102 patients / 1,145 events；固定 identity-v16 五个患者 fold；每个 held 患者所有事件完全隔离。
- masked-variable auxiliary：9 patients / 182 events；与 stable 患者及事件不重叠；每个 outer fold 排除同 fold auxiliary，auxiliary 永不评价。
- transform：本候选不拟合 PCA/scaler。
- prior：只由当前 stable outer-train 患者拟合；auxiliary 不影响 prior。
- foundation：只读取已经冻结的 target-free prefix；foundation trainable parameters=0。
- forbidden：private EEG、private target、TUSZ involvement label、clinical report text、历史 private prediction/metric、伪 SOZ 标签。

## 4. 训练合同

- optimizer：AdamW；learning rate `3e-3`；weight decay `1e-2`；100 epochs；gradient norm clip `1.0`。
- seeds：`20260828 + 1000 * outer_fold`；final refit seed `20265828`。
- loss：每患者一个 positive-set mass NLL；没有逐通道 zero-BCE、邻域训练标签、region loss、pairwise spread loss或 private calibration。
- candidate count：1；不扫描 seed、epoch、learning rate、weight decay、phase boundary、pooling、LaBraM block 或 auxiliary weight。

## 5. 预先冻结的 public 晋级门

候选相对于 v17 `masked_variable_auxiliary_full` 必须全部满足：

1. strict Top-1 不低于 `51/102=50.00%`；
2. official-neighborhood-4 至少 `80/102=78.43%`，且严格高于 v17 的 `78/102`；
3. macro average precision 严格高于 `0.5412731`；
4. far error count 不高于 `24`；
5. 五个 outer folds 中至少四折 strict 不低于 v17；
6. 输出完整、finite，PZ 永不进入候选。

任一失败即 `PUBLIC_NO_GO`：不建立 private direct-token 特征、不运行 private evaluator、不改变既有 v21 部署。全部通过才允许在 102+9 人上作一个 final refit，随后冻结 target-blind private prediction。

## 6. 文献对应关系与不能迁移的结论

- DeepSOZ（MICCAI 2023）支持患者级多发作聚合和时间证据，但其 detection attention 不能改称 SOZ onset。
- Deep Sets（NeurIPS 2017）支持对无序事件袋采用 permutation-invariant 聚合；不支持把事件当独立样本。
- LaBraM（ICLR 2024）支持使用预训练 channel-patch representation；其 TUSZ pretraining exposure 必须披露。
- Conformal Risk Control（ICLR 2024）要求独立 exchangeable calibration；本协议不在已开封 private 上选择拒答阈值。
- CerebraGloss 可借鉴 typed-fact report rendering，但不提供 electrode×time SOZ grounding，不参与本模型。

## 7. Claim boundary

即使通过，输出仍是 standard-19 physical scalp-electrode clinical SOZ-reference candidate。不得称 cortical SOZ、EZ、传播源或手术靶点。strict 与 neighborhood-4 必须并列；private 若再次运行只能称 post-open exploratory concordance。

