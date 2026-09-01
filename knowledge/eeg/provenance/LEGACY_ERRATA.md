# 1.0 历史设计稿勘误与迁移阻断项

`LearningEEG_ScalpEEG_SOZ_External_Knowledge_Base_v1.0.md` 不再是活动知识源。以下内容在迁移到
2.0 卡片时必须修正，不能逐字复制：

1. `definite_evolution` 必须按采用的标准/profile 写全连续变化次数、频率步长、每阶段周期数、
   稳定时间和持续时间要求；单次方向性变化不够。
2. ACNS 2021 IIC 不使用 `possible_iic/definite_iic` 分级，GRDA 不属于 IIC；常规癫痫监测与
   危重症术语不得静默混用。
3. TIRDA 与 temporal LRDA 只能标 `related_to`，不能同时标成 alias。
4. 相位反转提示共享电极的头皮电位局部极值候选，不是最大梯度、皮层源或 SOZ。
5. `F7-T7 + T7-P7` 的示例应分别保存 `field_electrodes=[F7,T7,P7]`、
   `phase_reversal_electrode=T7` 和区域解释，不能无依据排除 P7。
6. 通用卡片不得硬编码左/右，应使用占位符或无侧别概念。
7. `clinical_or_invasive_soz` 必须拆成多模态临床假设与侵入性 EEG 起始区；EZ 单独保存。
8. A–D 证据等级、概率到措辞阈值、评分公式、损失函数、模型输出头和数据集目标映射均不是
   通用医学事实。
9. MI/VR/ICA、LaBraM、DeepSOZ、oracle onset、私有标签和项目路径不进入活动知识。
10. 人读 Markdown、来源摘要和概念卡最终应由同一审核来源注册表确定性生成，不能继续作为
    三份独立真源。

所有 2.0 卡目前均为 `draft_clinical_review_required`；缺少精确条款/页码 locator 的卡不能用于
临床部署。
