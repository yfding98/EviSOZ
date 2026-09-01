# v60 开标后固定协议补充审计

## 1. 目的与不可改变项

本协议只补充审稿人导向的反证和公平 comparator，不修改正式 v29。以下项目保持冻结：

- standard-19、CAR19、200 Hz、`[-12,+48)` 秒输入合同；
- official pretrained LaBraM block-9；
- H/D carrier、C18 输出空间和 PZ mask；
- `0.5 H + 0.5 D` 概率融合；
- public 102-patient roster、五个 patient folds 和集合式 SOZ reference；
- private 88-event inference roster、51-event/23-cluster evaluable roster；
- strict physical-electrode Top-1 主终点以及 N2/N4 次终点。

研究者已经看过 private reference。因此下述新实验均属于 **post-open fixed-protocol audit**，不是 target-blind confirmation；private 结果不得用于选择网络、超参数、seed、阈值或替换 v29。

## 2. Raw200-Shallow comparator

### 2.1 输入和训练单位

- 每个事件输入：`[19,12000]`，200 Hz，60 秒 CAR19；
- 公共数据保留事件级波形，不在非线性特征提取前平均不同发作；
- 每个 epoch 每名 public 患者等概率抽取一个事件，使训练患者等权；
- OOF 推理对患者全部事件的候选概率等权平均；
- private 保持 88 个事件级预测，不读 significant/spread reference。

### 2.2 唯一网络配置

每个物理电极共享同一个 ShallowConvNet-style temporal-power scorer：

1. `Conv1d(1,32,kernel=101,stride=4,padding=50,bias=False)`；
2. 平方非线性；
3. `AvgPool1d(kernel=50,stride=12)`；
4. `log(clamp(power,1e-8))`；
5. 在 pre/early/late 三个固定相对事件区间分别计算均值和标准差；
6. `Linear(192,1)` 共享通道 scorer；
7. 加入 fold-local Jeffreys channel prior；
8. C18 candidate mask 后使用 positive-set probability-mass NLL。

该模型是为电极排序任务改写的 channel-local ShallowConvNet-style comparator，不称为 canonical EEGNet 或原版 ShallowConvNet。它没有 foundation/pretrained 参数。固定训练参数为：40 epochs、AdamW、`lr=1e-3`、`weight_decay=1e-3`、patient batch=8。只运行这一组配置。

### 2.3 评价

- public：patient OOF strict/N2/N4、Macro-AP；
- private：event-micro 和 patient-equal strict/N2/N4；
- v29 与 comparator 的患者/患者簇 paired bootstrap difference；
- 参数量、输入带宽和患者聚合方式必须同时报告。

## 3. 患者级标签置乱反证

标签置乱只在 public 训练折内进行：

- 以患者为最小单位置乱完整 positive set，不按事件置乱；
- held-out patient 的真实 reference 不进入对应训练折；
- 保持原 patient folds、输入 representation、candidate mask、loss 和训练预算；
- 每个 repetition/fold 使用独立且可复现的训练折内置乱；
- 正式 v29 不参与重选、集成或替换；
- private 数据和 reference 均不进入本实验。

至少运行 20 个 repetitions。报告 null strict/N2/N4/Macro-AP 分布、相对 prevalence-only 的差异以及正式 v29 在 null 分布中的位置。若置乱后仍稳定保留远高于 prior 的结果，必须停止可信性主张并检查患者泄漏或通道先验捷径。

## 4. Patient-partition stability

为补足仅改变 D-head 初始化的 v58，另做完整 H/D outer-partition audit：

- 使用 5 组不依赖 SOZ label、按患者事件数贪心平衡的替代五折；
- H 在每个替代 outer training fold 内重复原有 nested-L2 选择；
- D 保持固定 auxiliary roster、set-mass loss、优化器、epochs 和 seed policy；
- 每个替代 partition 都产生完整 public OOF prediction；
- private 只生成 reference-isolated 88-event prediction，再于独立 read-only audit 中评价；
- 所有 partition 均完整报告，不选择最优、不做 ensemble、不替换 v29。

该实验只审计低容量 H/D adaptation 对患者划分的敏感性，不审计 LaBraM 预训练随机性，也不会恢复 fresh private confirmation。

## 5. 允许与禁止的解释

允许：

- 比较冻结 foundation representation 与更公平的完整带宽 raw-waveform comparator；
- 用标签置乱反证训练—评价链条是否保留异常 target correspondence；
- 将所有 private comparator 结果称为 post-open descriptive audit。

禁止：

- 根据 private 结果改变 v29 或 comparator；
- 将 comparator 的较差结果解释为所有 raw-EEG 网络均无效；
- 将 private 称为 fresh/external validation；
- 将 N2/N4 称为 strict SOZ accuracy；
- 将本协议回溯描述为独立预注册。
