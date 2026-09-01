# DeepSOZ官方权重在本地TUSZ映射子集上的信号版本迁移结果

**日期：** 2026-08-14  
**状态：** 完整102人评估、独立审计通过  
**证据边界：** published-weight signal-version transfer with official held-out folds；不是原始数据精确复现，也不是独立外部验证。  
**数据防火墙：** private未读取；36名无SOZ标签患者未作为SOZ target；只使用患者实际位于官方test fold时的发表权重。

## 1. 为什么做这个诊断

当前LaBraM路线未达到80% strict或85% relaxed。为区分“foundation encoder/当前短窗管线失败”与
“DeepSOZ标签粒度和头皮可辨识性限制”，本轮固定重放DeepSOZ官方15个test fold权重，并尽可能复现
官方最终notebook的信号合同：19导REF、200 Hz、1.6--30 Hz四阶Butterworth/Gustafsson双向滤波、
逐导`±2 SD`裁剪、最长600秒、record内多发作联合标准化、每record最多10次发作和右补零。

本地TUSZ信号版本与论文原始版本不完全一致，因此结果只能回答：发表模型迁移到本地可追溯信号后，
是否明显优于当前LaBraM管线。它不能回答官方仓库在原始私有快照上是否可逐位复现。

## 2. 完整结果

所有模型使用相同102人、相同C18目标和PZ mask计分：

| Pipeline | Strict Top-1 | Neighborhood-2 | Neighborhood-4 |
|---|---:|---:|---:|
| DeepSOZ官方held-out ensemble | 48/102 = 47.06% | 64/102 = 62.75% | 76/102 = 74.51% |
| LaBraM identity-v16 | 51/102 = 50.00% | 68/102 = 66.67% | 77/102 = 75.49% |
| LaBraM auxiliary-v17 | 51/102 = 50.00% | 69/102 = 67.65% | 78/102 = 76.47% |

Wilson 95%区间为：DeepSOZ strict 37.66%--56.68%，DeepSOZ neighborhood-4
65.27%--81.97%；LaBraM-v17 strict 40.47%--59.53%，neighborhood-4 67.37%--83.65%。
这些区间描述同一反复使用public-development roster上的不确定性，不使其变成confirmatory证据。

### 2.1 患者配对差值

差值定义为`DeepSOZ - LaBraM`，20,000次patient bootstrap：

| Comparator | Endpoint | Difference | 95% CI | DeepSOZ-only correct | LaBraM-only correct |
|---|---|---:|---:|---:|---:|
| v16 | strict | -2.94 pp | [-14.71, +8.82] | 18 | 21 |
| v16 | neighborhood-2 | -3.92 pp | [-15.69, +7.84] | 17 | 21 |
| v16 | neighborhood-4 | -0.98 pp | [-11.76, +9.80] | 16 | 17 |
| v17 | strict | -2.94 pp | [-15.69, +8.82] | 19 | 22 |
| v17 | neighborhood-2 | -4.90 pp | [-16.67, +6.86] | 16 | 21 |
| v17 | neighborhood-4 | -1.96 pp | [-12.75, +8.82] | 15 | 17 |

没有终点支持DeepSOZ官方权重明显优于LaBraM；所有差值区间跨0，且点估计均略低于LaBraM。

### 2.2 官方三个单重复

| Repeat | N | Strict | Neighborhood-2 | Neighborhood-4 |
|---:|---:|---:|---:|---:|
| 0 | 99 | 47.47% | 59.60% | 67.68% |
| 1 | 100 | 47.00% | 57.00% | 63.00% |
| 2 | 98 | 43.88% | 63.27% | 73.47% |
| repeat macro mean | -- | 46.12% | 59.95% | 68.05% |

held-out ensemble相对单重复均值主要改善neighborhood-4（约+6.46 pp），strict只约+0.94 pp。
因此官方管线的集成主要减少近邻级错误，不能解决集合内精确电极排序。

## 3. 独立审计

独立审计器未调用主评估器的计分函数，并重新核对：

- 102名唯一患者、297个patient-fold预测；
- 每个预测只来自该患者实际出现的官方test fold；
- 8名重复覆盖不完整患者恰为`11604,12742,12858,12870,7032,7584,8608,9578`；
- 所有score有限、C18目标一致、PZ从不成为Top-1；
- ensemble等于患者实际held-out重复的score均值；
- exact、neighborhood-2和neighborhood-4逐行重算与保存结果完全一致；
- 5名患者因官方兼容路径对FZ/PZ作零填充；去除这5人后，主要结论不变。

审计结果为`pass`。

## 4. 关键诊断

### 4.1 更换DeepSOZ专用网络或恢复其600秒上下文不是充分解

官方模型、官方held-out fold和长上下文并未超过现有LaBraM。因本地信号版本不同，不能断言其架构
无效；但它直接否定了“只要把LaBraM换成DeepSOZ/CNN-LSTM式管线就会接近80%”这一恢复假设。
同理，现阶段没有证据支持仅换CBraMod、CerebraGloss或另一foundation backbone即可跨越约30 pp
strict缺口。

### 4.2 长上下文可能改变个别病例，但不是总体瓶颈

DeepSOZ实际消费1,313个事件，LaBraM冻结合同消费1,145个事件；仅56/102名患者的事件数相同，
46/102不同。因此这是完整pipeline比较，不是encoder-only消融。事件数分层表现高度非单调，且小层
样本数有限，不能据此挑新的maxSeiz或窗口。最多可得出：更多事件/更长上下文并不自动带来更高总体
strict定位。

这里三个常见事件数不是互相矛盾，而是不同冻结合同的分母：

| Count | 含义 |
|---:|---|
| 988 | 最早冻结的102人、因果`[-12,+48) s`、完整warm-up/QC signal-eligible core |
| 1,145 | identity recovery后、固定C18 complete-case LaBraM primary carrier |
| 1,528 | 当前102人在identity-v3 receipt中可追溯的全部annotation boundaries，包括按LaBraM因果合同排除但官方crop可以右补零处理的边界 |
| 1,313 | 上述边界按DeepSOZ官方每record最多10次发作裁剪后实际前向的crop数 |

独立重算得到543个records，其中19个records超过10次发作，官方上限共删除215个边界，
`1,528 - 215 = 1,313`。因此不能把1,313写成“新增训练样本”，也不能将两条pipeline差异归因给
encoder本身。

### 4.3 exact瓶颈与标签粒度一致

官方模型单重复到ensemble对strict几乎不增益，却明显改善neighborhood-4；正符合DeepSOZ多数标签是
临床区域展开positive set、没有集合内电极优先级的事实。single-positive仅10人，官方strict为1/10，
样本过小，不能据此训练新的single-positive专用分支。

### 4.4 两条管线错误互补，但当前不能据此调融合

DeepSOZ ensemble与LaBraM-v17 strict同时正确29人、同时错误32人；DeepSOZ-only正确19人、
LaBraM-only正确22人。两者Top-1完全一致仅26/102，其中strict为19/26、neighborhood-4为24/26。
这说明模型分歧可作为**未校准的epistemic uncertainty reason code**；它不授权在这102人上搜索
融合权重、选择性阈值或新ensemble。任何融合候选只能预注册后在新的label-fresh development cohort
上比较。

## 5. 对最新方法的冻结修订

下一版信息流应为：

```text
standard-19 EEG
  -> official pretrained LaBraM representation H
  -> source-native M / scalp-visible I / self-supervised V
  -> Q_obs signal observability gate（只降权/拒识）
  -> finite evidence bottleneck
  -> low-capacity patient-level set reasoner
  -> C18 ranking
  -> uncertainty layer:
       signal observability reason codes
       frozen-fold predictive disagreement
  -> deterministic facts-locked report
```

冻结判决：

1. 不以替换LaBraM为当前恢复主线；官方DeepSOZ作为外部实现风格comparator保留。
2. 不在102人/private上继续扫描长窗、maxSeiz、融合权重、图、loss、seed或阈值。
3. 36人只做target-free adaptation、QC、M/I/V native fidelity和Q_obs perturbation consistency。
4. Q_obs与模型分歧必须分开：前者是信号可观测性，后者是预测不一致；二者都未经独立校准。
5. private v18结果保持冻结，不运行新模型或新报告措辞。
6. 新的SOZ准确度提升只能由未来同终点、patient-level、label-fresh队列确认。

## 6. 对80%/85%目标的直接结论

当前三条完整公开管线均约47%--50% strict、74.5%--76.5% neighborhood-4。没有证据证明80% strict
或85% relaxed已经达到，也不能承诺通过换backbone达到。最有价值的下一步不是继续在102人上寻找
一个更高数字，而是：

- 在不重开private的条件下完成Q_obs、typed evidence和facts-locked reporting；
- 用新的连续入组队列获取与模型输入信息集匹配的patient-level scalp candidate set；
- 同时保存integrated clinical reference并按semiology/imaging availability分层；
- 模型与阈值冻结后一次性报告full-coverage strict、relaxed、Hit@K、coverage和patient-cluster CI。

若没有新的同终点标签，方法和安全性仍可推进，但“SOZ Top-1提升”在统计上无法闭环。

## 7. 机器证据

- 完整预测：`outputs/deepsoz_official_local_oof_full.json`
- 独立审计：`outputs/deepsoz_official_local_oof_full.audit.json`
- 患者对齐比较：`outputs/deepsoz_labram_endpoint_aligned_comparison.json`
- 评估器：`scripts/evaluate_official_deepsoz_oof.py`
- 审计器：`scripts/audit_official_deepsoz_oof.py`
- 比较器：`scripts/compare_deepsoz_labram.py`

当前`/mnt/hd1`为100%，上述新资产尚不能安全迁入仓库。释放空间后必须迁入正式目录并保留本结论的
claim boundary。
