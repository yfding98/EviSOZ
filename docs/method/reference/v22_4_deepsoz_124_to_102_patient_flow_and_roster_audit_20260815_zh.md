# v22.4 DeepSOZ 124→102患者流与双102-roster审计

**日期：** 2026-08-15  
**性质：** 数据谱系和评价分母修复；不训练、不选模、不改阈值、不访问private  
**机器结果：** `outputs/deepsoz_124_to_102_patient_flow_v22_4_20260815/result.json`

## 1. 决定性结论

DeepSOZ源清单的124名患者到当前102名C18定位患者之间，已经没有尚待“补映射”的稳定、完整C18
患者。22名未进入主定位分母的患者完全闭合为：

| 互斥类别 | 患者数 | 可否进入当前C18 gold |
|---|---:|---|
| 当前identity-v16稳定C18定位主队列 | 102 | 已进入 |
| 有合格信号、稳定target，但O2 reference缺失 | 1（258） | 否；不能补0或改成C17后冒充同终点 |
| 有合格信号、患者内target冲突，但仍有稳定正例可作masked auxiliary | 9 | 否；v17只作train-only partial-label消融，不能作gold评价 |
| 有合格信号、患者内target冲突且无稳定in-head正例 | 2 | 否；不能构造正例 |
| target稳定，但没有任何合格信号 | 4 | 否；不能绕过因果窗口/QC |
| 没有合格信号，且原target流程也无strict input | 6 | 否 |
| **合计** | **124** | 互斥且完备 |

因此，继续改crosswalk、放宽warm-up、把variable label取union/majority、给缺失O2补负类，都不能合法地
产生新的同终点患者。9名masked-variable auxiliary已经在v17使用，strict净增`0/102`，停止门失败。

## 2. Target-independent信号流

完整identity overlay先在不读取SOZ target的producer中重放：

```text
652 records / 124 identity patients / 1812 local seizure candidates
                    |
                    | direct standard-19 + causal [-12,+48) + QC
                    v
1364 eligible events / 114 patients
448 excluded events / 10 patients with zero eligible events
```

10名零信号患者的全部19个候选事件原因是：

- `insufficient_warmup`：8名、12个事件；
- `ambiguous_standard19`：1名、6个事件；
- `signal_qc`：1名、1个事件。

这里没有使用target值决定signal eligibility。`insufficient_warmup`不是可随性能需要删除的装饰条件：
当前因果IIR需要冻结的历史支持，事后放宽会改变预处理定义并再次消费已开标target。`signal_qc`患者也
不能因为只有一个事件就让坏导联进入CAR并污染所有通道。

## 3. Signal与target在最后一步相交

Target artifact在124名患者上给出：

```text
107 stable target-policy eligible
11 patient-variable target
6 no-strict-input target quarantine
```

与114名signal-eligible患者相交后：

```text
signal ∩ stable target = 103 patients
  -> 102 complete C18 localization primary
  -> 1 missing-O2 partial reference (258)

signal ∩ variable target = 11 patients
  -> 9 masked-variable auxiliary admitted
  -> 2 no stable unmasked in-head positive

no signal = 10 patients
  -> 4 otherwise target-stable
  -> 6 no-strict-input quarantine
```

这解释了为什么“114名有信号”不能写成114名SOZ训练/评价患者，也解释了为什么11名variable患者不能
通过简单多数投票变成新gold。

## 4. 两个“102名患者”不是同一个roster

这是本轮发现并必须修复的论文口径：

| 名称 | 患者/事件 | 用途 | 独有患者 |
|---|---:|---|---|
| legacy event-evidence core | 102 / 988 | 旧事件typed facts与v22 event reports | 258 |
| current localization primary（identity-v16） | 102 / 1,145 | 当前H-only训练、OOF定位、72.55%及risk--coverage | 10489 |
| 交集 | 101 patients | 可以同时拥有legacy event facts和当前定位 | 无 |

两者不能只因患者数相同而逐行join，也不能写成“同一个102人/988事件系统产生了当前定位结果”。正确
表述为：

> 当前SOZ定位指标在identity-v16的102名患者、1,145次合格事件袋上计算；逐事件typed-fact报告沿用
> legacy 102名患者、988事件证据core。二者交集101名患者，v22报告器对roster差异fail closed。

具体处理：

- patient 258在legacy core有4个event reports，但因O2 reference不完整，当前定位写
  `localization_unavailable`并隐藏排名；
- patient 10489在current localization primary有27个事件和患者级候选，但不在legacy event-fact core，
  因此只有patient report，不伪造event时间、双极边或形态事实；
- 其余101名患者可以安全绑定两层，但event事实仍不是H-only分数的因果解释。

## 5. 性能数字绑定哪个roster

下列现行结果全部绑定**current localization primary 102 patients / 1,145 events**：

- full-coverage H-only strict `47/102=46.08%`；
- full-coverage H-only neighborhood-4 `74/102=72.55%`；
- selective public `61/81=75.31% @ 79.41% coverage`；
- v22.3 public risk--coverage/AURC。

`988 events`只描述legacy event-evidence/report core，不是上述定位器的完整事件输入，也不是独立SOZ
样本数。Private结果不受这两个public roster名称修复影响。

## 6. 对“600例TUSZ子集”的最终写法

论文数据流程固定写：

```text
DeepSOZ source overlay: 652 records / 124 patients
local conservative mapping: 607 unique records
target-independent signal universe: 1364 events / 114 patients
current C18 localization primary: 1145 events / 102 patients
legacy event-evidence report core: 988 events / a different 102-patient roster
```

摘要和主结果不得写“600例患者”“1145例SOZ”或“988个独立SOZ样本”。所有public定位CI、bootstrap和
性能分母仍是current localization primary的102名患者。

## 7. 还能否从剩余22人继续训练

不能把22人视为一个待恢复训练池：

- 9名auxiliary已经被合法使用且strict失败；
- 2名没有稳定positive，无法定义positive-set loss；
- 1名partial reference只能做coverage/接口QA；
- 10名没有合格信号，不能进入EEG定位器。

它们仍可支持以下受限工作：

- variable-label sensitivity和partial-label方法负结果；
- target-free输入失败原因、missing-channel和abstention flow；
- 数据收集规范：完整C18、可用pre-onset context、unknown/variable状态保留。

它们不能恢复模型搜索授权。新的性能增量仍需要lineage-new、patient-level、同C18终点患者。

## 8. 实现与验证

- 审计器：`scripts/audit_deepsoz_124_to_102_patient_flow_v22_4.py`
- 结果：`outputs/deepsoz_124_to_102_patient_flow_v22_4_20260815/result.json`
- 测试：`tests/test_deepsoz_124_to_102_patient_flow_v22_4.py`
- 正式回归：`1 passed`

审计器只读取既有signal/target/join/report receipts和JSONL身份；没有读取raw EEG或模型prediction，
没有训练、选模、阈值搜索或private访问。
