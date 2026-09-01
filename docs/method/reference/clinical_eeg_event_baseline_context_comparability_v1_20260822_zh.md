# 事件级 baseline/context comparability 合同 v1

> 状态：2026-08-22 additive shadow sidecar；实现与定向测试完成，但未连接
> private、Findings 或 report route。它是方法约束，不是临床验证结果，也不提供
> 年龄/状态匹配的规范性正常值。2026-08-22 安全审计进一步确认：v1 尚无可信
> registry 将 canonical view、逐段质量、相似度测量向量和 calibration artifact 与
> sidecar 逐项交叉绑定；因此当前只输出 measurable/comparable candidate，所有
> distant-background、emergence、return/recovery 报告支持权限均强制失败关闭。

## 1. 现状审计

| 层 | 当前已有信息 | 尚不能证明的命题 |
|---|---|---|
| `event_eeg_findings_v3.context` | queried/local/distant intervals、一个全局 `background_status` 和 `contamination_risk` | 每段逐段质量、是否越过保护区、view/reference 是否相同、相似度及用途权限 |
| deterministic v1/v2/v3 producer | S0 内完整窗数量足够即设 `background_available`；背景频谱可测 | “较早”不等于未受相邻事件、伪迹或状态漂移污染，也不证明与事件目标技术可比 |
| recovery producer | `recovery_context_recording_seconds` 内有可用 offline amplitude 即输出 present `recovery_context_profile` | 没有和合格的事件前参考作同 view/reference 比较，不能证明 return/recovery |
| adaptive/profile | 有 final window、S0--S3、protection zone、左右删失 | `S0`/`S3` 是计算状态，不是合格背景或临床恢复 |
| BA-IEG P0/dense sidecar | 接收 background intervals，并可计算 robust change | interval 本身没有污染、参考、相似度和 comparability receipt；不足以授权报告措辞 |

当前确定性测试事件的 protection zone 是 `[92,128] s`，S3/recovery carrier 是
`[118,128] s`。后者仍位于 protection zone 内，因此新合同将其保留为“可测的
return candidate”，但明确拒绝 `recovery_support`。

## 2. 冻结语义：可测不等于可比

每个事件增加独立 sidecar：

```text
event_eeg_findings_v3（内容哈希绑定）
  + protection zone / onset-offset / censoring
  + physical context segments
      local_pre_event | distant_pre_event | distant_other_context
      event_emergence | post_event_return_candidate
  + per-segment view/reference/clock/bandwidth/unit set
  + quality + contamination receipts
  + calibrated similarity comparison
  -> technical comparability
  -> purpose-specific permissions
```

逐段必须保存 recording-relative `[start, stop)`、可用 view 的物理覆盖区间、与
protection zone 的关系/距离/重叠量、质量与污染资格、完整 view/reference 哈希。
背景角色只允许在保护区之前；local/distant 的距离阈值、最低参考时长和最低返回
观察时长都进入 source-development-locked policy receipt，不能由报告生成器修改。

技术可比采用严格的共同支持原则：canonical signal、view receipt、transform、
reference、unit set、rational clock、有效带宽和 quality mask 必须完全相同。任何
padding、imputation、edge contamination、质量不合格、候选事件污染或保护区重叠
均 fail closed。

## 3. 相似度与用途权限

相似度不能由裸分数直接授权。每个 comparison 绑定 feature set、方法、policy、
方向、阈值、不确定带和 calibration receipt，输出四态：

```text
matched | not_matched | uncertain | not_evaluable
```

下列规则是 **v2 trusted-receipt 目标语义**，不是当前 v1 已获得的报告授权：

- distant background 只有与同事件 local pre-event reference 在相同 view/reference
  下 `matched`，才可进入后续 comparison；
- event-emergence 与合格 reference 技术可比且校准为 `not_matched`，才可支持
  EEG-only 的相对变化；这仍不能创建 onset 或 SOZ 正证据；
- post-event target 必须位于保护区及观察到的 offset 之后、事件不右删失、观察
  时长合格，并与 reference 校准为 `matched`，才可支持“向本记录内参考返回”；
- `recovery_support` 仅表示上述 within-record relative return，不等于正常化、临床
  恢复或发作终止诊断。

当前 v1 即使收到自报为 `source_dev_locked` 的 ID/SHA，也只记录技术可比候选和
相似度候选，不能据此授权报告。原因是调用者目前仍可自报 view role、quality、
contamination、feature vector、similarity score 与 calibration 形状；只校验字段和
自哈希并不等于验证这些量来自可信 producer。代码据此固定拒绝：

```text
distant_background_reference
event_emergence_support
return_toward_reference
recovery_support
```

`context_measurement` 与 `within_record_relative_measurement` 仅表示工程侧存在可测
区间/技术可比 comparison candidate；它们不得被解释为临床支持或报告可用术语。
为避免下游误用 `permissions.*.authorized`，当前
`validate_baseline_context_claim_authorizations()` 会先核对 candidate 的 context/
comparison 绑定，再对任何非空 claim 列表统一返回
`v1_baseline_context_sidecar_is_candidate_only_not_report_authorization` 错误。也就是说，
v1 sidecar 可以被用于工程审计和未来 v2 资格化输入，但不能为当前报告授权任何句子。

无论是否可比，本合同永远拒绝：

```text
background_normality_statement
background_abnormality_statement
onset_support
soz_support
```

原因是本输入没有年龄、觉醒状态、药物和规范性参考，也不能让 offline context
创建 future-free onset 证据。

## 4. 与 BA-IEG 和报告的接口边界

P0 token 与 dense measurement sidecar 可以继续计算原始逐窗量，但
`baseline_delta`、background contrast、return/recovery 资格必须在未来显式绑定本
sidecar 的 authorized comparison ID。当前不改 P0 tensor、不改 v3 schema，也不接
报告路由，避免改变已冻结实验语义。

未来接线前必须增加三道 gate：

1. Findings measurement 的 `background_reference_ids` 映射到合格 context ID；
2. `termination_recovery` present atom 映射到 authorized return comparison；
3. report claim 通过 `validate_baseline_context_claim_authorizations`，禁止文本层绕过。

在增加 trusted canonical-view、quality/contamination、comparison-measurement 与
calibration registries 之前，第 2、3 道 gate 不得开启 emergence/recovery 支持。
claim gate 还必须要求每个 cited comparison 的 target/reference context endpoints
与 claim 的 context IDs 精确一致，不能分别通过 aggregate union 后交叉拼接。

## 5. 已实现与测试

- 实现：`src/clinical_eeg_long_recording/event_baseline_context_comparability.py`；
- 绑定并复核 v3 event hash、window/protection 和原 context projection；
- recompute segment eligibility、technical comparability、similarity status、distant
  qualification 和 aggregate permissions，不能只改布尔量伪造授权；
- 定向测试覆盖 candidate-only return、当前 S3 位于保护区、reference mismatch、
  normal/abnormal 永久拒绝、context projection 漂移、permission forgery、未校准
  阈值、mixed local/distant reference，以及 comparison-context 交叉拼接拒绝。

仍未完成的是：真实数据 quality/contamination/similarity producer、source-dev
阈值冻结、专家资格集验证以及与 Findings/report 的受控接线。因此不得把软件合同
通过解释为临床有效性已经成立。
