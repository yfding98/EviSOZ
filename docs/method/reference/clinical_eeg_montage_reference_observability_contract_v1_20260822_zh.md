# 临床 EEG 采集 montage/reference 可观测性合同 v1

日期：2026-08-22  
状态：研究性 canonical/adaptive shadow 合同；未连接 private/report route

## 1. 为什么必须独立建模

EDF 中名为 `FP1` 的数组不自动等于“可跨导比较的 FP1 电极电位”。如果输入已经是双极导联、不同通道使用不同参考，或参考无法从 signal label 观察到，再计算 CAR/Laplacian 会产生一个数值张量，却不能产生合法的头皮电极场。此错误会直接污染最早通道排序和 SOZ 证据链。

本合同只读取有序 EDF signal labels，不读取 EDF+ annotation、患者/记录头、Excel、医生标签、临床文本或视频。其结论是“参考兼容性”，不是物理参考已被验证，更不是皮层源定位。

## 2. 四类采集 montage

| 类别 | 可观察条件 | 派生 TCP/CAR/Laplacian |
|---|---|---|
| `common_compatible_referential` | 直接 Standard-19 电极信号使用同一可识别参考 token，无双极/未知 EEG signal 混入、无重复电极 | 可物化；逐输出仍需完整 observed carriers |
| `already_bipolar` | 只有可解析的 Standard-19 双极 signal | 禁止再次构造电极 CAR/Laplacian/TCP 场；应走 bipolar-native adapter |
| `mixed` | 直接/双极混用、多个参考 token、已知和未知参考混用，或同一电极重复 | 禁止派生场 |
| `unknown` | 直接电极的参考不可观察，或 EEG label 无法解析 | 禁止派生场 |

`REF/LE/AR/AVG/AV/CAR/A1/A2/M1/M2` 等 token 只表示 label 兼容。合同明确保存 `shared_edf_label_token_compatibility_not_physical_reference_verification_v1`。

## 3. 可重放矩阵诊断

采集参考首先表达为 signal-by-latent-node incidence matrix。每个派生参考也保存冻结线性矩阵。所有矩阵均记录：

- 行、列 unit 顺序及 `matrix_sha256`；
- numerical rank、row/column nullity；
- largest/smallest non-zero singular value condition number；
- 由“同一输出行中的非零 carriers”形成的 support graph、connected components。

该诊断能暴露仅凭 shape 看不出的结构。例如本项目 frozen TCP-20 对 Standard-19 的矩阵 rank 为 16，support graph 有 3 个 component，因为 FZ/PZ 没有进入任何 TCP carrier；CAR 和当前 Laplacian 的 rank 均为 18、一个 component。这些视图是相关变换，不能按独立证据票数相加。

## 4. 逐输出资格而非全局布尔值

每个输出保存 `quality_dependency_channel_ids`、缺失 carriers、资格和 reason codes：

- TCP lead：两个端点都必须直接 observed；输出仍然是 lead，不拆为端点电极证据；
- frozen Standard-19 CAR：任一 Standard-19 carrier 缺失时，所有 CAR 输出都不具备电极场证据资格；有限零 carrier 只可用于固定 tensor plumbing；
- Laplacian：target 及冻结邻居全部 observed 时，该局部输出才有资格；无关区域的缺导不自动取消它；
- montage 不是 `common_compatible_referential` 时，三个派生 family 在 tensor 构造前统一 fail closed。

所有空间输出只可支持 scalp montage consistency，不授权 source localization、cortical SOZ、epileptogenic zone 或手术靶点。

## 5. 坏导、step 和 gap 的污染传播

瞬时参考变换采用严格的非零 carrier union：一个 source quality primitive 影响所有矩阵中该 source 系数非零的输出，并保留同一 recording-relative half-open interval、severity 和 disabled evidence families。

投影 API 不能把调用者提供的字符串直接透传为质量语义。当前 validator 要求：
`quality_id` 是唯一安全标识符，通道列表为唯一 Standard-19 电极，时间端点为有限数值，
`severity` 仅可为 `limited|unusable`，disabled family 只能来自冻结的
`amplitude/morphology/spectral/spatial_field/high_frequency/waveform` 集合且不得重复；
`unusable` 必须关闭全部 evidence families。任一字段不满足即在参考传播前 fail closed。

例如 F7 step：

- 污染 `FP1-F7`、`F7-T7`；
- 污染全部 Standard-19 CAR 输出；
- 污染以 F7 为 target 或 neighbour 的 Laplacian 输出。

child view 可以增加 mask，不能删除 canonical/parent mask。空间参考本身不扩大或缩短时间区间。若 parent filter 对 step/gap 产生时间 ringing，其影响区间必须先由 parent preprocessing/QC 资格化并 mask；本合同不会用空间变换伪装已经解决该时间支持问题。这也是当前真实数据验证的独立 P0 gate。

## 6. 软件和版本边界

- 独立 receipt：`clinical_eeg_montage_reference_observability_v1`；
- 矩阵 receipt：`clinical_eeg_reference_matrix_observability_v1`；
- JSON Schema：`schemas/clinical_eeg_montage_reference_observability_v1.schema.json`；
- canonical materialization/producer 升为 v3；
- signal-view receipt 升为 v3；
- adaptive preprocessing/materialization 升为 v4，whole-record transform 升为 v2。

旧 canonical/adaptive receipt 因 schema 或 producer/method 不匹配而 fail closed。canonical materialization receipt 绑定 montage receipt SHA、类别和三种派生矩阵 SHA；adaptive observed-only CAR 另存其实际矩阵的 rank/condition/connectivity，并保持 `navigation_only=true`、`onset/findings_evidence_authorized=false`。

## 7. 尚不授权的事项

该合同没有证明 EDF header 与采集设备真实接线一致，也没有解决：未知私有 label 字典、EDF+D gap 的完整技术时钟恢复、IIR 伪迹 ringing 的临床安全 guard、坏导自动检测阈值、或真实患者上的 reader study。因此当前不能声称临床验证或 production ready，也不应连接报告主路由。
