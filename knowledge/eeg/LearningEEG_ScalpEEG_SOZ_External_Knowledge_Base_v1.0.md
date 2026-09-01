---
title: "LearningEEG–头皮 EEG–SOZ 外挂知识库"
subtitle: "面向长程 EEG、发作检测、头皮发作起始区定位、SOZ 报告生成与证据反馈闭环"
version: "1.0.0"
date: "2026-08-28"
language: "zh-CN"
status: "legacy_migration_design_not_runtime_knowledge"
superseded_by: "knowledge/eeg/manifest.json"
runtime_indexed: false
primary_source: "https://www.learningeeg.com/"
source_scope:
  - "LearningEEG 12 个教学章节与 Atlas"
  - "IFCN 电极与发作间期癫痫样放电规范"
  - "ACNS 2021 重症 EEG 术语"
  - "ILAE 发作分类及癫痫术前评估区域框架"
intended_use:
  - "RAG 外挂知识库"
  - "EEG 结构化标注规范"
  - "SOZ 定位模型的中间语义监督"
  - "证据约束的 EEG/SOZ 报告生成"
  - "报告—信号一致性检查"
not_intended_use:
  - "替代神经科或临床神经生理医师诊断"
  - "由单一头皮 EEG 证据直接确定临床 SOZ 或致痫区"
license_note: "本文档为基于公开教学内容与标准的重新组织和概括，不包含 LearningEEG 原始波形图片或大段原文。"
---

# LearningEEG–头皮 EEG–SOZ 外挂知识库

> **迁移说明（2026-08-29）**：本文是 1.0 历史设计稿，混合了领域知识、RAG/报告策略、
> 项目模型建议和评估方案，不再作为活动知识入口或运行时索引。解耦后的活动知识以
> [`manifest.json`](manifest.json) 为入口；模型、训练、项目数据合同和流程接入方案位于
> 知识库目录之外。本文仅用于逐条迁移与审计，不能直接整篇送入患者推理。

> **核心原则**：信号事实决定模式解释；模式解释支持头皮定位；多发作、多模态临床证据支持 SOZ 假设；任何单一头皮波形、单一双极导联、单一相位反转或单一 IED 都不能直接证明临床 SOZ，更不能直接证明致痫区（EZ）。

## 目录

- [0. 给检索模型与报告模型的强制指令](#sec-0)
- [1. 知识本体总览](#sec-1)
- [2. WAVE：生理基础与波形原子语言](#sec-2)
- [3. ELEC 与 TECH：电极、派生导联、Montage 与技术参数](#sec-3)
- [4. CTX 与 BG：正常清醒、嗜睡、睡眠和诱发条件](#sec-4)
- [5. ART：伪迹知识体系](#sec-5)
- [6. VAR：正常变异与良性 Mimics](#sec-6)
- [7. 年龄特异模块：新生儿与儿童](#sec-7)
- [8. ABN：非癫痫样异常](#sec-8)
- [9. IED：发作间期癫痫样放电](#sec-9)
- [10. RPP 与 IIC：节律性、周期性和高风险模式](#sec-10)
- [11. ICTAL：发作期时空知识体系](#sec-11)
- [12. LOC 与 CLIN：从导联到头皮起始区，再到 SOZ/EZ](#sec-12)
- [13. 强制禁止推理规则](#sec-13)
- [14. 规范术语、旧词与别名映射](#sec-14)
- [15. 机器可读标注协议](#sec-15)
- [16. RAG 外挂知识库卡片规范](#sec-16)
- [17. 证据约束的 EEG/SOZ 报告体系](#sec-17)
- [18. 报告—定位模型反馈闭环](#sec-18)
- [19. 与 LaBraM 变体和端到端系统的结合](#sec-19)
- [20. 评估体系](#sec-20)
- [21. 建议的知识库目录结构](#sec-21)
- [22. 检索问答模板](#sec-22)
- [23. 重点概念快速索引](#sec-23)
- [24. 来源与版本管理](#sec-24)
- [25. 最终系统原则](#sec-25)

---

<a id="sec-0"></a>
# 0. 给检索模型与报告模型的强制指令

### 0.1 推理分层

所有输出必须区分四个语义层级：

1. **OBS：可观察信号事实（Observation）**  
   只描述波形、时间、导联、电极、频率、形态、演变、空间电场、伪迹和状态。
2. **PAT：EEG 模式解释（Pattern interpretation）**  
   如 IED、局灶性慢化、LRDA、BIRDs、演变性发作模式。
3. **LOC：头皮 EEG 空间定位（Scalp localization）**  
   如“最早头皮可见改变位于左前—中颞区”。
4. **CLIN：临床 SOZ/EZ 推断（Clinical inference）**  
   必须明确证据来源、置信度和限制；不得仅由头皮 EEG 单项结果自动升级。

正确示例：

```text
OBS：128.4 s 起，F7–T7 与 T7–P7 首先出现 5–7 Hz 节律性尖锐 theta，
     随后频率减慢、振幅增高，并在 2.3 s 后招募左旁矢状区。
PAT：左颞区演变性发作模式。
LOC：最早头皮可见电图改变在左前—中颞区最明显。
CLIN：多次发作若重复呈现，可支持左颞区作为非侵入性 SOZ 候选；
      仍需结合临床症状、影像、SEEG、手术及结局验证。
```

禁止示例：

```text
F7–T7 最早异常，所以 F7 就是 SOZ。
左颞尖波证明左颞叶是致痫区。
T7 相位反转等于皮层起源位于 T7 下方。
```

### 0.2 四值标签系统

关键属性不得只用 0/1。统一使用：

| 值 | 含义 |
|---|---|
| `present` | 已在可评价数据中观察到 |
| `absent` | 已在可评价数据中主动检查且未观察到 |
| `unknown` | 信息不足、未标注或尚未判断 |
| `not_assessable` | 因伪迹、缺失通道、片段截断、参考方式等原因无法评价 |

例如缺失 Fz/Pz 时：

```yaml
fz_involvement: not_assessable
pz_involvement: not_assessable
```

不得写成：

```yaml
fz_involvement: absent
pz_involvement: absent
```

### 0.3 证据来源与标签来源必须分离

所有标签建议携带：

```yaml
value: "left_temporal"
status: "present"
source: "human"          # human | clinical_report | dataset | model | heuristic | llm
confidence: 0.86
review_status: "verified" # unreviewed | reviewed | verified | disputed
```

LLM 生成结论不能自动成为金标准；模型伪标签不能覆盖人工临床标签。

### 0.4 头皮可见起始与生物学起始的边界

必须使用以下表述之一：

- `earliest_scalp_visible_change`：最早头皮可见改变；
- `scalp_electrographic_onset_region`：头皮电图起始区；
- `noninvasive_soz_hypothesis`：非侵入性 SOZ 假设；
- `clinical_or_invasive_soz`：临床综合或侵入性确认 SOZ；
- `epileptogenic_zone`：致痫区。

它们不是同义词。

---

<a id="sec-1"></a>
# 1. 知识本体总览

## 1.1 顶层命名空间

| 命名空间 | 全称 | 内容 | 典型概念 |
|---|---|---|---|
| `TECH` | Technical | 采集、显示、滤波、质量 | sampling rate、filter、sensitivity |
| `ELEC` | Electrode & Montage | 电极、派生导联、montage | electrode、derivation、phase reversal |
| `WAVE` | Waveform | 波形原子属性 | frequency、amplitude、morphology |
| `CTX` | Context | 年龄、状态、诱发条件 | awake、N2、PMA、photic |
| `BG` | Background | 正常/异常背景 | PDR、AP gradient、reactivity |
| `ART` | Artifact | 非脑源信号 | ocular、myogenic、electrode pop |
| `VAR` | Normal Variant | 正常变异及良性 mimic | wicket、RMTD、BETS、mu |
| `ABN` | Non-epileptiform Abnormality | 非癫痫样异常 | focal slowing、attenuation、breach |
| `IED` | Interictal Epileptiform Discharge | 发作间期癫痫样放电 | spike、sharp、polyspike |
| `RPP` | Rhythmic & Periodic Pattern | 节律/周期模式与 IIC | LRDA、LPD、GPD、BIRDs |
| `ICTAL` | Ictal | 发作期时空动态 | onset、evolution、recruitment |
| `LOC` | Localization | 头皮空间定位 | laterality、region、field、spread |
| `CLIN` | Clinical | 临床区域和综合推断 | irritative zone、SOZ、EZ |
| `REP` | Reporting | 报告语言、证据边界 | observation、impression、limitation |

## 1.2 推荐关系类型

```text
recorded_with       使用某采集或导联方式记录
occurs_during       出现在某意识/睡眠状态
provoked_by         由某诱发操作触发
has_frequency       具有某频率
has_morphology      具有某形态
has_polarity        具有某极性
maximal_at          在某电极或区域最大
phase_reverses_at   在某电极形成相位反转
has_field_to        向某电极/区域形成空间电场
precedes            时间上先于另一事件
follows             时间上晚于另一事件
evolves_to          演变为另一模式
recruits            招募某电极/区域
propagates_to       传播至某区域
co_occurs_with      与另一模式同时存在
mimics              容易模仿另一模式
differentiated_by   依靠某特征鉴别
supports            支持某个推断
contradicts         与某推断矛盾
limited_by          受某因素限制
derived_from        来源于某标注或证据
verified_by         由某金标准验证
```

## 1.3 区域层次建议

```text
hemisphere
├── left
├── right
├── bilateral
├── midline
└── generalized

region
├── frontal
│   ├── frontopolar
│   ├── anterior_frontal
│   ├── frontal
│   └── frontocentral
├── temporal
│   ├── anterior_temporal
│   ├── mid_temporal
│   ├── posterior_temporal
│   └── inferior_temporal
├── central
├── centrotemporal
├── parietal
├── occipital
├── posterior_quadrant
├── parasagittal
├── hemispheric
├── multilobar
└── generalized
```

区域输出应同时保存：

```yaml
laterality: left
region_primary: temporal
region_subtype: anterior_mid_temporal
spatial_granularity: regional
```

---

<a id="sec-2"></a>
# 2. WAVE：生理基础与波形原子语言

## 2.1 头皮 EEG 信号的物理边界

- 头皮 EEG 主要反映大量皮层神经元突触后电位的同步总和，而非单个动作电位。
- 信号经脑组织、脑脊液、颅骨和头皮发生容积传导。
- 脑源深度、皮层面积、同步程度、偶极方向和颅骨状态都会改变头皮可见性。
- 头皮最大电位点、相位反转点或最大振幅导联不必等于真实皮层源。
- 深部、范围小、切向或快速扩散的发作可能在头皮上晚出现、模糊、双侧同步或完全不可见。

## 2.2 极性与显示方向

在传统 EEG 显示约定中，负电位通常向上、正电位通常向下；但必须记录软件设置和导联定义，不能凭视觉方向脱离 montage 判断极性。

```yaml
polarity:
  value: negative
  convention_verified: true
  montage_verified: true
```

## 2.3 频带

| ID | 英文 | 中文 | 常用范围 | 说明 |
|---|---|---|---:|---|
| `WAVE.BAND.DELTA` | delta | δ/慢 delta | 约 0.5–4 Hz | 成人清醒状态持续出现通常异常；睡眠、儿童、新生儿需结合年龄和状态 |
| `WAVE.BAND.THETA` | theta | θ | 约 4–8 Hz | 嗜睡、儿童可正常；成人清醒局灶或弥漫 theta 需评估 |
| `WAVE.BAND.ALPHA` | alpha | α | 约 8–13 Hz | 成人 PDR 常位于该范围；也可见 mu、wicket 等 |
| `WAVE.BAND.BETA` | beta | β | >13 Hz | 前部常见；过多可与药物、肌电或病理快活动有关 |
| `WAVE.BAND.GAMMA` | gamma | γ | 常指 >30 Hz | 常规头皮 EEG 易受肌电和低通滤波影响，必须谨慎 |

训练时优先保存连续频率值，而不是只保存频带：

```yaml
frequency_hz:
  start: 6.2
  end: 3.8
frequency_band_initial: theta
frequency_band_final: delta
```

## 2.4 振幅与灵敏度

| 概念 | 定义 |
|---|---|
| `amplitude_uv` | 信号真实物理振幅（µV） |
| `sensitivity_uv_per_mm` | 显示灵敏度，每毫米代表多少 µV |
| `display_height_mm` | 屏幕/纸面显示高度 |

必须区分“高振幅波形”与“因灵敏度设置看起来很高”。模型若输入截图，必须将灵敏度、时基和滤波作为元数据，否则无法可靠比较振幅或频率。

## 2.5 相、形态与尖锐度

| ID | 术语 | 定义 |
|---|---|---|
| `WAVE.PHASE.MONOPHASIC` | 单相 | 波形主要呈一个相位 |
| `WAVE.PHASE.BIPHASIC` | 双相 | 越过基线形成两个主要相位 |
| `WAVE.PHASE.TRIPHASIC` | 三相 | 三个主要相位；“三相形态”不等于单一病因 |
| `WAVE.PHASE.POLYPHASIC` | 多相 | 多个相位 |
| `WAVE.MORPH.MONOMORPHIC` | 单形性 | 相邻波形频率、持续时间和形态较一致 |
| `WAVE.MORPH.POLYMORPHIC` | 多形性 | 频率、振幅或形态变化明显 |
| `WAVE.SHAPE.SMOOTH` | 平滑 | 无明显尖锐峰 |
| `WAVE.SHAPE.SHARPLY_CONTOURED` | 尖锐轮廓 | 比背景尖锐，但未必满足 spike/sharp 定义 |
| `WAVE.SHAPE.SPIKE` | 棘波形态 | 典型持续约 20–70 ms |
| `WAVE.SHAPE.SHARP` | 尖波形态 | 典型持续约 70–200 ms |
| `WAVE.SHAPE.ARCIFORM` | 弓形/拱形 | mu、wicket 等常见 |
| `WAVE.SHAPE.SINUSOIDAL` | 正弦样 | 节律性波形可见 |
| `WAVE.SHAPE.SAWTOOTH` | 锯齿样 | 需结合年龄、状态和分布 |

“sharply contoured”只能描述形态，不能直接等同于 IED。

## 2.6 节律性、周期性与准节律性

| ID | 术语 | 操作定义 |
|---|---|---|
| `WAVE.RHYTHMIC` | rhythmic | 至少约 6 个连续、形态相近且彼此衔接的波形 |
| `WAVE.PERIODIC` | periodic | 至少约 6 个相似放电，放电间存在可识别间隔 |
| `WAVE.QUASI_RHYTHMIC` | quasi-rhythmic | 周期间隔有中等程度变化，仍保留大致节律性 |
| `WAVE.QUASI_PERIODIC` | quasi-periodic | 周期放电间隔有中等程度变化 |
| `WAVE.NONRHYTHMIC` | non-rhythmic | 不满足节律性或周期性 |

## 2.7 演变、波动与招募

### 明确演变 `definite_evolution`

必须在频率、形态或空间分布中至少存在清晰、连续且有方向的变化，例如：

- 7 Hz → 5 Hz → 3 Hz；
- 低振幅节律活动逐渐变为尖慢波复合；
- 左前颞 → 左中颞 → 左旁矢状 → 对侧；
- 新导联按时间顺序被持续招募。

### 波动 `fluctuation`

活动在振幅、频率或形态上来回变化，但不形成稳定方向或连续阶段。波动不自动等于发作演变。

### 振幅变化

`amplitude_increase` 或 `amplitude_decrease` 是重要描述，但单纯振幅变化通常不足以单独判定明确演变。

建议字段：

```yaml
evolution:
  status: present
  frequency: deceleration
  morphology: waveform_complexification
  spatial: focal_recruitment_then_contralateral_spread
  amplitude: increasing
  certainty: definite
```

---

<a id="sec-3"></a>
# 3. ELEC 与 TECH：电极、派生导联、Montage 与技术参数

## 3.1 Electrode、Channel 与 Derivation

| 术语 | 推荐中文 | 定义 |
|---|---|---|
| Electrode | 电极 | 头皮上的物理传感器，如 F7、T7 |
| Channel | 数据通道 | EDF/设备中的一条数字信号；可能是电极电位、双极差分、ECG、EOG 等 |
| Derivation | 派生导联/导联对 | 两个电极之差，或一个电极相对于某参考的信号 |
| Montage | 导联组合方式 | 电极被连接、相减和排列的规则 |

双极导联必须按“边”理解：

\[
x_{F7-T7}(t)=V_{F7}(t)-V_{T7}(t)
\]

因此 `F7-T7` 不是单一空间点。

## 3.2 10–20 系统常用前缀

| 前缀 | 区域 |
|---|---|
| Fp | 额极 |
| F | 额区 |
| C | 中央区 |
| P | 顶区 |
| O | 枕区 |
| T | 颞区 |
| Fz/Cz/Pz | 中线额/中央/顶 |
| 奇数 | 左侧 |
| 偶数 | 右侧 |
| z | 中线 |

## 3.3 旧名—新名与区域解释

| 旧名 | 规范名 | 推荐解释 |
|---|---|---|
| T3 | T7 | 左中颞区 |
| T4 | T8 | 右中颞区 |
| T5 | P7 | 左后颞区；临床中仍常按后颞解释 |
| T6 | P8 | 右后颞区 |
| F7 | F7 | 左前颞/额颞；不宜机械写成“左额” |
| F8 | F8 | 右前颞/额颞 |
| T1/T2 | 原名保留 | 附加下颞/前颞电极；不得无条件等同 FT9/FT10 |
| A1/A2 | 原名保留 | 耳部参考电极；不得无元数据时自动等同 M1/M2 |

建议保留原始名与规范名：

```yaml
electrode_original: T3
electrode_normalized: T7
anatomic_label: left_mid_temporal
```

## 3.4 Montage 类型

| ID | 名称 | 作用与风险 |
|---|---|---|
| `ELEC.MONTAGE.LONGITUDINAL_BIPOLAR` | 纵向双极 | 常见 double banana；利于观察沿链相位反转和传播 |
| `ELEC.MONTAGE.TRANSVERSE_BIPOLAR` | 横向双极 | 验证左右/冠状方向电场和 end-of-chain |
| `ELEC.MONTAGE.CIRCUMFERENTIAL` | 环周导联 | 对额极、颞、枕极局灶活动有辅助价值 |
| `ELEC.MONTAGE.REFERENTIAL` | 参考导联 | 用相对共同参考的振幅与电场评估最大点 |
| `ELEC.MONTAGE.AVERAGE_REFERENCE` | 平均参考 | 受坏导联、缺失导联和广泛活动影响 |
| `ELEC.MONTAGE.LAPLACIAN` | 拉普拉斯/源导联 | 提高局部性但依赖电极覆盖和插值 |
| `ELEC.MONTAGE.NEONATAL` | 新生儿 montage | 减少电极，突出中央/横向链；不能套用成人空间规则 |

单一 montage 不足以完成可靠定位。推荐至少比较双极与参考 montage。

## 3.5 相位反转

### 定义

在相邻双极导联共享一个电极时，两个导联显示方向相反，提示共享电极相对于相邻电极处于局部电位极值。

示例：

```text
F7–T7：向下
T7–P7：向上
```

可支持：

```yaml
phase_reversal_electrode: T7
```

不能自动支持：

```yaml
clinical_soz: T7
```

### 风险

- 参考极性、导联顺序或软件显示约定未确认；
- end-of-chain；
- 电极故障；
- 容积传导；
- 广泛场或远场；
- montage 中缺少相邻电极。

## 3.6 End-of-chain phenomenon

链首或链尾电极只有一个相邻导联，可能只显示“半个相位反转”。Fp1/Fp2、O1/O2、某些下颞端点尤其需要其他 montage 验证。

输出建议：

```yaml
localization_evidence:
  type: end_of_chain_maximum
  strength: weak
  requires_cross_montage_confirmation: true
```

## 3.7 空间电场 Field

真实脑源事件通常在相邻电极中形成解剖学连续、振幅逐步衰减且极性关系合理的空间电场。

建议检查：

- 是否跨越合理相邻电极；
- 是否在参考 montage 中有中心与衰减；
- 是否在双极 montage 中形成一致相位反转；
- 是否在其他 montage 中保留；
- 是否局限于单一坏电极；
- 是否与肌电、眼动、ECG 或设备同步。

## 3.8 技术参数

| 字段 | 说明 | SOZ 相关风险 |
|---|---|---|
| `sampling_rate_hz` | 采样率 | 过低会削弱快活动和尖波形态 |
| `high_pass_hz` / `low_frequency_filter` | 高通/低频滤波 | 过高可削弱慢波、改变尖锐瞬变 |
| `low_pass_hz` / `high_frequency_filter` | 低通/高频滤波 | 过低可掩盖 LVFA 和快活动 |
| `notch_hz` | 工频陷波 | 可改变接近 50/60 Hz 的信号和尖波边缘 |
| `sensitivity_uv_per_mm` | 灵敏度 | 截图幅度不可脱离该参数解释 |
| `page_duration_s` | 每页时长 | 影响频率与演变的视觉判断 |
| `impedance_kohm` | 电极阻抗 | 过高增加噪声与电极伪迹 |
| `reference` | 参考方式 | 直接影响极性、振幅和空间最大点 |
| `channel_order` | 通道顺序 | 决定双极链结构 |

## 3.9 缺失通道与跨 montage

缺失 Fz/Pz 等通道不一定使整个 SOZ 任务失效，但会降低对中线和旁矢状传播的可评价性。必须：

1. 保存原始可用电极 mask；
2. 只在可见区域内计算阴性证据；
3. 将缺失区域标为 `not_assessable`；
4. 避免用零填充后把零当生理静默；
5. 图模型中对缺失节点使用 mask，不让其参与归一化或 attention；
6. 报告中明确“区域级结论优于单电极结论”。

---

<a id="sec-4"></a>
# 4. CTX 与 BG：正常清醒、嗜睡、睡眠和诱发条件

## 4.1 背景读取顺序

推荐固定顺序：

```text
continuity
→ symmetry
→ anterior-posterior gradient
→ posterior dominant rhythm
→ variability
→ reactivity
→ state changes
→ sleep architecture
→ focal/generalized abnormalities
```

## 4.2 正常清醒背景

| ID | 术语 | 定义/要求 |
|---|---|---|
| `BG.CONTINUITY` | 连续性 | 成人和儿童正常背景应连续；新生儿除外 |
| `BG.SYMMETRY` | 对称性 | 左右频率和振幅大致相似 |
| `BG.AP_GRADIENT` | 前后梯度 | 前部相对快、低振幅，后部相对慢、高振幅 |
| `BG.PDR` | 后部优势节律 | 安静闭眼时枕区优势节律，睁眼衰减 |
| `BG.VARIABILITY` | 变异性 | 波形随时间自然变化 |
| `BG.REACTIVITY` | 反应性 | 对睁闭眼、声音、触碰等刺激产生合理改变 |
| `BG.ORGANIZATION` | 组织性 | 综合连续性、对称性、梯度和正常结构 |
| `CTX.STATE.AWAKE` | 清醒 | 眼睑活动、肌电、PDR、反应等支持 |
| `CTX.STATE.DROWSY` | 嗜睡 | PDR 碎片化/减弱、弥漫 theta、慢滚动眼动 |

成人 PDR 不宜仅用一个固定阈值机械判断，应结合年龄、药物、状态和个体差异。知识库中可保存网站教学范围，但模型训练应使用连续值和年龄条件。

## 4.3 诱发操作

### Photic driving

闪光频率与后部节律出现时间锁定，为正常但非必需反应。缺乏 photic driving 本身不一定异常。

### Photoparoxysmal response

若闪光诱发广泛性癫痫样放电，应与正常 photic driving 区分；需保存刺激频率、放电持续时间、是否在刺激停止后持续、是否有临床相关。

### Hyperventilation

过度换气可引起弥漫性慢化，儿童和青少年更明显。需区别：

- 生理性 HV build-up；
- 诱发的广泛性尖慢波；
- 局灶慢化加重；
- 努力不足导致无明显反应。

字段建议：

```yaml
activation:
  type: hyperventilation
  response: diffuse_slowing
  epileptiform_activation: absent
  effort_quality: adequate
```

## 4.4 睡眠状态

| 状态 | 关键特征 |
|---|---|
| Drowsiness | PDR 碎片化、弥漫 theta、慢滚动眼动 |
| N1 | vertex waves、POSTS、PDR 消失、背景衰减 |
| N2 | sleep spindles、K complexes，N1 波形可持续 |
| N3 | 0.5–2 Hz 高振幅同步 delta 占显著比例 |
| REM | 低振幅混合频率背景、快速眼动、肌张力降低（需 EMG 辅助） |
| Arousal | 从睡眠向较快活动和肌电增强转变 |

## 4.5 睡眠结构词汇

| ID | 术语 | 形态与分布 | 常见误判 |
|---|---|---|---|
| `BG.SLEEP.VERTEX` | 顶点波 | 中央最大、常对称、可高振幅 | 中央区 sharp/IED |
| `BG.SLEEP.POSTS` | 睡眠期正相枕尖瞬变 | 枕区正相、帆形、N1/N2 | 枕区 IED |
| `BG.SLEEP.SPINDLE` | 睡眠纺锤 | 约 11–16 Hz、中央/旁矢状、N2 | 快活动/发作 |
| `BG.SLEEP.K_COMPLEX` | K 复合波 | 高振幅双相慢波，常接 spindle | 广泛性尖慢波 |
| `BG.SLEEP.SLOW_WAVE` | 慢波睡眠活动 | 弥漫同步高振幅 delta | 广泛性慢化 |
| `BG.SLEEP.REM_EYE` | REM 眼动 | 双额对向波形，伴 REM 背景 | 额颞异常活动 |

## 4.6 状态标签来源

私有长程 EEG 没有人工睡眠标签时，应允许：

```yaml
state: unknown
state_source: model
state_confidence: 0.62
```

模型预测状态可用于条件化推理，但不得在未验证时当成临床真值。

---

<a id="sec-5"></a>
# 5. ART：伪迹知识体系

## 5.1 总体原则

伪迹是任何不反映脑活动的信号。它可以是生理性的、设备性的、环境性的或电极性的；可以节律、周期、局灶、高频、慢频，也可与真实发作同时存在。识别伪迹不能只依赖“没有演变”，而要综合空间电场、辅助通道、视频、时间锁定和跨 montage 一致性。

## 5.2 伪迹分类

| ID | 中文 | 典型表现 | 主要 SOZ 风险 |
|---|---|---|---|
| `ART.OCULAR.BLINK` | 眨眼 | 双额高振幅慢波，Fp1/Fp2 最大 | 假性额极/额颞尖慢波 |
| `ART.OCULAR.EYE_OPEN_CLOSE` | 睁闭眼 | 额极大波并伴 PDR 变化 | 与发作起始混淆 |
| `ART.OCULAR.LATERAL` | 水平眼动 | 左右额极/前颞极性相反 | 假性额颞偏侧活动 |
| `ART.OCULAR.ROVING` | 慢滚动眼动 | 嗜睡期双额缓慢对向波 | 假性慢化 |
| `ART.OCULAR.FLUTTER` | 眼睑颤动 | 前额节律性较快活动 | 假性发作节律 |
| `ART.MUSCLE.GENERAL` | 肌电 | 高频、不规则、前颞/额部常明显 | 假性 LVFA 或快活动 |
| `ART.MUSCLE.TEMPORALIS` | 颞肌伪迹 | 颞链高频，咬牙/紧张时增强 | 直接污染颞叶 SOZ 定位 |
| `ART.ORAL.CHEWING` | 咀嚼 | 节律性肌电与运动混合 | 假性颞区节律或传播 |
| `ART.ORAL.GLOSSOKINETIC` | 舌/舌咽运动 | 口腔电偶极造成前/颞慢波 | 假性局灶慢波 |
| `ART.MOVEMENT.BODY` | 体动 | 高振幅、杂乱、常伴多通道 | 假性广泛发作 |
| `ART.MOVEMENT.HEAD_SHAKE` | 摇头 | 可呈后部较慢节律活动 | 假性后部节律/眼动 |
| `ART.SKIN.SWEAT` | 汗液 | 常 <0.5 Hz 极慢漂移，可局灶或弥漫 | 假性 delta 慢化 |
| `ART.CARDIAC.ECG` | 心电 | 与 QRS 严格时锁，常左侧明显 | 假性周期性尖波 |
| `ART.CARDIAC.BALLISTIC` | 心搏机械伪迹 | 与脉搏/QRS 时锁 | 假性周期活动 |
| `ART.RESP.BREATHING` | 呼吸 | 与呼吸周期同步 | 假性周期性慢波 |
| `ART.RESP.VENTILATOR` | 呼吸机 | 与设备周期同步 | 假性周期放电 |
| `ART.ELECTRODE.POP` | 电极爆裂 | 单电极陡升后缓降，无合理 field | 假性 spike/phase reversal |
| `ART.ELECTRODE.LOOSE` | 电极松动 | 间歇漂移、工频、pop | 假性局灶异常 |
| `ART.ELECTRODE.SALT_BRIDGE` | 盐桥 | 两电极电位近似相同，双极导联变平 | 假性静默或缺失传播 |
| `ART.ELECTRODE.BAD_CONTACT` | 接触不良 | 高阻、噪声、工频 | 假性高频活动 |
| `ART.LINE.NOISE` | 50/60 Hz 工频 | 固定单调高频 | 假性 beta/LVFA |
| `ART.DEVICE.STIMULATION` | 植入设备刺激 | 固定形态、设备时锁 | 假性局灶快活动 |
| `ART.DEVICE.RNS` | RNS 刺激伪迹 | 与刺激脉冲有关 | 误判为发作活动 |

## 5.3 伪迹鉴别检查表

```text
1. 是否有合理、连续的脑源空间电场？
2. 是否局限于一个电极或一条含该电极的所有导联？
3. 是否与 EOG、ECG、EMG、呼吸、视频或设备事件时间锁定？
4. 是否随睁眼、咀嚼、说话、触碰、翻身而改变？
5. 是否在参考、纵向双极、横向或环周 montage 中保持合理空间关系？
6. 是否覆盖并遮挡了真实脑电，而非替代脑电？
7. 是否有频率、形态和空间招募的明确演变？
8. 是否存在背景打断和发作后改变？
```

## 5.4 多标签原则

允许真实发作与伪迹并存：

```yaml
event_family:
  - ICTAL.ELECTROGRAPHIC_SEIZURE
  - ART.MUSCLE.TEMPORALIS
artifact_obscuration: moderate
onset_assessability: partial
```

不得强制二选一。

---

<a id="sec-6"></a>
# 6. VAR：正常变异与良性 Mimics

## 6.1 关键概念表

| ID | 术语 | 典型特征 | 主要鉴别对象 | SOZ 规则 |
|---|---|---|---|---|
| `VAR.MU` | μ 节律 | 约 7–11 Hz 弓形、中央/旁矢状、运动或运动想象时抑制 | 中央 sharp、wicket | 不作为 SOZ 证据；对你的 MI 数据是重要生理特征 |
| `VAR.WICKET` | 门形波 | 颞区、嗜睡、弓形 alpha、短暂非演变簇发、无后随慢波、不打断背景 | 颞区 IED | 不作为颞叶 SOZ 证据 |
| `VAR.RMTD` | 嗜睡期中颞节律性 theta | 中颞区、theta、数秒、无演变，可单侧或双侧 | 颞叶发作、BIRDs、LRDA | 无演变时不标 ictal |
| `VAR.LAMBDA` | Lambda 波 | 清醒视觉扫描、双枕正相帆形、闭眼消失 | POSTS、枕区 IED | 结合清醒和视觉扫描 |
| `VAR.BETS_SSS` | BETS/小尖波 | 嗜睡/浅睡、低振幅、短暂、常颞区、无后随慢波 | 颞区 spike | 不作为 IED，除非有更强异常证据 |
| `VAR.POSITIVE_14_6` | 14/6 Hz 正相尖波 | 青少年/年轻人、嗜睡、后部、1–2 s 正相簇发 | 快活动、IED | 正常变异 |
| `VAR.PHANTOM_6HZ` | 6 Hz 幻影尖慢波 | 很低振幅 5–7 Hz 小尖慢波；WHAM/FOLD 亚型 | 广泛性尖慢波 | 通常不作为 SOZ 证据 |
| `VAR.POSTS` | POSTS | N1/N2 枕区正相帆形 | 枕区 IED、lambda | 睡眠结构 |
| `VAR.VERTEX` | 顶点波 | N1 中央最大、可高振幅 | 中央 IED | 睡眠结构 |
| `VAR.POSTERIOR_SLOW_YOUTH` | 青少年后部慢波 | PDR 中嵌入后部 theta/delta | 后部慢化 | 年龄依赖正常现象 |
| `VAR.SLOW_ALPHA` | 慢/半频 alpha 变异 | 与 PDR 谐波或次谐波关系 | 后部慢化 | 需与基础 PDR 对照 |
| `VAR.HYPNAGOGIC_HYPERSYNCHRONY` | 入睡期高同步 | 儿童高振幅弥漫节律慢活动 | 广泛性发作 | 状态依赖、无 ictal 演变 |
| `VAR.HYPNOPOMPIC_HYPERSYNCHRONY` | 醒转期高同步 | 儿童醒转时高振幅节律活动 | 发作 | 结合醒转状态 |
| `VAR.BREACH_ENHANCED_NORMAL` | 颅骨缺损增强的正常节律 | mu/wicket 等因 skull breach 变尖、增幅 | IED | 必须结合术后史与 field |

## 6.2 Mu 与你的五指运动想象任务

Mu 是感觉运动皮层的静息节律，真实或想象运动可引起对侧 mu event-related desynchronization（ERD）。在 MI 数据中：

- mu 变化是目标生理信号，不应被“尖锐外观”误删；
- ICA/伪迹去除需避免删除中央区真实 mu/beta 成分；
- VR 与非 VR 比较可分析 mu/beta ERD 强度、潜伏期、侧化和空间稳定性；
- 但 LearningEEG 的临床截图知识不能替代连续原始 MI 信号分析。

## 6.3 Wicket、RMTD 与颞叶发作鉴别

| 特征 | Wicket | RMTD | 颞叶发作 |
|---|---|---|---|
| 状态 | 多见嗜睡 | 嗜睡 | 任意状态 |
| 频率 | 常 alpha | theta | 可 delta/theta/alpha/fast |
| 形态 | 弓形、对称上下坡 | 尖锐轮廓节律 theta | 可尖锐、节律、LVFA 等 |
| 后随慢波 | 无 | 通常无 | 可有或无 |
| 背景打断 | 无 | 通常无 | 常有 |
| 明确演变 | 无 | 无 | 典型有频率/形态/空间演变 |
| 招募/传播 | 无 | 无 | 可见 |
| 发作后改变 | 无 | 无 | 可见慢化/衰减 |


---

<a id="sec-7"></a>
# 7. 年龄特异模块：新生儿与儿童

## 7.1 年龄 profile 必须显式保存

```yaml
age_profile: adult        # neonatal | pediatric | adult
chronological_age_days: null
gestational_age_weeks: null
postmenstrual_age_weeks: null
```

新生儿 EEG 的正常规则与成人显著不同；儿童背景、PDR、振幅和睡眠结构也随年龄快速变化。不得用成人阈值直接判断新生儿或幼儿异常。

## 7.2 新生儿基本术语

| ID | 术语 | 定义/意义 |
|---|---|---|
| `CTX.NEO.PMA` | postmenstrual age, PMA | 胎龄 + 出生后年龄；新生儿 EEG 判读的核心时间轴 |
| `CTX.NEO.AWAKE` | 新生儿清醒 | 眼睛睁开、行为状态和 EEG 综合判断 |
| `CTX.NEO.ACTIVE_SLEEP` | 活跃睡眠 | 类似成人 REM，闭眼、眼动、连续背景 |
| `CTX.NEO.QUIET_SLEEP` | 安静睡眠 | 更易出现生理性不连续模式 |
| `BG.NEO.SYNCHRONY` | 同步性 | 两侧爆发是否同时发生；不同于左右形态/幅度对称 |
| `BG.NEO.SYMMETRY` | 对称性 | 左右频率与振幅是否相似 |
| `BG.NEO.CONTINUITY` | 连续性 | 随 PMA 成熟，生理性不连续逐渐减少 |
| `BG.NEO.REACTIVITY` | 反应性 | 对刺激是否有变化，随成熟增强 |
| `BG.NEO.DELTA_BRUSH` | delta brush | delta 上叠加约 8–20 Hz 快活动，特定 PMA 可正常 |
| `BG.NEO.TRACE_DISCONTINU` | tracé discontinu | 较早 PMA 安静睡眠的低振幅间歇不连续模式 |
| `BG.NEO.TRACE_ALTERNANS` | tracé alternans | 较成熟新生儿安静睡眠中较高振幅的交替模式 |
| `BG.NEO.ENCOCHES_FRONTALES` | 额部尖瞬变 | 特定成熟阶段的双侧同步正常瞬变 |
| `BG.NEO.TEMPORAL_SAWTOOTH` | 颞区锯齿样 theta | 年龄依赖性正常新生儿波形 |
| `BG.NEO.MULTIFOCAL_SHARP_TRANSIENTS` | 多灶尖瞬变 | 新生儿早期可正常，但持续局灶、过多或超龄需警惕 |
| `ICTAL.NEO.ELECTROGRAPHIC_ONLY` | 新生儿纯电图发作 | 可无可见临床表现，需独立记录 EEG 事件 |

## 7.3 新生儿推理规则

1. 生理性不连续、弥漫慢活动和一定程度不同步在早产儿可正常；成人中则可能严重异常。
2. 同一个波形在 30、34、38、42 周 PMA 的意义不同。
3. 多灶尖瞬变可在成熟前出现，但若持续固定于一个区域、频繁、演变或伴背景异常，应考虑病理性。
4. 新生儿发作常表现为节律性、演变性活动，但临床表现可能缺失或很细微。
5. 新生儿 montage 电极较少，空间定位分辨率低；不得把单一新生儿导联最大点直接当 SOZ。
6. 新生儿发作分类、背景评分和预后判断应使用独立 profile。

## 7.4 儿童 PDR 发育参考

LearningEEG 教学中给出的常用里程碑可作为检索知识，而非不可变硬阈值：

```text
约 6 个月：4–5 Hz
约 1 岁：6 Hz
约 2 岁：7 Hz
约 3 岁：8 Hz
约 8 岁：9 Hz
约 10 岁：10 Hz
```

模型应保存年龄连续变量，并允许个体差异、药物、嗜睡和疾病影响。

## 7.5 儿童正常特征

- 背景振幅通常高于成人；
- 嗜睡可有显著高振幅 delta；
- vertex、K complex、spindle 可非常高振幅；
- 早期 spindle 可不同步，随年龄逐渐同步；
- posterior slow waves of youth 可单侧或双侧；
- hypnagogic/hypnopompic hypersynchrony 可模拟广泛性发作；
- 睡眠可显著激活 IED。

## 7.6 儿童癫痫综合征术语

| 主词 | 旧称/别名 | 典型 EEG 知识 | 与 SOZ 项目的关系 |
|---|---|---|---|
| `CLIN.SYN.EIDEE` | Ohtahara syndrome | 早期发育性癫痫性脑病、burst-suppression、多灶放电 | 不宜用成人局灶 SOZ 规则 |
| `CLIN.SYN.IESS` | West syndrome | infantile spasms、hypsarrhythmia | 痉挛与电极衰减/高幅紊乱需专门模型 |
| `CLIN.SYN.LGS` | Lennox–Gastaut syndrome | 慢背景、慢尖慢波、阵发性快活动 | 多灶/网络性，不宜强制单灶 SOZ |
| `CLIN.SYN.SELECTS` | BECTS/benign rolandic epilepsy | 中央颞区尖波、睡眠激活 | IED 区不必等同发作起始区 |
| `CLIN.SYN.ABSENCE` | childhood/juvenile absence | 广泛性尖慢波、快速起止 | 不应输出局灶 SOZ |
| `CLIN.SYN.JME` | juvenile myoclonic epilepsy | 广泛性 polyspike-wave、肌阵挛 | 不应因幅度偏侧误判局灶 |
| `CLIN.SYN.ESES` | CSWS 等历史相关术语 | 睡眠显著激活的尖慢波负荷 | 需要睡眠负荷而非单次事件定位 |

综合征名称应作为患者级临床实体，不应仅凭一张 EEG 图自动诊断。

---

<a id="sec-8"></a>
# 8. ABN：非癫痫样异常

## 8.1 广泛性慢化

| 程度 | 典型特征 | 解释 |
|---|---|---|
| Mild | PDR 减慢、AP 梯度欠佳、theta 增多，但反应性/变异性/状态转换保留 | 轻度弥漫性脑功能障碍 |
| Moderate | PDR 碎片化或缺失，theta–delta 优势，alpha 显著减少 | 中度弥漫性脑功能障碍 |
| Severe | 高度紊乱、delta 优势、可不连续，缺乏反应性和正常结构 | 严重弥漫性脑功能障碍 |

建议字段：

```yaml
generalized_slowing:
  status: present
  severity: moderate
  dominant_frequency: theta_delta
  pdr: absent
  reactivity: reduced
```

广泛性慢化通常不提供单一 SOZ 定位。

## 8.2 局灶性慢化

| ID | 术语 | 特征 | 临床意义 | SOZ 权重 |
|---|---|---|---|---|
| `ABN.FOCAL_POLYMORPHIC_SLOWING` | 局灶性多形性慢化 | theta/delta，形态变化，局限区域 | 非特异性区域功能障碍/结构病变 | 弱支持，不能单独定 SOZ |
| `ABN.FOCAL_RHYTHMIC_SLOWING` | 局灶性节律性慢化 | 单形性、节律性 | 可与皮层高兴奋性相关，需区分 LRDA/TIRDA | 中低权重 |
| `ABN.FOCAL_CONTINUOUS_SLOWING` | 持续性局灶慢化 | 全程或大部分持续 | 更强结构病变提示 | 仍非 SOZ 证据 |
| `ABN.FOCAL_INTERMITTENT_SLOWING` | 间歇性局灶慢化 | 状态依赖或间歇出现 | 可见于较小病变、药物或状态变化 | 弱支持 |
| `ABN.POSTICTAL_SLOWING` | 发作后慢化 | 发作终止后局灶或弥漫慢化 | 可辅助侧化/定位 | 若与发作起始一致则有价值 |

必须区分：

```text
局灶慢化 = 区域功能异常证据
局灶慢化 ≠ IED
局灶慢化 ≠ 发作起始
局灶慢化 ≠ SOZ
```

## 8.3 衰减、抑制与不连续

| ID | 术语 | 定义 |
|---|---|---|
| `ABN.ATTENUATION.REGIONAL` | 区域性衰减 | 某侧/某区持续或间歇低振幅，可见于硬膜下积液、病变、术后等 |
| `ABN.ATTENUATION.ICTAL` | 发作性衰减 | 某些 tonic seizure 或 electrodecrement 中突然衰减 |
| `ABN.DISCONTINUITY` | 不连续 | 较高振幅活动间夹显著低振幅时期；成人通常异常 |
| `ABN.SUPPRESSION` | 抑制 | 极低振幅背景 |
| `ABN.BURST_SUPPRESSION` | 爆发-抑制 | 爆发与抑制交替，可由严重脑损伤或深镇静产生 |

## 8.4 Breach activity/rhythm

颅骨缺损使下方脑电在头皮上振幅更高、轮廓更尖锐，常与局灶慢化共存。规则：

- breach 本身不是 IED；
- IED 可以嵌在 breach 区域中；
- 相同源在 breach 区可能看起来更局灶、更高振幅；
- 手术史、颅骨缺损位置和跨 montage 证据必须进入模型；
- SOZ 模型需避免把“高振幅尖锐 + 术后区域”自动判为起始。

## 8.5 Excess beta

弥漫或额部过多 beta 常与 benzodiazepine、barbiturate、镇静、焦虑或嗜睡相关。局灶、演变、低电压快活动需与肌电、breach、药物 beta 鉴别。

## 8.6 三相形态

推荐术语：

```text
GPDs with triphasic morphology
或
generalized periodic discharges with triphasic morphology
```

不要把“triphasic waves”当作唯一病因标签；三相形态可见于多种代谢、毒性、感染或其他脑病环境。

---

<a id="sec-9"></a>
# 9. IED：发作间期癫痫样放电

## 9.1 定义边界

IED 提示皮层高兴奋性和癫痫网络，但：

- IED 存在不等于患者一定有癫痫；
- IED 最大区主要定义 irritative zone；
- irritative zone 可与 SOZ 重叠，也可更广或不同；
- 单一 IED 不能证明临床 SOZ 或 EZ。

## 9.2 IFCN 风格六项形态/上下文特征

建议为每个候选放电分别标注：

1. `pointed_peak`：尖锐或棘样峰；
2. `slope_asymmetry`：上升/下降斜率不对称；
3. `aftergoing_slow_wave`：后随慢波；
4. `background_disruption`：打断背景；
5. `duration_outlier`：持续时间与背景不同；
6. `coherent_field`：解剖学一致的空间电场。

```yaml
ifcn_features:
  pointed_peak: present
  slope_asymmetry: present
  aftergoing_slow_wave: present
  background_disruption: present
  duration_outlier: present
  coherent_field: present
  feature_count: 6
```

特征计数可用于模型辅助，但不是脱离临床背景的绝对诊断器。

## 9.3 IED 基本术语

| ID | 英文 | 中文 | 操作定义 |
|---|---|---|---|
| `IED.SPIKE` | spike | 棘波 | 典型持续约 20–70 ms |
| `IED.SHARP` | sharp wave | 尖波 | 典型持续约 70–200 ms |
| `IED.SPIKE_WAVE` | spike-and-wave | 棘慢波复合 | 棘波后随慢波 |
| `IED.SHARP_SLOW` | sharp-and-slow-wave | 尖慢波复合 | 尖波后随慢波 |
| `IED.POLYSPIKE` | polyspike | 多棘波 | 两个或多个快速棘波连续出现 |
| `IED.POLYSPIKE_WAVE` | polyspike-and-wave | 多棘慢波 | 多棘波后随慢波 |
| `IED.PFA` | paroxysmal fast activity | 阵发性快活动 | 突发快活动，需结合综合征、状态和分布 |
| `IED.RUN` | run of IEDs | IED 簇发 | 连续 IED，但可无 ictal evolution |

## 9.4 空间分类

| ID | 含义 |
|---|---|
| `IED.FOCAL` | 单一区域局灶 IED |
| `IED.REGIONAL` | 相邻多个电极构成区域性 IED |
| `IED.MULTIFOCAL` | 两个或以上相互独立局灶 |
| `IED.HEMISPHERIC` | 半球范围但非广泛性 |
| `IED.BILATERAL_SYNCHRONOUS` | 双侧近同步，可有偏侧 |
| `IED.GENERALIZED` | 双侧广泛近同步放电 |
| `IED.INDEPENDENT_BILATERAL` | 双侧独立放电 |

## 9.5 IED 空间字段

```yaml
ied_spatial:
  laterality: left
  region: anterior_temporal
  maximal_electrode: F7
  phase_reversal_electrode: F7
  field: left_temporal_to_frontocentral
  montage_consistency: present
  state_activation: N2
```

必须同时保存 `maximal_electrode`、`phase_reversal_electrode` 与 `region`，不能用一个字段混代。

## 9.6 IED 与 mimics

重点鉴别：

- wicket waves；
- BETS/SSS；
- vertex waves；
- POSTS/lambda；
- electrode pop；
- ECG artifact；
- breach rhythm；
- sharply contoured focal slowing；
- muscle artifact；
- normal neonatal sharp transients。

## 9.7 IED 对 SOZ 的合理报告

推荐：

> 左前颞区可见频繁尖慢波，支持该区域存在发作间期皮层高兴奋性，并为左颞致痫网络提供辅助证据。

禁止：

> 左前颞尖波证明 SOZ 位于 F7。

---

<a id="sec-10"></a>
# 10. RPP 与 IIC：节律性、周期性和高风险模式

## 10.1 总体框架

Rhythmic and periodic patterns（RPP）可从非特异性脑功能障碍延伸到明确发作。其意义依赖：

- 分布：generalized、lateralized、bilateral independent、multifocal；
- 频率；
- 持续时间；
- prevalence/burden；
- morphology；
- plus modifiers；
- fluctuation/evolution；
- clinical correlate；
- state/stimulus dependence。

Ictal–interictal continuum（IIC）是介于明确发作间期与明确发作之间的描述框架，不是独立病因诊断。

## 10.2 主术语

| ID | 主词 | 中文 | 旧词/关联词 | SOZ 意义 |
|---|---|---|---|---|
| `RPP.GRDA` | generalized rhythmic delta activity | 广泛性节律性 delta | FIRDA/OIRDA 为分布性历史术语 | 多为非特异；一般不定位单一 SOZ |
| `RPP.GPD` | generalized periodic discharges | 广泛性周期性放电 | triphasic morphology 可作修饰 | 高风险取决于频率/plus/evolution |
| `RPP.LRDA` | lateralized rhythmic delta activity | 偏侧性节律性 delta | TIRDA 与其相关但非所有场景同义 | 可提示同侧皮层高兴奋性和发作风险 |
| `RPP.LPD` | lateralized periodic discharges | 偏侧性周期性放电 | PLED/PLEDs | 较高发作关联，常伴急性局灶病变 |
| `RPP.BIPD` | bilateral independent periodic discharges | 双侧独立周期性放电 | BIPLED | 两个独立侧活动，风险高 |
| `RPP.MULTIFOCAL_PD` | multifocal periodic discharges | 多灶周期性放电 | — | 多区域病理活动 |
| `RPP.BIRDS` | brief potentially ictal rhythmic discharges | 短暂潜在发作性节律放电 | — | 最接近 ictal 端的短事件 |
| `RPP.SIRPIDS` | stimulus-induced RPPs or seizures | 刺激诱发 RPP/发作 | — | 描述诱发关系，不是单一形态 |
| `RPP.IIC` | ictal-interictal continuum | 发作—发作间期连续体 | — | 表示不确定区间，需要上下文 |

## 10.3 Plus modifiers

周期性放电常用：

- `+F`：superimposed fast activity；
- `+R`：superimposed rhythmic activity；
- `+FR`：二者同时。

节律性活动常用：

- `+S`：叠加尖锐成分；
- `+F`：叠加快活动；
- `+FS`：二者同时。

plus 特征通常增加发作相关风险，但不自动把模式变成发作。

## 10.4 GRDA

- 广泛性节律 delta，常前额优势；
- 常与弥漫性脑病、代谢/感染/重症状态有关；
- 明确演变时需考虑发作；
- 不应因前额优势而输出额叶 SOZ；
- OIRDA 在儿童和特定癫痫背景中可有不同意义，需保留上下文。

## 10.5 LRDA 与 TIRDA

LRDA 是偏侧节律 delta，可提示同侧皮层高兴奋性。TIRDA 是颞区间歇性节律 delta 的临床历史术语，常与颞叶癫痫相关，但不能无条件与所有 temporal LRDA 等同。

建议：

```yaml
pattern: RPP.LRDA
region: left_temporal
legacy_alias: TIRDA
ictal_evolution: absent
soz_relevance: supportive_not_definitive
```

## 10.6 LPD/BIPD

LPD 常与急性局灶病变和较高发作风险相关。它可能：

- 处于 interictal 端；
- 处于 IIC；
- 演变成 electrographic seizure；
- 与临床症状时间锁定而成为 electroclinical event。

LPD 区域可帮助侧化病理网络，但仍不自动等同临床 SOZ。

## 10.7 BIRDs

操作性知识：

- 局灶或广泛性、外观接近 ictal 的节律活动；
- 常 >4 Hz；
- 至少约 6 个波；
- 持续约 0.5 s 至 <10 s；
- 不属于已知正常变异；
- 若明确演变，或与该患者的 IED/发作形态高度相似，则可信度更高。

必须与以下鉴别：

- RMTD；
- wicket；
- 短肌电簇发；
- 工频；
- non-evolving IED runs；
- 短暂设备刺激伪迹。

## 10.8 Electrographic seizure 的操作标准

常用 ACNS 风格规则包括：

```text
A. 癫痫样放电平均频率 >2.5 Hz，持续 ≥10 s；
或
B. 任意具有明确频率、形态或空间演变的模式，持续 ≥10 s。
```

若存在明确时间锁定的临床表现，较短 EEG 模式也可构成 electroclinical seizure。具体实现应以当前采用的标准版本为准，并在数据集说明中锁定规则。

## 10.9 IIC 标签建议

```yaml
iic_status: possible_iic
pattern: RPP.LRDA
frequency_hz: 2.0
plus_modifier: +S
evolution: absent
clinical_correlate: unknown
seizure_risk_level: high
```

允许值：

```text
below_iic
possible_iic
definite_iic
electrographic_seizure
electroclinical_seizure
indeterminate
```


---

<a id="sec-11"></a>
# 11. ICTAL：发作期时空知识体系

## 11.1 发作不是一个静态标签，而是一个状态机

```text
preictal
  ↓
earliest_scalp_visible_change
  ↓
electrographic_onset
  ↓
early_evolving_core
  ↓
regional_recruitment
  ↓
propagation
  ↓
contralateral_spread / bilateralization / generalization
  ↓
offset
  ↓
postictal_state
```

对每个阶段分别保存时间、空间、形态和置信度，避免把晚期高振幅传播区误当起始区。

## 11.2 时间事件定义

| ID | 术语 | 操作定义 |
|---|---|---|
| `ICTAL.PREICTAL` | 发作前期 | 发作前基线或潜在改变，不能默认存在 |
| `ICTAL.EARLIEST_SCALP_CHANGE` | 最早头皮可见改变 | 读者/算法在头皮信号中最早可靠识别的变化 |
| `ICTAL.EEG_ONSET` | 电图起始 | 满足预定义发作模式判据的起始时间 |
| `ICTAL.CLINICAL_ONSET` | 临床起始 | 最早可靠临床症状/体征时间 |
| `ICTAL.EARLY_PHASE` | 发作早期 | onset 后用于定位、尚未广泛传播的窗口 |
| `ICTAL.RECRUITMENT` | 招募 | 新电极/区域进入持续发作模式 |
| `ICTAL.PROPAGATION` | 传播 | 发作活动扩展到其他区域 |
| `ICTAL.BILATERALIZATION` | 双侧化 | 从偏侧/局灶扩展到双侧 |
| `ICTAL.OFFSET` | 电图终止 | 发作性模式停止 |
| `ICTAL.POSTICTAL` | 发作后期 | 终止后慢化、衰减、抑制或恢复阶段 |

必须允许：

```yaml
electrographic_onset_time: 128.4
clinical_onset_time: 130.1
electroclinical_delay_s: 1.7
```

临床起始可以早于、晚于或无法与头皮 EEG 对齐。

## 11.3 常见发作起始形态受控词

| ID | 主词 | 中文 | 鉴别重点 |
|---|---|---|---|
| `ICTAL.ONSET.RHYTHMIC_DELTA` | rhythmic delta onset | 节律性 delta 起始 | 与 LRDA、运动/呼吸伪迹鉴别 |
| `ICTAL.ONSET.RHYTHMIC_THETA` | rhythmic theta onset | 节律性 theta 起始 | 与 RMTD、wicket 鉴别 |
| `ICTAL.ONSET.RHYTHMIC_ALPHA` | rhythmic alpha onset | 节律性 alpha 起始 | 与 PDR、mu、wicket、肌电鉴别 |
| `ICTAL.ONSET.RHYTHMIC_SPIKES` | rhythmic spike onset | 节律性棘波起始 | 与 IED run、LPD 鉴别 |
| `ICTAL.ONSET.SPIKE_WAVE` | spike-wave onset | 尖/棘慢波起始 | 判断局灶/广泛及演变 |
| `ICTAL.ONSET.POLYSPIKE_WAVE` | polyspike-wave onset | 多棘慢波起始 | 常见于广泛性发作，也可局灶 |
| `ICTAL.ONSET.LVFA` | low-voltage fast activity | 低电压快活动 | 与颞肌、工频、药物 beta、breach 鉴别 |
| `ICTAL.ONSET.PFA` | paroxysmal fast activity | 阵发性快活动 | 结合睡眠、综合征及分布 |
| `ICTAL.ONSET.ELECTRODECREMENT` | electrodecrement | 电极衰减/背景突然衰减 | 可伴叠加快活动；与接触故障鉴别 |
| `ICTAL.ONSET.ATTENUATION_FAST` | attenuation with superimposed fast | 衰减伴快活动 | tonic seizure 等可见 |
| `ICTAL.ONSET.HERALD_SPIKE` | herald spike | 先导尖波 | 单一先导波后需有真正演变模式 |
| `ICTAL.ONSET.PERIODIC` | periodic discharge onset | 周期放电样起始 | 与 LPD/GPD/IIC 鉴别 |
| `ICTAL.ONSET.POORLY_FORMED` | poorly formed rhythmic onset | 低清晰度节律起始 | 需降低定位置信度 |
| `ICTAL.ONSET.INDETERMINATE` | indeterminate onset | 起始不明确 | 伪迹、截断、快速扩散或深部源 |

## 11.4 LVFA 的特别规则

LVFA 是 SOZ 研究中常见但高风险的术语。只有在以下条件中多数满足时，才应提高可信度：

- 在固定区域先于其他区域出现；
- 与背景清晰不同；
- 在多个相邻电极/导联中形成合理 field；
- 随后发生形态、频率或空间演变；
- 不与颞肌 EMG、咬牙、工频或设备刺激严格同步；
- 在不同 montage 中仍可见；
- 多次发作重复出现。

若起始阶段存在颞肌伪迹，应报告：

> 起始阶段可见左颞区低振幅快活动，但同期颞肌肌电干扰降低了对其脑源性的确定度。

不得直接报告：

> 左颞 LVFA 证明左颞 SOZ。

## 11.5 演变字段

```yaml
evolution:
  overall_status: present
  certainty: definite
  frequency:
    status: present
    direction: deceleration
    start_hz: 7.0
    end_hz: 3.0
  morphology:
    status: present
    sequence:
      - rhythmic_sharply_contoured_theta
      - rhythmic_spike_wave
      - polymorphic_delta
  spatial:
    status: present
    sequence:
      - left_anterior_temporal
      - left_mid_temporal
      - left_parasagittal
      - right_temporal
  amplitude:
    trend: increasing
```

## 11.6 空间角色标签

对每个电极、双极导联和区域分配角色，而不是只给 SOZ/non-SOZ：

```text
earliest_visible
onset_core
early_recruited
intermediate_recruited
late_recruited
propagated
bilateralized
postictal_only
uninvolved
artifact_obscured
unknown
not_assessable
```

示例：

```yaml
spatial_roles:
  - item: F7-T7
    level: derivation
    role: earliest_visible
    latency_s: 0.0
  - item: T7-P7
    level: derivation
    role: onset_core
    latency_s: 0.2
  - item: left_parasagittal
    level: region
    role: early_recruited
    latency_s: 2.3
  - item: right_temporal
    level: region
    role: propagated
    latency_s: 5.8
```

## 11.7 招募与传播图

将每次发作表示为有向时空图：

```text
节点：电极 / 双极导联 / 解剖区域
节点属性：进入发作模式的时间、形态、幅度、置信度、质量 mask
边：从较早节点到较晚节点的可能传播
边属性：时间差、频谱相似度、相位/方向指标、模型置信度
```

示例：

```yaml
propagation_edges:
  - source: left_anterior_temporal
    target: left_mid_temporal
    delay_s: 0.4
    confidence: 0.91
  - source: left_mid_temporal
    target: left_parasagittal
    delay_s: 1.9
    confidence: 0.77
  - source: left_temporal
    target: right_temporal
    delay_s: 5.6
    confidence: 0.64
```

传播方向是时序推断，不等同于有效连接或因果连接；必须标记为 `estimated_propagation`。

## 11.8 发作终止与发作后模式

| ID | 术语 | SOZ 价值 |
|---|---|---|
| `ICTAL.OFFSET.ABRUPT` | 突然终止 | 描述终止模式，不直接定位 |
| `ICTAL.OFFSET.GRADUAL` | 逐渐终止 | 可伴频率减慢/振幅变化 |
| `ICTAL.POSTICTAL.FOCAL_SLOWING` | 局灶发作后慢化 | 与起始侧一致时可辅助侧化 |
| `ICTAL.POSTICTAL.ATTENUATION` | 发作后衰减 | 可提示受累区域但非特异 |
| `ICTAL.POSTICTAL.SUPPRESSION` | 发作后抑制 | 广泛或局灶；需结合发作类型 |
| `ICTAL.POSTICTAL.PLED_LPD` | 发作后 LPD | 提示高兴奋区域，但非直接 SOZ |
| `ICTAL.POSTICTAL.RECOVERY` | 背景恢复 | 保存恢复时间和状态 |

## 11.9 头皮发作的空间分类

```text
focal
regional
hemispheric
multifocal
bilateral_independent
generalized_from_start
focal_to_bilateral
unknown_focal_vs_generalized
indeterminate_due_to_artifact
scalp_negative_or_no_clear_correlate
```

## 11.10 Electrographic 与 Electroclinical

| 术语 | 定义 |
|---|---|
| Electrographic seizure | EEG 满足发作判据，但无明确时间锁定临床表现或临床信息缺失 |
| Electroclinical seizure | EEG 模式与临床表现存在明确时间锁定关系 |
| Clinical seizure without clear scalp correlate | 有临床发作，但头皮 EEG 无明确相关，常见于深部/局限源或伪迹遮挡 |
| Subclinical seizure | 建议避免含糊使用，改用 electrographic seizure 并说明临床相关状态 |

## 11.11 发作临床表现 Semiology

建议按时间顺序保存，而不是只保存一个发作类型标签：

```text
behavioral_arrest
responsiveness_change
consciousness_change
aura
sensory_symptom
autonomic_symptom
emotional_symptom
cognitive_symptom
automatism
head_version
eye_deviation
tonic_posturing
clonic_movement
myoclonic_jerk
negative_myoclonus
hyperkinetic_behavior
speech_arrest
aphasia
bilateral_tonic_clonic
postictal_confusion
postictal_weakness
```

字段：

```yaml
semiology:
  - feature: behavioral_arrest
    onset_s: 130.1
    laterality: none
    confidence: 0.83
  - feature: right_head_version
    onset_s: 134.8
    laterality: right
    confidence: 0.91
```

临床表现可支持侧化/定位，但具体意义依赖症状类型、传播阶段和患者个体网络。

## 11.12 发作分类术语

患者级临床分类可使用：

```text
focal preserved consciousness seizure
focal impaired consciousness seizure
focal to bilateral tonic-clonic seizure
generalized tonic-clonic seizure
absence seizure
myoclonic seizure
tonic seizure
clonic seizure
atonic seizure
epileptic spasm
negative myoclonic seizure
unknown whether focal or generalized
unclassified
```

单纯 EEG 事件应优先使用 `electrographic seizure`、起始分布和形态描述，不应仅凭 EEG 自动断言意识状态。

## 11.13 Status 与 burden

长程 EEG 中建议另外保存：

```yaml
seizure_count: 12
total_seizure_duration_s: 840
recording_duration_s: 7200
seizure_burden_fraction: 0.1167
longest_seizure_s: 190
cluster_status: present
status_epilepticus_status: possible
```

Status epilepticus 的临床定义和不同发作类型时间阈值需按采用的指南版本执行，不应由本知识库单一规则自动决定。

---

<a id="sec-12"></a>
# 12. LOC 与 CLIN：从导联到头皮起始区，再到 SOZ/EZ

## 12.1 三层空间表示

### 电极节点层

```text
F7, T7, P7, F8, T8, P8, Cz, ...
```

### 派生导联边层

```text
F7-T7, T7-P7, F8-T8, T8-P8, ...
```

### 解剖区域层

```text
left_anterior_temporal
left_mid_temporal
left_posterior_temporal
left_temporal
right_temporal
left_frontal
left_posterior_quadrant
...
```

模型预测与标签应明确层级：

```yaml
target_value: F7-T7
target_level: derivation
```

不得在不经过空间解码时转换为：

```yaml
target_level: clinical_soz
```

## 12.2 临床术前评估区域

| 区域 | 推荐中文 | 定义 |
|---|---|---|
| Symptomatogenic zone | 症状产生区 | 激活后产生最早临床症状的皮层区域 |
| Irritative zone | 刺激区/发作间期放电区 | 产生 IED 的皮层区域 |
| Seizure onset zone, SOZ | 发作起始区 | 最早产生临床发作性活动的区域 |
| Epileptogenic lesion | 致痫性病变 | 与癫痫发生存在因果联系的结构病变 |
| Epileptogenic zone, EZ | 致痫区 | 为获得持续无发作所需切除/离断的最小皮层区域 |
| Eloquent cortex | 功能重要皮层 | 运动、语言、感觉等关键功能皮层 |

这些区域可重叠但不等同。EZ 无法由单一术前检查直接测得，通常是综合推断并由治疗结局间接验证。

## 12.3 项目级六层输出

| 层级 | 字段 | 允许声明 |
|---:|---|---|
| 1 | `earliest_scalp_visible_derivation` | 最早可见的双极导联/数据通道 |
| 2 | `earliest_scalp_visible_electrode_set` | 由多导联、相位反转和参考 montage 推断的电极集合 |
| 3 | `scalp_electrographic_onset_region` | 头皮 EEG 推测的电图起始区域，简称 SEOR |
| 4 | `noninvasive_soz_hypothesis` | 综合多次发作、IED、影像和临床形成的非侵入性 SOZ 假设 |
| 5 | `clinical_or_invasive_soz` | 临床团队综合或 SEEG 确认的 SOZ |
| 6 | `epileptogenic_zone` | 手术相关 EZ 推断及术后验证 |

建议将第 3 层命名为：

```text
SEOR = scalp electrographic onset region
```

避免头皮模型直接宣称 clinical SOZ。

## 12.4 Earliest derivation 到 electrode set 的解码

若 `F7-T7` 与 `T7-P7` 同时早期异常且在 T7 形成相位反转，可推断：

```yaml
earliest_derivations: [F7-T7, T7-P7]
phase_reversal_electrode: T7
earliest_electrode_set: [F7, T7]
scalp_region: left_anterior_mid_temporal
```

若只有 `F7-T7`，不能判断异常主要来自 F7 还是 T7；输出应保持不确定：

```yaml
earliest_electrode_set: [F7, T7]
within_derivation_localization: unresolved
```

## 12.5 头皮 EEG 定位限制

### 深部源

海马、岛叶、眶额等深部或切向源可能：

- 无明显头皮起始；
- 首先表现为非特异衰减；
- 在传播到新皮层后才出现节律活动；
- 形成假性对侧或双侧起始。

### 快速双侧同步

局灶源快速传播可在头皮上看似广泛性同步。需结合：

- 毫秒级或亚秒级偏侧；
- 重复发作的一致偏侧；
- 临床症状；
- IED 分布；
- 影像；
- MEG/高密度 EEG/SEEG。

### 颅骨缺损与术后改变

Breach 可放大某侧活动，造成看似更早或更高振幅。时序比振幅更重要，但时序也可能受信噪比影响。

### 缺失通道

缺失关键前颞、下颞或中线电极会降低空间分辨率；模型不能把不可见区域视为阴性。

### 伪迹遮挡

颞肌、运动和电极伪迹常在临床发作开始时最严重，可能让最早脑源活动不可评价。

## 12.6 SOZ 证据分级

以下为本项目的操作性分级，不替代临床指南。

### A 级：核心 ictal 定位证据

- 多次习惯性发作在同一区域出现最早头皮改变；
- 起始活动具有清晰的频率、形态或空间演变；
- 早期 field 在多个 montage 中一致；
- 最早区域在时序上稳定先于其他区域；
- 临床表现与该区域在解剖上相容；
- 多发作 laterality/region 高度一致。

### B 级：较强辅助证据

- 局灶 BIRDs 与发作区一致；
- 重复局灶 LVFA 且排除肌电/设备；
- 局灶发作后慢化或衰减与起始侧一致；
- 相位反转、参考 montage 最大点和区域预测一致；
- 招募/传播顺序在多次发作中稳定。

### C 级：支持性但非决定性证据

- 局灶 IED；
- 睡眠激活局灶 IED；
- TIRDA/temporal LRDA；
- 局灶多形性慢化；
- 同侧结构病变；
- 同侧 PET/SPECT/MEG 异常。

### D 级：限制、反证或降权因素

- 起始阶段严重肌电/运动/电极伪迹；
- 关键通道缺失；
- 片段从发作中段开始；
- 起始即双侧/广泛且无稳定偏侧；
- 不同发作之间定位明显不一致；
- breach effect；
- 单一 end-of-chain 现象；
- 深部源/头皮无明确相关；
- 单次事件或置信度低。

## 12.7 区域证据评分框架

可定义：

\[
\begin{aligned}
S(r)=&\;w_1E_{\text{reproducible ictal onset}}
+w_2E_{\text{early latency}}
+w_3E_{\text{evolution}}\\
&+w_4E_{\text{field/montage}}
+w_5E_{\text{postictal}}
+w_6E_{\text{interictal}}
+w_7E_{\text{clinical concordance}}\\
&-p_1P_{\text{artifact}}
-p_2P_{\text{missing channels}}
-p_3P_{\text{rapid spread}}
-p_4P_{\text{discordance}}
-p_5P_{\text{deep-source uncertainty}}
\end{aligned}
\]

权重应在训练集上学习，并在独立患者级验证集上校准。LLM 不得自由指定权重。

## 12.8 多发作聚合

患者级 SOZ 推断不能只取“最高概率单次发作”。建议保存：

```yaml
patient_aggregation:
  number_of_seizures: 5
  analyzable_seizures: 4
  excluded_seizures:
    - seizure_id: SZ02
      reason: severe_artifact
  region_votes:
    left_temporal: 3
    right_temporal: 0
    bilateral_temporal: 1
  weighted_region_probability:
    left_temporal: 0.78
    bilateral_temporal: 0.15
    other: 0.07
  consistency_score: 0.80
```

可采用：

- 质量加权平均；
- 贝叶斯层级聚合；
- set/attention pooling；
- 图级汇聚；
- mixture model 识别多个发作亚型；
- 不一致性显式建模。

## 12.9 一致性不是绝对真值

多次发作一致可增强证据，但也可能共同受到：

- 同一 montage 偏差；
- 同一缺失通道；
- 同一颅骨缺损；
- 同一伪迹来源；
- 传播后可见而深部起始不可见。

因此一致性是权重，不是自动验证。

---

<a id="sec-13"></a>
# 13. 强制禁止推理规则

| 禁止跳跃 | 原因 | 允许的替代表述 |
|---|---|---|
| `F7-T7 最早 → F7 是 SOZ` | 双极导联是两个电极之差 | “最早异常涉及 F7–T7，无法仅凭该导联区分 F7 与 T7” |
| `T7 相位反转 → T7 下方皮层是源` | 相位反转只提示头皮局部极值 | “T7 为头皮电位最大候选，需跨 montage 和其他证据验证” |
| `左颞 IED → 左颞 SOZ` | IED 主要定义 irritative zone | “左颞 IED 为左颞高兴奋网络的支持证据” |
| `左颞慢化 → 左颞 SOZ` | 慢化非特异 | “左颞慢化提示区域功能障碍/结构异常” |
| `最早头皮可见 → 生物学最早起始` | 深部/小范围活动可晚传到头皮 | “最早头皮可见改变” |
| `无头皮相关 → 无发作` | 深部或局限发作可 scalp-negative | “未见明确头皮相关，不能排除临床发作” |
| `双侧同步 → 一定广泛性癫痫` | 快速双侧同步可来自局灶源 | “外观广泛同步，局灶快速双侧传播仍需鉴别” |
| `高振幅区域 → 起始区域` | 振幅受传播、参考、breach 影响 | 优先比较时序、field 和演变 |
| `LLM 报告与模型一致 → 模型正确` | 可能共享同一输入偏差 | 必须回到信号逐项 grounding |
| `未标注 → 阴性` | 未标注可为 unknown | 使用四值标签 |
| `伪迹概率低 → 一定脑源` | 低伪迹概率非充分条件 | 仍需 field、演变和跨 montage |
| `单次发作 → 患者级 SOZ` | 需多发作和临床综合 | 输出单发作 SEOR 与置信度 |
| `模型置信度高 → 临床置信度高` | 神经网络可能失校准 | 使用独立校准和外部验证 |
| `DeepSOZ bipolar label → clinical SOZ` | 数据集标签层级可能是导联级 | 保留 `target_level` 与数据集定义 |
| `发作传播到某区 → 该区也是 onset core` | 传播区可高振幅但更晚 | 使用空间角色标签 |
| `POSTS/vertex/BETS 出现 → IED` | 正常睡眠/正常变异 | 结合状态、形态、field 和背景打断 |
| `TIRDA/LRDA → 已发生发作` | 可为高风险 interictal/IIC 模式 | 描述为支持性证据，检查演变与临床相关 |
| `LPD → status epilepticus` | LPD 可在 IIC 不同位置 | 按频率、plus、演变、持续及临床判断 |
| `截图相似 → 原始信号同类` | 截图受滤波、时基、标注影响 | 图像仅作原型，需原始信号验证 |


---

<a id="sec-14"></a>
# 14. 规范术语、旧词与别名映射

## 14.1 电极与 montage 术语

| 输入词/旧词 | 规范主词 | 处理规则 |
|---|---|---|
| T3/T4 | T7/T8 | 归一化保存，同时保留原始字段 |
| T5/T6 | P7/P8 | 归一化保存；临床区域仍标为 posterior temporal |
| double banana | longitudinal bipolar montage | `double_banana` 作为常用 alias |
| common average | average reference | 保存参与平均的电极和 mask |
| source montage | Laplacian/source derivation | 保存实现方法，避免与真实皮层 source 混淆 |
| channel | electrode 或 derivation | 必须根据数据格式明确层级，禁止混用 |
| phase reversal site | phase-reversal electrode | 不是 clinical source 或 SOZ |

## 14.2 RPP/IIC 旧词映射

| 旧词 | 规范主词 | 备注 |
|---|---|---|
| PLED/PLEDs | LPD/LPDs | 旧词仅作为 alias |
| BIPLED/BIPLEDs | BIPD/BIPDs | 双侧独立周期性放电 |
| triphasic waves | GPDs with triphasic morphology | 三相形态作为修饰，不作为单一病因 |
| FIRDA | frontally predominant GRDA | 保留历史 alias 和前额优势属性 |
| OIRDA | occipitally predominant GRDA / OIRDA | 儿童及特定场景保留独立临床信息 |
| TIRDA | temporal intermittent rhythmic delta activity | 与 temporal LRDA 建立 `related_to`，不强制完全等同 |
| subclinical seizure | electrographic seizure | 另行说明 clinical correlate 是否 absent/unknown |

## 14.3 正常变异与综合征映射

| 旧词/别名 | 规范主词 |
|---|---|
| BETS | small sharp spikes, SSS；知识库主词 `VAR.BETS_SSS` |
| BECTS / benign rolandic epilepsy | SeLECTS |
| West syndrome | IESS |
| Ohtahara syndrome | EIDEE |
| grand mal | generalized tonic-clonic；若明确局灶起始则 focal to bilateral tonic-clonic |
| secondarily generalized | focal to bilateral tonic-clonic |
| focal aware seizure | focal preserved consciousness seizure（与采用的 ILAE 版本一致） |
| focal impaired awareness seizure | focal impaired consciousness seizure（与采用的 ILAE 版本一致） |

## 14.4 SOZ 相关禁用简写

以下词必须展开或附限定词：

| 不推荐裸用 | 推荐写法 |
|---|---|
| SOZ channel | dataset-defined SOZ-related derivation / earliest scalp-visible derivation / clinical SOZ electrode hypothesis |
| onset channel | earliest scalp-visible derivation 或 earliest involved electrode set |
| source | scalp voltage maximum / estimated cortical source / clinical SOZ，三者必须区分 |
| localization | electrode-level / derivation-level / scalp-region-level / clinical-region-level |
| ground truth | label source + target level + validation method |
| normal | normal background / no target event / clinically normal record，明确粒度 |
| seizure-free | 指定 Engel/ILAE 结局、随访时长和是否停药 |

## 14.5 中英文核心词表

| 缩写 | 英文 | 中文 |
|---|---|---|
| EEG | electroencephalography | 脑电图 |
| PDR | posterior dominant rhythm | 后部优势节律 |
| AP gradient | anterior–posterior gradient | 前后梯度 |
| IED | interictal epileptiform discharge | 发作间期癫痫样放电 |
| RPP | rhythmic and periodic pattern | 节律性和周期性模式 |
| IIC | ictal–interictal continuum | 发作—发作间期连续体 |
| GRDA | generalized rhythmic delta activity | 广泛性节律性 delta 活动 |
| LRDA | lateralized rhythmic delta activity | 偏侧性节律性 delta 活动 |
| GPD | generalized periodic discharge | 广泛性周期性放电 |
| LPD | lateralized periodic discharge | 偏侧性周期性放电 |
| BIPD | bilateral independent periodic discharge | 双侧独立周期性放电 |
| BIRDs | brief potentially ictal rhythmic discharges | 短暂潜在发作性节律放电 |
| SIRPIDs | stimulus-induced RPPs or seizures | 刺激诱发 RPP 或发作 |
| LVFA | low-voltage fast activity | 低电压快活动 |
| PFA | paroxysmal fast activity | 阵发性快活动 |
| SOZ | seizure onset zone | 发作起始区 |
| EZ | epileptogenic zone | 致痫区 |
| SEOR | scalp electrographic onset region | 头皮电图起始区（本项目操作术语） |
| IZ | irritative zone | 刺激区/发作间期放电区 |
| PMA | postmenstrual age | 经后年龄 |
| ERD/ERS | event-related desynchronization/synchronization | 事件相关去同步/同步 |

---

<a id="sec-15"></a>
# 15. 机器可读标注协议

## 15.1 完整 YAML 示例

```yaml
ontology_version: "leeg-scalp-soz-1.0.0"

recording:
  patient_id: "P001"
  session_id: "S01"
  recording_id: "R001"
  file_id: "P001_S01_R001.edf"
  age_years: 35
  sex: "unknown"
  age_profile: "adult"             # neonatal | pediatric | adult
  postmenstrual_age_weeks: null
  recording_duration_s: 7200

technical:
  sampling_rate_hz: 256
  electrode_system: "10-20"
  montage_family: "longitudinal_bipolar"
  montage_name: "double_banana"
  reference_original: "REF"
  high_pass_hz: 0.5
  low_pass_hz: 70
  notch_hz: 50
  page_duration_s: 10
  sensitivity_uv_per_mm: 7
  electrode_name_standard: "10-10_normalized"
  channel_order_verified: true

channel_map:
  electrodes_available:
    - Fp1
    - Fp2
    - F7
    - F8
    - T7
    - T8
    - P7
    - P8
    - O1
    - O2
    - F3
    - F4
    - C3
    - C4
    - P3
    - P4
    - Cz
  missing_electrodes:
    - Fz
    - Pz
  derivations_available:
    - Fp1-F7
    - F7-T7
    - T7-P7
    - P7-O1
    - Fp2-F8
    - F8-T8
    - T8-P8
    - P8-O2
  legacy_aliases:
    T3: T7
    T4: T8
    T5: P7
    T6: P8

context:
  state:
    value: "N2"
    status: present
    source: model
    confidence: 0.86
  medications: []
  activation:
    photic: false
    hyperventilation: false
  clinical_event_marker_available: false
  video_available: false

quality:
  overall_quality: "moderate"
  overall_quality_score: 0.74
  bad_electrodes: []
  artifact_burden_fraction: 0.12
  artifact_types:
    - id: "ART.MUSCLE.TEMPORALIS"
      laterality: left
      burden: mild
  onset_assessability: partial
  limitations:
    - "Fz/Pz missing"
    - "mild left temporal myogenic artifact"

background:
  continuity:
    value: continuous
    status: present
  symmetry:
    value: mild_asymmetry
    status: present
  ap_gradient:
    value: not_assessable
    status: not_assessable
  pdr:
    status: unknown
    frequency_hz: null
  variability:
    value: present
    status: present
  reactivity:
    value: unknown
    status: unknown
  generalized_slowing:
    status: absent
  focal_slowing:
    status: present
    laterality: left
    region: temporal
    morphology: polymorphic
    persistence: intermittent

interictal_events:
  - event_id: "IED001"
    interval_s: [84.2, 84.5]
    label: "IED.SHARP_SLOW"
    status: present
    source: human
    confidence: 0.91
    spatial:
      laterality: left
      region: anterior_temporal
      maximal_electrode: F7
      phase_reversal_electrode: F7
      coherent_field: present
    ifcn_features:
      pointed_peak: present
      slope_asymmetry: present
      aftergoing_slow_wave: present
      background_disruption: present
      duration_outlier: present
      coherent_field: present
      feature_count: 6
    state_activation: N2
    soz_role: supportive_only

seizures:
  - seizure_id: "SZ03"
    source_segment:
      start_s: 110.0
      end_s: 170.0
      contains_true_onset: true
      oracle_onset_segment: true

    times:
      earliest_scalp_change_s: 128.4
      electrographic_onset_s: 128.6
      clinical_onset_s: null
      offset_s: 152.1
      onset_time_uncertainty_s: 0.4

    classification:
      eeg_event_class: "electrographic_seizure"
      spatial_class: "focal_to_bilateral"
      clinical_seizure_class: "unknown"

    onset_pattern:
      label: "ICTAL.ONSET.RHYTHMIC_THETA"
      initial_frequency_hz: [5.0, 7.0]
      morphology: "rhythmic_sharply_contoured"
      initial_amplitude: low
      background_disruption: present

    evolution:
      overall_status: present
      certainty: definite
      frequency:
        status: present
        direction: deceleration
      morphology:
        status: present
        sequence:
          - rhythmic_sharply_contoured_theta
          - rhythmic_spike_wave
      spatial:
        status: present
        sequence:
          - left_anterior_mid_temporal
          - left_parasagittal
          - right_temporal
      amplitude:
        trend: increasing

    spatial:
      earliest_derivations:
        - name: F7-T7
          latency_s: 0.0
          probability: 0.81
        - name: T7-P7
          latency_s: 0.2
          probability: 0.76
      phase_reversal_electrodes:
        - electrode: T7
          confidence: 0.74
      earliest_electrode_set:
        - F7
        - T7
      laterality: left
      scalp_onset_region: left_anterior_mid_temporal
      scalp_onset_region_probability: 0.78
      field_coherence: present
      montage_consistency: present

    spatial_roles:
      - item: F7-T7
        level: derivation
        role: earliest_visible
        latency_s: 0.0
      - item: T7-P7
        level: derivation
        role: onset_core
        latency_s: 0.2
      - item: left_parasagittal
        level: region
        role: early_recruited
        latency_s: 2.3
      - item: right_temporal
        level: region
        role: propagated
        latency_s: 5.8

    propagation_edges:
      - source: left_anterior_temporal
        target: left_mid_temporal
        delay_s: 0.4
        confidence: 0.91
      - source: left_mid_temporal
        target: left_parasagittal
        delay_s: 1.9
        confidence: 0.77
      - source: left_temporal
        target: right_temporal
        delay_s: 5.6
        confidence: 0.64

    artifacts:
      - type: ART.MUSCLE.TEMPORALIS
        laterality: left
        severity: mild
        overlaps_onset: true
    onset_assessability: partial

    postictal:
      focal_slowing:
        status: present
        laterality: left
        region: temporal
      attenuation: absent

    interpretation:
      pattern_label: focal_temporal_evolving_ictal_pattern
      seor: left_anterior_mid_temporal
      seor_confidence: 0.78
      clinical_soz_claim_allowed: false
      evidence_grade: A_B_mixed

patient_level:
  number_of_recordings: 3
  number_of_analyzed_seizures: 5
  number_of_analyzable_seizures: 4
  seizure_consistency_score: 0.80
  weighted_region_probability:
    left_temporal: 0.78
    bilateral_temporal: 0.15
    other: 0.07
  noninvasive_soz_hypothesis:
    value: left_temporal
    confidence: 0.75
    source: multimodal_model
  clinical_or_invasive_soz:
    status: unknown
    value: null
  epileptogenic_zone:
    status: unknown
    value: null

provenance:
  waveform_annotation_source: human
  dataset_soz_label_source: private_clinical_annotation
  dataset_soz_target_level: derivation
  report_source: model_generated
  report_human_verified: false
  last_reviewed: "2026-08-28"
```

## 15.2 最小事件对象

```json
{
  "event_id": "E001",
  "start_s": 128.4,
  "end_s": 132.8,
  "event_family": ["ICTAL.ELECTROGRAPHIC_SEIZURE"],
  "morphology": "rhythmic_sharply_contoured_theta",
  "laterality": "left",
  "region": "anterior_mid_temporal",
  "earliest_derivations": ["F7-T7", "T7-P7"],
  "evolution": "present",
  "artifact_obscuration": "mild",
  "confidence": 0.78,
  "source": "human"
}
```

## 15.3 标签粒度字段

每个目标必须指定：

```text
target_level:
  sample
  window
  channel
  electrode
  derivation
  event
  seizure
  recording
  session
  patient
  region
  clinical_zone
```

例如 TUSZ 通道时间标注、TUEV 事件、DeepSOZ 导联标签和私有临床区域标签不能放入同一个无层级的 `label` 列。

## 15.4 标签置信度与可评价性

```yaml
label_quality:
  confidence: 0.82
  annotator_count: 2
  inter_annotator_agreement: 0.76
  adjudicated: true
  assessability: partial
  uncertainty_reasons:
    - rapid_bilateral_spread
    - temporal_muscle_artifact
```

---

<a id="sec-16"></a>
# 16. RAG 外挂知识库卡片规范

## 16.1 单卡片结构

每个核心概念建议单独切成一张卡片：

```yaml
concept_id: "ICTAL.FOCAL_TEMPORAL_EVOLVING_RHYTHM"
preferred_en: "Focal temporal evolving ictal rhythm"
preferred_zh: "局灶性颞区演变性发作节律"
aliases:
  - "temporal ictal rhythm"
  - "temporal electrographic onset pattern"
namespace: "ICTAL"
semantic_layer: "pattern"
parent_ids:
  - "ICTAL.FOCAL_SEIZURE_PATTERN"

short_definition: >
  首先在颞区最明显，并在频率、形态或空间分布上发生明确演变的发作性 EEG 模式。

required_features:
  - temporal_predominance
  - definite_evolution
  - cerebral_field

supporting_features:
  - background_disruption
  - early_regional_recruitment
  - postictal_temporal_slowing
  - cross_seizure_reproducibility

exclusion_features:
  - pure_muscle_time_lock
  - single_electrode_pop
  - non_evolving_RMTD
  - isolated_wicket_run

mimics:
  - VAR.RMTD
  - VAR.WICKET
  - ART.MUSCLE.TEMPORALIS
  - RPP.LRDA

soz_relevance:
  strength: strong_if_reproducible
  allowed_inference:
    - supports_scalp_temporal_onset
    - supports_noninvasive_temporal_soz_hypothesis
  forbidden_inference:
    - proves_clinical_soz
    - proves_epileptogenic_zone

report_phrases:
  preferred:
    - "最早头皮可见的发作性改变位于左颞区。"
    - "该活动随后在频率、形态及空间分布上发生演变。"
  prohibited:
    - "该导联即为临床 SOZ。"

source:
  family: "LearningEEG_plus_official_standards"
  urls:
    - "https://www.learningeeg.com/seizures"
    - "https://www.learningeeg.com/montages"
  reviewed_date: "2026-08-28"
```

## 16.2 卡片必须包含的字段

```text
主词与 concept_id
中英文名称
alias 与旧词
所属命名空间和语义层
简短定义
必要特征
支持特征
排除特征
mimics
年龄/状态条件
空间分布
SOZ 证据强度
允许推理
禁止推理
报告推荐句
报告禁用句
来源 URL 与版本日期
```

## 16.3 推荐元数据

```yaml
rag_metadata:
  domain: eeg
  task:
    - seizure_detection
    - soz_localization
    - report_generation
  age_profile:
    - adult
  state:
    - awake
    - sleep
  evidence_level: B
  lexical_terms:
    - left temporal
    - rhythmic theta
    - evolution
  language:
    - zh-CN
    - en
```

## 16.4 切块建议

- 每个概念卡独立成块；
- 推荐约 500–1200 中文字/块；
- 保留 `concept_id`、主词、别名和父节点；
- 规则表与报告模板单独切块；
- 不要把整章切成一个巨块；
- 同一概念的定义、mimics、allowed/forbidden inference 尽量不拆散；
- 检索结果应同时返回“目标概念卡”和至少一个“鉴别诊断卡”。

## 16.5 检索策略

输入结构化证据：

```yaml
laterality: left
region: temporal
frequency_hz: 5-7
morphology: rhythmic_sharply_contoured
frequency_evolution: deceleration
spatial_evolution: present
artifact: mild_temporal_muscle
```

构造多路检索：

```text
语义检索：left temporal evolving rhythmic theta seizure
规则检索：temporal + evolution + field + artifact differential
鉴别检索：RMTD, wicket, LRDA, temporal muscle artifact
报告检索：scalp onset region + uncertainty wording
```

## 16.6 知识库不能篡改患者证据

RAG 只能提供：

- 术语定义；
- 模式判据；
- 鉴别诊断；
- 允许/禁止推理；
- 报告措辞；
- 证据权重建议。

RAG 不能：

- 为患者补写未观察到的波形；
- 把知识库典型分布强加给患者；
- 把模型概率改写成确定事实；
- 由“常见左颞模式”推断当前患者必然左颞；
- 将网站图例当作患者金标准。

---

<a id="sec-17"></a>
# 17. 证据约束的 EEG/SOZ 报告体系

## 17.1 推荐报告结构

```text
1. 技术条件与数据质量
2. 记录状态和背景活动
3. 发作间期异常
4. 发作事件与最早头皮可见改变
5. 起始形态、演变、招募与传播
6. 发作后改变
7. 单发作头皮定位结论（SEOR）
8. 多发作综合非侵入性 SOZ 假设
9. 一致证据、矛盾证据和限制
10. 临床边界声明
```

## 17.2 报告生成的输入协议

LLM 不应直接读取未约束的模型 logits 后自由写报告。应输入：

```yaml
observations:
  onset_time_s: 128.4
  earliest_derivations:
    - F7-T7
    - T7-P7
  phase_reversal_electrode: T7
  onset_frequency_hz: 5-7
  onset_morphology: rhythmic_sharply_contoured_theta
  evolution:
    frequency: deceleration
    morphology: present
    spatial: left_temporal_to_left_parasagittal
  postictal: left_temporal_slowing
  artifact: mild_left_temporal_muscle
  missing_electrodes:
    - Fz
    - Pz

model_probabilities:
  derivation:
    F7-T7: 0.81
    T7-P7: 0.76
  region:
    left_temporal: 0.78
    bilateral_temporal: 0.15
  calibration_status: calibrated

knowledge_constraints:
  must_distinguish_seor_from_clinical_soz: true
  must_report_artifact_limitation: true
  prohibited_claims:
    - "F7-T7 is the clinical SOZ"
```

## 17.3 结构化中间输出

```json
{
  "observed_evidence": {
    "earliest_derivations": ["F7-T7", "T7-P7"],
    "onset_pattern": "5-7 Hz rhythmic sharply contoured theta",
    "evolution": ["frequency slowing", "amplitude increase", "spatial recruitment"]
  },
  "interpretation": {
    "pattern": "focal temporal evolving ictal pattern",
    "scalp_onset_region": "left anterior-mid temporal"
  },
  "supporting_evidence": [
    "early left temporal field",
    "definite evolution",
    "left temporal postictal slowing"
  ],
  "contradictory_or_limiting_evidence": [
    "mild temporal muscle artifact",
    "Fz/Pz unavailable"
  ],
  "clinical_inference": {
    "noninvasive_soz_hypothesis": "left temporal",
    "confidence": "moderate",
    "clinical_soz_confirmed": false
  }
}
```

## 17.4 推荐句式

### 观察层

> 发作起始阶段，F7–T7 与 T7–P7 导联首先出现 5–7 Hz 节律性尖锐 theta 活动。

### 模式层

> 该活动随后表现出频率减慢、振幅增高及空间分布扩展，符合演变性发作模式。

### 头皮定位层

> 最早头皮可见的电图改变在左前—中颞区最明显。

### 多发作综合层

> 多次可评价发作显示一致的左颞区早期受累，支持左颞区作为非侵入性 SOZ 候选。

### 不确定性层

> 起始阶段受到轻度颞肌肌电干扰，且 Fz/Pz 缺失，因此单一电极层面的定位可信度低于区域层面的定位。

### 临床边界层

> 上述结论代表头皮 EEG 推测的电图起始区域，不能单独等同于侵入性确认的临床 SOZ 或致痫区。

## 17.5 置信度语言

| 定量范围示例 | 推荐文字 | 注意 |
|---:|---|---|
| ≥0.85 且已校准、证据一致 | strongly supports / 强支持 | 仍不等于“证明” |
| 0.70–0.85 | supports / 支持 | 报告主要支持证据和限制 |
| 0.50–0.70 | suggests / 提示、倾向 | 明确备选区域 |
| <0.50 或失校准 | indeterminate / 不确定 | 不强行给单一区域 |

数值阈值应按验证集校准，不是固定临床标准。

## 17.6 禁止句式

```text
“F7-T7 就是 SOZ。”
“左颞尖波证明左颞 SOZ。”
“相位反转点就是发作源。”
“无头皮 EEG 改变说明没有发作。”
“报告与模型一致，因此定位正确。”
“患者致痫区已由头皮 EEG 确定。”
“模型置信度 95%，所以临床确定性为 95%。”
“未见 Fz 受累。”（当 Fz 缺失时）
```

## 17.7 报告事实检查

每个自然语言主张必须映射回结构化证据：

| 报告主张 | 必需证据 |
|---|---|
| “最早出现” | 通道/区域 onset latency |
| “左颞区” | 导联解码 + field + region prediction |
| “频率减慢” | 时频/频率轨迹 |
| “振幅增加” | 经技术参数校正的幅度轨迹 |
| “向右传播” | 招募时间和传播图 |
| “相位反转于 T7” | 双极导联极性关系 + montage 验证 |
| “受肌电影响” | artifact head、EMG/视频或形态证据 |
| “多次发作一致” | 患者级发作聚合统计 |
| “SOZ 候选” | ictal 核心证据 + 多模态/多发作支持 |

---

<a id="sec-18"></a>
# 18. 报告—定位模型反馈闭环

## 18.1 错误闭环

```text
定位模型预测左颞
→ LLM 根据该预测生成左颞报告
→ 报告解析仍得到左颞
→ 认为两者一致所以正确
```

这是自我确认，不产生外部新证据。

## 18.2 正确闭环

```text
原始 EEG
→ 信号模型输出结构化观察证据
→ 知识检索提供规则和鉴别
→ 受约束报告生成
→ 报告解析为独立语义主张
→ 每项主张回到原始信号/辅助通道验证
→ 仅通过 grounding gate 的语义成为低权重软监督
```

## 18.3 报告解析字段

```text
report_channel_claim
report_electrode_claim
report_region_claim
report_laterality_claim
report_onset_time_claim
report_morphology_claim
report_evolution_claim
report_recruitment_claim
report_propagation_claim
report_artifact_claim
report_uncertainty_claim
report_clinical_boundary_claim
```

## 18.4 Grounding gate

一个语义主张只有在满足以下条件时才可反馈：

1. 有对应时间区间；
2. 有对应通道/电极/区域；
3. 有可复核的信号特征；
4. 未被缺失通道直接否定可评价性；
5. 与伪迹识别结果不冲突，或已明确降权；
6. 与多次发作聚合一致，或明确标注不一致；
7. 不是由报告模板自动补全的套话；
8. 置信度经过校准。

## 18.5 一致性损失

可使用：

\[
L_{cycle}=D_{KL}(p_{signal}(r)\parallel p_{report}(r))
\]

但必须同时有证据 grounding：

\[
L_{total}=L_{supervised}+\lambda_cL_{cycle}+\lambda_gL_{grounding}+\lambda_uL_{uncertainty}
\]

其中 `λc` 应较低，防止语言模型反向固化信号模型错误。

## 18.6 反事实检查

报告反馈前可进行：

- 移除最早导联后，区域结论是否合理下降；
- 打乱传播时序后，“最早/传播”主张是否不再成立；
- 隐藏 artifact 元数据后，报告是否过度确定；
- 替换左右标签后，模型是否只跟随文字而忽略信号；
- 使用不同 montage，区域结论是否稳定；
- 输入正常变异样本，是否错误生成 IED/SOZ 报告。

---

<a id="sec-19"></a>
# 19. 与 LaBraM 变体和端到端系统的结合

## 19.1 推荐模型输出头

```text
1. Signal quality / bad-channel head
2. Artifact type and burden head
3. State/background head
4. IED morphology and spatial-field head
5. RPP/IIC head
6. Seizure onset-time head
7. Earliest derivation role head
8. Electrode-set decoder
9. Region and laterality head
10. Evolution head
11. Recruitment/propagation graph head
12. Postictal pattern head
13. Multi-seizure patient-level aggregation head
14. Evidence-grounded report head
15. Uncertainty/calibration head
```

## 19.2 多任务损失示意

\[
\begin{aligned}
L =\;&\lambda_1L_{quality}
+\lambda_2L_{artifact}
+\lambda_3L_{state}
+\lambda_4L_{IED}\\
&+\lambda_5L_{seizure\ detection}
+\lambda_6L_{onset\ time}
+\lambda_7L_{derivation\ role}\\
&+\lambda_8L_{electrode\ set}
+\lambda_9L_{region}
+\lambda_{10}L_{laterality}\\
&+\lambda_{11}L_{evolution}
+\lambda_{12}L_{propagation}
+\lambda_{13}L_{patient\ aggregation}\\
&+\lambda_{14}L_{report\ grounding}
+\lambda_{15}L_{calibration}
\end{aligned}
\]

## 19.3 DeepSOZ-style Top-1 标签的处理

你现有的 `soz_bipolar` 建议保存为：

```yaml
label_value: F7-T7
label_family: dataset_defined_soz_related_label
target_level: derivation
source_dataset: DeepSOZ_or_private
clinical_meaning: "dataset-specific bipolar SOZ target"
```

模型可继续优化 Top-1，但报告生成前必须经过：

```text
bipolar derivation
→ electrode-set ambiguity
→ phase reversal/field validation
→ regional mapping
→ multi-seizure aggregation
→ clinical boundary check
```

## 19.4 Oracle onset 与真实长程流程的差异

Oracle onset 上的高准确度不能直接代表端到端性能。完整系统需分别评估：

```text
长程发作检测召回
onset 时间误差
onset 片段包含率
单发作 SEOR 定位
多发作患者级 SOZ 假设
报告事实性和证据 grounding
```

错误传播路径：

```text
漏检发作
→ onset 片段错误
→ 定位偏移
→ 报告错误
```

应报告 oracle 与 predicted onset 两套结果。

## 19.5 公共数据的角色分工

```text
TUEG
→ 大规模无标签/弱标签临床 EEG 表征预训练

TUAR
→ 伪迹和质量控制

TUEV
→ spike/sharp、周期放电、眼动、伪迹、背景事件识别

TUSL
→ seizure / slowing / complex background 区分

TUSZ
→ 发作检测、时间定位、通道招募、传播弱监督

TUEP
→ 患者级 epilepsy/no-epilepsy 和临床 metadata

DeepSOZ + 私有数据
→ SOZ 相关导联/区域的核心监督和临床验证
```

TUSZ 的最早通道标注可作为弱起始证据，但不能自动当 clinical SOZ 金标准。

---

<a id="sec-20"></a>
# 20. 评估体系

## 20.1 发作检测与时间定位

- event sensitivity/recall；
- false alarms per hour；
- onset latency error；
- offset error；
- event-based F1；
- duration-weighted overlap；
- seizure burden error。

## 20.2 导联与区域定位

- Top-1 / Top-k derivation accuracy；
- electrode-set precision/recall；
- region accuracy；
- laterality accuracy；
- hierarchical distance error；
- mean rank / MRR；
- region-level balanced accuracy；
- calibration：ECE、Brier score、NLL；
- abstention/selective risk。

## 20.3 多发作患者级定位

- patient-level region accuracy；
- seizure consistency score；
- probability aggregation calibration；
- per-patient macro metrics；
- discordant-seizure detection；
- bootstrap patient-level confidence interval；
- 与临床综合结论、SEEG、切除区和结局的一致性。

## 20.4 传播评估

- recruitment time MAE；
- node role F1；
- edge precision/recall；
- temporal order correlation；
- graph edit distance；
- early-vs-late channel classification。

## 20.5 报告评估

通用 BLEU/ROUGE 不足以评价临床报告。建议：

- clinical entity precision/recall；
- laterality correctness；
- region correctness；
- onset time grounding；
- morphology correctness；
- evolution correctness；
- propagation correctness；
- artifact/limitation coverage；
- contradiction rate；
- hallucination rate；
- unsupported-claim rate；
- certainty calibration；
- clinical boundary compliance；
- 医师 Likert 评分；
- 医师间一致性；
- 错误严重度分级。

## 20.6 闭环是否真正有效

报告反馈模块必须与以下基线比较：

```text
A. 纯信号定位模型
B. 信号模型 + 普通多任务辅助头
C. 信号模型 + 文本蒸馏但无报告反馈
D. 信号模型 + 报告循环一致性
E. 信号模型 + grounding gate 的证据反馈
```

只有 E 在独立患者级测试集上稳定提高定位、校准或鲁棒性，才能说明闭环带来真实增益，而非自我蒸馏。

---

<a id="sec-21"></a>
# 21. 建议的知识库目录结构

```text
eeg_knowledge_system/
├── README.md
├── ontology/
│   ├── terms.yaml
│   ├── relations.yaml
│   ├── region_hierarchy.yaml
│   ├── event_hierarchy.yaml
│   └── clinical_zone_hierarchy.yaml
├── terminology/
│   ├── aliases_en_zh.yaml
│   ├── legacy_to_current.yaml
│   ├── electrode_aliases.yaml
│   └── prohibited_terms.yaml
├── annotation/
│   ├── recording_schema.yaml
│   ├── event_schema.yaml
│   ├── seizure_schema.yaml
│   ├── patient_schema.yaml
│   └── label_status.yaml
├── rag/
│   ├── physiology_cards/
│   ├── montage_cards/
│   ├── background_cards/
│   ├── artifact_cards/
│   ├── normal_variant_cards/
│   ├── abnormality_cards/
│   ├── ied_cards/
│   ├── iic_cards/
│   ├── ictal_cards/
│   └── soz_reasoning_cards/
├── reporting/
│   ├── observation_phrases.yaml
│   ├── interpretation_phrases.yaml
│   ├── uncertainty_phrases.yaml
│   ├── report_template.yaml
│   └── prohibited_claims.yaml
├── reasoning/
│   ├── inference_rules.yaml
│   ├── evidence_hierarchy.yaml
│   ├── contradiction_rules.yaml
│   └── grounding_rules.yaml
└── provenance/
    ├── source_registry.yaml
    ├── ontology_versions.yaml
    └── change_log.yaml
```

---

<a id="sec-22"></a>
# 22. 检索问答模板

## 22.1 “这是 IED 还是正常变异？”

检索应返回：

1. 目标候选模式卡；
2. 至少两个主要 mimic 卡；
3. 状态条件；
4. IFCN 特征；
5. field 和背景打断要求；
6. SOZ 推理边界。

输出格式：

```text
观察事实：...
支持 IED 的特征：...
支持正常变异的特征：...
缺失/不可评价特征：...
最可能解释：...
SOZ 意义：仅作为 irritative-zone 支持/不支持 SOZ。
```

## 22.2 “最早导联能否当 SOZ？”

固定回答逻辑：

```text
先判断标签层级 → 双极导联解码 → 检查相位反转和 field
→ 检查 onset 是否完整 → 检查伪迹/缺失通道
→ 映射为 SEOR → 多发作聚合 → 临床证据整合
```

## 22.3 “左颞慢化说明什么？”

固定返回：

- 局灶性功能障碍/结构异常的非特异证据；
- 多形性、节律性、持续性、间歇性分别说明；
- 与 IED/LRDA/TIRDA 鉴别；
- 与同侧 ictal onset、postictal slowing 和影像一致时可辅助；
- 单独不能确定 SOZ。

## 22.4 “为何报告不能直接写临床 SOZ？”

固定返回：

- 头皮 EEG 受容积传导和深部源限制；
- earliest scalp-visible 不等于 biological onset；
- 双极导联不是单电极位置；
- IED/慢化/传播区与 SOZ 不同；
- 临床 SOZ 需多模态与侵入性证据；
- EZ 需要手术结局间接验证。

---

<a id="sec-23"></a>
# 23. 重点概念快速索引

## 23.1 高优先级 P0

```text
ELEC.ELECTRODE
ELEC.DERIVATION
ELEC.MONTAGE
ELEC.PHASE_REVERSAL
ELEC.END_OF_CHAIN
ELEC.FIELD
TECH.FILTER
TECH.SENSITIVITY
BG.PDR
BG.STATE
ART.MUSCLE.TEMPORALIS
ART.ELECTRODE.POP
ART.OCULAR
VAR.WICKET
VAR.RMTD
VAR.BETS_SSS
VAR.MU
ABN.FOCAL_SLOWING
ABN.BREACH
IED.SPIKE
IED.SHARP
IED.IFCN_FEATURES
RPP.LRDA
RPP.LPD
RPP.BIRDS
ICTAL.ONSET
ICTAL.EVOLUTION
ICTAL.RECRUITMENT
ICTAL.PROPAGATION
ICTAL.POSTICTAL
LOC.EARLIEST_DERIVATION
LOC.ELECTRODE_SET
LOC.SEOR
CLIN.IRRITATIVE_ZONE
CLIN.SOZ
CLIN.EZ
REP.GROUNDING
REP.UNCERTAINTY
```

## 23.2 P1

```text
BG.SLEEP_ARCHITECTURE
BG.REACTIVITY
ABN.GENERALIZED_SLOWING
RPP.GPD
RPP.GRDA
RPP.IIC
ICTAL.SEMIOLOGY
ICTAL.STATUS_BURDEN
LOC.MULTI_SEIZURE_AGGREGATION
REP.CONTRADICTION_CHECK
MODEL.CALIBRATION
```

## 23.3 P2

```text
NEONATAL.PMA
NEONATAL.TRACE_DISCONTINU
NEONATAL.TRACE_ALTERNANS
PEDIATRIC.PDR_DEVELOPMENT
PEDIATRIC.SYNDROMES
CRITICAL_CARE.BURDEN
STIMULUS_INDUCED_PATTERNS
```

---

<a id="sec-24"></a>
# 24. 来源与版本管理

## 24.1 LearningEEG 页面

- 首页与章节目录：<https://www.learningeeg.com/>
- Atlas：<https://www.learningeeg.com/atlas>
- Physiology & Terminology：<https://www.learningeeg.com/physiology-terminology>
- Montages & Technical：<https://www.learningeeg.com/montages>
- Normal Awake：<https://www.learningeeg.com/normal-awake>
- Normal Asleep：<https://www.learningeeg.com/normal-asleep>
- Artifacts：<https://www.learningeeg.com/artifacts>
- Normal Variants：<https://www.learningeeg.com/normal-variants>
- Neonatal：<https://www.learningeeg.com/neonatal>
- Pediatric：<https://www.learningeeg.com/pediatric>
- Non-Epileptiform Abnormalities：<https://www.learningeeg.com/nonepileptiform>
- Epileptiform Discharges：<https://www.learningeeg.com/epileptiform>
- Rhythmicity, Periodicity & the IIC：<https://www.learningeeg.com/rhythmicity-periodicity>
- Seizures & Status：<https://www.learningeeg.com/seizures>

## 24.2 标准与关键参考来源

正式实现时应锁定具体版本、发布日期和 DOI。建议至少登记以下来源：

- ILAE Updated classification of epileptic seizures (2025)：<https://www.ilae.org/updated-classification-epileptic-seizures-2025>
- ILAE Definition & Classification 入口：<https://www.ilae.org/guidelines/definition-and-classification>
- ACNS Standardized Critical Care EEG Terminology: 2021 Version（全文）：<https://pmc.ncbi.nlm.nih.gov/articles/PMC8135051/>
- ACNS/CCEMRC 2021 术语培训入口：<https://www.acns.org/research/critical-care-eeg-monitoring-research-consortium-ccemrc/education>
- IFCN standardized EEG electrode array（PubMed）：<https://pubmed.ncbi.nlm.nih.gov/28778476/>
- IFCN IED 六项标准临床验证：<https://pubmed.ncbi.nlm.nih.gov/32321764/>
- Presurgical evaluation of epilepsy（六区域框架）：<https://academic.oup.com/brain/article/124/9/1683/303186>
- The Epileptogenic Zone: Concept and Definition：<https://pmc.ncbi.nlm.nih.gov/articles/PMC5963498/>
- 本地医院报告规范、专家共识、数据集官方 README 和标注协议。

本知识库中的 ACNS/IFCN/ILAE 条目是面向项目实现的概括；需要临床部署时，应回到原标准逐条核验。

## 24.3 来源优先级

```text
正式指南/共识/标准
> 数据集官方协议
> 同行评议研究
> LearningEEG 教学概括
> 本项目操作定义
> LLM 生成内容
```

若冲突，以更高层级、更新且适用于当前任务的来源为准，并记录变更。

## 24.4 版权边界

本知识库使用自行概括的定义、术语、推理规则和结构化 schema。不得在未获授权时：

- 批量复制 LearningEEG 波形图片；
- 公开重新发布其完整图集；
- 大段复制网页原文；
- 把网站图片打包为公开训练集；
- 删除原始署名后重新分发。

可优先使用：

- 自己渲染的 EEG 截图；
- 自己的 EDF/信号；
- 规范化术语；
- 自行概括的知识卡；
- 指向原页面的来源链接。

## 24.5 版本更新记录模板

```yaml
version: 1.0.1
date: 2026-09-15
changes:
  - "updated seizure classification aliases"
  - "added scalp-negative seizure card"
  - "revised F7/F8 anatomic mapping"
reviewers:
  - role: clinical_neurophysiologist
    status: pending
breaking_changes: false
```

---

<a id="sec-25"></a>
# 25. 最终系统原则

```text
1. 先描述，再解释；先解释，再定位；先定位，再做临床推断。
2. 双极导联是边，不是点。
3. 相位反转是头皮电位极值证据，不是皮层源证明。
4. IED 主要支持 irritative zone，不自动等同 SOZ。
5. 局灶慢化是非特异功能异常证据，不自动等同 SOZ。
6. 发作演变、早期时序、空间 field 和跨发作重复性是核心 ictal 证据。
7. 传播区可能高振幅，但晚于起始区。
8. earliest scalp-visible 不等于 biological onset。
9. unknown、absent 与 not_assessable 必须区分。
10. 报告中的每个事实必须回到信号证据 grounding。
11. 报告—模型一致不等于外部正确性。
12. 临床 SOZ 和 EZ 必须由多模态、侵入性与结局证据综合验证。
```

本知识库面向以下完整链路：

```text
长程 EEG
→ 数据质量与伪迹控制
→ 状态和背景识别
→ 发作检测
→ onset 时间定位
→ 波形形态与演变
→ 最早导联角色
→ 电极集合解码
→ 头皮电图起始区 SEOR
→ 招募与传播图
→ 多发作患者级聚合
→ 非侵入性 SOZ 假设
→ 有证据边界的诊断报告
→ 逐项信号 grounding
→ 低权重、可验证的语义反馈
```

---

## 附录 A：推荐检索标签

```yaml
tags:
  - eeg
  - scalp_eeg
  - seizure_detection
  - seizure_onset
  - soz_localization
  - electrographic_onset
  - montage
  - phase_reversal
  - field
  - artifact
  - normal_variant
  - interictal_epileptiform_discharge
  - rhythmic_periodic_pattern
  - ictal_interictal_continuum
  - evolution
  - propagation
  - clinical_report
  - rag
  - evidence_grounding
  - uncertainty
```

## 附录 B：系统提示词片段

```text
你是 EEG–SOZ 证据推理模块。必须依次输出：
(1) 可观察信号事实；
(2) EEG 模式解释；
(3) 头皮 EEG 空间定位；
(4) 非侵入性 SOZ 假设；
(5) 支持证据；
(6) 矛盾/限制证据；
(7) 临床边界声明。

禁止把双极导联当作单一电极位置，禁止把相位反转、IED、局灶慢化、
最早头皮可见改变或报告—模型一致性直接当作临床 SOZ/EZ 的证明。
对缺失通道使用 not_assessable；对未判断内容使用 unknown；
只有在主动检查且数据可评价时才能使用 absent。
```

## 附录 C：报告输出模板

```markdown
### 技术与质量
- 采样率：{sampling_rate_hz} Hz
- Montage：{montage_name}
- 缺失/坏导联：{missing_or_bad_channels}
- 主要伪迹：{artifact_summary}
- 起始可评价性：{onset_assessability}

### 背景与状态
{background_summary}

### 发作间期异常
{interictal_summary}

### 发作事件
- 最早头皮可见改变：{earliest_scalp_change}
- 起始形态：{onset_morphology}
- 演变：{evolution_summary}
- 招募与传播：{recruitment_propagation_summary}
- 发作后改变：{postictal_summary}

### 头皮 EEG 定位
{seor_summary}

### 多发作综合
{multi_seizure_summary}

### 支持证据
{supporting_evidence}

### 限制与矛盾证据
{limitations_and_contradictions}

### 结论
{noninvasive_soz_hypothesis}

> 本结论代表基于头皮 EEG 的非侵入性推测，不能单独等同于侵入性确认的临床 SOZ 或致痫区。
```
