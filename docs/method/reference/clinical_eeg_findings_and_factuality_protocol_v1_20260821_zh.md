# 长程 EEG 的同源多视图预处理、逐事件 Findings 与报告事实一致性协议

**日期：** 2026-08-21；更新：2026-08-23  
**状态：** v1 方法、标注与评价协议；canonical 多视图、P0 ragged token、冻结 v2 基础合同及原生确定性 producer、additive `event_eeg_findings_v3` schema/validator 与保守 v2→v3 迁移、逐 atom/measurement/waveform raw-dependency 闭包与 evidence/decision 双时钟、完整事件分母 ledger、原子 claim factuality evaluator 与 source-bound case materializer已实现。v3 把 term-bound occurrence/burden/within-event variability、rhythmicity/periodicity 独立资格、acquisition capability、竞争信号假设和显式事件结局提升为 first-class；旧 v2 未记录的部分迁移后一律保持不可评价，不能被猜成阴性或阳性。已有 signal-only synthetic EDF 的 canonical→adaptive→v2 Findings→ledger 纵向回归；v3 当前证明 wire/迁移/反例合同，不代表已有原生临床 producer。source-bound 层从冻结 EvidenceGraph/record graph/render/full roster 内生构造 claim、推导、flow 和固定权重，执行全句/原子分句 ownership 与带外 source replay；它证明来源/序列化闭合，不证明上游临床事实正确。P0 已改为一个 recording-relative 物理时间网格、逐 view 按各自采样时钟向内映射并保存实际 support；dense measurement sidecar 同步保存 nominal window 与实际 sample support，不再要求 onset-causal 与 offline-context 共用采样率。2026-08-22 新增 permission-locked、非致密的 physical-time sparse association graph、analysis-unit onset interval/rank heads、retained-K 合法 physical-time boundary-path marginalization primitive、first-class `pattern_candidates[]/pattern_instance_id` 合同、可执行 `EEG-ClaimGround`，以及禁止双极端点归因、按跨参考稳定性逐级回退的多参考 scalp-field primitive。K-path 与多参考组件当前只证明 future/offline fail-closed、物理时间 pooling、rank stability 和分辨率回退合同，不代表边界概率已校准、空间场已临床资格化，也尚未接入生产。2026-08-23 已实现 `event_findings_atom_roster_v1` 的 draft-shadow policy/schema、source/policy/inventory/self-hash 绑定、28 个固定 core 与 12 个条件 child roster 的确定性 sidecar 及结构反例门；它把 activation、Finding 四态和 technical failure 分开。下游 bundle/claim-plan gate 和私有生产接线仍未连接，且 roster closure 只证明结构完整，不是临床正确性。完整的 morphology/rhythm/periodicity/field/evolution 临床 Findings heads、患者隔离的目标域资格化、私有数据生产路由及前瞻性医师评价尚未完成；`ADAPTIVE_REPORT_ROUTE_CONNECTED` 保持 `False`。  
> **2026-08-23 roster 审计勘误与后续实现：** 原 atom-roster sidecar 仍只证明固定 key 与给定 source 行可重放；其外已依次实现 candidate-blind 的 40-item typed-unit interval-union 结构分母、冻结 34 个术语/41 个 operational query 的逐 term-query 分母，以及唯一 primary/受限 secondary Finding binding ledger。三者均为 public/synthetic shadow，仍不得授权临床阴性、报告 promotion 或生产接线。详见 4.9 节。

**任务边界：** 30--60 min 头皮 EEG；检测至少一次完整电图发作；逐事件提取 EEG-only Findings；跨事件形成研究性头皮电图起始拓扑/排序报告。已历史开标的私有 141 例及医生参考只能作锁定后的事后描述性一致性审计；确认性终点必须来自 fresh patient/site 或从未打开的预注册患者级 holdout。内部旧变量名中的 `SOZ` 仅为兼容标识，不代表皮层 SOZ 或致痫区。

## 1. 三个问题的直接结论

1. **初筛与 Findings 需要统一证据根，但不应强迫使用同一个处理张量。** 二者必须共享 EDF 内容哈希、物理通道、单位、原始采样坐标、记录相对时间、采集带宽和质量账本；detector 继续使用其 checkpoint 的原生输入分布，形态、节律和空间场则使用保留相应物理语义的独立派生视图。
2. **统一 Findings 不应等同于三个粗粒度 concept 分类。** 三个 concept 可保留为医生阅读栏目，但学习与评价应落到可测量的原子证据：形态、频率、节律/周期性、数量与负荷、空间场、演变、随后累及顺序、终止/恢复、质量和可评价性。每条证据区分 `measured → model_candidate → report_eligible_automated`；最后一级只表示自动 producer 通过逐术语报告资格门，不代表医师确认。
3. **通道 Top-1 只回答报告事实性的一个很小子问题。** 完整评价至少覆盖事件发现完整性、输入防火墙、信号到测量、测量到临床术语、逐事件到记录级假设、结构化证据到文本、冻结后医生参考一致性和临床可读性七层。主结果应是分层 dashboard，不合并成一个容易被优化失真的总分。

## 2. 权威来源与适用边界

本协议使用下列资料定义术语、本体和报告结构；它们不证明本文网络、窗长或阈值有效。

临床知识库在流水线中只允许提供受控术语、单位、资格规则和矛盾检查，不能检索相似病例并为当前记录补患者事实。

| 来源 | 可用于本项目 | 不可外推 |
|---|---|---|
| [IFCN 临床 EEG 术语与报告格式修订版，Kane et al. 2017](https://doi.org/10.1016/j.cnp.2017.07.002) | EEG 现象按频率、波幅、相位关系、波形、定位、数量及其变异性描述；事实描述与解释分层；伪迹需披露 | 不提供自动算法阈值，也不授权在缺少病史时生成临床诊断 |
| [SCORE，Beniczky et al. 2013](https://doi.org/10.1111/epi.12135)及[第二版 2017](https://doi.org/10.1016/j.clinph.2017.07.418) | 结构化 episodes、位置、时间特征、图形和诊断意义；支持“先结构化、后叙述” | SCORE 中的临床、睡眠、诱发和多导栏目不能进入本 EEG-only 流水线 |
| [ACNS EEG 报告指南 7，Tatum et al. 2016](https://doi.org/10.1097/WNP.0000000000000319) | 支持标准化报告结构、技术条件披露、描述性术语和在重要/争议波形处附数字波形样例 | 其中病史、睡眠和临床相关栏目依赖本项目没有的输入，不能用占位推断或由模型补写 |
| [ACNS 2021 critical-care EEG terminology](https://doi.org/10.1097/WNP.0000000000000806) | 将频率、形态、位置和演变拆开；提供可复现的 evolution 工程定义；强调波幅单独变化不构成 definite evolution | 原文主要面向重症 EEG，不能把其全部类别和阈值直接当普通癫痫长程 EEG 的目标域金标准 |
| [ILAE/IFCN 长程视频 EEG 最低标准，Tatum et al. 2022](https://doi.org/10.1111/epi.16977) | 长程记录、技术条件、审阅和报告质量的流程参照 | 本任务不读取视频、临床行为或病史，不能使用这些模态相关建议补事实 |
| [SzCORE，2024](https://doi.org/10.1111/epi.18113) | 连续 EEG 自动发作检测的事件级验证、数据划分和可比评价框架 | 只约束 detection benchmark，不能验证逐事件形态、随后累及顺序或 SOZ |
| [Reisinger et al.，2025](https://doi.org/10.1111/epi.18521) | 证明采样率、窗长、量化位数和通道数会同时改变事件敏感度、误报、检测延迟和能耗，支持 provider-native detector view 与 accuracy--compute 联合消融 | 研究面向少通道低功耗检测；不提供头皮多导 Findings、报告术语或 SOZ 标签，也不能直接给本项目选择参数 |
| [癫痫神经技术与 AI 研究报告建议，Viana et al. 2025](https://doi.org/10.1002/epi4.70194) | 用于约束研究问题、数据、方法、统计、复现性和结果披露，特别适合投稿前审计 | 属于研究报告建议，不定义逐事件 EEG 术语，也不证明本方法有效 |
| [Widmann et al. 数字滤波实践，2015](https://doi.org/10.1016/j.jneumeth.2014.08.002) | 说明滤波器、相位和边缘处理会改变时域波形，支持 transform receipt 与 guard/缓存设计 | 不指定本项目最佳截止频率，也不证明任何临床术语阈值 |
| [HED EEG annotation schema，Hermes et al. 2025](https://doi.org/10.1038/s41597-025-05791-2) | 层级事件、上下文、时间和 provenance 的机器表达 | HED schema 是语义基础设施，不是临床 Findings 或 SOZ 真值 |
| [SCORE-AI，Tveit et al. 2023](https://doi.org/10.1001/jamaneurol.2023.1645) | 说明大规模结构化 SCORE 标签可支持记录级 EEG 自动解释 | 其记录级正常/异常任务不提供本项目逐发作边界、募集和 SOZ 金标准 |
| [IFCN IED 六项标准临床验证，Kural et al. 2020](https://doi.org/10.1212/WNL.0000000000009439) | 支持将尖锐瞬变拆成二/三相尖峰形态、相对背景时限差异、波形不对称、后随慢波、背景扰动和生理性头皮电场六项可审阅原子 | 数据是预筛选的间期尖锐瞬变；`4/6` 或 `5/6` 不能直接资格化发作期 spike、ictal pattern 或 SOZ |
| [IFCN 判据专家一致性复核，Yuan et al. 2025](https://doi.org/10.1016/j.clinph.2025.02.275) | 9 名专家、200 个候选的前后比较提示应保留逐项判据、不确定性和校准，而非只训练一个 IED boolean | 使用判据前后 AUC `0.90/0.91`、AC1 `0.48/0.47`，未显著改善总体性能、一致性或校准；不能把六项计数当无噪声金标准 |
| [SpikeNet2，Li et al. 2025](https://doi.org/10.1056/aioa2401221) | 以 13,523 名患者、32,433 个专家标注事件和 hard-negative mining 训练/外部验证，可作为 spike-candidate teacher、事件级 comparator 和难负例策略参考 | 任务是 interictal epileptiform discharge/spike 与 EEG-level 分类，不提供 ictal evolution、最早场、逐事件 SOZ 或报告金标准；checkpoint/data lineage 不清时也不能进入主无泄漏结果 |
| [混合 AI EEG 背景分析与报告生成，Tung et al. 2025](https://doi.org/10.1109/JBHI.2024.3496996) | 支持将可复核算法、学习模型与报告生成分层；其 PDR、广泛/局灶背景异常及外部 TUAB 验证可作为模块化实现和评价参照 | 任务主要是 EEG 背景而非逐发作 onset/evolution/SOZ；以其他 LLM 判定生成文字不能替代 signal→measurement→claim 重放或 EEG 专家盲评 |
| [EEG--language pretraining，Gijsen & Ritter，ICML 2025](https://proceedings.mlr.press/v267/gijsen25a.html) | 证明患者报告与长 EEG 的弱配对可通过 crop/text-segment MIL 改善低标签 phenotyping；支持把 report-level alignment 作为表示学习对照 | 医生报告没有事件时间对齐，并包含病史、药物和解释；MIL attention 不是逐事件 Findings 金标准，且 TUH 系谱必须与 TUSZ/TUEV/DeepSOZ 评价患者去重 |
| [CELM 长程 EEG-to-report，2026 预印本](https://arxiv.org/abs/2601.22197) | 9,922 份报告、约 11,000 h、9,048 名患者的患者级拆分，展示了最长约 3 h EEG 的 epoch aggregation、sequence-aware alignment、zero-context 生成和六名专家评价；可作为长记录报告生成强基线 | 每个 10 s epoch 被高度压缩，报告与局部事件无显式时间 grounding；论文也承认 BLEU/ROUGE/METEOR 主要反映词面且稀有 epileptiform section 较弱，不能证明每条 onset/SOZ claim 由当前样本支持 |
| [NeuroNarrator，2026 预印本](https://arxiv.org/abs/2603.16880) | 提供 segment-level spectro-spatial/temporal-context EEG-to-text 对照，并强调短暂事件需要时间锚点 | 其 160K 文本主要由 GPT-4.1 根据数据集标签、人口学信息、PSD 与通道能量模板合成，不是医师逐段金标准；零相位 10 s 处理和历史上下文也不能资格化因果 onset，GPT 生成再由 GPT adjudicate 不能替代专家/信号重放 |
| [EEG-to-Report 标注与 feature--text 框架，Tran 2026 预印本](https://doi.org/10.21203/rs.3.rs-10452606/v1) | 其浏览器内时间段/通道选择、语音转写和 segment timing + channel context + feature + note JSON 对齐方式可作为本项目专家资格集的人机界面参考 | 论文只报告 pilot annotation framework；feature-to-text 训练仍是 future work，自动报告模块也不能提供本项目 Findings/SOZ 金标准或事实性证据 |
| [Shin et al. 2026 局灶性皮质发育不良 ictal scalp-EEG 侧化/定位](https://doi.org/10.3389/fnins.2026.1915535) | 69 名特定病理患者中，以 CAR、重叠 1 s 窗、五频带、形态/连接特征做早期 post-onset 侧化与定位；支持把明确测量特征和 onset-relative trajectory 设为可复现基线 | 单一 FCD 人群、样本量有限且是侧化/粗定位分类，不提供跨病理逐事件 Findings、长记录 detector、自由报告或 claim grounding |
| [头皮 EEG 特异性间期/发作起始模式综述，Baumgartner et al. 2025](https://doi.org/10.1016/j.yebeh.2025.110298) | 支持把 5--9 Hz 颞区节律、阵发性快活动、重复癫痫样放电等组织成可审阅的 pattern candidates，并提示颞外发作常在头皮上不可定位 | 属于叙述性综述，模式与病因/深部起始不是简单一一对应；不能把 pattern 名称直接变成病因、皮层 SOZ 或术语资格标签 |
| [非同步 scalp video-EEG 与后续 SEEG 发作类型配对研究，Bolzan et al. 2024](https://doi.org/10.1002/epi4.12886) | 41 名 MRI 阴性药物难治性局灶癫痫患者中，报告的 `9.75 s` 是临床起始到头皮 EEG 起始的延迟；提示头皮 onset pattern 具有可见性限制，应保存“深部/更早活动可能不可见”的限制状态 | scalp video-EEG 与 SEEG 不是同步记录，而是先后检查后按发作类型配对；该延迟不是 SEEG→scalp 延迟，本 EEG-only 流水线没有视频/临床起始，因而不能测量或生成该数值；也不能据其建立自动皮层定位规则 |
| [头皮 EEG--SEEG 起始对应系统综述，Santos et al. 2025](https://doi.org/10.1016/j.eplepsyres.2025.107666) | 支持将 scalp-visible onset 与潜在深部 onset 分轨，并将最终输出限定为 scalp-visible onset hypothesis | 仅纳入 8 项异质性回顾性研究、250 名患者；不能提供本项目逐事件映射规则，也不能用头皮排序替代 SEEG/多模态术前评估 |
| [ILAE 2025 发作分类立场文件](https://doi.org/10.1111/epi.18338) | 约束 focal/generalized/unknown/unclassified 等术语的临床含义，并提醒意识状态和按时间顺序的发作表现依赖临床观察 | 本项目没有视频、行为、意识或反应性输入，不得从 EEG 单独生成意识状态、运动/非运动 semiology 或完整 ILAE 临床发作类型 |
| [RadGraph，Jain et al. 2021](https://arxiv.org/abs/2106.14463) | 医学报告应先表示实体、断言状态和关系，再评价文本表面 | 胸片实体/关系本体不能直接迁移成 EEG 术语，也不提供信号级 grounding |
| [RadCliQ/胸片报告评价，Yu et al. 2023](https://doi.org/10.1016/j.patter.2023.100802) | 说明 BLEU 等词面指标与临床错误不等价，评价应与专家错误对齐 | 其权重和胸片标签器不适用于 EEG；本项目不能在无配对 EEG 报告时照搬分数 |
| [GREEN，Ostmeier et al. 2024](https://doi.org/10.18653/v1/2024.findings-emnlp.21) | 可借鉴 major/minor error、遗漏和矛盾的错误分类 | 生成式 evaluator 不能替代本地可重放的 signal→measurement→claim 合同或双医师盲评 |
| [RadFact/MAIRA-2，Bannur et al. 2024](https://doi.org/10.48550/arXiv.2406.04449) | 可借鉴双向 entailment precision/recall 与 evidence grounding，将其改造成 event×time×channel×temporal-role 的 EEG claim grounding | 胸片图像/文本 checkpoint 与空间 box 不能直接给中文 EEG 报告打分；原子结构化检查仍是主判据 |

ACNS 对 evolution 的定义尤其适合作为可检验的候选标签：在频率、形态或位置的**同一类别**中至少出现两次明确连续变化。频率变化需同方向、每次至少 `0.5 Hz`，每个频率状态至少持续 3 个周期；形态演变要求连续两次进入新形态，每个形态或其过渡形式至少持续 3 个周期；位置演变要求依次进入或退出至少两个不同的标准 10--20 电极位置，每个新增位置至少持续 3 个周期。若该演变维度连续 5 min 或更久保持不变，则之前与之后的变化不能拼接成一次 evolution。仅波幅变化不能升级为 definite evolution。本项目将这些条件实现为可配置 candidate rule，并在目标域专家子集上重新验证，而不把 critical-care RPP/ESz 定义原样视为普通长程癫痫 EEG 中 seizure、onset 或 SOZ 的必要/充分条件。`rhythmicity`、bandpower 或波幅轨迹可作为测量，但不是 ACNS definite-evolution 的第四条资格轴；`equivocal evolution` 若保留，只能标为项目自定义候选状态，不能伪装成 ACNS 标准术语。

ACNS 账本必须拆开：`evolution_candidate_receipt` 只回答频率/形态/位置的连续变化；ESz criterion A 原文允许平均 `>2.5 Hz` 且连续 `>=10 s` 的 epileptiform discharges，且其注释明确某些 sharply contoured discharges 即使主成分 `>200 ms` 也可满足该条；criterion B 另行要求 definite evolution 且持续 `>=10 s`。因此，本项目若在 criterion A 前额外强制“通过独立 discharge-pattern/目标域资格回执并排除伪迹”，只能称为**项目更保守的自动报告安全门**，不能冒充 ACNS 原文完整定义，更不能误写成必须先获得间期 IED 资格。两条 ACNS criterion、项目安全门、伪迹/良性变异排除和目标长程人群 qualification 分别落账。未通过项目门时只称电图事件候选；短于 10 秒也不能因 ACNS 来源而自动命名 BIRDs。ECSz 依赖与 EEG 时间锁定的临床相关表现，或某些定义中的治疗反应；本 EEG-only 流水线没有这些输入，因此禁止生成 ECSz/ECSE、意识、运动/非运动或行为学分类。

## 3. 预处理：统一什么，不统一什么

### 3.1 必须统一的 canonical evidence root

每条 EDF 建立一个不可变 `CanonicalEEGRecord`：

```text
signal_sha256
native physical samples and unit-to-volt scale
native channel names -> canonical physical electrode map
acquisition reference if known, otherwise explicit unknown; electrode coordinates/10-20 coverage
native sample index <-> recording-relative rational clock
native sample rate and acquisition high/low-pass metadata
missing/duplicate/non-EEG channel ledger
flat/clipping/step/line-noise/edge-invalid quality primitives
annotation/excel/doctor-label firewall receipt
```

这里的 `physical` 只表示按 EDF 标定恢复的电位差、物理电极身份和可追溯单位，不表示获得了无参考绝对电位或皮层源。采集参考未知时必须显式记为 unknown；CAR、双极和 Laplacian 均是派生观察视图，不是源定位算法。

所有 onset、窗口、Finding、波形附件和报告时间都必须回写到这个时间轴。嵌套扩窗不能对同一个物理样本反复独立滤波；应对整条记录或固定 guard tile 一次处理并缓存，扩窗只增加索引和 token。

### 3.2 不应统一成一个张量的任务视图

| 视图 | 目标 | 允许的专用处理 | 不能支持的事实 |
|---|---|---|---|
| `D_provider` | 连续 seizure detector | 严格复现各 checkpoint 的 fs、montage、滤波、裁剪、归一化和 padding | z-score/裁剪后的物理波幅、临床形态、插补通道事实 |
| `B_coarse` | 全记录 QC、候选拆并、`S0/S1/S2/S3` 计算状态和背景检索 | 低成本连续滤波、降采样和秒级特征 | spike/HFO、明确传播或某电极为源 |
| `F_native` | 瞬变、sharpness、斜率、物理幅度 | 保留原采样率、单位和真实采集带宽；带回执的去趋势 | 上采样生成的高频、全局 z-score 后的振幅 |
| `F_onset_causal` | earliest interval、lead-lag、起始正证据 | 只访问当前及更早样本；保存 warm-up、原始样本依赖区间和处理延迟；已知群延迟不得通过简单前移时间戳制造更早证据 | 零相位前振铃、延迟输出向前平移或 onset 后归一化把晚期活动搬到起始前 |
| `F_context_offline` | 频率、节律、形态可读性、演变和终止 | 带 guard 的零相位/双向全事件表示；保存实际有效带宽 | 由 late spread 创建新 SOZ 正证据 |
| `S_car/S_bipolar/S_laplacian` | 空间场、相位反转和参考稳定性 | 显式 reference matrix，多参考扰动 | 将一个双极导联的变化直接归因给某一端电极 |
| `W_display` | 医师复核波形 | 冻结显示滤波、增益、通道顺序和图像哈希 | 图中“看起来像”的新事实 |

每个 view 保存父信号哈希、请求/有效区间、采样映射、滤波、采集参考与派生参考、单位、clipping/normalization、padding/guard/quality mask、软件版本和输出哈希。每条 native measurement/waveform 进一步保存 `raw_sample_dependency`，Finding 保存依赖 ID 闭包；onset-positive 输出同时区分 reported evidence interval、decision availability、`processing_latency` 与 `confirmation_latency`。若延迟滤波结果没有原生/瞬时证据锚定，只能扩大起始区间或保留较晚边界，不能减去群延迟后把主张时间前移。缺失电极即使为 detector 兼容而插补，也必须是 `observed=false, evidence_eligible=false`。

`F_onset_causal` 与 `F_context_offline` 不是重复算两次后任意择优。前者是正 onset/SOZ 支持的唯一时间方向来源；后者只能描述完整事件、佐证已有候选或产生目标明确的反证。若某通道的“最早变化”仅在零相位/双向路径出现，必须降为 `temporal_direction_unstable`，不能进入精细 Top-k。

### 3.2.1 多采样时钟的物理时间合同

“统一时间轴”不等于“统一采样率”。主方法先在 recording-relative seconds 上定义事件和 nominal fine/coarse/context tile；对每个 view 独立执行：

```text
nominal [t0,t1]
  -> start = ceil(t0 * fs_view)
  -> stop  = floor(t1 * fs_view)
  -> actual support = [start/fs_view, stop/fs_view]
```

实际 support 必须位于 nominal tile 内并进入 token、deterministic target 和 receipt。禁止为对齐不同 view 而在事件局部重采样、time-warp、补零或借用窗外样本；极短尾窗在某个慢时钟上不足最小样本数时，只从该 view 的 ragged token 中省略，不能制造证据。onset-causal view 保留 native clock，offline-context 可按全记录、带 anti-alias receipt 的策略降采样。local encoder 的因果 mask 必须依据 token 的实际 raw-support end/decision time，而不是 token 行号；后者按 view/unit/scale 排序，并非全局时间顺序。

### 3.3 统一策略的验收实验

预注册四臂：

```text
A  detector tensor 直接复用于 Findings
B  各模块独立截窗读 EDF、独立滤波
C  canonical root + 单一临床 view
D  canonical root + detector/morphology/rhythm/spatial 多视图（主方法）
```

同时评价 detector sensitivity/FA per hour/onset error、物理量重放误差、Findings typed F1、参考扰动稳定性、时间漂移、事实错误、RTF 和峰值内存。只有 D 在不损害检测性能的条件下改善真实性或 accuracy--compute Pareto，才可成为论文主方法。

## 4. 每个 onset 事件的统一 Findings 最小信息集

### 4.1 事件、时间和上下文

- `event_id`、recording-relative 时间系统和信号/view receipts；
- detector support interval 与 posterior，不把 detector anchor 当精确 onset；
- detector roster 中每个候选及完整 recording-level detection opportunity；漏检、合并、拆分和未进入 Findings 的候选不得从评价分母消失；
- 搜索区间、最终事件区间、onset/offset **区间分布**；起始未观察、记录左删失或 detector 晚报时允许 onset boundary 为空并携带 `censored/not_observed` 状态；
- 统一非临床计算状态及后验：`S0=background-compatible signal → S1=candidate emergence → S2=sustained/evolving candidate signal → S3=return/after-event candidate`；质量 `Q` 独立 overlay。状态名不授权“背景正常、ictal、seizure、termination 或 postictal”；另设 `event_qualification_status = unqualified_candidate | qualified_electrographic_event | qualified_electrographic_seizure | not_evaluable`，只有资格回执才允许将计算状态投影为临床事件阶段；
- 每段保存 `signal_temporal_context = outside_candidate_protection | pre_candidate | candidate_emergence | sustained_candidate | late_involvement | return_candidate | unknown`、所属 `event_id/event_group_id`、保护区版本及与保护区的交叠；`outside_candidate_protection` 或 detector-low 仍不自动等于 interictal，需独立资格后才可使用该词；
- left/right/search-cap censoring 和候选 merge/split 状态；
- local/distant background intervals、检索相似度和污染风险；
- interval 本身不授权 baseline/recovery：必须通过独立的[事件级 baseline/context comparability 合同](clinical_eeg_event_baseline_context_comparability_v1_20260822_zh.md)，逐段绑定 protection zone、质量/污染、同 view/reference/clock/bandwidth、校准相似度及用途权限；2026-08-22 安全审计确认 v1 尚缺可信 canonical-view、quality/contamination、comparison-measurement 和 calibration registry，因此当前只保存 measurable/comparable candidate，distant-background、emergence、return/recovery 报告支持全部 fail closed；不可比 context 不得支持 return/recovery，任何 within-record comparison 都不得生成背景正常/异常结论；
- 所有时间的下界、中位数、上界、分辨率和 calibration status。

### 4.2 质量与可评价性

- 每电极/导联 availability、usable fraction 和缺失原因；
- flat、clipping、step、line noise、高频污染、运动样/电极样候选区间；
- 对每个显著瞬变保存 `cerebral-compatible` 与 `artifact-compatible` 两个可竞争假设及其证据，未能排除伪迹时不能只保留脑源解释；
- 每个 feature family 独立的 `available / limited / not_evaluable`；
- native fs、采集低通和实际分析带宽；
- 记录边缘、滤波 guard、padding 和插补 mask。
- 对每个候选并列保存 `cerebral-compatible | physiologic/benign-compatible | uncertain-pattern-compatible | artifact-compatible` 竞争假设；具体生理/良性名称若依赖年龄、睡眠、反应性或临床上下文则保持 `not_evaluable`；
- EEG-only 独立事件结局状态：`qualified_scalp_electrographic_ictal_pattern | candidate_not_qualified_as_a_scalp_electrographic_ictal_pattern_within_queried_support | obscured_by_artifact | not_possible_to_determine`，不能把后三类全部压成一个无原因的 nonlocalizable。旧 v3 wire 的 `no_demonstrable_scalp_ictal_change` 只作历史来源审计，active report adapter 必须降级为 `candidate_only` 并附前述资格失败理由；没有视频/按钮等独立临床事件锚点时不得输出“临床事件无脑电对应”。

`not_evaluable` 不是阴性，缺失也不是“未见异常”。伪迹来源只能写成 EEG 波形候选，不能从波形推断被试行为。

### 4.3 原子 Finding families

| Family | 必存原子测量 | 可资格化的临床表达 | 主要防错条件 |
|---|---|---|---|
| morphology | component duration、phase count、rise/fall slope、sharpness、after-going slow component 的 latency/duration/field、峰谷幅度、重复模板一致性 | spike-like、sharp-like、polyspike、spike-wave、attenuation/electrodecrement、LVFA candidate | 单点峰、单导 motif 或带宽不足不能升级；spike/sharp 必须过形态、电场和伪迹门 |
| rhythm | autocorrelation、cycle consistency、instantaneous/dominant frequency、spectral entropy、形态一致性和持续时间 | rhythmic activity；项目自定义 indeterminate-rhythmic candidate | bandpower 增大不自动等于节律；`quasi-rhythmic` 不作为无独立资格门的临床捷径；工频和肌源污染先过滤 |
| periodicity | successive-discharge intervals、interval variability、element duration/morphology、modifier 与持续时间 | periodic discharges/activity | 必须有可分辨重复 elements 与量化 inter-discharge interval；不能由单一谱峰或 autocorrelation 直接升级 |
| spectral/amplitude | dominant frequency/range、bandpower、RMS/peak-to-peak、line length、frequency/amplitude slope、相对背景变化 | θ/δ/α/β 范围活动、频率增减、波幅变化 | 临床频带词必须绑定实际频率；物理波幅仅来自保单位 view |
| occurrence/burden/variability | discharge/burst count、rate、occupancy、run-duration distribution、event incidence、mode frequency 与跨事件变异 | “在可评价时段内出现 n 次/占比 x%”“多次事件呈两种头皮起始模式” | 分母绑定有效可评价时间和去重后的完整 roster；重叠候选、复制事件、不可评价时段不得增加数量或负荷 |
| spatial field | active units、reference-specific 正/负 polarity map、观测极值与梯度、field extent、phase-reversal pair、共享电极、左右同步/不对称、coverage 与 reference perturbation stability | 局灶/区域优势场、双侧广泛场 | 单一 montage 或一个双极 edge 不足以生成物理电极源结论；“最大”只能指该参考下观测极值 |
| evolution | frequency/morphology/location 的有序状态与 change points、每状态持续周期、改变方向/幅度和总持续时间；rhythmicity/波幅另存测量 | ACNS-derived definite-evolution candidate；项目自定义 indeterminate | 只有波幅或 rhythmicity 改变不能称 definite evolution；location change 不自动等于因果传播 |
| later-involvement/order | 每 unit/field involvement interval、区间 lead-lag、partial order、near-synchronous set | 资格化后方可写“首先见于……随后累及……” | near-synchronous tolerance 必须绑定采样/边界分辨率；区间重叠只能写近同步/顺序不可分；association edge 和 location evolution 都不是因果传播 |
| offset/return candidate | last-pattern/offset interval、abrupt/gradual cessation、return-to-comparable-background、after-event slowing/attenuation measurement | 事件资格后方可写电图终止和事件后 EEG 改变 | 右删失、伪迹跨边界、事件连续性不清或无可比背景时不能声称自然终止/恢复 |

频带功率、主频和“节律”必须分轨。例如 `theta_power_ratio` 增高只是 `4--<8 Hz` 频带的数值事实；只有同时满足实际主频范围、周期一致性、持续时间和质量/伪迹门，才可升级为“θ 范围节律性活动”。在缺少年龄、警觉状态、反应性和临床上下文时，不应把它直接命名为正常或异常的“θ 节律”。

HFO 与 LVFA 必须使用不同资格门。HFO 通常需要远高于常规低密度头皮 EEG 的有效带宽，默认只作可选探索分支；LVFA 是低幅快活动，需独立检查物理低幅、快活动频率、采集带宽和肌源性污染，不能因 HFO 不可评价而一并关闭，也不能借 LVFA 名称绕过高频/伪迹门。

DC shift、large slow wave，以及彼此分开的 irregular delta 与 irregular theta 等 acquisition-sensitive pattern 同样必须有逐术语带宽/耦合/时间常数资格；常规高通或未知采集链下 DC shift 为 `not_evaluable`，不能静默记为 absent。`rhythmic` 与 `periodic` 使用独立资格门；`ACNS-2021-rule-derived definite-evolution candidate` 只允许 frequency、morphology 或 exact location 的连续明确变化。项目自定义的 distribution/field-extent 变化、rhythmicity、burden 和 amplitude 只能作为并列测量，不能冒充 exact location evolution。

截至 2026-08-22，仓库新增了两个独立的 signal-only sidecar primitive：一条从连续、同 view、内容绑定的主频状态生成 ACNS-derived **frequency-evolution candidate**；另一条显式分割 bounded waveform elements 并保存相邻 element interval ledger，生成 **element-interval periodicity candidate**。二者都只允许 `model_candidate/course-or-context-only`，不支持 onset/SOZ 正证据，也尚未注入 `event_eeg_findings_v3` 或取得目标域术语资格；因此当前 native v3 的临床 periodicity/evolution gate 仍不得写成 report-eligible。

#### 4.3.1 原子证据之上的复合 pattern 层

旧三 concept 中混合在一起的 `rhythmic δ/θ`、LVFA/PFA、repetitive spike/sharp、attenuation/electrodecrement、spike/polyspike--slow-wave 和 widespread near-synchronous 等，应成为显式 `pattern_instance`，而不是新的不可解释单标签 head。每个 pattern 至少绑定 required atom IDs、counterevidence IDs、实际频率/持续时间/物理波幅、空间场、多参考稳定性、起始区间、质量/带宽机会和资格规则。pattern 名称只说明头皮 EEG 可见表型；不得直接映射到病因、深部结构、皮层 SOZ/EZ 或手术靶点。

`spike/sharp-like` 也不能由单一 sharpness 阈值产生。按 [IFCN 临床 EEG glossary](https://doi.org/10.1016/j.cnp.2017.07.002)，spike 的主成分时长为 `20--<70 ms`，sharp wave 为 `70--200 ms`；这只是描述性时长边界，二者在 glossary 中仍要求清楚区别于背景并具癫痫样意义，不能把任意落在时长范围内的峰命名为 spike/sharp wave。参考 [IFCN 六特征临床验证，Kural et al. 2020](https://doi.org/10.1212/WNL.0000000000009439)，每个瞬变必须保存六项独立四态（`present | absent_with_opportunity | uncertain | not_evaluable`）：

```text
di_or_triphasic_sharp_or_spiky_morphology
duration_differs_from_background
waveform_asymmetry
slow_after_wave
surrounding_background_disruption
physiologic_scalp_field
```

第一项严格指二相或三相、具有 pointed sharp/spiky peak 的波形，不能把任意“多相”波形算入。每项绑定自己的测量、波形、背景、montage 和质量证据；另存良性变异/伪迹反证、criteria count、冻结 operating-point/qualification receipt。后随慢波需有 latency、duration 和 field；“脑源相容场”需有正负 scalp polarity/topography，单纯双极相位倒转不等价。Kural 研究中 `>=5/6` 的 sensor-space operating point 得到 `95.65%` specificity 和 `81.48%` sensitivity；`4/6` 虽有较高总体 accuracy，但 specificity 仅 `85%`。但该结果来自预筛选的间期 sharp transients、100 名患者和视频 EEG 临床参考，不能直接资格化发作期 spike-like、ictal onset pattern 或当前目标域。因此 `5/6` 最多作为**保护区外 IED-candidate** 高特异 promotion gate 的开发初值，仍需目标域患者隔离 qualification；对所有 spike/sharp-like 瞬变保存六项原子只是为了可审计，并不代表应用了 IED 阈值。事件保护区内或持续候选事件组成部分中的 spike-like/repetitive spike 只保留形态事实，不能命名 IED；IED 还需独立的间期上下文资格。

[Yuan et al. 2025](https://doi.org/10.1016/j.clinph.2025.02.275)进一步表明，即便由 9 名专家评价同一批 200 个候选，显式使用 IFCN 判据前后也没有显著改善总体 AUC、专家一致性或校准。因此主方法不再把 `4/6` 或 `5/6` 设计为固定硬标签 head：六项分别输出概率/四态和证据区间，`criteria_count` 只作可解释基线特征；是否可写入报告由目标域、患者隔离的逐术语资格门决定。

[IFCN/ILAE 2023 最低记录标准](https://doi.org/10.1111/epi.17448)的 `256 Hz` 和 `0.5--70 Hz` 是技术建议；当前 `200 Hz/0.5--45 Hz` 的自动降级是本项目安全策略，不是指南直接证明所有 spike 均不可评价。receipt 必须分开记录 acquisition analog bandwidth、stored sampling/Nyquist、analysis filter 和 display filter，再按术语做 eligibility；插值不能恢复已经丢失的形态或频率内容。

为避免“哪个特征最醒目就被当成 SOZ”，先给 Finding 分配与具体候选无关的**内在时间/用途资格**：

1. `onset_eligible`：资格化 onset 区间内最早可分辨的持续变化、相容空间场、多参考稳定性和逐 unit/field 区间；
2. `early_context`：同一早期区间内的频率、形态、节律性或 attenuation/LVFA 候选，本身不授权精细空间 Top-1；
3. `later_involvement`：晚期高幅活动、后续场和 return candidate，只描述事件轨迹；
4. `non_event_context`：保护区外的背景或独立 IED candidate；
5. `limitation`：广泛近同步、参考不稳定、关键导缺失、顺序不可分、伪迹或边界高熵。

“支持/反驳”不是 Finding 的固有属性，必须作为相对于某个研究假设的关系另存：

```text
hypothesis_evidence_relation_id
hypothesis_id / axis / candidate_id
relation: supports | contradicts
evidence_ids
producer/policy receipt
```

例如右侧早期场可以支持右侧候选并反驳左侧候选；仅写全局 `contradiction` 会丢失目标。`later_involvement` 只可支持“后续累及/快速双侧化”等对应谓词，永不支持 SOZ laterality/region/channel；IED 和非事件背景只作独立 context/concordance，不能进入 SOZ 空间轴的正支持或提升其输出分辨率。

每条 Finding 必须保存：

```text
evidence_id / family / controlled term ID + ontology/source/version/rule ID
assertion_level: measured | model_candidate | report_eligible_automated
status: present | absent_with_opportunity | uncertain | not_evaluable
signal_temporal_context / event and protection-zone ownership
intrinsic_evidence_role: onset_eligible | early_context | later_involvement | non_event_context | limitation
S0/S1/S2/S3 membership, event qualification status and physical time interval
lead/electrode/region/laterality support with mapping status
reference-specific polarity/field support and coverage
boundary/quality/background/model/reference uncertainties
qualification receipt and waveform evidence IDs
```

每条 `measurements[]` 不能只存数值和单位，还必须逐条绑定：

```text
measurement_id / value / unit / numerical uncertainty
source_view_id + view/transform/tensor hashes
canonical source unit IDs and recording/sample interval
actual montage/reference and effective bandwidth
background_reference_ids when computing a contrast
quality/edge/padding mask IDs
producer/method/policy receipt
non-null raw_sample_dependency + dependency ID/hash
```

原生 v2 合同现已把上述绑定下沉到原始采样级：每条 native measurement 的 `source_binding.raw_sample_dependency` 与每条 waveform evidence 均按 canonical channel 保存 `raw_start_sample`、`raw_stop_sample_exclusive`、`reported_evidence_start_sample`、`reported_evidence_stop_sample_exclusive`、`unshifted_decision_available_stop_sample_exclusive`，以及 view/transform/receipt lineage 与内容哈希；每条 Finding 的 `raw_sample_dependency_ids` 必须精确等于其 measurements 和 waveform evidence 所引用依赖的去重闭包。v1→v2 migration 无法恢复这些事实时只能写 `null/[]` 并记录 migration-loss code，不能猜测。

这里显式区分两个时钟和两类延迟：`evidence_recording_interval` 是报告所描述的波形区间，`decision_available_recording_seconds` 是持续性条件满足后算法最早可作出该判断的时间；`processing_latency` 只记录 FIR 等处理延迟，`confirmation_latency` 只记录 sustained-confirmation 相对 reported interval 的额外等待。raw support 组件分别标记 `baseline_reference`、`reported_evidence_interval` 与 `sustained_confirmation`。对 onset-positive causal evidence，validator 强制 raw support stop 不晚于**未平移的** decision availability，且不能用群延迟减法前移 reported onset；offline dependency 必须声明 future access 且 `onset_support_eligible=false`。

这样 detector 的归一化张量不能支持物理波幅，低带宽 view 不能支持越带宽频率，单一双极 edge 也不能静默升级为电极场；“更早的波形区间”也不会被误写成“算法在该时刻已经确认”。

需区分 `Finding family` 与 measurement binding 中的底层 `source evidence family`：后者当前只描述 view 对 amplitude/morphology/spectral/spatial-field/high-frequency/waveform 的物理资格，不是 Findings 本体枚举。`rhythm` 必须逐项绑定其 spectral/waveform 测量，`evolution` 必须绑定每个频率/形态/位置 change point，`spatial_recruitment` 必须绑定空间场与逐 unit 时间区间，`termination_recovery` 必须绑定 offset/背景对比。不得因为 schema 中没有同名 source family 而只保存一个无法重放的派生分数；后续 wire 升版可将该字段重命名为 `source_evidence_family` 以消除歧义。

当前 v2/v3 wire schema 已统一使用 `report_eligible_automated`；它的唯一允许语义是“自动 producer 通过逐术语报告资格门”，**不等于本病例经过医师确认**。历史文档或产物中的 `clinically_qualified` 只能在显式 legacy migration 中读取，不得由新 producer、qualifier、renderer 或 evaluator 写出。当前 schema 的 `evidence_role=onset_support|spread_support|contradiction|context_only` 只是兼容投影：`onset_support` 只表示 `onset_eligible`，不表示支持所有假设；`contradiction` 必须同时有上述 target-relative relation 才完整。

原子证据之上可以增加一个**可组合的发作模式候选层**，但它不是第四个不可解释的大类 classifier。首轮只允许由已资格化原子项组合出：`5--9 Hz temporal rhythmic candidate`、分开的 `rhythmic-delta onset candidate` 与 `rhythmic-theta onset candidate`、分开的 `ictal low-voltage-fast candidate` 与 `fast-spike/paroxysmal-fast candidate`、`repetitive spike/sharp-like candidate`、`attenuation/electrodecrement candidate`、`generalized spike-/polyspike-wave scalp-pattern candidate`、`widespread near-synchronous candidate` 以及 `bilateral asynchronous/independent/multifocal scalp-pattern candidate`。跨 delta/theta 边界或落在预注册边界不确定区的 run 只能保留 generic rhythmic candidate；ACNS RDA 只命名 delta。每个 pattern 必须保存组成原子、持续时间、实际主频/幅度、空间场、参考稳定性、起始区间和反证；无法满足其中任一必要项时仍停留在原子描述。文献中关于“提示内侧颞叶”“提示局灶性皮质发育不良”等病因或深部定位关系只用于提出待验证假设，不能进入 EEG-only 正文。这样既保留临床医生熟悉的 onset-pattern 阅读方式，又避免在金标准不足时用一个 pattern 标签掩盖测量、时序和空间错误。

目标本体和 v2 schema 已支持 `absent_with_opportunity`；v1 兼容 schema 仍只有 `present | uncertain | not_evaluable`。即使在 v2 中，也只有绑定有效 `evaluation_opportunity`、足够带宽/质量覆盖和 sensitivity qualification receipt 的 producer 或专家显式阴性才可使用 `absent_with_opportunity`；当前确定性 producer 未具备该灵敏度资格，因而不生成阴性 Finding。`not_evaluable` 只表示无法评价，不能冒充阴性；边界的 `not_observed/censored` 独立保存，不与 Finding 阴性合并。

起始候选的正证据必须进一步 fail-closed：只允许 `status=present`、`intrinsic_evidence_role=onset_eligible`、处于资格化 onset/S1 区间且属于冻结 onset-family allowlist 的空间场、最早 involvement interval 或 evolution 证据，并要求显式 target-relative `supports` relation。晚期扩散、S3、background、IED、`uncertain` 或非空间 Finding 只能描述轨迹、提供 context 或形成有目标的反证，不能独立支持精细 Top-1。

### 4.4 空间起始与逐事件研究假设

原子 Findings 之后才形成：

- 每导联/物理电极/区域的 onset interval；
- earliest distinguishable set，而非强迫唯一最早点；
- field-involvement partial-order graph；只有事件和空间关系均通过资格门后才可显示为 recruitment；
- laterality、region、channel 候选排序；
- 单事件的 `localized_or_lateralized_scalp_visible_onset_pattern`、`localized_scalp_onset_with_rapid_bilateral_later_involvement`、`bilateral_near_synchronous_or_rapid_bilateral_later_involvement_ambiguous`、`widespread_bilateral_near_synchronous_scalp_onset_pattern` 或 `scalp_onset_nonlocalizable` 表型。若旧 wire 仍使用 `focal/generalized_synchronous` 内部 ID，它们只作兼容编码，禁止原样进入用户可见正文。

`multiple_scalp_onset_modes` 不能作为一个事件的表型；而且只有患者隔离、穷尽的 event→mode assignment 与逐事件 onset-field 双专家 gold 闭合并在 held-out gold 上资格化后，才可由跨事件记录级模型自动产生。缺少该 gold 时只能输出 `latent event-heterogeneity/discordant onset evidence`，只评价重采样、参考扰动和 leave-one-event-out 稳定性，不报告 mode purity、校准的多模式概率或 mode accuracy。上述输出必须标为研究性头皮可见起始假设；“广泛双侧近同步头皮电图起始表型”不等同于 ILAE generalized seizure，也不能推出全面性癫痫综合征。任何表型均不等于真实皮层 SOZ、致痫区或手术靶点。

### 4.5 本任务必须排除的字段

EDF annotation、Excel 起始/显著/扩散通道、临床行为、视频、意识、病史、用药、睡眠分析、诱发试验、ECG/EMG/EOG 和身份字段均无权进入 Findings、SOZ 推理、Qwen 或正文。Excel 与医生标签只在输出冻结后进入独立 evaluator。

### 4.6 事件证据与整条记录证据分轨

除逐发作 `event_eeg_findings_v3` 外，可并行生成一个 EEG-only 记录上下文 bundle，保存全记录可判读率、背景 prototypes、非发作期 spike/sharp-like 候选的数量与空间分布。v2 是 v3 的冻结基础投影与当前确定性 producer，v1 只保留为兼容 baseline。该分支有三条硬限制：

记录上下文还应分别保存 detector 原始候选数、去重后独立事件有效数、合格/不可评价事件数、各 EEG-derived event mode 的次数与占时比例，以及独立瞬变候选率。数量统计必须绑定 detector roster、保护区和去重哈希；同一波形的重叠候选或复制事件不能增加发生率、模式支持或 SOZ 置信度。

1. 保护区外的 spike/sharp-like 首先标为 `non_event_context candidate`；detector-low、背景 bank 成员或 `S0` 本身不足以命名 interictal；
2. 只有独立通过时间上下文和 IED 资格后才可称 interictal IED；它仍属于 irritative/context finding，不进入 SOZ laterality/region/channel 的正支持，只能另列空间一致/冲突描述；
3. 事件保护区内、onset 候选中或持续事件组成部分中的 spike/sharp-like 不得复用 IED 标签；未完成目标域 qualification 时只能输出候选数值。

这样既利用 30--60 min 长记录中的远端信息，又不把“间期最大导联”“发作期晚期高幅导联”和“最早头皮起始场”混为同一标签。

record-level validator 还必须冻结 event ownership、保护区、重叠/聚类规则和 waveform identity 去重。背景 bank 不能只依赖 detector 低分，应联合独立异常/QC 门并在扩窗后迭代清除事件保护区；背景永不参与 field-involvement/recruitment edge，非连续 context 也不能拼成 event duration。

### 4.7 `event_eeg_findings_v2` 合同与原生确定性 producer 已落地，但私有生产路由尚未接线

当前 v1 是已运行生产路径中的过渡 wire schema，不是最终本体。独立 v2 JSON Schema、运行时 validator、显式 v1→v2 migrator，以及不经过 migrator、直接消费 canonical/adaptive/task-view receipts 的原生确定性 v2 producer 均已实现；下列字段已成为 v2 的原生机器合同：

- 非临床 `S0/S1/S2/S3 posterior` 与独立 `event_qualification_status`，不再把 wire 字段命名为 `early_ictal/evolved_ictal`；
- `signal_temporal_context`、event/protection-zone ownership 和 `F_onset_causal/F_context_offline` view role；
- `present | absent_with_opportunity | uncertain | not_evaluable` 四态，以及逐 family `evaluation_opportunity_id + sensitivity_receipt_id`；
- `intrinsic_evidence_role` 与 target-relative `supports/contradicts(hypothesis, axis, candidate)` 分离；
- observed、imputed、evidence-eligible、missing reason 分离；
- term 的 ontology/source/version/rule ID、受控单位 registry、`measurement_id` 和逐测量数值不确定性；
- measurement/waveform 的 non-null `raw_sample_dependency`、Finding 的依赖 ID 精确闭包，以及 reported evidence interval 与 decision availability 分离的双时钟；
- IFCN 六项各自四态并逐项绑定 measurement/waveform/background/QC，而非六个 host-supplied boolean；
- onset/offset 的 observed/censored/not-observed/indeterminate 原生状态；
- capability receipt 的目标域、患者数、precision 下界、coverage、montage/带宽和 held-out 指标。

v1 的兼容层现已额外要求：任何 lead/electrode/region/laterality Top-k 候选至少绑定一个 `spatial_field` 或 earliest `spatial_recruitment` anchor；spectral/rhythm/morphology/high-frequency atom 可以补充描述和置信信息，但不能独立创建空间 SOZ 候选。v2 已将该规则改写为显式 target-relative evidence edge，并要求 onset 正证据绑定 causal view。原生确定性 v2 producer 已能输出四态 opportunity、逐 unit involvement、partial order、target-relative relations 与研究性头皮起始候选；未获术语资格的 morphology/HFO 等 family 保守输出 `not_evaluable`，不会伪造 spike、IED、ACNS evolution 或临床确诊。v1→v2 迁移不能恢复 causal/offline view role、S0--S3 posterior、插补来源、扩展输入防火墙、资格指标或 raw-sample dependency，因此对应字段只能写为 `unknown/not_evaluable/null/[]`，记录明确 loss code，清空定位支持并禁止 report-eligible 术语；迁移器不会把 legacy 字段猜成 v2 真值。

v2 现已把复合 pattern 提升为一等公民：Finding 可携带 nullable `pattern_instance_id`，顶层 `pattern_candidates[]` 逐候选保存 term、状态、required atoms、counterevidence、source-domain scope 和 qualification-rule receipt。同一个 physical instance 可保留多个竞争 `model_candidate`；实例的 atom ownership 是其候选 required-atom 集合的并集。runtime 对悬空/跨实例 atom、required/counterevidence 重叠、course/offline 原子伪装成 onset-causal、四态机会和 report-eligible receipt 逐项 fail closed。旧 v2 若完全没有新字段仅补空数组/null，并保持旧 source-binding digest 兼容；legacy migration 不能创造 pattern 候选。

这仍不等于 v2/v3 已进入私有报告生产。现有 signal-only synthetic EDF 纵向测试证明 canonical→adaptive window→原生确定性 v2 Findings→event ledger，以及逐 measurement/waveform/atom raw-dependency 与 evidence/decision 双时钟的软件合同可执行；它不证明临床术语模型、BA-IEG、真实患者性能或报告路由已经完成。生产级术语/能力 registry、mode-aware v3 record aggregator、raw closure 向 production event artifact/claim/report manifest 的哈希透传与私有批处理接线仍缺失，因此 production profile 继续 fail closed。

### 4.8 additive `event_eeg_findings_v3`：补齐数量、竞争假设与事件结局，但不冒充原生模型

v3 不复制或放松 v2。运行时先剥离新增块、精确恢复 v2 projection 并完整执行 v2 的 EEG-only、机会、receipt、raw dependency 和 onset-causal 校验，然后才验证新增块。每个 occurrence roster、可评价秒数、rate、burden interval union 与 within-event variability 都绑定同一 term 或显式 composite pattern required atoms；交换另一个 term 的 evidence 必须失败。rhythmicity 与 periodicity 不得共享 Finding、term decision 或临床资格 receipt。HFO、DC shift、LVFA 等 acquisition-sensitive term 必须先通过各自物理机会门；例如有效上限带宽不高于 80 Hz 时不能将 HFO 写为 evaluable。

active EEG-only 事件结局显式区分 `qualified_electrographic_seizure | qualified_electrographic_event | candidate_only | candidate_not_qualified_as_a_scalp_electrographic_ictal_pattern_within_queried_support | obscured_by_artifact | not_possible_to_determine`。合格事件必须绑定 selected、supported 的 cerebral-ictal 竞争假设；“候选未达模式资格”只陈述冻结 query/opportunity 下的资格失败，并需要相应 sensitivity receipt，不能转写为“临床发作无脑电改变”；“伪迹遮蔽”必须绑定与事件窗重叠的伪迹；“无法判断”不能转写为阴性。legacy v3 wire `no_demonstrable_scalp_ictal_change` 只作历史只读并在 active adapter 中降级。positive scalp-onset/SOZ 路径仍只有 v2 继承的 `F_onset_causal` evidence edge，新增 differential 或 outcome 块不能从 offline/future evidence 创造空间候选。

当前 v3 原生 clinical producer 尚未实现；v2→v3 migrator 只保存完整 base projection，并把旧 schema 无法恢复的 rhythm kind、数量/负荷/变异、采集能力和竞争假设显式置为不可评价。因此软件层已补齐目标本体，不等于真实波形的术语识别、资格化或 record-level 跨事件 variability 已完成。

### 4.9 Findings denominator：source-accounting shadow 与独立 item-scope v1

`event_eeg_findings_v2/v3` 对已发射对象有严格约束，但仅有 `minItems` 和引用闭包仍不能发现整个 morphology family、某一预期导联或某类评价机会被上游静默删除。为使 precision、grounding 和报告完整性拥有固定分母，每个事件必须附一个独立、内容寻址的 `event_findings_atom_roster_v1` sidecar；不修改冻结 v2/v3 wire，也不让 payload 自己声明“我本来只打算评价这些”。

roster 使用两层闭合：

1. **固定 core slots：** 每个事件都必须逐项给出状态，覆盖 Q、C1、C2、C3 的预定义临床问题；允许 `not_evaluable` 或显式 technical failure，但不允许缺行。
2. **条件 child rosters：** 发作间期 IED、发作期 sharp-contoured component、rhythmic run、periodic element、composite pattern、evolution transition、spatial involvement、artifact interval，以及 HFO/DC shift/LVFA/极慢活动四个彼此独立的 acquisition-sensitive gate，只在各自 opportunity/enumerator 激活后逐实例闭合，并保存去重分母。

四个概念必须分开：

```text
structural scope: event_mandatory | unit_mandatory | instance_dependent | acquisition_sensitive
activation_status: always_expected | triggered | not_triggered | trigger_not_evaluable
finding_status: present | absent_with_opportunity | uncertain | not_evaluable
processing_disposition: completed | technical_failure
```

因此“候选模型没有输出实例”既不等于 `absent_with_opportunity`，producer 异常也不等于信号本身不可评价。阴性必须同时绑定完整的时间/空间 evaluation opportunity 和目标域 sensitivity receipt；缺任一项只能是 uncertain/not-evaluable。对 `unit_mandatory` 项，分母来自 host-trusted canonical expected-unit roster，缺导也要以 `not_evaluable + missing reason` 出现，不能只登记模型选中的活跃通道。每个 core/child key 恰好一行，结构缺失、额外行、重复行、source/policy/hash 漂移均 fail closed。

2026-08-23 的反例审计仍确认 `event_findings_atom_roster_v1` 只能称为 `source-accounting shadow`：它读取同一 v3 payload 自带的 opportunities/instances，机会区间使用 convex hull，也没有唯一 primary-slot/允许 secondary-slot binding ledger。它只证明“已注册 core/child key 与给定 source 行可确定性重放”，不能独立证明上游没有静默删除 expected unit 或 opportunity。

在该审计后，已新增与 Findings payload 完全分离的 `clinical_eeg_event_findings_denominator_v1`。它由 host-trusted EEG scope inventory 在 producer 运行前枚举，API 不接收 Findings、payload opportunities、pattern candidates、事件结局、报告、annotation、Excel、医生标签或临床文本。冻结 policy 将 `28 core + 12 child` 恰好映射到六类物理 scope；每事件形成 `29 + 11N` 个 cell，其中 `N` 是 typed canonical physical units 数。event-global 使用 `event:GLOBAL`，物理单元保留 `lead:*` 与 `electrode:*` 类型，二者同名也不能碰撞。

每个 cell 保存 required/evaluable 的 recording-relative 半开区间 union、秒数、覆盖率、capability/sensitivity/failure/source lineage、`sufficient | limited | not_evaluable` 和自哈希。overlap/touch 合并但真实 QC gap 保留；例如 `[0,5)∪[7,10)` 仍为两个 segment 和 8 秒，不再变成 10 秒凸包。缺 capability、缺 scope、缺 unit、technical failure 或空背景不能删 cell，只能 fail closed 或保留为 `not_evaluable`。receipt 绑定固定 policy SHA、当前 enumerator code SHA、event-scope receipt、unit-inventory receipt、可选 failure ledger、EEG-only firewall 和整个 trusted source-inventory SHA；validator 使用同一 host inventory exact replay，跨 event/record/signal 换包与重算外层哈希均不能通过。

这里的 exact replay 必须准确表述为 **host-trusted source-inventory exact replay**，还不是对底层 canonical/QC/capability/sensitivity receipt 内容逐一重新计算。v1 也是 **roster item × typed unit × interval-union** 的结构分母，而不是逐 `term_query` 分母；其中 `absence_authorized` 只允许 atom-roster 的结构四态账本，receipt 固定 `clinical_term_absence_authorized=false`、`clinical_correctness_claimed=false`。v1 单独不能生成临床阴性或晋升报告术语。

在 v1 之上现已实现 `clinical_eeg_event_findings_term_query_denominator_v2`。它冻结 `34` 个 canonical term 和 `41` 个 operational query；query key 同时固定 term、claim kind、temporal context、typed physical unit、scope、view、reference 与 bandwidth，不再允许一个 item-level sensitivity 被 spike、sharp wave、IED 或不同 evolution 轴共同借用。枚举器不接收 Findings candidate；未实现 query 仍保留为 `not_evaluable`。capability/sensitivity receipt 必须逐 query cell 完整匹配，时间机会保留真实 gap，并从已验证 v1 source inventory/receipt exact replay。当前 `8` 个 query 是 candidate shadow、`5` 个 measurement shadow、`3` 个 structural shadow、`25` 个未实现；所有 query 的 `report_promotion_authorized=false`。因此该分母只是未来阴性判断的必要机会条件，本身不产生 `absent_with_opportunity`、临床正确性或正文资格。

同时已实现 `clinical_eeg_event_finding_binding_ledger_v1`。它要求每个 source Finding 恰好一个 primary disposition、每个独立 query cell 恰好一条 binding row，且 `status` 与 `assertion_level` 必须来自同一条 primary Finding；secondary link 只能保存受控 provenance，不能增加 vote、TP、recall、burden 或 promotion。另用 physical-instance inventory 将同一物理 occurrence 与多个竞争 semantic candidate 分开，阻止多参考重复计票、semantic clone、multi-primary、跨 signal、跨 domain capability laundering，以及把 technical failure 拼成临床 absence。ledger 对 v3 Findings、atom roster、item denominator/join、term-query denominator、physical instance 与 producer inventory 做 host-side exact replay；它仍只是结构绑定 shadow，不证明上游术语判断正确。

审计同时冻结了四个 promotion 前置条件：① actual `provenance.inference_exclusions` 与 EEG-only firewall 的内容哈希绑定；② report-eligible Finding 恰好一个冻结 primary binding，unbound/ambiguous 一律不能晋升；③ onset claim 同时具备 present causal onset atom、qualified event outcome 和 selected supported cerebral-ictal hypothesis；④ expected-unit inventory、technical-failure ledger 和 independent opportunity denominator 均由带 lineage 的可信 receipt 提供。当前 item-scope denominator、term-query denominator 与 Finding binding ledger 均已实现为 public/synthetic shadow；底层 typed receipt registry 的内容重放、`trusted_event_evidence_v3` promotion gate、真实 clinical heads 的患者隔离资格化及 production bundle 接线仍未完成。

因此 atom-roster、item-scope denominator、term-query denominator 或 binding ledger 均不得单独满足报告 promotion。实验冻结门要求实际 typed receipt 内容重放和完整 trusted-event bundle，而非只登记 receipt kind 字符串；`production_bundle_gate_connected=false`、`report_eligible_term_allowlist=[]` 保持不变。当前实现可以证明“注册的问题、单元、逐术语查询和 primary ownership 均有账”，仍不证明回答临床正确，也不要求正文逐项倾倒。正文继续只选择 Critical/Major 阳性、结论相关反证与会改变定位分辨率的 limitation，Supporting 数值主要进入结构化 sidecar/波形附件。

## 5. 金标准不足时的可训练方案

### 5.1 不再训练三个不可分解的 concept classifier

历史实现中的 `M/I/V` 分别是 morphology、scalp-visible ictal involvement 和 observable change/evolution；现有审计没有把三者全部资格化成功，因此不能把新方案写成旧三分支已经恢复。这里把它们拆成原子证据后，重新组织为三个**阅读组**：

- **形态阅读组：** duration、phase、sharpness、slope、物理幅度、field support；
- **时间--频谱演变阅读组：** frequency、bandpower、rhythmicity、entropy、change point 和 `S0/S1/S2/S3` trajectory；
- **空间场--随后累及阅读组：** 多导联 field、reference stability、per-unit onset interval、partial order、laterality/region coverage。

最终呈现采用 `3 + Q`，但它们不是四个标量分类器，也不是 IFCN、SCORE 或 ACNS 已发布的官方四分类本体；这是本项目为了自动化测量、审阅和消融而提出的工程组织框架：

| 阅读栏目 | 原子组成 |
|---|---|
| C1 Phenomenology／看到了什么 | morphology primitive、frequency/amplitude、分别资格化的 rhythm 与 periodicity、occurrence/burden |
| C2 Spatiotemporal onset／何时何地首次可见 | pattern-specific emergence interval、spatial field、polarity、reference stability、earliest distinguishable set |
| C3 Course／随后如何变化 | frequency/morphology/location evolution、sequential involvement、termination/recovery |
| Q Evaluability／能否评价 | quality、bandwidth、coverage、artifact counter-hypothesis、censoring、provenance |

模型训练在原子测量、区间、候选术语和集合标签上完成；三个组只负责报告组织、专家审阅和消融。这避免用一个粗标签同时掩盖检测、形态、空间和时间错误。`present | absent_with_opportunity | uncertain | not_evaluable` 四态、`measured | model_candidate | report_eligible_automated` 三级 assertion、causal/offline evidence role、raw-sample dependency 与 qualification receipt 同样是本项目的可信 AI 扩展，不应写成指南原生字段。

其中旧 `I` 被拆为可见发作支持、空间场、质量与可评价性，旧 `V` 被拆为频率/形态/位置轨迹和区间关系。任何历史上未通过原生任务门的 M/I/V logit 都不能直接接回新 reasoner。

### 5.2 三级监督

#### Level 1：无需临床标签的确定性测量

冻结 DSP/统计 producer，输出频率、波幅、line length、谱熵、autocorrelation、cycle consistency、change point、field extent 和 lead-lag interval。通过合成信号、人工注入波形和物理重放测试验证数值，不把阈值越界自动命名为临床术语。

#### Level 2：自监督与弱监督候选

- masked time/time-frequency modeling、相邻时间排序、background--emergence contrast；临床 EEG 自监督工作（[Banville et al. 2021](https://doi.org/10.1088/1741-2552/abca18)）只支持“未标注 EEG 可学习可迁移表征”，不把 latent 当临床术语资格；
- 同一物理区间的跨 view 一致性、channel dropout、左右镜像等变性；
- TUSZ 仅监督 seizure interval、检测和弱边界；
- TUEV 的 SPSW/PLED/GPED 仅作选中 TCP 双极 edge 区间的 morphology/periodic candidate，未标区不是阴性，也不能把 edge 标签拆给一个或两个物理电极；历史 PLED/GPED code 保留在 lineage，正文只有经过目标域资格后才使用当前 LPD/GPD 术语；
- TUAR/TUEV artifact 只训练质量门；
- IIIC/HMS 可训练 rhythmic/periodic candidate 与专家分歧，但 ICU 域不直接资格化目标域 ictal onset；
- [Lin et al. 2025 开放 IED 空间数据](https://doi.org/10.1038/s41597-025-04572-1)可训练 IED morphology/粗区域候选，不提供 ictal evolution/SOZ；
- [SpikeNet2](https://doi.org/10.1056/aioa2401221)若能获得内容可审计且与评价患者无重叠的 artifact，可作冻结 spike-candidate teacher/comparator，并迁移其多尺度 hard-negative mining；其输出仍只能是间期 spike `model_candidate`，不得授权发作期演变或 SOZ；
- DeepSOZ 只用于 patient/record positive-set/partial-label MIL 的头皮通道集合，不提供逐事件 Findings 或皮层 SOZ。未标通道不是负例；若损失仅把概率质量集中到已知阳性集合、同时压低其余通道，必须准确称为 incomplete-positive weak ranking objective，不能冒称 PU 学习。主方法应显式建模标注缺失机制，采用 partial-label/PU 或只对 verified negative 建 pairwise 约束，并做 `unlabeled-as-negative / positive-mass / PU-partial-label` 消融。

近期 [EEG-to-Report 标注框架](https://doi.org/10.21203/rs.3.rs-10452606/v1)可用于降低专家采集 segment/channel/feature--text 对齐数据的工程成本，但不能把其自动计算特征或 pilot 文本当真值。本项目导出的监督单元仍是原子 interval/field/partial-order/四态及其 raw dependency，而不是未经证据拆分的自由文本。

弱标签不能直接多数投票。对每个 term 单独建立 source-development-only probabilistic label model，显式登记 labeling function 的来源域、目标字段、相关性/冲突和 evaluation opportunity；未标区间继续是 unknown，而不是自动负例。其后验及熵只监督 `model_candidate` noisy student，并与原始连续 measurement、跨参考一致性和少量专家 anchor 联合训练。只有每个 term 至少有三个语义非冗余 labeling functions、candidate-blind 随机专家 anchor、patient-level cross-fit，并通过预注册的可识别性、LF 相关性与整体标签翻转检查时，概率 label model 才可启用。必须预注册 `deterministic rule / simple vote / dependency-aware label model / noisy student / expert-only` 消融；若 label model 在患者隔离开发集不能优于简单规则或校准更差，就删除该层。任何 private/Excel/医生结论均不得成为 labeling function。

LLM、attention、VQ code 和动态图边不能生成临床伪真值。Qwen 可辅助把已有人审文本整理成待核对结构，但不能作为 qualification annotator。

#### Level 3：小规模目标域双专家资格集

不需要逐 patch 标完整金标准；需要对“哪些术语能进入报告”建立严格资格集：

1. 以患者为最高拆分单位，建立 **candidate-blind 随机核心集**：按 `patient → full recording → time×channel opportunity` 抽样，包含 detector-negative 区间；不能先按模型候选事件筛选再称随机；
2. 建立**完整记录复核子集**：对抽中的整条记录穷尽标注事件 roster；若要监督多模式推理，还必须显式标 event→mode assignment 与逐事件 onset-field；
3. 另建不用于自然患病率/总体性能直接估计的**主动学习补充集**：纳入全部模型阳性并富集罕见术语、模型分歧、reference instability、镜像不一致和多事件冲突；
4. 三框均保存 patient、sampling stratum 和 selection probability；随机核心/完整记录复核提供总体分母，主动集单独报告或按预注册 design-based/inverse-probability 方法分析，不能直接混池；
5. 两名 EEG 专家独立标 interval/field/partial order，并显式标 `present / absent_with_opportunity / uncertain / not_evaluable`；阴性必须同时标出实际存在的 candidate-blind evaluation opportunity；
6. 专家不可见 private、Excel、DeepSOZ reference、模型是否命中或其他事件的医生结论；
7. 先做小型 pilot 估计术语发生率和一致性，再按患者聚类置信区间、阳性 precision 下界以及阴性 sensitivity/NPV/漏检风险下界决定正式样本量，不把一个任意事件数写成充分；
8. 对每个术语分别冻结 eligible views、带宽、最小时间/电场支持、positive precision gate、negative sensitivity/NPV gate、uncertainty policy 和 receipt hash；receipt 同时记录目标域、参考标准、患者数、coverage 和 held-out 结果；
9. 未达到预注册资格下界的术语保留为数值或 `model_candidate`，不进入正文。

主动学习样本提高训练效率，但不能单独估计自然队列性能，也不能与随机核心直接合并后计算普通 patient-macro 指标。资格数据还需分为 threshold/calibration development 与 patient-disjoint held-out qualification；保留双专家原始四态、区间和 field 分布，必要时第三人仲裁，但不得用仲裁共识抹去观察者不确定性。逐术语同时报告 AC1/kappa、分歧率、对每位专家及共识的 sensitivity analysis；样本量按术语 prevalence 与 patient-cluster CI 决定，任意固定事件数只能作为 pilot 初值。正式资格与最终 source-eval 必须保留未参与选样、调参的患者。

TUSZ、TUEV、TUAR、TUAB 与 DeepSOZ 共享 TUH/TUEG 系谱；跨语料预训练、资格化和评价前必须联合使用稳定 patient ledger、原始 content hash 与抗重采样/裁剪的 waveform fingerprint 去重，不能仅相信 corpus 名称不同或 exact hash 不同就认为患者隔离。

公开 EEG foundation checkpoint 也可能已在 TUH/TUAB/TUEV 等信号上预训练。若其训练清单不能证明与 DeepSOZ/TUSZ source-eval 患者和记录无重叠，该 checkpoint 只能进入探索性 backbone 对照，不能作为无泄漏主结果。主结果优先使用可审计的 source-development-only 自监督预训练，并将 source-eval 从表示学习、归一化统计、codebook、hard-example mining 和模型选择中全部封存。

### 5.3 推荐模型：BA-IEG

```text
canonical multi-view EEG
  -> detector candidate + S0/S1/S2/S3 computational-state posterior
  -> asymmetric variable acquisition and background bank
  -> native fine proposals + coarse/context physical-time tokens
  -> deterministic measurement bank
  -> continuous local time-frequency encoder
  -> sparse physical spatiotemporal graph
  -> K legal boundary-path phase pooling
  -> atomic measurement/candidate heads
  -> term-specific clinical qualification gates
  -> event_eeg_findings_v3 EvidenceGraph
  -> latent heterogeneity clustering across events
     (supervised mode-aware MIL only after event→mode + event-onset-field gold)
  -> record SOZ hypothesis/claim graph
  -> constrained Qwen graph-to-text + deterministic fallback
```

局部 encoder 可吸收 TFM/CBraMod 的时频 patch 和自监督机制；TimeFilter 只启发 temporal/spatial/spatiotemporal relation router。主图使用 10--20 物理邻接和少量质量门控动态边，不能把 learned edge 称为传播。长事件以真实物理时间、变 token 密度编码，不做 time-warp。

### 5.4 可投稿的创新假设

以下可作为待验证创新，而不是已证实结论：

1. canonical physical evidence root + task-native views，同时保持 detector 性能和 Findings 物理真实性；
2. 边界后验驱动的非对称取窗与 token 分辨率分配，而不是固定 60/120/300 s；
3. `S0/S1/S2/S3` 计算状态路径边缘化，将边界不确定性传播到每条 Finding、field-involvement 边和 SOZ 候选，并由独立 qualification 决定是否可使用 ictal/onset/recovery 术语；
4. `measured → candidate → qualified` 的逐术语选择性生成；
5. 多参考电场一致性 + onset interval partial order，区分最早可见场与晚期高幅扩散；
6. event→mode + 逐事件 onset-field 专家 gold 闭合后的 mode-aware 多事件 MIL；在此之前创新主张限于可审计的 latent event-heterogeneity stability，不把不同发作虚假平均；
7. claim-level evidence locking，使信号证据、研究推理和语言表面可分别评价。

## 6. 报告事实一致性：不能只看 SOZ Top-1

### 6.0 两条不可循环替代的事实轴

事实评价先严格拆成：

1. **serialization fidelity：** Qwen/确定性 renderer 是否忠实序列化冻结 EvidenceGraph；这可对全部报告自动评价。
2. **evidence correctness：** EvidenceGraph、SOZ 假设和报告是否由原始 EEG 支持，并与独立专家/冻结参考一致；这需要信号重放和患者隔离的参考子集。

报告可能 100% 忠实复述一个错误的上游 Finding，也可能偶然命中 SOZ Top-1 但引用了 late spread 或错误事件。两者均不能称事实一致。语言风格是与两条事实轴正交的评价维度，只在有同一 EEG 配对全文时评价；给定的单份真实报告只作结构/措辞样例，不能作为其他患者的 BLEU 参考。

当前 141 份私有报告的既有语言审计正好说明为什么必须分层：`private_clinical_eeg_language_quality_v2_3_20260820.json` 中 ledger 内已登记 claim 的 micro grounding precision 为 `500/500=1.0`，但 claim 覆盖报告正文字符的比例仅均值 `12.26%`、中位数 `13.43%`、范围 `0--28.27%`；完整配对医生报告数为 0，冻结后空间可比较字段数也为 0。因此它只能证明“少量显式登记的 claim 没有引用未知 fact ID”，不能证明整篇报告、上游 Findings 或 SOZ 结论事实一致，更不能替代本节 L0--L6 评价。

### 6.1 七层评价

| 层 | 核心问题 | 主要指标 |
|---|---|---|
| L0 事件发现完整性 | 是否找到并正确拆并整条记录中的发作 | event sensitivity、FA/24h、完整发作覆盖率、onset/offset coverage-width、merge/split error、漏报事件数 |
| L1 输入防火墙 | 是否只用了当前 EEG | annotation/Excel/clinical input count=0；标签扰动后输出 byte-identical；signal/view receipt 完整率 |
| L2 信号→测量 | 数值能否从波形重放 | frequency/amplitude error、interval IoU/coverage-width、unit/time/hash mismatch、reference perturbation stability |
| L3 测量→Finding | 临床词是否被测量支持 | 当前先报 per-family positive precision/coverage、qualified precision、越级率和 not-evaluable calibration；只有显式 evaluation opportunity/negative gold 完成后才报 recall/F1 |
| L4 事件→记录假设 | 多事件推理和头皮起始排序是否合理 | laterality/region/phenotype 分层结果；穷尽逐电极 gold 下的 PR-AUC/set F1/mAP/calibration；incomplete-positive 参考下仅 annotated-positive rank、Hit/recall@k、MRR、positive mass 与 PU/缺失标签敏感性；无 event→mode gold 时 mode 只报稳定性；另报 onset--late-involvement confusion 与 resolution--risk |
| L5 claim graph→文本 | 文字是否忠实于图、推理链是否完整 | `LinkageClosure`、supported-claim precision、salient-claim recall、relation F1、event/mode attribution、numeric/time/negation/epistemic exactness、`ChainPrecision`、`SalientChainRecall`、contradiction/omission |
| L6 post-hoc descriptive / fresh confirmatory | 是否与医生参考一致并可审阅 | 已历史开标私有 141 只报 locked descriptive Excel laterality/region/unclear 与 incomplete-positive channel agreement；fresh patient/site 或从未打开 holdout 才报 confirmatory 双医师 major/minor error、编辑时间、完整性/可读性/定位效用 |

L3 和 L5 都必须各报两个不可互相替代的 recall。L3 的 **module-conditional recall** 只在 reference-matched event 上评价具体 Finding head，用于定位模块错误；**end-to-end Finding recall** 的分母则是连续长记录独立审阅得到的全部 gold salient atoms，detector 漏检、窗口未覆盖、技术失败和整类 Finding 未发射均记为漏检。L5 的 **serialization recall** 以 Qwen 前冻结且哈希化的 `render_required_claims` 为分母；**clinical salient-claim recall** 以专家在看不到模型输出时标出的 Critical/Major claims 为分母。空 EvidenceGraph 或只生成少量安全句不能因此得到高分。

建议将四态 Finding 评价为机会感知的分层任务，而不是压成单一二分类：`present` gold 计算 type/interval/field 的 precision--recall；`absent_with_opportunity` 只在独立 opportunity 与 sensitivity 充分的分母上计算 specificity/NPV；`uncertain` 和 `not_evaluable` 单列四态混淆矩阵、coverage、selective risk/AURC 与校准。结构 roster closure 是进入这些指标的前置门，不作为临床正确性的加分项。

所有统计以患者为最高 bootstrap/拆分单位；发作多的患者不能获得更高隐式权重。

除总体 patient-macro 外，必须按 site/device、采样率/有效带宽、montage/reference class、QC burden、事件数和事件 mode 报告预注册 worst-group performance 与 patient-cluster CI；否则总体均值可能掩盖某一采集条件下的系统性错误局灶化或错误安心结论。

原子事实匹配不能只比较术语字符串。预测 claim 与专家/冻结参考 claim 至少在 `polarity + event/mode + physical time interval + unit/region + concept + attribute/value + epistemic status` 上相容；时间、频率和波幅使用预注册容差，空间误差同时报告 10--20 邻接图距离。匹配函数按 claim type 冻结：允许合法的 `electrode→region→laterality→phenotype` 层级回退，以 ontology distance 计部分匹配；删失/不确定时间以 interval distribution 的 coverage 与 sharpness 评价，而非只做点值 exact match。先构造 `Match(p,g)∈[0,1]`，再以 Hungarian/最大权匹配得到一对一集合 `A*`，避免多条重复句子反复匹配同一事实而虚增得分。借鉴 [FActScore](https://doi.org/10.18653/v1/2023.emnlp-main.741) 的分解思想，可报告 severity-weighted factual precision/recall，但证据库必须是本地冻结的 EEG EvidenceGraph，而非通用知识库或另一份自由文本：

\[
P_w=\frac{\sum_{(p,g)\in A^*}w_p Match(p,g)}{\sum_{p\in Pred}w_p},\qquad
R_w=\frac{\sum_{(p,g)\in A^*}w_g Match(p,g)}{\sum_{g\in Gold}w_g}.
\]

可预注册 `EEG-ClaimGround` 作为 EEG 特有的语义 grounding，而不是复用胸片 box：

\[
G(p,e)=I_{event/mode}\cdot IoU_{time}\cdot Jaccard_{channel/region}
\cdot I_{authorized\ temporal\ role}.
\]

SOZ 正支持的最后一项只有在 `onset_support_eligible=true`、`future_sample_access=false` 时为 1；offline context 只可 grounding 到演变、后续累及、终止或反证。波形附件需使用同一 claim、事件、相对时间、通道集合和图像哈希闭合。

该最小 evaluator 已于 2026-08-22 实现：每个 aligned claim 输出 event/mode exact match、recording-relative temporal IoU、channel/region set Jaccard、逐 evidence 时间授权和乘积 grounding score。正 onset/SOZ 的 predicted claim 必须提供完整 per-evidence binding，且同时满足 onset-causal、future-free、onset-authorized 与 onset-support-eligible，才能获得 strict support；legacy role-only claim 可继续解析和 Hungarian 对齐，但 `authorization_complete=false`，不能得到严格事实支持。reference legacy gold 暂不强迫完整绑定，以保留迁移入口；结果必须显式披露这一不对称。

canonical evidence-ID exact set equality 只用于 `renderer output ↔ 同一 frozen claim plan`，名称固定为 `LinkageClosure`。它证明序列化/来源闭合，不证明 raw EEG→Finding 的临床正确性；独立专家可能选择临床等价但 ID 不同的 evidence span，因此 evidence correctness 必须按 interval、unit/region、value、polarity、temporal role 和 ontology relation 做一对一语义匹配，不能要求 evidence-ID 相等。

同时公开不加权 precision/recall/F1、`hallucination=1-P` 和 `omission=1-R`；严重性权重必须由盲法专家在 source-development 上冻结。安全约束不参与加权平均：forbidden input count、imputed/QC-fail evidence count 和 unsupported critical claim count 的目标均为 0。

L0 的事件匹配同样必须一对一。可用 `C(p,g)=λ(1-IoU)+(1-λ)min(d_onset/τ,1)` 构造成本，再做 Hungarian matching；医生只给 onset 区间时，预测落在区间内的 `d_onset=0`。必须同时报告 event sensitivity、FA/h、onset signed/absolute/P90 error、duration IoU 与 merge/split/duplicate error，不能只在“成功进入 Findings 的候选”上计算条件性能。

记录级结论还需双向评价推理链：

\[
ChainPrecision=\frac{\sum_{d\in D_{pred}}w_d I(\text{premises supported and rule allowed})}{\sum_{d\in D_{pred}}w_d}.
\]

另以独立专家/reference 中预先定义的显著链为分母计算 `SalientChainRecall`。零输出、漏掉关键反证或漏掉整次事件在 recall 中记 0。两者共同检查 SOZ/广泛双侧近同步头皮表型/有资格的多模式结论是否只引用 onset-eligible 证据、是否错误使用 late spread、是否跨事件或 mode 混接；一个偶然命中医生 Top-1 的结论若证据链无效，仍记为推理错误。旧实现字段 `ChainValidity` 只能作为 predicted-chain precision 的兼容别名，不能单独作为完整推理事实性结论。

不应同时宣布五个未校正的共同主终点。预注册采用 gate + 最多三个 confirmatory primary 的层级，并为 family-wise error 做 Holm 等校正：固定 FA/h 的 detection 先作上游资格门；主终点可分别覆盖同预算 adaptive-window benefit、独立专家支持的 evidence-to-report precision/recall，以及有适配且穷尽 gold 的记录级 phenotype/topography。PR-AUC、set 指标、Brier/ECE/AURC 和 prediction-set coverage 只有相应 reference 穷尽且正负/不确定状态可识别时才能进入确认性分析；在现有 incomplete-positive 医生通道集上只能使用 annotated-positive rank/Hit/recall@k/MRR/positive mass 与 PU/标签缺失敏感性。Top-1、BLEU、ROUGE 和 BERTScore 均为次要指标。诊断准确性研究的 reference-standard、拆分和透明报告按 [STARD-AI 2025](https://doi.org/10.1038/s41591-025-03953-8) 审计。

### 6.2 Claim graph 是文本事实性的主合同

每个句子先由结构化 claim plan 决定：

```json
{
  "claim_id": "C-001",
  "claim_kind": "observation | event_inference | mode_inference | record_hypothesis",
  "subject": {"type": "eeg_event", "id": "EV-01"},
  "predicate": "earliest_sustained_change_maximal_at",
  "object_or_value": {"region": "left_temporal", "electrodes": ["T7", "P7"]},
  "event_id": "EV-01",
  "mode_id": "MODE-A",
  "time_interval": {"lower": 622.4, "upper": 624.1},
  "epistemic_status": "report_eligible_automated | research_ai_hypothesis",
  "evidence_ids": ["E-031", "E-032"],
  "allowed_surface_frames": ["event_onset_maximal_at_v1"]
}
```

“随后”“早于”“传至”“同步”本身也是关系 claim；仅同时存在两个合法实体不能授权 Qwen 创建关系。文本 validator 应逐 claim 核对实体、谓词、事件/mode、时间、否定、认识状态和 evidence IDs。Qwen 失败或越权时使用同一 sentence plan 的确定性 lexicalizer，保证每条可读取记录仍有报告。

### 6.3 SOZ 评价应是层级和风险敏感的

指标必须按 reference 完整性分轨，而不是把“未标”当作负例：

- **穷尽逐电极双专家 gold：** hard onset 可报 PR-AUC、set Jaccard/F1、mAP、nDCG@k、Hit@3/5 与 MRR；soft spread 独立报 graded nDCG/weighted AP；
- **现有 incomplete-positive 医生显著通道：** 只报 annotated-positive rank、Hit/recall@k、MRR、positive probability mass 与 PU/缺失标签敏感性分析；未标通道保持 unknown，因此禁止 FP-dependent PR-AUC、set Jaccard/F1、mAP、Brier/ECE 和 prediction-set coverage；
- 医生 spread channel 是独立且也可能不完整的 graded relevance，不能与显著通道合并成穷尽 hard/soft gold；
- laterality、region、channel 的层级守恒；
- neighbor、同侧远端、对侧远端错误；
- localized/lateralized scalp-visible、widespread bilateral near-synchronous、bilateral-ambiguous、nonlocalizable 表型；multiple-mode 只有 event→mode + 逐事件 onset-field gold 闭合后评价，否则只报 latent heterogeneity stability；
- Brier/NLL/ECE、risk--coverage/AURC 与 prediction-set coverage/size 仅在相应层级 reference 穷尽且有合法 calibration receipt 时报告；否则只称 score/rank；
- electrode→region→hemisphere→phenotype 的 resolution--risk 曲线；
- leave-one-event-out、复制事件、打乱事件顺序和跨 montage 稳定性。

Excel“起始”字段先盲法映射成 laterality/region/unclear closed vocabulary，只作报告冻结后的描述性参考一致性。它是医生事后总结参考而非不可错的事实真值，也不是完整参考报告，不能拿来计算 BLEU。私有 141 及其标签已经历史开标，即使新模型、阈值、prompt 和模板锁定，也只能称 `post-hoc locked descriptive audit`，不得称 independent external/confirmatory validation；确认性结果必须来自 fresh patient/site 或从未打开的预注册患者级 holdout。

### 6.4 语言质量与事实性分开报告

- 只有同一 EEG、完整配对、去标识化的医生报告才可计算 corpus BLEU-1--4、ROUGE-L、METEOR 和本地可审计 BERTScore；
- EEGGraph claim/relation F1、unsupported claim 和 contradiction 才评价事实；
- BLEU/ROUGE 低可能只是医生措辞不同，高也可能复制了错误或无关模板；
- 两名 EEG 医师盲评事实正确性、遗漏、过度断言、不确定性表达、临床风格、修改量和审核时间；
- 机器合同违规数目标应为 0，但不能据此声称临床幻觉风险为 0。

### 6.5 必做反事实

1. 修改 EDF annotation、Excel 或医生标签，报告和 hypothesis hashes 不变；
2. 左右镜像信号后 laterality/electrode 排序相应镜像；
3. 复制同一事件不提高支持率和置信度；
4. 打乱事件顺序不改变 mode-aware 结论；
5. 交换 onset 与 late-involvement 段，随后累及的偏序关系反转或失去资格；
6. 只改变波幅，不能触发 definite evolution；
7. HFO 带宽资格失败时 HFO 自动变 `not_evaluable`，LVFA 按自己的物理低幅/快活动带宽与伪迹门独立判定；
8. 将 observed channel 改为 imputed，相关 Finding 和 SOZ 证据资格归零；
9. 改变 reference 后不稳定的电极结论降到区域/侧别；
10. Qwen 停服、越界或漏 claim，仍生成事实等价的确定性报告。
11. 合格事件或显著证据从 detector→Finding→record graph→文本的端到端 salient-evidence recall 单独记账，不能只在已检测候选上计算高忠实度。

## 7. 预注册实验顺序

1. 完成 canonical/view receipt 与跨视图时间、单位、带宽、mask 验证；
2. 完成 source-development detector operating point，保持 source-eval 封存；
3. 先输出原生 deterministic-only `event_eeg_findings_v2`，验证可重放、边界删失、四态可评价性与 causal/offline 权限；v1 只保留为兼容 baseline；
4. 比较固定 `[-12,+48]`、60/120/300 watchdog 和 `S0/S1/S2/S3` 计算状态自适应窗；
5. 比较 deterministic-only、SSL encoder、弱监督 heads、专家 qualification fusion；
6. 比较单尺度、固定多尺度、自适应 token budget；
7. 比较无/有 reference perturbation、boundary marginalization 和 background bank；
8. 比较单事件、简单平均、reciprocal-rank 和 latent heterogeneity clustering；只有 event→mode + 逐事件 onset-field 专家 gold 闭合后才比较 supervised mode-aware MIL；
9. 比较自由 Qwen、block fact IDs、claim graph direct text、sentence-plan + constrained Qwen 和确定性 lexicalizer；
10. 冻结全部模型、阈值、术语门和模板后，只读打开 source-eval/private 医生参考。

主方法只有在 patient-disjoint 评价中同时改善 Findings/SOZ/事实性或形成更优 accuracy--compute Pareto，且不增加 unsupported clinical claims 时才能晋升。不能用语言模型流畅度补偿信号证据失败。

## 8. 当前实现与缺口

当前已实现的是**机器合同与确定性研究 baseline，而非已验证的临床 Findings 模型**：canonical EEG receipt、时间/单位/带宽/插补/质量及跨视图 validator；同一 mother signal 派生 `findings_native_morphology`、`onset_causal`、`context_offline` 及 referential/TCP/CAR/Laplacian 视图，detector role 被禁止支持临床 Finding。独立 montage/reference observability receipt 已分类共同参考兼容、原生双极、mixed 与 unknown，保存派生矩阵 rank/condition/connectivity/per-output carriers，并把母级 flat/clipping/step/gap 沿所有非零 carrier 传播；它仍是 label-observable shadow 合同，不证明设备接线或真实数据临床资格。adaptive event 到 fine/coarse/context 物理时间 ragged token 的 P0 v3 materializer、canonical/adaptive hash binding、dense deterministic-target sidecar v2 与共享 numerical kernel也已存在。P0 的 nominal tile 在 recording-relative seconds 上唯一，onset-causal/native 与 context-offline/downsampled view 独立向内映射并保存实际 support；不再假设八个 view 共用一个采样时钟。独立 `event_eeg_findings_v2` schema/validator、fail-closed v1→v2 migrator，以及直接消费 canonical/adaptive/task views、从不经过 v1 migrator 的原生确定性 v2 producer也已存在。该 producer 能生成质量、物理幅值、频率、节律度、change-point、reference-specific spatial field、最早/后续 involvement、partial order 与 target-relative relation，并已为 measurement/waveform/atom 建立 raw-sample dependency 闭包和 reported-evidence/decision-available 双时钟；未获资格的 morphology/HFO、阴性断言及临床术语保持关闭。v1 的空间 anchor、时间方向和输入防火墙兼容门仍保留作复现 baseline。

`event_processing_ledger_v2` 已实现冻结 detector roster 的一一闭合，保留成功、跳过、不可评价和技术失败的原始分母，并区分 zero-detector、all-not-evaluable、partial/all-technical-failure 与完整 Findings；后级 stage hash 不能在前级缺失时“复活”。独立 factuality evaluator 已覆盖 Hungarian 原子 claim 一对一匹配、supported precision、salient recall、hallucination/omission、closed-rule predicted-chain `ChainValidity`（论文语义仅为 `ChainPrecision` shadow）、五阶段 evidence attrition、patient-macro 与 patient bootstrap；独立显著链分母的 `SalientChainRecall` 尚未实现。v1 deterministic aggregator/renderer 已有 claim/evidence/waveform 闭包测试，但它尚不是 v2 私有生产报告路由。

新增 source-bound materializer 不再信任调用者填写的 claim、derivation、stage 或权重：它从冻结 EvidenceGraph projection、record hypothesis graph、deterministic sentence ledger 与完整 event roster 重新派生 factuality case，检查每句/每个语义原子分句的唯一 claim ownership，并以带外源重放抵抗同步修改 text+ledger+self-hash 或保留 evidence ID 后篡改观察对象/时间。该能力仍只覆盖 source replay、serialization 和全 roster evidence-flow，不替代双专家 Finding/SOZ 真值。

纵向验证必须分开表述：一组 signal-only synthetic EDF 测试覆盖 canonical→adaptive search/window→原生 v2 Findings→event ledger，并检查不同事件保留不同变长窗口、修改未来信号不回写更早 causal 证据、offline context 不能创建起始正证据、raw closure/双时钟约束，以及单事件不可评价不取消其他事件；另一组 synthetic v1 baseline 测试覆盖 Findings→record claim graph→deterministic renderer。2026-08-22 上一轮完整 `tests/test_clinical_eeg_*.py` 回归为 `810 passed, 2 skipped`；其中 retained-K boundary-path marginalizer 为 `5` 项专项测试，多参考场 primitive 为 `9` 项，source-bound factuality materializer 为 `10` 项且与 claim evaluator/multievent graph/render 的四链交叉回归为 `53 passed`。本轮 montage/quality、baseline/context 和 v3 policy 安全修订后，重新执行的 baseline、montage、Findings v3、deterministic v3、claim factuality、source-bound materializer、canonical signal-view 与 policy 8 个直接相关测试文件共 `86 passed`；Python 编译和 JSON 解析通过，未重新跑完全部历史长耗时套件。上述均是软件回归，不是私有 production、真实患者性能或临床验证；当前仍没有一个测试能够证明“私有真实长程 EDF 已沿 v2 路由生成临床报告”。

联合接线审计后，EDF-container 哈希与 canonical physical-signal 哈希已分开命名并由 `canonical_adaptive_signal_binding_v1` 桥接；缺 `FZ/PZ` 使用零 carrier、证据资格为零且不插值；P0 和原生 v2 producer 都拒绝未绑定或来自另一 canonical EEG 的 search receipt。dense sidecar 的 replayable deterministic targets 是模型监督对象，**不能与 raw-sample dependency provenance 混称为同一种 sidecar**。raw provenance 已从 view/transform 的 FIR support、processing latency、warm-up 和“不前移证据时间”进一步下沉到每条 measurement/waveform，并由 Finding 依赖 ID 精确闭合；双时钟 validator 同时区分 reported evidence、decision availability、FIR processing latency 与 sustained-confirmation latency。capability/term-decision 双回执、registry、claim graph 与 renderer 的现有验证主要是 synthetic fixture 合同，不是训练式临床术语 producer 或患者隔离资格证据。因此真实状态是“模块和合成纵向切片通过测试”，不是 production 闭环。仍缺：

- 已晋升 detector provider 与 canonical root 的 provider-native transform/source binding、冻结 operating point，以及统一连续长程 benchmark 的 sensitivity、FA/h、onset error、RTF 与资源结果；当前启发式 preselector 不得称为 SOTA；
- 将 canonical/adaptive/v2 producer 与 event ledger 接入私有 `_run_one` 主路由，并完成 per-event isolation、zero/all-ineligible report outcome、waveform、claim graph、renderer 和 profile-bound manifest 的内容哈希闭包；
- 将已验证的 raw-dependency/双时钟闭包透传到私有 waveform/claim/report manifest；设备 acquisition phase 通常未知，仍需联合 acquisition+digital-filter+sample-grid 边界不确定性和 causal/offline 跨路 temporal-direction stability gate；
- 将已实现的 event-level 多参考 rank/field stability、earliest-set 与分辨率回退 primitive 接到真实 v2 event/record reasoner；当前工程 gate 不是训练式 field/polarity/phase-reversal head，也没有患者隔离的临床资格；
- 可重放的背景 prototype similarity、IED/context 独立 reasoner、保护区外资格门，以及带 evaluation opportunity 和 sensitivity receipt 的显式 `absent`；
- 将 shadow `event_findings_atom_roster_v1` 升级为独立 denominator：实现可信 `term × unit × interval-union` opportunity roster、Finding primary/secondary binding ledger、EEG-only exclusion hash、canonical montage/failure lineage 和 canonical unit mapping；在此之前它只作 source accounting，不能接入 bundle/claim-plan promotion；
- spike/sharp、IFCN 六项、ACNS evolution/ESz 等**临床术语 signal candidate producer**和患者隔离的 capability/term-decision registry；现有 schema、fixture 和资格规则不等于模型已训练或术语已获临床验证；
- BA-IEG heads、校准后的 `S0/S1/S2/S3`/event qualification、latent heterogeneity stability，以及取得 event→mode + 逐事件 onset-field gold 后的 mode-aware hierarchical MIL/v2 record reasoner 训练和患者隔离验证；
- 目标域双专家术语资格集、source-eval/私有冻结后评价、双医师 reader study，以及语言质量与外部临床正确性的正式评估；
- 私有批处理生产路由的实际切换；`adaptive_event_findings_v2` 非 dry-run 继续 fail closed，避免旁路 adaptive artifact 静默回落到固定 `[-12,+48] s` legacy 报告。

`ADAPTIVE_REPORT_ROUTE_CONNECTED=False`。因此本文档给出的是可执行研究方案，不表示已经达到 SOTA、临床准确度或可直接诊疗使用。
