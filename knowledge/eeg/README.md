# EEG 外部知识系统

本目录保存与具体模型、训练方法和项目数据合同解耦的 EEG / 头皮可见发作起始 / SOZ
通用知识。它提供术语、本体、鉴别规则、推理边界、报告措辞和来源追踪；它不保存患者事实，
也不参与原始 EEG 测量、标签生成、模型打分或金标准构造。

## 活动边界

知识系统只允许承担三类作用：

1. 规定“要观察什么”和“术语如何使用”；
2. 对已经形成且冻结的患者证据做鉴别、边界检查和一般医学解释；
3. 约束报告措辞，并给出规则来源。

知识系统不得：

- 根据典型知识补写当前患者未观察到的波形、临床表现或检查结果；
- 修改事件时间、通道/电极/区域排序、模型概率、标签或医师结论；
- 把双极导联、相位反转、IED、局灶慢化或最早头皮可见改变直接升级为皮层 SOZ/EZ；
- 进入训练/评估的目标侧，或从评估表、医师标签和报告中反向泄漏答案。

## 目录

```text
knowledge/eeg/
├── manifest.json                    活动版本、层级和入口
├── ontology/                        领域概念、语义层与空间层级
├── terminology/                     别名、旧词和高风险表述
├── annotation/                      认识状态与来源语义
├── cards/                           消费端中立的原子知识卡
├── reasoning/                       推理边界、证据角色和 grounding
├── reporting/                       claim 级报告语言策略
├── provenance/                      来源注册与版本优先级
├── schemas/                         机器可读合同
├── knowledge_base.jsonl             兼容现有调用的来源摘要层
└── LearningEEG_..._v1.0.md          历史迁移设计稿，不是活动入口
```

本结构保留原设计建议的职责分层，但使用 JSON/JSONL 而不是 YAML：仓库当前运行时仅依赖
Python 标准库读取 JSON，采用 JSON 可避免新增解析依赖，并能直接做 schema 和 CI 校验。

## 五类对象必须分开

| 对象 | 目录/文件 | 能否成为患者事实 |
|---|---|---|
| 领域概念 | `ontology/`、`terminology/` | 否 |
| 一般判读规则 | `cards/`、`reasoning/` | 否 |
| 文献与数据集语义 | `knowledge_base.jsonl`、`provenance/` | 否 |
| 患者信号证据 | 项目 fact/evidence ledger（本目录之外） | 是，须有证据 ID 与来源 |
| 项目适配与模型策略 | 项目代码、配置和报告（本目录之外） | 不是知识库内容 |

患者 claim 必须同时绑定两条不同的追踪链：

```text
patient claim -> patient evidence IDs
patient claim -> knowledge card IDs -> source IDs
```

第二条链只能解释第一条链，不能替代第一条链。

## 活动入口

- [`manifest.json`](manifest.json)：版本与文件角色的唯一入口；
- [`cards/core_safety_cards.jsonl`](cards/core_safety_cards.jsonl)：首批高风险概念卡；
- [`reasoning/inference_rules.json`](reasoning/inference_rules.json)：OBS→PAT→LOC→CLIN 的升级条件；
- [`reporting/claim_policy.json`](reporting/claim_policy.json)：允许和禁止的报告 claim；
- [`knowledge_base.jsonl`](knowledge_base.jsonl)：25 条带引用的来源摘要，暂保留旧调用路径。

`knowledge_base.jsonl` 是来源摘要层，不是概念本体，也不是患者标签。数据集说明类条目只能解释
相应数据集的标注语义，不应在无关病例中为了“collection 覆盖率”被强制检索。

检索排序、top-k 和组合策略属于消费端配置，不是领域知识。本仓库的声明性影子策略位于
`configs/eeg_knowledge/retrieval_policy_v2.json`，当前尚无 v2 中立 retriever 或 qualification
receipt 运行时实现。

## 版本与审核

活动知识版本为 `2.0.0-draft`。在临床部署前，每张概念卡都必须完成临床神经生理医师审核，
并补齐 claim 级来源定位（标准条款、章节或页码）、适用人群、标准版本和复核日期。教学网站
用于解释和导航，正式指南/共识/标准优先于教学概括。

`manifest.json` 记录全部活动入口/合同文件的 SHA-256，并用规范化 hash map 形成
`active_bundle_sha256`。这能发现未同步的文件漂移，但不是外部签名；当前目录仍是未提交 Git 的
workspace draft，在 commit/build receipt 建立前不能称正式可复现发布。

运行结构校验：

```bash
rtk python3 scripts/validate_eeg_knowledge_system.py
```

本知识系统仅用于研究和专业复核支持，不替代临床判读、术前多模态评估或侵入性验证。
