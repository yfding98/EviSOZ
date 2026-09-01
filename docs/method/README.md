# EviSOZ-LM 方法文档

本目录的受控方法参考统一位于 [`reference/`](reference/)。为兼容早期会话，当前目录仍保留两份
入口副本：

- [`evisoz_lm_repository_aligned_design_v1_20260830_zh.md`](evisoz_lm_repository_aligned_design_v1_20260830_zh.md)：完整方法架构、数据治理、Stage-0 门和分阶段训练合同。
- [`evisoz_evidence_json_runtime_usage_v1_20260901_zh.md`](evisoz_evidence_json_runtime_usage_v1_20260901_zh.md)：Evidence JSON 的实际 schema、生产/消费入口和真实数据运行时地图。

## 路径与版本约定

文档中的 `outputs/...` 是原工作区的外部受控 artifact 根，默认对应：

```text
/mnt/hd1/dyf/workspace/laptop/EEG_Seizure/outputs/
```

它们不是本 Git 项目的输入，也不会因为文档迁移而自动获得训练或评估授权。目标仓库的生成结果应写入被 `.gitignore` 忽略的 `outputs/`，并在每次重放时生成新的版本化 receipt。

文档中的源码相对路径以本仓库根目录
`/mnt/hd1/dyf/workspace/laptop/EviSOZ` 为准；如果某个历史段落引用了当前 clean worktree 未包含的 legacy 路径，应先按 [`AGENTS.md`](../../AGENTS.md) 审计其依赖，不得把它当作已验证的运行入口。

两份文档均记录了 2026-09-01 前后的历史 gate 状态；新会话应以当前外部 Stage-0 gate、目标仓库 schema registry 和最新 receipt 为准，不覆盖历史记录。

## 迁移后的参考层

[`reference/README.md`](reference/README.md) 列出完整方案、Evidence JSON runtime map、SOZ 目标合同、
canonical v29 H/D 协议、报告/知识库边界以及所需的历史审计协议。后续代码和实验应优先引用
`docs/method/reference` 中的版本化文件；父工作区 `research/02_method` 只作为未迁移历史记录来源。
