# EviSOZ-LM 方法文档

本目录只保留当前 EviSOZ-LM 实施所需的两份规范文档：

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
