# NeuroSOZ v34：受约束 Qwen3.6 与癫痫知识库报告语言层

**日期：** 2026-08-16  
**角色：** 表达、解释和信息组织；不参与 SOZ 预测。  
**发布原则：** 事实核心优先，LLM 可选；任何失败均 fail-closed 到确定性正文。

## 1. 为什么增加 LLM，但不让 LLM 定位

此前确定性模板具有事实稳定、可回放和无幻觉的优点，但组织方式固定，医学概念解释能力有限。新增语言层只解决：

1. 将已经锁定的患者事实按临床阅读顺序重组；
2. 用白名单文献解释 SOZ、头皮电极、致痫区和 EEG 描述边界；
3. 改善候选与弃权报告的专业表达。

以下任务仍完全由冻结模型或确定性规则完成：

- SOZ 候选分数、顺序、Top-K 和区域投影；
- 候选显示或弃权；
- 波形区间和导联观察；
- concept qualification；
- 患者事实、gold label 和评价指标。

因此 LLM 不会改变任何 SOZ 指标，也不能把一般知识转成患者特异事实。

## 2. 信息流

```text
v32 facts-locked report
  ├─ patient/unit identity
  ├─ candidate or abstention
  ├─ locked typed facts
  └─ clinical boundary
            │
            │ 不加载 Raw EEG / SOZ gold / evaluation / hidden ranking
            ▼
authorized knowledge selector
  └─ 只从白名单文献摘要中选择最多4项
            │
            ▼
local Qwen3.6 language-only generation
  └─ strict JSON schema, temperature=0, thinking disabled
            │
            ▼
independent publication validator
  ├─ locked identity/action/candidates/region exact equality
  ├─ section-to-fact binding
  ├─ electrode/number/region subset check
  ├─ clinical-sensitive-term support check
  ├─ forbidden assertion check
  └─ authorized citation check
       │ pass                         │ fail/error
       ▼                              ▼
publish Qwen narrative          deterministic fallback
       └──────────────┬───────────────┘
                      ▼
HTML “知识辅助解读”独立卡片
事实核心、候选表、波形和逐句依据均不被替换
```

## 3. 本地模型与运行依赖

| 项目 | 冻结值 |
|---|---|
| 模型 | `Qwen3.6-35B-A3B-GPTQ-Int4` |
| 模型路径 | `models/Qwen3.6-35B-A3B-GPTQ-Int4` |
| 推理框架 | vLLM 0.19.1 |
| 环境 | `.venv-vllm-qwen36` |
| 服务 | `qwen36-soz`，仅监听 `127.0.0.1` |
| GPU | RTX A6000 48GB；权重实测约 21.06 GiB |
| context | 8,192 tokens |
| temperature | 0.0 |
| thinking | disabled；不请求、不读取、不保留 chain-of-thought |
| response format | strict JSON schema |

vLLM 0.19.1 对 Qwen3.6 hybrid attention/Mamba 模型存在一个启动边界：当 `max-num-seqs=1` 时，内部临时 KV cache 恰为 `(2,2,...)`，布局判定失败。启动脚本把 `max_cudagraph_capture_size` 固定为 1，避免临时缓存歧义；正式模型权重与最终 KV 容量不变。

当前 workspace 模型路径是指向 `/tmp/Qwen3.6-35B-A3B-GPTQ-Int4` 的符号链接，适合本机实验但不适合长期归档。正式复现包应把模型放入持久化只读存储，并冻结 config、generation config、weight index、tokenizer 及全部 shard hashes；本次独立审计至少记录前四项 hash。

## 4. 权威知识白名单

来源配置位于 `configs/constrained_llm_reporting_v1.json`，知识条目来自 `knowledge/eeg/knowledge_base.jsonl`。允许来源包括：

- IFCN/临床 EEG 术语与报告格式文献；
- Lüders 等关于 SOZ、irritative zone、symptomatogenic zone 和 EZ 的概念区分；
- ILAE 发作类型共识；
- ACNS 2021 terminology；
- Niedermeyer 与 Tatum 临床 EEG 教材；
- TUSZ 数据集论文。

每个 source ID 同时绑定 citation、authority tier 和 allowed use。当前实现应准确称为**带引用的权威来源白名单摘要库**，而不是完整知识图谱或权威全文 RAG。一般医学原则只能放在 `knowledge_notes`，不能为患者新增电极、时间、区域、形态、传播、症状、影像或治疗事实。

## 5. 输入合同

LLM 只能读取：

```text
LOCKED_FACTS = {
  analysis_scope,
  waveform_observation,
  localization_result,
  reference_opinion,
  evidence_applicability,
  clinical_boundary
}
```

不能读取：

```text
Raw EEG
DeepSOZ/private SOZ labels
evaluation hit/miss
known-spread result
abstained hidden ranking
model gradients or weights
```

四个发布章节及允许 fact IDs 固定为：

| 章节 | 唯一允许的患者事实 |
|---|---|
| 分析范围 | `analysis_scope` |
| 波形复核要点 | `waveform_observation` |
| 定位参考 | `localization_result` + `reference_opinion` |
| 证据边界 | `evidence_applicability` + `clinical_boundary` |

## 6. 发布验证器

### 6.1 Exact-lock

以下字段必须与 v32 source report 逐值一致：

- unit ID、patient ID；
- `localization_action`；
- 候选电极数组及顺序；
- `top1_region_zh`；
- 四个 safety acknowledgements 均为 false。

### 6.2 Patient-fact surface

每节单独检查：

- 输出电极必须是该节 locked facts 中出现过的电极；
- 输出数字必须是该节 locked facts 中出现过的数字；
- 输出区域必须是该节 locked facts 中出现过的区域；
- “演变、传播、起始、形态、节律、伪迹、症状、影像、解剖、诊断、治疗、手术、皮层 SOZ、致痫区”等敏感词，仅在该节原事实已出现时允许使用；
- 禁止确认皮层 SOZ/EZ/手术靶点、传播路径、波形导致模型评分、直接诊疗用途等断言。

### 6.3 Knowledge grounding

- 每条知识说明必须引用 1--2 个本病例已授权的 source IDs；
- 不得出现患者电极、数字或区域；
- 与所引 source summary/rules/limitations 需达到最低词面支持；
- citation 由知识库提供，LLM 不自行生成文献条目。

### 6.4 Fail-closed

网络错误、空正文、JSON 错误、schema 错误、锁定字段漂移、无授权 citation 或患者事实越界均触发单病例确定性回退。临床发布物不保存未验证模型草稿，只保存可重放的 canonical hash。

## 7. HTML 展示逻辑

原有内容保持不变：

1. 临床可读摘要；
2. Top-K 候选或弃权；
3. 处理后 19 导波形；
4. 证据适用性和临床边界；
5. 逐句证据依据及二级技术字段。

新增独立卡片：

```text
知识辅助解读
  ├─ Qwen3.6受约束语言层 / 安全回退状态
  ├─ 四段事实绑定叙述
  └─ 可展开的医学知识依据与正式 citation
```

页面明确写明语言层未参与 SOZ 预测、评分或候选选择。弃权病例不会暴露隐藏排序。

## 8. 实测状态

### 8.1 双病例真实 pilot

| 病例 | 定位状态 | 结果 |
|---|---|---|
| `PRIV-E0003` | 显示 P8、FP2、C4、C3、P4；P8→右颞区 | Qwen 输出通过全部发布门 |
| `PRIV-E0001` | localization abstain；无候选 | Qwen 输出通过全部发布门，未暴露排序 |

pilot 产物：`outputs/constrained_llm_soz_reports_v34_qwen36_pilot_real_v3_20260816/`。

### 8.2 全量报告

全量对象为 public 102 个 patient reports 与 private 88 个 event reports：

| 项目 | 结果 |
|---|---:|
| Qwen 正常发布 | 188/190 = 98.95% |
| 确定性安全回退 | 2/190 = 1.05% |
| 独立离线重验证 | 190/190 通过 |
| Qwen 患者事实段落与确定性版本逐字相同 | 744/752 = 98.94% |
| 四段均逐字相同的 Qwen 报告 | 184/188 = 97.87% |
| 知识说明直接等于来源摘要 | 346/355 = 97.46% |

两个回退对象是 public patient `7793` 和 `9370`。Qwen 在 `waveform_review` 中新增了未在该节 locked fact 出现的“起始”一词，敏感词门拒绝发布；两份草稿均未保留正文，只保存 canonical hash。没有通过放宽门槛重试。

### 机器变异攻击审计 v41

在不调用LLM的情况下，对全部102份public患者报告和88份private事件报告的合法payload逐一实施12类预设高风险变异：候选篡改、unit ID篡改、新电极、新数字、新区域、皮层SOZ断言、传播路径断言、非法fact ID、非法知识来源、额外payload字段、虚假安全声明及`<think>`标记注入。总计2,280次变异均被独立发布validator拒绝，unsafe escape为0，拒绝后发布动作均为确定性fallback。

工件：`outputs/trustworthy_soz_reporting_mutation_audit_v41_20260816/result.json`。

该结果证明的是当前预设攻击面上的机器事实锁和回退执行，不证明临床事实完整性、专业可读性、临床效用或自动化偏倚安全性。该语言实验仍绑定v21候选profile，不能借用v29定位性能。

该结果证明安全链可运行，但也说明当前语言层的增量主要是白名单知识选择和固定结构组织，而不是显著的自由改写。它不能作为“LLM 明显提升报告表达”的证据；是否有临床阅读增益必须由 template-versus-Qwen 盲法 reader study 回答。

正式 HTML 共 191 个页面（1 个索引 + 190 个报告），包含 188 个 Qwen badge、2 个安全回退 badge 和 189 张可绑定波形；152 份显示候选，38 份弃权。发布入口：`outputs/trustworthy_soz_clinician_html_v34_20260816/index.html`。

## 9. 仍然存在的限制

1. 规则验证不是完备的自然语言蕴含证明；需要专家抽检和 adversarial evaluation。
2. 知识摘要需要癫痫专科医生逐条确认，最好补 document version、section/page 和原文摘录。
3. LLM 可能提高流畅度并同时增加 automation bias，必须用盲法 reader study 评价。
4. LLM 层不能修复 v21/v29 的 strict localization error、标签语义差异或独立验证缺失。
5. 在 reader study 前，输出仍是科研辅助报告，不是正式 EEG 诊断报告。

## 10. 实现与验证入口

- 策略：`configs/constrained_llm_reporting_v1.json`
- 核心合同：`src/soz/constrained_llm_reporting.py`
- 材料化：`scripts/materialize_constrained_llm_soz_reports_v34.py`
- 本地服务：`scripts/serve_qwen36_vllm.sh`
- HTML：`scripts/render_trustworthy_soz_reports_v23.py`
- 独立审计：`scripts/audit_constrained_llm_soz_reports_v34.py`
- 测试：`tests/test_constrained_llm_soz_reporting.py`
- 正式语言包：`outputs/constrained_llm_soz_reports_v34_qwen36_20260816/`
- 审计回执：`outputs/constrained_llm_soz_reports_v34_qwen36_audit_20260816/manifest.json`
- 正式 HTML：`outputs/trustworthy_soz_clinician_html_v34_20260816/index.html`
