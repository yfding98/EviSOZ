#!/usr/bin/env python3
"""Render frozen typed-fact SOZ candidate reports as a clinician-readable HTML pack.

The renderer is presentation-only. It reads the already materialized public
patient and private event report JSONL files. It never reads EEG, SOZ labels,
model weights, hidden rankings, or evaluation rows, and it does not alter any
candidate, score, threshold, abstention decision, or clinical clause.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from html import escape
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs/trustworthy_soz_qualified_reports_v22_20260815"
DEFAULT_OUTPUT = ROOT / "outputs/trustworthy_soz_clinician_html_v23_20260815"
REPORT_SCHEMAS = {
    "trustworthy_soz_qualified_report_v22",
    "trustworthy_soz_qualified_report_v24",
    "trustworthy_soz_clinical_reference_report_v32",
}
MANIFEST_SCHEMAS = {
    "trustworthy_soz_qualified_reporting_manifest_v22",
    "trustworthy_soz_qualified_reporting_manifest_v24",
    "trustworthy_soz_clinical_reference_reporting_manifest_v32",
}
WAVEFORM_MANIFEST_SCHEMA = "trustworthy_soz_processed_waveform_figures_v32"
OUTPUT_SCHEMA = "trustworthy_soz_clinician_html_manifest_v23"
CONSTRAINED_LLM_MANIFEST_SCHEMA = "trustworthy_soz_constrained_llm_manifest_v1"
CONSTRAINED_LLM_RECORD_SCHEMA = "trustworthy_soz_constrained_llm_narrative_v1"
CONSTRAINED_LLM_PAYLOAD_SCHEMA = "trustworthy_soz_constrained_llm_payload_v1"
SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


CSS = """
:root { color-scheme: light; --ink:#172033; --muted:#5b6475; --line:#d8deea;
  --blue:#2457d6; --blue-soft:#eef3ff; --amber:#9a5d00; --amber-soft:#fff7e6;
  --red:#a52a2a; --red-soft:#fff0f0; --green:#176b4d; --green-soft:#edf9f4; }
* { box-sizing: border-box; }
body { margin:0; background:#f4f6fa; color:var(--ink); font-family:-apple-system,
  BlinkMacSystemFont,"Segoe UI","Noto Sans SC","Microsoft YaHei",sans-serif;
  line-height:1.65; }
main { max-width:1040px; margin:32px auto; padding:0 20px 48px; }
.card { background:white; border:1px solid var(--line); border-radius:14px;
  box-shadow:0 8px 26px rgba(25,39,72,.06); padding:26px 30px; margin:18px 0; }
h1 { margin:0 0 4px; font-size:27px; } h2 { margin:0 0 14px; font-size:19px; }
.sub,.muted { color:var(--muted); } .meta { display:grid;
  grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:8px 24px; }
.tag { display:inline-block; border-radius:999px; padding:3px 10px; margin-right:7px;
  font-size:13px; font-weight:650; }
.candidate { color:var(--green); background:var(--green-soft); }
.abstain { color:var(--amber); background:var(--amber-soft); }
.warning { color:var(--red); background:var(--red-soft); border-left:4px solid var(--red);
  padding:12px 15px; border-radius:8px; }
.decision { background:var(--blue-soft); border-left:4px solid var(--blue);
  padding:14px 16px; border-radius:8px; }
table { width:100%; border-collapse:collapse; margin-top:10px; }
th,td { border-bottom:1px solid var(--line); padding:9px 10px; text-align:left; }
th { color:var(--muted); font-size:13px; } code { overflow-wrap:anywhere; }
.clause { padding:12px 0; border-bottom:1px solid var(--line); }
.clause:last-child { border-bottom:0; }
.clause-type { color:var(--blue); font-size:13px; font-weight:700; }
details { margin-top:6px; color:var(--muted); } a { color:var(--blue); }
.index-table td:last-child { width:110px; }
.waveform { width:100%; height:auto; display:block; border:1px solid var(--line);
  border-radius:10px; background:white; }
.reference-opinion { background:var(--green-soft); border-left:4px solid var(--green);
  padding:14px 16px; border-radius:8px; }
.technical { font-size:13px; color:var(--muted); }
.evidence-details { margin-top:9px; padding:9px 12px; background:#f8fafc;
  border:1px solid var(--line); border-radius:8px; }
.evidence-grid { display:grid; grid-template-columns:minmax(90px,130px) 1fr;
  gap:5px 12px; margin-top:9px; }
.evidence-label { color:var(--muted); font-size:13px; font-weight:700; }
.technical-audit { margin:10px 0 0; padding-top:7px; border-top:1px dashed var(--line);
  font-size:12px; }
.language-layer { border-left:4px solid #7157bd; }
.language-badge { color:#563d9b; background:#f1edff; }
.language-section { padding:10px 0; border-bottom:1px solid var(--line); }
.language-section:last-of-type { border-bottom:0; }
.language-heading { color:#563d9b; font-size:14px; font-weight:700; }
.knowledge-note { margin:9px 0; padding:10px 12px; background:#f8f6ff;
  border-radius:8px; }
.citation { margin-top:4px; color:var(--muted); font-size:12px; }
@media print { body { background:white; } main { max-width:none; margin:0; }
  .card { box-shadow:none; break-inside:avoid; } .no-print { display:none; } }
"""


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.resolve(strict=True).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"row {line_number} is not an object: {path}")
            rows.append(value)
    if not rows:
        raise ValueError(f"empty report file: {path}")
    return rows


def _canonical_sha256(value: Mapping[str, object]) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} must be non-empty text")
    return value.strip()


def _object(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be an object")
    return value


def _safe_unit_id(value: object) -> str:
    unit_id = _text(value, name="unit_id")
    if not SAFE_ID.fullmatch(unit_id):
        raise ValueError(f"unsafe unit_id: {unit_id!r}")
    return unit_id


def _validate_record(record: Mapping[str, object]) -> tuple[str, str]:
    if record.get("schema_version") not in REPORT_SCHEMAS:
        raise ValueError("qualified report schema drifted")
    unit_id = _safe_unit_id(record.get("unit_id"))
    patient_id = _safe_unit_id(record.get("patient_id"))
    if record.get("facts_locked") is not True or record.get("llm_used") is not False:
        raise ValueError(f"report {unit_id} is not facts-locked or declares LLM use")
    clauses = record.get("clauses")
    sentence_map = record.get("sentence_fact_map")
    if not isinstance(clauses, list) or not clauses:
        raise TypeError(f"report {unit_id} has no clauses")
    if not isinstance(sentence_map, list) or len(sentence_map) != len(clauses):
        raise ValueError(f"report {unit_id} clause/fact-map count mismatch")
    localization = _object(record.get("localization"), name="localization")
    action = localization.get("action")
    displayed = localization.get("displayed_candidates")
    if action not in {"display_candidate", "localization_abstain", "localization_unavailable"}:
        raise ValueError(f"unsupported localization action: {action!r}")
    if not isinstance(displayed, list):
        raise TypeError("displayed_candidates must be a list")
    if action != "display_candidate" and displayed:
        raise ValueError("abstained/unavailable report exposes a hidden ranking")
    return unit_id, patient_id


def _candidate_table(localization: Mapping[str, object]) -> str:
    candidates = localization.get("displayed_candidates")
    if not isinstance(candidates, list) or not candidates:
        return ""
    rows = []
    for rank, raw in enumerate(candidates, start=1):
        item = _object(raw, name="candidate")
        channel = escape(_text(item.get("channel"), name="candidate channel"))
        score = item.get("normalized_candidate_score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise TypeError("candidate score must be numeric")
        rows.append(f"<tr><td>{rank}</td><td><strong>{channel}</strong></td><td>{float(score):.4f}</td></tr>")
    return (
        "<table><thead><tr><th>顺位</th><th>头皮电极候选</th>"
        "<th>模型排序分数（仅用于相对比较）</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


_EVIDENCE_GUIDANCE = {
    "analysis_scope": (
        "信号处理记录与报告身份绑定",
        "客观处理信息；用于确认本报告分析了哪一段信号及采用何种处理方式",
        "核对处理后波形的导联、采样率、参考方式和时间范围",
    ),
    "waveform_observation": (
        "独立且不读取SOZ标注的波形变化检测",
        "算法观察；不等同于医生确认的发作起始、传播路径或SOZ起点",
        "在处理后波形中复核黄色标记区间及相关导联",
    ),
    "localization_result": (
        "本报告使用的LaBraM头皮电极定位模型",
        "相对候选排序；排序分数不是经临床校准的SOZ概率",
        "结合完整发作期脑电、症状学和影像学复核候选区域",
    ),
    "reference_opinion": (
        "定位结果与预先固定的临床表达规则",
        "复核建议；不是新的模型预测或确定性诊断",
        "由癫痫专科医生决定是否采纳以及是否需要进一步检查",
    ),
    "evidence_applicability": (
        "证据资格审查记录",
        "未完成独立临床验证的具体形态、节律、伪迹或空间扩展分类仅供参考",
        "医生可在原始及处理后脑电中独立复核，不应据此单独定位SOZ",
    ),
    "clinical_boundary": (
        "预先固定的医学使用边界",
        "适用于所有报告；不随模型候选或病例结果改变",
        "不得将头皮电极候选直接解释为皮层SOZ、致痫区或手术靶点",
    ),
}


def _fact_details(paths: object, *, clause_type: str, waveform_available: bool) -> str:
    if not isinstance(paths, list) or not paths or not all(isinstance(item, str) for item in paths):
        raise TypeError("clause fact_paths must be a non-empty string list")
    items = "".join(f"<li><code>{escape(item)}</code></li>" for item in paths)
    source, status, review = _EVIDENCE_GUIDANCE.get(
        clause_type,
        (
            "结构化报告记录",
            "该内容来自锁定的报告事实，不由HTML展示层重新推断",
            "结合原始脑电及完整临床资料复核",
        ),
    )
    waveform_link = (
        '<div class="evidence-label">快捷复核</div><div><a href="#processed-waveform">定位到处理后波形</a></div>'
        if waveform_available and clause_type == "waveform_observation"
        else ""
    )
    return (
        '<details class="evidence-details"><summary>查看证据依据</summary>'
        '<div class="evidence-grid">'
        f'<div class="evidence-label">证据来源</div><div>{escape(source)}</div>'
        f'<div class="evidence-label">证据状态</div><div>{escape(status)}</div>'
        f'<div class="evidence-label">建议复核</div><div>{escape(review)}</div>'
        f"{waveform_link}</div>"
        '<details class="technical-audit"><summary>查看技术审计字段（供研究人员）</summary>'
        f"<ul>{items}</ul></details></details>"
    )


_CLAUSE_LABELS = {
    "analysis_scope": "分析范围",
    "waveform_observation": "波形观察",
    "localization_result": "定位结果",
    "reference_opinion": "参考意见",
    "evidence_applicability": "证据适用性",
    "clinical_boundary": "临床边界",
}


def _validate_language_record(
    language_record: Mapping[str, object],
    source_record: Mapping[str, object],
) -> tuple[dict[str, object], str, str | None, dict[str, str]]:
    if language_record.get("schema_version") != CONSTRAINED_LLM_RECORD_SCHEMA:
        raise ValueError("constrained language record schema drifted")
    unit_id = _safe_unit_id(source_record.get("unit_id"))
    patient_id = _safe_unit_id(source_record.get("patient_id"))
    if language_record.get("unit_id") != unit_id or language_record.get("patient_id") != patient_id:
        raise ValueError(f"constrained language identity mismatch: {unit_id}")
    if language_record.get("cohort") != source_record.get("cohort"):
        raise ValueError(f"constrained language cohort mismatch: {unit_id}")
    if language_record.get("source_report_schema") != source_record.get("schema_version"):
        raise ValueError(f"constrained language source schema mismatch: {unit_id}")
    if language_record.get("source_report_sha256") != _canonical_sha256(source_record):
        raise ValueError(f"constrained language source hash mismatch: {unit_id}")
    if language_record.get("localization") != source_record.get("localization"):
        raise ValueError(f"constrained language changed localization: {unit_id}")
    access = _object(language_record.get("access_receipt"), name="language access receipt")
    required_false = (
        "raw_eeg_loaded",
        "soz_gold_labels_loaded",
        "evaluation_rows_loaded",
        "model_scores_or_localization_changed",
        "patient_facts_added",
    )
    if any(access.get(field) is not False for field in required_false):
        raise ValueError(f"constrained language access contract failed: {unit_id}")
    generation = _object(language_record.get("generation"), name="language generation")
    generator = str(generation.get("generator", ""))
    if generator not in {"qwen3.6_constrained_language_only", "deterministic_fallback"}:
        raise ValueError(f"unsupported constrained language generator: {generator!r}")
    payload = _object(language_record.get("published_narrative"), name="published narrative")
    if payload.get("schema_version") != CONSTRAINED_LLM_PAYLOAD_SCHEMA:
        raise ValueError(f"constrained language payload schema drifted: {unit_id}")
    localization = _object(source_record.get("localization"), name="localization")
    candidates = [
        str(item["channel"])
        for item in localization.get("displayed_candidates", [])
        if isinstance(item, Mapping)
    ]
    locked = {
        "unit_id": unit_id,
        "patient_id": patient_id,
        "localization_action": localization.get("action"),
        "candidate_channels": candidates,
        "top1_region_zh": localization.get("top1_region_projection_zh"),
    }
    if any(payload.get(field) != value for field, value in locked.items()):
        raise ValueError(f"constrained language payload changed locked fields: {unit_id}")
    receipt = _object(language_record.get("knowledge_receipt"), name="knowledge receipt")
    raw_citations = receipt.get("citations")
    if not isinstance(raw_citations, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in raw_citations.items()
    ):
        raise TypeError(f"constrained language citations are invalid: {unit_id}")
    fallback_reason = generation.get("fallback_reason")
    if fallback_reason is not None and not isinstance(fallback_reason, str):
        raise TypeError("language fallback reason must be text or null")
    return payload, generator, fallback_reason, dict(raw_citations)


def _language_layer_html(
    language_record: Mapping[str, object] | None,
    source_record: Mapping[str, object],
) -> tuple[str, str | None]:
    if language_record is None:
        return "", None
    payload, generator, fallback_reason, citations = _validate_language_record(
        language_record, source_record
    )
    sections = payload.get("sections")
    notes = payload.get("knowledge_notes")
    if not isinstance(sections, list) or len(sections) != 4:
        raise ValueError("constrained language section count drifted")
    if not isinstance(notes, list) or not notes:
        raise ValueError("constrained language knowledge notes are missing")
    section_html: list[str] = []
    for raw in sections:
        section = _object(raw, name="language section")
        heading = escape(_text(section.get("heading_zh"), name="language heading"))
        body = escape(_text(section.get("text_zh"), name="language text"))
        section_html.append(
            f'<div class="language-section"><div class="language-heading">{heading}</div>'
            f'<div>{body}</div></div>'
        )
    note_html: list[str] = []
    for raw in notes:
        note = _object(raw, name="knowledge note")
        body = escape(_text(note.get("text_zh"), name="knowledge note text"))
        source_ids = note.get("source_ids")
        if not isinstance(source_ids, list) or not source_ids or not all(
            isinstance(source_id, str) and source_id in citations for source_id in source_ids
        ):
            raise ValueError("knowledge note cites an unavailable source")
        source_text = "；".join(escape(citations[source_id]) for source_id in source_ids)
        note_html.append(
            f'<div class="knowledge-note">{body}<div class="citation">依据：{source_text}</div></div>'
        )
    if generator == "qwen3.6_constrained_language_only":
        badge = "Qwen3.6受约束语言层"
        notice = "以下文字仅重组已锁定的患者事实，并使用白名单文献解释一般医学原则；未参与SOZ预测、评分或候选选择。"
    else:
        if fallback_reason == "llm_not_called":
            badge = "确定性语言版本"
            notice = "当前版本未调用LLM，以下内容由确定性规则组织；SOZ预测与候选保持不变。"
        elif fallback_reason == "llm_validation_failed":
            badge = "规则化安全回退"
            notice = "本病例的模型草稿未通过安全发布条件，以下内容由确定性规则生成；SOZ预测与候选仍保持不变。"
        else:
            badge = "语言服务安全回退"
            notice = "本病例未获得可发布的模型正文，以下内容由确定性规则生成；SOZ预测与候选仍保持不变。"
    html = (
        '<section class="card language-layer"><h2>知识辅助解读</h2>'
        f'<span class="tag language-badge">{badge}</span><p class="muted">{notice}</p>'
        + "".join(section_html)
        + '<details><summary>查看医学知识依据</summary>'
        + "".join(note_html)
        + "</details></section>"
    )
    return html, generator


def _render_report(
    record: Mapping[str, object],
    *,
    title_scope: str,
    waveform_href: str | None = None,
    language_record: Mapping[str, object] | None = None,
) -> str:
    unit_id, patient_id = _validate_record(record)
    localization = _object(record.get("localization"), name="localization")
    action = str(localization["action"])
    action_zh = {
        "display_candidate": "显示候选",
        "localization_abstain": "低置信弃权",
        "localization_unavailable": "定位不可用",
    }[action]
    badge = "candidate" if action == "display_candidate" else "abstain"
    region = localization.get("top1_region_projection_zh")
    margin = localization.get("top1_top2_margin")
    threshold = localization.get("frozen_threshold")
    margin_text = "不可用" if margin is None else f"{float(margin):.4f}"
    threshold_text = "不可用" if threshold is None else f"{float(threshold):.4f}"
    clauses = record["clauses"]
    clause_html = []
    for raw in clauses:
        clause = _object(raw, name="clause")
        raw_clause_type = _text(clause.get("type"), name="clause type")
        clause_type = escape(_CLAUSE_LABELS.get(raw_clause_type, raw_clause_type))
        clause_text = escape(_text(clause.get("text"), name="clause text"))
        clause_html.append(
            f'<div class="clause"><div class="clause-type">{clause_type}</div>'
            f"<div>{clause_text}</div>"
            f"{_fact_details(clause.get('fact_paths'), clause_type=raw_clause_type, waveform_available=waveform_href is not None)}"
            "</div>"
        )
    clinical_text = escape(_text(record.get("clinical_text_zh"), name="clinical_text_zh"))
    reference_opinion = record.get("reference_opinion_zh")
    applicability = record.get("evidence_applicability_zh")
    boundary_text = record.get("clinical_boundary_zh")
    is_v32 = record.get("schema_version") == "trustworthy_soz_clinical_reference_report_v32"
    location_text = "—" if region is None else escape(str(region))
    waveform_html = ""
    if waveform_href is not None:
        waveform = _object(record.get("waveform_figure"), name="waveform_figure")
        event_id = escape(_text(waveform.get("event_id"), name="waveform event_id"))
        representative = bool(waveform.get("representative_event"))
        scope_note = (
            "下图为患者级分析中的一段代表性事件；最终候选由全部合格发作综合获得。"
            if representative
            else "下图与当前事件使用同一处理窗口和参考方式。"
        )
        waveform_html = (
            '<section class="card" id="processed-waveform"><h2>处理后脑电波形</h2>'
            f'<p class="muted">{scope_note} 图示事件：{event_id}。黄色区域仅为算法变化区间。</p>'
            f'<img class="waveform" src="{escape(waveform_href)}" alt="处理后标准19导脑电波形"></section>'
        )
    applicability_html = ""
    if is_v32 and isinstance(applicability, str) and isinstance(boundary_text, str):
        applicability_html = (
            '<section class="card"><h2>证据适用性与使用范围</h2>'
            f'<p>{escape(applicability)}</p><p>{escape(boundary_text)}</p></section>'
        )
    opinion_html = ""
    if is_v32 and isinstance(reference_opinion, str):
        opinion_html = f'<div class="reference-opinion"><strong>{escape(reference_opinion)}</strong></div>'
    language_html, _ = _language_layer_html(language_record, record)
    technical_meta = (
        f'<details class="technical"><summary>查看模型显示参数</summary>'
        f'<div>Top1–Top2 margin：{margin_text}；显示阈值：{threshold_text}。'
        "上述数值不是校准后的正确概率。</div></details>"
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(title_scope)} · {escape(unit_id)}</title><style>{CSS}</style></head>
<body><main>
<section class="card"><div class="sub">科研辅助输出 · 非确诊报告</div>
<h1>发作期头皮脑电SOZ候选辅助分析报告</h1><span class="tag {badge}">{action_zh}</span>
<div class="meta"><div><strong>分析单位：</strong>{escape(unit_id)}</div>
<div><strong>患者标识：</strong>{escape(patient_id)}</div><div><strong>队列：</strong>{escape(title_scope)}</div>
<div><strong>首位空间投影：</strong>{location_text}</div></div>{technical_meta}</section>
<section class="card"><h2>临床可读摘要</h2><div class="decision">{clinical_text}</div>
{_candidate_table(localization)}{opinion_html}</section>
{waveform_html}
{language_html}
{applicability_html}
<section class="card"><h2>诊断链条与依据</h2>{''.join(clause_html)}</section>
<section class="card warning"><strong>使用范围：</strong>本报告提供头皮电极层面的辅助定位参考，
不能替代临床脑电判读，也不能单独确定皮层SOZ、致痫区或手术靶点。弃权不表示不存在SOZ。</section>
<p class="no-print"><a href="index.html">返回报告索引</a></p>
</main></body></html>"""


def _render_index(rows: Sequence[Mapping[str, object]], *, language_layer_loaded: bool) -> str:
    body = []
    for row in rows:
        unit_id = str(row["unit_id"])
        patient_id = str(row["patient_id"])
        action = str(row["action"])
        action_zh = "显示候选" if action == "display_candidate" else ("弃权" if action == "localization_abstain" else "不可用")
        body.append(
            f"<tr><td>{escape(row['scope'])}</td><td>{escape(unit_id)}</td>"
            f"<td>{escape(patient_id)}</td><td>{escape(action_zh)}</td>"
            f'<td><a href="{escape(str(row["file"]))}">打开报告</a></td></tr>'
        )
    provenance = (
        "报告事实核心由结构化证据确定性生成；可选知识辅助层仅重组锁定事实，不参与SOZ预测。"
        if language_layer_loaded
        else "报告正文由结构化证据确定性生成。"
    )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>SOZ候选辅助报告索引</title>
<style>{CSS}</style></head><body><main><section class="card"><div class="sub">候选、弃权、处理后波形与逐句依据</div>
<h1>发作期头皮脑电SOZ候选辅助报告索引</h1><p>本索引不包含SOZ标签或评价结果。{provenance}</p>
<table class="index-table"><thead><tr><th>队列</th><th>分析单位</th><th>患者</th><th>状态</th><th>报告</th></tr></thead>
<tbody>{''.join(body)}</tbody></table></section></main></body></html>"""


def render(
    input_directory: Path,
    output_directory: Path,
    waveform_directory: Path | None = None,
    constrained_llm_directory: Path | None = None,
) -> dict[str, object]:
    source = input_directory.resolve(strict=True)
    source_manifest = _read_json(source / "manifest.json")
    if source_manifest.get("schema_version") not in MANIFEST_SCHEMAS:
        raise ValueError("qualified report manifest schema drifted")
    cohorts = (
        ("public_patient", "公开开发队列", source / "public_patient_reports.jsonl"),
        ("private_event", "回顾性验证队列", source / "private_event_reports.jsonl"),
    )
    language_records: dict[tuple[str, str], dict[str, object]] = {}
    language_source: Path | None = None
    if constrained_llm_directory is not None:
        language_source = constrained_llm_directory.resolve(strict=True)
        language_manifest = _read_json(language_source / "manifest.json")
        if language_manifest.get("schema_version") != CONSTRAINED_LLM_MANIFEST_SCHEMA:
            raise ValueError("constrained language manifest schema drifted")
        language_access = _object(
            language_manifest.get("access_receipt"), name="language manifest access receipt"
        )
        if any(
            language_access.get(field) is not False
            for field in ("raw_eeg_loaded", "soz_gold_labels_loaded", "evaluation_rows_loaded", "localization_changed")
        ):
            raise ValueError("constrained language manifest access contract failed")
        for scope, filename in (
            ("public_patient", "public_patient_reports.jsonl"),
            ("private_event", "private_event_reports.jsonl"),
        ):
            path = language_source / filename
            if not path.is_file():
                continue
            for row in _read_jsonl(path):
                key = (scope, _safe_unit_id(row.get("unit_id")))
                if key in language_records:
                    raise ValueError("duplicate constrained language identity")
                language_records[key] = row
    waveform_source: Path | None = None
    waveform_entries: dict[tuple[str, str], dict[str, object]] = {}
    if waveform_directory is not None:
        waveform_source = waveform_directory.resolve(strict=True)
        waveform_manifest = _read_json(waveform_source / "manifest.json")
        if waveform_manifest.get("schema_version") != WAVEFORM_MANIFEST_SCHEMA:
            raise ValueError("waveform manifest schema drifted")
        entries = waveform_manifest.get("entries")
        if not isinstance(entries, list):
            raise TypeError("waveform manifest has no entries")
        for entry in entries:
            if not isinstance(entry, dict):
                raise TypeError("waveform entry is not an object")
            key = (str(entry.get("scope", "")), str(entry.get("unit_id", "")))
            if not all(key) or key in waveform_entries:
                raise ValueError("waveform identity is empty or duplicated")
            waveform_entries[key] = entry
    target = output_directory.resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    index_rows: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    used_language_keys: set[tuple[str, str]] = set()
    try:
        for scope, title_scope, path in cohorts:
            destination = staging / scope
            destination.mkdir()
            for record in _read_jsonl(path):
                unit_id, patient_id = _validate_record(record)
                language_record = language_records.get((scope, unit_id))
                if language_source is not None and language_record is None:
                    raise ValueError(f"missing constrained language record: {scope}/{unit_id}")
                action = str(_object(record["localization"], name="localization")["action"])
                relative = Path(scope) / f"{unit_id}.html"
                waveform_href = None
                waveform = record.get("waveform_figure")
                if waveform is not None:
                    if waveform_source is None or not isinstance(waveform, Mapping):
                        raise ValueError("report waveform requires a validated waveform directory")
                    entry = waveform_entries.get((scope, unit_id))
                    if entry is None or dict(waveform) != entry:
                        raise ValueError("report/waveform manifest binding mismatch")
                    figure_file = Path(_text(entry.get("figure_file"), name="figure_file"))
                    if figure_file.is_absolute() or ".." in figure_file.parts or figure_file.suffix.lower() != ".png":
                        raise ValueError("unsafe waveform figure path")
                    source_figure = (waveform_source / figure_file).resolve(strict=True)
                    source_figure.relative_to(waveform_source)
                    destination_figure = staging / "waveforms" / scope / f"{unit_id}.png"
                    destination_figure.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_figure, destination_figure)
                    waveform_href = f"../waveforms/{scope}/{unit_id}.png"
                    counts[f"{scope}_with_waveform"] += 1
                (staging / relative).write_text(
                    _render_report(
                        record,
                        title_scope=title_scope,
                        waveform_href=waveform_href,
                        language_record=language_record,
                    ),
                    encoding="utf-8",
                )
                if language_record is not None:
                    _, generator = _language_layer_html(language_record, record)
                    counts[str(generator)] += 1
                    used_language_keys.add((scope, unit_id))
                index_rows.append(
                    {"scope": scope, "unit_id": unit_id, "patient_id": patient_id, "action": action, "file": relative.as_posix()}
                )
                counts[f"{scope}_reports"] += 1
                counts[action] += 1
        if used_language_keys != set(language_records):
            raise ValueError("constrained language directory contains unmatched reports")
        (staging / "index.html").write_text(
            _render_index(index_rows, language_layer_loaded=language_source is not None),
            encoding="utf-8",
        )
        manifest: dict[str, object] = {
            "schema_version": OUTPUT_SCHEMA,
            "status": (
                "completed_clinician_readable_render_with_constrained_language"
                if language_source is not None
                else "completed_deterministic_clinician_readable_render"
            ),
            "source_manifest_schema": source_manifest["schema_version"],
            "counts": dict(sorted(counts.items())),
            "index": "index.html",
            "access_receipt": {
                "raw_eeg_loaded": False,
                "soz_targets_loaded": False,
                "evaluation_rows_loaded": False,
                "model_weights_loaded": False,
                "pre_rendered_target_blind_waveform_figures_loaded": waveform_source is not None,
                "scores_thresholds_or_decisions_changed": False,
                "hidden_abstained_ranking_exposed": False,
                "constrained_language_layer_loaded": language_source is not None,
                "llm_used_for_language_only": counts["qwen3.6_constrained_language_only"] > 0,
                "llm_used_for_soz_prediction": False,
                "llm_used": counts["qwen3.6_constrained_language_only"] > 0,
            },
            "boundary": "research_soz_candidate_support_not_confirmed_diagnosis_or_surgical_target",
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, target)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--input-directory", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--waveform-directory", type=Path)
    parser.add_argument("--constrained-llm-directory", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = render(
        args.input_directory,
        args.output_directory,
        args.waveform_directory,
        args.constrained_llm_directory,
    )
    print(json.dumps({"output": str(args.output_directory), **result["counts"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
