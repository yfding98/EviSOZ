"""Fact-locked LLM narration for SOZ candidate research reports.

The LLM may reorganize and paraphrase already materialized report facts and
may explain general concepts using a curated knowledge subset.  It never sees
SOZ targets or evaluation rows and cannot change localization decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Mapping, Sequence
import urllib.error
import urllib.parse
import urllib.request

from code.auto_annotate.eeg_knowledge_base import (
    EEGKnowledgeBase,
    EEGKnowledgePassage,
    load_eeg_knowledge_base,
)
from src.soz.geometry import STANDARD_19


SOURCE_REPORT_SCHEMA = "trustworthy_soz_clinical_reference_report_v32"
OUTPUT_SCHEMA = "trustworthy_soz_constrained_llm_narrative_v1"
LLM_PAYLOAD_SCHEMA = "trustworthy_soz_constrained_llm_payload_v1"
POLICY_SCHEMA = "constrained_llm_reporting_policy_v1"

SECTION_ORDER = (
    "case_scope",
    "waveform_review",
    "localization_reference",
    "uncertainty_and_boundary",
)
SECTION_FACT_TYPES = {
    "case_scope": ("analysis_scope",),
    "waveform_review": ("waveform_observation",),
    "localization_reference": ("localization_result", "reference_opinion"),
    "uncertainty_and_boundary": ("evidence_applicability", "clinical_boundary"),
}
SECTION_HEADINGS = {
    "case_scope": "分析范围",
    "waveform_review": "波形复核要点",
    "localization_reference": "定位参考",
    "uncertainty_and_boundary": "证据边界",
}

REGION_TERMS = (
    "左额区",
    "右额区",
    "左颞区",
    "右颞区",
    "左顶区",
    "右顶区",
    "枕区",
    "中央中线区",
    "中央区",
    "中线区",
)
CHANNEL_RE = re.compile(
    r"(?<![A-Z0-9])(" + "|".join(sorted(STANDARD_19, key=len, reverse=True)) + r")(?![A-Z0-9])",
    re.I,
)
NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?")
SENTENCE_SPLIT_RE = re.compile(r"[。！？；]")
FORBIDDEN_ASSERTIONS = (
    re.compile(r"(?:确认|证实|确定).{0,10}(?:皮层)?SOZ"),
    re.compile(r"(?:确认|证实|确定).{0,10}(?:致痫区|手术靶点)"),
    re.compile(r"(?:皮层SOZ|致痫区|手术靶点)(?:位于|为)"),
    re.compile(r"由(?:FP1|FP2|F7|F8|F3|F4|FZ|T7|T8|C3|C4|CZ|P7|P8|P3|P4|PZ|O1|O2).{0,8}传播至", re.I),
    re.compile(r"传播路径(?:为|是)"),
    re.compile(r"模型(?:因为|由于).{0,40}(?:选择|判断|定位)"),
    re.compile(r"可(?:直接|独立)用于(?:诊断|手术|治疗)"),
)
PATIENT_SENSITIVE_TERMS = (
    "演变",
    "传播",
    "起始",
    "起源",
    "发作类型",
    "意识",
    "症状",
    "影像",
    "棘波",
    "尖波",
    "尖慢波",
    "低电压快活动",
    "节律",
    "频段",
    "伪迹",
    "病灶",
    "病因",
    "解剖",
    "通道",
    "诊断",
    "治疗",
    "手术",
    "致痫区",
    "皮层SOZ",
)


@dataclass(frozen=True)
class ReportingKnowledge:
    base: EEGKnowledgeBase
    policy_sha256: str
    passages: tuple[EEGKnowledgePassage, ...]
    authority_by_id: Mapping[str, str]
    allowed_use_by_id: Mapping[str, str]


def _canonical_sha256(value: Mapping[str, object]) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def load_reporting_knowledge(
    knowledge_path: Path,
    policy_path: Path,
) -> tuple[ReportingKnowledge, dict[str, object]]:
    policy = _read_json(policy_path)
    if policy.get("schema_version") != POLICY_SCHEMA:
        raise ValueError("constrained LLM policy schema drifted")
    base = load_eeg_knowledge_base(knowledge_path)
    by_id = {passage.id: passage for passage in base.passages}
    allowed = policy.get("allowed_sources")
    if not isinstance(allowed, list) or not allowed:
        raise TypeError("policy allowed_sources must be a non-empty list")
    passages: list[EEGKnowledgePassage] = []
    authority: dict[str, str] = {}
    allowed_use: dict[str, str] = {}
    for raw in allowed:
        if not isinstance(raw, dict):
            raise TypeError("allowed source entry must be an object")
        source_id = str(raw.get("id", "")).strip()
        if source_id not in by_id or source_id in authority:
            raise ValueError(f"unknown or duplicate reporting source: {source_id}")
        authority[source_id] = str(raw.get("authority", "")).strip()
        allowed_use[source_id] = str(raw.get("allowed_use", "")).strip()
        if not authority[source_id] or not allowed_use[source_id]:
            raise ValueError(f"reporting source metadata incomplete: {source_id}")
        passages.append(by_id[source_id])
    required = policy.get("required_source_ids")
    if not isinstance(required, list) or not set(map(str, required)).issubset(authority):
        raise ValueError("required reporting knowledge is missing from the whitelist")
    policy_sha = hashlib.sha256(policy_path.resolve(strict=True).read_bytes()).hexdigest()
    return (
        ReportingKnowledge(
            base=base,
            policy_sha256=policy_sha,
            passages=tuple(passages),
            authority_by_id=authority,
            allowed_use_by_id=allowed_use,
        ),
        policy,
    )


def _char_bigrams(text: str) -> set[str]:
    compact = re.sub(r"\s+", "", text.lower())
    return {compact[index : index + 2] for index in range(max(0, len(compact) - 1))}


def select_reporting_knowledge(
    knowledge: ReportingKnowledge,
    report: Mapping[str, object],
    *,
    max_sources: int,
    required_source_ids: Sequence[str],
) -> tuple[EEGKnowledgePassage, ...]:
    if max_sources < len(required_source_ids):
        raise ValueError("max_sources is below required source count")
    query = _char_bigrams(str(report.get("clinical_summary_zh", "")))

    def score(passage: EEGKnowledgePassage) -> tuple[int, int, str]:
        body = " ".join(
            [
                passage.title,
                passage.summary_zh,
                *passage.application_rules,
                *passage.limitations,
            ]
        )
        overlap = len(query.intersection(_char_bigrams(body)))
        cohort_bonus = int(
            "public" in str(report.get("cohort", "")).lower()
            and passage.id == "tusz-shah-2018"
        )
        return overlap + 8 * cohort_bonus, passage.year, passage.id

    by_id = {passage.id: passage for passage in knowledge.passages}
    selected = [by_id[source_id] for source_id in required_source_ids]
    for passage in sorted(knowledge.passages, key=score, reverse=True):
        if passage not in selected:
            selected.append(passage)
        if len(selected) >= max_sources:
            break
    return tuple(selected)


def build_fact_inventory(report: Mapping[str, object]) -> tuple[dict[str, object], ...]:
    if report.get("schema_version") != SOURCE_REPORT_SCHEMA:
        raise ValueError("source clinical report schema drifted")
    clauses = report.get("clauses")
    if not isinstance(clauses, list) or not clauses:
        raise TypeError("source clinical report has no clauses")
    facts: list[dict[str, object]] = []
    seen_types: set[str] = set()
    for index, raw in enumerate(clauses, start=1):
        if not isinstance(raw, dict):
            raise TypeError("report clause must be an object")
        fact_type = str(raw.get("type", "")).strip()
        text = str(raw.get("text", "")).strip()
        paths = raw.get("fact_paths")
        if not fact_type or fact_type in seen_types or not text:
            raise ValueError("report clause type/text is empty or duplicated")
        if not isinstance(paths, list) or not paths or not all(isinstance(item, str) for item in paths):
            raise TypeError("report clause fact_paths are invalid")
        seen_types.add(fact_type)
        facts.append(
            {
                "fact_id": f"F{index}",
                "fact_type": fact_type,
                "text_zh": text,
                "fact_paths": list(paths),
            }
        )
    expected = {item for values in SECTION_FACT_TYPES.values() for item in values}
    if seen_types != expected:
        raise ValueError(f"clinical fact types drifted: {sorted(seen_types)}")
    return tuple(facts)


def _knowledge_payload(
    passages: Sequence[EEGKnowledgePassage],
    knowledge: ReportingKnowledge,
) -> list[dict[str, object]]:
    return [
        {
            "source_id": passage.id,
            "authority": knowledge.authority_by_id[passage.id],
            "allowed_use": knowledge.allowed_use_by_id[passage.id],
            "citation": passage.citation,
            "summary_zh": passage.summary_zh,
            "application_rules": list(passage.application_rules),
            "limitations": list(passage.limitations),
        }
        for passage in passages
    ]


def build_llm_request(
    report: Mapping[str, object],
    facts: Sequence[Mapping[str, object]],
    passages: Sequence[EEGKnowledgePassage],
    knowledge: ReportingKnowledge,
) -> tuple[str, str]:
    localization = report.get("localization")
    if not isinstance(localization, Mapping):
        raise TypeError("report localization is missing")
    displayed = localization.get("displayed_candidates")
    if not isinstance(displayed, list):
        raise TypeError("displayed candidates must be a list")
    candidates = [str(item["channel"]) for item in displayed if isinstance(item, Mapping)]
    contract = {
        "schema_version": LLM_PAYLOAD_SCHEMA,
        "unit_id": str(report["unit_id"]),
        "patient_id": str(report["patient_id"]),
        "localization_action": str(localization["action"]),
        "candidate_channels_must_equal": candidates,
        "top1_region_zh_must_equal": localization.get("top1_region_projection_zh"),
        "section_order": list(SECTION_ORDER),
        "section_fact_contract": {
            section: [
                str(fact["fact_id"])
                for fact in facts
                if fact["fact_type"] in SECTION_FACT_TYPES[section]
            ]
            for section in SECTION_ORDER
        },
        "safety_acknowledgements_must_equal": {
            "patient_facts_added": False,
            "soz_prediction_changed": False,
            "diagnosis_generated": False,
            "treatment_recommendation_generated": False,
        },
    }
    system = (
        "你是癫痫头皮脑电科研报告的受约束临床叙述器，不是SOZ预测器，也不是诊断医生。"
        "患者级事实只能来自LOCKED_FACTS；KNOWLEDGE只能解释一般医学原则，不能推断本患者的新通道、"
        "时间、区域、形态、节律、伪迹、传播、症状、影像或治疗事实。不得修改候选顺序、弃权状态或区域。"
        "每个sections元素必须只引用合同指定的fact_ids；knowledge_notes必须引用允许的source_ids。"
        "不得建立波形观察导致定位分数的因果关系。输出简洁、专业、适合医生复核。"
        "患者章节中涉及演变、传播、起始、形态、节律、伪迹、症状、影像、解剖定位或治疗的词，"
        "只有在该节绑定LOCKED_FACTS原文已经出现时才可使用；不要扩写SOZ缩写。"
        "只输出一个符合指定结构的JSON对象，不要Markdown，不要思维过程。"
    )
    user = json.dumps(
        {
            "TASK": "在不新增事实的前提下改善表达、解释和信息组织",
            "OUTPUT_CONTRACT": contract,
            "LOCKED_FACTS": list(facts),
            "AUTHORIZED_KNOWLEDGE": _knowledge_payload(passages, knowledge),
            "OUTPUT_EXAMPLE_SHAPE": {
                "schema_version": LLM_PAYLOAD_SCHEMA,
                "unit_id": contract["unit_id"],
                "patient_id": contract["patient_id"],
                "localization_action": contract["localization_action"],
                "candidate_channels": candidates,
                "top1_region_zh": contract["top1_region_zh_must_equal"],
                "sections": [
                    {
                        "section_id": section,
                        "heading_zh": SECTION_HEADINGS[section],
                        "text_zh": "只复述该section获授权fact_ids中的患者事实",
                        "fact_ids": contract["section_fact_contract"][section],
                    }
                    for section in SECTION_ORDER
                ],
                "knowledge_notes": [
                    {
                        "text_zh": "不含患者特异事实的一般医学解释",
                        "source_ids": [passages[0].id],
                    }
                ],
                "safety_acknowledgements": contract["safety_acknowledgements_must_equal"],
            },
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return system, user


def llm_json_schema() -> dict[str, object]:
    section_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["section_id", "heading_zh", "text_zh", "fact_ids"],
        "properties": {
            "section_id": {"type": "string", "enum": list(SECTION_ORDER)},
            "heading_zh": {"type": "string", "maxLength": 24},
            "text_zh": {"type": "string", "minLength": 1, "maxLength": 500},
            "fact_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 2},
        },
    }
    note_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["text_zh", "source_ids"],
        "properties": {
            "text_zh": {"type": "string", "minLength": 1, "maxLength": 300},
            "source_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 2},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "unit_id",
            "patient_id",
            "localization_action",
            "candidate_channels",
            "top1_region_zh",
            "sections",
            "knowledge_notes",
            "safety_acknowledgements",
        ],
        "properties": {
            "schema_version": {"type": "string", "const": LLM_PAYLOAD_SCHEMA},
            "unit_id": {"type": "string"},
            "patient_id": {"type": "string"},
            "localization_action": {
                "type": "string",
                "enum": ["display_candidate", "localization_abstain", "localization_unavailable"],
            },
            "candidate_channels": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
            "top1_region_zh": {"type": ["string", "null"]},
            "sections": {"type": "array", "items": section_schema, "minItems": 4, "maxItems": 4},
            "knowledge_notes": {"type": "array", "items": note_schema, "minItems": 1, "maxItems": 2},
            "safety_acknowledgements": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "patient_facts_added",
                    "soz_prediction_changed",
                    "diagnosis_generated",
                    "treatment_recommendation_generated",
                ],
                "properties": {
                    "patient_facts_added": {"type": "boolean", "const": False},
                    "soz_prediction_changed": {"type": "boolean", "const": False},
                    "diagnosis_generated": {"type": "boolean", "const": False},
                    "treatment_recommendation_generated": {"type": "boolean", "const": False},
                },
            },
        },
    }


def _channels(text: str) -> set[str]:
    return {match.upper() for match in CHANNEL_RE.findall(text.upper())}


def _numbers(text: str) -> set[float]:
    result: set[float] = set()
    for token in NUMBER_RE.findall(text):
        value = float(token)
        if math.isfinite(value):
            result.add(round(value, 6))
    return result


def _regions(text: str) -> set[str]:
    return {term for term in REGION_TERMS if term in text}


def _validate_no_unsupported_surface(text: str, support_text: str, *, name: str) -> None:
    if not text.strip() or len(text) > 500 or "<think>" in text.lower() or "```" in text:
        raise ValueError(f"{name} is empty, too long, or contains non-report markup")
    if not _channels(text).issubset(_channels(support_text)):
        raise ValueError(f"{name} introduces an unsupported electrode")
    if not _numbers(text).issubset(_numbers(support_text)):
        raise ValueError(f"{name} introduces an unsupported numeric fact")
    if not _regions(text).issubset(_regions(support_text)):
        raise ValueError(f"{name} introduces an unsupported region")
    for pattern in FORBIDDEN_ASSERTIONS:
        if pattern.search(text):
            raise ValueError(f"{name} contains a forbidden clinical assertion")


def _validate_sensitive_patient_terms(text: str, support_text: str, *, name: str) -> None:
    introduced = [
        term for term in PATIENT_SENSITIVE_TERMS if term in text and term not in support_text
    ]
    if introduced:
        raise ValueError(f"{name} introduces an unsupported sensitive term: {introduced}")


def validate_llm_payload(
    payload: Mapping[str, object],
    report: Mapping[str, object],
    facts: Sequence[Mapping[str, object]],
    passages: Sequence[EEGKnowledgePassage],
) -> dict[str, object]:
    expected_payload_keys = {
        "schema_version",
        "unit_id",
        "patient_id",
        "localization_action",
        "candidate_channels",
        "top1_region_zh",
        "sections",
        "knowledge_notes",
        "safety_acknowledgements",
    }
    if set(payload) != expected_payload_keys:
        raise ValueError("LLM payload keys drifted")
    localization = report.get("localization")
    if not isinstance(localization, Mapping):
        raise TypeError("source localization is missing")
    expected_candidates = [
        str(item["channel"])
        for item in localization.get("displayed_candidates", [])
        if isinstance(item, Mapping)
    ]
    exact = {
        "schema_version": LLM_PAYLOAD_SCHEMA,
        "unit_id": str(report["unit_id"]),
        "patient_id": str(report["patient_id"]),
        "localization_action": str(localization["action"]),
        "candidate_channels": expected_candidates,
        "top1_region_zh": localization.get("top1_region_projection_zh"),
    }
    for key, value in exact.items():
        if payload.get(key) != value:
            raise ValueError(f"LLM changed locked field: {key}")
    expected_safety = {
        "patient_facts_added": False,
        "soz_prediction_changed": False,
        "diagnosis_generated": False,
        "treatment_recommendation_generated": False,
    }
    if payload.get("safety_acknowledgements") != expected_safety:
        raise ValueError("LLM safety acknowledgements drifted")
    facts_by_id = {str(fact["fact_id"]): fact for fact in facts}
    type_to_id = {str(fact["fact_type"]): str(fact["fact_id"]) for fact in facts}
    sections = payload.get("sections")
    if not isinstance(sections, list) or len(sections) != len(SECTION_ORDER):
        raise ValueError("LLM section count drifted")
    for expected_section, raw in zip(SECTION_ORDER, sections, strict=True):
        if not isinstance(raw, Mapping) or raw.get("section_id") != expected_section:
            raise ValueError("LLM section order drifted")
        if set(raw) != {"section_id", "heading_zh", "text_zh", "fact_ids"}:
            raise ValueError("LLM section keys drifted")
        if raw.get("heading_zh") != SECTION_HEADINGS[expected_section]:
            raise ValueError("LLM section heading drifted")
        expected_ids = [type_to_id[item] for item in SECTION_FACT_TYPES[expected_section]]
        if raw.get("fact_ids") != expected_ids:
            raise ValueError("LLM section fact contract drifted")
        support = "。".join(str(facts_by_id[fact_id]["text_zh"]) for fact_id in expected_ids)
        section_text = raw.get("text_zh")
        if not isinstance(section_text, str):
            raise TypeError("LLM section text must be a string")
        _validate_no_unsupported_surface(section_text, support, name=f"section:{expected_section}")
        _validate_sensitive_patient_terms(
            section_text, support, name=f"section:{expected_section}"
        )
    allowed_knowledge = {passage.id: passage for passage in passages}
    notes = payload.get("knowledge_notes")
    if not isinstance(notes, list) or not 1 <= len(notes) <= 2:
        raise ValueError("LLM knowledge note count drifted")
    for index, raw in enumerate(notes):
        if not isinstance(raw, Mapping):
            raise TypeError("LLM knowledge note must be an object")
        if set(raw) != {"text_zh", "source_ids"}:
            raise ValueError("LLM knowledge note keys drifted")
        source_ids = raw.get("source_ids")
        if not isinstance(source_ids, list) or not source_ids or not all(
            isinstance(item, str) and item in allowed_knowledge for item in source_ids
        ):
            raise ValueError("LLM cited unauthorized knowledge")
        if len(source_ids) > 2 or len(source_ids) != len(set(source_ids)):
            raise ValueError("LLM knowledge source count or uniqueness drifted")
        text = raw.get("text_zh")
        if not isinstance(text, str):
            raise TypeError("LLM knowledge note text must be a string")
        _validate_no_unsupported_surface(text, "", name=f"knowledge_note:{index}")
        source_text = " ".join(
            " ".join(
                [
                    allowed_knowledge[source_id].summary_zh,
                    *allowed_knowledge[source_id].application_rules,
                    *allowed_knowledge[source_id].limitations,
                ]
            )
            for source_id in source_ids
        )
        note_bigrams = _char_bigrams(text)
        if note_bigrams and len(note_bigrams.intersection(_char_bigrams(source_text))) / len(note_bigrams) < 0.08:
            raise ValueError("LLM knowledge note has insufficient lexical support")
    return json.loads(json.dumps(payload, ensure_ascii=False))


def deterministic_fallback_payload(
    report: Mapping[str, object],
    facts: Sequence[Mapping[str, object]],
    passages: Sequence[EEGKnowledgePassage],
) -> dict[str, object]:
    localization = report["localization"]
    if not isinstance(localization, Mapping):
        raise TypeError("source localization is missing")
    by_type = {str(fact["fact_type"]): fact for fact in facts}
    sections = []
    for section in SECTION_ORDER:
        section_facts = [by_type[item] for item in SECTION_FACT_TYPES[section]]
        sections.append(
            {
                "section_id": section,
                "heading_zh": SECTION_HEADINGS[section],
                "text_zh": "。".join(str(fact["text_zh"]).rstrip("。") for fact in section_facts) + "。",
                "fact_ids": [str(fact["fact_id"]) for fact in section_facts],
            }
        )
    boundary = next(passage for passage in passages if passage.id == "lueders-epileptogenic-zone-2006")
    return {
        "schema_version": LLM_PAYLOAD_SCHEMA,
        "unit_id": str(report["unit_id"]),
        "patient_id": str(report["patient_id"]),
        "localization_action": str(localization["action"]),
        "candidate_channels": [
            str(item["channel"])
            for item in localization.get("displayed_candidates", [])
            if isinstance(item, Mapping)
        ],
        "top1_region_zh": localization.get("top1_region_projection_zh"),
        "sections": sections,
        "knowledge_notes": [
            {
                "text_zh": boundary.summary_zh,
                "source_ids": [boundary.id],
            }
        ],
        "safety_acknowledgements": {
            "patient_facts_added": False,
            "soz_prediction_changed": False,
            "diagnosis_generated": False,
            "treatment_recommendation_generated": False,
        },
    }


def call_local_qwen_chat(
    *,
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    enable_thinking: bool,
    timeout_seconds: float,
    retries: int,
) -> tuple[dict[str, object], dict[str, object]]:
    parsed = urllib.parse.urlparse(base_url)
    if (parsed.hostname or "").lower() not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("constrained SOZ narration only permits a local LLM endpoint")
    endpoint = base_url.rstrip("/")
    if endpoint.endswith("/v1"):
        endpoint += "/chat/completions"
    elif not endpoint.endswith("/chat/completions"):
        endpoint += "/v1/chat/completions"
    request_payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        # Qwen3.6 otherwise spends the constrained output budget in the
        # separate reasoning field and may return an empty publishable body.
        # Private chain-of-thought is neither needed nor retained here.
        "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": LLM_PAYLOAD_SCHEMA,
                "strict": True,
                "schema": llm_json_schema(),
            },
        },
    }
    body = json.dumps(request_payload, ensure_ascii=False).encode("utf-8")
    last_error: Exception | None = None
    for attempt in range(max(0, int(retries)) + 1):
        request = urllib.request.Request(
            endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=float(timeout_seconds)) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
            choices = response_payload.get("choices")
            if not isinstance(choices, list) or not choices:
                raise RuntimeError("local Qwen response has no choices")
            message = choices[0].get("message")
            if not isinstance(message, Mapping):
                raise RuntimeError("local Qwen response has no message")
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise RuntimeError("local Qwen response content is empty")
            parsed_payload = json.loads(content)
            if not isinstance(parsed_payload, dict):
                raise TypeError("local Qwen response is not a JSON object")
            metadata = {
                "id": response_payload.get("id"),
                "model": response_payload.get("model", model),
                "usage": response_payload.get("usage"),
                "finish_reason": choices[0].get("finish_reason"),
                "attempt": attempt + 1,
                "endpoint_host": parsed.hostname,
            }
            return parsed_payload, metadata
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, RuntimeError, TypeError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(4.0, 2.0**attempt))
    raise RuntimeError(f"local Qwen request failed: {last_error}")


def build_augmented_record(
    *,
    report: Mapping[str, object],
    facts: Sequence[Mapping[str, object]],
    passages: Sequence[EEGKnowledgePassage],
    knowledge: ReportingKnowledge,
    candidate_payload: Mapping[str, object] | None,
    model_metadata: Mapping[str, object] | None,
    generation_error: str | None,
) -> dict[str, object]:
    fallback_reason: str | None = None
    validation_errors: list[str] = []
    if candidate_payload is None:
        fallback_reason = generation_error or "llm_not_called"
        published = deterministic_fallback_payload(report, facts, passages)
        generator = "deterministic_fallback"
    else:
        try:
            published = validate_llm_payload(candidate_payload, report, facts, passages)
            generator = "qwen3.6_constrained_language_only"
        except (TypeError, ValueError) as exc:
            validation_errors.append(f"{type(exc).__name__}: {exc}")
            fallback_reason = "llm_validation_failed"
            published = deterministic_fallback_payload(report, facts, passages)
            generator = "deterministic_fallback"
    localization = report.get("localization")
    if not isinstance(localization, Mapping):
        raise TypeError("source localization is missing")
    return {
        "schema_version": OUTPUT_SCHEMA,
        "unit_id": report["unit_id"],
        "patient_id": report["patient_id"],
        "cohort": report["cohort"],
        "source_report_schema": report["schema_version"],
        "source_report_sha256": _canonical_sha256(report),
        "localization": json.loads(json.dumps(localization, ensure_ascii=False)),
        "fact_inventory": list(facts),
        "knowledge_receipt": {
            "knowledge_base_sha256": knowledge.base.sha256,
            "policy_sha256": knowledge.policy_sha256,
            "source_ids": [passage.id for passage in passages],
            "citations": {passage.id: passage.citation for passage in passages},
        },
        "published_narrative": published,
        "generation": {
            "generator": generator,
            # The publishable artifact never retains an unvalidated draft. A
            # canonical hash is sufficient to audit replay without allowing a
            # rejected hallucination to become a second patient-fact surface.
            "llm_candidate_retained_for_audit": False,
            "llm_candidate_sha256": (
                _canonical_sha256(candidate_payload) if candidate_payload is not None else None
            ),
            "llm_candidate_payload": None,
            "model_metadata": dict(model_metadata or {}),
            "generation_error": generation_error,
            "validation_errors": validation_errors,
            "fallback_reason": fallback_reason,
        },
        "access_receipt": {
            "raw_eeg_loaded": False,
            "soz_gold_labels_loaded": False,
            "evaluation_rows_loaded": False,
            "model_scores_or_localization_changed": False,
            "patient_facts_added": False,
            "llm_used_for_language_only": candidate_payload is not None,
        },
    }
