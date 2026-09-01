"""Facts-locked Qwen narration for ``clinical_eeg_report_v1``.

This module is deliberately independent from ``src.soz``.  It consumes a
validated clinical EEG fact ledger, sends only de-identified facts to a local
OpenAI-compatible endpoint, validates every generated surface, and falls back
to deterministic text on any error.  It never reads raw EEG and never signs a
clinical report.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Mapping, Sequence
import urllib.error
import urllib.parse
import urllib.request

from .style import ClinicalEEGStyleProfile
from .schema import (
    CANONICAL_ELECTRODES,
    FACT_TYPE_LABEL_ZH,
    FACT_TYPE_TO_SECTION,
    validate_report_payload,
)


NARRATIVE_SCHEMA = "clinical_eeg_report_narrative_v1"
PIPELINE_RECORD_SCHEMA = "clinical_eeg_report_pipeline_record_v1"
POLICY_SCHEMA = "clinical_eeg_report_policy_v1"

FINDING_SECTION_ORDER = (
    "background",
    "interictal",
)
FINDING_FACT_SECTIONS = {
    "background": {"background"},
    "interictal": {"interictal"},
}
EEG_EVENT_COLUMN_ORDER = (
    "onset",
    "evolution_spread",
    "termination_postictal",
)
EEG_EVENT_FACT_TYPES = {
    "onset": {"ictal_onset_pattern"},
    "evolution_spread": {
        "algorithmic_sustained_eeg_change",
        "later_scalp_visible_eeg_change",
        "ictal_evolution",
        "ictal_spread",
    },
    "termination_postictal": {"ictal_termination", "postictal_pattern"},
}
# ``source_eeg_annotation_timing`` is retained in the legacy input schema so
# already materialized ledgers can still be validated and migrated.  It is an
# audit/label-side fact, not an EEG-signal fact, and is removed from the active
# generation view before aliases, prompts, narrative coverage, source hashes or
# renderers see the report.  Keeping this set separate from deterministic
# layout facts prevents a future renderer from interpreting "not sent to the
# LLM" as permission to display it in the clinical body.
AUDIT_ONLY_FACT_TYPES = frozenset({"source_eeg_annotation_timing"})
DETERMINISTIC_LAYOUT_FACT_TYPES = frozenset(
    {"electrographic_event_occurrence"}
)
IMPRESSION_SECTION_ORDER = (
    "overall",
    "interictal",
    "ictal",
    "limitations",
)
# Missing workflow fields are transport state, not clinical observations.  New
# narratives therefore encode an unsupported block as an empty string and the
# renderer omits it.  The legacy spellings remain accepted at the validation
# boundary so already frozen language records can be safely re-rendered; they
# are never projected into a new clinical surface.
EMPTY_FINDING_TEXT = {"background": "", "interictal": ""}
EMPTY_EVENT_TEXT = {
    "onset": "",
    "evolution_spread": "",
    "termination_postictal": "",
}
EMPTY_IMPRESSION_TEXT = ""

LEGACY_EMPTY_FINDING_TEXT = {
    "background": "未提供可表述的背景活动结构化事实，待临床脑电医师补充。",
    "interictal": "未提供可表述的发作间期结构化事实，待临床脑电医师补充。",
}
LEGACY_EMPTY_EVENT_TEXT = {
    "onset": "未提供该脑电事件的起始结构化事实。",
    "evolution_spread": "未提供该脑电事件的演变或空间扩展结构化事实。",
    "termination_postictal": "未提供该脑电事件的终止或事件后脑电结构化事实。",
}
LEGACY_EMPTY_IMPRESSION_TEXT = "脑电图印象尚待临床脑电医师审核确认。"

_NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?")
_LLM_ALIAS_RE = re.compile(r"(?<![A-Za-z0-9])(?:LLMFACT\d{4}|EV\d+)(?![A-Za-z0-9])", re.IGNORECASE)
_CHANNEL_LABELS = tuple(
    sorted(
        set(CANONICAL_ELECTRODES).union(
            {"A1", "A2", "T3", "T4", "T5", "T6", "SPHL", "SPHR"}
        ),
        key=lambda item: (-len(item), item),
    )
)
_CHANNEL_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    + "|".join(re.escape(label) for label in _CHANNEL_LABELS)
    + r")(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_REGION_TERMS = (
    "左额极区",
    "右额极区",
    "左额区",
    "右额区",
    "左中央区",
    "右中央区",
    "左前颞区",
    "右前颞区",
    "左中颞区",
    "右中颞区",
    "左后颞区",
    "右后颞区",
    "左颞区",
    "右颞区",
    "左顶区",
    "右顶区",
    "左枕区",
    "右枕区",
    "枕区",
    "中央区",
    "中线区",
)
_NON_EEG_CONTENT_RE = re.compile(
    r"(?:临床表现|临床事件|临床发作|临床相关|临床诊断|结合病史|"
    r"既往史|病史|用药|药物治疗|影像|MRI|年龄|性别|科室|转诊|检查原因|"
    r"视频(?:表现|事件|症状)|症状|意识(?:状态|受损|保留|丧失)|"
    r"行为|动作|自动症|呼之不应|惯常(?:事件|发作)|"
    r"心电(?!伪迹)|肌电(?!伪迹)|眼电|"
    r"睡眠|清醒|困倦|\bN[123]\b|\bREM\b|纺锤波?|K复合波|顶尖波|"
    r"诱发(?:试验|实验|反应|事件)|闪光刺激|过度换气|睁闭眼|睁眼|闭眼|"
    r"\bECG\b|\bEMG\b|\bEOG\b)",
    re.IGNORECASE,
)
_FORBIDDEN_ASSERTIONS = (
    re.compile(r"(?:确认|证实|确定).{0,12}(?:皮层)?SOZ", re.IGNORECASE),
    re.compile(r"(?:皮层SOZ|致痫区|手术靶点)(?:位于|为|确定)", re.IGNORECASE),
    re.compile(r"(?:建议|应当|需要).{0,12}(?:用药|停药|加药|减药|手术|治疗)"),
    re.compile(r"可(?:直接|独立)用于(?:诊断|治疗|手术)"),
    re.compile(r"(?:患者姓名|病历号|住院号|床号|联系电话)"),
    _NON_EEG_CONTENT_RE,
    re.compile(r"(?:支持|符合|提示).{0,10}(?:癫痫诊断|临床诊断|临床发作)"),
)
_MISSING_STATE_PHRASES = {
    "not_recorded": ("未记录", "未提供", "无记录"),
    "not_assessable": ("无法评估", "不能评估", "不可评估"),
    "uncertain": ("不确定", "尚不明确", "待复核", "可能"),
}
_NEGATIVE_CERTAINTY_PHRASES = ("正常", "未见异常", "无异常")
_CANDIDATE_PHRASES = ("候选", "算法", "待复核", "技术记录", "初步")
_NEUTRAL_TEMPORAL_FACT_TYPES = {
    "algorithmic_sustained_eeg_change",
    "later_scalp_visible_eeg_change",
}
_NEUTRAL_ONSET_PROMOTION_RE = re.compile(
    r"(?:发作(?:期(?:脑电)?)?起始|脑电(?:发作)?起始|"
    r"(?:发作期|脑电)(?:变化)?起点|临床(?:确认的?)?起始|"
    r"(?:皮层|癫痫灶|SOZ)(?:起始|起点|起源)|最早(?:头皮可见|电极|导联)|"
    r"(?:^|[^A-Za-z0-9])(?:onset|origin)(?:$|[^A-Za-z0-9]))",
    re.IGNORECASE,
)
_NEUTRAL_PROPAGATION_PROMOTION_RE = re.compile(
    r"(?:传播|扩散|蔓延|空间扩展|(?:^|[^A-Za-z0-9])(?:propagation|spread)"
    r"(?:$|[^A-Za-z0-9]))",
    re.IGNORECASE,
)
_DIRECT_IDENTIFIER_PATTERNS = (
    re.compile(
        r"(?:患者姓名|姓名|病历号|住院号|门诊号|身份证号?|手机号|联系电话|床号)"
        r"\s*[:：=]\s*[^\s，。；,;]{1,80}"
    ),
    re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    re.compile(r"(?<!\d)\d{17}[0-9Xx](?!\d)"),
    re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    re.compile(r"(?<!\d)(?:19|20)\d{2}[-/.](?:0?[1-9]|1[0-2])[-/.](?:0?[1-9]|[12]\d|3[01])(?!\d)"),
)


def eeg_only_scope_receipt() -> dict[str, Any]:
    """Return the fixed, machine-auditable scope declaration for v1."""

    return {
        "contract_id": "current_soz_scalp_eeg_only",
        "input_scope": "current_soz_processed_scalp_eeg_and_acquisition_metadata_only",
        "generated_fact_sections": ["metadata", "background", "interictal", "ictal", "impression"],
        "clinical_information_mode": "omitted_not_available_to_pipeline",
        "activation_mode": "omitted_not_available_to_pipeline",
        "sleep_mode": "omitted_not_available_to_pipeline",
        "video_used": False,
        "ecg_used": False,
        "emg_used": False,
        "eog_used": False,
        "edf_annotation_used": False,
        "spreadsheet_observation_used": False,
        "physician_ground_truth_used": False,
        "audit_only_fact_types_excluded_before_generation": sorted(
            AUDIT_ONLY_FACT_TYPES
        ),
        "current_record_evidence_binding_verified": False,
        "evidence_binding_limitation": (
            "source/evidence IDs are scope-filtered, but v1 has no explicit "
            "evidence_modality or current-record binding"
        ),
    }


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_policy(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != POLICY_SCHEMA:
        raise ValueError("unsupported clinical EEG report policy")
    generation = value.get("generation")
    fact_policy = value.get("fact_policy")
    safety = value.get("safety")
    deployment = value.get("deployment")
    if (
        not isinstance(generation, dict)
        or not isinstance(fact_policy, dict)
        or not isinstance(safety, dict)
        or not isinstance(deployment, dict)
    ):
        raise TypeError("clinical EEG report policy is incomplete")
    if generation.get("temperature") != 0.0 or generation.get("enable_thinking") is not False:
        raise ValueError("clinical report generation must be deterministic and non-thinking")
    if safety.get("validation_failure_action") != "deterministic_fallback":
        raise ValueError("unsafe clinical report failure action")
    if safety.get("llm_may_add_patient_facts") is not False:
        raise ValueError("LLM patient-fact mutation must remain disabled")
    if safety.get("llm_may_generate_non_eeg_content") is not False:
        raise ValueError("non-EEG narration must remain disabled")
    if safety.get("llm_may_generate_sleep_or_activation_content") is not False:
        raise ValueError("unsupported sleep/activation narration must remain disabled")
    if safety.get("non_eeg_input_action") != "schema_reject":
        raise ValueError("non-EEG inputs must fail closed at the schema boundary")
    if fact_policy.get("unsupported_sections_omitted_from_report") is not True:
        raise ValueError("unsupported SOZ report sections must be omitted")
    if fact_policy.get("deterministic_epistemic_qualifier_prefix") != "待复核候选：":
        raise ValueError("the only permitted deterministic narration repair drifted")
    for key in (
        "sleep_eeg_available_to_generator",
        "activation_experiment_available_to_generator",
        "clinical_information_available_to_generator",
    ):
        if fact_policy.get(key) is not False:
            raise ValueError(f"unsupported generator input was enabled: {key}")
    return value


def _report_dict(report: Any) -> dict[str, Any]:
    if hasattr(report, "to_dict"):
        value = report.to_dict()
    elif isinstance(report, Mapping):
        value = dict(report)
    else:
        raise TypeError("report must be a validated ClinicalEEGReport or mapping")
    if not isinstance(value, dict):
        raise TypeError("report serialization must be an object")
    value = validate_report_payload(value).to_dict()
    if value.get("schema_version") != "clinical_eeg_report_v1":
        raise ValueError("clinical EEG fact schema drifted")
    expected_keys = {
        "schema_version",
        "report_id",
        "patient_pseudonym",
        "facts",
        "eeg_event_ids",
        "impression_fact_ids",
    }
    if set(value) != expected_keys:
        legacy = sorted(
            set(value).intersection(
                {"case_context", "clinical_event_ids", "clinical_events", "ecg", "emg", "video"}
            )
        )
        if legacy:
            raise ValueError(f"non-EEG report fields are forbidden: {legacy}")
        raise ValueError("clinical EEG report envelope drifted")
    facts = value.get("facts")
    if not isinstance(facts, list) or not facts:
        raise ValueError("clinical EEG fact ledger is empty")
    for fact in facts:
        if not isinstance(fact, Mapping):
            raise TypeError("fact ledger entries must be objects")
        fact_type = str(fact.get("fact_type", ""))
        if fact_type not in FACT_TYPE_TO_SECTION:
            raise ValueError(f"non-EEG or unknown fact type is forbidden: {fact_type}")
    value["facts"] = [
        fact
        for fact in facts
        if str(fact.get("fact_type", "")) not in AUDIT_ONLY_FACT_TYPES
    ]
    # Revalidate the reduced view so references cannot become dangling when a
    # legacy audit-only fact is quarantined.  The original input file/hash can
    # still be retained by the outer materialization manifest for audit.
    value = validate_report_payload(value).to_dict()
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def eeg_only_generation_report_view(report: Any) -> dict[str, Any]:
    """Return the canonical signal-only report view used by all generators.

    The function deliberately validates the complete legacy envelope before
    removing audit-only fact types.  Malformed or prompt-injection-bearing
    annotation facts therefore still fail closed, while valid annotation
    values cannot influence fact aliases, prompts, narrative, source hashes,
    waveform bindings or rendered report content.
    """

    return _report_dict(report)


def _event_aliases(report: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    event_ids = report.get("eeg_event_ids")
    if not isinstance(event_ids, list) or not all(isinstance(item, str) and item for item in event_ids):
        raise TypeError("eeg_event_ids must be a string list")
    forward = {event_id: f"EV{index}" for index, event_id in enumerate(event_ids, start=1)}
    reverse = {alias: event_id for event_id, alias in forward.items()}
    return forward, reverse


def _fact_aliases(report: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    facts = report.get("facts")
    if not isinstance(facts, list):
        raise TypeError("facts must be a list")
    forward: dict[str, str] = {}
    for index, fact in enumerate(facts, start=1):
        if not isinstance(fact, Mapping):
            raise TypeError("fact ledger entries must be objects")
        fact_id = str(fact.get("fact_id", ""))
        if not fact_id or fact_id in forward:
            raise ValueError("fact IDs must be non-empty and unique")
        forward[fact_id] = f"LLMFACT{index:04d}"
    return forward, {alias: fact_id for fact_id, alias in forward.items()}


def _deidentify_fact_value(
    value: Any,
    fact_aliases: Mapping[str, str],
    event_aliases: Mapping[str, str],
) -> Any:
    if not isinstance(value, Mapping):
        return value
    result = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    for key in ("target_fact_ids", "abnormality_fact_ids", "supported_fact_ids"):
        raw = result.get(key)
        if isinstance(raw, list):
            result[key] = [fact_aliases[str(item)] for item in raw]
    raw_events = result.get("eeg_event_ids")
    if isinstance(raw_events, list):
        result["eeg_event_ids"] = [event_aliases[str(item)] for item in raw_events]
    return result


def _deidentified_fact_inventory(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    aliases, _ = _event_aliases(report)
    fact_aliases, _ = _fact_aliases(report)
    inventory: list[dict[str, Any]] = []
    for raw in report["facts"]:
        if not isinstance(raw, dict):
            raise TypeError("fact ledger entries must be objects")
        # Technique metadata and event timing are rendered deterministically.
        # They are deliberately withheld from the narrator so it cannot move
        # them into free text or turn missing workflow fields into findings.
        if (
            raw.get("section") == "metadata"
            or raw.get("fact_type") in DETERMINISTIC_LAYOUT_FACT_TYPES
        ):
            continue
        event_id = raw.get("eeg_event_id")
        verification = raw.get("verification")
        inventory.append(
            {
                "fact_id": fact_aliases[str(raw.get("fact_id"))],
                "section": raw.get("section"),
                "fact_type": raw.get("fact_type"),
                "state": raw.get("state"),
                "value": _deidentify_fact_value(raw.get("value"), fact_aliases, aliases),
                "verification_status": (
                    verification.get("status") if isinstance(verification, dict) else None
                ),
                "eeg_event_alias": aliases.get(str(event_id)) if event_id is not None else None,
            }
        )
    # The LLM never receives report_id, patient pseudonym, provenance source
    # identifiers, evidence paths, reviewer identity, or signature metadata.
    return inventory


def _private_tokens(report: Mapping[str, Any]) -> set[str]:
    """Collect envelope and audit identifiers that must not enter a prompt."""

    tokens = {
        str(report.get("report_id", "")),
        str(report.get("patient_pseudonym", "")),
    }
    for raw in report.get("eeg_event_ids", []):
        tokens.add(str(raw))
    for fact in report.get("facts", []):
        if not isinstance(fact, Mapping):
            continue
        tokens.add(str(fact.get("fact_id", "")))
        event_id = fact.get("eeg_event_id")
        if event_id is not None:
            tokens.add(str(event_id))
        provenance = fact.get("provenance")
        if isinstance(provenance, Mapping):
            tokens.add(str(provenance.get("source_id", "")))
        verification = fact.get("verification")
        if isinstance(verification, Mapping) and verification.get("verified_by") is not None:
            tokens.add(str(verification.get("verified_by")))
        for evidence_id in fact.get("evidence_ids", []):
            tokens.add(str(evidence_id))
    return {token for token in tokens if token}


def _string_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        result: list[str] = []
        for item in value.values():
            result.extend(_string_values(item))
        return result
    if isinstance(value, list):
        result = []
        for item in value:
            result.extend(_string_values(item))
        return result
    return []


def _assert_prompt_inventory_deidentified(
    inventory: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
) -> None:
    """Fail closed if free text reintroduces a known or obvious identifier.

    Explicit identifier fields are removed or aliased before this check.  The
    scan exists for accidental copies inside ``text_zh``/description fields;
    it is deliberately conservative because a deterministic report remains
    available when narration is blocked.
    """

    strings = _string_values(list(inventory))
    for token in sorted(_private_tokens(report), key=len, reverse=True):
        if any(
            text == token
            or (
                len(token) >= 4
                and re.search(
                    rf"(?<![A-Za-z0-9._:/-]){re.escape(token)}(?![A-Za-z0-9._:/-])",
                    text,
                )
            )
            for text in strings
        ):
            raise ValueError("de-identified fact inventory contains a protected internal identifier")
    surface = "\n".join(strings)
    if any(pattern.search(surface) for pattern in _DIRECT_IDENTIFIER_PATTERNS):
        raise ValueError("de-identified fact inventory contains a direct-identifier pattern")
    if _NON_EEG_CONTENT_RE.search(surface):
        raise ValueError("fact inventory contains content outside the EEG-only scope")


def _llm_fact_assignments(
    report: Mapping[str, Any],
    fact_aliases: Mapping[str, str],
    event_aliases: Mapping[str, str],
) -> tuple[
    dict[str, list[str]],
    dict[str, dict[str, list[str]]],
    dict[str, list[str]],
]:
    facts = [fact for fact in report["facts"] if isinstance(fact, Mapping)]
    findings = {
        section_id: [
            fact_aliases[str(fact["fact_id"])]
            for fact in facts
            if str(fact.get("section")) in FINDING_FACT_SECTIONS[section_id]
            and fact.get("eeg_event_id") is None
        ]
        for section_id in FINDING_SECTION_ORDER
    }
    events: dict[str, dict[str, list[str]]] = {}
    for event_id, alias in event_aliases.items():
        events[alias] = {
            column_id: [
                fact_aliases[str(fact["fact_id"])]
                for fact in facts
                if str(fact.get("eeg_event_id")) == event_id
                and str(fact.get("fact_type")) in EEG_EVENT_FACT_TYPES[column_id]
            ]
            for column_id in EEG_EVENT_COLUMN_ORDER
        }
    impression_type_sections = {
        "study_classification": "overall",
        "interictal_impression": "interictal",
        "ictal_eeg_impression": "ictal",
        "recording_limitation": "limitations",
    }
    approved = set(map(str, report.get("impression_fact_ids", [])))
    impression: dict[str, list[str]] = {key: [] for key in IMPRESSION_SECTION_ORDER}
    for fact in facts:
        fact_id = str(fact.get("fact_id"))
        if fact_id not in approved:
            continue
        section_id = impression_type_sections[str(fact.get("fact_type"))]
        impression[section_id].append(fact_aliases[fact_id])
    impression = {key: value for key, value in impression.items() if value}
    if not impression:
        impression = {"overall": []}
    return findings, events, impression


def _schema_choice(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(values) == 1:
        return dict(values[0])
    return {"oneOf": [dict(value) for value in values]}


def llm_json_schema(
    finding_fact_ids: Mapping[str, Sequence[str]],
    event_fact_ids: Mapping[str, Mapping[str, Sequence[str]]],
    impression_fact_ids: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    # vLLM 0.19.1 structured decoding does not implement ``uniqueItems``.
    # The publication validator below independently rejects duplicate IDs, so
    # omitting that grammar keyword changes compatibility, not the safety gate.
    finding_blocks = []
    for section_id in FINDING_SECTION_ORDER:
        ids = list(finding_fact_ids[section_id])
        text_schema = (
            {"type": "string", "const": EMPTY_FINDING_TEXT[section_id]}
            if not ids
            else {"type": "string", "minLength": 1, "maxLength": 1200}
        )
        finding_blocks.append(
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["section_id", "text_zh", "fact_ids"],
                "properties": {
                    "section_id": {"type": "string", "const": section_id},
                    "text_zh": text_schema,
                    "fact_ids": {"type": "array", "const": ids},
                },
            }
        )
    event_blocks = []
    for event_alias, columns in event_fact_ids.items():
        properties: dict[str, Any] = {
            "eeg_event_alias": {"type": "string", "const": event_alias},
        }
        for label, text_key, ids_key, max_length in (
            ("onset", "onset_text_zh", "onset_fact_ids", 1200),
            (
                "evolution_spread",
                "evolution_spread_text_zh",
                "evolution_spread_fact_ids",
                1600,
            ),
            (
                "termination_postictal",
                "termination_postictal_text_zh",
                "termination_postictal_fact_ids",
                1200,
            ),
        ):
            ids = list(columns[label])
            properties[text_key] = (
                {"type": "string", "const": EMPTY_EVENT_TEXT[label]}
                if not ids
                else {"type": "string", "minLength": 1, "maxLength": max_length}
            )
            properties[ids_key] = {"type": "array", "const": ids}
        event_blocks.append(
            {
                "type": "object",
                "additionalProperties": False,
                "required": list(properties),
                "properties": properties,
            }
        )
    impression_blocks = []
    for section_id, raw_ids in impression_fact_ids.items():
        ids = list(raw_ids)
        text_schema = (
            {"type": "string", "const": EMPTY_IMPRESSION_TEXT}
            if not ids
            else {"type": "string", "minLength": 1, "maxLength": 1200}
        )
        impression_blocks.append(
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["section_id", "text_zh", "fact_ids"],
                "properties": {
                    "section_id": {"type": "string", "const": section_id},
                    "text_zh": text_schema,
                    "fact_ids": {"type": "array", "const": ids},
                },
            }
        )
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "findings",
            "events",
            "impression",
            "safety_acknowledgements",
        ],
        "properties": {
            "schema_version": {"type": "string", "const": NARRATIVE_SCHEMA},
            "findings": {
                "type": "array",
                "items": _schema_choice(finding_blocks),
                "minItems": len(FINDING_SECTION_ORDER),
                "maxItems": len(FINDING_SECTION_ORDER),
            },
            "events": {
                "type": "array",
                "items": (
                    _schema_choice(event_blocks)
                    if event_blocks
                    else {"type": "object", "additionalProperties": False}
                ),
                "minItems": len(event_blocks),
                "maxItems": len(event_blocks),
            },
            "impression": {
                "type": "array",
                "items": _schema_choice(impression_blocks),
                "minItems": len(impression_blocks),
                "maxItems": len(impression_blocks),
            },
            "safety_acknowledgements": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "patient_facts_added",
                    "fact_states_changed",
                    "eeg_event_order_or_count_changed",
                    "non_eeg_content_generated",
                    "treatment_recommendation_generated",
                    "unverified_diagnosis_generated",
                ],
                "properties": {
                    "patient_facts_added": {"type": "boolean", "const": False},
                    "fact_states_changed": {"type": "boolean", "const": False},
                    "eeg_event_order_or_count_changed": {"type": "boolean", "const": False},
                    "non_eeg_content_generated": {"type": "boolean", "const": False},
                    "treatment_recommendation_generated": {"type": "boolean", "const": False},
                    "unverified_diagnosis_generated": {"type": "boolean", "const": False},
                },
            },
        },
    }


def build_llm_request(
    report: Any,
    style: ClinicalEEGStyleProfile,
) -> tuple[str, str, dict[str, Any]]:
    source = _report_dict(report)
    inventory = _deidentified_fact_inventory(source)
    _assert_prompt_inventory_deidentified(inventory, source)
    event_aliases, reverse_aliases = _event_aliases(source)
    fact_aliases, _ = _fact_aliases(source)
    finding_ids, event_ids, impression_ids = _llm_fact_assignments(
        source,
        fact_aliases,
        event_aliases,
    )
    schema = llm_json_schema(finding_ids, event_ids, impression_ids)
    system_prompt = (
        "你是SOZ脑电流水线的受约束中文叙述器，只生成待脑电医师审核的脑电草稿。"
        "FACT_LEDGER是唯一病例事实来源；STYLE_PROFILE只定义栏目和文风，不是病例证据。"
        "只能陈述当前EEG信号直接支持的背景、间期和纯脑电事件事实。不得生成病史、"
        "临床资料、视频表现、症状、动作、意识、心电、肌电、临床诊断、临床相关性或治疗。"
        "睡眠脑电、诱发实验及人工临床资料不在当前信号处理范围，不得生成相应内容或占位栏目。"
        "EDF annotation、Excel、医生标签、原始自由文本、路径及姓名不得发送给叙述器，"
        "也不得由版式层显示在生成报告中；它们只可在报告冻结后进入独立评估旁路。"
        "不得新增、删除、改变事实状态，不得改变脑电事件数或顺序。not_recorded不是阴性，"
        "not_assessable不是正常，uncertain必须保留不确定语气。任何非physician_verified"
        "事实必须明确写成算法/技术候选或待复核观察；凡一个文本块含任一此类事实，"
        "该文本块必须逐字包含‘待复核候选’，不能只用一般不确定措辞。脑电图印象只能复述"
        "IMPRESSION_FACT_IDS中的医师确认事实。不得给治疗建议，不得把头皮起始候选升级"
        "为皮层SOZ、致痫区或手术靶点。algorithmic_sustained_eeg_change和"
        "later_scalp_visible_eeg_change只能写在演变与扩展栏，且只能描述算法标记的持续"
        "波形变化或后续头皮可见的时序关系；不得称为发作起始、起源、传播、扩散、"
        "传播路径或传播速度。文本中的阿拉伯数字只能逐字复制该文本块所引用事实值中"
        "已经存在的数字；若数值范围的min与max相同，只写一次该数值，不写成x–x。"
        "不得添加事件序号、数量统计或换算另一套时间坐标。每个文本块"
        "必须列出实际使用的fact_ids。fact alias只能出现在JSON的fact_ids数组中，"
        "不得把LLMFACT编号或EV编号写入任何text_zh正文。"
        "若某文本块的fact_ids为空，text_zh必须为空字符串；缺失内部字段"
        "不是临床所见，不得生成‘未提供结构化事实’、‘待补充’等占位语。"
        "只输出符合JSON Schema的一个对象；不要Markdown、解释或思维过程。"
    )
    user_payload = {
        "TASK": "按真实脑电报告的客观、简洁风格组织已锁定EEG事实",
        "STYLE_PROFILE": style.prompt_payload(),
        "FACT_LEDGER": inventory,
        "EVENT_ORDER": list(reverse_aliases),
        "IMPRESSION_FACT_IDS": [
            fact_aliases[str(item)] for item in source.get("impression_fact_ids", [])
        ],
        "BLOCK_LANGUAGE_REQUIREMENTS": {
            "findings": {
                section_id: {
                    "candidate_qualifier_required": any(
                        item["fact_id"] in set(finding_ids[section_id])
                        and item.get("verification_status") != "physician_verified"
                        for item in inventory
                    ),
                    "required_exact_phrase_if_true": "待复核候选",
                }
                for section_id in FINDING_SECTION_ORDER
            },
            "eeg_events": {
                event_alias: {
                    column_id: {
                        "candidate_qualifier_required": any(
                            item["fact_id"] in set(columns[column_id])
                            and item.get("verification_status") != "physician_verified"
                            for item in inventory
                        ),
                        "required_exact_phrase_if_true": "待复核候选",
                    }
                    for column_id in EEG_EVENT_COLUMN_ORDER
                }
                for event_alias, columns in event_ids.items()
            },
        },
        "OUTPUT_RULES": {
            "finding_section_order": list(FINDING_SECTION_ORDER),
            "eeg_event_columns": list(EEG_EVENT_COLUMN_ORDER),
            "impression_section_order": list(IMPRESSION_SECTION_ORDER),
            "no_patient_identity_or_signature": True,
            "no_layout_markup": True,
            "fact_id_lists_must_not_repeat_items": True,
            "fact_id_assignments_are_fixed_by_output_schema": True,
            "fact_aliases_are_for_fact_id_arrays_only_not_text": True,
            "numeric_literals_must_be_exactly_copied_from_assigned_fact_values": True,
            "do_not_add_event_ordinals_counts_or_time_coordinate_conversions": True,
            "metadata_and_event_occurrence_are_layout_only_and_not_sent": True,
            "sleep_activation_and_clinical_fields_are_omitted": True,
            "forbid_non_eeg_content": True,
            "empty_finding_text_must_equal": EMPTY_FINDING_TEXT,
            "empty_event_text_must_equal": EMPTY_EVENT_TEXT,
            "empty_impression_text_must_equal": EMPTY_IMPRESSION_TEXT,
            "missing_workflow_fields_are_omitted_from_clinical_render": True,
            "forbid_missing_structured_fact_placeholder_wording": True,
        },
        "OUTPUT_JSON_SCHEMA": schema,
    }
    return (
        system_prompt,
        json.dumps(user_payload, ensure_ascii=False, separators=(",", ":")),
        schema,
    )


def _surface(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _channels(text: str) -> set[str]:
    return {match.group(0).upper() for match in _CHANNEL_RE.finditer(text)}


def _numbers(text: str) -> set[float]:
    # Fact/event aliases are transport identifiers, not clinical numbers.  A
    # model may redundantly echo ``LLMFACT0003`` in prose even though the
    # strict schema already carries it in ``fact_ids``.  Remove the complete
    # alias before scanning so its numeric suffix cannot be misread as a new
    # duration, frequency or amplitude.  Real standalone numbers remain under
    # the exact-copy lock below.
    without_aliases = _LLM_ALIAS_RE.sub("", text)
    without_channels = _CHANNEL_RE.sub("", without_aliases)
    result: set[float] = set()
    for token in _NUMBER_RE.findall(without_channels):
        number = float(token)
        if math.isfinite(number):
            result.add(round(number, 6))
    return result


def _regions(text: str) -> set[str]:
    return {term for term in _REGION_TERMS if term in text}


def _validate_text_surface(text: str, facts: Sequence[Mapping[str, Any]], *, name: str) -> None:
    if not isinstance(text, str) or not text.strip() or "<think>" in text.lower() or "```" in text:
        raise ValueError(f"{name} is empty or contains non-report markup")
    semantic_surface = text
    for phrase in _CANDIDATE_PHRASES:
        semantic_surface = semantic_surface.replace(phrase, "")
    semantic_surface = re.sub(r"[^A-Za-z0-9\u4e00-\u9fff]+", "", semantic_surface)
    if facts and len(semantic_surface) < 4:
        raise ValueError(f"{name} contains only a qualifier and no EEG observation")
    support = " ".join(_surface(fact.get("value")) for fact in facts)
    unsupported_channels = _channels(text).difference(_channels(support))
    if unsupported_channels:
        raise ValueError(
            f"{name} introduces an unsupported electrode: "
            f"{sorted(unsupported_channels)}"
        )
    unsupported_numbers = _numbers(text).difference(_numbers(support))
    if unsupported_numbers:
        raise ValueError(
            f"{name} introduces an unsupported numeric fact: "
            f"{sorted(unsupported_numbers)}"
        )
    unsupported_regions = _regions(text).difference(_regions(support))
    if unsupported_regions:
        raise ValueError(
            f"{name} introduces an unsupported region: {sorted(unsupported_regions)}"
        )
    for pattern in _FORBIDDEN_ASSERTIONS:
        if pattern.search(text):
            raise ValueError(f"{name} contains a forbidden clinical assertion")
    fact_types = {str(fact.get("fact_type", "")) for fact in facts}
    if fact_types.intersection(_NEUTRAL_TEMPORAL_FACT_TYPES):
        if _NEUTRAL_ONSET_PROMOTION_RE.search(text):
            raise ValueError(f"{name} promotes a neutral temporal observation to onset")
        if (
            "ictal_spread" not in fact_types
            and _NEUTRAL_PROPAGATION_PROMOTION_RE.search(text)
        ):
            raise ValueError(
                f"{name} promotes a neutral temporal observation to propagation"
            )
    for fact in facts:
        state = str(fact.get("state", ""))
        if state in _MISSING_STATE_PHRASES:
            if not any(phrase in text for phrase in _MISSING_STATE_PHRASES[state]):
                raise ValueError(f"{name} loses fact state {state}")
            unsupported_negative = [
                phrase
                for phrase in _NEGATIVE_CERTAINTY_PHRASES
                if phrase in text and phrase not in support
            ]
            if state in {"not_recorded", "not_assessable"} and unsupported_negative:
                raise ValueError(f"{name} turns missing evidence into a negative finding")
        verification = fact.get("verification")
        status = verification.get("status") if isinstance(verification, Mapping) else None
        if status != "physician_verified" and not any(phrase in text for phrase in _CANDIDATE_PHRASES):
            raise ValueError(f"{name} promotes an unverified fact")


def _selected_facts(
    fact_ids: Any,
    facts_by_id: Mapping[str, Mapping[str, Any]],
    *,
    name: str,
    allow_empty: bool = False,
) -> list[Mapping[str, Any]]:
    if not isinstance(fact_ids, list) or (not fact_ids and not allow_empty) or not all(
        isinstance(item, str) for item in fact_ids
    ):
        qualifier = "string list" if allow_empty else "non-empty string list"
        raise TypeError(f"{name} fact_ids must be a {qualifier}")
    if len(fact_ids) != len(set(fact_ids)):
        raise ValueError(f"{name} repeats a fact_id")
    unknown = [fact_id for fact_id in fact_ids if fact_id not in facts_by_id]
    if unknown:
        raise ValueError(f"{name} cites unknown facts: {unknown}")
    return [facts_by_id[fact_id] for fact_id in fact_ids]


def validate_narrative_payload(payload: Mapping[str, Any], report: Any) -> dict[str, Any]:
    source = _report_dict(report)
    normalized_payload = json.loads(json.dumps(payload, ensure_ascii=False, allow_nan=False))
    _, reverse_fact_aliases = _fact_aliases(source)

    def restore(values: Any) -> None:
        if not isinstance(values, list):
            return
        for index, value in enumerate(values):
            if isinstance(value, str) and value in reverse_fact_aliases:
                values[index] = reverse_fact_aliases[value]

    for block in normalized_payload.get("findings", []):
        if isinstance(block, dict):
            restore(block.get("fact_ids"))
    for block in normalized_payload.get("events", []):
        if isinstance(block, dict):
            for key in (
                "onset_fact_ids",
                "evolution_spread_fact_ids",
                "termination_postictal_fact_ids",
            ):
                restore(block.get(key))
    for block in normalized_payload.get("impression", []):
        if isinstance(block, dict):
            restore(block.get("fact_ids"))
    payload = normalized_payload
    expected_keys = {
        "schema_version",
        "findings",
        "events",
        "impression",
        "safety_acknowledgements",
    }
    if set(payload) != expected_keys or payload.get("schema_version") != NARRATIVE_SCHEMA:
        raise ValueError("clinical EEG narrative schema drifted")
    expected_safety = {
        "patient_facts_added": False,
        "fact_states_changed": False,
        "eeg_event_order_or_count_changed": False,
        "non_eeg_content_generated": False,
        "treatment_recommendation_generated": False,
        "unverified_diagnosis_generated": False,
    }
    if payload.get("safety_acknowledgements") != expected_safety:
        raise ValueError("clinical EEG narrative safety acknowledgements drifted")
    facts_by_id = {
        str(fact["fact_id"]): fact for fact in source["facts"] if isinstance(fact, Mapping)
    }
    used_non_impression: set[str] = set()
    findings = payload.get("findings")
    if not isinstance(findings, list) or len(findings) != len(FINDING_SECTION_ORDER):
        raise ValueError("finding section count drifted")
    for section_id, block in zip(FINDING_SECTION_ORDER, findings, strict=True):
        if not isinstance(block, Mapping) or set(block) != {"section_id", "text_zh", "fact_ids"}:
            raise ValueError("finding block shape drifted")
        if block.get("section_id") != section_id:
            raise ValueError("finding section order drifted")
        available = [
            fact
            for fact in source["facts"]
            if isinstance(fact, Mapping)
            and str(fact.get("section")) in FINDING_FACT_SECTIONS[section_id]
            and fact.get("eeg_event_id") is None
        ]
        facts = _selected_facts(
            block.get("fact_ids"),
            facts_by_id,
            name=f"finding:{section_id}",
            allow_empty=not available,
        )
        if not available:
            accepted_empty_text = {
                EMPTY_FINDING_TEXT[section_id],
                LEGACY_EMPTY_FINDING_TEXT[section_id],
            }
            if facts or block.get("text_zh") not in accepted_empty_text:
                raise ValueError(f"finding:{section_id} must use the fixed empty-section text")
            continue
        if any(str(fact.get("section")) not in FINDING_FACT_SECTIONS[section_id] for fact in facts):
            raise ValueError(f"finding:{section_id} cites a fact from another section")
        if any(fact.get("eeg_event_id") is not None for fact in facts):
            raise ValueError(f"finding:{section_id} cites event-specific facts")
        _validate_text_surface(str(block.get("text_zh", "")), facts, name=f"finding:{section_id}")
        used_non_impression.update(str(fact["fact_id"]) for fact in facts)

    aliases, reverse_aliases = _event_aliases(source)
    events = payload.get("events")
    if not isinstance(events, list) or len(events) != len(reverse_aliases):
        raise ValueError("EEG event count drifted")
    for expected_alias, block in zip(reverse_aliases, events, strict=True):
        if not isinstance(block, Mapping) or block.get("eeg_event_alias") != expected_alias:
            raise ValueError("EEG event order drifted")
        expected_keys_event = {
            "eeg_event_alias",
            "onset_text_zh",
            "onset_fact_ids",
            "evolution_spread_text_zh",
            "evolution_spread_fact_ids",
            "termination_postictal_text_zh",
            "termination_postictal_fact_ids",
        }
        if set(block) != expected_keys_event:
            raise ValueError("EEG event block shape drifted")
        event_id = reverse_aliases[expected_alias]
        specs = (
            ("onset", "onset_text_zh", "onset_fact_ids"),
            (
                "evolution_spread",
                "evolution_spread_text_zh",
                "evolution_spread_fact_ids",
            ),
            (
                "termination_postictal",
                "termination_postictal_text_zh",
                "termination_postictal_fact_ids",
            ),
        )
        for label, text_key, ids_key in specs:
            available = [
                fact
                for fact in source["facts"]
                if isinstance(fact, Mapping)
                and str(fact.get("eeg_event_id")) == event_id
                and str(fact.get("fact_type")) in EEG_EVENT_FACT_TYPES[label]
            ]
            facts = _selected_facts(
                block.get(ids_key),
                facts_by_id,
                name=f"event:{expected_alias}:{label}",
                allow_empty=not available,
            )
            if not available:
                accepted_empty_text = {
                    EMPTY_EVENT_TEXT[label],
                    LEGACY_EMPTY_EVENT_TEXT[label],
                }
                if facts or block.get(text_key) not in accepted_empty_text:
                    raise ValueError(f"event:{expected_alias}:{label} must use fixed empty text")
                continue
            if any(
                str(fact.get("section")) != "ictal"
                or str(fact.get("fact_type")) not in EEG_EVENT_FACT_TYPES[label]
                for fact in facts
            ):
                raise ValueError(f"EEG event:{expected_alias}:{label} cites another column")
            if any(str(fact.get("eeg_event_id")) != event_id for fact in facts):
                raise ValueError(f"EEG event:{expected_alias}:{label} cites another event")
            _validate_text_surface(str(block.get(text_key, "")), facts, name=f"event:{expected_alias}:{label}")
            used_non_impression.update(str(fact["fact_id"]) for fact in facts)

    impression_fact_ids = source.get("impression_fact_ids")
    if not isinstance(impression_fact_ids, list):
        raise TypeError("impression_fact_ids must be a list")
    allowed_impression = set(map(str, impression_fact_ids))
    impression = payload.get("impression")
    if not isinstance(impression, list) or not impression:
        raise ValueError("clinical impression is empty")
    if not allowed_impression:
        accepted_empty_impressions = [
            [{"section_id": "overall", "text_zh": text, "fact_ids": []}]
            for text in (EMPTY_IMPRESSION_TEXT, LEGACY_EMPTY_IMPRESSION_TEXT)
        ]
        if impression not in accepted_empty_impressions:
            raise ValueError("unreviewed report must use the fixed pending-impression text")
        impression = []
    seen_impression_sections: list[str] = []
    used_impression: set[str] = set()
    for block in impression:
        if not isinstance(block, Mapping) or set(block) != {"section_id", "text_zh", "fact_ids"}:
            raise ValueError("impression block shape drifted")
        section_id = str(block.get("section_id"))
        if section_id not in IMPRESSION_SECTION_ORDER or section_id in seen_impression_sections:
            raise ValueError("impression section is unknown or duplicated")
        seen_impression_sections.append(section_id)
        facts = _selected_facts(block.get("fact_ids"), facts_by_id, name=f"impression:{section_id}")
        ids = {str(fact["fact_id"]) for fact in facts}
        if not ids.issubset(allowed_impression) or any(str(fact.get("section")) != "impression" for fact in facts):
            raise ValueError("impression cites a non-approved fact")
        if any(_impression_section(fact) != section_id for fact in facts):
            raise ValueError("impression fact is assigned to the wrong subsection")
        for fact in facts:
            verification = fact.get("verification")
            if not isinstance(verification, Mapping) or verification.get("status") != "physician_verified":
                raise ValueError("impression cites a non-physician-verified fact")
        _validate_text_surface(str(block.get("text_zh", "")), facts, name=f"impression:{section_id}")
        used_impression.update(ids)
    order_indices = [IMPRESSION_SECTION_ORDER.index(value) for value in seen_impression_sections]
    if order_indices != sorted(order_indices):
        raise ValueError("impression section order drifted")
    if used_impression != allowed_impression:
        raise ValueError("clinical impression omits or adds approved impression facts")

    required_non_impression = {
        str(fact["fact_id"])
        for fact in source["facts"]
        if isinstance(fact, Mapping)
        and str(fact.get("section")) != "metadata"
        and str(fact.get("section")) != "impression"
        and str(fact.get("fact_type")) not in DETERMINISTIC_LAYOUT_FACT_TYPES
    }
    if used_non_impression != required_non_impression:
        missing = sorted(required_non_impression.difference(used_non_impression))
        extra = sorted(used_non_impression.difference(required_non_impression))
        raise ValueError(f"narrative fact coverage drifted: missing={missing}, extra={extra}")
    serialized = json.loads(json.dumps(payload, ensure_ascii=False, allow_nan=False))
    # Ensure the real identifiers never leaked back from a model response.
    surface = json.dumps(serialized, ensure_ascii=False)
    for identity_key in ("report_id", "patient_pseudonym"):
        identity = str(source.get(identity_key, ""))
        if identity and identity in surface:
            raise ValueError(f"narrative leaks {identity_key}")
    return serialized


def _fact_text(fact: Mapping[str, Any]) -> str:
    state = str(fact.get("state", ""))
    value = fact.get("value")
    label = FACT_TYPE_LABEL_ZH.get(str(fact.get("fact_type", "")), str(fact.get("fact_type", "观察")))
    if value is None:
        if state == "absent":
            base = f"{label}：已评估，未见相应表现"
        elif state == "not_recorded":
            base = f"{label}：未记录"
        elif state == "not_assessable":
            base = f"{label}：现有记录无法评估"
        else:
            base = f"{label}：未提供可表述的事实值"
    elif isinstance(value, Mapping):
        for key in ("text_zh", "statement", "description_zh", "summary_zh"):
            text = value.get(key)
            if isinstance(text, str) and text.strip():
                base = text.strip().rstrip("。")
                break
        else:
            base = "；".join(f"{key}={_surface(item)}" for key, item in value.items())
    else:
        base = _surface(value).strip().rstrip("。")
    if state == "not_recorded" and not any(word in base for word in _MISSING_STATE_PHRASES[state]):
        base = "未记录：" + base
    elif state == "not_assessable" and not any(word in base for word in _MISSING_STATE_PHRASES[state]):
        base = "无法评估：" + base
    elif state == "uncertain" and not any(word in base for word in _MISSING_STATE_PHRASES[state]):
        base = "尚不明确，待复核：" + base
    verification = fact.get("verification")
    status = verification.get("status") if isinstance(verification, Mapping) else None
    if status != "physician_verified" and not any(word in base for word in _CANDIDATE_PHRASES):
        base = "待复核候选：" + base
    return base + "。"


def _impression_section(fact: Mapping[str, Any]) -> str:
    fact_type = str(fact.get("fact_type", "")).lower()
    if fact_type == "interictal_impression":
        return "interictal"
    if fact_type == "ictal_eeg_impression":
        return "ictal"
    if fact_type == "recording_limitation":
        return "limitations"
    return "overall"


def deterministic_fallback_payload(report: Any) -> dict[str, Any]:
    source = _report_dict(report)
    facts = [fact for fact in source["facts"] if isinstance(fact, Mapping)]
    findings: list[dict[str, Any]] = []
    for section_id in FINDING_SECTION_ORDER:
        selected = [
            fact
            for fact in facts
            if str(fact.get("section")) in FINDING_FACT_SECTIONS[section_id]
            and fact.get("eeg_event_id") is None
        ]
        findings.append(
            {
                "section_id": section_id,
                "text_zh": (
                    "".join(_fact_text(fact) for fact in selected)
                    if selected
                    else EMPTY_FINDING_TEXT[section_id]
                ),
                "fact_ids": [str(fact["fact_id"]) for fact in selected],
            }
        )
    aliases, _ = _event_aliases(source)
    events: list[dict[str, Any]] = []
    for event_id, alias in aliases.items():
        selected = {
            column_id: [
                fact
                for fact in facts
                if str(fact.get("section")) == "ictal"
                and str(fact.get("eeg_event_id")) == event_id
                and str(fact.get("fact_type")) in EEG_EVENT_FACT_TYPES[column_id]
            ]
            for column_id in EEG_EVENT_COLUMN_ORDER
        }
        events.append(
            {
                "eeg_event_alias": alias,
                "onset_text_zh": (
                    "".join(_fact_text(fact) for fact in selected["onset"])
                    if selected["onset"]
                    else EMPTY_EVENT_TEXT["onset"]
                ),
                "onset_fact_ids": [str(fact["fact_id"]) for fact in selected["onset"]],
                "evolution_spread_text_zh": (
                    "".join(_fact_text(fact) for fact in selected["evolution_spread"])
                    if selected["evolution_spread"]
                    else EMPTY_EVENT_TEXT["evolution_spread"]
                ),
                "evolution_spread_fact_ids": [
                    str(fact["fact_id"]) for fact in selected["evolution_spread"]
                ],
                "termination_postictal_text_zh": (
                    "".join(_fact_text(fact) for fact in selected["termination_postictal"])
                    if selected["termination_postictal"]
                    else EMPTY_EVENT_TEXT["termination_postictal"]
                ),
                "termination_postictal_fact_ids": [
                    str(fact["fact_id"]) for fact in selected["termination_postictal"]
                ],
            }
        )
    impression_ids = set(map(str, source.get("impression_fact_ids", [])))
    grouped: dict[str, list[Mapping[str, Any]]] = {key: [] for key in IMPRESSION_SECTION_ORDER}
    for fact in facts:
        if str(fact.get("fact_id")) in impression_ids:
            grouped[_impression_section(fact)].append(fact)
    impression = [
        {
            "section_id": section_id,
            "text_zh": "".join(_fact_text(fact) for fact in grouped[section_id]),
            "fact_ids": [str(fact["fact_id"]) for fact in grouped[section_id]],
        }
        for section_id in IMPRESSION_SECTION_ORDER
        if grouped[section_id]
    ]
    if not impression:
        impression = [
            {"section_id": "overall", "text_zh": EMPTY_IMPRESSION_TEXT, "fact_ids": []}
        ]
    payload = {
        "schema_version": NARRATIVE_SCHEMA,
        "findings": findings,
        "events": events,
        "impression": impression,
        "safety_acknowledgements": {
            "patient_facts_added": False,
            "fact_states_changed": False,
            "eeg_event_order_or_count_changed": False,
            "non_eeg_content_generated": False,
            "treatment_recommendation_generated": False,
            "unverified_diagnosis_generated": False,
        },
    }
    return validate_narrative_payload(payload, source)


def call_local_qwen_chat(
    *,
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    json_schema: Mapping[str, Any],
    max_tokens: int,
    temperature: float,
    enable_thinking: bool,
    timeout_seconds: float,
    retries: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    parsed = urllib.parse.urlparse(base_url)
    if (parsed.hostname or "").lower() not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("clinical EEG narration only permits a local LLM endpoint")
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
        "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": NARRATIVE_SCHEMA,
                "strict": True,
                "schema": dict(json_schema),
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
            candidate = json.loads(content)
            if not isinstance(candidate, dict):
                raise TypeError("local Qwen response is not a JSON object")
            return candidate, {
                "id": response_payload.get("id"),
                "model": response_payload.get("model", model),
                "usage": response_payload.get("usage"),
                "finish_reason": choices[0].get("finish_reason"),
                "attempt": attempt + 1,
                "endpoint_host": parsed.hostname,
            }
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, RuntimeError, TypeError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(4.0, 2.0**attempt))
    raise RuntimeError(f"local Qwen request failed: {last_error}")


def _prefix_required_epistemic_qualifiers(
    payload: Mapping[str, Any],
    report: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Add only a fixed epistemic qualifier when Qwen omitted it.

    This narrow deterministic normalization cannot add an EEG observation,
    number, electrode, region, diagnosis, or interpretation.  All other
    candidate defects still fail validation and trigger the full fallback.
    """

    normalized = json.loads(json.dumps(payload, ensure_ascii=False, allow_nan=False))
    repairs: list[str] = []
    facts = [fact for fact in report["facts"] if isinstance(fact, Mapping)]

    def requires_qualifier(selected: Sequence[Mapping[str, Any]]) -> bool:
        return any(
            not isinstance(fact.get("verification"), Mapping)
            or fact["verification"].get("status") != "physician_verified"
            for fact in selected
        )

    def prefix(block: Any, text_key: str, *, receipt: str) -> None:
        if not isinstance(block, dict):
            return
        text = block.get(text_key)
        if not isinstance(text, str) or not text.strip():
            return
        if any(phrase in text for phrase in _CANDIDATE_PHRASES):
            return
        block[text_key] = "待复核候选：" + text.strip()
        repairs.append(receipt)

    findings = normalized.get("findings")
    if isinstance(findings, list):
        by_section = {
            str(block.get("section_id")): block
            for block in findings
            if isinstance(block, dict)
        }
        for section_id in FINDING_SECTION_ORDER:
            selected = [
                fact
                for fact in facts
                if str(fact.get("section")) in FINDING_FACT_SECTIONS[section_id]
                and fact.get("eeg_event_id") is None
            ]
            if selected and requires_qualifier(selected):
                prefix(
                    by_section.get(section_id),
                    "text_zh",
                    receipt=f"finding:{section_id}:prefixed_waiting_for_review_qualifier",
                )

    events = normalized.get("events")
    aliases, _ = _event_aliases(report)
    if isinstance(events, list):
        by_alias = {
            str(block.get("eeg_event_alias")): block
            for block in events
            if isinstance(block, dict)
        }
        for event_id, event_alias in aliases.items():
            block = by_alias.get(event_alias)
            for column_id in EEG_EVENT_COLUMN_ORDER:
                selected = [
                    fact
                    for fact in facts
                    if str(fact.get("eeg_event_id")) == event_id
                    and str(fact.get("fact_type")) in EEG_EVENT_FACT_TYPES[column_id]
                ]
                if selected and requires_qualifier(selected):
                    prefix(
                        block,
                        f"{column_id}_text_zh",
                        receipt=(
                            f"event:{event_alias}:{column_id}:"
                            "prefixed_waiting_for_review_qualifier"
                        ),
                    )
    return normalized, repairs


def build_pipeline_record(
    *,
    report: Any,
    style: ClinicalEEGStyleProfile,
    policy: Mapping[str, Any],
    candidate_payload: Mapping[str, Any] | None,
    model_metadata: Mapping[str, Any] | None = None,
    generation_error: str | None = None,
) -> dict[str, Any]:
    source = _report_dict(report)
    validation_errors: list[str] = []
    deterministic_safety_repairs: list[str] = []
    fallback_reason: str | None = None
    if candidate_payload is None:
        narrative = deterministic_fallback_payload(source)
        generator = "deterministic_fallback"
        fallback_reason = generation_error or "llm_not_called"
    else:
        try:
            normalized_candidate, deterministic_safety_repairs = (
                _prefix_required_epistemic_qualifiers(candidate_payload, source)
            )
            narrative = validate_narrative_payload(normalized_candidate, source)
            generator = (
                "qwen3.6_facts_locked_draft_with_safety_qualifier"
                if deterministic_safety_repairs
                else "qwen3.6_facts_locked_draft"
            )
        except (TypeError, ValueError) as exc:
            validation_errors.append(f"{type(exc).__name__}: {exc}")
            narrative = deterministic_fallback_payload(source)
            generator = "deterministic_fallback"
            fallback_reason = "llm_validation_failed"
    return {
        "schema_version": PIPELINE_RECORD_SCHEMA,
        "report_id": source["report_id"],
        "patient_pseudonym": source["patient_pseudonym"],
        "source_schema": source["schema_version"],
        "source_sha256": _canonical_sha256(source),
        "scope_receipt": eeg_only_scope_receipt(),
        "style_receipt": {
            "profile_id": style.profile_id,
            "sha256": style.sha256,
            "patient_facts_retained": False,
        },
        "narrative": narrative,
        "generation": {
            "generator": generator,
            "model_release": policy.get("model_release"),
            "model_metadata": dict(model_metadata or {}),
            "candidate_retained": False,
            "candidate_sha256": (
                _canonical_sha256(candidate_payload) if candidate_payload is not None else None
            ),
            "candidate_payload": None,
            "generation_error": generation_error,
            "validation_errors": validation_errors,
            "deterministic_safety_repairs": deterministic_safety_repairs,
            "fallback_reason": fallback_reason,
        },
        "release": {
            "status": "ai_draft",
            "clinical_export_allowed": False,
            "physician_review_required": True,
            "physician_signature_generated_by_llm": False,
        },
        "access_receipt": {
            "raw_eeg_loaded_by_narrator": False,
            "patient_identity_sent_to_llm": False,
            "signature_sent_to_llm": False,
            "non_eeg_context_sent_to_llm": False,
            "sleep_eeg_sent_to_llm": False,
            "activation_experiment_sent_to_llm": False,
            "event_occurrence_sent_to_llm": False,
            "unsupported_sections_omitted_from_report": True,
            "treatment_generated": False,
        },
    }


__all__ = [
    "AUDIT_ONLY_FACT_TYPES",
    "FINDING_SECTION_ORDER",
    "IMPRESSION_SECTION_ORDER",
    "NARRATIVE_SCHEMA",
    "PIPELINE_RECORD_SCHEMA",
    "build_llm_request",
    "build_pipeline_record",
    "call_local_qwen_chat",
    "deterministic_fallback_payload",
    "eeg_only_generation_report_view",
    "eeg_only_scope_receipt",
    "llm_json_schema",
    "load_policy",
    "validate_narrative_payload",
]
