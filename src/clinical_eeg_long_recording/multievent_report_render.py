"""Deterministic Chinese lexicalizer for the multievent EEG claim graph.

The renderer deliberately has no access to raw EDF annotations, spreadsheets,
clinical text or patient metadata.  It accepts only a host-validated
``clinical_eeg_multievent_soz_report_v1`` graph and realizes the already
authorized sentence plan.  Qwen may choose that plan upstream; it cannot add
entities, relations, times or conclusions here.

This is a research scalp-EEG renderer.  It never equates a scalp-visible onset
candidate with the cortical SOZ, epileptogenic zone or a surgical target.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator

from .multievent_soz_claim_validation import (
    validate_multievent_soz_report_payload,
)


MULTIEVENT_REPORT_RENDER_SCHEMA_VERSION = "clinical_eeg_multievent_report_render_v1"
LEXICALIZER_ID = "deterministic_claim_lexicalizer_v1"
_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA_PATH = (
    _ROOT / "schemas" / "clinical_eeg_multievent_report_render_v1.schema.json"
)

_SECTION_ORDER = (
    "technical_quality",
    "eeg_findings",
    "ictal_findings",
    "cross_event_summary",
    "impression",
    "limitations",
    "waveform_index",
)
_SECTION_LABELS = {
    "technical_quality": "技术与可判读性",
    "eeg_findings": "脑电所见",
    "ictal_findings": "事件所见",
    "cross_event_summary": "多事件汇总",
    "impression": "脑电图印象",
    "limitations": "局限性",
    "waveform_index": "波形证据",
}
_ENTITY_LABELS = {
    "left": "左侧",
    "right": "右侧",
    "bilateral": "双侧",
    "midline": "中线",
    "indeterminate": "侧别不定",
    "left_temporal": "左侧颞区",
    "right_temporal": "右侧颞区",
    "left_frontal": "左侧额区",
    "right_frontal": "右侧额区",
    "left_central": "左侧中央区",
    "right_central": "右侧中央区",
    "left_parietal": "左侧顶区",
    "right_parietal": "右侧顶区",
    "left_occipital": "左侧枕区",
    "right_occipital": "右侧枕区",
    "focal": "局灶性",
    "focal_with_rapid_bilateralization": "局灶起始伴快速双侧化",
    "bilateral_synchronous_or_rapid_bilateralization_ambiguous": ("双侧近同步或快速双侧化不易区分"),
    "generalized_synchronous": "头皮广泛近同步起始",
    "multiple_scalp_onset_modes": "多种头皮起始模式",
    "scalp_onset_nonlocalizable": "头皮起始不可定位",
}
_FORBIDDEN_SURFACE_FRAGMENTS = (
    "睡眠脑电",
    "睡眠分期",
    "诱发试验",
    "过度换气",
    "闪光刺激",
    "心电",
    "肌电",
    "眼电",
    "病史",
    "临床表现",
    "意识",
    "用药",
    "治疗建议",
    "手术建议",
    "Excel",
    "EDF annotation",
    "医生标注",
)


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@lru_cache(maxsize=1)
def _schema_validator() -> Draft202012Validator:
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _schema_path(error: object) -> str:
    path = getattr(error, "absolute_path", ())
    return "$" + "".join(
        f"[{item}]" if isinstance(item, int) else f".{item}" for item in path
    )


def _format_recording_time(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("recording-relative time must be finite and non-negative")
    milliseconds = int(round(seconds * 1000.0))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{millis:03d}"


def _time_text(value: Mapping[str, object]) -> str:
    kind = str(value["kind"])
    if kind == "none":
        return ""
    lower = float(value["lower"])
    upper = float(value["upper"])
    if kind == "recording_interval":
        if abs(lower - upper) <= 1e-9:
            return f"相对记录 {_format_recording_time(lower)}"
        return (
            "相对记录 " f"{_format_recording_time(lower)}–{_format_recording_time(upper)}"
        )
    if abs(lower - upper) <= 1e-9:
        return f"相对延迟 {lower:.2f} 秒"
    return f"相对延迟 {lower:.2f}–{upper:.2f} 秒"


def _entity_label(entity_type: str, identifier: str) -> str:
    if entity_type == "electrode":
        return f"{identifier} 通道"
    if entity_type == "mode":
        return f"模式 {identifier}"
    if entity_type == "eeg_event":
        return identifier
    return _ENTITY_LABELS.get(identifier, identifier.replace("_", " "))


def _spatial_text(entities: Sequence[Mapping[str, object]]) -> str:
    by_type: dict[str, list[str]] = defaultdict(list)
    for entity in entities:
        by_type[str(entity["type"])].append(str(entity["id"]))
    parts: list[str] = []
    if by_type.get("region"):
        parts.extend(_entity_label("region", item) for item in by_type["region"])
    elif by_type.get("laterality"):
        parts.extend(
            _entity_label("laterality", item) for item in by_type["laterality"]
        )
    if by_type.get("electrode"):
        electrodes = "、".join(by_type["electrode"])
        parts.append(f"{electrodes} 通道")
    return "（".join(parts[:1]) + (f"（{'、'.join(parts[1:])}）" if len(parts) > 1 else "")


def _phenotype_text(entities: Sequence[Mapping[str, object]]) -> str:
    phenotypes = [
        _entity_label("phenotype", str(row["id"]))
        for row in entities
        if row["type"] == "phenotype"
    ]
    return "、".join(phenotypes)


def _measurement(claim: Mapping[str, object], name: str) -> float | None:
    for row in claim["object_or_value"]["measurements"]:
        if row["name"] == name:
            return float(row["value"])
    return None


def _measurement_summary(claim: Mapping[str, object]) -> str:
    rows = claim["object_or_value"]["measurements"]
    if not rows:
        return ""
    rendered = []
    for row in rows:
        value = float(row["value"])
        rendered_value = str(int(value)) if value.is_integer() else f"{value:.3g}"
        rendered.append(f"{row['name']}={rendered_value} {row['unit']}")
    return "；".join(rendered)


def _event_label(event_id: str | None, ordinals: Mapping[str, int]) -> str:
    if event_id is None:
        return "该事件"
    ordinal = ordinals.get(event_id)
    return f"第 {ordinal} 次事件" if ordinal is not None else f"事件 {event_id}"


def _counterevidence_count(
    sentence_claims: Sequence[Mapping[str, object]],
) -> int:
    return sum(
        1
        for claim in sentence_claims
        if claim["claim_kind"] == "evidence_relation"
        and claim["predicate"] == "contradicts_claim"
    )


def _support_count(sentence_claims: Sequence[Mapping[str, object]]) -> int:
    return sum(
        1
        for claim in sentence_claims
        if claim["claim_kind"] == "evidence_relation"
        and claim["predicate"] == "supports_claim"
    )


def _uncertainty_suffix(sentence_claims: Sequence[Mapping[str, object]]) -> str:
    count = _counterevidence_count(sentence_claims)
    if count:
        return f"；同时存在 {count} 项相反证据，结论保留不确定性"
    return ""


def _core_claim(
    claims: Sequence[Mapping[str, object]],
    *,
    allowed_kinds: set[str],
) -> Mapping[str, object] | None:
    rows = [row for row in claims if str(row["claim_kind"]) in allowed_kinds]
    if len(rows) > 1:
        raise ValueError("one sentence contains multiple incompatible core claims")
    return rows[0] if rows else None


def _render_sentence(
    sentence: Mapping[str, object],
    *,
    claim_by_id: Mapping[str, Mapping[str, object]],
    hypothesis_by_id: Mapping[str, Mapping[str, object]],
    event_ordinals: Mapping[str, int],
) -> str:
    ordered = [claim_by_id[str(item)] for item in sentence["claim_order"]]
    template = str(sentence["template_id"])
    observations = [row for row in ordered if row["claim_kind"] == "observation"]
    core = _core_claim(
        ordered,
        allowed_kinds={"event_inference", "mode_inference", "record_hypothesis"},
    )

    if template == "event_detected_v1":
        claim = observations[0]
        event = _event_label(claim["event_id"], event_ordinals)
        when = _time_text(claim["time"])
        return f"{event}于{when}记录到电图事件候选。"

    if template in {"event_onset_maximal_at_v1", "event_competing_onset_fields_v1"}:
        fragments = []
        for claim in observations:
            event = _event_label(claim["event_id"], event_ordinals)
            when = _time_text(claim["time"])
            spatial = _spatial_text(claim["object_or_value"]["entities"])
            fragments.append(f"{event}于{when}，最早持续改变以{spatial}较突出")
        return "；".join(fragments) + "。"

    if template == "event_rhythm_morphology_v1":
        claim = observations[0]
        event = _event_label(claim["event_id"], event_ordinals)
        when = _time_text(claim["time"])
        spatial = _spatial_text(claim["object_or_value"]["entities"])
        measurements = _measurement_summary(claim)
        details = f"，{measurements}" if measurements else ""
        return f"{event}于{when}见{spatial}波形或节律改变候选{details}。"

    if template == "event_evolution_v1":
        claim = observations[0]
        event = _event_label(claim["event_id"], event_ordinals)
        axis = "频率" if claim["predicate"] == "evolves_in_frequency" else "波幅"
        when = _time_text(claim["time"])
        return f"{event}于{when}记录到{axis}变化轨迹。"

    if template == "event_onset_then_recruitment_v2":
        claim = observations[0]
        event = _event_label(claim["event_id"], event_ordinals)
        return f"{event}记录到早期场与后续累及的先后关系，{_time_text(claim['time'])}。"

    if template == "event_near_synchronous_v1":
        claim = observations[0]
        event = _event_label(claim["event_id"], event_ordinals)
        return f"{event}相关场的先后次序不可分辨，呈近同步候选。"

    if template == "event_termination_v1":
        claim = observations[0]
        event = _event_label(claim["event_id"], event_ordinals)
        return f"{event}于{_time_text(claim['time'])}见活动回落候选。"

    if template == "event_recovery_v1":
        claim = observations[0]
        event = _event_label(claim["event_id"], event_ordinals)
        return f"{event}后于{_time_text(claim['time'])}见信号回归可比背景候选。"

    if template == "event_limitation_v1":
        claim = observations[0]
        event = _event_label(claim["event_id"], event_ordinals)
        return f"{event}受信号质量或可观测性限制，空间判读降级。"

    if template == "event_hypothesis_v1":
        if core is None:
            raise ValueError("event_hypothesis_v1 lacks its core claim")
        event = _event_label(core["event_id"], event_ordinals)
        phenotype = _phenotype_text(core["object_or_value"]["entities"])
        spatial = _spatial_text(core["object_or_value"]["entities"])
        detail = "、".join(item for item in (phenotype, spatial) if item)
        return f"{event}的研究性头皮起始表型倾向{detail}{_uncertainty_suffix(ordered)}。"

    if template in {
        "mode_recurrence_v1",
        "mode_recurrence_with_counterevidence_v1",
    }:
        if core is None:
            raise ValueError("mode recurrence sentence lacks its core claim")
        support_events = _measurement(core, "supporting_events")
        total_events = _measurement(core, "total_events")
        ratio = ""
        if support_events is not None and total_events is not None:
            ratio = f"在 {int(support_events)}/{int(total_events)} 次可用事件中"
        phenotype = _phenotype_text(core["object_or_value"]["entities"])
        spatial = _spatial_text(core["object_or_value"]["entities"])
        detail = "、".join(item for item in (phenotype, spatial) if item)
        return (
            f"按冻结的启发式相似性分组，{ratio}事件呈现{detail}"
            f"{_uncertainty_suffix(ordered)}；该分组仅表示潜在异质性，"
            "未获得独立事件模式资格。"
        )

    if template == "record_generalized_synchronous_v1":
        if core is None:
            return "记录到多导广泛近同步改变候选。"
        return "综合各次事件，头皮 EEG 呈广泛近同步起始表型，不形成局灶通道结论。"
    if template == "record_nonlocalizable_v1":
        if core is None:
            return "记录内未形成稳定的局灶头皮早期场候选。"
        return "综合各次事件，未形成稳定的局灶头皮起始场，研究性头皮起始不可定位。"
    if template == "record_technical_limited_v1":
        if core is None:
            return "本记录受技术或可观测性限制，部分 EEG 事实不可评价。"
        return "本记录受技术或可观测性限制，不能形成可评价的头皮起始候选。"

    if core is None:
        raise ValueError(f"template {template!r} lacks a core inference claim")
    hypothesis_id = str(core["hypothesis_id"])
    hypothesis = hypothesis_by_id[hypothesis_id]
    phenotype = str(hypothesis["phenotype"] or "")
    entities = core["object_or_value"]["entities"]
    spatial = _spatial_text(entities)

    if template == "record_multiple_modes_v1":
        return "综合各次事件可分为多种头皮起始模式，不强制归并为单一局灶通道结论。"
    if template == "record_alternative_hypothesis_v1":
        return f"另保留研究性头皮起始替代候选：{spatial}{_uncertainty_suffix(ordered)}。"
    if template in {
        "record_primary_focal_hypothesis_v1",
        "record_primary_hypothesis_with_counterevidence_v1",
    }:
        if phenotype not in {"focal", "focal_with_rapid_bilateralization"}:
            raise ValueError(
                "focal primary template conflicts with hypothesis phenotype"
            )
        support = _support_count(ordered)
        support_text = f"基于 {support} 项起始支持关系，" if support else ""
        return (
            f"{support_text}本记录的研究性头皮 EEG 起始候选倾向{spatial}"
            f"{_uncertainty_suffix(ordered)}。"
        )
    raise ValueError(f"unsupported deterministic surface frame: {template}")


def _surface_guard(text: str) -> None:
    hits = [item for item in _FORBIDDEN_SURFACE_FRAGMENTS if item in text]
    if hits:
        raise ValueError(f"renderer emitted forbidden non-EEG content: {hits}")


def validate_multievent_report_render(payload: object) -> dict[str, Any]:
    """Validate a persisted deterministic render and its claim ledger."""

    if type(payload) is not dict:
        raise TypeError("multievent report render must be an object")
    errors = sorted(
        _schema_validator().iter_errors(payload), key=lambda item: list(item.path)
    )
    if errors:
        rendered = "; ".join(
            f"{_schema_path(error)}: {error.message}" for error in errors[:8]
        )
        raise ValueError(
            f"multievent report render schema validation failed: {rendered}"
        )
    data: dict[str, Any] = deepcopy(payload)
    ledger = data["sentence_ledger"]
    sentence_ids = [str(row["sentence_id"]) for row in ledger]
    if len(sentence_ids) != len(set(sentence_ids)):
        raise ValueError("sentence ledger contains duplicate sentence IDs")
    claim_counts: Counter[str] = Counter(
        str(claim_id) for row in ledger for claim_id in row["claim_ids"]
    )
    if any(count > 1 for count in claim_counts.values()):
        raise ValueError("sentence ledger duplicates a claim across sentences")
    by_section: dict[str, list[str]] = defaultdict(list)
    for row in ledger:
        section_id = str(row["section_id"])
        if section_id not in data["sections"]:
            raise ValueError("sentence ledger references an absent report section")
        _surface_guard(str(row["text_zh"]))
        by_section[section_id].append(str(row["text_zh"]))
    expected_sections = {
        section: "".join(by_section[section])
        for section in _SECTION_ORDER
        if section in by_section
    }
    if data["sections"] != expected_sections:
        raise ValueError("rendered sections do not exactly close the sentence ledger")
    expected_text = "\n\n".join(
        f"{_SECTION_LABELS[section]}\n{expected_sections[section]}"
        for section in _SECTION_ORDER
        if section in expected_sections
    )
    if data["report_text_zh"] != expected_text:
        raise ValueError("report surface does not exactly close rendered sections")
    _surface_guard(str(data["report_text_zh"]))
    digest_source = deepcopy(data)
    digest_source["report_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["report_sha256"] != _canonical_sha256(digest_source):
        raise ValueError("report_sha256 does not bind the rendered report")
    return data


def render_multievent_soz_report_zh(
    payload: object,
    *,
    trusted_producer_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_calibration_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_capability_qualification_receipts: (
        Mapping[str, Mapping[str, object]] | None
    ) = None,
    trusted_term_decision_receipts: (Mapping[str, Mapping[str, object]] | None) = None,
) -> dict[str, Any]:
    """Validate a claim graph and realize its authorized Chinese sentences."""

    graph = validate_multievent_soz_report_payload(
        payload,
        trusted_producer_receipts=trusted_producer_receipts,
        trusted_calibration_receipts=trusted_calibration_receipts,
        trusted_capability_qualification_receipts=(
            trusted_capability_qualification_receipts
        ),
        trusted_term_decision_receipts=trusted_term_decision_receipts,
    )
    if graph["report_policy"]["lexicalizer"] != LEXICALIZER_ID:
        raise ValueError("claim graph does not authorize this lexicalizer")

    claim_by_id = {str(row["claim_id"]): row for row in graph["claims"]}
    hypothesis_by_id = {str(row["hypothesis_id"]): row for row in graph["hypotheses"]}
    evidence_by_id = {str(row["evidence_id"]): row for row in graph["evidence_catalog"]}
    event_ordinals = {
        str(row["event_id"]): index + 1
        for index, row in enumerate(
            sorted(
                graph["events"],
                key=lambda item: float(item["analysis_interval"]["lower"]),
            )
        )
    }

    sections: dict[str, list[str]] = {item: [] for item in _SECTION_ORDER}
    ledger: list[dict[str, Any]] = []
    for sentence in graph["sentence_plan"]["sentences"]:
        text = _render_sentence(
            sentence,
            claim_by_id=claim_by_id,
            hypothesis_by_id=hypothesis_by_id,
            event_ordinals=event_ordinals,
        )
        _surface_guard(text)
        section_id = str(sentence["section_id"])
        sections[section_id].append(text)
        claim_ids = [str(item) for item in sentence["claim_ids"]]
        evidence_ids = sorted(
            {
                str(evidence_id)
                for claim_id in claim_ids
                for evidence_id in claim_by_id[claim_id]["evidence_ids"]
            }
        )
        waveform_ids = sorted(
            {
                str(waveform_id)
                for evidence_id in evidence_ids
                for waveform_id in evidence_by_id[evidence_id]["waveform_evidence_ids"]
            }
        )
        ledger.append(
            {
                "sentence_id": str(sentence["sentence_id"]),
                "section_id": section_id,
                "template_id": str(sentence["template_id"]),
                "text_zh": text,
                "claim_ids": claim_ids,
                "evidence_ids": evidence_ids,
                "waveform_evidence_ids": waveform_ids,
            }
        )

    rendered_sections = {
        section: "".join(sentences)
        for section, sentences in sections.items()
        if sentences
    }
    report_blocks = [
        f"{_SECTION_LABELS[section]}\n{rendered_sections[section]}"
        for section in _SECTION_ORDER
        if section in rendered_sections
    ]
    report_text = "\n\n".join(report_blocks)
    _surface_guard(report_text)
    body: dict[str, Any] = {
        "schema_version": MULTIEVENT_REPORT_RENDER_SCHEMA_VERSION,
        "record_id": str(graph["record_id"]),
        "language": "zh-CN",
        "render_mode": "deterministic_claim_lexicalizer",
        "lexicalizer_id": LEXICALIZER_ID,
        "source_claim_graph_sha256": _canonical_sha256(graph),
        "sections": rendered_sections,
        "report_text_zh": report_text,
        "sentence_ledger": ledger,
        "source_firewall": deepcopy(graph["provenance"]["inference_exclusions"]),
        "report_sha256": "CONTENT-ADDRESS-PENDING",
    }
    digest_source = deepcopy(body)
    body["report_sha256"] = _canonical_sha256(digest_source)
    return validate_multievent_report_render(body)


__all__ = [
    "LEXICALIZER_ID",
    "MULTIEVENT_REPORT_RENDER_SCHEMA_VERSION",
    "render_multievent_soz_report_zh",
    "validate_multievent_report_render",
]
