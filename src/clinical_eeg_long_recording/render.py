"""Deterministic renderers for a long-recording clinical EEG AI draft.

EEG findings and the automatic EEG impression are projected only from typed,
validated signal facts.  A narrowly bounded event-language projection may consume a
``build_pipeline_record`` narrative after revalidating it against the current
event FACT ledger.  It can contribute wording only: event identity, timing,
facts, impression, waveforms and research rankings remain deterministic.

External EDF annotations, spreadsheet observations and physician ground truth
are rejected by this rendering boundary.  They belong exclusively to the
post-freeze evaluation path and can never be displayed in a generated report.
"""

from __future__ import annotations

import hashlib
from html import escape as html_escape
import json
import math
from pathlib import Path, PurePosixPath
import struct
from typing import Any, Mapping, Sequence
from xml.sax.saxutils import escape as xml_escape
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from .record_findings import (
    build_recording_eeg_findings_summary,
    build_research_multievent_channel_consistency,
)
from .report_outcome import (
    classify_recording_eeg_outcome,
    qualified_ictal_onset_value,
    qualified_sustained_change_value,
)


BUNDLE_SCHEMA = "trustworthy_long_term_clinical_eeg_bundle_v1"
FILTERED_BUNDLE_SCHEMA = (
    "trustworthy_long_term_clinical_eeg_bundle_v2_signal_eligibility_partition"
)
LANGUAGE_LAYER_SCHEMA = "long_term_event_language_layer_v1"

_FACT_LOCKED_QWEN_GENERATOR = "qwen3.6_facts_locked_draft"
_LANGUAGE_TEXT_FIELDS = (
    "onset_text_zh",
    "evolution_spread_text_zh",
    "termination_postictal_text_zh",
)

_AI_DRAFT = "AI 草稿 · 未经脑电医师签署 · 不得直接用于诊疗"
_NOT_ASSESSABLE = "本流水线未对长程全记录执行系统分析，无法评估。"
_SETTINGS_NOT_ASSESSABLE = "未由全部事件的同源 EEG metadata facts 一致支持，无法评估"
# Empty report cells stay empty.  Missing-reason codes belong to the audit
# ledger and must not be translated into repetitive clinical prose.
_NO_QUALIFIED_DERIVATION = ""
_NO_QUALIFIED_EVENT_FINDINGS = (
    "本记录未形成通过资格门槛的事件级 EEG 信号所见，"
    "现有信号证据不足以确认电图发作或判断 SOZ。"
)
_EEG_FINDINGS_SCOPE = (
    "本报告的脑电所见与自动脑电印象仅依据当前流水线处理的头皮 EEG 信号。"
    "当前版本对全记录执行候选粗筛，并对候选窗提取信号特征；"
    "未对全记录背景活动或发作间期放电执行系统资格化分析，因此不列空白结论。"
    "其他未获信号事实支持的栏目在正文省略，缺失原因仅保留于审计记录。"
)
_SOZ_DISCLAIMER = (
    "以下头皮电极排序为研究模型输出，仅供波形复核，不进入临床定位结论。"
)

_EVENT_CLASS_ZH = {
    "electrographic_seizure": "电图发作候选",
    "electrographic_event": "脑电事件候选",
    "uncertain_electrographic_pattern": "意义不确定的脑电变化候选",
}
_PATTERN_ZH = {
    "low_voltage_fast_activity": "低电压快活动",
    "rhythmic_activity": "节律性活动",
    "repetitive_spikes": "重复棘波",
    "electrodecrement": "电压递减",
    "attenuation": "衰减",
    "irregular_activity": "不规则活动",
    "spike": "棘波",
    "sharp_wave": "尖波",
    "spike_and_slow_wave": "棘慢波",
    "sharp_and_slow_wave": "尖慢波",
    "polyspike": "多棘波",
    "fast_activity": "快活动",
    "theta_activity": "θ活动",
    "delta_activity": "δ活动",
    "mixed": "混合波形",
    "indeterminate": "波形类型不确定",
    "suppression": "抑制",
    "slowing": "慢波活动",
    "periodic_discharge": "周期性放电",
    "return_to_baseline": "恢复至基线",
    "other": "其他受控波形类型",
    "unknown": "波形类型不确定",
}
_CHANGE_ZH = {
    "frequency": "频率",
    "amplitude": "波幅",
    "morphology": "形态",
    "spatial_distribution": "空间分布",
}
_FREQUENCY_BAND_ZH = {
    "delta": "δ频段",
    "theta": "θ频段",
    "alpha": "α频段",
    "beta": "β频段",
    "gamma": "γ频段",
    "broadband": "宽频段",
    "unknown": "频段不确定",
}
_RHYTHMICITY_ZH = {
    "rhythmic": "节律性",
    "quasi_rhythmic": "近节律性",
    "nonrhythmic": "非节律性",
    "indeterminate": "节律性不确定",
}
_LATERALITY_ZH = {
    "left": "左侧",
    "right": "右侧",
    "bilateral": "双侧",
    "midline": "中线",
    "none": "无明确侧别",
    "indeterminate": "侧别不确定",
}
_DISTRIBUTION_ZH = {
    "focal": "局灶分布",
    "multifocal": "多灶分布",
    "hemispheric": "半球性分布",
    "bilateral_independent": "双侧独立分布",
    "bilateral_synchronous": "双侧同步分布",
    "generalized": "广泛性分布",
    "diffuse": "弥漫分布",
}
_REGION_ZH = {
    "frontal": "额区",
    "temporal": "颞区",
    "central": "中央区",
    "parietal": "顶区",
    "occipital": "枕区",
    "frontotemporal": "额颞区",
    "centrotemporal": "中央颞区",
    "temporoparietal": "颞顶区",
    "posterior": "后头部",
    "diffuse": "弥漫",
    "midline": "中线区",
    "unknown": "区域不确定",
}
def _mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a canonical mapping")
    return value


def _canonical_dict(value: object, context: str) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    return dict(_mapping(value, context))


def _validated_bundle(value: object) -> dict[str, Any]:
    source = _canonical_dict(value, "long-term EEG bundle")
    if source.get("schema_version") not in (BUNDLE_SCHEMA, FILTERED_BUNDLE_SCHEMA):
        raise ValueError(
            f"renderer requires {BUNDLE_SCHEMA} or {FILTERED_BUNDLE_SCHEMA}"
        )
    # Keep validation at the rendering boundary.  The import is local so this
    # module remains usable while schema dataclasses import rendering helpers.
    from .aggregation import validate_trustworthy_long_term_clinical_eeg_bundle

    validated = validate_trustworthy_long_term_clinical_eeg_bundle(source)
    return _canonical_dict(validated, "validated long-term EEG bundle")


def _finite(value: object, context: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{context} must be >= {minimum:g}")
    return result


def _clock(value: object) -> str:
    seconds = _finite(value, "recording-relative time", minimum=0.0)
    milliseconds = int(round(seconds * 1000.0))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{millis:03d}"


def _duration(value: object) -> str:
    seconds = _finite(value, "duration", minimum=0.0)
    return f"{seconds:.3f} 秒"


def _range_text(value: object, *, unit: str) -> str:
    item = _mapping(value, "numeric range")
    low = _finite(item.get("min"), "range.min", minimum=0.0)
    high = _finite(item.get("max"), "range.max", minimum=0.0)
    if high < low:
        raise ValueError("numeric range is reversed")
    if math.isclose(low, high, rel_tol=0.0, abs_tol=1e-12):
        return f"{low:g} {unit}"
    return f"{low:g}–{high:g} {unit}"


def _closed_label(value: object, labels: Mapping[str, str], context: str) -> str:
    if not isinstance(value, str) or value not in labels:
        raise ValueError(f"unsupported {context} code")
    return labels[value]


def _join_codes(
    values: object,
    *,
    context: str,
    labels: Mapping[str, str] | None = None,
    empty: str = "未提供",
) -> str:
    if not isinstance(values, list):
        raise TypeError(f"{context} must be a canonical list")
    if not values:
        return empty
    result: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value:
            raise TypeError(f"{context} entries must be non-empty strings")
        result.append(_closed_label(value, labels, context) if labels is not None else value)
    return "、".join(result)


def _validated_bundle_with_context(
    bundle: object,
    context: object | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    source = _validated_bundle(bundle)
    if context is not None:
        raise ValueError(
            "EEG-only renderer rejects external annotation or spreadsheet context"
        )
    return source, None


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalize_fact_locked_language_text(
    text: str,
    event: Mapping[str, Any],
) -> str:
    """Collapse fact-equivalent degenerate ranges in validated LLM wording.

    A locked numeric range with identical bounds is clinically written as one
    value (``1 Hz``), not ``1–1 Hz``.  Only spellings derived from the current
    event FACT ledger are eligible, so this presentation normalization cannot
    introduce or alter a numeric fact.
    """

    normalized = text
    for fact in _facts(event):
        value = fact.get("value")
        if not isinstance(value, Mapping):
            continue
        for field in ("frequency_hz", "amplitude_uv"):
            numeric_range = value.get(field)
            if not isinstance(numeric_range, Mapping):
                continue
            low = _finite(numeric_range.get("min"), f"{field}.min", minimum=0.0)
            high = _finite(numeric_range.get("max"), f"{field}.max", minimum=0.0)
            if not math.isclose(low, high, rel_tol=0.0, abs_tol=1e-12):
                continue
            spellings = {f"{low:g}", str(numeric_range.get("min"))}
            for spelling in spellings:
                for separator in ("–", "-", "—", "~", "～", "至", "到"):
                    normalized = normalized.replace(
                        f"{spelling}{separator}{spelling}", spelling
                    )
                    normalized = normalized.replace(
                        f"{spelling} {separator} {spelling}", spelling
                    )
    return normalized


def _fact_locked_event_language(
    bundle: Mapping[str, Any],
    language_layer: object | None,
) -> dict[str, dict[str, str]]:
    """Return the only LLM surface that may reach a report.

    The complete language record is deliberately revalidated against the
    event's current report payload at the rendering boundary.  Any fallback,
    safety repair, validation error, event-binding mismatch, prompt-firewall
    drift or malformed record is silently omitted so language-service failure
    cannot prevent the deterministic report from being produced.
    """

    if language_layer is None or not isinstance(language_layer, Mapping):
        return {}
    if language_layer.get("schema_version") != LANGUAGE_LAYER_SCHEMA:
        return {}
    raw_records = language_layer.get("event_records")
    if not isinstance(raw_records, list):
        return {}

    from src.clinical_eeg_report.generation import (
        PIPELINE_RECORD_SCHEMA,
        validate_narrative_payload,
    )

    events_by_id = {
        str(event["eeg_event_id"]): event for event in _events(bundle)
    }
    result: dict[str, dict[str, str]] = {}
    seen_event_ids: set[str] = set()
    firewall_fields = {
        "event_envelope_values_absent",
        "source_context_values_absent",
        "edf_annotation_values_absent",
        "excel_observation_values_absent",
        "waveform_values_absent",
        "research_ranking_values_absent",
    }
    forbidden_access_true = {
        "raw_eeg_loaded_by_narrator",
        "patient_identity_sent_to_llm",
        "signature_sent_to_llm",
        "non_eeg_context_sent_to_llm",
        "sleep_eeg_sent_to_llm",
        "activation_experiment_sent_to_llm",
        "event_occurrence_sent_to_llm",
        "treatment_generated",
    }

    for raw_record in raw_records:
        try:
            record = _mapping(raw_record, "event language record")
            event_id = record.get("eeg_event_id")
            if not isinstance(event_id, str) or event_id not in events_by_id:
                continue
            if event_id in seen_event_ids:
                result.pop(event_id, None)
                continue
            seen_event_ids.add(event_id)
            event = events_by_id[event_id]
            record_start = _finite(
                record.get("recording_event_start_offset_seconds"),
                "language record event start",
                minimum=0.0,
            )
            event_start = _finite(
                event.get("recording_event_start_offset_seconds"),
                "bundle event start",
                minimum=0.0,
            )
            if not math.isclose(record_start, event_start, rel_tol=0.0, abs_tol=1e-6):
                continue

            audit = _mapping(record.get("request_audit"), "language request audit")
            firewall = _mapping(audit.get("firewall"), "language prompt firewall")
            if (
                audit.get("request_outcome") != "candidate_received"
                or audit.get("prompt_or_schema_content_persisted") is not False
                or any(firewall.get(key) is not True for key in firewall_fields)
            ):
                continue

            pipeline_record = _mapping(
                record.get("language_record"), "clinical EEG pipeline record"
            )
            generation = _mapping(
                pipeline_record.get("generation"), "language generation receipt"
            )
            if (
                pipeline_record.get("schema_version") != PIPELINE_RECORD_SCHEMA
                or generation.get("generator") != _FACT_LOCKED_QWEN_GENERATOR
                or generation.get("validation_errors") != []
                or generation.get("deterministic_safety_repairs") != []
                or generation.get("fallback_reason") is not None
                or generation.get("generation_error") is not None
                or generation.get("candidate_retained") is not False
                or generation.get("candidate_payload") is not None
            ):
                continue
            access = _mapping(
                pipeline_record.get("access_receipt"), "language access receipt"
            )
            if any(access.get(key) is not False for key in forbidden_access_true):
                continue

            report = _mapping(event.get("event_report_payload"), "event report payload")
            if (
                report.get("eeg_event_ids") != [event_id]
                or pipeline_record.get("report_id") != report.get("report_id")
                or pipeline_record.get("patient_pseudonym")
                != report.get("patient_pseudonym")
                or pipeline_record.get("source_schema") != report.get("schema_version")
                or pipeline_record.get("source_sha256") != _canonical_sha256(report)
            ):
                continue
            narrative = validate_narrative_payload(
                _mapping(pipeline_record.get("narrative"), "fact-locked narrative"),
                report,
            )
            narrative_events = narrative.get("events")
            if not isinstance(narrative_events, list) or len(narrative_events) != 1:
                continue
            narrative_event = _mapping(
                narrative_events[0], "fact-locked event narrative"
            )
            facts_by_id = {
                str(fact.get("fact_id")): fact for fact in _facts(event)
            }
            id_field_by_text = {
                "onset_text_zh": "onset_fact_ids",
                "evolution_spread_text_zh": "evolution_spread_fact_ids",
                "termination_postictal_text_zh": "termination_postictal_fact_ids",
            }
            onset_visible = qualified_ictal_onset_value(event) is not None
            sustained_visible = qualified_sustained_change_value(event) is not None
            termination_visible = any(
                fact.get("fact_type") in {"ictal_termination", "postictal_pattern"}
                and _event_fact_language_authorized(event, fact)
                for fact in _facts(event)
            )
            texts: dict[str, str] = {}
            for field in _LANGUAGE_TEXT_FIELDS:
                # Count only wording that the current renderer can actually put
                # on the report surface.  Neutral sustained-change rows are
                # rendered deterministically, and absent clinical columns must
                # not inflate the Qwen projection receipt merely because a
                # legacy language record still contains text for them.
                if field == "onset_text_zh" and not onset_visible:
                    continue
                if field == "evolution_spread_text_zh" and not (
                    onset_visible and not sustained_visible
                ):
                    continue
                if field == "termination_postictal_text_zh" and not termination_visible:
                    continue
                value = narrative_event.get(field)
                fact_ids = narrative_event.get(id_field_by_text[field])
                if not isinstance(value, str) or not isinstance(fact_ids, list):
                    raise ValueError("fact-locked event narrative field is malformed")
                # Empty columns are workflow state and are omitted, including
                # legacy records that used a fixed “missing structured fact”
                # sentence.  They must never occupy a clinical report cell.
                if not fact_ids:
                    continue
                if not value.strip():
                    raise ValueError("fact-backed event narrative text is empty")
                cited = [facts_by_id.get(str(fact_id)) for fact_id in fact_ids]
                if any(
                    fact is None
                    or not _event_fact_language_authorized(event, fact)
                    for fact in cited
                ):
                    # Withhold only the unauthorized wording block.  Other
                    # independently qualified blocks for the event survive.
                    continue
                texts[field] = _normalize_fact_locked_language_text(value, event)
            if texts:
                result[event_id] = texts
        except (KeyError, TypeError, ValueError):
            # A language-layer defect removes language authority only.  The
            # deterministic report remains available and unchanged.
            continue
    return result


def _events(bundle: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = bundle.get("events")
    if not isinstance(raw, list):
        raise TypeError("bundle.events must be a canonical list")
    events = [_mapping(item, f"bundle.events[{index}]") for index, item in enumerate(raw)]
    numbers = [item.get("event_number") for item in events]
    if numbers != list(range(1, len(events) + 1)):
        raise ValueError("bundle events must be in canonical recording-time order")
    if bundle.get("event_count") != len(events):
        raise ValueError("bundle event_count does not match events")
    return events


def _analysis_candidate_counts(bundle: Mapping[str, Any]) -> tuple[int, int, int]:
    events = _events(bundle)
    if "analysis_selection" not in bundle:
        return len(events), len(events), 0
    selected = bundle.get("detector_selected_candidate_count")
    analyzable = bundle.get("analysis_analyzable_candidate_count")
    rejected = bundle.get("analysis_rejected_candidate_count")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (
        selected,
        analyzable,
        rejected,
    )):
        raise TypeError("analysis candidate counts must be integers")
    assert isinstance(selected, int)
    assert isinstance(analyzable, int)
    assert isinstance(rejected, int)
    if (
        min(selected, analyzable, rejected) < 0
        or selected != analyzable + rejected
        or analyzable != len(events)
    ):
        raise ValueError("analysis candidate counts do not close")
    return selected, analyzable, rejected


_REJECTION_REASON_ZH = {
    "ambiguous_standard19": "标准 19 电极映射不唯一",
    "invalid_sfreq": "采样率不满足分析合同",
    "mixed_sfreq": "标准 19 通道采样率不一致",
    "sample_count_mismatch": "标准 19 通道样本数不一致",
    "insufficient_warmup": "候选前因果预热时长不足",
    "insufficient_post": "候选后可用信号时长不足",
    "payload_shape": "候选信号载荷形状不符合分析合同",
    "signal_qc": "候选窗信号质量门槛未通过",
    "reference_or_signal_contract": "参考或信号合同不满足",
}


def _analysis_rejection_rows(
    bundle: Mapping[str, Any],
) -> list[tuple[str, str, str, str]]:
    raw_selection = bundle.get("analysis_selection")
    if raw_selection is None:
        return []
    selection = _mapping(raw_selection, "analysis selection")
    raw_events = selection.get("events")
    if not isinstance(raw_events, list):
        raise TypeError("analysis selection events must be an array")
    rows: list[tuple[str, str, str, str]] = []
    for raw in raw_events:
        item = _mapping(raw, "analysis selection event")
        if item.get("analysis_disposition") != "rejected_signal_eligibility":
            continue
        receipt = _mapping(item.get("rejection_receipt"), "analysis rejection receipt")
        code = receipt.get("eligibility_code")
        if code not in _REJECTION_REASON_ZH:
            raise ValueError("analysis rejection reason code is unsupported")
        details = _mapping(receipt.get("signal_qc_details"), "signal QC details")
        detail_parts: list[str] = []
        flatline = details.get("flatline_channels")
        clipping = details.get("clipping_channels")
        if isinstance(flatline, list) and flatline:
            detail_parts.append("持续平直通道：" + "、".join(str(x) for x in flatline))
        if isinstance(clipping, list) and clipping:
            detail_parts.append("持续极值/削顶通道：" + "、".join(str(x) for x in clipping))
        if not detail_parts:
            detail_parts.append("未形成可安全进入后续 SOZ 分析的信号窗")
        rows.append(
            (
                str(item.get("eeg_event_id")),
                _clock(item.get("candidate_anchor_offset_seconds")),
                _REJECTION_REASON_ZH[str(code)] + "；" + "；".join(detail_parts),
                "未进入事件级排序/所见提取；不表示未发作，也不表示正常脑电",
            )
        )
    return rows


def _coverage(bundle: Mapping[str, Any]) -> tuple[str, str]:
    manifest = _mapping(bundle.get("detection_manifest"), "detection_manifest")
    raw_intervals = manifest.get("scan_coverage_intervals")
    if not isinstance(raw_intervals, list) or not raw_intervals:
        raise ValueError("detection manifest has no scan coverage intervals")
    intervals: list[tuple[float, float]] = []
    for index, raw in enumerate(raw_intervals):
        interval = _mapping(raw, f"scan coverage interval {index}")
        start = _finite(interval.get("start_offset_seconds"), "coverage start", minimum=0.0)
        stop = _finite(interval.get("stop_offset_seconds"), "coverage stop", minimum=0.0)
        if stop <= start:
            raise ValueError("scan coverage interval is empty or reversed")
        intervals.append((start, stop))
    duration = _finite(bundle.get("recording_duration_seconds"), "recording duration", minimum=0.0)
    covered = sum(stop - start for start, stop in intervals)
    percent = min(100.0, covered / duration * 100.0) if duration else 0.0
    interval_text = "；".join(f"{_clock(start)}–{_clock(stop)}" for start, stop in intervals)
    return interval_text, f"{percent:.2f}%"


_DETECTOR_DESCRIPTION_ZH = {
    (
        "heuristic_preselector",
        "not_evaluated_for_deployment",
    ): "启发式待复核预筛，非已验证发作检测模型",
    (
        "research_candidate",
        "not_deployment_qualified",
    ): "研究候选检测器，尚未取得部署资格",
    (
        "deployment_qualified",
        "passed_external_promotion_gate",
    ): "已通过冻结外部验证部署门槛的检测器",
}


def _detector_display(bundle: Mapping[str, Any]) -> str:
    """Expose the validated detector receipt without upgrading its authority."""

    manifest = _mapping(bundle.get("detection_manifest"), "detection_manifest")
    receipt = _mapping(manifest.get("detector_receipt"), "detector_receipt")
    detector_id = receipt.get("detector_id")
    role = receipt.get("detector_role")
    promotion = receipt.get("promotion_status")
    if not isinstance(detector_id, str) or not detector_id:
        raise TypeError("detector receipt is missing detector_id")
    if not isinstance(role, str) or not isinstance(promotion, str):
        raise TypeError("detector receipt role or promotion status is malformed")
    description = _DETECTOR_DESCRIPTION_ZH.get((role, promotion))
    if description is None:
        raise ValueError("detector receipt role and promotion status are inconsistent")
    return f"{detector_id}（{description}）"


def _facts(event: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    report = _mapping(event.get("event_report_payload"), "event report payload")
    raw = report.get("facts")
    if not isinstance(raw, list):
        raise TypeError("event report facts must be a canonical list")
    return [_mapping(item, "event report fact") for item in raw]


def _fact_values(event: Mapping[str, Any], fact_type: str) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    event_id = event.get("eeg_event_id")
    result: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for fact in _facts(event):
        if fact.get("fact_type") != fact_type or fact.get("eeg_event_id") != event_id:
            continue
        value = fact.get("value")
        if isinstance(value, Mapping):
            result.append((fact, value))
    return result


def _one_fact(event: Mapping[str, Any], fact_type: str) -> tuple[Mapping[str, Any], Mapping[str, Any]] | None:
    values = _fact_values(event, fact_type)
    if len(values) > 1:
        raise ValueError(f"event has repeated {fact_type} facts")
    return values[0] if values else None


def _one_metadata_fact(
    event: Mapping[str, Any], fact_type: str
) -> Mapping[str, Any] | None:
    values: list[Mapping[str, Any]] = []
    for fact in _facts(event):
        if fact.get("fact_type") != fact_type or "eeg_event_id" in fact:
            continue
        value = fact.get("value")
        if not isinstance(value, Mapping):
            raise TypeError(f"{fact_type} metadata fact value must be a mapping")
        values.append(value)
    if len(values) > 1:
        raise ValueError(f"event has repeated {fact_type} metadata facts")
    return values[0] if values else None


def _recording_signal_settings(bundle: Mapping[str, Any]) -> dict[str, str]:
    events = _events(bundle)
    if not events:
        return {
            "sampling_rate": _SETTINGS_NOT_ASSESSABLE,
            "filter": _SETTINGS_NOT_ASSESSABLE,
            "reference_montage": _SETTINGS_NOT_ASSESSABLE,
        }

    signatures: list[tuple[float, float, float, float | None, tuple[str, ...], str | None]] = []
    for event in events:
        acquisition = _one_metadata_fact(event, "acquisition_settings")
        setup = _one_metadata_fact(event, "electrode_setup")
        if acquisition is None or setup is None:
            return {
                "sampling_rate": _SETTINGS_NOT_ASSESSABLE,
                "filter": _SETTINGS_NOT_ASSESSABLE,
                "reference_montage": _SETTINGS_NOT_ASSESSABLE,
            }
        sampling_rate = _finite(
            acquisition.get("sampling_rate_hz"),
            "metadata sampling rate",
            minimum=0.0,
        )
        low_cut = _finite(
            acquisition.get("low_cut_hz"), "metadata low cut", minimum=0.0
        )
        high_cut = _finite(
            acquisition.get("high_cut_hz"), "metadata high cut", minimum=0.0
        )
        notch_raw = acquisition.get("notch_hz")
        notch = (
            _finite(notch_raw, "metadata notch", minimum=0.0)
            if notch_raw is not None
            else None
        )
        raw_montages = setup.get("montages")
        if not isinstance(raw_montages, list) or not all(
            isinstance(item, str) and item for item in raw_montages
        ):
            raise TypeError("electrode_setup montages must be a non-empty string list")
        reference = setup.get("reference")
        if reference is not None and (
            not isinstance(reference, str) or not reference
        ):
            raise TypeError("electrode_setup reference must be a non-empty string")
        signatures.append(
            (
                sampling_rate,
                low_cut,
                high_cut,
                notch,
                tuple(raw_montages),
                reference,
            )
        )

    if any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError("event EEG acquisition/reference metadata facts are inconsistent")

    sampling_rate, low_cut, high_cut, notch, montages, reference = signatures[0]
    montage_labels = {
        "longitudinal_bipolar": "纵向双极",
        "transverse_bipolar": "横向双极",
        "common_average": "共平均",
        "average": "平均参考",
        "referential": "参考导联",
    }
    montage_text = "、".join(
        montage_labels.get(item, item) for item in montages
    )
    reference_text = montage_labels.get(reference, reference) if reference else "未提供"
    filter_text = f"{low_cut:g}–{high_cut:g} Hz"
    if notch is not None:
        filter_text += f"；陷波 {notch:g} Hz"
    return {
        "sampling_rate": f"{sampling_rate:g} Hz",
        "filter": filter_text,
        "reference_montage": f"{montage_text}；参考={reference_text}",
    }


def _event_interval(event: Mapping[str, Any]) -> tuple[float, float]:
    start = _finite(
        event.get("recording_event_start_offset_seconds"),
        "recording event start",
        minimum=0.0,
    )
    stop = _finite(
        event.get("recording_event_stop_offset_seconds"),
        "recording event stop",
        minimum=0.0,
    )
    if stop <= start:
        raise ValueError("recording event interval is empty or reversed")
    occurrence = _one_fact(event, "electrographic_event_occurrence")
    if occurrence is None:
        raise ValueError("event has no electrographic occurrence fact")
    _, value = occurrence
    if value.get("time_coordinate") != "segment_start_seconds":
        raise ValueError("event occurrence must explicitly use the segment-start timebase")
    segment_start = _finite(
        event.get("segment_start_offset_seconds"),
        "segment start",
        minimum=0.0,
    )
    local_start = _finite(value.get("start_offset_seconds"), "segment-local event start", minimum=0.0)
    local_duration = _finite(value.get("duration_seconds"), "event duration", minimum=0.0)
    if not math.isclose(segment_start + local_start, start, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("segment-local event start does not close to recording time")
    if not math.isclose(start + local_duration, stop, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("event duration does not close to recording time")
    return start, stop


def _event_class(event: Mapping[str, Any]) -> str:
    occurrence = _one_fact(event, "electrographic_event_occurrence")
    assert occurrence is not None
    return _closed_label(occurrence[1].get("event_class"), _EVENT_CLASS_ZH, "event class")


def _qualified_signal_value(value: Mapping[str, Any]) -> bool:
    """Return whether optional quantitative descriptors passed all hard gates."""

    raw = value.get("qualification")
    if raw is None:
        return False
    qualification = _mapping(raw, "signal finding qualification")
    required_true = (
        "artifact_gate_passed",
        "sustained_change_gate_passed",
        "reproducibility_gate_passed",
        "source_signal_only",
    )
    required_false = ("external_context_used", "research_ranking_used")
    if any(qualification.get(key) is not True for key in required_true):
        raise ValueError("signal finding qualification has an unpassed evidence gate")
    if any(qualification.get(key) is not False for key in required_false):
        raise ValueError("signal finding qualification used a forbidden input")
    return True


def _physician_or_independently_qualified_fact(
    fact: Mapping[str, Any],
    value: Mapping[str, Any],
    *,
    morphology_required: bool = False,
    spatial_required: bool = False,
) -> bool:
    """Gate clinical morphology/evolution/spread wording at render time."""

    verification = fact.get("verification")
    if isinstance(verification, Mapping) and verification.get("status") == "physician_verified":
        return True
    qualification = value.get("qualification")
    if not isinstance(qualification, Mapping):
        return False
    if not _qualified_signal_value(value):
        return False
    if morphology_required and qualification.get("morphology_terms_qualified") is not True:
        return False
    if spatial_required and qualification.get("spatial_spread_terms_qualified") is not True:
        return False
    return True


def _event_fact_language_authorized(
    event: Mapping[str, Any],
    fact: Mapping[str, Any],
) -> bool:
    """Return whether one fact may contribute clinical-facing event wording."""

    value = fact.get("value")
    if not isinstance(value, Mapping):
        return False
    fact_type = str(fact.get("fact_type"))
    if fact_type == "algorithmic_sustained_eeg_change":
        return qualified_sustained_change_value(event) is value or (
            qualified_sustained_change_value(event) == value
        )
    if fact_type == "later_scalp_visible_eeg_change":
        return _qualified_signal_value(value)
    if fact_type == "ictal_onset_pattern":
        qualified = qualified_ictal_onset_value(event)
        return qualified is value or qualified == value
    if fact_type == "ictal_evolution":
        dimensions = value.get("change_dimensions")
        morphology_required = isinstance(dimensions, list) and "morphology" in dimensions
        spatial_required = isinstance(dimensions, list) and "spatial_distribution" in dimensions
        return _physician_or_independently_qualified_fact(
            fact,
            value,
            morphology_required=morphology_required,
            spatial_required=spatial_required,
        )
    if fact_type == "ictal_spread":
        return _physician_or_independently_qualified_fact(
            fact,
            value,
            spatial_required=True,
        )
    if fact_type == "ictal_termination":
        return _physician_or_independently_qualified_fact(fact, value)
    if fact_type == "postictal_pattern":
        return _physician_or_independently_qualified_fact(fact, value)
    return False


def _spatial_tendency(value: Mapping[str, Any]) -> str | None:
    pieces: list[str] = []
    if "laterality" in value:
        pieces.append(
            _closed_label(value.get("laterality"), _LATERALITY_ZH, "laterality")
        )
    if "regions" in value:
        pieces.append(
            _join_codes(
                value.get("regions"),
                context="signal regions",
                labels=_REGION_ZH,
            )
        )
    if "distribution" in value:
        pieces.append(
            _closed_label(
                value.get("distribution"), _DISTRIBUTION_ZH, "distribution"
            )
        )
    return "、".join(pieces) if pieces else None


def _qualified_onset_text(event: Mapping[str, Any]) -> tuple[str, str] | None:
    value = qualified_ictal_onset_value(event)
    if value is None:
        return None
    pieces = [
        _closed_label(value.get("onset_type"), _PATTERN_ZH, "onset type"),
        _closed_label(value.get("morphology"), _PATTERN_ZH, "onset morphology"),
    ]
    if "frequency_hz" in value:
        pieces.append(_range_text(value["frequency_hz"], unit="Hz"))
    if "amplitude_uv" in value:
        pieces.append(_range_text(value["amplitude_uv"], unit="μV"))
    spatial = _spatial_tendency(value)
    if spatial:
        pieces.append(spatial)
    derivations = _join_codes(
        value.get("derivations", []),
        context="onset derivations",
        empty="",
    )
    electrodes = _join_codes(
        value.get("electrodes", []),
        context="onset electrodes",
        empty="",
    )
    support = []
    if derivations:
        support.append(f"起始支持双极导联：{derivations}")
    if electrodes:
        support.append(f"起始支持头皮电极：{electrodes}")
    return "；".join(pieces), "；".join(support) or _NO_QUALIFIED_DERIVATION


def _sustained_text(event: Mapping[str, Any]) -> tuple[str, str]:
    value = qualified_sustained_change_value(event)
    if value is None:
        start, stop = _event_interval(event)
        return (
            "粗筛待复核候选支持窗（"
            f"{_duration(stop - start)}；未由此确认发作）",
            _NO_QUALIFIED_DERIVATION,
        )
    segment_start = _finite(event.get("segment_start_offset_seconds"), "segment start", minimum=0.0)
    local_start = _finite(value.get("start_offset_seconds"), "sustained change start", minimum=0.0)
    local_stop = _finite(value.get("end_offset_seconds"), "sustained change stop", minimum=0.0)
    if local_stop <= local_start:
        raise ValueError("sustained EEG change interval is empty or reversed")
    text = (
        f"{_event_class(event)}；量化持续变化候选 "
        f"{_clock(segment_start + local_start)}–"
        f"{_clock(segment_start + local_stop)}（{_duration(local_stop - local_start)}）"
    )
    descriptors: list[str] = []
    if "frequency_band" in value:
        descriptors.append(
            _closed_label(
                value.get("frequency_band"),
                _FREQUENCY_BAND_ZH,
                "frequency band",
            )
        )
    if "frequency_hz" in value:
        descriptors.append(f"主频 {_range_text(value['frequency_hz'], unit='Hz')}")
    if "rhythmicity" in value:
        descriptors.append(
            _closed_label(
                value.get("rhythmicity"), _RHYTHMICITY_ZH, "rhythmicity"
            )
        )
    if "amplitude_uv" in value:
        descriptors.append(f"波幅 {_range_text(value['amplitude_uv'], unit='μV')}")
    if descriptors:
        text += "；" + "，".join(descriptors)
    derivations = _join_codes(
        value.get("derivations"),
        context="sustained change derivations",
        empty=_NO_QUALIFIED_DERIVATION,
    )
    # The current producer is longitudinal-bipolar.  Even if a historical
    # generic ledger carries endpoint/spatial sibling fields, this long-record
    # renderer never promotes them: one bipolar difference cannot identify
    # either endpoint as a maximal electrode, region, laterality or SOZ.
    electrode_pieces: list[str] = []
    if derivations != _NO_QUALIFIED_DERIVATION:
        electrode_pieces.append(f"分析导联：{derivations}")
    return text, "；".join(electrode_pieces) or _NO_QUALIFIED_DERIVATION


def _later_change_text(event: Mapping[str, Any]) -> str:
    pieces: list[str] = []
    sustained_value = qualified_sustained_change_value(event)
    if sustained_value is not None:
        trajectory = sustained_value.get("quantitative_trajectory")
        if trajectory is not None:
            item = _mapping(trajectory, "quantitative sustained-change trajectory")
            if item.get("amplitude_change_alone_is_not_ictal_evolution") is not True:
                raise ValueError("quantitative trajectory lost its non-ictal boundary")
            offset = _finite(
                item.get("comparison_offset_seconds"),
                "trajectory comparison offset",
                minimum=0.0,
            )
            dimensions = _join_codes(
                item.get("change_dimensions"),
                context="trajectory dimensions",
                labels=_CHANGE_ZH,
            )
            early_frequency = _finite(
                item.get("early_frequency_hz"), "early trajectory frequency", minimum=0.0
            )
            late_frequency = _finite(
                item.get("late_frequency_hz"), "late trajectory frequency", minimum=0.0
            )
            early_amplitude = _finite(
                item.get("early_amplitude_uv"), "early trajectory amplitude", minimum=0.0
            )
            late_amplitude = _finite(
                item.get("late_amplitude_uv"), "late trajectory amplitude", minimum=0.0
            )
            pieces.append(
                f"持续变化起点后 {offset:g} 秒的前后段量化轨迹：{dimensions}；"
                f"主频 {early_frequency:g}→{late_frequency:g} Hz，"
                f"波幅 {early_amplitude:g}→{late_amplitude:g} μV"
            )
        later_derivations = sustained_value.get("later_derivation_changes")
        if later_derivations is not None:
            if not isinstance(later_derivations, list):
                raise TypeError("later derivation changes must be a list")
            values: list[str] = []
            for raw in later_derivations:
                observation = _mapping(raw, "later derivation observation")
                derivation = observation.get("derivation")
                if not isinstance(derivation, str) or not derivation:
                    raise TypeError("later derivation observation is missing a derivation")
                delay = _finite(
                    observation.get("delay_seconds"),
                    "later derivation delay",
                    minimum=0.0,
                )
                values.append(f"{derivation}（+{delay:.3f} 秒）")
            pieces.append(
                "后续达到同一量化门槛的双极导联："
                + "、".join(values)
            )
        return_offset = sustained_value.get(
            "candidate_return_to_baseline_offset_seconds"
        )
        if return_offset is not None:
            offset = _finite(
                return_offset, "candidate return-to-baseline offset", minimum=0.0
            )
            pieces.append(
                f"持续变化起点后 {offset:g} 秒：量化变化回到冻结基线门槛以下"
                "（回归基线候选，非确认发作终止）"
            )

    # Historical ledgers retain their original typed facts.  Their renderer
    # remains bounded, while the current quantitative producer no longer emits
    # these ictal-labelled fact types.
    for fact, value in sorted(
        _fact_values(event, "ictal_evolution"),
        key=lambda item: int(item[1].get("sequence_index", 0)),
    ):
        dimensions_raw = value.get("change_dimensions")
        morphology_required = (
            isinstance(dimensions_raw, list) and "morphology" in dimensions_raw
        )
        spatial_required = (
            isinstance(dimensions_raw, list)
            and "spatial_distribution" in dimensions_raw
        )
        if not _physician_or_independently_qualified_fact(
            fact,
            value,
            morphology_required=morphology_required,
            spatial_required=spatial_required,
        ):
            continue
        offset = _finite(value.get("onset_offset_seconds"), "evolution offset", minimum=0.0)
        dimensions = _join_codes(
            value.get("change_dimensions"), context="evolution dimensions", labels=_CHANGE_ZH
        )
        detail: list[str] = []
        if "frequency_hz" in value:
            detail.append(_range_text(value["frequency_hz"], unit="Hz"))
        if "amplitude_uv" in value:
            detail.append(_range_text(value["amplitude_uv"], unit="μV"))
        if "morphology" in value:
            detail.append(
                _closed_label(value.get("morphology"), _PATTERN_ZH, "evolution morphology")
            )
        suffix = f"（{'，'.join(detail)}）" if detail else ""
        pieces.append(f"候选变化起点后 {offset:g} 秒：{dimensions}改变{suffix}")

    later = _one_fact(event, "later_scalp_visible_eeg_change")
    if later is not None and _qualified_signal_value(later[1]):
        observations = later[1].get("observations")
        if not isinstance(observations, list):
            raise TypeError("later scalp-visible observations must be a list")
        values = []
        for raw in observations:
            observation = _mapping(raw, "later scalp-visible observation")
            electrode = observation.get("electrode")
            if not isinstance(electrode, str) or not electrode:
                raise TypeError("later scalp-visible electrode is missing")
            delay = _finite(observation.get("delay_seconds"), "later change delay", minimum=0.0)
            values.append(f"{electrode}（+{delay:.3f} 秒）")
        pieces.append(
            "后续达到同一量化门槛："
            + "、".join(values)
        )
    spread = _one_fact(event, "ictal_spread")
    if spread is not None and _physician_or_independently_qualified_fact(
        spread[0],
        spread[1],
        spatial_required=True,
    ):
        value = spread[1]
        offset = _finite(value.get("onset_offset_seconds"), "spread offset", minimum=0.0)
        destinations = _join_codes(value.get("to_electrodes"), context="spread electrodes")
        pieces.append(f"候选变化起点后 {offset:g} 秒：头皮可见范围至 {destinations}")
    termination = _one_fact(event, "ictal_termination")
    if termination is not None and _physician_or_independently_qualified_fact(
        termination[0], termination[1]
    ):
        termination_value = termination[1]
        offset = _finite(termination_value.get("offset_seconds"), "termination offset", minimum=0.0)
        if _qualified_signal_value(termination_value):
            pieces.append(
                f"候选变化起点后 {offset:g} 秒：量化变化回到冻结基线门槛以下"
                "（待复核终止候选）"
            )
        else:
            pieces.append(f"候选变化起点后 {offset:g} 秒：电图变化终止")
    postictal = _one_fact(event, "postictal_pattern")
    if postictal is not None and _physician_or_independently_qualified_fact(
        postictal[0], postictal[1]
    ):
        pattern = _closed_label(postictal[1].get("pattern"), _PATTERN_ZH, "post-event pattern")
        pieces.append(f"随后：{pattern}")
    return "；".join(pieces)


def _event_narrative(
    event: Mapping[str, Any],
    language: Mapping[str, Mapping[str, str]],
) -> tuple[str, str]:
    wording = language.get(str(event["eeg_event_id"]), {})
    onset = _qualified_onset_text(event)
    sustained_value = qualified_sustained_change_value(event)
    sustained_text, sustained_derivations = _sustained_text(event)
    lines: list[str] = []
    support: list[str] = []

    if onset is not None:
        onset_text, onset_support = onset
        lines.append(
            "头皮可见起始："
            + wording.get("onset_text_zh", onset_text)
        )
        if onset_support != _NO_QUALIFIED_DERIVATION:
            support.append(onset_support)

    if sustained_value is not None:
        # Neutral sustained-change facts are already complete and typed.  Old
        # per-event Qwen records often restated the same safety disclaimer for
        # every row, so use the deterministic fact projection here.  Qwen may
        # still lend wording to independently qualified onset/evolution facts.
        lines.append("信号所见：" + sustained_text)
        later = _later_change_text(event)
        if later:
            lines.append("量化轨迹与后续变化：" + later)
        if sustained_derivations != _NO_QUALIFIED_DERIVATION:
            support.append(sustained_derivations)
    elif onset is not None:
        later = wording.get("evolution_spread_text_zh") or _later_change_text(event)
        if later:
            lines.append("演变与后续变化：" + later)

    termination_wording = wording.get("termination_postictal_text_zh")
    termination_authorized = any(
        fact.get("fact_type") in {"ictal_termination", "postictal_pattern"}
        and _event_fact_language_authorized(event, fact)
        for fact in _facts(event)
    )
    if termination_wording and termination_authorized:
        lines.append("终止与事件后：" + termination_wording)
    return "\n".join(lines), "；".join(dict.fromkeys(support)) or _NO_QUALIFIED_DERIVATION


def _has_qualified_event_finding(event: Mapping[str, Any]) -> bool:
    return (
        qualified_ictal_onset_value(event) is not None
        or qualified_sustained_change_value(event) is not None
    )


def _event_rows(
    bundle: Mapping[str, Any],
    language: Mapping[str, Mapping[str, str]],
) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for event in _events(bundle):
        if not _has_qualified_event_finding(event):
            continue
        start, stop = _event_interval(event)
        narrative, derivations = _event_narrative(event, language)
        rows.append(
            (
                f"事件{event['event_number']}\n候选支持窗："
                f"{_clock(start)}–{_clock(stop)}",
                narrative,
                derivations,
            )
        )
    return rows


def _event_findings_intro(
    bundle: Mapping[str, Any],
    *,
    detector_selected: int,
    analyzable_count: int,
    rejected_count: int,
    finding_count: int,
) -> str:
    if finding_count <= 0 or finding_count > analyzable_count:
        raise ValueError("event Findings count is inconsistent")
    qualified = (
        f"其中 {finding_count} 个形成通过资格门槛的事件级 EEG 信号所见，列于下表。"
    )
    if "analysis_selection" in bundle:
        return (
            f"全记录信号粗筛形成 {detector_selected} 个待复核候选；"
            f"{analyzable_count} 个通过信号资格并进入分析，{rejected_count} 个未进入。"
            + qualified
            + "下表时间均相对于原记录起点；表内区间为候选支持窗，不等同于经医师"
            "确认的发作起止。通过资格的候选在锚点前 12 秒至后 48 秒片段上分析。"
        )
    return (
        f"全记录信号粗筛形成 {len(_events(bundle))} 个待复核候选。"
        + qualified
        + "下表时间均相对于原记录起点；表内区间为候选支持窗，"
        "不等同于经医师确认的发作起止。每个候选另在锚点前 12 秒至后 48 秒片段上分析。"
    )


def _counted_labels(
    counts: Mapping[str, Any],
    labels: Mapping[str, str],
) -> str:
    values = [
        (str(code), int(count))
        for code, count in counts.items()
        if str(code) in labels and isinstance(count, int) and count > 0
    ]
    values.sort(key=lambda item: (-item[1], item[0]))
    return "、".join(f"{labels[code]} {count} 个" for code, count in values)


def _record_findings_text(
    summary: Mapping[str, Any],
    *,
    detector_selected: int,
    analyzable_count: int,
    rejected_count: int,
) -> str:
    qualified = int(summary["qualified_signal_event_count"])
    without = int(summary["events_without_qualified_signal_findings"])
    if detector_selected == 0:
        return (
            "全记录粗筛未形成待复核候选；该结果不等同于未见脑电异常。"
        )
    pieces = [
        f"全记录粗筛形成 {detector_selected} 个待复核候选；"
        f"{analyzable_count} 个进入事件级信号分析，{rejected_count} 个因信号或时间"
        "支持资格不足未进入。"
    ]
    if analyzable_count:
        pieces.append(
            f"已分析候选中 {qualified} 个形成通过伪迹、持续性和跨窗"
            f"复现门槛的量化头皮 EEG 变化，{without} 个未形成合格信号所见。"
        )
    support = _mapping(summary.get("cross_event_support"), "cross-event support")
    if qualified:
        band_text = _counted_labels(
            _mapping(support.get("frequency_band_counts"), "frequency band counts"),
            _FREQUENCY_BAND_ZH,
        )
        rhythm_text = _counted_labels(
            _mapping(support.get("rhythmicity_counts"), "rhythmicity counts"),
            _RHYTHMICITY_ZH,
        )
        if band_text:
            pieces.append("合格候选的主导频段分布：" + band_text + "。")
        if rhythm_text:
            pieces.append("节律性分布：" + rhythm_text + "。")
        recurring = support.get("recurring_bipolar_derivations")
        if not isinstance(recurring, list):
            raise TypeError("recurring bipolar derivations must be a list")
        if recurring:
            recurrence_text = "、".join(
                f"{item['derivation']} {int(item['event_count'])}/{qualified} 个候选"
                for item in recurring[:6]
            )
            pieces.append(
                "跨候选重复出现的主要双极导联："
                + recurrence_text
                + "。该重复性仅是导联级量化支持，不直接等同于侧别、脑区或 SOZ。"
            )
        morphology_text = _counted_labels(
            _mapping(
                support.get(
                    "morphology_counts_from_independently_qualified_onsets_only"
                ),
                "qualified morphology counts",
            ),
            _PATTERN_ZH,
        )
        if morphology_text:
            pieces.append(
                "仅在通过独立起始/形态资格门槛的候选中，起始形态为："
                + morphology_text
                + "。"
            )
    return "".join(pieces)


def _automatic_eeg_impression(bundle: Mapping[str, Any]) -> dict[str, str]:
    """Synthesize all signal-qualified events, then apply the SOZ hard gate."""

    detector_selected, analyzable_count, rejected_count = (
        _analysis_candidate_counts(bundle)
    )
    summary = build_recording_eeg_findings_summary(bundle)
    findings = _record_findings_text(
        summary,
        detector_selected=detector_selected,
        analyzable_count=analyzable_count,
        rejected_count=rejected_count,
    )

    diagnostic_outcome = classify_recording_eeg_outcome(bundle)
    diagnostic_status = str(diagnostic_outcome["report_status"])
    if diagnostic_status == "completed_localizable":
        localization = (
            "仅依据已经通过独立电图发作、起始及空间电场门槛的事实形成以下"
            "头皮 EEG 倾向："
            + str(diagnostic_outcome["conclusion_zh"])
        )
    elif diagnostic_status == "completed_nonlocalizable":
        localization = (
            "独立合格的电图起始事实未形成可合并的局灶空间结论："
            + str(diagnostic_outcome["conclusion_zh"])
        )
    else:
        recurring = _mapping(
            summary.get("cross_event_support"), "cross-event support"
        ).get("recurring_bipolar_derivations")
        if detector_selected > 0 and analyzable_count == 0:
            localization = (
                "所有粗筛候选均未通过信号/时间支持资格，未形成可用于 SOZ 推理的"
                "候选窗事实；SOZ 定位证据不足，无法判断。"
            )
        elif isinstance(recurring, list) and recurring:
            localization = (
                "部分双极导联在多个合格量化变化候选中重复出现，"
                "但双极电位差无法单独决定最大电极、侧别或脑区；又未形成通过"
                "独立电图发作起始和空间电场门槛的事实，因此 SOZ 定位证据不足。"
            )
        else:
            localization = (
                "当前只有双极导联级量化变化或无合格变化；未形成通过独立电图发作、"
                "起始和空间电场门槛的侧别/区域事实，SOZ 定位无法评估。"
            )

    uncertainty = (
        "量化持续变化只表示头皮双极导联级信号改变；未通过各自独立"
        "资格器的形态、起始、演变、空间扩展、终止及事件后改变不进入 Findings。"
        "头皮 EEG 最多提供 SOZ 定位倾向，"
        "不能单独确定皮层 SOZ，更不能单独判定致痫区（EZ）或治疗靶点。"
        "本流程不能单独判定正常或异常脑电图。"
    )
    return {
        "findings": findings,
        "localization": localization,
        "uncertainty": uncertainty,
        "diagnostic_conclusion": str(diagnostic_outcome["conclusion_zh"]),
        "diagnostic_status": str(diagnostic_outcome["report_status"]),
    }


def _waveform_attachment(event: Mapping[str, Any]) -> Mapping[str, Any] | None:
    value = event.get("waveform_attachment")
    if value is None:
        return None
    return _mapping(value, "waveform attachment")


def _waveform_key_scheme(
    events: Sequence[Mapping[str, Any]], values: Mapping[str, object] | None
) -> tuple[Mapping[str, object] | None, str | None]:
    attachments = [(event, _waveform_attachment(event)) for event in events]
    attachments = [(event, attachment) for event, attachment in attachments if attachment is not None]
    if values is None:
        return None, None
    if not isinstance(values, Mapping):
        raise TypeError("waveform values must be a mapping")
    key_sets = {
        "evidence_id": {str(attachment["evidence_id"]) for _, attachment in attachments},
        "attachment_id": {str(attachment["attachment_id"]) for _, attachment in attachments},
        "eeg_event_id": {str(event["eeg_event_id"]) for event, _ in attachments},
    }
    supplied = set(values)
    matches = [name for name, expected in key_sets.items() if supplied == expected]
    if not attachments and not supplied:
        return values, "evidence_id"
    if not matches:
        raise ValueError("waveform mapping keys must exactly match all rendered attachments")
    return values, matches[0]


def _waveform_value(
    event: Mapping[str, Any],
    attachment: Mapping[str, Any],
    values: Mapping[str, object],
    scheme: str,
) -> object:
    key = event["eeg_event_id"] if scheme == "eeg_event_id" else attachment[scheme]
    return values[str(key)]


def _safe_href(value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TypeError("waveform href must be a non-empty trimmed string")
    if "\\" in value or ":" in value or value.startswith("/"):
        raise ValueError("waveform href must be a safe relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.suffix.lower() != ".png"
    ):
        raise ValueError("waveform href must be a canonical relative PNG path")
    return value


def _png_payload(value: object, expected_sha256: str) -> tuple[bytes, int, int]:
    if not isinstance(value, (str, Path)):
        raise TypeError("waveform path must be a filesystem path")
    path = Path(value)
    if path.is_symlink() or not path.is_file() or path.suffix.lower() != ".png":
        raise ValueError("waveform path must be a regular PNG file")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("waveform PNG SHA-256 does not match its canonical attachment")
    if len(payload) < 24 or not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("waveform attachment is not a PNG")
    width, height = struct.unpack(">II", payload[16:24])
    if width <= 0 or height <= 0:
        raise ValueError("waveform PNG has invalid dimensions")
    return payload, width, height


def _waveform_window_configuration(
    attachment: Mapping[str, Any],
) -> tuple[float, float, float]:
    display_window = attachment.get("event_window_seconds")
    if not isinstance(display_window, list) or len(display_window) != 2:
        raise ValueError("waveform display window is malformed")
    display_start = _finite(display_window[0], "waveform display start")
    display_stop = _finite(display_window[1], "waveform display stop")
    if display_stop <= display_start:
        raise ValueError("waveform display window is empty or reversed")
    anchor = _finite(
        attachment.get("event_anchor_offset_seconds"),
        "waveform anchor offset",
        minimum=0.0,
    )
    if anchor > display_stop - display_start:
        raise ValueError("waveform anchor lies outside the display window")
    return display_start, display_stop, anchor


def _waveform_evidence_interval(
    attachment: Mapping[str, Any],
) -> tuple[float, float]:
    evidence_window = attachment.get("evidence_interval_seconds_relative_to_anchor")
    if not isinstance(evidence_window, list) or len(evidence_window) != 2:
        raise ValueError("waveform evidence interval is malformed")
    window_start = _finite(evidence_window[0], "waveform evidence start")
    window_stop = _finite(evidence_window[1], "waveform evidence stop")
    if window_stop <= window_start:
        raise ValueError("waveform evidence interval is empty or reversed")
    return window_start, window_stop


def _waveform_context_lines(
    events: Sequence[Mapping[str, Any]],
    settings: Mapping[str, str],
) -> list[str]:
    groups: dict[tuple[float, float, float], list[int]] = {}
    for event in events:
        attachment = _waveform_attachment(event)
        if attachment is None:
            continue
        configuration = _waveform_window_configuration(attachment)
        _waveform_evidence_interval(attachment)
        groups.setdefault(configuration, []).append(int(event["event_number"]))
    if not groups:
        return []

    lines = [
        "本节波形共同采集设置："
        f"采样率 {settings['sampling_rate']}；滤波 {settings['filter']}；"
        f"参考/蒙太奇 {settings['reference_montage']}。"
    ]
    one_configuration = len(groups) == 1
    for (display_start, display_stop, anchor), event_numbers in groups.items():
        label = (
            "共同窗口配置"
            if one_configuration
            else "窗口配置（"
            + "、".join(f"事件{number}" for number in event_numbers)
            + "）"
        )
        lines.append(
            f"{label}：显示范围为候选锚点 {display_start:+.3f} 至 "
            f"{display_stop:+.3f} 秒；候选锚点位于图窗起点后 {anchor:.3f} 秒。"
        )
    lines.append(
        "图中标记对应各事件的待复核候选支持区间，具体原记录区间见各图说明；"
        "标记仅用于定位待复核的信号变化。"
    )
    return lines


def _waveform_caption(
    event: Mapping[str, Any],
    attachment: Mapping[str, Any],
) -> str:
    _waveform_window_configuration(attachment)
    window_start, window_stop = _waveform_evidence_interval(attachment)
    start, stop = _event_interval(event)
    anchor = _finite(
        event.get("candidate_anchor_offset_seconds"),
        "candidate anchor offset",
        minimum=0.0,
    )
    if not (
        math.isclose(start, anchor + window_start, rel_tol=0.0, abs_tol=1e-6)
        and math.isclose(stop, anchor + window_stop, rel_tol=0.0, abs_tol=1e-6)
    ):
        raise ValueError("waveform support interval is not bound to the event interval")
    return (
        f"事件{event['event_number']}：原记录候选支持窗 "
        f"{_clock(start)}–{_clock(stop)}。"
    )


def _soz_rows(bundle: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for event in _events(bundle):
        receipt = event.get("research_soz_ranking_receipt")
        if receipt is None:
            continue
        ranking = _mapping(receipt, "research electrode ranking receipt")
        if ranking.get("interpretation_status") != "research_scalp_electrode_ranking_not_clinical_soz":
            raise ValueError("research electrode ranking interpretation boundary is missing")
        raw_electrodes = ranking.get("ranked_electrodes")
        if not isinstance(raw_electrodes, list):
            raise TypeError("ranked_electrodes must be a list")
        displayed = []
        for index, raw in enumerate(raw_electrodes, start=1):
            item = _mapping(raw, "ranked scalp electrode")
            if item.get("rank") != index:
                raise ValueError("research electrode ranking order drifted")
            electrode = item.get("electrode")
            if not isinstance(electrode, str) or not electrode:
                raise TypeError("ranked scalp electrode is missing")
            score = _finite(item.get("score"), "research electrode score")
            if index <= 3:
                displayed.append(f"{index}. {electrode}（评分 {score:.4g}）")
        rows.append(
            (
                f"事件{event['event_number']}",
                "；".join(displayed) or "无候选",
                "非临床事实，仅供研究复核",
            )
        )
    return rows


def _research_channel_consistency_text(bundle: Mapping[str, Any]) -> str:
    summary = build_research_multievent_channel_consistency(bundle, top_k=3)
    ranked = int(summary["ranked_event_count"])
    if ranked == 0:
        return ""
    support = summary.get("channel_support")
    if not isinstance(support, list):
        raise TypeError("research channel consistency support must be a list")
    if not support:
        return ""
    items = []
    for raw in support[:8]:
        item = _mapping(raw, "research channel support")
        rate = _finite(item.get("event_support_rate"), "event support rate", minimum=0.0)
        items.append(
            f"{item['electrode']}：{int(item['event_support_count'])}/{ranked} 个候选"
            f"（{rate * 100:.1f}%，top-1 {int(item['top1_event_count'])} 次）"
        )
    return (
        f"研究排名覆盖 {ranked} 个已分析候选；top-3 通道跨候选支持率："
        + "；".join(items)
        + "。该一致性只表示 v29 研究排名的重复性。"
    )


def _html_table(
    css_class: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
) -> str:
    heading = "".join(f"<th>{html_escape(value)}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html_escape(value)}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return (
        f'<table class="{html_escape(css_class, quote=True)}"><thead><tr>{heading}</tr></thead>'
        f"<tbody>{body}</tbody></table>"
    )


def render_long_term_html(
    bundle: object,
    context: object | None = None,
    waveform_hrefs: Mapping[str, str] | None = None,
    language_layer: object | None = None,
) -> str:
    """Render a signal-only draft; any external context fails closed."""

    source, _ = _validated_bundle_with_context(bundle, context)
    events = _events(source)
    detector_selected, analyzable_count, rejected_count = (
        _analysis_candidate_counts(source)
    )
    rejection_rows = _analysis_rejection_rows(source)
    event_language = _fact_locked_event_language(source, language_layer)
    href_values, href_scheme = _waveform_key_scheme(events, waveform_hrefs)
    coverage_intervals, coverage_percent = _coverage(source)
    signal_settings = _recording_signal_settings(source)
    impression = _automatic_eeg_impression(source)
    finding_rows = _event_rows(source, event_language)
    if finding_rows:
        intro = _event_findings_intro(
            source,
            detector_selected=detector_selected,
            analyzable_count=analyzable_count,
            rejected_count=rejected_count,
            finding_count=len(finding_rows),
        )
        findings_html = (
            f"<p>{html_escape(intro)}</p>"
            + _html_table(
                "events",
                (
                    "事件/原记录相对候选支持窗",
                    "候选事件脑电描述",
                    "相关双极导联",
                ),
                finding_rows,
            )
        )
    else:
        findings_html = f'<p>{html_escape(_NO_QUALIFIED_EVENT_FINDINGS)}</p>'
    waveform_context_html = "".join(
        f'<p class="audit">{html_escape(line)}</p>'
        for line in _waveform_context_lines(events, signal_settings)
    )
    waveform_parts: list[str] = []
    for event in events:
        attachment = _waveform_attachment(event)
        if attachment is None:
            waveform_parts.append(
                f'<section class="waveform"><h3>事件{event["event_number"]}</h3>'
                '<p class="placeholder">未附可验证波形。</p></section>'
            )
            continue
        caption = _waveform_caption(event, attachment)
        if href_values is None or href_scheme is None:
            body = '<p class="placeholder">波形附件未嵌入当前 HTML；请核对受控附件。</p>'
        else:
            href = _safe_href(_waveform_value(event, attachment, href_values, href_scheme))
            body = (
                f'<img src="{html_escape(href, quote=True)}" '
                f'alt="事件{int(event["event_number"])}脑电波形证据">'
            )
        waveform_parts.append(
            f'<figure class="waveform">{body}<figcaption>{html_escape(caption)}</figcaption></figure>'
        )
    waveform_html = "".join(waveform_parts) or '<p class="placeholder">本记录无事件波形附件。</p>'
    soz_rows = _soz_rows(source)
    research_consistency = _research_channel_consistency_text(source)
    research_appendix = (
        "<h2>研究性附录</h2>"
        "<h3>多候选头皮电极排序一致性（研究性）</h3>"
        f'<p class="boundary">{html_escape(_SOZ_DISCLAIMER)}</p>'
        f"<p>{html_escape(research_consistency)}</p>"
        + _html_table(
            "soz",
            ("候选", "研究性 top-3 头皮电极排序（非临床事实）", "用途边界"),
            soz_rows,
        )
        if soz_rows and research_consistency
        else ""
    )
    rejection_html = (
            "<h3>未进入事件级分析的粗筛候选</h3>"
        '<p class="boundary">以下为信号/时间支持资格拒绝，不表示未发作，'
        "也不得解释为正常脑电。</p>"
        + _html_table(
            "analysis-rejections",
            ("候选事件 ID", "候选锚点", "纯信号原因", "解释边界"),
            rejection_rows,
        )
        if rejection_rows
        else ""
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>长程头皮脑电分析报告（AI草稿）</title>
<style>
@page {{ size: A4; margin: 15mm; }}
body {{ margin: 0 auto; max-width: 1080px; padding: 26px; color: #20242b; font-family: "Noto Serif CJK SC","SimSun",serif; line-height: 1.6; }}
h1 {{ text-align: center; font-size: 26px; margin: 0; }}
.draft {{ text-align: center; color: #962f2f; font-weight: 700; margin: 5px 0 20px; }}
h2 {{ border-top: 3px solid #30353d; padding-top: 9px; font-size: 20px; }}
h3 {{ font-size: 16px; }}
p {{ white-space: pre-wrap; }}
table {{ width: 100%; border-collapse: collapse; table-layout: fixed; margin: 10px 0 18px; }}
thead {{ display: table-header-group; }}
tr, figure {{ break-inside: avoid; page-break-inside: avoid; }}
th,td {{ border: 1px solid #4b515a; padding: 7px; vertical-align: top; overflow-wrap: anywhere; }}
th {{ background: #e8edf2; text-align: center; }}
.metadata th {{ width: 18%; }}
.events th:nth-child(1) {{ width: 23%; }}
.events th:nth-child(2) {{ width: 52%; }}
.events th:nth-child(3) {{ width: 25%; }}
.events td:first-child {{ white-space: pre-line; }}
.placeholder,.audit {{ color: #606a75; }}
.waveform {{ margin: 15px 0 24px; }}
.waveform img {{ width: 100%; height: auto; display: block; border: 1px solid #b5bdc7; }}
.waveform figcaption {{ margin-top: 6px; color: #505b67; font-size: 13px; overflow-wrap: anywhere; }}
.boundary {{ border: 1px solid #ad7b20; background: #fff8e8; padding: 10px; font-weight: 600; }}
.review {{ margin-top: 28px; border: 1px solid #9ba3ad; padding: 12px; }}
</style>
</head>
<body>
<h1>长程头皮脑电分析报告</h1>
<div class="draft">{html_escape(_AI_DRAFT)}</div>
<h2>记录信息</h2>
{_html_table("metadata", ("记录字段", "内容"), (
    ("去标识记录 ID", str(source["recording_id"])),
    ("记录总时长", f"{_duration(source['recording_duration_seconds'])}（{_clock(source['recording_duration_seconds'])}）"),
    ("粗筛覆盖区间", coverage_intervals),
    ("粗筛覆盖比例", coverage_percent),
    ("粗筛器", _detector_display(source)),
    ("粗筛输出含义", "仅形成待复核候选，不确认发作"),
    ("采样率", signal_settings["sampling_rate"]),
    ("滤波", signal_settings["filter"]),
    ("参考/蒙太奇", signal_settings["reference_montage"]),
))}
<p class="audit">{html_escape(_EEG_FINDINGS_SCOPE)}</p>
<h2>脑电图表现</h2>
<h3>候选事件脑电（待复核）</h3>
{findings_html}
{rejection_html}
<h2>相关 EEG 波形证据</h2>
{waveform_context_html}
{waveform_html}
<p class="audit">波形图是待复核证据附件，不构成独立诊断。</p>
<h2>脑电图印象</h2>
<p>一、全记录候选信号所见：{html_escape(impression['findings'])}</p>
<p>二、头皮分布与定位推理：{html_escape(impression['localization'])}</p>
<p>三、SOZ 定位结论：{html_escape(impression['diagnostic_conclusion'])}</p>
<p>四、不确定性与结论边界：{html_escape(impression['uncertainty'])}</p>
<p class="boundary">最终结论须由脑电医师复核原始长程 EEG 后签署。</p>
{research_appendix}
<div class="review"><strong>审核状态：</strong>AI 草稿<br><strong>审核医师：</strong>________________<br><strong>签署日期：</strong>________________</div>
</body>
</html>
"""


def _w_run(text: str, *, bold: bool = False, size: int = 20) -> str:
    properties = (
        "<w:rPr>"
        '<w:rFonts w:ascii="SimSun" w:eastAsia="宋体" w:hAnsi="SimSun"/>'
        + ("<w:b/>" if bold else "")
        + f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>'
        "</w:rPr>"
    )
    return f'<w:r>{properties}<w:t xml:space="preserve">{xml_escape(text)}</w:t></w:r>'


def _w_paragraph(
    text: str,
    *,
    bold: bool = False,
    size: int = 20,
    align: str | None = None,
    keep_next: bool = False,
) -> str:
    properties = []
    if align:
        properties.append(f'<w:jc w:val="{align}"/>')
    if keep_next:
        properties.append("<w:keepNext/>")
    ppr = f"<w:pPr>{''.join(properties)}</w:pPr>" if properties else ""
    lines = text.splitlines() or [""]
    runs = "<w:br/>".join(_w_run(line, bold=bold, size=size) for line in lines)
    return f"<w:p>{ppr}{runs}</w:p>"


def _w_cell(text: str, *, bold: bool, width: int, shade: str | None = None) -> str:
    shading = f'<w:shd w:val="clear" w:color="auto" w:fill="{shade}"/>' if shade else ""
    return (
        f'<w:tc><w:tcPr><w:tcW w:w="{width}" w:type="dxa"/>{shading}</w:tcPr>'
        f"{_w_paragraph(text, bold=bold)}</w:tc>"
    )


def _w_table(
    headers: Sequence[tuple[str, int]],
    rows: Sequence[Sequence[str]],
) -> str:
    if any(len(row) != len(headers) for row in rows):
        raise ValueError("DOCX table row width drifted")
    header_cells = "".join(
        _w_cell(label, bold=True, width=width, shade="DCE6F1") for label, width in headers
    )
    body = [
        f"<w:tr><w:trPr><w:tblHeader/><w:cantSplit/></w:trPr>{header_cells}</w:tr>"
    ]
    for row in rows:
        cells = "".join(
            _w_cell(str(value), bold=False, width=width)
            for value, (_, width) in zip(row, headers, strict=True)
        )
        body.append(f"<w:tr><w:trPr><w:cantSplit/></w:trPr>{cells}</w:tr>")
    return (
        '<w:tbl><w:tblPr><w:tblW w:w="0" w:type="auto"/><w:tblBorders>'
        '<w:top w:val="single" w:sz="6" w:color="555555"/>'
        '<w:left w:val="single" w:sz="6" w:color="555555"/>'
        '<w:bottom w:val="single" w:sz="6" w:color="555555"/>'
        '<w:right w:val="single" w:sz="6" w:color="555555"/>'
        '<w:insideH w:val="single" w:sz="4" w:color="777777"/>'
        '<w:insideV w:val="single" w:sz="4" w:color="777777"/>'
        f"</w:tblBorders></w:tblPr>{''.join(body)}</w:tbl>"
    )


def _w_image_paragraph(
    *,
    relationship_id: str,
    drawing_id: int,
    width_px: int,
    height_px: int,
) -> str:
    # Fit inside the printable page while preserving the original aspect ratio.
    max_width_emu = 6_250_000
    max_height_emu = 7_400_000
    scale = min(max_width_emu / width_px, max_height_emu / height_px)
    width_emu = max(1, int(round(width_px * scale)))
    height_emu = max(1, int(round(height_px * scale)))
    name = f"EEG waveform {drawing_id}"
    safe_name = xml_escape(name, {'"': "&quot;"})
    return (
        '<w:p><w:pPr><w:jc w:val="center"/><w:keepNext/></w:pPr><w:r><w:drawing>'
        '<wp:inline distT="0" distB="0" distL="0" distR="0">'
        f'<wp:extent cx="{width_emu}" cy="{height_emu}"/>'
        '<wp:effectExtent l="0" t="0" r="0" b="0"/>'
        f'<wp:docPr id="{drawing_id}" name="{safe_name}" descr="受控 EEG 波形证据"/>'
        '<wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/>'
        '</wp:cNvGraphicFramePr><a:graphic>'
        '<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        '<pic:pic><pic:nvPicPr>'
        f'<pic:cNvPr id="{drawing_id}" name="{safe_name}"/><pic:cNvPicPr/>'
        '</pic:nvPicPr><pic:blipFill>'
        f'<a:blip r:embed="{relationship_id}"/><a:stretch><a:fillRect/></a:stretch>'
        '</pic:blipFill><pic:spPr><a:xfrm><a:off x="0" y="0"/>'
        f'<a:ext cx="{width_emu}" cy="{height_emu}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
        '</pic:spPr></pic:pic></a:graphicData></a:graphic>'
        '</wp:inline></w:drawing></w:r></w:p>'
    )


def _heading(text: str) -> str:
    return _w_paragraph(text, bold=True, size=27, keep_next=True)


def _document_xml(
    bundle: Mapping[str, Any],
    event_language: Mapping[str, Mapping[str, str]],
    image_records: Sequence[tuple[Mapping[str, Any], Mapping[str, Any], bytes, int, int]],
) -> str:
    events = _events(bundle)
    detector_selected, analyzable_count, rejected_count = (
        _analysis_candidate_counts(bundle)
    )
    rejection_rows = _analysis_rejection_rows(bundle)
    coverage_intervals, coverage_percent = _coverage(bundle)
    signal_settings = _recording_signal_settings(bundle)
    impression = _automatic_eeg_impression(bundle)
    finding_rows = _event_rows(bundle, event_language)
    parts = [
        _w_paragraph("长程头皮脑电分析报告", bold=True, size=32, align="center"),
        _w_paragraph(_AI_DRAFT, bold=True, size=20, align="center"),
        _heading("记录信息"),
        _w_table(
            (("记录字段", 2400), ("内容", 6800)),
            (
                ("去标识记录 ID", str(bundle["recording_id"])),
                (
                    "记录总时长",
                    f"{_duration(bundle['recording_duration_seconds'])}（{_clock(bundle['recording_duration_seconds'])}）",
                ),
                ("粗筛覆盖区间", coverage_intervals),
                ("粗筛覆盖比例", coverage_percent),
                ("粗筛器", _detector_display(bundle)),
                ("粗筛输出含义", "仅形成待复核候选，不确认发作"),
                ("采样率", signal_settings["sampling_rate"]),
                ("滤波", signal_settings["filter"]),
                ("参考/蒙太奇", signal_settings["reference_montage"]),
            ),
        ),
        _w_paragraph(_EEG_FINDINGS_SCOPE, size=18),
        _heading("脑电图表现"),
    ]
    parts.append(
        _w_paragraph("候选事件脑电（待复核）", bold=True, size=23, keep_next=True)
    )
    if finding_rows:
        parts.extend(
            [
                _w_paragraph(
                    _event_findings_intro(
                        bundle,
                        detector_selected=detector_selected,
                        analyzable_count=analyzable_count,
                        rejected_count=rejected_count,
                        finding_count=len(finding_rows),
                    )
                ),
                _w_table(
                (
                    ("事件/原记录相对候选支持窗", 2200),
                    ("候选事件脑电描述", 4700),
                    ("相关双极导联", 2300),
                ),
                    finding_rows,
                ),
            ]
        )
    else:
        parts.append(_w_paragraph(_NO_QUALIFIED_EVENT_FINDINGS))
    if rejection_rows:
        parts.extend(
            [
                _w_paragraph(
                    "未进入事件级分析的粗筛候选", bold=True, size=23, keep_next=True
                ),
                _w_paragraph(
                    "以下为信号/时间支持资格拒绝，不表示未发作，也不得解释为正常脑电。",
                    bold=True,
                    size=18,
                ),
                _w_table(
                    (
                        ("候选事件 ID", 1900),
                        ("候选锚点", 1400),
                        ("纯信号原因", 3400),
                        ("解释边界", 2500),
                    ),
                    rejection_rows,
                ),
            ]
        )
    parts.append(_heading("相关 EEG 波形证据"))
    parts.extend(
        _w_paragraph(line, size=18)
        for line in _waveform_context_lines(events, signal_settings)
    )
    image_by_event = {str(event["eeg_event_id"]): item for item in image_records for event in (item[0],)}
    next_image = 0
    for event in events:
        attachment = _waveform_attachment(event)
        if attachment is None:
            parts.append(_w_paragraph(f"事件{event['event_number']}：未附可验证波形。"))
            continue
        parts.append(_w_paragraph(f"事件{event['event_number']}波形证据", bold=True, keep_next=True))
        record = image_by_event.get(str(event["eeg_event_id"]))
        if record is None:
            parts.append(_w_paragraph("波形附件未嵌入当前 DOCX；请核对受控附件。"))
        else:
            next_image += 1
            _, _, _, width, height = record
            parts.append(
                _w_image_paragraph(
                    relationship_id=f"rId{next_image + 1}",
                    drawing_id=next_image,
                    width_px=width,
                    height_px=height,
                )
            )
        parts.append(
            _w_paragraph(
                _waveform_caption(event, attachment), size=18
            )
        )
    if not events:
        parts.append(_w_paragraph("本记录无事件波形附件。"))
    parts.extend(
        [
            _w_paragraph("波形图是待复核证据附件，不构成独立诊断。", size=18),
            _heading("脑电图印象"),
            _w_paragraph(
                f"一、全记录候选信号所见：{impression['findings']}"
            ),
            _w_paragraph(
                f"二、头皮分布与定位推理：{impression['localization']}"
            ),
            _w_paragraph(
                f"三、SOZ 定位结论：{impression['diagnostic_conclusion']}"
            ),
            _w_paragraph(
                f"四、不确定性与结论边界：{impression['uncertainty']}"
            ),
            _w_paragraph(
                "最终结论须由脑电医师复核原始长程 EEG 后签署。",
                bold=True,
            ),
        ]
    )
    soz_rows = _soz_rows(bundle)
    research_consistency = _research_channel_consistency_text(bundle)
    if soz_rows and research_consistency:
        parts.extend(
            [
                _heading("研究性附录"),
                _w_paragraph(
                    "多候选头皮电极排序一致性（研究性）",
                    bold=True,
                    size=23,
                    keep_next=True,
                ),
                _w_paragraph(_SOZ_DISCLAIMER, bold=True),
                _w_paragraph(research_consistency),
                _w_table(
                    (
                        ("候选", 1600),
                        ("研究性 top-3 头皮电极排序（非临床事实）", 5200),
                        ("用途边界", 2400),
                    ),
                    soz_rows,
                ),
            ]
        )
    parts.extend(
        [
            _w_paragraph("审核状态：AI 草稿", bold=True),
            _w_paragraph("审核医师：________________"),
            _w_paragraph("签署日期：________________"),
        ]
    )
    body = "".join(parts)
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
        'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        f"<w:body>{body}"
        '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="850" w:right="700" w:bottom="850" w:left="700" '
        'w:header="400" w:footer="400" w:gutter="0"/>'
        '</w:sectPr></w:body></w:document>'
    )


def _styles_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/><w:qFormat/>
    <w:rPr><w:rFonts w:ascii="SimSun" w:eastAsia="宋体" w:hAnsi="SimSun"/><w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr>
  </w:style>
</w:styles>"""


def _zip_text(archive: ZipFile, name: str, value: str) -> None:
    info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o600 << 16
    archive.writestr(info, value.encode("utf-8"))


def _zip_bytes(archive: ZipFile, name: str, value: bytes) -> None:
    info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o600 << 16
    archive.writestr(info, value)


def render_long_term_docx(
    path: Path,
    bundle: object,
    context: object | None = None,
    waveform_paths: Mapping[str, str | Path] | None = None,
    language_layer: object | None = None,
) -> None:
    """Render a signal-only deterministic draft; external context is rejected."""

    source, _ = _validated_bundle_with_context(bundle, context)
    events = _events(source)
    event_language = _fact_locked_event_language(source, language_layer)
    path_values, path_scheme = _waveform_key_scheme(events, waveform_paths)
    image_records: list[tuple[Mapping[str, Any], Mapping[str, Any], bytes, int, int]] = []
    if path_values is not None and path_scheme is not None:
        for event in events:
            attachment = _waveform_attachment(event)
            if attachment is None:
                continue
            raw_path = _waveform_value(event, attachment, path_values, path_scheme)
            expected_sha = attachment.get("figure_sha256")
            if not isinstance(expected_sha, str):
                raise ValueError("waveform attachment SHA-256 is missing")
            payload, width, height = _png_payload(raw_path, expected_sha)
            image_records.append((event, attachment, payload, width, height))
    document = _document_xml(source, event_language, image_records)
    output_path = Path(path)
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output_path, "w") as archive:
        _zip_text(
            archive,
            "[Content_Types].xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="png" ContentType="image/png"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>""",
        )
        _zip_text(
            archive,
            "_rels/.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>""",
        )
        _zip_text(archive, "word/document.xml", document)
        _zip_text(archive, "word/styles.xml", _styles_xml())
        relationships = "".join(
            '<Relationship '
            f'Id="rId{index + 1}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
            f'Target="media/eeg_waveform_{index:02d}.png"/>'
            for index, _ in enumerate(image_records, start=1)
        )
        _zip_text(
            archive,
            "word/_rels/document.xml.rels",
            f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
  {relationships}
</Relationships>""",
        )
        for index, (_, _, payload, _, _) in enumerate(image_records, start=1):
            _zip_bytes(archive, f"word/media/eeg_waveform_{index:02d}.png", payload)


__all__ = ["render_long_term_docx", "render_long_term_html"]
