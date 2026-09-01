"""Fail-closed downstream projection for ``event_eeg_findings_v3``.

The frozen v1/v2 report graph has no faithful predicates for several v3
concepts.  In particular, periodicity is not the old combined
``rhythm_or_morphology_observed`` predicate, an explicit event outcome is not
``event_detected``, and event-window occurrence/burden is not record-level
recurrence or burden.  This module therefore creates a separate, typed
sidecar instead of silently squeezing those concepts into the legacy graph.

Every emitted concept claim owns exactly one deterministic Chinese atomic
clause and retains its exact source object, evidence and evaluation-
opportunity identifiers.  Evidence time permissions are copied from the
validated v3/v2 evidence graph.  Only a causally supported ``cerebral_ictal``
competing hypothesis can grant positive onset-support permission, and every
one of its supporting Findings must independently be onset-causal,
past-and-present-only and future-free.

Validation is source-bound: it validates the supplied v3 payload, rebuilds
the complete projection, and requires byte-equivalent structured content.
Changing a clause and merely recomputing the sidecar's self hash therefore
fails.  The module performs no I/O and never reads EDF annotations,
spreadsheets, physician labels, clinical text or private source data.  It is
an engineering serialization boundary, not evidence of clinical validity.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from .event_findings_v3_validation import (
    project_event_eeg_findings_v3_to_v2,
    validate_event_eeg_findings_v3_payload,
)
from .eeg_only_event_outcome_semantics import (
    EEG_ONLY_CANDIDATE_NOT_QUALIFIED_REASON_CODE,
    LEGACY_NO_DEMONSTRABLE_SCALP_ICTAL_CHANGE,
    event_outcome_uses_deprecated_clinical_anchor_wording,
    normalize_eeg_only_event_outcome,
)


EVENT_FINDINGS_V3_DOWNSTREAM_PROJECTION_SCHEMA_VERSION = (
    "clinical_eeg_event_findings_v3_downstream_projection_v1"
)
EVENT_FINDINGS_V3_DOWNSTREAM_PROJECTOR_ID = (
    "event_findings_v3_typed_sidecar_projector_v1"
)
EVENT_FINDINGS_V3_LEGACY_MAPPING_SCHEMA_VERSION = (
    "clinical_eeg_event_findings_v3_legacy_mapping_receipt_v1"
)

_CONCEPTS = {
    "occurrence",
    "burden",
    "variability",
    "rhythmicity",
    "periodicity",
    "acquisition_capability",
    "competing_hypothesis",
    "event_outcome",
}
_OUTCOMES = {
    "qualified_electrographic_seizure",
    "qualified_electrographic_event",
    "candidate_only",
    LEGACY_NO_DEMONSTRABLE_SCALP_ICTAL_CHANGE,
    "obscured_by_artifact",
    "not_possible_to_determine",
}
_EEG_ONLY_FIREWALL: Mapping[str, bool] = {
    "eeg_signal_claims_only": True,
    "edf_annotations_used": False,
    "excel_used": False,
    "doctor_labels_used": False,
    "clinical_text_used": False,
    "patient_metadata_used": False,
    "video_used": False,
    "ecg_emg_eog_used": False,
    "sleep_staging_used": False,
    "provocation_used": False,
}
_RECORD_AGGREGATE_INPUTS = (
    "complete_frozen_detector_event_roster",
    "complete_event_to_mode_roster",
    "deduplication_ownership_receipt",
    "evaluable_record_time_denominator",
)
_PROHIBITED_RECORD_INFERENCES = (
    "record_occurrence",
    "record_burden",
    "cross_event_variability",
    "record_onset_mode_variability",
)
_FORBIDDEN_LEGACY_PREDICATES = (
    "event_detected",
    "rhythm_or_morphology_observed",
    "record_onset_nonlocalizable",
)
_FORBIDDEN_SURFACE_FRAGMENTS = (
    "皮层SOZ",
    "皮层 SOZ",
    "致痫区",
    "致痫灶",
    "癫痫灶",
    "临床诊断",
    "诊断为",
    "正常脑电",
    "排除癫痫",
    "无癫痫",
    "治疗",
    "用药",
    "手术",
    "睡眠",
    "诱发",
    "过度换气",
    "闪光刺激",
    "心电",
    "肌电",
    "眼电",
    "病史",
    "临床表现",
    "行为",
    "医生",
    "标注",
    "Excel",
    "EDF",
)
_LEGACY_REASON_BY_CONCEPT = {
    "occurrence": "event_window_occurrence_is_not_legacy_record_recurrence",
    "burden": "denominator_bound_event_burden_has_no_legacy_predicate",
    "variability": "event_variability_is_not_record_mode_variability",
    "rhythmicity": "rhythmicity_cannot_use_combined_legacy_rhythm_predicate",
    "periodicity": "periodicity_cannot_use_combined_legacy_rhythm_predicate",
    "acquisition_capability": "acquisition_evaluability_has_no_legacy_predicate",
    "competing_hypothesis": "differential_signal_hypothesis_has_no_legacy_predicate",
    "event_outcome": "event_outcome_is_not_event_detected_or_nonlocalizable",
}


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _reject_nonfinite(value: object, context: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{context} must be finite")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_nonfinite(item, f"{context}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nonfinite(item, f"{context}[{index}]")


def _seal(value: dict[str, Any], field: str, domain: str) -> None:
    value[field] = "CONTENT-ADDRESS-PENDING"
    value[field] = _sha256({"binding_domain": domain, "value": value})


def _number(value: object) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("rendered quantity must be finite")
    if abs(number) < 0.5e-9:
        number = 0.0
    return f"{number:.6f}".rstrip("0").rstrip(".")


def _safe_clause(text: str, *, concept: str) -> str:
    if not isinstance(text, str) or not text.strip() or text != text.strip():
        raise ValueError("atomic clause must be non-empty and trimmed")
    if not text.endswith("。"):
        raise ValueError("atomic Chinese fallback clause must end with a full stop")
    if concept == "periodicity" and "节律" in text:
        raise ValueError("periodicity fallback must not use rhythmicity wording")
    forbidden = [item for item in _FORBIDDEN_SURFACE_FRAGMENTS if item in text]
    if forbidden:
        raise ValueError(
            f"atomic clause contains unsupported clinical content: {forbidden}"
        )
    return text


def _source_ref(
    *, object_kind: str, object_id: str, json_pointer: str
) -> dict[str, str]:
    return {
        "object_kind": str(object_kind),
        "object_id": str(object_id),
        "json_pointer": str(json_pointer),
    }


def _raw_dependencies(
    payload: Mapping[str, Any], finding: Mapping[str, Any]
) -> list[dict[str, Any]]:
    waveform_map = {
        str(row["waveform_evidence_id"]): row for row in payload["waveform_evidence"]
    }
    candidates: list[Mapping[str, Any]] = []
    for measurement in finding["measurements"]:
        dependency = measurement["source_binding"]["raw_sample_dependency"]
        if dependency is not None:
            candidates.append(dependency)
    for waveform_id in finding["waveform_evidence_ids"]:
        dependency = waveform_map[str(waveform_id)]["raw_sample_dependency"]
        if dependency is not None:
            candidates.append(dependency)

    by_id: dict[str, dict[str, Any]] = {}
    for dependency in candidates:
        dependency_id = str(dependency["dependency_id"])
        row = {
            "dependency_id": dependency_id,
            "view_role": str(dependency["view_role"]),
            "dependency_policy": str(dependency["dependency_policy"]),
            "future_sample_access": bool(dependency["future_sample_access"]),
            "onset_evidence_authorized": bool(dependency["onset_evidence_authorized"]),
            "onset_support_eligible": bool(dependency["onset_support_eligible"]),
            "evidence_recording_interval": deepcopy(
                dependency["evidence_recording_interval"]
            ),
        }
        previous = by_id.get(dependency_id)
        if previous is not None and previous != row:
            raise ValueError(
                f"raw dependency {dependency_id!r} has inconsistent permissions"
            )
        by_id[dependency_id] = row
    return [by_id[item] for item in sorted(by_id)]


def _evidence_time_permissions(
    payload: Mapping[str, Any], evidence_ids: Sequence[str]
) -> list[dict[str, Any]]:
    finding_map = {str(row["evidence_id"]): row for row in payload["findings"]}
    result: list[dict[str, Any]] = []
    for evidence_id in sorted(set(str(item) for item in evidence_ids)):
        finding = finding_map[evidence_id]
        dependencies = _raw_dependencies(payload, finding)
        causal = bool(dependencies) and all(
            row["view_role"] == "onset_causal"
            and row["dependency_policy"] == "past_and_present_only"
            and not row["future_sample_access"]
            and row["onset_evidence_authorized"]
            and row["onset_support_eligible"]
            for row in dependencies
        )
        result.append(
            {
                "evidence_id": evidence_id,
                "finding_status": str(finding["status"]),
                "assertion_level": str(finding["assertion_level"]),
                "intrinsic_evidence_role": str(finding["intrinsic_evidence_role"]),
                "raw_dependencies": dependencies,
                "onset_time_permission": causal,
                "positive_onset_support_permitted": bool(
                    finding["status"] == "present"
                    and finding["intrinsic_evidence_role"] == "onset_eligible"
                    and causal
                ),
            }
        )
    return result


def _occurrence_clause(row: Mapping[str, Any]) -> str:
    status = str(row["status"])
    if status == "measured":
        count = int(row["count"])
        rate = _number(row["rate_per_minute"])
        if count == 0:
            return f"在该事件的可评价时段内未记录到去重模式候选（{rate} 次/分钟）。"
        return f"在该事件的可评价时段内记录到 {count} 个去重模式候选" f"（{rate} 次/分钟）。"
    if status == "limited":
        return "该事件窗内模式候选的出现次数评价受限。"
    return "该事件窗内模式候选的出现次数无法评价。"


def _burden_clause(row: Mapping[str, Any]) -> str:
    status = str(row["status"])
    if status == "measured":
        observed = _number(row["observed_seconds"])
        evaluable = _number(row["evaluable_seconds"])
        percentage = _number(float(row["proportion"]) * 100.0)
        return f"该模式候选在 {evaluable} 秒可评价时段内累计占用 " f"{observed} 秒（{percentage}%）。"
    if status == "limited":
        return "该事件窗内模式候选的时长占比评价受限。"
    return "该事件窗内模式候选的时长占比无法评价。"


_VARIABILITY_LABELS = {
    "frequency": "频率",
    "amplitude": "波幅",
    "morphology": "形态",
    "spatial_distribution": "空间分布",
    "inter_occurrence_interval": "出现间隔",
    "duration": "持续时间",
}


def _variability_clause(row: Mapping[str, Any] | None, status: str) -> str:
    if row is None:
        if status == "limited":
            return "该事件窗内模式候选的变异性评价受限。"
        return "该事件窗内模式候选的变异性无法评价。"
    label = _VARIABILITY_LABELS[str(row["dimension"])]
    dimension_status = str(row["status"])
    if dimension_status == "stable":
        return f"该事件窗内模式候选的{label}表现稳定。"
    if dimension_status == "variable":
        return f"该事件窗内模式候选的{label}存在变异。"
    if dimension_status == "indeterminate":
        return f"该事件窗内模式候选的{label}变异性不能确定。"
    return f"该事件窗内模式候选的{label}变异性无法评价。"


def _rhythm_clause(kind: str, status: str) -> str:
    if kind == "rhythmicity":
        noun = "节律性活动"
    else:
        noun = "周期性重复活动"
    if status == "qualified_present":
        return f"记录到通过自动资格门的{noun}候选。"
    if status == "qualified_absent_with_opportunity":
        return f"在充分评价机会下未记录到{noun}候选。"
    if status == "candidate_only":
        return f"记录到{noun}候选，尚未达到自动报告资格门。"
    return f"该段的{noun}无法评价。"


_CAPABILITY_LABELS = {
    "dc_shift": "直流漂移类特征",
    "high_frequency_oscillation": "高频振荡类特征",
    "low_voltage_fast_activity": "低电压快活动类特征",
    "other_acquisition_sensitive": "采集敏感特征",
}


def _capability_clause(row: Mapping[str, Any] | None, status: str) -> str:
    if row is None:
        return "该事件的采集敏感特征评价条件无法确定。"
    label = _CAPABILITY_LABELS[str(row["feature_class"])]
    if status == "evaluable":
        return f"该段对{label}具备已记录的可评价采集条件。"
    if status == "limited":
        return f"该段对{label}的采集评价条件受限。"
    return f"该段对{label}的采集评价条件无法确定。"


_HYPOTHESIS_LABELS = {
    "cerebral_ictal": "脑源性发作期模式",
    "cerebral_nonictal": "脑源性非发作期模式",
    "physiologic": "生理性模式",
    "benign_variant": "良性变异模式",
    "artifact": "伪迹",
    "technical": "技术性模式",
    "uncertain_pattern": "未定模式",
}
_DISPOSITION_LABELS = {
    "supported": "得到支持",
    "possible": "仍有可能",
    "disfavored": "较不支持",
    "not_evaluable": "无法评价",
}


def _hypothesis_clause(row: Mapping[str, Any] | None, *, selected: bool) -> str:
    if row is None:
        return "该事件的竞争性信号假设无法评价。"
    label = _HYPOTHESIS_LABELS[str(row["category"])]
    disposition = _DISPOSITION_LABELS[str(row["disposition"])]
    rank = row["rank"]
    rank_text = f"第 {int(rank)} 位" if rank is not None else "未排序"
    if selected:
        rank_text += "，已选"
    support = len(row["supporting_evidence_ids"])
    contradiction = len(row["contradictory_evidence_ids"])
    return (
        f"竞争性信号假设“{label}”（{rank_text}）{disposition}；"
        f"支持证据 {support} 项，相反证据 {contradiction} 项。"
    )


_OUTCOME_CLAUSES = {
    "qualified_electrographic_seizure": ("该段达到自动限定的电图发作资格门。"),
    "qualified_electrographic_event": ("该段达到自动限定的电图事件资格门。"),
    "candidate_only": ("该段保留为电图事件候选，尚未达到自动报告资格门。"),
    LEGACY_NO_DEMONSTRABLE_SCALP_ICTAL_CHANGE: (
        "在已查询的头皮信号支持内，该候选未达到预注册的电图发作样模式资格；" "该表述不预设或判断一个独立临床事件是否存在。"
    ),
    "obscured_by_artifact": ("该段关键时段受伪迹遮蔽，无法判断头皮发作期电图改变。"),
    "not_possible_to_determine": ("该段因可评价性不足，无法判断是否存在头皮发作期电图改变。"),
}


def _legacy_mapping(concept: str) -> dict[str, Any]:
    return {
        "status": "unsupported",
        "target_schema_version": "clinical_eeg_multievent_soz_report_v1",
        "target_predicate": None,
        "reason_code": _LEGACY_REASON_BY_CONCEPT[concept],
    }


def _build_projection(source: Mapping[str, Any]) -> dict[str, Any]:
    source_v2 = project_event_eeg_findings_v3_to_v2(source)
    event_id = str(source["event_id"])
    claims: list[dict[str, Any]] = []
    clauses: list[dict[str, str]] = []
    used_claim_ids: set[str] = set()
    used_owner_ids: set[str] = set()
    used_clause_ids: set[str] = set()

    def emit(
        *,
        concept: str,
        discriminator: str,
        source_object_refs: list[dict[str, str]],
        source_evidence_ids: Sequence[str],
        source_opportunity_ids: Sequence[str],
        assertion_status: str,
        epistemic_status: str,
        value: object,
        clause_text: str,
        positive_onset_support_permitted: bool = False,
    ) -> None:
        if concept not in _CONCEPTS:
            raise ValueError(f"unsupported v3 downstream concept: {concept}")
        safe_text = _safe_clause(clause_text, concept=concept)
        evidence_ids = sorted(set(str(item) for item in source_evidence_ids))
        opportunity_ids = sorted(set(str(item) for item in source_opportunity_ids))
        permissions = _evidence_time_permissions(source, evidence_ids)
        seed = {
            "event_id": event_id,
            "concept": concept,
            "discriminator": discriminator,
            "source_object_refs": source_object_refs,
        }
        seed_sha = _sha256(seed)
        claim_id = f"V3C-{seed_sha[:24]}"
        owner_id = f"V3OWNER-{_sha256({'claim_id': claim_id})[:24]}"
        clause_id = f"V3CLAUSE-{_sha256({'claim_id': claim_id})[:24]}"
        if claim_id in used_claim_ids:
            raise ValueError(f"duplicate downstream claim identity: {claim_id}")
        if owner_id in used_owner_ids or clause_id in used_clause_ids:
            raise ValueError("atomic clause ownership identity collision")
        used_claim_ids.add(claim_id)
        used_owner_ids.add(owner_id)
        used_clause_ids.add(clause_id)
        claims.append(
            {
                "claim_id": claim_id,
                "claim_owner_id": owner_id,
                "concept": concept,
                "event_id": event_id,
                "assertion_status": str(assertion_status),
                "epistemic_status": str(epistemic_status),
                "source_object_refs": deepcopy(source_object_refs),
                "source_evidence_ids": evidence_ids,
                "source_opportunity_ids": opportunity_ids,
                "evidence_time_permissions": permissions,
                "positive_onset_support_permitted": bool(
                    positive_onset_support_permitted
                ),
                "value": deepcopy(value),
                "atomic_clause_id": clause_id,
                "legacy_mapping": _legacy_mapping(concept),
            }
        )
        clauses.append(
            {
                "clause_id": clause_id,
                "claim_id": claim_id,
                "claim_owner_id": owner_id,
                "text_zh": safe_text,
            }
        )

    quantity = source["occurrence_burden_variability"]
    summaries = quantity["summaries"]
    if summaries:
        for summary_index, summary in enumerate(summaries):
            summary_id = str(summary["summary_id"])
            base_pointer = f"/occurrence_burden_variability/summaries/{summary_index}"
            occurrence = summary["occurrence"]
            occurrence_status = str(occurrence["status"])
            emit(
                concept="occurrence",
                discriminator=f"{summary_id}:occurrence",
                source_object_refs=[
                    _source_ref(
                        object_kind="pattern_quantity_summary",
                        object_id=summary_id,
                        json_pointer=f"{base_pointer}/occurrence",
                    )
                ],
                source_evidence_ids=occurrence["supporting_evidence_ids"],
                source_opportunity_ids=occurrence["evaluation_opportunity_ids"],
                assertion_status=(
                    "present"
                    if occurrence_status == "measured" and int(occurrence["count"]) > 0
                    else "absent_with_opportunity"
                    if occurrence_status == "measured"
                    else "uncertain"
                    if occurrence_status == "limited"
                    else "not_evaluable"
                ),
                epistemic_status=(
                    "measured"
                    if occurrence_status == "measured"
                    else "limited"
                    if occurrence_status == "limited"
                    else "not_evaluable"
                ),
                value={
                    "scope": "signal_only_event_window",
                    "term_id": summary["term_id"],
                    "scope_interval": deepcopy(summary["scope_interval"]),
                    "occurrence": deepcopy(occurrence),
                },
                clause_text=_occurrence_clause(occurrence),
            )
            burden = summary["burden"]
            burden_status = str(burden["status"])
            emit(
                concept="burden",
                discriminator=f"{summary_id}:burden",
                source_object_refs=[
                    _source_ref(
                        object_kind="pattern_quantity_summary",
                        object_id=summary_id,
                        json_pointer=f"{base_pointer}/burden",
                    )
                ],
                source_evidence_ids=burden["supporting_evidence_ids"],
                source_opportunity_ids=burden["evaluation_opportunity_ids"],
                assertion_status=(
                    "present"
                    if burden_status == "measured"
                    and float(burden["observed_seconds"]) > 0
                    else "absent_with_opportunity"
                    if burden_status == "measured"
                    else "uncertain"
                    if burden_status == "limited"
                    else "not_evaluable"
                ),
                epistemic_status=(
                    "measured"
                    if burden_status == "measured"
                    else "limited"
                    if burden_status == "limited"
                    else "not_evaluable"
                ),
                value={
                    "scope": "signal_only_event_window",
                    "term_id": summary["term_id"],
                    "scope_interval": deepcopy(summary["scope_interval"]),
                    "burden": deepcopy(burden),
                },
                clause_text=_burden_clause(burden),
            )
            variability = summary["variability"]
            dimensions = variability["dimensions"]
            if dimensions:
                for dimension_index, dimension in enumerate(dimensions):
                    dimension_status = str(dimension["status"])
                    emit(
                        concept="variability",
                        discriminator=(
                            f"{summary_id}:variability:{dimension['dimension']}"
                        ),
                        source_object_refs=[
                            _source_ref(
                                object_kind="variability_dimension",
                                object_id=(f"{summary_id}:{dimension['dimension']}"),
                                json_pointer=(
                                    f"{base_pointer}/variability/dimensions/"
                                    f"{dimension_index}"
                                ),
                            )
                        ],
                        source_evidence_ids=dimension["supporting_evidence_ids"],
                        source_opportunity_ids=[],
                        assertion_status=(
                            "present"
                            if dimension_status in {"stable", "variable"}
                            else "uncertain"
                            if dimension_status == "indeterminate"
                            else "not_evaluable"
                        ),
                        epistemic_status=(
                            "measured"
                            if dimension_status in {"stable", "variable"}
                            else "limited"
                            if dimension_status == "indeterminate"
                            else "not_evaluable"
                        ),
                        value={
                            "scope": "signal_only_event_window",
                            "term_id": summary["term_id"],
                            "scope_interval": deepcopy(summary["scope_interval"]),
                            "variability": deepcopy(dimension),
                        },
                        clause_text=_variability_clause(
                            dimension, str(variability["status"])
                        ),
                    )
            else:
                variability_status = str(variability["status"])
                emit(
                    concept="variability",
                    discriminator=f"{summary_id}:variability",
                    source_object_refs=[
                        _source_ref(
                            object_kind="variability_summary",
                            object_id=f"{summary_id}:variability",
                            json_pointer=f"{base_pointer}/variability",
                        )
                    ],
                    source_evidence_ids=variability["supporting_evidence_ids"],
                    source_opportunity_ids=[],
                    assertion_status=(
                        "uncertain"
                        if variability_status == "limited"
                        else "not_evaluable"
                    ),
                    epistemic_status=(
                        "limited"
                        if variability_status == "limited"
                        else "not_evaluable"
                    ),
                    value={
                        "scope": "signal_only_event_window",
                        "term_id": summary["term_id"],
                        "scope_interval": deepcopy(summary["scope_interval"]),
                        "variability": deepcopy(variability),
                    },
                    clause_text=_variability_clause(None, variability_status),
                )
    else:
        block_status = str(quantity["status"])
        for concept in ("occurrence", "burden", "variability"):
            clause = {
                "occurrence": "该事件窗内模式候选的出现次数无法评价。",
                "burden": "该事件窗内模式候选的时长占比无法评价。",
                "variability": "该事件窗内模式候选的变异性无法评价。",
            }[concept]
            emit(
                concept=concept,
                discriminator=f"quantity-block:{concept}",
                source_object_refs=[
                    _source_ref(
                        object_kind="occurrence_burden_variability",
                        object_id="occurrence_burden_variability",
                        json_pointer="/occurrence_burden_variability",
                    )
                ],
                source_evidence_ids=[],
                source_opportunity_ids=[],
                assertion_status=(
                    "uncertain" if block_status == "limited" else "not_evaluable"
                ),
                epistemic_status=(
                    "limited" if block_status == "limited" else "not_evaluable"
                ),
                value={
                    "scope": "signal_only_event_window",
                    "block_status": block_status,
                    "reason_codes": deepcopy(quantity["reason_codes"]),
                },
                clause_text=clause,
            )

    rhythm_block = source["rhythm_periodicity_qualification"]
    for kind in ("rhythmicity", "periodicity"):
        gate = rhythm_block[kind]
        gate_status = str(gate["qualification_status"])
        emit(
            concept=kind,
            discriminator=kind,
            source_object_refs=[
                _source_ref(
                    object_kind="rhythm_qualification_gate",
                    object_id=kind,
                    json_pointer=f"/rhythm_periodicity_qualification/{kind}",
                )
            ],
            source_evidence_ids=gate["finding_ids"],
            source_opportunity_ids=gate["evaluation_opportunity_ids"],
            assertion_status=(
                "present"
                if gate_status == "qualified_present"
                else "absent_with_opportunity"
                if gate_status == "qualified_absent_with_opportunity"
                else "candidate_only"
                if gate_status == "candidate_only"
                else "not_evaluable"
            ),
            epistemic_status=(
                "automated_qualified"
                if gate_status
                in {"qualified_present", "qualified_absent_with_opportunity"}
                else "model_candidate"
                if gate_status == "candidate_only"
                else "not_evaluable"
            ),
            value=deepcopy(gate),
            clause_text=_rhythm_clause(kind, gate_status),
        )

    acquisition = source["acquisition_capabilities"]
    if acquisition["capabilities"]:
        for capability_index, capability in enumerate(acquisition["capabilities"]):
            capability_status = str(capability["status"])
            capability_id = str(capability["capability_id"])
            emit(
                concept="acquisition_capability",
                discriminator=capability_id,
                source_object_refs=[
                    _source_ref(
                        object_kind="acquisition_capability",
                        object_id=capability_id,
                        json_pointer=(
                            f"/acquisition_capabilities/capabilities/"
                            f"{capability_index}"
                        ),
                    )
                ],
                source_evidence_ids=[],
                source_opportunity_ids=capability["evaluation_opportunity_ids"],
                assertion_status=(
                    "present"
                    if capability_status == "evaluable"
                    else "uncertain"
                    if capability_status == "limited"
                    else "not_evaluable"
                ),
                epistemic_status=(
                    "acquisition_metadata"
                    if capability_status == "evaluable"
                    else "limited"
                    if capability_status == "limited"
                    else "not_evaluable"
                ),
                value=deepcopy(capability),
                clause_text=_capability_clause(capability, capability_status),
            )
    else:
        acquisition_status = str(acquisition["status"])
        emit(
            concept="acquisition_capability",
            discriminator="acquisition-capabilities-block",
            source_object_refs=[
                _source_ref(
                    object_kind="acquisition_capabilities",
                    object_id="acquisition_capabilities",
                    json_pointer="/acquisition_capabilities",
                )
            ],
            source_evidence_ids=[],
            source_opportunity_ids=[],
            assertion_status=(
                "uncertain" if acquisition_status == "limited" else "not_evaluable"
            ),
            epistemic_status=(
                "limited" if acquisition_status == "limited" else "not_evaluable"
            ),
            value=deepcopy(acquisition),
            clause_text=_capability_clause(None, acquisition_status),
        )

    competing = source["competing_hypotheses"]
    selected_hypothesis_id = competing["selected_hypothesis_id"]
    if competing["hypotheses"]:
        for hypothesis_index, hypothesis in enumerate(competing["hypotheses"]):
            hypothesis_id = str(hypothesis["hypothesis_id"])
            support_ids = [str(item) for item in hypothesis["supporting_evidence_ids"]]
            contradiction_ids = [
                str(item) for item in hypothesis["contradictory_evidence_ids"]
            ]
            permissions = _evidence_time_permissions(source, support_ids)
            every_support_is_causal = bool(permissions) and all(
                row["positive_onset_support_permitted"] for row in permissions
            )
            contradiction_permissions = _evidence_time_permissions(
                source, contradiction_ids
            )
            every_contradiction_is_time_closed = all(
                row["onset_time_permission"] for row in contradiction_permissions
            )
            positive_permission = bool(
                hypothesis["category"] == "cerebral_ictal"
                and hypothesis["disposition"] == "supported"
                and hypothesis["onset_claim_eligible"]
                and every_support_is_causal
                and every_contradiction_is_time_closed
            )
            disposition = str(hypothesis["disposition"])
            selected = hypothesis_id == selected_hypothesis_id
            emit(
                concept="competing_hypothesis",
                discriminator=hypothesis_id,
                source_object_refs=[
                    _source_ref(
                        object_kind="competing_hypothesis",
                        object_id=hypothesis_id,
                        json_pointer=(
                            f"/competing_hypotheses/hypotheses/" f"{hypothesis_index}"
                        ),
                    )
                ],
                source_evidence_ids=support_ids + contradiction_ids,
                source_opportunity_ids=[],
                assertion_status=(
                    "present"
                    if disposition == "supported"
                    else "uncertain"
                    if disposition == "possible"
                    else "disfavored"
                    if disposition == "disfavored"
                    else "not_evaluable"
                ),
                epistemic_status=(
                    "source_bounded_differential"
                    if disposition != "not_evaluable"
                    else "not_evaluable"
                ),
                value={
                    **deepcopy(hypothesis),
                    "selected": selected,
                },
                clause_text=_hypothesis_clause(hypothesis, selected=selected),
                positive_onset_support_permitted=positive_permission,
            )
    else:
        competing_status = str(competing["status"])
        emit(
            concept="competing_hypothesis",
            discriminator="competing-hypotheses-block",
            source_object_refs=[
                _source_ref(
                    object_kind="competing_hypotheses",
                    object_id="competing_hypotheses",
                    json_pointer="/competing_hypotheses",
                )
            ],
            source_evidence_ids=[],
            source_opportunity_ids=[],
            assertion_status=(
                "uncertain" if competing_status == "limited" else "not_evaluable"
            ),
            epistemic_status=(
                "source_bounded_differential"
                if competing_status == "limited"
                else "not_evaluable"
            ),
            value=deepcopy(competing),
            clause_text=_hypothesis_clause(None, selected=False),
        )

    event_outcome = source["event_outcome"]
    source_outcome = str(event_outcome["outcome"])
    if source_outcome not in _OUTCOMES:
        raise ValueError(f"unknown v3 event outcome: {source_outcome}")
    legacy_eeg_only_downgrade = event_outcome_uses_deprecated_clinical_anchor_wording(
        source_outcome
    )
    outcome = normalize_eeg_only_event_outcome(source_outcome)
    projected_event_outcome = deepcopy(event_outcome)
    projected_event_outcome["outcome"] = outcome
    if legacy_eeg_only_downgrade:
        projected_event_outcome["reason_codes"] = sorted(
            {
                *(str(item) for item in projected_event_outcome["reason_codes"]),
                EEG_ONLY_CANDIDATE_NOT_QUALIFIED_REASON_CODE,
            }
        )
    emit(
        concept="event_outcome",
        discriminator=outcome,
        source_object_refs=[
            _source_ref(
                object_kind="event_outcome",
                object_id=source_outcome,
                json_pointer="/event_outcome",
            )
        ],
        source_evidence_ids=event_outcome["evidence_ids"],
        source_opportunity_ids=[],
        assertion_status=(
            "not_evaluable"
            if outcome in {"obscured_by_artifact", "not_possible_to_determine"}
            else "absent_with_opportunity"
            if legacy_eeg_only_downgrade
            else "candidate_only"
            if outcome == "candidate_only"
            else "present"
        ),
        epistemic_status=(
            "automated_qualified"
            if outcome
            in {
                "qualified_electrographic_seizure",
                "qualified_electrographic_event",
            }
            else "source_bounded_qualification_failure"
            if legacy_eeg_only_downgrade
            else "model_candidate"
            if outcome == "candidate_only"
            else "not_evaluable"
        ),
        value=projected_event_outcome,
        clause_text=_OUTCOME_CLAUSES[source_outcome],
    )

    # The only concept allowed to grant positive onset support is one
    # independently closed competing hypothesis.  Everything else remains
    # false even when it describes a qualified event or a positive Finding.
    if any(
        claim["positive_onset_support_permitted"]
        and claim["concept"] != "competing_hypothesis"
        for claim in claims
    ):
        raise ValueError("non-hypothesis v3 concept granted onset permission")

    legacy_receipt: dict[str, Any] = {
        "schema_version": EVENT_FINDINGS_V3_LEGACY_MAPPING_SCHEMA_VERSION,
        "status": "unsupported",
        "target_schema_version": "clinical_eeg_multievent_soz_report_v1",
        "unsupported_claim_ids": [row["claim_id"] for row in claims],
        "forbidden_legacy_predicates": list(_FORBIDDEN_LEGACY_PREDICATES),
        "reason_codes": [
            "v3_typed_semantics_not_representable_by_frozen_legacy_contract",
            "use_separate_v3_downstream_sidecar",
        ],
        "receipt_sha256": "",
    }
    _seal(
        legacy_receipt,
        "receipt_sha256",
        "clinical-eeg-event-findings-v3-legacy-mapping-receipt-v1",
    )

    migration = source["v3_migration"]
    result: dict[str, Any] = {
        "schema_version": (EVENT_FINDINGS_V3_DOWNSTREAM_PROJECTION_SCHEMA_VERSION),
        "projector_id": EVENT_FINDINGS_V3_DOWNSTREAM_PROJECTOR_ID,
        "event_id": event_id,
        "source_binding": {
            "source_schema_version": "event_eeg_findings_v3",
            "source_event_findings_v3_sha256": _sha256(source),
            "frozen_v2_projection_sha256": _sha256(source_v2),
            "source_semantics": (
                "native_v3" if migration is None else "lossy_v2_migration"
            ),
            "migration_loss_codes": (
                [] if migration is None else deepcopy(migration["loss_codes"])
            ),
        },
        "eeg_only_firewall": dict(_EEG_ONLY_FIREWALL),
        "scope_receipt": {
            "claim_scope": "single_signal_only_event_window",
            "event_window_quantities_are_record_aggregates": False,
            "event_outcomes_are_legacy_event_detected": False,
            "periodicity_is_legacy_rhythm_or_morphology": False,
        },
        "concept_claims": claims,
        "atomic_clauses": clauses,
        "atomic_clause_ownership": {
            "policy_id": "one_claim_one_atomic_clause_owner_v1",
            "claim_count": len(claims),
            "clause_count": len(clauses),
            "all_claims_owned_exactly_once": True,
            "all_clause_owners_unique": True,
        },
        "record_aggregate_requirements": {
            "status": "not_projected_from_single_event",
            "required_input_ids": list(_RECORD_AGGREGATE_INPUTS),
            "prohibited_single_event_inferences": list(_PROHIBITED_RECORD_INFERENCES),
            "reason_codes": ["complete_record_denominators_and_rosters_required"],
        },
        "legacy_mapping_receipt": legacy_receipt,
        "projection_sha256": "",
    }
    _seal(
        result,
        "projection_sha256",
        "clinical-eeg-event-findings-v3-downstream-projection-v1",
    )
    return result


def project_event_eeg_findings_v3_downstream(
    payload: object,
    *,
    trusted_producer_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_calibration_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_capability_qualification_receipts: (
        Mapping[str, Mapping[str, object]] | None
    ) = None,
    trusted_sensitivity_receipts: (Mapping[str, Mapping[str, object]] | None) = None,
    trusted_term_decision_receipts: (Mapping[str, Mapping[str, object]] | None) = None,
    trusted_registry_bindings: (Mapping[str, Mapping[str, object]] | None) = None,
) -> dict[str, Any]:
    """Validate v3 and build its typed, source-bound downstream sidecar."""

    source = validate_event_eeg_findings_v3_payload(
        payload,
        trusted_producer_receipts=trusted_producer_receipts,
        trusted_calibration_receipts=trusted_calibration_receipts,
        trusted_capability_qualification_receipts=(
            trusted_capability_qualification_receipts
        ),
        trusted_sensitivity_receipts=trusted_sensitivity_receipts,
        trusted_term_decision_receipts=trusted_term_decision_receipts,
        trusted_registry_bindings=trusted_registry_bindings,
    )
    return _build_projection(source)


def validate_event_eeg_findings_v3_downstream_projection(
    value: object,
    *,
    source_event_findings_v3: object,
    trusted_producer_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_calibration_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_capability_qualification_receipts: (
        Mapping[str, Mapping[str, object]] | None
    ) = None,
    trusted_sensitivity_receipts: (Mapping[str, Mapping[str, object]] | None) = None,
    trusted_term_decision_receipts: (Mapping[str, Mapping[str, object]] | None) = None,
    trusted_registry_bindings: (Mapping[str, Mapping[str, object]] | None) = None,
) -> dict[str, Any]:
    """Replay the projection from its frozen v3 source and fail on drift."""

    if type(value) is not dict:
        raise TypeError("v3 downstream projection must be an object")
    candidate: dict[str, Any] = deepcopy(value)
    _reject_nonfinite(candidate)
    expected = project_event_eeg_findings_v3_downstream(
        source_event_findings_v3,
        trusted_producer_receipts=trusted_producer_receipts,
        trusted_calibration_receipts=trusted_calibration_receipts,
        trusted_capability_qualification_receipts=(
            trusted_capability_qualification_receipts
        ),
        trusted_sensitivity_receipts=trusted_sensitivity_receipts,
        trusted_term_decision_receipts=trusted_term_decision_receipts,
        trusted_registry_bindings=trusted_registry_bindings,
    )
    if candidate != expected:
        raise ValueError(
            "v3 downstream projection does not replay exactly from its source"
        )
    return deepcopy(expected)


__all__ = [
    "EVENT_FINDINGS_V3_DOWNSTREAM_PROJECTION_SCHEMA_VERSION",
    "EVENT_FINDINGS_V3_DOWNSTREAM_PROJECTOR_ID",
    "EVENT_FINDINGS_V3_LEGACY_MAPPING_SCHEMA_VERSION",
    "project_event_eeg_findings_v3_downstream",
    "validate_event_eeg_findings_v3_downstream_projection",
]
