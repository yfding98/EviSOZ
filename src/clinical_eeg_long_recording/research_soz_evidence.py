"""Uncalibrated descriptive strength for research scalp-ranking artifacts.

This module deliberately separates *whether a Top-k ranking is emitted* from
*how consistently the per-event rankings support its leading electrode*.
Every validated research prediction retains its Top-k hypotheses.  The three
levels below are descriptive routing labels only; they are neither clinical
confidence nor calibrated probabilities.

No raw EEG, EDF annotation, spreadsheet, doctor label, or narrative text is
accepted by this API.  Its sole input is a validated artifact produced by
``research_soz_prediction``.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Any, Mapping

from .research_soz_prediction import (
    validate_research_soz_prediction_artifact,
)


RESEARCH_SOZ_EVIDENCE_SCHEMA_VERSION = (
    "clinical_eeg_research_soz_descriptive_strength_v1_1"
)
RESEARCH_SOZ_EVIDENCE_POLICY_ID = (
    "single_mode_repeated_event_support_descriptive_v1"
)

STABLE_LEADING_CANDIDATE = "stable_leading_candidate_descriptive"
LIMITED_CROSS_EVENT_CONSISTENCY = "limited_cross_event_consistency"
MULTIMODAL_OR_WEAK_RANKED_HYPOTHESES = (
    "multimodal_or_weak_ranked_hypotheses"
)
DESCRIPTIVE_EVIDENCE_LEVELS: tuple[str, ...] = (
    STABLE_LEADING_CANDIDATE,
    LIMITED_CROSS_EVENT_CONSISTENCY,
    MULTIMODAL_OR_WEAK_RANKED_HYPOTHESES,
)

# Engineering cut points for readable descriptive routing.  They are not
# tuned on the private cohort and must not be interpreted as clinical or
# probabilistic thresholds.  A future patient-disjoint source-development
# calibration receipt can replace this provisional policy without changing
# the underlying Top-k prediction artifact.
MIN_STABLE_EVENT_COUNT = 3
MIN_STABLE_TOP1_SUPPORT_RATE = 2.0 / 3.0
MIN_STABLE_TOP3_SUPPORT_RATE = 0.8

_SHA256_HEX = frozenset("0123456789abcdef")

_LEVEL_ZH = {
    STABLE_LEADING_CANDIDATE: "首位候选跨事件相对稳定",
    LIMITED_CROSS_EVENT_CONSISTENCY: "候选存在但跨事件一致性有限",
    MULTIMODAL_OR_WEAK_RANKED_HYPOTHESES: "多模式或弱证据排序假设",
}


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _content_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256 hex digest")
    return value


def _finite_rate(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{context} must be a finite rate from zero to one")
    return result


def _render_deterministic_conclusion(binding: Mapping[str, Any]) -> str:
    electrodes = binding["ranked_electrodes"]
    joined = "、".join(electrodes)
    top1_percent = f'{float(binding["top1_support_rate"]) * 100.0:.1f}%'
    top3_percent = f'{float(binding["top3_support_rate"]) * 100.0:.1f}%'
    level_zh = _LEVEL_ZH[str(binding["evidence_level"])]
    return (
        f'基于 {int(binding["input_event_count"])} 个 EEG 事件的 C18 排序，'
        f'头皮 EEG 起始候选通道 Top-{int(binding["top_k"])} 依次为：{joined}。'
        f'首位通道 {electrodes[0]} 的跨事件 Top-1 支持率为 {top1_percent}，'
        f'Top-3 支持率为 {top3_percent}；完整链接聚类得到 '
        f'{int(binding["mode_cluster_count"])} 个模式簇。证据层描述为“{level_zh}”。'
        "该结论仅为未校准的研究性头皮通道排序，不是临床诊断或治疗靶点。"
    )


def classify_research_soz_descriptive_strength(
    prediction: Mapping[str, Any],
    *,
    recording_id: str | None = None,
) -> dict[str, Any]:
    """Classify cross-event consistency without suppressing the Top-k output.

    ``stable`` requires at least three events, a single complete-link event
    mode, and repeated support for the leading electrode.  A multi-mode
    result, or a single event for which cross-event stability is unknowable,
    is routed to the weak/multimodal level.  Other single-mode predictions are
    marked limited.  These provisional engineering rules are intentionally
    transparent and remain uncalibrated until a patient-disjoint source-dev
    receipt is attached in a future version.
    """

    validated = validate_research_soz_prediction_artifact(prediction)
    event_count = int(validated["input_event_count"])
    diagnostics = validated["aggregate_diagnostics"]
    consistency = validated["cross_event_consistency"]
    top1_support = float(diagnostics["top1_support_rate"])
    top3_support = float(diagnostics["top3_support_rate"])
    mode_count = int(consistency["mode_cluster_count"])
    multimodal = bool(consistency["multimodal"])

    stable_checks = {
        "minimum_event_count_met": event_count >= MIN_STABLE_EVENT_COUNT,
        "single_mode_met": mode_count == 1 and not multimodal,
        "top1_repeated_support_met": (
            top1_support >= MIN_STABLE_TOP1_SUPPORT_RATE
        ),
        "top3_repeated_support_met": (
            top3_support >= MIN_STABLE_TOP3_SUPPORT_RATE
        ),
    }
    if all(stable_checks.values()):
        level = STABLE_LEADING_CANDIDATE
    elif event_count < 2 or multimodal or mode_count > 1:
        level = MULTIMODAL_OR_WEAK_RANKED_HYPOTHESES
    else:
        level = LIMITED_CROSS_EVENT_CONSISTENCY

    reasons: list[str] = []
    if event_count < 2:
        reasons.append("cross_event_consistency_not_estimable_from_single_event")
    if multimodal or mode_count > 1:
        reasons.append("multiple_complete_link_event_modes_detected")
    if event_count < MIN_STABLE_EVENT_COUNT:
        reasons.append("fewer_than_three_ranked_events")
    if top1_support < MIN_STABLE_TOP1_SUPPORT_RATE:
        reasons.append("leading_electrode_support_below_descriptive_stable_cutpoint")
    if top3_support < MIN_STABLE_TOP3_SUPPORT_RATE:
        reasons.append("leading_electrode_top3_support_below_descriptive_stable_cutpoint")
    if not reasons:
        reasons.append("single_mode_repeated_leading_electrode_support")

    ranked_electrodes = [
        str(row["electrode"]) for row in validated["ranked_hypotheses"]
    ]
    conclusion_binding = {
        "input_event_count": event_count,
        "top_k": int(validated["top_k"]),
        "ranked_electrodes": ranked_electrodes,
        "top1_electrode": diagnostics["top1_electrode"],
        "top1_support_rate": top1_support,
        "top3_support_rate": top3_support,
        "mode_cluster_count": mode_count,
        "evidence_level": level,
    }

    result: dict[str, Any] = {
        "schema_version": RESEARCH_SOZ_EVIDENCE_SCHEMA_VERSION,
        "policy_id": RESEARCH_SOZ_EVIDENCE_POLICY_ID,
        "recording_id": recording_id,
        "prediction_artifact_id": validated["artifact_id"],
        "prediction_content_sha256": validated["content_sha256"],
        "evidence_level": level,
        "evidence_level_semantics": "uncalibrated_descriptive_only",
        "descriptive_inputs": {
            "input_event_count": event_count,
            "top1_electrode": diagnostics["top1_electrode"],
            "top1_support_rate": top1_support,
            "top3_support_rate": top3_support,
            "mode_cluster_count": mode_count,
            "multimodal": multimodal,
            "jensen_shannon_consistency": consistency[
                "jensen_shannon_consistency"
            ],
            "normalized_entropy": diagnostics["normalized_entropy"],
            "top1_margin": diagnostics["top1_margin"],
        },
        "descriptive_rule": {
            "minimum_stable_event_count": MIN_STABLE_EVENT_COUNT,
            "minimum_stable_top1_support_rate": MIN_STABLE_TOP1_SUPPORT_RATE,
            "minimum_stable_top3_support_rate": MIN_STABLE_TOP3_SUPPORT_RATE,
            "stable_requires_single_complete_link_mode": True,
            "cutpoint_status": (
                "provisional_engineering_description_not_private_cohort_tuned"
            ),
        },
        "reason_codes": reasons,
        "deterministic_research_conclusion": {
            "template_id": "scalp_eeg_onset_candidate_topk_bound_zh_v1",
            "language": "zh-CN",
            "text": _render_deterministic_conclusion(conclusion_binding),
            "binding": conclusion_binding,
        },
        "reporting_policy": {
            "top_k_output_required_for_every_valid_prediction": True,
            "leading_candidate_is_research_scalp_eeg_hypothesis": True,
            "diagnosis_or_treatment_target_claim_prohibited": True,
            "confidence_or_probability_language_prohibited": True,
        },
        "llm_projection_receipt": {
            "llm_input_eligible": True,
            "llm_invoked": False,
            "llm_may_add_facts": False,
            "eligible_input_scope": (
                "deterministic_conclusion_and_bound_rank_statistics_only"
            ),
            "required_projection": "language_only_preserving_all_bound_facts",
            "qwen_service_called": False,
        },
        "future_calibration_receipt": {
            "status": "not_attached",
            "receipt": None,
            "intended_source": "patient_disjoint_tusz_source_development_partition",
            "source_evaluation_partition_must_remain_frozen": True,
            "required_before_confidence_or_probability_language": True,
        },
        "input_boundary": {
            "research_prediction_artifact_only": True,
            "raw_eeg_used": False,
            "edf_annotations_used": False,
            "excel_fields_used": False,
            "doctor_labels_used": False,
            "free_text_used": False,
        },
    }
    result["content_sha256"] = _content_sha256(result)
    return validate_research_soz_descriptive_strength(result)


def validate_research_soz_descriptive_strength(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the compact sidecar and its uncalibrated claim boundary."""

    if not isinstance(payload, Mapping):
        raise TypeError("research SOZ descriptive strength must be a mapping")
    required = {
        "schema_version",
        "policy_id",
        "recording_id",
        "prediction_artifact_id",
        "prediction_content_sha256",
        "evidence_level",
        "evidence_level_semantics",
        "descriptive_inputs",
        "descriptive_rule",
        "reason_codes",
        "deterministic_research_conclusion",
        "reporting_policy",
        "llm_projection_receipt",
        "future_calibration_receipt",
        "input_boundary",
        "content_sha256",
    }
    if set(payload) != required:
        raise ValueError("research SOZ descriptive strength has unexpected keys")
    if payload["schema_version"] != RESEARCH_SOZ_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("unexpected descriptive strength schema version")
    if payload["policy_id"] != RESEARCH_SOZ_EVIDENCE_POLICY_ID:
        raise ValueError("unexpected descriptive strength policy")
    if payload["evidence_level"] not in DESCRIPTIVE_EVIDENCE_LEVELS:
        raise ValueError("unexpected descriptive strength level")
    if payload["evidence_level_semantics"] != "uncalibrated_descriptive_only":
        raise ValueError("descriptive strength was promoted beyond its boundary")
    if payload["recording_id"] is not None and (
        not isinstance(payload["recording_id"], str)
        or not payload["recording_id"]
        or len(payload["recording_id"]) > 128
    ):
        raise ValueError("recording_id must be null or a bounded opaque string")
    if not isinstance(payload["prediction_artifact_id"], str) or not payload[
        "prediction_artifact_id"
    ]:
        raise ValueError("prediction_artifact_id must be a non-empty string")
    _sha256(
        payload["prediction_content_sha256"], "prediction_content_sha256"
    )
    saved_hash = _sha256(payload["content_sha256"], "content_sha256")
    hashable = dict(payload)
    hashable.pop("content_sha256")
    if _content_sha256(hashable) != saved_hash:
        raise ValueError("descriptive strength content hash mismatch")

    inputs = payload["descriptive_inputs"]
    if not isinstance(inputs, Mapping):
        raise TypeError("descriptive_inputs must be a mapping")
    if (
        isinstance(inputs.get("input_event_count"), bool)
        or not isinstance(inputs.get("input_event_count"), int)
        or inputs["input_event_count"] < 1
    ):
        raise ValueError("descriptive input event count must be positive")
    for key in (
        "top1_support_rate",
        "top3_support_rate",
        "jensen_shannon_consistency",
        "normalized_entropy",
        "top1_margin",
    ):
        _finite_rate(inputs.get(key), f"descriptive_inputs.{key}")
    if (
        isinstance(inputs.get("mode_cluster_count"), bool)
        or not isinstance(inputs.get("mode_cluster_count"), int)
        or inputs["mode_cluster_count"] < 1
    ):
        raise ValueError("mode_cluster_count must be positive")
    if not isinstance(inputs.get("multimodal"), bool):
        raise TypeError("multimodal must be boolean")
    if inputs["multimodal"] is not (inputs["mode_cluster_count"] > 1):
        raise ValueError("multimodal flag does not match mode count")

    reporting = payload["reporting_policy"]
    if not isinstance(reporting, Mapping) or any(
        reporting.get(key) is not True
        for key in (
            "top_k_output_required_for_every_valid_prediction",
            "leading_candidate_is_research_scalp_eeg_hypothesis",
            "diagnosis_or_treatment_target_claim_prohibited",
            "confidence_or_probability_language_prohibited",
        )
    ):
        raise ValueError("descriptive strength reporting boundary is incomplete")
    calibration = payload["future_calibration_receipt"]
    if (
        not isinstance(calibration, Mapping)
        or calibration.get("status") != "not_attached"
        or calibration.get("receipt") is not None
        or calibration.get("required_before_confidence_or_probability_language")
        is not True
    ):
        raise ValueError("unexpected calibration receipt state")
    boundary = payload["input_boundary"]
    if not isinstance(boundary, Mapping):
        raise TypeError("input_boundary must be a mapping")
    if boundary.get("research_prediction_artifact_only") is not True or any(
        boundary.get(key) is not False
        for key in (
            "raw_eeg_used",
            "edf_annotations_used",
            "excel_fields_used",
            "doctor_labels_used",
            "free_text_used",
        )
    ):
        raise ValueError("descriptive strength admits a prohibited input")
    if not isinstance(payload["reason_codes"], list) or not payload[
        "reason_codes"
    ]:
        raise ValueError("reason_codes must be a non-empty list")
    conclusion = payload["deterministic_research_conclusion"]
    if not isinstance(conclusion, Mapping):
        raise TypeError("deterministic_research_conclusion must be a mapping")
    if set(conclusion) != {"template_id", "language", "text", "binding"}:
        raise ValueError("deterministic research conclusion has unexpected keys")
    if (
        conclusion["template_id"] != "scalp_eeg_onset_candidate_topk_bound_zh_v1"
        or conclusion["language"] != "zh-CN"
        or not isinstance(conclusion["binding"], Mapping)
        or conclusion["text"] != _render_deterministic_conclusion(
            conclusion["binding"]
        )
    ):
        raise ValueError("deterministic research conclusion is not fact-bound")
    binding = conclusion["binding"]
    if set(binding) != {
        "input_event_count",
        "top_k",
        "ranked_electrodes",
        "top1_electrode",
        "top1_support_rate",
        "top3_support_rate",
        "mode_cluster_count",
        "evidence_level",
    }:
        raise ValueError("deterministic conclusion binding has unexpected keys")
    ranked_electrodes = binding["ranked_electrodes"]
    if (
        not isinstance(ranked_electrodes, list)
        or len(ranked_electrodes) != binding["top_k"]
        or not ranked_electrodes
        or binding["top1_electrode"] != ranked_electrodes[0]
        or len(ranked_electrodes) != len(set(ranked_electrodes))
    ):
        raise ValueError("deterministic conclusion Top-k binding is invalid")
    if (
        binding["input_event_count"] != inputs["input_event_count"]
        or binding["top1_electrode"] != inputs["top1_electrode"]
        or binding["top1_support_rate"] != inputs["top1_support_rate"]
        or binding["top3_support_rate"] != inputs["top3_support_rate"]
        or binding["mode_cluster_count"] != inputs["mode_cluster_count"]
        or binding["evidence_level"] != payload["evidence_level"]
    ):
        raise ValueError("deterministic conclusion disagrees with descriptive inputs")
    llm_receipt = payload["llm_projection_receipt"]
    if not isinstance(llm_receipt, Mapping) or (
        llm_receipt.get("llm_input_eligible") is not True
        or llm_receipt.get("llm_invoked") is not False
        or llm_receipt.get("llm_may_add_facts") is not False
        or llm_receipt.get("qwen_service_called") is not False
        or llm_receipt.get("eligible_input_scope")
        != "deterministic_conclusion_and_bound_rank_statistics_only"
        or llm_receipt.get("required_projection")
        != "language_only_preserving_all_bound_facts"
    ):
        raise ValueError("LLM projection receipt violates fact-locking")
    return deepcopy(dict(payload))


__all__ = [
    "DESCRIPTIVE_EVIDENCE_LEVELS",
    "LIMITED_CROSS_EVENT_CONSISTENCY",
    "MIN_STABLE_EVENT_COUNT",
    "MIN_STABLE_TOP1_SUPPORT_RATE",
    "MIN_STABLE_TOP3_SUPPORT_RATE",
    "MULTIMODAL_OR_WEAK_RANKED_HYPOTHESES",
    "RESEARCH_SOZ_EVIDENCE_POLICY_ID",
    "RESEARCH_SOZ_EVIDENCE_SCHEMA_VERSION",
    "STABLE_LEADING_CANDIDATE",
    "classify_research_soz_descriptive_strength",
    "validate_research_soz_descriptive_strength",
]
