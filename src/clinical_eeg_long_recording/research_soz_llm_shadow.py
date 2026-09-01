"""Fact-locked Qwen shadow reasoner for research scalp-channel ranking.

The language model never sees raw EEG, annotations, spreadsheets, clinical
text, or ground truth.  It receives a validated cross-event ranking artifact
and a small controlled projection of signal findings.  Its output is a strict
JSON reordering of the already supplied Top-k candidate set with event-level
citations; free-form prose is not accepted.

This module deliberately does not call a model service and does not promote
the shadow result into a report.  Promotion requires a patient-isolated TUSZ
comparison against the deterministic rank aggregator.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

from .research_soz_prediction import (
    C18_ELECTRODES,
    validate_research_soz_prediction_artifact,
)


RESEARCH_SOZ_LLM_SHADOW_INPUT_SCHEMA_VERSION = (
    "clinical_eeg_research_soz_llm_shadow_input_v1"
)
RESEARCH_SOZ_LLM_SHADOW_OUTPUT_SCHEMA_VERSION = (
    "clinical_eeg_research_soz_llm_shadow_output_v1"
)
RESEARCH_SOZ_LLM_SHADOW_METHOD_ID = "qwen36_fact_locked_topk_reranker_shadow_v1"

_WINDOW_STATUSES = {
    "complete_variable_window",
    "right_censored_variable_window",
    "left_censored_nonlocalizing_window",
}
_QUALITY_STATUSES = {"qualified", "limited", "unusable"}
_BANDS = {"delta", "theta", "alpha", "beta", "low_gamma", "unavailable"}
_RHYTHMICITY = {"rhythmic", "quasi_rhythmic", "non_rhythmic", "unavailable"}
_BOUNDARY = {"resolved_candidate", "left_censored", "right_censored", "unresolved"}
_FINDING_CODES = {
    "sustained_signal_change",
    "rhythmic_pattern",
    "periodic_pattern",
    "frequency_evolution_candidate",
    "morphology_evolution_candidate",
    "spatial_recruitment_candidate",
    "amplitude_trajectory_description",
    "termination_candidate",
    "postevent_change_candidate",
}
_REASON_CODES = {
    "cross_event_top1_support",
    "cross_event_top3_support",
    "rank_margin_support",
    "rank_entropy_limit",
    "mode_consistency_support",
    "mode_conflict_limit",
    "early_change_support",
    "signal_quality_support",
    "left_censoring_limit",
    "right_censoring_limit",
}
_UNCERTAINTY_CODES = {
    "stable_leading_candidate",
    "candidate_with_limited_cross_event_consistency",
    "multimodal_or_weak_topk_hypotheses",
}
_DERIVATION_RE = re.compile(r"^[A-Z0-9]{1,4}-[A-Z0-9]{1,4}$")
_CANONICAL_DERIVATION_ELECTRODES = frozenset((*C18_ELECTRODES, "PZ"))
_SHA256_HEX = frozenset("0123456789abcdef")


def _canonical_bytes(value: object) -> bytes:
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


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_digest(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256 hex digest")
    return value


def _mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping")
    return value


def _finite(value: object, context: str, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum or result > maximum:
        raise ValueError(f"{context} is outside its allowed range")
    return result


def _identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 128:
        raise TypeError(f"{context} must be a non-empty opaque identifier")
    if any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:-" for character in value):
        raise ValueError(f"{context} contains non-identifier characters")
    return value


def _all_event_ids(artifact: Mapping[str, Any]) -> set[str]:
    result: set[str] = set()
    for cluster in artifact["event_mode_clusters"]:
        for event_id in cluster["event_ids"]:
            result.add(str(event_id))
    if len(result) != int(artifact["input_event_count"]):
        raise ValueError("research artifact event clusters do not cover input events")
    return result


def _event_reference_map(event_ids: set[str]) -> dict[str, str]:
    """Create prompt-local references so source identifiers never reach Qwen."""

    return {
        f"EEG-EVENT-{index:04d}": event_id
        for index, event_id in enumerate(sorted(event_ids), start=1)
    }


def _validate_event_evidence(
    values: object, *, expected_event_ids: set[str]
) -> list[dict[str, Any]]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError("event_evidence must be a sequence")
    required = {
        "event_id",
        "window_status",
        "signal_quality_status",
        "finding_codes",
        "dominant_frequency_band",
        "rhythmicity",
        "supporting_bipolar_derivations",
        "onset_boundary_status",
        "evidence_weight",
    }
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(values):
        row = _mapping(raw, f"event_evidence[{index}]")
        if set(row) != required:
            raise ValueError("event evidence has missing or unknown fields")
        event_id = _identifier(row["event_id"], "event evidence event_id")
        if event_id not in expected_event_ids or event_id in seen:
            raise ValueError("event evidence IDs must exactly cover artifact events")
        seen.add(event_id)
        if row["window_status"] not in _WINDOW_STATUSES:
            raise ValueError("event evidence window status is invalid")
        if row["signal_quality_status"] not in _QUALITY_STATUSES:
            raise ValueError("event evidence signal quality is invalid")
        if row["dominant_frequency_band"] not in _BANDS:
            raise ValueError("event evidence frequency band is invalid")
        if row["rhythmicity"] not in _RHYTHMICITY:
            raise ValueError("event evidence rhythmicity is invalid")
        if row["onset_boundary_status"] not in _BOUNDARY:
            raise ValueError("event evidence boundary status is invalid")
        codes = row["finding_codes"]
        if (
            not isinstance(codes, list)
            or codes != sorted(set(codes))
            or not set(codes) <= _FINDING_CODES
        ):
            raise ValueError("event evidence finding codes are invalid")
        derivations = row["supporting_bipolar_derivations"]
        if (
            not isinstance(derivations, list)
            or derivations != sorted(set(derivations))
            or any(
                not isinstance(item, str)
                or not _DERIVATION_RE.fullmatch(item)
                or any(
                    electrode not in _CANONICAL_DERIVATION_ELECTRODES
                    for electrode in item.split("-", maxsplit=1)
                )
                or len(set(item.split("-", maxsplit=1))) != 2
                for item in derivations
            )
        ):
            raise ValueError("event evidence bipolar derivations are invalid")
        weight = _finite(row["evidence_weight"], "event evidence weight", minimum=0.0, maximum=1.0)
        result.append(
            {
                "event_id": event_id,
                "window_status": row["window_status"],
                "signal_quality_status": row["signal_quality_status"],
                "finding_codes": list(codes),
                "dominant_frequency_band": row["dominant_frequency_band"],
                "rhythmicity": row["rhythmicity"],
                "supporting_bipolar_derivations": list(derivations),
                "onset_boundary_status": row["onset_boundary_status"],
                "evidence_weight": weight,
            }
        )
    if seen != expected_event_ids:
        raise ValueError("event evidence must exactly cover all ranked events")
    result.sort(key=lambda item: item["event_id"])
    return result


def _model_projection(
    artifact: Mapping[str, Any], *, source_to_reference: Mapping[str, str]
) -> dict[str, Any]:
    return {
        "input_event_count": artifact["input_event_count"],
        "ranked_hypotheses": [
            {
                "rank": row["rank"],
                "electrode": row["electrode"],
                "top1_support_rate": row["top1_support_rate"],
                "top3_support_rate": row["top3_support_rate"],
                "aggregate_rank_proxy": row["aggregate_rank_proxy"],
            }
            for row in artifact["ranked_hypotheses"]
        ],
        "aggregate_diagnostics": deepcopy(artifact["aggregate_diagnostics"]),
        "cross_event_consistency": deepcopy(artifact["cross_event_consistency"]),
        "event_mode_clusters": [
            {
                "cluster_id": cluster["cluster_id"],
                "event_ids": [
                    source_to_reference[str(event_id)]
                    for event_id in cluster["event_ids"]
                ],
                "ranked_hypotheses": [
                    {
                        "rank": row["rank"],
                        "electrode": row["electrode"],
                        "aggregate_rank_proxy": row["aggregate_rank_proxy"],
                    }
                    for row in cluster["ranked_hypotheses"]
                ],
            }
            for cluster in artifact["event_mode_clusters"]
        ],
    }


def llm_shadow_json_schema(
    *,
    top_k: int,
    candidate_electrodes: Sequence[str],
    event_references: Sequence[str],
    input_sha256: str,
) -> dict[str, Any]:
    candidate_enum = list(candidate_electrodes)
    event_enum = list(event_references)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if (
        len(candidate_enum) != top_k
        or len(set(candidate_enum)) != top_k
        or any(candidate not in C18_ELECTRODES for candidate in candidate_enum)
    ):
        raise ValueError("candidate_electrodes must be a unique canonical Top-k")
    if (
        not event_enum
        or event_enum != sorted(set(event_enum))
        or any(not re.fullmatch(r"EEG-EVENT-[0-9]{4}", item) for item in event_enum)
    ):
        raise ValueError("event_references must be sorted unique prompt-local IDs")
    input_sha256 = _sha256_digest(input_sha256, "input_sha256")
    row = {
        "type": "object",
        "additionalProperties": False,
        "required": ["rank", "electrode", "support_event_ids", "reason_codes"],
        "properties": {
            "rank": {"type": "integer", "minimum": 1, "maximum": top_k},
            "electrode": {"type": "string", "enum": candidate_enum},
            "support_event_ids": {
                "type": "array",
                "minItems": 1,
                "uniqueItems": True,
                "items": {"type": "string", "enum": event_enum},
            },
            "reason_codes": {
                "type": "array",
                "minItems": 1,
                "uniqueItems": True,
                "items": {"type": "string", "enum": sorted(_REASON_CODES)},
            },
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "method_id",
            "input_sha256",
            "uncertainty_code",
            "ranked_candidates",
        ],
        "properties": {
            "schema_version": {
                "type": "string",
                "const": RESEARCH_SOZ_LLM_SHADOW_OUTPUT_SCHEMA_VERSION,
            },
            "method_id": {"type": "string", "const": RESEARCH_SOZ_LLM_SHADOW_METHOD_ID},
            "input_sha256": {"type": "string", "const": input_sha256},
            "uncertainty_code": {"type": "string", "enum": sorted(_UNCERTAINTY_CODES)},
            "ranked_candidates": {
                "type": "array",
                "minItems": top_k,
                "maxItems": top_k,
                "items": row,
            },
        },
    }


def build_research_soz_llm_shadow_request(
    *,
    recording_id: str,
    research_prediction_artifact: Mapping[str, Any],
    event_evidence: Sequence[Mapping[str, Any]],
) -> tuple[str, str, dict[str, Any], dict[str, Any]]:
    """Build a prompt/schema pair that cannot ask Qwen to invent candidates."""

    recording_id = _identifier(recording_id, "recording_id")
    artifact = validate_research_soz_prediction_artifact(research_prediction_artifact)
    event_ids = _all_event_ids(artifact)
    evidence = _validate_event_evidence(event_evidence, expected_event_ids=event_ids)
    reference_to_source = _event_reference_map(event_ids)
    source_to_reference = {
        source: reference for reference, source in reference_to_source.items()
    }
    top_candidates = [str(row["electrode"]) for row in artifact["ranked_hypotheses"]]
    payload = {
        "schema_version": RESEARCH_SOZ_LLM_SHADOW_INPUT_SCHEMA_VERSION,
        "method_id": RESEARCH_SOZ_LLM_SHADOW_METHOD_ID,
        "deterministic_prediction": _model_projection(
            artifact, source_to_reference=source_to_reference
        ),
        "event_evidence": [
            {
                **row,
                "event_id": source_to_reference[row["event_id"]],
            }
            for row in evidence
        ],
        "constraints": {
            "allowed_candidate_electrodes": top_candidates,
            "must_return_exact_permutation_of_allowed_candidates": True,
            "must_cite_supporting_event_ids": True,
            "free_text_prohibited": True,
            "probability_claim_prohibited": True,
            "aggregate_rank_proxy_semantics": (
                "ordinal_only_uncalibrated_not_probability"
            ),
            "cortical_soz_ez_or_treatment_claim_prohibited": True,
            "ground_truth_annotations_excel_or_clinical_context_available": False,
        },
    }
    system_prompt = (
        "你是研究性头皮 EEG 起始候选通道排序器。只根据用户提供的结构化 EEG 事件证据，"
        "在给定候选集合内重新排序。不得新增通道、事件、形态、时间或临床事实；不得输出"
        "皮层 SOZ、致痫区、治疗靶点或个体正确概率。每个候选必须引用支持事件 ID。"
        "只返回符合 JSON Schema 的 JSON，不得返回自然语言。"
    )
    user_prompt = json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
    input_sha256 = _sha256(payload)
    schema = llm_shadow_json_schema(
        top_k=len(top_candidates),
        candidate_electrodes=top_candidates,
        event_references=sorted(reference_to_source),
        input_sha256=input_sha256,
    )
    receipt = {
        "schema_version": RESEARCH_SOZ_LLM_SHADOW_INPUT_SCHEMA_VERSION,
        "method_id": RESEARCH_SOZ_LLM_SHADOW_METHOD_ID,
        "recording_id": recording_id,
        "research_prediction_artifact_sha256": artifact["content_sha256"],
        "input_sha256": input_sha256,
        "system_prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
        "user_prompt_sha256": hashlib.sha256(user_prompt.encode("utf-8")).hexdigest(),
        "json_schema_sha256": _sha256(schema),
        "event_reference_map": reference_to_source,
        "scope_receipt": {
            "eeg_structured_evidence_only": True,
            "raw_eeg_sent": False,
            "recording_id_sent": False,
            "source_event_ids_sent": False,
            "edf_annotations_sent": False,
            "excel_sent": False,
            "doctor_labels_or_ground_truth_sent": False,
            "clinical_context_sent": False,
            "rank_proxy_calibrated": False,
            "probability_claim_prohibited": True,
            "may_override_deterministic_prediction": False,
            "promotion_status": "shadow_pending_patient_level_qualification",
        },
    }
    receipt["content_sha256"] = _sha256(receipt)
    return system_prompt, user_prompt, schema, receipt


def validate_research_soz_llm_shadow_output(
    payload: object,
    *,
    research_prediction_artifact: Mapping[str, Any],
    request_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate that the LLM only reorders supplied candidates and cites events."""

    artifact = validate_research_soz_prediction_artifact(research_prediction_artifact)
    receipt = _mapping(request_receipt, "LLM shadow request receipt")
    expected_receipt_fields = {
        "schema_version",
        "method_id",
        "recording_id",
        "research_prediction_artifact_sha256",
        "input_sha256",
        "system_prompt_sha256",
        "user_prompt_sha256",
        "json_schema_sha256",
        "event_reference_map",
        "scope_receipt",
        "content_sha256",
    }
    if set(receipt) != expected_receipt_fields:
        raise ValueError("LLM shadow request receipt has missing or unknown fields")
    if receipt["schema_version"] != RESEARCH_SOZ_LLM_SHADOW_INPUT_SCHEMA_VERSION:
        raise ValueError("LLM shadow request receipt schema drifted")
    if receipt["method_id"] != RESEARCH_SOZ_LLM_SHADOW_METHOD_ID:
        raise ValueError("LLM shadow request receipt method drifted")
    saved_receipt_sha256 = _sha256_digest(
        receipt["content_sha256"], "request receipt content SHA-256"
    )
    hashable_receipt = dict(receipt)
    hashable_receipt.pop("content_sha256")
    if _sha256(hashable_receipt) != saved_receipt_sha256:
        raise ValueError("LLM shadow request receipt content hash mismatch")
    _identifier(receipt["recording_id"], "request receipt recording_id")
    if (
        _sha256_digest(
            receipt["research_prediction_artifact_sha256"],
            "request receipt artifact SHA-256",
        )
        != artifact["content_sha256"]
    ):
        raise ValueError("LLM shadow request receipt belongs to another artifact")
    expected_input_sha256 = _sha256_digest(
        receipt["input_sha256"], "request receipt input SHA-256"
    )
    for key in ("system_prompt_sha256", "user_prompt_sha256", "json_schema_sha256"):
        _sha256_digest(receipt[key], f"request receipt {key}")
    event_ids = _all_event_ids(artifact)
    expected_reference_map = _event_reference_map(event_ids)
    reference_map = _mapping(
        receipt["event_reference_map"], "request receipt event_reference_map"
    )
    if dict(reference_map) != expected_reference_map:
        raise ValueError("LLM shadow event reference map does not match artifact events")
    expected_candidates = [
        str(row["electrode"]) for row in artifact["ranked_hypotheses"]
    ]
    expected_schema = llm_shadow_json_schema(
        top_k=len(expected_candidates),
        candidate_electrodes=expected_candidates,
        event_references=sorted(expected_reference_map),
        input_sha256=expected_input_sha256,
    )
    if receipt["json_schema_sha256"] != _sha256(expected_schema):
        raise ValueError("LLM shadow request receipt schema hash is inconsistent")
    scope = _mapping(receipt["scope_receipt"], "request receipt scope_receipt")
    expected_scope = {
        "eeg_structured_evidence_only": True,
        "raw_eeg_sent": False,
        "recording_id_sent": False,
        "source_event_ids_sent": False,
        "edf_annotations_sent": False,
        "excel_sent": False,
        "doctor_labels_or_ground_truth_sent": False,
        "clinical_context_sent": False,
        "rank_proxy_calibrated": False,
        "probability_claim_prohibited": True,
        "may_override_deterministic_prediction": False,
        "promotion_status": "shadow_pending_patient_level_qualification",
    }
    if dict(scope) != expected_scope:
        raise ValueError("LLM shadow request receipt does not preserve its scope firewall")
    value = _mapping(payload, "LLM shadow output")
    required = {
        "schema_version",
        "method_id",
        "input_sha256",
        "uncertainty_code",
        "ranked_candidates",
    }
    if set(value) != required:
        raise ValueError("LLM shadow output has missing or unknown fields")
    if value["schema_version"] != RESEARCH_SOZ_LLM_SHADOW_OUTPUT_SCHEMA_VERSION:
        raise ValueError("LLM shadow output schema drifted")
    if value["method_id"] != RESEARCH_SOZ_LLM_SHADOW_METHOD_ID:
        raise ValueError("LLM shadow method drifted")
    if (
        _sha256_digest(value["input_sha256"], "LLM shadow input SHA-256")
        != expected_input_sha256
    ):
        raise ValueError("LLM shadow output is not bound to this request")
    if (
        not isinstance(value["uncertainty_code"], str)
        or value["uncertainty_code"] not in _UNCERTAINTY_CODES
    ):
        raise ValueError("LLM shadow uncertainty code is invalid")
    event_references = set(expected_reference_map)
    rows = value["ranked_candidates"]
    if not isinstance(rows, list) or len(rows) != len(expected_candidates):
        raise ValueError("LLM shadow ranked candidates must preserve Top-k length")
    result_rows: list[dict[str, Any]] = []
    seen_candidates: set[str] = set()
    for expected_rank, raw in enumerate(rows, start=1):
        row = _mapping(raw, f"ranked_candidates[{expected_rank - 1}]")
        if set(row) != {"rank", "electrode", "support_event_ids", "reason_codes"}:
            raise ValueError("LLM shadow candidate row has invalid fields")
        if (
            isinstance(row["rank"], bool)
            or not isinstance(row["rank"], int)
            or row["rank"] != expected_rank
        ):
            raise ValueError("LLM shadow ranks must be contiguous")
        electrode = row["electrode"]
        if (
            not isinstance(electrode, str)
            or electrode not in expected_candidates
            or electrode in seen_candidates
        ):
            raise ValueError("LLM shadow invented or repeated an electrode")
        seen_candidates.add(str(electrode))
        support_ids = row["support_event_ids"]
        if (
            not isinstance(support_ids, list)
            or not support_ids
            or any(not isinstance(item, str) for item in support_ids)
            or support_ids != sorted(set(support_ids))
            or not set(support_ids) <= event_references
        ):
            raise ValueError("LLM shadow support event citations are invalid")
        reason_codes = row["reason_codes"]
        if (
            not isinstance(reason_codes, list)
            or not reason_codes
            or any(not isinstance(item, str) for item in reason_codes)
            or reason_codes != sorted(set(reason_codes))
            or not set(reason_codes) <= _REASON_CODES
        ):
            raise ValueError("LLM shadow reason codes are invalid")
        result_rows.append(
            {
                "rank": expected_rank,
                "electrode": electrode,
                "support_event_ids": [
                    expected_reference_map[reference] for reference in support_ids
                ],
                "reason_codes": list(reason_codes),
            }
        )
    if seen_candidates != set(expected_candidates):
        raise ValueError("LLM shadow output must be a permutation of supplied Top-k")
    return {
        "schema_version": value["schema_version"],
        "method_id": value["method_id"],
        "input_sha256": value["input_sha256"],
        "uncertainty_code": value["uncertainty_code"],
        "ranked_candidates": result_rows,
        "promotion_receipt": {
            "role": "unpromoted_shadow_result",
            "output_semantics": "ordinal_scalp_eeg_topk_only",
            "rank_proxy_calibrated": False,
            "probability_claim_prohibited": True,
            "may_override_deterministic_prediction": False,
            "patient_level_tusz_qualification_complete": False,
        },
    }


__all__ = [
    "RESEARCH_SOZ_LLM_SHADOW_INPUT_SCHEMA_VERSION",
    "RESEARCH_SOZ_LLM_SHADOW_METHOD_ID",
    "RESEARCH_SOZ_LLM_SHADOW_OUTPUT_SCHEMA_VERSION",
    "build_research_soz_llm_shadow_request",
    "llm_shadow_json_schema",
    "validate_research_soz_llm_shadow_output",
]
