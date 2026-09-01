"""Validator for the real-data EviSOZ shadow inference receipt."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.evisoz.data.artifact_ref import canonical_json_sha256


REAL_SHADOW_INFERENCE_RECEIPT_SCHEMA_VERSION = "evisoz_real_shadow_inference_receipt_v1"
_HASH_PLACEHOLDER = "0" * 64


def _hash_source(value: dict[str, Any]) -> dict[str, Any]:
    body = deepcopy(value)
    body["receipt_sha256"] = _HASH_PLACEHOLDER
    return body


def validate_real_shadow_inference_receipt(value: object) -> dict[str, Any]:
    """Validate the no-generation real shadow receipt and safety boundary."""

    required = {
        "schema_version", "status", "stage0_status", "event_count", "patient_count",
        "source", "baseline", "adapter", "safety", "receipt_sha256",
    }
    if type(value) is not dict or set(value) != required:
        raise ValueError("real shadow inference receipt fields drifted")
    data = deepcopy(value)
    if data["schema_version"] != REAL_SHADOW_INFERENCE_RECEIPT_SCHEMA_VERSION:
        raise ValueError("real shadow inference receipt schema drifted")
    if data["status"] != "real_data_shadow_evidence_and_report_plan_only":
        raise ValueError("real shadow inference receipt status drifted")
    if data["stage0_status"] != "NO_GO":
        raise ValueError("real shadow receipt must remain Stage-0 NO_GO")
    for key in ("event_count", "patient_count"):
        if type(data[key]) is not int or data[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    source = data["source"]
    if type(source) is not dict or set(source) != {
        "bound_evidence_root", "loader_receipt_sha256", "evaluation_receipt_sha256"
    }:
        raise ValueError("real shadow receipt source fields drifted")
    for key in ("bound_evidence_root", "loader_receipt_sha256", "evaluation_receipt_sha256"):
        if not isinstance(source[key], str) or not source[key]:
            raise ValueError(f"real shadow receipt source.{key} is invalid")
    for key in ("loader_receipt_sha256", "evaluation_receipt_sha256"):
        if len(source[key]) != 64:
            raise ValueError(f"real shadow receipt source.{key} hash is invalid")
    baseline = data["baseline"]
    if type(baseline) is not dict or baseline.get("method_id") != "canonical_v29_equal_H_D_probability_ensemble":
        raise ValueError("real shadow receipt baseline identity drifted")
    for key in ("manifest_sha256", "predictions_sha256", "candidate_mask_sha256"):
        if not isinstance(baseline.get(key), str) or len(baseline[key]) != 64:
            raise ValueError(f"real shadow receipt baseline.{key} is invalid")
    if baseline.get("baseline_role") != "frozen_v29_identity_reference_only":
        raise ValueError("real shadow receipt baseline role drifted")
    adapter = data["adapter"]
    if type(adapter) is not dict or adapter.get("node") != "official_labram_base_patch200_200" or adapter.get("edge") != "signed_tcp22_independent_temporal_patch_encoder":
        raise ValueError("real shadow receipt adapter identity drifted")
    if not isinstance(adapter.get("node_units"), list) or len(adapter["node_units"]) != 19:
        raise ValueError("real shadow receipt Standard19 roster drifted")
    if not isinstance(adapter.get("edge_units"), list) or len(adapter["edge_units"]) != 22:
        raise ValueError("real shadow receipt TCP22 roster drifted")
    if adapter.get("token_dim") != 128 or adapter.get("projection_mode") != "fixed_shadow" or adapter.get("endpoint_expansion") is not False:
        raise ValueError("real shadow receipt adapter contract drifted")
    if data["safety"] != {
        "formal_training": False,
        "residual_enabled": False,
        "residual_identity_all_events": True,
        "teacher_runtime_opened": False,
        "physician_report_text_opened": False,
        "canonical_shadow_report_opened": True,
        "knowledge_card_text_opened": False,
        "qwen_generation": False,
        "tcp22_edge_to_node_label_expansion": False,
        "predicted_packets_are_patient_facts": False,
    }:
        raise ValueError("real shadow receipt safety policy drifted")
    if not isinstance(data["receipt_sha256"], str) or len(data["receipt_sha256"]) != 64:
        raise ValueError("real shadow receipt hash is invalid")
    if data["receipt_sha256"] != canonical_json_sha256(_hash_source(data)):
        raise ValueError("real shadow receipt hash drifted")
    return data


__all__ = [
    "REAL_SHADOW_INFERENCE_RECEIPT_SCHEMA_VERSION",
    "validate_real_shadow_inference_receipt",
]
