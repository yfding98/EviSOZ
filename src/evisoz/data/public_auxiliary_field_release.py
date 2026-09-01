"""Capability-only field release for public auxiliary cohorts.

This receipt describes which *typed fields could be exposed* after a future
authorization.  It deliberately contains no patient label values and grants
no training permission; it is therefore safe to materialize before overlap
and TUEV-evaluation identity closure are complete.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from .artifact_ref import build_json_artifact_ref, canonical_json_sha256, validate_artifact_ref


PUBLIC_AUXILIARY_FIELD_RELEASE_SCHEMA_VERSION = (
    "evisoz_public_auxiliary_field_release_v1"
)
_HASH_PLACEHOLDER = "0" * 64
_PENDING_ID = "CONTENT-ADDRESS-PENDING"
_ID_PREFIX = "EVISOZ-PUBFIELDS-"


def _hash_source(value: Mapping[str, object]) -> dict[str, object]:
    result = deepcopy(dict(value))
    result["receipt_sha256"] = _HASH_PLACEHOLDER
    return result


def _id_source(value: Mapping[str, object]) -> dict[str, object]:
    result = _hash_source(value)
    result["release_id"] = _PENDING_ID
    return result


def _capability_rows() -> list[dict[str, Any]]:
    no_loss = {
        "typed_slot_loss": False,
        "node_localization_loss": False,
        "report_text_loss": False,
    }
    rows = [
        {
            "field_id": "TUSZ-ANNOTATED-CHANNEL",
            "field_path": "ictal_findings.annotated_channel",
            "semantic_role": "other",
            "payload_schema_version": "evisoz_candidate_value_v1",
            "allowed_roles": ["development_cv"],
            "loss_allowed": dict(no_loss),
            "report_target_allowed": False,
            "prompt_or_rag_allowed": False,
        },
        {
            "field_id": "TUSZ-ICTAL-INTERVAL",
            "field_path": "ictal_findings.interval",
            "semantic_role": "evolution",
            "payload_schema_version": "evisoz_categorical_label_value_v1",
            "allowed_roles": ["development_cv"],
            "loss_allowed": dict(no_loss),
            "report_target_allowed": False,
            "prompt_or_rag_allowed": False,
        },
        {
            "field_id": "TUSZ-SEIZURE-TYPE",
            "field_path": "ictal_findings.seizure_type",
            "semantic_role": "other",
            "payload_schema_version": "evisoz_categorical_label_value_v1",
            "allowed_roles": ["development_cv"],
            "loss_allowed": dict(no_loss),
            "report_target_allowed": False,
            "prompt_or_rag_allowed": False,
        },
        {
            "field_id": "TUSZ-SIGNAL-QUALITY",
            "field_path": "ictal_findings.signal_quality",
            "semantic_role": "quality",
            "payload_schema_version": "evisoz_categorical_label_value_v1",
            "allowed_roles": ["development_cv"],
            "loss_allowed": dict(no_loss),
            "report_target_allowed": False,
            "prompt_or_rag_allowed": False,
        },
        {
            "field_id": "TUSZ-WEAK-MOTIF",
            "field_path": "ictal_findings.motif_event",
            "semantic_role": "morphology",
            "payload_schema_version": "evisoz_candidate_value_v1",
            "allowed_roles": ["development_cv"],
            "loss_allowed": dict(no_loss),
            "report_target_allowed": False,
            "prompt_or_rag_allowed": False,
        },
    ]
    return sorted(rows, key=lambda row: row["field_id"])


def build_public_auxiliary_field_release(
    *,
    projection: Mapping[str, object],
) -> dict[str, Any]:
    """Build a capability-only release bound to the public split projection."""

    if type(projection) is not dict:
        raise TypeError("public auxiliary projection must be an object")
    if projection.get("schema_version") != "evisoz_public_auxiliary_exposure_projection_v1":
        raise ValueError("public auxiliary projection schema drifted")
    projection_ref = build_json_artifact_ref(
        projection,
        artifact_kind="public_auxiliary_exposure_projection",
        payload_schema_version="evisoz_public_auxiliary_exposure_projection_v1",
    )
    body: dict[str, Any] = {
        "schema_version": PUBLIC_AUXILIARY_FIELD_RELEASE_SCHEMA_VERSION,
        "release_id": _PENDING_ID,
        "status": "capability_catalog_materialized_training_disabled",
        "projection_ref": projection_ref,
        "datasets": [
            {
                "dataset_id": "tusz",
                "report_scope": "ictal_findings",
                "patient_count": projection["counts"]["tusz_source_train_patient_count"],
                "field_roster": _capability_rows(),
                "field_values_materialized": False,
                "raw_patient_identifiers_stored": False,
            }
        ],
        "counts": {
            "dataset_count": 1,
            "field_count": len(_capability_rows()),
            "patient_count": projection["counts"]["tusz_source_train_patient_count"],
        },
        "permissions": {
            "field_values_training_authorized": False,
            "report_target_authorized": False,
            "self_supervised_signal_use_authorized": False,
            "outer_fold_exclusion_required": True,
            "knowledge_can_create_patient_fact": False,
            "raw_patient_identifiers_stored": False,
        },
        "missing_closure_codes": [
            "field_values_not_materialized",
            "near_or_partial_content_overlap_not_materialized",
            "tuev_eval_patient_identity_opaque",
        ],
        "receipt_sha256": _HASH_PLACEHOLDER,
    }
    body["release_id"] = _ID_PREFIX + canonical_json_sha256(_id_source(body))[:24]
    body["receipt_sha256"] = canonical_json_sha256(_hash_source(body))
    return validate_public_auxiliary_field_release(body)


def validate_public_auxiliary_field_release(value: object) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {
        "schema_version",
        "release_id",
        "status",
        "projection_ref",
        "datasets",
        "counts",
        "permissions",
        "missing_closure_codes",
        "receipt_sha256",
    }:
        raise ValueError("public auxiliary field release fields drifted")
    data = deepcopy(value)
    if data["schema_version"] != PUBLIC_AUXILIARY_FIELD_RELEASE_SCHEMA_VERSION:
        raise ValueError("public auxiliary field release schema drifted")
    if data["status"] != "capability_catalog_materialized_training_disabled":
        raise ValueError("public auxiliary field release status drifted")
    validate_artifact_ref(data["projection_ref"])
    datasets = data["datasets"]
    if not isinstance(datasets, list) or len(datasets) != 1:
        raise ValueError("public auxiliary field release dataset roster drifted")
    dataset = datasets[0]
    if type(dataset) is not dict or set(dataset) != {
        "dataset_id",
        "report_scope",
        "patient_count",
        "field_roster",
        "field_values_materialized",
        "raw_patient_identifiers_stored",
    }:
        raise ValueError("public auxiliary field release dataset fields drifted")
    if dataset["dataset_id"] != "tusz" or dataset["report_scope"] != "ictal_findings":
        raise ValueError("public auxiliary field release dataset identity drifted")
    if (
        isinstance(dataset["patient_count"], bool)
        or not isinstance(dataset["patient_count"], int)
        or dataset["patient_count"] <= 0
        or dataset["field_values_materialized"] is not False
        or dataset["raw_patient_identifiers_stored"] is not False
    ):
        raise ValueError("public auxiliary field release dataset policy drifted")
    fields = dataset["field_roster"]
    if fields != _capability_rows():
        raise ValueError("public auxiliary field release field roster drifted")
    for row in fields:
        if any(row["loss_allowed"].values()) or row["report_target_allowed"] or row["prompt_or_rag_allowed"]:
            raise ValueError("public auxiliary field release unexpectedly grants use")
    expected_counts = {
        "dataset_count": 1,
        "field_count": len(fields),
        "patient_count": dataset["patient_count"],
    }
    if data["counts"] != expected_counts:
        raise ValueError("public auxiliary field release counts drifted")
    if data["permissions"] != {
        "field_values_training_authorized": False,
        "report_target_authorized": False,
        "self_supervised_signal_use_authorized": False,
        "outer_fold_exclusion_required": True,
        "knowledge_can_create_patient_fact": False,
        "raw_patient_identifiers_stored": False,
    }:
        raise ValueError("public auxiliary field release permissions drifted")
    if data["missing_closure_codes"] != [
        "field_values_not_materialized",
        "near_or_partial_content_overlap_not_materialized",
        "tuev_eval_patient_identity_opaque",
    ]:
        raise ValueError("public auxiliary field release missing closures drifted")
    expected_id = _ID_PREFIX + canonical_json_sha256(_id_source(data))[:24]
    if data["release_id"] != expected_id:
        raise ValueError("public auxiliary field release ID drifted")
    if data["receipt_sha256"] != canonical_json_sha256(_hash_source(data)):
        raise ValueError("public auxiliary field release receipt drifted")
    return data


__all__ = [
    "PUBLIC_AUXILIARY_FIELD_RELEASE_SCHEMA_VERSION",
    "build_public_auxiliary_field_release",
    "validate_public_auxiliary_field_release",
]
