"""Endpoint-aligned hidden-chunk targets for additive NS-CEC v1.5.

The older BA-IEG counterfactual contract deliberately used entropy and
same-model stability endpoints.  Those are useful engineering proxies, but
they cannot establish that another EEG interval improves an externally
identifiable task.  This additive module defines the stricter target used by
the v1.5 NS-CEC research candidate:

* an interval-valued public boundary loss;
* a patient-level, incomplete-positive onset-ranking loss;
* deterministic native-EEG Finding *opportunity* (not a clinical Finding);
* explicit harm and observed resource-cost vectors; and
* entropy/stability deltas kept in a separately labelled proxy ablation.

Every row is conditional on one visible state and one mutually exclusive
hidden-chunk action.  A downstream checkpoint must be frozen and patient
cross-fitted.  The public boundary reference and incomplete-positive set are
target-only: the predictor receipt rejects feature names that could encode
them, and stores no feature values.  Patient-level channel positives are
applied once after complete patient reassembly; they are never broadcast as
per-event negatives or gold.

This file is a replayable software contract.  It does not read the referenced
artifacts, authorize optimization, certify that caller-supplied target values
were recomputed from raw EEG, authorize a Finding, or authorize a report
claim.  A separate artifact verifier and data-governance gate remain required
before real training.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import re
from typing import Any, Final, Mapping, Sequence

from .ba_ieg_counterfactual_utility_supervision_v1 import (
    counterfactual_interval_roster_sha256_v1,
)


NSCEC_EXTERNAL_ENDPOINT_TARGET_SCHEMA_VERSION_V1: Final[
    str
] = "nscec_external_endpoint_aligned_hidden_chunk_target_v1"
NSCEC_EXTERNAL_ENDPOINT_METHOD_ID_V1: Final[
    str
] = "nscec_patient_cross_fitted_external_endpoint_counterfactual_v1"
NSCEC_PREDICTOR_INPUT_RECEIPT_SCHEMA_VERSION_V1: Final[
    str
] = "nscec_visible_only_action_predictor_input_receipt_v1"
NSCEC_CROSSFIT_CHECKPOINT_BINDING_SCHEMA_VERSION_V1: Final[
    str
] = "nscec_patient_cross_fitted_frozen_downstream_checkpoint_binding_v1"

NSCEC_EXTERNAL_ENDPOINT_DELTA_NAMES_V1: Final[tuple[str, ...]] = (
    "external_boundary_interval_loss_reduction",
    "public_incomplete_positive_onset_rank_loss_reduction",
)
NSCEC_NATIVE_FINDING_OPPORTUNITY_NAMES_V1: Final[tuple[str, ...]] = (
    "frequency_measurement_opportunity",
    "amplitude_measurement_opportunity",
    "morphology_primitive_opportunity",
    "repetition_cycle_opportunity",
    "earliest_field_opportunity",
    "spatial_field_opportunity",
    "evolution_measurement_opportunity",
    "offset_recovery_opportunity",
    "matched_context_opportunity",
)
NSCEC_HARM_NAMES_V1: Final[tuple[str, ...]] = (
    "bad_quality_exposure_fraction",
    "event_mixing_risk",
    "late_spread_leakage_risk",
    "gap_or_censor_risk",
)
NSCEC_COST_NAMES_V1: Final[tuple[str, ...]] = (
    "unique_eeg_seconds",
    "native_samples",
    "model_tokens",
    "gpu_seconds",
    "io_bytes",
    "wall_seconds",
)
NSCEC_PROXY_ABLATION_NAMES_V1: Final[tuple[str, ...]] = (
    "onset_entropy_nats",
    "offset_entropy_nats",
    "earliest_field_stability",
    "onset_rank_stability",
)
NSCEC_BOUNDARY_INTERVAL_LOSS_METRIC_ID_V1: Final[str] = "CENSOR-AWARE-INTERVAL-LOSS-V1"
NSCEC_INCOMPLETE_POSITIVE_RANK_LOSS_METRIC_ID_V1: Final[
    str
] = "NEG-LOG-ANNOTATED-POSITIVE-MASS-V1"

_ACTION_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "action_id",
        "action_type",
        "side",
        "current_event_interval_recording_seconds",
        "visible_intervals_recording_seconds",
        "proposed_intervals_recording_seconds",
        "full_candidate_envelope_recording_seconds",
        "predictor_input_receipt_sha256",
        "hidden_chunk_receipt_sha256",
        "target_independent_candidate_roster_sha256",
        "hidden_chunk_was_masked_from_predictor_input",
    }
)
_ACTION_TYPES: Final[frozenset[str]] = frozenset(
    {"query_left", "query_right", "retrieve_distant_background"}
)
_SPLIT_ROLES: Final[Mapping[str, str]] = {
    "source_train": "target_contract_only_optimization_not_authorized",
    "source_dev": "calibration_or_evaluation_only_no_gradient",
}
_SHA256_ALPHABET: Final[frozenset[str]] = frozenset("0123456789abcdef")
_TOLERANCE: Final[float] = 1e-8

# Boundary posteriors and EEG-derived uncertainty are legal visible-state
# features.  Only names that indicate target/reference access are rejected.
_FORBIDDEN_FEATURE_PATTERNS: Final[tuple[re.Pattern[str], ...]] = tuple(
    re.compile(pattern)
    for pattern in (
        r"(^|_)ground(_|$)",
        r"(^|_)gold(_|$)",
        r"(^|_)gt(_|$)",
        r"(^|_)target(_|$)",
        r"reference_(boundary|interval|onset|offset|channel|unit)",
        r"(boundary|onset|offset)_reference",
        r"positive_(set|label|channel|unit)",
        r"(set|label|channel|unit)_positive",
        r"incomplete_positive",
        r"(boundary|rank)_loss",
        r"external_(boundary|rank|endpoint)",
        r"(doctor|clinician|physician|expert|annotation|spreadsheet|excel)",
        r"significant_(channel|unit)",
        r"(hidden|revealed)_chunk",
        r"counterfactual_(delta|gain|target)",
        r"(soz|onset)_label",
    )
)

_PREDICTOR_FIREWALL: Final[dict[str, bool]] = {
    "visible_eeg_derived_state_only": True,
    "hidden_or_revealed_chunk_used": False,
    "external_boundary_reference_used": False,
    "public_positive_set_or_label_used": False,
    "counterfactual_delta_used": False,
    "edf_annotation_used": False,
    "spreadsheet_or_doctor_text_used": False,
    "clinical_context_or_private_label_used": False,
    "video_sleep_activation_or_other_physiology_used": False,
}

_ROW_AUTHORIZATION: Final[dict[str, object]] = {
    "software_contract_replayable": True,
    "referenced_artifact_bytes_verified_here": False,
    "caller_supplied_target_values_verified_from_raw_eeg_here": False,
    "real_target_materialization_certified": False,
    "router_optimization_authorized": False,
    "model_or_method_promotion_authorized": False,
    "may_authorize_positive_onset_or_soz_evidence": False,
    "may_create_report_eligible_finding": False,
    "may_create_or_strengthen_report_claim": False,
}

_ONE_STEP_SEMANTICS: Final[dict[str, bool]] = {
    "delta_is_conditional_on_exact_visible_parent_state": True,
    "counterfactual_actions_within_context_are_mutually_exclusive": True,
    "may_sum_alternative_action_deltas": False,
    "may_sum_deltas_across_decision_steps": False,
    "next_step_requires_actual_selected_action_reveal_and_full_recompute": True,
}


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _exact_object(value: object, fields: set[str], context: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{context} must be an object")
    missing = fields - set(value)
    extra = set(value) - fields
    if missing or extra:
        raise ValueError(
            f"{context} fields drifted; missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )
    return deepcopy(value)


def _identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TypeError(f"{context} must be a non-empty trimmed identifier")
    if len(value) > 256 or any(character in value for character in ("/", "\\")):
        raise ValueError(f"{context} is not a safe identifier")
    return value


def _sha256(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or set(value).difference(_SHA256_ALPHABET)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _optional_sha256(value: object, context: str) -> str | None:
    if value is None:
        return None
    return _sha256(value, context)


def _finite(
    value: object,
    context: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    if minimum is not None and result < minimum - _TOLERANCE:
        raise ValueError(f"{context} must be >= {minimum}")
    if maximum is not None and result > maximum + _TOLERANCE:
        raise ValueError(f"{context} must be <= {maximum}")
    return result


def _nonnegative_int(value: object, context: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context} must be an integer")
    minimum = 1 if positive else 0
    if value < minimum:
        raise ValueError(f"{context} must be >= {minimum}")
    return value


def _sorted_identifiers(
    value: object,
    context: str,
    *,
    allow_empty: bool,
) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be a list")
    result = [
        _identifier(item, f"{context}[{index}]") for index, item in enumerate(value)
    ]
    if result != sorted(set(result)):
        raise ValueError(f"{context} must be unique and canonically sorted")
    if not allow_empty and not result:
        raise ValueError(f"{context} cannot be empty")
    return result


def _seal(body: Mapping[str, Any], *, id_field: str, prefix: str) -> dict[str, Any]:
    result = deepcopy(dict(body))
    result[id_field] = "CONTENT-ADDRESS-PENDING"
    result[id_field] = f"{prefix}-" + _canonical_sha256(result)[:24]
    return result


def _feature_name(value: object, context: str) -> str:
    name = _identifier(value, context)
    normalized = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    if not normalized:
        raise ValueError(f"{context} has no normalized content")
    for pattern in _FORBIDDEN_FEATURE_PATTERNS:
        if pattern.search(normalized):
            raise ValueError(
                f"{context} encodes target/reference information forbidden from "
                "the NS-CEC action predictor"
            )
    return name


def _build_predictor_input_receipt(
    *,
    feature_schema_id: object,
    feature_names: object,
    feature_values_sha256: object,
    visible_state_sha256: object,
    visible_support_roster_sha256: object,
    hidden_chunk_mask_receipt_sha256: object,
    target_independent_candidate_roster_sha256: object,
    firewall: object,
) -> dict[str, Any]:
    schema_id = _feature_name(feature_schema_id, "feature_schema_id")
    if not isinstance(feature_names, list) or not feature_names:
        raise ValueError("feature_names must be a non-empty list")
    names = [
        _feature_name(name, f"feature_names[{index}]")
        for index, name in enumerate(feature_names)
    ]
    if len(set(names)) != len(names):
        raise ValueError("feature_names must be unique")
    if firewall != _PREDICTOR_FIREWALL:
        raise ValueError("predictor input violates the target/reference firewall")
    proxy_visible = any(
        "entropy" in name.lower() or "stability" in name.lower() for name in names
    )
    body = {
        "schema_version": NSCEC_PREDICTOR_INPUT_RECEIPT_SCHEMA_VERSION_V1,
        "receipt_id": "CONTENT-ADDRESS-PENDING",
        "feature_schema_id": schema_id,
        "feature_names": names,
        "feature_values_sha256": _sha256(
            feature_values_sha256, "feature_values_sha256"
        ),
        "visible_state_sha256": _sha256(visible_state_sha256, "visible_state_sha256"),
        "visible_support_roster_sha256": _sha256(
            visible_support_roster_sha256, "visible_support_roster_sha256"
        ),
        "hidden_chunk_mask_receipt_sha256": _sha256(
            hidden_chunk_mask_receipt_sha256,
            "hidden_chunk_mask_receipt_sha256",
        ),
        "target_independent_candidate_roster_sha256": _sha256(
            target_independent_candidate_roster_sha256,
            "target_independent_candidate_roster_sha256",
        ),
        "entropy_or_stability_visible_state_features_present": proxy_visible,
        "target_reference_or_delta_feature_names_present": False,
        "firewall": deepcopy(_PREDICTOR_FIREWALL),
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    sealed = _seal(body, id_field="receipt_id", prefix="NSCECPRED")
    receipt_source = deepcopy(sealed)
    receipt_source["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    sealed["receipt_sha256"] = _canonical_sha256(receipt_source)
    return sealed


def materialize_nscec_predictor_input_receipt_v1(
    *,
    feature_schema_id: str,
    feature_names: Sequence[str],
    feature_values_sha256: str,
    visible_state_sha256: str,
    visible_support_roster_sha256: str,
    hidden_chunk_mask_receipt_sha256: str,
    target_independent_candidate_roster_sha256: str,
) -> dict[str, Any]:
    """Seal a visible-only predictor descriptor with no target values."""

    return _build_predictor_input_receipt(
        feature_schema_id=feature_schema_id,
        feature_names=list(feature_names),
        feature_values_sha256=feature_values_sha256,
        visible_state_sha256=visible_state_sha256,
        visible_support_roster_sha256=visible_support_roster_sha256,
        hidden_chunk_mask_receipt_sha256=hidden_chunk_mask_receipt_sha256,
        target_independent_candidate_roster_sha256=(
            target_independent_candidate_roster_sha256
        ),
        firewall=deepcopy(_PREDICTOR_FIREWALL),
    )


def validate_nscec_predictor_input_receipt_v1(payload: object) -> dict[str, Any]:
    fields = {
        "schema_version",
        "receipt_id",
        "feature_schema_id",
        "feature_names",
        "feature_values_sha256",
        "visible_state_sha256",
        "visible_support_roster_sha256",
        "hidden_chunk_mask_receipt_sha256",
        "target_independent_candidate_roster_sha256",
        "entropy_or_stability_visible_state_features_present",
        "target_reference_or_delta_feature_names_present",
        "firewall",
        "receipt_sha256",
    }
    data = _exact_object(payload, fields, "predictor input receipt")
    if data["schema_version"] != NSCEC_PREDICTOR_INPUT_RECEIPT_SCHEMA_VERSION_V1:
        raise ValueError("predictor input receipt schema drifted")
    expected = _build_predictor_input_receipt(
        feature_schema_id=data["feature_schema_id"],
        feature_names=data["feature_names"],
        feature_values_sha256=data["feature_values_sha256"],
        visible_state_sha256=data["visible_state_sha256"],
        visible_support_roster_sha256=data["visible_support_roster_sha256"],
        hidden_chunk_mask_receipt_sha256=data["hidden_chunk_mask_receipt_sha256"],
        target_independent_candidate_roster_sha256=data[
            "target_independent_candidate_roster_sha256"
        ],
        firewall=data["firewall"],
    )
    if _canonical_json(data) != _canonical_json(expected):
        raise ValueError("predictor input receipt does not replay")
    return expected


def _build_crossfit_binding(
    *,
    patient_uid: object,
    crossfit_fold_id: object,
    checkpoint_id: object,
    checkpoint_artifact_sha256: object,
    checkpoint_training_run_receipt_sha256: object,
    checkpoint_freeze_receipt_sha256: object,
    preprocessing_artifact_sha256: object,
    source_data_manifest_sha256: object,
    held_out_patient_uids: object,
    training_patient_uids: object,
    preprocessing_fit_patient_uids: object,
    checkpoint_training_patient_roster_complete: object,
    all_patient_exposure_sources_declared: object,
    external_or_unknown_patient_adaptation_used: object,
    checkpoint_frozen_before_counterfactual_sweep: object,
) -> dict[str, Any]:
    patient = _identifier(patient_uid, "patient_uid")
    fold = _identifier(crossfit_fold_id, "crossfit_fold_id")
    held_out = _sorted_identifiers(
        held_out_patient_uids, "held_out_patient_uids", allow_empty=False
    )
    training = _sorted_identifiers(
        training_patient_uids, "training_patient_uids", allow_empty=False
    )
    preprocess = _sorted_identifiers(
        preprocessing_fit_patient_uids,
        "preprocessing_fit_patient_uids",
        allow_empty=True,
    )
    if patient not in held_out:
        raise ValueError(
            "current target patient is not held out by the downstream fold"
        )
    if set(held_out).intersection(training) or set(held_out).intersection(preprocess):
        raise ValueError("held-out patient leaked into checkpoint/preprocessing fit")
    if not set(preprocess).issubset(training):
        raise ValueError(
            "preprocessing-fit patients must be a subset of checkpoint training"
        )
    if checkpoint_training_patient_roster_complete is not True:
        raise ValueError("checkpoint training-patient roster is not declared complete")
    if all_patient_exposure_sources_declared is not True:
        raise ValueError("all downstream patient-exposure sources must be declared")
    if external_or_unknown_patient_adaptation_used is not False:
        raise ValueError("unknown/external patient adaptation breaks cross-fitting")
    if checkpoint_frozen_before_counterfactual_sweep is not True:
        raise ValueError("downstream checkpoint was not frozen before target sweep")
    body = {
        "schema_version": NSCEC_CROSSFIT_CHECKPOINT_BINDING_SCHEMA_VERSION_V1,
        "binding_id": "CONTENT-ADDRESS-PENDING",
        "patient_uid": patient,
        "crossfit_fold_id": fold,
        "checkpoint_id": _identifier(checkpoint_id, "checkpoint_id"),
        "checkpoint_artifact_sha256": _sha256(
            checkpoint_artifact_sha256, "checkpoint_artifact_sha256"
        ),
        "checkpoint_training_run_receipt_sha256": _sha256(
            checkpoint_training_run_receipt_sha256,
            "checkpoint_training_run_receipt_sha256",
        ),
        "checkpoint_freeze_receipt_sha256": _sha256(
            checkpoint_freeze_receipt_sha256,
            "checkpoint_freeze_receipt_sha256",
        ),
        "preprocessing_artifact_sha256": _sha256(
            preprocessing_artifact_sha256, "preprocessing_artifact_sha256"
        ),
        "source_data_manifest_sha256": _sha256(
            source_data_manifest_sha256, "source_data_manifest_sha256"
        ),
        "held_out_patient_uids": held_out,
        "training_patient_uids": training,
        "preprocessing_fit_patient_uids": preprocess,
        "checkpoint_training_patient_roster_complete": True,
        "all_patient_exposure_sources_declared": True,
        "external_or_unknown_patient_adaptation_used": False,
        "checkpoint_frozen_before_counterfactual_sweep": True,
        "current_patient_excluded_from_checkpoint_and_preprocessing_fit": True,
        "binding_sha256": "CONTENT-ADDRESS-PENDING",
    }
    sealed = _seal(body, id_field="binding_id", prefix="NSCECCFOLD")
    receipt_source = deepcopy(sealed)
    receipt_source["binding_sha256"] = "CONTENT-ADDRESS-PENDING"
    sealed["binding_sha256"] = _canonical_sha256(receipt_source)
    return sealed


def materialize_nscec_crossfit_checkpoint_binding_v1(
    *,
    patient_uid: str,
    crossfit_fold_id: str,
    checkpoint_id: str,
    checkpoint_artifact_sha256: str,
    checkpoint_training_run_receipt_sha256: str,
    checkpoint_freeze_receipt_sha256: str,
    preprocessing_artifact_sha256: str,
    source_data_manifest_sha256: str,
    held_out_patient_uids: Sequence[str],
    training_patient_uids: Sequence[str],
    preprocessing_fit_patient_uids: Sequence[str],
    checkpoint_training_patient_roster_complete: bool,
    all_patient_exposure_sources_declared: bool,
    external_or_unknown_patient_adaptation_used: bool,
    checkpoint_frozen_before_counterfactual_sweep: bool,
) -> dict[str, Any]:
    """Bind one target patient to a frozen patient-excluding checkpoint."""

    return _build_crossfit_binding(
        patient_uid=patient_uid,
        crossfit_fold_id=crossfit_fold_id,
        checkpoint_id=checkpoint_id,
        checkpoint_artifact_sha256=checkpoint_artifact_sha256,
        checkpoint_training_run_receipt_sha256=(checkpoint_training_run_receipt_sha256),
        checkpoint_freeze_receipt_sha256=checkpoint_freeze_receipt_sha256,
        preprocessing_artifact_sha256=preprocessing_artifact_sha256,
        source_data_manifest_sha256=source_data_manifest_sha256,
        held_out_patient_uids=list(held_out_patient_uids),
        training_patient_uids=list(training_patient_uids),
        preprocessing_fit_patient_uids=list(preprocessing_fit_patient_uids),
        checkpoint_training_patient_roster_complete=(
            checkpoint_training_patient_roster_complete
        ),
        all_patient_exposure_sources_declared=all_patient_exposure_sources_declared,
        external_or_unknown_patient_adaptation_used=(
            external_or_unknown_patient_adaptation_used
        ),
        checkpoint_frozen_before_counterfactual_sweep=(
            checkpoint_frozen_before_counterfactual_sweep
        ),
    )


def validate_nscec_crossfit_checkpoint_binding_v1(payload: object) -> dict[str, Any]:
    fields = {
        "schema_version",
        "binding_id",
        "patient_uid",
        "crossfit_fold_id",
        "checkpoint_id",
        "checkpoint_artifact_sha256",
        "checkpoint_training_run_receipt_sha256",
        "checkpoint_freeze_receipt_sha256",
        "preprocessing_artifact_sha256",
        "source_data_manifest_sha256",
        "held_out_patient_uids",
        "training_patient_uids",
        "preprocessing_fit_patient_uids",
        "checkpoint_training_patient_roster_complete",
        "all_patient_exposure_sources_declared",
        "external_or_unknown_patient_adaptation_used",
        "checkpoint_frozen_before_counterfactual_sweep",
        "current_patient_excluded_from_checkpoint_and_preprocessing_fit",
        "binding_sha256",
    }
    data = _exact_object(payload, fields, "crossfit checkpoint binding")
    if data["schema_version"] != NSCEC_CROSSFIT_CHECKPOINT_BINDING_SCHEMA_VERSION_V1:
        raise ValueError("crossfit checkpoint binding schema drifted")
    expected = _build_crossfit_binding(
        patient_uid=data["patient_uid"],
        crossfit_fold_id=data["crossfit_fold_id"],
        checkpoint_id=data["checkpoint_id"],
        checkpoint_artifact_sha256=data["checkpoint_artifact_sha256"],
        checkpoint_training_run_receipt_sha256=data[
            "checkpoint_training_run_receipt_sha256"
        ],
        checkpoint_freeze_receipt_sha256=data["checkpoint_freeze_receipt_sha256"],
        preprocessing_artifact_sha256=data["preprocessing_artifact_sha256"],
        source_data_manifest_sha256=data["source_data_manifest_sha256"],
        held_out_patient_uids=data["held_out_patient_uids"],
        training_patient_uids=data["training_patient_uids"],
        preprocessing_fit_patient_uids=data["preprocessing_fit_patient_uids"],
        checkpoint_training_patient_roster_complete=data[
            "checkpoint_training_patient_roster_complete"
        ],
        all_patient_exposure_sources_declared=data[
            "all_patient_exposure_sources_declared"
        ],
        external_or_unknown_patient_adaptation_used=data[
            "external_or_unknown_patient_adaptation_used"
        ],
        checkpoint_frozen_before_counterfactual_sweep=data[
            "checkpoint_frozen_before_counterfactual_sweep"
        ],
    )
    if _canonical_json(data) != _canonical_json(expected):
        raise ValueError("crossfit checkpoint binding does not replay")
    return expected


def _validate_action(value: object) -> dict[str, Any]:
    action = _exact_object(value, set(_ACTION_FIELDS), "hidden-chunk action")
    if action["action_type"] not in _ACTION_TYPES:
        raise ValueError("NS-CEC external target only supports acquisition actions")
    # Reuse the older, already adversarially tested physical geometry validator.
    counterfactual_interval_roster_sha256_v1(action, revealed=False)
    counterfactual_interval_roster_sha256_v1(action, revealed=True)
    return action


def _status_loss_endpoint(
    value: object,
    context: str,
    *,
    kind: str,
) -> dict[str, Any]:
    if kind == "boundary":
        fields = {
            "status",
            "loss_value",
            "loss_unit",
            "metric_id",
            "metric_definition_sha256",
            "public_reference_receipt_sha256",
            "prediction_receipt_sha256",
            "reference_split",
            "reference_is_interval_valued",
            "censoring_supported",
            "reference_used_as_predictor_feature",
        }
    else:
        fields = {
            "status",
            "loss_value",
            "loss_unit",
            "metric_id",
            "metric_definition_sha256",
            "public_positive_set_receipt_sha256",
            "patient_context_roster_sha256",
            "patient_aggregate_prediction_receipt_sha256",
            "reference_split",
            "positive_set_scope",
            "patient_complete_reassembly",
            "positive_set_applied_once",
            "per_event_label_broadcast_used",
            "unlabeled_typed_units_treated_as_negative",
            "spread_labels_used",
            "reference_used_as_predictor_feature",
        }
    data = _exact_object(value, fields, context)
    status = data["status"]
    if status not in {"evaluable", "not_evaluable"}:
        raise ValueError(f"{context}.status must be evaluable or not_evaluable")
    loss = data["loss_value"]
    if status == "evaluable":
        loss = _finite(loss, f"{context}.loss_value", minimum=0.0)
    elif loss is not None:
        raise ValueError(f"{context} not_evaluable endpoint cannot carry a loss")
    unit = data["loss_unit"]
    if unit not in {"seconds", "dimensionless"}:
        raise ValueError(f"{context}.loss_unit is unsupported")
    result = deepcopy(data)
    result["loss_value"] = loss
    result["metric_id"] = _identifier(data["metric_id"], f"{context}.metric_id")
    result["metric_definition_sha256"] = _sha256(
        data["metric_definition_sha256"], f"{context}.metric_definition_sha256"
    )
    result["reference_split"] = _identifier(
        data["reference_split"], f"{context}.reference_split"
    )
    if data["reference_used_as_predictor_feature"] is not False:
        raise ValueError(f"{context} leaked a target reference into predictor features")
    if kind == "boundary":
        if result["metric_id"] != NSCEC_BOUNDARY_INTERVAL_LOSS_METRIC_ID_V1:
            raise ValueError(f"{context} does not use the frozen interval-loss metric")
        if result["loss_unit"] != "seconds":
            raise ValueError(f"{context} boundary interval loss must use seconds")
        result["public_reference_receipt_sha256"] = _optional_sha256(
            data["public_reference_receipt_sha256"],
            f"{context}.public_reference_receipt_sha256",
        )
        result["prediction_receipt_sha256"] = _optional_sha256(
            data["prediction_receipt_sha256"],
            f"{context}.prediction_receipt_sha256",
        )
        if status == "evaluable" and (
            result["public_reference_receipt_sha256"] is None
            or result["prediction_receipt_sha256"] is None
        ):
            raise ValueError(f"{context} evaluable boundary endpoint lacks receipts")
        if data["reference_is_interval_valued"] is not True:
            raise ValueError(
                f"{context} must use an interval-valued boundary reference"
            )
        if data["censoring_supported"] is not True:
            raise ValueError(f"{context} must preserve boundary censoring")
    else:
        if result["metric_id"] != NSCEC_INCOMPLETE_POSITIVE_RANK_LOSS_METRIC_ID_V1:
            raise ValueError(
                f"{context} does not use the frozen incomplete-positive rank loss"
            )
        if result["loss_unit"] != "dimensionless":
            raise ValueError(
                f"{context} incomplete-positive rank loss is dimensionless"
            )
        result["public_positive_set_receipt_sha256"] = _optional_sha256(
            data["public_positive_set_receipt_sha256"],
            f"{context}.public_positive_set_receipt_sha256",
        )
        result["patient_context_roster_sha256"] = _optional_sha256(
            data["patient_context_roster_sha256"],
            f"{context}.patient_context_roster_sha256",
        )
        result["patient_aggregate_prediction_receipt_sha256"] = _optional_sha256(
            data["patient_aggregate_prediction_receipt_sha256"],
            f"{context}.patient_aggregate_prediction_receipt_sha256",
        )
        if status == "evaluable" and any(
            result[name] is None
            for name in (
                "public_positive_set_receipt_sha256",
                "patient_context_roster_sha256",
                "patient_aggregate_prediction_receipt_sha256",
            )
        ):
            raise ValueError(f"{context} evaluable rank endpoint lacks receipts")
        expected_flags = {
            "positive_set_scope": "patient_level_incomplete_positive",
            "patient_complete_reassembly": True,
            "positive_set_applied_once": True,
            "per_event_label_broadcast_used": False,
            "unlabeled_typed_units_treated_as_negative": False,
            "spread_labels_used": False,
        }
        for name, expected in expected_flags.items():
            if data[name] != expected:
                raise ValueError(
                    f"{context}.{name} violates incomplete-positive patient-level semantics"
                )
    return result


def _metric_vector(
    value: object,
    names: Sequence[str],
    context: str,
    *,
    maximum: float | None,
) -> dict[str, float | None]:
    data = _exact_object(value, set(names), context)
    result: dict[str, float | None] = {}
    for name in names:
        raw = data[name]
        if raw is None:
            result[name] = None
        else:
            result[name] = _finite(
                raw,
                f"{context}.{name}",
                minimum=0.0,
                maximum=maximum,
            )
    return result


def _validate_snapshot(value: object, context: str) -> dict[str, Any]:
    fields = {
        "downstream_checkpoint_binding_sha256",
        "input_evidence_union_sha256",
        "evidence_interval_roster_sha256",
        "endpoint_recompute_receipt_sha256",
        "external_boundary_endpoint",
        "public_incomplete_positive_rank_endpoint",
        "native_finding_opportunity",
        "native_finding_opportunity_roster_sha256",
        "native_finding_remeasurement_receipt_sha256",
        "native_finding_target_or_clinical_term_used",
        "harm_metrics",
        "harm_metric_definition_sha256",
        "harm_evaluator_receipt_sha256",
        "proxy_ablation_metrics",
        "proxy_ablation_metric_definition_sha256",
        "proxy_ablation_receipt_sha256",
    }
    data = _exact_object(value, fields, context)
    result = {
        "downstream_checkpoint_binding_sha256": _sha256(
            data["downstream_checkpoint_binding_sha256"],
            f"{context}.downstream_checkpoint_binding_sha256",
        ),
        "input_evidence_union_sha256": _sha256(
            data["input_evidence_union_sha256"],
            f"{context}.input_evidence_union_sha256",
        ),
        "evidence_interval_roster_sha256": _sha256(
            data["evidence_interval_roster_sha256"],
            f"{context}.evidence_interval_roster_sha256",
        ),
        "endpoint_recompute_receipt_sha256": _sha256(
            data["endpoint_recompute_receipt_sha256"],
            f"{context}.endpoint_recompute_receipt_sha256",
        ),
        "external_boundary_endpoint": _status_loss_endpoint(
            data["external_boundary_endpoint"],
            f"{context}.external_boundary_endpoint",
            kind="boundary",
        ),
        "public_incomplete_positive_rank_endpoint": _status_loss_endpoint(
            data["public_incomplete_positive_rank_endpoint"],
            f"{context}.public_incomplete_positive_rank_endpoint",
            kind="rank",
        ),
        "native_finding_opportunity": _metric_vector(
            data["native_finding_opportunity"],
            NSCEC_NATIVE_FINDING_OPPORTUNITY_NAMES_V1,
            f"{context}.native_finding_opportunity",
            maximum=1.0,
        ),
        "native_finding_opportunity_roster_sha256": _sha256(
            data["native_finding_opportunity_roster_sha256"],
            f"{context}.native_finding_opportunity_roster_sha256",
        ),
        "native_finding_remeasurement_receipt_sha256": _sha256(
            data["native_finding_remeasurement_receipt_sha256"],
            f"{context}.native_finding_remeasurement_receipt_sha256",
        ),
        "native_finding_target_or_clinical_term_used": data[
            "native_finding_target_or_clinical_term_used"
        ],
        "harm_metrics": _metric_vector(
            data["harm_metrics"],
            NSCEC_HARM_NAMES_V1,
            f"{context}.harm_metrics",
            maximum=1.0,
        ),
        "harm_metric_definition_sha256": _sha256(
            data["harm_metric_definition_sha256"],
            f"{context}.harm_metric_definition_sha256",
        ),
        "harm_evaluator_receipt_sha256": _sha256(
            data["harm_evaluator_receipt_sha256"],
            f"{context}.harm_evaluator_receipt_sha256",
        ),
        "proxy_ablation_metrics": _metric_vector(
            data["proxy_ablation_metrics"],
            NSCEC_PROXY_ABLATION_NAMES_V1,
            f"{context}.proxy_ablation_metrics",
            maximum=None,
        ),
        "proxy_ablation_metric_definition_sha256": _sha256(
            data["proxy_ablation_metric_definition_sha256"],
            f"{context}.proxy_ablation_metric_definition_sha256",
        ),
        "proxy_ablation_receipt_sha256": _sha256(
            data["proxy_ablation_receipt_sha256"],
            f"{context}.proxy_ablation_receipt_sha256",
        ),
    }
    if result["native_finding_target_or_clinical_term_used"] is not False:
        raise ValueError(
            f"{context} native opportunity must not use a target or clinical term"
        )
    for name in ("earliest_field_stability", "onset_rank_stability"):
        value_number = result["proxy_ablation_metrics"][name]
        if value_number is not None and value_number > 1.0 + _TOLERANCE:
            raise ValueError(
                f"{context}.proxy_ablation_metrics.{name} must be in [0,1]"
            )
    if all(item is None for item in result["native_finding_opportunity"].values()):
        raise ValueError(f"{context} has no native Finding opportunity denominator")
    return result


def _validate_cost(value: object, *, physical_seconds: float) -> dict[str, Any]:
    fields = set(NSCEC_COST_NAMES_V1) | {
        "cost_measurement_receipt_sha256",
        "cost_values_are_observed_not_predicted",
    }
    data = _exact_object(value, fields, "observed action cost")
    result: dict[str, Any] = {
        "unique_eeg_seconds": _finite(
            data["unique_eeg_seconds"], "cost.unique_eeg_seconds", minimum=0.0
        ),
        "native_samples": _nonnegative_int(
            data["native_samples"], "cost.native_samples", positive=True
        ),
        "model_tokens": _nonnegative_int(
            data["model_tokens"], "cost.model_tokens", positive=True
        ),
        "gpu_seconds": _finite(data["gpu_seconds"], "cost.gpu_seconds", minimum=0.0),
        "io_bytes": _nonnegative_int(data["io_bytes"], "cost.io_bytes", positive=True),
        "wall_seconds": _finite(data["wall_seconds"], "cost.wall_seconds", minimum=0.0),
        "cost_measurement_receipt_sha256": _sha256(
            data["cost_measurement_receipt_sha256"],
            "cost.cost_measurement_receipt_sha256",
        ),
        "cost_values_are_observed_not_predicted": data[
            "cost_values_are_observed_not_predicted"
        ],
    }
    if abs(result["unique_eeg_seconds"] - physical_seconds) > _TOLERANCE:
        raise ValueError("observed unique EEG seconds do not match revealed intervals")
    if result["cost_values_are_observed_not_predicted"] is not True:
        raise ValueError("counterfactual target costs must be observed, not predicted")
    return result


def _conditional_context(
    value: object,
    *,
    patient_uid: str,
    recording_id: str,
    event_id: str,
) -> dict[str, Any]:
    fields = {
        "decision_index",
        "parent_state_receipt_sha256",
        "previous_selected_action_target_id",
        "state_materialized_after_actual_previous_query",
        "one_step_semantics",
    }
    data = _exact_object(value, fields, "counterfactual context")
    index = _nonnegative_int(data["decision_index"], "decision_index")
    previous = data["previous_selected_action_target_id"]
    if index == 0:
        if previous is not None:
            raise ValueError(
                "first counterfactual context cannot name a previous target"
            )
    else:
        previous = _identifier(previous, "previous_selected_action_target_id")
    if data["state_materialized_after_actual_previous_query"] is not True:
        raise ValueError(
            "counterfactual rows require an actually materialized parent state"
        )
    if data["one_step_semantics"] != _ONE_STEP_SEMANTICS:
        raise ValueError("counterfactual one-step/non-additivity semantics drifted")
    parent = _sha256(data["parent_state_receipt_sha256"], "parent_state_receipt_sha256")
    context_id = (
        "NSCECCTX-"
        + _canonical_sha256(
            {
                "patient_uid": patient_uid,
                "recording_id": recording_id,
                "event_id": event_id,
                "decision_index": index,
                "parent_state_receipt_sha256": parent,
                "previous_selected_action_target_id": previous,
            }
        )[:24]
    )
    return {
        "context_id": context_id,
        "decision_index": index,
        "parent_state_receipt_sha256": parent,
        "previous_selected_action_target_id": previous,
        "state_materialized_after_actual_previous_query": True,
        "one_step_semantics": deepcopy(_ONE_STEP_SEMANTICS),
    }


def _paired_optional_delta(
    before: float | None,
    after: float | None,
    *,
    direction: str,
    context: str,
) -> tuple[float | None, bool]:
    if (before is None) != (after is None):
        raise ValueError(f"{context} evaluability changed across the hidden action")
    if before is None:
        return None, False
    assert after is not None
    delta = before - after if direction == "reduction" else after - before
    if abs(delta) <= _TOLERANCE:
        delta = 0.0
    return float(delta), True


def _vector_delta(
    base: Mapping[str, float | None],
    revealed: Mapping[str, float | None],
    names: Sequence[str],
    *,
    direction: str,
    context: str,
) -> tuple[dict[str, float | None], dict[str, bool]]:
    values: dict[str, float | None] = {}
    masks: dict[str, bool] = {}
    for name in names:
        values[name], masks[name] = _paired_optional_delta(
            base[name],
            revealed[name],
            direction=direction,
            context=f"{context}.{name}",
        )
    return values, masks


def _same_endpoint_definition(
    base: Mapping[str, Any],
    revealed: Mapping[str, Any],
    *,
    kind: str,
) -> None:
    common = {
        "status",
        "loss_unit",
        "metric_id",
        "metric_definition_sha256",
        "reference_split",
        "reference_used_as_predictor_feature",
    }
    if kind == "boundary":
        common |= {
            "public_reference_receipt_sha256",
            "reference_is_interval_valued",
            "censoring_supported",
        }
    else:
        common |= {
            "public_positive_set_receipt_sha256",
            "patient_context_roster_sha256",
            "positive_set_scope",
            "patient_complete_reassembly",
            "positive_set_applied_once",
            "per_event_label_broadcast_used",
            "unlabeled_typed_units_treated_as_negative",
            "spread_labels_used",
        }
    for name in common:
        if base[name] != revealed[name]:
            raise ValueError(
                f"{kind} endpoint definition/reference changed across action"
            )


def _build_target(
    *,
    patient_uid: object,
    recording_id: object,
    event_id: object,
    model_split: object,
    source_data_manifest_sha256: object,
    predictor_input_receipt: object,
    downstream_crossfit_checkpoint_binding: object,
    counterfactual_context: object,
    action: object,
    base_snapshot: object,
    revealed_snapshot: object,
    observed_cost: object,
) -> dict[str, Any]:
    patient = _identifier(patient_uid, "patient_uid")
    recording = _identifier(recording_id, "recording_id")
    event = _identifier(event_id, "event_id")
    if model_split not in _SPLIT_ROLES:
        raise ValueError(
            "NS-CEC external targets are restricted to public source splits"
        )
    split = str(model_split)
    source_manifest = _sha256(
        source_data_manifest_sha256, "source_data_manifest_sha256"
    )
    predictor = validate_nscec_predictor_input_receipt_v1(predictor_input_receipt)
    checkpoint = validate_nscec_crossfit_checkpoint_binding_v1(
        downstream_crossfit_checkpoint_binding
    )
    if checkpoint["patient_uid"] != patient:
        raise ValueError("crossfit checkpoint binding belongs to another patient")
    if checkpoint["source_data_manifest_sha256"] != source_manifest:
        raise ValueError("checkpoint binding and target use different source manifests")
    context = _conditional_context(
        counterfactual_context,
        patient_uid=patient,
        recording_id=recording,
        event_id=event,
    )
    validated_action = _validate_action(action)
    expected_base_roster = counterfactual_interval_roster_sha256_v1(
        validated_action, revealed=False
    )
    expected_revealed_roster = counterfactual_interval_roster_sha256_v1(
        validated_action, revealed=True
    )
    if (
        predictor["receipt_sha256"]
        != validated_action["predictor_input_receipt_sha256"]
    ):
        raise ValueError(
            "action does not bind the supplied visible-only predictor receipt"
        )
    if predictor["visible_support_roster_sha256"] != expected_base_roster:
        raise ValueError(
            "predictor receipt does not bind the exact visible support roster"
        )
    if (
        predictor["target_independent_candidate_roster_sha256"]
        != validated_action["target_independent_candidate_roster_sha256"]
    ):
        raise ValueError("predictor/action candidate rosters disagree")
    if (
        predictor["hidden_chunk_mask_receipt_sha256"]
        == validated_action["hidden_chunk_receipt_sha256"]
    ):
        raise ValueError(
            "hidden chunk receipt cannot be the visible predictor mask receipt"
        )

    base = _validate_snapshot(base_snapshot, "base_snapshot")
    revealed = _validate_snapshot(revealed_snapshot, "revealed_snapshot")
    checkpoint_receipt = checkpoint["binding_sha256"]
    if (
        base["downstream_checkpoint_binding_sha256"] != checkpoint_receipt
        or revealed["downstream_checkpoint_binding_sha256"] != checkpoint_receipt
    ):
        raise ValueError("snapshots do not use the bound frozen crossfit checkpoint")
    if base["evidence_interval_roster_sha256"] != expected_base_roster:
        raise ValueError("base snapshot does not bind exact visible support")
    if revealed["evidence_interval_roster_sha256"] != expected_revealed_roster:
        raise ValueError(
            "revealed snapshot does not bind exact visible-plus-action support"
        )
    if base["input_evidence_union_sha256"] == revealed["input_evidence_union_sha256"]:
        raise ValueError("hidden action did not change the downstream evidence union")
    if (
        base["endpoint_recompute_receipt_sha256"]
        == revealed["endpoint_recompute_receipt_sha256"]
    ):
        raise ValueError("base/revealed endpoints were not independently recomputed")

    boundary_base = base["external_boundary_endpoint"]
    boundary_revealed = revealed["external_boundary_endpoint"]
    rank_base = base["public_incomplete_positive_rank_endpoint"]
    rank_revealed = revealed["public_incomplete_positive_rank_endpoint"]
    _same_endpoint_definition(boundary_base, boundary_revealed, kind="boundary")
    _same_endpoint_definition(rank_base, rank_revealed, kind="rank")
    for endpoint in (boundary_base, rank_base):
        if endpoint["reference_split"] != split:
            raise ValueError("external target reference split differs from model split")
    if (
        rank_base["status"] == "evaluable"
        and rank_base["patient_aggregate_prediction_receipt_sha256"]
        == rank_revealed["patient_aggregate_prediction_receipt_sha256"]
    ):
        raise ValueError(
            "patient-level rank endpoint was not recomputed after the action"
        )
    if (
        boundary_base["status"] == "evaluable"
        and boundary_base["prediction_receipt_sha256"]
        == boundary_revealed["prediction_receipt_sha256"]
    ):
        raise ValueError(
            "boundary prediction endpoint was not recomputed after the action"
        )

    boundary_delta, boundary_evaluable = _paired_optional_delta(
        boundary_base["loss_value"],
        boundary_revealed["loss_value"],
        direction="reduction",
        context="external boundary interval loss",
    )
    rank_delta, rank_evaluable = _paired_optional_delta(
        rank_base["loss_value"],
        rank_revealed["loss_value"],
        direction="reduction",
        context="public incomplete-positive rank loss",
    )
    if not boundary_evaluable and not rank_evaluable:
        raise ValueError("target row has no evaluable external endpoint")

    if (
        base["native_finding_opportunity_roster_sha256"]
        != revealed["native_finding_opportunity_roster_sha256"]
    ):
        raise ValueError("native Finding opportunity denominator changed across action")
    opportunity_delta, opportunity_mask = _vector_delta(
        base["native_finding_opportunity"],
        revealed["native_finding_opportunity"],
        NSCEC_NATIVE_FINDING_OPPORTUNITY_NAMES_V1,
        direction="increase",
        context="native Finding opportunity",
    )
    harm_raw_delta, harm_mask = _vector_delta(
        base["harm_metrics"],
        revealed["harm_metrics"],
        NSCEC_HARM_NAMES_V1,
        direction="increase",
        context="harm",
    )
    if (
        base["harm_metric_definition_sha256"]
        != revealed["harm_metric_definition_sha256"]
    ):
        raise ValueError("harm metric definition changed across the hidden action")
    harm_increase = {
        name: None if value is None else max(0.0, value)
        for name, value in harm_raw_delta.items()
    }

    proxy_delta: dict[str, float | None] = {}
    proxy_mask: dict[str, bool] = {}
    if (
        base["proxy_ablation_metric_definition_sha256"]
        != revealed["proxy_ablation_metric_definition_sha256"]
    ):
        raise ValueError("proxy ablation metric definition changed across the action")
    for name in NSCEC_PROXY_ABLATION_NAMES_V1:
        direction = "reduction" if name.endswith("entropy_nats") else "increase"
        proxy_delta[name], proxy_mask[name] = _paired_optional_delta(
            base["proxy_ablation_metrics"][name],
            revealed["proxy_ablation_metrics"][name],
            direction=direction,
            context=f"proxy ablation.{name}",
        )

    physical_seconds = sum(
        float(stop) - float(start)
        for start, stop in validated_action["proposed_intervals_recording_seconds"]
    )
    cost = _validate_cost(observed_cost, physical_seconds=physical_seconds)
    body: dict[str, Any] = {
        "schema_version": NSCEC_EXTERNAL_ENDPOINT_TARGET_SCHEMA_VERSION_V1,
        "target_id": "CONTENT-ADDRESS-PENDING",
        "method_id": NSCEC_EXTERNAL_ENDPOINT_METHOD_ID_V1,
        "patient_uid": patient,
        "recording_id": recording,
        "event_id": event,
        "model_split": split,
        "optimization_role": _SPLIT_ROLES[split],
        "source_data_manifest_sha256": source_manifest,
        "predictor_input_receipt": predictor,
        "downstream_crossfit_checkpoint_binding": checkpoint,
        "counterfactual_context": context,
        "action": validated_action,
        "base_snapshot": base,
        "revealed_snapshot": revealed,
        "primary_external_endpoint_delta": {
            NSCEC_EXTERNAL_ENDPOINT_DELTA_NAMES_V1[0]: boundary_delta,
            NSCEC_EXTERNAL_ENDPOINT_DELTA_NAMES_V1[1]: rank_delta,
        },
        "primary_external_endpoint_evaluable": {
            NSCEC_EXTERNAL_ENDPOINT_DELTA_NAMES_V1[0]: boundary_evaluable,
            NSCEC_EXTERNAL_ENDPOINT_DELTA_NAMES_V1[1]: rank_evaluable,
        },
        "native_finding_opportunity_gain": opportunity_delta,
        "native_finding_opportunity_evaluable": opportunity_mask,
        "harm_raw_signed_increase": harm_raw_delta,
        "harm_nonnegative_increase_target": harm_increase,
        "harm_evaluable": harm_mask,
        "observed_cost_vector": cost,
        "proxy_entropy_stability_ablation": {
            "role": "proxy_ablation_only_not_primary_utility",
            "included_in_primary_external_endpoint_target": False,
            "may_authorize_nscec_promotion": False,
            "signed_delta": proxy_delta,
            "evaluable": proxy_mask,
        },
        "one_step_nonadditivity": deepcopy(_ONE_STEP_SEMANTICS),
        "authorization": deepcopy(_ROW_AUTHORIZATION),
    }
    body["target_id"] = "NSCECTGT-" + _canonical_sha256(body)[:24]
    return body


def materialize_nscec_external_endpoint_counterfactual_target_v1(
    *,
    patient_uid: str,
    recording_id: str,
    event_id: str,
    model_split: str,
    source_data_manifest_sha256: str,
    predictor_input_receipt: Mapping[str, Any],
    downstream_crossfit_checkpoint_binding: Mapping[str, Any],
    counterfactual_context: Mapping[str, Any],
    action: Mapping[str, Any],
    base_snapshot: Mapping[str, Any],
    revealed_snapshot: Mapping[str, Any],
    observed_cost: Mapping[str, Any],
) -> dict[str, Any]:
    """Materialize one one-step, endpoint-aligned hidden-action row."""

    return _build_target(
        patient_uid=patient_uid,
        recording_id=recording_id,
        event_id=event_id,
        model_split=model_split,
        source_data_manifest_sha256=source_data_manifest_sha256,
        predictor_input_receipt=predictor_input_receipt,
        downstream_crossfit_checkpoint_binding=(downstream_crossfit_checkpoint_binding),
        counterfactual_context=counterfactual_context,
        action=action,
        base_snapshot=base_snapshot,
        revealed_snapshot=revealed_snapshot,
        observed_cost=observed_cost,
    )


def validate_nscec_external_endpoint_counterfactual_target_v1(
    payload: object,
) -> dict[str, Any]:
    """Validate a serialized row by exact deterministic replay."""

    fields = {
        "schema_version",
        "target_id",
        "method_id",
        "patient_uid",
        "recording_id",
        "event_id",
        "model_split",
        "optimization_role",
        "source_data_manifest_sha256",
        "predictor_input_receipt",
        "downstream_crossfit_checkpoint_binding",
        "counterfactual_context",
        "action",
        "base_snapshot",
        "revealed_snapshot",
        "primary_external_endpoint_delta",
        "primary_external_endpoint_evaluable",
        "native_finding_opportunity_gain",
        "native_finding_opportunity_evaluable",
        "harm_raw_signed_increase",
        "harm_nonnegative_increase_target",
        "harm_evaluable",
        "observed_cost_vector",
        "proxy_entropy_stability_ablation",
        "one_step_nonadditivity",
        "authorization",
    }
    data = _exact_object(payload, fields, "NS-CEC external endpoint target")
    if data["schema_version"] != NSCEC_EXTERNAL_ENDPOINT_TARGET_SCHEMA_VERSION_V1:
        raise ValueError("NS-CEC external endpoint target schema drifted")
    if data["method_id"] != NSCEC_EXTERNAL_ENDPOINT_METHOD_ID_V1:
        raise ValueError("NS-CEC external endpoint target method drifted")
    expected = _build_target(
        patient_uid=data["patient_uid"],
        recording_id=data["recording_id"],
        event_id=data["event_id"],
        model_split=data["model_split"],
        source_data_manifest_sha256=data["source_data_manifest_sha256"],
        predictor_input_receipt=data["predictor_input_receipt"],
        downstream_crossfit_checkpoint_binding=data[
            "downstream_crossfit_checkpoint_binding"
        ],
        counterfactual_context={
            name: data["counterfactual_context"][name]
            for name in (
                "decision_index",
                "parent_state_receipt_sha256",
                "previous_selected_action_target_id",
                "state_materialized_after_actual_previous_query",
                "one_step_semantics",
            )
        },
        action=data["action"],
        base_snapshot=data["base_snapshot"],
        revealed_snapshot=data["revealed_snapshot"],
        observed_cost=data["observed_cost_vector"],
    )
    if _canonical_json(data) != _canonical_json(expected):
        raise ValueError("NS-CEC external endpoint target does not replay")
    return expected


def validate_nscec_mutually_exclusive_action_rows_v1(
    payloads: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate alternatives for one state without manufacturing a return.

    The returned object intentionally contains row IDs and no summed endpoint
    delta.  A trajectory may select exactly one row, reveal it, and recompute
    the next state; it may not add alternative or later one-step deltas.
    """

    if isinstance(payloads, (str, bytes)) or not payloads:
        raise ValueError("counterfactual action rows must be non-empty")
    rows = [
        validate_nscec_external_endpoint_counterfactual_target_v1(row)
        for row in payloads
    ]
    context_ids = {row["counterfactual_context"]["context_id"] for row in rows}
    if len(context_ids) != 1:
        raise ValueError(
            "mutually exclusive action rows must share one visible context"
        )
    action_ids = [str(row["action"]["action_id"]) for row in rows]
    target_ids = [str(row["target_id"]) for row in rows]
    if len(set(action_ids)) != len(action_ids) or len(set(target_ids)) != len(
        target_ids
    ):
        raise ValueError("counterfactual action rows must be unique")
    base_hashes = {_canonical_sha256(row["base_snapshot"]) for row in rows}
    checkpoint_hashes = {
        row["downstream_crossfit_checkpoint_binding"]["binding_sha256"] for row in rows
    }
    candidate_rosters = {
        row["action"]["target_independent_candidate_roster_sha256"] for row in rows
    }
    if len(base_hashes) != 1 or len(checkpoint_hashes) != 1:
        raise ValueError("alternative actions do not share one frozen base endpoint")
    if len(candidate_rosters) != 1:
        raise ValueError("alternative actions were not drawn from one frozen roster")
    descriptor = {
        "schema_version": "nscec_mutually_exclusive_hidden_action_set_v1",
        "context_id": next(iter(context_ids)),
        "target_ids": sorted(target_ids),
        "action_ids": sorted(action_ids),
        "one_step_nonadditivity": deepcopy(_ONE_STEP_SEMANTICS),
        "summed_endpoint_delta": None,
        "summed_endpoint_delta_permitted": False,
    }
    descriptor["action_set_sha256"] = _canonical_sha256(descriptor)
    return descriptor


__all__ = [
    "NSCEC_BOUNDARY_INTERVAL_LOSS_METRIC_ID_V1",
    "NSCEC_COST_NAMES_V1",
    "NSCEC_CROSSFIT_CHECKPOINT_BINDING_SCHEMA_VERSION_V1",
    "NSCEC_EXTERNAL_ENDPOINT_DELTA_NAMES_V1",
    "NSCEC_EXTERNAL_ENDPOINT_METHOD_ID_V1",
    "NSCEC_EXTERNAL_ENDPOINT_TARGET_SCHEMA_VERSION_V1",
    "NSCEC_HARM_NAMES_V1",
    "NSCEC_INCOMPLETE_POSITIVE_RANK_LOSS_METRIC_ID_V1",
    "NSCEC_NATIVE_FINDING_OPPORTUNITY_NAMES_V1",
    "NSCEC_PREDICTOR_INPUT_RECEIPT_SCHEMA_VERSION_V1",
    "NSCEC_PROXY_ABLATION_NAMES_V1",
    "materialize_nscec_crossfit_checkpoint_binding_v1",
    "materialize_nscec_external_endpoint_counterfactual_target_v1",
    "materialize_nscec_predictor_input_receipt_v1",
    "validate_nscec_crossfit_checkpoint_binding_v1",
    "validate_nscec_external_endpoint_counterfactual_target_v1",
    "validate_nscec_mutually_exclusive_action_rows_v1",
    "validate_nscec_predictor_input_receipt_v1",
]
