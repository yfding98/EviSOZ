"""Additive patient-OOF checkpoint-training exposure binding for G0 A1.

The frozen A1 prediction-first roster proves that each prediction declares the
patient-held-out fold set, but it does not name the checkpoint that served each
fold and it does not expose that checkpoint's training-patient roster.  The A1
roster therefore cannot, by itself, prove patient-OOF checkpoint training.

This module leaves the frozen roster untouched and adds three fail-closed
contracts:

* a fold-specific manifest binding content-addressed checkpoint and
  preprocessing artifacts to the complete inventory-derived training-patient
  roster;
* a supplemental per-record/per-fold provider usage binding proving which
  fold exposure served (or was assigned to) every prediction denominator; and
* an explicit pending receipt for existing prediction rosters that lack those
  supplemental provider receipts.

All inputs are prediction/inventory lineage.  There is no parameter for public
event intervals, channel/SOZ targets, EDF annotations, spreadsheets, clinical
text, or raw EEG values.  These contracts do not authorize training, G0
promotion, accuracy claims, or clinical use.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any, Final, Mapping, Sequence

from .ba_ieg_g0_a1_candidate_roster_v1 import (
    validate_ba_ieg_g0_a1_oof_inventory_v1,
    validate_ba_ieg_g0_a1_prediction_roster_v1,
)


BA_IEG_G0_A1_FOLD_CHECKPOINT_EXPOSURE_MANIFEST_SCHEMA_V1: Final[
    str
] = "ba_ieg_g0_a1_fold_checkpoint_training_exposure_manifest_v1"
BA_IEG_G0_A1_CHECKPOINT_USAGE_BINDING_SCHEMA_V1: Final[
    str
] = "ba_ieg_g0_a1_prediction_fold_checkpoint_usage_binding_v1"
BA_IEG_G0_A1_CHECKPOINT_EXPOSURE_PENDING_SCHEMA_V1: Final[
    str
] = "ba_ieg_g0_a1_checkpoint_training_exposure_pending_status_v1"

_SHA256_ALPHABET: Final[frozenset[str]] = frozenset("0123456789abcdef")
_PREPROCESS_FIT_MODES: Final[frozenset[str]] = frozenset(
    {"learned_on_fold_training_patients", "fixed_stateless"}
)
_USAGE_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "served_prediction",
        "failed_before_checkpoint_load",
        "failed_after_checkpoint_load_before_fold_prediction",
        "failed_after_fold_prediction_before_record_commit",
    }
)
_FAILURE_STAGE_TO_USAGE_STATUSES: Final[dict[str, frozenset[str]]] = {
    "inventory": frozenset({"failed_before_checkpoint_load"}),
    "materialization": frozenset({"failed_before_checkpoint_load"}),
    "preprocess": frozenset({"failed_before_checkpoint_load"}),
    "checkpoint_resolution": frozenset({"failed_before_checkpoint_load"}),
    "checkpoint_load": frozenset({"failed_before_checkpoint_load"}),
    "inference": frozenset({"failed_after_checkpoint_load_before_fold_prediction"}),
    "aggregation": frozenset({"failed_after_fold_prediction_before_record_commit"}),
    "serialization": frozenset({"failed_after_fold_prediction_before_record_commit"}),
    "write": frozenset({"failed_after_fold_prediction_before_record_commit"}),
}
_RAW_EXPOSURE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "inference_fold_id",
        "checkpoint_id",
        "checkpoint_artifact_sha256",
        "checkpoint_training_run_receipt_sha256",
        "preprocess_id",
        "preprocess_artifact_sha256",
        "preprocess_fit_receipt_sha256",
        "preprocess_fit_mode",
        "training_patient_uids",
        "preprocess_fit_patient_uids",
        "training_exposure_roster_complete",
        "checkpoint_training_provenance_scope",
        "external_pretraining_or_patient_adaptation_used",
    }
)
_SEALED_EXPOSURE_FIELDS: Final[frozenset[str]] = frozenset(
    set(_RAW_EXPOSURE_FIELDS)
    | {
        "fold_exposure_id",
        "inventory_receipt_sha256",
        "fold_assignment_receipt_sha256",
        "provider_id",
        "held_out_patient_uids",
        "training_patient_roster_sha256",
        "preprocess_fit_patient_roster_sha256",
        "receipt_sha256",
    }
)
_RAW_USAGE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "patient_uid",
        "recording_id",
        "inference_fold_id",
        "fold_exposure_receipt_sha256",
        "provider_fold_assignment_receipt_sha256",
        "execution_status",
        "provider_checkpoint_load_receipt_sha256",
        "provider_fold_prediction_receipt_sha256",
        "provider_fold_failure_receipt_sha256",
    }
)
_SEALED_USAGE_FIELDS: Final[frozenset[str]] = frozenset(
    set(_RAW_USAGE_FIELDS)
    | {
        "usage_id",
        "prediction_outcome",
        "prediction_failure_stage",
        "prediction_artifact_sha256",
        "prediction_result_receipt_sha256",
        "provider_id",
        "provider_prediction_receipt_sha256",
        "receipt_sha256",
    }
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _strict_object(
    value: object, fields: frozenset[str], context: str
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != set(fields):
        raise ValueError(f"{context} fields drifted")
    return deepcopy(value)


def _identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TypeError(f"{context} must be a non-empty trimmed string")
    return value


def _sha256(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or set(value).difference(_SHA256_ALPHABET)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _fold_id(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"{context} must be a non-negative integer")
    return value


def _patient_roster(value: object, context: str) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be a list")
    result = [_identifier(item, f"{context} patient UID") for item in value]
    if result != sorted(set(result)):
        raise ValueError(f"{context} must be unique and canonically sorted")
    return result


def _seal(value: Mapping[str, Any], *, id_field: str, prefix: str) -> dict[str, Any]:
    result = deepcopy(dict(value))
    result[id_field] = f"{prefix}-PENDING"
    result["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    id_source = deepcopy(result)
    result[id_field] = prefix + "-" + _canonical_sha256(id_source)[:24]
    hash_source = deepcopy(result)
    hash_source["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    result["receipt_sha256"] = _canonical_sha256(hash_source)
    return result


def _replay_seal(
    value: Mapping[str, Any], *, id_field: str, prefix: str, context: str
) -> None:
    expected = _seal(value, id_field=id_field, prefix=prefix)
    if (
        value[id_field] != expected[id_field]
        or value["receipt_sha256"] != expected["receipt_sha256"]
    ):
        raise ValueError(f"{context} content address does not replay")


def _validate_inventory_prediction_source(
    inventory: Mapping[str, Any], prediction_roster: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_inventory = validate_ba_ieg_g0_a1_oof_inventory_v1(dict(inventory))
    predictions = validate_ba_ieg_g0_a1_prediction_roster_v1(dict(prediction_roster))
    if (
        predictions["inventory_receipt_sha256"] != source_inventory["receipt_sha256"]
        or predictions["fold_assignment_receipt_sha256"]
        != source_inventory["fold_assignment_receipt_sha256"]
    ):
        raise ValueError(
            "inventory, fold assignment and prediction roster do not share one source"
        )
    inventory_records = {
        row["recording_id"]: row for row in source_inventory["records"]
    }
    for prediction in predictions["records"]:
        source = inventory_records.get(prediction["recording_id"])
        if source is None or (
            prediction["patient_uid"] != source["patient_uid"]
            or prediction["inference_fold_ids"] != source["held_out_fold_ids"]
            or prediction["source_signal_sha256"] != source["source_signal_sha256"]
        ):
            raise ValueError(
                "prediction record does not replay its inventory fold source"
            )
    return source_inventory, predictions


def _inventory_fold_patient_sets(
    inventory: Mapping[str, Any],
) -> dict[int, dict[str, list[str]]]:
    patient_folds: dict[str, tuple[int, ...]] = {}
    for row in inventory["records"]:
        patient_folds[row["patient_uid"]] = tuple(row["held_out_fold_ids"])
    fold_ids = sorted({fold for folds in patient_folds.values() for fold in folds})
    if not fold_ids:
        raise ValueError("OOF inventory exposes no inference fold")
    all_patients = set(patient_folds)
    result: dict[int, dict[str, list[str]]] = {}
    for fold in fold_ids:
        held_out = sorted(
            patient for patient, folds in patient_folds.items() if fold in folds
        )
        training = sorted(all_patients.difference(held_out))
        if not held_out:
            raise ValueError("inventory fold has no held-out patient")
        if not training:
            raise ValueError("inventory fold has no eligible training patient")
        result[fold] = {"held_out": held_out, "training": training}
    return result


def _training_roster_sha256(
    *,
    inventory_receipt_sha256: str,
    fold_assignment_receipt_sha256: str,
    fold_id: int,
    roster_role: str,
    patient_uids: Sequence[str],
) -> str:
    return _canonical_sha256(
        {
            "schema": "ba_ieg_g0_a1_fold_patient_exposure_roster_v1",
            "inventory_receipt_sha256": inventory_receipt_sha256,
            "fold_assignment_receipt_sha256": fold_assignment_receipt_sha256,
            "inference_fold_id": fold_id,
            "roster_role": roster_role,
            "patient_uids": list(patient_uids),
        }
    )


def _normalize_raw_exposure(
    value: object,
    *,
    index: int,
    inventory: Mapping[str, Any],
    provider_id: str,
    expected_by_fold: Mapping[int, Mapping[str, list[str]]],
) -> dict[str, Any]:
    row = _strict_object(value, _RAW_EXPOSURE_FIELDS, f"fold exposure {index}")
    fold = _fold_id(row["inference_fold_id"], "inference fold ID")
    expected = expected_by_fold.get(fold)
    if expected is None:
        raise ValueError("checkpoint exposure names a fold absent from the inventory")
    training = _patient_roster(row["training_patient_uids"], "training roster")
    if training != expected["training"]:
        raise ValueError(
            "checkpoint training roster does not equal inventory-derived "
            "non-held-out patients"
        )
    if row["training_exposure_roster_complete"] is not True:
        raise ValueError("checkpoint training exposure roster is not declared complete")
    if (
        row["checkpoint_training_provenance_scope"]
        != ("clean_room_source_fold_only_no_external_pretraining_or_patient_adaptation")
        or row["external_pretraining_or_patient_adaptation_used"] is not False
    ):
        raise ValueError(
            "only clean-room source-fold checkpoints are supported; third-party, "
            "pretrained or unknown external exposure must remain pending"
        )
    fit_mode = row["preprocess_fit_mode"]
    if fit_mode not in _PREPROCESS_FIT_MODES:
        raise ValueError("preprocess fit mode is unsupported")
    preprocess_patients = _patient_roster(
        row["preprocess_fit_patient_uids"], "preprocess-fit roster"
    )
    expected_preprocess = (
        training if fit_mode == "learned_on_fold_training_patients" else []
    )
    if preprocess_patients != expected_preprocess:
        raise ValueError("preprocess-fit roster violates its fold-local fit mode")
    held_out = list(expected["held_out"])
    if set(held_out).intersection(training) or set(held_out).intersection(
        preprocess_patients
    ):
        raise ValueError("held-out patient leaked into checkpoint/preprocess fitting")
    body = {
        "fold_exposure_id": "BAIEG-G0-A1-FOLD-EXPOSURE-PENDING",
        "inference_fold_id": fold,
        "inventory_receipt_sha256": inventory["receipt_sha256"],
        "fold_assignment_receipt_sha256": inventory["fold_assignment_receipt_sha256"],
        "provider_id": provider_id,
        "checkpoint_id": _identifier(row["checkpoint_id"], "checkpoint ID"),
        "checkpoint_artifact_sha256": _sha256(
            row["checkpoint_artifact_sha256"], "checkpoint artifact"
        ),
        "checkpoint_training_run_receipt_sha256": _sha256(
            row["checkpoint_training_run_receipt_sha256"],
            "checkpoint training-run receipt",
        ),
        "preprocess_id": _identifier(row["preprocess_id"], "preprocess ID"),
        "preprocess_artifact_sha256": _sha256(
            row["preprocess_artifact_sha256"], "preprocess artifact"
        ),
        "preprocess_fit_receipt_sha256": _sha256(
            row["preprocess_fit_receipt_sha256"], "preprocess-fit receipt"
        ),
        "preprocess_fit_mode": fit_mode,
        "held_out_patient_uids": held_out,
        "training_patient_uids": training,
        "preprocess_fit_patient_uids": preprocess_patients,
        "training_patient_roster_sha256": _training_roster_sha256(
            inventory_receipt_sha256=inventory["receipt_sha256"],
            fold_assignment_receipt_sha256=inventory["fold_assignment_receipt_sha256"],
            fold_id=fold,
            roster_role="checkpoint_complete_source_patient_training_exposure",
            patient_uids=training,
        ),
        "preprocess_fit_patient_roster_sha256": _training_roster_sha256(
            inventory_receipt_sha256=inventory["receipt_sha256"],
            fold_assignment_receipt_sha256=inventory["fold_assignment_receipt_sha256"],
            fold_id=fold,
            roster_role=f"preprocess_fit::{fit_mode}",
            patient_uids=preprocess_patients,
        ),
        "training_exposure_roster_complete": True,
        "checkpoint_training_provenance_scope": (
            "clean_room_source_fold_only_no_external_pretraining_or_patient_adaptation"
        ),
        "external_pretraining_or_patient_adaptation_used": False,
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    return _seal(body, id_field="fold_exposure_id", prefix="BAIEGG0A1FOLDEXP")


def build_ba_ieg_g0_a1_fold_checkpoint_exposure_manifest_v1(
    *,
    inventory: Mapping[str, Any],
    provider_id: str,
    fold_exposures: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind every inventory fold to complete checkpoint/preprocess exposure."""

    source_inventory = validate_ba_ieg_g0_a1_oof_inventory_v1(dict(inventory))
    provider = _identifier(provider_id, "provider ID")
    if not isinstance(fold_exposures, Sequence) or isinstance(
        fold_exposures, (str, bytes)
    ):
        raise TypeError("fold checkpoint exposures must be a sequence")
    expected_by_fold = _inventory_fold_patient_sets(source_inventory)
    exposures = [
        _normalize_raw_exposure(
            dict(row),
            index=index,
            inventory=source_inventory,
            provider_id=provider,
            expected_by_fold=expected_by_fold,
        )
        for index, row in enumerate(fold_exposures)
    ]
    exposures.sort(key=lambda row: row["inference_fold_id"])
    if [row["inference_fold_id"] for row in exposures] != sorted(expected_by_fold):
        raise ValueError("checkpoint exposures do not exactly cover inventory folds")
    if len({row["checkpoint_id"] for row in exposures}) != len(exposures):
        raise ValueError("fold checkpoint IDs must be unique")
    if len({row["checkpoint_artifact_sha256"] for row in exposures}) != len(exposures):
        raise ValueError("fold checkpoint artifacts must be unique")
    if len({row["checkpoint_training_run_receipt_sha256"] for row in exposures}) != len(
        exposures
    ):
        raise ValueError("fold checkpoint training-run receipts must be unique")
    body = {
        "schema_version": BA_IEG_G0_A1_FOLD_CHECKPOINT_EXPOSURE_MANIFEST_SCHEMA_V1,
        "manifest_id": "BAIEG-G0-A1-EXPOSURE-MANIFEST-PENDING",
        "model_split": "source_train",
        "inventory_receipt_sha256": source_inventory["receipt_sha256"],
        "fold_assignment_receipt_sha256": source_inventory[
            "fold_assignment_receipt_sha256"
        ],
        "provider_id": provider,
        "fold_exposures": exposures,
        "counts": {
            "folds": len(exposures),
            "unique_checkpoint_artifacts": len(exposures),
            "held_out_patient_fold_pairs": sum(
                len(row["held_out_patient_uids"]) for row in exposures
            ),
            "checkpoint_training_patient_fold_pairs": sum(
                len(row["training_patient_uids"]) for row in exposures
            ),
        },
        "scope_receipt": {
            "complete_inventory_derived_fold_training_rosters": True,
            "checkpoint_and_preprocess_artifacts_content_addressed": True,
            "checkpoint_authority": (
                "clean_room_source_fold_only_no_external_pretraining_or_patient_adaptation"
            ),
            "third_party_or_pretrained_checkpoint_supported": False,
            "unknown_external_training_exposure_forces_pending": True,
            "global_external_patient_exposure_verified": False,
            "public_event_intervals_opened": 0,
            "edf_annotations_opened": 0,
            "channel_or_soz_targets_opened": 0,
            "private_doctor_or_clinical_text_opened": 0,
            "raw_eeg_embedded": False,
            "training_authorized": False,
            "g0_promotion_authorized": False,
        },
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    result = _seal(body, id_field="manifest_id", prefix="BAIEGG0A1EXPMAN")
    validate_ba_ieg_g0_a1_fold_checkpoint_exposure_manifest_v1(
        result, inventory=source_inventory
    )
    return result


def validate_ba_ieg_g0_a1_fold_checkpoint_exposure_manifest_v1(
    payload: object, *, inventory: Mapping[str, Any]
) -> dict[str, Any]:
    fields = frozenset(
        {
            "schema_version",
            "manifest_id",
            "model_split",
            "inventory_receipt_sha256",
            "fold_assignment_receipt_sha256",
            "provider_id",
            "fold_exposures",
            "counts",
            "scope_receipt",
            "receipt_sha256",
        }
    )
    data = _strict_object(payload, fields, "checkpoint exposure manifest")
    source_inventory = validate_ba_ieg_g0_a1_oof_inventory_v1(dict(inventory))
    if (
        data["schema_version"]
        != BA_IEG_G0_A1_FOLD_CHECKPOINT_EXPOSURE_MANIFEST_SCHEMA_V1
        or data["model_split"] != "source_train"
    ):
        raise ValueError("checkpoint exposure manifest schema/split drifted")
    if (
        data["inventory_receipt_sha256"] != source_inventory["receipt_sha256"]
        or data["fold_assignment_receipt_sha256"]
        != source_inventory["fold_assignment_receipt_sha256"]
    ):
        raise ValueError("checkpoint exposure manifest crosses inventory/fold source")
    provider = _identifier(data["provider_id"], "provider ID")
    expected_by_fold = _inventory_fold_patient_sets(source_inventory)
    if not isinstance(data["fold_exposures"], list):
        raise TypeError("checkpoint exposure rows must be a list")
    exposures: list[dict[str, Any]] = []
    for index, raw in enumerate(data["fold_exposures"]):
        row = _strict_object(raw, _SEALED_EXPOSURE_FIELDS, f"sealed exposure {index}")
        raw_row = {name: row[name] for name in _RAW_EXPOSURE_FIELDS}
        expected = _normalize_raw_exposure(
            raw_row,
            index=index,
            inventory=source_inventory,
            provider_id=provider,
            expected_by_fold=expected_by_fold,
        )
        if row != expected:
            raise ValueError("fold checkpoint exposure content address does not replay")
        exposures.append(row)
    if exposures != sorted(exposures, key=lambda row: row["inference_fold_id"]):
        raise ValueError("fold checkpoint exposures are not canonically sorted")
    if [row["inference_fold_id"] for row in exposures] != sorted(expected_by_fold):
        raise ValueError("checkpoint exposures do not exactly cover inventory folds")
    if len({row["checkpoint_id"] for row in exposures}) != len(exposures):
        raise ValueError("fold checkpoint IDs are not unique")
    if len({row["checkpoint_artifact_sha256"] for row in exposures}) != len(exposures):
        raise ValueError("fold checkpoint artifacts are not unique")
    if len({row["checkpoint_training_run_receipt_sha256"] for row in exposures}) != len(
        exposures
    ):
        raise ValueError("fold checkpoint training-run receipts are not unique")
    expected_counts = {
        "folds": len(exposures),
        "unique_checkpoint_artifacts": len(exposures),
        "held_out_patient_fold_pairs": sum(
            len(row["held_out_patient_uids"]) for row in exposures
        ),
        "checkpoint_training_patient_fold_pairs": sum(
            len(row["training_patient_uids"]) for row in exposures
        ),
    }
    if data["counts"] != expected_counts:
        raise ValueError("checkpoint exposure counts do not replay")
    expected_scope = {
        "complete_inventory_derived_fold_training_rosters": True,
        "checkpoint_and_preprocess_artifacts_content_addressed": True,
        "checkpoint_authority": (
            "clean_room_source_fold_only_no_external_pretraining_or_patient_adaptation"
        ),
        "third_party_or_pretrained_checkpoint_supported": False,
        "unknown_external_training_exposure_forces_pending": True,
        "global_external_patient_exposure_verified": False,
        "public_event_intervals_opened": 0,
        "edf_annotations_opened": 0,
        "channel_or_soz_targets_opened": 0,
        "private_doctor_or_clinical_text_opened": 0,
        "raw_eeg_embedded": False,
        "training_authorized": False,
        "g0_promotion_authorized": False,
    }
    if data["scope_receipt"] != expected_scope:
        raise ValueError("checkpoint exposure firewall/authority drifted")
    _replay_seal(
        data,
        id_field="manifest_id",
        prefix="BAIEGG0A1EXPMAN",
        context="checkpoint exposure manifest",
    )
    return data


def build_ba_ieg_g0_a1_checkpoint_exposure_pending_status_v1(
    *, inventory: Mapping[str, Any], prediction_roster: Mapping[str, Any]
) -> dict[str, Any]:
    """Return the honest status for a native A1 roster with no addendum."""

    source_inventory, predictions = _validate_inventory_prediction_source(
        inventory, prediction_roster
    )
    body = {
        "schema_version": BA_IEG_G0_A1_CHECKPOINT_EXPOSURE_PENDING_SCHEMA_V1,
        "status_id": "BAIEG-G0-A1-CHECKPOINT-EXPOSURE-PENDING",
        "status": (
            "pending_missing_fold_checkpoint_training_exposure_and_"
            "per_prediction_fold_usage"
        ),
        "inventory_receipt_sha256": source_inventory["receipt_sha256"],
        "fold_assignment_receipt_sha256": source_inventory[
            "fold_assignment_receipt_sha256"
        ],
        "prediction_roster_receipt_sha256": predictions["receipt_sha256"],
        "provider_id": predictions["provider_id"],
        "provider_prediction_receipt_sha256": predictions[
            "provider_prediction_receipt_sha256"
        ],
        "inference_fold_ids": sorted(
            {
                fold
                for row in predictions["records"]
                for fold in row["inference_fold_ids"]
            }
        ),
        "prediction_schema_native_per_fold_checkpoint_usage": False,
        "missing_evidence": [
            "fold_checkpoint_and_preprocess_training_exposure_manifest",
            "per_record_per_fold_provider_checkpoint_assignment_receipts",
            "per_record_per_fold_checkpoint_load_and_prediction_receipts",
            "clean_room_no_external_pretraining_or_patient_adaptation_provenance",
        ],
        "scope_receipt": {
            "native_prediction_roster_mutated": False,
            "patient_oof_checkpoint_training_exposure_verified": False,
            "third_party_or_unknown_external_training_exposure_accepted": False,
            "global_external_patient_exposure_verified": False,
            "fail_closed_pending": True,
            "public_event_intervals_opened": 0,
            "channel_or_soz_targets_opened": 0,
            "training_authorized": False,
            "g0_promotion_authorized": False,
        },
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    result = _seal(body, id_field="status_id", prefix="BAIEGG0A1EXPPEND")
    validate_ba_ieg_g0_a1_checkpoint_exposure_pending_status_v1(
        result, inventory=source_inventory, prediction_roster=predictions
    )
    return result


def validate_ba_ieg_g0_a1_checkpoint_exposure_pending_status_v1(
    payload: object,
    *,
    inventory: Mapping[str, Any],
    prediction_roster: Mapping[str, Any],
) -> dict[str, Any]:
    fields = frozenset(
        {
            "schema_version",
            "status_id",
            "status",
            "inventory_receipt_sha256",
            "fold_assignment_receipt_sha256",
            "prediction_roster_receipt_sha256",
            "provider_id",
            "provider_prediction_receipt_sha256",
            "inference_fold_ids",
            "prediction_schema_native_per_fold_checkpoint_usage",
            "missing_evidence",
            "scope_receipt",
            "receipt_sha256",
        }
    )
    data = _strict_object(payload, fields, "checkpoint exposure pending status")
    source_inventory, predictions = _validate_inventory_prediction_source(
        inventory, prediction_roster
    )
    build_body = {
        "schema_version": BA_IEG_G0_A1_CHECKPOINT_EXPOSURE_PENDING_SCHEMA_V1,
        "status_id": "BAIEG-G0-A1-CHECKPOINT-EXPOSURE-PENDING",
        "status": (
            "pending_missing_fold_checkpoint_training_exposure_and_"
            "per_prediction_fold_usage"
        ),
        "inventory_receipt_sha256": source_inventory["receipt_sha256"],
        "fold_assignment_receipt_sha256": source_inventory[
            "fold_assignment_receipt_sha256"
        ],
        "prediction_roster_receipt_sha256": predictions["receipt_sha256"],
        "provider_id": predictions["provider_id"],
        "provider_prediction_receipt_sha256": predictions[
            "provider_prediction_receipt_sha256"
        ],
        "inference_fold_ids": sorted(
            {
                fold
                for row in predictions["records"]
                for fold in row["inference_fold_ids"]
            }
        ),
        "prediction_schema_native_per_fold_checkpoint_usage": False,
        "missing_evidence": [
            "fold_checkpoint_and_preprocess_training_exposure_manifest",
            "per_record_per_fold_provider_checkpoint_assignment_receipts",
            "per_record_per_fold_checkpoint_load_and_prediction_receipts",
            "clean_room_no_external_pretraining_or_patient_adaptation_provenance",
        ],
        "scope_receipt": {
            "native_prediction_roster_mutated": False,
            "patient_oof_checkpoint_training_exposure_verified": False,
            "third_party_or_unknown_external_training_exposure_accepted": False,
            "global_external_patient_exposure_verified": False,
            "fail_closed_pending": True,
            "public_event_intervals_opened": 0,
            "channel_or_soz_targets_opened": 0,
            "training_authorized": False,
            "g0_promotion_authorized": False,
        },
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    expected = _seal(build_body, id_field="status_id", prefix="BAIEGG0A1EXPPEND")
    if data != expected:
        raise ValueError("checkpoint exposure pending status does not replay")
    return data


def _normalize_usage(
    value: object,
    *,
    index: int,
    prediction_by_record: Mapping[str, Mapping[str, Any]],
    exposure_by_fold: Mapping[int, Mapping[str, Any]],
    provider_id: str,
    provider_prediction_receipt_sha256: str,
) -> dict[str, Any]:
    row = _strict_object(value, _RAW_USAGE_FIELDS, f"prediction-fold usage {index}")
    patient = _identifier(row["patient_uid"], "usage patient UID")
    recording = _identifier(row["recording_id"], "usage recording ID")
    fold = _fold_id(row["inference_fold_id"], "usage inference fold")
    prediction = prediction_by_record.get(recording)
    if prediction is None or prediction["patient_uid"] != patient:
        raise ValueError("checkpoint usage crosses prediction patient/recording")
    if fold not in prediction["inference_fold_ids"]:
        raise ValueError("checkpoint usage fold was not assigned to the prediction")
    exposure = exposure_by_fold.get(fold)
    if exposure is None:
        raise ValueError("checkpoint usage fold has no exposure manifest row")
    if row["fold_exposure_receipt_sha256"] != exposure["receipt_sha256"]:
        raise ValueError("checkpoint usage references the wrong fold exposure")
    if (
        patient in exposure["training_patient_uids"]
        or patient in exposure["preprocess_fit_patient_uids"]
    ):
        raise ValueError("prediction patient leaked into its serving checkpoint")
    status = row["execution_status"]
    if status not in _USAGE_STATUSES:
        raise ValueError("checkpoint usage status is unsupported")
    load_receipt = row["provider_checkpoint_load_receipt_sha256"]
    fold_prediction_receipt = row["provider_fold_prediction_receipt_sha256"]
    failure_receipt = row["provider_fold_failure_receipt_sha256"]
    if prediction["outcome"] == "technical_failure":
        failure_stage = prediction["failure_stage"]
        allowed_statuses = _FAILURE_STAGE_TO_USAGE_STATUSES.get(failure_stage)
        if allowed_statuses is None:
            raise ValueError(
                "prediction failure stage has no fail-closed checkpoint-exposure "
                "mapping; keep the binding pending"
            )
        if status not in allowed_statuses:
            raise ValueError(
                "technical-failure checkpoint receipts disagree with prediction "
                "failure stage"
            )
        failure_receipt = _sha256(
            failure_receipt, "provider fold inference-attempt/failure receipt"
        )
        if status == "failed_before_checkpoint_load":
            if load_receipt is not None or fold_prediction_receipt is not None:
                raise ValueError("pre-load failure claims a later execution receipt")
        elif status == "failed_after_checkpoint_load_before_fold_prediction":
            load_receipt = _sha256(load_receipt, "provider checkpoint-load receipt")
            if fold_prediction_receipt is not None:
                raise ValueError(
                    "pre-prediction failure claims a fold prediction receipt"
                )
        else:
            load_receipt = _sha256(load_receipt, "provider checkpoint-load receipt")
            fold_prediction_receipt = _sha256(
                fold_prediction_receipt, "provider fold-prediction receipt"
            )
    else:
        if status != "served_prediction":
            raise ValueError("completed prediction lacks a served checkpoint usage")
        if failure_receipt is not None:
            raise ValueError("completed prediction acquired a failure receipt")
        load_receipt = _sha256(load_receipt, "provider checkpoint-load receipt")
        fold_prediction_receipt = _sha256(
            fold_prediction_receipt, "provider fold-prediction receipt"
        )
    body = {
        "usage_id": "BAIEG-G0-A1-CHECKPOINT-USAGE-PENDING",
        "patient_uid": patient,
        "recording_id": recording,
        "inference_fold_id": fold,
        "fold_exposure_receipt_sha256": exposure["receipt_sha256"],
        "provider_fold_assignment_receipt_sha256": _sha256(
            row["provider_fold_assignment_receipt_sha256"],
            "provider fold-assignment receipt",
        ),
        "execution_status": status,
        "provider_checkpoint_load_receipt_sha256": load_receipt,
        "provider_fold_prediction_receipt_sha256": fold_prediction_receipt,
        "provider_fold_failure_receipt_sha256": failure_receipt,
        "prediction_outcome": prediction["outcome"],
        "prediction_failure_stage": prediction["failure_stage"],
        "prediction_artifact_sha256": prediction["prediction_artifact_sha256"],
        "prediction_result_receipt_sha256": prediction[
            "prediction_result_receipt_sha256"
        ],
        "provider_id": provider_id,
        "provider_prediction_receipt_sha256": provider_prediction_receipt_sha256,
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    return _seal(body, id_field="usage_id", prefix="BAIEGG0A1USE")


def build_ba_ieg_g0_a1_checkpoint_training_exposure_binding_v1(
    *,
    inventory: Mapping[str, Any],
    prediction_roster: Mapping[str, Any],
    exposure_manifest: Mapping[str, Any],
    prediction_fold_usage_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Verify additive per-fold checkpoint usage without opening any target."""

    source_inventory, predictions = _validate_inventory_prediction_source(
        inventory, prediction_roster
    )
    exposures = validate_ba_ieg_g0_a1_fold_checkpoint_exposure_manifest_v1(
        dict(exposure_manifest), inventory=source_inventory
    )
    if exposures["provider_id"] != predictions["provider_id"]:
        raise ValueError(
            "checkpoint exposure provider differs from prediction provider"
        )
    if (
        exposures["inventory_receipt_sha256"] != predictions["inventory_receipt_sha256"]
        or exposures["fold_assignment_receipt_sha256"]
        != predictions["fold_assignment_receipt_sha256"]
    ):
        raise ValueError("checkpoint exposure crosses prediction fold source")
    if not isinstance(prediction_fold_usage_rows, Sequence) or isinstance(
        prediction_fold_usage_rows, (str, bytes)
    ):
        raise TypeError("prediction-fold checkpoint usage must be a sequence")
    prediction_by_record = {row["recording_id"]: row for row in predictions["records"]}
    exposure_by_fold = {
        row["inference_fold_id"]: row for row in exposures["fold_exposures"]
    }
    usage = [
        _normalize_usage(
            dict(row),
            index=index,
            prediction_by_record=prediction_by_record,
            exposure_by_fold=exposure_by_fold,
            provider_id=predictions["provider_id"],
            provider_prediction_receipt_sha256=predictions[
                "provider_prediction_receipt_sha256"
            ],
        )
        for index, row in enumerate(prediction_fold_usage_rows)
    ]
    usage.sort(
        key=lambda row: (
            row["patient_uid"],
            row["recording_id"],
            row["inference_fold_id"],
        )
    )
    expected_keys = {
        (row["recording_id"], fold)
        for row in predictions["records"]
        for fold in row["inference_fold_ids"]
    }
    observed_keys = {(row["recording_id"], row["inference_fold_id"]) for row in usage}
    if observed_keys != expected_keys or len(usage) != len(expected_keys):
        raise ValueError(
            "checkpoint usage does not exactly cover every prediction-fold denominator"
        )
    for receipt_field in (
        "provider_fold_assignment_receipt_sha256",
        "provider_checkpoint_load_receipt_sha256",
        "provider_fold_prediction_receipt_sha256",
        "provider_fold_failure_receipt_sha256",
    ):
        receipts = [
            row[receipt_field] for row in usage if row[receipt_field] is not None
        ]
        if len(receipts) != len(set(receipts)):
            raise ValueError(f"{receipt_field} is reused across prediction-fold rows")
    body = {
        "schema_version": BA_IEG_G0_A1_CHECKPOINT_USAGE_BINDING_SCHEMA_V1,
        "binding_id": "BAIEG-G0-A1-CHECKPOINT-BINDING-PENDING",
        "verification_status": (
            "verified_clean_room_source_fold_additive_supplemental_binding"
        ),
        "model_split": "source_train",
        "inventory_receipt_sha256": source_inventory["receipt_sha256"],
        "fold_assignment_receipt_sha256": source_inventory[
            "fold_assignment_receipt_sha256"
        ],
        "prediction_roster_receipt_sha256": predictions["receipt_sha256"],
        "provider_id": predictions["provider_id"],
        "provider_prediction_receipt_sha256": predictions[
            "provider_prediction_receipt_sha256"
        ],
        "exposure_manifest_receipt_sha256": exposures["receipt_sha256"],
        "prediction_schema_native_per_fold_checkpoint_usage": False,
        "supplemental_prediction_fold_usage": usage,
        "counts": {
            "prediction_records": len(predictions["records"]),
            "prediction_fold_denominators": len(usage),
            "served_prediction_fold_rows": sum(
                row["execution_status"] == "served_prediction" for row in usage
            ),
            "technical_failure_fold_rows": sum(
                row["prediction_outcome"] == "technical_failure" for row in usage
            ),
            "patient_exposure_violations": 0,
        },
        "scope_receipt": {
            "native_prediction_roster_mutated": False,
            "supplemental_per_prediction_fold_checkpoint_lineage_complete": True,
            "every_prediction_patient_absent_from_assigned_checkpoint_training": True,
            "every_prediction_patient_absent_from_assigned_preprocess_fit": True,
            "inventory_fold_prediction_receipts_share_one_source": True,
            "checkpoint_authority_is_clean_room_source_fold_only": True,
            "third_party_pretrained_or_unknown_external_exposure_eligible": False,
            "global_external_patient_exposure_verified": False,
            "public_event_intervals_opened": 0,
            "edf_annotations_opened": 0,
            "channel_or_soz_targets_opened": 0,
            "private_doctor_or_clinical_text_opened": 0,
            "raw_eeg_embedded": False,
            "training_authorized": False,
            "g0_promotion_authorized": False,
        },
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    result = _seal(body, id_field="binding_id", prefix="BAIEGG0A1CKPTBIND")
    validate_ba_ieg_g0_a1_checkpoint_training_exposure_binding_v1(
        result,
        inventory=source_inventory,
        prediction_roster=predictions,
        exposure_manifest=exposures,
    )
    return result


def validate_ba_ieg_g0_a1_checkpoint_training_exposure_binding_v1(
    payload: object,
    *,
    inventory: Mapping[str, Any],
    prediction_roster: Mapping[str, Any],
    exposure_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    fields = frozenset(
        {
            "schema_version",
            "binding_id",
            "verification_status",
            "model_split",
            "inventory_receipt_sha256",
            "fold_assignment_receipt_sha256",
            "prediction_roster_receipt_sha256",
            "provider_id",
            "provider_prediction_receipt_sha256",
            "exposure_manifest_receipt_sha256",
            "prediction_schema_native_per_fold_checkpoint_usage",
            "supplemental_prediction_fold_usage",
            "counts",
            "scope_receipt",
            "receipt_sha256",
        }
    )
    data = _strict_object(payload, fields, "checkpoint exposure binding")
    source_inventory, predictions = _validate_inventory_prediction_source(
        inventory, prediction_roster
    )
    exposures = validate_ba_ieg_g0_a1_fold_checkpoint_exposure_manifest_v1(
        dict(exposure_manifest), inventory=source_inventory
    )
    if (
        data["schema_version"] != BA_IEG_G0_A1_CHECKPOINT_USAGE_BINDING_SCHEMA_V1
        or data["verification_status"]
        != "verified_clean_room_source_fold_additive_supplemental_binding"
        or data["model_split"] != "source_train"
    ):
        raise ValueError("checkpoint exposure binding schema/status drifted")
    expected_sources = {
        "inventory_receipt_sha256": source_inventory["receipt_sha256"],
        "fold_assignment_receipt_sha256": source_inventory[
            "fold_assignment_receipt_sha256"
        ],
        "prediction_roster_receipt_sha256": predictions["receipt_sha256"],
        "provider_id": predictions["provider_id"],
        "provider_prediction_receipt_sha256": predictions[
            "provider_prediction_receipt_sha256"
        ],
        "exposure_manifest_receipt_sha256": exposures["receipt_sha256"],
    }
    if any(data[name] != value for name, value in expected_sources.items()):
        raise ValueError("checkpoint binding crosses inventory/fold/prediction source")
    if exposures["provider_id"] != predictions["provider_id"]:
        raise ValueError("checkpoint binding mixes providers")
    if data["prediction_schema_native_per_fold_checkpoint_usage"] is not False:
        raise ValueError("native A1 prediction schema overclaims checkpoint lineage")
    prediction_by_record = {row["recording_id"]: row for row in predictions["records"]}
    exposure_by_fold = {
        row["inference_fold_id"]: row for row in exposures["fold_exposures"]
    }
    if not isinstance(data["supplemental_prediction_fold_usage"], list):
        raise TypeError("supplemental checkpoint usage must be a list")
    usage: list[dict[str, Any]] = []
    for index, raw in enumerate(data["supplemental_prediction_fold_usage"]):
        row = _strict_object(raw, _SEALED_USAGE_FIELDS, f"sealed usage {index}")
        raw_row = {name: row[name] for name in _RAW_USAGE_FIELDS}
        expected = _normalize_usage(
            raw_row,
            index=index,
            prediction_by_record=prediction_by_record,
            exposure_by_fold=exposure_by_fold,
            provider_id=predictions["provider_id"],
            provider_prediction_receipt_sha256=predictions[
                "provider_prediction_receipt_sha256"
            ],
        )
        if row != expected:
            raise ValueError("prediction-fold checkpoint usage does not replay")
        usage.append(row)
    expected_order = sorted(
        usage,
        key=lambda row: (
            row["patient_uid"],
            row["recording_id"],
            row["inference_fold_id"],
        ),
    )
    if usage != expected_order:
        raise ValueError("prediction-fold checkpoint usage is not canonically sorted")
    expected_keys = {
        (row["recording_id"], fold)
        for row in predictions["records"]
        for fold in row["inference_fold_ids"]
    }
    observed_keys = {(row["recording_id"], row["inference_fold_id"]) for row in usage}
    if observed_keys != expected_keys or len(usage) != len(expected_keys):
        raise ValueError("checkpoint binding lost a prediction-fold denominator")
    expected_counts = {
        "prediction_records": len(predictions["records"]),
        "prediction_fold_denominators": len(usage),
        "served_prediction_fold_rows": sum(
            row["execution_status"] == "served_prediction" for row in usage
        ),
        "technical_failure_fold_rows": sum(
            row["prediction_outcome"] == "technical_failure" for row in usage
        ),
        "patient_exposure_violations": 0,
    }
    if data["counts"] != expected_counts:
        raise ValueError("checkpoint exposure binding counts do not replay")
    expected_scope = {
        "native_prediction_roster_mutated": False,
        "supplemental_per_prediction_fold_checkpoint_lineage_complete": True,
        "every_prediction_patient_absent_from_assigned_checkpoint_training": True,
        "every_prediction_patient_absent_from_assigned_preprocess_fit": True,
        "inventory_fold_prediction_receipts_share_one_source": True,
        "checkpoint_authority_is_clean_room_source_fold_only": True,
        "third_party_pretrained_or_unknown_external_exposure_eligible": False,
        "global_external_patient_exposure_verified": False,
        "public_event_intervals_opened": 0,
        "edf_annotations_opened": 0,
        "channel_or_soz_targets_opened": 0,
        "private_doctor_or_clinical_text_opened": 0,
        "raw_eeg_embedded": False,
        "training_authorized": False,
        "g0_promotion_authorized": False,
    }
    if data["scope_receipt"] != expected_scope:
        raise ValueError("checkpoint exposure binding firewall/authority drifted")
    _replay_seal(
        data,
        id_field="binding_id",
        prefix="BAIEGG0A1CKPTBIND",
        context="checkpoint exposure binding",
    )
    return data


__all__ = [
    "BA_IEG_G0_A1_FOLD_CHECKPOINT_EXPOSURE_MANIFEST_SCHEMA_V1",
    "BA_IEG_G0_A1_CHECKPOINT_USAGE_BINDING_SCHEMA_V1",
    "BA_IEG_G0_A1_CHECKPOINT_EXPOSURE_PENDING_SCHEMA_V1",
    "build_ba_ieg_g0_a1_fold_checkpoint_exposure_manifest_v1",
    "validate_ba_ieg_g0_a1_fold_checkpoint_exposure_manifest_v1",
    "build_ba_ieg_g0_a1_checkpoint_exposure_pending_status_v1",
    "validate_ba_ieg_g0_a1_checkpoint_exposure_pending_status_v1",
    "build_ba_ieg_g0_a1_checkpoint_training_exposure_binding_v1",
    "validate_ba_ieg_g0_a1_checkpoint_training_exposure_binding_v1",
]
