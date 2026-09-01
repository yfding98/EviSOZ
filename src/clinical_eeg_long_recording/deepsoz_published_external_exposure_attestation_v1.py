"""Fail-closed exposure attestation for published external DeepSOZ folds.

This contract deliberately does *not* use the clean-room G0a exposure schema.
It records what can be established from the authors' published repository:

* exact train/test array bytes and their disjoint patient rosters;
* exact published checkpoint bytes copied into the local provider bundle; and
* the legacy aggregate posterior's declared fold IDs and runtime fold rows.

The historical posterior bundle did not persist the native per-fold posterior
payloads or per-record checkpoint-load receipts.  Consequently this contract
must keep actual historical checkpoint usage, strict G0a, training permission,
and clinical/production permission closed.  A content ID for a missing payload
is not treated as a replayable execution receipt.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any, Final, Mapping, Sequence

from .deepsoz_temporal_adapter import (
    PUBLISHED_DEEPSOZ_FOLD_WEIGHT_SHA256,
    PUBLISHED_DEEPSOZ_TEST_FOLD_NPY_SHA256,
    PUBLISHED_DEEPSOZ_WEIGHTS_MANIFEST_SHA256,
)


DEEPSOZ_PUBLISHED_EXTERNAL_EXPOSURE_ATTESTATION_SCHEMA_V1: Final[str] = (
    "deepsoz_published_external_patient_fold_exposure_attestation_v1"
)
DEEPSOZ_PUBLISHED_EXTERNAL_EXPOSURE_ATTESTATION_CLASS_V1: Final[str] = (
    "published_external_patient_fold_split_and_checkpoint_artifact_attested"
)
DEEPSOZ_PUBLISHED_EXTERNAL_EXPOSURE_STATUS_V1: Final[str] = (
    "published_external_split_checkpoint_bytes_attested_historical_usage_pending"
)
DEEPSOZ_PROVIDER_ID: Final[str] = "deepsoz_temporal_oof_candidate_v1"
DEEPSOZ_UPSTREAM_REPOSITORY_URL: Final[str] = (
    "https://github.com/deeksha-ms/DeepSOZ.git"
)
DEEPSOZ_UPSTREAM_COMMIT: Final[str] = (
    "913c921f8a08fa4df76ca0708126f565860f1068"
)

PUBLISHED_DEEPSOZ_TRAIN_FOLD_NPY_SHA256: Final[dict[int, str]] = {
    0: "64b426090a8f598c8a969dbea8a4a208712b7df76284d9978d7ea9776bd732d3",
    1: "d5a03fca457b08789d85b1334c4e486a79495e031888a95c5204a4ec50011e78",
    2: "2ef273aeddee912d6ba7a83eff4f84388fd5fd00dd74808ee528fbf3dc5c34a8",
    3: "147d5177b267c62e1368ea0a74073deb619f354847b9ee03062d8033a1bae5b9",
    4: "d7aebbfda7b6c110d325ee78a98240402a7c6be21da89df6cf2f328abe478b3f",
    5: "199adb3b83a9981387d60fa9964be3ea88bdf963b0ce8d23c98b9567535fccfe",
    6: "9079dca9a0829bd43c8a718c648ba9c6a84fe8ae39a78e9b36df66ee53330f72",
    7: "2a7b9321a70ad6b0febcdb19e5faa87de118057fc0c3fa6a3bb57edee7c73a7c",
    8: "90322a35c41740bc6a132b4781c3c31e0bc821a83aeaaee6df4f3e2a21d61e7e",
    9: "110bbca3eb9ed599e49dc0b11e49d1edafe87af428e6a04357da6ebfb8ab2ae3",
    10: "795edc03874d326b9852e28c6a432cf796b01d0dd42b4334cb9929dc4ac229d7",
    11: "da91a7e15d0f1c5055463c08734774710f4a870e8d43e34b5ea329b10a94458f",
    12: "5eb9a3737ce157cc3fb19154b96084aeab9e298734b1f0c7c3d299cc85dd66a3",
    13: "9154217e5b9a5f4a753c18678beb708ecf262e12cfc5d78d152d35a0becc01d7",
    14: "1419932a35a717ee5fd5f24d79fe3f82c5a1a9e4c1c8fc6cfbe83b0f0e0e6b47",
}

PUBLISHED_DEEPSOZ_SELECTED_CHECKPOINT_BASENAME: Final[dict[int, str]] = {
    **{
        fold: f"txlstm_szpool_finetuned_cv{fold % 5}_0.0001.pth.tar"
        for fold in range(15)
    },
    10: "txlstm_nomask_szpool_finetuned_cv0_0.0001.pth.tar",
}

_SHA256_ALPHABET: Final[frozenset[str]] = frozenset("0123456789abcdef")
_FOLD_RAW_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "fold_index",
        "train_relative_path",
        "train_file_sha256",
        "train_patient_ids",
        "train_patient_roster_sha256",
        "test_relative_path",
        "test_file_sha256",
        "test_patient_ids",
        "test_patient_roster_sha256",
        "fold_universe_patient_roster_sha256",
        "train_patient_count",
        "test_patient_count",
        "fold_universe_patient_count",
        "train_test_disjoint",
        "test_is_exact_complement_of_train",
        "checkpoint_upstream_relative_path",
        "checkpoint_local_filename",
        "checkpoint_sha256",
        "checkpoint_upstream_git_tracked",
        "checkpoint_upstream_worktree_clean",
        "checkpoint_local_upstream_byte_identical",
        "published_directory_colocation_attested",
        "training_run_receipt_available",
        "complete_checkpoint_training_exposure_verified",
        "clean_room_verified",
    }
)
_FOLD_FIELDS: Final[frozenset[str]] = frozenset(
    set(_FOLD_RAW_FIELDS) | {"fold_receipt_sha256"}
)
_USAGE_RAW_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "record_fold_ordinal",
        "patient_id",
        "recording_id",
        "fold_index",
        "fold_receipt_sha256",
        "checkpoint_sha256",
        "aggregate_posterior_artifact_id",
        "aggregate_posterior_file_sha256",
        "fold_posterior_artifact_id",
        "patient_in_published_test_roster",
        "patient_absent_from_published_train_roster",
        "aggregate_declares_fold",
        "original_runtime_declares_fold_inference",
        "legacy_fold_posterior_content_id_present",
        "native_fold_posterior_payload_available",
        "native_fold_checkpoint_hash_replayed",
        "historical_checkpoint_load_receipt_available",
        "actual_posterior_usage_verified",
        "usage_verification_status",
    }
)
_USAGE_FIELDS: Final[frozenset[str]] = frozenset(
    set(_USAGE_RAW_FIELDS) | {"usage_receipt_sha256"}
)
_UPSTREAM_RAW_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "repository_url",
        "upstream_remote_url",
        "pinned_commit",
        "observed_head_commit",
        "head_matches_pinned_commit",
        "bound_paths_git_tracked",
        "bound_paths_worktree_clean",
        "selected_checkpoint_bytes_match_normalized_local",
        "training_code_artifacts",
        "training_run_receipts_available",
        "exact_environment_receipt_available",
        "original_preprocessing_execution_receipt_available",
        "checkpoint_training_patient_roster_declared_complete",
        "model_license_verified",
        "commit_signature_verified",
    }
)
_UPSTREAM_FIELDS: Final[frozenset[str]] = frozenset(
    set(_UPSTREAM_RAW_FIELDS) | {"upstream_binding_sha256"}
)
_BATCH_RAW_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "batch_root_name",
        "batch_receipt_id",
        "batch_receipt_file_sha256",
        "posterior_index_file_sha256",
        "reference_free_validation_receipt_sha256",
        "selected_split",
        "recording_count",
        "patient_count",
        "record_fold_usage_count",
        "materializer_code_sha256",
        "adapter_code_sha256",
        "weights_manifest_sha256",
        "all_aggregate_artifact_content_ids_verified",
        "all_original_runtime_fold_indices_verified",
        "native_per_fold_posterior_payloads_persisted",
        "actual_per_fold_checkpoint_usage_replayable",
    }
)
_BATCH_FIELDS: Final[frozenset[str]] = frozenset(
    set(_BATCH_RAW_FIELDS) | {"posterior_batch_binding_sha256"}
)

_MISSING_EVIDENCE: Final[list[str]] = [
    "author_training_run_receipt_binding_each_checkpoint_to_the_complete_training_exposure_roster",
    "author_preprocessing_fit_and_environment_receipts_for_each_checkpoint",
    "native_persisted_per_record_per_fold_posterior_payloads_containing_checkpoint_sha256",
    "native_per_record_per_fold_checkpoint_load_and_prediction_receipts",
    "historical_materializer_execution_receipt_binding_fold_payloads_to_loaded_checkpoints",
    "v1_4_stable_origin_support_lineage_geometry_matched_background_and_shortcut_gates",
]


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
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


def _patient(value: object, context: str) -> str:
    text = _identifier(str(value), context)
    if not text.isdigit():
        raise ValueError(f"{context} must be numeric")
    return str(int(text))


def _patient_roster(value: object, context: str) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be a list")
    result = [_patient(item, context) for item in value]
    if result != sorted(set(result), key=int):
        raise ValueError(f"{context} must be unique and numerically sorted")
    return result


def _nonnegative_int(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"{context} must be a non-negative integer")
    return value


def _seal_subreceipt(
    value: Mapping[str, Any], *, field: str
) -> dict[str, Any]:
    result = deepcopy(dict(value))
    result[field] = "CONTENT-ADDRESS-PENDING"
    result[field] = canonical_sha256(result)
    return result


def _normalize_fold(value: object, index: int) -> dict[str, Any]:
    row = _strict_object(value, _FOLD_RAW_FIELDS, f"fold exposure {index}")
    fold = _nonnegative_int(row["fold_index"], "fold index")
    if fold >= 15:
        raise ValueError("fold index is outside 0..14")
    train = _patient_roster(row["train_patient_ids"], "train patient roster")
    test = _patient_roster(row["test_patient_ids"], "test patient roster")
    if len(train) != 100 or len(test) != 24:
        raise ValueError("published fold must contain 100 train and 24 test patients")
    if set(train).intersection(test) or len(set(train).union(test)) != 124:
        raise ValueError("published train/test fold is not a disjoint 124-patient split")
    if (
        row["train_patient_count"] != 100
        or row["test_patient_count"] != 24
        or row["fold_universe_patient_count"] != 124
        or row["train_test_disjoint"] is not True
        or row["test_is_exact_complement_of_train"] is not True
    ):
        raise ValueError("published fold count/disjointness claims drifted")
    if (
        row["train_file_sha256"] != PUBLISHED_DEEPSOZ_TRAIN_FOLD_NPY_SHA256[fold]
        or row["test_file_sha256"] != PUBLISHED_DEEPSOZ_TEST_FOLD_NPY_SHA256[fold]
        or row["checkpoint_sha256"] != PUBLISHED_DEEPSOZ_FOLD_WEIGHT_SHA256[fold]
        or row["checkpoint_local_filename"] != f"fold{fold}.pth.tar"
        or not str(row["train_relative_path"]).startswith(f"final_models/fold{fold}/")
        or not str(row["test_relative_path"]).startswith(f"final_models/fold{fold}/")
        or row["checkpoint_upstream_relative_path"]
        != (
            f"final_models/fold{fold}/"
            f"{PUBLISHED_DEEPSOZ_SELECTED_CHECKPOINT_BASENAME[fold]}"
        )
    ):
        raise ValueError("published fold file/checkpoint identity drifted")
    for field in (
        "train_patient_roster_sha256",
        "test_patient_roster_sha256",
        "fold_universe_patient_roster_sha256",
        "train_file_sha256",
        "test_file_sha256",
        "checkpoint_sha256",
    ):
        _sha256(row[field], field)
    if row["train_patient_roster_sha256"] != canonical_sha256(train):
        raise ValueError("train patient roster hash drifted")
    if row["test_patient_roster_sha256"] != canonical_sha256(test):
        raise ValueError("test patient roster hash drifted")
    universe = sorted(set(train).union(test), key=int)
    if row["fold_universe_patient_roster_sha256"] != canonical_sha256(universe):
        raise ValueError("fold universe patient roster hash drifted")
    for field in (
        "checkpoint_upstream_git_tracked",
        "checkpoint_upstream_worktree_clean",
        "checkpoint_local_upstream_byte_identical",
        "published_directory_colocation_attested",
    ):
        if row[field] is not True:
            raise ValueError("published checkpoint byte/working-tree attestation failed")
    for field in (
        "training_run_receipt_available",
        "complete_checkpoint_training_exposure_verified",
        "clean_room_verified",
    ):
        if row[field] is not False:
            raise ValueError("external published fold overclaims training provenance")
    return _seal_subreceipt(row, field="fold_receipt_sha256")


def _normalize_upstream(value: object) -> dict[str, Any]:
    row = _strict_object(value, _UPSTREAM_RAW_FIELDS, "upstream repository binding")
    if (
        row["repository_url"] != DEEPSOZ_UPSTREAM_REPOSITORY_URL
        or row["upstream_remote_url"] != DEEPSOZ_UPSTREAM_REPOSITORY_URL
        or row["pinned_commit"] != DEEPSOZ_UPSTREAM_COMMIT
        or row["observed_head_commit"] != DEEPSOZ_UPSTREAM_COMMIT
    ):
        raise ValueError("upstream repository identity drifted")
    for field in (
        "head_matches_pinned_commit",
        "bound_paths_git_tracked",
        "bound_paths_worktree_clean",
        "selected_checkpoint_bytes_match_normalized_local",
    ):
        if row[field] is not True:
            raise ValueError("upstream repository byte binding is incomplete")
    for field in (
        "training_run_receipts_available",
        "exact_environment_receipt_available",
        "original_preprocessing_execution_receipt_available",
        "checkpoint_training_patient_roster_declared_complete",
        "model_license_verified",
        "commit_signature_verified",
    ):
        if row[field] is not False:
            raise ValueError("upstream repository overclaims unavailable provenance")
    artifacts = row["training_code_artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("upstream training code artifact inventory is empty")
    paths: list[str] = []
    for artifact in artifacts:
        if type(artifact) is not dict or set(artifact) != {
            "relative_path",
            "file_sha256",
            "git_tracked",
            "worktree_clean",
        }:
            raise ValueError("upstream training code artifact schema drifted")
        paths.append(_identifier(artifact["relative_path"], "training code path"))
        _sha256(artifact["file_sha256"], "training code hash")
        if artifact["git_tracked"] is not True or artifact["worktree_clean"] is not True:
            raise ValueError("upstream training code is not tracked and clean")
    if paths != sorted(set(paths)):
        raise ValueError("training code artifacts must be uniquely sorted")
    return _seal_subreceipt(row, field="upstream_binding_sha256")


def _normalize_batch(value: object) -> dict[str, Any]:
    row = _strict_object(value, _BATCH_RAW_FIELDS, "posterior batch binding")
    if (
        row["selected_split"] != "source_train"
        or row["weights_manifest_sha256"]
        != PUBLISHED_DEEPSOZ_WEIGHTS_MANIFEST_SHA256
    ):
        raise ValueError("posterior batch split/weight identity drifted")
    for field in (
        "batch_receipt_file_sha256",
        "posterior_index_file_sha256",
        "reference_free_validation_receipt_sha256",
        "materializer_code_sha256",
        "adapter_code_sha256",
        "weights_manifest_sha256",
    ):
        _sha256(row[field], field)
    for field in (
        "recording_count",
        "patient_count",
        "record_fold_usage_count",
    ):
        if _nonnegative_int(row[field], field) < 1:
            raise ValueError("posterior batch count must be positive")
    if (
        row["all_aggregate_artifact_content_ids_verified"] is not True
        or row["all_original_runtime_fold_indices_verified"] is not True
        or row["native_per_fold_posterior_payloads_persisted"] is not False
        or row["actual_per_fold_checkpoint_usage_replayable"] is not False
    ):
        raise ValueError("posterior batch legacy-usage boundary drifted")
    return _seal_subreceipt(row, field="posterior_batch_binding_sha256")


def _normalize_usage(
    value: object,
    index: int,
    *,
    folds: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    row = _strict_object(value, _USAGE_RAW_FIELDS, f"prediction-fold usage {index}")
    ordinal = _nonnegative_int(row["record_fold_ordinal"], "record-fold ordinal")
    if ordinal != index + 1:
        raise ValueError("record-fold usage ordinal drifted")
    patient = _patient(row["patient_id"], "usage patient")
    _identifier(row["recording_id"], "usage recording")
    fold = _nonnegative_int(row["fold_index"], "usage fold")
    exposure = folds.get(fold)
    if exposure is None:
        raise ValueError("usage references an unknown fold")
    if (
        row["fold_receipt_sha256"] != exposure["fold_receipt_sha256"]
        or row["checkpoint_sha256"] != exposure["checkpoint_sha256"]
        or patient not in exposure["test_patient_ids"]
        or patient in exposure["train_patient_ids"]
    ):
        raise ValueError("usage patient/checkpoint does not replay its published fold")
    for field in (
        "fold_receipt_sha256",
        "checkpoint_sha256",
        "aggregate_posterior_file_sha256",
    ):
        _sha256(row[field], field)
    for field in (
        "patient_in_published_test_roster",
        "patient_absent_from_published_train_roster",
        "aggregate_declares_fold",
        "original_runtime_declares_fold_inference",
        "legacy_fold_posterior_content_id_present",
    ):
        if row[field] is not True:
            raise ValueError("published external held-out usage declaration failed")
    for field in (
        "native_fold_posterior_payload_available",
        "native_fold_checkpoint_hash_replayed",
        "historical_checkpoint_load_receipt_available",
        "actual_posterior_usage_verified",
    ):
        if row[field] is not False:
            raise ValueError("legacy posterior usage overclaims missing execution evidence")
    if row["usage_verification_status"] != (
        "declared_fold_content_id_and_runtime_only_pending_native_fold_payload"
    ):
        raise ValueError("legacy posterior usage status drifted")
    aggregate_id = _identifier(
        row["aggregate_posterior_artifact_id"], "aggregate posterior artifact ID"
    )
    fold_id = _identifier(
        row["fold_posterior_artifact_id"], "fold posterior artifact ID"
    )
    if not aggregate_id.startswith("DSZOOF-") or not fold_id.startswith("DSZPOST-"):
        raise ValueError("posterior artifact ID prefix drifted")
    return _seal_subreceipt(row, field="usage_receipt_sha256")


def _build(
    *,
    upstream_repository: Mapping[str, Any],
    posterior_batch: Mapping[str, Any],
    fold_exposures: Sequence[Mapping[str, Any]],
    prediction_fold_usage: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    upstream = _normalize_upstream(upstream_repository)
    batch = _normalize_batch(posterior_batch)
    if not isinstance(fold_exposures, Sequence) or isinstance(
        fold_exposures, (str, bytes)
    ):
        raise TypeError("fold exposures must be a sequence")
    folds = [_normalize_fold(row, index) for index, row in enumerate(fold_exposures)]
    folds.sort(key=lambda row: row["fold_index"])
    if [row["fold_index"] for row in folds] != list(range(15)):
        raise ValueError("published external exposure must cover all 15 folds")
    fold_lookup = {row["fold_index"]: row for row in folds}
    if not isinstance(prediction_fold_usage, Sequence) or isinstance(
        prediction_fold_usage, (str, bytes)
    ):
        raise TypeError("prediction-fold usage must be a sequence")
    usage = [
        _normalize_usage(row, index, folds=fold_lookup)
        for index, row in enumerate(prediction_fold_usage)
    ]
    usage_keys = [
        (row["recording_id"], row["fold_index"])
        for row in usage
    ]
    if len(usage_keys) != len(set(usage_keys)):
        raise ValueError("prediction-fold usage contains duplicate denominators")
    if batch["record_fold_usage_count"] != len(usage):
        raise ValueError("posterior batch record-fold count differs from usage ledger")
    recording_ids = {row["recording_id"] for row in usage}
    patient_ids = {row["patient_id"] for row in usage}
    if (
        batch["recording_count"] != len(recording_ids)
        or batch["patient_count"] != len(patient_ids)
    ):
        raise ValueError("posterior batch record/patient count differs from usage ledger")
    body: dict[str, Any] = {
        "schema_version": DEEPSOZ_PUBLISHED_EXTERNAL_EXPOSURE_ATTESTATION_SCHEMA_V1,
        "attestation_id": "DEEPSOZ-PUBLISHED-EXTERNAL-EXPOSURE-PENDING",
        "provider_id": DEEPSOZ_PROVIDER_ID,
        "attestation_class": (
            DEEPSOZ_PUBLISHED_EXTERNAL_EXPOSURE_ATTESTATION_CLASS_V1
        ),
        "verification_status": DEEPSOZ_PUBLISHED_EXTERNAL_EXPOSURE_STATUS_V1,
        "upstream_repository": upstream,
        "posterior_batch": batch,
        "fold_exposures": folds,
        "prediction_fold_usage": usage,
        "counts": {
            "published_folds": len(folds),
            "published_train_patient_fold_memberships": sum(
                len(row["train_patient_ids"]) for row in folds
            ),
            "published_test_patient_fold_memberships": sum(
                len(row["test_patient_ids"]) for row in folds
            ),
            "posterior_records": len(recording_ids),
            "posterior_patients": len(patient_ids),
            "prediction_fold_usage_rows": len(usage),
            "inference_patient_train_roster_violations": 0,
            "checkpoint_byte_mismatches": 0,
            "actual_posterior_usage_verified_rows": 0,
            "actual_posterior_usage_pending_rows": len(usage),
        },
        "evidence_gates": {
            "published_external_split_files_exact_hash_verified": True,
            "every_inference_patient_in_serving_test_roster": True,
            "every_inference_patient_absent_from_serving_train_roster": True,
            "published_checkpoint_bytes_exact_hash_verified": True,
            "historical_aggregate_declared_fold_artifact_ids_bound": True,
            "historical_runtime_fold_indices_bound": True,
            "published_external_patient_held_out_assignment_attested": True,
            "checkpoint_training_run_provenance_verified": False,
            "complete_checkpoint_training_exposure_verified": False,
            "actual_posterior_usage_verified": False,
            "clean_room_verified": False,
            "strict_g0a_checkpoint_exposure_verified": False,
            "strict_g0a_verified": False,
            "model_training_authorized": False,
            "navigation_training_authorized": False,
            "clinical_or_production_use_authorized": False,
        },
        "missing_evidence": list(_MISSING_EVIDENCE),
        "source_firewall": {
            "eeg_samples_opened_by_attestation": False,
            "public_event_intervals_opened": False,
            "source_train_reference_opened": False,
            "source_eval_opened": False,
            "edf_annotations_opened": False,
            "excel_or_doctor_labels_opened": False,
            "clinical_text_opened": False,
            "raw_posterior_values_copied": False,
        },
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    id_source = deepcopy(body)
    body["attestation_id"] = "DSZPUBEXTEXP-" + canonical_sha256(id_source)[:24]
    body["receipt_sha256"] = canonical_sha256(body)
    return body


def build_deepsoz_published_external_exposure_attestation_v1(
    *,
    upstream_repository: Mapping[str, Any],
    posterior_batch: Mapping[str, Any],
    fold_exposures: Sequence[Mapping[str, Any]],
    prediction_fold_usage: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a published-external attestation while keeping strict gates shut."""

    result = _build(
        upstream_repository=upstream_repository,
        posterior_batch=posterior_batch,
        fold_exposures=fold_exposures,
        prediction_fold_usage=prediction_fold_usage,
    )
    return validate_deepsoz_published_external_exposure_attestation_v1(result)


def validate_deepsoz_published_external_exposure_attestation_v1(
    payload: object,
) -> dict[str, Any]:
    """Replay a sealed attestation without accepting a clean-room promotion."""

    fields = frozenset(
        {
            "schema_version",
            "attestation_id",
            "provider_id",
            "attestation_class",
            "verification_status",
            "upstream_repository",
            "posterior_batch",
            "fold_exposures",
            "prediction_fold_usage",
            "counts",
            "evidence_gates",
            "missing_evidence",
            "source_firewall",
            "receipt_sha256",
        }
    )
    data = _strict_object(payload, fields, "published external attestation")
    if (
        data["schema_version"]
        != DEEPSOZ_PUBLISHED_EXTERNAL_EXPOSURE_ATTESTATION_SCHEMA_V1
        or data["provider_id"] != DEEPSOZ_PROVIDER_ID
        or data["attestation_class"]
        != DEEPSOZ_PUBLISHED_EXTERNAL_EXPOSURE_ATTESTATION_CLASS_V1
        or data["verification_status"]
        != DEEPSOZ_PUBLISHED_EXTERNAL_EXPOSURE_STATUS_V1
    ):
        raise ValueError("published external attestation identity drifted")
    upstream = _strict_object(
        data["upstream_repository"], _UPSTREAM_FIELDS, "sealed upstream binding"
    )
    batch = _strict_object(
        data["posterior_batch"], _BATCH_FIELDS, "sealed posterior batch binding"
    )
    if not isinstance(data["fold_exposures"], list) or not isinstance(
        data["prediction_fold_usage"], list
    ):
        raise TypeError("sealed fold/usage ledgers must be lists")
    raw_upstream = {key: upstream[key] for key in _UPSTREAM_RAW_FIELDS}
    raw_batch = {key: batch[key] for key in _BATCH_RAW_FIELDS}
    raw_folds = []
    for row in data["fold_exposures"]:
        sealed = _strict_object(row, _FOLD_FIELDS, "sealed fold exposure")
        raw_folds.append({key: sealed[key] for key in _FOLD_RAW_FIELDS})
    raw_usage = []
    for row in data["prediction_fold_usage"]:
        sealed = _strict_object(row, _USAGE_FIELDS, "sealed usage row")
        raw_usage.append({key: sealed[key] for key in _USAGE_RAW_FIELDS})
    expected = _build(
        upstream_repository=raw_upstream,
        posterior_batch=raw_batch,
        fold_exposures=raw_folds,
        prediction_fold_usage=raw_usage,
    )
    if data != expected:
        raise ValueError("published external attestation does not replay")
    return data


__all__ = [
    "DEEPSOZ_PROVIDER_ID",
    "DEEPSOZ_PUBLISHED_EXTERNAL_EXPOSURE_ATTESTATION_CLASS_V1",
    "DEEPSOZ_PUBLISHED_EXTERNAL_EXPOSURE_ATTESTATION_SCHEMA_V1",
    "DEEPSOZ_PUBLISHED_EXTERNAL_EXPOSURE_STATUS_V1",
    "DEEPSOZ_UPSTREAM_COMMIT",
    "DEEPSOZ_UPSTREAM_REPOSITORY_URL",
    "PUBLISHED_DEEPSOZ_SELECTED_CHECKPOINT_BASENAME",
    "PUBLISHED_DEEPSOZ_TRAIN_FOLD_NPY_SHA256",
    "build_deepsoz_published_external_exposure_attestation_v1",
    "canonical_sha256",
    "validate_deepsoz_published_external_exposure_attestation_v1",
]
