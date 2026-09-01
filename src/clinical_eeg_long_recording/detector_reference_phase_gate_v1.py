"""Controller-authorized detector label-bearing reference phases.

``inner_validation`` and ``final_refit`` references are label-bearing data.
They therefore cannot be opened on the strength of a caller supplied digest.
This module replays an out-of-band trust-anchor file, a self-hashed gate
artifact, the complete candidate-checkpoint byte inventory, reference-free
predictions for every candidate epoch and recording, and the actual prior
phase exposure receipts.  Final-refit additionally replays the selected-epoch
metric inventory and the from-scratch refit prerequisites.

The first implementation incorrectly accepted the expected trust-anchor hash
from the same caller that supplied the gate bundle, and prediction rows had no
artifact path/byte binding.  This implementation instead verifies an Ed25519
signature rooted in the checked phase registry, replays every checkpoint and
typed prediction artifact byte, seals a logical pre-reference timing receipt,
and only then authorizes the first exact ``TERM,seiz`` reference read.

The supported prediction interchange is deliberately narrow and provider
neutral: normalized seizure intervals plus a typed terminal outcome.  Dense
providers must export that interval contract through a frozen adapter;
unregistered contracts fail closed.  Epoch selection is recomputed with one
fixed exact interval scorer and never accepts a caller-supplied metric value.
The module has no signing API and the repository contains no controller
private key.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Final, Mapping, Sequence

try:  # A missing cryptographic dependency is a hard authorization failure.
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
except ImportError:  # pragma: no cover - exercised by explicit fail-closed injection.
    InvalidSignature = None  # type: ignore[assignment]
    Ed25519PublicKey = None  # type: ignore[assignment]


DETECTOR_REFERENCE_PHASE_GATE_VALIDATOR_ID_V1: Final[str] = (
    "ed25519_controller_signed_prediction_first_detector_reference_gate_v1"
)
DETECTOR_REFERENCE_PHASE_GATE_ARTIFACT_SCHEMA_V1: Final[str] = (
    "clinical_eeg_detector_reference_phase_gate_artifact_v1"
)
DETECTOR_REFERENCE_PHASE_GATE_TRUST_ANCHOR_SCHEMA_V1: Final[str] = (
    "clinical_eeg_detector_reference_phase_gate_trust_anchor_v1"
)
DETECTOR_PHASE_ACTUAL_EXPOSURE_RECEIPT_SCHEMA_V1: Final[str] = (
    "clinical_eeg_detector_phase_actual_exposure_receipt_v1"
)
DETECTOR_SELECTION_CHECKPOINT_INVENTORY_SCHEMA_V1: Final[str] = (
    "clinical_eeg_detector_selection_checkpoint_inventory_v1"
)
DETECTOR_REFERENCE_FREE_PREDICTION_INVENTORY_SCHEMA_V1: Final[str] = (
    "clinical_eeg_detector_reference_free_prediction_inventory_v1"
)
DETECTOR_SELECTED_EPOCH_FREEZE_SCHEMA_V1: Final[str] = (
    "clinical_eeg_detector_selected_epoch_freeze_v1"
)
DETECTOR_FINAL_REFIT_PREREQUISITE_SCHEMA_V1: Final[str] = (
    "clinical_eeg_detector_final_refit_prerequisite_v1"
)
DETECTOR_REFERENCE_PHASE_GATE_EXECUTION_STATUS_V1: Final[str] = (
    "nonselection_reference_phases_executable_only_through_controller_signed_"
    "byte_replayed_prediction_first_and_exact_metric_recompute_v1"
)
DETECTOR_CONTROLLER_LEDGER_SCHEMA_V1: Final[str] = (
    "clinical_eeg_detector_controller_release_ledger_v1"
)
DETECTOR_CONTROLLER_SIGNATURE_ALGORITHM_V1: Final[str] = "ed25519"
DETECTOR_PREDICTION_TERMINAL_ARTIFACT_SCHEMA_V1: Final[str] = (
    "clinical_eeg_provider_neutral_interval_prediction_terminal_v1"
)
DETECTOR_PREDICTION_CONTRACT_ID_V1: Final[str] = (
    "provider_neutral_normalized_seizure_intervals_and_typed_terminal_v1"
)
DETECTOR_PRE_REFERENCE_RELEASE_RECEIPT_SCHEMA_V1: Final[str] = (
    "clinical_eeg_detector_pre_reference_release_timing_receipt_v1"
)
DETECTOR_SELECTION_METRIC_RECEIPT_SCHEMA_V1: Final[str] = (
    "clinical_eeg_detector_exact_selection_metric_receipt_v1"
)
DETECTOR_SELECTION_SCORER_ID_V1: Final[str] = (
    "patient_macro_event_sensitive_false_alarm_dual_track_scorer_v1"
)
DETECTOR_SELECTION_SCORER_VERSION_V1: Final[str] = "1.1.0"
DETECTOR_FINAL_REFIT_TYPED_PREREQUISITE_SCHEMA_V1: Final[str] = (
    "clinical_eeg_detector_typed_final_refit_prerequisite_v1"
)

_SHA256_ALPHABET: Final[frozenset[str]] = frozenset("0123456789abcdef")
_PHASES: Final[frozenset[str]] = frozenset({"inner_validation", "final_refit"})
_TERMINAL_OUTCOMES: Final[frozenset[str]] = frozenset(
    {
        "completed_with_predictions",
        "completed_zero_alarm",
        "partial_coverage",
        "technical_failure",
    }
)
_ROSTER_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "patient_count",
        "recording_count",
        "duration_seconds_fraction",
        "patient_roster_sha256",
        "analysis_identity_roster_sha256",
        "local_edf_path_roster_sha256",
        "record_duration_binding_sha256",
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


def detector_reference_phase_gate_source_sha256_v1() -> str:
    digest = hashlib.sha256()
    with Path(__file__).resolve(strict=True).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or set(value).difference(_SHA256_ALPHABET)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _identifier(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 1024
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"{context} must be a normalized identifier")
    return value


def _strict_dict(value: object, fields: set[str] | frozenset[str], context: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != set(fields):
        raise ValueError(f"{context} fields drifted")
    return deepcopy(value)


def _positive_int(value: object, context: str, *, allow_zero: bool = False) -> int:
    if type(value) is not int or value < (0 if allow_zero else 1):
        raise ValueError(f"{context} must be a {'nonnegative' if allow_zero else 'positive'} integer")
    return value


def _outer_fold_id(value: object, context: str = "outer fold ID") -> int:
    if type(value) is not int or not 0 <= value < 5:
        raise ValueError(f"{context} must be one of 0..4")
    return value


def _zero_int(value: object, context: str) -> int:
    if type(value) is not int or value != 0:
        raise PermissionError(f"{context} must be the integer zero")
    return value


def _fraction(value: object, context: str) -> Fraction:
    if (
        type(value) is not list
        or len(value) != 2
        or type(value[0]) is not int
        or type(value[1]) is not int
        or value[1] <= 0
    ):
        raise ValueError(f"{context} must be a reduced positive fraction")
    result = Fraction(value[0], value[1])
    if result <= 0 or [result.numerator, result.denominator] != value:
        raise ValueError(f"{context} must be a reduced positive fraction")
    return result


def _validate_roster(value: object, context: str) -> dict[str, Any]:
    roster = _strict_dict(value, _ROSTER_FIELDS, context)
    _positive_int(roster["patient_count"], f"{context} patient_count")
    _positive_int(roster["recording_count"], f"{context} recording_count")
    _fraction(roster["duration_seconds_fraction"], f"{context} duration")
    for field in _ROSTER_FIELDS.difference(
        {"patient_count", "recording_count", "duration_seconds_fraction"}
    ):
        _sha256(roster[field], f"{context} {field}")
    if roster["patient_count"] > roster["recording_count"]:
        raise ValueError(f"{context} has more patients than recordings")
    return roster


def _validate_fold_ids(value: object, context: str) -> list[int]:
    if type(value) is not list or not value:
        raise ValueError(f"{context} must be a non-empty fold array")
    if any(type(item) is not int or not 0 <= item < 5 for item in value):
        raise ValueError(f"{context} contains an invalid fold")
    if value != sorted(set(value)):
        raise ValueError(f"{context} must be unique and sorted")
    return list(value)


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    body = deepcopy(dict(value))
    if "receipt_sha256" in body:
        raise ValueError("receipt_sha256 is assigned by the sealer")
    body["receipt_sha256"] = _canonical_sha256(body)
    return body


def _validate_self_hash(value: object, fields: set[str], context: str) -> dict[str, Any]:
    body = _strict_dict(value, fields | {"receipt_sha256"}, context)
    observed = _sha256(body["receipt_sha256"], f"{context} receipt")
    if observed != _canonical_sha256(
        {key: item for key, item in body.items() if key != "receipt_sha256"}
    ):
        raise ValueError(f"{context} does not replay")
    return body


def _safe_relative_path(value: object, context: str) -> PurePosixPath:
    text = _identifier(value, context)
    if "\\" in text:
        raise ValueError(f"{context} must use POSIX separators")
    path = PurePosixPath(text)
    if path.is_absolute() or not path.parts or "." in path.parts or ".." in path.parts:
        raise PermissionError(f"{context} escapes the gate bundle")
    return path


def _read_bundle_bytes(bundle_root: Path, relative_path: object, context: str) -> bytes:
    relative = _safe_relative_path(relative_path, context)
    root_input = Path(bundle_root)
    if root_input.is_symlink():
        raise ValueError("phase-gate bundle root must not be a symlink")
    root = root_input.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("phase-gate bundle root must be a directory")
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"{context} must not contain a symlink")
    candidate = cursor.resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise PermissionError(f"{context} escaped the gate bundle") from error
    if not candidate.is_file() or candidate.is_symlink():
        raise ValueError(f"{context} must be a regular file")
    return candidate.read_bytes()


def _load_strict_json(payload: bytes, context: str) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{context} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"{context} contains non-finite JSON token {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{context} is not strict UTF-8 JSON") from error
    if type(value) is not dict:
        raise ValueError(f"{context} must be a JSON object")
    return value


def _expected_rows(value: Sequence[Mapping[str, Any]], context: str) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise ValueError(f"{context}[{index}] must be an object")
        result.append(
            {
                "analysis_identity_id": _identifier(
                    raw.get("analysis_identity_id"), f"{context}[{index}] identity"
                ),
                "local_patient_id": _identifier(
                    raw.get("local_patient_id"), f"{context}[{index}] patient"
                ),
                "source_edf_relative_path": _identifier(
                    raw.get("source_edf_relative_path", raw.get("local_edf_path")),
                    f"{context}[{index}] EDF path",
                ),
            }
        )
    result.sort(key=lambda row: (row["analysis_identity_id"], row["source_edf_relative_path"]))
    identities = [row["analysis_identity_id"] for row in result]
    paths = [row["source_edf_relative_path"] for row in result]
    if len(set(identities)) != len(identities) or len(set(paths)) != len(paths):
        raise ValueError(f"{context} contains duplicate identities or paths")
    return result


def build_detector_phase_actual_exposure_receipt_v1(
    *,
    provider_id: str,
    detector_variant_id: str,
    outer_fold_id: int,
    phase: str,
    authorized_fold_ids: Sequence[int],
    authorized_roster: Mapping[str, Any],
    reference_authority_receipt_sha256: str,
    reference_file_count: int,
    reference_bytes_read: int,
    target_event_inventory_sha256: str,
) -> dict[str, Any]:
    """Seal the actual target exposure from a completed prior phase."""

    return _seal(
        {
            "schema_version": DETECTOR_PHASE_ACTUAL_EXPOSURE_RECEIPT_SCHEMA_V1,
            "provider_id": _identifier(provider_id, "exposure provider"),
            "detector_variant_id": _identifier(detector_variant_id, "exposure variant"),
            "outer_fold_id": _outer_fold_id(outer_fold_id),
            "phase": phase,
            "authorized_fold_ids": list(authorized_fold_ids),
            "authorized_roster": deepcopy(dict(authorized_roster)),
            "reference_authority_receipt_sha256": _sha256(
                reference_authority_receipt_sha256, "reference authority receipt"
            ),
            "reference_file_count": _positive_int(
                reference_file_count, "reference file count"
            ),
            "reference_bytes_read": _positive_int(
                reference_bytes_read, "reference bytes read"
            ),
            "target_event_inventory_sha256": _sha256(
                target_event_inventory_sha256, "target event inventory"
            ),
            "outer_heldout_reference_files_opened": 0,
            "source_dev_reference_files_opened": 0,
            "source_eval_reference_files_opened": 0,
            "private_reference_files_opened": 0,
        }
    )


_EXPOSURE_FIELDS = {
    "schema_version",
    "provider_id",
    "detector_variant_id",
    "outer_fold_id",
    "phase",
    "authorized_fold_ids",
    "authorized_roster",
    "reference_authority_receipt_sha256",
    "reference_file_count",
    "reference_bytes_read",
    "target_event_inventory_sha256",
    "outer_heldout_reference_files_opened",
    "source_dev_reference_files_opened",
    "source_eval_reference_files_opened",
    "private_reference_files_opened",
}


def _validate_exposure(
    value: object,
    *,
    provider_id: str,
    variant_id: str,
    outer_fold_id: int,
    phase: str,
    expected_fold_ids: Sequence[int],
    expected_roster: Mapping[str, Any],
) -> dict[str, Any]:
    outer_fold_id = _outer_fold_id(outer_fold_id)
    data = _validate_self_hash(value, _EXPOSURE_FIELDS, f"{phase} actual exposure")
    if (
        data["schema_version"] != DETECTOR_PHASE_ACTUAL_EXPOSURE_RECEIPT_SCHEMA_V1
        or data["provider_id"] != provider_id
        or data["detector_variant_id"] != variant_id
        or _outer_fold_id(data["outer_fold_id"], "exposure outer fold")
        != outer_fold_id
        or data["phase"] != phase
        or _validate_fold_ids(data["authorized_fold_ids"], "exposure folds")
        != list(expected_fold_ids)
        or _validate_roster(data["authorized_roster"], "exposure roster")
        != dict(expected_roster)
    ):
        raise ValueError(f"{phase} actual exposure lineage or roster drifted")
    _sha256(data["reference_authority_receipt_sha256"], "reference authority receipt")
    _sha256(data["target_event_inventory_sha256"], "target inventory")
    if (
        _positive_int(data["reference_file_count"], "reference file count")
        != expected_roster["recording_count"]
    ):
        raise ValueError(f"{phase} actual exposure denominator is incomplete")
    _positive_int(data["reference_bytes_read"], "reference bytes read")
    for field in (
        "outer_heldout_reference_files_opened",
        "source_dev_reference_files_opened",
        "source_eval_reference_files_opened",
        "private_reference_files_opened",
    ):
        _zero_int(data[field], f"forbidden reference exposure {field}")
    return data


def build_detector_selection_checkpoint_inventory_v1(
    *,
    provider_id: str,
    detector_variant_id: str,
    outer_fold_id: int,
    authorized_fold_ids: Sequence[int],
    authorized_roster: Mapping[str, Any],
    actual_exposure_receipt_sha256: str,
    architecture_code_sha256: str,
    training_code_sha256: str,
    preprocessing_fit_artifact_sha256: str,
    random_seed_and_rng_state_receipt_sha256: str,
    candidate_checkpoints: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    candidates = sorted(
        [deepcopy(dict(row)) for row in candidate_checkpoints],
        key=lambda row: row["epoch"],
    )
    return _seal(
        {
            "schema_version": DETECTOR_SELECTION_CHECKPOINT_INVENTORY_SCHEMA_V1,
            "provider_id": _identifier(provider_id, "checkpoint provider"),
            "detector_variant_id": _identifier(detector_variant_id, "checkpoint variant"),
            "outer_fold_id": _outer_fold_id(outer_fold_id),
            "training_phase": "selection_fit",
            "authorized_fold_ids": list(authorized_fold_ids),
            "authorized_roster": deepcopy(dict(authorized_roster)),
            "actual_exposure_receipt_sha256": _sha256(
                actual_exposure_receipt_sha256, "selection exposure receipt"
            ),
            "architecture_code_sha256": _sha256(architecture_code_sha256, "architecture code"),
            "training_code_sha256": _sha256(training_code_sha256, "training code"),
            "preprocessing_fit_artifact_sha256": _sha256(
                preprocessing_fit_artifact_sha256, "preprocessing fit artifact"
            ),
            "random_seed_and_rng_state_receipt_sha256": _sha256(
                random_seed_and_rng_state_receipt_sha256, "RNG receipt"
            ),
            "candidate_checkpoints": candidates,
            "candidate_epoch_roster_sha256": _canonical_sha256(
                [row["epoch"] for row in candidates]
            ),
            "outer_heldout_reference_access_count": 0,
            "source_dev_reference_access_count": 0,
            "source_eval_reference_access_count": 0,
            "private_reference_access_count": 0,
        }
    )


_CHECKPOINT_FIELDS = {
    "schema_version",
    "provider_id",
    "detector_variant_id",
    "outer_fold_id",
    "training_phase",
    "authorized_fold_ids",
    "authorized_roster",
    "actual_exposure_receipt_sha256",
    "architecture_code_sha256",
    "training_code_sha256",
    "preprocessing_fit_artifact_sha256",
    "random_seed_and_rng_state_receipt_sha256",
    "candidate_checkpoints",
    "candidate_epoch_roster_sha256",
    "outer_heldout_reference_access_count",
    "source_dev_reference_access_count",
    "source_eval_reference_access_count",
    "private_reference_access_count",
}
_CHECKPOINT_ROW_FIELDS = {
    "epoch",
    "checkpoint_relative_path",
    "checkpoint_file_sha256",
    "checkpoint_file_bytes",
}


def _validate_checkpoint_inventory(
    value: object,
    *,
    bundle_root: Path,
    provider_id: str,
    variant_id: str,
    outer_fold_id: int,
    expected_fold_ids: Sequence[int],
    expected_roster: Mapping[str, Any],
    exposure_receipt_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    outer_fold_id = _outer_fold_id(outer_fold_id)
    data = _validate_self_hash(value, _CHECKPOINT_FIELDS, "selection checkpoint inventory")
    if (
        data["schema_version"] != DETECTOR_SELECTION_CHECKPOINT_INVENTORY_SCHEMA_V1
        or data["provider_id"] != provider_id
        or data["detector_variant_id"] != variant_id
        or _outer_fold_id(data["outer_fold_id"], "checkpoint outer fold")
        != outer_fold_id
        or data["training_phase"] != "selection_fit"
        or _validate_fold_ids(data["authorized_fold_ids"], "selection checkpoint folds")
        != list(expected_fold_ids)
        or _validate_roster(data["authorized_roster"], "selection checkpoint roster")
        != dict(expected_roster)
        or data["actual_exposure_receipt_sha256"] != exposure_receipt_sha256
    ):
        raise ValueError("selection checkpoint lineage or exposure drifted")
    for field in (
        "architecture_code_sha256",
        "training_code_sha256",
        "preprocessing_fit_artifact_sha256",
        "random_seed_and_rng_state_receipt_sha256",
    ):
        _sha256(data[field], f"selection checkpoint {field}")
    for field in (
        "outer_heldout_reference_access_count",
        "source_dev_reference_access_count",
        "source_eval_reference_access_count",
        "private_reference_access_count",
    ):
        _zero_int(data[field], f"selection checkpoint {field}")
    raw_candidates = data["candidate_checkpoints"]
    if type(raw_candidates) is not list or not raw_candidates:
        raise ValueError("selection checkpoint candidate inventory is empty")
    candidates: list[dict[str, Any]] = []
    bytes_read = 0
    previous_epoch = 0
    paths: set[str] = set()
    for index, raw in enumerate(raw_candidates):
        row = _strict_dict(raw, _CHECKPOINT_ROW_FIELDS, f"checkpoint candidate {index}")
        epoch = _positive_int(row["epoch"], f"checkpoint candidate {index} epoch")
        if epoch <= previous_epoch:
            raise ValueError("checkpoint candidate epochs must be strictly increasing")
        previous_epoch = epoch
        relative = _safe_relative_path(
            row["checkpoint_relative_path"], f"checkpoint candidate {index} path"
        ).as_posix()
        if relative in paths:
            raise ValueError("checkpoint candidate paths must be unique")
        paths.add(relative)
        expected_hash = _sha256(row["checkpoint_file_sha256"], "checkpoint bytes")
        expected_bytes = _positive_int(row["checkpoint_file_bytes"], "checkpoint size")
        payload = _read_bundle_bytes(bundle_root, relative, "checkpoint artifact")
        if len(payload) != expected_bytes or hashlib.sha256(payload).hexdigest() != expected_hash:
            raise ValueError("selection checkpoint artifact bytes drifted")
        row["checkpoint_relative_path"] = relative
        candidates.append(row)
        bytes_read += len(payload)
    epochs = [row["epoch"] for row in candidates]
    if data["candidate_epoch_roster_sha256"] != _canonical_sha256(epochs):
        raise ValueError("checkpoint candidate epoch roster drifted")
    return data, candidates, bytes_read


def build_detector_reference_free_prediction_inventory_v1(
    *,
    provider_id: str,
    detector_variant_id: str,
    outer_fold_id: int,
    authorized_fold_ids: Sequence[int],
    authorized_roster: Mapping[str, Any],
    selection_checkpoint_inventory_receipt_sha256: str,
    candidate_epochs: Sequence[int],
    prediction_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = sorted(
        [deepcopy(dict(row)) for row in prediction_rows],
        key=lambda row: (row["checkpoint_epoch"], row["analysis_identity_id"]),
    )
    epochs = list(candidate_epochs)
    patients = sorted({row["local_patient_id"] for row in rows})
    identities = sorted({row["analysis_identity_id"] for row in rows})
    return _seal(
        {
            "schema_version": DETECTOR_REFERENCE_FREE_PREDICTION_INVENTORY_SCHEMA_V1,
            "provider_id": _identifier(provider_id, "prediction provider"),
            "detector_variant_id": _identifier(detector_variant_id, "prediction variant"),
            "outer_fold_id": _outer_fold_id(outer_fold_id),
            "inference_phase": "inner_validation_before_reference_open",
            "authorized_fold_ids": list(authorized_fold_ids),
            "authorized_roster": deepcopy(dict(authorized_roster)),
            "selection_checkpoint_inventory_receipt_sha256": _sha256(
                selection_checkpoint_inventory_receipt_sha256,
                "selection checkpoint inventory receipt",
            ),
            "candidate_epochs": epochs,
            "prediction_rows": rows,
            "prediction_row_count": len(rows),
            "prediction_recording_count": len(identities),
            "prediction_patient_count": len(patients),
            "prediction_row_roster_sha256": _canonical_sha256(rows),
            "complete_cartesian_epoch_record_inventory": True,
            "reference_fields_present": False,
            "reference_access_count_before_freeze": 0,
        }
    )


_PREDICTION_FIELDS = {
    "schema_version",
    "provider_id",
    "detector_variant_id",
    "outer_fold_id",
    "inference_phase",
    "authorized_fold_ids",
    "authorized_roster",
    "selection_checkpoint_inventory_receipt_sha256",
    "candidate_epochs",
    "prediction_rows",
    "prediction_row_count",
    "prediction_recording_count",
    "prediction_patient_count",
    "prediction_row_roster_sha256",
    "complete_cartesian_epoch_record_inventory",
    "reference_fields_present",
    "reference_access_count_before_freeze",
}
_PREDICTION_ROW_FIELDS = {
    "analysis_identity_id",
    "local_patient_id",
    "source_edf_relative_path",
    "checkpoint_epoch",
    "checkpoint_file_sha256",
    "prediction_payload_sha256",
    "terminal_outcome",
    "reference_access_count_before_freeze",
}


def _validate_prediction_inventory(
    value: object,
    *,
    provider_id: str,
    variant_id: str,
    outer_fold_id: int,
    expected_fold_ids: Sequence[int],
    expected_roster: Mapping[str, Any],
    expected_rows: Sequence[Mapping[str, Any]],
    checkpoint_receipt_sha256: str,
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    outer_fold_id = _outer_fold_id(outer_fold_id)
    data = _validate_self_hash(value, _PREDICTION_FIELDS, "prediction-first inventory")
    epochs = [row["epoch"] for row in candidates]
    observed_candidate_epochs = [
        _positive_int(epoch, "prediction candidate epoch")
        for epoch in data["candidate_epochs"]
    ] if type(data["candidate_epochs"]) is list else []
    _zero_int(
        data["reference_access_count_before_freeze"],
        "prediction inventory reference access count",
    )
    if (
        data["schema_version"] != DETECTOR_REFERENCE_FREE_PREDICTION_INVENTORY_SCHEMA_V1
        or data["provider_id"] != provider_id
        or data["detector_variant_id"] != variant_id
        or _outer_fold_id(data["outer_fold_id"], "prediction outer fold")
        != outer_fold_id
        or data["inference_phase"] != "inner_validation_before_reference_open"
        or _validate_fold_ids(data["authorized_fold_ids"], "prediction folds")
        != list(expected_fold_ids)
        or _validate_roster(data["authorized_roster"], "prediction roster")
        != dict(expected_roster)
        or data["selection_checkpoint_inventory_receipt_sha256"]
        != checkpoint_receipt_sha256
        or observed_candidate_epochs != epochs
        or data["complete_cartesian_epoch_record_inventory"] is not True
        or data["reference_fields_present"] is not False
    ):
        raise PermissionError("prediction-first inventory lineage/firewall drifted")
    expected = _expected_rows(expected_rows, "expected prediction records")
    expected_by_identity = {row["analysis_identity_id"]: row for row in expected}
    checkpoint_hash_by_epoch = {
        row["epoch"]: row["checkpoint_file_sha256"] for row in candidates
    }
    raw_rows = data["prediction_rows"]
    if type(raw_rows) is not list:
        raise ValueError("prediction-first rows must be an array")
    normalized: list[dict[str, Any]] = []
    observed_pairs: set[tuple[int, str]] = set()
    for index, raw in enumerate(raw_rows):
        row = _strict_dict(raw, _PREDICTION_ROW_FIELDS, f"prediction row {index}")
        epoch = _positive_int(row["checkpoint_epoch"], f"prediction row {index} epoch")
        identity = _identifier(row["analysis_identity_id"], "prediction identity")
        pair = (epoch, identity)
        if pair in observed_pairs:
            raise ValueError("prediction-first inventory contains a duplicate epoch/record")
        observed_pairs.add(pair)
        expected_row = expected_by_identity.get(identity)
        if (
            expected_row is None
            or row["local_patient_id"] != expected_row["local_patient_id"]
            or row["source_edf_relative_path"] != expected_row["source_edf_relative_path"]
            or row["checkpoint_file_sha256"] != checkpoint_hash_by_epoch.get(epoch)
        ):
            raise ValueError("prediction-first row roster/checkpoint drifted")
        _sha256(row["prediction_payload_sha256"], "prediction payload")
        if row["terminal_outcome"] not in _TERMINAL_OUTCOMES:
            raise ValueError("prediction-first terminal outcome is invalid")
        _zero_int(
            row["reference_access_count_before_freeze"],
            "prediction-first row reference access count",
        )
        normalized.append(row)
    normalized.sort(key=lambda row: (row["checkpoint_epoch"], row["analysis_identity_id"]))
    expected_pairs = {(epoch, row["analysis_identity_id"]) for epoch in epochs for row in expected}
    if observed_pairs != expected_pairs or normalized != raw_rows:
        raise ValueError("prediction-first inventory is not the complete epoch/record Cartesian product")
    patients = {row["local_patient_id"] for row in normalized}
    identities = {row["analysis_identity_id"] for row in normalized}
    if (
        _positive_int(data["prediction_row_count"], "prediction row count")
        != len(normalized)
        or _positive_int(
            data["prediction_recording_count"], "prediction recording count"
        )
        != len(expected)
        or _positive_int(data["prediction_patient_count"], "prediction patient count")
        != expected_roster["patient_count"]
        or len(identities) != expected_roster["recording_count"]
        or len(patients) != expected_roster["patient_count"]
        or data["prediction_row_roster_sha256"] != _canonical_sha256(normalized)
    ):
        raise ValueError("prediction-first inventory denominator/hash drifted")
    return data


def build_detector_selected_epoch_freeze_v1(
    *,
    provider_id: str,
    detector_variant_id: str,
    outer_fold_id: int,
    selection_checkpoint_inventory_receipt_sha256: str,
    prediction_first_inventory_receipt_sha256: str,
    inner_validation_actual_exposure_receipt_sha256: str,
    candidate_epochs: Sequence[int],
    metric_inventory: Sequence[Mapping[str, Any]],
    selected_checkpoint_file_sha256: str,
) -> dict[str, Any]:
    epochs = [
        _positive_int(epoch, "candidate epoch") for epoch in candidate_epochs
    ]
    if not epochs or epochs != sorted(set(epochs)):
        raise ValueError("candidate epochs must be a non-empty sorted unique array")
    metrics: list[dict[str, Any]] = []
    for index, raw in enumerate(metric_inventory):
        row = _strict_dict(raw, {"epoch", "value"}, f"epoch metric {index}")
        epoch = _positive_int(row["epoch"], f"epoch metric {index} epoch")
        value = row["value"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("epoch metric value must be numeric")
        metric = float(value)
        if not math.isfinite(metric) or metric < 0:
            raise ValueError("epoch metric value must be finite and nonnegative")
        metrics.append({"epoch": epoch, "value": metric})
    metrics.sort(key=lambda row: row["epoch"])
    if [row["epoch"] for row in metrics] != epochs:
        raise ValueError("metric inventory must cover every candidate epoch exactly once")
    selected_epoch = min(metrics, key=lambda row: (row["value"], row["epoch"]))["epoch"]
    return _seal(
        {
            "schema_version": DETECTOR_SELECTED_EPOCH_FREEZE_SCHEMA_V1,
            "provider_id": _identifier(provider_id, "selected-epoch provider"),
            "detector_variant_id": _identifier(detector_variant_id, "selected-epoch variant"),
            "outer_fold_id": _outer_fold_id(outer_fold_id),
            "selection_checkpoint_inventory_receipt_sha256": _sha256(
                selection_checkpoint_inventory_receipt_sha256, "checkpoint receipt"
            ),
            "prediction_first_inventory_receipt_sha256": _sha256(
                prediction_first_inventory_receipt_sha256, "prediction receipt"
            ),
            "inner_validation_actual_exposure_receipt_sha256": _sha256(
                inner_validation_actual_exposure_receipt_sha256, "inner exposure receipt"
            ),
            "candidate_epochs": epochs,
            "metric_name": "patient_macro_dense_detection_loss",
            "metric_inventory": metrics,
            "metric_inventory_sha256": _canonical_sha256(metrics),
            "selection_rule": "minimum_metric_then_earliest_epoch",
            "selected_epoch": selected_epoch,
            "selected_checkpoint_file_sha256": _sha256(
                selected_checkpoint_file_sha256, "selected checkpoint"
            ),
            "frozen_after_complete_inner_validation_metric_inventory": True,
            "outer_heldout_reference_access_count": 0,
            "source_dev_reference_access_count": 0,
            "source_eval_reference_access_count": 0,
            "private_reference_access_count": 0,
        }
    )


_SELECTED_EPOCH_FIELDS = {
    "schema_version",
    "provider_id",
    "detector_variant_id",
    "outer_fold_id",
    "selection_checkpoint_inventory_receipt_sha256",
    "prediction_first_inventory_receipt_sha256",
    "inner_validation_actual_exposure_receipt_sha256",
    "candidate_epochs",
    "metric_name",
    "metric_inventory",
    "metric_inventory_sha256",
    "selection_rule",
    "selected_epoch",
    "selected_checkpoint_file_sha256",
    "frozen_after_complete_inner_validation_metric_inventory",
    "outer_heldout_reference_access_count",
    "source_dev_reference_access_count",
    "source_eval_reference_access_count",
    "private_reference_access_count",
}


def _validate_selected_epoch(
    value: object,
    *,
    provider_id: str,
    variant_id: str,
    outer_fold_id: int,
    checkpoint: Mapping[str, Any],
    prediction: Mapping[str, Any],
    inner_exposure: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    outer_fold_id = _outer_fold_id(outer_fold_id)
    data = _validate_self_hash(value, _SELECTED_EPOCH_FIELDS, "selected-epoch freeze")
    epochs = [row["epoch"] for row in candidates]
    observed_candidate_epochs = [
        _positive_int(epoch, "selected-epoch candidate")
        for epoch in data["candidate_epochs"]
    ] if type(data["candidate_epochs"]) is list else []
    if (
        data["schema_version"] != DETECTOR_SELECTED_EPOCH_FREEZE_SCHEMA_V1
        or data["provider_id"] != provider_id
        or data["detector_variant_id"] != variant_id
        or _outer_fold_id(data["outer_fold_id"], "selected-epoch outer fold")
        != outer_fold_id
        or data["selection_checkpoint_inventory_receipt_sha256"] != checkpoint["receipt_sha256"]
        or data["prediction_first_inventory_receipt_sha256"] != prediction["receipt_sha256"]
        or data["inner_validation_actual_exposure_receipt_sha256"] != inner_exposure["receipt_sha256"]
        or observed_candidate_epochs != epochs
        or data["metric_name"] != "patient_macro_dense_detection_loss"
        or data["selection_rule"] != "minimum_metric_then_earliest_epoch"
        or data["frozen_after_complete_inner_validation_metric_inventory"] is not True
    ):
        raise ValueError("selected-epoch freeze lineage or rule drifted")
    raw_metrics = data["metric_inventory"]
    if type(raw_metrics) is not list or len(raw_metrics) != len(epochs):
        raise ValueError("selected-epoch metric denominator drifted")
    metrics: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_metrics):
        row = _strict_dict(raw, {"epoch", "value"}, f"epoch metric {index}")
        if type(row["epoch"]) is not int or row["epoch"] != epochs[index]:
            raise ValueError("selected-epoch metric roster drifted")
        if isinstance(row["value"], bool) or not isinstance(row["value"], (int, float)):
            raise ValueError("selected-epoch metric must be numeric")
        row["value"] = float(row["value"])
        if not math.isfinite(row["value"]) or row["value"] < 0:
            raise ValueError("selected-epoch metric must be finite and nonnegative")
        metrics.append(row)
    if metrics != raw_metrics or data["metric_inventory_sha256"] != _canonical_sha256(metrics):
        raise ValueError("selected-epoch metric inventory hash/canonicalization drifted")
    selected = min(metrics, key=lambda row: (row["value"], row["epoch"]))["epoch"]
    checkpoint_hash = {
        row["epoch"]: row["checkpoint_file_sha256"] for row in candidates
    }[selected]
    if data["selected_epoch"] != selected or data["selected_checkpoint_file_sha256"] != checkpoint_hash:
        raise ValueError("selected epoch/checkpoint does not replay from the metric inventory")
    for field in (
        "outer_heldout_reference_access_count",
        "source_dev_reference_access_count",
        "source_eval_reference_access_count",
        "private_reference_access_count",
    ):
        _zero_int(data[field], f"selected-epoch freeze {field}")
    return data


def build_detector_final_refit_prerequisite_v1(
    *,
    provider_id: str,
    detector_variant_id: str,
    outer_fold_id: int,
    authorized_fold_ids: Sequence[int],
    authorized_roster: Mapping[str, Any],
    selected_epoch_freeze_receipt_sha256: str,
    planned_epoch_count: int,
    architecture_code_sha256: str,
    training_code_sha256: str,
    preprocessing_fit_artifact_sha256: str,
    random_seed_and_rng_state_receipt_sha256: str,
) -> dict[str, Any]:
    return _seal(
        {
            "schema_version": DETECTOR_FINAL_REFIT_PREREQUISITE_SCHEMA_V1,
            "provider_id": _identifier(provider_id, "final-refit provider"),
            "detector_variant_id": _identifier(detector_variant_id, "final-refit variant"),
            "outer_fold_id": _outer_fold_id(outer_fold_id),
            "training_phase": "final_refit",
            "authorized_fold_ids": list(authorized_fold_ids),
            "authorized_roster": deepcopy(dict(authorized_roster)),
            "selected_epoch_freeze_receipt_sha256": _sha256(
                selected_epoch_freeze_receipt_sha256, "selected-epoch receipt"
            ),
            "planned_epoch_count": _positive_int(
                planned_epoch_count, "planned final-refit epoch count"
            ),
            "reinitialize_from_scratch": True,
            "selection_checkpoint_used_as_initializer": False,
            "final_refit_training_started": False,
            "architecture_code_sha256": _sha256(architecture_code_sha256, "architecture code"),
            "training_code_sha256": _sha256(training_code_sha256, "training code"),
            "preprocessing_fit_artifact_sha256": _sha256(
                preprocessing_fit_artifact_sha256, "preprocessing fit artifact"
            ),
            "random_seed_and_rng_state_receipt_sha256": _sha256(
                random_seed_and_rng_state_receipt_sha256, "final-refit RNG receipt"
            ),
            "outer_heldout_reference_access_count": 0,
            "source_dev_reference_access_count": 0,
            "source_eval_reference_access_count": 0,
            "private_reference_access_count": 0,
        }
    )


_FINAL_REFIT_FIELDS = {
    "schema_version",
    "provider_id",
    "detector_variant_id",
    "outer_fold_id",
    "training_phase",
    "authorized_fold_ids",
    "authorized_roster",
    "selected_epoch_freeze_receipt_sha256",
    "planned_epoch_count",
    "reinitialize_from_scratch",
    "selection_checkpoint_used_as_initializer",
    "final_refit_training_started",
    "architecture_code_sha256",
    "training_code_sha256",
    "preprocessing_fit_artifact_sha256",
    "random_seed_and_rng_state_receipt_sha256",
    "outer_heldout_reference_access_count",
    "source_dev_reference_access_count",
    "source_eval_reference_access_count",
    "private_reference_access_count",
}


def _validate_final_refit(
    value: object,
    *,
    provider_id: str,
    variant_id: str,
    outer_fold_id: int,
    expected_fold_ids: Sequence[int],
    expected_roster: Mapping[str, Any],
    selected_epoch: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    outer_fold_id = _outer_fold_id(outer_fold_id)
    data = _validate_self_hash(value, _FINAL_REFIT_FIELDS, "final-refit prerequisite")
    if (
        data["schema_version"] != DETECTOR_FINAL_REFIT_PREREQUISITE_SCHEMA_V1
        or data["provider_id"] != provider_id
        or data["detector_variant_id"] != variant_id
        or _outer_fold_id(data["outer_fold_id"], "final-refit outer fold")
        != outer_fold_id
        or data["training_phase"] != "final_refit"
        or _validate_fold_ids(data["authorized_fold_ids"], "final-refit folds")
        != list(expected_fold_ids)
        or _validate_roster(data["authorized_roster"], "final-refit roster")
        != dict(expected_roster)
        or data["selected_epoch_freeze_receipt_sha256"] != selected_epoch["receipt_sha256"]
        or _positive_int(
            data["planned_epoch_count"], "planned final-refit epoch count"
        )
        != _positive_int(selected_epoch["selected_epoch"], "selected epoch")
        or data["reinitialize_from_scratch"] is not True
        or data["selection_checkpoint_used_as_initializer"] is not False
        or data["final_refit_training_started"] is not False
    ):
        raise PermissionError("final-refit prerequisite lineage/reinitialization drifted")
    for field in ("architecture_code_sha256", "training_code_sha256", "preprocessing_fit_artifact_sha256"):
        if data[field] != checkpoint[field]:
            raise ValueError(f"final-refit {field} differs from selection architecture/transform")
    _sha256(data["random_seed_and_rng_state_receipt_sha256"], "final-refit RNG receipt")
    for field in (
        "outer_heldout_reference_access_count",
        "source_dev_reference_access_count",
        "source_eval_reference_access_count",
        "private_reference_access_count",
    ):
        _zero_int(data[field], f"final-refit prerequisite {field}")
    return data


_GATE_FIELDS = {
    "schema_version",
    "artifact_id",
    "gate_type",
    "registry_id",
    "registry_receipt_sha256",
    "fold_plan_receipt_sha256",
    "provider_id",
    "detector_variant_id",
    "outer_fold_id",
    "opens_phase",
    "authorized_fold_ids",
    "authorized_roster",
    "prerequisites",
    "scope_receipt",
}
_INNER_PREREQUISITE_FIELDS = {
    "selection_fit_actual_exposure_receipt",
    "selection_checkpoint_inventory_receipt",
    "prediction_first_inventory_receipt",
}
_FINAL_PREREQUISITE_FIELDS = _INNER_PREREQUISITE_FIELDS | {
    "inner_validation_actual_exposure_receipt",
    "selected_epoch_freeze_receipt",
    "final_refit_prerequisite_receipt",
}
_GATE_SCOPE = {
    "external_trust_anchor_required": True,
    "checkpoint_artifact_bytes_replay_required": True,
    "complete_prediction_first_inventory_required": True,
    "actual_prior_phase_exposure_receipts_required": True,
    "reference_sidecars_opened_during_gate_replay": False,
    "outer_heldout_reference_access_authorized": False,
    "source_dev_or_eval_reference_access_authorized": False,
    "private_reference_access_authorized": False,
}


def build_detector_reference_phase_gate_artifact_v1(
    *,
    artifact_id: str,
    registry_id: str,
    registry_receipt_sha256: str,
    fold_plan_receipt_sha256: str,
    provider_id: str,
    detector_variant_id: str,
    outer_fold_id: int,
    opens_phase: str,
    authorized_fold_ids: Sequence[int],
    authorized_roster: Mapping[str, Any],
    prerequisites: Mapping[str, Any],
) -> dict[str, Any]:
    if opens_phase not in _PHASES:
        raise ValueError("typed phase gate opens only inner_validation/final_refit")
    return _seal(
        {
            "schema_version": DETECTOR_REFERENCE_PHASE_GATE_ARTIFACT_SCHEMA_V1,
            "artifact_id": _identifier(artifact_id, "gate artifact ID"),
            "gate_type": {
                "inner_validation": "selection_checkpoint_and_prediction_first_inventory",
                "final_refit": "selected_epoch_and_from_scratch_final_refit_prerequisites",
            }[opens_phase],
            "registry_id": _identifier(registry_id, "gate registry ID"),
            "registry_receipt_sha256": _sha256(registry_receipt_sha256, "gate registry receipt"),
            "fold_plan_receipt_sha256": _sha256(fold_plan_receipt_sha256, "gate fold-plan receipt"),
            "provider_id": _identifier(provider_id, "gate provider"),
            "detector_variant_id": _identifier(detector_variant_id, "gate variant"),
            "outer_fold_id": _outer_fold_id(outer_fold_id),
            "opens_phase": opens_phase,
            "authorized_fold_ids": list(authorized_fold_ids),
            "authorized_roster": deepcopy(dict(authorized_roster)),
            "prerequisites": deepcopy(dict(prerequisites)),
            "scope_receipt": deepcopy(_GATE_SCOPE),
        }
    )


_TRUST_FIELDS = {
    "schema_version",
    "trust_anchor_id",
    "authority_id",
    "authority_class",
    "registry_id",
    "registry_receipt_sha256",
    "gate_artifact_binding",
    "provider_id",
    "detector_variant_id",
    "outer_fold_id",
    "opens_phase",
    "authorized_fold_ids",
    "authorized_roster",
    "decision",
    "authorized_before_first_reference_open",
}
_GATE_BINDING_FIELDS = {
    "relative_path",
    "file_sha256",
    "file_bytes",
    "schema_version",
    "receipt_sha256",
}


def build_detector_reference_phase_gate_trust_anchor_v1(
    *,
    trust_anchor_id: str,
    authority_id: str,
    registry_id: str,
    registry_receipt_sha256: str,
    gate_artifact_relative_path: str,
    gate_artifact_file_bytes: bytes,
    gate_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    gate = dict(gate_artifact)
    return _seal(
        {
            "schema_version": DETECTOR_REFERENCE_PHASE_GATE_TRUST_ANCHOR_SCHEMA_V1,
            "trust_anchor_id": _identifier(trust_anchor_id, "trust-anchor ID"),
            "authority_id": _identifier(authority_id, "trust-anchor authority"),
            "authority_class": "external_immutable_execution_controller",
            "registry_id": _identifier(registry_id, "trust-anchor registry ID"),
            "registry_receipt_sha256": _sha256(registry_receipt_sha256, "trust-anchor registry receipt"),
            "gate_artifact_binding": {
                "relative_path": _safe_relative_path(
                    gate_artifact_relative_path, "trusted gate artifact path"
                ).as_posix(),
                "file_sha256": hashlib.sha256(gate_artifact_file_bytes).hexdigest(),
                "file_bytes": len(gate_artifact_file_bytes),
                "schema_version": gate["schema_version"],
                "receipt_sha256": gate["receipt_sha256"],
            },
            "provider_id": gate["provider_id"],
            "detector_variant_id": gate["detector_variant_id"],
            "outer_fold_id": _outer_fold_id(
                gate["outer_fold_id"], "trusted gate outer fold"
            ),
            "opens_phase": gate["opens_phase"],
            "authorized_fold_ids": deepcopy(gate["authorized_fold_ids"]),
            "authorized_roster": deepcopy(gate["authorized_roster"]),
            "decision": "authorized_by_external_controller",
            "authorized_before_first_reference_open": True,
        }
    )


_CONTROLLER_SIGNATURE_AUTHORITY_FIELDS = {
    "algorithm",
    "controller_key_id",
    "public_key_hex",
    "private_key_material_in_repository",
    "canonical_signed_message",
    "missing_crypto_dependency_behavior",
}
_CONTROLLER_LEDGER_BODY_FIELDS = {
    "schema_version",
    "ledger_id",
    "registry_id",
    "registry_receipt_sha256",
    "fold_plan_receipt_sha256",
    "controller_key_id",
    "provider_id",
    "detector_variant_id",
    "prediction_contract_id",
    "outer_fold_id",
    "release_phase",
    "authorized_fold_ids",
    "authorized_roster",
    "selection_fit_fold_ids",
    "selection_fit_roster",
    "selection_fit_phase_receipt_sha256",
    "prediction_fold_ids",
    "prediction_roster",
    "architecture_code_sha256",
    "training_code_sha256",
    "preprocessing_fit_artifact_sha256",
    "candidate_epochs",
    "candidate_checkpoints",
    "prediction_artifacts",
    "reference_release_authority",
    "prior_inner_validation_binding",
    "final_refit_prerequisite",
    "scorer_binding",
    "forbidden_access_counts",
    "ledger_body_sha256",
}
_CONTROLLER_SIGNATURE_FIELDS = {"algorithm", "controller_key_id", "signature_hex"}
_CHECKPOINT_LEDGER_ROW_FIELDS = {
    "epoch",
    "checkpoint_relative_path",
    "checkpoint_file_sha256",
    "checkpoint_file_bytes",
}
_PREDICTION_LEDGER_ROW_FIELDS = {
    "analysis_identity_id",
    "local_patient_id",
    "source_edf_relative_path",
    "recording_duration_seconds_fraction",
    "checkpoint_epoch",
    "checkpoint_file_sha256",
    "prediction_artifact_relative_path",
    "prediction_artifact_file_sha256",
    "prediction_artifact_file_bytes",
    "terminal_schema_version",
    "terminal_outcome",
}
_REFERENCE_RELEASE_FIELDS = {
    "authority_id",
    "dataset_split",
    "reference_projection",
    "reference_root_canonical_path_sha256",
    "authorized_reference_relative_path_roster_sha256",
    "authorized_reference_file_count",
    "release_phase",
    "first_reference_open_only_after_pre_reference_receipt",
}
_SCORER_BINDING_V1 = {
    "scorer_id": DETECTOR_SELECTION_SCORER_ID_V1,
    "scorer_version": DETECTOR_SELECTION_SCORER_VERSION_V1,
    "prediction_contract_id": DETECTOR_PREDICTION_CONTRACT_ID_V1,
    "reference_projection": "exact_global_TERM_seiz_intervals_only",
    "primary_target": "seizure_event_detection_on_exact_global_TERM_seiz",
    "patient_macro_event_sensitivity_minimum_fraction": [1, 2],
    "patient_macro_event_duration_coverage_minimum_fraction": [1, 2],
    "patient_macro_false_alarm_events_per_hour_maximum_fraction": [5, 1],
    "patient_macro_false_positive_seconds_per_hour_maximum_fraction": [300, 1],
    "event_hit_rule": (
        "maximum_cardinality_one_to_one_temporal_overlap_then_maximum_overlap_"
        "then_minimum_onset_error"
    ),
    "alarm_fragmentation_semantics": (
        "each_prediction_interval_matches_at_most_one_reference_event_and_all_"
        "unmatched_prediction_intervals_count_as_false_alarms"
    ),
    "constraint_rule": (
        "all_preregistered_sensitivity_coverage_false_alarm_count_and_"
        "false_positive_duration_constraints_required_for_promotion_qualification"
    ),
    "research_epoch_order": [
        "promotion_qualification_tier_first_but_never_null_research_selection",
        "maximum_patient_macro_event_sensitivity",
        "maximum_patient_macro_event_duration_coverage",
        "minimum_patient_macro_false_alarm_events_per_hour",
        "minimum_patient_macro_false_positive_seconds_per_hour",
        "minimum_patient_macro_onset_absolute_error_seconds",
        "minimum_patient_macro_interval_symmetric_difference_fraction",
        "earlier_epoch",
    ],
    "promotion_qualified_epoch_order": [
        "maximum_patient_macro_event_sensitivity",
        "maximum_patient_macro_event_duration_coverage",
        "minimum_patient_macro_false_alarm_events_per_hour",
        "minimum_patient_macro_false_positive_seconds_per_hour",
        "minimum_patient_macro_onset_absolute_error_seconds",
        "minimum_patient_macro_interval_symmetric_difference_fraction",
        "earlier_epoch",
    ],
    "record_to_patient_aggregation": "exact_sum_denominators_then_patient_metric",
    "patient_to_epoch_aggregation": "arithmetic_macro_mean_over_exact_patient_denominators",
    "technical_failure_semantics": "all_reference_events_missed_full_duration_false_positive_penalty",
    "partial_coverage_semantics": "all_reference_events_missed_full_duration_false_positive_penalty",
    "zero_prediction_semantics": "valid_empty_interval_set",
    "interval_symmetric_difference_role": "secondary_not_primary",
    "selection_rule": (
        "always_select_research_epoch_with_qualification_tier_then_patient_macro_"
        "event_sensitivity_and_false_alarm_lexicographic_v1"
    ),
}
_FORBIDDEN_ACCESS_COUNTS = {
    "outer_heldout_reference_access_count": 0,
    "source_dev_reference_access_count": 0,
    "source_eval_reference_access_count": 0,
    "private_reference_access_count": 0,
    "reference_access_count_before_prediction_freeze": 0,
}
_PREDICTION_ARTIFACT_FIELDS = {
    "schema_version",
    "prediction_contract_id",
    "provider_id",
    "detector_variant_id",
    "outer_fold_id",
    "checkpoint_epoch",
    "checkpoint_file_sha256",
    "analysis_identity_id",
    "local_patient_id",
    "source_edf_relative_path",
    "recording_duration_seconds_fraction",
    "terminal",
    "reference_access_count_before_terminal",
}
_TERMINAL_FIELDS = {
    "outcome",
    "evaluated_support_intervals",
    "predicted_seizure_intervals",
    "failure_code",
}
_INTERVAL_FIELDS = {"start_seconds_fraction", "stop_seconds_fraction"}
_TYPED_TERMINAL_OUTCOMES = {
    "success",
    "zero_prediction",
    "technical_failure",
    "partial",
}


def _fraction_v1(
    value: object, context: str, *, positive: bool = False
) -> Fraction:
    if (
        type(value) is not list
        or len(value) != 2
        or type(value[0]) is not int
        or type(value[1]) is not int
        or value[1] <= 0
    ):
        raise ValueError(f"{context} must be a reduced rational pair")
    result = Fraction(value[0], value[1])
    if [result.numerator, result.denominator] != value:
        raise ValueError(f"{context} must be reduced")
    if (positive and result <= 0) or (not positive and result < 0):
        raise ValueError(f"{context} has invalid sign")
    return result


def _fraction_json_v1(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


def _lower_hex(value: object, byte_count: int, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != byte_count * 2
        or set(value).difference(_SHA256_ALPHABET)
    ):
        raise ValueError(f"{context} must be {byte_count}-byte lowercase hex")
    return value


def _canonical_root_path_sha256(root_value: Path, context: str) -> str:
    root_input = Path(root_value)
    if root_input.is_symlink():
        raise ValueError(f"{context} root must not be a symlink")
    root = root_input.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"{context} root must be a directory")
    return hashlib.sha256(str(root).encode("utf-8")).hexdigest()


def _reference_relative_path_v1(source_edf_relative_path: object) -> str:
    value = _identifier(source_edf_relative_path, "source EDF relative path")
    if "\\" in value:
        raise ValueError("source EDF relative path must use POSIX separators")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or path.parts[0] != "train"
        or "." in path.parts
        or ".." in path.parts
        or path.suffix.lower() != ".edf"
    ):
        raise PermissionError("source EDF path is outside public source-train")
    return path.with_suffix(".csv_bi").as_posix()


def validate_detector_controller_signature_authority_v1(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    authority = _strict_dict(
        value,
        _CONTROLLER_SIGNATURE_AUTHORITY_FIELDS,
        "controller signature authority",
    )
    if (
        authority["algorithm"] != DETECTOR_CONTROLLER_SIGNATURE_ALGORITHM_V1
        or authority["private_key_material_in_repository"] is not False
        or authority["canonical_signed_message"]
        != "canonical_utf8_json_of_ledger_body_including_ledger_body_sha256"
        or authority["missing_crypto_dependency_behavior"] != "fail_closed"
    ):
        raise PermissionError("controller signature authority policy drifted")
    _identifier(authority["controller_key_id"], "controller key ID")
    public_hex = _lower_hex(authority["public_key_hex"], 32, "Ed25519 public key")
    if Ed25519PublicKey is None:
        raise PermissionError("cryptography Ed25519 support is unavailable; gate fails closed")
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_hex))
    except ValueError as error:
        raise ValueError("controller Ed25519 public key is invalid") from error
    return authority


def detector_controller_ledger_signing_bytes_v1(
    ledger_body: Mapping[str, Any],
) -> bytes:
    body = _strict_dict(
        ledger_body, _CONTROLLER_LEDGER_BODY_FIELDS, "controller ledger body"
    )
    observed = _sha256(body["ledger_body_sha256"], "controller ledger body")
    expected = _canonical_sha256(
        {key: item for key, item in body.items() if key != "ledger_body_sha256"}
    )
    if observed != expected:
        raise ValueError("controller ledger body does not replay")
    return _canonical_json_bytes(body)


def build_detector_reference_release_authority_v1(
    *,
    authority_id: str,
    reference_root: Path,
    release_phase: str,
    authorized_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if release_phase not in _PHASES:
        raise ValueError("controller release phase is invalid")
    paths = sorted(
        _reference_relative_path_v1(row.get("source_edf_relative_path", row.get("local_edf_path")))
        for row in authorized_rows
    )
    if len(paths) != len(set(paths)) or not paths:
        raise ValueError("authorized reference path roster is empty or duplicated")
    return {
        "authority_id": _identifier(authority_id, "reference release authority ID"),
        "dataset_split": "public_TUSZ_source_train_only",
        "reference_projection": "exact_global_TERM_seiz_intervals_only",
        "reference_root_canonical_path_sha256": _canonical_root_path_sha256(
            reference_root, "reference release"
        ),
        "authorized_reference_relative_path_roster_sha256": _canonical_sha256(paths),
        "authorized_reference_file_count": len(paths),
        "release_phase": release_phase,
        "first_reference_open_only_after_pre_reference_receipt": True,
    }


def build_detector_prediction_terminal_artifact_v1(
    *,
    provider_id: str,
    detector_variant_id: str,
    outer_fold_id: int,
    checkpoint_epoch: int,
    checkpoint_file_sha256: str,
    analysis_identity_id: str,
    local_patient_id: str,
    source_edf_relative_path: str,
    recording_duration_seconds_fraction: Sequence[int],
    outcome: str,
    evaluated_support_intervals: Sequence[Mapping[str, Any]],
    predicted_seizure_intervals: Sequence[Mapping[str, Any]],
    failure_code: str | None,
) -> dict[str, Any]:
    artifact = _seal(
        {
            "schema_version": DETECTOR_PREDICTION_TERMINAL_ARTIFACT_SCHEMA_V1,
            "prediction_contract_id": DETECTOR_PREDICTION_CONTRACT_ID_V1,
            "provider_id": _identifier(provider_id, "prediction provider"),
            "detector_variant_id": _identifier(detector_variant_id, "prediction variant"),
            "outer_fold_id": _outer_fold_id(outer_fold_id),
            "checkpoint_epoch": _positive_int(checkpoint_epoch, "prediction epoch"),
            "checkpoint_file_sha256": _sha256(
                checkpoint_file_sha256, "prediction checkpoint"
            ),
            "analysis_identity_id": _identifier(
                analysis_identity_id, "prediction analysis identity"
            ),
            "local_patient_id": _identifier(local_patient_id, "prediction patient"),
            "source_edf_relative_path": _identifier(
                source_edf_relative_path, "prediction EDF path"
            ),
            "recording_duration_seconds_fraction": list(
                recording_duration_seconds_fraction
            ),
            "terminal": {
                "outcome": outcome,
                "evaluated_support_intervals": [deepcopy(dict(row)) for row in evaluated_support_intervals],
                "predicted_seizure_intervals": [deepcopy(dict(row)) for row in predicted_seizure_intervals],
                "failure_code": failure_code,
            },
            "reference_access_count_before_terminal": 0,
        }
    )
    _validate_prediction_terminal_artifact_v1(artifact)
    return artifact


def _normalize_intervals_v1(
    value: object,
    *,
    duration: Fraction,
    context: str,
    allow_empty: bool,
) -> tuple[list[dict[str, list[int]]], list[tuple[Fraction, Fraction]]]:
    if type(value) is not list or (not allow_empty and not value):
        raise ValueError(f"{context} must be an {'optionally empty' if allow_empty else 'non-empty'} array")
    normalized: list[dict[str, list[int]]] = []
    fractions: list[tuple[Fraction, Fraction]] = []
    previous_stop = Fraction(-1, 1)
    for index, raw in enumerate(value):
        row = _strict_dict(raw, _INTERVAL_FIELDS, f"{context}[{index}]")
        start = _fraction_v1(row["start_seconds_fraction"], f"{context}[{index}] start")
        stop = _fraction_v1(row["stop_seconds_fraction"], f"{context}[{index}] stop", positive=True)
        if start < 0 or stop <= start or stop > duration or start < previous_stop:
            raise ValueError(f"{context} is outside duration, overlapping, or unsorted")
        previous_stop = stop
        normalized.append(
            {
                "start_seconds_fraction": _fraction_json_v1(start),
                "stop_seconds_fraction": _fraction_json_v1(stop),
            }
        )
        fractions.append((start, stop))
    if normalized != value:
        raise ValueError(f"{context} is not canonically reduced")
    return normalized, fractions


def _interval_is_covered_v1(
    interval: tuple[Fraction, Fraction], supports: Sequence[tuple[Fraction, Fraction]]
) -> bool:
    return any(left <= interval[0] and interval[1] <= right for left, right in supports)


def _validate_prediction_terminal_artifact_v1(value: object) -> dict[str, Any]:
    data = _validate_self_hash(
        value,
        _PREDICTION_ARTIFACT_FIELDS,
        "typed prediction terminal artifact",
    )
    if (
        data["schema_version"] != DETECTOR_PREDICTION_TERMINAL_ARTIFACT_SCHEMA_V1
        or data["prediction_contract_id"] != DETECTOR_PREDICTION_CONTRACT_ID_V1
    ):
        raise PermissionError("unsupported prediction provider contract")
    _identifier(data["provider_id"], "prediction provider")
    _identifier(data["detector_variant_id"], "prediction variant")
    _outer_fold_id(data["outer_fold_id"], "prediction outer fold")
    _positive_int(data["checkpoint_epoch"], "prediction epoch")
    _sha256(data["checkpoint_file_sha256"], "prediction checkpoint")
    _identifier(data["analysis_identity_id"], "prediction identity")
    _identifier(data["local_patient_id"], "prediction patient")
    _identifier(data["source_edf_relative_path"], "prediction source path")
    duration = _fraction_v1(
        data["recording_duration_seconds_fraction"],
        "prediction recording duration",
        positive=True,
    )
    terminal = _strict_dict(data["terminal"], _TERMINAL_FIELDS, "prediction terminal")
    outcome = terminal["outcome"]
    if outcome not in _TYPED_TERMINAL_OUTCOMES:
        raise ValueError("prediction terminal outcome is invalid")
    _, support = _normalize_intervals_v1(
        terminal["evaluated_support_intervals"],
        duration=duration,
        context="prediction evaluated support",
        allow_empty=True,
    )
    _, predicted = _normalize_intervals_v1(
        terminal["predicted_seizure_intervals"],
        duration=duration,
        context="predicted seizure intervals",
        allow_empty=True,
    )
    full_support = support == [(Fraction(0, 1), duration)]
    if any(not _interval_is_covered_v1(interval, support) for interval in predicted):
        raise ValueError("predicted seizure interval is outside evaluated support")
    failure_code = terminal["failure_code"]
    if outcome == "success":
        if not full_support or not predicted or failure_code is not None:
            raise ValueError("success terminal requires full support and non-empty predictions")
    elif outcome == "zero_prediction":
        if not full_support or predicted or failure_code is not None:
            raise ValueError("zero-prediction terminal must be full-support and empty")
    elif outcome == "technical_failure":
        if predicted or not isinstance(failure_code, str):
            raise ValueError("technical-failure terminal must be typed and prediction-free")
        _identifier(failure_code, "technical failure code")
    else:
        if full_support or not support or not isinstance(failure_code, str):
            raise ValueError("partial terminal must identify incomplete non-empty support")
        _identifier(failure_code, "partial terminal code")
    _zero_int(
        data["reference_access_count_before_terminal"],
        "prediction reference access before terminal",
    )
    return data


def build_detector_controller_ledger_body_v1(
    *,
    ledger_id: str,
    registry_id: str,
    registry_receipt_sha256: str,
    fold_plan_receipt_sha256: str,
    controller_key_id: str,
    provider_id: str,
    detector_variant_id: str,
    outer_fold_id: int,
    release_phase: str,
    authorized_fold_ids: Sequence[int],
    authorized_roster: Mapping[str, Any],
    selection_fit_fold_ids: Sequence[int],
    selection_fit_roster: Mapping[str, Any],
    selection_fit_phase_receipt_sha256: str,
    prediction_fold_ids: Sequence[int],
    prediction_roster: Mapping[str, Any],
    architecture_code_sha256: str,
    training_code_sha256: str,
    preprocessing_fit_artifact_sha256: str,
    candidate_checkpoints: Sequence[Mapping[str, Any]],
    prediction_artifacts: Sequence[Mapping[str, Any]],
    reference_release_authority: Mapping[str, Any],
    prior_inner_validation_binding: Mapping[str, Any] | None = None,
    final_refit_prerequisite: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if release_phase not in _PHASES:
        raise ValueError("controller ledger release phase is invalid")
    candidates = sorted(
        [deepcopy(dict(row)) for row in candidate_checkpoints],
        key=lambda row: row["epoch"],
    )
    epochs = [row["epoch"] for row in candidates]
    predictions = sorted(
        [deepcopy(dict(row)) for row in prediction_artifacts],
        key=lambda row: (row["checkpoint_epoch"], row["analysis_identity_id"]),
    )
    body: dict[str, Any] = {
        "schema_version": DETECTOR_CONTROLLER_LEDGER_SCHEMA_V1,
        "ledger_id": _identifier(ledger_id, "controller ledger ID"),
        "registry_id": _identifier(registry_id, "controller ledger registry"),
        "registry_receipt_sha256": _sha256(registry_receipt_sha256, "registry receipt"),
        "fold_plan_receipt_sha256": _sha256(fold_plan_receipt_sha256, "fold-plan receipt"),
        "controller_key_id": _identifier(controller_key_id, "controller key ID"),
        "provider_id": _identifier(provider_id, "ledger provider"),
        "detector_variant_id": _identifier(detector_variant_id, "ledger detector variant"),
        "prediction_contract_id": DETECTOR_PREDICTION_CONTRACT_ID_V1,
        "outer_fold_id": _outer_fold_id(outer_fold_id),
        "release_phase": release_phase,
        "authorized_fold_ids": list(authorized_fold_ids),
        "authorized_roster": deepcopy(dict(authorized_roster)),
        "selection_fit_fold_ids": list(selection_fit_fold_ids),
        "selection_fit_roster": deepcopy(dict(selection_fit_roster)),
        "selection_fit_phase_receipt_sha256": _sha256(
            selection_fit_phase_receipt_sha256, "selection phase receipt"
        ),
        "prediction_fold_ids": list(prediction_fold_ids),
        "prediction_roster": deepcopy(dict(prediction_roster)),
        "architecture_code_sha256": _sha256(architecture_code_sha256, "architecture code"),
        "training_code_sha256": _sha256(training_code_sha256, "training code"),
        "preprocessing_fit_artifact_sha256": _sha256(
            preprocessing_fit_artifact_sha256, "preprocessing fit artifact"
        ),
        "candidate_epochs": epochs,
        "candidate_checkpoints": candidates,
        "prediction_artifacts": predictions,
        "reference_release_authority": deepcopy(dict(reference_release_authority)),
        "prior_inner_validation_binding": (
            None
            if prior_inner_validation_binding is None
            else deepcopy(dict(prior_inner_validation_binding))
        ),
        "final_refit_prerequisite": (
            None
            if final_refit_prerequisite is None
            else deepcopy(dict(final_refit_prerequisite))
        ),
        "scorer_binding": deepcopy(_SCORER_BINDING_V1),
        "forbidden_access_counts": deepcopy(_FORBIDDEN_ACCESS_COUNTS),
        "ledger_body_sha256": "CONTENT-ADDRESS-PENDING",
    }
    if release_phase == "inner_validation":
        if prior_inner_validation_binding is not None or final_refit_prerequisite is not None:
            raise ValueError("inner-validation ledger cannot contain final-refit prerequisites")
    elif prior_inner_validation_binding is None or final_refit_prerequisite is None:
        raise ValueError("final-refit ledger requires signed prior-phase and refit bindings")
    body["ledger_body_sha256"] = _canonical_sha256(
        {key: item for key, item in body.items() if key != "ledger_body_sha256"}
    )
    detector_controller_ledger_signing_bytes_v1(body)
    return body


def _expected_prediction_rows_v1(
    rows: Sequence[Mapping[str, Any]], context: str
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        duration = _fraction_v1(
            raw.get("recording_duration_seconds_fraction"),
            f"{context}[{index}] duration",
            positive=True,
        )
        result.append(
            {
                "analysis_identity_id": _identifier(
                    raw.get("analysis_identity_id"), f"{context}[{index}] identity"
                ),
                "local_patient_id": _identifier(
                    raw.get("local_patient_id"), f"{context}[{index}] patient"
                ),
                "source_edf_relative_path": _identifier(
                    raw.get("source_edf_relative_path", raw.get("local_edf_path")),
                    f"{context}[{index}] EDF path",
                ),
                "recording_duration_seconds_fraction": _fraction_json_v1(duration),
            }
        )
    result.sort(key=lambda row: (row["analysis_identity_id"], row["source_edf_relative_path"]))
    if len({row["analysis_identity_id"] for row in result}) != len(result):
        raise ValueError(f"{context} contains duplicate identities")
    return result


def _validate_reference_release_v1(
    value: object,
    *,
    reference_root: Path,
    release_phase: str,
    authorized_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    release = _strict_dict(value, _REFERENCE_RELEASE_FIELDS, "reference release authority")
    expected_paths = sorted(
        _reference_relative_path_v1(row.get("source_edf_relative_path", row.get("local_edf_path")))
        for row in authorized_rows
    )
    if (
        release["dataset_split"] != "public_TUSZ_source_train_only"
        or release["reference_projection"] != "exact_global_TERM_seiz_intervals_only"
        or release["reference_root_canonical_path_sha256"]
        != _canonical_root_path_sha256(reference_root, "reference release")
        or release["authorized_reference_relative_path_roster_sha256"]
        != _canonical_sha256(expected_paths)
        or release["authorized_reference_file_count"] != len(expected_paths)
        or release["release_phase"] != release_phase
        or release["first_reference_open_only_after_pre_reference_receipt"] is not True
    ):
        raise PermissionError("controller reference-root/release authority drifted")
    _identifier(release["authority_id"], "reference release authority ID")
    return release


@dataclass(frozen=True)
class DetectorReferenceGateReplayV1:
    proof: dict[str, Any]
    prediction_artifacts: tuple[dict[str, Any], ...]
    checkpoint_hash_by_epoch: dict[int, str]


def _verify_controller_signature_v1(
    *,
    ledger: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    signature_authority = validate_detector_controller_signature_authority_v1(authority)
    signed = _strict_dict(
        ledger,
        _CONTROLLER_LEDGER_BODY_FIELDS | {"controller_signature"},
        "signed controller ledger",
    )
    body = {key: deepcopy(item) for key, item in signed.items() if key != "controller_signature"}
    message = detector_controller_ledger_signing_bytes_v1(body)
    signature = _strict_dict(
        signed["controller_signature"],
        _CONTROLLER_SIGNATURE_FIELDS,
        "controller ledger signature",
    )
    if (
        signature["algorithm"] != DETECTOR_CONTROLLER_SIGNATURE_ALGORITHM_V1
        or signature["controller_key_id"] != signature_authority["controller_key_id"]
        or body["controller_key_id"] != signature_authority["controller_key_id"]
    ):
        raise PermissionError("controller signature key lineage drifted")
    signature_hex = _lower_hex(signature["signature_hex"], 64, "Ed25519 signature")
    if Ed25519PublicKey is None or InvalidSignature is None:
        raise PermissionError("cryptography Ed25519 support is unavailable; gate fails closed")
    verifier = Ed25519PublicKey.from_public_bytes(
        bytes.fromhex(signature_authority["public_key_hex"])
    )
    try:
        verifier.verify(bytes.fromhex(signature_hex), message)
    except InvalidSignature as error:
        raise PermissionError("controller ledger Ed25519 signature is invalid") from error
    return signed


def _validate_final_refit_prerequisite_v1(
    value: object,
    *,
    bundle_root: Path,
    provider_id: str,
    variant_id: str,
    outer_fold_id: int,
    authorized_fold_ids: Sequence[int],
    authorized_roster: Mapping[str, Any],
    metric_receipt: Mapping[str, Any],
    checkpoint_hashes: Mapping[int, str],
    architecture_code_sha256: str,
    training_code_sha256: str,
    preprocessing_fit_artifact_sha256: str,
) -> tuple[dict[str, Any], int]:
    fields = {
        "schema_version",
        "provider_id",
        "detector_variant_id",
        "outer_fold_id",
        "training_phase",
        "authorized_fold_ids",
        "authorized_roster",
        "selected_epoch_metric_receipt_sha256",
        "selected_epoch",
        "planned_epoch_count",
        "reinitialize_from_scratch",
        "selection_checkpoint_used_as_initializer",
        "optimizer_or_rng_resume_state_loaded",
        "final_refit_training_started",
        "fresh_initialization_artifact",
        "architecture_code_sha256",
        "training_code_sha256",
        "preprocessing_fit_artifact_sha256",
        "random_seed_and_rng_state_receipt_sha256",
        "forbidden_access_counts",
    }
    data = _validate_self_hash(value, fields, "typed final-refit prerequisite")
    selected_epoch = _positive_int(metric_receipt.get("selected_epoch"), "recomputed selected epoch")
    if (
        data["schema_version"] != DETECTOR_FINAL_REFIT_TYPED_PREREQUISITE_SCHEMA_V1
        or data["provider_id"] != provider_id
        or data["detector_variant_id"] != variant_id
        or data["outer_fold_id"] != outer_fold_id
        or data["training_phase"] != "final_refit"
        or _validate_fold_ids(data["authorized_fold_ids"], "final-refit folds")
        != list(authorized_fold_ids)
        or _validate_roster(data["authorized_roster"], "final-refit roster")
        != dict(authorized_roster)
        or data["selected_epoch_metric_receipt_sha256"] != metric_receipt["receipt_sha256"]
        or data["selected_epoch"] != selected_epoch
        or data["planned_epoch_count"] != selected_epoch
        or data["reinitialize_from_scratch"] is not True
        or data["selection_checkpoint_used_as_initializer"] is not False
        or data["optimizer_or_rng_resume_state_loaded"] is not False
        or data["final_refit_training_started"] is not False
    ):
        raise PermissionError("final-refit typed metric/exposure/reinitialization lineage drifted")
    if metric_receipt["selected_checkpoint_file_sha256"] != checkpoint_hashes[selected_epoch]:
        raise ValueError("final-refit selected checkpoint differs from recomputed metric")
    for field, expected in (
        ("architecture_code_sha256", architecture_code_sha256),
        ("training_code_sha256", training_code_sha256),
        ("preprocessing_fit_artifact_sha256", preprocessing_fit_artifact_sha256),
    ):
        if data[field] != expected:
            raise ValueError(f"final-refit {field} drifted")
    _sha256(data["random_seed_and_rng_state_receipt_sha256"], "final-refit RNG receipt")
    if data["forbidden_access_counts"] != _FORBIDDEN_ACCESS_COUNTS:
        raise PermissionError("final-refit forbidden access count drifted")
    initial = _strict_dict(
        data["fresh_initialization_artifact"],
        {"relative_path", "file_sha256", "file_bytes"},
        "fresh initialization artifact",
    )
    payload = _read_bundle_bytes(
        bundle_root, initial["relative_path"], "fresh initialization artifact"
    )
    if (
        len(payload) != _positive_int(initial["file_bytes"], "fresh initialization bytes")
        or hashlib.sha256(payload).hexdigest()
        != _sha256(initial["file_sha256"], "fresh initialization hash")
    ):
        raise ValueError("fresh initialization artifact bytes drifted")
    if initial["file_sha256"] in set(checkpoint_hashes.values()):
        raise PermissionError("fresh initialization artifact is a selection checkpoint")
    return data, len(payload)


def replay_detector_reference_phase_gate_v1(
    *,
    bundle_root: Path,
    controller_ledger_relative_path: str,
    controller_signature_authority: Mapping[str, Any],
    registry_id: str,
    registry_receipt_sha256: str,
    fold_plan_receipt_sha256: str,
    outer_fold_id: int,
    opens_phase: str,
    reference_root: Path,
    authorized_fold_ids: Sequence[int],
    authorized_roster: Mapping[str, Any],
    authorized_rows: Sequence[Mapping[str, Any]],
    selection_fit_fold_ids: Sequence[int],
    selection_fit_roster: Mapping[str, Any],
    selection_fit_phase_receipt_sha256: str,
    inner_validation_fold_ids: Sequence[int],
    inner_validation_roster: Mapping[str, Any],
    inner_validation_rows: Sequence[Mapping[str, Any]],
    prior_inner_validation_phase_receipt: Mapping[str, Any] | None = None,
    selected_epoch_metric_receipt: Mapping[str, Any] | None = None,
) -> DetectorReferenceGateReplayV1:
    """Verify signed authority and replay all prediction bytes before reference open."""

    if opens_phase not in _PHASES:
        raise ValueError("phase-gate replay opens only inner_validation/final_refit")
    outer_fold_id = _outer_fold_id(outer_fold_id)
    authorized_fold_ids = _validate_fold_ids(list(authorized_fold_ids), "authorized folds")
    selection_fit_fold_ids = _validate_fold_ids(list(selection_fit_fold_ids), "selection folds")
    inner_validation_fold_ids = _validate_fold_ids(list(inner_validation_fold_ids), "inner folds")
    authorized_roster = _validate_roster(dict(authorized_roster), "authorized roster")
    selection_fit_roster = _validate_roster(dict(selection_fit_roster), "selection roster")
    inner_validation_roster = _validate_roster(dict(inner_validation_roster), "prediction roster")
    expected_predictions = _expected_prediction_rows_v1(
        inner_validation_rows, "inner-validation prediction rows"
    )
    if len(expected_predictions) != inner_validation_roster["recording_count"]:
        raise ValueError("inner-validation prediction denominator drifted")
    if len(_expected_prediction_rows_v1(authorized_rows, "authorized release rows")) != authorized_roster["recording_count"]:
        raise ValueError("authorized release denominator drifted")

    ledger_payload = _read_bundle_bytes(
        bundle_root, controller_ledger_relative_path, "controller-signed ledger"
    )
    ledger = _verify_controller_signature_v1(
        ledger=_load_strict_json(ledger_payload, "controller-signed ledger"),
        authority=controller_signature_authority,
    )
    if (
        ledger["schema_version"] != DETECTOR_CONTROLLER_LEDGER_SCHEMA_V1
        or ledger["registry_id"] != registry_id
        or ledger["registry_receipt_sha256"] != registry_receipt_sha256
        or ledger["fold_plan_receipt_sha256"] != fold_plan_receipt_sha256
        or ledger["outer_fold_id"] != outer_fold_id
        or ledger["release_phase"] != opens_phase
        or ledger["prediction_contract_id"] != DETECTOR_PREDICTION_CONTRACT_ID_V1
        or _validate_fold_ids(ledger["authorized_fold_ids"], "ledger authorized folds")
        != authorized_fold_ids
        or _validate_roster(ledger["authorized_roster"], "ledger authorized roster")
        != authorized_roster
        or _validate_fold_ids(ledger["selection_fit_fold_ids"], "ledger selection folds")
        != selection_fit_fold_ids
        or _validate_roster(ledger["selection_fit_roster"], "ledger selection roster")
        != selection_fit_roster
        or ledger["selection_fit_phase_receipt_sha256"]
        != _sha256(selection_fit_phase_receipt_sha256, "selection phase receipt")
        or _validate_fold_ids(ledger["prediction_fold_ids"], "ledger prediction folds")
        != inner_validation_fold_ids
        or _validate_roster(ledger["prediction_roster"], "ledger prediction roster")
        != inner_validation_roster
        or ledger["scorer_binding"] != _SCORER_BINDING_V1
        or ledger["forbidden_access_counts"] != _FORBIDDEN_ACCESS_COUNTS
    ):
        raise PermissionError("controller ledger fold/provider/reference lineage drifted")
    provider_id = _identifier(ledger["provider_id"], "ledger provider")
    variant_id = _identifier(ledger["detector_variant_id"], "ledger detector variant")
    for field in (
        "architecture_code_sha256",
        "training_code_sha256",
        "preprocessing_fit_artifact_sha256",
    ):
        _sha256(ledger[field], f"ledger {field}")
    release = _validate_reference_release_v1(
        ledger["reference_release_authority"],
        reference_root=reference_root,
        release_phase=opens_phase,
        authorized_rows=authorized_rows,
    )

    raw_candidates = ledger["candidate_checkpoints"]
    if type(raw_candidates) is not list or not raw_candidates:
        raise ValueError("controller ledger checkpoint inventory is empty")
    candidates: list[dict[str, Any]] = []
    checkpoint_hashes: dict[int, str] = {}
    checkpoint_bytes = 0
    checkpoint_paths: set[str] = set()
    for index, raw in enumerate(raw_candidates):
        row = _strict_dict(raw, _CHECKPOINT_LEDGER_ROW_FIELDS, f"ledger checkpoint {index}")
        epoch = _positive_int(row["epoch"], f"ledger checkpoint {index} epoch")
        relative = _safe_relative_path(row["checkpoint_relative_path"], "checkpoint path").as_posix()
        if relative in checkpoint_paths or epoch in checkpoint_hashes:
            raise ValueError("checkpoint path or epoch is duplicated")
        checkpoint_paths.add(relative)
        payload = _read_bundle_bytes(bundle_root, relative, "candidate checkpoint")
        expected_hash = _sha256(row["checkpoint_file_sha256"], "candidate checkpoint hash")
        if (
            len(payload) != _positive_int(row["checkpoint_file_bytes"], "candidate checkpoint bytes")
            or hashlib.sha256(payload).hexdigest() != expected_hash
        ):
            raise ValueError("candidate checkpoint artifact bytes drifted")
        row["checkpoint_relative_path"] = relative
        candidates.append(row)
        checkpoint_hashes[epoch] = expected_hash
        checkpoint_bytes += len(payload)
    candidates.sort(key=lambda row: row["epoch"])
    epochs = [row["epoch"] for row in candidates]
    if ledger["candidate_epochs"] != epochs or epochs != sorted(set(epochs)):
        raise ValueError("candidate epoch inventory drifted")

    expected_by_identity = {row["analysis_identity_id"]: row for row in expected_predictions}
    expected_pairs = {(epoch, identity) for epoch in epochs for identity in expected_by_identity}
    raw_prediction_rows = ledger["prediction_artifacts"]
    if type(raw_prediction_rows) is not list:
        raise ValueError("controller prediction inventory must be an array")
    observed_pairs: set[tuple[int, str]] = set()
    prediction_paths: set[str] = set()
    normalized_artifacts: list[dict[str, Any]] = []
    prediction_bytes = 0
    outcome_counts = {outcome: 0 for outcome in sorted(_TYPED_TERMINAL_OUTCOMES)}
    full_terminal_by_epoch = {epoch: 0 for epoch in epochs}
    ledger_rows: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_prediction_rows):
        row = _strict_dict(raw, _PREDICTION_LEDGER_ROW_FIELDS, f"ledger prediction {index}")
        epoch = _positive_int(row["checkpoint_epoch"], f"ledger prediction {index} epoch")
        identity = _identifier(row["analysis_identity_id"], "ledger prediction identity")
        pair = (epoch, identity)
        if pair in observed_pairs:
            raise ValueError("prediction inventory duplicates an epoch/record pair")
        observed_pairs.add(pair)
        expected = expected_by_identity.get(identity)
        if (
            expected is None
            or row["local_patient_id"] != expected["local_patient_id"]
            or row["source_edf_relative_path"] != expected["source_edf_relative_path"]
            or row["recording_duration_seconds_fraction"]
            != expected["recording_duration_seconds_fraction"]
            or row["checkpoint_file_sha256"] != checkpoint_hashes.get(epoch)
            or row["terminal_schema_version"]
            != DETECTOR_PREDICTION_TERMINAL_ARTIFACT_SCHEMA_V1
            or row["terminal_outcome"] not in _TYPED_TERMINAL_OUTCOMES
        ):
            raise PermissionError("prediction ledger roster/checkpoint/terminal drifted")
        relative = _safe_relative_path(
            row["prediction_artifact_relative_path"], "prediction artifact path"
        ).as_posix()
        if relative in prediction_paths:
            raise ValueError("prediction artifact path is duplicated")
        prediction_paths.add(relative)
        payload = _read_bundle_bytes(bundle_root, relative, "typed prediction artifact")
        expected_hash = _sha256(
            row["prediction_artifact_file_sha256"], "prediction artifact hash"
        )
        if (
            len(payload) != _positive_int(row["prediction_artifact_file_bytes"], "prediction artifact bytes")
            or hashlib.sha256(payload).hexdigest() != expected_hash
        ):
            raise ValueError("prediction artifact actual bytes drifted")
        artifact = _validate_prediction_terminal_artifact_v1(
            _load_strict_json(payload, "typed prediction artifact")
        )
        if (
            artifact["provider_id"] != provider_id
            or artifact["detector_variant_id"] != variant_id
            or artifact["outer_fold_id"] != outer_fold_id
            or artifact["checkpoint_epoch"] != epoch
            or artifact["checkpoint_file_sha256"] != checkpoint_hashes[epoch]
            or artifact["analysis_identity_id"] != identity
            or artifact["local_patient_id"] != expected["local_patient_id"]
            or artifact["source_edf_relative_path"] != expected["source_edf_relative_path"]
            or artifact["recording_duration_seconds_fraction"]
            != expected["recording_duration_seconds_fraction"]
            or artifact["terminal"]["outcome"] != row["terminal_outcome"]
        ):
            raise PermissionError("prediction terminal artifact lineage drifted")
        outcome = artifact["terminal"]["outcome"]
        outcome_counts[outcome] += 1
        if outcome in {"success", "zero_prediction"}:
            full_terminal_by_epoch[epoch] += 1
        normalized_artifacts.append(artifact)
        row["prediction_artifact_relative_path"] = relative
        ledger_rows.append(row)
        prediction_bytes += len(payload)
    ledger_rows.sort(key=lambda row: (row["checkpoint_epoch"], row["analysis_identity_id"]))
    if observed_pairs != expected_pairs or ledger_rows != raw_prediction_rows:
        raise ValueError("prediction inventory is not the complete prediction-first Cartesian denominator")
    if any(count == 0 for count in full_terminal_by_epoch.values()):
        raise PermissionError("candidate epoch has no fully scorable terminal; all-technical/partial inference fails closed")

    prediction_inventory_sha256 = _canonical_sha256(
        [
            {
                "checkpoint_epoch": row["checkpoint_epoch"],
                "analysis_identity_id": row["analysis_identity_id"],
                "prediction_artifact_file_sha256": row["prediction_artifact_file_sha256"],
                "terminal_outcome": row["terminal_outcome"],
            }
            for row in ledger_rows
        ]
    )
    final_prerequisite_sha256: str | None = None
    selected_epoch: int | None = None
    fresh_initialization_bytes = 0
    if opens_phase == "inner_validation":
        if ledger["prior_inner_validation_binding"] is not None or ledger["final_refit_prerequisite"] is not None:
            raise PermissionError("inner-validation ledger contains final-refit authority")
        if prior_inner_validation_phase_receipt is not None or selected_epoch_metric_receipt is not None:
            raise PermissionError("inner-validation replay received forbidden post-reference inputs")
    else:
        if prior_inner_validation_phase_receipt is None or selected_epoch_metric_receipt is None:
            raise PermissionError("final-refit gate requires prior inner receipt and recomputed metric")
        prior_binding = _strict_dict(
            ledger["prior_inner_validation_binding"],
            {
                "phase_receipt_sha256",
                "selection_metric_receipt_sha256",
                "prediction_artifact_inventory_sha256",
            },
            "prior inner-validation binding",
        )
        if (
            prior_binding["phase_receipt_sha256"]
            != prior_inner_validation_phase_receipt.get("receipt_sha256")
            or prior_binding["selection_metric_receipt_sha256"]
            != selected_epoch_metric_receipt.get("receipt_sha256")
            or prior_binding["prediction_artifact_inventory_sha256"]
            != prediction_inventory_sha256
        ):
            raise PermissionError("final-refit prior-phase signed binding drifted")
        validate_detector_selection_metric_receipt_v1(
            selected_epoch_metric_receipt,
            prediction_artifacts=normalized_artifacts,
            checkpoint_hash_by_epoch=checkpoint_hashes,
            prediction_artifact_inventory_sha256=prediction_inventory_sha256,
            reference_records=prior_inner_validation_phase_receipt.get("records"),
        )
        final_prerequisite, initialization_bytes = _validate_final_refit_prerequisite_v1(
            ledger["final_refit_prerequisite"],
            bundle_root=Path(bundle_root),
            provider_id=provider_id,
            variant_id=variant_id,
            outer_fold_id=outer_fold_id,
            authorized_fold_ids=authorized_fold_ids,
            authorized_roster=authorized_roster,
            metric_receipt=selected_epoch_metric_receipt,
            checkpoint_hashes=checkpoint_hashes,
            architecture_code_sha256=ledger["architecture_code_sha256"],
            training_code_sha256=ledger["training_code_sha256"],
            preprocessing_fit_artifact_sha256=ledger["preprocessing_fit_artifact_sha256"],
        )
        final_prerequisite_sha256 = final_prerequisite["receipt_sha256"]
        selected_epoch = selected_epoch_metric_receipt["selected_epoch"]
        fresh_initialization_bytes = initialization_bytes

    pre_reference = _seal(
        {
            "schema_version": DETECTOR_PRE_REFERENCE_RELEASE_RECEIPT_SCHEMA_V1,
            "controller_ledger_body_sha256": ledger["ledger_body_sha256"],
            "controller_ledger_file_sha256": hashlib.sha256(ledger_payload).hexdigest(),
            "controller_signature_verified": True,
            "controller_signature_verified_logical_sequence": 1,
            "checkpoint_and_prediction_actual_byte_replay_complete": True,
            "artifact_byte_replay_complete_logical_sequence": 2,
            "pre_reference_release_receipt_sealed_logical_sequence": 3,
            "candidate_checkpoint_count": len(candidates),
            "candidate_checkpoint_bytes_replayed": checkpoint_bytes,
            "fresh_initialization_artifact_bytes_replayed": fresh_initialization_bytes,
            "prediction_artifact_count": len(normalized_artifacts),
            "prediction_artifact_bytes_replayed": prediction_bytes,
            "prediction_artifact_inventory_sha256": prediction_inventory_sha256,
            "complete_prediction_first_cartesian_denominator": True,
            "typed_terminal_outcome_counts": outcome_counts,
            "reference_bytes_opened_before_receipt": 0,
            "reference_files_opened_before_receipt": 0,
        }
    )
    proof: dict[str, Any] = {
        "gate_type": (
            "controller_signed_prediction_first_inner_validation_release"
            if opens_phase == "inner_validation"
            else "controller_signed_recomputed_epoch_from_scratch_final_refit_release"
        ),
        "gate_validator_id": DETECTOR_REFERENCE_PHASE_GATE_VALIDATOR_ID_V1,
        "gate_validator_source_sha256": detector_reference_phase_gate_source_sha256_v1(),
        "provider_id": provider_id,
        "detector_variant_id": variant_id,
        "outer_fold_id": outer_fold_id,
        "opens_phase": opens_phase,
        "authorized_fold_ids": list(authorized_fold_ids),
        "authorized_roster": deepcopy(dict(authorized_roster)),
        "controller_signed_ledger": deepcopy(ledger),
        "controller_ledger_file_sha256": hashlib.sha256(ledger_payload).hexdigest(),
        "controller_ledger_file_bytes": len(ledger_payload),
        "controller_signature_verified": True,
        "reference_release_authority": release,
        "pre_reference_release_receipt": pre_reference,
        "prediction_artifact_inventory_sha256": prediction_inventory_sha256,
        "selected_epoch_metric_receipt_sha256": (
            None if selected_epoch_metric_receipt is None else selected_epoch_metric_receipt["receipt_sha256"]
        ),
        "selected_epoch": selected_epoch,
        "final_refit_prerequisite_receipt_sha256": final_prerequisite_sha256,
        "reference_sidecars_opened_during_gate_replay": 0,
        "validated_before_first_reference_open": True,
    }
    proof["gate_proof_sha256"] = _canonical_sha256(proof)
    return DetectorReferenceGateReplayV1(
        proof=proof,
        prediction_artifacts=tuple(normalized_artifacts),
        checkpoint_hash_by_epoch=checkpoint_hashes,
    )


def _interval_intersection_duration_v1(
    left: Sequence[tuple[Fraction, Fraction]],
    right: Sequence[tuple[Fraction, Fraction]],
) -> Fraction:
    i = 0
    j = 0
    total = Fraction(0, 1)
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        stop = min(left[i][1], right[j][1])
        if stop > start:
            total += stop - start
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total


def _one_to_one_event_matches_v1(
    predicted: Sequence[tuple[Fraction, Fraction]],
    target: Sequence[tuple[Fraction, Fraction]],
) -> tuple[tuple[int, int], ...]:
    """Return a deterministic maximum-cardinality ordered interval matching."""

    # Both interval rosters are sorted and internally non-overlapping.  An
    # optimal overlap matching therefore has a non-crossing representative;
    # this dynamic program prevents one long alarm from hitting many reference
    # events and prevents fragmented alarms from sharing one reference event.
    empty: tuple[int, Fraction, Fraction, tuple[tuple[int, int], ...]] = (
        0,
        Fraction(0, 1),
        Fraction(0, 1),
        (),
    )
    table: dict[
        tuple[int, int],
        tuple[int, Fraction, Fraction, tuple[tuple[int, int], ...]],
    ] = {}

    def order_key(
        value: tuple[int, Fraction, Fraction, tuple[tuple[int, int], ...]],
    ) -> tuple[object, ...]:
        matches, overlap, onset_error, pairs = value
        return (-matches, -overlap, onset_error, pairs)

    for target_index in range(len(target), -1, -1):
        for prediction_index in range(len(predicted), -1, -1):
            if target_index == len(target) or prediction_index == len(predicted):
                table[(target_index, prediction_index)] = empty
                continue
            candidates = [
                table[(target_index + 1, prediction_index)],
                table[(target_index, prediction_index + 1)],
            ]
            overlap = _interval_intersection_duration_v1(
                [predicted[prediction_index]], [target[target_index]]
            )
            if overlap > 0:
                suffix = table[(target_index + 1, prediction_index + 1)]
                candidates.append(
                    (
                        suffix[0] + 1,
                        suffix[1] + overlap,
                        suffix[2]
                        + abs(
                            predicted[prediction_index][0]
                            - target[target_index][0]
                        ),
                        ((target_index, prediction_index),) + suffix[3],
                    )
                )
            table[(target_index, prediction_index)] = min(
                candidates, key=order_key
            )
    return table[(0, 0)][3]


def _reference_intervals_v1(
    row: Mapping[str, Any], duration: Fraction
) -> list[tuple[Fraction, Fraction]]:
    raw = row.get("seizure_intervals")
    if type(raw) is not list:
        raise ValueError("reference seizure intervals must be an array")
    result: list[tuple[Fraction, Fraction]] = []
    previous_stop = Fraction(-1, 1)
    for event in raw:
        if type(event) is not dict or set(event) != {"start_seconds", "stop_seconds"}:
            raise ValueError("reference seizure interval schema drifted")
        start = Fraction(str(event["start_seconds"]))
        stop = Fraction(str(event["stop_seconds"]))
        if start < 0 or stop <= start or stop > duration or start < previous_stop:
            raise ValueError("reference seizure intervals are invalid")
        result.append((start, stop))
        previous_stop = stop
    return result


def build_detector_selection_metric_receipt_v1(
    *,
    prediction_artifacts: Sequence[Mapping[str, Any]],
    checkpoint_hash_by_epoch: Mapping[int, str],
    prediction_artifact_inventory_sha256: str,
    reference_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Recompute a frozen patient-macro event scorer from exact references.

    Research epoch selection and detector promotion qualification are
    deliberately distinct.  A complete, evaluable inventory always yields one
    ``research_selected_epoch``.  Passing all frozen event-sensitivity,
    event-duration-coverage, false-alarm-count and false-positive-duration
    constraints is considered first, but failure to pass never makes research
    selection null.  ``promotion_qualified_epoch`` is nullable and promotion
    itself remains false until the later source-dev operating-point authority.
    """

    _sha256(prediction_artifact_inventory_sha256, "prediction artifact inventory")
    reference_by_identity: dict[str, Mapping[str, Any]] = {}
    for row in list(reference_records):
        identity = _identifier(row.get("analysis_identity_id"), "reference identity")
        if identity in reference_by_identity:
            raise ValueError("reference metric roster duplicates an identity")
        reference_by_identity[identity] = row
    artifacts = [
        _validate_prediction_terminal_artifact_v1(row)
        for row in prediction_artifacts
    ]
    if not artifacts or not checkpoint_hash_by_epoch:
        raise ValueError("selection scorer inventory is empty")
    expected_provider = artifacts[0]["provider_id"]
    expected_variant = artifacts[0]["detector_variant_id"]
    expected_outer_fold = artifacts[0]["outer_fold_id"]
    for artifact in artifacts:
        epoch = artifact["checkpoint_epoch"]
        if (
            artifact["provider_id"] != expected_provider
            or artifact["detector_variant_id"] != expected_variant
            or artifact["outer_fold_id"] != expected_outer_fold
            or checkpoint_hash_by_epoch.get(epoch)
            != artifact["checkpoint_file_sha256"]
        ):
            raise PermissionError(
                "selection scorer provider/variant/fold/checkpoint lineage drifted"
            )
    patient_by_identity: dict[str, str] = {}
    for artifact in artifacts:
        identity = artifact["analysis_identity_id"]
        patient = artifact["local_patient_id"]
        if identity in patient_by_identity and patient_by_identity[identity] != patient:
            raise ValueError("prediction patient binding drifted across epochs")
        patient_by_identity[identity] = patient
    if set(reference_by_identity) != set(patient_by_identity):
        raise ValueError("selection scorer reference/prediction recording denominator differs")
    expected_pairs = {
        (epoch, identity)
        for epoch in checkpoint_hash_by_epoch
        for identity in reference_by_identity
    }
    observed_pairs = {
        (row["checkpoint_epoch"], row["analysis_identity_id"])
        for row in artifacts
    }
    if observed_pairs != expected_pairs or len(observed_pairs) != len(artifacts):
        raise ValueError("selection scorer lacks the complete epoch/record denominator")

    patients = sorted(set(patient_by_identity.values()))
    terminal_counts_by_epoch: dict[int, dict[str, int]] = {
        epoch: {outcome: 0 for outcome in sorted(_TYPED_TERMINAL_OUTCOMES)}
        for epoch in checkpoint_hash_by_epoch
    }
    aggregate: dict[tuple[int, str], dict[str, Any]] = {}
    for epoch in checkpoint_hash_by_epoch:
        for patient in patients:
            aggregate[(epoch, patient)] = {
                "recording_count": 0,
                "recording_duration": Fraction(0, 1),
                "reference_event_count": 0,
                "hit_event_count": 0,
                "reference_event_duration": Fraction(0, 1),
                "covered_event_duration": Fraction(0, 1),
                "false_alarm_event_count": 0,
                "false_positive_duration": Fraction(0, 1),
                "onset_absolute_error_sum": Fraction(0, 1),
                "symmetric_difference_duration": Fraction(0, 1),
            }
    record_rows: list[dict[str, Any]] = []
    for artifact in sorted(
        artifacts,
        key=lambda row: (row["checkpoint_epoch"], row["analysis_identity_id"]),
    ):
        epoch = artifact["checkpoint_epoch"]
        identity = artifact["analysis_identity_id"]
        patient = artifact["local_patient_id"]
        duration = _fraction_v1(
            artifact["recording_duration_seconds_fraction"],
            "metric recording duration",
            positive=True,
        )
        reference = reference_by_identity[identity]
        if reference.get("recording_duration_seconds_fraction") != _fraction_json_v1(duration):
            raise ValueError("metric reference duration differs from prediction")
        target = _reference_intervals_v1(reference, duration)
        target_duration = sum(
            (stop - start for start, stop in target), Fraction(0, 1)
        )
        outcome = artifact["terminal"]["outcome"]
        terminal_counts_by_epoch[epoch][outcome] += 1
        if outcome in {"technical_failure", "partial"}:
            predicted: list[tuple[Fraction, Fraction]] = []
            intersection = Fraction(0, 1)
            hit_count = 0
            false_alarm_count = 1
            false_positive = duration
            false_negative = target_duration
            onset_error_sum = duration * len(target)
            symmetric = duration
        else:
            predicted = [
                (
                    _fraction_v1(
                        row["start_seconds_fraction"], "predicted interval start"
                    ),
                    _fraction_v1(
                        row["stop_seconds_fraction"],
                        "predicted interval stop",
                        positive=True,
                    ),
                )
                for row in artifact["terminal"]["predicted_seizure_intervals"]
            ]
            intersection = _interval_intersection_duration_v1(predicted, target)
            predicted_duration = sum(
                (stop - start for start, stop in predicted), Fraction(0, 1)
            )
            false_positive = predicted_duration - intersection
            false_negative = target_duration - intersection
            symmetric = false_positive + false_negative
            matches = _one_to_one_event_matches_v1(predicted, target)
            hit_count = len(matches)
            matched_prediction_indices = {
                prediction_index for _, prediction_index in matches
            }
            false_alarm_count = len(predicted) - len(matched_prediction_indices)
            onset_error_sum = duration * (len(target) - hit_count) + sum(
                (
                    abs(
                        predicted[prediction_index][0]
                        - target[target_index][0]
                    )
                    for target_index, prediction_index in matches
                ),
                Fraction(0, 1),
            )
        stats = aggregate[(epoch, patient)]
        stats["recording_count"] += 1
        stats["recording_duration"] += duration
        stats["reference_event_count"] += len(target)
        stats["hit_event_count"] += hit_count
        stats["reference_event_duration"] += target_duration
        stats["covered_event_duration"] += intersection
        stats["false_alarm_event_count"] += false_alarm_count
        stats["false_positive_duration"] += false_positive
        stats["onset_absolute_error_sum"] += onset_error_sum
        stats["symmetric_difference_duration"] += symmetric
        record_rows.append(
            {
                "epoch": epoch,
                "analysis_identity_id": identity,
                "local_patient_id": patient,
                "terminal_outcome": outcome,
                "reference_event_denominator": len(target),
                "hit_event_numerator": hit_count,
                "reference_event_duration_seconds_fraction": _fraction_json_v1(
                    target_duration
                ),
                "covered_event_duration_seconds_fraction": _fraction_json_v1(
                    intersection
                ),
                "false_alarm_event_count": false_alarm_count,
                "false_positive_duration_seconds_fraction": _fraction_json_v1(
                    false_positive
                ),
                "false_negative_duration_seconds_fraction": _fraction_json_v1(
                    false_negative
                ),
                "onset_absolute_error_sum_seconds_fraction": _fraction_json_v1(
                    onset_error_sum
                ),
                "interval_symmetric_difference_seconds_fraction": _fraction_json_v1(
                    symmetric
                ),
            }
        )

    patient_rows: list[dict[str, Any]] = []
    epoch_patient_metrics: dict[int, list[dict[str, Any]]] = {
        epoch: [] for epoch in checkpoint_hash_by_epoch
    }
    for epoch in sorted(checkpoint_hash_by_epoch):
        for patient in patients:
            stats = aggregate[(epoch, patient)]
            duration = stats["recording_duration"]
            if stats["recording_count"] <= 0 or duration <= 0:
                raise ValueError("selection scorer patient denominator is incomplete")
            event_count = stats["reference_event_count"]
            seizure_duration = stats["reference_event_duration"]
            sensitivity = (
                None
                if event_count == 0
                else Fraction(stats["hit_event_count"], event_count)
            )
            coverage = (
                None
                if seizure_duration == 0
                else stats["covered_event_duration"] / seizure_duration
            )
            onset_error = (
                None
                if event_count == 0
                else stats["onset_absolute_error_sum"] / event_count
            )
            false_alarms_per_hour = Fraction(
                stats["false_alarm_event_count"] * 3600, 1
            ) / duration
            false_positive_seconds_per_hour = (
                stats["false_positive_duration"] * 3600 / duration
            )
            symmetric_fraction = stats["symmetric_difference_duration"] / duration
            metrics = {
                "event_sensitivity": sensitivity,
                "event_duration_coverage": coverage,
                "false_alarm_events_per_hour": false_alarms_per_hour,
                "false_positive_seconds_per_hour": false_positive_seconds_per_hour,
                "onset_absolute_error_seconds": onset_error,
                "interval_symmetric_difference_fraction": symmetric_fraction,
            }
            epoch_patient_metrics[epoch].append(metrics)
            patient_rows.append(
                {
                    "epoch": epoch,
                    "local_patient_id": patient,
                    "recording_denominator": stats["recording_count"],
                    "recording_duration_seconds_fraction": _fraction_json_v1(duration),
                    "reference_event_denominator": event_count,
                    "hit_event_numerator": stats["hit_event_count"],
                    "event_sensitivity_fraction": (
                        None if sensitivity is None else _fraction_json_v1(sensitivity)
                    ),
                    "event_duration_coverage_fraction": (
                        None if coverage is None else _fraction_json_v1(coverage)
                    ),
                    "false_alarm_events_per_hour_fraction": _fraction_json_v1(
                        false_alarms_per_hour
                    ),
                    "false_positive_seconds_per_hour_fraction": _fraction_json_v1(
                        false_positive_seconds_per_hour
                    ),
                    "onset_absolute_error_seconds_fraction": (
                        None if onset_error is None else _fraction_json_v1(onset_error)
                    ),
                    "interval_symmetric_difference_fraction": _fraction_json_v1(
                        symmetric_fraction
                    ),
                }
            )

    def macro(values: Sequence[Fraction], context: str) -> Fraction:
        if not values:
            raise PermissionError(f"selection scorer is not evaluable: {context} denominator is zero")
        return sum(values, Fraction(0, 1)) / len(values)

    epoch_rows: list[dict[str, Any]] = []
    research_ordering: dict[int, tuple[Fraction, ...]] = {}
    promotion_ordering: dict[int, tuple[Fraction, ...]] = {}
    qualified: list[int] = []
    sensitivity_minimum = _fraction_v1(
        _SCORER_BINDING_V1["patient_macro_event_sensitivity_minimum_fraction"],
        "frozen sensitivity minimum",
    )
    coverage_minimum = _fraction_v1(
        _SCORER_BINDING_V1[
            "patient_macro_event_duration_coverage_minimum_fraction"
        ],
        "frozen coverage minimum",
    )
    false_alarm_maximum = _fraction_v1(
        _SCORER_BINDING_V1[
            "patient_macro_false_alarm_events_per_hour_maximum_fraction"
        ],
        "frozen false-alarm maximum",
    )
    false_positive_maximum = _fraction_v1(
        _SCORER_BINDING_V1[
            "patient_macro_false_positive_seconds_per_hour_maximum_fraction"
        ],
        "frozen false-positive-duration maximum",
    )
    for epoch in sorted(checkpoint_hash_by_epoch):
        rows = epoch_patient_metrics[epoch]
        seizure_rows = [row for row in rows if row["event_sensitivity"] is not None]
        sensitivity = macro(
            [row["event_sensitivity"] for row in seizure_rows],
            "seizure-bearing patient sensitivity",
        )
        coverage = macro(
            [row["event_duration_coverage"] for row in seizure_rows],
            "seizure-bearing patient coverage",
        )
        false_alarm = macro(
            [row["false_alarm_events_per_hour"] for row in rows],
            "patient false alarm",
        )
        false_positive_rate = macro(
            [row["false_positive_seconds_per_hour"] for row in rows],
            "patient false-positive duration",
        )
        onset_error = macro(
            [row["onset_absolute_error_seconds"] for row in seizure_rows],
            "seizure-bearing patient onset error",
        )
        symmetric = macro(
            [row["interval_symmetric_difference_fraction"] for row in rows],
            "patient interval symmetric difference",
        )
        is_qualified = (
            sensitivity >= sensitivity_minimum
            and coverage >= coverage_minimum
            and false_alarm <= false_alarm_maximum
            and false_positive_rate <= false_positive_maximum
        )
        if is_qualified:
            qualified.append(epoch)
        common_order = (
            -sensitivity,
            -coverage,
            false_alarm,
            false_positive_rate,
            onset_error,
            symmetric,
            Fraction(epoch, 1),
        )
        research_ordering[epoch] = (
            Fraction(0 if is_qualified else 1, 1),
            *common_order,
        )
        promotion_ordering[epoch] = common_order
        epoch_rows.append(
            {
                "epoch": epoch,
                "recording_denominator": len(reference_by_identity),
                "patient_denominator": len(rows),
                "seizure_bearing_patient_denominator": len(seizure_rows),
                "terminal_outcome_counts": terminal_counts_by_epoch[epoch],
                "patient_macro_event_sensitivity_fraction": _fraction_json_v1(
                    sensitivity
                ),
                "patient_macro_event_duration_coverage_fraction": _fraction_json_v1(
                    coverage
                ),
                "patient_macro_false_alarm_events_per_hour_fraction": _fraction_json_v1(
                    false_alarm
                ),
                "patient_macro_false_positive_seconds_per_hour_fraction": _fraction_json_v1(
                    false_positive_rate
                ),
                "patient_macro_onset_absolute_error_seconds_fraction": _fraction_json_v1(
                    onset_error
                ),
                "patient_macro_interval_symmetric_difference_fraction": _fraction_json_v1(
                    symmetric
                ),
                "absolute_qualification_constraints_met": is_qualified,
            }
        )
    research_selected_epoch = min(
        research_ordering, key=lambda epoch: research_ordering[epoch]
    )
    promotion_qualified_epoch = (
        None
        if not qualified
        else min(qualified, key=lambda epoch: promotion_ordering[epoch])
    )
    reference_inventory = [
        {
            "analysis_identity_id": identity,
            "reference_file_sha256": _sha256(
                reference_by_identity[identity].get("reference_file_sha256"),
                "metric reference file",
            ),
            "event_inventory_sha256": _sha256(
                reference_by_identity[identity].get("event_inventory_sha256"),
                "metric event inventory",
            ),
        }
        for identity in sorted(reference_by_identity)
    ]
    return _seal(
        {
            "schema_version": DETECTOR_SELECTION_METRIC_RECEIPT_SCHEMA_V1,
            "scorer_binding": deepcopy(_SCORER_BINDING_V1),
            "scorer_config_sha256": _canonical_sha256(_SCORER_BINDING_V1),
            "prediction_artifact_inventory_sha256": prediction_artifact_inventory_sha256,
            "reference_record_inventory_sha256": _canonical_sha256(reference_inventory),
            "candidate_epochs": sorted(checkpoint_hash_by_epoch),
            "recording_denominator_per_epoch": len(reference_by_identity),
            "patient_denominator_per_epoch": len(patients),
            "record_metric_rows": record_rows,
            "patient_metric_rows": patient_rows,
            "epoch_metric_rows": epoch_rows,
            "metric_values_caller_supplied": False,
            "research_selected_epoch": research_selected_epoch,
            "research_selected_checkpoint_file_sha256": checkpoint_hash_by_epoch[
                research_selected_epoch
            ],
            "promotion_qualified_epoch": promotion_qualified_epoch,
            "promotion_qualified_checkpoint_file_sha256": (
                None
                if promotion_qualified_epoch is None
                else checkpoint_hash_by_epoch[promotion_qualified_epoch]
            ),
            "selected_epoch": research_selected_epoch,
            "selected_checkpoint_file_sha256": checkpoint_hash_by_epoch[
                research_selected_epoch
            ],
            "selected_epoch_semantics": "research_selected_epoch",
            "promotion_qualification_state": (
                "no_promotion_qualified_epoch"
                if promotion_qualified_epoch is None
                else "promotion_qualified_epoch_available"
            ),
            "detector_promotion_authorized": False,
            "promotion_blocker": "source_dev_operating_point_and_promotion_gate_not_run",
            "selection_rule": _SCORER_BINDING_V1["selection_rule"],
            "outer_heldout_reference_access_count": 0,
            "source_dev_reference_access_count": 0,
            "source_eval_reference_access_count": 0,
            "private_reference_access_count": 0,
        }
    )


def validate_detector_selection_metric_receipt_v1(
    value: Mapping[str, Any],
    *,
    prediction_artifacts: Sequence[Mapping[str, Any]],
    checkpoint_hash_by_epoch: Mapping[int, str],
    prediction_artifact_inventory_sha256: str,
    reference_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected = build_detector_selection_metric_receipt_v1(
        prediction_artifacts=prediction_artifacts,
        checkpoint_hash_by_epoch=checkpoint_hash_by_epoch,
        prediction_artifact_inventory_sha256=prediction_artifact_inventory_sha256,
        reference_records=reference_records,
    )
    if dict(value) != expected:
        raise ValueError("selected-epoch metric receipt does not recompute from predictions and reference")
    return expected


def build_detector_final_refit_prerequisite_v1(
    *,
    provider_id: str,
    detector_variant_id: str,
    outer_fold_id: int,
    authorized_fold_ids: Sequence[int],
    authorized_roster: Mapping[str, Any],
    selected_epoch_metric_receipt: Mapping[str, Any],
    fresh_initialization_artifact: Mapping[str, Any],
    architecture_code_sha256: str,
    training_code_sha256: str,
    preprocessing_fit_artifact_sha256: str,
    random_seed_and_rng_state_receipt_sha256: str,
) -> dict[str, Any]:
    metric = dict(selected_epoch_metric_receipt)
    selected_epoch = _positive_int(metric.get("selected_epoch"), "selected epoch")
    _sha256(metric.get("receipt_sha256"), "selected metric receipt")
    initial = _strict_dict(
        fresh_initialization_artifact,
        {"relative_path", "file_sha256", "file_bytes"},
        "fresh initialization artifact",
    )
    _safe_relative_path(initial["relative_path"], "fresh initialization path")
    _sha256(initial["file_sha256"], "fresh initialization hash")
    _positive_int(initial["file_bytes"], "fresh initialization bytes")
    return _seal(
        {
            "schema_version": DETECTOR_FINAL_REFIT_TYPED_PREREQUISITE_SCHEMA_V1,
            "provider_id": _identifier(provider_id, "final-refit provider"),
            "detector_variant_id": _identifier(detector_variant_id, "final-refit variant"),
            "outer_fold_id": _outer_fold_id(outer_fold_id),
            "training_phase": "final_refit",
            "authorized_fold_ids": list(authorized_fold_ids),
            "authorized_roster": deepcopy(dict(authorized_roster)),
            "selected_epoch_metric_receipt_sha256": metric["receipt_sha256"],
            "selected_epoch": selected_epoch,
            "planned_epoch_count": selected_epoch,
            "reinitialize_from_scratch": True,
            "selection_checkpoint_used_as_initializer": False,
            "optimizer_or_rng_resume_state_loaded": False,
            "final_refit_training_started": False,
            "fresh_initialization_artifact": deepcopy(initial),
            "architecture_code_sha256": _sha256(architecture_code_sha256, "architecture code"),
            "training_code_sha256": _sha256(training_code_sha256, "training code"),
            "preprocessing_fit_artifact_sha256": _sha256(
                preprocessing_fit_artifact_sha256, "preprocessing fit artifact"
            ),
            "random_seed_and_rng_state_receipt_sha256": _sha256(
                random_seed_and_rng_state_receipt_sha256, "final-refit RNG receipt"
            ),
            "forbidden_access_counts": deepcopy(_FORBIDDEN_ACCESS_COUNTS),
        }
    )


def validate_detector_reference_phase_gate_proof_v1(
    value: Mapping[str, Any],
    *,
    controller_signature_authority: Mapping[str, Any],
    outer_fold_id: int,
    opens_phase: str,
    authorized_fold_ids: Sequence[int],
    authorized_roster: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate serialized signed-proof structure, never issue authority.

    This intentionally does not open checkpoint or prediction paths.  Formal
    consumers must obtain :class:`DetectorReferenceGateReplayV1` from
    :func:`replay_detector_reference_phase_gate_v1`; the fold materializer then
    wraps the fully replayed phase in its separate process-local opaque type.
    """

    proof = deepcopy(dict(value))
    observed = _sha256(proof.pop("gate_proof_sha256", None), "gate proof")
    if observed != _canonical_sha256(proof):
        raise ValueError("detector phase-gate proof does not replay")
    proof["gate_proof_sha256"] = observed
    ledger = _verify_controller_signature_v1(
        ledger=proof.get("controller_signed_ledger"),
        authority=controller_signature_authority,
    )
    pre = proof.get("pre_reference_release_receipt")
    if not isinstance(pre, Mapping):
        raise ValueError("pre-reference release timing receipt is missing")
    pre_data = _validate_self_hash(
        pre,
        {
            "schema_version",
            "controller_ledger_body_sha256",
            "controller_ledger_file_sha256",
            "controller_signature_verified",
            "controller_signature_verified_logical_sequence",
            "checkpoint_and_prediction_actual_byte_replay_complete",
            "artifact_byte_replay_complete_logical_sequence",
            "pre_reference_release_receipt_sealed_logical_sequence",
            "candidate_checkpoint_count",
            "candidate_checkpoint_bytes_replayed",
            "fresh_initialization_artifact_bytes_replayed",
            "prediction_artifact_count",
            "prediction_artifact_bytes_replayed",
            "prediction_artifact_inventory_sha256",
            "complete_prediction_first_cartesian_denominator",
            "typed_terminal_outcome_counts",
            "reference_bytes_opened_before_receipt",
            "reference_files_opened_before_receipt",
        },
        "pre-reference release receipt",
    )
    raw_candidates = ledger.get("candidate_checkpoints")
    raw_predictions = ledger.get("prediction_artifacts")
    if type(raw_candidates) is not list or type(raw_predictions) is not list:
        raise ValueError("signed ledger artifact inventories are not arrays")
    candidates = [
        _strict_dict(row, _CHECKPOINT_LEDGER_ROW_FIELDS, f"proof checkpoint {index}")
        for index, row in enumerate(raw_candidates)
    ]
    predictions = [
        _strict_dict(row, _PREDICTION_LEDGER_ROW_FIELDS, f"proof prediction {index}")
        for index, row in enumerate(raw_predictions)
    ]
    prediction_inventory_sha256 = _canonical_sha256(
        [
            {
                "checkpoint_epoch": row["checkpoint_epoch"],
                "analysis_identity_id": row["analysis_identity_id"],
                "prediction_artifact_file_sha256": row[
                    "prediction_artifact_file_sha256"
                ],
                "terminal_outcome": row["terminal_outcome"],
            }
            for row in predictions
        ]
    )
    outcome_counts = {
        outcome: sum(row["terminal_outcome"] == outcome for row in predictions)
        for outcome in sorted(_TYPED_TERMINAL_OUTCOMES)
    }
    expected_initialization_bytes = 0
    if opens_phase == "final_refit":
        prerequisite = ledger.get("final_refit_prerequisite")
        if not isinstance(prerequisite, Mapping):
            raise ValueError("final-refit signed prerequisite is missing")
        initial = prerequisite.get("fresh_initialization_artifact")
        if not isinstance(initial, Mapping):
            raise ValueError("final-refit fresh initialization binding is missing")
        expected_initialization_bytes = _positive_int(
            initial.get("file_bytes"), "proof fresh initialization bytes"
        )
    if (
        proof.get("gate_validator_id") != DETECTOR_REFERENCE_PHASE_GATE_VALIDATOR_ID_V1
        or proof.get("gate_validator_source_sha256")
        != detector_reference_phase_gate_source_sha256_v1()
        or proof.get("provider_id") != ledger["provider_id"]
        or proof.get("detector_variant_id") != ledger["detector_variant_id"]
        or proof.get("outer_fold_id") != outer_fold_id
        or proof.get("opens_phase") != opens_phase
        or proof.get("authorized_fold_ids") != list(authorized_fold_ids)
        or proof.get("authorized_roster") != dict(authorized_roster)
        or proof.get("controller_signature_verified") is not True
        or proof.get("reference_sidecars_opened_during_gate_replay") != 0
        or proof.get("validated_before_first_reference_open") is not True
        or ledger["outer_fold_id"] != outer_fold_id
        or ledger["release_phase"] != opens_phase
        or proof.get("reference_release_authority")
        != ledger["reference_release_authority"]
        or pre_data["controller_ledger_body_sha256"]
        != ledger["ledger_body_sha256"]
        or pre_data["controller_ledger_file_sha256"]
        != proof.get("controller_ledger_file_sha256")
        or not isinstance(proof.get("controller_ledger_file_bytes"), int)
        or proof.get("controller_ledger_file_bytes") <= 0
        or proof.get("prediction_artifact_inventory_sha256")
        != prediction_inventory_sha256
        or pre_data["prediction_artifact_inventory_sha256"]
        != prediction_inventory_sha256
        or pre_data["candidate_checkpoint_count"] != len(candidates)
        or pre_data["candidate_checkpoint_bytes_replayed"]
        != sum(row["checkpoint_file_bytes"] for row in candidates)
        or pre_data["fresh_initialization_artifact_bytes_replayed"]
        != expected_initialization_bytes
        or pre_data["prediction_artifact_count"] != len(predictions)
        or pre_data["prediction_artifact_bytes_replayed"]
        != sum(row["prediction_artifact_file_bytes"] for row in predictions)
        or pre_data["typed_terminal_outcome_counts"] != outcome_counts
        or pre_data["schema_version"] != DETECTOR_PRE_REFERENCE_RELEASE_RECEIPT_SCHEMA_V1
        or pre_data["controller_signature_verified"] is not True
        or pre_data["checkpoint_and_prediction_actual_byte_replay_complete"] is not True
        or pre_data["complete_prediction_first_cartesian_denominator"] is not True
        or pre_data["controller_signature_verified_logical_sequence"] != 1
        or pre_data["artifact_byte_replay_complete_logical_sequence"] != 2
        or pre_data["pre_reference_release_receipt_sealed_logical_sequence"] != 3
        or pre_data["reference_bytes_opened_before_receipt"] != 0
        or pre_data["reference_files_opened_before_receipt"] != 0
    ):
        raise PermissionError("detector phase-gate proof scope/timing drifted")
    return proof


__all__ = [
    "DETECTOR_CONTROLLER_LEDGER_SCHEMA_V1",
    "DETECTOR_CONTROLLER_SIGNATURE_ALGORITHM_V1",
    "DETECTOR_FINAL_REFIT_TYPED_PREREQUISITE_SCHEMA_V1",
    "DETECTOR_PREDICTION_CONTRACT_ID_V1",
    "DETECTOR_PREDICTION_TERMINAL_ARTIFACT_SCHEMA_V1",
    "DETECTOR_PRE_REFERENCE_RELEASE_RECEIPT_SCHEMA_V1",
    "DETECTOR_REFERENCE_PHASE_GATE_EXECUTION_STATUS_V1",
    "DETECTOR_REFERENCE_PHASE_GATE_VALIDATOR_ID_V1",
    "DETECTOR_SELECTION_METRIC_RECEIPT_SCHEMA_V1",
    "DETECTOR_SELECTION_SCORER_ID_V1",
    "DETECTOR_SELECTION_SCORER_VERSION_V1",
    "DetectorReferenceGateReplayV1",
    "build_detector_controller_ledger_body_v1",
    "build_detector_final_refit_prerequisite_v1",
    "build_detector_prediction_terminal_artifact_v1",
    "build_detector_reference_release_authority_v1",
    "build_detector_selection_metric_receipt_v1",
    "detector_controller_ledger_signing_bytes_v1",
    "detector_reference_phase_gate_source_sha256_v1",
    "replay_detector_reference_phase_gate_v1",
    "validate_detector_controller_signature_authority_v1",
    "validate_detector_reference_phase_gate_proof_v1",
    "validate_detector_selection_metric_receipt_v1",
]
