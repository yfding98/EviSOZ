"""Append-only artifact writer for reference-free DeepSOZ batch validation.

This module is intentionally not exported by the package ``__init__``.  It
binds the existing sealed DeepSOZ posterior-batch validator to an
identity/split-only roster receipt, then writes validation evidence to a new
directory.  It has no reference, annotation, spreadsheet, clinical-text,
calibration or source-evaluation input.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .continuous_detection_roster import (
    validate_continuous_detector_split_roster,
)
from .deepsoz_posterior_batch_validation import (
    DEEPSOZ_MATERIALIZER_CODE_SHA256,
    DEEPSOZ_PROVIDER_ID,
    ValidatedDeepSOZPosteriorBatch,
    revalidate_deepsoz_posterior_batch_without_references,
    validate_deepsoz_posterior_batch_without_references,
)


DEEPSOZ_REFERENCE_FREE_VALIDATION_RECEIPT_FILENAME = "validation_receipt.json"
DEEPSOZ_REFERENCE_FREE_VALIDATION_WRITE_RECEIPT_FILENAME = "write_receipt.json"
DEEPSOZ_REFERENCE_FREE_VALIDATION_WRITE_SCHEMA_VERSION = (
    "deepsoz_reference_free_batch_validation_write_receipt_v1"
)
DEEPSOZ_REFERENCE_FREE_VALIDATION_WRITE_METHOD_ID = (
    "identity_roster_bound_append_only_reference_free_validation_write_v1"
)

_ALLOWED_SPLITS = frozenset({"source_train", "source_dev"})
_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_REFERENCE_ACCESS = {
    "reference_path_argument_accepted": False,
    "reference_files_opened": 0,
    "edf_annotations_opened": 0,
    "excel_files_opened": 0,
    "clinical_text_opened": 0,
    "source_eval_opened": 0,
}
_WRITE_SCOPE = {
    "identity_and_split_only_roster_used": True,
    "reference_parameter_accepted": False,
    "annotation_parameter_accepted": False,
    "excel_parameter_accepted": False,
    "clinical_parameter_accepted": False,
    "source_eval_profile_or_parameter_accepted": False,
    "reference_files_opened": 0,
    "edf_annotations_opened": 0,
    "excel_files_opened": 0,
    "clinical_text_opened": 0,
    "source_eval_opened": 0,
    "calibration_performed": False,
    "source_eval_scoring_performed": False,
    "output_is_validation_evidence_only": True,
    "production_or_sota_claim_authorized": False,
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


def _file_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _SHA256_CHARACTERS for character in value)
    )


def _identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TypeError(f"{context} must be a non-empty trimmed string")
    if len(value) > 1024 or any(ord(character) < 32 for character in value):
        raise ValueError(f"{context} is invalid")
    return value


def _loads_json(payload: bytes, context: str) -> Any:
    def no_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{context} contains duplicate key {key!r}")
            result[key] = value
        return result

    def no_constant(value: str) -> None:
        raise ValueError(f"{context} contains non-finite constant {value}")

    try:
        return json.loads(
            payload,
            object_pairs_hook=no_duplicate,
            parse_constant=no_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{context} is not valid UTF-8 JSON") from error


def _regular_file(path: Path, context: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{context} must be a regular non-symlink file")
    return path.resolve(strict=True)


@dataclass(frozen=True, slots=True)
class DeepSOZIdentityRosterBinding:
    roster_id: str
    roster_receipt_path: str
    roster_receipt_file_sha256: str
    roster_receipt_content_sha256: str
    source_manifest_file_sha256: str
    selected_split: str
    patient_ids: tuple[str, ...]
    recording_ids: tuple[str, ...]
    patient_roster_sha256: str
    recording_roster_sha256: str


def load_deepsoz_identity_roster_binding(
    roster_receipt_path: str | Path,
    *,
    selected_split: str,
) -> DeepSOZIdentityRosterBinding:
    """Load only the validated identity/split projection for one closed split."""

    split = _identifier(selected_split, "selected split")
    if split not in _ALLOWED_SPLITS:
        raise ValueError(
            "reference-free DeepSOZ validation only permits source_train or source_dev"
        )
    path = _regular_file(Path(roster_receipt_path), "split-roster receipt")
    raw = path.read_bytes()
    roster = validate_continuous_detector_split_roster(
        _loads_json(raw, "split-roster receipt")
    )
    scope = roster["scope_receipt"]
    if (
        scope["identity_and_split_metadata_only"] is not True
        or scope["target_annotation_excel_fields_projected"] is not False
        or scope["seizure_intervals_retained"] is not False
        or scope["soz_or_channel_labels_retained"] is not False
        or scope["edf_annotations_used"] is not False
        or scope["excel_or_clinical_text_used"] is not False
        or scope["roster_is_reference_or_performance_evidence"] is not False
    ):
        raise ValueError("split-roster receipt is not identity/split-only")
    selected = roster["split_rosters"].get(split)
    if not isinstance(selected, dict):
        raise ValueError(f"split-roster receipt has no {split} inventory")
    return DeepSOZIdentityRosterBinding(
        roster_id=roster["roster_id"],
        roster_receipt_path=str(path),
        roster_receipt_file_sha256=_file_sha256(raw),
        roster_receipt_content_sha256=_canonical_sha256(roster),
        source_manifest_file_sha256=roster["source_manifest_file_sha256"],
        selected_split=split,
        patient_ids=tuple(selected["patient_ids"]),
        recording_ids=tuple(selected["recording_ids"]),
        patient_roster_sha256=selected["patient_roster_sha256"],
        recording_roster_sha256=selected["recording_roster_sha256"],
    )


def _bind_validation_to_roster(
    batch: ValidatedDeepSOZPosteriorBatch,
    binding: DeepSOZIdentityRosterBinding,
) -> dict[str, Any]:
    sealed = revalidate_deepsoz_posterior_batch_without_references(batch)
    receipt = sealed.validation_receipt()
    if (
        receipt["selected_split"] != binding.selected_split
        or receipt["manifest_sha256"] != binding.source_manifest_file_sha256
        or receipt["recording_count"] != len(binding.recording_ids)
        or receipt["patient_count"] != len(binding.patient_ids)
        or receipt["recording_ids_sha256"] != binding.recording_roster_sha256
        or receipt["patient_ids_sha256"] != binding.patient_roster_sha256
        or receipt["inventory_completeness_verified"] is not True
        or receipt["reference_access"] != _REFERENCE_ACCESS
    ):
        raise ValueError("DeepSOZ validation receipt and identity roster disagree")
    return receipt


def validate_deepsoz_reference_free_validation_write_receipt(
    payload: object,
) -> dict[str, Any]:
    required = {
        "schema_version",
        "write_id",
        "method_id",
        "provider_id",
        "selected_split",
        "validation_id",
        "validation_receipt_sha256",
        "validation_receipt_filename",
        "validation_receipt_file_sha256",
        "split_roster_id",
        "split_roster_receipt_file_sha256",
        "split_roster_receipt_content_sha256",
        "source_manifest_file_sha256",
        "expected_recording_roster_sha256",
        "expected_patient_roster_sha256",
        "recording_count",
        "patient_count",
        "written_filenames",
        "append_only_new_directory",
        "overwrite_performed",
        "reference_access",
        "scope_receipt",
        "receipt_sha256",
    }
    if type(payload) is not dict or set(payload) != required:
        raise ValueError("DeepSOZ validation write receipt fields drifted")
    data = deepcopy(payload)
    if (
        data["schema_version"] != DEEPSOZ_REFERENCE_FREE_VALIDATION_WRITE_SCHEMA_VERSION
        or data["method_id"] != DEEPSOZ_REFERENCE_FREE_VALIDATION_WRITE_METHOD_ID
        or data["provider_id"] != DEEPSOZ_PROVIDER_ID
        or data["selected_split"] not in _ALLOWED_SPLITS
        or data["validation_receipt_filename"]
        != DEEPSOZ_REFERENCE_FREE_VALIDATION_RECEIPT_FILENAME
        or data["written_filenames"]
        != [
            DEEPSOZ_REFERENCE_FREE_VALIDATION_RECEIPT_FILENAME,
            DEEPSOZ_REFERENCE_FREE_VALIDATION_WRITE_RECEIPT_FILENAME,
        ]
        or data["append_only_new_directory"] is not True
        or data["overwrite_performed"] is not False
        or data["reference_access"] != _REFERENCE_ACCESS
        or data["scope_receipt"] != _WRITE_SCOPE
    ):
        raise ValueError("DeepSOZ validation write identity or firewall drifted")
    for field in (
        "validation_receipt_sha256",
        "validation_receipt_file_sha256",
        "split_roster_receipt_file_sha256",
        "split_roster_receipt_content_sha256",
        "source_manifest_file_sha256",
        "expected_recording_roster_sha256",
        "expected_patient_roster_sha256",
        "receipt_sha256",
    ):
        if not _is_sha256(data[field]):
            raise ValueError(f"DeepSOZ validation write {field} is invalid")
    for field in ("write_id", "validation_id", "split_roster_id"):
        _identifier(data[field], f"DeepSOZ validation write {field}")
    for field in ("recording_count", "patient_count"):
        if (
            isinstance(data[field], bool)
            or not isinstance(data[field], int)
            or data[field] < 1
        ):
            raise ValueError(f"DeepSOZ validation write {field} is invalid")
    digest = deepcopy(data)
    digest["write_id"] = "DEEPSOZ-VALIDATION-WRITE-PENDING"
    digest["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["write_id"] != "DSZVALIDWRITE-" + _canonical_sha256(digest)[:24]:
        raise ValueError("DeepSOZ validation write ID is not content-bound")
    digest = deepcopy(data)
    digest["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["receipt_sha256"] != _canonical_sha256(digest):
        raise ValueError("DeepSOZ validation write receipt hash drifted")
    return data


def write_deepsoz_reference_free_validation_append_only(
    batch: ValidatedDeepSOZPosteriorBatch,
    *,
    roster_binding: DeepSOZIdentityRosterBinding,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Write validation and write receipts once into a new two-file directory."""

    validation = _bind_validation_to_roster(batch, roster_binding)
    validation_bytes = (
        json.dumps(
            validation,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    body: dict[str, Any] = {
        "schema_version": DEEPSOZ_REFERENCE_FREE_VALIDATION_WRITE_SCHEMA_VERSION,
        "write_id": "DEEPSOZ-VALIDATION-WRITE-PENDING",
        "method_id": DEEPSOZ_REFERENCE_FREE_VALIDATION_WRITE_METHOD_ID,
        "provider_id": DEEPSOZ_PROVIDER_ID,
        "selected_split": validation["selected_split"],
        "validation_id": validation["validation_id"],
        "validation_receipt_sha256": validation["receipt_sha256"],
        "validation_receipt_filename": (
            DEEPSOZ_REFERENCE_FREE_VALIDATION_RECEIPT_FILENAME
        ),
        "validation_receipt_file_sha256": _file_sha256(validation_bytes),
        "split_roster_id": roster_binding.roster_id,
        "split_roster_receipt_file_sha256": (roster_binding.roster_receipt_file_sha256),
        "split_roster_receipt_content_sha256": (
            roster_binding.roster_receipt_content_sha256
        ),
        "source_manifest_file_sha256": (roster_binding.source_manifest_file_sha256),
        "expected_recording_roster_sha256": (roster_binding.recording_roster_sha256),
        "expected_patient_roster_sha256": roster_binding.patient_roster_sha256,
        "recording_count": validation["recording_count"],
        "patient_count": validation["patient_count"],
        "written_filenames": [
            DEEPSOZ_REFERENCE_FREE_VALIDATION_RECEIPT_FILENAME,
            DEEPSOZ_REFERENCE_FREE_VALIDATION_WRITE_RECEIPT_FILENAME,
        ],
        "append_only_new_directory": True,
        "overwrite_performed": False,
        "reference_access": deepcopy(_REFERENCE_ACCESS),
        "scope_receipt": deepcopy(_WRITE_SCOPE),
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    body["write_id"] = "DSZVALIDWRITE-" + _canonical_sha256(body)[:24]
    body["receipt_sha256"] = _canonical_sha256(body)
    write_receipt = validate_deepsoz_reference_free_validation_write_receipt(body)
    write_bytes = (
        json.dumps(
            write_receipt,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")

    output = Path(output_directory)
    if output.exists() or output.is_symlink():
        raise FileExistsError(
            "DeepSOZ validation output already exists; append-only write refused"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    validation_path = output / DEEPSOZ_REFERENCE_FREE_VALIDATION_RECEIPT_FILENAME
    write_path = output / DEEPSOZ_REFERENCE_FREE_VALIDATION_WRITE_RECEIPT_FILENAME
    with validation_path.open("xb") as handle:
        handle.write(validation_bytes)
    with write_path.open("xb") as handle:
        handle.write(write_bytes)
    if {path.name for path in output.iterdir()} != set(body["written_filenames"]):
        raise RuntimeError("DeepSOZ validation output inventory drifted")
    return {
        "output_directory": str(output.resolve(strict=True)),
        "validation_receipt_path": str(validation_path.resolve(strict=True)),
        "write_receipt_path": str(write_path.resolve(strict=True)),
        "validation_receipt_file_sha256": _file_sha256(validation_path.read_bytes()),
        "write_receipt_file_sha256": _file_sha256(write_path.read_bytes()),
        "validation_id": validation["validation_id"],
        "validation_receipt_sha256": validation["receipt_sha256"],
        "write_id": write_receipt["write_id"],
        "write_receipt_sha256": write_receipt["receipt_sha256"],
        "append_only_new_directory": True,
        "overwrite_performed": False,
    }


def validate_and_write_deepsoz_batch_reference_free(
    *,
    posterior_batch_root: str | Path,
    split_roster_receipt: str | Path,
    selected_split: str,
    output_directory: str | Path,
    provider_registry_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Run sealed validation and persist receipts without any reference join."""

    binding = load_deepsoz_identity_roster_binding(
        split_roster_receipt,
        selected_split=selected_split,
    )
    sealed = validate_deepsoz_posterior_batch_without_references(
        posterior_batch_root,
        expected_split=binding.selected_split,
        expected_manifest_sha256=binding.source_manifest_file_sha256,
        expected_recording_ids=binding.recording_ids,
        expected_patient_ids=binding.patient_ids,
        expected_materializer_code_sha256=DEEPSOZ_MATERIALIZER_CODE_SHA256,
        require_complete_inventory=True,
        provider_registry_path=provider_registry_path,
    )
    validation = _bind_validation_to_roster(sealed, binding)
    write_summary = write_deepsoz_reference_free_validation_append_only(
        sealed,
        roster_binding=binding,
        output_directory=output_directory,
    )
    write_receipt_path = Path(write_summary["write_receipt_path"])
    write_receipt = validate_deepsoz_reference_free_validation_write_receipt(
        _loads_json(write_receipt_path.read_bytes(), "validation write receipt")
    )
    return validation, write_receipt, write_summary


__all__ = [
    "DEEPSOZ_REFERENCE_FREE_VALIDATION_RECEIPT_FILENAME",
    "DEEPSOZ_REFERENCE_FREE_VALIDATION_WRITE_METHOD_ID",
    "DEEPSOZ_REFERENCE_FREE_VALIDATION_WRITE_RECEIPT_FILENAME",
    "DEEPSOZ_REFERENCE_FREE_VALIDATION_WRITE_SCHEMA_VERSION",
    "DeepSOZIdentityRosterBinding",
    "load_deepsoz_identity_roster_binding",
    "validate_and_write_deepsoz_batch_reference_free",
    "validate_deepsoz_reference_free_validation_write_receipt",
    "write_deepsoz_reference_free_validation_append_only",
]
