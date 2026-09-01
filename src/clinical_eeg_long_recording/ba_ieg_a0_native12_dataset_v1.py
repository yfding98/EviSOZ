"""Source-train-only A0 native-12 dataset manifest and resume contracts.

The manifest is a terminal inventory, not a success-only index.  Every
selected A0 record is committed atomically with every expected event in one
of two typed states: ``materialized`` or ``failed``.  Zero-event records are
first-class record terminals.  A full manifest is representable only when the
exact frozen A0 denominator has been replayed; an explicit subset is always
named ``partial_smoke_terminal_inventory`` and cannot claim 70/318/908.

Artifact references are canonical relative paths with byte size and SHA-256.
Resume verification reopens every referenced file without following symlinks
and checks immutable bytes.  This module contains no detector admission path,
no EDF/Excel annotation input, and no model/report generation code.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
from typing import Any, Final, Mapping, Sequence

from .ba_ieg_a0_oracle_navigation_candidate_roster_v1 import (
    BA_IEG_NAVIGATION_ARM_A0,
    validate_ba_ieg_a0_oracle_navigation_candidate_roster_v1,
)
from .ba_ieg_a0_navigation_window_v1 import (
    BA_IEG_A0_EVALUATION_SEMANTICS_V1,
    validate_ba_ieg_a0_navigation_window_v1,
)
from .ba_ieg_event_model_input_projection_v2 import (
    BA_IEG_EVENT_MODEL_INPUT_PROJECTION_SCHEMA_VERSION_V2,
)
from .ba_ieg_training_contract import (
    BA_IEG_P0_MATERIALIZATION_SCHEMA_VERSION_A0_NATIVE_12,
    BA_IEG_P0_VIEW_PROFILE_NATIVE_12,
)


BA_IEG_A0_NATIVE12_DATASET_MANIFEST_SCHEMA_V2: Final[str] = (
    "ba_ieg_a0_source_train_native12_terminal_dataset_manifest_v2"
)
BA_IEG_A0_NATIVE12_DATASET_METHOD_ID_V1: Final[str] = (
    "record_atomic_oracle_navigation_native12_projection_v2_dataset_v1"
)
BA_IEG_A0_NATIVE12_P0_SCHEMA_V1: Final[str] = (
    BA_IEG_P0_MATERIALIZATION_SCHEMA_VERSION_A0_NATIVE_12
)

_SHA256_CHARACTERS: Final[frozenset[str]] = frozenset("0123456789abcdef")
_DATASET_STATES: Final[frozenset[str]] = frozenset(
    {"complete_terminal_inventory", "partial_smoke_terminal_inventory"}
)
_SUCCESS_ARTIFACT_KINDS: Final[tuple[tuple[str, str], ...]] = (
    ("canonical_identity_binding", ".json"),
    ("navigation_window", ".json"),
    ("p0_receipt", ".json"),
    ("input_metadata", ".json"),
    ("input_tensors", ".npz"),
    ("projection_v2_receipt", ".json"),
    ("raw_dependency", ".json"),
    ("raw_dependency_reference", ".json"),
    ("deterministic_target_metadata", ".json"),
    ("deterministic_target_tensors", ".npz"),
    ("deterministic_target_reference", ".json"),
    ("boundary_target", ".json"),
)
# ``raw_dependency.json`` is the legacy canonical-JSON disk-v2 transport.
# New materializations use the additive disk-v3 transport: the *same exact*
# canonical disk-v2 artifact bytes are carried in a deterministic single-member
# gzip file.  Keep the legacy suffix admissible so already committed smoke
# records remain resumable; no other artifact kind gains an alternate suffix.
_SUCCESS_ARTIFACT_ALLOWED_SUFFIXES: Final[
    dict[str, tuple[str, ...]]
] = {
    **{
        kind: (suffix,)
        for kind, suffix in _SUCCESS_ARTIFACT_KINDS
        if kind != "raw_dependency"
    },
    "raw_dependency": (".json", ".json.gz"),
}
_EVENT_FAILURE_CODES: Final[frozenset[str]] = frozenset(
    {
        "record_canonical_edf_load_failed",
        "record_canonical_identity_binding_failed",
        "a0_navigation_window_failed",
        "p0_invalid_canonical_bundle",
        "p0_invalid_a0_navigation_window",
        "p0_a0_identity_binding_mismatch",
        "p0_recording_clock_mismatch",
        "p0_event_interval_unavailable",
        "p0_view_clock_or_reference_mismatch",
        "p0_no_evidence_eligible_tokens",
        "p0_tokenization_failed",
        "projection_v2_failed",
        "raw_dependency_disk_failed",
        "deterministic_target_disk_failed",
        "boundary_target_failed",
        "artifact_publication_failed",
        "unexpected_event_failure",
    }
)
_RECORD_FAILURE_CODES: Final[frozenset[str]] = frozenset(
    {
        "canonical_edf_load_failed",
        "canonical_identity_binding_failed",
        "record_materialization_interrupted",
        "record_artifact_publication_failed",
        "unexpected_record_failure",
    }
)
_SCOPE_RECEIPT: Final[dict[str, Any]] = {
    "model_split": "source_train",
    "source_train_only": True,
    "source_dev_present": False,
    "source_eval_present": False,
    "private_data_present": False,
    "navigation_arm": BA_IEG_NAVIGATION_ARM_A0,
    "evaluation_semantics": BA_IEG_A0_EVALUATION_SEMANTICS_V1,
    "oracle_navigation_not_detector_frozen": True,
    "detector_output_used": False,
    "native_12_required": True,
    "projection_v2_required": True,
    "raw_dependency_sidecar_required_for_success": True,
    "deterministic_target_json_npz_required_for_success": True,
    "boundary_target_separate_from_model_input": True,
    "fixed_12_48_support_role": "initial_bootstrap_watchdog_only",
    "fixed_watchdog_is_final_analysis_window": False,
    "final_support_requires_iterative_rule_adaptive_acquisition": True,
    "iterative_rule_adaptive_acquisition_status": "not_materialized",
    "event_success_only_denominator_used": False,
    "zero_event_records_retained": True,
    "append_only_record_atomic_resume": True,
    "edf_annotations_used": False,
    "spreadsheet_used": False,
    "clinical_text_used": False,
    "localization_channel_target_used_for_materialization": False,
}

_A0_BOUNDARY_SCOPE: Final[dict[str, Any]] = {
    "authority": "public_tusz_seizure_interval",
    "evaluation_semantics": BA_IEG_A0_EVALUATION_SEMANTICS_V1,
    "public_seizure_interval_used_for_navigation": True,
    "boundary_target_attached_after_model_input_freeze": True,
    "boundary_target_available_to_model_forward": False,
    "candidate_selection_or_window_is_target_free": False,
    "localization_channel_target_used": False,
    "seizure_type_used": False,
    "edf_annotation_used": False,
    "spreadsheet_used": False,
    "clinical_text_used": False,
}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_CHARACTERS for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _identifier(value: object, name: str, *, prefix: str | None = None) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 2048
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"{name} must be a valid non-empty trimmed identifier")
    if prefix is not None and not value.startswith(prefix):
        raise ValueError(f"{name} must start with {prefix}")
    return value


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _relative_path(
    value: object, *, suffixes: Sequence[str], name: str
) -> str:
    text = _identifier(value, name)
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or str(path) != text
        or any(part in {"", ".", ".."} for part in path.parts)
        or not any(text.endswith(suffix) for suffix in suffixes)
        or "\\" in text
    ):
        expected = "/".join(suffixes)
        raise ValueError(
            f"{name} must be a canonical relative {expected} path"
        )
    return text


def _finalize_receipt(
    body: Mapping[str, Any], *, id_field: str, id_prefix: str
) -> dict[str, Any]:
    result = deepcopy(dict(body))
    result[id_field] = "CONTENT-ADDRESS-PENDING"
    result["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    result[id_field] = id_prefix + _canonical_sha256(result)[:24]
    result["receipt_sha256"] = _canonical_sha256(result)
    return result


def _validate_content_address(
    data: Mapping[str, Any], *, id_field: str, id_prefix: str
) -> None:
    _identifier(data[id_field], id_field, prefix=id_prefix)
    _sha256(data["receipt_sha256"], "receipt_sha256")
    digest_source = deepcopy(dict(data))
    digest_source["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["receipt_sha256"] != _canonical_sha256(digest_source):
        raise ValueError("terminal receipt hash does not bind content")
    id_source = deepcopy(dict(data))
    id_source[id_field] = "CONTENT-ADDRESS-PENDING"
    id_source["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data[id_field] != id_prefix + _canonical_sha256(id_source)[:24]:
        raise ValueError("terminal receipt ID does not bind content")


def build_ba_ieg_a0_artifact_reference_v1(
    *, kind: str, relative_path: str, file_size_bytes: int, file_sha256: str
) -> dict[str, Any]:
    expected = dict(_SUCCESS_ARTIFACT_KINDS)
    if kind not in expected:
        raise ValueError("unknown A0 dataset artifact kind")
    return {
        "kind": kind,
        "relative_path": _relative_path(
            relative_path,
            suffixes=_SUCCESS_ARTIFACT_ALLOWED_SUFFIXES[kind],
            name=f"{kind}.relative_path",
        ),
        "file_size_bytes": _positive_integer(
            file_size_bytes, f"{kind}.file_size_bytes"
        ),
        "file_sha256": _sha256(file_sha256, f"{kind}.file_sha256"),
    }


def _validate_artifact_reference(
    payload: object, *, expected_kind: str, record_id: str
) -> dict[str, Any]:
    if type(payload) is not dict or set(payload) != {
        "kind",
        "relative_path",
        "file_size_bytes",
        "file_sha256",
    }:
        raise ValueError("A0 artifact reference fields drifted")
    data = build_ba_ieg_a0_artifact_reference_v1(**payload)
    if data["kind"] != expected_kind:
        raise ValueError("A0 artifact kind drifted")
    path = PurePosixPath(data["relative_path"])
    if len(path.parts) < 3 or path.parts[:2] != ("records", record_id):
        raise ValueError("A0 event artifact is outside its record namespace")
    return data


def _event_identity_from_roster(event_row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "event_id": event_row["model_event_id"],
        "event_receipt_sha256": event_row["event_receipt_sha256"],
        "model_recording_id": event_row["model_recording_id"],
        "patient_uid": event_row["patient_uid"],
        "model_split": "source_train",
    }


def build_ba_ieg_a0_boundary_target_v1(
    *,
    event_row: Mapping[str, Any],
    navigation_window: Mapping[str, Any],
    source_p0_materialization_receipt_sha256: str,
    source_event_model_input_receipt_sha256: str,
    annotation_resolution_seconds: float,
) -> dict[str, Any]:
    """Attach weak public interval supervision after the A0 input is frozen.

    Unlike the target-independent detector arm, this A0-specific target is
    honest that the same public interval supplied oracle navigation.  It must
    therefore never be validated as, or converted into, a detector-frozen
    segmental target without an explicit downstream A0 adapter.
    """

    window = validate_ba_ieg_a0_navigation_window_v1(navigation_window)
    identity = _event_identity_from_roster(event_row)
    if (
        identity["event_id"] != window["event_identity"]["event_id"]
        or identity["model_recording_id"]
        != window["event_identity"]["model_recording_id"]
        or identity["patient_uid"] != window["event_identity"]["patient_uid"]
        or event_row["event_receipt_sha256"]
        != window["event_identity"]["event_receipt_sha256"]
    ):
        raise ValueError("A0 boundary event/window identity drifted")
    p0_receipt = _sha256(
        source_p0_materialization_receipt_sha256,
        "source_p0_materialization_receipt_sha256",
    )
    input_receipt = _sha256(
        source_event_model_input_receipt_sha256,
        "source_event_model_input_receipt_sha256",
    )
    resolution = float(annotation_resolution_seconds)
    if not resolution > 0 or not resolution < 60 or not (
        resolution == resolution and abs(resolution) != float("inf")
    ):
        raise ValueError("annotation_resolution_seconds must lie in (0,60)")
    onset, offset = map(float, event_row["seizure_interval_seconds"])
    support_start, support_stop = map(
        float, window["timing"]["analysis_interval_recording_seconds"]
    )

    def observed_interval(boundary: float) -> list[float]:
        half = 0.5 * resolution
        lower = max(support_start, boundary - half)
        upper = min(support_stop, boundary + half)
        if upper < lower:
            raise ValueError("A0 boundary is outside frozen analysis support")
        return [lower, upper]

    onset_interval = observed_interval(onset)
    if offset <= support_stop + 1e-9:
        offset_status = "observed_interval"
        offset_interval: list[float] | None = observed_interval(offset)
    else:
        offset_status = "right_censored"
        offset_interval = None
    body = {
        "schema_version": "ba_ieg_a0_oracle_navigation_boundary_target_v1",
        "target_id": "CONTENT-ADDRESS-PENDING",
        "event_identity": identity,
        "binding": {
            "event_receipt_sha256": event_row["event_receipt_sha256"],
            "navigation_window_receipt_sha256": window["receipt_sha256"],
            "source_p0_materialization_receipt_sha256": p0_receipt,
            "source_event_model_input_receipt_sha256": input_receipt,
        },
        "authority": "public_tusz_seizure_interval",
        "event_status": "present",
        "onset": {
            "status": "observed_interval",
            "interval_seconds": onset_interval,
        },
        "offset": {
            "status": offset_status,
            "interval_seconds": offset_interval,
        },
        "bout_count_status": "single_bout",
        "annotation_resolution_seconds": resolution,
        "frozen_analysis_support_seconds": [support_start, support_stop],
        "scope_receipt": deepcopy(_A0_BOUNDARY_SCOPE),
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    return validate_ba_ieg_a0_boundary_target_v1(
        _finalize_receipt(
            body, id_field="target_id", id_prefix="BAIEG-A0-BOUNDARY-"
        ),
        event_row=event_row,
        navigation_window=window,
        expected_source_p0_materialization_receipt_sha256=p0_receipt,
        expected_source_event_model_input_receipt_sha256=input_receipt,
    )


def validate_ba_ieg_a0_boundary_target_v1(
    payload: object,
    *,
    event_row: Mapping[str, Any] | None = None,
    navigation_window: Mapping[str, Any] | None = None,
    expected_source_p0_materialization_receipt_sha256: str | None = None,
    expected_source_event_model_input_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    required = {
        "schema_version",
        "target_id",
        "event_identity",
        "binding",
        "authority",
        "event_status",
        "onset",
        "offset",
        "bout_count_status",
        "annotation_resolution_seconds",
        "frozen_analysis_support_seconds",
        "scope_receipt",
        "receipt_sha256",
    }
    if type(payload) is not dict or set(payload) != required:
        raise ValueError("A0 boundary target fields drifted")
    data = deepcopy(payload)
    if (
        data["schema_version"]
        != "ba_ieg_a0_oracle_navigation_boundary_target_v1"
        or data["authority"] != "public_tusz_seizure_interval"
        or data["event_status"] != "present"
        or data["bout_count_status"] != "single_bout"
        or data["scope_receipt"] != _A0_BOUNDARY_SCOPE
    ):
        raise ValueError("A0 boundary target authority/status drifted")
    identity = data["event_identity"]
    if type(identity) is not dict or set(identity) != {
        "event_id",
        "event_receipt_sha256",
        "model_recording_id",
        "patient_uid",
        "model_split",
    } or identity["model_split"] != "source_train":
        raise ValueError("A0 boundary target identity drifted")
    _identifier(identity["event_id"], "boundary event_id")
    _identifier(identity["model_recording_id"], "boundary model_recording_id")
    _identifier(identity["patient_uid"], "boundary patient_uid")
    _sha256(identity["event_receipt_sha256"], "boundary event receipt")
    binding = data["binding"]
    if type(binding) is not dict or set(binding) != {
        "event_receipt_sha256",
        "navigation_window_receipt_sha256",
        "source_p0_materialization_receipt_sha256",
        "source_event_model_input_receipt_sha256",
    }:
        raise ValueError("A0 boundary target binding drifted")
    for name, value in binding.items():
        _sha256(value, f"boundary binding {name}")
    if binding["event_receipt_sha256"] != identity["event_receipt_sha256"]:
        raise ValueError("A0 boundary event receipt was rebound")
    resolution = float(data["annotation_resolution_seconds"])
    support = data["frozen_analysis_support_seconds"]
    if (
        not 0 < resolution < 60
        or not isinstance(support, list)
        or len(support) != 2
        or not float(support[0]) < float(support[1])
    ):
        raise ValueError("A0 boundary resolution/support is invalid")
    onset = data["onset"]
    offset = data["offset"]
    if (
        type(onset) is not dict
        or set(onset) != {"status", "interval_seconds"}
        or onset["status"] != "observed_interval"
        or not isinstance(onset["interval_seconds"], list)
        or len(onset["interval_seconds"]) != 2
        or type(offset) is not dict
        or set(offset) != {"status", "interval_seconds"}
        or offset["status"] not in {"observed_interval", "right_censored"}
        or (offset["status"] == "right_censored")
        is not (offset["interval_seconds"] is None)
    ):
        raise ValueError("A0 boundary onset/offset status drifted")
    for interval in (
        onset["interval_seconds"],
        offset["interval_seconds"],
    ):
        if interval is not None and (
            len(interval) != 2
            or float(interval[0]) > float(interval[1])
            or float(interval[0]) < float(support[0]) - 1e-9
            or float(interval[1]) > float(support[1]) + 1e-9
        ):
            raise ValueError("A0 boundary interval exceeds frozen support")
    _validate_content_address(
        data, id_field="target_id", id_prefix="BAIEG-A0-BOUNDARY-"
    )
    if event_row is not None and identity != _event_identity_from_roster(event_row):
        raise ValueError("A0 boundary target disagrees with event roster")
    if navigation_window is not None:
        window = validate_ba_ieg_a0_navigation_window_v1(navigation_window)
        if (
            binding["navigation_window_receipt_sha256"]
            != window["receipt_sha256"]
            or support
            != window["timing"]["analysis_interval_recording_seconds"]
        ):
            raise ValueError("A0 boundary target disagrees with navigation support")
    for expected, name in (
        (
            expected_source_p0_materialization_receipt_sha256,
            "source_p0_materialization_receipt_sha256",
        ),
        (
            expected_source_event_model_input_receipt_sha256,
            "source_event_model_input_receipt_sha256",
        ),
    ):
        if expected is not None and binding[name] != _sha256(expected, name):
            raise ValueError(f"A0 boundary target {name} was rebound")
    return data


def build_ba_ieg_a0_event_failure_terminal_v1(
    *,
    event_row: Mapping[str, Any],
    failure_code: str,
    failure_stage: str,
    canonical_identity_binding_receipt_sha256: str | None,
) -> dict[str, Any]:
    """Build one typed terminal without dropping the failed A0 event."""

    if failure_code not in _EVENT_FAILURE_CODES:
        raise ValueError("unknown typed A0 event failure code")
    stage = _identifier(failure_stage, "failure_stage")
    identity = _event_identity_from_roster(event_row)
    if canonical_identity_binding_receipt_sha256 is not None:
        _sha256(
            canonical_identity_binding_receipt_sha256,
            "canonical_identity_binding_receipt_sha256",
        )
    failure = _finalize_receipt(
        {
            "schema_version": "ba_ieg_a0_event_typed_failure_v1",
            "failure_id": "CONTENT-ADDRESS-PENDING",
            "failure_code": failure_code,
            "failure_stage": stage,
            "event_identity": identity,
            "canonical_identity_binding_receipt_sha256": (
                canonical_identity_binding_receipt_sha256
            ),
            "scope_receipt": {
                "event_retained_in_terminal_denominator": True,
                "failure_interpreted_as_negative_eeg": False,
                "detector_result_claimed": False,
                "clinical_conclusion_claimed": False,
            },
            "receipt_sha256": "CONTENT-ADDRESS-PENDING",
        },
        id_field="failure_id",
        id_prefix="BAIEG-A0-FAIL-",
    )
    terminal = _finalize_receipt(
        {
            "schema_version": "ba_ieg_a0_event_terminal_v1",
            "terminal_id": "CONTENT-ADDRESS-PENDING",
            "status": "failed",
            "event_identity": identity,
            "canonical_identity_binding_receipt_sha256": (
                canonical_identity_binding_receipt_sha256
            ),
            "success_binding": None,
            "artifacts": None,
            "failure_receipt": failure,
            "receipt_sha256": "CONTENT-ADDRESS-PENDING",
        },
        id_field="terminal_id",
        id_prefix="BAIEG-A0-EVTTERM-",
    )
    return _validate_event_terminal(terminal, event_row=event_row)


def build_ba_ieg_a0_event_success_terminal_v1(
    *,
    event_row: Mapping[str, Any],
    canonical_identity_binding_receipt_sha256: str,
    a0_navigation_window_receipt_sha256: str,
    p0_materialization_receipt_sha256: str,
    event_model_input_receipt_sha256: str,
    projection_v2_receipt_sha256: str,
    raw_dependency_sidecar_sha256: str,
    deterministic_target_sidecar_receipt_sha256: str,
    deterministic_target_receipt_sha256: str,
    boundary_target_receipt_sha256: str,
    artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build one fully bound native-12/projection-v2 success terminal."""

    identity = _event_identity_from_roster(event_row)
    hashes = {
        "canonical_identity_binding_receipt_sha256": (
            canonical_identity_binding_receipt_sha256
        ),
        "a0_navigation_window_receipt_sha256": (
            a0_navigation_window_receipt_sha256
        ),
        "p0_materialization_receipt_sha256": p0_materialization_receipt_sha256,
        "event_model_input_receipt_sha256": event_model_input_receipt_sha256,
        "projection_v2_receipt_sha256": projection_v2_receipt_sha256,
        "raw_dependency_sidecar_sha256": raw_dependency_sidecar_sha256,
        "deterministic_target_sidecar_receipt_sha256": (
            deterministic_target_sidecar_receipt_sha256
        ),
        "deterministic_target_receipt_sha256": (
            deterministic_target_receipt_sha256
        ),
        "boundary_target_receipt_sha256": boundary_target_receipt_sha256,
    }
    for name, value in hashes.items():
        _sha256(value, name)
    expected_kinds = dict(_SUCCESS_ARTIFACT_KINDS)
    if set(artifacts) != set(expected_kinds):
        raise ValueError("successful A0 event has missing/unknown artifacts")
    normalized_artifacts = {
        kind: _validate_artifact_reference(
            artifacts[kind],
            expected_kind=kind,
            record_id=identity["model_recording_id"],
        )
        for kind in expected_kinds
    }
    success_binding = {
        **hashes,
        "p0_schema_version": BA_IEG_A0_NATIVE12_P0_SCHEMA_V1,
        "view_profile": BA_IEG_P0_VIEW_PROFILE_NATIVE_12,
        "projection_schema_version": (
            BA_IEG_EVENT_MODEL_INPUT_PROJECTION_SCHEMA_VERSION_V2
        ),
        "navigation_arm": BA_IEG_NAVIGATION_ARM_A0,
        "evaluation_semantics": BA_IEG_A0_EVALUATION_SEMANTICS_V1,
        "model_split": "source_train",
    }
    terminal = _finalize_receipt(
        {
            "schema_version": "ba_ieg_a0_event_terminal_v1",
            "terminal_id": "CONTENT-ADDRESS-PENDING",
            "status": "materialized",
            "event_identity": identity,
            "canonical_identity_binding_receipt_sha256": (
                canonical_identity_binding_receipt_sha256
            ),
            "success_binding": success_binding,
            "artifacts": normalized_artifacts,
            "failure_receipt": None,
            "receipt_sha256": "CONTENT-ADDRESS-PENDING",
        },
        id_field="terminal_id",
        id_prefix="BAIEG-A0-EVTTERM-",
    )
    return _validate_event_terminal(terminal, event_row=event_row)


def _validate_failure_receipt(
    payload: object, *, event_identity: Mapping[str, Any]
) -> dict[str, Any]:
    required = {
        "schema_version",
        "failure_id",
        "failure_code",
        "failure_stage",
        "event_identity",
        "canonical_identity_binding_receipt_sha256",
        "scope_receipt",
        "receipt_sha256",
    }
    if type(payload) is not dict or set(payload) != required:
        raise ValueError("typed A0 event failure fields drifted")
    data = deepcopy(payload)
    if (
        data["schema_version"] != "ba_ieg_a0_event_typed_failure_v1"
        or data["failure_code"] not in _EVENT_FAILURE_CODES
        or data["event_identity"] != event_identity
        or data["scope_receipt"]
        != {
            "event_retained_in_terminal_denominator": True,
            "failure_interpreted_as_negative_eeg": False,
            "detector_result_claimed": False,
            "clinical_conclusion_claimed": False,
        }
    ):
        raise ValueError("typed A0 event failure contract drifted")
    _identifier(data["failure_stage"], "failure_stage")
    binding = data["canonical_identity_binding_receipt_sha256"]
    if binding is not None:
        _sha256(binding, "failure canonical identity binding")
    _validate_content_address(
        data, id_field="failure_id", id_prefix="BAIEG-A0-FAIL-"
    )
    return data


def _validate_event_terminal(
    payload: object, *, event_row: Mapping[str, Any]
) -> dict[str, Any]:
    required = {
        "schema_version",
        "terminal_id",
        "status",
        "event_identity",
        "canonical_identity_binding_receipt_sha256",
        "success_binding",
        "artifacts",
        "failure_receipt",
        "receipt_sha256",
    }
    if type(payload) is not dict or set(payload) != required:
        raise ValueError("A0 event terminal fields drifted")
    data = deepcopy(payload)
    expected_identity = _event_identity_from_roster(event_row)
    if (
        data["schema_version"] != "ba_ieg_a0_event_terminal_v1"
        or data["event_identity"] != expected_identity
        or data["status"] not in {"materialized", "failed"}
    ):
        raise ValueError("A0 event terminal identity/status drifted")
    binding_receipt = data["canonical_identity_binding_receipt_sha256"]
    if binding_receipt is not None:
        _sha256(binding_receipt, "canonical_identity_binding_receipt_sha256")
    if data["status"] == "failed":
        if data["success_binding"] is not None or data["artifacts"] is not None:
            raise ValueError("failed A0 event cannot carry success artifacts")
        failure = _validate_failure_receipt(
            data["failure_receipt"], event_identity=expected_identity
        )
        if failure["canonical_identity_binding_receipt_sha256"] != binding_receipt:
            raise ValueError("failed A0 terminal binding drifted")
        data["failure_receipt"] = failure
    else:
        if binding_receipt is None or data["failure_receipt"] is not None:
            raise ValueError("successful A0 event needs canonical identity binding")
        success = data["success_binding"]
        expected_success_keys = {
            "canonical_identity_binding_receipt_sha256",
            "a0_navigation_window_receipt_sha256",
            "p0_materialization_receipt_sha256",
            "event_model_input_receipt_sha256",
            "projection_v2_receipt_sha256",
            "raw_dependency_sidecar_sha256",
            "deterministic_target_sidecar_receipt_sha256",
            "deterministic_target_receipt_sha256",
            "boundary_target_receipt_sha256",
            "p0_schema_version",
            "view_profile",
            "projection_schema_version",
            "navigation_arm",
            "evaluation_semantics",
            "model_split",
        }
        if type(success) is not dict or set(success) != expected_success_keys:
            raise ValueError("successful A0 event binding fields drifted")
        for name in expected_success_keys:
            if name.endswith("sha256"):
                _sha256(success[name], name)
        if (
            success["canonical_identity_binding_receipt_sha256"]
            != binding_receipt
            or success["p0_schema_version"] != BA_IEG_A0_NATIVE12_P0_SCHEMA_V1
            or success["view_profile"] != BA_IEG_P0_VIEW_PROFILE_NATIVE_12
            or success["projection_schema_version"]
            != BA_IEG_EVENT_MODEL_INPUT_PROJECTION_SCHEMA_VERSION_V2
            or success["navigation_arm"] != BA_IEG_NAVIGATION_ARM_A0
            or success["evaluation_semantics"]
            != BA_IEG_A0_EVALUATION_SEMANTICS_V1
            or success["model_split"] != "source_train"
        ):
            raise ValueError("successful A0 event authority/profile drifted")
        artifacts = data["artifacts"]
        if type(artifacts) is not dict or set(artifacts) != dict(
            _SUCCESS_ARTIFACT_KINDS
        ).keys():
            raise ValueError("successful A0 event artifact roster drifted")
        data["artifacts"] = {
            kind: _validate_artifact_reference(
                artifacts[kind],
                expected_kind=kind,
                record_id=expected_identity["model_recording_id"],
            )
            for kind, _suffix in _SUCCESS_ARTIFACT_KINDS
        }
    _validate_content_address(
        data, id_field="terminal_id", id_prefix="BAIEG-A0-EVTTERM-"
    )
    return data


def build_ba_ieg_a0_record_terminal_v1(
    *,
    record_row: Mapping[str, Any],
    event_rows: Sequence[Mapping[str, Any]],
    event_terminals: Sequence[Mapping[str, Any]],
    canonical_identity_binding_receipt_sha256: str | None,
    record_failure_code: str | None = None,
    record_failure_stage: str | None = None,
) -> dict[str, Any]:
    """Freeze one record only after all of its expected events are terminal."""

    record_id = record_row["model_recording_id"]
    expected_rows = list(event_rows)
    if [row["model_event_id"] for row in expected_rows] != list(
        record_row["model_event_ids"]
    ):
        raise ValueError("record event rows disagree with the A0 record roster")
    if len(event_terminals) != len(expected_rows):
        raise ValueError("record commit lost an expected A0 event terminal")
    terminals = [
        _validate_event_terminal(value, event_row=row)
        for value, row in zip(event_terminals, expected_rows)
    ]
    if canonical_identity_binding_receipt_sha256 is not None:
        _sha256(
            canonical_identity_binding_receipt_sha256,
            "canonical_identity_binding_receipt_sha256",
        )
    failure: dict[str, Any] | None = None
    if record_failure_code is not None or record_failure_stage is not None:
        if (
            record_failure_code not in _RECORD_FAILURE_CODES
            or record_failure_stage is None
            or canonical_identity_binding_receipt_sha256 is not None
        ):
            raise ValueError("record failure requires a typed code/stage and no binding")
        if any(terminal["status"] != "failed" for terminal in terminals):
            raise ValueError("failed record cannot contain materialized events")
        failure = {
            "failure_code": record_failure_code,
            "failure_stage": _identifier(record_failure_stage, "record failure stage"),
        }
    elif canonical_identity_binding_receipt_sha256 is None:
        raise ValueError("non-failed record requires canonical identity binding")
    elif any(
        terminal["canonical_identity_binding_receipt_sha256"]
        != canonical_identity_binding_receipt_sha256
        for terminal in terminals
    ):
        raise ValueError("record/event canonical identity bindings drifted")

    body = {
        "schema_version": "ba_ieg_a0_record_terminal_v1",
        "record_terminal_id": "CONTENT-ADDRESS-PENDING",
        "status": "failed" if failure is not None else "committed",
        "record_identity": {
            "model_recording_id": record_id,
            "source_recording_id": record_row["source_recording_id"],
            "patient_uid": record_row["patient_uid"],
            "model_split": "source_train",
            "record_roster_receipt_sha256": record_row[
                "record_roster_receipt_sha256"
            ],
            "source_container_sha256": record_row["source_container_sha256"],
        },
        "canonical_identity_binding_receipt_sha256": (
            canonical_identity_binding_receipt_sha256
        ),
        "expected_event_count": len(expected_rows),
        "event_terminals": terminals,
        "record_failure": failure,
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    return _validate_record_terminal(
        _finalize_receipt(
            body,
            id_field="record_terminal_id",
            id_prefix="BAIEG-A0-RECTERM-",
        ),
        record_row=record_row,
        event_rows=expected_rows,
    )


def _validate_record_terminal(
    payload: object,
    *,
    record_row: Mapping[str, Any],
    event_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    required = {
        "schema_version",
        "record_terminal_id",
        "status",
        "record_identity",
        "canonical_identity_binding_receipt_sha256",
        "expected_event_count",
        "event_terminals",
        "record_failure",
        "receipt_sha256",
    }
    if type(payload) is not dict or set(payload) != required:
        raise ValueError("A0 record terminal fields drifted")
    data = deepcopy(payload)
    expected_identity = {
        "model_recording_id": record_row["model_recording_id"],
        "source_recording_id": record_row["source_recording_id"],
        "patient_uid": record_row["patient_uid"],
        "model_split": "source_train",
        "record_roster_receipt_sha256": record_row[
            "record_roster_receipt_sha256"
        ],
        "source_container_sha256": record_row["source_container_sha256"],
    }
    if (
        data["schema_version"] != "ba_ieg_a0_record_terminal_v1"
        or data["status"] not in {"committed", "failed"}
        or data["record_identity"] != expected_identity
        or data["expected_event_count"] != len(event_rows)
        or not isinstance(data["event_terminals"], list)
        or len(data["event_terminals"]) != len(event_rows)
    ):
        raise ValueError("A0 record terminal identity/count drifted")
    terminals = [
        _validate_event_terminal(value, event_row=row)
        for value, row in zip(data["event_terminals"], event_rows)
    ]
    data["event_terminals"] = terminals
    binding = data["canonical_identity_binding_receipt_sha256"]
    if data["status"] == "failed":
        failure = data["record_failure"]
        if (
            binding is not None
            or type(failure) is not dict
            or set(failure) != {"failure_code", "failure_stage"}
            or failure["failure_code"] not in _RECORD_FAILURE_CODES
            or any(terminal["status"] != "failed" for terminal in terminals)
        ):
            raise ValueError("failed A0 record terminal is inconsistent")
        _identifier(failure["failure_stage"], "record failure stage")
    else:
        _sha256(binding, "record canonical identity binding")
        if data["record_failure"] is not None or any(
            terminal["canonical_identity_binding_receipt_sha256"] != binding
            for terminal in terminals
        ):
            raise ValueError("committed A0 record binding drifted")
    _validate_content_address(
        data,
        id_field="record_terminal_id",
        id_prefix="BAIEG-A0-RECTERM-",
    )
    return data


def _events_by_record(roster: Mapping[str, Any]) -> dict[str, list[Mapping[str, Any]]]:
    result = {row["model_recording_id"]: [] for row in roster["records"]}
    for event in roster["events"]:
        result[event["model_recording_id"]].append(event)
    return result


def _assemble_manifest(
    *,
    candidate_roster: Mapping[str, Any],
    record_terminals: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    validate_ba_ieg_a0_oracle_navigation_candidate_roster_v1(candidate_roster)
    roster = deepcopy(dict(candidate_roster))
    record_lookup = {
        row["model_recording_id"]: row for row in roster["records"]
    }
    event_lookup = _events_by_record(roster)
    if not isinstance(record_terminals, Sequence) or isinstance(
        record_terminals, (str, bytes)
    ):
        raise TypeError("record_terminals must be a sequence")
    supplied_ids: list[str] = []
    normalized: list[dict[str, Any]] = []
    for item in record_terminals:
        if not isinstance(item, Mapping):
            raise TypeError("record terminal must be an object")
        identity = item.get("record_identity")
        record_id = identity.get("model_recording_id") if isinstance(identity, Mapping) else None
        if record_id not in record_lookup:
            raise ValueError("record terminal is absent from the frozen A0 roster")
        supplied_ids.append(record_id)
        normalized.append(
            _validate_record_terminal(
                item,
                record_row=record_lookup[record_id],
                event_rows=event_lookup[record_id],
            )
        )
    if len(supplied_ids) != len(set(supplied_ids)):
        raise ValueError("A0 dataset contains duplicate record terminals")
    normalized.sort(
        key=lambda item: list(record_lookup).index(
            item["record_identity"]["model_recording_id"]
        )
    )
    selected_ids = [item["record_identity"]["model_recording_id"] for item in normalized]
    full_ids = list(record_lookup)
    complete = selected_ids == full_ids
    dataset_state = (
        "complete_terminal_inventory"
        if complete
        else "partial_smoke_terminal_inventory"
    )
    event_terminals = [
        terminal
        for record in normalized
        for terminal in record["event_terminals"]
    ]
    selected_zero = sum(record_lookup[item]["expected_unique_occurrence_count"] == 0 for item in selected_ids)
    counts = {
        "frozen_denominator": deepcopy(roster["denominator_contract"]),
        "selected_records": len(normalized),
        "selected_events": len(event_terminals),
        "selected_zero_event_records": selected_zero,
        "terminal_records": len(normalized),
        "record_failures": sum(item["status"] == "failed" for item in normalized),
        "terminal_events": len(event_terminals),
        "materialized_events": sum(
            item["status"] == "materialized" for item in event_terminals
        ),
        "failed_events": sum(item["status"] == "failed" for item in event_terminals),
    }
    body = {
        "schema_version": BA_IEG_A0_NATIVE12_DATASET_MANIFEST_SCHEMA_V2,
        "manifest_id": "CONTENT-ADDRESS-PENDING",
        "method_id": BA_IEG_A0_NATIVE12_DATASET_METHOD_ID_V1,
        "dataset_state": dataset_state,
        "complete_frozen_denominator": complete,
        "model_split": "source_train",
        "navigation_arm": BA_IEG_NAVIGATION_ARM_A0,
        "evaluation_semantics": BA_IEG_A0_EVALUATION_SEMANTICS_V1,
        "view_profile": BA_IEG_P0_VIEW_PROFILE_NATIVE_12,
        "p0_schema_version": BA_IEG_A0_NATIVE12_P0_SCHEMA_V1,
        "projection_schema_version": (
            BA_IEG_EVENT_MODEL_INPUT_PROJECTION_SCHEMA_VERSION_V2
        ),
        "a0_candidate_roster_receipt_sha256": roster["receipt_sha256"],
        "a0_oracle_navigation_receipt_sha256": roster[
            "oracle_navigation_receipt_sha256"
        ],
        "source_identity_binding_receipt_sha256": roster[
            "identity_binding_sha256"
        ],
        "selected_record_ids": selected_ids,
        "counts": counts,
        "record_terminals": normalized,
        "record_terminal_roster_sha256": _canonical_sha256(
            [item["receipt_sha256"] for item in normalized]
        ),
        "scope_receipt": deepcopy(_SCOPE_RECEIPT),
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    return _finalize_receipt(
        body, id_field="manifest_id", id_prefix="BAIEG-A0-DATASET-"
    )


def build_ba_ieg_a0_native12_dataset_manifest_v2(
    *,
    candidate_roster: Mapping[str, Any],
    record_terminals: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return validate_ba_ieg_a0_native12_dataset_manifest_v2(
        _assemble_manifest(
            candidate_roster=candidate_roster,
            record_terminals=record_terminals,
        ),
        candidate_roster=candidate_roster,
    )


def validate_ba_ieg_a0_native12_dataset_manifest_v2(
    payload: object,
    *,
    candidate_roster: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "schema_version",
        "manifest_id",
        "method_id",
        "dataset_state",
        "complete_frozen_denominator",
        "model_split",
        "navigation_arm",
        "evaluation_semantics",
        "view_profile",
        "p0_schema_version",
        "projection_schema_version",
        "a0_candidate_roster_receipt_sha256",
        "a0_oracle_navigation_receipt_sha256",
        "source_identity_binding_receipt_sha256",
        "selected_record_ids",
        "counts",
        "record_terminals",
        "record_terminal_roster_sha256",
        "scope_receipt",
        "receipt_sha256",
    }
    if type(payload) is not dict or set(payload) != required:
        raise ValueError("A0 native-12 dataset manifest has missing/unknown fields")
    data = deepcopy(payload)
    validate_ba_ieg_a0_oracle_navigation_candidate_roster_v1(candidate_roster)
    roster = deepcopy(dict(candidate_roster))
    if (
        data["schema_version"] != BA_IEG_A0_NATIVE12_DATASET_MANIFEST_SCHEMA_V2
        or data["method_id"] != BA_IEG_A0_NATIVE12_DATASET_METHOD_ID_V1
        or data["dataset_state"] not in _DATASET_STATES
        or type(data["complete_frozen_denominator"]) is not bool
        or data["model_split"] != "source_train"
        or data["navigation_arm"] != BA_IEG_NAVIGATION_ARM_A0
        or data["evaluation_semantics"] != BA_IEG_A0_EVALUATION_SEMANTICS_V1
        or data["view_profile"] != BA_IEG_P0_VIEW_PROFILE_NATIVE_12
        or data["p0_schema_version"] != BA_IEG_A0_NATIVE12_P0_SCHEMA_V1
        or data["projection_schema_version"]
        != BA_IEG_EVENT_MODEL_INPUT_PROJECTION_SCHEMA_VERSION_V2
        or data["scope_receipt"] != _SCOPE_RECEIPT
        or data["a0_candidate_roster_receipt_sha256"] != roster["receipt_sha256"]
        or data["a0_oracle_navigation_receipt_sha256"]
        != roster["oracle_navigation_receipt_sha256"]
        or data["source_identity_binding_receipt_sha256"]
        != roster["identity_binding_sha256"]
    ):
        raise ValueError("A0 native-12 dataset authority/profile drifted")
    if not isinstance(data["record_terminals"], list):
        raise ValueError("A0 dataset record terminals must be an array")
    replayed = _assemble_manifest(
        candidate_roster=roster,
        record_terminals=data["record_terminals"],
    )
    if replayed != data:
        raise ValueError("A0 native-12 dataset manifest did not replay exactly")
    return data


def _read_reference_file(root: Path, reference: Mapping[str, Any]) -> None:
    relative = reference["relative_path"]
    root_resolved = root.resolve(strict=True)
    logical = root_resolved / relative
    parent = logical.parent.resolve(strict=True)
    if parent != root_resolved and root_resolved not in parent.parents:
        raise ValueError("A0 resume artifact path escapes dataset root")
    path = parent / logical.name
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError("A0 resume artifact must be a single-link regular file")
        if before.st_size != reference["file_size_bytes"]:
            raise ValueError("A0 resume artifact size drifted")
        digest = hashlib.sha256()
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError("A0 resume artifact was truncated")
            remaining -= len(chunk)
            digest.update(chunk)
        if os.read(descriptor, 1):
            raise ValueError("A0 resume artifact grew during verification")
        after = os.fstat(descriptor)
        for name in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"):
            if getattr(before, name) != getattr(after, name):
                raise ValueError("A0 resume artifact changed during verification")
    finally:
        os.close(descriptor)
    if digest.hexdigest() != reference["file_sha256"]:
        raise ValueError("A0 resume artifact SHA-256 drifted")


def verify_ba_ieg_a0_record_terminal_artifacts_v1(
    root: str | Path,
    record_terminal: Mapping[str, Any],
    *,
    candidate_roster: Mapping[str, Any],
) -> dict[str, Any]:
    """Hash-check an already committed record before append-only resume reuse."""

    validate_ba_ieg_a0_oracle_navigation_candidate_roster_v1(candidate_roster)
    roster = deepcopy(dict(candidate_roster))
    record_id = record_terminal["record_identity"]["model_recording_id"]
    record_rows = [
        row for row in roster["records"] if row["model_recording_id"] == record_id
    ]
    if len(record_rows) != 1:
        raise ValueError("resume record is absent from A0 roster")
    events = _events_by_record(roster)[record_id]
    validated = _validate_record_terminal(
        record_terminal, record_row=record_rows[0], event_rows=events
    )
    root_path = Path(root)
    if not root_path.resolve(strict=True).is_dir():
        raise ValueError("A0 dataset root must be an existing directory")
    seen_paths: set[str] = set()
    for terminal in validated["event_terminals"]:
        if terminal["status"] != "materialized":
            continue
        for reference in terminal["artifacts"].values():
            if reference["relative_path"] in seen_paths:
                raise ValueError("A0 record commit reuses one artifact path")
            seen_paths.add(reference["relative_path"])
            _read_reference_file(root_path, reference)
    return validated


def write_ba_ieg_a0_native12_dataset_manifest_v2(
    payload: Mapping[str, Any],
    destination: str | Path,
    *,
    candidate_roster: Mapping[str, Any],
) -> Path:
    """Append-only canonical-JSON publication for a terminal dataset manifest."""

    validated = validate_ba_ieg_a0_native12_dataset_manifest_v2(
        payload, candidate_roster=candidate_roster
    )
    path = Path(destination)
    if path.suffix != ".json" or not path.parent.resolve(strict=True).is_dir():
        raise ValueError("dataset manifest destination must be .json in an existing directory")
    if os.path.lexists(path):
        raise FileExistsError("dataset manifest already exists")
    encoded = _canonical_json_bytes(validated)
    descriptor = -1
    temporary: str | None = None
    try:
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.tmp-", dir=path.parent
        )
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = -1
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary is not None and os.path.lexists(temporary):
            os.unlink(temporary)
    return path


__all__ = [
    "BA_IEG_A0_NATIVE12_DATASET_MANIFEST_SCHEMA_V2",
    "BA_IEG_A0_NATIVE12_DATASET_METHOD_ID_V1",
    "BA_IEG_A0_NATIVE12_P0_SCHEMA_V1",
    "build_ba_ieg_a0_artifact_reference_v1",
    "build_ba_ieg_a0_boundary_target_v1",
    "build_ba_ieg_a0_event_failure_terminal_v1",
    "build_ba_ieg_a0_event_success_terminal_v1",
    "build_ba_ieg_a0_native12_dataset_manifest_v2",
    "build_ba_ieg_a0_record_terminal_v1",
    "validate_ba_ieg_a0_native12_dataset_manifest_v2",
    "validate_ba_ieg_a0_boundary_target_v1",
    "verify_ba_ieg_a0_record_terminal_artifacts_v1",
    "write_ba_ieg_a0_native12_dataset_manifest_v2",
]
