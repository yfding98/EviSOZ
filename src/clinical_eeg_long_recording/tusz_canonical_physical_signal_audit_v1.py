"""Resumable canonical-physical-signal duplicate audit for TUSZ.

The complete TUSZ roster v2 closes exact *container-byte* equivalence.  Two
EDF containers can nevertheless differ in identity/header bytes while
carrying the same physical EEG samples.  This module adds a separate,
signal-only audit layer:

* every exact-container analysis identity is materialized through the
  canonical physical EEG loader;
* the float32-volts tensor hash is combined with the physical sampling clock;
* deterministic shards retain success and failure outcomes for every row;
* a full audit quarantines physical equivalence classes crossing a patient or
  an official split, and deduplicates same-patient/same-split aliases; and
* a downstream analysis projection is authorized only when every source row
  has a successful canonical outcome.

No ``csv_bi`` contents, EDF+ annotations, spreadsheet, report, clinical text,
or doctor label is accepted by any public API in this module.
"""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

from .canonical_edf_materialization import (
    CanonicalEDFPhysicalSourceIdentity,
    load_canonical_edf_physical_source_identity,
)
from .tusz_complete_detector_roster_v2 import (
    TUSZ_ANALYSIS_IDENTITY_FIELDS_V2,
    validate_tusz_analysis_identity_projection_v2,
    validate_tusz_complete_detector_roster_v2,
)


TUSZ_CANONICAL_PHYSICAL_OUTCOME_V1_SCHEMA_VERSION = (
    "tusz_canonical_physical_signal_outcome_v1"
)
TUSZ_CANONICAL_PHYSICAL_OUTCOME_V1_METHOD_ID = (
    "canonical_float32_volts_tensor_plus_sampling_clock_v1"
)
TUSZ_CANONICAL_PHYSICAL_SHARD_V1_SCHEMA_VERSION = (
    "tusz_canonical_physical_signal_audit_shard_v1"
)
TUSZ_CANONICAL_PHYSICAL_SHARD_V1_METHOD_ID = (
    "canonical_projection_index_modulo_resumable_shard_v1"
)
TUSZ_CANONICAL_PHYSICAL_AUDIT_V1_SCHEMA_VERSION = (
    "tusz_canonical_physical_signal_duplicate_audit_v1"
)
TUSZ_CANONICAL_PHYSICAL_AUDIT_V1_METHOD_ID = (
    "complete_signal_equivalence_class_cross_boundary_quarantine_v1"
)
TUSZ_CANONICAL_PHYSICAL_PROJECTION_V1_SCHEMA_VERSION = (
    "tusz_canonical_physical_analysis_projection_v1"
)
TUSZ_CANONICAL_PHYSICAL_PROJECTION_V1_METHOD_ID = (
    "one_unit_per_safe_physical_equivalence_class_v1"
)

_PHYSICAL_FIELDS = {
    "canonical_source_tensor_sha256",
    "canonical_source_signal_sha256",
    "canonical_source_header_receipt_sha256",
    "canonical_reader_policy",
    "observed_channel_ids",
    "sampling_rate_fraction",
    "sample_count",
    "duration_seconds_fraction",
    "canonical_physical_equivalence_sha256",
}
_OUTCOME_FIELDS = {
    "schema_version",
    "method_id",
    "terminal_status",
    "source_analysis_projection_receipt_sha256",
    *TUSZ_ANALYSIS_IDENTITY_FIELDS_V2,
    "container_sha256_recomputed",
    "physical_signal",
    "failure",
    "scope_receipt",
    "receipt_sha256",
}
_PHYSICAL_PROJECTION_EXTRA_FIELDS = (
    "canonical_physical_equivalence_id",
    "canonical_physical_source_tensor_sha256",
    "physical_equivalence_canonical_analysis_identity_id",
    "source_physical_identity_multiplicity",
)
_MODEL_TO_OFFICIAL_SPLIT = {
    "source_train": "train",
    "source_dev": "dev",
    "source_eval": "eval",
}


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_prefixed_sha256(value: object, prefix: str) -> bool:
    return (
        isinstance(value, str)
        and value.startswith(prefix)
        and _is_sha256(value[len(prefix) :])
    )


def _identifier(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"{context} must be a non-empty normalized string")
    return value


def _nonnegative_integer(value: object, context: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{context} must be a non-negative integer")
    return value


def _positive_integer(value: object, context: str) -> int:
    result = _nonnegative_integer(value, context)
    if result < 1:
        raise ValueError(f"{context} must be positive")
    return result


def _fraction(value: object, context: str) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or type(value[0]) is not int
        or type(value[1]) is not int
        or value[0] < 0
        or value[1] <= 0
    ):
        raise ValueError(f"{context} must be a non-negative reduced fraction")
    reduced = Fraction(value[0], value[1])
    if [reduced.numerator, reduced.denominator] != value:
        raise ValueError(f"{context} must be reduced")
    return reduced.numerator, reduced.denominator


def _scope_receipt() -> dict[str, object]:
    return {
        "canonical_physical_eeg_is_only_payload_authority": True,
        "edf_signal_header_is_only_metadata_authority": True,
        "edf_container_hash_verification_required": True,
        "edf_patient_or_recording_header_values_retained": False,
        "edf_annotations_read": False,
        "csv_bi_contents_read": False,
        "seizure_interval_or_label_values_read": False,
        "spreadsheet_or_clinical_text_read": False,
        "doctor_labels_or_reports_read": False,
    }


def _exact_duplicate_scope_receipt() -> dict[str, object]:
    return {
        "canonical_physical_equivalence_scope": (
            "exact_float32_volts_tensor_channel_roster_sampling_clock_and_"
            "sample_count_only"
        ),
        "identical_payload_discordant_sampling_clock_quarantined": True,
        "near_duplicate_or_partial_overlap_audit_performed": False,
        "near_duplicate_or_partial_overlap_exclusion_claim_authorized": False,
    }


def _physical_projection_role_permissions() -> dict[str, dict[str, bool]]:
    return {
        "source_train": {
            "model_fit_identity_authorized": True,
            "development_calibration_identity_authorized": False,
            "locked_evaluation_identity_export_authorized": False,
            "model_execution_authorized_by_projection": False,
            "host_admission_required": False,
            "reference_access_authorized": False,
        },
        "source_dev": {
            "model_fit_identity_authorized": False,
            "development_calibration_identity_authorized": True,
            "locked_evaluation_identity_export_authorized": False,
            "model_execution_authorized_by_projection": False,
            "host_admission_required": False,
            "reference_access_authorized": False,
        },
        "source_eval": {
            "model_fit_identity_authorized": False,
            "development_calibration_identity_authorized": False,
            "locked_evaluation_identity_export_authorized": True,
            "model_execution_authorized_by_projection": False,
            "host_admission_required": True,
            "reference_access_authorized": False,
        },
    }


def _physical_projection_reference_access_receipt() -> dict[str, object]:
    return {
        "reference_path_argument_accepted": False,
        "reference_files_opened": 0,
        "csv_bi_files_opened": 0,
        "csv_bi_bytes_read": 0,
        "csv_bi_contents_read": False,
        "seizure_interval_or_label_values_read": False,
        "edf_annotations_read": False,
        "spreadsheet_or_clinical_text_read": False,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _safe_edf_path(tusz_root: str | Path, local_edf_path: str) -> Path:
    root_path = Path(tusz_root)
    if root_path.is_symlink() or not root_path.is_dir():
        raise ValueError("TUSZ root must be a regular non-symlink directory")
    root = root_path.resolve(strict=True)
    relative = PurePosixPath(local_edf_path)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or "\\" in local_edf_path
        or relative.suffix.lower() != ".edf"
    ):
        raise ValueError("TUSZ projection EDF path is unsafe")
    candidate = root.joinpath(*relative.parts)
    if candidate.is_symlink():
        raise ValueError("TUSZ projected EDF must not be a symbolic link")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("TUSZ projected EDF escapes the dataset root") from exc
    if not resolved.is_file():
        raise ValueError("TUSZ projected EDF is not a regular file")
    return resolved


def _physical_equivalence_sha256(
    *,
    source_tensor_sha256: str,
    observed_channel_ids: Sequence[str],
    sampling_rate_fraction: Sequence[int],
    sample_count: int,
) -> str:
    if not _is_sha256(source_tensor_sha256):
        raise ValueError("canonical tensor hash must be SHA-256")
    rate = list(sampling_rate_fraction)
    _fraction(rate, "canonical sampling rate")
    _positive_integer(sample_count, "canonical sample count")
    return _canonical_sha256(
        {
            "domain": "canonical-physical-eeg-equivalence-v1",
            "canonical_physical_unit": "V",
            "float_payload": "float32-le",
            "source_tensor_sha256": source_tensor_sha256,
            "observed_channel_ids": list(observed_channel_ids),
            "sampling_rate_fraction": rate,
            "sample_count": sample_count,
        }
    )


def _outcome_identity(
    projection_row: Mapping[str, object],
    *,
    source_projection_receipt_sha256: str,
) -> dict[str, object]:
    if not _is_sha256(source_projection_receipt_sha256):
        raise ValueError("source projection receipt must be SHA-256")
    return {
        "source_analysis_projection_receipt_sha256": (source_projection_receipt_sha256),
        **{
            field: deepcopy(projection_row[field])
            for field in TUSZ_ANALYSIS_IDENTITY_FIELDS_V2
        },
    }


def materialize_tusz_canonical_physical_outcome_v1(
    *,
    projection_row: Mapping[str, object],
    source_projection_receipt_sha256: str,
    tusz_root: str | Path,
    record_loader: Callable[[str | Path], CanonicalEDFPhysicalSourceIdentity] = (
        load_canonical_edf_physical_source_identity
    ),
    container_hasher: Callable[[Path], str] = _sha256_file,
) -> dict[str, Any]:
    """Materialize one signal-only success outcome.

    Exceptions are intentionally not swallowed.  The shard runner converts a
    caught exception to a typed terminal failure outcome, preserving the full
    denominator without serializing free-form exception text.
    """

    identity = _outcome_identity(
        projection_row,
        source_projection_receipt_sha256=source_projection_receipt_sha256,
    )
    path = _safe_edf_path(tusz_root, str(projection_row["local_edf_path"]))
    before = path.stat()
    container_sha256 = container_hasher(path)
    if container_sha256 != projection_row["source_edf_container_sha256"]:
        raise ValueError("TUSZ EDF container hash differs from frozen projection")
    record = record_loader(path)
    after = path.stat()
    before_binding = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_binding = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_binding != after_binding:
        raise ValueError("TUSZ EDF changed during canonical physical audit")

    source_header = record.source_header_receipt
    header_rows = source_header["channel_signal_headers"]
    if not header_rows:
        raise ValueError("canonical record has no observed physical channels")
    rates = {
        (
            int(row["sampling_rate_numerator"]),
            int(row["sampling_rate_denominator"]),
        )
        for row in header_rows
    }
    counts = {int(row["sample_count"]) for row in header_rows}
    if len(rates) != 1 or len(counts) != 1:
        raise ValueError("canonical physical audit requires one shared EEG clock")
    rate_numerator, rate_denominator = next(iter(rates))
    sample_count = next(iter(counts))
    rate = Fraction(rate_numerator, rate_denominator)
    duration = Fraction(sample_count, 1) / rate
    observed = list(record.observed_channel_ids)
    tensor_hash = str(source_header["source_tensor_sha256"])
    physical_hash = _physical_equivalence_sha256(
        source_tensor_sha256=tensor_hash,
        observed_channel_ids=observed,
        sampling_rate_fraction=[rate.numerator, rate.denominator],
        sample_count=sample_count,
    )
    body: dict[str, Any] = {
        "schema_version": TUSZ_CANONICAL_PHYSICAL_OUTCOME_V1_SCHEMA_VERSION,
        "method_id": TUSZ_CANONICAL_PHYSICAL_OUTCOME_V1_METHOD_ID,
        "terminal_status": "success",
        **identity,
        "container_sha256_recomputed": container_sha256,
        "physical_signal": {
            "canonical_source_tensor_sha256": tensor_hash,
            "canonical_source_signal_sha256": source_header["source_signal_sha256"],
            "canonical_source_header_receipt_sha256": source_header["receipt_sha256"],
            "canonical_reader_policy": source_header["reader_policy"],
            "observed_channel_ids": observed,
            "sampling_rate_fraction": [rate.numerator, rate.denominator],
            "sample_count": sample_count,
            "duration_seconds_fraction": [
                duration.numerator,
                duration.denominator,
            ],
            "canonical_physical_equivalence_sha256": physical_hash,
        },
        "failure": None,
        "scope_receipt": _scope_receipt(),
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    body["receipt_sha256"] = _canonical_sha256(body)
    return validate_tusz_canonical_physical_outcome_v1(
        body,
        projection_row=projection_row,
        source_projection_receipt_sha256=source_projection_receipt_sha256,
    )


def build_tusz_canonical_physical_failure_outcome_v1(
    *,
    projection_row: Mapping[str, object],
    source_projection_receipt_sha256: str,
    failure_stage: str,
    exception_type: str,
) -> dict[str, Any]:
    """Create a terminal, privacy-minimal failure row for denominator closure."""

    identity = _outcome_identity(
        projection_row,
        source_projection_receipt_sha256=source_projection_receipt_sha256,
    )
    body: dict[str, Any] = {
        "schema_version": TUSZ_CANONICAL_PHYSICAL_OUTCOME_V1_SCHEMA_VERSION,
        "method_id": TUSZ_CANONICAL_PHYSICAL_OUTCOME_V1_METHOD_ID,
        "terminal_status": "failure",
        **identity,
        "container_sha256_recomputed": None,
        "physical_signal": None,
        "failure": {
            "failure_stage": _identifier(failure_stage, "failure stage"),
            "reason_code": "canonical_physical_materialization_or_binding_failed",
            "exception_type": _identifier(exception_type, "exception type"),
            "free_form_exception_text_retained": False,
        },
        "scope_receipt": _scope_receipt(),
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    body["receipt_sha256"] = _canonical_sha256(body)
    return validate_tusz_canonical_physical_outcome_v1(
        body,
        projection_row=projection_row,
        source_projection_receipt_sha256=source_projection_receipt_sha256,
    )


def validate_tusz_canonical_physical_outcome_v1(
    payload: object,
    *,
    projection_row: Mapping[str, object] | None = None,
    source_projection_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    if type(payload) is not dict or set(payload) != _OUTCOME_FIELDS:
        raise ValueError("canonical physical outcome fields drifted")
    data = deepcopy(payload)
    if (
        data["schema_version"] != TUSZ_CANONICAL_PHYSICAL_OUTCOME_V1_SCHEMA_VERSION
        or data["method_id"] != TUSZ_CANONICAL_PHYSICAL_OUTCOME_V1_METHOD_ID
        or data["terminal_status"] not in {"success", "failure"}
    ):
        raise ValueError("canonical physical outcome schema/status drifted")
    if not _is_sha256(data["source_analysis_projection_receipt_sha256"]):
        raise ValueError("canonical physical source projection hash is invalid")
    if (
        source_projection_receipt_sha256 is not None
        and data["source_analysis_projection_receipt_sha256"]
        != source_projection_receipt_sha256
    ):
        raise ValueError("canonical physical outcome binds another projection")
    for field in TUSZ_ANALYSIS_IDENTITY_FIELDS_V2:
        if projection_row is not None and data[field] != projection_row[field]:
            raise ValueError("canonical physical outcome identity binding drifted")
    if not _is_sha256(data["source_edf_container_sha256"]):
        raise ValueError("canonical physical source container hash is invalid")
    if data["scope_receipt"] != _scope_receipt():
        raise ValueError("canonical physical outcome violates EEG-only scope")

    if data["terminal_status"] == "success":
        if (
            data["failure"] is not None
            or not _is_sha256(data["container_sha256_recomputed"])
            or data["container_sha256_recomputed"]
            != data["source_edf_container_sha256"]
        ):
            raise ValueError(
                "successful canonical outcome has invalid container binding"
            )
        physical = data["physical_signal"]
        if type(physical) is not dict or set(physical) != _PHYSICAL_FIELDS:
            raise ValueError("successful canonical physical payload fields drifted")
        for field in (
            "canonical_source_tensor_sha256",
            "canonical_source_signal_sha256",
            "canonical_source_header_receipt_sha256",
            "canonical_physical_equivalence_sha256",
        ):
            if not _is_sha256(physical[field]):
                raise ValueError(f"canonical physical {field} is invalid")
        _identifier(physical["canonical_reader_policy"], "canonical reader policy")
        observed = physical["observed_channel_ids"]
        if (
            not isinstance(observed, list)
            or not observed
            or len(observed) != len(set(observed))
            or not all(isinstance(item, str) and item for item in observed)
        ):
            raise ValueError("canonical observed-channel roster is invalid")
        rate = list(physical["sampling_rate_fraction"])
        rate_numerator, _ = _fraction(rate, "canonical sampling rate")
        if rate_numerator <= 0:
            raise ValueError("canonical sampling rate must be positive")
        sample_count = _positive_integer(
            physical["sample_count"], "canonical sample count"
        )
        duration_numerator, duration_denominator = _fraction(
            physical["duration_seconds_fraction"], "canonical duration"
        )
        expected_duration = Fraction(sample_count, 1) / Fraction(*rate)
        if [duration_numerator, duration_denominator] != [
            expected_duration.numerator,
            expected_duration.denominator,
        ]:
            raise ValueError("canonical physical duration does not match its clock")
        expected_physical_hash = _physical_equivalence_sha256(
            source_tensor_sha256=physical["canonical_source_tensor_sha256"],
            observed_channel_ids=observed,
            sampling_rate_fraction=rate,
            sample_count=sample_count,
        )
        if physical["canonical_physical_equivalence_sha256"] != expected_physical_hash:
            raise ValueError("canonical physical equivalence hash is not replayable")
    else:
        if (
            data["container_sha256_recomputed"] is not None
            or data["physical_signal"] is not None
        ):
            raise ValueError(
                "failed canonical outcome must not carry physical evidence"
            )
        failure = data["failure"]
        required_failure = {
            "failure_stage",
            "reason_code",
            "exception_type",
            "free_form_exception_text_retained",
        }
        if type(failure) is not dict or set(failure) != required_failure:
            raise ValueError("canonical physical failure fields drifted")
        _identifier(failure["failure_stage"], "failure stage")
        _identifier(failure["exception_type"], "exception type")
        if (
            failure["reason_code"]
            != "canonical_physical_materialization_or_binding_failed"
            or failure["free_form_exception_text_retained"] is not False
        ):
            raise ValueError("canonical physical failure policy drifted")

    digest = deepcopy(data)
    digest["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if not _is_sha256(data["receipt_sha256"]) or data[
        "receipt_sha256"
    ] != _canonical_sha256(digest):
        raise ValueError("canonical physical outcome receipt hash drifted")
    return data


def select_tusz_canonical_physical_shard_rows_v1(
    source_projection: object,
    *,
    shard_count: int,
    shard_index: int,
    source_roster: object | None = None,
) -> list[dict[str, Any]]:
    projection = validate_tusz_analysis_identity_projection_v2(
        source_projection,
        source_roster=source_roster,
    )
    return _select_tusz_canonical_physical_shard_rows_from_validated_v1(
        projection,
        shard_count=shard_count,
        shard_index=shard_index,
    )


def _select_tusz_canonical_physical_shard_rows_from_validated_v1(
    projection: Mapping[str, Any],
    *,
    shard_count: int,
    shard_index: int,
) -> list[dict[str, Any]]:
    count = _positive_integer(shard_count, "shard count")
    index = _nonnegative_integer(shard_index, "shard index")
    if index >= count:
        raise ValueError("shard index must be smaller than shard count")
    return [
        deepcopy(row)
        for global_index, row in enumerate(projection["records"])
        if global_index % count == index
    ]


def _shard_binding(
    *,
    roster: Mapping[str, Any],
    projection: Mapping[str, Any],
    shard_count: int,
    shard_index: int,
    selected_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    identities = [row["analysis_identity_id"] for row in selected_rows]
    return {
        "source_roster_id": roster["roster_id"],
        "source_roster_receipt_sha256": roster["receipt_sha256"],
        "source_analysis_projection_id": projection["projection_id"],
        "source_analysis_projection_receipt_sha256": projection["receipt_sha256"],
        "source_analysis_identity_count": len(projection["records"]),
        "partition_rule": "canonical_projection_global_index_modulo_shard_count",
        "shard_count": shard_count,
        "shard_index": shard_index,
        "selected_identity_count": len(identities),
        "selected_identity_roster_sha256": _canonical_sha256(identities),
    }


def build_tusz_canonical_physical_shard_v1(
    *,
    source_roster: object,
    source_projection: object,
    shard_count: int,
    shard_index: int,
    outcomes: Sequence[Mapping[str, object]],
) -> dict[str, Any]:
    roster = validate_tusz_complete_detector_roster_v2(source_roster)
    projection = validate_tusz_analysis_identity_projection_v2(
        source_projection,
        source_roster=roster,
    )
    return _build_tusz_canonical_physical_shard_from_validated_v1(
        roster=roster,
        projection=projection,
        shard_count=shard_count,
        shard_index=shard_index,
        outcomes=outcomes,
    )


def _build_tusz_canonical_physical_shard_from_validated_v1(
    *,
    roster: Mapping[str, Any],
    projection: Mapping[str, Any],
    shard_count: int,
    shard_index: int,
    outcomes: Sequence[Mapping[str, object]],
) -> dict[str, Any]:
    selected = _select_tusz_canonical_physical_shard_rows_from_validated_v1(
        projection,
        shard_count=shard_count,
        shard_index=shard_index,
    )
    if len(outcomes) != len(selected):
        raise ValueError("canonical physical shard does not cover every selected row")
    validated = [
        validate_tusz_canonical_physical_outcome_v1(
            outcome,
            projection_row=row,
            source_projection_receipt_sha256=projection["receipt_sha256"],
        )
        for row, outcome in zip(selected, outcomes)
    ]
    success_count = sum(row["terminal_status"] == "success" for row in validated)
    failure_count = len(validated) - success_count
    body: dict[str, Any] = {
        "schema_version": TUSZ_CANONICAL_PHYSICAL_SHARD_V1_SCHEMA_VERSION,
        "method_id": TUSZ_CANONICAL_PHYSICAL_SHARD_V1_METHOD_ID,
        "shard_id": "TUSZ-CANONICAL-PHYSICAL-SHARD-V1-PENDING",
        "source_binding": _shard_binding(
            roster=roster,
            projection=projection,
            shard_count=shard_count,
            shard_index=shard_index,
            selected_rows=selected,
        ),
        "outcomes": validated,
        "outcome_inventory": {
            "terminal_outcome_count": len(validated),
            "success_count": success_count,
            "failure_count": failure_count,
            "every_selected_identity_has_one_terminal_outcome": True,
            "canonical_hash_success_for_all": failure_count == 0,
            "outcome_receipt_roster_sha256": _canonical_sha256(
                [row["receipt_sha256"] for row in validated]
            ),
        },
        "scope_receipt": _scope_receipt(),
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    body["shard_id"] = "TUSZPHYSSHARDV1-" + _canonical_sha256(body)[:24]
    body["receipt_sha256"] = _canonical_sha256(body)
    return validate_tusz_canonical_physical_shard_v1(
        body,
    )


def build_tusz_canonical_physical_shards_v1(
    *,
    source_roster: object,
    source_projection: object,
    outcomes_by_shard: Sequence[Sequence[Mapping[str, object]]],
) -> list[dict[str, Any]]:
    """Build a complete partition while validating large source inputs once."""

    roster = validate_tusz_complete_detector_roster_v2(source_roster)
    projection = validate_tusz_analysis_identity_projection_v2(
        source_projection,
        source_roster=roster,
    )
    if not outcomes_by_shard:
        raise ValueError("canonical physical shard batch must not be empty")
    shard_count = len(outcomes_by_shard)
    return [
        _build_tusz_canonical_physical_shard_from_validated_v1(
            roster=roster,
            projection=projection,
            shard_count=shard_count,
            shard_index=shard_index,
            outcomes=outcomes,
        )
        for shard_index, outcomes in enumerate(outcomes_by_shard)
    ]


def validate_tusz_canonical_physical_shard_v1(
    payload: object,
    *,
    source_roster: object | None = None,
    source_projection: object | None = None,
) -> dict[str, Any]:
    required = {
        "schema_version",
        "method_id",
        "shard_id",
        "source_binding",
        "outcomes",
        "outcome_inventory",
        "scope_receipt",
        "receipt_sha256",
    }
    if type(payload) is not dict or set(payload) != required:
        raise ValueError("canonical physical shard fields drifted")
    data = deepcopy(payload)
    if (
        data["schema_version"] != TUSZ_CANONICAL_PHYSICAL_SHARD_V1_SCHEMA_VERSION
        or data["method_id"] != TUSZ_CANONICAL_PHYSICAL_SHARD_V1_METHOD_ID
    ):
        raise ValueError("canonical physical shard schema/method drifted")
    binding = data["source_binding"]
    required_binding = {
        "source_roster_id",
        "source_roster_receipt_sha256",
        "source_analysis_projection_id",
        "source_analysis_projection_receipt_sha256",
        "source_analysis_identity_count",
        "partition_rule",
        "shard_count",
        "shard_index",
        "selected_identity_count",
        "selected_identity_roster_sha256",
    }
    if type(binding) is not dict or set(binding) != required_binding:
        raise ValueError("canonical physical shard source binding drifted")
    count = _positive_integer(binding["shard_count"], "shard count")
    index = _nonnegative_integer(binding["shard_index"], "shard index")
    if index >= count or binding["partition_rule"] != (
        "canonical_projection_global_index_modulo_shard_count"
    ):
        raise ValueError("canonical physical shard partition binding drifted")
    for field in (
        "source_roster_receipt_sha256",
        "source_analysis_projection_receipt_sha256",
        "selected_identity_roster_sha256",
    ):
        if not _is_sha256(binding[field]):
            raise ValueError("canonical physical shard binding hash is invalid")
    _positive_integer(
        binding["source_analysis_identity_count"], "source identity count"
    )
    _nonnegative_integer(binding["selected_identity_count"], "selected count")

    selected: list[dict[str, Any]] | None = None
    if source_roster is not None or source_projection is not None:
        if source_roster is None or source_projection is None:
            raise ValueError("source roster and projection must be supplied together")
        roster = validate_tusz_complete_detector_roster_v2(source_roster)
        projection = validate_tusz_analysis_identity_projection_v2(
            source_projection,
            source_roster=roster,
        )
        selected = _select_tusz_canonical_physical_shard_rows_from_validated_v1(
            projection,
            shard_count=count,
            shard_index=index,
        )
        expected_binding = _shard_binding(
            roster=roster,
            projection=projection,
            shard_count=count,
            shard_index=index,
            selected_rows=selected,
        )
        if binding != expected_binding:
            raise ValueError(
                "canonical physical shard source binding is not replayable"
            )

    outcomes = data["outcomes"]
    if (
        not isinstance(outcomes, list)
        or len(outcomes) != binding["selected_identity_count"]
    ):
        raise ValueError("canonical physical shard outcome count drifted")
    validated: list[dict[str, Any]] = []
    for outcome_index, outcome in enumerate(outcomes):
        row = selected[outcome_index] if selected is not None else None
        validated.append(
            validate_tusz_canonical_physical_outcome_v1(
                outcome,
                projection_row=row,
                source_projection_receipt_sha256=binding[
                    "source_analysis_projection_receipt_sha256"
                ],
            )
        )
    identities = [row["analysis_identity_id"] for row in validated]
    if (
        len(identities) != len(set(identities))
        or _canonical_sha256(identities) != binding["selected_identity_roster_sha256"]
    ):
        raise ValueError("canonical physical shard identity roster drifted")
    success_count = sum(row["terminal_status"] == "success" for row in validated)
    failure_count = len(validated) - success_count
    expected_inventory = {
        "terminal_outcome_count": len(validated),
        "success_count": success_count,
        "failure_count": failure_count,
        "every_selected_identity_has_one_terminal_outcome": True,
        "canonical_hash_success_for_all": failure_count == 0,
        "outcome_receipt_roster_sha256": _canonical_sha256(
            [row["receipt_sha256"] for row in validated]
        ),
    }
    if data["outcome_inventory"] != expected_inventory:
        raise ValueError("canonical physical shard outcome inventory drifted")
    if data["scope_receipt"] != _scope_receipt():
        raise ValueError("canonical physical shard violates EEG-only scope")
    digest = deepcopy(data)
    digest["shard_id"] = "TUSZ-CANONICAL-PHYSICAL-SHARD-V1-PENDING"
    digest["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["shard_id"] != "TUSZPHYSSHARDV1-" + _canonical_sha256(digest)[:24]:
        raise ValueError("canonical physical shard ID drifted")
    digest = deepcopy(data)
    digest["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if not _is_sha256(data["receipt_sha256"]) or data[
        "receipt_sha256"
    ] != _canonical_sha256(digest):
        raise ValueError("canonical physical shard receipt hash drifted")
    return data


def _identical_payload_classes(
    successful_outcomes: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Audit one canonical tensor payload against every declared clock.

    A sampling clock remains part of strict physical equivalence.  However,
    byte-identical channel/sample payloads carrying two different clocks are
    too suspicious to admit on both sides of a patient/split boundary.  They
    are therefore represented as a separate conflict class and quarantined,
    rather than being mislabeled as physically equivalent.
    """

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for outcome in successful_outcomes:
        tensor_hash = str(
            outcome["physical_signal"]["canonical_source_tensor_sha256"]
        )
        grouped.setdefault(tensor_hash, []).append(outcome)

    classes: list[dict[str, Any]] = []
    for tensor_hash, members in grouped.items():
        ordered = sorted(members, key=lambda row: str(row["analysis_identity_id"]))
        by_clock: dict[tuple[int, int, int], list[Mapping[str, Any]]] = {}
        for outcome in ordered:
            physical = outcome["physical_signal"]
            rate = physical["sampling_rate_fraction"]
            clock = (int(rate[0]), int(rate[1]), int(physical["sample_count"]))
            by_clock.setdefault(clock, []).append(outcome)

        clocks: list[dict[str, Any]] = []
        for clock, clock_members in sorted(
            by_clock.items(),
            key=lambda item: (Fraction(item[0][0], item[0][1]), item[0][2]),
        ):
            rate_numerator, rate_denominator, sample_count = clock
            clock_identities = [
                str(row["analysis_identity_id"]) for row in clock_members
            ]
            durations = {
                tuple(row["physical_signal"]["duration_seconds_fraction"])
                for row in clock_members
            }
            physical_hashes = {
                str(
                    row["physical_signal"][
                        "canonical_physical_equivalence_sha256"
                    ]
                )
                for row in clock_members
            }
            if len(durations) != 1 or len(physical_hashes) != 1:
                raise ValueError(
                    "one identical-payload clock has inconsistent physical receipts"
                )
            duration = next(iter(durations))
            clocks.append(
                {
                    "sampling_rate_fraction": [rate_numerator, rate_denominator],
                    "sample_count": sample_count,
                    "duration_seconds_fraction": list(duration),
                    "canonical_physical_equivalence_sha256": next(
                        iter(physical_hashes)
                    ),
                    "member_count": len(clock_members),
                    "member_analysis_identity_ids": clock_identities,
                    "member_identity_roster_sha256": _canonical_sha256(
                        clock_identities
                    ),
                }
            )

        identities = [str(row["analysis_identity_id"]) for row in ordered]
        discordant = len(clocks) > 1
        classes.append(
            {
                "identical_payload_class_id": f"TUSZPAYLOAD-{tensor_hash}",
                "canonical_source_tensor_sha256": tensor_hash,
                "member_count": len(ordered),
                "member_analysis_identity_ids": identities,
                "member_patient_aliases": sorted(
                    {str(row["local_patient_id"]) for row in ordered}
                ),
                "member_official_splits": sorted(
                    {str(row["official_split"]) for row in ordered}
                ),
                "sampling_clock_count": len(clocks),
                "sampling_clocks": clocks,
                "discordant_sampling_clock": discordant,
                "analysis_policy": (
                    "quarantine_all_identical_payload_members"
                    if discordant
                    else "defer_to_strict_physical_equivalence_policy"
                ),
                "quarantined_analysis_identity_ids": identities if discordant else [],
                "quarantine_reason": (
                    "identical_canonical_payload_has_discordant_sampling_clock"
                    if discordant
                    else None
                ),
                "member_identity_roster_sha256": _canonical_sha256(identities),
            }
        )
    classes.sort(key=lambda row: row["identical_payload_class_id"])
    return classes


def _physical_equivalence_classes(
    successful_outcomes: Sequence[Mapping[str, Any]],
    *,
    discordant_clock_tensor_hashes: set[str] | None = None,
) -> list[dict[str, Any]]:
    clock_conflicts = discordant_clock_tensor_hashes or set()
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for outcome in successful_outcomes:
        physical_hash = outcome["physical_signal"][
            "canonical_physical_equivalence_sha256"
        ]
        grouped.setdefault(str(physical_hash), []).append(outcome)
    classes: list[dict[str, Any]] = []
    for physical_hash, members in grouped.items():
        ordered = sorted(members, key=lambda row: str(row["analysis_identity_id"]))
        patients = sorted({str(row["local_patient_id"]) for row in ordered})
        splits = sorted({str(row["official_split"]) for row in ordered})
        identities = [str(row["analysis_identity_id"]) for row in ordered]
        paths = [str(row["local_edf_path"]) for row in ordered]
        containers = sorted(
            {str(row["source_edf_container_sha256"]) for row in ordered}
        )
        tensors = {
            str(row["physical_signal"]["canonical_source_tensor_sha256"])
            for row in ordered
        }
        if len(tensors) != 1:
            raise ValueError(
                "one physical equivalence class has multiple tensor hashes"
            )
        if len(ordered) == 1:
            boundary_type = "singleton"
            boundary_eligible = True
        elif len(patients) == 1 and len(splits) == 1:
            boundary_type = "same_split_same_patient_physical_alias"
            boundary_eligible = True
        elif len(patients) > 1 and len(splits) > 1:
            boundary_type = "cross_official_split_and_patient_physical_duplicate"
            boundary_eligible = False
        elif len(splits) > 1:
            boundary_type = "cross_official_split_physical_duplicate"
            boundary_eligible = False
        else:
            boundary_type = "cross_patient_same_split_physical_duplicate"
            boundary_eligible = False
        tensor_hash = next(iter(tensors))
        clock_conflict = tensor_hash in clock_conflicts
        eligible = boundary_eligible and not clock_conflict
        canonical = identities[0] if eligible else None
        if clock_conflict and not boundary_eligible:
            quarantine_reason = (
                "canonical_physical_signal_crosses_patient_or_split_and_"
                "identical_payload_has_discordant_sampling_clock"
            )
        elif clock_conflict:
            quarantine_reason = (
                "identical_canonical_payload_has_discordant_sampling_clock"
            )
        elif not boundary_eligible:
            quarantine_reason = "canonical_physical_signal_crosses_patient_or_split"
        else:
            quarantine_reason = None
        classes.append(
            {
                "physical_equivalence_class_id": f"TUSZPHYS-{physical_hash}",
                "canonical_physical_equivalence_sha256": physical_hash,
                "canonical_source_tensor_sha256": tensor_hash,
                "member_count": len(ordered),
                "member_analysis_identity_ids": identities,
                "member_local_edf_paths": paths,
                "member_source_container_sha256s": containers,
                "member_patient_aliases": patients,
                "member_official_splits": splits,
                "boundary_type": boundary_type,
                "identical_payload_clock_conflict": clock_conflict,
                "identical_payload_clock_policy": (
                    "discordant_clock_quarantine"
                    if clock_conflict
                    else "single_clock"
                ),
                "analysis_eligible": eligible,
                "analysis_canonical_identity_id": canonical,
                "excluded_same_patient_alias_identity_ids": (
                    identities[1:]
                    if (
                        boundary_type
                        == "same_split_same_patient_physical_alias"
                        and eligible
                    )
                    else []
                ),
                "quarantine_reason": quarantine_reason,
                "member_identity_roster_sha256": _canonical_sha256(identities),
            }
        )
    classes.sort(key=lambda row: row["physical_equivalence_class_id"])
    return classes


def _physical_equivalence_inventory(
    classes: Sequence[Mapping[str, Any]],
    identical_payload_classes: Sequence[Mapping[str, Any]],
    *,
    source_identity_count: int,
    failure_count: int,
) -> dict[str, Any]:
    duplicates = [row for row in classes if row["member_count"] > 1]
    same_patient = [
        row
        for row in classes
        if row["boundary_type"] == "same_split_same_patient_physical_alias"
        and row["analysis_eligible"]
    ]
    quarantined = [row for row in classes if not row["analysis_eligible"]]
    cross_boundary = [
        row
        for row in classes
        if row["boundary_type"]
        in {
            "cross_official_split_and_patient_physical_duplicate",
            "cross_official_split_physical_duplicate",
            "cross_patient_same_split_physical_duplicate",
        }
    ]
    clock_conflicts = [
        row
        for row in identical_payload_classes
        if row["discordant_sampling_clock"]
    ]
    clock_conflict_tensors = {
        str(row["canonical_source_tensor_sha256"]) for row in clock_conflicts
    }
    successful_count = sum(int(row["member_count"]) for row in classes)
    excluded_count = sum(
        len(row["excluded_same_patient_alias_identity_ids"]) for row in same_patient
    )
    quarantined_count = sum(int(row["member_count"]) for row in quarantined)
    cross_boundary_count = sum(int(row["member_count"]) for row in cross_boundary)
    clock_conflict_count = sum(
        int(row["member_count"]) for row in clock_conflicts
    )
    eligible_count = sum(bool(row["analysis_eligible"]) for row in classes)
    return {
        "source_analysis_identity_count": source_identity_count,
        "successful_canonical_identity_count": successful_count,
        "failed_canonical_identity_count": failure_count,
        "physical_equivalence_class_count": len(classes),
        "singleton_class_count": sum(row["member_count"] == 1 for row in classes),
        "physical_duplicate_class_count": len(duplicates),
        "physical_duplicate_member_identity_count": sum(
            int(row["member_count"]) for row in duplicates
        ),
        "same_patient_alias_class_count": len(same_patient),
        "same_patient_alias_excluded_identity_count": excluded_count,
        "analysis_quarantine_class_count": len(quarantined),
        "analysis_quarantined_identity_count": quarantined_count,
        "cross_boundary_quarantine_class_count": len(cross_boundary),
        "cross_boundary_quarantined_identity_count": cross_boundary_count,
        "identical_payload_class_count": len(identical_payload_classes),
        "identical_payload_duplicate_class_count": sum(
            row["member_count"] > 1 for row in identical_payload_classes
        ),
        "identical_payload_discordant_clock_class_count": len(clock_conflicts),
        "identical_payload_discordant_clock_identity_count": clock_conflict_count,
        "identical_payload_discordant_clock_quarantine_verified": all(
            not physical["analysis_eligible"]
            for physical in classes
            if physical["canonical_source_tensor_sha256"]
            in clock_conflict_tensors
        ),
        "analysis_eligible_canonical_identity_count": eligible_count,
        "terminal_outcome_accounting_verified": (
            successful_count + failure_count == source_identity_count
        ),
        "analysis_path_accounting_verified": (
            eligible_count + excluded_count + quarantined_count == successful_count
        ),
        "class_roster_sha256": _canonical_sha256(classes),
        "identical_payload_class_roster_sha256": _canonical_sha256(
            identical_payload_classes
        ),
    }


def build_tusz_canonical_physical_duplicate_audit_v1(
    *,
    source_roster: object,
    source_projection: object,
    shards: Sequence[Mapping[str, object]],
) -> dict[str, Any]:
    roster = validate_tusz_complete_detector_roster_v2(source_roster)
    projection = validate_tusz_analysis_identity_projection_v2(
        source_projection,
        source_roster=roster,
    )
    if not shards:
        raise ValueError("canonical physical audit requires at least one shard")
    validated = [validate_tusz_canonical_physical_shard_v1(shard) for shard in shards]
    shard_counts = {row["source_binding"]["shard_count"] for row in validated}
    if len(shard_counts) != 1:
        raise ValueError("canonical physical audit shards use different partitions")
    shard_count = next(iter(shard_counts))
    by_index = {row["source_binding"]["shard_index"]: row for row in validated}
    if len(by_index) != len(validated) or set(by_index) != set(range(shard_count)):
        raise ValueError("canonical physical audit does not contain every shard once")
    ordered_shards = [by_index[index] for index in range(shard_count)]
    for shard_index, shard in enumerate(ordered_shards):
        selected = _select_tusz_canonical_physical_shard_rows_from_validated_v1(
            projection,
            shard_count=shard_count,
            shard_index=shard_index,
        )
        expected_binding = _shard_binding(
            roster=roster,
            projection=projection,
            shard_count=shard_count,
            shard_index=shard_index,
            selected_rows=selected,
        )
        if shard["source_binding"] != expected_binding:
            raise ValueError("canonical physical audit shard source binding drifted")
    outcomes = [outcome for shard in ordered_shards for outcome in shard["outcomes"]]
    by_identity = {row["analysis_identity_id"]: row for row in outcomes}
    source_identities = [row["analysis_identity_id"] for row in projection["records"]]
    if len(by_identity) != len(outcomes) or set(by_identity) != set(source_identities):
        raise ValueError("canonical physical audit terminal outcomes do not close")
    canonical_outcomes = [by_identity[identity] for identity in source_identities]
    successes = [
        row for row in canonical_outcomes if row["terminal_status"] == "success"
    ]
    failures = [
        row for row in canonical_outcomes if row["terminal_status"] == "failure"
    ]
    identical_payload_classes = _identical_payload_classes(successes)
    discordant_clock_tensor_hashes = {
        str(row["canonical_source_tensor_sha256"])
        for row in identical_payload_classes
        if row["discordant_sampling_clock"]
    }
    classes = _physical_equivalence_classes(
        successes,
        discordant_clock_tensor_hashes=discordant_clock_tensor_hashes,
    )
    inventory = _physical_equivalence_inventory(
        classes,
        identical_payload_classes,
        source_identity_count=len(source_identities),
        failure_count=len(failures),
    )
    complete = (
        not failures
        and inventory["terminal_outcome_accounting_verified"]
        and inventory["analysis_path_accounting_verified"]
        and inventory[
            "identical_payload_discordant_clock_quarantine_verified"
        ]
    )
    body: dict[str, Any] = {
        "schema_version": TUSZ_CANONICAL_PHYSICAL_AUDIT_V1_SCHEMA_VERSION,
        "method_id": TUSZ_CANONICAL_PHYSICAL_AUDIT_V1_METHOD_ID,
        "audit_id": "TUSZ-CANONICAL-PHYSICAL-AUDIT-V1-PENDING",
        "source_binding": {
            "source_roster_id": roster["roster_id"],
            "source_roster_receipt_sha256": roster["receipt_sha256"],
            "source_analysis_projection_id": projection["projection_id"],
            "source_analysis_projection_receipt_sha256": projection["receipt_sha256"],
            "source_analysis_identity_count": len(projection["records"]),
        },
        "shard_inventory": {
            "shard_count": shard_count,
            "shard_receipt_roster_sha256": _canonical_sha256(
                [row["receipt_sha256"] for row in ordered_shards]
            ),
            "all_partition_indices_present_once": True,
        },
        "outcomes": canonical_outcomes,
        "identical_payload_classes": identical_payload_classes,
        "physical_equivalence_classes": classes,
        "physical_equivalence_inventory": inventory,
        "scope_receipt": {
            **_scope_receipt(),
            "canonical_physical_signal_duplicate_audit_complete": complete,
            "analysis_projection_authorized": complete,
            "failed_rows_cannot_be_silently_excluded": True,
            "cross_patient_or_split_physical_duplicates_quarantined": True,
            **_exact_duplicate_scope_receipt(),
        },
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    body["audit_id"] = "TUSZPHYSAUDITV1-" + _canonical_sha256(body)[:24]
    body["receipt_sha256"] = _canonical_sha256(body)
    return validate_tusz_canonical_physical_duplicate_audit_v1(
        body,
        source_roster=roster,
        source_projection=projection,
    )


def validate_tusz_canonical_physical_duplicate_audit_v1(
    payload: object,
    *,
    source_roster: object | None = None,
    source_projection: object | None = None,
) -> dict[str, Any]:
    required = {
        "schema_version",
        "method_id",
        "audit_id",
        "source_binding",
        "shard_inventory",
        "outcomes",
        "identical_payload_classes",
        "physical_equivalence_classes",
        "physical_equivalence_inventory",
        "scope_receipt",
        "receipt_sha256",
    }
    if type(payload) is not dict or set(payload) != required:
        raise ValueError("canonical physical duplicate audit fields drifted")
    data = deepcopy(payload)
    if (
        data["schema_version"] != TUSZ_CANONICAL_PHYSICAL_AUDIT_V1_SCHEMA_VERSION
        or data["method_id"] != TUSZ_CANONICAL_PHYSICAL_AUDIT_V1_METHOD_ID
    ):
        raise ValueError("canonical physical duplicate audit schema/method drifted")
    binding = data["source_binding"]
    required_binding = {
        "source_roster_id",
        "source_roster_receipt_sha256",
        "source_analysis_projection_id",
        "source_analysis_projection_receipt_sha256",
        "source_analysis_identity_count",
    }
    if type(binding) is not dict or set(binding) != required_binding:
        raise ValueError("canonical physical audit source binding drifted")
    _positive_integer(binding["source_analysis_identity_count"], "source count")
    for field in (
        "source_roster_receipt_sha256",
        "source_analysis_projection_receipt_sha256",
    ):
        if not _is_sha256(binding[field]):
            raise ValueError("canonical physical audit binding hash is invalid")
    projection_rows: list[Mapping[str, Any]] | None = None
    validated_source_roster: Mapping[str, Any] | None = None
    validated_source_projection: Mapping[str, Any] | None = None
    if source_roster is not None or source_projection is not None:
        if source_roster is None or source_projection is None:
            raise ValueError("source roster and projection must be supplied together")
        roster = validate_tusz_complete_detector_roster_v2(source_roster)
        projection = validate_tusz_analysis_identity_projection_v2(
            source_projection,
            source_roster=roster,
        )
        expected_binding = {
            "source_roster_id": roster["roster_id"],
            "source_roster_receipt_sha256": roster["receipt_sha256"],
            "source_analysis_projection_id": projection["projection_id"],
            "source_analysis_projection_receipt_sha256": projection["receipt_sha256"],
            "source_analysis_identity_count": len(projection["records"]),
        }
        if binding != expected_binding:
            raise ValueError(
                "canonical physical audit source binding is not replayable"
            )
        projection_rows = projection["records"]
        validated_source_roster = roster
        validated_source_projection = projection

    shard_inventory = data["shard_inventory"]
    required_shard_inventory = {
        "shard_count",
        "shard_receipt_roster_sha256",
        "all_partition_indices_present_once",
    }
    if (
        type(shard_inventory) is not dict
        or set(shard_inventory) != required_shard_inventory
        or shard_inventory["all_partition_indices_present_once"] is not True
        or not _is_sha256(shard_inventory["shard_receipt_roster_sha256"])
    ):
        raise ValueError("canonical physical audit shard inventory drifted")
    _positive_integer(shard_inventory["shard_count"], "audit shard count")

    outcomes = data["outcomes"]
    if (
        not isinstance(outcomes, list)
        or len(outcomes) != binding["source_analysis_identity_count"]
    ):
        raise ValueError("canonical physical audit outcome count drifted")
    validated_outcomes: list[dict[str, Any]] = []
    for index, outcome in enumerate(outcomes):
        row = projection_rows[index] if projection_rows is not None else None
        validated_outcomes.append(
            validate_tusz_canonical_physical_outcome_v1(
                outcome,
                projection_row=row,
                source_projection_receipt_sha256=binding[
                    "source_analysis_projection_receipt_sha256"
                ],
            )
        )
    identities = [row["analysis_identity_id"] for row in validated_outcomes]
    if len(identities) != len(set(identities)):
        raise ValueError("canonical physical audit repeats an identity")
    if (
        validated_source_roster is not None
        and validated_source_projection is not None
    ):
        outcomes_by_identity = {
            row["analysis_identity_id"]: row for row in validated_outcomes
        }
        rebuilt_shard_receipts: list[str] = []
        for shard_index in range(shard_inventory["shard_count"]):
            selected = _select_tusz_canonical_physical_shard_rows_from_validated_v1(
                validated_source_projection,
                shard_count=shard_inventory["shard_count"],
                shard_index=shard_index,
            )
            rebuilt = _build_tusz_canonical_physical_shard_from_validated_v1(
                roster=validated_source_roster,
                projection=validated_source_projection,
                shard_count=shard_inventory["shard_count"],
                shard_index=shard_index,
                outcomes=[
                    outcomes_by_identity[row["analysis_identity_id"]]
                    for row in selected
                ],
            )
            rebuilt_shard_receipts.append(rebuilt["receipt_sha256"])
        if shard_inventory["shard_receipt_roster_sha256"] != _canonical_sha256(
            rebuilt_shard_receipts
        ):
            raise ValueError(
                "canonical physical audit shard receipt roster is not replayable"
            )
    successes = [
        row for row in validated_outcomes if row["terminal_status"] == "success"
    ]
    failures = [
        row for row in validated_outcomes if row["terminal_status"] == "failure"
    ]
    identical_payload_classes = _identical_payload_classes(successes)
    if data["identical_payload_classes"] != identical_payload_classes:
        raise ValueError("canonical identical-payload classes are not replayable")
    discordant_clock_tensor_hashes = {
        str(row["canonical_source_tensor_sha256"])
        for row in identical_payload_classes
        if row["discordant_sampling_clock"]
    }
    classes = _physical_equivalence_classes(
        successes,
        discordant_clock_tensor_hashes=discordant_clock_tensor_hashes,
    )
    if data["physical_equivalence_classes"] != classes:
        raise ValueError("canonical physical equivalence classes are not replayable")
    inventory = _physical_equivalence_inventory(
        classes,
        identical_payload_classes,
        source_identity_count=len(validated_outcomes),
        failure_count=len(failures),
    )
    if data["physical_equivalence_inventory"] != inventory:
        raise ValueError("canonical physical equivalence inventory drifted")
    complete = (
        not failures
        and inventory["terminal_outcome_accounting_verified"]
        and inventory["analysis_path_accounting_verified"]
        and inventory[
            "identical_payload_discordant_clock_quarantine_verified"
        ]
    )
    expected_scope = {
        **_scope_receipt(),
        "canonical_physical_signal_duplicate_audit_complete": complete,
        "analysis_projection_authorized": complete,
        "failed_rows_cannot_be_silently_excluded": True,
        "cross_patient_or_split_physical_duplicates_quarantined": True,
        **_exact_duplicate_scope_receipt(),
    }
    if data["scope_receipt"] != expected_scope:
        raise ValueError("canonical physical duplicate audit scope drifted")
    digest = deepcopy(data)
    digest["audit_id"] = "TUSZ-CANONICAL-PHYSICAL-AUDIT-V1-PENDING"
    digest["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["audit_id"] != "TUSZPHYSAUDITV1-" + _canonical_sha256(digest)[:24]:
        raise ValueError("canonical physical duplicate audit ID drifted")
    digest = deepcopy(data)
    digest["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if not _is_sha256(data["receipt_sha256"]) or data[
        "receipt_sha256"
    ] != _canonical_sha256(digest):
        raise ValueError("canonical physical duplicate audit receipt hash drifted")
    return data


def build_tusz_canonical_physical_analysis_projection_v1(
    *,
    audit: object,
    source_roster: object,
    source_projection: object,
) -> dict[str, Any]:
    roster = validate_tusz_complete_detector_roster_v2(source_roster)
    projection = validate_tusz_analysis_identity_projection_v2(
        source_projection,
        source_roster=roster,
    )
    validated_audit = validate_tusz_canonical_physical_duplicate_audit_v1(
        audit,
        source_roster=roster,
        source_projection=projection,
    )
    if not validated_audit["scope_receipt"]["analysis_projection_authorized"]:
        raise PermissionError(
            "canonical physical analysis projection requires a complete audit"
        )
    by_identity = {row["analysis_identity_id"]: row for row in projection["records"]}
    outcome_by_identity = {
        row["analysis_identity_id"]: row for row in validated_audit["outcomes"]
    }
    records: list[dict[str, Any]] = []
    for equivalence in validated_audit["physical_equivalence_classes"]:
        canonical_id = equivalence["analysis_canonical_identity_id"]
        if canonical_id is None:
            continue
        source = by_identity[canonical_id]
        outcome = outcome_by_identity[canonical_id]
        records.append(
            {
                **deepcopy(source),
                "canonical_physical_equivalence_id": equivalence[
                    "physical_equivalence_class_id"
                ],
                "canonical_physical_source_tensor_sha256": outcome["physical_signal"][
                    "canonical_source_tensor_sha256"
                ],
                "physical_equivalence_canonical_analysis_identity_id": canonical_id,
                "source_physical_identity_multiplicity": equivalence["member_count"],
            }
        )
    records.sort(
        key=lambda row: (
            row["official_split"],
            row["local_patient_id"],
            row["local_edf_path"],
        )
    )
    inventory = validated_audit["physical_equivalence_inventory"]
    body: dict[str, Any] = {
        "schema_version": TUSZ_CANONICAL_PHYSICAL_PROJECTION_V1_SCHEMA_VERSION,
        "method_id": TUSZ_CANONICAL_PHYSICAL_PROJECTION_V1_METHOD_ID,
        "projection_id": "TUSZ-CANONICAL-PHYSICAL-PROJECTION-V1-PENDING",
        "source_binding": {
            "source_roster_id": roster["roster_id"],
            "source_roster_receipt_sha256": roster["receipt_sha256"],
            "source_analysis_projection_id": projection["projection_id"],
            "source_analysis_projection_receipt_sha256": projection["receipt_sha256"],
            "source_canonical_physical_audit_id": validated_audit["audit_id"],
            "source_canonical_physical_audit_receipt_sha256": validated_audit[
                "receipt_sha256"
            ],
        },
        "identity_fields": [
            *TUSZ_ANALYSIS_IDENTITY_FIELDS_V2,
            *_PHYSICAL_PROJECTION_EXTRA_FIELDS,
        ],
        "records": records,
        "projection_inventory": {
            "source_analysis_identity_count": len(projection["records"]),
            "projected_analysis_identity_count": len(records),
            "same_patient_alias_excluded_identity_count": inventory[
                "same_patient_alias_excluded_identity_count"
            ],
            "analysis_quarantined_identity_count": inventory[
                "analysis_quarantined_identity_count"
            ],
            "cross_boundary_quarantined_identity_count": inventory[
                "cross_boundary_quarantined_identity_count"
            ],
            "identical_payload_discordant_clock_quarantined_identity_count": (
                inventory["identical_payload_discordant_clock_identity_count"]
            ),
            "path_accounting_verified": (
                len(records)
                + inventory["same_patient_alias_excluded_identity_count"]
                + inventory["analysis_quarantined_identity_count"]
                == len(projection["records"])
            ),
            "projected_identity_roster_sha256": _canonical_sha256(
                [row["analysis_identity_id"] for row in records]
            ),
        },
        "role_permissions": _physical_projection_role_permissions(),
        "reference_access_receipt": _physical_projection_reference_access_receipt(),
        "scope_receipt": {
            "canonical_physical_signal_duplicate_audit_complete": True,
            "one_unit_per_safe_physical_equivalence_class": True,
            "cross_patient_or_split_physical_duplicates_quarantined": True,
            "same_patient_same_split_physical_aliases_deduplicated": True,
            "reference_join_authorized": False,
            "model_performance_claim_authorized": False,
            **_exact_duplicate_scope_receipt(),
        },
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    body["projection_id"] = "TUSZPHYSPROJV1-" + _canonical_sha256(body)[:24]
    body["receipt_sha256"] = _canonical_sha256(body)
    # Validate the newly built object structurally here.  The optional
    # source-aware validator path below rebuilds through this function; using
    # it here would recurse indefinitely.
    return validate_tusz_canonical_physical_analysis_projection_v1(body)


def validate_tusz_canonical_physical_analysis_projection_v1(
    payload: object,
    *,
    audit: object | None = None,
    source_roster: object | None = None,
    source_projection: object | None = None,
) -> dict[str, Any]:
    required = {
        "schema_version",
        "method_id",
        "projection_id",
        "source_binding",
        "identity_fields",
        "records",
        "projection_inventory",
        "role_permissions",
        "reference_access_receipt",
        "scope_receipt",
        "receipt_sha256",
    }
    if type(payload) is not dict or set(payload) != required:
        raise ValueError("canonical physical analysis projection fields drifted")
    data = deepcopy(payload)
    if (
        data["schema_version"] != TUSZ_CANONICAL_PHYSICAL_PROJECTION_V1_SCHEMA_VERSION
        or data["method_id"] != TUSZ_CANONICAL_PHYSICAL_PROJECTION_V1_METHOD_ID
    ):
        raise ValueError("canonical physical analysis projection schema drifted")
    binding = data["source_binding"]
    required_binding = {
        "source_roster_id",
        "source_roster_receipt_sha256",
        "source_analysis_projection_id",
        "source_analysis_projection_receipt_sha256",
        "source_canonical_physical_audit_id",
        "source_canonical_physical_audit_receipt_sha256",
    }
    if type(binding) is not dict or set(binding) != required_binding:
        raise ValueError("canonical physical projection source binding drifted")
    _identifier(binding["source_roster_id"], "source roster ID")
    _identifier(
        binding["source_analysis_projection_id"],
        "source analysis projection ID",
    )
    if not (
        isinstance(binding["source_canonical_physical_audit_id"], str)
        and binding["source_canonical_physical_audit_id"].startswith(
            "TUSZPHYSAUDITV1-"
        )
        and len(binding["source_canonical_physical_audit_id"])
        == len("TUSZPHYSAUDITV1-") + 24
        and all(
            character in "0123456789abcdef"
            for character in binding["source_canonical_physical_audit_id"][
                len("TUSZPHYSAUDITV1-") :
            ]
        )
    ):
        raise ValueError("canonical physical projection audit ID is invalid")
    for field in (
        "source_roster_receipt_sha256",
        "source_analysis_projection_receipt_sha256",
        "source_canonical_physical_audit_receipt_sha256",
    ):
        if not _is_sha256(binding[field]):
            raise ValueError("canonical physical projection source hash is invalid")
    if data["identity_fields"] != [
        *TUSZ_ANALYSIS_IDENTITY_FIELDS_V2,
        *_PHYSICAL_PROJECTION_EXTRA_FIELDS,
    ]:
        raise ValueError("canonical physical projection identity fields drifted")
    records = data["records"]
    expected_fields = set(data["identity_fields"])
    if not isinstance(records, list):
        raise ValueError("canonical physical projection records must be a list")
    identities: list[str] = []
    for row in records:
        if type(row) is not dict or set(row) != expected_fields:
            raise ValueError("canonical physical projection row fields drifted")
        model_split = row["model_split"]
        official_split = row["official_split"]
        if (
            model_split not in _MODEL_TO_OFFICIAL_SPLIT
            or official_split != _MODEL_TO_OFFICIAL_SPLIT[model_split]
        ):
            raise ValueError("canonical physical projection split mapping drifted")
        patient_id = _identifier(
            row["local_patient_id"], "canonical physical patient alias"
        )
        local_edf_path = _identifier(
            row["local_edf_path"], "canonical physical EDF path"
        )
        parsed_path = PurePosixPath(local_edf_path)
        if (
            parsed_path.is_absolute()
            or ".." in parsed_path.parts
            or "\\" in local_edf_path
            or len(parsed_path.parts) < 3
            or parsed_path.parts[0] != official_split
            or parsed_path.parts[1] != patient_id
            or parsed_path.suffix.lower() != ".edf"
        ):
            raise ValueError("canonical physical projection path binding drifted")
        container_sha256 = row["source_edf_container_sha256"]
        if not _is_sha256(container_sha256):
            raise ValueError("canonical physical projection container hash is invalid")
        identity = row["analysis_identity_id"]
        if (
            identity != f"TUSZANALYSIS-{container_sha256}"
            or row["exact_container_equivalence_id"]
            != f"TUSZEXACT-{container_sha256}"
        ):
            raise ValueError(
                "canonical physical projection exact-container identity drifted"
            )
        identities.append(identity)
        if (
            not _is_sha256(row["canonical_physical_source_tensor_sha256"])
            or row["physical_equivalence_canonical_analysis_identity_id"] != identity
            or not _is_prefixed_sha256(
                row["canonical_physical_equivalence_id"], "TUSZPHYS-"
            )
        ):
            raise ValueError("canonical physical projection row binding drifted")
        _positive_integer(
            row["source_official_path_multiplicity"],
            "source official path multiplicity",
        )
        if type(row["analysis_unit_weight"]) is not int or row[
            "analysis_unit_weight"
        ] != 1:
            raise ValueError("canonical physical projection unit weight must equal one")
        _positive_integer(
            row["source_physical_identity_multiplicity"],
            "physical identity multiplicity",
        )
    if len(identities) != len(set(identities)):
        raise ValueError("canonical physical projection identities repeat")
    canonical_order = sorted(
        records,
        key=lambda row: (
            row["official_split"],
            row["local_patient_id"],
            row["local_edf_path"],
        ),
    )
    if records != canonical_order:
        raise ValueError("canonical physical projection order drifted")
    inventory = data["projection_inventory"]
    required_inventory = {
        "source_analysis_identity_count",
        "projected_analysis_identity_count",
        "same_patient_alias_excluded_identity_count",
        "analysis_quarantined_identity_count",
        "cross_boundary_quarantined_identity_count",
        "identical_payload_discordant_clock_quarantined_identity_count",
        "path_accounting_verified",
        "projected_identity_roster_sha256",
    }
    if type(inventory) is not dict or set(inventory) != required_inventory:
        raise ValueError("canonical physical projection inventory drifted")
    for field in (
        "source_analysis_identity_count",
        "projected_analysis_identity_count",
        "same_patient_alias_excluded_identity_count",
        "analysis_quarantined_identity_count",
        "cross_boundary_quarantined_identity_count",
        "identical_payload_discordant_clock_quarantined_identity_count",
    ):
        _nonnegative_integer(inventory[field], field)
    if (
        inventory["projected_analysis_identity_count"] != len(records)
        or inventory["projected_identity_roster_sha256"]
        != _canonical_sha256(identities)
        or inventory["path_accounting_verified"] is not True
        or len(records)
        + inventory["same_patient_alias_excluded_identity_count"]
        + inventory["analysis_quarantined_identity_count"]
        != inventory["source_analysis_identity_count"]
    ):
        raise ValueError("canonical physical projection accounting drifted")
    if (
        inventory["cross_boundary_quarantined_identity_count"]
        > inventory["analysis_quarantined_identity_count"]
        or inventory[
            "identical_payload_discordant_clock_quarantined_identity_count"
        ]
        > inventory["analysis_quarantined_identity_count"]
    ):
        raise ValueError("canonical physical projection quarantine counts drifted")
    if data["role_permissions"] != _physical_projection_role_permissions():
        raise ValueError("canonical physical projection role permissions drifted")
    if (
        data["reference_access_receipt"]
        != _physical_projection_reference_access_receipt()
    ):
        raise ValueError("canonical physical projection reference access drifted")
    expected_scope = {
        "canonical_physical_signal_duplicate_audit_complete": True,
        "one_unit_per_safe_physical_equivalence_class": True,
        "cross_patient_or_split_physical_duplicates_quarantined": True,
        "same_patient_same_split_physical_aliases_deduplicated": True,
        "reference_join_authorized": False,
        "model_performance_claim_authorized": False,
        **_exact_duplicate_scope_receipt(),
    }
    if data["scope_receipt"] != expected_scope:
        raise ValueError("canonical physical projection scope drifted")

    if audit is not None or source_roster is not None or source_projection is not None:
        if audit is None or source_roster is None or source_projection is None:
            raise ValueError(
                "audit, source roster and projection must be supplied together"
            )
        expected = build_tusz_canonical_physical_analysis_projection_v1(
            audit=audit,
            source_roster=source_roster,
            source_projection=source_projection,
        )
        if data != expected:
            raise ValueError("canonical physical analysis projection is not replayable")
        return data

    digest = deepcopy(data)
    digest["projection_id"] = "TUSZ-CANONICAL-PHYSICAL-PROJECTION-V1-PENDING"
    digest["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["projection_id"] != "TUSZPHYSPROJV1-" + _canonical_sha256(digest)[:24]:
        raise ValueError("canonical physical projection ID drifted")
    digest = deepcopy(data)
    digest["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if not _is_sha256(data["receipt_sha256"]) or data[
        "receipt_sha256"
    ] != _canonical_sha256(digest):
        raise ValueError("canonical physical projection receipt hash drifted")
    return data


__all__ = [
    "TUSZ_CANONICAL_PHYSICAL_AUDIT_V1_SCHEMA_VERSION",
    "TUSZ_CANONICAL_PHYSICAL_OUTCOME_V1_SCHEMA_VERSION",
    "TUSZ_CANONICAL_PHYSICAL_PROJECTION_V1_SCHEMA_VERSION",
    "TUSZ_CANONICAL_PHYSICAL_SHARD_V1_SCHEMA_VERSION",
    "build_tusz_canonical_physical_analysis_projection_v1",
    "build_tusz_canonical_physical_duplicate_audit_v1",
    "build_tusz_canonical_physical_failure_outcome_v1",
    "build_tusz_canonical_physical_shard_v1",
    "build_tusz_canonical_physical_shards_v1",
    "materialize_tusz_canonical_physical_outcome_v1",
    "select_tusz_canonical_physical_shard_rows_v1",
    "validate_tusz_canonical_physical_analysis_projection_v1",
    "validate_tusz_canonical_physical_duplicate_audit_v1",
    "validate_tusz_canonical_physical_outcome_v1",
    "validate_tusz_canonical_physical_shard_v1",
]
