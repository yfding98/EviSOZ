"""Post-freeze source-development reference join for detector calibration.

This module owns the *second* phase of continuous-detector calibration input
materialization.  A reference-free posterior-batch validator must finish for
the complete ``source_dev`` inventory before any path to a TUSZ ``.csv_bi``
file is constructed or opened.  The public join consequently accepts a
sealed validation object, never a posterior directory or a caller-supplied
``validation_passed`` boolean.

Only global ``TERM,seiz`` intervals are projected.  Channel annotations,
clinical text, Excel, private data and source-evaluation references have no
slot in the output.  This is an evaluation/calibration boundary; it never
runs a detector or changes a frozen posterior.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Final, Mapping, Sequence

from .continuous_detection_calibration import (
    validate_continuous_calibration_rows,
)
from .deepsoz_posterior_batch_validation import (
    ValidatedDeepSOZPosteriorRecording,
    revalidate_deepsoz_posterior_batch_without_references,
)


SOURCE_DEV_REFERENCE_JOIN_SCHEMA_VERSION: Final[str] = (
    "continuous_detection_source_dev_reference_join_v1"
)
SOURCE_DEV_REFERENCE_PARSER_ID: Final[str] = (
    "tusz_csv_bi_exact_TERM_seiz_projection_v1"
)
SOURCE_DEV_REFERENCE_MAPPING_ID: Final[str] = (
    "same_recording_relative_path_edf_suffix_to_csv_bi_v1"
)
SOURCE_DEV_CALIBRATION_ROWS_FILENAME: Final[str] = "calibration_rows.jsonl"
SOURCE_DEV_REFERENCE_JOIN_RECEIPT_FILENAME: Final[str] = "join_receipt.json"

_REFERENCE_FIELDS: Final[frozenset[str]] = frozenset(
    {"channel", "start_time", "stop_time", "label"}
)
_FORBIDDEN_PATH_PARTS: Final[frozenset[str]] = frozenset(
    {"train", "eval", "source_eval", "private", "private_inference"}
)

_JOIN_RECEIPT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "join_id",
        "parser_id",
        "mapping_id",
        "source_split",
        "provider_id",
        "stage1_validation_id",
        "stage1_validation_receipt_sha256",
        "stage1_manifest_sha256",
        "stage1_batch_receipt_file_sha256",
        "stage1_posterior_index_file_sha256",
        "stage1_posterior_artifact_inventory_sha256",
        "stage1_record_binding_roster_sha256",
        "input_recording_roster_sha256",
        "input_patient_roster_sha256",
        "input_posterior_artifact_roster_sha256",
        "reference_file_inventory_sha256",
        "reference_event_inventory_sha256",
        "calibration_rows_sha256",
        "output_calibration_jsonl_sha256",
        "recording_count",
        "patient_count",
        "reference_file_count",
        "selected_term_seiz_event_count",
        "ignored_non_term_seiz_row_count",
        "seizure_free_recording_count",
        "posterior_validation_completed_before_first_reference_open",
        "reference_files_opened",
        "full_stage1_inventory_joined",
        "calibration_rows_revalidated",
        "scope_receipt",
        "receipt_sha256",
    }
)

_JOIN_SCOPE_RECEIPT: Final[dict[str, object]] = {
    "source_dev_only": True,
    "global_term_seiz_intervals_only": True,
    "channel_annotations_used": False,
    "edf_annotations_used": False,
    "excel_or_clinical_labels_used": False,
    "private_data_used": False,
    "source_eval_used": False,
    "references_used_for_calibration_only": True,
    "posterior_artifacts_mutated": False,
    "detector_rerun": False,
    "research_only": True,
    "clinical_use_authorized": False,
    "production_qualified": False,
    "sota_claim_authorized": False,
}


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


def _finite(value: object, context: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{context} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{context} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TypeError(f"{context} must be a non-empty trimmed string")
    if len(value) > 1024 or any(ord(character) < 32 for character in value):
        raise ValueError(f"{context} is invalid")
    return value


@dataclass(frozen=True)
class ParsedSourceDevReference:
    """Immutable projection of one reference file.

    ``events_json`` is canonical JSON rather than a mutable list so callers
    cannot alter the validated intervals between parsing and receipt sealing.
    """

    reference_file_sha256: str
    events_json: str
    selected_term_seiz_row_count: int
    ignored_non_term_seiz_row_count: int

    def events(self) -> list[dict[str, float]]:
        value = json.loads(self.events_json)
        if not isinstance(value, list):  # pragma: no cover - construction invariant
            raise RuntimeError("sealed reference events are not an array")
        return value


def parse_tusz_term_seiz_reference_bytes(
    payload: bytes,
    *,
    duration_seconds: float,
) -> ParsedSourceDevReference:
    """Parse only exact global ``TERM,seiz`` rows from one TUSZ CSV.

    Structural CSV errors fail closed even when they occur on an ignored row.
    Physiological/channel annotations are otherwise ignored and never copied
    into the returned object.
    """

    if not isinstance(payload, bytes) or not payload:
        raise TypeError("TUSZ reference payload must be non-empty bytes")
    duration = _finite(duration_seconds, "recording duration")
    if duration <= 0:
        raise ValueError("recording duration must be positive")
    try:
        text = payload.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("TUSZ reference is not valid UTF-8") from error
    if "\x00" in text:
        raise ValueError("TUSZ reference contains a NUL byte")
    data_lines = [
        line
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not data_lines:
        raise ValueError("TUSZ reference has no CSV header")
    reader = csv.DictReader(io.StringIO("\n".join(data_lines), newline=""))
    fieldnames = reader.fieldnames
    if fieldnames is None:
        raise ValueError("TUSZ reference has no CSV header")
    normalized_fields = [str(value).strip() for value in fieldnames]
    if (
        any(not value for value in normalized_fields)
        or len(normalized_fields) != len(set(normalized_fields))
        or not _REFERENCE_FIELDS.issubset(normalized_fields)
    ):
        raise ValueError("TUSZ reference header is missing or duplicates fields")
    # DictReader retains original header spelling.  Build an exact normalized
    # lookup once instead of accepting near-match aliases.
    field_lookup = dict(zip(normalized_fields, fieldnames))

    intervals: list[dict[str, float]] = []
    ignored = 0
    for line_number, row in enumerate(reader, start=2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(
                f"TUSZ reference row {line_number} has a malformed column count"
            )
        channel = str(row[field_lookup["channel"]]).strip()
        label = str(row[field_lookup["label"]]).strip()
        if channel != "TERM" or label != "seiz":
            ignored += 1
            continue
        start = _finite(
            str(row[field_lookup["start_time"]]).strip(),
            f"TUSZ TERM,seiz row {line_number} start_time",
        )
        stop = _finite(
            str(row[field_lookup["stop_time"]]).strip(),
            f"TUSZ TERM,seiz row {line_number} stop_time",
        )
        if start < 0 or stop <= start or stop > duration + 1e-9:
            raise ValueError(
                f"TUSZ TERM,seiz row {line_number} lies outside recording duration"
            )
        intervals.append({"start_seconds": start, "stop_seconds": stop})

    canonical = sorted(
        intervals, key=lambda event: (event["start_seconds"], event["stop_seconds"])
    )
    if intervals != canonical:
        raise ValueError("TUSZ TERM,seiz intervals are not in chronological order")
    previous_stop = -math.inf
    for event in canonical:
        if event["start_seconds"] < previous_stop - 1e-9:
            raise ValueError("TUSZ TERM,seiz intervals overlap or contradict")
        previous_stop = event["stop_seconds"]

    return ParsedSourceDevReference(
        reference_file_sha256=hashlib.sha256(payload).hexdigest(),
        events_json=_canonical_json_bytes(canonical).decode("utf-8"),
        selected_term_seiz_row_count=len(canonical),
        ignored_non_term_seiz_row_count=ignored,
    )


def source_dev_reference_relative_path(recording_id: str) -> PurePosixPath:
    """Map one validated ``dev/.../*.edf`` identity to ``*.csv_bi``.

    This function is deliberately called only by the phase-2 join after a
    complete phase-1 validation receipt has been accepted.
    """

    identifier = _identifier(recording_id, "recording_id")
    if "\\" in identifier:
        raise ValueError("recording_id must use POSIX separators")
    path = PurePosixPath(identifier)
    lowered = tuple(part.lower() for part in path.parts)
    if (
        path.is_absolute()
        or not path.parts
        or path.parts[0] != "dev"
        or ".." in path.parts
        or "." in path.parts
        or path.suffix.lower() != ".edf"
        or any(part in _FORBIDDEN_PATH_PARTS for part in lowered[1:])
    ):
        raise ValueError("recording_id is not a safe source-development EDF path")
    return path.with_suffix(".csv_bi")


def read_source_dev_reference_bytes(
    reference_root: Path,
    relative_path: PurePosixPath,
) -> bytes:
    """Read one canonical regular reference file without following symlinks."""

    root_input = Path(reference_root)
    if root_input.is_symlink():
        raise ValueError("TUSZ reference root must not be a symlink")
    root = root_input.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("TUSZ reference root must be a directory")
    candidate = root.joinpath(*relative_path.parts)
    cursor = root
    for part in relative_path.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError("TUSZ reference path must not contain symlinks")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("TUSZ reference escaped its canonical root") from error
    if not resolved.is_file():
        raise ValueError("TUSZ reference path is not a regular file")
    return resolved.read_bytes()


def _require_sha256(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} must be lowercase SHA-256")
    return value


def _integer(value: object, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TypeError(f"{context} must be an integer >= {minimum}")
    return value


def _canonical_json_text(value: object) -> str:
    return _canonical_json_bytes(value).decode("utf-8")


def _calibration_jsonl(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(_canonical_json_text(dict(row)) + "\n" for row in rows)


def _assert_source_dev_recording_identity(recording_id: str) -> None:
    """Validate the recording identity without deriving a reference path."""

    identifier = _identifier(recording_id, "recording_id")
    if "\\" in identifier:
        raise ValueError("recording_id must use POSIX separators")
    path = PurePosixPath(identifier)
    lowered = tuple(part.lower() for part in path.parts)
    if (
        path.is_absolute()
        or not path.parts
        or path.parts[0] != "dev"
        or ".." in path.parts
        or "." in path.parts
        or path.suffix.lower() != ".edf"
        or any(part in _FORBIDDEN_PATH_PARTS for part in lowered[1:])
    ):
        raise ValueError("recording_id is not a safe source-development EDF path")


def _validate_stage1_before_reference_access(
    validation: object,
) -> tuple[
    dict[str, Any],
    tuple[ValidatedDeepSOZPosteriorRecording, ...],
    str,
]:
    """Delegate closure replay to Stage-1, then check mapping eligibility."""

    sealed = revalidate_deepsoz_posterior_batch_without_references(validation)
    if not Path(sealed.batch_root).is_absolute():
        raise ValueError("sealed Stage-1 batch root must be absolute")
    binding_rows: list[list[str]] = []
    for record in sealed.recordings:
        _assert_source_dev_recording_identity(record.recording_id)
        binding_rows.append(
            [
                record.recording_id,
                _require_sha256(
                    record.record_binding_sha256,
                    "posterior record binding",
                ),
            ]
        )
    return (
        sealed.validation_receipt(),
        sealed.recordings,
        _canonical_sha256(sorted(binding_rows)),
    )


@dataclass(frozen=True)
class JoinedSourceDevCalibrationRows:
    """Immutable calibration rows plus their post-freeze join receipt."""

    calibration_rows_json: str
    join_receipt_json: str

    def calibration_rows(self) -> list[dict[str, Any]]:
        value = json.loads(self.calibration_rows_json)
        if not isinstance(value, list):  # pragma: no cover - construction invariant
            raise RuntimeError("sealed calibration rows are not an array")
        return value

    def join_receipt(self) -> dict[str, Any]:
        value = json.loads(self.join_receipt_json)
        if not isinstance(value, dict):  # pragma: no cover - construction invariant
            raise RuntimeError("sealed reference join receipt is not an object")
        return value

    def calibration_jsonl(self) -> str:
        return _calibration_jsonl(self.calibration_rows())


def join_source_dev_references(
    validation: object,
    reference_root: str | Path,
    *,
    reference_reader: Callable[[Path, PurePosixPath], bytes] = (
        read_source_dev_reference_bytes
    ),
) -> JoinedSourceDevCalibrationRows:
    """Join exact source-dev global seizure intervals after Stage-1 closure.

    The ordering in this function is a security property.  The exact sealed
    Stage-1 type, receipt and every recording binding are revalidated before
    ``reference_root`` is resolved, a ``.csv_bi`` path is derived, or the
    supplied reader is called.
    """

    stage1, records, record_binding_roster_sha256 = (
        _validate_stage1_before_reference_access(validation)
    )

    # Everything above this line is reference-free.  Resolve the public
    # source-development reference root only after the complete batch passed.
    raw_reference_root = Path(reference_root)
    if raw_reference_root.is_symlink():
        raise ValueError("TUSZ source-dev reference root must not be a symlink")
    canonical_reference_root = raw_reference_root.resolve(strict=True)
    if not canonical_reference_root.is_dir():
        raise ValueError("TUSZ source-dev reference root must be a directory")
    if not callable(reference_reader):
        raise TypeError("reference_reader must be callable")

    rows: list[dict[str, Any]] = []
    reference_inventory: list[list[object]] = []
    reference_event_inventory: list[list[object]] = []
    selected_event_count = 0
    ignored_row_count = 0
    seizure_free_recording_count = 0
    reference_open_count = 0
    for record in records:
        relative_path = source_dev_reference_relative_path(record.recording_id)
        payload = reference_reader(canonical_reference_root, relative_path)
        reference_open_count += 1
        if not isinstance(payload, bytes) or not payload:
            raise TypeError("reference_reader must return non-empty bytes")
        parsed = parse_tusz_term_seiz_reference_bytes(
            payload,
            duration_seconds=record.duration_seconds,
        )
        events = parsed.events()
        selected_event_count += parsed.selected_term_seiz_row_count
        ignored_row_count += parsed.ignored_non_term_seiz_row_count
        seizure_free_recording_count += int(not events)
        reference_inventory.append(
            [
                record.recording_id,
                relative_path.as_posix(),
                parsed.reference_file_sha256,
                parsed.selected_term_seiz_row_count,
                parsed.ignored_non_term_seiz_row_count,
            ]
        )
        reference_event_inventory.append([record.recording_id, events])
        rows.append(
            {
                "patient_id": record.patient_id,
                "recording_id": record.recording_id,
                "split": "source_dev",
                "duration_seconds": record.duration_seconds,
                "source_signal_sha256": record.canonical_source_signal_sha256,
                "posterior_artifact_id": record.posterior_artifact_id,
                "provider_receipt": record.provider_receipt(),
                "posterior_timeline": record.posterior_timeline(),
                "reference_events": events,
            }
        )

    provider_id = _identifier(stage1["provider_id"], "provider_id")
    validated_rows = validate_continuous_calibration_rows(
        rows,
        provider_id=provider_id,
    )
    if len(validated_rows) != len(records):
        raise RuntimeError("calibration row validation lost frozen recordings")
    rows_json = _canonical_json_text(validated_rows)
    jsonl = _calibration_jsonl(validated_rows)
    recording_ids = [str(row["recording_id"]) for row in validated_rows]
    patient_ids = sorted({str(row["patient_id"]) for row in validated_rows})
    artifact_ids = [str(row["posterior_artifact_id"]) for row in validated_rows]
    receipt: dict[str, Any] = {
        "schema_version": SOURCE_DEV_REFERENCE_JOIN_SCHEMA_VERSION,
        "join_id": "SOURCE-DEV-REFERENCE-JOIN-PENDING",
        "parser_id": SOURCE_DEV_REFERENCE_PARSER_ID,
        "mapping_id": SOURCE_DEV_REFERENCE_MAPPING_ID,
        "source_split": "source_dev",
        "provider_id": provider_id,
        "stage1_validation_id": stage1["validation_id"],
        "stage1_validation_receipt_sha256": stage1["receipt_sha256"],
        "stage1_manifest_sha256": stage1["manifest_sha256"],
        "stage1_batch_receipt_file_sha256": stage1[
            "batch_receipt_file_sha256"
        ],
        "stage1_posterior_index_file_sha256": stage1[
            "posterior_index_file_sha256"
        ],
        "stage1_posterior_artifact_inventory_sha256": stage1[
            "posterior_artifact_inventory_sha256"
        ],
        "stage1_record_binding_roster_sha256": record_binding_roster_sha256,
        "input_recording_roster_sha256": _canonical_sha256(
            sorted(recording_ids)
        ),
        "input_patient_roster_sha256": _canonical_sha256(patient_ids),
        "input_posterior_artifact_roster_sha256": _canonical_sha256(
            sorted(artifact_ids)
        ),
        "reference_file_inventory_sha256": _canonical_sha256(
            reference_inventory
        ),
        "reference_event_inventory_sha256": _canonical_sha256(
            reference_event_inventory
        ),
        "calibration_rows_sha256": _canonical_sha256(validated_rows),
        "output_calibration_jsonl_sha256": hashlib.sha256(
            jsonl.encode("utf-8")
        ).hexdigest(),
        "recording_count": len(validated_rows),
        "patient_count": len(patient_ids),
        "reference_file_count": len(reference_inventory),
        "selected_term_seiz_event_count": selected_event_count,
        "ignored_non_term_seiz_row_count": ignored_row_count,
        "seizure_free_recording_count": seizure_free_recording_count,
        "posterior_validation_completed_before_first_reference_open": True,
        "reference_files_opened": reference_open_count,
        "full_stage1_inventory_joined": True,
        "calibration_rows_revalidated": True,
        "scope_receipt": deepcopy(_JOIN_SCOPE_RECEIPT),
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    receipt["join_id"] = "SRCDEVJOIN-" + _canonical_sha256(receipt)[:24]
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    joined = JoinedSourceDevCalibrationRows(
        calibration_rows_json=rows_json,
        join_receipt_json=_canonical_json_text(receipt),
    )
    validate_source_dev_reference_join(joined)
    return joined


def validate_source_dev_reference_join(
    joined: object,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate a sealed Stage-2 join before it is written or calibrated."""

    if type(joined) is not JoinedSourceDevCalibrationRows:
        raise TypeError("source-dev join must use the exact sealed result type")
    rows = joined.calibration_rows()
    receipt = joined.join_receipt()
    if type(receipt) is not dict or set(receipt) != _JOIN_RECEIPT_FIELDS:
        raise ValueError("source-dev reference join receipt schema drifted")
    if (
        receipt["schema_version"] != SOURCE_DEV_REFERENCE_JOIN_SCHEMA_VERSION
        or receipt["parser_id"] != SOURCE_DEV_REFERENCE_PARSER_ID
        or receipt["mapping_id"] != SOURCE_DEV_REFERENCE_MAPPING_ID
        or receipt["source_split"] != "source_dev"
        or receipt["scope_receipt"] != _JOIN_SCOPE_RECEIPT
    ):
        raise ValueError("source-dev reference join scope drifted")
    for field in (
        "stage1_validation_receipt_sha256",
        "stage1_manifest_sha256",
        "stage1_batch_receipt_file_sha256",
        "stage1_posterior_index_file_sha256",
        "stage1_posterior_artifact_inventory_sha256",
        "stage1_record_binding_roster_sha256",
        "input_recording_roster_sha256",
        "input_patient_roster_sha256",
        "input_posterior_artifact_roster_sha256",
        "reference_file_inventory_sha256",
        "reference_event_inventory_sha256",
        "calibration_rows_sha256",
        "output_calibration_jsonl_sha256",
        "receipt_sha256",
    ):
        _require_sha256(receipt[field], f"join {field}")
    for field in (
        "posterior_validation_completed_before_first_reference_open",
        "full_stage1_inventory_joined",
        "calibration_rows_revalidated",
    ):
        if receipt[field] is not True:
            raise ValueError(f"source-dev join gate is false: {field}")
    provider_id = _identifier(receipt["provider_id"], "join provider_id")
    validated_rows = validate_continuous_calibration_rows(
        rows,
        provider_id=provider_id,
    )
    if validated_rows != rows or _canonical_json_text(rows) != (
        joined.calibration_rows_json
    ):
        raise ValueError("sealed calibration rows are not canonical")
    recording_ids = [str(row["recording_id"]) for row in rows]
    patient_ids = sorted({str(row["patient_id"]) for row in rows})
    artifact_ids = [str(row["posterior_artifact_id"]) for row in rows]
    event_inventory = [
        [str(row["recording_id"]), row["reference_events"]] for row in rows
    ]
    expected_counts = {
        "recording_count": len(rows),
        "patient_count": len(patient_ids),
        "reference_file_count": len(rows),
        "reference_files_opened": len(rows),
        "selected_term_seiz_event_count": sum(
            len(row["reference_events"]) for row in rows
        ),
        "seizure_free_recording_count": sum(
            int(not row["reference_events"]) for row in rows
        ),
    }
    for field, expected in expected_counts.items():
        if _integer(receipt[field], f"join {field}") != expected:
            raise ValueError(f"source-dev join {field} drifted")
    _integer(
        receipt["ignored_non_term_seiz_row_count"],
        "join ignored non-TERM,seiz row count",
    )
    if receipt["input_recording_roster_sha256"] != _canonical_sha256(
        sorted(recording_ids)
    ):
        raise ValueError("source-dev join recording roster hash drifted")
    if receipt["input_patient_roster_sha256"] != _canonical_sha256(patient_ids):
        raise ValueError("source-dev join patient roster hash drifted")
    if receipt["input_posterior_artifact_roster_sha256"] != _canonical_sha256(
        sorted(artifact_ids)
    ):
        raise ValueError("source-dev join artifact roster hash drifted")
    if receipt["reference_event_inventory_sha256"] != _canonical_sha256(
        event_inventory
    ):
        raise ValueError("source-dev join event inventory hash drifted")
    if receipt["calibration_rows_sha256"] != _canonical_sha256(rows):
        raise ValueError("source-dev calibration row hash drifted")
    if receipt["output_calibration_jsonl_sha256"] != hashlib.sha256(
        _calibration_jsonl(rows).encode("utf-8")
    ).hexdigest():
        raise ValueError("source-dev calibration JSONL hash drifted")
    receipt_digest = deepcopy(receipt)
    receipt_digest["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if receipt["receipt_sha256"] != _canonical_sha256(receipt_digest):
        raise ValueError("source-dev reference join receipt hash drifted")
    id_digest = deepcopy(receipt_digest)
    id_digest["join_id"] = "SOURCE-DEV-REFERENCE-JOIN-PENDING"
    if receipt["join_id"] != "SRCDEVJOIN-" + _canonical_sha256(id_digest)[:24]:
        raise ValueError("source-dev reference join ID is not content-bound")
    if _canonical_json_text(receipt) != joined.join_receipt_json:
        raise ValueError("sealed source-dev reference join receipt is not canonical")
    return validated_rows, receipt


def write_source_dev_reference_join_append_only(
    joined: object,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Write one immutable two-file calibration bundle without overwriting."""

    rows, receipt = validate_source_dev_reference_join(joined)
    raw_output = Path(output_directory)
    if raw_output.exists() or raw_output.is_symlink():
        raise FileExistsError(raw_output)
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    raw_output.mkdir(exist_ok=False)
    rows_path = raw_output / SOURCE_DEV_CALIBRATION_ROWS_FILENAME
    receipt_path = raw_output / SOURCE_DEV_REFERENCE_JOIN_RECEIPT_FILENAME
    jsonl = _calibration_jsonl(rows)
    with rows_path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(jsonl)
    with receipt_path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
        handle.write("\n")
    return {
        "output_directory": str(raw_output.resolve(strict=True)),
        "calibration_rows_path": str(rows_path.resolve(strict=True)),
        "join_receipt_path": str(receipt_path.resolve(strict=True)),
        "calibration_rows_file_sha256": hashlib.sha256(
            rows_path.read_bytes()
        ).hexdigest(),
        "join_receipt_file_sha256": hashlib.sha256(
            receipt_path.read_bytes()
        ).hexdigest(),
        "join_id": receipt["join_id"],
        "overwrite_performed": False,
    }


__all__ = [
    "JoinedSourceDevCalibrationRows",
    "ParsedSourceDevReference",
    "SOURCE_DEV_CALIBRATION_ROWS_FILENAME",
    "SOURCE_DEV_REFERENCE_JOIN_SCHEMA_VERSION",
    "SOURCE_DEV_REFERENCE_JOIN_RECEIPT_FILENAME",
    "SOURCE_DEV_REFERENCE_MAPPING_ID",
    "SOURCE_DEV_REFERENCE_PARSER_ID",
    "join_source_dev_references",
    "parse_tusz_term_seiz_reference_bytes",
    "read_source_dev_reference_bytes",
    "source_dev_reference_relative_path",
    "validate_source_dev_reference_join",
    "write_source_dev_reference_join_append_only",
]
