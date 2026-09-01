"""Provider-neutral, prediction-first runner for continuous EEG detection.

Stage-P materializes one terminal outcome for every selected long-recording
identity before any public reference interval is opened.  Records are
append-only and independently replayable, so a technical failure never erases
the denominator and an interrupted batch can resume without overwriting prior
predictions.

This module deliberately does not score detection accuracy.  It produces the
prediction-side inventory that a later, separately authorized reference join
may score.  A green batch receipt therefore proves inventory closure and
artifact integrity only, not detector admission or clinical performance.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Callable, Final, Mapping, Sequence
import uuid


STAGE_P_RUN_CONTRACT_SCHEMA_V1: Final[str] = (
    "clinical_eeg_continuous_detection_stage_p_run_contract_v1"
)
STAGE_P_ATTEMPT_SCHEMA_V1: Final[str] = (
    "clinical_eeg_continuous_detection_stage_p_attempt_v1"
)
STAGE_P_TERMINAL_SCHEMA_V1: Final[str] = (
    "clinical_eeg_continuous_detection_stage_p_terminal_v1"
)
STAGE_P_BATCH_SCHEMA_V1: Final[str] = (
    "clinical_eeg_continuous_detection_stage_p_batch_receipt_v1"
)
STAGE_P_METHOD_ID_V1: Final[str] = (
    "provider_neutral_prediction_first_terminal_outcome_runner_v1"
)

STAGE_P_TERMINAL_OUTCOMES: Final[tuple[str, ...]] = (
    "completed_with_alarms",
    "completed_zero_alarm",
    "partial_coverage",
    "technical_failure",
)
STAGE_P_SUCCESS_OUTCOMES: Final[frozenset[str]] = frozenset(
    STAGE_P_TERMINAL_OUTCOMES[:-1]
)
STAGE_P_SERVICE_STATES: Final[tuple[str, str]] = (
    "cold_process_start",
    "warm_same_process",
)
_SHA256_ALPHABET = frozenset("0123456789abcdef")
_TIMING_FIELDS = (
    "edf_io_seconds",
    "preprocessing_seconds",
    "inference_seconds",
    "decoder_seconds",
    "end_to_end_seconds",
)
_EEG_ONLY_SCOPE = {
    "eeg_samples_used": True,
    "edf_signal_header_used": True,
    "edf_annotations_used": False,
    "spreadsheet_used": False,
    "doctor_labels_used": False,
    "clinical_text_used": False,
    "reference_events_opened_during_stage_p": False,
}


# The supplied path is a fresh, attempt-specific artifact directory.  Artifact
# paths returned by the processor are relative to that directory; the runner
# prefixes them into the immutable record-level lineage before publication.
StagePProcessor = Callable[[Mapping[str, Any], Path, str], Mapping[str, Any]]
StagePResumeValidator = Callable[
    [Mapping[str, Any], Mapping[str, Any], Path], None
]


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identifier(value: object, name: str) -> str:
    text = str(value)
    if not text or text != text.strip():
        raise ValueError(f"{name} must be a non-empty trimmed identifier")
    return text


def _sha256(value: object, name: str) -> str:
    text = str(value)
    if len(text) != 64 or set(text) - _SHA256_ALPHABET:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return text


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"{name} must be a non-negative integer")
    return value


def _regular_file(path: Path, name: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be a regular non-symlink file")
    return path


def _read_json(path: Path, name: str) -> dict[str, Any]:
    raw = _regular_file(path, name).read_bytes()
    if not raw:
        raise ValueError(f"{name} is empty")

    def no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate JSON field {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=no_duplicates,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"{name} contains non-finite JSON constant {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_create_only(path: Path, value: object) -> None:
    """Atomically publish JSON without ever replacing an existing artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.link(temporary, path)
    except FileExistsError:
        raise FileExistsError(f"append-only artifact already exists: {path}")
    finally:
        temporary.unlink(missing_ok=True)
    _fsync_directory(path.parent)


def _safe_relative_path(value: object, name: str) -> str:
    text = _identifier(value, name)
    relative = PurePosixPath(text)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"{name} must be a safe relative path")
    if text != relative.as_posix():
        raise ValueError(f"{name} must use canonical POSIX separators")
    return text


def _duration_fraction(value: object) -> tuple[int, int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
    ):
        raise TypeError("recording duration fraction must be [numerator, denominator]")
    numerator, denominator = value
    if (
        isinstance(numerator, bool)
        or not isinstance(numerator, int)
        or numerator <= 0
        or isinstance(denominator, bool)
        or not isinstance(denominator, int)
        or denominator <= 0
    ):
        raise ValueError("recording duration fraction must contain positive integers")
    common = math.gcd(numerator, denominator)
    return numerator // common, denominator // common


def _normalize_record(value: object, *, model_split: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("Stage-P record must be a mapping")
    recording_id = _identifier(value.get("recording_id"), "recording_id")
    row_split = _identifier(value.get("model_split"), "record model_split")
    if row_split != model_split:
        raise ValueError("Stage-P record crosses the frozen model split")
    numerator, denominator = _duration_fraction(
        value.get("recording_duration_fraction")
    )
    container = _sha256(
        value.get("source_edf_container_sha256"), "source EDF container"
    )
    analysis_identity = _identifier(
        value.get("analysis_identity"), "analysis identity"
    )
    body: dict[str, Any] = {
        "recording_id": recording_id,
        "analysis_identity": analysis_identity,
        "model_split": row_split,
        "source_edf_container_sha256": container,
        "recording_duration_fraction": [numerator, denominator],
        "record_key": hashlib.sha256(recording_id.encode("utf-8")).hexdigest()[:24],
        "record_receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    body["record_receipt_sha256"] = _canonical_sha256(body)
    return body


def build_stage_p_run_contract_v1(
    *,
    provider_id: str,
    model_split: str,
    records: Sequence[Mapping[str, Any]],
    source_identity_projection_sha256: str,
    provider_execution_receipt_sha256: str,
    checkpoint_sha256: str,
    provider_code_sha256: str,
    preprocessing_contract_sha256: str,
    decoder_contract_sha256: str,
    runtime_hardware_contract_sha256: str,
) -> dict[str, Any]:
    """Freeze an identity-only, reference-free prediction inventory."""

    provider = _identifier(provider_id, "provider_id")
    split = _identifier(model_split, "model_split")
    if split not in {"source_train", "source_dev", "source_eval"}:
        raise ValueError("Stage-P model split is unsupported")
    normalized = sorted(
        (_normalize_record(row, model_split=split) for row in records),
        key=lambda row: row["recording_id"],
    )
    if not normalized:
        raise ValueError("Stage-P run contract requires at least one record")
    if len({row["recording_id"] for row in normalized}) != len(normalized):
        raise ValueError("Stage-P run contract repeats a recording identity")
    if len({row["record_key"] for row in normalized}) != len(normalized):
        raise ValueError("Stage-P record-key collision")
    body: dict[str, Any] = {
        "schema_version": STAGE_P_RUN_CONTRACT_SCHEMA_V1,
        "method_id": STAGE_P_METHOD_ID_V1,
        "run_id": "STAGE-P-RUN-PENDING",
        "provider_id": provider,
        "model_split": split,
        "source_identity_projection_sha256": _sha256(
            source_identity_projection_sha256, "source identity projection"
        ),
        "provider_execution_receipt_sha256": _sha256(
            provider_execution_receipt_sha256, "provider execution receipt"
        ),
        "checkpoint_sha256": _sha256(checkpoint_sha256, "checkpoint"),
        "provider_code_sha256": _sha256(provider_code_sha256, "provider code"),
        "preprocessing_contract_sha256": _sha256(
            preprocessing_contract_sha256, "preprocessing contract"
        ),
        "decoder_contract_sha256": _sha256(
            decoder_contract_sha256, "decoder contract"
        ),
        "runtime_hardware_contract_sha256": _sha256(
            runtime_hardware_contract_sha256, "runtime hardware contract"
        ),
        "selected_record_count": len(normalized),
        "record_roster_sha256": _canonical_sha256(normalized),
        "records": normalized,
        "prediction_first_before_reference_join": True,
        "terminal_outcome_roster": list(STAGE_P_TERMINAL_OUTCOMES),
        "scope_receipt": deepcopy(_EEG_ONLY_SCOPE),
        "contract_sha256": "CONTENT-ADDRESS-PENDING",
    }
    id_source = deepcopy(body)
    body["run_id"] = "STAGEP-" + _canonical_sha256(id_source)[:24]
    body["contract_sha256"] = _canonical_sha256(body)
    return validate_stage_p_run_contract_v1(body)


def validate_stage_p_run_contract_v1(value: object) -> dict[str, Any]:
    required = {
        "schema_version",
        "method_id",
        "run_id",
        "provider_id",
        "model_split",
        "source_identity_projection_sha256",
        "provider_execution_receipt_sha256",
        "checkpoint_sha256",
        "provider_code_sha256",
        "preprocessing_contract_sha256",
        "decoder_contract_sha256",
        "runtime_hardware_contract_sha256",
        "selected_record_count",
        "record_roster_sha256",
        "records",
        "prediction_first_before_reference_join",
        "terminal_outcome_roster",
        "scope_receipt",
        "contract_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("Stage-P run contract fields drifted")
    data = deepcopy(dict(value))
    if (
        data["schema_version"] != STAGE_P_RUN_CONTRACT_SCHEMA_V1
        or data["method_id"] != STAGE_P_METHOD_ID_V1
        or data["prediction_first_before_reference_join"] is not True
        or data["terminal_outcome_roster"] != list(STAGE_P_TERMINAL_OUTCOMES)
        or data["scope_receipt"] != _EEG_ONLY_SCOPE
    ):
        raise ValueError("Stage-P run contract permissions or semantics drifted")
    records = [
        _normalize_record(row, model_split=str(data["model_split"]))
        for row in data["records"]
    ]
    if records != data["records"] or records != sorted(
        records, key=lambda row: row["recording_id"]
    ):
        raise ValueError("Stage-P record roster is not canonical")
    if data["selected_record_count"] != len(records) or data[
        "record_roster_sha256"
    ] != _canonical_sha256(records):
        raise ValueError("Stage-P run contract record denominator drifted")
    for field in (
        "source_identity_projection_sha256",
        "provider_execution_receipt_sha256",
        "checkpoint_sha256",
        "provider_code_sha256",
        "preprocessing_contract_sha256",
        "decoder_contract_sha256",
        "runtime_hardware_contract_sha256",
    ):
        _sha256(data[field], field)
    contract = data["contract_sha256"]
    _sha256(contract, "Stage-P run contract")
    digest = deepcopy(data)
    digest["contract_sha256"] = "CONTENT-ADDRESS-PENDING"
    if contract != _canonical_sha256(digest):
        raise ValueError("Stage-P run contract self-hash is invalid")
    return data


def _validate_artifacts(
    artifacts: object, *, record_directory: Path
) -> list[dict[str, str]]:
    if not isinstance(artifacts, list):
        raise TypeError("Stage-P artifacts must be a list")
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in artifacts:
        if not isinstance(item, Mapping) or set(item) != {
            "relative_path",
            "file_sha256",
            "semantic",
        }:
            raise ValueError("Stage-P artifact row fields drifted")
        relative = _safe_relative_path(item["relative_path"], "artifact path")
        if relative in seen:
            raise ValueError("Stage-P artifact path is repeated")
        seen.add(relative)
        expected = _sha256(item["file_sha256"], "artifact file")
        semantic = _identifier(item["semantic"], "artifact semantic")
        path = record_directory.joinpath(*PurePosixPath(relative).parts)
        if record_directory.resolve() not in path.resolve(strict=True).parents:
            raise ValueError("Stage-P artifact escaped its record directory")
        _regular_file(path, "Stage-P record artifact")
        if _file_sha256(path) != expected:
            raise ValueError("Stage-P record artifact file hash mismatch")
        rows.append(
            {
                "relative_path": relative,
                "file_sha256": expected,
                "semantic": semantic,
            }
        )
    rows.sort(key=lambda row: (row["semantic"], row["relative_path"]))
    if rows != artifacts:
        raise ValueError("Stage-P artifact roster must be canonically sorted")
    return rows


def _normalize_timing(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(_TIMING_FIELDS):
        raise ValueError("Stage-P timing fields drifted")
    timing = {
        name: _finite_nonnegative(value[name], f"timing {name}")
        for name in _TIMING_FIELDS
    }
    component_sum = sum(timing[name] for name in _TIMING_FIELDS[:-1])
    if component_sum > timing["end_to_end_seconds"] + 1e-6:
        raise ValueError("Stage-P timing components exceed end-to-end time")
    return timing


def _terminal_from_success(
    result: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    record: Mapping[str, Any],
    record_directory: Path,
    attempt: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "outcome_status",
        "recording_duration_seconds",
        "event_proposal_count",
        "modeled_target_coverage_seconds",
        "provider_result_receipt_sha256",
        "provider_prediction_receipt_sha256",
        "timing_seconds",
        "artifacts",
    }
    if set(result) != required:
        raise ValueError("Stage-P processor result fields drifted")
    outcome = str(result["outcome_status"])
    if outcome not in STAGE_P_SUCCESS_OUTCOMES:
        raise ValueError("Stage-P processor may return only non-failure outcomes")
    duration = _finite_nonnegative(
        result["recording_duration_seconds"], "recording duration"
    )
    if duration <= 0.0:
        raise ValueError("Stage-P recording duration must be positive")
    expected_duration = (
        record["recording_duration_fraction"][0]
        / record["recording_duration_fraction"][1]
    )
    if not math.isclose(
        duration, expected_duration, rel_tol=1e-12, abs_tol=1e-9
    ):
        raise ValueError(
            "Stage-P processor recording duration disagrees with the frozen roster"
        )
    modeled = _finite_nonnegative(
        result["modeled_target_coverage_seconds"], "modeled target coverage"
    )
    if modeled > duration + 1e-6:
        raise ValueError("Stage-P modeled coverage exceeds recording duration")
    events = _nonnegative_integer(result["event_proposal_count"], "event count")
    if outcome == "completed_zero_alarm" and events != 0:
        raise ValueError("Stage-P zero-alarm status disagrees with event count")
    if outcome == "completed_with_alarms" and events == 0:
        raise ValueError("Stage-P alarm status disagrees with event count")
    if outcome == "completed_with_alarms" and modeled + 1e-6 < duration:
        raise ValueError("completed-with-alarms requires complete target coverage")
    if outcome == "completed_zero_alarm" and modeled + 1e-6 < duration:
        raise ValueError("completed-zero-alarm requires complete target coverage")
    if outcome == "partial_coverage" and modeled + 1e-6 >= duration:
        raise ValueError("partial-coverage status requires incomplete coverage")
    body: dict[str, Any] = {
        "schema_version": STAGE_P_TERMINAL_SCHEMA_V1,
        "method_id": STAGE_P_METHOD_ID_V1,
        "run_contract_sha256": contract["contract_sha256"],
        "provider_id": contract["provider_id"],
        "model_split": contract["model_split"],
        "recording_id": record["recording_id"],
        "record_key": record["record_key"],
        "record_receipt_sha256": record["record_receipt_sha256"],
        "attempt_id": attempt["attempt_id"],
        "execution_session_id": attempt["execution_session_id"],
        "service_state": attempt["service_state"],
        "outcome_status": outcome,
        "recording_duration_seconds": duration,
        "event_proposal_count": events,
        "modeled_target_coverage_seconds": modeled,
        "provider_result_receipt_sha256": _sha256(
            result["provider_result_receipt_sha256"], "provider result receipt"
        ),
        "provider_prediction_receipt_sha256": _sha256(
            result["provider_prediction_receipt_sha256"],
            "provider prediction receipt",
        ),
        "timing_seconds": _normalize_timing(result["timing_seconds"]),
        "artifacts": _validate_artifacts(
            result["artifacts"], record_directory=record_directory
        ),
        "technical_failure": None,
        "scope_receipt": deepcopy(_EEG_ONLY_SCOPE),
        "terminal_receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    body["terminal_receipt_sha256"] = _canonical_sha256(body)
    return body


def _bind_attempt_artifacts(
    result: Mapping[str, Any],
    *,
    attempt: Mapping[str, Any],
    attempt_artifact_directory: Path,
) -> dict[str, Any]:
    """Validate processor-local files and project paths to the record root."""

    if "artifacts" not in result or not isinstance(result["artifacts"], list):
        raise TypeError("Stage-P processor artifacts must be a list")
    prefix = PurePosixPath("artifacts") / str(attempt["attempt_id"])
    projected: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in result["artifacts"]:
        if not isinstance(item, Mapping) or set(item) != {
            "relative_path",
            "file_sha256",
            "semantic",
        }:
            raise ValueError("Stage-P processor artifact fields drifted")
        relative = _safe_relative_path(
            item["relative_path"], "processor artifact path"
        )
        if relative in seen:
            raise ValueError("Stage-P processor artifact path is repeated")
        seen.add(relative)
        expected = _sha256(item["file_sha256"], "processor artifact file")
        semantic = _identifier(item["semantic"], "processor artifact semantic")
        local = attempt_artifact_directory.joinpath(
            *PurePosixPath(relative).parts
        )
        resolved = local.resolve(strict=True)
        if attempt_artifact_directory.resolve() not in resolved.parents:
            raise ValueError("Stage-P processor artifact escaped its attempt")
        _regular_file(resolved, "Stage-P processor artifact")
        if _file_sha256(resolved) != expected:
            raise ValueError("Stage-P processor artifact file hash mismatch")
        projected.append(
            {
                "relative_path": (prefix / PurePosixPath(relative)).as_posix(),
                "file_sha256": expected,
                "semantic": semantic,
            }
        )
    projected.sort(key=lambda row: (row["semantic"], row["relative_path"]))
    value = deepcopy(dict(result))
    value["artifacts"] = projected
    return value


def _terminal_from_failure(
    error: Exception,
    *,
    contract: Mapping[str, Any],
    record: Mapping[str, Any],
    attempt: Mapping[str, Any],
) -> dict[str, Any]:
    numerator, denominator = record["recording_duration_fraction"]
    message = str(error)
    body: dict[str, Any] = {
        "schema_version": STAGE_P_TERMINAL_SCHEMA_V1,
        "method_id": STAGE_P_METHOD_ID_V1,
        "run_contract_sha256": contract["contract_sha256"],
        "provider_id": contract["provider_id"],
        "model_split": contract["model_split"],
        "recording_id": record["recording_id"],
        "record_key": record["record_key"],
        "record_receipt_sha256": record["record_receipt_sha256"],
        "attempt_id": attempt["attempt_id"],
        "execution_session_id": attempt["execution_session_id"],
        "service_state": attempt["service_state"],
        "outcome_status": "technical_failure",
        "recording_duration_seconds": numerator / denominator,
        "event_proposal_count": 0,
        "modeled_target_coverage_seconds": 0.0,
        "provider_result_receipt_sha256": None,
        "provider_prediction_receipt_sha256": None,
        "timing_seconds": None,
        "artifacts": [],
        "technical_failure": {
            "exception_type": type(error).__name__,
            "message_sha256": hashlib.sha256(message.encode("utf-8")).hexdigest(),
            "message_preview": message[:512],
            "failure_stage": "record_processor",
            "retryable_by_policy": True,
        },
        "scope_receipt": deepcopy(_EEG_ONLY_SCOPE),
        "terminal_receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    body["terminal_receipt_sha256"] = _canonical_sha256(body)
    return body


def validate_stage_p_terminal_v1(
    value: object,
    *,
    contract: Mapping[str, Any],
    record: Mapping[str, Any],
    record_directory: Path,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("Stage-P terminal must be a mapping")
    data = deepcopy(dict(value))
    required = {
        "schema_version",
        "method_id",
        "run_contract_sha256",
        "provider_id",
        "model_split",
        "recording_id",
        "record_key",
        "record_receipt_sha256",
        "attempt_id",
        "execution_session_id",
        "service_state",
        "outcome_status",
        "recording_duration_seconds",
        "event_proposal_count",
        "modeled_target_coverage_seconds",
        "provider_result_receipt_sha256",
        "provider_prediction_receipt_sha256",
        "timing_seconds",
        "artifacts",
        "technical_failure",
        "scope_receipt",
        "terminal_receipt_sha256",
    }
    if set(data) != required:
        raise ValueError("Stage-P terminal fields drifted")
    if (
        data["schema_version"] != STAGE_P_TERMINAL_SCHEMA_V1
        or data["method_id"] != STAGE_P_METHOD_ID_V1
        or data["run_contract_sha256"] != contract["contract_sha256"]
        or data["provider_id"] != contract["provider_id"]
        or data["model_split"] != contract["model_split"]
        or data["recording_id"] != record["recording_id"]
        or data["record_key"] != record["record_key"]
        or data["record_receipt_sha256"] != record["record_receipt_sha256"]
        or data["scope_receipt"] != _EEG_ONLY_SCOPE
        or data["service_state"] not in STAGE_P_SERVICE_STATES
        or data["outcome_status"] not in STAGE_P_TERMINAL_OUTCOMES
    ):
        raise ValueError("Stage-P terminal identity or permission drifted")
    _identifier(data["attempt_id"], "attempt_id")
    _identifier(data["execution_session_id"], "execution_session_id")
    duration = _finite_nonnegative(
        data["recording_duration_seconds"], "terminal recording duration"
    )
    modeled = _finite_nonnegative(
        data["modeled_target_coverage_seconds"], "terminal modeled coverage"
    )
    if duration <= 0.0 or modeled > duration + 1e-6:
        raise ValueError("Stage-P terminal coverage is invalid")
    expected_duration = (
        record["recording_duration_fraction"][0]
        / record["recording_duration_fraction"][1]
    )
    if not math.isclose(
        duration, expected_duration, rel_tol=1e-12, abs_tol=1e-9
    ):
        raise ValueError("Stage-P terminal duration drifted from the frozen roster")
    events = _nonnegative_integer(data["event_proposal_count"], "event count")
    if data["outcome_status"] == "technical_failure":
        if (
            data["provider_result_receipt_sha256"] is not None
            or data["provider_prediction_receipt_sha256"] is not None
            or data["timing_seconds"] is not None
            or data["artifacts"] != []
            or not isinstance(data["technical_failure"], Mapping)
            or set(data["technical_failure"])
            != {
                "exception_type",
                "message_sha256",
                "message_preview",
                "failure_stage",
                "retryable_by_policy",
            }
            or events != 0
            or modeled != 0.0
        ):
            raise ValueError("Stage-P technical-failure semantics drifted")
        _sha256(data["technical_failure"]["message_sha256"], "failure message")
    else:
        if data["technical_failure"] is not None:
            raise ValueError("successful Stage-P terminal carries a failure")
        _sha256(data["provider_result_receipt_sha256"], "provider result")
        _sha256(data["provider_prediction_receipt_sha256"], "provider prediction")
        _normalize_timing(data["timing_seconds"])
        _validate_artifacts(data["artifacts"], record_directory=record_directory)
        if data["outcome_status"] == "completed_zero_alarm" and events != 0:
            raise ValueError("Stage-P terminal zero-alarm semantics drifted")
        if data["outcome_status"] == "completed_with_alarms" and events == 0:
            raise ValueError("Stage-P terminal alarm semantics drifted")
        if data["outcome_status"] in {
            "completed_zero_alarm",
            "completed_with_alarms",
        } and modeled + 1e-6 < duration:
            raise ValueError("Stage-P complete terminal has incomplete coverage")
        if data["outcome_status"] == "partial_coverage" and modeled + 1e-6 >= duration:
            raise ValueError("Stage-P partial terminal has complete coverage")
    receipt = _sha256(data["terminal_receipt_sha256"], "terminal receipt")
    digest = deepcopy(data)
    digest["terminal_receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if receipt != _canonical_sha256(digest):
        raise ValueError("Stage-P terminal self-hash is invalid")
    return data


def _attempt_payload(
    *,
    contract: Mapping[str, Any],
    record: Mapping[str, Any],
    ordinal: int,
    session_id: str,
    service_state: str,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": STAGE_P_ATTEMPT_SCHEMA_V1,
        "method_id": STAGE_P_METHOD_ID_V1,
        "attempt_id": (
            f"STAGEP-{record['record_key']}-ATTEMPT-{ordinal:06d}"
        ),
        "attempt_ordinal": ordinal,
        "execution_session_id": session_id,
        "service_state": service_state,
        "run_contract_sha256": contract["contract_sha256"],
        "recording_id": record["recording_id"],
        "record_receipt_sha256": record["record_receipt_sha256"],
        "status": "attempt_started_terminal_not_yet_published",
        "attempt_receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    body["attempt_receipt_sha256"] = _canonical_sha256(body)
    return body


def _next_attempt_ordinal(record_directory: Path) -> int:
    directory = record_directory / "attempts"
    if not directory.exists():
        return 1
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Stage-P attempts path is not a regular directory")
    ordinals: list[int] = []
    for path in directory.iterdir():
        if path.is_symlink() or not path.is_file() or path.suffix != ".json":
            raise ValueError("Stage-P attempts directory contains an invalid entry")
        stem = path.stem
        if not stem.startswith("attempt-") or not stem[8:].isdigit():
            raise ValueError("Stage-P attempt filename is invalid")
        payload = _read_json(path, "Stage-P attempt")
        receipt = payload.get("attempt_receipt_sha256")
        _sha256(receipt, "attempt receipt")
        digest = deepcopy(payload)
        digest["attempt_receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
        if receipt != _canonical_sha256(digest):
            raise ValueError("Stage-P attempt self-hash is invalid")
        ordinals.append(int(stem[8:]))
    if len(ordinals) != len(set(ordinals)):
        raise ValueError("Stage-P attempt ordinal is duplicated")
    return max(ordinals, default=0) + 1


def _build_batch_receipt(
    *,
    contract: Mapping[str, Any],
    output_root: Path,
    terminals: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    outcomes = {
        outcome: sum(row["outcome_status"] == outcome for row in terminals)
        for outcome in STAGE_P_TERMINAL_OUTCOMES
    }
    inventory: list[dict[str, Any]] = []
    for record, terminal in zip(contract["records"], terminals):
        path = output_root / "records" / record["record_key"] / "terminal.json"
        inventory.append(
            {
                "recording_id": record["recording_id"],
                "record_key": record["record_key"],
                "outcome_status": terminal["outcome_status"],
                "service_state": terminal["service_state"],
                "terminal_receipt_sha256": terminal["terminal_receipt_sha256"],
                "terminal_file_sha256": _file_sha256(path),
            }
        )
    expected_seconds = sum(
        row["recording_duration_fraction"][0]
        / row["recording_duration_fraction"][1]
        for row in contract["records"]
    )
    warm = [
        row
        for row in terminals
        if row["service_state"] == "warm_same_process"
        and row["outcome_status"] in STAGE_P_SUCCESS_OUTCOMES
    ]
    warm_eeg_seconds = sum(row["recording_duration_seconds"] for row in warm)
    warm_wall_seconds = sum(row["timing_seconds"]["end_to_end_seconds"] for row in warm)
    body: dict[str, Any] = {
        "schema_version": STAGE_P_BATCH_SCHEMA_V1,
        "method_id": STAGE_P_METHOD_ID_V1,
        "batch_id": "STAGE-P-BATCH-PENDING",
        "run_id": contract["run_id"],
        "run_contract_sha256": contract["contract_sha256"],
        "provider_id": contract["provider_id"],
        "model_split": contract["model_split"],
        "selected_record_count": len(contract["records"]),
        "terminal_record_count": len(terminals),
        "outcome_counts": outcomes,
        "expected_recording_seconds": expected_seconds,
        "modeled_target_coverage_seconds": sum(
            row["modeled_target_coverage_seconds"] for row in terminals
        ),
        "warm_successful_record_count": len(warm),
        "warm_recording_seconds": warm_eeg_seconds,
        "warm_end_to_end_seconds": warm_wall_seconds,
        "warm_end_to_end_rtf": (
            warm_wall_seconds / warm_eeg_seconds if warm_eeg_seconds > 0.0 else None
        ),
        "terminal_inventory": inventory,
        "complete_inventory_closure": len(terminals) == len(contract["records"]),
        "prediction_frozen_before_reference_join": True,
        "scientific_status": (
            "prediction_inventory_closed_not_detection_accuracy_or_admission"
        ),
        "scope_receipt": deepcopy(_EEG_ONLY_SCOPE),
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    body["batch_id"] = "STAGEPBATCH-" + _canonical_sha256(body)[:24]
    body["receipt_sha256"] = _canonical_sha256(body)
    return validate_stage_p_batch_receipt_v1(body, contract=contract)


def validate_stage_p_batch_receipt_v1(
    value: object, *, contract: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("Stage-P batch receipt must be a mapping")
    data = deepcopy(dict(value))
    required = {
        "schema_version",
        "method_id",
        "batch_id",
        "run_id",
        "run_contract_sha256",
        "provider_id",
        "model_split",
        "selected_record_count",
        "terminal_record_count",
        "outcome_counts",
        "expected_recording_seconds",
        "modeled_target_coverage_seconds",
        "warm_successful_record_count",
        "warm_recording_seconds",
        "warm_end_to_end_seconds",
        "warm_end_to_end_rtf",
        "terminal_inventory",
        "complete_inventory_closure",
        "prediction_frozen_before_reference_join",
        "scientific_status",
        "scope_receipt",
        "receipt_sha256",
    }
    if set(data) != required:
        raise ValueError("Stage-P batch receipt fields drifted")
    if (
        data["schema_version"] != STAGE_P_BATCH_SCHEMA_V1
        or data["method_id"] != STAGE_P_METHOD_ID_V1
        or data["run_id"] != contract["run_id"]
        or data["run_contract_sha256"] != contract["contract_sha256"]
        or data["provider_id"] != contract["provider_id"]
        or data["model_split"] != contract["model_split"]
        or data["scope_receipt"] != _EEG_ONLY_SCOPE
        or data["complete_inventory_closure"] is not True
        or data["prediction_frozen_before_reference_join"] is not True
        or data["scientific_status"]
        != "prediction_inventory_closed_not_detection_accuracy_or_admission"
    ):
        raise ValueError("Stage-P batch identity or permissions drifted")
    selected = _nonnegative_integer(data["selected_record_count"], "selected count")
    terminal = _nonnegative_integer(data["terminal_record_count"], "terminal count")
    if selected != len(contract["records"]) or terminal != selected:
        raise ValueError("Stage-P batch denominator is incomplete")
    counts = data["outcome_counts"]
    if not isinstance(counts, Mapping) or set(counts) != set(
        STAGE_P_TERMINAL_OUTCOMES
    ) or any(_nonnegative_integer(value, "outcome count") < 0 for value in counts.values()):
        raise ValueError("Stage-P batch outcome counts are invalid")
    if sum(counts.values()) != selected:
        raise ValueError("Stage-P batch outcome counts do not close")
    inventory = data["terminal_inventory"]
    if not isinstance(inventory, list) or len(inventory) != selected:
        raise ValueError("Stage-P batch terminal inventory is incomplete")
    if [row["recording_id"] for row in inventory] != [
        row["recording_id"] for row in contract["records"]
    ]:
        raise ValueError("Stage-P batch terminal inventory order drifted")
    for row in inventory:
        if not isinstance(row, Mapping) or set(row) != {
            "recording_id",
            "record_key",
            "outcome_status",
            "service_state",
            "terminal_receipt_sha256",
            "terminal_file_sha256",
        }:
            raise ValueError("Stage-P batch inventory row fields drifted")
        if (
            row["outcome_status"] not in STAGE_P_TERMINAL_OUTCOMES
            or row["service_state"] not in STAGE_P_SERVICE_STATES
        ):
            raise ValueError("Stage-P batch inventory state is invalid")
        _sha256(row["terminal_receipt_sha256"], "terminal receipt")
        _sha256(row["terminal_file_sha256"], "terminal file")
    for field in (
        "expected_recording_seconds",
        "modeled_target_coverage_seconds",
        "warm_recording_seconds",
        "warm_end_to_end_seconds",
    ):
        _finite_nonnegative(data[field], field)
    warm_count = _nonnegative_integer(
        data["warm_successful_record_count"], "warm successful count"
    )
    if warm_count > selected:
        raise ValueError("Stage-P warm count exceeds selected records")
    if data["warm_end_to_end_rtf"] is not None:
        observed = _finite_nonnegative(data["warm_end_to_end_rtf"], "warm RTF")
        if data["warm_recording_seconds"] <= 0.0 or not math.isclose(
            observed,
            data["warm_end_to_end_seconds"] / data["warm_recording_seconds"],
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("Stage-P warm RTF is inconsistent")
    receipt = _sha256(data["receipt_sha256"], "batch receipt")
    digest = deepcopy(data)
    digest["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if receipt != _canonical_sha256(digest):
        raise ValueError("Stage-P batch receipt self-hash is invalid")
    return data


def run_stage_p_prediction_batch_v1(
    *,
    output_directory: str | Path,
    run_contract: Mapping[str, Any],
    processor: StagePProcessor,
    resume_validator: StagePResumeValidator | None = None,
) -> dict[str, Any]:
    """Run or resume a complete prediction-first inventory.

    ``processor`` may write provider-specific artifacts only beneath the
    supplied fresh attempt directory and returns paths relative to that
    directory.  Ordinary ``Exception`` instances become
    terminal technical failures; ``BaseException`` (for example an interrupt)
    is deliberately not swallowed, leaving an append-only attempt for resume.
    """

    contract = validate_stage_p_run_contract_v1(run_contract)
    if not callable(processor):
        raise TypeError("Stage-P processor must be callable")
    if resume_validator is not None and not callable(resume_validator):
        raise TypeError("Stage-P resume validator must be callable")
    root = Path(output_directory).resolve()
    if root.is_symlink():
        raise ValueError("Stage-P output directory cannot be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    contract_path = root / "run_contract.json"
    if contract_path.exists() or contract_path.is_symlink():
        observed = validate_stage_p_run_contract_v1(
            _read_json(contract_path, "Stage-P run contract")
        )
        if observed != contract:
            raise ValueError("Stage-P resume run contract drifted")
    else:
        _write_json_create_only(contract_path, contract)

    session_id = "STAGEPSESSION-" + uuid.uuid4().hex
    attempted_in_process = 0
    terminals: list[dict[str, Any]] = []
    for record in contract["records"]:
        record_directory = root / "records" / record["record_key"]
        record_directory.mkdir(parents=True, exist_ok=True)
        terminal_path = record_directory / "terminal.json"
        if terminal_path.exists() or terminal_path.is_symlink():
            terminal = validate_stage_p_terminal_v1(
                _read_json(terminal_path, "Stage-P terminal"),
                contract=contract,
                record=record,
                record_directory=record_directory,
            )
            if resume_validator is not None:
                resume_validator(record, terminal, record_directory)
            terminals.append(terminal)
            continue

        service_state = (
            "cold_process_start" if attempted_in_process == 0 else "warm_same_process"
        )
        attempted_in_process += 1
        ordinal = _next_attempt_ordinal(record_directory)
        attempt = _attempt_payload(
            contract=contract,
            record=record,
            ordinal=ordinal,
            session_id=session_id,
            service_state=service_state,
        )
        _write_json_create_only(
            record_directory / "attempts" / f"attempt-{ordinal:06d}.json",
            attempt,
        )
        attempt_artifact_directory = (
            record_directory / "artifacts" / str(attempt["attempt_id"])
        )
        attempt_artifact_directory.mkdir(parents=True, exist_ok=False)
        try:
            raw = processor(record, attempt_artifact_directory, service_state)
            if not isinstance(raw, Mapping):
                raise TypeError("Stage-P processor must return a mapping")
            raw = _bind_attempt_artifacts(
                raw,
                attempt=attempt,
                attempt_artifact_directory=attempt_artifact_directory,
            )
            terminal = _terminal_from_success(
                raw,
                contract=contract,
                record=record,
                record_directory=record_directory,
                attempt=attempt,
            )
        except Exception as error:  # terminal denominator preservation is intentional
            terminal = _terminal_from_failure(
                error,
                contract=contract,
                record=record,
                attempt=attempt,
            )
        _write_json_create_only(terminal_path, terminal)
        terminals.append(terminal)

    if len(terminals) != len(contract["records"]):
        raise RuntimeError("Stage-P terminal denominator did not close")
    batch = _build_batch_receipt(
        contract=contract,
        output_root=root,
        terminals=terminals,
    )
    batch_path = root / "batch_receipt.json"
    if batch_path.exists() or batch_path.is_symlink():
        observed = validate_stage_p_batch_receipt_v1(
            _read_json(batch_path, "Stage-P batch receipt"), contract=contract
        )
        if observed != batch:
            raise ValueError("Stage-P resumed batch receipt drifted")
    else:
        _write_json_create_only(batch_path, batch)
    return batch


__all__ = [
    "STAGE_P_ATTEMPT_SCHEMA_V1",
    "STAGE_P_BATCH_SCHEMA_V1",
    "STAGE_P_METHOD_ID_V1",
    "STAGE_P_RUN_CONTRACT_SCHEMA_V1",
    "STAGE_P_SERVICE_STATES",
    "STAGE_P_SUCCESS_OUTCOMES",
    "STAGE_P_TERMINAL_OUTCOMES",
    "STAGE_P_TERMINAL_SCHEMA_V1",
    "build_stage_p_run_contract_v1",
    "run_stage_p_prediction_batch_v1",
    "validate_stage_p_batch_receipt_v1",
    "validate_stage_p_run_contract_v1",
    "validate_stage_p_terminal_v1",
]
