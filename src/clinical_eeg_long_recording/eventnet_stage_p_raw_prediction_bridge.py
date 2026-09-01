"""Strict reference-free bridge from EventNet Stage-P to decoder-grid input.

Stage-P stores provider artifacts beneath the attempt that produced the
record-level terminal.  The older EventNet raw-bundle validator expects a
flat record directory, so it cannot safely consume Stage-P by path rewriting
alone.  This module validates the complete Stage-P denominator, resolves the
terminal's committed attempt, and returns the existing immutable raw-bundle
carrier with tensor paths pointing at the original NPZ files.

No API in this module accepts an annotation, seizure-reference, spreadsheet,
clinical-text, private-data, or source-evaluation path.  Prediction artifacts
must be frozen and complete before this bridge runs.  The bridge neither
copies nor rewrites prediction tensors and does not confer detection,
Findings, SOZ, clinical, or deployment authority.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from .continuous_detection_stage_p_runner_v1 import (
    STAGE_P_ATTEMPT_SCHEMA_V1,
    STAGE_P_METHOD_ID_V1,
    STAGE_P_SERVICE_STATES,
    STAGE_P_SUCCESS_OUTCOMES,
    validate_stage_p_batch_receipt_v1,
    validate_stage_p_run_contract_v1,
    validate_stage_p_terminal_v1,
)
from .detector_provider_contract import to_continuous_decoder_provider_receipt
from .eventnet_continuous_benchmark_projection import (
    validate_eventnet_benchmark_prediction_projection,
)
from .eventnet_full_record_adapter import (
    EVENTNET_PROVIDER_ID,
    EVENTNET_SAMPLING_RATE_HZ,
    validate_eventnet_prediction_receipt,
)
from .eventnet_raw_prediction_bundle import (
    EVENTNET_BENCHMARK_PROJECTION_FILENAME,
    EVENTNET_PREDICTION_RECEIPT_FILENAME,
    EVENTNET_RAW_BUNDLE_METHOD_ID,
    EVENTNET_RAW_BUNDLE_SCHEMA_VERSION,
    EVENTNET_TENSOR_FILENAME,
    ValidatedEventNetRawPrediction,
    ValidatedEventNetRawPredictionBundle,
    validate_eventnet_tensor_sidecar_bytes,
)


EVENTNET_STAGE_P_RAW_BRIDGE_SCHEMA_VERSION = (
    "eventnet_stage_p_raw_prediction_bridge_v1"
)
EVENTNET_STAGE_P_RAW_BRIDGE_METHOD_ID = (
    "eventnet_stage_p_committed_attempt_to_raw_bundle_reference_free_v1"
)
EVENTNET_STAGE_P_INPUT_RECEIPT_SCHEMA_VERSION = (
    "eventnet_stage_p_input_receipt_v1"
)

_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_ROOT_FILENAMES = {
    "batch_receipt.json",
    "records",
    "run_contract.json",
    "stage_p_input_receipt.json",
}
_INPUT_FIELDS = {
    "schema_version",
    "provider_id",
    "model_split",
    "selected_record_count",
    "source_identity_projection_sha256",
    "complete_roster_projection_receipt_sha256",
    "provider_execution_receipt",
    "checkpoint_sha256",
    "preprocessing_contract_sha256",
    "decoder_contract_sha256",
    "runtime_hardware_receipt",
    "run_contract_sha256",
    "reference_files_opened",
    "edf_annotations_read",
    "spreadsheet_or_clinical_text_read",
    "receipt_sha256",
}
_ATTEMPT_FIELDS = {
    "schema_version",
    "method_id",
    "attempt_id",
    "attempt_ordinal",
    "execution_session_id",
    "service_state",
    "run_contract_sha256",
    "recording_id",
    "record_receipt_sha256",
    "status",
    "attempt_receipt_sha256",
}
_ARTIFACT_FILENAMES_BY_SEMANTIC = {
    "eventnet_benchmark_prediction_projection": (
        EVENTNET_BENCHMARK_PROJECTION_FILENAME
    ),
    "eventnet_prediction_receipt": EVENTNET_PREDICTION_RECEIPT_FILENAME,
    "eventnet_prediction_tensors": EVENTNET_TENSOR_FILENAME,
}
_REFERENCE_ACCESS = {
    "reference_path_argument_accepted": False,
    "reference_files_opened": 0,
    "edf_annotations_opened": 0,
    "excel_files_opened": 0,
    "clinical_text_opened": 0,
    "private_data_opened": 0,
    "source_eval_opened": 0,
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


def _file_sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def eventnet_stage_p_raw_bridge_code_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _SHA256_CHARACTERS for character in value)
    )


def _sha256(value: object, context: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return str(value)


def _identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TypeError(f"{context} must be a non-empty trimmed string")
    if len(value) > 1024 or any(ord(character) < 32 for character in value):
        raise ValueError(f"{context} is invalid")
    return value


def _integer(value: object, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TypeError(f"{context} must be an integer >= {minimum}")
    return value


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _loads_json(payload: bytes | str, context: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_no_duplicate_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{context} is not valid UTF-8 JSON") from error
    if type(value) is not dict:
        raise TypeError(f"{context} must contain one JSON object")
    return value


def _regular_file(path: Path, context: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{context} must be a regular non-symlink file")
    return path


def _regular_directory(path: Path, context: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{context} must be a regular non-symlink directory")
    return path


def _resolved_regular_root(path: str | Path) -> Path:
    supplied = Path(path)
    _regular_directory(supplied, "EventNet Stage-P root")
    return supplied.resolve(strict=True)


def _strict_json_file(path: Path, context: str) -> tuple[dict[str, Any], bytes]:
    payload = _regular_file(path, context).read_bytes()
    if not payload:
        raise ValueError(f"{context} is empty")
    return _loads_json(payload, context), payload


def _normalized_expected_roster(values: Iterable[str]) -> list[str]:
    source = [
        _identifier(value, "expected source-dev recording ID") for value in values
    ]
    if not source:
        raise ValueError("expected source-dev recording roster is empty")
    if len(source) != len(set(source)):
        raise ValueError("expected source-dev recording roster repeats an identity")
    result = sorted(source)
    for recording_id in result:
        _patient_alias(recording_id)
    return result


def _patient_alias(recording_id: str) -> str:
    relative = PurePosixPath(recording_id)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or len(relative.parts) < 3
        or relative.parts[0] != "dev"
        or relative.suffix.lower() != ".edf"
    ):
        raise ValueError("EventNet Stage-P recording ID is unsafe or off source-dev")
    return _identifier(relative.parts[1], "source-dev patient alias")


def _validate_self_hash(
    value: Mapping[str, Any], *, field: str, placeholder: str, context: str
) -> str:
    observed = _sha256(value.get(field), f"{context} {field}")
    digest = deepcopy(dict(value))
    digest[field] = placeholder
    if observed != _canonical_sha256(digest):
        raise ValueError(f"{context} self-hash drifted")
    return observed


def _validate_stage_p_input_receipt(
    value: object, *, contract: Mapping[str, Any]
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _INPUT_FIELDS:
        raise ValueError("EventNet Stage-P input receipt fields drifted")
    data = deepcopy(value)
    if (
        data["schema_version"] != EVENTNET_STAGE_P_INPUT_RECEIPT_SCHEMA_VERSION
        or data["provider_id"] != contract["provider_id"]
        or data["model_split"] != contract["model_split"]
        or data["selected_record_count"] != contract["selected_record_count"]
        or data["source_identity_projection_sha256"]
        != contract["source_identity_projection_sha256"]
        or data["checkpoint_sha256"] != contract["checkpoint_sha256"]
        or data["preprocessing_contract_sha256"]
        != contract["preprocessing_contract_sha256"]
        or data["decoder_contract_sha256"]
        != contract["decoder_contract_sha256"]
        or data["run_contract_sha256"] != contract["contract_sha256"]
        or data["reference_files_opened"] != 0
        or data["edf_annotations_read"] is not False
        or data["spreadsheet_or_clinical_text_read"] is not False
    ):
        raise ValueError("EventNet Stage-P input permission or contract binding drifted")
    _integer(data["selected_record_count"], "Stage-P input selected count", minimum=1)
    for field in (
        "source_identity_projection_sha256",
        "checkpoint_sha256",
        "preprocessing_contract_sha256",
        "decoder_contract_sha256",
        "run_contract_sha256",
    ):
        _sha256(data[field], f"Stage-P input {field}")
    projection_receipt = data["complete_roster_projection_receipt_sha256"]
    if projection_receipt is not None:
        _sha256(projection_receipt, "Stage-P complete-roster projection")
    execution = deepcopy(data["provider_execution_receipt"])
    projected_execution = to_continuous_decoder_provider_receipt(execution)
    if (
        projected_execution["provider_id"] != EVENTNET_PROVIDER_ID
        or projected_execution["code_sha256"] != contract["provider_code_sha256"]
        or _canonical_sha256(execution)
        != contract["provider_execution_receipt_sha256"]
    ):
        raise ValueError("EventNet Stage-P provider execution binding drifted")
    runtime = data["runtime_hardware_receipt"]
    if type(runtime) is not dict or not _is_sha256(runtime.get("receipt_sha256")):
        raise ValueError("EventNet Stage-P runtime-hardware receipt is invalid")
    _validate_self_hash(
        runtime,
        field="receipt_sha256",
        placeholder="CONTENT-ADDRESS-PENDING",
        context="EventNet Stage-P runtime-hardware receipt",
    )
    if runtime["receipt_sha256"] != contract["runtime_hardware_contract_sha256"]:
        raise ValueError("EventNet Stage-P runtime-hardware binding drifted")
    _validate_self_hash(
        data,
        field="receipt_sha256",
        placeholder="CONTENT-ADDRESS-PENDING",
        context="EventNet Stage-P input receipt",
    )
    data["provider_execution_receipt"] = execution
    return data


def _validate_attempts(
    *,
    record_directory: Path,
    contract: Mapping[str, Any],
    record: Mapping[str, Any],
    terminal: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    attempts_directory = _regular_directory(
        record_directory / "attempts", "EventNet Stage-P attempts directory"
    )
    paths = sorted(attempts_directory.iterdir(), key=lambda path: path.name)
    if not paths:
        raise ValueError("EventNet Stage-P record has no attempts")
    attempts: list[dict[str, Any]] = []
    for expected_ordinal, path in enumerate(paths, start=1):
        expected_name = f"attempt-{expected_ordinal:06d}.json"
        if path.name != expected_name:
            raise ValueError("EventNet Stage-P attempt ordinals are not contiguous")
        data, _payload = _strict_json_file(path, "EventNet Stage-P attempt")
        if set(data) != _ATTEMPT_FIELDS:
            raise ValueError("EventNet Stage-P attempt fields drifted")
        expected_id = (
            f"STAGEP-{record['record_key']}-ATTEMPT-{expected_ordinal:06d}"
        )
        if (
            data["schema_version"] != STAGE_P_ATTEMPT_SCHEMA_V1
            or data["method_id"] != STAGE_P_METHOD_ID_V1
            or data["attempt_id"] != expected_id
            or data["attempt_ordinal"] != expected_ordinal
            or data["service_state"] not in STAGE_P_SERVICE_STATES
            or data["run_contract_sha256"] != contract["contract_sha256"]
            or data["recording_id"] != record["recording_id"]
            or data["record_receipt_sha256"] != record["record_receipt_sha256"]
            or data["status"]
            != "attempt_started_terminal_not_yet_published"
        ):
            raise ValueError("EventNet Stage-P attempt identity drifted")
        _identifier(data["execution_session_id"], "Stage-P execution session")
        _validate_self_hash(
            data,
            field="attempt_receipt_sha256",
            placeholder="CONTENT-ADDRESS-PENDING",
            context="EventNet Stage-P attempt",
        )
        attempts.append(data)
    committed = attempts[-1]
    if (
        terminal["attempt_id"] != committed["attempt_id"]
        or terminal["execution_session_id"] != committed["execution_session_id"]
        or terminal["service_state"] != committed["service_state"]
    ):
        raise ValueError(
            "EventNet Stage-P terminal does not bind the latest committed attempt"
        )
    return committed, attempts


def _validate_artifact_layout(
    *,
    record_directory: Path,
    terminal: Mapping[str, Any],
    attempts: list[dict[str, Any]],
) -> dict[str, tuple[Path, str]]:
    if len(terminal["artifacts"]) != len(_ARTIFACT_FILENAMES_BY_SEMANTIC):
        raise ValueError("EventNet Stage-P committed artifact count drifted")
    artifact_rows: dict[str, tuple[Path, str]] = {}
    attempt_id = terminal["attempt_id"]
    for row in terminal["artifacts"]:
        semantic = row["semantic"]
        if semantic not in _ARTIFACT_FILENAMES_BY_SEMANTIC or semantic in artifact_rows:
            raise ValueError("EventNet Stage-P committed artifact semantics drifted")
        filename = _ARTIFACT_FILENAMES_BY_SEMANTIC[semantic]
        expected_relative = f"artifacts/{attempt_id}/{filename}"
        if row["relative_path"] != expected_relative:
            raise ValueError("EventNet Stage-P committed artifact path drifted")
        expected_hash = _sha256(row["file_sha256"], "Stage-P artifact file")
        path = _regular_file(
            record_directory / "artifacts" / attempt_id / filename,
            f"EventNet Stage-P {semantic}",
        )
        if _file_sha256_bytes(path.read_bytes()) != expected_hash:
            raise ValueError("EventNet Stage-P committed artifact hash drifted")
        artifact_rows[semantic] = (path, expected_hash)
    if set(artifact_rows) != set(_ARTIFACT_FILENAMES_BY_SEMANTIC):
        raise ValueError("EventNet Stage-P committed artifact roster is incomplete")

    artifacts_root = _regular_directory(
        record_directory / "artifacts", "EventNet Stage-P artifacts root"
    )
    known_attempt_ids = {attempt["attempt_id"] for attempt in attempts}
    observed_attempt_ids: set[str] = set()
    for path in artifacts_root.iterdir():
        _regular_directory(path, "EventNet Stage-P attempt-artifact directory")
        if path.name not in known_attempt_ids:
            raise ValueError("EventNet Stage-P artifacts contain an unknown attempt")
        observed_attempt_ids.add(path.name)
    if attempt_id not in observed_attempt_ids:
        raise ValueError("EventNet Stage-P committed attempt artifacts are absent")
    committed_directory = artifacts_root / attempt_id
    if {path.name for path in committed_directory.iterdir()} != set(
        _ARTIFACT_FILENAMES_BY_SEMANTIC.values()
    ):
        raise ValueError("EventNet Stage-P committed artifact directory drifted")
    return artifact_rows


def _validate_record_artifacts(
    *,
    record_directory: Path,
    contract: Mapping[str, Any],
    record: Mapping[str, Any],
    terminal: Mapping[str, Any],
    execution_receipt: Mapping[str, Any],
) -> tuple[ValidatedEventNetRawPrediction, dict[str, Any], dict[str, Any]]:
    committed, attempts = _validate_attempts(
        record_directory=record_directory,
        contract=contract,
        record=record,
        terminal=terminal,
    )
    artifacts = _validate_artifact_layout(
        record_directory=record_directory,
        terminal=terminal,
        attempts=attempts,
    )
    prediction_path, prediction_file_sha256 = artifacts[
        "eventnet_prediction_receipt"
    ]
    prediction_payload = prediction_path.read_bytes()
    if _file_sha256_bytes(prediction_payload) != prediction_file_sha256:
        raise ValueError("EventNet Stage-P prediction receipt changed during bridge")
    prediction = validate_eventnet_prediction_receipt(
        _loads_json(prediction_payload, "EventNet Stage-P prediction receipt")
    )
    generic = prediction["generic_full_record_result"]
    coverage = generic["coverage_receipt"]
    runtime = prediction["runtime_receipt"]
    if (
        prediction["provider_id"] != EVENTNET_PROVIDER_ID
        or prediction["recording_id"] != record["recording_id"]
        or prediction["provider_execution_receipt"] != execution_receipt
        or prediction["checkpoint_receipt"]["sha256"]
        != contract["checkpoint_sha256"]
        or prediction["provider_execution_receipt"]["code_sha256"]
        != contract["provider_code_sha256"]
        or prediction["receipt_sha256"]
        != terminal["provider_prediction_receipt_sha256"]
        or _canonical_sha256(generic)
        != terminal["provider_result_receipt_sha256"]
        or generic["outcome_status"] != terminal["outcome_status"]
        or generic["recording_duration_seconds"]
        != terminal["recording_duration_seconds"]
        or prediction["decoder_receipt"]["event_proposal_count"]
        != terminal["event_proposal_count"]
        or coverage["modeled_target_coverage_seconds"]
        != terminal["modeled_target_coverage_seconds"]
        or coverage["posterior_target_coverage_complete"] is not True
        or runtime["recording_duration_seconds"]
        != terminal["recording_duration_seconds"]
        or runtime["service_state"] != terminal["service_state"]
    ):
        raise ValueError("EventNet Stage-P prediction/terminal binding drifted")

    projection_path, projection_file_sha256 = artifacts[
        "eventnet_benchmark_prediction_projection"
    ]
    projection_payload = projection_path.read_bytes()
    if _file_sha256_bytes(projection_payload) != projection_file_sha256:
        raise ValueError("EventNet Stage-P benchmark projection changed during bridge")
    projection = validate_eventnet_benchmark_prediction_projection(
        _loads_json(projection_payload, "EventNet Stage-P benchmark projection")
    )
    projection_execution = projection["execution_receipt"]
    expected_service_state = (
        "cold"
        if terminal["service_state"] == "cold_process_start"
        else "warm"
    )
    timing = terminal["timing_seconds"]
    if (
        projection["provider_id"] != EVENTNET_PROVIDER_ID
        or projection["recording_id"] != record["recording_id"]
        or projection["prediction_id"] != prediction["prediction_id"]
        or projection["prediction_receipt_sha256"]
        != prediction["receipt_sha256"]
        or projection["duration_seconds"] != terminal["recording_duration_seconds"]
        or projection_execution["complete_recording_coverage"] is not True
        or projection_execution["service_state"] != expected_service_state
        or projection_execution["edf_io_seconds"] != timing["edf_io_seconds"]
        or projection_execution["preprocessing_seconds"]
        != timing["preprocessing_seconds"]
        or projection_execution["inference_seconds"] != timing["inference_seconds"]
        or projection_execution["postprocessing_seconds"] != timing["decoder_seconds"]
        or projection_execution["total_wall_seconds"]
        != timing["end_to_end_seconds"]
    ):
        raise ValueError("EventNet Stage-P projection/terminal binding drifted")

    tensor_path, tensor_file_sha256 = artifacts["eventnet_prediction_tensors"]
    tensor_payload = tensor_path.read_bytes()
    arrays, tensor_replay = validate_eventnet_tensor_sidecar_bytes(
        tensor_payload,
        expected_file_sha256=tensor_file_sha256,
        prediction_receipt=prediction,
    )
    duration = float(terminal["recording_duration_seconds"])
    sample_count = int(arrays["center_probability"].size)
    source_signal_sha256 = prediction["canonical_detector_input_binding"][
        "canonical_source_signal_sha256"
    ]
    patient_alias = _patient_alias(record["recording_id"])
    sealed = ValidatedEventNetRawPrediction(
        recording_id=record["recording_id"],
        patient_alias=patient_alias,
        record_key=record["record_key"],
        prediction_id=prediction["prediction_id"],
        prediction_receipt_sha256=prediction["receipt_sha256"],
        source_signal_sha256=source_signal_sha256,
        recording_duration_seconds=duration,
        sample_count=sample_count,
        prediction_receipt_json=_canonical_json(prediction),
        tensor_path=str(tensor_path.resolve(strict=True)),
        tensor_file_sha256=tensor_file_sha256,
        tensor_replay_receipt_json=_canonical_json(tensor_replay),
    )
    inventory = {
        "recording_id": record["recording_id"],
        "patient_alias": patient_alias,
        "record_key": record["record_key"],
        "prediction_id": prediction["prediction_id"],
        "prediction_receipt_sha256": prediction["receipt_sha256"],
        "source_signal_sha256": source_signal_sha256,
        "tensor_file_sha256": tensor_file_sha256,
        "tensor_replay_receipt_sha256": _canonical_sha256(tensor_replay),
        "sample_count": sample_count,
        "recording_duration_seconds": duration,
    }
    lineage = {
        "recording_id": record["recording_id"],
        "record_key": record["record_key"],
        "terminal_receipt_sha256": terminal["terminal_receipt_sha256"],
        "attempt_id": committed["attempt_id"],
        "attempt_ordinal": committed["attempt_ordinal"],
        "attempt_receipt_sha256": committed["attempt_receipt_sha256"],
        "attempt_count": len(attempts),
        "prediction_receipt_file_sha256": prediction_file_sha256,
        "prediction_tensor_file_sha256": tensor_file_sha256,
        "benchmark_projection_id": projection["projection_id"],
        "benchmark_projection_receipt_sha256": projection["receipt_sha256"],
        "benchmark_projection_file_sha256": projection_file_sha256,
    }
    return sealed, inventory, lineage


def validate_eventnet_stage_p_raw_prediction_bundle_without_references(
    stage_p_root: str | Path,
    *,
    expected_recording_ids: Iterable[str],
    expected_run_contract_sha256: str | None = None,
) -> ValidatedEventNetRawPredictionBundle:
    """Close a complete source-dev Stage-P inventory without reference access."""

    root = _resolved_regular_root(stage_p_root)
    if {path.name for path in root.iterdir()} != _ROOT_FILENAMES:
        raise ValueError("EventNet Stage-P root has missing or unknown entries")
    contract_payload, contract_bytes = _strict_json_file(
        root / "run_contract.json", "EventNet Stage-P run contract"
    )
    contract = validate_stage_p_run_contract_v1(contract_payload)
    contract_id_source = deepcopy(contract)
    contract_id_source["run_id"] = "STAGE-P-RUN-PENDING"
    contract_id_source["contract_sha256"] = "CONTENT-ADDRESS-PENDING"
    if contract["run_id"] != "STAGEP-" + _canonical_sha256(contract_id_source)[:24]:
        raise ValueError("EventNet Stage-P run ID is not content-bound")
    if (
        contract["provider_id"] != EVENTNET_PROVIDER_ID
        or contract["model_split"] != "source_dev"
    ):
        raise ValueError("EventNet Stage-P bridge accepts source-dev EventNet only")
    if expected_run_contract_sha256 is not None:
        expected_contract = _sha256(
            expected_run_contract_sha256, "expected Stage-P run contract"
        )
        if contract["contract_sha256"] != expected_contract:
            raise ValueError("EventNet Stage-P run contract disagrees with expectation")

    expected = _normalized_expected_roster(expected_recording_ids)
    observed = [record["recording_id"] for record in contract["records"]]
    if observed != expected:
        missing = sorted(set(expected).difference(observed))
        extra = sorted(set(observed).difference(expected))
        raise ValueError(
            "EventNet Stage-P source-dev inventory mismatch: "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )

    input_payload, input_bytes = _strict_json_file(
        root / "stage_p_input_receipt.json", "EventNet Stage-P input receipt"
    )
    input_receipt = _validate_stage_p_input_receipt(
        input_payload, contract=contract
    )
    execution_receipt = input_receipt["provider_execution_receipt"]
    batch_payload, batch_bytes = _strict_json_file(
        root / "batch_receipt.json", "EventNet Stage-P batch receipt"
    )
    batch = validate_stage_p_batch_receipt_v1(batch_payload, contract=contract)
    batch_id_source = deepcopy(batch)
    batch_id_source["batch_id"] = "STAGE-P-BATCH-PENDING"
    batch_id_source["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if batch["batch_id"] != "STAGEPBATCH-" + _canonical_sha256(batch_id_source)[:24]:
        raise ValueError("EventNet Stage-P batch ID is not content-bound")
    if (
        batch["outcome_counts"]["technical_failure"] != 0
        or batch["outcome_counts"]["partial_coverage"] != 0
        or sum(
            batch["outcome_counts"][outcome]
            for outcome in STAGE_P_SUCCESS_OUTCOMES
            if outcome != "partial_coverage"
        )
        != contract["selected_record_count"]
    ):
        raise ValueError(
            "EventNet Stage-P raw bridge requires a complete successful denominator"
        )
    expected_seconds = sum(
        record["recording_duration_fraction"][0]
        / record["recording_duration_fraction"][1]
        for record in contract["records"]
    )
    if not math.isclose(
        float(batch["expected_recording_seconds"]),
        expected_seconds,
        rel_tol=1e-12,
        abs_tol=1e-9,
    ):
        raise ValueError("EventNet Stage-P batch duration denominator drifted")

    records_root = _regular_directory(
        root / "records", "EventNet Stage-P records root"
    )
    expected_record_keys = {record["record_key"] for record in contract["records"]}
    if {path.name for path in records_root.iterdir()} != expected_record_keys:
        raise ValueError("EventNet Stage-P records root disagrees with run contract")
    batch_inventory = batch["terminal_inventory"]
    sealed: list[ValidatedEventNetRawPrediction] = []
    record_inventory: list[dict[str, Any]] = []
    stage_lineage_inventory: list[dict[str, Any]] = []
    total_duration = 0.0
    total_samples = 0
    total_modeled = 0.0
    for index, record in enumerate(contract["records"]):
        record_directory = _regular_directory(
            records_root / record["record_key"],
            "EventNet Stage-P record directory",
        )
        if {path.name for path in record_directory.iterdir()} != {
            "artifacts",
            "attempts",
            "terminal.json",
        }:
            raise ValueError("EventNet Stage-P record directory layout drifted")
        terminal_payload, terminal_bytes = _strict_json_file(
            record_directory / "terminal.json", "EventNet Stage-P terminal"
        )
        terminal = validate_stage_p_terminal_v1(
            terminal_payload,
            contract=contract,
            record=record,
            record_directory=record_directory,
        )
        indexed = batch_inventory[index]
        terminal_file_sha256 = _file_sha256_bytes(terminal_bytes)
        if (
            indexed["recording_id"] != record["recording_id"]
            or indexed["record_key"] != record["record_key"]
            or indexed["outcome_status"] != terminal["outcome_status"]
            or indexed["service_state"] != terminal["service_state"]
            or indexed["terminal_receipt_sha256"]
            != terminal["terminal_receipt_sha256"]
            or indexed["terminal_file_sha256"] != terminal_file_sha256
        ):
            raise ValueError("EventNet Stage-P batch/terminal inventory drifted")
        if terminal["outcome_status"] not in {
            "completed_with_alarms",
            "completed_zero_alarm",
        }:
            raise ValueError("EventNet Stage-P terminal is not fully modeled")
        prediction, inventory, lineage = _validate_record_artifacts(
            record_directory=record_directory,
            contract=contract,
            record=record,
            terminal=terminal,
            execution_receipt=execution_receipt,
        )
        lineage["terminal_file_sha256"] = terminal_file_sha256
        sealed.append(prediction)
        record_inventory.append(inventory)
        stage_lineage_inventory.append(lineage)
        total_duration += prediction.recording_duration_seconds
        total_samples += prediction.sample_count
        total_modeled += float(terminal["modeled_target_coverage_seconds"])

    if not math.isclose(total_duration, expected_seconds, rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError("EventNet Stage-P bridged recording duration does not close")
    if not math.isclose(
        total_modeled,
        float(batch["modeled_target_coverage_seconds"]),
        rel_tol=1e-12,
        abs_tol=1e-9,
    ) or not math.isclose(total_modeled, total_duration, rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError("EventNet Stage-P modeled coverage does not close")

    patient_aliases = sorted({row["patient_alias"] for row in record_inventory})
    stage_lineage: dict[str, Any] = {
        "schema_version": EVENTNET_STAGE_P_RAW_BRIDGE_SCHEMA_VERSION,
        "method_id": EVENTNET_STAGE_P_RAW_BRIDGE_METHOD_ID,
        "run_id": contract["run_id"],
        "run_contract_sha256": contract["contract_sha256"],
        "run_contract_file_sha256": _file_sha256_bytes(contract_bytes),
        "stage_p_input_receipt_sha256": input_receipt["receipt_sha256"],
        "stage_p_input_receipt_file_sha256": _file_sha256_bytes(input_bytes),
        "upstream_complete_projection_receipt_sha256": input_receipt[
            "complete_roster_projection_receipt_sha256"
        ],
        "stage_p_batch_id": batch["batch_id"],
        "stage_p_batch_receipt_sha256": batch["receipt_sha256"],
        "stage_p_batch_receipt_file_sha256": _file_sha256_bytes(batch_bytes),
        "record_roster_sha256": contract["record_roster_sha256"],
        "selected_record_count": contract["selected_record_count"],
        "committed_attempt_inventory": stage_lineage_inventory,
        "committed_attempt_inventory_sha256": _canonical_sha256(
            stage_lineage_inventory
        ),
        "tensor_materialization_policy": "original_stage_p_npz_path_no_copy_no_rewrite",
        "reference_or_annotation_input_used": False,
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    stage_lineage["receipt_sha256"] = _canonical_sha256(stage_lineage)
    receipt: dict[str, Any] = {
        "schema_version": EVENTNET_RAW_BUNDLE_SCHEMA_VERSION,
        "validation_id": "EVENTNET-RAW-BUNDLE-VALIDATION-PENDING",
        "method_id": EVENTNET_RAW_BUNDLE_METHOD_ID,
        "provider_id": EVENTNET_PROVIDER_ID,
        "validator_code_sha256": eventnet_stage_p_raw_bridge_code_sha256(),
        "provider_adapter_code_sha256": contract["provider_code_sha256"],
        "weights_manifest_sha256": execution_receipt["checkpoint_sha256"],
        "source_split": "source_dev",
        "batch_root": str(root),
        "batch_id": batch["batch_id"],
        "batch_receipt_sha256": batch["receipt_sha256"],
        "batch_receipt_file_sha256": _file_sha256_bytes(batch_bytes),
        "expected_recording_roster_sha256": _canonical_sha256(expected),
        "observed_recording_roster_sha256": _canonical_sha256(observed),
        "patient_alias_roster_sha256": _canonical_sha256(patient_aliases),
        "record_count": len(sealed),
        "patient_alias_count": len(patient_aliases),
        "recording_duration_seconds": total_duration,
        "provider_sample_count": total_samples,
        "provider_sampling_rate_hz": EVENTNET_SAMPLING_RATE_HZ,
        "record_inventory": record_inventory,
        "record_inventory_sha256": _canonical_sha256(record_inventory),
        "all_npz_files_loaded_with_allow_pickle_false": True,
        "all_npz_members_exact_float32_finite_unit_interval": True,
        "all_tensor_payload_hashes_replayed": True,
        "released_per_tile_smoothing_replayed_exactly": True,
        "complete_target_sample_coverage": True,
        "complete_expected_source_dev_inventory_verified": True,
        "reference_access": deepcopy(_REFERENCE_ACCESS),
        "scope_receipt": {
            "eeg_signal_and_acquisition_metadata_only": True,
            "raw_predictions_frozen_before_reference_join": True,
            "source_dev_only": True,
            "source_eval_opened": False,
            "private_data_opened": False,
            "edf_annotations_opened": False,
            "excel_or_clinical_text_opened": False,
            "prediction_is_confirmed_seizure_or_clinical_onset": False,
            "findings_or_soz_evidence_authorized": False,
            "production_or_sota_claim_authorized": False,
        },
        "stage_p_lineage": stage_lineage,
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    receipt["validation_id"] = "EVNRAWVAL-" + _canonical_sha256(receipt)[:24]
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    return ValidatedEventNetRawPredictionBundle(
        batch_root=str(root),
        recordings=tuple(sealed),
        validation_receipt_json=_canonical_json(receipt),
    )


__all__ = [
    "EVENTNET_STAGE_P_RAW_BRIDGE_METHOD_ID",
    "EVENTNET_STAGE_P_RAW_BRIDGE_SCHEMA_VERSION",
    "eventnet_stage_p_raw_bridge_code_sha256",
    "validate_eventnet_stage_p_raw_prediction_bundle_without_references",
]
