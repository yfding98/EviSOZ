"""Reference-free closure validation for frozen EventNet prediction bundles.

The EventNet materializer writes one JSON prediction receipt and one NPZ
sidecar per complete EEG recording.  A JSON hash alone does not establish that
the numeric sidecar is safe to load or that it still has the semantics claimed
by the receipt.  This module closes that gap *before* a caller may construct or
open a source-development seizure-reference path.

The public validator deliberately has no annotation, reference, spreadsheet,
clinical-text, private-data or source-evaluation path argument.  It validates
the exact complete ``source_dev`` prediction inventory, loads NPZ bytes with
``allow_pickle=False``, checks every numeric member against the receipt, and
replays EventNet's released per-tile Gaussian smoothing.  The returned carrier
is immutable; every later tensor load rechecks the frozen file hash and tensor
payload receipts.

This is a research evidence boundary.  It does not turn EventNet proposals
into confirmed seizures, clinical onsets, Findings, SOZ evidence, or a
production-qualified detector.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .detector_provider_contract import to_continuous_decoder_provider_receipt
from .eventnet_continuous_benchmark_projection import (
    validate_eventnet_benchmark_prediction_projection,
)
from .eventnet_full_record_adapter import (
    EVENTNET_CENTER_SMOOTHING_SIGMA_SAMPLES,
    EVENTNET_PROVIDER_ID,
    EVENTNET_SAMPLING_RATE_HZ,
    validate_eventnet_prediction_receipt,
)


EVENTNET_RAW_BUNDLE_SCHEMA_VERSION = "eventnet_raw_prediction_bundle_v1"
EVENTNET_RAW_BUNDLE_METHOD_ID = (
    "eventnet_npz_prediction_receipt_target_grid_smoothing_closure_v1"
)
EVENTNET_MATERIALIZED_BATCH_SCHEMA_VERSION = (
    "eventnet_public_eeg_only_prediction_batch_v1"
)
EVENTNET_TENSOR_FILENAME = "prediction_tensors.npz"
EVENTNET_PREDICTION_RECEIPT_FILENAME = "prediction_receipt.json"
EVENTNET_BENCHMARK_PROJECTION_FILENAME = "benchmark_prediction_projection.json"

_TENSOR_KEYS = (
    "center_probability",
    "duration_fraction",
    "smoothed_center_probability",
)
_TENSOR_SEMANTICS = {
    "center_probability": "event_center_probability",
    "duration_fraction": "event_duration_fraction_of_300_seconds",
    "smoothed_center_probability": ("event_center_probability_gaussian_smoothed"),
}
_BATCH_FIELDS = {
    "schema_version",
    "batch_id",
    "provider_id",
    "model_split",
    "source_eval_opened",
    "provider_execution_receipt",
    "manifest_projection",
    "selected_record_count",
    "completed_record_count",
    "technical_failure_count",
    "recording_duration_seconds",
    "batch_wall_seconds",
    "batch_real_time_factor",
    "inventory",
    "scientific_status",
    "scope_receipt",
    "receipt_sha256",
}
_INVENTORY_FIELDS = {
    "recording_id",
    "record_key",
    "model_split",
    "prediction_id",
    "prediction_receipt_sha256",
    "prediction_receipt_file_sha256",
    "prediction_tensor_file_sha256",
    "benchmark_projection_id",
    "benchmark_projection_receipt_sha256",
    "benchmark_projection_file_sha256",
    "recording_duration_seconds",
    "canonical_root_materialization_seconds",
    "adapter_wall_seconds",
    "outcome_status",
    "event_proposal_count",
}
_BATCH_SCOPE = {
    "eeg_samples_used": True,
    "edf_signal_header_used": True,
    "edf_annotations_used": False,
    "spreadsheet_used": False,
    "doctor_labels_used": False,
    "clinical_text_used": False,
    "identity_fields_used_for_model_inference": False,
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
_SHA256_CHARACTERS = frozenset("0123456789abcdef")


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


def eventnet_raw_bundle_validator_code_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


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


def _finite(value: object, context: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{context} must be finite and >= {minimum}")
    return result


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


def _loads_json(payload: bytes | str, context: str) -> Any:
    try:
        return json.loads(
            payload,
            object_pairs_hook=_no_duplicate_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{context} is not valid UTF-8 JSON") from error


def _regular_file(path: Path, context: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{context} must be a regular non-symlink file")
    return path


def _regular_directory(path: Path, context: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{context} must be a regular non-symlink directory")
    return path


def _normalized_roster(values: Iterable[str], context: str) -> list[str]:
    result = sorted({_identifier(value, context) for value in values})
    if not result:
        raise ValueError(f"{context} roster is empty")
    return result


def _patient_alias_from_recording_id(recording_id: str) -> str:
    relative = PurePosixPath(recording_id)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or len(relative.parts) < 3
        or relative.parts[0] != "dev"
        or relative.suffix.lower() != ".edf"
    ):
        raise ValueError("EventNet source-dev recording ID is unsafe or off-split")
    return _identifier(relative.parts[1], "source-dev path patient alias")


def _payload_receipt(array: np.ndarray, semantic: str) -> dict[str, Any]:
    carrier = np.ascontiguousarray(array, dtype="<f4")
    return {
        "semantic": semantic,
        "dtype": "float32_little_endian",
        "shape": list(carrier.shape),
        "payload_sha256": hashlib.sha256(carrier.tobytes(order="C")).hexdigest(),
        "minimum": float(carrier.min(initial=np.inf)),
        "maximum": float(carrier.max(initial=-np.inf)),
    }


def _validate_tensor_receipt(
    array: np.ndarray,
    receipt: object,
    *,
    key: str,
) -> dict[str, Any]:
    if type(receipt) is not dict:
        raise TypeError(f"EventNet {key} tensor receipt must be an object")
    expected = _payload_receipt(array, _TENSOR_SEMANTICS[key])
    if receipt != expected:
        raise ValueError(f"EventNet {key} tensor payload disagrees with receipt")
    return expected


def _validate_target_tiles(
    value: object,
    *,
    sample_count: int,
    recording_duration_seconds: float,
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("EventNet raw bundle has no target tiles")
    tiles: list[dict[str, Any]] = []
    cursor = 0
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise TypeError(f"EventNet tile {index} must be an object")
        tile = deepcopy(dict(raw))
        start = _integer(tile.get("target_start_sample"), "tile target start")
        stop = _integer(
            tile.get("target_stop_sample_exclusive"), "tile target stop", minimum=1
        )
        if start != cursor or stop <= start or stop > sample_count:
            raise ValueError(
                "EventNet target tiles are not contiguous complete coverage"
            )
        if tile.get("tile_id") != f"EVNTILE-{index:06d}":
            raise ValueError("EventNet target tile identity drifted")
        start_seconds = _finite(
            tile.get("target_start_offset_seconds"), "tile target start seconds"
        )
        stop_seconds = _finite(
            tile.get("target_stop_offset_seconds"), "tile target stop seconds"
        )
        if (
            abs(start_seconds - start / EVENTNET_SAMPLING_RATE_HZ) > 1e-12
            or abs(
                stop_seconds
                - min(recording_duration_seconds, stop / EVENTNET_SAMPLING_RATE_HZ)
            )
            > 1e-12
        ):
            raise ValueError("EventNet target tile physical clock drifted")
        tiles.append(tile)
        cursor = stop
    if cursor != sample_count:
        raise ValueError("EventNet target tiles do not close the sample timeline")
    return tiles


def validate_eventnet_tensor_sidecar_bytes(
    payload: bytes,
    *,
    expected_file_sha256: str,
    prediction_receipt: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Validate and snapshot one NPZ sidecar without executing pickle code."""

    if not isinstance(payload, bytes) or not payload:
        raise TypeError("EventNet NPZ sidecar must be non-empty bytes")
    if not _is_sha256(expected_file_sha256):
        raise ValueError("EventNet expected NPZ SHA-256 is invalid")
    file_sha256 = _file_sha256_bytes(payload)
    if file_sha256 != expected_file_sha256:
        raise ValueError("EventNet NPZ file SHA-256 drifted")
    prediction = validate_eventnet_prediction_receipt(dict(prediction_receipt))
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as carrier:
            if set(carrier.files) != set(_TENSOR_KEYS):
                raise ValueError("EventNet NPZ has missing or unknown numeric members")
            arrays = {key: np.array(carrier[key], copy=True) for key in _TENSOR_KEYS}
    except (OSError, EOFError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith("EventNet NPZ"):
            raise
        raise ValueError("EventNet NPZ cannot be safely decoded") from error

    shapes = {array.shape for array in arrays.values()}
    if len(shapes) != 1:
        raise ValueError("EventNet NPZ numeric members do not share one clock")
    for key, array in arrays.items():
        if array.dtype != np.dtype("<f4") or array.ndim != 1 or array.size < 1:
            raise ValueError(
                f"EventNet {key} must be one-dimensional little-endian float32"
            )
        if not np.isfinite(array).all() or np.any(array < 0) or np.any(array > 1):
            raise ValueError(f"EventNet {key} is not a finite unit-interval tensor")
        _validate_tensor_receipt(
            array,
            prediction["output_tensors"][key],
            key=key,
        )

    sample_count = int(arrays["center_probability"].size)
    preprocessing = prediction["preprocessing_receipt"]
    if (
        preprocessing.get("target_sampling_rate_hz") != EVENTNET_SAMPLING_RATE_HZ
        or preprocessing.get("provider_sample_count") != sample_count
    ):
        raise ValueError("EventNet NPZ clock disagrees with preprocessing receipt")
    duration = _finite(
        prediction["runtime_receipt"]["recording_duration_seconds"],
        "EventNet recording duration",
        minimum=1e-12,
    )
    if sample_count != int(round(duration * EVENTNET_SAMPLING_RATE_HZ)):
        raise ValueError("EventNet NPZ sample count disagrees with recording duration")
    tiles = _validate_target_tiles(
        prediction["tile_receipts"],
        sample_count=sample_count,
        recording_duration_seconds=duration,
    )

    replayed = np.empty(sample_count, dtype=np.float32)
    center = arrays["center_probability"]
    for tile in tiles:
        start = int(tile["target_start_sample"])
        stop = int(tile["target_stop_sample_exclusive"])
        replayed[start:stop] = gaussian_filter1d(
            center[start:stop],
            EVENTNET_CENTER_SMOOTHING_SIGMA_SAMPLES,
        ).astype(np.float32, copy=False)
    smoothed = arrays["smoothed_center_probability"]
    if not np.array_equal(replayed, smoothed):
        maximum_error = float(np.max(np.abs(replayed - smoothed)))
        raise ValueError(
            "EventNet released per-tile Gaussian smoothing replay failed: "
            f"max_abs_error={maximum_error}"
        )
    decoder_tensor = prediction["decoder_receipt"].get("smoothed_center_tensor")
    _validate_tensor_receipt(
        smoothed,
        decoder_tensor,
        key="smoothed_center_probability",
    )
    full = prediction["generic_full_record_result"]
    if full["coverage_receipt"]["posterior_target_coverage_complete"] is not True:
        raise ValueError("EventNet raw sidecar lacks complete target coverage")

    replay_receipt = {
        "npz_file_sha256": file_sha256,
        "np_load_allow_pickle": False,
        "exact_numeric_member_names": list(_TENSOR_KEYS),
        "dtype": "float32_little_endian",
        "sample_count": sample_count,
        "provider_sampling_rate_hz": EVENTNET_SAMPLING_RATE_HZ,
        "recording_duration_seconds": duration,
        "tensor_payload_receipts": {
            key: _payload_receipt(arrays[key], _TENSOR_SEMANTICS[key])
            for key in _TENSOR_KEYS
        },
        "per_tile_released_smoothing_replayed": True,
        "per_tile_released_smoothing_exact_sample_equality": True,
        "complete_target_sample_coverage": True,
        "reference_or_annotation_input_used": False,
    }
    return arrays, replay_receipt


def _load_validated_tensor_file(
    path: Path,
    *,
    expected_file_sha256: str,
    prediction_receipt: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    file_path = _regular_file(path, "EventNet NPZ sidecar")
    return validate_eventnet_tensor_sidecar_bytes(
        file_path.read_bytes(),
        expected_file_sha256=expected_file_sha256,
        prediction_receipt=prediction_receipt,
    )


@dataclass(frozen=True, slots=True)
class ValidatedEventNetRawPrediction:
    recording_id: str
    patient_alias: str
    record_key: str
    prediction_id: str
    prediction_receipt_sha256: str
    source_signal_sha256: str
    recording_duration_seconds: float
    sample_count: int
    prediction_receipt_json: str
    tensor_path: str
    tensor_file_sha256: str
    tensor_replay_receipt_json: str

    def prediction_receipt(self) -> dict[str, Any]:
        value = _loads_json(self.prediction_receipt_json, "sealed prediction receipt")
        return validate_eventnet_prediction_receipt(value)

    def tensor_replay_receipt(self) -> dict[str, Any]:
        value = _loads_json(
            self.tensor_replay_receipt_json, "sealed tensor replay receipt"
        )
        if type(value) is not dict:
            raise RuntimeError("sealed EventNet tensor replay receipt is corrupted")
        return value

    def load_tensors(self) -> dict[str, np.ndarray]:
        arrays, replay = _load_validated_tensor_file(
            Path(self.tensor_path),
            expected_file_sha256=self.tensor_file_sha256,
            prediction_receipt=self.prediction_receipt(),
        )
        if replay != self.tensor_replay_receipt():
            raise ValueError("EventNet NPZ replay receipt changed after bundle freeze")
        return arrays


@dataclass(frozen=True, slots=True)
class ValidatedEventNetRawPredictionBundle:
    batch_root: str
    recordings: tuple[ValidatedEventNetRawPrediction, ...]
    validation_receipt_json: str

    def validation_receipt(self) -> dict[str, Any]:
        value = _loads_json(
            self.validation_receipt_json, "sealed raw-bundle validation receipt"
        )
        if type(value) is not dict:
            raise RuntimeError("sealed EventNet raw-bundle receipt is corrupted")
        return value


def _validate_batch_receipt(payload: object) -> dict[str, Any]:
    if type(payload) is not dict or set(payload) != _BATCH_FIELDS:
        raise ValueError("EventNet materialized batch fields drifted")
    data = deepcopy(payload)
    if (
        data["schema_version"] != EVENTNET_MATERIALIZED_BATCH_SCHEMA_VERSION
        or data["provider_id"] != EVENTNET_PROVIDER_ID
        or data["model_split"] != "source_dev"
        or data["source_eval_opened"] is not False
        or data["manifest_projection"] != ["model_split", "local_edf_path"]
        or data["scope_receipt"] != _BATCH_SCOPE
    ):
        raise ValueError(
            "EventNet raw bundle split, projection, or EEG-only scope drifted"
        )
    execution = to_continuous_decoder_provider_receipt(
        data["provider_execution_receipt"]
    )
    if execution["provider_id"] != EVENTNET_PROVIDER_ID:
        raise ValueError("EventNet batch provider execution receipt drifted")
    selected = _integer(data["selected_record_count"], "selected record count")
    completed = _integer(data["completed_record_count"], "completed record count")
    failures = _integer(data["technical_failure_count"], "technical failure count")
    if selected < 1 or completed != selected or failures != 0:
        raise ValueError("EventNet raw bundle is not a complete successful inventory")
    _finite(data["recording_duration_seconds"], "batch duration", minimum=1e-12)
    wall = _finite(data["batch_wall_seconds"], "batch wall time")
    rtf = _finite(data["batch_real_time_factor"], "batch RTF")
    if abs(rtf - wall / float(data["recording_duration_seconds"])) > 1e-12:
        raise ValueError("EventNet batch RTF is not replayable")
    if not isinstance(data["inventory"], list) or len(data["inventory"]) != selected:
        raise ValueError("EventNet batch inventory count drifted")
    digest = deepcopy(data)
    digest["batch_id"] = "EVENTNET-BATCH-PENDING"
    digest["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["batch_id"] != "EVNBATCH-" + _canonical_sha256(digest)[:24]:
        raise ValueError("EventNet batch ID is not content-bound")
    digest = deepcopy(data)
    digest["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["receipt_sha256"] != _canonical_sha256(digest):
        raise ValueError("EventNet batch receipt SHA-256 drifted")
    return data


def validate_eventnet_raw_prediction_bundle_without_references(
    batch_root: str | Path,
    *,
    expected_recording_ids: Iterable[str],
) -> ValidatedEventNetRawPredictionBundle:
    """Validate a complete frozen source-dev EventNet batch before references."""

    root = _regular_directory(
        Path(batch_root).resolve(strict=True), "EventNet batch root"
    )
    batch_path = _regular_file(root / "batch_receipt.json", "EventNet batch receipt")
    batch_bytes = batch_path.read_bytes()
    batch = _validate_batch_receipt(_loads_json(batch_bytes, "EventNet batch receipt"))
    expected = _normalized_roster(expected_recording_ids, "source-dev recording ID")
    observed = [
        _identifier(row.get("recording_id"), "EventNet inventory recording ID")
        if isinstance(row, Mapping)
        else ""
        for row in batch["inventory"]
    ]
    if observed != sorted(observed) or len(observed) != len(set(observed)):
        raise ValueError(
            "EventNet batch inventory is not canonically sorted and unique"
        )
    if observed != expected:
        missing = sorted(set(expected).difference(observed))
        extra = sorted(set(observed).difference(expected))
        raise ValueError(
            "EventNet source-dev inventory mismatch: "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )

    sealed: list[ValidatedEventNetRawPrediction] = []
    inventory_receipts: list[dict[str, Any]] = []
    total_duration = 0.0
    total_samples = 0
    for index, raw in enumerate(batch["inventory"]):
        if type(raw) is not dict or set(raw) != _INVENTORY_FIELDS:
            raise ValueError(f"EventNet batch inventory row {index} fields drifted")
        row = deepcopy(raw)
        recording_id = observed[index]
        record_key = hashlib.sha256(recording_id.encode("utf-8")).hexdigest()[:24]
        if row["record_key"] != record_key or row["model_split"] != "source_dev":
            raise ValueError("EventNet record key or split binding drifted")
        for field in (
            "prediction_receipt_sha256",
            "prediction_receipt_file_sha256",
            "prediction_tensor_file_sha256",
            "benchmark_projection_receipt_sha256",
            "benchmark_projection_file_sha256",
        ):
            if not _is_sha256(row[field]):
                raise ValueError(f"EventNet inventory {field} is invalid")
        record_directory = _regular_directory(
            root / "records" / record_key, "EventNet record directory"
        )
        expected_names = {
            EVENTNET_PREDICTION_RECEIPT_FILENAME,
            EVENTNET_TENSOR_FILENAME,
            EVENTNET_BENCHMARK_PROJECTION_FILENAME,
        }
        if {path.name for path in record_directory.iterdir()} != expected_names:
            raise ValueError("EventNet record directory has missing or unknown files")

        prediction_path = _regular_file(
            record_directory / EVENTNET_PREDICTION_RECEIPT_FILENAME,
            "EventNet prediction receipt",
        )
        prediction_bytes = prediction_path.read_bytes()
        if (
            _file_sha256_bytes(prediction_bytes)
            != row["prediction_receipt_file_sha256"]
        ):
            raise ValueError("EventNet prediction receipt file SHA-256 drifted")
        prediction = validate_eventnet_prediction_receipt(
            _loads_json(prediction_bytes, "EventNet prediction receipt")
        )
        if (
            prediction["recording_id"] != recording_id
            or prediction["prediction_id"] != row["prediction_id"]
            or prediction["receipt_sha256"] != row["prediction_receipt_sha256"]
            or prediction["provider_execution_receipt"]
            != batch["provider_execution_receipt"]
        ):
            raise ValueError("EventNet prediction and batch inventory disagree")

        projection_path = _regular_file(
            record_directory / EVENTNET_BENCHMARK_PROJECTION_FILENAME,
            "EventNet benchmark projection",
        )
        projection_bytes = projection_path.read_bytes()
        if (
            _file_sha256_bytes(projection_bytes)
            != row["benchmark_projection_file_sha256"]
        ):
            raise ValueError("EventNet benchmark projection file SHA-256 drifted")
        projection = validate_eventnet_benchmark_prediction_projection(
            _loads_json(projection_bytes, "EventNet benchmark projection")
        )
        if (
            projection["projection_id"] != row["benchmark_projection_id"]
            or projection["receipt_sha256"]
            != row["benchmark_projection_receipt_sha256"]
            or projection["prediction_id"] != prediction["prediction_id"]
            or projection["prediction_receipt_sha256"] != prediction["receipt_sha256"]
            or projection["recording_id"] != recording_id
        ):
            raise ValueError("EventNet benchmark projection and prediction disagree")

        tensor_path = _regular_file(
            record_directory / EVENTNET_TENSOR_FILENAME,
            "EventNet prediction tensor sidecar",
        )
        arrays, tensor_replay = _load_validated_tensor_file(
            tensor_path,
            expected_file_sha256=row["prediction_tensor_file_sha256"],
            prediction_receipt=prediction,
        )
        duration = float(prediction["runtime_receipt"]["recording_duration_seconds"])
        if (
            float(row["recording_duration_seconds"]) != duration
            or float(projection["duration_seconds"]) != duration
            or row["outcome_status"]
            != prediction["generic_full_record_result"]["outcome_status"]
            or row["event_proposal_count"]
            != prediction["decoder_receipt"]["event_proposal_count"]
        ):
            raise ValueError("EventNet duration or released-decoder outcome drifted")
        source_signal_sha256 = prediction["canonical_detector_input_binding"][
            "canonical_source_signal_sha256"
        ]
        sample_count = int(arrays["center_probability"].size)
        patient_alias = _patient_alias_from_recording_id(recording_id)
        sealed.append(
            ValidatedEventNetRawPrediction(
                recording_id=recording_id,
                patient_alias=patient_alias,
                record_key=record_key,
                prediction_id=prediction["prediction_id"],
                prediction_receipt_sha256=prediction["receipt_sha256"],
                source_signal_sha256=source_signal_sha256,
                recording_duration_seconds=duration,
                sample_count=sample_count,
                prediction_receipt_json=_canonical_json(prediction),
                tensor_path=str(tensor_path.resolve(strict=True)),
                tensor_file_sha256=row["prediction_tensor_file_sha256"],
                tensor_replay_receipt_json=_canonical_json(tensor_replay),
            )
        )
        inventory_receipts.append(
            {
                "recording_id": recording_id,
                "patient_alias": patient_alias,
                "record_key": record_key,
                "prediction_id": prediction["prediction_id"],
                "prediction_receipt_sha256": prediction["receipt_sha256"],
                "source_signal_sha256": source_signal_sha256,
                "tensor_file_sha256": row["prediction_tensor_file_sha256"],
                "tensor_replay_receipt_sha256": _canonical_sha256(tensor_replay),
                "sample_count": sample_count,
                "recording_duration_seconds": duration,
            }
        )
        total_duration += duration
        total_samples += sample_count

    if abs(total_duration - float(batch["recording_duration_seconds"])) > 1e-9:
        raise ValueError("EventNet batch recording duration does not replay")
    record_root = _regular_directory(root / "records", "EventNet records root")
    if {path.name for path in record_root.iterdir()} != {
        row["record_key"] for row in inventory_receipts
    }:
        raise ValueError("EventNet records root contains unindexed record directories")
    patient_aliases = sorted({row["patient_alias"] for row in inventory_receipts})
    receipt: dict[str, Any] = {
        "schema_version": EVENTNET_RAW_BUNDLE_SCHEMA_VERSION,
        "validation_id": "EVENTNET-RAW-BUNDLE-VALIDATION-PENDING",
        "method_id": EVENTNET_RAW_BUNDLE_METHOD_ID,
        "provider_id": EVENTNET_PROVIDER_ID,
        "validator_code_sha256": eventnet_raw_bundle_validator_code_sha256(),
        "provider_adapter_code_sha256": batch["provider_execution_receipt"][
            "code_sha256"
        ],
        "weights_manifest_sha256": batch["provider_execution_receipt"][
            "checkpoint_sha256"
        ],
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
        "record_inventory": inventory_receipts,
        "record_inventory_sha256": _canonical_sha256(inventory_receipts),
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
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    receipt["validation_id"] = "EVNRAWVAL-" + _canonical_sha256(receipt)[:24]
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    return ValidatedEventNetRawPredictionBundle(
        batch_root=str(root),
        recordings=tuple(sealed),
        validation_receipt_json=_canonical_json(receipt),
    )


def revalidate_eventnet_raw_prediction_bundle_without_references(
    value: object,
) -> ValidatedEventNetRawPredictionBundle:
    """Revalidate the sealed in-memory carrier without accepting file paths."""

    if type(value) is not ValidatedEventNetRawPredictionBundle:
        raise TypeError("EventNet decoder grid requires the sealed raw-bundle type")
    receipt = value.validation_receipt()
    if (
        receipt.get("schema_version") != EVENTNET_RAW_BUNDLE_SCHEMA_VERSION
        or receipt.get("method_id") != EVENTNET_RAW_BUNDLE_METHOD_ID
        or receipt.get("provider_id") != EVENTNET_PROVIDER_ID
        or receipt.get("source_split") != "source_dev"
        or receipt.get("reference_access") != _REFERENCE_ACCESS
        or receipt.get("record_count") != len(value.recordings)
        or not _is_sha256(receipt.get("validator_code_sha256"))
        or not _is_sha256(receipt.get("provider_adapter_code_sha256"))
        or not _is_sha256(receipt.get("weights_manifest_sha256"))
    ):
        raise ValueError("sealed EventNet raw-bundle receipt identity drifted")
    inventory = [
        {
            "recording_id": row.recording_id,
            "patient_alias": row.patient_alias,
            "record_key": row.record_key,
            "prediction_id": row.prediction_id,
            "prediction_receipt_sha256": row.prediction_receipt_sha256,
            "source_signal_sha256": row.source_signal_sha256,
            "tensor_file_sha256": row.tensor_file_sha256,
            "tensor_replay_receipt_sha256": _canonical_sha256(
                row.tensor_replay_receipt()
            ),
            "sample_count": row.sample_count,
            "recording_duration_seconds": row.recording_duration_seconds,
        }
        for row in value.recordings
    ]
    if receipt.get("record_inventory") != inventory or receipt.get(
        "record_inventory_sha256"
    ) != _canonical_sha256(inventory):
        raise ValueError("sealed EventNet raw-bundle record inventory drifted")
    digest = deepcopy(receipt)
    digest["validation_id"] = "EVENTNET-RAW-BUNDLE-VALIDATION-PENDING"
    digest["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if receipt.get("validation_id") != "EVNRAWVAL-" + _canonical_sha256(digest)[:24]:
        raise ValueError("sealed EventNet raw-bundle validation ID drifted")
    digest = deepcopy(receipt)
    digest["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if receipt.get("receipt_sha256") != _canonical_sha256(digest):
        raise ValueError("sealed EventNet raw-bundle validation hash drifted")
    return value


__all__ = [
    "EVENTNET_RAW_BUNDLE_METHOD_ID",
    "EVENTNET_RAW_BUNDLE_SCHEMA_VERSION",
    "ValidatedEventNetRawPrediction",
    "ValidatedEventNetRawPredictionBundle",
    "eventnet_raw_bundle_validator_code_sha256",
    "revalidate_eventnet_raw_prediction_bundle_without_references",
    "validate_eventnet_raw_prediction_bundle_without_references",
    "validate_eventnet_tensor_sidecar_bytes",
]
