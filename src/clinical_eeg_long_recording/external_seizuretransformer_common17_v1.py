"""Research-only common17 adapter for an external SeizureTransformer checkpoint.

This is deliberately separate from the clean-room ST16 provider.  The external
artifact is a 19-referential-channel checkpoint with unknown exact training
exposure.  It is mechanically projected to the directly observed common17
axis by deleting only the FZ/PZ columns of the first convolution.  The adapter
replays the public repository's referential preprocessing and postprocessing
as closely as its source permits, while recording the remaining provenance
uncertainty explicitly.
"""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pyedflib
import scipy
from safetensors.torch import load_file
from scipy.ndimage import binary_closing, binary_opening
from scipy.signal import butter, iirnotch, lfilter, resample
import torch

from .continuous_detection_benchmark import _aggregate_metrics
from .eventnet_common17_streaming_v1 import (
    COMMON17_CHANNEL_ORDER,
    _edf_layout,
    _read_source_matrix,
    load_common17_manifest,
)
from .onset_collar_scoring_v1 import aggregate_onset_collar_metrics
from .st16_common17_source_dev_evaluation_v1 import _official_timescoring_metrics
from third_party.SeizureTransformer.time_step_level.model import (
    SeizureTransformer,
)


SCHEMA_VERSION = "external_seizuretransformer_common17_adapter_v1"
PROVIDER_ID = "external_seizuretransformer_projected_ref17_exposure_unknown_v1"
PROJECTION_SCHEMA_VERSION = (
    "external_seizuretransformer_common17_target_free_projection_v1"
)
LEGACY_PREDICTION_ROW_SCHEMA_VERSION = (
    "external_seizuretransformer_common17_prediction_row_v1"
)
PREDICTION_ROW_SCHEMA_VERSION = (
    "external_seizuretransformer_common17_prediction_row_v2"
)
LEGACY_PREDICTION_MANIFEST_SCHEMA_VERSION = (
    "external_seizuretransformer_common17_prediction_manifest_v1"
)
PREDICTION_MANIFEST_SCHEMA_VERSION = (
    "external_seizuretransformer_common17_prediction_manifest_v2"
)
SCORE_SCHEMA_VERSION = "external_seizuretransformer_common17_source_dev_score_v2"
PREDICTION_AUDIT_SCHEMA_VERSION = (
    "external_seizuretransformer_common17_prediction_inventory_audit_v1"
)
RUN_CONTRACT_SCHEMA_VERSION = (
    "external_seizuretransformer_common17_inference_run_contract_v1"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "2cdc841001a0fbcdf1dfcbb02b3a26fa7af14002e01ebf9815fa09c82be06f61"
)
TARGET_FS_HZ = 256
WINDOW_SAMPLES = 60 * TARGET_FS_HZ
STANDARD19 = (
    "FP1",
    "F3",
    "C3",
    "P3",
    "O1",
    "F7",
    "T7",
    "P7",
    "FZ",
    "CZ",
    "PZ",
    "FP2",
    "F4",
    "C4",
    "P4",
    "O2",
    "F8",
    "T8",
    "P8",
)
COMMON17 = tuple(channel for channel in STANDARD19 if channel not in {"FZ", "PZ"})
COMMON17_INDICES = tuple(
    index for index, channel in enumerate(STANDARD19) if channel in COMMON17
)
DEFAULT_THRESHOLDS = (
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
    0.99,
)
_PENDING = "CONTENT-ADDRESS-PENDING"

PREPROCESSING_CONTRACT = (
    "public_repository_referential_whole_record_zscore_then_FFT_"
    "resample_to_256Hz_then_per_nonoverlap_60s_tile_causal_0p5_120Hz_"
    "bandpass_1Hz_notch_60Hz_notch_v1"
)
POSTPROCESSING_CONTRACT = (
    "strict_probability_threshold_then_binary_opening_5_samples_"
    "closing_5_samples_then_remove_events_shorter_than_2_seconds"
)

if COMMON17 != tuple(COMMON17_CHANNEL_ORDER):
    raise RuntimeError("external checkpoint and common17 manifest axes disagree")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).resolve(strict=True).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _content_address(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(value))
    result["receipt_sha256"] = _PENDING
    result["receipt_sha256"] = _canonical_sha256(result)
    return result


def _verify_content_address(
    value: Mapping[str, Any], *, context: str
) -> dict[str, Any]:
    result = deepcopy(dict(value))
    supplied = result.get("receipt_sha256")
    result["receipt_sha256"] = _PENDING
    if not isinstance(supplied, str) or _canonical_sha256(result) != supplied:
        raise ValueError(f"{context} failed content-address replay")
    result["receipt_sha256"] = supplied
    return result


def _atomic_bytes(path: Path, payload: bytes, *, replace: bool) -> None:
    target = path.resolve(strict=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not replace and (target.exists() or target.is_symlink()):
        raise FileExistsError(target)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(temporary, target)
        else:
            os.link(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, value: object, *, replace: bool) -> None:
    _atomic_bytes(path, _canonical_bytes(value) + b"\n", replace=replace)


def _atomic_json_gzip(path: Path, value: object, *, replace: bool) -> None:
    # mtime=0 makes the compressed carrier byte-replayable across runs.
    payload = gzip.compress(_canonical_bytes(value) + b"\n", mtime=0)
    _atomic_bytes(path, payload, replace=replace)


def _load_json_gzip(path: Path) -> dict[str, Any]:
    with gzip.open(path.resolve(strict=True), "rt", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError("compressed prediction row must be an object")
    return value


def build_target_free_source_dev_projection(
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Remove all reference/target fields before the inference process starts."""

    manifest_source = Path(manifest_path).resolve(strict=True)
    manifest = load_common17_manifest(manifest_source, require_complete=True)
    records = [
        {
            "analysis_identity_id": str(row["analysis_identity_id"]),
            "patient_id": str(row["patient_id"]),
            "edf_relative_path": str(row["edf_relative_path"]),
            "model_split": "source_dev",
            "target_sample_count_256hz": int(row["target_sample_count_256hz"]),
            "recording_duration_seconds_fraction": list(
                row["recording_duration_seconds_fraction"]
            ),
        }
        for row in manifest["records"]
        if row["model_split"] == "source_dev"
    ]
    records.sort(key=lambda row: row["analysis_identity_id"])
    if (
        len(records) != 1821
        or len({row["analysis_identity_id"] for row in records}) != len(records)
        or any(Path(row["edf_relative_path"]).parts[0] != "dev" for row in records)
    ):
        raise ValueError("common17 source-dev target-free projection roster drifted")
    return _content_address(
        {
            "schema_version": PROJECTION_SCHEMA_VERSION,
            "source_manifest_path": str(manifest_source),
            "source_manifest_file_sha256": _file_sha256(manifest_source),
            "source_manifest_receipt_sha256": manifest["receipt_sha256"],
            "split": "source_dev",
            "reference_annotation_or_target_fields_present": False,
            "complete_expected_record_count": len(records),
            "records": records,
            "receipt_sha256": _PENDING,
        }
    )


def load_target_free_projection(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve(strict=True)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError("target-free projection must be an object")
    projection = _verify_content_address(value, context="target-free projection")
    if (
        projection.get("schema_version") != PROJECTION_SCHEMA_VERSION
        or projection.get("split") != "source_dev"
        or projection.get("reference_annotation_or_target_fields_present") is not False
        or projection.get("complete_expected_record_count")
        != len(projection.get("records", []))
    ):
        raise PermissionError("target-free projection contract drifted")
    allowed_fields = {
        "analysis_identity_id",
        "patient_id",
        "edf_relative_path",
        "model_split",
        "target_sample_count_256hz",
        "recording_duration_seconds_fraction",
    }
    for row in projection["records"]:
        if set(row) != allowed_fields or row["model_split"] != "source_dev":
            raise PermissionError("prediction row contains reference or unknown fields")
    return projection


def project_state_to_common17(
    state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    first_key = "encoder.convs.0.weight"
    if first_key not in state or tuple(state[first_key].shape) != (32, 19, 11):
        raise ValueError("external checkpoint first convolution is not 19-channel")
    projected = dict(state)
    projected[first_key] = state[first_key][
        :, COMMON17_INDICES, :
    ].contiguous()
    changed = [
        key
        for key in state
        if tuple(state[key].shape) != tuple(projected[key].shape)
    ]
    if changed != [first_key] or tuple(projected[first_key].shape) != (32, 17, 11):
        raise RuntimeError("external checkpoint common17 projection drifted")
    return projected


def load_projected_model(
    checkpoint_path: str | Path, *, device: torch.device
) -> tuple[SeizureTransformer, str]:
    source = Path(checkpoint_path).resolve(strict=True)
    checkpoint_sha = _file_sha256(source)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("external SeizureTransformer checkpoint bytes drifted")
    state = load_file(str(source), device="cpu")
    model = SeizureTransformer(
        in_channels=17,
        in_samples=WINDOW_SAMPLES,
        dim_feedforward=2048,
        num_layers=8,
        num_heads=4,
        drop_rate=0.1,
    )
    model.load_state_dict(project_state_to_common17(state), strict=True)
    model.to(device).eval()
    return model, checkpoint_sha


def read_official_repo_referential_common17(path: str | Path) -> np.ndarray:
    """Replay public-repo referential normalization then FFT resampling.

    The public code normalizes each complete EDF channel to mean zero/unit
    population standard deviation *before* ``scipy.signal.resample``.  It then
    resets causal filters independently for every 60-second model tile.  That
    tile filtering is performed by :func:`official_repo_filtered_tiles`.
    """

    source_path = Path(path).resolve(strict=True)
    with pyedflib.EdfReader(str(source_path)) as reader:
        layout = _edf_layout(reader)
        source = _read_source_matrix(
            reader, layout, start=0, count=layout.source_sample_count
        )
    center = np.mean(source, axis=1, keepdims=True)
    scale = np.std(source, axis=1, keepdims=True)
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("external checkpoint preprocessing found zero/nonfinite std")
    normalized = (source - center) / scale
    target_count = (
        layout.source_sample_count
        * TARGET_FS_HZ
        * layout.sampling_rate.denominator
    ) // layout.sampling_rate.numerator
    if target_count < 1:
        raise ValueError("external checkpoint preprocessing produced empty support")
    if layout.sampling_rate == Fraction(TARGET_FS_HZ, 1):
        transformed = normalized
    else:
        transformed = resample(normalized, int(target_count), axis=1)
    output = np.ascontiguousarray(transformed[:, :target_count], dtype=np.float32)
    if output.shape != (17, target_count) or not np.isfinite(output).all():
        raise RuntimeError("external checkpoint referential transform drifted")
    return output


def official_repo_filtered_tiles(signal: np.ndarray) -> tuple[np.ndarray, int]:
    """Return non-overlapping, tail-zero-padded, causally filtered 60-s tiles."""

    values = np.asarray(signal)
    if values.ndim != 2 or values.shape[0] != 17 or values.shape[1] < 1:
        raise ValueError("external checkpoint signal must have shape [17,time]")
    sample_count = int(values.shape[1])
    tile_count = max(1, math.ceil(sample_count / WINDOW_SAMPLES))
    padded = np.zeros((17, tile_count * WINDOW_SAMPLES), dtype=np.float64)
    padded[:, :sample_count] = values
    tiles = padded.reshape(17, tile_count, WINDOW_SAMPLES).transpose(1, 0, 2)
    nyquist = 0.5 * TARGET_FS_HZ
    band_b, band_a = butter(3, [0.5 / nyquist, 120.0 / nyquist], btype="band")
    notch_1_b, notch_1_a = iirnotch(1.0, Q=30, fs=TARGET_FS_HZ)
    notch_60_b, notch_60_a = iirnotch(60.0, Q=30, fs=TARGET_FS_HZ)
    filtered = lfilter(band_b, band_a, tiles, axis=-1)
    filtered = lfilter(notch_1_b, notch_1_a, filtered, axis=-1)
    filtered = lfilter(notch_60_b, notch_60_a, filtered, axis=-1)
    output = np.ascontiguousarray(filtered, dtype=np.float32)
    if output.shape != (tile_count, 17, WINDOW_SAMPLES):
        raise RuntimeError("external checkpoint tile geometry drifted")
    return output, sample_count


def predict_preprocessed_record(
    signal: np.ndarray,
    *,
    model: SeizureTransformer,
    device: torch.device,
    batch_size: int,
    precision: str,
) -> np.ndarray:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("prediction batch_size must be a positive integer")
    if precision not in {"float32", "bfloat16"}:
        raise ValueError("precision must be float32 or bfloat16")
    if precision == "bfloat16" and (
        device.type != "cuda" or not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("bfloat16 inference requires BF16-capable CUDA")
    tiles, observed_sample_count = official_repo_filtered_tiles(signal)
    outputs: list[np.ndarray] = []
    for start in range(0, len(tiles), batch_size):
        inputs = torch.from_numpy(tiles[start : start + batch_size]).to(device)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=precision == "bfloat16",
        ):
            probability = model(inputs)
        outputs.append(probability.float().cpu().numpy())
    posterior = np.ascontiguousarray(
        np.concatenate(outputs, axis=0).reshape(-1)[:observed_sample_count],
        dtype=np.float32,
    )
    if (
        posterior.shape != (observed_sample_count,)
        or not np.isfinite(posterior).all()
        or float(posterior.min()) < 0
        or float(posterior.max()) > 1
    ):
        raise RuntimeError("external checkpoint posterior is malformed")
    return posterior


def decode_official_repository_events(
    probability: np.ndarray,
    *,
    threshold: float,
    duration_seconds: float,
) -> list[dict[str, float]]:
    """Replay threshold, 5-sample open/close, and 2-second event floor."""

    values = np.asarray(probability)
    if values.ndim != 1 or not len(values) or not 0 < float(threshold) < 1:
        raise ValueError("posterior/threshold is invalid")
    positive = values > float(threshold)  # public repository uses strict >
    structure = np.ones(5, dtype=np.bool_)
    positive = binary_opening(positive, structure=structure)
    positive = binary_closing(positive, structure=structure)
    transitions = np.diff(
        np.concatenate(([False], positive, [False])).astype(np.int8)
    )
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1)
    events: list[dict[str, float]] = []
    for start, stop in zip(starts.tolist(), stops.tolist(), strict=True):
        if stop - start < 2 * TARGET_FS_HZ:
            continue
        start_seconds = start / TARGET_FS_HZ
        stop_seconds = min(stop / TARGET_FS_HZ, float(duration_seconds))
        if stop_seconds > start_seconds:
            events.append(
                {
                    "start_seconds": float(start_seconds),
                    "stop_seconds": float(stop_seconds),
                }
            )
    return events


def _legacy_threshold_key(value: float) -> str:
    """Return the non-injective key used by the frozen v1 prediction run."""

    return f"{float(value):.3f}"


def _threshold_key(value: float) -> str:
    """Return an exact, round-trip-safe binary64 threshold key."""

    result = float(value)
    if not math.isfinite(result) or not 0 < result < 1:
        raise ValueError("threshold key requires a finite value in (0,1)")
    return f"binary64:{result.hex()}"


def _threshold_key_for_manifest(value: float, schema_version: str) -> str:
    if schema_version == LEGACY_PREDICTION_MANIFEST_SCHEMA_VERSION:
        return _legacy_threshold_key(value)
    if schema_version == PREDICTION_MANIFEST_SCHEMA_VERSION:
        return _threshold_key(value)
    raise ValueError("external prediction manifest schema is unsupported")


def _normalize_thresholds(values: Sequence[float]) -> tuple[float, ...]:
    raw = tuple(float(value) for value in values)
    result = tuple(sorted(set(raw)))
    if (
        not result
        or len(result) != len(raw)
        or any(not math.isfinite(value) or not 0 < value < 1 for value in result)
    ):
        raise ValueError("threshold grid must contain unique finite values in (0,1)")
    keys = tuple(_threshold_key(value) for value in result)
    if len(set(keys)) != len(keys):
        raise ValueError("threshold grid keys are not one-to-one")
    return result


def _validate_inference_parameters(
    *, device_name: str, batch_size: int, precision: str
) -> torch.device:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("prediction batch_size must be a positive integer")
    if precision not in {"float32", "bfloat16"}:
        raise ValueError("precision must be float32 or bfloat16")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    if precision == "bfloat16" and (
        device.type != "cuda" or not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("bfloat16 inference requires BF16-capable CUDA")
    return device


def _build_run_contract(
    *,
    checkpoint_sha256: str,
    projection_receipt_sha256: str,
    thresholds: Sequence[float],
    device: torch.device,
    batch_size: int,
    precision: str,
) -> dict[str, Any]:
    source = Path(__file__).resolve(strict=True)
    model_source = (
        source.parents[2]
        / "third_party/SeizureTransformer/time_step_level/model.py"
    ).resolve(strict=True)
    contract = {
        "schema_version": RUN_CONTRACT_SCHEMA_VERSION,
        "checkpoint_sha256": checkpoint_sha256,
        "projection_receipt_sha256": projection_receipt_sha256,
        "thresholds": [float(value) for value in thresholds],
        "threshold_keys": [_threshold_key(float(value)) for value in thresholds],
        "device_name": str(device),
        "device_type": device.type,
        "batch_size": batch_size,
        "precision": precision,
        "model_configuration": {
            "in_channels": 17,
            "in_samples": WINDOW_SAMPLES,
            "dim_feedforward": 2048,
            "num_layers": 8,
            "num_heads": 4,
            "drop_rate": 0.1,
        },
        "preprocessing_contract": PREPROCESSING_CONTRACT,
        "postprocessing_contract": POSTPROCESSING_CONTRACT,
        "adapter_source_sha256": _file_sha256(source),
        "vendored_model_source_sha256": _file_sha256(model_source),
        "dependency_versions": {
            "numpy": str(np.__version__),
            "scipy": str(scipy.__version__),
            "torch": str(torch.__version__),
            "torch_cuda": None if torch.version.cuda is None else str(torch.version.cuda),
            "pyedflib": str(pyedflib.__version__),
        },
        "offline_retrospective_only": True,
        "whole_record_future_context_used": True,
        "bidirectional_60s_model_context": True,
        "checkpoint_native_preprocessing_verified": False,
        "checkpoint_channel_axis_verified": False,
    }
    return {**contract, "run_contract_sha256": _canonical_sha256(contract)}


def _prediction_row_is_reusable(
    row: Mapping[str, Any],
    *,
    identity_row: Mapping[str, Any],
    roster_index: int,
    checkpoint_sha256: str,
    projection_receipt_sha256: str,
    thresholds: Sequence[float],
    run_contract: Mapping[str, Any],
) -> bool:
    expected_keys = {_threshold_key(float(value)) for value in thresholds}
    return bool(
        row.get("schema_version") == PREDICTION_ROW_SCHEMA_VERSION
        and row.get("provider_id") == PROVIDER_ID
        and row.get("claim_status")
        == "external_exposure_unknown_diagnostic_only"
        and row.get("status") == "prediction_complete"
        and row.get("analysis_identity_id")
        == identity_row.get("analysis_identity_id")
        and row.get("patient_id") == identity_row.get("patient_id")
        and row.get("recording_id") == identity_row.get("edf_relative_path")
        and row.get("prediction_roster_index") == roster_index
        and row.get("checkpoint_sha256") == checkpoint_sha256
        and row.get("projection_receipt_sha256")
        == projection_receipt_sha256
        and row.get("thresholds") == list(thresholds)
        and row.get("run_contract_sha256")
        == run_contract.get("run_contract_sha256")
        and row.get("device_name") == run_contract.get("device_name")
        and row.get("device_type") == run_contract.get("device_type")
        and row.get("batch_size") == run_contract.get("batch_size")
        and row.get("precision") == run_contract.get("precision")
        and row.get("reference_annotation_or_target_opened") is False
        and row.get("source_eval_opened") is False
        and row.get("sample_count")
        == int(identity_row.get("target_sample_count_256hz", -1))
        and isinstance(row.get("events_by_threshold"), Mapping)
        and set(row["events_by_threshold"]) == expected_keys
    )


def _safe_source_path(root: Path, relative_text: str) -> Path:
    relative = Path(relative_text)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not relative.parts
        or relative.parts[0] != "dev"
        or relative.suffix.lower() != ".edf"
    ):
        raise PermissionError("external checkpoint path crosses source-dev root")
    candidate = root / relative
    for depth in range(1, len(relative.parts) + 1):
        if root.joinpath(*relative.parts[:depth]).is_symlink():
            raise PermissionError("source-dev path must not traverse a symlink")
    result = candidate.resolve(strict=True)
    result.relative_to(root)
    if result.is_symlink() or not result.is_file():
        raise PermissionError("source-dev EDF must be a regular non-symlink file")
    return result


def predict_source_dev(
    *,
    checkpoint_path: str | Path,
    projection_path: str | Path,
    tusz_root: str | Path,
    output_dir: str | Path,
    device_name: str,
    batch_size: int,
    precision: str,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    maximum_records: int | None = None,
    retry_failures: bool = False,
) -> dict[str, Any]:
    """Materialize target-free source-dev event predictions for a threshold grid."""

    projection_source = Path(projection_path).resolve(strict=True)
    projection = load_target_free_projection(projection_source)
    rows = list(projection["records"])
    full_count = len(rows)
    if maximum_records is not None:
        if (
            isinstance(maximum_records, bool)
            or not isinstance(maximum_records, int)
            or maximum_records < 1
        ):
            raise ValueError("maximum_records must be positive")
        rows = rows[:maximum_records]
    normalized_thresholds = _normalize_thresholds(thresholds)
    device = _validate_inference_parameters(
        device_name=device_name, batch_size=batch_size, precision=precision
    )
    model, checkpoint_sha = load_projected_model(checkpoint_path, device=device)
    run_contract = _build_run_contract(
        checkpoint_sha256=checkpoint_sha,
        projection_receipt_sha256=projection["receipt_sha256"],
        thresholds=normalized_thresholds,
        device=device,
        batch_size=batch_size,
        precision=precision,
    )
    root = Path(tusz_root).resolve(strict=True)
    output = Path(output_dir).resolve(strict=False)
    output.mkdir(parents=True, exist_ok=True)
    result_rows: list[dict[str, Any]] = []
    reused_completed_count = 0
    retried_failure_count = 0
    began = time.perf_counter()
    for index, identity_row in enumerate(rows):
        identity = str(identity_row["analysis_identity_id"])
        row_path = output / "records" / f"{identity}.json.gz"
        replace_existing_failure = False
        superseded_failure: dict[str, Any] | None = None
        if row_path.is_file() and not row_path.is_symlink():
            reused = _verify_content_address(
                _load_json_gzip(row_path), context=f"prediction row {identity}"
            )
            if _prediction_row_is_reusable(
                reused,
                identity_row=identity_row,
                roster_index=index,
                checkpoint_sha256=checkpoint_sha,
                projection_receipt_sha256=projection["receipt_sha256"],
                thresholds=normalized_thresholds,
                run_contract=run_contract,
            ):
                result_rows.append(reused)
                reused_completed_count += 1
                continue
            retryable_failure = bool(
                retry_failures
                and reused.get("schema_version") == PREDICTION_ROW_SCHEMA_VERSION
                and reused.get("provider_id") == PROVIDER_ID
                and reused.get("status") == "typed_technical_failure"
                and reused.get("analysis_identity_id") == identity
                and reused.get("prediction_roster_index") == index
                and reused.get("checkpoint_sha256") == checkpoint_sha
                and reused.get("projection_receipt_sha256")
                == projection["receipt_sha256"]
            )
            if retryable_failure:
                replace_existing_failure = True
                retried_failure_count += 1
                superseded_failure = {
                    "receipt_sha256": reused["receipt_sha256"],
                    "run_contract_sha256": reused.get("run_contract_sha256"),
                    "failure_type": reused.get("failure_type"),
                    "failure_message": reused.get("failure_message"),
                }
            else:
                raise FileExistsError(
                    "incompatible or failed prediction row exists; use a new "
                    f"output directory or --retry-failures: {row_path}"
                )
        row_began = time.perf_counter()
        base = {
            "schema_version": PREDICTION_ROW_SCHEMA_VERSION,
            "provider_id": PROVIDER_ID,
            "claim_status": "external_exposure_unknown_diagnostic_only",
            "analysis_identity_id": identity,
            "patient_id": identity_row["patient_id"],
            "recording_id": identity_row["edf_relative_path"],
            "prediction_roster_index": index,
            "projection_receipt_sha256": projection["receipt_sha256"],
            "checkpoint_sha256": checkpoint_sha,
            "thresholds": list(normalized_thresholds),
            "threshold_keys": [
                _threshold_key(value) for value in normalized_thresholds
            ],
            "run_contract_sha256": run_contract["run_contract_sha256"],
            "device_name": str(device),
            "device_type": device.type,
            "batch_size": batch_size,
            "precision": precision,
            "reference_annotation_or_target_opened": False,
            "source_eval_opened": False,
            "offline_retrospective_only": True,
            "whole_record_future_context_used": True,
            "bidirectional_60s_model_context": True,
        }
        if superseded_failure is not None:
            base["superseded_failure_attempt"] = superseded_failure
        try:
            edf_path = _safe_source_path(root, str(identity_row["edf_relative_path"]))
            io_began = time.perf_counter()
            signal = read_official_repo_referential_common17(edf_path)
            io_seconds = time.perf_counter() - io_began
            if signal.shape[1] != int(identity_row["target_sample_count_256hz"]):
                raise RuntimeError("prediction signal sample count disagrees with projection")
            inference_began = time.perf_counter()
            posterior = predict_preprocessed_record(
                signal,
                model=model,
                device=device,
                batch_size=batch_size,
                precision=precision,
            )
            inference_seconds = time.perf_counter() - inference_began
            duration_fraction = identity_row["recording_duration_seconds_fraction"]
            duration_seconds = float(duration_fraction[0]) / float(duration_fraction[1])
            decode_began = time.perf_counter()
            events_by_threshold = {
                _threshold_key(threshold): decode_official_repository_events(
                    posterior,
                    threshold=threshold,
                    duration_seconds=duration_seconds,
                )
                for threshold in normalized_thresholds
            }
            decode_seconds = time.perf_counter() - decode_began
            row = _content_address(
                {
                    **base,
                    "status": "prediction_complete",
                    "sample_count": int(len(posterior)),
                    "posterior_minimum": float(posterior.min()),
                    "posterior_maximum": float(posterior.max()),
                    "posterior_mean": float(posterior.mean(dtype=np.float64)),
                    "events_by_threshold": events_by_threshold,
                    "preprocessing_seconds": io_seconds,
                    "inference_seconds": inference_seconds,
                    "decoding_seconds": decode_seconds,
                    "wall_seconds": time.perf_counter() - row_began,
                    "receipt_sha256": _PENDING,
                }
            )
        except Exception as exc:  # noqa: BLE001 - typed failure is benchmark data
            row = _content_address(
                {
                    **base,
                    "status": "typed_technical_failure",
                    "failure_type": type(exc).__name__,
                    "failure_message": str(exc),
                    "wall_seconds": time.perf_counter() - row_began,
                    "receipt_sha256": _PENDING,
                }
            )
        _atomic_json_gzip(
            row_path, row, replace=replace_existing_failure
        )
        result_rows.append(row)
        print(
            json.dumps(
                {
                    "stage": "external_seizuretransformer_common17_predict",
                    "completed": index + 1,
                    "selected": len(rows),
                    "status": row["status"],
                    "analysis_identity_id": identity,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    manifest = _content_address(
        {
            "schema_version": PREDICTION_MANIFEST_SCHEMA_VERSION,
            "provider_id": PROVIDER_ID,
            "claim_status": "external_exposure_unknown_diagnostic_only",
            "checkpoint_path": str(Path(checkpoint_path).resolve(strict=True)),
            "checkpoint_sha256": checkpoint_sha,
            "projection_path": str(projection_source),
            "projection_receipt_sha256": projection["receipt_sha256"],
            "run_contract": run_contract,
            "run_contract_sha256": run_contract["run_contract_sha256"],
            "source_eval_opened": False,
            "reference_annotation_or_target_opened": False,
            "preprocessing_contract": PREPROCESSING_CONTRACT,
            "postprocessing_contract": POSTPROCESSING_CONTRACT,
            "offline_retrospective_only": True,
            "whole_record_future_context_used": True,
            "bidirectional_60s_model_context": True,
            "checkpoint_native_preprocessing_verified": False,
            "checkpoint_channel_axis_verified": False,
            "channel_projection": {
                "source_axis": list(STANDARD19),
                "target_axis": list(COMMON17),
                "deleted_axes": ["FZ", "PZ"],
                "zero_fill_interpolation_or_CZ_substitution": False,
                "mathematically_equivalent_to_standardized_FZ_PZ_zero_inputs": True,
                "changed_state_tensors": ["encoder.convs.0.weight"],
            },
            "short_record_policy": {
                "local_policy": "include_and_zero_pad_tail_to_60s",
                "public_eval_script_policy": "skip_records_shorter_than_60s",
                "selected_short_record_count": sum(
                    int(row["target_sample_count_256hz"]) < WINDOW_SAMPLES
                    for row in rows
                ),
                "protocol_difference_declared": True,
            },
            "device_name": str(device),
            "device_type": device.type,
            "precision": precision,
            "batch_size": batch_size,
            "thresholds": list(normalized_thresholds),
            "full_expected_record_count": full_count,
            "selected_record_count": len(rows),
            "complete_full_inventory": len(rows) == full_count,
            "prediction_complete_count": sum(
                row["status"] == "prediction_complete" for row in result_rows
            ),
            "typed_technical_failure_count": sum(
                row["status"] == "typed_technical_failure" for row in result_rows
            ),
            "reused_completed_prediction_count": reused_completed_count,
            "retried_failure_count": retried_failure_count,
            "retry_failures_enabled": retry_failures,
            "prediction_rows": result_rows,
            "wall_seconds": time.perf_counter() - began,
            "checkpoint_training_exposure_known": False,
            "TUSZ_source_dev_or_eval_exposure_excluded": False,
            "eligible_as_independent_test_or_primary_SOTA": False,
            "receipt_sha256": _PENDING,
        }
    )
    _atomic_json(output / "prediction_manifest.json", manifest, replace=True)
    return manifest


def _validated_prediction_manifest(
    path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate a v1 legacy or v2 bound prediction inventory row by row."""

    source = Path(path).resolve(strict=True)
    raw = json.loads(source.read_text(encoding="utf-8"))
    prediction = _verify_content_address(raw, context="external prediction manifest")
    schema = prediction.get("schema_version")
    if (
        schema
        not in {
            LEGACY_PREDICTION_MANIFEST_SCHEMA_VERSION,
            PREDICTION_MANIFEST_SCHEMA_VERSION,
        }
        or prediction.get("provider_id") != PROVIDER_ID
        or prediction.get("reference_annotation_or_target_opened") is not False
        or prediction.get("source_eval_opened") is not False
    ):
        raise PermissionError("external prediction manifest contract drifted")

    projection_source = Path(str(prediction.get("projection_path"))).resolve(
        strict=True
    )
    projection = load_target_free_projection(projection_source)
    if projection["receipt_sha256"] != prediction.get(
        "projection_receipt_sha256"
    ):
        raise ValueError("prediction manifest projection binding drifted")
    checkpoint_source = Path(str(prediction.get("checkpoint_path"))).resolve(
        strict=True
    )
    checkpoint_sha256 = _file_sha256(checkpoint_source)
    if (
        checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256
        or checkpoint_sha256 != prediction.get("checkpoint_sha256")
    ):
        raise ValueError("prediction manifest checkpoint binding drifted")

    thresholds = tuple(float(value) for value in prediction.get("thresholds", []))
    if thresholds != _normalize_thresholds(thresholds):
        raise ValueError("prediction manifest threshold order drifted")
    threshold_keys = {
        _threshold_key_for_manifest(value, str(schema)) for value in thresholds
    }
    rows = prediction.get("prediction_rows")
    if not isinstance(rows, list):
        raise TypeError("prediction manifest rows must be an array")
    selected_count = prediction.get("selected_record_count")
    full_count = prediction.get("full_expected_record_count")
    if (
        isinstance(selected_count, bool)
        or not isinstance(selected_count, int)
        or isinstance(full_count, bool)
        or not isinstance(full_count, int)
        or selected_count != len(rows)
        or full_count != len(projection["records"])
        or not 0 < selected_count <= full_count
        or prediction.get("complete_full_inventory") is not (
            selected_count == full_count
        )
    ):
        raise ValueError("prediction manifest roster counts drifted")

    expected_row_schema = (
        LEGACY_PREDICTION_ROW_SCHEMA_VERSION
        if schema == LEGACY_PREDICTION_MANIFEST_SCHEMA_VERSION
        else PREDICTION_ROW_SCHEMA_VERSION
    )
    expected_projection_rows = projection["records"][:selected_count]
    complete_count = 0
    failure_count = 0
    identities: set[str] = set()
    for index, (row, identity_row) in enumerate(
        zip(rows, expected_projection_rows, strict=True)
    ):
        if not isinstance(row, Mapping):
            raise TypeError("prediction row must be an object")
        validated_row = _verify_content_address(
            row, context=f"embedded prediction row {index}"
        )
        identity = str(identity_row["analysis_identity_id"])
        if (
            validated_row.get("schema_version") != expected_row_schema
            or validated_row.get("provider_id") != PROVIDER_ID
            or validated_row.get("claim_status")
            != "external_exposure_unknown_diagnostic_only"
            or validated_row.get("analysis_identity_id") != identity
            or validated_row.get("patient_id") != identity_row["patient_id"]
            or validated_row.get("recording_id")
            != identity_row["edf_relative_path"]
            or validated_row.get("prediction_roster_index") != index
            or validated_row.get("projection_receipt_sha256")
            != projection["receipt_sha256"]
            or validated_row.get("checkpoint_sha256") != checkpoint_sha256
            or validated_row.get("thresholds") != list(thresholds)
            or validated_row.get("reference_annotation_or_target_opened") is not False
            or validated_row.get("source_eval_opened") is not False
            or validated_row.get("status")
            not in {"prediction_complete", "typed_technical_failure"}
            or identity in identities
        ):
            raise PermissionError(f"prediction row {index} lineage drifted")
        identities.add(identity)
        if schema == PREDICTION_MANIFEST_SCHEMA_VERSION and (
            validated_row.get("run_contract_sha256")
            != prediction.get("run_contract_sha256")
            or validated_row.get("device_name") != prediction.get("device_name")
            or validated_row.get("device_type") != prediction.get("device_type")
            or validated_row.get("batch_size") != prediction.get("batch_size")
            or validated_row.get("precision") != prediction.get("precision")
            or validated_row.get("offline_retrospective_only") is not True
            or validated_row.get("whole_record_future_context_used") is not True
            or validated_row.get("bidirectional_60s_model_context") is not True
        ):
            raise PermissionError("v2 prediction row run contract drifted")
        if validated_row["status"] == "prediction_complete":
            complete_count += 1
            events = validated_row.get("events_by_threshold")
            if (
                validated_row.get("sample_count")
                != int(identity_row["target_sample_count_256hz"])
                or not isinstance(events, Mapping)
                or set(events) != threshold_keys
                or any(not isinstance(value, list) for value in events.values())
            ):
                raise ValueError("completed prediction row payload drifted")
        else:
            failure_count += 1
            if "events_by_threshold" in validated_row:
                raise ValueError("technical failure row must not contain events")

    if (
        complete_count != prediction.get("prediction_complete_count")
        or failure_count != prediction.get("typed_technical_failure_count")
        or complete_count + failure_count != selected_count
    ):
        raise ValueError("prediction manifest status counts drifted")

    run_contract_bound = schema == PREDICTION_MANIFEST_SCHEMA_VERSION
    if run_contract_bound:
        contract = prediction.get("run_contract")
        if not isinstance(contract, Mapping):
            raise TypeError("v2 prediction manifest lacks a run contract")
        contract_body = dict(contract)
        supplied_contract_sha = contract_body.pop("run_contract_sha256", None)
        if (
            supplied_contract_sha != _canonical_sha256(contract_body)
            or supplied_contract_sha != prediction.get("run_contract_sha256")
            or contract.get("schema_version") != RUN_CONTRACT_SCHEMA_VERSION
            or contract.get("device_name") != prediction.get("device_name")
            or contract.get("device_type") != prediction.get("device_type")
            or contract.get("batch_size") != prediction.get("batch_size")
            or contract.get("precision") != prediction.get("precision")
            or contract.get("thresholds") != list(thresholds)
            or prediction.get("offline_retrospective_only") is not True
            or prediction.get("whole_record_future_context_used") is not True
            or prediction.get("bidirectional_60s_model_context") is not True
        ):
            raise ValueError("prediction run contract failed replay")

    validation = {
        "manifest_schema_version": schema,
        "row_schema_version": expected_row_schema,
        "row_receipts_replayed": len(rows),
        "unique_identity_count": len(identities),
        "continuous_roster_indices": True,
        "prediction_complete_count": complete_count,
        "typed_technical_failure_count": failure_count,
        "run_contract_content_bound": run_contract_bound,
        "legacy_resume_configuration_unbound": not run_contract_bound,
        "offline_retrospective_only": True,
        "whole_record_future_context_used": True,
        "bidirectional_60s_model_context": True,
    }
    return prediction, projection, validation


def audit_prediction_inventory(
    *,
    prediction_manifest_path: str | Path,
    observed_device_name: str | None = None,
    fresh_single_configuration_run_asserted: bool = False,
) -> dict[str, Any]:
    """Seal row carriers for a completed run without opening EEG or references."""

    source = Path(prediction_manifest_path).resolve(strict=True)
    prediction, _projection, validation = _validated_prediction_manifest(source)
    records_directory = (source.parent / "records").resolve(strict=True)
    embedded_rows = list(prediction["prediction_rows"])
    expected_names = {
        f"{row['analysis_identity_id']}.json.gz" for row in embedded_rows
    }
    files = sorted(records_directory.glob("*.json.gz"))
    if (
        len(files) != len(embedded_rows)
        or {path.name for path in files} != expected_names
        or any(path.is_symlink() or not path.is_file() for path in files)
    ):
        raise ValueError("prediction row carrier roster drifted")
    embedded_by_id = {
        str(row["analysis_identity_id"]): row for row in embedded_rows
    }
    inventory: list[dict[str, str]] = []
    for path in files:
        identity = path.name[: -len(".json.gz")]
        carrier = _verify_content_address(
            _load_json_gzip(path), context=f"prediction carrier {identity}"
        )
        if carrier != embedded_by_id[identity]:
            raise ValueError("prediction carrier differs from embedded row")
        inventory.append(
            {
                "analysis_identity_id": identity,
                "file_sha256": _file_sha256(path),
                "row_receipt_sha256": carrier["receipt_sha256"],
            }
        )
    return _content_address(
        {
            "schema_version": PREDICTION_AUDIT_SCHEMA_VERSION,
            "stage": "read_only_prediction_inventory_seal",
            "prediction_manifest_path": str(source),
            "prediction_manifest_file_sha256": _file_sha256(source),
            "prediction_manifest_receipt_sha256": prediction["receipt_sha256"],
            "validation": validation,
            "carrier_file_count": len(files),
            "carrier_inventory_sha256": _canonical_sha256(inventory),
            "complete_full_inventory": prediction["complete_full_inventory"],
            "all_predictions_complete": (
                prediction["prediction_complete_count"]
                == prediction["selected_record_count"]
                and prediction["typed_technical_failure_count"] == 0
            ),
            "manifest_precision": prediction.get("precision"),
            "manifest_batch_size": prediction.get("batch_size"),
            "observed_device_name_assertion": observed_device_name,
            "fresh_single_configuration_run_asserted": bool(
                fresh_single_configuration_run_asserted
            ),
            "execution_assertions_content_bound_by_legacy_manifest": False,
            "assertion_source": (
                "orchestrating_agent_runtime_observation_not_legacy_manifest"
            ),
            "EDF_or_reference_opened": False,
            "source_eval_opened": False,
            "offline_retrospective_only": True,
            "checkpoint_native_preprocessing_verified": False,
            "checkpoint_channel_axis_verified": False,
            "independent_test_or_primary_SOTA_claim_authorized": False,
            "receipt_sha256": _PENDING,
        }
    )


def _duration_seconds(row: Mapping[str, Any]) -> float:
    fraction = row["recording_duration_seconds_fraction"]
    return float(fraction[0]) / float(fraction[1])


def score_source_dev_predictions(
    *,
    prediction_manifest_path: str | Path,
    reference_manifest_path: str | Path,
    project_root: str | Path,
) -> dict[str, Any]:
    prediction_source = Path(prediction_manifest_path).resolve(strict=True)
    prediction, _projection, prediction_validation = (
        _validated_prediction_manifest(prediction_source)
    )
    reference_source = Path(reference_manifest_path).resolve(strict=True)
    reference = load_common17_manifest(reference_source, require_complete=True)
    reference_by_id = {
        row["analysis_identity_id"]: row
        for row in reference["records"]
        if row["model_split"] == "source_dev"
    }
    prediction_rows = list(prediction["prediction_rows"])
    expected_ids = sorted(reference_by_id)
    observed_ids = sorted(row["analysis_identity_id"] for row in prediction_rows)
    if prediction.get("complete_full_inventory") is True and observed_ids != expected_ids:
        raise ValueError("external prediction/reference full rosters disagree")
    thresholds = tuple(float(value) for value in prediction["thresholds"])
    curve: list[dict[str, Any]] = []
    for threshold in thresholds:
        key = _threshold_key_for_manifest(
            threshold, str(prediction["schema_version"])
        )
        rows: list[dict[str, Any]] = []
        for prediction_row in prediction_rows:
            reference_row = reference_by_id.get(prediction_row["analysis_identity_id"])
            if reference_row is None:
                raise ValueError("external prediction identity is absent from references")
            predicted_events = (
                prediction_row["events_by_threshold"][key]
                if prediction_row["status"] == "prediction_complete"
                else []
            )
            rows.append(
                {
                    "patient_id": reference_row["patient_id"],
                    "recording_id": reference_row["edf_relative_path"],
                    "split": "source_dev",
                    "duration_seconds": _duration_seconds(reference_row),
                    "reference_events": [
                        {
                            "start_seconds": float(event["start_seconds"]),
                            "stop_seconds": float(event["stop_seconds"]),
                        }
                        for event in reference_row["seizure_events"]
                    ],
                    "predicted_events": predicted_events,
                }
            )
        strict = _aggregate_metrics(rows, (1.0, 3.0, 5.0, 10.0))
        onset_collar = aggregate_onset_collar_metrics(rows)
        szcore = _official_timescoring_metrics(
            rows, project_root=Path(project_root).resolve(strict=True)
        )
        curve.append(
            {
                "threshold": threshold,
                "strict": strict,
                "onset_collar": onset_collar,
                "szcore_compatible": szcore,
            }
        )
    best = max(
        curve,
        key=lambda point: (
            -1.0 if point["strict"]["event_f1"] is None else point["strict"]["event_f1"],
            -1.0
            if point["strict"]["event_sensitivity"] is None
            else point["strict"]["event_sensitivity"],
            -math.inf
            if point["strict"]["alarm_false_alarms_per_24h"] is None
            else -point["strict"]["alarm_false_alarms_per_24h"],
            point["threshold"],
        ),
    )
    paper_threshold = next(
        (point for point in curve if abs(point["threshold"] - 0.8) <= 1e-12), None
    )
    return _content_address(
        {
            "schema_version": SCORE_SCHEMA_VERSION,
            "provider_id": PROVIDER_ID,
            "claim_status": "external_exposure_unknown_source_dev_diagnostic_only",
            "prediction_manifest_path": str(prediction_source),
            "prediction_manifest_file_sha256": _file_sha256(prediction_source),
            "prediction_manifest_receipt_sha256": prediction["receipt_sha256"],
            "prediction_manifest_validation": prediction_validation,
            "reference_manifest_path": str(reference_source),
            "reference_manifest_file_sha256": _file_sha256(reference_source),
            "reference_manifest_receipt_sha256": reference["receipt_sha256"],
            "evaluated_record_count": len(prediction_rows),
            "full_source_dev_inventory": prediction.get("complete_full_inventory"),
            "typed_technical_failures_scored_as_empty_predictions": True,
            "selection_rule": (
                "same_source_dev_max_strict_event_f1_then_sensitivity_then_lower_"
                "false_alarms_per_24h_then_higher_threshold"
            ),
            "best_source_dev_diagnostic_point": best,
            "public_repository_threshold_0p8_point": paper_threshold,
            "metric_curve": curve,
            "strict_overlap_and_onset_collar_are_separate": True,
            "szcore_tolerance_is_not_called_onset_accuracy": True,
            "offline_retrospective_only": True,
            "whole_record_future_context_used": True,
            "bidirectional_60s_model_context": True,
            "checkpoint_native_preprocessing_verified": False,
            "checkpoint_channel_axis_verified": False,
            "checkpoint_training_exposure_known": False,
            "TUSZ_source_dev_or_eval_exposure_excluded": False,
            "independent_test_claim_authorized": False,
            "primary_SOTA_claim_authorized": False,
            "receipt_sha256": _PENDING,
        }
    )


__all__ = [
    "COMMON17",
    "DEFAULT_THRESHOLDS",
    "EXPECTED_CHECKPOINT_SHA256",
    "PROVIDER_ID",
    "audit_prediction_inventory",
    "build_target_free_source_dev_projection",
    "decode_official_repository_events",
    "load_projected_model",
    "load_target_free_projection",
    "official_repo_filtered_tiles",
    "predict_preprocessed_record",
    "predict_source_dev",
    "project_state_to_common17",
    "read_official_repo_referential_common17",
    "score_source_dev_predictions",
]
