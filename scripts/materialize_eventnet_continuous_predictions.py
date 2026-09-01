#!/usr/bin/env python3
"""Materialize EEG-only EventNet predictions on public source-train/dev records.

The legacy 652-record CSV input remains supported.  A complete-roster route
also accepts a validated identity-only projection and verifies each EDF's
container SHA-256 before inference.  Seizure intervals, annotations, SOZ
labels and clinical fields are never read or passed to the provider.
Source-eval identities may exist in the projection, but execution is sealed
until a separate host admission receipt is implemented.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import hashlib
import json
import platform
from pathlib import Path, PurePosixPath
import sys
import tempfile
import time
from typing import Any, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording.canonical_edf_materialization import (  # noqa: E402
    load_canonical_edf_record,
)
from src.clinical_eeg_long_recording.continuous_detection_stage_p_runner_v1 import (  # noqa: E402
    build_stage_p_run_contract_v1,
    run_stage_p_prediction_batch_v1,
)
from src.clinical_eeg_long_recording.detector_provider_contract import (  # noqa: E402
    authorize_provider_execution,
)
from src.clinical_eeg_long_recording.eventnet_full_record_adapter import (  # noqa: E402
    EVENTNET_ADAPTER_METHOD_ID,
    EVENTNET_CHECKPOINT_SHA256,
    EVENTNET_PROVIDER_ID,
    EventNetFullRecordSession,
    eventnet_research_provider_definition,
    validate_eventnet_prediction_receipt,
)
from src.clinical_eeg_long_recording.eventnet_continuous_benchmark_projection import (  # noqa: E402
    project_eventnet_prediction_to_benchmark,
    validate_eventnet_benchmark_prediction_projection,
)
from src.clinical_eeg_long_recording.eventnet_tusz_complete_identity_projection_v1 import (  # noqa: E402
    validate_eventnet_tusz_complete_identity_projection_v1,
)
from src.clinical_eeg_long_recording.tusz_complete_detector_roster_v1 import (  # noqa: E402
    inspect_edf_container_header_v1,
)
from src.clinical_eeg_long_recording.tusz_complete_detector_roster_v2 import (  # noqa: E402
    TUSZ_ANALYSIS_IDENTITY_PROJECTION_V2_SCHEMA_VERSION,
    validate_tusz_analysis_identity_projection_v2,
)


DEFAULT_MANIFEST = ROOT / "deepsoz_tusz_652_record_manifest.csv"
DEFAULT_TUSZ_ROOT = Path("/mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf")
DEFAULT_CHECKPOINT = ROOT / "models/eventnet_2024_official/model.pth"
DEFAULT_OUTPUT = ROOT / "outputs/eventnet_2024_public_stage_p_v1"
STAGE_P_INPUT_RECEIPT_SCHEMA_VERSION = "eventnet_stage_p_input_receipt_v1"


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _selected_rows(
    manifest: Path,
    *,
    split: str,
    recording_id: str | None,
    max_records: int,
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {
            "model_split",
            "local_edf_path",
        }.issubset(reader.fieldnames):
            raise ValueError("public manifest lacks split/path fields")
        for source in reader:
            row_split = str(source["model_split"]).strip()
            row_path = str(source["local_edf_path"]).strip()
            if row_split != split:
                continue
            if recording_id is not None and row_path != recording_id:
                continue
            # Immediate input-only projection: no other source column survives.
            selected.append({"model_split": row_split, "local_edf_path": row_path})
    selected.sort(key=lambda row: row["local_edf_path"])
    if max_records > 0:
        selected = selected[:max_records]
    if not selected:
        raise ValueError("no public record matched the requested smoke scope")
    paths = [row["local_edf_path"] for row in selected]
    if len(paths) != len(set(paths)):
        raise ValueError("selected public recording paths are not unique")
    return selected


def _load_complete_roster_projection(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(
            "complete-roster projection must be a regular non-symlink JSON file"
        )
    raw = path.read_bytes()
    if not raw:
        raise ValueError("complete-roster projection JSON is empty")
    value = json.loads(
        raw.decode("utf-8"), object_pairs_hook=_no_duplicate_object
    )
    if value.get("schema_version") == (
        TUSZ_ANALYSIS_IDENTITY_PROJECTION_V2_SCHEMA_VERSION
    ):
        return validate_tusz_analysis_identity_projection_v2(value)
    return validate_eventnet_tusz_complete_identity_projection_v1(value)


def _selected_complete_roster_rows(
    projection: dict[str, Any],
    *,
    split: str,
    recording_id: str | None,
    max_records: int,
) -> list[dict[str, str]]:
    selected = [
        deepcopy(row)
        for row in projection["records"]
        if row["model_split"] == split
        and (recording_id is None or row["local_edf_path"] == recording_id)
    ]
    selected.sort(key=lambda row: row["local_edf_path"])
    if max_records > 0:
        selected = selected[:max_records]
    if not selected:
        raise ValueError("no complete-roster record matched the requested scope")
    return selected


def _authorize_projected_split_execution(
    projection: dict[str, Any], split: str
) -> None:
    if projection.get("schema_version") == (
        TUSZ_ANALYSIS_IDENTITY_PROJECTION_V2_SCHEMA_VERSION
    ):
        permission = projection["role_permissions"][split]
        if split == "source_eval":
            if (
                permission["locked_evaluation_identity_export_authorized"]
                is not True
                or permission["host_admission_required"] is not True
                or permission["reference_access_authorized"] is not False
            ):
                raise ValueError("source-eval analysis-v2 permission receipt drifted")
            raise PermissionError(
                "source_eval identity export is valid, but EventNet model execution "
                "requires a host admission receipt; this CLI accepts no bypass"
            )
        role_field = (
            "model_fit_identity_authorized"
            if split == "source_train"
            else "development_calibration_identity_authorized"
        )
        if (
            permission[role_field] is not True
            or permission["host_admission_required"] is not False
            or permission["reference_access_authorized"] is not False
            or permission["model_execution_authorized_by_projection"] is not False
        ):
            raise PermissionError(
                f"analysis-v2 identity role is not admissible for {split}"
            )
        # The identity projection intentionally does not authorize a particular
        # provider.  EventNet's separate research execution receipt below does.
        return
    permission = projection["split_permissions"][split]
    if split == "source_eval":
        if (
            permission["identity_export_authorized"] is not True
            or permission["eventnet_model_execution_authorized"] is not False
            or permission["eventnet_model_execution_admission_required"] is not True
            or permission["host_admission_receipt_present"] is not False
        ):
            raise ValueError("source-eval projection permission receipt drifted")
        raise PermissionError(
            "source_eval identity export is valid, but EventNet model execution "
            "requires a host admission receipt; this CLI accepts no bypass"
        )
    if (
        permission["identity_export_authorized"] is not True
        or permission["eventnet_model_execution_authorized"] is not True
        or permission["eventnet_model_execution_admission_required"] is not False
        or permission["host_admission_receipt_present"] is not False
    ):
        raise PermissionError(f"EventNet execution is not authorized for {split}")


def _stable_file_identity(path: Path) -> tuple[int, int, int, int, int]:
    value = path.stat()
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _verify_projected_edf_container(
    path: Path, expected_sha256: str
) -> tuple[int, int, int, int, int]:
    before = _stable_file_identity(path)
    observed = _file_sha256(path)
    after = _stable_file_identity(path)
    if before != after:
        raise ValueError("projected EDF changed during container verification")
    if observed != expected_sha256:
        raise ValueError("projected EDF container SHA-256 drifted")
    return after


def _safe_edf(root: Path, relative: str) -> Path:
    value = PurePosixPath(relative)
    if value.is_absolute() or ".." in value.parts or value.suffix.lower() != ".edf":
        raise ValueError("unsafe public EDF relative path")
    resolved_root = root.resolve(strict=True)
    resolved = resolved_root.joinpath(*value.parts).resolve(strict=True)
    resolved.relative_to(resolved_root)
    return resolved


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(content)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        np.savez_compressed(handle, **arrays)
        temporary = Path(handle.name)
    temporary.replace(path)


def _strict_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a regular JSON artifact: {path}")
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_no_duplicate_object,
        parse_constant=lambda item: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {item}")
        ),
    )
    if not isinstance(value, dict):
        raise TypeError(f"JSON artifact must contain an object: {path}")
    return value


def _write_json_once_or_validate(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        if _strict_json(path) != dict(value):
            raise ValueError(f"resumed Stage-P input artifact drifted: {path}")
        return
    _atomic_json(path, value)


def _runtime_hardware_receipt(device: str) -> dict[str, Any]:
    requested = str(device)
    cuda_index: int | None = None
    cuda_name: str | None = None
    cuda_capability: list[int] | None = None
    if requested.startswith("cuda") and torch.cuda.is_available():
        cuda_index = torch.device(requested).index
        if cuda_index is None:
            cuda_index = torch.cuda.current_device()
        cuda_name = torch.cuda.get_device_name(cuda_index)
        cuda_capability = list(torch.cuda.get_device_capability(cuda_index))
    body: dict[str, Any] = {
        "schema_version": "eventnet_stage_p_runtime_hardware_contract_v1",
        "device_request": requested,
        "python_version": platform.python_version(),
        "platform_machine": platform.machine(),
        "platform_system": platform.system(),
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_runtime_version": torch.version.cuda,
        "cuda_device_index": cuda_index,
        "cuda_device_name": cuda_name,
        "cuda_compute_capability": cuda_capability,
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    body["receipt_sha256"] = _canonical_sha256(body)
    return body


def _preflight_stage_p_records(
    rows: list[dict[str, str]],
    *,
    tusz_root: Path,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Path],
    dict[str, tuple[int, int, int, int, int]],
]:
    """Bind duration/container identity without opening any reference sidecar."""

    records: list[dict[str, Any]] = []
    paths: dict[str, Path] = {}
    stable_identities: dict[str, tuple[int, int, int, int, int]] = {}
    for row in rows:
        recording_id = row["local_edf_path"]
        edf_path = _safe_edf(tusz_root, recording_id)
        expected_container = row.get("source_edf_container_sha256")
        if expected_container is None:
            before = _stable_file_identity(edf_path)
            expected_container = _file_sha256(edf_path)
            after = _stable_file_identity(edf_path)
            if before != after:
                raise ValueError("legacy EDF changed during container hashing")
            stable = after
        else:
            stable = _verify_projected_edf_container(
                edf_path, expected_container
            )
        header = inspect_edf_container_header_v1(edf_path)
        if _stable_file_identity(edf_path) != stable:
            raise ValueError("EDF changed during Stage-P duration preflight")
        analysis_identity = row.get("analysis_identity_id") or (
            "TUSZ-STAGEP-"
            + _canonical_sha256(
                {
                    "recording_id": recording_id,
                    "source_edf_container_sha256": expected_container,
                }
            )[:24]
        )
        records.append(
            {
                "recording_id": recording_id,
                "analysis_identity": analysis_identity,
                "model_split": row["model_split"],
                "source_edf_container_sha256": expected_container,
                "recording_duration_fraction": header[
                    "recording_duration_fraction"
                ],
            }
        )
        paths[recording_id] = edf_path
        stable_identities[recording_id] = stable
    return records, paths, stable_identities


def _stage_p_source_projection_sha256(
    *,
    records: list[dict[str, Any]],
    complete_projection: Mapping[str, Any] | None,
) -> str:
    return _canonical_sha256(
        {
            "schema_version": "eventnet_stage_p_reference_free_source_projection_v1",
            "upstream_complete_projection_receipt_sha256": (
                complete_projection["receipt_sha256"]
                if complete_projection is not None
                else None
            ),
            "records": records,
            "reference_files_opened": 0,
            "edf_annotations_read": False,
            "spreadsheet_or_clinical_text_read": False,
        }
    )


def _stage_p_artifact_paths(
    terminal: Mapping[str, Any], record_directory: Path
) -> dict[str, Path]:
    rows = terminal["artifacts"]
    by_semantic = {
        str(row["semantic"]): record_directory.joinpath(
            *PurePosixPath(str(row["relative_path"])).parts
        )
        for row in rows
    }
    expected = {
        "eventnet_benchmark_prediction_projection",
        "eventnet_prediction_receipt",
        "eventnet_prediction_tensors",
    }
    if set(by_semantic) != expected:
        raise ValueError("EventNet Stage-P artifact semantics drifted")
    return by_semantic


def _validate_stage_p_eventnet_resume(
    record: Mapping[str, Any],
    terminal: Mapping[str, Any],
    record_directory: Path,
) -> None:
    """Replay provider-specific prediction/tensor lineage before resume skip."""

    if terminal["outcome_status"] == "technical_failure":
        return
    paths = _stage_p_artifact_paths(terminal, record_directory)
    prediction = validate_eventnet_prediction_receipt(
        _strict_json(paths["eventnet_prediction_receipt"])
    )
    projection = validate_eventnet_benchmark_prediction_projection(
        _strict_json(paths["eventnet_benchmark_prediction_projection"])
    )
    if (
        prediction["recording_id"] != record["recording_id"]
        or prediction["receipt_sha256"]
        != terminal["provider_prediction_receipt_sha256"]
        or projection["prediction_receipt_sha256"]
        != prediction["receipt_sha256"]
        or prediction["runtime_receipt"]["service_state"]
        != terminal["service_state"]
    ):
        raise ValueError("EventNet Stage-P prediction identity drifted")
    expected_projection_state = (
        "cold" if terminal["service_state"] == "cold_process_start" else "warm"
    )
    if projection["execution_receipt"]["service_state"] != expected_projection_state:
        raise ValueError("EventNet Stage-P warm/cold projection drifted")
    generic = prediction["generic_full_record_result"]
    if _canonical_sha256(generic) != terminal["provider_result_receipt_sha256"]:
        raise ValueError("EventNet Stage-P provider-result binding drifted")
    runtime = prediction["runtime_receipt"]
    if (
        runtime["checkpoint_load_included_in_full_adapter_wall"] is not False
        or runtime["checkpoint_static_audit_and_safe_load_seconds"] != 0.0
    ):
        raise ValueError("EventNet Stage-P warm-session timing drifted")

    tensor_path = paths["eventnet_prediction_tensors"]
    if tensor_path.is_symlink() or not tensor_path.is_file():
        raise ValueError("EventNet Stage-P tensor artifact is not a regular file")
    with np.load(tensor_path, allow_pickle=False) as archive:
        names = {
            "center_probability",
            "duration_fraction",
            "smoothed_center_probability",
        }
        if set(archive.files) != names:
            raise ValueError("EventNet Stage-P tensor keys drifted")
        for name in sorted(names):
            array = np.ascontiguousarray(archive[name], dtype="<f4")
            source = prediction["output_tensors"][name]
            if (
                list(array.shape) != source["shape"]
                or hashlib.sha256(array.tobytes(order="C")).hexdigest()
                != source["payload_sha256"]
            ):
                raise ValueError(f"EventNet Stage-P tensor payload drifted: {name}")


class _EventNetStagePProcessor:
    def __init__(
        self,
        *,
        paths: Mapping[str, Path],
        stable_identities: Mapping[str, tuple[int, int, int, int, int]],
        checkpoint_path: Path,
        provider_execution_receipt: Mapping[str, Any],
        device: str,
    ) -> None:
        self.paths = dict(paths)
        self.stable_identities = dict(stable_identities)
        self.checkpoint_path = checkpoint_path
        self.provider_execution_receipt = dict(provider_execution_receipt)
        self.device = device
        self.session: EventNetFullRecordSession | None = None

    def __call__(
        self,
        record: Mapping[str, Any],
        artifact_directory: Path,
        service_state: str,
    ) -> dict[str, Any]:
        recording_id = str(record["recording_id"])
        edf_path = self.paths[recording_id]
        stable = self.stable_identities[recording_id]
        if _stable_file_identity(edf_path) != stable:
            stable = _verify_projected_edf_container(
                edf_path, str(record["source_edf_container_sha256"])
            )
            self.stable_identities[recording_id] = stable

        canonical_started = time.perf_counter()
        canonical_record = load_canonical_edf_record(edf_path)
        canonical_seconds = time.perf_counter() - canonical_started
        if _stable_file_identity(edf_path) != stable:
            raise ValueError("EDF changed during canonical signal materialization")
        if self.session is None:
            self.session = EventNetFullRecordSession(
                checkpoint_path=self.checkpoint_path,
                provider_execution_receipt=self.provider_execution_receipt,
                device=self.device,
            )
        prediction = self.session.predict(
            canonical_record=canonical_record,
            recording_id=recording_id,
            service_state=service_state,
        )
        receipt = validate_eventnet_prediction_receipt(prediction.receipt)
        projection = project_eventnet_prediction_to_benchmark(
            receipt,
            edf_io_seconds=canonical_seconds,
            service_state=(
                "cold" if service_state == "cold_process_start" else "warm"
            ),
        )
        receipt_path = artifact_directory / "prediction_receipt.json"
        projection_path = artifact_directory / "benchmark_prediction_projection.json"
        tensor_path = artifact_directory / "prediction_tensors.npz"
        _atomic_json(receipt_path, receipt)
        _atomic_json(projection_path, projection)
        _atomic_npz(
            tensor_path,
            center_probability=prediction.center_probability,
            duration_fraction=prediction.duration_fraction,
            smoothed_center_probability=prediction.smoothed_center_probability,
        )
        runtime = receipt["runtime_receipt"]
        generic = receipt["generic_full_record_result"]
        coverage = generic["coverage_receipt"]
        artifacts = [
            {
                "relative_path": projection_path.name,
                "file_sha256": _file_sha256(projection_path),
                "semantic": "eventnet_benchmark_prediction_projection",
            },
            {
                "relative_path": receipt_path.name,
                "file_sha256": _file_sha256(receipt_path),
                "semantic": "eventnet_prediction_receipt",
            },
            {
                "relative_path": tensor_path.name,
                "file_sha256": _file_sha256(tensor_path),
                "semantic": "eventnet_prediction_tensors",
            },
        ]
        artifacts.sort(key=lambda row: (row["semantic"], row["relative_path"]))
        return {
            "outcome_status": generic["outcome_status"],
            "recording_duration_seconds": generic["recording_duration_seconds"],
            "event_proposal_count": receipt["decoder_receipt"][
                "event_proposal_count"
            ],
            "modeled_target_coverage_seconds": coverage[
                "modeled_target_coverage_seconds"
            ],
            "provider_result_receipt_sha256": _canonical_sha256(generic),
            "provider_prediction_receipt_sha256": receipt["receipt_sha256"],
            "timing_seconds": {
                "edf_io_seconds": canonical_seconds,
                "preprocessing_seconds": (
                    runtime["canonical_carrier_binding_seconds"]
                    + runtime["provider_preprocessing_and_tiling_seconds"]
                ),
                "inference_seconds": runtime["model_inference_seconds"],
                "decoder_seconds": runtime["direct_event_decoding_seconds"],
                "end_to_end_seconds": (
                    canonical_seconds + runtime["full_adapter_wall_seconds"]
                ),
            },
            "artifacts": artifacts,
        }


def _guard_stage_p_output(output: Path) -> None:
    if not output.exists():
        return
    if output.is_symlink() or not output.is_dir():
        raise ValueError("Stage-P output must be a regular directory")
    entries = {path.name for path in output.iterdir()}
    if entries and "run_contract.json" not in entries:
        allowed_interrupted_preflight = {"stage_p_input_receipt.json"}
        if not entries.issubset(allowed_interrupted_preflight):
            raise ValueError(
                "refusing to mix Stage-P artifacts with a legacy/non-Stage-P output"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        required=True,
        choices=("source_train", "source_dev", "source_eval"),
    )
    parser.add_argument("--recording-id")
    parser.add_argument("--max-records", type=int, default=0)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--manifest", type=Path)
    source.add_argument("--complete-roster-projection", type=Path)
    parser.add_argument("--tusz-root", type=Path, default=DEFAULT_TUSZ_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args()
    if arguments.max_records < 0:
        raise ValueError("max-records must be non-negative")

    complete_projection: dict[str, Any] | None = None
    if arguments.complete_roster_projection is not None:
        complete_projection = _load_complete_roster_projection(
            arguments.complete_roster_projection
        )
        rows = _selected_complete_roster_rows(
            complete_projection,
            split=arguments.split,
            recording_id=arguments.recording_id,
            max_records=arguments.max_records,
        )
        _authorize_projected_split_execution(
            complete_projection, arguments.split
        )
    else:
        if arguments.split == "source_eval":
            raise PermissionError(
                "legacy manifest execution supports source_train/source_dev only; "
                "source_eval remains sealed"
            )
        rows = _selected_rows(
            arguments.manifest or DEFAULT_MANIFEST,
            split=arguments.split,
            recording_id=arguments.recording_id,
            max_records=arguments.max_records,
        )
    provider_definition = eventnet_research_provider_definition()
    execution = authorize_provider_execution(
        provider_definition,
        requested_role="research",
    )
    output = arguments.output_directory.resolve()
    _guard_stage_p_output(output)
    output.mkdir(parents=True, exist_ok=True)
    records, paths, stable_identities = _preflight_stage_p_records(
        rows,
        tusz_root=arguments.tusz_root,
    )
    checkpoint_sha256 = _file_sha256(arguments.checkpoint)
    if checkpoint_sha256 != EVENTNET_CHECKPOINT_SHA256:
        raise ValueError("EventNet checkpoint SHA-256 drifted before Stage-P")
    runtime_hardware = _runtime_hardware_receipt(arguments.device)
    source_projection_sha256 = _stage_p_source_projection_sha256(
        records=records,
        complete_projection=complete_projection,
    )
    preprocessing_contract_sha256 = _canonical_sha256(
        {
            "schema_version": "eventnet_stage_p_preprocessing_binding_v1",
            "adapter_method_id": EVENTNET_ADAPTER_METHOD_ID,
            "adapter_code_sha256": provider_definition["adapter_code_sha256"],
            "canonical_physical_signal_only": True,
        }
    )
    decoder_contract_sha256 = _canonical_sha256(
        {
            "schema_version": "eventnet_stage_p_decoder_binding_v1",
            "adapter_method_id": EVENTNET_ADAPTER_METHOD_ID,
            "adapter_code_sha256": provider_definition["adapter_code_sha256"],
            "direct_event_decoder": True,
        }
    )
    run_contract = build_stage_p_run_contract_v1(
        provider_id=EVENTNET_PROVIDER_ID,
        model_split=arguments.split,
        records=records,
        source_identity_projection_sha256=source_projection_sha256,
        provider_execution_receipt_sha256=_canonical_sha256(execution),
        checkpoint_sha256=checkpoint_sha256,
        provider_code_sha256=provider_definition["adapter_code_sha256"],
        preprocessing_contract_sha256=preprocessing_contract_sha256,
        decoder_contract_sha256=decoder_contract_sha256,
        runtime_hardware_contract_sha256=runtime_hardware["receipt_sha256"],
    )
    input_receipt: dict[str, Any] = {
        "schema_version": STAGE_P_INPUT_RECEIPT_SCHEMA_VERSION,
        "provider_id": EVENTNET_PROVIDER_ID,
        "model_split": arguments.split,
        "selected_record_count": len(records),
        "source_identity_projection_sha256": source_projection_sha256,
        "complete_roster_projection_receipt_sha256": (
            complete_projection["receipt_sha256"]
            if complete_projection is not None
            else None
        ),
        "provider_execution_receipt": execution,
        "checkpoint_sha256": checkpoint_sha256,
        "preprocessing_contract_sha256": preprocessing_contract_sha256,
        "decoder_contract_sha256": decoder_contract_sha256,
        "runtime_hardware_receipt": runtime_hardware,
        "run_contract_sha256": run_contract["contract_sha256"],
        "reference_files_opened": 0,
        "edf_annotations_read": False,
        "spreadsheet_or_clinical_text_read": False,
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    input_receipt["receipt_sha256"] = _canonical_sha256(input_receipt)
    _write_json_once_or_validate(
        output / "stage_p_input_receipt.json", input_receipt
    )
    processor = _EventNetStagePProcessor(
        paths=paths,
        stable_identities=stable_identities,
        checkpoint_path=arguments.checkpoint,
        provider_execution_receipt=execution,
        device=arguments.device,
    )
    batch = run_stage_p_prediction_batch_v1(
        output_directory=output,
        run_contract=run_contract,
        processor=processor,
        resume_validator=_validate_stage_p_eventnet_resume,
    )
    print(
        f"batch={batch['batch_id']} records={batch['terminal_record_count']} "
        f"outcomes={batch['outcome_counts']} "
        f"warm_rtf={batch['warm_end_to_end_rtf']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
