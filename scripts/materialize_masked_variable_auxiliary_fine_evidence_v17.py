#!/usr/bin/env python3
"""Materialize independent fine temporal evidence for v17 auxiliary events."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soz.data.deepsoz_masked_variable_auxiliary_cache_v17 import (  # noqa: E402
    FORMAL_AUXILIARY_ADMISSION_ARTIFACT_SHA256,
    FORMAL_SIGNAL_UNIVERSE_ARTIFACT_SHA256,
    MANIFEST_FILENAME,
    atomic_publish_safetensors,
    cache_lineage_axes,
    canonical_sha256,
    event_tensor_sha256,
    file_sha256,
    load_auxiliary_cache_contract,
    prepare_output_directory,
    require_formal_cache_scope,
    resolve_raw_root,
    safe_edf_path,
    select_cache_events,
    tensor_bitwise_equal,
    tensor_sha256,
    validate_raw_replay,
)
from src.soz.data.edf import load_standard19_edf_event  # noqa: E402
from src.soz.fine_temporal_evidence import (  # noqa: E402
    FINE_TEMPORAL_EVIDENCE_SCHEMA,
    FINE_TEMPORAL_FEATURE_NAMES,
    extract_fine_temporal_evidence,
)
from src.soz.geometry import STANDARD_19  # noqa: E402


DEFAULT_ADMISSION = (
    ROOT / "outputs/deepsoz_masked_variable_auxiliary_join_v1_20260812"
)
DEFAULT_SIGNAL_UNIVERSE = (
    ROOT / "outputs/deepsoz_target_independent_signal_universe_v1_20260812"
)
DEFAULT_TUSZ_ROOT = Path("/mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf")
DEFAULT_OUTPUT = (
    ROOT / "outputs/deepsoz_masked_variable_auxiliary_fine_evidence_v17_20260812"
)

FULL_SCHEMA = "soz_deepsoz_masked_variable_auxiliary_fine_evidence_v17"
SMOKE_SCHEMA = "soz_deepsoz_masked_variable_auxiliary_fine_evidence_v17_smoke"
TENSOR_FILENAME = "evidence.safetensors"
EVENT_TENSOR_NAMES = (
    "features",
    "composite_trace",
    "dominant_frequency_hz",
    "node_change_detected",
    "node_change_latency_sec",
    "bipolar_change_detected",
    "bipolar_change_latency_sec",
)
SHARED_TENSOR_NAME = "window_center_sec"
EVENT_TENSOR_SHAPES = {
    "features": (19, 20),
    "composite_trace": (19, 237),
    "dominant_frequency_hz": (19, 237),
    "node_change_detected": (19,),
    "node_change_latency_sec": (19,),
    "bipolar_change_detected": (20,),
    "bipolar_change_latency_sec": (20,),
}
EVENT_TENSOR_DTYPES = {
    "features": torch.float32,
    "composite_trace": torch.float32,
    "dominant_frequency_hz": torch.float32,
    "node_change_detected": torch.bool,
    "node_change_latency_sec": torch.float32,
    "bipolar_change_detected": torch.bool,
    "bipolar_change_latency_sec": torch.float32,
}
SHARED_TENSOR_SHAPE = (237,)


def _evidence_values(evidence: object) -> dict[str, torch.Tensor]:
    values = {
        "features": evidence.features.detach().cpu().contiguous(),
        "composite_trace": evidence.composite_trace.detach().cpu().contiguous(),
        "dominant_frequency_hz": (
            evidence.dominant_frequency_hz.detach().cpu().contiguous()
        ),
        "node_change_detected": (
            evidence.node_change_detected.detach().cpu().contiguous()
        ),
        "node_change_latency_sec": (
            evidence.node_change_latency_sec.detach().cpu().contiguous()
        ),
        "bipolar_change_detected": (
            evidence.bipolar_change_detected.detach().cpu().contiguous()
        ),
        "bipolar_change_latency_sec": (
            evidence.bipolar_change_latency_sec.detach().cpu().contiguous()
        ),
    }
    for name, value in values.items():
        if tuple(value.shape) != EVENT_TENSOR_SHAPES[name]:
            raise RuntimeError(f"fine-evidence shape drifted: {name}")
        if value.dtype != EVENT_TENSOR_DTYPES[name]:
            raise RuntimeError(f"fine-evidence dtype drifted: {name}")
    finite_names = ("features", "composite_trace", "dominant_frequency_hz")
    if any(not torch.isfinite(values[name]).all() for name in finite_names):
        raise RuntimeError("fine-evidence model inputs contain non-finite values")
    return values


def materialize(
    *,
    admission_directory: Path,
    signal_universe_directory: Path,
    expected_admission_artifact_sha256: str,
    expected_signal_universe_artifact_sha256: str,
    tusz_root: Path,
    output_directory: Path,
    device: torch.device,
    limit: int | None,
    progress_every: int,
) -> tuple[Path, Mapping[str, object]]:
    if device.type not in {"cpu", "cuda"} or device.index is not None:
        raise ValueError("device must be cpu or cuda without an explicit index")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if type(progress_every) is not int or progress_every < 1:
        raise ValueError("progress_every must be a positive integer")

    contract = load_auxiliary_cache_contract(
        admission_directory,
        signal_universe_directory,
        expected_admission_artifact_sha256=expected_admission_artifact_sha256,
        expected_signal_universe_artifact_sha256=(
            expected_signal_universe_artifact_sha256
        ),
    )
    selected, full_scope = select_cache_events(contract, limit)
    require_formal_cache_scope(contract, selected, full_scope=full_scope)
    raw_root = resolve_raw_root(tusz_root)
    target = prepare_output_directory(
        output_directory,
        input_paths=(contract.admission_path, contract.signal_path, raw_root),
    )

    values_by_name: dict[str, list[torch.Tensor]] = {
        name: [] for name in EVENT_TENSOR_NAMES
    }
    expected_grid: torch.Tensor | None = None
    rows: list[dict[str, object]] = []
    started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    else:
        torch.set_num_threads(max(1, min(torch.get_num_threads(), 4)))

    for ordinal, event in enumerate(selected):
        source = safe_edf_path(raw_root, event["relative_edf_path"])
        loaded = load_standard19_edf_event(
            source,
            float(event["global_t0_sec"]),
            config=contract.preprocess_config,
        )
        replay_sha = validate_raw_replay(loaded, event)
        with torch.inference_mode():
            evidence = extract_fine_temporal_evidence(
                loaded.window.data.to(device),
                sfreq_hz=loaded.window.sfreq_hz,
            )
        values = _evidence_values(evidence)
        grid = evidence.window_center_sec.detach().cpu().contiguous()
        if tuple(grid.shape) != SHARED_TENSOR_SHAPE or grid.dtype != torch.float32:
            raise RuntimeError("fine temporal grid shape/dtype drifted")
        if not torch.isfinite(grid).all():
            raise RuntimeError("fine temporal grid contains non-finite values")
        if expected_grid is None:
            expected_grid = grid
        elif not tensor_bitwise_equal(expected_grid, grid):
            raise RuntimeError("fine temporal grid differs between auxiliary events")
        for name in EVENT_TENSOR_NAMES:
            values_by_name[name].append(values[name])
        rows.append(
            {
                "ordinal": ordinal,
                "event_id": str(event["event_id"]),
                "patient_id": str(event["patient_id"]),
                "official_split": str(event["official_split"]),
                "source_model_split": str(event["source_model_split"]),
                "aux_outer_fold": int(event["aux_outer_fold"]),
                "event_record_sha256": str(event["event_record_sha256"]),
                "admission_event_record_sha256": str(
                    event["admission_event_record_sha256"]
                ),
                "processed_window_sha256": replay_sha,
                "fine_evidence_sha256": event_tensor_sha256(
                    tuple(values[name] for name in EVENT_TENSOR_NAMES)
                ),
                "node_change_detected_count": int(
                    values["node_change_detected"].sum().item()
                ),
                "bipolar_change_detected_count": int(
                    values["bipolar_change_detected"].sum().item()
                ),
            }
        )
        position = ordinal + 1
        if position % progress_every == 0 or position == len(selected):
            elapsed = time.monotonic() - started
            print(
                json.dumps(
                    {
                        "event": position,
                        "total": len(selected),
                        "elapsed_sec": round(elapsed, 2),
                        "seconds_per_event": round(elapsed / position, 3),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    if expected_grid is None:
        raise RuntimeError("fine-evidence materialization selected no events")
    tensors = {
        name: torch.stack(values_by_name[name]).contiguous()
        for name in EVENT_TENSOR_NAMES
    }
    tensors[SHARED_TENSOR_NAME] = expected_grid.contiguous()
    for name in EVENT_TENSOR_NAMES:
        expected_shape = (len(selected), *EVENT_TENSOR_SHAPES[name])
        if tuple(tensors[name].shape) != expected_shape:
            raise RuntimeError(f"complete fine-evidence tensor shape drifted: {name}")
    event_ids = [str(row["event_id"]) for row in rows]
    if event_ids != [str(event["event_id"]) for event in selected]:
        raise RuntimeError("auxiliary fine-evidence event order drifted")
    if full_scope and event_ids != list(contract.event_ids):
        raise RuntimeError("full fine cache is not the admitted auxiliary roster")

    admission_receipt = contract.admission.receipt
    elapsed = time.monotonic() - started
    peak = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0

    def build_manifest(tensor_path: Path) -> Mapping[str, object]:
        specs = {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "tensor_sha256": tensor_sha256(value),
            }
            for name, value in tensors.items()
        }
        return {
            "schema_version": FULL_SCHEMA if full_scope else SMOKE_SCHEMA,
            "evidence_schema_version": FINE_TEMPORAL_EVIDENCE_SCHEMA,
            "purpose": (
                "independent_masked_variable_auxiliary_fine_temporal_evidence_"
                "for_v17_development"
            ),
            "development_only": True,
            "public_confirmation_forbidden": True,
            "full_scope": full_scope,
            "smoke_only": not full_scope,
            "independent_auxiliary_cache": True,
            "event_count": len(selected),
            "patient_count": len({str(event["patient_id"]) for event in selected}),
            "event_ids": event_ids,
            "event_order_sha256": canonical_sha256(event_ids),
            "admitted_event_roster_complete": full_scope
            and tuple(event_ids) == contract.event_ids,
            "events": rows,
            "standard_19": list(STANDARD_19),
            "channel_order": list(STANDARD_19),
            "event_join_key": "event_id",
            "patient_join_key": "patient_id",
            "feature_names": list(FINE_TEMPORAL_FEATURE_NAMES),
            "preprocess_schema": contract.signal.receipt["preprocess_schema"],
            "preprocess_config": asdict(contract.preprocess_config),
            "preprocess_config_sha256": contract.signal.receipt[
                "preprocess_config_sha256"
            ],
            "reference_representation": "primary_reference_then_CAR19",
            "input_shape_per_event": [19, 12_000],
            "features_event_shape": [19, 20],
            "temporal_contract": {
                "input_shape": [19, 12_000],
                "sampling_frequency_hz": 200.0,
                "event_interval_sec": [-12.0, 48.0],
                "analysis_window_sec": 1.0,
                "analysis_stride_sec": 0.25,
                "effective_temporal_resolution_sec": 1.0,
                "output_grid_stride_sec": 0.25,
                "output_grid_points": 237,
                "tusz_t0_is_alignment_not_soz_onset": True,
                "change_is_not_cortical_onset": True,
                "relative_delay_is_not_propagation_truth": True,
            },
            "tensor_file": TENSOR_FILENAME,
            "tensor_file_sha256": file_sha256(tensor_path),
            "tensor_file_size_bytes": tensor_path.stat().st_size,
            "tensor_specs": specs,
            "foundation_backbone": "official_pretrained_LaBraM_Base_not_replaced",
            "foundation_model_invoked_by_this_cache": False,
            "foundation_trainable_parameters_during_materialization": 0,
            "deterministic_fine_extractor_trainable_parameters": 0,
            "materialization_device": str(device),
            "elapsed_sec": elapsed,
            "seconds_per_event": elapsed / len(selected),
            "peak_cuda_memory_bytes": int(peak),
            "lineage_axes": cache_lineage_axes(),
            "lineage": {
                "admission_artifact_sha256": contract.admission.artifact_sha256,
                "admission_receipt_sha256": contract.admission.receipt_sha256,
                "source_target_join_artifact_sha256": admission_receipt[
                    "source_join_artifact_sha256"
                ],
                "source_target_join_receipt_sha256": admission_receipt[
                    "source_join_receipt_sha256"
                ],
                "signal_universe_artifact_sha256": contract.signal.artifact_sha256,
                "signal_universe_receipt_sha256": contract.signal.receipt_sha256,
                "signal_universe_eligible_event_roster_sha256": (
                    contract.signal.receipt["eligible_event_roster_sha256"]
                ),
                "signal_preprocess_config_sha256": contract.signal.receipt[
                    "preprocess_config_sha256"
                ],
                "tusz_root": str(raw_root),
            },
            "cache_separation_receipt": {
                "existing_1149_event_cache_loaded": False,
                "existing_1149_event_tensor_concatenated": False,
                "existing_1149_event_cache_overwritten": False,
                "legacy_reused_event_count": 0,
                "new_auxiliary_event_count": len(selected),
                "output_contains_only_admitted_auxiliary_events": True,
            },
            "access_receipt": {
                "admission_only_artifact_loaded": True,
                "target_bearing_join_artifact_loaded": False,
                "direct_target_values_loaded": False,
                "upstream_target_conditioned_roster": True,
                "raw_public_eeg_loaded": True,
                "raw_public_event_count": len(selected),
                "tusz_channel_annotation_values_loaded": False,
                "historical_prediction_artifacts_loaded": False,
                "stable_1149_representation_cache_loaded": False,
                "private_eeg_loaded": False,
                "private_target_values_loaded": False,
                "foundation_training_performed": False,
                "foundation_trainable_parameters": 0,
                "reasoner_training_performed": False,
            },
        }

    return atomic_publish_safetensors(
        target,
        tensor_filename=TENSOR_FILENAME,
        tensors=tensors,
        build_manifest=build_manifest,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--admission-directory", type=Path, default=DEFAULT_ADMISSION)
    parser.add_argument(
        "--signal-universe-directory", type=Path, default=DEFAULT_SIGNAL_UNIVERSE
    )
    parser.add_argument(
        "--expected-admission-artifact-sha256",
        default=FORMAL_AUXILIARY_ADMISSION_ARTIFACT_SHA256,
    )
    parser.add_argument(
        "--expected-signal-universe-artifact-sha256",
        default=FORMAL_SIGNAL_UNIVERSE_ARTIFACT_SHA256,
    )
    parser.add_argument("--tusz-root", type=Path, default=DEFAULT_TUSZ_ROOT)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    path, manifest = materialize(
        admission_directory=args.admission_directory,
        signal_universe_directory=args.signal_universe_directory,
        expected_admission_artifact_sha256=(
            args.expected_admission_artifact_sha256
        ),
        expected_signal_universe_artifact_sha256=(
            args.expected_signal_universe_artifact_sha256
        ),
        tusz_root=args.tusz_root,
        output_directory=args.output_directory,
        device=torch.device(args.device),
        limit=args.limit,
        progress_every=args.progress_every,
    )
    print(
        json.dumps(
            {
                "status": "MASKED_VARIABLE_AUXILIARY_FINE_EVIDENCE_V17_MATERIALIZED",
                "path": str(path),
                "manifest_sha256": file_sha256(path / MANIFEST_FILENAME),
                "event_count": manifest["event_count"],
                "full_scope": manifest["full_scope"],
                "independent_auxiliary_cache": True,
                "target_values_loaded": False,
                "private_loaded": False,
                "foundation_trainable_parameters": 0,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
