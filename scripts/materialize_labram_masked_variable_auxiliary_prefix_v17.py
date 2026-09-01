#!/usr/bin/env python3
"""Materialize an independent admitted-auxiliary LaBraM block-9 cache."""

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
from src.soz.geometry import STANDARD_19  # noqa: E402
from src.soz.models.labram import (  # noqa: E402
    AUDITED_LABRAM_BASE_SHA256,
    AUDITED_LABRAM_MODELING_SHA256,
    LABRAM_RAW_HEADER_POSITION_BINDING_POLICY,
    OfficialLaBraMEncoder,
    bind_labram_record_positions,
)
from src.soz.models.labram_peft import (  # noqa: E402
    OfficialLaBraMFrozenPrefixEncoder,
    OfficialLaBraMMinimalPEFTSuffix,
)


DEFAULT_ADMISSION = (
    ROOT / "outputs/deepsoz_masked_variable_auxiliary_join_v1_20260812"
)
DEFAULT_SIGNAL_UNIVERSE = (
    ROOT / "outputs/deepsoz_target_independent_signal_universe_v1_20260812"
)
DEFAULT_TUSZ_ROOT = Path("/mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf")
DEFAULT_MODELING = Path("/mnt/hd1/dyf/workspace/LaBraM/modeling_finetune.py")
DEFAULT_CHECKPOINT = Path(
    "/mnt/hd1/dyf/workspace/LaBraM/checkpoints/labram-base.pth"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/deepsoz_masked_variable_auxiliary_labram_prefix_v17_20260812"
)

FULL_SCHEMA = "soz_deepsoz_masked_variable_auxiliary_labram_prefix_v17"
SMOKE_SCHEMA = "soz_deepsoz_masked_variable_auxiliary_labram_prefix_v17_smoke"
TENSOR_FILENAME = "prefix.safetensors"
PREFIX_TENSOR_NAME = "prefix_tokens"
PREFIX_EVENT_SHAPE = (15, 77, 200)
FOUNDATION_PREFIX_BLOCKS = tuple(range(10))


def _split_calls(eeg: torch.Tensor) -> torch.Tensor:
    if tuple(eeg.shape) != (19, 12_000) or eeg.dtype != torch.float32:
        raise ValueError("LaBraM auxiliary event must be float32 [19,12000]")
    calls = eeg.reshape(19, 15, 4, 200).permute(1, 0, 2, 3).contiguous()
    restored = calls.permute(1, 0, 2, 3).reshape(19, 12_000)
    if not tensor_bitwise_equal(restored, eeg):
        raise RuntimeError("LaBraM four-second calls do not reassemble exactly")
    return calls


def materialize(
    *,
    admission_directory: Path,
    signal_universe_directory: Path,
    expected_admission_artifact_sha256: str,
    expected_signal_universe_artifact_sha256: str,
    tusz_root: Path,
    modeling_path: Path,
    checkpoint_path: Path,
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
    model_path = Path(modeling_path).resolve(strict=True)
    checkpoint = Path(checkpoint_path).resolve(strict=True)
    target = prepare_output_directory(
        output_directory,
        input_paths=(
            contract.admission_path,
            contract.signal_path,
            raw_root,
            model_path,
            checkpoint,
        ),
    )

    prefix_encoder = OfficialLaBraMFrozenPrefixEncoder(
        modeling_path=model_path,
        checkpoint_path=checkpoint,
        expected_sha256=AUDITED_LABRAM_BASE_SHA256,
        expected_modeling_sha256=AUDITED_LABRAM_MODELING_SHA256,
    ).to(device).eval()
    if any(parameter.requires_grad for parameter in prefix_encoder.parameters()):
        raise RuntimeError("frozen LaBraM blocks 0-9 expose trainable weights")

    rows: list[dict[str, object]] = []
    prefixes: list[torch.Tensor] = []
    equivalence_error: float | None = None
    started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for ordinal, event in enumerate(selected):
        source = safe_edf_path(raw_root, event["relative_edf_path"])
        loaded = load_standard19_edf_event(
            source,
            float(event["global_t0_sec"]),
            config=contract.preprocess_config,
        )
        replay_sha = validate_raw_replay(loaded, event)
        calls = _split_calls(loaded.window.data).to(device)
        binding = bind_labram_record_positions(
            loaded.edf_receipt.raw_channel_names,
            semantic_channels=loaded.edf_receipt.semantic_channels,
        )
        with torch.inference_mode():
            prefix = prefix_encoder.forward_with_record_binding(calls, binding)
        prefix = prefix.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if tuple(prefix.shape) != PREFIX_EVENT_SHAPE:
            raise RuntimeError("LaBraM block-9 prefix shape drifted")
        if not torch.isfinite(prefix).all():
            raise RuntimeError("LaBraM block-9 prefix contains non-finite values")

        if equivalence_error is None:
            suffix = OfficialLaBraMMinimalPEFTSuffix(
                modeling_path=model_path,
                checkpoint_path=checkpoint,
                expected_sha256=AUDITED_LABRAM_BASE_SHA256,
                expected_modeling_sha256=AUDITED_LABRAM_MODELING_SHA256,
            ).to(device).eval()
            official = OfficialLaBraMEncoder(
                modeling_path=model_path,
                checkpoint_path=checkpoint,
                expected_sha256=AUDITED_LABRAM_BASE_SHA256,
                expected_modeling_sha256=AUDITED_LABRAM_MODELING_SHA256,
                tile_seconds=4,
                position_names=binding.position_names,
            ).to(device).eval()
            for parameter in official.parameters():
                parameter.requires_grad_(False)
            # The zero-initialized suffix deliberately exposes exactly its four
            # LoRA tensors as trainable; its forward method enforces that
            # contract.  ``inference_mode`` below guarantees that this
            # operator-level equivalence control performs no training.  The
            # official comparison encoder itself must remain fully frozen.
            if suffix.n_trainable_parameters != 6_400 or any(
                parameter.requires_grad for parameter in official.parameters()
            ):
                raise RuntimeError("LaBraM equivalence model scope changed")
            with torch.inference_mode():
                restored = suffix(prefix.to(device))
                expected = official(calls)
            equivalence_error = float((restored - expected).abs().amax().cpu())
            if equivalence_error > 1e-6:
                raise RuntimeError(
                    "block-9 prefix does not recover official LaBraM output: "
                    f"{equivalence_error}"
                )
            del suffix, official, restored, expected

        prefixes.append(prefix)
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
                "position_binding_policy": (
                    LABRAM_RAW_HEADER_POSITION_BINDING_POLICY
                ),
                "position_names": list(binding.position_names),
                "position_ids": list(binding.position_ids),
                "prefix_tensor_sha256": tensor_sha256(prefix),
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

    prefix_tensor = torch.stack(prefixes).contiguous()
    if tuple(prefix_tensor.shape) != (len(selected), *PREFIX_EVENT_SHAPE):
        raise RuntimeError("auxiliary prefix cache has the wrong complete shape")
    event_ids = [str(row["event_id"]) for row in rows]
    if event_ids != [str(event["event_id"]) for event in selected]:
        raise RuntimeError("auxiliary prefix event order drifted")
    if full_scope and event_ids != list(contract.event_ids):
        raise RuntimeError("full prefix cache is not the admitted auxiliary roster")
    if equivalence_error is None:
        raise RuntimeError("LaBraM equivalence check did not execute")

    admission_receipt = contract.admission.receipt
    elapsed = time.monotonic() - started
    peak = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0

    def build_manifest(tensor_path: Path) -> Mapping[str, object]:
        return {
            "schema_version": FULL_SCHEMA if full_scope else SMOKE_SCHEMA,
            "purpose": (
                "independent_masked_variable_auxiliary_frozen_labram_block9_"
                "prefix_for_v17_development"
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
            "preprocess_schema": contract.signal.receipt["preprocess_schema"],
            "preprocess_config": asdict(contract.preprocess_config),
            "preprocess_config_sha256": contract.signal.receipt[
                "preprocess_config_sha256"
            ],
            "reference_representation": "primary_reference_then_CAR19",
            "foundation_backbone": "official_pretrained_LaBraM_Base_not_replaced",
            "foundation_checkpoint_sha256": AUDITED_LABRAM_BASE_SHA256,
            "foundation_modeling_sha256": AUDITED_LABRAM_MODELING_SHA256,
            "foundation_prefix_blocks": list(FOUNDATION_PREFIX_BLOCKS),
            "foundation_prefix_stop_exclusive": 10,
            "foundation_trainable_parameters_during_materialization": 0,
            "position_binding_policy": LABRAM_RAW_HEADER_POSITION_BINDING_POLICY,
            "input_shape_per_event": [19, 12_000],
            "sampling_frequency_hz": 200.0,
            "event_interval_sec": [-12.0, 48.0],
            "call_count_per_event": 15,
            "call_duration_sec": 4.0,
            "call_input_shape": [19, 4, 200],
            "call_output_shape": [77, 200],
            "prefix_event_shape": list(PREFIX_EVENT_SHAPE),
            "prefix_tensor_shape": list(prefix_tensor.shape),
            "tensor_name": PREFIX_TENSOR_NAME,
            "tensor_file": TENSOR_FILENAME,
            "tensor_file_sha256": file_sha256(tensor_path),
            "tensor_file_size_bytes": tensor_path.stat().st_size,
            "prefix_tensor_sha256": tensor_sha256(prefix_tensor),
            "tensor_specs": {
                PREFIX_TENSOR_NAME: {
                    "shape": list(prefix_tensor.shape),
                    "dtype": str(prefix_tensor.dtype),
                    "tensor_sha256": tensor_sha256(prefix_tensor),
                }
            },
            "zero_adapter_official_equivalence_max_abs_error": equivalence_error,
            "zero_adapter_official_equivalence_verified": True,
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
        tensors={PREFIX_TENSOR_NAME: prefix_tensor},
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
    parser.add_argument("--modeling-path", type=Path, default=DEFAULT_MODELING)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress-every", type=int, default=25)
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
        modeling_path=args.modeling_path,
        checkpoint_path=args.checkpoint_path,
        output_directory=args.output_directory,
        device=torch.device(args.device),
        limit=args.limit,
        progress_every=args.progress_every,
    )
    print(
        json.dumps(
            {
                "status": "LABRAM_MASKED_VARIABLE_AUXILIARY_PREFIX_V17_MATERIALIZED",
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
