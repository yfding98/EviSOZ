#!/usr/bin/env python3
"""Append only recovered events to the frozen v11 LaBraM block-9 cache."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
import time
from typing import Mapping, Sequence

import torch
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soz.data.edf import CausalEDFConfig, load_standard19_edf_event  # noqa: E402
from src.soz.data.identity_v12_cache_extension import (  # noqa: E402
    LEGACY_EVENT_COUNT,
    append_event_tensor_exact,
    canonical_bytes,
    canonical_sha256,
    file_sha256,
    load_identity_v12_extension_contract,
    load_legacy_representation_cache,
    select_appended_events,
    tensor_bitwise_equal,
    tensor_sha256,
)
from src.soz.frozen_h_crosswalk import _signal_tensor_sha256  # noqa: E402
from src.soz.models.labram import (  # noqa: E402
    AUDITED_LABRAM_BASE_SHA256,
    AUDITED_LABRAM_MODELING_SHA256,
    OfficialLaBraMEncoder,
    bind_labram_record_positions,
)
from src.soz.models.labram_peft import (  # noqa: E402
    OfficialLaBraMFrozenPrefixEncoder,
    OfficialLaBraMMinimalPEFTSuffix,
)


DEFAULT_UNION = ROOT / "outputs/public_development_union_identity_v12_20260812"
DEFAULT_SIGNAL = ROOT / "outputs/deepsoz_signal_preflight_identity_v3_20260812"
DEFAULT_LEGACY_CACHE = (
    ROOT / "outputs/public_development_labram_prefix_v11_20260811"
)
DEFAULT_TUSZ_ROOT = Path("/mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf")
DEFAULT_MODELING = Path("/mnt/hd1/dyf/workspace/LaBraM/modeling_finetune.py")
DEFAULT_CHECKPOINT = Path("/mnt/hd1/dyf/workspace/LaBraM/checkpoints/labram-base.pth")
DEFAULT_OUTPUT = (
    ROOT / "outputs/public_development_labram_prefix_identity_v12_20260812"
)

EXPECTED_UNION_MANIFEST_SHA256 = (
    "645c55541c37dfc204fdd48c21e0a3c81fe7201f76b862556d1c4dc3bfa4d429"
)
EXPECTED_SIGNAL_ARTIFACT_SHA256 = (
    "2a6bb8a7be20993949e7250b10c83d11fe027ff1afc0fa0919124f7fa371ef8e"
)
EXPECTED_LEGACY_MANIFEST_SHA256 = (
    "b3ce8913a33848b7a706f8b30ccedf09ad8b2f6ae27412b1ae56d187866ff71f"
)
EXPECTED_LEGACY_TENSOR_FILE_SHA256 = (
    "40396fabac11ead6ac870ee69f428951f0577445c291a45b58e37c8fc6bf12bc"
)
LEGACY_SCHEMA = "soz_public_development_labram_block9_prefix_v11_full"
FULL_SCHEMA = "soz_public_development_labram_prefix_identity_v12"
SMOKE_SCHEMA = "soz_public_development_labram_prefix_identity_v12_smoke"
MANIFEST_NAME = "manifest.json"
TENSOR_NAME = "prefix.safetensors"
PREFIX_TENSOR_NAME = "prefix_tokens"
PREFIX_EVENT_SHAPE = (15, 77, 200)


def _safe_edf(root: Path, relative_value: str) -> Path:
    relative = PurePosixPath(str(relative_value))
    if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".edf":
        raise ValueError("union EDF path is not a safe relative EDF path")
    source = root.joinpath(*relative.parts)
    for component in (source, *source.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise ValueError("union EDF path cannot traverse symlinks")
    resolved = source.resolve(strict=True)
    if resolved.relative_to(root).as_posix() != relative.as_posix():
        raise ValueError("union EDF path escaped the pinned TUSZ root")
    return resolved


def _split_calls(eeg: torch.Tensor) -> torch.Tensor:
    if tuple(eeg.shape) != (19, 12_000) or eeg.dtype != torch.float32:
        raise ValueError("LaBraM union event must be float32 [19,12000]")
    calls = eeg.reshape(19, 15, 4, 200).permute(1, 0, 2, 3).contiguous()
    if not tensor_bitwise_equal(
        calls.permute(1, 0, 2, 3).reshape(19, 12_000), eeg
    ):
        raise RuntimeError("LaBraM four-second calls do not exactly reassemble EEG")
    return calls


def _validate_legacy_prefix(
    tensors: Mapping[str, torch.Tensor],
    manifest: Mapping[str, object],
) -> tuple[torch.Tensor, str]:
    if set(tensors) != {PREFIX_TENSOR_NAME}:
        raise ValueError("legacy LaBraM prefix tensor names changed")
    prefix = tensors[PREFIX_TENSOR_NAME].detach().cpu().contiguous()
    if (
        tuple(prefix.shape) != (LEGACY_EVENT_COUNT, *PREFIX_EVENT_SHAPE)
        or prefix.dtype != torch.float32
        or list(prefix.shape) != manifest.get("prefix_tensor_shape")
        or tensor_sha256(prefix) != manifest.get("prefix_tensor_sha256")
        or manifest.get("tensor_name") != PREFIX_TENSOR_NAME
    ):
        raise ValueError("legacy LaBraM prefix tensor integrity failed")
    rows = manifest.get("events")
    if not isinstance(rows, list):
        raise TypeError("legacy LaBraM prefix manifest lacks event rows")
    event_hashes: list[str] = []
    for ordinal, row in enumerate(rows):
        digest = tensor_sha256(prefix[ordinal])
        if digest != row.get("prefix_tensor_sha256"):
            raise ValueError(
                f"legacy LaBraM event SHA failed: {row.get('event_id')}"
            )
        event_hashes.append(digest)
    return prefix, canonical_sha256(event_hashes)


def _validate_raw_replay(loaded: object, event: Mapping[str, object]) -> str:
    edf_receipt_sha = canonical_sha256(asdict(loaded.edf_receipt))
    signal_receipt_sha = canonical_sha256(asdict(loaded.signal_receipt))
    replay_sha = _signal_tensor_sha256(loaded.window.data)
    checks = {
        "EDF": loaded.edf_receipt.edf_sha256 == event["edf_sha256"],
        "EDF receipt": edf_receipt_sha == event["edf_receipt_sha256"],
        "signal receipt": signal_receipt_sha == event["signal_receipt_sha256"],
        "processed window": replay_sha == event["processed_window_sha256"],
        "shape": tuple(loaded.window.data.shape) == (19, 12_000),
        "sampling": float(loaded.window.sfreq_hz) == 200.0,
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"LaBraM raw replay failed {event['event_id']}: {failed}")
    return replay_sha


def materialize(
    *,
    union_directory: Path,
    signal_directory: Path,
    legacy_cache_directory: Path,
    tusz_root: Path,
    modeling_path: Path,
    checkpoint_path: Path,
    output_directory: Path,
    device: torch.device,
    append_limit: int | None,
    progress_every: int,
) -> tuple[Path, Mapping[str, object]]:
    if device.type not in {"cpu", "cuda"} or device.index is not None:
        raise ValueError("device must be cpu or cuda without an explicit index")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if type(progress_every) is not int or progress_every < 1:
        raise ValueError("progress_every must be a positive integer")
    contract = load_identity_v12_extension_contract(
        union_directory,
        signal_directory,
        expected_union_manifest_sha256=EXPECTED_UNION_MANIFEST_SHA256,
        expected_signal_artifact_sha256=EXPECTED_SIGNAL_ARTIFACT_SHA256,
    )
    legacy = load_legacy_representation_cache(
        legacy_cache_directory,
        contract=contract,
        expected_schema=LEGACY_SCHEMA,
        tensor_filename=TENSOR_NAME,
        expected_manifest_sha256=EXPECTED_LEGACY_MANIFEST_SHA256,
        expected_tensor_file_sha256=EXPECTED_LEGACY_TENSOR_FILE_SHA256,
    )
    selected_append, full_scope = select_appended_events(contract, append_limit)
    config_payload = contract.signal_bundle.receipt.get("preprocess_config")
    if not isinstance(config_payload, Mapping):
        raise TypeError("identity-v3 signal receipt lacks preprocessing config")
    config = CausalEDFConfig(**dict(config_payload))

    raw_root = Path(os.path.abspath(tusz_root)).resolve(strict=True)
    if not raw_root.is_dir() or raw_root.is_symlink():
        raise FileNotFoundError(raw_root)
    target = Path(os.path.abspath(output_directory))
    if os.path.lexists(target):
        raise FileExistsError(target)
    for source in (
        contract.union_path,
        Path(signal_directory).resolve(),
        legacy.path,
        raw_root,
        Path(modeling_path).resolve(),
        Path(checkpoint_path).resolve(),
    ):
        if target == source or target in source.parents or source in target.parents:
            raise ValueError("LaBraM prefix extension output overlaps an input")
    if not target.parent.is_dir():
        raise FileNotFoundError(target.parent)

    legacy_prefix, legacy_event_sha_roster = _validate_legacy_prefix(
        load_file(str(legacy.tensor_path), device="cpu"), legacy.manifest
    )
    prefix_encoder = OfficialLaBraMFrozenPrefixEncoder(
        modeling_path=modeling_path,
        checkpoint_path=checkpoint_path,
        expected_sha256=AUDITED_LABRAM_BASE_SHA256,
        expected_modeling_sha256=AUDITED_LABRAM_MODELING_SHA256,
    ).to(device).eval()
    if any(parameter.requires_grad for parameter in prefix_encoder.parameters()):
        raise RuntimeError("frozen LaBraM prefix exposes trainable weights")

    new_prefixes: list[torch.Tensor] = []
    new_rows: list[dict[str, object]] = []
    append_equivalence_error: float | None = None
    started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for position, event in enumerate(selected_append, start=1):
        path = _safe_edf(raw_root, str(event["relative_edf_path"]))
        loaded = load_standard19_edf_event(
            path, float(event["global_t0_sec"]), config=config
        )
        replay_sha = _validate_raw_replay(loaded, event)
        calls = _split_calls(loaded.window.data).to(device)
        binding = bind_labram_record_positions(
            loaded.edf_receipt.raw_channel_names,
            semantic_channels=loaded.edf_receipt.semantic_channels,
        )
        with torch.inference_mode():
            prefix = prefix_encoder.forward_with_record_binding(calls, binding)
        prefix = prefix.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if tuple(prefix.shape) != PREFIX_EVENT_SHAPE:
            raise RuntimeError("LaBraM prefix returned the wrong event shape")

        if append_equivalence_error is None:
            suffix = OfficialLaBraMMinimalPEFTSuffix(
                modeling_path=modeling_path,
                checkpoint_path=checkpoint_path,
                expected_sha256=AUDITED_LABRAM_BASE_SHA256,
                expected_modeling_sha256=AUDITED_LABRAM_MODELING_SHA256,
            ).to(device).eval()
            official = OfficialLaBraMEncoder(
                modeling_path=modeling_path,
                checkpoint_path=checkpoint_path,
                expected_sha256=AUDITED_LABRAM_BASE_SHA256,
                expected_modeling_sha256=AUDITED_LABRAM_MODELING_SHA256,
                tile_seconds=4,
                position_names=binding.position_names,
            ).to(device).eval()
            with torch.inference_mode():
                restored = suffix(prefix.to(device))
                expected = official(calls)
            append_equivalence_error = float(
                (restored - expected).abs().amax().cpu()
            )
            if append_equivalence_error > 1e-6:
                raise RuntimeError(
                    "zero-adapter recovered prefix differs from official LaBraM: "
                    f"{append_equivalence_error}"
                )
            del suffix, official, restored, expected

        new_prefixes.append(prefix)
        new_rows.append(
            {
                "ordinal": int(event["ordinal"]),
                "event_id": str(event["event_id"]),
                "patient_id": str(event["patient_id"]),
                "outer_fold": int(event["outer_fold"]),
                "legacy_model_split": str(event["legacy_model_split"]),
                "processed_window_sha256": replay_sha,
                "position_names": list(binding.position_names),
                "position_ids": list(binding.position_ids),
                "prefix_tensor_sha256": tensor_sha256(prefix),
            }
        )
        if position % progress_every == 0 or position == len(selected_append):
            elapsed = time.monotonic() - started
            print(
                json.dumps(
                    {
                        "new_event": position,
                        "new_total": len(selected_append),
                        "elapsed_sec": round(elapsed, 2),
                        "seconds_per_new_event": round(elapsed / position, 3),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    appended_prefix = torch.stack(new_prefixes).contiguous()
    combined_prefix = append_event_tensor_exact(legacy_prefix, appended_prefix)
    expected_count = LEGACY_EVENT_COUNT + len(selected_append)
    if tuple(combined_prefix.shape) != (expected_count, *PREFIX_EVENT_SHAPE):
        raise RuntimeError("extended LaBraM prefix tensor has the wrong shape")
    legacy_rows = [dict(row) for row in legacy.manifest["events"]]
    output_rows = [*legacy_rows, *new_rows]
    selected_events = (*contract.legacy_events, *selected_append)
    event_ids = [str(event["event_id"]) for event in selected_events]
    if [str(row["event_id"]) for row in output_rows] != event_ids:
        raise RuntimeError("extended LaBraM prefix event order changed")
    if output_rows[:LEGACY_EVENT_COUNT] != legacy_rows:
        raise RuntimeError("legacy LaBraM prefix manifest rows changed")

    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        tensor_path = staging / TENSOR_NAME
        save_file({PREFIX_TENSOR_NAME: combined_prefix}, str(tensor_path))
        elapsed = time.monotonic() - started
        peak = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
        legacy_equivalence_error = float(
            legacy.manifest["zero_adapter_official_equivalence_max_abs_error"]
        )
        equivalence_error = max(
            legacy_equivalence_error,
            float(append_equivalence_error),
        )
        manifest: dict[str, object] = {
            "schema_version": FULL_SCHEMA if full_scope else SMOKE_SCHEMA,
            "purpose": (
                "target_free_append_only_frozen_labram_block9_prefix_for_"
                "identity_v12_development"
            ),
            "development_only": True,
            "public_confirmation_forbidden": True,
            "full_scope": full_scope,
            "smoke_only": not full_scope,
            "event_count": len(selected_events),
            "patient_count": len({str(event["patient_id"]) for event in selected_events}),
            "legacy_reused_event_count": LEGACY_EVENT_COUNT,
            "newly_computed_event_count": len(selected_append),
            "event_ids": event_ids,
            "event_order_sha256": canonical_sha256(event_ids),
            "events": output_rows,
            "foundation_backbone": legacy.manifest["foundation_backbone"],
            "foundation_checkpoint_sha256": AUDITED_LABRAM_BASE_SHA256,
            "foundation_modeling_sha256": AUDITED_LABRAM_MODELING_SHA256,
            "foundation_trainable_parameters_during_materialization": 0,
            "foundation_prefix_blocks": list(legacy.manifest["foundation_prefix_blocks"]),
            "foundation_prefix_stop_exclusive": legacy.manifest[
                "foundation_prefix_stop_exclusive"
            ],
            "input_shape_per_event": [19, 12_000],
            "call_count_per_event": 15,
            "call_input_shape": [19, 4, 200],
            "call_output_shape": [77, 200],
            "prefix_event_shape": list(PREFIX_EVENT_SHAPE),
            "prefix_tensor_shape": list(combined_prefix.shape),
            "tensor_name": PREFIX_TENSOR_NAME,
            "tensor_file": TENSOR_NAME,
            "tensor_file_sha256": file_sha256(tensor_path),
            "tensor_file_size_bytes": tensor_path.stat().st_size,
            "prefix_tensor_sha256": tensor_sha256(combined_prefix),
            "zero_adapter_official_equivalence_max_abs_error": equivalence_error,
            "zero_adapter_official_equivalence_verified": True,
            "legacy_zero_adapter_equivalence_max_abs_error": (
                legacy_equivalence_error
            ),
            "appended_zero_adapter_equivalence_max_abs_error": (
                append_equivalence_error
            ),
            "materialization_device": str(device),
            "elapsed_sec": elapsed,
            "seconds_per_new_event": elapsed / len(selected_append),
            "peak_cuda_memory_bytes": int(peak),
            "cache_extension_receipt": {
                "append_only": True,
                "legacy_event_rows_exact_prefix": output_rows[:LEGACY_EVENT_COUNT]
                == legacy_rows,
                "legacy_event_ids_exact_prefix": event_ids[:LEGACY_EVENT_COUNT]
                == list(legacy.manifest["event_ids"]),
                "legacy_tensor_prefix_exact": tensor_bitwise_equal(
                    combined_prefix[:LEGACY_EVENT_COUNT], legacy_prefix
                ),
                "legacy_event_sha_roster_sha256": legacy_event_sha_roster,
                "old_raw_eeg_replayed": False,
                "new_raw_eeg_event_count": len(selected_append),
            },
            "lineage": {
                "public_union_manifest_sha256": contract.union_manifest_sha256,
                "signal_identity_recovery_artifact_sha256": (
                    contract.signal_bundle.artifact_sha256
                ),
                "signal_identity_recovery_receipt_sha256": (
                    contract.signal_bundle.receipt_sha256
                ),
                "signal_preprocess_config_sha256": (
                    contract.signal_bundle.receipt["preprocess_config_sha256"]
                ),
                "legacy_cache_manifest_sha256": legacy.manifest_sha256,
                "legacy_cache_tensor_file_sha256": legacy.tensor_file_sha256,
                "tusz_root": str(raw_root),
            },
            "access_receipt": {
                "legacy_target_free_tensor_cache_loaded": True,
                "raw_public_eeg_loaded": True,
                "raw_public_event_count": len(selected_append),
                "legacy_raw_public_event_count": 0,
                "deepsoz_target_values_loaded": False,
                "tusz_channel_annotation_values_loaded": False,
                "historical_prediction_artifacts_loaded": False,
                "private_eeg_loaded": False,
                "private_target_values_loaded": False,
                "foundation_training_performed": False,
                "reasoner_training_performed": False,
            },
        }
        if not all(
            manifest["cache_extension_receipt"][key]
            for key in (
                "append_only",
                "legacy_event_rows_exact_prefix",
                "legacy_event_ids_exact_prefix",
                "legacy_tensor_prefix_exact",
            )
        ):
            raise RuntimeError("LaBraM prefix append-only receipt failed")
        (staging / MANIFEST_NAME).write_bytes(canonical_bytes(manifest, newline=True))
        os.rename(staging, target)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return target, manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--union-directory", type=Path, default=DEFAULT_UNION)
    parser.add_argument("--signal-directory", type=Path, default=DEFAULT_SIGNAL)
    parser.add_argument(
        "--legacy-cache-directory", type=Path, default=DEFAULT_LEGACY_CACHE
    )
    parser.add_argument("--tusz-root", type=Path, default=DEFAULT_TUSZ_ROOT)
    parser.add_argument("--modeling-path", type=Path, default=DEFAULT_MODELING)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--append-limit", type=int)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    path, manifest = materialize(
        union_directory=args.union_directory,
        signal_directory=args.signal_directory,
        legacy_cache_directory=args.legacy_cache_directory,
        tusz_root=args.tusz_root,
        modeling_path=args.modeling_path,
        checkpoint_path=args.checkpoint_path,
        output_directory=args.output_directory,
        device=torch.device(args.device),
        append_limit=args.append_limit,
        progress_every=args.progress_every,
    )
    print(
        json.dumps(
            {
                "status": "LABRAM_PREFIX_IDENTITY_V12_EXTENDED",
                "path": str(path),
                "manifest_sha256": file_sha256(path / MANIFEST_NAME),
                "event_count": manifest["event_count"],
                "legacy_reused_event_count": manifest["legacy_reused_event_count"],
                "newly_computed_event_count": manifest[
                    "newly_computed_event_count"
                ],
                "full_scope": manifest["full_scope"],
                "zero_adapter_equivalence_max_abs_error": manifest[
                    "zero_adapter_official_equivalence_max_abs_error"
                ],
                "target_values_loaded": False,
                "private_loaded": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
