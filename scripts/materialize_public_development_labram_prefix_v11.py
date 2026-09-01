#!/usr/bin/env python3
"""Uniformly materialize frozen LaBraM block-9 prefixes for all 988 events."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
import time
from typing import Mapping, Sequence

import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soz.data.edf import CausalEDFConfig, load_standard19_edf_event  # noqa: E402
from src.soz.frozen_h_crosswalk import _signal_tensor_sha256  # noqa: E402
from src.soz.ictal_native_eval import (  # noqa: E402
    load_bound_deepsoz_signal_preflight_artifact,
)
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
from src.soz.v11_development_union import (  # noqa: E402
    EXPECTED_PUBLIC_DEVELOPMENT_UNION_MANIFEST_SHA256,
    load_public_development_union,
)


DEFAULT_UNION = ROOT / "outputs/public_development_union_v11_20260811"
DEFAULT_SIGNAL = ROOT / "outputs/deepsoz_signal_preflight_v2_20260809_current"
DEFAULT_TUSZ_ROOT = Path("/mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf")
DEFAULT_MODELING = Path("/mnt/hd1/dyf/workspace/LaBraM/modeling_finetune.py")
DEFAULT_CHECKPOINT = Path("/mnt/hd1/dyf/workspace/LaBraM/checkpoints/labram-base.pth")
DEFAULT_OUTPUT = ROOT / "outputs/public_development_labram_prefix_v11_20260811"
EXPECTED_SIGNAL_ARTIFACT_SHA256 = (
    "a2fdf45dd122e39ec6e73b3a3edafa1264669875fd2d8cd2b9cb7e8313d1ee66"
)
EXPECTED_SIGNAL_RECEIPT_SHA256 = (
    "10128ad30d2163838222d0b4a27d9889a767276a9b697812e3cf568a3d9fd446"
)
FULL_SCHEMA = "soz_public_development_labram_block9_prefix_v11_full"
SMOKE_SCHEMA = "soz_public_development_labram_block9_prefix_v11_smoke"
MANIFEST_NAME = "manifest.json"
TENSOR_NAME = "prefix.safetensors"
PREFIX_TENSOR_NAME = "prefix_tokens"


def _canonical_bytes(value: object, *, newline: bool = False) -> bytes:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return raw + (b"\n" if newline else b"")


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    metadata = _canonical_bytes(
        {"dtype": str(tensor.dtype), "shape": list(tensor.shape)}
    )
    raw = tensor.view(torch.uint8).numpy().tobytes()
    digest = hashlib.sha256()
    digest.update(len(metadata).to_bytes(8, "big"))
    digest.update(metadata)
    digest.update(len(raw).to_bytes(8, "big"))
    digest.update(raw)
    return digest.hexdigest()


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
    if tuple(eeg.shape) != (19, 12000) or eeg.dtype != torch.float32:
        raise ValueError("LaBraM union event must be float32 [19,12000]")
    calls = eeg.reshape(19, 15, 4, 200).permute(1, 0, 2, 3).contiguous()
    if not torch.equal(calls.permute(1, 0, 2, 3).reshape(19, 12000), eeg):
        raise RuntimeError("LaBraM four-second calls do not exactly reassemble EEG")
    return calls


def materialize(
    *,
    union_directory: Path,
    signal_directory: Path,
    tusz_root: Path,
    modeling_path: Path,
    checkpoint_path: Path,
    output_directory: Path,
    device: torch.device,
    limit: int | None,
    progress_every: int,
) -> tuple[Path, Mapping[str, object]]:
    if device.type not in {"cpu", "cuda"} or device.index is not None:
        raise ValueError("device must be cpu or cuda without explicit index")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    union = load_public_development_union(
        union_directory,
        expected_manifest_sha256=EXPECTED_PUBLIC_DEVELOPMENT_UNION_MANIFEST_SHA256,
    )
    signal = load_bound_deepsoz_signal_preflight_artifact(
        signal_directory,
        expected_artifact_sha256=EXPECTED_SIGNAL_ARTIFACT_SHA256,
        expected_receipt_sha256=EXPECTED_SIGNAL_RECEIPT_SHA256,
    )
    config_payload = signal.receipt.get("preprocess_config")
    if not isinstance(config_payload, Mapping):
        raise TypeError("signal preflight lacks preprocessing configuration")
    config = CausalEDFConfig(**dict(config_payload))
    signal_rows = signal.receipt.get("events")
    if not isinstance(signal_rows, list):
        raise TypeError("signal preflight event rows are missing")
    signal_by_id = {str(row["event_id"]): row for row in signal_rows}
    if len(signal_by_id) != 988:
        raise ValueError("signal preflight event roster changed")

    raw_root = Path(os.path.abspath(tusz_root)).resolve(strict=True)
    target = Path(os.path.abspath(output_directory))
    if target.exists():
        raise FileExistsError(target)
    for source in (union.path, Path(signal_directory).resolve(), raw_root):
        if target == source or target in source.parents or source in target.parents:
            raise ValueError("LaBraM prefix output overlaps immutable input")
    if not target.parent.is_dir():
        raise FileNotFoundError(target.parent)
    if limit is None:
        selected = union.events
        full_scope = True
    else:
        if isinstance(limit, bool) or not 1 <= int(limit) < len(union.events):
            raise ValueError("--limit must be a smoke prefix in [1,987]")
        selected = union.events[: int(limit)]
        full_scope = False

    prefix_encoder = OfficialLaBraMFrozenPrefixEncoder(
        modeling_path=modeling_path,
        checkpoint_path=checkpoint_path,
        expected_sha256=AUDITED_LABRAM_BASE_SHA256,
        expected_modeling_sha256=AUDITED_LABRAM_MODELING_SHA256,
    ).to(device).eval()
    if any(parameter.requires_grad for parameter in prefix_encoder.parameters()):
        raise RuntimeError("frozen LaBraM prefix unexpectedly exposes trainable weights")

    prefix_rows = []
    output_rows = []
    equivalence_error: float | None = None
    started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for position, event in enumerate(selected, start=1):
        source_row = signal_by_id.get(event.event_id)
        if source_row is None:
            raise ValueError("union event disappeared from signal preflight")
        path = _safe_edf(raw_root, event.relative_edf_path)
        loaded = load_standard19_edf_event(path, event.global_t0_sec, config=config)
        replay_sha = _signal_tensor_sha256(loaded.window.data)
        edf_receipt_sha = _canonical_sha(asdict(loaded.edf_receipt))
        signal_receipt_sha = _canonical_sha(asdict(loaded.signal_receipt))
        checks = {
            "EDF": loaded.edf_receipt.edf_sha256 == event.edf_sha256,
            "EDF receipt": edf_receipt_sha == event.edf_receipt_sha256,
            "signal receipt": signal_receipt_sha == event.signal_receipt_sha256,
            "processed window": replay_sha == event.processed_window_sha256,
        }
        failed = tuple(name for name, passed in checks.items() if not passed)
        if failed:
            raise ValueError(f"LaBraM raw replay failed {event.event_id}: {failed}")
        calls = _split_calls(loaded.window.data).to(device)
        binding = bind_labram_record_positions(
            loaded.edf_receipt.raw_channel_names,
            semantic_channels=loaded.edf_receipt.semantic_channels,
        )
        with torch.inference_mode():
            prefix = prefix_encoder.forward_with_record_binding(calls, binding)
        prefix = prefix.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if tuple(prefix.shape) != (15, 77, 200):
            raise RuntimeError("LaBraM prefix returned the wrong event shape")

        if equivalence_error is None:
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
            equivalence_error = float((restored - expected).abs().amax().cpu())
            if equivalence_error > 1e-6:
                raise RuntimeError(
                    "zero-adapter prefix/suffix differs from official LaBraM: "
                    f"{equivalence_error}"
                )
            del suffix, official, restored, expected
        prefix_rows.append(prefix)
        output_rows.append(
            {
                "ordinal": event.ordinal,
                "event_id": event.event_id,
                "patient_id": event.patient_id,
                "outer_fold": event.outer_fold,
                "legacy_model_split": event.legacy_model_split,
                "processed_window_sha256": replay_sha,
                "position_names": list(binding.position_names),
                "position_ids": list(binding.position_ids),
                "prefix_tensor_sha256": _tensor_sha(prefix),
            }
        )
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

    prefixes = torch.stack(prefix_rows).contiguous()
    if tuple(prefixes.shape) != (len(selected), 15, 77, 200):
        raise RuntimeError("unified LaBraM prefix tensor has the wrong shape")
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        tensor_path = staging / TENSOR_NAME
        save_file({PREFIX_TENSOR_NAME: prefixes}, str(tensor_path))
        elapsed = time.monotonic() - started
        peak = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
        manifest = {
            "schema_version": FULL_SCHEMA if full_scope else SMOKE_SCHEMA,
            "purpose": "uniform_target_free_frozen_labram_block9_prefix_for_v11",
            "development_only": True,
            "public_confirmation_forbidden": True,
            "full_scope": full_scope,
            "smoke_only": not full_scope,
            "event_count": len(selected),
            "patient_count": len({event.patient_id for event in selected}),
            "event_ids": [event.event_id for event in selected],
            "event_order_sha256": _canonical_sha(
                [event.event_id for event in selected]
            ),
            "events": output_rows,
            "foundation_backbone": "official_pretrained_LaBraM_Base_not_replaced",
            "foundation_checkpoint_sha256": AUDITED_LABRAM_BASE_SHA256,
            "foundation_modeling_sha256": AUDITED_LABRAM_MODELING_SHA256,
            "foundation_trainable_parameters_during_materialization": 0,
            "foundation_prefix_blocks": list(range(10)),
            "foundation_prefix_stop_exclusive": 10,
            "input_shape_per_event": [19, 12000],
            "call_count_per_event": 15,
            "call_input_shape": [19, 4, 200],
            "call_output_shape": [77, 200],
            "prefix_event_shape": [15, 77, 200],
            "prefix_tensor_shape": list(prefixes.shape),
            "tensor_name": PREFIX_TENSOR_NAME,
            "tensor_file": TENSOR_NAME,
            "tensor_file_sha256": _file_sha(tensor_path),
            "tensor_file_size_bytes": tensor_path.stat().st_size,
            "prefix_tensor_sha256": _tensor_sha(prefixes),
            "zero_adapter_official_equivalence_max_abs_error": equivalence_error,
            "zero_adapter_official_equivalence_verified": True,
            "materialization_device": str(device),
            "elapsed_sec": elapsed,
            "seconds_per_event": elapsed / len(selected),
            "peak_cuda_memory_bytes": int(peak),
            "lineage": {
                "public_union_manifest_sha256": union.manifest_sha256,
                "signal_preflight_artifact_sha256": signal.artifact_sha256,
                "signal_preflight_receipt_sha256": signal.receipt_sha256,
                "signal_preprocess_config_sha256": signal.receipt[
                    "preprocess_config_sha256"
                ],
                "tusz_root": str(raw_root),
            },
            "access_receipt": {
                "raw_public_eeg_loaded": True,
                "raw_public_event_count": len(selected),
                "deepsoz_target_values_loaded": False,
                "tusz_channel_annotation_values_loaded": False,
                "historical_prediction_artifacts_loaded": False,
                "private_eeg_loaded": False,
                "private_target_values_loaded": False,
                "training_performed": False,
            },
        }
        (staging / MANIFEST_NAME).write_bytes(_canonical_bytes(manifest, newline=True))
        os.replace(staging, target)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return target, manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--union-directory", type=Path, default=DEFAULT_UNION)
    parser.add_argument("--signal-directory", type=Path, default=DEFAULT_SIGNAL)
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
    if args.progress_every < 1:
        raise ValueError("--progress-every must be positive")
    path, manifest = materialize(
        union_directory=args.union_directory,
        signal_directory=args.signal_directory,
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
                "status": "PUBLIC_DEVELOPMENT_LABRAM_PREFIX_V11_MATERIALIZED",
                "path": str(path),
                "manifest_sha256": _file_sha(path / MANIFEST_NAME),
                "event_count": manifest["event_count"],
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
