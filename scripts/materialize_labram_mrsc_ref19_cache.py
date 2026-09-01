#!/usr/bin/env python3
"""Materialize target-free C-REF19 evidence for the frozen v11.1 roster.

The output is the missing sensitivity-side carrier required to compute MRSC
final-score reference disagreement.  It contains the paired C-REF19 LaBraM
block-9 prefix and the same deterministic fine temporal features used by the
v11.1 C-CAR19 anchor.  It does not load SOZ targets, evaluate accuracy, train,
select a model, calibrate MRSC, or access private data.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path, PurePosixPath
import tempfile
import time
from typing import Mapping

from safetensors.torch import save_file
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from src.soz.data.edf import CausalEDFConfig, load_standard19_edf_event  # noqa: E402
from src.soz.fine_temporal_evidence import (  # noqa: E402
    FINE_TEMPORAL_FEATURE_NAMES,
    extract_fine_temporal_evidence,
)
from src.soz.models.labram import bind_labram_record_positions  # noqa: E402
from src.soz.models.labram_peft import OfficialLaBraMFrozenPrefixEncoder  # noqa: E402
from src.soz.preprocessing_arm_runtime import (  # noqa: E402
    CAUSAL_REFERENCE_PAIR_ROLE,
    CAUSAL_REFERENCE_PAIR_SCHEMA,
    CAUSAL_REFERENCE_SENSITIVITY_ARM_ID,
)


DEFAULT_ANCHOR_BRIDGE = (
    ROOT / "outputs/labram_v11_1_anchor_target_excluding_20260812"
)
DEFAULT_UNION = ROOT / "outputs/public_development_union_v11_20260811"
DEFAULT_SOURCE_RECEIPT = (
    ROOT / "outputs/tusz_ictal_master_manifest_v4_20260809_preflight/receipt.json"
)
DEFAULT_TUSZ_ROOT = Path("/mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf")
DEFAULT_MODELING = Path("/mnt/hd1/dyf/workspace/LaBraM/modeling_finetune.py")
DEFAULT_CHECKPOINT = Path("/mnt/hd1/dyf/workspace/LaBraM/checkpoints/labram-base.pth")
DEFAULT_OUTPUT = ROOT / "outputs/labram_mrsc_ref19_cache_20260812"

SCHEMA = "soz_labram_mrsc_ref19_target_free_cache_v1"
ANCHOR_BRIDGE_SCHEMA = "soz_labram_v11_1_target_excluding_anchor_bridge_v1"
UNION_SCHEMA = "soz_public_development_union_v11"
SOURCE_SCHEMA = "tusz_ictal_training_manifest_v4.0.0"


def _load_json(path: Path, *, name: str) -> dict[str, object]:
    source = path.resolve(strict=True)
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"{name} must be a canonical regular file")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{name} must contain one JSON object")
    return payload


def _safe_edf(root: Path, relative_value: object) -> Path:
    relative = PurePosixPath(str(relative_value))
    if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".edf":
        raise ValueError("Event contains an unsafe relative EDF path")
    source = root.joinpath(*relative.parts)
    for component in (source, *source.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise ValueError("EDF path cannot traverse a symbolic link")
    resolved = source.resolve(strict=True)
    if resolved.relative_to(root).as_posix() != relative.as_posix():
        raise ValueError("EDF path escaped the pinned TUSZ root")
    return resolved


def _event_calls(window: torch.Tensor) -> torch.Tensor:
    if tuple(window.shape) != (19, 12_000):
        raise ValueError("MRSC reference cache requires one [19,12000] event")
    return window.reshape(19, 15, 4, 200).permute(1, 0, 2, 3).contiguous()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    fd, staging = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging, path)
    except Exception:
        try:
            os.unlink(staging)
        except FileNotFoundError:
            pass
        raise


def materialize(
    *,
    anchor_bridge: Path,
    union_directory: Path,
    source_receipt: Path,
    tusz_root: Path,
    modeling_path: Path,
    checkpoint_path: Path,
    output_directory: Path,
    device_name: str,
    progress_every: int,
) -> dict[str, object]:
    bridge = _load_json(anchor_bridge / "manifest.json", name="anchor bridge")
    union = _load_json(union_directory / "manifest.json", name="signal union")
    source = _load_json(source_receipt, name="TUSZ source receipt")
    if bridge.get("schema_version") != ANCHOR_BRIDGE_SCHEMA or (
        bridge.get("status") != "completed_target_excluding_anchor_bridge"
    ):
        raise ValueError("Unsupported target-excluding anchor bridge")
    if union.get("schema_version") != UNION_SCHEMA:
        raise ValueError("Unsupported signal-only public-development union")
    if source.get("schema_version") != SOURCE_SCHEMA:
        raise ValueError("Unsupported TUSZ source receipt")
    access = bridge.get("access_receipt")
    if not isinstance(access, Mapping) or access.get("target_tensor_values_loaded") is not False:
        raise ValueError("Anchor bridge is not target-excluding")
    union_access = union.get("access_receipt")
    if not isinstance(union_access, Mapping):
        raise TypeError("Signal union lacks an access receipt")
    for field in (
        "deepsoz_target_values_loaded",
        "private_eeg_loaded",
        "private_target_values_loaded",
        "source_eval_target_values_loaded",
        "prediction_artifacts_loaded",
    ):
        if union_access.get(field) is not False:
            raise ValueError(f"Signal union is not target/private free: {field}")

    patient_ids_raw = bridge.get("patient_ids")
    event_ids_raw = bridge.get("event_ids")
    if not isinstance(patient_ids_raw, list) or not isinstance(event_ids_raw, list):
        raise TypeError("Anchor bridge lacks patient/event identities")
    patient_ids = tuple(str(value) for value in patient_ids_raw)
    event_ids = tuple(str(value) for value in event_ids_raw)
    if len(patient_ids) != 101 or len(set(patient_ids)) != 101:
        raise ValueError("Anchor bridge patient roster changed")
    if len(event_ids) != 984 or len(set(event_ids)) != 984:
        raise ValueError("Anchor bridge event roster changed")
    patient_to_index = {patient: index for index, patient in enumerate(patient_ids)}

    raw_events = union.get("events")
    if not isinstance(raw_events, list):
        raise TypeError("Signal union lacks event rows")
    selected: list[Mapping[str, object]] = []
    event_patient_indices: list[int] = []
    for row in raw_events:
        if not isinstance(row, Mapping):
            raise TypeError("Signal union contains a non-object event row")
        patient = str(row.get("patient_id", "")).strip()
        if patient not in patient_to_index:
            continue
        selected.append(row)
        event_patient_indices.append(patient_to_index[patient])
    selected_ids = tuple(str(row.get("event_id", "")) for row in selected)
    if selected_ids != event_ids:
        raise ValueError("Signal union and target-excluding bridge event order differ")

    preprocess = source.get("preprocess_config")
    if not isinstance(preprocess, Mapping):
        raise TypeError("TUSZ source receipt lacks preprocessing configuration")
    car_config = CausalEDFConfig(**dict(preprocess))
    if not car_config.apply_car19:
        raise ValueError("Frozen source preprocessing is not C-CAR19")
    ref_config = replace(car_config, apply_car19=False)

    raw_root = tusz_root.resolve(strict=True)
    if not raw_root.is_dir() or raw_root.is_symlink():
        raise ValueError("TUSZ root must be a canonical directory")
    target = output_directory.absolute()
    if target.exists():
        raise FileExistsError(target)
    if not target.parent.is_dir():
        raise FileNotFoundError(target.parent)

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    encoder = OfficialLaBraMFrozenPrefixEncoder(
        modeling_path=modeling_path,
        checkpoint_path=checkpoint_path,
    ).to(device)
    encoder.eval()
    encoder_dtype = next(encoder.parameters()).dtype

    prefixes = torch.empty((len(selected), 15, 77, 200), dtype=torch.float32)
    fine_features = torch.empty(
        (len(selected), 19, len(FINE_TEMPORAL_FEATURE_NAMES)), dtype=torch.float32
    )
    source_references: set[str] = set()
    started = time.monotonic()
    for index, row in enumerate(selected):
        event_id = str(row.get("event_id", "")).strip()
        onset = float(row.get("global_t0_sec"))
        if not event_id or not math.isfinite(onset):
            raise ValueError("Selected event has invalid identity/timing")
        edf = _safe_edf(raw_root, row.get("relative_edf_path"))
        loaded = load_standard19_edf_event(edf, onset, config=ref_config)
        ref_waveform = loaded.window.data.detach().cpu().float().contiguous()
        if tuple(ref_waveform.shape) != (19, 12_000) or not torch.isfinite(ref_waveform).all():
            raise ValueError("C-REF19 preprocessing returned an invalid event")
        binding = bind_labram_record_positions(
            loaded.edf_receipt.raw_channel_names,
            semantic_channels=loaded.edf_receipt.semantic_channels,
        )
        with torch.no_grad():
            prefix = encoder.forward_with_record_binding(
                _event_calls(ref_waveform).to(device=device, dtype=encoder_dtype),
                binding,
            ).detach().cpu().float()
        if tuple(prefix.shape) != (15, 77, 200):
            raise RuntimeError("Frozen LaBraM returned an invalid reference prefix")
        fine = extract_fine_temporal_evidence(ref_waveform, sfreq_hz=200.0)
        prefixes[index].copy_(prefix)
        fine_features[index].copy_(fine.features.cpu())
        source_references.update(str(value) for value in loaded.signal_receipt.source_references)
        completed = index + 1
        if progress_every > 0 and (completed % progress_every == 0 or completed == len(selected)):
            print(json.dumps({
                "completed": completed,
                "total": len(selected),
                "elapsed_sec": round(time.monotonic() - started, 2),
            }, sort_keys=True), flush=True)

    if not torch.isfinite(prefixes).all() or not torch.isfinite(fine_features).all():
        raise RuntimeError("C-REF19 cache contains non-finite evidence")
    target.mkdir()
    tensor_path = target / "ref19_evidence.safetensors"
    staging = target / ".ref19_evidence.safetensors.tmp"
    try:
        save_file(
            {
                "ref_prefix_tokens": prefixes.contiguous(),
                "ref_fine_features": fine_features.contiguous(),
                "event_patient_index": torch.tensor(event_patient_indices, dtype=torch.long),
            },
            str(staging),
        )
        os.replace(staging, tensor_path)
        manifest: dict[str, object] = {
            "schema_version": SCHEMA,
            "status": "completed_target_free_ref19_evidence_cache",
            "reference_pair": {
                "schema_version": CAUSAL_REFERENCE_PAIR_SCHEMA,
                "role": CAUSAL_REFERENCE_PAIR_ROLE,
                "primary": "C-CAR19",
                "sensitivity": CAUSAL_REFERENCE_SENSITIVITY_ARM_ID,
                "shared_filter_resample_crop_contract": True,
            },
            "patient_count": len(patient_ids),
            "event_count": len(event_ids),
            "patient_ids": list(patient_ids),
            "event_ids": list(event_ids),
            "tensor_file": tensor_path.name,
            "tensor_specs": {
                "ref_prefix_tokens": list(prefixes.shape),
                "ref_fine_features": list(fine_features.shape),
                "event_patient_index": [len(event_patient_indices)],
            },
            "fine_feature_names": list(FINE_TEMPORAL_FEATURE_NAMES),
            "source_reference_names": sorted(source_references),
            "device": str(device),
            "elapsed_sec": time.monotonic() - started,
            "preprocess_config": asdict(ref_config),
            "access_receipt": {
                "anchor_bridge_target_excluding": True,
                "deepsoz_target_values_loaded": False,
                "private_eeg_loaded": False,
                "private_target_values_loaded": False,
                "training_performed": False,
                "accuracy_or_error_metrics_computed": False,
                "model_or_threshold_selection_performed": False,
                "foundation_optimizer_parameters": 0,
            },
            "claim_boundary": {
                "reference_evidence_is_sensitivity_not_second_ground_truth": True,
                "fine_change_is_not_soz_onset_or_propagation_truth": True,
                "cache_does_not_contain_soz_targets": True,
                "cache_does_not_change_car19_anchor_scores": True,
            },
        }
        _atomic_json(target / "manifest.json", manifest)
    except Exception:
        if staging.exists():
            staging.unlink()
        for child in target.iterdir():
            child.unlink()
        target.rmdir()
        raise
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-bridge", type=Path, default=DEFAULT_ANCHOR_BRIDGE)
    parser.add_argument("--union-directory", type=Path, default=DEFAULT_UNION)
    parser.add_argument("--source-receipt", type=Path, default=DEFAULT_SOURCE_RECEIPT)
    parser.add_argument("--tusz-root", type=Path, default=DEFAULT_TUSZ_ROOT)
    parser.add_argument("--modeling-path", type=Path, default=DEFAULT_MODELING)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()
    manifest = materialize(
        anchor_bridge=args.anchor_bridge,
        union_directory=args.union_directory,
        source_receipt=args.source_receipt,
        tusz_root=args.tusz_root,
        modeling_path=args.modeling_path,
        checkpoint_path=args.checkpoint_path,
        output_directory=args.output_directory,
        device_name=args.device,
        progress_every=args.progress_every,
    )
    print(json.dumps({
        "output": str(args.output_directory),
        "event_count": manifest["event_count"],
        "elapsed_sec": manifest["elapsed_sec"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
