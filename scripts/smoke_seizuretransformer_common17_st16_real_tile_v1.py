#!/usr/bin/env python3
"""One-step ST16 BF16 smoke on a real FZ/PZ-missing source-train EEG tile.

This is deliberately not a detector benchmark and emits no accuracy metric or
promotable checkpoint.  It verifies the real EDF -> typed common17 lineage ->
named LB16 transform -> vendored 60 s architecture -> differentiable update
path while source-dev/source-eval and all non-EEG inputs remain unopened.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording.canonical_edf_materialization import (
    load_canonical_edf_record,
)
from src.clinical_eeg_long_recording.detector_signal_lineage_authority_v1 import (
    authorize_detector_signal_lineage_from_canonical_record,
    require_validated_detector_signal_lineage_authority,
)
from src.clinical_eeg_long_recording.eventnet_common17_streaming_v1 import (
    load_common17_manifest,
)
from src.clinical_eeg_long_recording import (
    seizuretransformer_cleanroom_registry_v1 as st,
)
from src.clinical_eeg_long_recording.st16_common17_axis_contract_v1 import (
    CANONICAL_ST16_TYPED_UNITS,
)
from third_party.SeizureTransformer.time_step_level.model import (
    SeizureTransformer,
)


PENDING = "CONTENT-ADDRESS-PENDING"
DEFAULT_MANIFEST = ROOT / "outputs/eventnet_common17_streaming_v1_20260824/manifest.json"
DEFAULT_TUSZ_ROOT = Path("/mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf")
DEFAULT_OUTPUT = ROOT / "outputs/clinical_eeg_st16_common17_real_tile_smoke_v1_20260825/receipt.json"


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _content_address(value: dict) -> dict:
    result = deepcopy(value)
    result["receipt_sha256"] = PENDING
    result["receipt_sha256"] = _sha(result)
    return result


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_edf(root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "train" or relative.suffix.lower() != ".edf":
        raise PermissionError("real ST16 smoke accepts a source-train relative EDF only")
    path = (root / relative).resolve(strict=True)
    path.relative_to(root)
    if path.is_symlink() or not path.is_file():
        raise ValueError("real ST16 smoke EDF must be a regular non-symlink file")
    return path


def _select_record(manifest: dict) -> tuple[dict, int]:
    candidates = [
        row
        for row in manifest["records"]
        if row["model_split"] == "source_train"
        and row["seizure_events"]
        and row["target_sample_count_256hz"] >= st.TILE_SAMPLES
        and "FZ" not in row["audited_observed_channel_ids"]
        and "PZ" not in row["audited_observed_channel_ids"]
    ]
    if not candidates:
        raise ValueError("no FZ/PZ-missing source-train event record supports 60 seconds")
    candidates.sort(
        key=lambda row: (
            row["recording_duration_seconds_fraction"][0]
            / row["recording_duration_seconds_fraction"][1],
            row["edf_relative_path"],
        )
    )
    return candidates[0], len(candidates)


def _target_spans(record: dict, sample_count: int) -> list[tuple[int, int]]:
    spans = []
    for event in record["seizure_events"]:
        start = max(0, min(sample_count, math.floor(float(event["start_seconds"]) * st.TARGET_FS_HZ)))
        stop = max(start, min(sample_count, math.ceil(float(event["stop_seconds"]) * st.TARGET_FS_HZ)))
        if stop > start:
            spans.append((start, stop))
    spans.sort()
    return spans


def _choose_positive_tile(spans: list[tuple[int, int]], sample_count: int) -> int:
    for start, length in st.enumerate_seizuretransformer_training_tiles(sample_count):
        stop = start + length
        if any(event_start < stop and event_stop > start for event_start, event_stop in spans):
            return start
    raise ValueError("selected event record has no fully observed positive ST16 tile")


def _atomic_new_json(path: Path, value: object) -> None:
    target = path.resolve(strict=False)
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def run(*, manifest_path: Path, tusz_root: Path, device_name: str) -> dict:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8" and device_name.startswith("cuda"):
        raise RuntimeError("CUDA smoke requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    manifest = load_common17_manifest(manifest_path, require_complete=True)
    record, eligible_missing_midline_count = _select_record(manifest)
    root = tusz_root.resolve(strict=True)
    edf_path = _safe_edf(root, record["edf_relative_path"])
    canonical = load_canonical_edf_record(edf_path)
    signal_authority = authorize_detector_signal_lineage_from_canonical_record(canonical)
    lineage = require_validated_detector_signal_lineage_authority(signal_authority)
    observed = tuple(lineage["observed_roster_authority"]["observed_standard_channel_ids"])
    if "FZ" in observed or "PZ" in observed or set(observed) != set(record["audited_observed_channel_ids"]):
        raise PermissionError("real EDF does not replay the manifest missing-midline roster")
    if lineage["canonical_physical_signal"]["source_tensor_sha256"] != record["canonical_source_tensor_sha256"]:
        raise PermissionError("real EDF tensor does not replay the common17 manifest")

    registry_path = ROOT / st.CONFIG_RELATIVE_PATH
    registry = st.load_registry(registry_path)
    referential_volts = np.asarray(canonical.observed_signal_volts.detach().cpu().numpy())
    transformed = st.apply_full_record_transform(
        referential_volts,
        variant_id=st.ST16_VARIANT_ID,
        signal_lineage_authority=signal_authority,
        registry=registry,
    )
    if tuple(transformed.receipt["output"]["typed_units"]) != CANONICAL_ST16_TYPED_UNITS:
        raise ValueError("real transform emitted the wrong ST16 axis positions")
    if transformed.signal.shape[0] != 16:
        raise ValueError("real ST16 transform did not emit 16 axes")

    spans = _target_spans(record, transformed.signal.shape[1])
    tile_start = _choose_positive_tile(spans, transformed.signal.shape[1])
    tile_stop = tile_start + st.TILE_SAMPLES
    target_np, mask_np, target_receipt = st.build_seizuretransformer_dense_target_pure_primitive(
        spans, target_start_sample=tile_start
    )
    if not np.all(mask_np == 1) or not int(np.count_nonzero(target_np)):
        raise ValueError("real ST16 smoke tile target is not a fully observed positive tile")

    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("real ST16 BF16 smoke requires the available CUDA GPU")
    torch.manual_seed(20260825)
    torch.cuda.manual_seed(20260825)
    torch.cuda.empty_cache()
    free_before, total_memory = torch.cuda.mem_get_info(device)
    torch.cuda.reset_peak_memory_stats(device)
    model = SeizureTransformer(
        in_channels=16,
        in_samples=st.TILE_SAMPLES,
        dim_feedforward=2048,
        num_layers=8,
        num_heads=4,
        drop_rate=0.1,
    ).to(device)
    optimizer = torch.optim.RAdam(
        model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8,
        weight_decay=2e-5, foreach=False,
    )
    inputs = torch.from_numpy(transformed.signal[:, tile_start:tile_stop].copy())[None].to(device)
    target = torch.from_numpy(target_np.copy())[None].to(device=device, dtype=torch.float32)
    positive = float(target.sum().item())
    negative = float(target.numel() - positive)
    positive_weight = min(50.0, negative / positive)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        probability = model(inputs)
    clipped = probability.float().clamp(1e-7, 1.0 - 1e-7)
    per_sample = -(positive_weight * target * torch.log(clipped) + (1.0 - target) * torch.log1p(-clipped))
    denominator = torch.where(target == 1, positive_weight, 1.0).sum()
    loss = per_sample.sum() / denominator
    if not bool(torch.isfinite(loss)):
        raise ValueError("real ST16 smoke loss is nonfinite")
    loss.backward()
    parameter_with_gradient_count = sum(parameter.grad is not None for parameter in model.parameters())
    if not parameter_with_gradient_count:
        raise RuntimeError("real ST16 smoke produced no gradients")
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    first_parameter = next(model.parameters())
    first_before = first_parameter.detach().clone()
    optimizer.step()
    torch.cuda.synchronize(device)
    step_seconds = time.perf_counter() - started
    first_update_linf = float((first_parameter.detach() - first_before).abs().max().cpu())
    free_after, _ = torch.cuda.mem_get_info(device)
    if first_update_linf <= 0.0:
        raise RuntimeError("real ST16 smoke optimizer did not update the model")

    result = _content_address(
        {
            "schema_version": "clinical_eeg_st16_common17_real_tile_smoke_v1",
            "status": "pass_real_missing_midline_source_train_BF16_one_step",
            "source_bindings": {
                "manifest_path": str(manifest_path.resolve()),
                "manifest_file_sha256": _file_sha(manifest_path.resolve()),
                "manifest_receipt_sha256": manifest["receipt_sha256"],
                "registry_path": str(registry_path.relative_to(ROOT)),
                "registry_file_sha256": _file_sha(registry_path),
                "registry_sha256": registry["registry_sha256"],
                "analysis_identity_id": record["analysis_identity_id"],
                "source_edf_relative_path_sha256": hashlib.sha256(record["edf_relative_path"].encode()).hexdigest(),
                "source_tensor_sha256": record["canonical_source_tensor_sha256"],
                "eligible_missing_FZ_PZ_positive_record_count": eligible_missing_midline_count,
            },
            "channel_and_transform_contract": {
                "observed_standard_channel_ids": list(observed),
                "FZ_observed": False,
                "PZ_observed": False,
                "zero_fill_interpolation_or_imputation_used": False,
                "ST18_intermediate_used": False,
                "ST16_typed_units": list(CANONICAL_ST16_TYPED_UNITS),
                "polarity": "first_named_referential_axis_minus_second_named_referential_axis",
                "transform_receipt_sha256": transformed.receipt["receipt_sha256"],
                "transform_output_payload_sha256": transformed.receipt["output"]["payload_receipt"]["payload_sha256"],
                "transformed_shape": list(transformed.signal.shape),
            },
            "tile_and_target": {
                "tile_start_sample": tile_start,
                "tile_stop_sample_exclusive": tile_stop,
                "tile_start_seconds": tile_start / st.TARGET_FS_HZ,
                "tile_stop_seconds": tile_stop / st.TARGET_FS_HZ,
                "input_shape": list(inputs.shape),
                "output_shape": list(probability.shape),
                "positive_target_sample_count": int(positive),
                "target_receipt_sha256": target_receipt["receipt_sha256"],
                "global_TERM_seiz_source_train_target_only": True,
            },
            "one_step_numeric_receipt": {
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "parameter_tensor_with_gradient_count": parameter_with_gradient_count,
                "loss": float(loss.detach().cpu()),
                "positive_weight": positive_weight,
                "preclip_gradient_global_L2_norm": float(gradient_norm.detach().cpu()),
                "first_parameter_update_Linf": first_update_linf,
                "step_seconds": step_seconds,
                "precision": "CUDA_bfloat16_autocast_forward_float32_loss_and_master_parameters",
                "process_peak_allocated_MiB": torch.cuda.max_memory_allocated(device) / 1048576.0,
                "process_peak_reserved_MiB": torch.cuda.max_memory_reserved(device) / 1048576.0,
                "device_free_before_MiB": free_before / 1048576.0,
                "device_free_after_MiB": free_after / 1048576.0,
                "device_total_MiB": total_memory / 1048576.0,
            },
            "firewall": {
                "source_train_EEG_and_global_TERM_seiz_target_used": True,
                "source_dev_or_source_eval_opened": False,
                "EDF_annotation_used": False,
                "Excel_doctor_text_clinical_history_video_behavior_used": False,
                "sleep_activation_ECG_EMG_EOG_used": False,
            },
            "claim_boundary": {
                "promotable_checkpoint_materialized": False,
                "detector_prediction_inventory_materialized": False,
                "detection_accuracy_or_efficiency_metric_materialized": False,
                "formal_training_started": False,
                "clinical_or_production_use_authorized": False,
            },
            "receipt_sha256": PENDING,
        }
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--tusz-root", type=Path, default=DEFAULT_TUSZ_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(manifest_path=args.manifest, tusz_root=args.tusz_root, device_name=args.device)
    _atomic_new_json(args.output, result)
    print(json.dumps({"output": str(args.output.resolve()), "receipt_sha256": result["receipt_sha256"], "status": result["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
