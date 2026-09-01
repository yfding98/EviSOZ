#!/usr/bin/env python3
"""Run the locked public source-only official LaBraM masked-code DAPT."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soz.data.labram_source_dapt import (  # noqa: E402
    PatientUniformEpochSampler,
    SourceDAPTWindowDataset,
    load_source_dapt_manifest,
)
from src.soz.models.labram_peft import (  # noqa: E402
    LABRAM_PEFT_ALPHA,
    LABRAM_PEFT_BLOCKS,
    LABRAM_PEFT_RANK,
    LABRAM_PEFT_TRAINABLE_PARAMETERS,
)
from src.soz.models.labram_source_dapt import (  # noqa: E402
    AUDITED_VQNSP_SHA256,
    LABRAM_DAPT_CODE_DIM,
    LABRAM_DAPT_INPUT_SCALE_FROM_VOLTS,
    LABRAM_DAPT_OBJECTIVE,
    OfficialFrozenLaBraMVQTokenizer,
    OfficialLaBraMSourceDAPT,
    exact_random_mask,
    masked_neural_code_objective,
    verify_zero_lora_official_pretraining_parity,
)


DEFAULT_MANIFEST = ROOT / "outputs/labram_source_only_dapt_manifest_v1_20260811/manifest.json"
DEFAULT_DEEPSOZ_SPLIT = ROOT / "outputs/deepsoz_tusz_patient_splits_v1/split_manifest.csv"
DEFAULT_TUSZ_EDF_ROOT = Path("/mnt/hd1/dyf/dataset/TUSZ/v2.0.3/edf")
DEFAULT_LABRAM_ROOT = Path("/mnt/hd1/dyf/workspace/LaBraM")
DEFAULT_OUTPUT = ROOT / "outputs/labram_source_only_dapt_v1_20260811"

TRAIN_WINDOWS_PER_PATIENT = 128
DEV_WINDOWS_PER_PATIENT = 32
EPOCHS = 20
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
GRADIENT_CLIP = 3.0
SEED = 20260811


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(
        path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_new_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(temporary, flags, 0o644)
        try:
            offset = 0
            while offset < len(content):
                written = os.write(descriptor, content[offset:])
                if written < 1:
                    raise OSError("Short write while publishing DAPT artifact")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.link(temporary, path, follow_symlinks=False)
    finally:
        temporary.unlink(missing_ok=True)
    _fsync_parent(path)


def _atomic_idempotent_bytes(path: Path, content: bytes) -> None:
    if path.exists():
        if path.is_symlink() or path.read_bytes() != content:
            raise FileExistsError(f"Existing final artifact differs: {path}")
        return
    _atomic_new_bytes(path, content)


def _save_adapter_new(path: Path, state: Mapping[str, torch.Tensor]) -> str:
    if path.is_symlink():
        raise FileExistsError(path)
    if path.exists():
        existing = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(existing, Mapping) or set(existing) != set(state):
            raise FileExistsError("Existing selected adapter has different keys")
        for key, expected in state.items():
            actual = existing[key]
            if not isinstance(actual, torch.Tensor) or not torch.equal(
                actual, expected.detach().cpu()
            ):
                raise FileExistsError("Existing selected adapter differs from resumed state")
        return hashlib.sha256(path.read_bytes()).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            torch.save(dict(state), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path, follow_symlinks=False)
    finally:
        temporary.unlink(missing_ok=True)
    _fsync_parent(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def _save_torch_payload_new(path: Path, payload: Mapping[str, object]) -> str:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path, follow_symlinks=False)
    finally:
        temporary.unlink(missing_ok=True)
    _fsync_parent(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--deepsoz-split-roster", type=Path, default=DEFAULT_DEEPSOZ_SPLIT)
    parser.add_argument("--tusz-edf-root", type=Path, default=DEFAULT_TUSZ_EDF_ROOT)
    parser.add_argument("--labram-root", type=Path, default=DEFAULT_LABRAM_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--train-windows-per-patient", type=int, default=TRAIN_WINDOWS_PER_PATIENT
    )
    parser.add_argument(
        "--dev-windows-per-patient", type=int, default=DEV_WINDOWS_PER_PATIENT
    )
    parser.add_argument(
        "--smoke-steps",
        type=int,
        default=0,
        help="Run only N optimizer steps and publish a non-promotable smoke receipt.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Load one real window, verify official parity/VQ/scope, and do not optimize.",
    )
    parser.add_argument(
        "--timing-pilot",
        action="store_true",
        help="One non-promotable epoch: exactly 1 train/dev window per patient.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Resume an epoch-atomic trusted checkpoint with an exact config hash.",
    )
    return parser.parse_args()


def _checkpoint_rank(path: Path) -> int:
    if path.name == "initial_state.pt":
        return -1
    match = re.fullmatch(r"epoch_(\d{3})_state\.pt", path.name)
    if match is None:
        raise ValueError(f"Unknown epoch-checkpoint filename: {path.name}")
    return int(match.group(1))


def _require_latest_resume_checkpoint(path: Path, *, output_dir: Path) -> None:
    if (output_dir / "run_receipt.json").exists():
        raise ValueError("Completed DAPT output cannot be resumed")
    checkpoint_dir = output_dir / "epoch_checkpoints"
    candidates = tuple(sorted(checkpoint_dir.glob("*.pt"), key=lambda item: item.name))
    if not candidates:
        raise ValueError("Resume output contains no epoch-atomic checkpoints")
    ranked = [(_checkpoint_rank(candidate), candidate.resolve(strict=True)) for candidate in candidates]
    latest_rank, latest = max(ranked, key=lambda item: item[0])
    del latest_rank
    if path.resolve(strict=True) != latest:
        raise ValueError(f"Resume must use the latest epoch checkpoint: {latest}")


def _validate_args(args: argparse.Namespace) -> None:
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    if args.train_windows_per_patient < 1 or args.dev_windows_per_patient < 1:
        raise ValueError("Per-patient window budgets must be positive")
    if args.num_workers < 0 or args.smoke_steps < 0:
        raise ValueError("num-workers/smoke-steps must be non-negative")
    if sum(bool(value) for value in (args.preflight_only, args.smoke_steps, args.timing_pilot)) > 1:
        raise ValueError("preflight-only, smoke-steps, and timing-pilot are mutually exclusive")
    if args.timing_pilot and (
        args.epochs != 1
        or args.train_windows_per_patient != 1
        or args.dev_windows_per_patient != 1
    ):
        raise ValueError(
            "timing-pilot is locked to --epochs 1 and one train/dev window per patient"
        )
    if args.resume_checkpoint is not None and (
        args.preflight_only or args.smoke_steps or args.timing_pilot
    ):
        raise ValueError("Only a full pretext-training run may be resumed")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    full_run = not args.preflight_only and not args.smoke_steps and not args.timing_pilot
    if full_run:
        locked = {
            "manifest": (args.manifest.resolve(), DEFAULT_MANIFEST.resolve()),
            "deepsoz_split_roster": (
                args.deepsoz_split_roster.resolve(),
                DEFAULT_DEEPSOZ_SPLIT.resolve(),
            ),
            "tusz_edf_root": (
                args.tusz_edf_root.resolve(),
                DEFAULT_TUSZ_EDF_ROOT.resolve(),
            ),
            "labram_root": (
                args.labram_root.resolve(),
                DEFAULT_LABRAM_ROOT.resolve(),
            ),
            "output_dir": (args.output_dir.resolve(), DEFAULT_OUTPUT.resolve()),
        }
        drifted = [name for name, (actual, expected) in locked.items() if actual != expected]
        scalar_contract = (
            args.device == "cuda"
            and args.epochs == EPOCHS
            and args.batch_size == BATCH_SIZE
            and args.train_windows_per_patient == TRAIN_WINDOWS_PER_PATIENT
            and args.dev_windows_per_patient == DEV_WINDOWS_PER_PATIENT
            and args.num_workers == 2
        )
        if drifted or not scalar_contract:
            raise ValueError(
                "Full DAPT is locked to the canonical paths, CUDA, 20 epochs, "
                "batch 4, 128/32 windows per patient, and 2 workers; "
                f"drifted_paths={drifted}"
            )
    if args.resume_checkpoint is None:
        if args.output_dir.exists() or args.output_dir.is_symlink():
            raise FileExistsError(
                f"Refusing to reuse an existing DAPT output directory: {args.output_dir}"
            )
    else:
        checkpoint = args.resume_checkpoint.resolve(strict=True)
        if args.output_dir.is_symlink():
            raise ValueError("Resume output directory cannot be a symlink")
        output = args.output_dir.resolve(strict=True)
        if checkpoint.parent.parent != output:
            raise ValueError("Resume checkpoint must be under OUTPUT/epoch_checkpoints/")
        _require_latest_resume_checkpoint(checkpoint, output_dir=output)


def _build_run_config(
    *,
    args: argparse.Namespace,
    manifest_sha256: str,
    foundation_checkpoint_sha256: str,
    vq_checkpoint_sha256: str,
) -> tuple[dict[str, object], str]:
    mode = (
        "preflight_only"
        if args.preflight_only
        else "non_promotable_smoke"
        if args.smoke_steps
        else "non_promotable_timing_pilot"
        if args.timing_pilot
        else "full_pretext_training"
    )
    config: dict[str, object] = {
        "schema_version": "soz_labram_source_only_dapt_run_config_v1",
        "mode": mode,
        "manifest_sha256": manifest_sha256,
        "foundation_checkpoint_sha256": foundation_checkpoint_sha256,
        "vq_checkpoint_sha256": vq_checkpoint_sha256,
        "objective": LABRAM_DAPT_OBJECTIVE,
        "lora_blocks": list(LABRAM_PEFT_BLOCKS),
        "lora_rank": LABRAM_PEFT_RANK,
        "lora_alpha": LABRAM_PEFT_ALPHA,
        "trainable_parameter_count": LABRAM_PEFT_TRAINABLE_PARAMETERS,
        "seed": SEED,
        "device": args.device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "train_windows_per_patient": args.train_windows_per_patient,
        "dev_windows_per_patient": args.dev_windows_per_patient,
        "num_workers": args.num_workers,
        "smoke_steps": args.smoke_steps,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": LEARNING_RATE,
            "betas": [0.9, 0.98],
            "eps": 1e-8,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip": GRADIENT_CLIP,
        },
        "selection_metric": "pretext_dev_patient_macro_loss_only",
        "implementation_sha256": {
            "runner": _file_sha256(Path(__file__).resolve()),
            "data": _file_sha256(
                ROOT / "src/soz/data/labram_source_dapt.py"
            ),
            "model": _file_sha256(
                ROOT / "src/soz/models/labram_source_dapt.py"
            ),
            "peft": _file_sha256(ROOT / "src/soz/models/labram_peft.py"),
        },
    }
    return config, hashlib.sha256(_json_bytes(config)).hexdigest()


_EPOCH_CHECKPOINT_KEYS = {
    "schema_version",
    "manifest_sha256",
    "run_config_sha256",
    "epoch_completed",
    "completed_optimizer_steps",
    "adapter_state",
    "optimizer_state",
    "best_adapter_state",
    "best_pretext_dev_loss",
    "best_epoch",
    "zero_lora_pretext_dev",
    "epoch_history",
    "torch_cpu_rng_state",
    "torch_cuda_rng_state_all",
    "numpy_rng_state",
    "last_gradient_receipt",
}


def _capture_rng_state(*, device: torch.device) -> dict[str, object]:
    return {
        "torch_cpu_rng_state": torch.get_rng_state().clone(),
        "torch_cuda_rng_state_all": (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if device.type == "cuda"
            else []
        ),
        "numpy_rng_state": np.random.get_state(),
    }


def _restore_rng_state(payload: Mapping[str, object], *, device: torch.device) -> None:
    cpu_state = payload["torch_cpu_rng_state"]
    cuda_states = payload["torch_cuda_rng_state_all"]
    numpy_state = payload["numpy_rng_state"]
    if not isinstance(cpu_state, torch.Tensor) or cpu_state.dtype != torch.uint8:
        raise TypeError("Resume checkpoint torch CPU RNG state is invalid")
    if not isinstance(cuda_states, list) or any(
        not isinstance(state, torch.Tensor) or state.dtype != torch.uint8
        for state in cuda_states
    ):
        raise TypeError("Resume checkpoint CUDA RNG-state list is invalid")
    if not isinstance(numpy_state, tuple) or len(numpy_state) != 5:
        raise TypeError("Resume checkpoint NumPy RNG state is invalid")
    if device.type == "cuda":
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError("Resume CUDA RNG-state count differs from visible devices")
        torch.cuda.set_rng_state_all(cuda_states)
    elif cuda_states:
        raise ValueError("CUDA checkpoint cannot be resumed as a CPU run")
    torch.set_rng_state(cpu_state)
    np.random.set_state(numpy_state)


def _load_resume_checkpoint(
    path: Path,
    *,
    expected_manifest_sha256: str,
    expected_run_config_sha256: str,
) -> Mapping[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or set(payload) != _EPOCH_CHECKPOINT_KEYS:
        raise ValueError("Epoch checkpoint schema/keys changed")
    if payload["schema_version"] != "soz_labram_source_only_dapt_epoch_checkpoint_v1":
        raise ValueError("Epoch checkpoint schema changed")
    if payload["manifest_sha256"] != expected_manifest_sha256:
        raise ValueError("Resume checkpoint manifest hash differs from this run")
    if payload["run_config_sha256"] != expected_run_config_sha256:
        raise ValueError("Resume checkpoint config/code hash differs from this run")
    epoch = payload["epoch_completed"]
    steps = payload["completed_optimizer_steps"]
    history = payload["epoch_history"]
    if (
        isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or epoch < -1
        or isinstance(steps, bool)
        or not isinstance(steps, int)
        or not isinstance(history, list)
        or (epoch == -1 and (steps != 0 or history != []))
        or (epoch >= 0 and (steps < 1 or len(history) != epoch + 1))
    ):
        raise ValueError("Resume checkpoint epoch/history coordinates are invalid")
    # Validate all serialized random states before any optimizer update can run.
    if not isinstance(payload["torch_cpu_rng_state"], torch.Tensor):
        raise TypeError("Resume checkpoint lacks torch CPU RNG state")
    if not isinstance(payload["torch_cuda_rng_state_all"], list):
        raise TypeError("Resume checkpoint lacks CUDA RNG states")
    if not isinstance(payload["numpy_rng_state"], tuple):
        raise TypeError("Resume checkpoint lacks NumPy RNG state")
    if payload["last_gradient_receipt"] is not None and not isinstance(
        payload["last_gradient_receipt"], Mapping
    ):
        raise TypeError("Resume checkpoint last-gradient receipt is invalid")
    for key in ("adapter_state", "optimizer_state", "best_adapter_state"):
        if not isinstance(payload[key], Mapping):
            raise TypeError(f"Resume checkpoint {key} must be a mapping")
    baseline = payload["zero_lora_pretext_dev"]
    if not isinstance(baseline, Mapping) or not math.isfinite(
        float(baseline.get("patient_macro_pretext_loss", float("nan")))
    ):
        raise ValueError("Resume checkpoint lacks a finite zero-LoRA dev baseline")
    if not math.isfinite(float(payload["best_pretext_dev_loss"])):
        raise ValueError("Resume checkpoint best pretext-dev loss is non-finite")
    best_epoch = payload["best_epoch"]
    if best_epoch != "zero_lora_baseline" and (
        isinstance(best_epoch, bool)
        or not isinstance(best_epoch, int)
        or not 0 <= best_epoch <= epoch
    ):
        raise ValueError("Resume checkpoint best epoch is invalid")
    return payload


def _fixed_sample_masks(batch: Mapping[str, object], device: torch.device) -> torch.Tensor:
    masks = []
    grid_values = batch["grid_index"].tolist()
    for uid, grid_index in zip(batch["record_uid"], grid_values):
        digest = hashlib.sha256(f"{SEED}\0{uid}\0{grid_index}".encode("ascii")).digest()
        mask_seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
        masks.append(exact_random_mask(1, seed=mask_seed, device=device)[0])
    return torch.stack(masks)


def _train_sample_masks(
    batch: Mapping[str, object], *, epoch: int, device: torch.device
) -> torch.Tensor:
    masks = []
    grid_values = batch["grid_index"].tolist()
    for patient, uid, grid_index in zip(
        batch["patient_id"], batch["record_uid"], grid_values
    ):
        digest = hashlib.sha256(
            f"{SEED}\0train\0{epoch}\0{patient}\0{uid}\0{grid_index}".encode(
                "ascii"
            )
        ).digest()
        mask_seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
        masks.append(exact_random_mask(1, seed=mask_seed, device=device)[0])
    return torch.stack(masks)


def _preflight(
    *,
    train_dataset: SourceDAPTWindowDataset,
    model: OfficialLaBraMSourceDAPT,
    tokenizer: OfficialFrozenLaBraMVQTokenizer,
    labram_root: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, object]:
    sample = train_dataset[0]
    eeg = sample["eeg"].unsqueeze(0).to(device=device, dtype=torch.float32)
    positions = sample["position_ids"].to(device=device)
    mask = exact_random_mask(1, seed=SEED, device=device)
    if tuple(eeg.shape) != (1, 19, 8, 200) or not torch.isfinite(eeg).all():
        raise RuntimeError("Real source-DAPT smoke window has invalid geometry/values")
    channel_std = eeg.reshape(19, -1).std(dim=1)
    channel_ptp = eeg.reshape(19, -1).amax(dim=1) - eeg.reshape(19, -1).amin(dim=1)
    if torch.count_nonzero(channel_std > 0).item() < 19:
        raise RuntimeError("Real source-DAPT smoke window contains a constant channel")
    parity = verify_zero_lora_official_pretraining_parity(
        model,
        official_root=labram_root,
        checkpoint_path=checkpoint_path,
        patches_volts=eeg,
        position_ids=positions,
        bool_masked_pos=mask,
    )
    with torch.no_grad():
        codes = tokenizer(eeg, positions)
    unique_codes = int(torch.unique(codes).numel())
    if unique_codes < 2:
        raise RuntimeError("Official VQ-NSP collapsed to one code on the real smoke window")
    return {
        "real_window_shape": list(eeg.shape),
        "real_window_abs_max_volts": float(eeg.abs().max().cpu()),
        "real_window_min_channel_std_volts": float(channel_std.min().cpu()),
        "real_window_min_channel_ptp_volts": float(channel_ptp.min().cpu()),
        "model_input_scale_from_volts": LABRAM_DAPT_INPUT_SCALE_FROM_VOLTS,
        "position_ids": positions.cpu().tolist(),
        "zero_lora_official_pretraining_parity": parity,
        "vq_code_dim": tokenizer.code_dim,
        "vq_codebook_size": tokenizer.codebook_size,
        "vq_unique_codes_in_smoke_window": unique_codes,
        "vq_code_min": int(codes.min()),
        "vq_code_max": int(codes.max()),
        "trainable_parameter_count": model.n_trainable_parameters,
        "trainable_parameter_names": list(model.trainable_parameter_names),
    }


def _gradient_receipt(model: OfficialLaBraMSourceDAPT) -> dict[str, object]:
    values: dict[str, object] = {}
    for block in LABRAM_PEFT_BLOCKS:
        adapter = model._lora(block)
        for factor in ("A", "B"):
            gradient = getattr(adapter, f"lora_{factor}").grad
            key = f"blocks.{block}.lora_{factor}"
            values[key] = {
                "present": gradient is not None,
                "finite": bool(gradient is not None and torch.isfinite(gradient).all()),
                "absolute_sum": (
                    None if gradient is None else float(gradient.detach().abs().sum().cpu())
                ),
            }
    return values


def _train_epoch(
    *,
    model: OfficialLaBraMSourceDAPT,
    tokenizer: OfficialFrozenLaBraMVQTokenizer,
    loader: DataLoader,
    sampler: PatientUniformEpochSampler,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    device: torch.device,
    smoke_steps: int,
    completed_steps: int,
) -> tuple[dict[str, float], int, dict[str, object] | None]:
    sampler.set_epoch(epoch)
    model.train()
    tokenizer.eval()
    if model.backbone.training or model.lm_head.training:
        raise RuntimeError("Frozen LaBraM backbone/head left eval mode during DAPT")
    if tokenizer.tokenizer.training:
        raise RuntimeError("Frozen VQ tokenizer left eval mode during DAPT")
    tokenizer._assert_frozen()
    totals = defaultdict(float)
    samples = 0
    last_gradient: dict[str, object] | None = None
    for step, batch in enumerate(loader):
        eeg = batch["eeg"].to(device=device, dtype=torch.float32, non_blocking=True)
        positions = batch["position_ids"].to(device=device, non_blocking=True)
        mask = _train_sample_masks(batch, epoch=epoch, device=device)
        optimizer.zero_grad(set_to_none=True)
        output = masked_neural_code_objective(model, tokenizer, eeg, positions, mask)
        output.loss.backward()
        last_gradient = _gradient_receipt(model)
        for item in last_gradient.values():
            if not item["present"] or not item["finite"]:
                raise RuntimeError("LaBraM source-DAPT LoRA gradient is missing/non-finite")
        if completed_steps == 0:
            for block in LABRAM_PEFT_BLOCKS:
                if last_gradient[f"blocks.{block}.lora_B"]["absolute_sum"] <= 0:
                    raise RuntimeError("Initial zero-LoRA step must give each LoRA-B a non-zero gradient")
                if last_gradient[f"blocks.{block}.lora_A"]["absolute_sum"] != 0:
                    raise RuntimeError("Initial zero-LoRA step must leave LoRA-A gradient exactly zero")
        elif completed_steps == 1:
            for block in LABRAM_PEFT_BLOCKS:
                for factor in ("A", "B"):
                    if last_gradient[f"blocks.{block}.lora_{factor}"]["absolute_sum"] <= 0:
                        raise RuntimeError("Second DAPT step must give every LoRA factor a non-zero gradient")
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            GRADIENT_CLIP,
        )
        optimizer.step()
        batch_size = eeg.shape[0]
        samples += batch_size
        totals["loss"] += float(output.loss.detach()) * batch_size
        totals["masked_loss"] += float(output.masked_loss.detach()) * batch_size
        totals["complementary_loss"] += float(output.complementary_loss.detach()) * batch_size
        totals["masked_accuracy"] += float(output.masked_accuracy.detach()) * batch_size
        totals["complementary_accuracy"] += float(output.complementary_accuracy.detach()) * batch_size
        completed_steps += 1
        if smoke_steps and completed_steps >= smoke_steps:
            break
    if samples < 1:
        raise RuntimeError("Source-DAPT training epoch produced no samples")
    return ({key: value / samples for key, value in totals.items()}, completed_steps, last_gradient)


def _evaluate_patient_macro(
    *,
    model: OfficialLaBraMSourceDAPT,
    tokenizer: OfficialFrozenLaBraMVQTokenizer,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    tokenizer.eval()
    if model.backbone.training or model.lm_head.training or tokenizer.tokenizer.training:
        raise RuntimeError("Frozen pretext modules must remain in eval mode")
    tokenizer._assert_frozen()
    patient_losses: dict[str, list[float]] = defaultdict(list)
    patient_accuracies: dict[str, list[float]] = defaultdict(list)
    with torch.no_grad():
        for batch in loader:
            masks = _fixed_sample_masks(batch, device)
            for index, patient in enumerate(batch["patient_id"]):
                eeg = batch["eeg"][index : index + 1].to(device=device, dtype=torch.float32)
                positions = batch["position_ids"][index : index + 1].to(device=device)
                output = masked_neural_code_objective(
                    model, tokenizer, eeg, positions, masks[index : index + 1]
                )
                patient_losses[str(patient)].append(float(output.loss))
                patient_accuracies[str(patient)].append(
                    0.5
                    * (
                        float(output.masked_accuracy)
                        + float(output.complementary_accuracy)
                    )
                )
    if len(patient_losses) != 12 or set(patient_losses) != set(patient_accuracies):
        raise RuntimeError("Pretext validation must cover exactly 12 patients")
    macro_loss = float(
        np.mean([np.mean(values) for values in patient_losses.values()])
    )
    macro_accuracy = float(
        np.mean([np.mean(values) for values in patient_accuracies.values()])
    )
    if not math.isfinite(macro_loss) or not math.isfinite(macro_accuracy):
        raise RuntimeError("Pretext validation metric is non-finite")
    return {
        "patient_macro_pretext_loss": macro_loss,
        "patient_macro_code_accuracy": macro_accuracy,
        "patient_count": float(len(patient_losses)),
        "windows_per_patient": float(min(map(len, patient_losses.values()))),
    }


def main() -> int:
    total_started = time.perf_counter()
    args = parse_args()
    _validate_args(args)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(SEED)
    device = torch.device(args.device)

    manifest = load_source_dapt_manifest(
        args.manifest,
        deepsoz_split_roster=args.deepsoz_split_roster,
        tusz_root=args.tusz_edf_root,
        verify_file_inventory=True,
    )
    train_dataset = SourceDAPTWindowDataset(manifest, split="pretext_train")
    dev_dataset = SourceDAPTWindowDataset(manifest, split="pretext_dev")
    train_sampler = PatientUniformEpochSampler(
        train_dataset,
        windows_per_patient=args.train_windows_per_patient,
        seed=SEED,
    )
    dev_sampler = PatientUniformEpochSampler(
        dev_dataset,
        windows_per_patient=args.dev_windows_per_patient,
        seed=SEED + 17,
    )
    dev_sampler.set_epoch(0)
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
        persistent_workers=args.num_workers > 0,
        multiprocessing_context=("spawn" if args.num_workers > 0 else None),
        drop_last=False,
    )
    dev_loader = DataLoader(
        dev_dataset,
        sampler=dev_sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
        persistent_workers=args.num_workers > 0,
        multiprocessing_context=("spawn" if args.num_workers > 0 else None),
        drop_last=False,
    )

    labram_root = args.labram_root.resolve(strict=True)
    checkpoint = labram_root / "checkpoints/labram-base.pth"
    tokenizer_checkpoint = labram_root / "checkpoints/vqnsp.pth"
    model = OfficialLaBraMSourceDAPT(
        modeling_path=labram_root / "modeling_finetune.py",
        checkpoint_path=checkpoint,
    ).to(device=device, dtype=torch.float32)
    tokenizer = OfficialFrozenLaBraMVQTokenizer(
        official_root=labram_root,
        checkpoint_path=tokenizer_checkpoint,
        expected_sha256=AUDITED_VQNSP_SHA256,
    ).to(device=device, dtype=torch.float32)
    run_config, run_config_sha256 = _build_run_config(
        args=args,
        manifest_sha256=manifest.sha256,
        foundation_checkpoint_sha256=model.checkpoint_sha256,
        vq_checkpoint_sha256=tokenizer.checkpoint_sha256,
    )
    preflight_started = time.perf_counter()
    preflight = _preflight(
        train_dataset=train_dataset,
        model=model,
        tokenizer=tokenizer,
        labram_root=labram_root,
        checkpoint_path=checkpoint,
        device=device,
    )
    preflight_seconds = time.perf_counter() - preflight_started
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    receipt: dict[str, object] = {
        "schema_version": "soz_labram_source_only_dapt_run_v1",
        "mode": run_config["mode"],
        "run_config": run_config,
        "run_config_sha256": run_config_sha256,
        "manifest_path": str(manifest.path),
        "manifest_sha256": manifest.sha256,
        "device": str(device),
        "objective": LABRAM_DAPT_OBJECTIVE,
        "foundation_checkpoint_sha256": model.checkpoint_sha256,
        "vq_checkpoint_sha256": tokenizer.checkpoint_sha256,
        "vq_code_dim": LABRAM_DAPT_CODE_DIM,
        "lora_blocks": list(LABRAM_PEFT_BLOCKS),
        "lora_rank": LABRAM_PEFT_RANK,
        "lora_alpha": LABRAM_PEFT_ALPHA,
        "trainable_parameter_count": LABRAM_PEFT_TRAINABLE_PARAMETERS,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": LEARNING_RATE,
            "betas": [0.9, 0.98],
            "eps": 1e-8,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip": GRADIENT_CLIP,
        },
        "selection_metric": "pretext_dev_patient_macro_loss_only",
        "adapter_serialization": "torch.save_internal_trusted_artifact_adapter_tensors_only",
        "target_values_loaded": False,
        "private_data_loaded": False,
        "annotation_times_used": False,
        "preflight": preflight,
        "preflight_wall_time_seconds": preflight_seconds,
        "epochs": [],
    }
    if args.resume_checkpoint is None:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        _atomic_new_bytes(
            args.output_dir / "run_config.json", _json_bytes(run_config)
        )
    else:
        frozen_config_path = args.output_dir / "run_config.json"
        frozen_config = frozen_config_path.read_bytes()
        if frozen_config != _json_bytes(run_config):
            raise ValueError("Resume run_config.json differs from current exact config")
    if args.preflight_only:
        receipt["training_started"] = False
        receipt["training_completed"] = False
        receipt["qualification_pending"] = True
        receipt["representation_qualified"] = False
        receipt["soz_promotion"] = False
        receipt["candidate_promotable"] = False
        if device.type == "cuda":
            receipt["cuda_max_memory_allocated_bytes"] = int(
                torch.cuda.max_memory_allocated(device)
            )
            receipt["cuda_max_memory_reserved_bytes"] = int(
                torch.cuda.max_memory_reserved(device)
            )
        receipt["wall_time_seconds"] = time.perf_counter() - total_started
        _atomic_new_bytes(args.output_dir / "run_receipt.json", _json_bytes(receipt))
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=LEARNING_RATE,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=WEIGHT_DECAY,
    )
    best_loss = math.inf
    best_epoch: int | str | None = None
    best_state = model.lora_state_dict()
    zero_lora_pretext_dev: Mapping[str, float] | None = None
    completed_steps = 0
    start_epoch = 0
    last_gradient = None
    if args.resume_checkpoint is not None:
        resumed = _load_resume_checkpoint(
            args.resume_checkpoint.resolve(strict=True),
            expected_manifest_sha256=manifest.sha256,
            expected_run_config_sha256=run_config_sha256,
        )
        model.load_lora_state_dict(resumed["adapter_state"])
        optimizer.load_state_dict(resumed["optimizer_state"])
        best_state = dict(resumed["best_adapter_state"])
        best_loss = float(resumed["best_pretext_dev_loss"])
        best_epoch = resumed["best_epoch"]
        zero_lora_pretext_dev = resumed["zero_lora_pretext_dev"]
        completed_steps = int(resumed["completed_optimizer_steps"])
        start_epoch = int(resumed["epoch_completed"]) + 1
        last_gradient = resumed["last_gradient_receipt"]
        receipt["epochs"] = list(resumed["epoch_history"])
        _restore_rng_state(resumed, device=device)
        receipt["resumed_from_checkpoint"] = str(
            args.resume_checkpoint.resolve(strict=True)
        )
        receipt["resumed_epoch_completed"] = start_epoch - 1
        receipt["zero_lora_pretext_dev"] = zero_lora_pretext_dev
    elif not args.smoke_steps and not args.timing_pilot:
        baseline_started = time.perf_counter()
        baseline = _evaluate_patient_macro(
            model=model, tokenizer=tokenizer, loader=dev_loader, device=device
        )
        receipt["zero_lora_pretext_dev_wall_time_seconds"] = (
            time.perf_counter() - baseline_started
        )
        receipt["zero_lora_pretext_dev"] = baseline
        zero_lora_pretext_dev = baseline
        best_loss = baseline["patient_macro_pretext_loss"]
        best_epoch = "zero_lora_baseline"
        initial_payload = {
            "schema_version": "soz_labram_source_only_dapt_epoch_checkpoint_v1",
            "manifest_sha256": manifest.sha256,
            "run_config_sha256": run_config_sha256,
            "epoch_completed": -1,
            "completed_optimizer_steps": 0,
            "adapter_state": model.lora_state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_adapter_state": best_state,
            "best_pretext_dev_loss": best_loss,
            "best_epoch": best_epoch,
            "zero_lora_pretext_dev": zero_lora_pretext_dev,
            "epoch_history": [],
            "last_gradient_receipt": None,
            **_capture_rng_state(device=device),
        }
        initial_checkpoint_path = (
            args.output_dir / "epoch_checkpoints" / "initial_state.pt"
        )
        initial_checkpoint_sha256 = _save_torch_payload_new(
            initial_checkpoint_path, initial_payload
        )
        _atomic_new_bytes(
            args.output_dir / "initial_progress.json",
            _json_bytes(
                {
                    "schema_version": "soz_labram_source_only_dapt_initial_progress_v1",
                    "manifest_sha256": manifest.sha256,
                    "run_config_sha256": run_config_sha256,
                    "epoch_completed": -1,
                    "completed_optimizer_steps": 0,
                    "epoch_checkpoint": str(initial_checkpoint_path.resolve()),
                    "epoch_checkpoint_sha256": initial_checkpoint_sha256,
                    "zero_lora_pretext_dev": zero_lora_pretext_dev,
                    "candidate_promotable": False,
                    "representation_qualified": False,
                    "soz_promotion": False,
                }
            ),
        )

    for epoch in range(start_epoch, args.epochs):
        train_started = time.perf_counter()
        train_metrics, completed_steps, last_gradient = _train_epoch(
            model=model,
            tokenizer=tokenizer,
            loader=train_loader,
            sampler=train_sampler,
            optimizer=optimizer,
            epoch=epoch,
            device=device,
            smoke_steps=args.smoke_steps,
            completed_steps=completed_steps,
        )
        row: dict[str, object] = {
            "epoch": epoch,
            "train": train_metrics,
            "train_wall_time_seconds": time.perf_counter() - train_started,
        }
        if not args.smoke_steps:
            validation_started = time.perf_counter()
            validation = _evaluate_patient_macro(
                model=model, tokenizer=tokenizer, loader=dev_loader, device=device
            )
            row["pretext_dev"] = validation
            row["pretext_dev_wall_time_seconds"] = (
                time.perf_counter() - validation_started
            )
            if args.timing_pilot:
                best_state = model.lora_state_dict()
            else:
                candidate_loss = validation["patient_macro_pretext_loss"]
                if candidate_loss < best_loss:
                    best_loss = candidate_loss
                    best_epoch = epoch
                    best_state = model.lora_state_dict()
        receipt["epochs"].append(row)
        if not args.smoke_steps and not args.timing_pilot:
            epoch_payload = {
                "schema_version": "soz_labram_source_only_dapt_epoch_checkpoint_v1",
                "manifest_sha256": manifest.sha256,
                "run_config_sha256": run_config_sha256,
                "epoch_completed": epoch,
                "completed_optimizer_steps": completed_steps,
                "adapter_state": model.lora_state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_adapter_state": best_state,
                "best_pretext_dev_loss": (
                    None if args.timing_pilot else best_loss
                ),
                "best_epoch": best_epoch,
                "zero_lora_pretext_dev": zero_lora_pretext_dev,
                "epoch_history": list(receipt["epochs"]),
                "last_gradient_receipt": last_gradient,
                **_capture_rng_state(device=device),
            }
            checkpoint_path = (
                args.output_dir
                / "epoch_checkpoints"
                / f"epoch_{epoch:03d}_state.pt"
            )
            epoch_checkpoint_sha256 = _save_torch_payload_new(
                checkpoint_path, epoch_payload
            )
            progress = {
                "schema_version": "soz_labram_source_only_dapt_epoch_progress_v1",
                "manifest_sha256": manifest.sha256,
                "run_config_sha256": run_config_sha256,
                "epoch_completed": epoch,
                "completed_optimizer_steps": completed_steps,
                "epoch_checkpoint": str(checkpoint_path.resolve()),
                "epoch_checkpoint_sha256": epoch_checkpoint_sha256,
                "mode": run_config["mode"],
                "candidate_promotable": False,
                "representation_qualified": False,
                "soz_promotion": False,
                "epoch_metrics": row,
            }
            _atomic_new_bytes(
                args.output_dir / f"epoch_{epoch:03d}_progress.json",
                _json_bytes(progress),
            )
        if args.smoke_steps and completed_steps >= args.smoke_steps:
            break

    receipt["training_started"] = True
    receipt["optimizer_steps"] = completed_steps
    receipt["last_gradient_receipt"] = last_gradient
    receipt["best_epoch_by_pretext_dev_loss_only"] = best_epoch
    receipt["best_pretext_dev_patient_macro_loss"] = (
        None if args.smoke_steps or args.timing_pilot else best_loss
    )
    if args.smoke_steps or args.timing_pilot:
        best_state = model.lora_state_dict()
    receipt["training_completed"] = bool(
        not args.smoke_steps
        and not args.timing_pilot
        and len(receipt["epochs"]) == args.epochs
    )
    receipt["timing_pilot_completed"] = bool(
        args.timing_pilot and len(receipt["epochs"]) == 1
    )
    receipt["pretext_point_estimate_improved_over_zero_lora"] = bool(
        not args.smoke_steps
        and not args.timing_pilot
        and isinstance(best_epoch, int)
    )
    receipt["qualification_pending"] = True
    receipt["representation_qualified"] = False
    receipt["soz_promotion"] = False
    receipt["candidate_promotable"] = False
    adapter_sha = _save_adapter_new(args.output_dir / "selected_lora.pt", best_state)
    receipt["selected_adapter_sha256"] = adapter_sha
    if device.type == "cuda":
        receipt["cuda_max_memory_allocated_bytes"] = int(
            torch.cuda.max_memory_allocated(device)
        )
        receipt["cuda_max_memory_reserved_bytes"] = int(
            torch.cuda.max_memory_reserved(device)
        )
    receipt["wall_time_seconds"] = time.perf_counter() - total_started
    _atomic_idempotent_bytes(
        args.output_dir / "run_receipt.json", _json_bytes(receipt)
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
