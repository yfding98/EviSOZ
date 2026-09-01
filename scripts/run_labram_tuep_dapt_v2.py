#!/usr/bin/env python3
"""Run locked target-free TUEP LaBraM DAPT-v2 continuation training."""

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
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soz.data.labram_tuep_dapt_v2 import (  # noqa: E402
    EXPECTED_DEV_PATIENTS,
    EXPECTED_QUALIFICATION_PATIENTS,
    EXPECTED_TRAIN_PATIENTS,
    PatientUniformEpochSampler,
    TUEPDAPTV2WindowDataset,
    load_tuep_dapt_v2_manifest,
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
    OfficialFrozenLaBraMVQTokenizer,
    OfficialLaBraMSourceDAPT,
    exact_random_mask,
    verify_zero_lora_official_pretraining_parity,
)
from src.soz.models.labram_source_dapt_v2 import (  # noqa: E402
    LABRAM_DAPT_V2_ENTROPY_FLOOR_WEIGHT,
    LABRAM_DAPT_V2_MINIMUM_PERPLEXITY_RATIO,
    LABRAM_DAPT_V2_OBJECTIVE,
    LABRAM_DAPT_V2_TEACHER_KL_WEIGHT,
    LABRAM_DAPT_V2_TEACHER_TEMPERATURE,
    diversity_preserving_masked_neural_code_objective,
)


DEFAULT_MANIFEST = (
    ROOT / "outputs/labram_tuep_dapt_v2_manifest_20260811/manifest.json"
)
DEFAULT_TUEP_ROOT = Path("/mnt/hd1/dyf/dataset/tuh_eeg_epilepsy/v2.0.1")
DEFAULT_LABRAM_ROOT = Path("/mnt/hd1/dyf/workspace/LaBraM")
DEFAULT_OUTPUT = ROOT / "outputs/labram_tuep_dapt_v2_20260811"
EXPECTED_MANIFEST_SHA256 = (
    "83abb54ec7a22820368ddc06d78d6f32338595d5405834ecf0d7daed980e165b"
)

TRAIN_WINDOWS_PER_PATIENT = 64
DEV_WINDOWS_PER_PATIENT = 16
EPOCHS = 10
BATCH_SIZE = 4
NUM_WORKERS = 2
LEARNING_RATE = 5e-5
ADAM_BETAS = (0.9, 0.98)
ADAM_EPS = 1e-8
WEIGHT_DECAY = 1e-2
GRADIENT_CLIP = 3.0
SEED = 20260811
HARD_PREDICTION_MINIMUM_PERPLEXITY_RATIO = 0.95
HARD_PREDICTION_LOG_PERPLEXITY_MARGIN = math.log(
    HARD_PREDICTION_MINIMUM_PERPLEXITY_RATIO
)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _file_sha256(path_value: str | Path) -> str:
    path = Path(path_value).resolve(strict=True)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"DAPT-v2 lineage input is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_new_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path, follow_symlinks=False)
    finally:
        temporary.unlink(missing_ok=True)
    _fsync_directory(path.parent)


def _atomic_idempotent_bytes(path: Path, content: bytes) -> None:
    if path.exists():
        if path.is_symlink() or path.read_bytes() != content:
            raise FileExistsError(f"Existing final DAPT-v2 artifact differs: {path}")
        return
    _atomic_new_bytes(path, content)


def _save_torch_new(path: Path, payload: Mapping[str, object]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path, follow_symlinks=False)
    finally:
        temporary.unlink(missing_ok=True)
    _fsync_directory(path.parent)
    return _file_sha256(path)


def _save_adapter_new(path: Path, state: Mapping[str, torch.Tensor]) -> str:
    canonical = {key: value.detach().cpu().clone() for key, value in state.items()}
    if path.is_symlink():
        raise FileExistsError(path)
    if path.exists():
        existing = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(existing, Mapping) or set(existing) != set(canonical):
            raise FileExistsError("Existing DAPT-v2 adapter schema differs")
        if any(
            not isinstance(existing[key], torch.Tensor)
            or not torch.equal(existing[key], expected)
            for key, expected in canonical.items()
        ):
            raise FileExistsError("Existing DAPT-v2 adapter tensors differ")
        return _file_sha256(path)
    return _save_torch_new(path, canonical)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--tuep-root", type=Path, default=DEFAULT_TUEP_ROOT)
    parser.add_argument("--labram-root", type=Path, default=DEFAULT_LABRAM_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--smoke-steps",
        type=int,
        default=0,
        help="Run N optimizer steps and publish a non-selectable smoke receipt.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Resume only the latest epoch-atomic checkpoint of the formal run.",
    )
    return parser.parse_args()


def _checkpoint_rank(path: Path) -> int:
    if path.name == "initial_state.pt":
        return -1
    match = re.fullmatch(r"epoch_(\d{3})_state\.pt", path.name)
    if match is None:
        raise ValueError(f"Unknown DAPT-v2 checkpoint filename: {path.name}")
    return int(match.group(1))


def _require_latest_checkpoint(path: Path, output_dir: Path) -> None:
    if (output_dir / "run_receipt.json").exists():
        raise ValueError("A completed DAPT-v2 run cannot be resumed")
    candidates = tuple((output_dir / "epoch_checkpoints").glob("*.pt"))
    if not candidates:
        raise ValueError("DAPT-v2 resume output has no epoch checkpoint")
    latest = max(candidates, key=_checkpoint_rank).resolve(strict=True)
    if path.resolve(strict=True) != latest:
        raise ValueError(f"DAPT-v2 resume must use latest checkpoint: {latest}")
    rank = _checkpoint_rank(latest)
    progress_path = (
        output_dir / "initial_progress.json"
        if rank == -1
        else output_dir / f"epoch_{rank:03d}_progress.json"
    )
    progress = json.loads(progress_path.resolve(strict=True).read_text(encoding="utf-8"))
    if (
        not isinstance(progress, Mapping)
        or Path(str(progress.get("epoch_checkpoint"))).resolve(strict=True) != latest
        or progress.get("epoch_checkpoint_sha256") != _file_sha256(latest)
        or progress.get("epoch_completed") != rank
    ):
        raise ValueError("Latest DAPT-v2 checkpoint/progress binding is invalid")


def validate_args(args: argparse.Namespace) -> str:
    if args.num_workers < 0 or args.smoke_steps < 0:
        raise ValueError("num-workers and smoke-steps must be non-negative")
    if args.preflight_only and args.smoke_steps:
        raise ValueError("preflight-only and smoke-steps are mutually exclusive")
    if args.resume_checkpoint is not None and (args.preflight_only or args.smoke_steps):
        raise ValueError("Only formal DAPT-v2 training may be resumed")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    for name, actual, expected in (
        ("manifest", args.manifest.resolve(), DEFAULT_MANIFEST.resolve()),
        ("tuep_root", args.tuep_root.resolve(), DEFAULT_TUEP_ROOT.resolve()),
        ("labram_root", args.labram_root.resolve(), DEFAULT_LABRAM_ROOT.resolve()),
    ):
        if actual != expected:
            raise ValueError(f"DAPT-v2 {name} path is frozen to {expected}")
    formal = not args.preflight_only and args.smoke_steps == 0
    if formal:
        if (
            args.output_dir.resolve() != DEFAULT_OUTPUT.resolve()
            or args.device != "cuda"
            or args.num_workers != NUM_WORKERS
        ):
            raise ValueError(
                "Formal DAPT-v2 is locked to canonical output, CUDA, and 2 workers"
            )
        mode = "formal_training"
    else:
        if args.output_dir.resolve() == DEFAULT_OUTPUT.resolve():
            raise ValueError(
                "Preflight/smoke must use a non-formal --output-dir"
            )
        mode = "preflight_only" if args.preflight_only else "non_selectable_smoke"
    if args.resume_checkpoint is None:
        if args.output_dir.exists() or args.output_dir.is_symlink():
            raise FileExistsError(f"Refusing to reuse DAPT-v2 output: {args.output_dir}")
    else:
        if args.output_dir.is_symlink():
            raise ValueError("DAPT-v2 resume output cannot be a symlink")
        output = args.output_dir.resolve(strict=True)
        checkpoint = args.resume_checkpoint.resolve(strict=True)
        if checkpoint.parent.parent != output:
            raise ValueError("Resume checkpoint must be under OUTPUT/epoch_checkpoints")
        _require_latest_checkpoint(checkpoint, output)
    return mode


def _build_train_dev_datasets(manifest: object) -> tuple[object, object]:
    """Construct exactly train/dev signal datasets; never qualification signal."""

    train = TUEPDAPTV2WindowDataset(manifest, split="pretext_train")
    dev = TUEPDAPTV2WindowDataset(manifest, split="pretext_dev")
    payload = getattr(manifest, "payload")
    split = payload["pretext_split_contract"]
    expected_train = set(split["train_patient_ids"])
    expected_dev = set(split["dev_patient_ids"])
    qualification = set(split["qualification_patient_ids"])
    if (
        set(train.patient_to_indices) != expected_train
        or set(dev.patient_to_indices) != expected_dev
        or set(train.patient_to_indices) & qualification
        or set(dev.patient_to_indices) & qualification
        or len(qualification) != EXPECTED_QUALIFICATION_PATIENTS
    ):
        raise RuntimeError("DAPT-v2 runner signal split boundary changed")
    return train, dev


def _verify_train_dev_file_inventory(manifest: object) -> dict[str, int]:
    """Verify only train/dev filesystem inventory; never traverse qualification."""

    payload = getattr(manifest, "payload")
    root = Path(getattr(manifest, "source_root")).resolve(strict=True)
    split = payload["pretext_split_contract"]
    train_patients = set(split["train_patient_ids"])
    dev_patients = set(split["dev_patient_ids"])
    allowed_patients = train_patients | dev_patients
    declared_rows = [
        row
        for row in payload["records"]
        if row["pretext_split"] in {"pretext_train", "pretext_dev"}
    ]
    declared_paths = {str(row["relative_edf_path"]) for row in declared_rows}
    current_paths: set[str] = set()
    for storage_directory in ("00_epilepsy", "01_no_epilepsy"):
        base = root / storage_directory
        for patient in allowed_patients:
            patient_root = base / patient
            if not patient_root.is_dir():
                continue
            for candidate in patient_root.rglob("*.edf"):
                resolved = candidate.resolve(strict=True)
                if candidate.is_symlink() or not resolved.is_relative_to(root):
                    raise ValueError("DAPT-v2 train/dev inventory has symlink/path escape")
                current_paths.add(resolved.relative_to(root).as_posix())
    if current_paths != declared_paths:
        raise ValueError("Runtime DAPT-v2 train/dev EDF inventory differs from manifest")
    for row in declared_rows:
        source = (root / str(row["relative_edf_path"])).resolve(strict=True)
        if source.stat().st_size != int(row["file_size_bytes"]):
            raise ValueError("Runtime DAPT-v2 train/dev EDF size changed")
    return {
        "train_patient_count": len(train_patients),
        "dev_patient_count": len(dev_patients),
        "train_dev_record_count": len(declared_rows),
        "qualification_record_count_opened": 0,
    }


def _fixed_dev_masks(batch: Mapping[str, object], device: torch.device) -> torch.Tensor:
    masks: list[torch.Tensor] = []
    for record_uid, grid_index in zip(batch["record_uid"], batch["grid_index"].tolist()):
        digest = hashlib.sha256(
            f"{SEED}\0dev\0{record_uid}\0{grid_index}".encode("ascii")
        ).digest()
        mask_seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
        masks.append(exact_random_mask(1, seed=mask_seed, device=device)[0])
    return torch.stack(masks)


def _train_masks(
    batch: Mapping[str, object], *, epoch: int, device: torch.device
) -> torch.Tensor:
    masks: list[torch.Tensor] = []
    for patient, record_uid, grid_index in zip(
        batch["patient_id"], batch["record_uid"], batch["grid_index"].tolist()
    ):
        digest = hashlib.sha256(
            f"{SEED}\0train\0{epoch}\0{patient}\0{record_uid}\0{grid_index}".encode(
                "ascii"
            )
        ).digest()
        mask_seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
        masks.append(exact_random_mask(1, seed=mask_seed, device=device)[0])
    return torch.stack(masks)


def _preflight(
    *,
    train_dataset: object,
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
        raise RuntimeError("DAPT-v2 preflight window is invalid")
    parity = verify_zero_lora_official_pretraining_parity(
        model,
        official_root=labram_root,
        checkpoint_path=checkpoint_path,
        patches_volts=eeg,
        position_ids=positions,
        bool_masked_pos=mask,
    )
    output = diversity_preserving_masked_neural_code_objective(
        model,
        tokenizer,
        eeg,
        positions.unsqueeze(0),
        mask,
    )
    if (
        float(output.teacher_kl_loss.detach().abs()) > 1e-7
        or float(output.entropy_floor_loss.detach()) != 0.0
        or not torch.allclose(output.loss, output.official_ce_loss, rtol=0.0, atol=1e-7)
    ):
        raise RuntimeError("Zero-LoRA DAPT-v2 objective does not reduce to official CE")
    codes = output.neural_codes
    if torch.unique(codes).numel() < 2:
        raise RuntimeError("DAPT-v2 preflight VQ predictions collapsed")
    channel_std = eeg.reshape(19, -1).std(dim=1)
    if torch.any(channel_std <= 0):
        raise RuntimeError("DAPT-v2 preflight contains a constant channel")
    result = {
        "real_window_shape": list(eeg.shape),
        "real_window_abs_max_volts": float(eeg.abs().max()),
        "real_window_min_channel_std_volts": float(channel_std.min()),
        "model_input_scale_from_volts": LABRAM_DAPT_INPUT_SCALE_FROM_VOLTS,
        "patient_id": str(sample["patient_id"]),
        "record_uid": str(sample["record_uid"]),
        "grid_index": int(sample["grid_index"]),
        "zero_lora_official_pretraining_parity": parity,
        "zero_lora_official_ce": float(output.official_ce_loss.detach()),
        "zero_lora_teacher_kl": float(output.teacher_kl_loss.detach()),
        "zero_lora_entropy_floor_loss": float(output.entropy_floor_loss.detach()),
        "teacher_batch_marginal_entropy": float(
            output.teacher_batch_marginal_entropy
        ),
        "batch_marginal_entropy_floor": float(output.batch_marginal_entropy_floor),
        "vq_codebook_size": tokenizer.codebook_size,
        "vq_code_dim": tokenizer.code_dim,
        "vq_unique_codes": int(torch.unique(codes).numel()),
        "trainable_parameter_count": model.n_trainable_parameters,
        "trainable_parameter_names": list(model.trainable_parameter_names),
        "qualification_signal_split_loaded": False,
    }
    del output
    return result


def _gradient_receipt(model: OfficialLaBraMSourceDAPT) -> dict[str, object]:
    receipt: dict[str, object] = {}
    for block in LABRAM_PEFT_BLOCKS:
        adapter = model._lora(block)
        for factor in ("A", "B"):
            gradient = getattr(adapter, f"lora_{factor}").grad
            receipt[f"blocks.{block}.lora_{factor}"] = {
                "present": gradient is not None,
                "finite": bool(gradient is not None and torch.isfinite(gradient).all()),
                "absolute_sum": (
                    None
                    if gradient is None
                    else float(gradient.detach().abs().sum().cpu())
                ),
            }
    return receipt


def _train_epoch(
    *,
    model: OfficialLaBraMSourceDAPT,
    tokenizer: OfficialFrozenLaBraMVQTokenizer,
    loader: DataLoader,
    sampler: PatientUniformEpochSampler,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    device: torch.device,
    completed_steps: int,
    smoke_steps: int,
) -> tuple[dict[str, float], int, Mapping[str, object] | None]:
    sampler.set_epoch(epoch)
    model.train()
    tokenizer.eval()
    tokenizer._assert_frozen()
    totals: defaultdict[str, float] = defaultdict(float)
    samples = 0
    last_gradient: Mapping[str, object] | None = None
    metric_names = (
        "loss",
        "official_ce_loss",
        "masked_ce_loss",
        "complementary_ce_loss",
        "teacher_kl_loss",
        "masked_teacher_kl_loss",
        "complementary_teacher_kl_loss",
        "entropy_floor_loss",
        "student_batch_marginal_entropy",
        "teacher_batch_marginal_entropy",
        "batch_marginal_entropy_floor",
        "student_batch_marginal_effective_perplexity",
        "teacher_batch_marginal_effective_perplexity",
        "student_batch_marginal_top_probability",
        "teacher_batch_marginal_top_probability",
        "masked_accuracy",
        "complementary_accuracy",
    )
    for batch in loader:
        eeg = batch["eeg"].to(device=device, dtype=torch.float32, non_blocking=True)
        positions = batch["position_ids"].to(device=device, non_blocking=True)
        masks = _train_masks(batch, epoch=epoch, device=device)
        optimizer.zero_grad(set_to_none=True)
        output = diversity_preserving_masked_neural_code_objective(
            model, tokenizer, eeg, positions, masks
        )
        output.loss.backward()
        last_gradient = _gradient_receipt(model)
        if any(not item["present"] or not item["finite"] for item in last_gradient.values()):
            raise RuntimeError("DAPT-v2 LoRA gradient is missing/non-finite")
        if completed_steps == 0:
            for block in LABRAM_PEFT_BLOCKS:
                if last_gradient[f"blocks.{block}.lora_B"]["absolute_sum"] <= 0:
                    raise RuntimeError("Initial DAPT-v2 LoRA-B gradient must be non-zero")
                if last_gradient[f"blocks.{block}.lora_A"]["absolute_sum"] != 0:
                    raise RuntimeError("Initial zero-LoRA DAPT-v2 LoRA-A gradient must be zero")
        elif completed_steps == 1:
            for block in LABRAM_PEFT_BLOCKS:
                for factor in ("A", "B"):
                    if last_gradient[f"blocks.{block}.lora_{factor}"]["absolute_sum"] <= 0:
                        raise RuntimeError("Second DAPT-v2 step must update all LoRA factors")
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            GRADIENT_CLIP,
        )
        optimizer.step()
        batch_size = eeg.shape[0]
        samples += batch_size
        for name in metric_names:
            totals[name] += float(getattr(output, name).detach()) * batch_size
        completed_steps += 1
        if smoke_steps and completed_steps >= smoke_steps:
            break
    if samples < 1:
        raise RuntimeError("DAPT-v2 training epoch produced no samples")
    return (
        {name: totals[name] / samples for name in metric_names},
        completed_steps,
        last_gradient,
    )


def _hard_code_metrics(counts: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(counts, dtype=np.int64)
    if values.shape != (8192,) or (values < 0).any() or values.sum() < 1:
        raise ValueError("Hard predicted-code counts must be non-empty [8192]")
    probabilities = values[values > 0].astype(np.float64) / float(values.sum())
    log_perplexity = -float(
        np.sum(probabilities * np.log(probabilities), dtype=np.float64)
    )
    return {
        "hard_prediction_unique_count": int(np.count_nonzero(values)),
        "hard_prediction_log_perplexity": log_perplexity,
        "hard_prediction_effective_perplexity": math.exp(log_perplexity),
        "hard_prediction_top_fraction": float(values.max()) / float(values.sum()),
    }


def _evaluate_dev(
    *,
    model: OfficialLaBraMSourceDAPT,
    tokenizer: OfficialFrozenLaBraMVQTokenizer,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    tokenizer.eval()
    tokenizer._assert_frozen()
    patient_ce: dict[str, list[float]] = defaultdict(list)
    patient_accuracy: dict[str, list[float]] = defaultdict(list)
    patient_counts: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for batch in loader:
            masks = _fixed_dev_masks(batch, device)
            for index, patient_value in enumerate(batch["patient_id"]):
                patient = str(patient_value)
                eeg = batch["eeg"][index : index + 1].to(
                    device=device, dtype=torch.float32
                )
                positions = batch["position_ids"][index].to(device=device)
                mask = masks[index : index + 1]
                codes = tokenizer(eeg, positions)
                masked_logits, complementary_logits = model.complementary_logits(
                    eeg, positions, mask
                )
                masked_target = codes[mask]
                complementary_target = codes[~mask]
                ce = float(F.cross_entropy(masked_logits, masked_target)) + float(
                    F.cross_entropy(complementary_logits, complementary_target)
                )
                masked_prediction = masked_logits.argmax(dim=-1)
                complementary_prediction = complementary_logits.argmax(dim=-1)
                correct = int((masked_prediction == masked_target).sum()) + int(
                    (complementary_prediction == complementary_target).sum()
                )
                predictions = torch.cat(
                    (masked_prediction, complementary_prediction), dim=0
                )
                if predictions.numel() != 152:
                    raise RuntimeError("DAPT-v2 dev must produce 152 hard codes/window")
                counts = torch.bincount(predictions, minlength=8192).cpu().numpy()
                patient_ce[patient].append(ce)
                patient_accuracy[patient].append(correct / 152.0)
                if patient not in patient_counts:
                    patient_counts[patient] = np.zeros(8192, dtype=np.int64)
                patient_counts[patient] += counts
    patients = tuple(sorted(patient_ce))
    if (
        len(patients) != EXPECTED_DEV_PATIENTS
        or set(patient_ce) != set(patient_accuracy)
        or set(patient_ce) != set(patient_counts)
        or any(len(patient_ce[patient]) != DEV_WINDOWS_PER_PATIENT for patient in patients)
        or any(
            int(patient_counts[patient].sum())
            != DEV_WINDOWS_PER_PATIENT * 152
            for patient in patients
        )
    ):
        raise RuntimeError("DAPT-v2 dev replay is not exactly 24 patients x 16 windows")
    patient_values: dict[str, object] = {}
    for patient in patients:
        patient_values[patient] = {
            "official_ce": float(np.mean(patient_ce[patient], dtype=np.float64)),
            "accuracy": float(
                np.mean(patient_accuracy[patient], dtype=np.float64)
            ),
            **_hard_code_metrics(patient_counts[patient]),
        }
    macro_ce = float(
        np.mean([patient_values[p]["official_ce"] for p in patients], dtype=np.float64)
    )
    macro_accuracy = float(
        np.mean([patient_values[p]["accuracy"] for p in patients], dtype=np.float64)
    )
    macro_log_ppl = float(
        np.mean(
            [patient_values[p]["hard_prediction_log_perplexity"] for p in patients],
            dtype=np.float64,
        )
    )
    aggregate_counts = np.sum(
        np.stack([patient_counts[patient] for patient in patients]),
        axis=0,
        dtype=np.int64,
    )
    return {
        "patient_count": len(patients),
        "windows_per_patient": DEV_WINDOWS_PER_PATIENT,
        "patient_macro_official_ce": macro_ce,
        "patient_macro_accuracy": macro_accuracy,
        "patient_macro_hard_prediction_log_perplexity": macro_log_ppl,
        "patient_macro_hard_prediction_effective_perplexity": math.exp(
            macro_log_ppl
        ),
        "aggregate_hard_prediction": _hard_code_metrics(aggregate_counts),
        "patient_values": patient_values,
    }


def evaluate_epoch_eligibility(
    baseline: Mapping[str, object], candidate: Mapping[str, object]
) -> dict[str, object]:
    keys = (
        "patient_macro_official_ce",
        "patient_macro_accuracy",
        "patient_macro_hard_prediction_log_perplexity",
    )
    if any(
        key not in baseline
        or key not in candidate
        or not math.isfinite(float(baseline[key]))
        or not math.isfinite(float(candidate[key]))
        for key in keys
    ):
        raise ValueError("DAPT-v2 selection metrics are missing/non-finite")
    ce_pass = float(candidate[keys[0]]) < float(baseline[keys[0]])
    accuracy_pass = float(candidate[keys[1]]) >= float(baseline[keys[1]])
    log_ppl_delta = float(candidate[keys[2]]) - float(baseline[keys[2]])
    # Compare against the directly constructed threshold so an exact boundary
    # is not rejected by a second subtractive round-off.
    diversity_threshold = (
        float(baseline[keys[2]]) + HARD_PREDICTION_LOG_PERPLEXITY_MARGIN
    )
    diversity_pass = float(candidate[keys[2]]) >= diversity_threshold
    return {
        "ce_strictly_below_zero": ce_pass,
        "accuracy_not_below_zero": accuracy_pass,
        "hard_prediction_log_perplexity_delta": log_ppl_delta,
        "hard_prediction_log_perplexity_margin": (
            HARD_PREDICTION_LOG_PERPLEXITY_MARGIN
        ),
        "hard_prediction_perplexity_ratio_minimum": (
            HARD_PREDICTION_MINIMUM_PERPLEXITY_RATIO
        ),
        "hard_prediction_diversity_gate": diversity_pass,
        "eligible": bool(ce_pass and accuracy_pass and diversity_pass),
    }


def select_epoch_by_frozen_rule(
    baseline: Mapping[str, object], epoch_rows: Sequence[Mapping[str, object]]
) -> int | str:
    best_epoch: int | str = "zero_lora_baseline"
    best_ce: float | None = None
    for row in epoch_rows:
        epoch = row.get("epoch")
        dev = row.get("pretext_dev")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or not isinstance(dev, Mapping):
            raise ValueError("DAPT-v2 epoch selection row is invalid")
        eligibility = evaluate_epoch_eligibility(baseline, dev)
        if eligibility["eligible"]:
            ce = float(dev["patient_macro_official_ce"])
            if best_ce is None or ce < best_ce:
                best_ce = ce
                best_epoch = epoch
    return best_epoch


def _capture_rng_state(device: torch.device) -> dict[str, object]:
    return {
        "torch_cpu_rng_state": torch.get_rng_state().clone(),
        "torch_cuda_rng_state_all": (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if device.type == "cuda"
            else []
        ),
        "numpy_rng_state": np.random.get_state(),
    }


def _restore_rng_state(payload: Mapping[str, object], device: torch.device) -> None:
    cpu = payload["torch_cpu_rng_state"]
    cuda = payload["torch_cuda_rng_state_all"]
    numpy_state = payload["numpy_rng_state"]
    if not isinstance(cpu, torch.Tensor) or cpu.dtype != torch.uint8:
        raise TypeError("DAPT-v2 resume CPU RNG state is invalid")
    if not isinstance(cuda, list) or any(
        not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 for value in cuda
    ):
        raise TypeError("DAPT-v2 resume CUDA RNG state is invalid")
    if not isinstance(numpy_state, tuple) or len(numpy_state) != 5:
        raise TypeError("DAPT-v2 resume NumPy RNG state is invalid")
    torch.set_rng_state(cpu)
    if device.type == "cuda":
        if len(cuda) != torch.cuda.device_count():
            raise ValueError("DAPT-v2 resume CUDA device count changed")
        torch.cuda.set_rng_state_all(cuda)
    elif cuda:
        raise ValueError("CUDA DAPT-v2 checkpoint cannot resume on CPU")
    np.random.set_state(numpy_state)


_CHECKPOINT_KEYS = {
    "schema_version",
    "manifest_sha256",
    "run_config_sha256",
    "epoch_completed",
    "completed_optimizer_steps",
    "adapter_state",
    "optimizer_state",
    "best_adapter_state",
    "best_epoch",
    "best_eligible_ce",
    "zero_lora_pretext_dev",
    "epoch_history",
    "last_gradient_receipt",
    "torch_cpu_rng_state",
    "torch_cuda_rng_state_all",
    "numpy_rng_state",
}


def _load_resume_checkpoint(
    path: Path, *, manifest_sha256: str, run_config_sha256: str
) -> Mapping[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or set(payload) != _CHECKPOINT_KEYS:
        raise ValueError("DAPT-v2 checkpoint fields changed")
    if payload["schema_version"] != "soz_labram_tuep_dapt_v2_epoch_checkpoint_v1":
        raise ValueError("DAPT-v2 checkpoint schema changed")
    if payload["manifest_sha256"] != manifest_sha256:
        raise ValueError("DAPT-v2 checkpoint manifest hash mismatch")
    if payload["run_config_sha256"] != run_config_sha256:
        raise ValueError("DAPT-v2 checkpoint run-config hash mismatch")
    epoch = payload["epoch_completed"]
    history = payload["epoch_history"]
    steps = payload["completed_optimizer_steps"]
    if (
        isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or not -1 <= epoch < EPOCHS
        or not isinstance(history, list)
        or len(history) != max(epoch + 1, 0)
        or isinstance(steps, bool)
        or not isinstance(steps, int)
        or steps < 0
    ):
        raise ValueError("DAPT-v2 checkpoint epoch/history coordinates are invalid")
    if any(
        not isinstance(row, Mapping) or row.get("epoch") != index
        for index, row in enumerate(history)
    ):
        raise ValueError("DAPT-v2 checkpoint epoch history is not contiguous")
    expected_steps = 0 if epoch == -1 else (epoch + 1) * (
        EXPECTED_TRAIN_PATIENTS * TRAIN_WINDOWS_PER_PATIENT // BATCH_SIZE
    )
    if steps != expected_steps:
        raise ValueError("DAPT-v2 checkpoint optimizer-step coordinate changed")
    for key in ("adapter_state", "optimizer_state", "best_adapter_state"):
        if not isinstance(payload[key], Mapping):
            raise TypeError(f"DAPT-v2 checkpoint {key} must be a mapping")
    baseline = payload["zero_lora_pretext_dev"]
    if not isinstance(baseline, Mapping):
        raise TypeError("DAPT-v2 checkpoint lacks zero-LoRA dev baseline")
    if payload["best_epoch"] != select_epoch_by_frozen_rule(baseline, history):
        raise ValueError("DAPT-v2 checkpoint selected epoch contradicts frozen rule")
    expected_best_ce = None
    if isinstance(payload["best_epoch"], int):
        expected_best_ce = float(history[payload["best_epoch"]]["pretext_dev"]["patient_macro_official_ce"])
    if payload["best_eligible_ce"] != expected_best_ce:
        raise ValueError("DAPT-v2 checkpoint best eligible CE is inconsistent")
    return payload


def _run_config(
    *,
    mode: str,
    args: argparse.Namespace,
    manifest_sha256: str,
    foundation_sha256: str,
    vq_sha256: str,
) -> tuple[dict[str, object], str]:
    config: dict[str, object] = {
        "schema_version": "soz_labram_tuep_dapt_v2_run_config_v1",
        "mode": mode,
        "seed": SEED,
        "manifest_sha256": manifest_sha256,
        "foundation_checkpoint_sha256": foundation_sha256,
        "vq_checkpoint_sha256": vq_sha256,
        "objective": LABRAM_DAPT_V2_OBJECTIVE,
        "teacher_temperature": LABRAM_DAPT_V2_TEACHER_TEMPERATURE,
        "teacher_kl_weight": LABRAM_DAPT_V2_TEACHER_KL_WEIGHT,
        "soft_marginal_entropy_floor_weight": LABRAM_DAPT_V2_ENTROPY_FLOOR_WEIGHT,
        "soft_marginal_minimum_perplexity_ratio": (
            LABRAM_DAPT_V2_MINIMUM_PERPLEXITY_RATIO
        ),
        "lora_blocks": list(LABRAM_PEFT_BLOCKS),
        "lora_rank": LABRAM_PEFT_RANK,
        "lora_alpha": LABRAM_PEFT_ALPHA,
        "trainable_parameter_count": LABRAM_PEFT_TRAINABLE_PARAMETERS,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "train_patient_count": EXPECTED_TRAIN_PATIENTS,
        "train_windows_per_patient": TRAIN_WINDOWS_PER_PATIENT,
        "dev_patient_count": EXPECTED_DEV_PATIENTS,
        "dev_windows_per_patient": DEV_WINDOWS_PER_PATIENT,
        "num_workers": args.num_workers,
        "device": args.device,
        "smoke_steps": args.smoke_steps,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": LEARNING_RATE,
            "betas": list(ADAM_BETAS),
            "eps": ADAM_EPS,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip": GRADIENT_CLIP,
        },
        "selection_rule": {
            "zero_lora_fallback": True,
            "ce": "candidate_strictly_below_zero",
            "accuracy": "candidate_greater_than_or_equal_to_zero",
            "hard_prediction_macro_log_perplexity_delta_minimum": (
                HARD_PREDICTION_LOG_PERPLEXITY_MARGIN
            ),
            "eligible_argmin": "patient_macro_official_ce_then_earlier_epoch",
        },
        "signal_splits_loaded": ["pretext_train", "pretext_dev"],
        "qualification_signal_split_loaded": False,
        "implementation_sha256": {
            "runner": _file_sha256(Path(__file__)),
            "data": _file_sha256(ROOT / "src/soz/data/labram_tuep_dapt_v2.py"),
            "objective": _file_sha256(
                ROOT / "src/soz/models/labram_source_dapt_v2.py"
            ),
            "official_dapt_model": _file_sha256(
                ROOT / "src/soz/models/labram_source_dapt.py"
            ),
            "peft": _file_sha256(ROOT / "src/soz/models/labram_peft.py"),
        },
    }
    digest = hashlib.sha256(_json_bytes(config)).hexdigest()
    return config, digest


def main() -> int:
    started = time.perf_counter()
    args = parse_args()
    mode = validate_args(args)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(SEED)
    device = torch.device(args.device)

    manifest = load_tuep_dapt_v2_manifest(
        args.manifest,
        tuep_root=args.tuep_root,
        # The loader validates the hash-bound manifest structure but must not
        # traverse the held qualification filesystem.  Train/dev inventory is
        # replayed explicitly below.
        verify_file_inventory=False,
    )
    if manifest.sha256 != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Formal DAPT-v2 manifest digest changed")
    for field in (
        "target_values_loaded",
        "diagnostic_directory_labels_used",
        "private_data_loaded",
        "annotation_sidecars_opened",
    ):
        if manifest.payload[field] is not False:
            raise ValueError(f"DAPT-v2 manifest safety flag changed: {field}")
    if manifest.payload["continuous_grid_contract"]["annotation_time_used"] is not False:
        raise ValueError("DAPT-v2 manifest used annotation time")
    train_dev_inventory = _verify_train_dev_file_inventory(manifest)
    train_dataset, dev_dataset = _build_train_dev_datasets(manifest)
    train_sampler = PatientUniformEpochSampler(
        train_dataset,
        windows_per_patient=TRAIN_WINDOWS_PER_PATIENT,
        seed=SEED,
    )
    dev_sampler = PatientUniformEpochSampler(
        dev_dataset,
        windows_per_patient=DEV_WINDOWS_PER_PATIENT,
        seed=SEED + 17,
    )
    dev_sampler.set_epoch(0)
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=BATCH_SIZE,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        multiprocessing_context=("spawn" if args.num_workers > 0 else None),
        drop_last=False,
    )
    dev_loader = DataLoader(
        dev_dataset,
        sampler=dev_sampler,
        batch_size=BATCH_SIZE,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        multiprocessing_context=("spawn" if args.num_workers > 0 else None),
        drop_last=False,
    )
    if len(train_sampler) != 120 * 64 or len(dev_sampler) != 24 * 16:
        raise RuntimeError("DAPT-v2 patient-uniform sampler budgets changed")

    labram_root = args.labram_root.resolve(strict=True)
    checkpoint_path = labram_root / "checkpoints/labram-base.pth"
    tokenizer_path = labram_root / "checkpoints/vqnsp.pth"
    model = OfficialLaBraMSourceDAPT(
        modeling_path=labram_root / "modeling_finetune.py",
        checkpoint_path=checkpoint_path,
    ).to(device=device, dtype=torch.float32)
    tokenizer = OfficialFrozenLaBraMVQTokenizer(
        official_root=labram_root,
        checkpoint_path=tokenizer_path,
        expected_sha256=AUDITED_VQNSP_SHA256,
    ).to(device=device, dtype=torch.float32)
    config, config_sha256 = _run_config(
        mode=mode,
        args=args,
        manifest_sha256=manifest.sha256,
        foundation_sha256=model.checkpoint_sha256,
        vq_sha256=tokenizer.checkpoint_sha256,
    )
    preflight_started = time.perf_counter()
    preflight = _preflight(
        train_dataset=train_dataset,
        model=model,
        tokenizer=tokenizer,
        labram_root=labram_root,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    preflight_seconds = time.perf_counter() - preflight_started

    receipt: dict[str, object] = {
        "schema_version": "soz_labram_tuep_dapt_v2_run_v1",
        "mode": mode,
        "run_config": config,
        "run_config_sha256": config_sha256,
        "manifest_path": str(manifest.path),
        "manifest_sha256": manifest.sha256,
        "objective": LABRAM_DAPT_V2_OBJECTIVE,
        "foundation_checkpoint_sha256": model.checkpoint_sha256,
        "vq_checkpoint_sha256": tokenizer.checkpoint_sha256,
        "vq_code_dim": LABRAM_DAPT_CODE_DIM,
        "preflight": preflight,
        "preflight_wall_time_seconds": preflight_seconds,
        "target_values_loaded": False,
        "diagnostic_directory_labels_used": False,
        "private_data_loaded": False,
        "annotation_sidecars_opened": False,
        "annotation_times_used": False,
        "qualification_signal_split_loaded": False,
        "qualification_patient_signals_seen": 0,
        "train_dev_inventory": train_dev_inventory,
        "soz_promotion": False,
        "candidate_promotable": False,
        "representation_qualified": False,
        "epochs": [],
    }
    if args.resume_checkpoint is None and mode != "formal_training":
        args.output_dir.mkdir(parents=True, exist_ok=False)
        _atomic_new_bytes(args.output_dir / "run_config.json", _json_bytes(config))
    elif args.resume_checkpoint is not None:
        if (args.output_dir / "run_config.json").read_bytes() != _json_bytes(config):
            raise ValueError("DAPT-v2 resume run_config.json changed")

    if args.preflight_only:
        receipt.update(
            {
                "training_started": False,
                "training_completed": False,
                "qualification_pending": True,
                "wall_time_seconds": time.perf_counter() - started,
            }
        )
        _atomic_new_bytes(
            args.output_dir / "run_receipt.json", _json_bytes(receipt)
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=LEARNING_RATE,
        betas=ADAM_BETAS,
        eps=ADAM_EPS,
        weight_decay=WEIGHT_DECAY,
    )
    completed_steps = 0
    start_epoch = 0
    last_gradient: Mapping[str, object] | None = None
    zero_baseline: Mapping[str, object] | None = None
    best_epoch: int | str = "zero_lora_baseline"
    best_eligible_ce: float | None = None
    best_state = model.lora_state_dict()

    if args.resume_checkpoint is not None:
        resumed = _load_resume_checkpoint(
            args.resume_checkpoint.resolve(strict=True),
            manifest_sha256=manifest.sha256,
            run_config_sha256=config_sha256,
        )
        model.load_lora_state_dict(resumed["adapter_state"])
        optimizer.load_state_dict(resumed["optimizer_state"])
        best_state = dict(resumed["best_adapter_state"])
        best_epoch = resumed["best_epoch"]
        best_eligible_ce = resumed["best_eligible_ce"]
        zero_baseline = resumed["zero_lora_pretext_dev"]
        receipt["epochs"] = list(resumed["epoch_history"])
        receipt["zero_lora_pretext_dev"] = zero_baseline
        completed_steps = int(resumed["completed_optimizer_steps"])
        start_epoch = int(resumed["epoch_completed"]) + 1
        last_gradient = resumed["last_gradient_receipt"]
        _restore_rng_state(resumed, device)
        receipt["resumed_from_checkpoint"] = str(
            args.resume_checkpoint.resolve(strict=True)
        )
    elif mode == "formal_training":
        baseline_started = time.perf_counter()
        zero_baseline = _evaluate_dev(
            model=model,
            tokenizer=tokenizer,
            loader=dev_loader,
            device=device,
        )
        receipt["zero_lora_pretext_dev"] = zero_baseline
        receipt["zero_lora_pretext_dev_wall_time_seconds"] = (
            time.perf_counter() - baseline_started
        )
        # Publish the formal directory only after the read-only zero baseline
        # succeeds, then immediately create a resumable initial checkpoint.
        args.output_dir.mkdir(parents=True, exist_ok=False)
        _atomic_new_bytes(args.output_dir / "run_config.json", _json_bytes(config))
        initial_payload = {
            "schema_version": "soz_labram_tuep_dapt_v2_epoch_checkpoint_v1",
            "manifest_sha256": manifest.sha256,
            "run_config_sha256": config_sha256,
            "epoch_completed": -1,
            "completed_optimizer_steps": 0,
            "adapter_state": model.lora_state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_adapter_state": best_state,
            "best_epoch": best_epoch,
            "best_eligible_ce": None,
            "zero_lora_pretext_dev": zero_baseline,
            "epoch_history": [],
            "last_gradient_receipt": None,
            **_capture_rng_state(device),
        }
        initial_path = args.output_dir / "epoch_checkpoints/initial_state.pt"
        initial_sha = _save_torch_new(initial_path, initial_payload)
        _atomic_new_bytes(
            args.output_dir / "initial_progress.json",
            _json_bytes(
                {
                    "schema_version": "soz_labram_tuep_dapt_v2_initial_progress_v1",
                    "manifest_sha256": manifest.sha256,
                    "run_config_sha256": config_sha256,
                    "epoch_completed": -1,
                    "completed_optimizer_steps": 0,
                    "epoch_checkpoint": str(initial_path.resolve()),
                    "epoch_checkpoint_sha256": initial_sha,
                    "qualification_pending": True,
                    "qualification_signal_split_loaded": False,
                    "candidate_promotable": False,
                    "soz_promotion": False,
                }
            ),
        )

    for epoch in range(start_epoch, EPOCHS):
        train_started = time.perf_counter()
        train_metrics, completed_steps, last_gradient = _train_epoch(
            model=model,
            tokenizer=tokenizer,
            loader=train_loader,
            sampler=train_sampler,
            optimizer=optimizer,
            epoch=epoch,
            device=device,
            completed_steps=completed_steps,
            smoke_steps=args.smoke_steps,
        )
        row: dict[str, object] = {
            "epoch": epoch,
            "train": train_metrics,
            "train_wall_time_seconds": time.perf_counter() - train_started,
        }
        if mode == "formal_training":
            if zero_baseline is None:
                raise RuntimeError("Formal DAPT-v2 lacks zero-LoRA baseline")
            dev_started = time.perf_counter()
            dev = _evaluate_dev(
                model=model,
                tokenizer=tokenizer,
                loader=dev_loader,
                device=device,
            )
            eligibility = evaluate_epoch_eligibility(zero_baseline, dev)
            row["pretext_dev"] = dev
            row["selection_eligibility"] = eligibility
            row["pretext_dev_wall_time_seconds"] = time.perf_counter() - dev_started
            if eligibility["eligible"]:
                candidate_ce = float(dev["patient_macro_official_ce"])
                if best_eligible_ce is None or candidate_ce < best_eligible_ce:
                    best_eligible_ce = candidate_ce
                    best_epoch = epoch
                    best_state = model.lora_state_dict()
        receipt["epochs"].append(row)

        if mode == "formal_training":
            if best_epoch != select_epoch_by_frozen_rule(
                zero_baseline, receipt["epochs"]
            ):
                raise RuntimeError("Online DAPT-v2 selection disagrees with frozen replay")
            checkpoint_payload = {
                "schema_version": "soz_labram_tuep_dapt_v2_epoch_checkpoint_v1",
                "manifest_sha256": manifest.sha256,
                "run_config_sha256": config_sha256,
                "epoch_completed": epoch,
                "completed_optimizer_steps": completed_steps,
                "adapter_state": model.lora_state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_adapter_state": best_state,
                "best_epoch": best_epoch,
                "best_eligible_ce": best_eligible_ce,
                "zero_lora_pretext_dev": zero_baseline,
                "epoch_history": list(receipt["epochs"]),
                "last_gradient_receipt": last_gradient,
                **_capture_rng_state(device),
            }
            checkpoint = (
                args.output_dir / "epoch_checkpoints" / f"epoch_{epoch:03d}_state.pt"
            )
            checkpoint_sha = _save_torch_new(checkpoint, checkpoint_payload)
            _atomic_new_bytes(
                args.output_dir / f"epoch_{epoch:03d}_progress.json",
                _json_bytes(
                    {
                        "schema_version": "soz_labram_tuep_dapt_v2_epoch_progress_v1",
                        "manifest_sha256": manifest.sha256,
                        "run_config_sha256": config_sha256,
                        "epoch_completed": epoch,
                        "completed_optimizer_steps": completed_steps,
                        "epoch_checkpoint": str(checkpoint.resolve()),
                        "epoch_checkpoint_sha256": checkpoint_sha,
                        "best_epoch": best_epoch,
                        "best_eligible_ce": best_eligible_ce,
                        "epoch_metrics": row,
                        "qualification_pending": True,
                        "qualification_signal_split_loaded": False,
                        "candidate_promotable": False,
                        "soz_promotion": False,
                    }
                ),
            )
        if args.smoke_steps and completed_steps >= args.smoke_steps:
            break

    receipt["training_started"] = True
    receipt["optimizer_steps"] = completed_steps
    receipt["last_gradient_receipt"] = last_gradient
    if mode == "formal_training":
        receipt["training_completed"] = len(receipt["epochs"]) == EPOCHS
        if receipt["training_completed"] is not True:
            raise RuntimeError("Formal DAPT-v2 did not complete all 10 epochs")
        receipt["best_epoch_by_frozen_eligibility_then_ce"] = best_epoch
        receipt["best_eligible_patient_macro_official_ce"] = best_eligible_ce
        receipt["selection_fallback_to_zero_lora"] = best_epoch == "zero_lora_baseline"
        receipt["eligible_epoch_count"] = sum(
            int(row["selection_eligibility"]["eligible"])
            for row in receipt["epochs"]
        )
        adapter_path = args.output_dir / "selected_lora.pt"
    else:
        receipt["training_completed"] = False
        receipt["smoke_completed"] = bool(
            args.smoke_steps and completed_steps >= args.smoke_steps
        )
        best_state = model.lora_state_dict()
        adapter_path = args.output_dir / "smoke_lora.pt"
    receipt["selected_adapter_path"] = str(adapter_path.resolve())
    receipt["selected_adapter_sha256"] = _save_adapter_new(adapter_path, best_state)
    receipt["qualification_pending"] = True
    receipt["wall_time_seconds"] = time.perf_counter() - started
    if device.type == "cuda":
        receipt["cuda_max_memory_allocated_bytes"] = int(
            torch.cuda.max_memory_allocated(device)
        )
        receipt["cuda_max_memory_reserved_bytes"] = int(
            torch.cuda.max_memory_reserved(device)
        )
    _atomic_idempotent_bytes(
        args.output_dir / "run_receipt.json", _json_bytes(receipt)
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
