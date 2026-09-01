#!/usr/bin/env python3
"""Run the single frozen LaBraM A0/A1/A2 comparison after S1 release.

The entry point fails before loading a feature tensor, model, checkpoint, or
CUDA context unless the development cohort has passed the two-reader plus
third-reader release gate and its label-only target artifact exists.  It never
opens S1 calibration/locked labels, DeepSOZ targets, TUSZ involvement targets,
or private data.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Mapping, Sequence

import torch
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_tusz_eeg_only_s1_label_release import audit_cohort  # noqa: E402
from scripts.materialize_tusz_eeg_only_s1_labram_prefix import (  # noqa: E402
    FULL_SCHEMA as PREFIX_SCHEMA,
    TENSOR_NAME as PREFIX_TENSOR_NAME,
)
from scripts.materialize_tusz_eeg_only_s1_targets import (  # noqa: E402
    SCHEMA_VERSION as TARGET_SCHEMA,
    STATUS as TARGET_STATUS,
)
from src.soz.geometry import N_STANDARD_CHANNELS, STANDARD_19  # noqa: E402
from src.soz.metrics import (  # noqa: E402
    deepsoz_style_top1_metrics,
    patient_localization_metrics,
)
from src.soz.models.labram_peft import (  # noqa: E402
    LABRAM_PEFT_TRAINABLE_PARAMETERS,
    OfficialLaBraMMinimalPEFTSuffix,
)
from src.soz.models.labram_static_suffix import (  # noqa: E402
    OfficialLaBraMStaticAdapterSuffix,
)
from src.soz.s1_labram_recovery import (  # noqa: E402
    S1_PROJECTOR_TRAINABLE_PARAMETERS,
    S1SharedChannelProjector,
    aggregate_complete_patient_bags,
    event_logits_from_prefix,
    fold_channel_prior_logits,
    s1_patient_objective,
    validate_protocol_partitions,
)


DEFAULT_READER_PACK = ROOT / "outputs/tusz_eeg_only_s1_reader_pack_v1_20260813"
DEFAULT_PREFIX = ROOT / "outputs/tusz_eeg_only_s1_labram_prefix_v1_20260813"
DEFAULT_TARGET = ROOT / "outputs/tusz_eeg_only_s1_development_targets_v1_20260813"
DEFAULT_PROTOCOL = ROOT / "configs/tusz_eeg_only_s1_a0_a1_a2_protocol_v1.json"
DEFAULT_MODELING = Path("/mnt/hd1/dyf/workspace/LaBraM/modeling_finetune.py")
DEFAULT_CHECKPOINT = Path("/mnt/hd1/dyf/workspace/LaBraM/checkpoints/labram-base.pth")
DEFAULT_OUTPUT = ROOT / "outputs/tusz_eeg_only_s1_labram_a0_a1_a2_v1_20260813"

SCHEMA_VERSION = "tusz_eeg_only_s1_labram_a0_a1_a2_oof_v1"
ARMS = ("A0", "A1", "A2")
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260813


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _canonical_bytes(value: object) -> bytes:
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


def _require_released_target(
    target_directory: Path,
    reader_pack: Path,
) -> Path:
    """Stop before any model/GPU access when S1 supervision is unavailable."""

    target = target_directory.absolute()
    manifest = target / "manifest.json"
    tensor = target / "targets.safetensors"
    if not manifest.is_file() or not tensor.is_file():
        release = audit_cohort(reader_pack, "s1_development")
        raise RuntimeError(
            "S1-development targets are not released; training is forbidden "
            f"({release.get('valid_completed_patient_count')}/"
            f"{release.get('expected_patient_count')} completed)"
        )
    return target.resolve(strict=True)


@dataclass(frozen=True)
class S1FormalInputs:
    prefix_tokens: torch.Tensor
    event_patient_index: torch.Tensor
    event_ids: tuple[str, ...]
    case_ids: tuple[str, ...]
    patient_ids: tuple[str, ...]
    patient_case_ids: tuple[str, ...]
    targets: torch.Tensor
    target_mask: torch.Tensor
    spread_targets: torch.Tensor
    spread_mask: torch.Tensor
    patient_folds: torch.Tensor
    excluded_patients: tuple[Mapping[str, str], ...]
    protocol: Mapping[str, object]

    def __post_init__(self) -> None:
        patients = len(self.patient_ids)
        events = len(self.event_ids)
        if patients < 3 or len(set(self.patient_ids)) != patients:
            raise ValueError("S1 available patient roster is too small or duplicated")
        if len(self.patient_case_ids) != patients or len(set(self.patient_case_ids)) != patients:
            raise ValueError("S1 patient case roster changed")
        if tuple(self.prefix_tokens.shape) != (events, 15, 77, 200):
            raise ValueError("S1 prefix carrier must have shape [E,15,77,200]")
        if self.prefix_tokens.requires_grad or not torch.isfinite(
            self.prefix_tokens
        ).all():
            raise ValueError("S1 prefix carrier must be detached and finite")
        if self.event_patient_index.dtype != torch.long or tuple(
            self.event_patient_index.shape
        ) != (events,):
            raise TypeError("S1 event routing must be long [E]")
        if len(self.case_ids) != events or len(set(self.event_ids)) != events:
            raise ValueError("S1 event identity carrier changed")
        expected = (patients, N_STANDARD_CHANNELS)
        for name, value in (
            ("targets", self.targets),
            ("target_mask", self.target_mask),
            ("spread_targets", self.spread_targets),
            ("spread_mask", self.spread_mask),
        ):
            if tuple(value.shape) != expected:
                raise ValueError(f"S1 {name} must have shape [P,19]")
        if self.target_mask.dtype != torch.bool or self.spread_mask.dtype != torch.bool:
            raise TypeError("S1 target/spread masks must be bool")
        if tuple(self.patient_folds.shape) != (patients,) or (
            self.patient_folds.dtype != torch.long
        ):
            raise TypeError("S1 patient folds must be long [P]")
        if set(self.patient_folds.tolist()) != {0, 1, 2}:
            raise ValueError("Every S1 OOF fold must contain an available patient")
        aggregate_complete_patient_bags(
            torch.zeros(events, N_STANDARD_CHANNELS),
            self.event_patient_index,
            patients,
        )
        if not (((self.targets == 1) & self.target_mask).any(dim=1)).all():
            raise ValueError("Every released S1 patient needs an observed positive")
        if bool(
            (((self.targets == 1) & self.target_mask) & ((self.spread_targets == 1) & self.spread_mask)).any()
        ):
            raise ValueError("S1 SOZ candidate and spread carriers overlap")


def _load_formal_inputs(
    *,
    reader_pack: Path,
    prefix_directory: Path,
    target_directory: Path,
    protocol_path: Path,
) -> S1FormalInputs:
    target_root = _require_released_target(target_directory, reader_pack)
    protocol = _read_json(protocol_path.resolve(strict=True))
    if protocol.get("schema_version") != "tusz_eeg_only_s1_a0_a1_a2_protocol_v1" or (
        protocol.get("status") != "frozen_before_s1_label_release"
    ):
        raise ValueError("S1 A0/A1/A2 protocol is not the frozen version")
    hierarchy = protocol.get("hierarchy")
    if not isinstance(hierarchy, Mapping):
        raise TypeError("S1 protocol lacks hierarchy")
    validate_protocol_partitions(
        hierarchy.get("region_partition", {}),
        hierarchy.get("laterality_partition", {}),
    )

    target_manifest = _read_json(target_root / "manifest.json")
    if target_manifest.get("schema_version") != TARGET_SCHEMA or (
        target_manifest.get("status") != TARGET_STATUS
    ) or target_manifest.get("cohort") != "s1_development":
        raise ValueError("S1 target artifact is not released development supervision")
    if tuple(target_manifest.get("candidate_channels", ())) != STANDARD_19:
        raise ValueError("S1 target channel order changed")
    access = target_manifest.get("access_receipt")
    if not isinstance(access, Mapping) or any(
        access.get(name) is not False
        for name in (
            "raw_eeg_loaded",
            "labram_features_loaded",
            "model_predictions_loaded",
            "deepsoz_targets_loaded",
            "tusz_involvement_targets_loaded",
            "private_eeg_loaded",
            "private_targets_loaded",
            "training_performed",
        )
    ):
        raise ValueError("S1 target release access firewall changed")
    target_tensors = load_file(str(target_root / "targets.safetensors"), device="cpu")
    if set(target_tensors) != {
        "targets",
        "target_mask",
        "spread_targets",
        "spread_mask",
    }:
        raise ValueError("S1 target tensor vocabulary changed")
    target_patients = target_manifest.get("patients")
    if not isinstance(target_patients, list) or not target_patients:
        raise ValueError("S1 released target patient roster is empty")
    patient_ids = tuple(str(row["patient_id"]) for row in target_patients)
    patient_cases = tuple(str(row["case_id"]) for row in target_patients)
    if len(set(patient_ids)) != len(patient_ids) or len(set(patient_cases)) != len(patient_cases):
        raise ValueError("S1 target patient identities are duplicated")

    prefix_root = prefix_directory.resolve(strict=True)
    prefix_manifest = _read_json(prefix_root / "manifest.json")
    if prefix_manifest.get("schema_version") != PREFIX_SCHEMA or (
        prefix_manifest.get("status") != "target_blind_frozen_labram_block9_s1_prefix_ready"
    ) or prefix_manifest.get("full_scope") is not True:
        raise ValueError("S1 prefix is not the full target-blind cache")
    prefix_access = prefix_manifest.get("access_receipt")
    if not isinstance(prefix_access, Mapping) or any(
        prefix_access.get(name) is not False
        for name in (
            "completed_s1_labels_loaded",
            "deepsoz_targets_loaded",
            "tusz_channel_time_targets_loaded",
            "private_eeg_loaded",
            "private_targets_loaded",
            "training_performed",
        )
    ):
        raise ValueError("S1 prefix access firewall changed")
    all_prefix = load_file(
        str(prefix_root / str(prefix_manifest["tensor_file"])), device="cpu"
    ).get(PREFIX_TENSOR_NAME)
    events = prefix_manifest.get("events")
    if not isinstance(events, list) or all_prefix is None or (
        all_prefix.shape[0] != len(events)
    ):
        raise ValueError("S1 prefix event roster/tensor changed")

    patient_lookup = {patient: index for index, patient in enumerate(patient_ids)}
    selected_indices: list[int] = []
    event_patient: list[int] = []
    event_ids: list[str] = []
    case_ids: list[str] = []
    event_counts = [0] * len(patient_ids)
    for index, row in enumerate(events):
        if row.get("cohort") != "s1_development":
            continue
        patient_id = str(row.get("patient_id"))
        if patient_id not in patient_lookup:
            continue
        patient_index = patient_lookup[patient_id]
        case_id = str(row.get("case_id"))
        if case_id != patient_cases[patient_index]:
            raise ValueError("S1 prefix and target case identity disagree")
        selected_indices.append(index)
        event_patient.append(patient_index)
        event_ids.append(str(row.get("event_id")))
        case_ids.append(case_id)
        event_counts[patient_index] += 1
    declared_counts = [int(row["available_event_count"]) for row in target_patients]
    if event_counts != declared_counts:
        raise ValueError("S1 complete patient bags do not replay target release counts")
    selected_tensor = all_prefix.index_select(
        0, torch.tensor(selected_indices, dtype=torch.long)
    ).float().contiguous()

    raw_folds = protocol.get("folds")
    if not isinstance(raw_folds, Mapping) or set(raw_folds) != {"fold0", "fold1", "fold2"}:
        raise ValueError("S1 protocol folds changed")
    fold_by_case: dict[str, int] = {}
    for fold in range(3):
        values = raw_folds[f"fold{fold}"]
        if not isinstance(values, list):
            raise TypeError("S1 protocol fold roster must be an array")
        for case_id in values:
            if str(case_id) in fold_by_case:
                raise ValueError("S1 protocol case appears in multiple folds")
            fold_by_case[str(case_id)] = fold
    if len(fold_by_case) != 36:
        raise ValueError("S1 protocol must freeze all 36 development cases")
    if any(case not in fold_by_case for case in patient_cases):
        raise ValueError("A released S1 target is absent from the frozen folds")
    patient_folds = torch.tensor(
        [fold_by_case[case] for case in patient_cases], dtype=torch.long
    )
    excluded = target_manifest.get("excluded_patients", [])
    if not isinstance(excluded, list):
        raise TypeError("S1 excluded patient roster must be an array")
    return S1FormalInputs(
        prefix_tokens=selected_tensor,
        event_patient_index=torch.tensor(event_patient, dtype=torch.long),
        event_ids=tuple(event_ids),
        case_ids=tuple(case_ids),
        patient_ids=patient_ids,
        patient_case_ids=patient_cases,
        targets=target_tensors["targets"].float().contiguous(),
        target_mask=target_tensors["target_mask"].bool().contiguous(),
        spread_targets=target_tensors["spread_targets"].float().contiguous(),
        spread_mask=target_tensors["spread_mask"].bool().contiguous(),
        patient_folds=patient_folds,
        excluded_patients=tuple(dict(row) for row in excluded),
        protocol=protocol,
    )


@dataclass(frozen=True)
class _PatientSubset:
    prefix: torch.Tensor
    event_patient_index: torch.Tensor
    targets: torch.Tensor
    target_mask: torch.Tensor
    global_patient_indices: torch.Tensor


def _subset(inputs: S1FormalInputs, patient_indices: torch.Tensor) -> _PatientSubset:
    if patient_indices.dtype != torch.long or patient_indices.ndim != 1 or (
        patient_indices.numel() < 1
    ):
        raise TypeError("S1 patient subset indices must be non-empty long [P]")
    lookup = torch.full((len(inputs.patient_ids),), -1, dtype=torch.long)
    lookup[patient_indices] = torch.arange(patient_indices.numel(), dtype=torch.long)
    remapped = lookup[inputs.event_patient_index]
    event_indices = torch.nonzero(remapped >= 0, as_tuple=False).flatten()
    return _PatientSubset(
        prefix=inputs.prefix_tokens.index_select(0, event_indices),
        event_patient_index=remapped.index_select(0, event_indices),
        targets=inputs.targets.index_select(0, patient_indices),
        target_mask=inputs.target_mask.index_select(0, patient_indices),
        global_patient_indices=patient_indices,
    )


def _seeded_projector(seed: int, device: torch.device) -> S1SharedChannelProjector:
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        return S1SharedChannelProjector().to(device)


def _suffix(
    arm: str,
    *,
    modeling: Path,
    checkpoint: Path,
    seed: int,
    device: torch.device,
) -> torch.nn.Module:
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if arm == "A1":
            result: torch.nn.Module = OfficialLaBraMMinimalPEFTSuffix(
                modeling_path=modeling,
                checkpoint_path=checkpoint,
            )
        else:
            result = OfficialLaBraMStaticAdapterSuffix(
                modeling_path=modeling,
                checkpoint_path=checkpoint,
            )
    return result.to(device)


def _chunks(count: int, size: int) -> tuple[torch.Tensor, ...]:
    return tuple(
        torch.arange(start, min(start + size, count), dtype=torch.long)
        for start in range(0, count, size)
    )


def _collect_event_logits(
    subset: _PatientSubset,
    suffix: torch.nn.Module,
    projector: S1SharedChannelProjector,
    prior: torch.Tensor,
    *,
    microbatch: int,
    device: torch.device,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    with torch.no_grad():
        for indices in _chunks(subset.prefix.shape[0], microbatch):
            rows.append(
                event_logits_from_prefix(
                    suffix,
                    projector,
                    subset.prefix.index_select(0, indices).to(device),
                    prior.to(device),
                ).cpu()
            )
    return torch.cat(rows, dim=0)


def _fit(
    subset: _PatientSubset,
    *,
    arm: str,
    modeling: Path,
    checkpoint: Path,
    config: Mapping[str, object],
    seed: int,
    device: torch.device,
) -> tuple[torch.nn.Module, S1SharedChannelProjector, torch.Tensor, dict[str, object]]:
    suffix = _suffix(
        arm,
        modeling=modeling,
        checkpoint=checkpoint,
        seed=seed + 1,
        device=device,
    )
    projector = _seeded_projector(seed, device)
    prior = fold_channel_prior_logits(subset.targets, subset.target_mask).cpu()
    head_lr = float(config["head_learning_rate"])
    lora_lr = float(config["lora_learning_rate"])
    weight_decay = float(config["weight_decay"])
    epochs = int(config["epochs"])
    maximum_gradient_norm = float(config["maximum_gradient_norm"])
    microbatch = int(config["event_microbatch"])
    groups: list[dict[str, object]] = [
        {
            "params": tuple(projector.parameters()),
            "lr": head_lr,
            "weight_decay": weight_decay,
        }
    ]
    if arm == "A1":
        groups.append(
            {
                "params": tuple(
                    value for value in suffix.parameters() if value.requires_grad
                ),
                "lr": lora_lr,
                "weight_decay": weight_decay,
            }
        )
    trainable = sum(
        parameter.numel()
        for group in groups
        for parameter in group["params"]  # type: ignore[index]
    )
    expected = S1_PROJECTOR_TRAINABLE_PARAMETERS + (
        LABRAM_PEFT_TRAINABLE_PARAMETERS if arm == "A1" else 0
    )
    if trainable != expected:
        raise RuntimeError("S1 trainable parameter contract changed")
    optimizer = torch.optim.AdamW(groups)
    curve: list[dict[str, float]] = []
    maximum_replay_error = 0.0
    patients = subset.targets.shape[0]
    counts = torch.bincount(subset.event_patient_index, minlength=patients)
    for epoch in range(epochs):
        projector.train()
        suffix.train()
        first = _collect_event_logits(
            subset,
            suffix,
            projector,
            prior,
            microbatch=microbatch,
            device=device,
        )
        patient_leaf = aggregate_complete_patient_bags(
            first, subset.event_patient_index, patients
        ).detach().requires_grad_(True)
        objective = s1_patient_objective(
            patient_leaf,
            subset.targets,
            subset.target_mask,
            arm=arm,
        )
        patient_gradient = torch.autograd.grad(
            objective.total, patient_leaf, create_graph=False
        )[0]
        event_upstream = patient_gradient.index_select(
            0, subset.event_patient_index
        ) / counts.index_select(0, subset.event_patient_index).float().unsqueeze(1)
        optimizer.zero_grad(set_to_none=True)
        replay_rows: list[torch.Tensor] = []
        for indices in _chunks(subset.prefix.shape[0], microbatch):
            logits = event_logits_from_prefix(
                suffix,
                projector,
                subset.prefix.index_select(0, indices).to(device),
                prior.to(device),
            )
            replay_rows.append(logits.detach().cpu())
            logits.backward(event_upstream.index_select(0, indices).to(device))
        replay = torch.cat(replay_rows, dim=0)
        replay_error = float((first - replay).abs().max())
        maximum_replay_error = max(maximum_replay_error, replay_error)
        if replay_error > 1e-6:
            raise RuntimeError("S1 two-pass patient objective replay became stochastic")
        parameters = [
            parameter
            for group in groups
            for parameter in group["params"]  # type: ignore[index]
        ]
        norm = torch.nn.utils.clip_grad_norm_(parameters, maximum_gradient_norm)
        if not torch.isfinite(norm):
            raise RuntimeError("S1 gradient norm is non-finite")
        if any(
            parameter.grad is not None
            for parameter in suffix.parameters()
            if not parameter.requires_grad
        ):
            raise RuntimeError("A frozen LaBraM parameter received a gradient")
        optimizer.step()
        curve.append(
            {
                "epoch": float(epoch + 1),
                "total": float(objective.total.detach()),
                "exact_set_mass": float(objective.exact_set_mass.detach()),
                "hierarchy_set_mass": float(objective.hierarchy_set_mass.detach()),
                "gradient_norm_before_clip": float(norm.detach()),
                "two_pass_replay_max_abs_error": replay_error,
            }
        )
    optimizer.zero_grad(set_to_none=True)
    projector.eval()
    suffix.eval()
    return suffix, projector, prior, {
        "arm": arm,
        "epochs": epochs,
        "seed": seed,
        "trainable_parameter_count": trainable,
        "foundation_trainable_parameter_count": (
            LABRAM_PEFT_TRAINABLE_PARAMETERS if arm == "A1" else 0
        ),
        "shared_projector_trainable_parameter_count": S1_PROJECTOR_TRAINABLE_PARAMETERS,
        "complete_patient_bag_before_loss": True,
        "patient_equal_loss": True,
        "maximum_two_pass_replay_error": maximum_replay_error,
        "curve": curve,
    }


def _metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    spread: torch.Tensor,
    spread_mask: torch.Tensor,
) -> dict[str, object]:
    top1 = deepsoz_style_top1_metrics(
        logits,
        targets,
        mask,
        spread_targets=spread,
        spread_mask=spread_mask,
    )
    ranking = patient_localization_metrics(
        logits, targets, mask, k_values=(1, 3)
    )
    return {
        "top1": asdict(top1),
        "ranking": asdict(ranking),
        "far_error_expected_count": top1.n_samples * (1.0 - top1.relaxed_accuracy),
    }


def _patient_deltas(
    candidate: torch.Tensor,
    baseline: torch.Tensor,
    inputs: S1FormalInputs,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for patient in range(len(inputs.patient_ids)):
        index = slice(patient, patient + 1)
        candidate_metrics = _metrics(
            candidate[index],
            inputs.targets[index],
            inputs.target_mask[index],
            inputs.spread_targets[index],
            inputs.spread_mask[index],
        )
        baseline_metrics = _metrics(
            baseline[index],
            inputs.targets[index],
            inputs.target_mask[index],
            inputs.spread_targets[index],
            inputs.spread_mask[index],
        )
        rows.append(
            torch.tensor(
                [
                    candidate_metrics["top1"]["strict_accuracy"]
                    - baseline_metrics["top1"]["strict_accuracy"],
                    candidate_metrics["ranking"]["macro_average_precision"]
                    - baseline_metrics["ranking"]["macro_average_precision"],
                    candidate_metrics["far_error_expected_count"]
                    - baseline_metrics["far_error_expected_count"],
                ],
                dtype=torch.float64,
            )
        )
    return torch.stack(rows)


def _paired_bootstrap(
    candidate: torch.Tensor,
    baseline: torch.Tensor,
    inputs: S1FormalInputs,
) -> dict[str, object]:
    rows = _patient_deltas(candidate, baseline, inputs)
    generator = torch.Generator().manual_seed(BOOTSTRAP_SEED)
    indices = torch.randint(
        0,
        rows.shape[0],
        (BOOTSTRAP_REPLICATES, rows.shape[0]),
        generator=generator,
    )
    samples = rows[indices].mean(dim=1)
    names = ("strict_top1", "macro_average_precision", "far_error_expected_count")
    return {
        name: {
            "delta": float(rows[:, column].mean()),
            "ci95": [
                float(torch.quantile(samples[:, column], 0.025)),
                float(torch.quantile(samples[:, column], 0.975)),
            ],
            "improved_patient_count": int((rows[:, column] > 0).sum()),
            "worsened_patient_count": int((rows[:, column] < 0).sum()),
            "equal_patient_count": int((rows[:, column] == 0).sum()),
        }
        for column, name in enumerate(names)
    }


def _select_arm(
    metrics: Mapping[str, Mapping[str, object]],
    bootstraps: Mapping[str, Mapping[str, object]],
) -> tuple[str, str]:
    supported = [
        arm
        for arm in ("A1", "A2")
        if float(bootstraps[arm]["strict_top1"]["ci95"][0]) > 0.0  # type: ignore[index]
    ]
    if not supported:
        return "A0", "no_candidate_has_strict_top1_bootstrap_ci95_lower_above_zero"

    def key(arm: str) -> tuple[float, float, float, int]:
        value = metrics[arm]
        strict = float(value["top1"]["strict_accuracy"])  # type: ignore[index]
        ap = float(value["ranking"]["macro_average_precision"])  # type: ignore[index]
        far = float(value["far_error_expected_count"])
        parameters = S1_PROJECTOR_TRAINABLE_PARAMETERS + (
            LABRAM_PEFT_TRAINABLE_PARAMETERS if arm == "A1" else 0
        )
        return (-strict, -ap, far, parameters)

    selected = min(supported, key=key)
    return selected, "strict_supported_then_strict_AP_far_error_parameter_order"


def _state(
    prefix: str,
    suffix: torch.nn.Module,
    projector: S1SharedChannelProjector,
    prior: torch.Tensor,
    *,
    include_lora: bool,
) -> dict[str, torch.Tensor]:
    state = {
        f"{prefix}.projector.{name}": value.detach().cpu().contiguous()
        for name, value in projector.state_dict().items()
    }
    state[f"{prefix}.channel_prior_logits"] = prior.detach().cpu().contiguous()
    if include_lora:
        state_loader = getattr(suffix, "lora_state_dict", None)
        if not callable(state_loader):
            raise TypeError("A1 suffix must expose lora_state_dict")
        for name, value in state_loader().items():
            state[f"{prefix}.lora.{name}"] = value.contiguous()
    return state


def run(
    *,
    reader_pack: Path,
    prefix_directory: Path,
    target_directory: Path,
    protocol_path: Path,
    modeling: Path,
    checkpoint: Path,
    output_directory: Path,
    device: torch.device,
) -> tuple[Path, Mapping[str, object]]:
    inputs = _load_formal_inputs(
        reader_pack=reader_pack,
        prefix_directory=prefix_directory,
        target_directory=target_directory,
        protocol_path=protocol_path,
    )
    # Only now, after target release and all joins pass, may assets/CUDA open.
    model_path = modeling.resolve(strict=True)
    checkpoint_path = checkpoint.resolve(strict=True)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    optimization = inputs.protocol.get("optimization")
    if not isinstance(optimization, Mapping):
        raise TypeError("S1 protocol lacks optimization settings")
    base_seed = int(optimization["seed"])
    oof = {arm: torch.full((len(inputs.patient_ids), 19), torch.nan) for arm in ARMS}
    state: dict[str, torch.Tensor] = {}
    folds: list[dict[str, object]] = []
    for fold in range(3):
        train_indices = torch.nonzero(inputs.patient_folds != fold, as_tuple=False).flatten()
        held_indices = torch.nonzero(inputs.patient_folds == fold, as_tuple=False).flatten()
        train = _subset(inputs, train_indices)
        held = _subset(inputs, held_indices)
        fold_record: dict[str, object] = {
            "fold": fold,
            "train_patient_count": int(train_indices.numel()),
            "held_patient_count": int(held_indices.numel()),
            "train_event_count": int(train.prefix.shape[0]),
            "held_event_count": int(held.prefix.shape[0]),
            "arms": {},
        }
        for arm_index, arm in enumerate(ARMS):
            suffix, projector, prior, fit = _fit(
                train,
                arm=arm,
                modeling=model_path,
                checkpoint=checkpoint_path,
                config=optimization,
                seed=base_seed + 100 * fold,
                device=device,
            )
            held_events = _collect_event_logits(
                held,
                suffix,
                projector,
                prior,
                microbatch=int(optimization["event_microbatch"]),
                device=device,
            )
            held_patients = aggregate_complete_patient_bags(
                held_events,
                held.event_patient_index,
                held.targets.shape[0],
            )
            oof[arm].index_copy_(0, held_indices, held_patients)
            state.update(
                _state(
                    f"fold{fold}.{arm}",
                    suffix,
                    projector,
                    prior,
                    include_lora=arm == "A1",
                )
            )
            fold_record["arms"][arm] = fit  # type: ignore[index]
            del suffix, projector
            if device.type == "cuda":
                torch.cuda.empty_cache()
        folds.append(fold_record)
    if any(not torch.isfinite(value).all() for value in oof.values()):
        raise RuntimeError("S1 OOF prediction carrier is incomplete")
    metrics = {
        arm: _metrics(
            logits,
            inputs.targets,
            inputs.target_mask,
            inputs.spread_targets,
            inputs.spread_mask,
        )
        for arm, logits in oof.items()
    }
    bootstraps = {
        arm: _paired_bootstrap(oof[arm], oof["A0"], inputs)
        for arm in ("A1", "A2")
    }
    selected, selection_reason = _select_arm(metrics, bootstraps)

    all_indices = torch.arange(len(inputs.patient_ids), dtype=torch.long)
    full = _subset(inputs, all_indices)
    suffix, projector, prior, full_fit = _fit(
        full,
        arm=selected,
        modeling=model_path,
        checkpoint=checkpoint_path,
        config=optimization,
        seed=base_seed,
        device=device,
    )
    state.update(
        _state(
            f"selected_refit.{selected}",
            suffix,
            projector,
            prior,
            include_lora=selected == "A1",
        )
    )
    state.update({f"oof.{arm}": value for arm, value in oof.items()})
    state.update(
        {
            "targets": inputs.targets,
            "target_mask": inputs.target_mask,
            "spread_targets": inputs.spread_targets,
            "spread_mask": inputs.spread_mask,
            "patient_folds": inputs.patient_folds,
            "event_patient_index": inputs.event_patient_index,
        }
    )

    target = output_directory.absolute()
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    published = False
    try:
        save_file(state, str(staging / "results.safetensors"))
        manifest: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "status": "completed_s1_development_oof_and_selected_refit",
            "backbone": "official_pretrained_LaBraM_Base_not_replaced",
            "patient_ids": list(inputs.patient_ids),
            "patient_case_ids": list(inputs.patient_case_ids),
            "event_ids": list(inputs.event_ids),
            "available_supervised_patient_count": len(inputs.patient_ids),
            "excluded_indeterminate_or_unavailable_patients": list(inputs.excluded_patients),
            "folds": folds,
            "metrics": metrics,
            "paired_patient_bootstrap_vs_A0": bootstraps,
            "selected_arm": selected,
            "selection_reason": selection_reason,
            "selected_refit": full_fit,
            "primary_endpoint": "patient_level_strict_top1_membership",
            "official_neighborhood_role": "sensitivity_secondary_not_training_target",
            "results_tensor_file": "results.safetensors",
            "access_receipt": {
                "s1_development_targets_loaded": True,
                "s1_calibration_labels_loaded": False,
                "s1_locked_labels_loaded": False,
                "deepsoz_targets_loaded": False,
                "tusz_involvement_targets_loaded": False,
                "private_eeg_loaded": False,
                "private_targets_loaded": False,
                "training_performed": True,
                "foundation_trained_from_scratch": False,
            },
        }
        (staging / "manifest.json").write_bytes(_canonical_bytes(manifest))
        os.replace(staging, target)
        published = True
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)
    return target, manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--reader-pack", type=Path, default=DEFAULT_READER_PACK)
    parser.add_argument("--prefix-directory", type=Path, default=DEFAULT_PREFIX)
    parser.add_argument("--target-directory", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--modeling-path", type=Path, default=DEFAULT_MODELING)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    path, manifest = run(
        reader_pack=args.reader_pack,
        prefix_directory=args.prefix_directory,
        target_directory=args.target_directory,
        protocol_path=args.protocol,
        modeling=args.modeling_path,
        checkpoint=args.checkpoint_path,
        output_directory=args.output_directory,
        device=torch.device(args.device),
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "selected_arm": manifest["selected_arm"],
                "available_supervised_patient_count": manifest[
                    "available_supervised_patient_count"
                ],
                "output": str(path),
                "private_loaded": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
