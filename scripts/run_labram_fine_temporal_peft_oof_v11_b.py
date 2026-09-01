#!/usr/bin/env python3
"""Run the single fixed fold-local LaBraM PEFT capacity diagnostic v11-B."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time
from typing import Mapping, Sequence

import safetensors
from safetensors.torch import load_file, save_file
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_labram_fine_temporal_nested_oof_v11 import (  # noqa: E402
    DEFAULT_FINE,
    DEFAULT_PREFIX,
    DEFAULT_SOURCE,
    DEFAULT_SPLIT,
    DEFAULT_TARGET,
    DEFAULT_UNION,
    EXPECTED_FINE_MANIFEST_SHA256,
    EXPECTED_FINE_TENSOR_FILE_SHA256,
    EXPECTED_PREFIX_MANIFEST_SHA256,
    EXPECTED_PREFIX_TENSOR_FILE_SHA256,
    EXPECTED_SOURCE_SHA256,
    EXPECTED_SPLIT_SHA256,
    EXPECTED_TARGET_ARTIFACT_SHA256,
    EXPECTED_TARGET_README_SHA256,
    EXPECTED_TARGET_RECEIPT_SHA256,
    EXPECTED_TARGET_SUMMARY_SHA256,
    OUTER_FOLDS,
    _canonical_bytes,
    _file_sha,
    _fit_reasoner,
    _load_json_manifest,
    _require_target_free_cache,
    _state_sha,
    _transform_state,
)
from scripts.run_labram_fine_temporal_nested_oof_v11_1 import (  # noqa: E402
    _absolute_bootstrap,
    _complete_candidate_label_rows,
    _evaluate,
    _paired_bootstrap,
)
from src.soz.data.deepsoz_target_v2 import (  # noqa: E402
    TARGET_V2_POLICY_SHA256,
    load_verified_deepsoz_target_v2_artifact,
)
from src.soz.fine_temporal_evidence import FINE_TEMPORAL_FEATURE_NAMES  # noqa: E402
from src.soz.labram_peft_recovery import suffix_node_tokens  # noqa: E402
from src.soz.models.labram import (  # noqa: E402
    AUDITED_LABRAM_BASE_SHA256,
    AUDITED_LABRAM_MODELING_SHA256,
)
from src.soz.models.labram_peft import (  # noqa: E402
    LABRAM_PEFT_BLOCKS,
    LABRAM_PEFT_TRAINABLE_PARAMETERS,
    OfficialLaBraMMinimalPEFTSuffix,
)
from src.soz.v11_development_union import (  # noqa: E402
    EXPECTED_PUBLIC_DEVELOPMENT_UNION_MANIFEST_SHA256,
    load_public_development_union,
)
from src.soz.v11_reasoner import (  # noqa: E402
    FoldFeatureTransform,
    SharedPositiveSetReasoner,
    V11_CANDIDATE_MASK,
    fit_fold_transform,
    positive_set_mass_loss,
    robust_pool_complete_patient_bags,
)
from src.soz.v11b_peft import (  # noqa: E402
    apply_fold_transform_differentiable,
    clone_reasoner_pair,
    differentiable_pool_complete_patient_bags,
    differentiable_suffix_phase_contrasts,
    patient_loss_and_h_upstream,
)


PROTOCOL_PATH = (
    ROOT
    / "research/02_method/labram_fold_local_peft_capacity_protocol_v11_b_20260811_zh.md"
)
EXPECTED_PROTOCOL_SHA256 = (
    "6bd0d80a9b64c6134e71d120cdff0b3fd4924f1ab470091798660087e768c1d7"
)
DEFAULT_OUTPUT = ROOT / "outputs/labram_fine_temporal_peft_oof_v11_b_20260811_r3"
DEFAULT_MODELING = Path("/mnt/hd1/dyf/workspace/LaBraM/modeling_finetune.py")
DEFAULT_CHECKPOINT = Path(
    "/mnt/hd1/dyf/workspace/LaBraM/checkpoints/labram-base.pth"
)
V11_1_REFERENCE = (
    ROOT / "outputs/labram_fine_temporal_nested_oof_v11_1_20260811_r2"
)
EXPECTED_V11_1_MANIFEST_SHA256 = (
    "f399678e5756ae30cbe5f9f87d9d8bb5b220b16015e1b2a0417110f20e70195c"
)
EXPECTED_V11_1_OOF_SHA256 = (
    "6443680b18b53b0c552b9634e7c9e2547284c9d08cccd5cd99c35b9e1a27ac08"
)
SCHEMA = "soz_labram_fine_temporal_peft_oof_v11_b_r3"
MATCHED_FROZEN = "matched_frozen_final_suffix"
PEFT = "peft_qkv_blocks10_11_r4"
BASE_SEED = 20260811
FORMAL_EPOCHS = 20
HEAD_LR = 3e-3
LORA_LR = 1e-4
WEIGHT_DECAY = 1e-2
MAX_GRAD_NORM = 1.0
EVENT_MICROBATCH = 4
WARM_START_L2 = 0.20
NONINFERIORITY_MARGIN = 0.05
PRIMARY_PATIENT_COUNT = 101
PRIMARY_EVENT_COUNT = 984
EXCLUDED_PATIENT = "258"
EXCLUDED_PARTIAL_REFERENCE_PATIENT = EXCLUDED_PATIENT


def _fold_seed_manifest() -> dict[str, int]:
    return {str(fold): BASE_SEED + fold for fold in OUTER_FOLDS}


def _masked_oof_scores_for_publish(logits: torch.Tensor) -> torch.Tensor:
    """Return finite, argmax-safe scores with PZ at the dtype minimum."""

    if logits.ndim != 2 or logits.shape[1] != 19:
        raise ValueError("v11-B OOF logits must have shape [P,19]")
    if not logits.is_floating_point() or not torch.isfinite(logits).all():
        raise ValueError("v11-B OOF logits must be finite floating-point values")
    candidate_mask = V11_CANDIDATE_MASK.to(device=logits.device)
    sentinel = torch.finfo(logits.dtype).min
    return logits.detach().masked_fill(~candidate_mask, sentinel).contiguous()


def _chunks(count: int, size: int) -> tuple[torch.Tensor, ...]:
    if count < 1 or size < 1:
        raise ValueError("chunk count and size must be positive")
    return tuple(
        torch.arange(start, min(start + size, count), dtype=torch.long)
        for start in range(0, count, size)
    )


@dataclass(frozen=True)
class V11BInputs:
    prefix: torch.Tensor
    fine_event: torch.Tensor
    reliability: torch.Tensor
    event_patient_index: torch.Tensor
    fine_patient: torch.Tensor
    patient_ids: tuple[str, ...]
    targets: torch.Tensor
    target_mask: torch.Tensor
    patient_folds: torch.Tensor
    event_counts: torch.Tensor
    formal_scope: bool
    target_free_union_manifest_sha256: str
    target_receipt_sha256: str
    target_artifact_sha256: str

    def __post_init__(self) -> None:
        patients = len(self.patient_ids)
        events = int(self.prefix.shape[0])
        if tuple(self.prefix.shape[1:]) != (15, 77, 200):
            raise ValueError("v11-B prefix must have shape [E,15,77,200]")
        if tuple(self.fine_event.shape) != (events, 19, 20):
            raise ValueError("v11-B fine events must have shape [E,19,20]")
        if tuple(self.reliability.shape) != (events, 19):
            raise ValueError("v11-B reliability must have shape [E,19]")
        if tuple(self.event_patient_index.shape) != (events,):
            raise ValueError("v11-B event-patient index must be [E]")
        if tuple(self.fine_patient.shape) != (patients, 19, 20):
            raise ValueError("v11-B fine patients must have shape [P,19,20]")
        if tuple(self.targets.shape) != (patients, 19) or tuple(
            self.target_mask.shape
        ) != (patients, 19):
            raise ValueError("v11-B target carrier must be [P,19]")
        if tuple(self.patient_folds.shape) != (patients,) or tuple(
            self.event_counts.shape
        ) != (patients,):
            raise ValueError("v11-B patient fold/count carrier must be [P]")
        if not torch.equal(
            self.target_mask,
            V11_CANDIDATE_MASK.view(1, -1).expand_as(self.target_mask),
        ):
            raise ValueError("v11-B requires the fixed 18-candidate mask")
        if int(self.event_counts.sum()) != events:
            raise ValueError("v11-B event counts do not cover the complete bags")


@dataclass(frozen=True)
class TrainBatch:
    prefix: torch.Tensor
    zero_h: torch.Tensor
    reliability: torch.Tensor
    event_patient_index: torch.Tensor
    fine_patient: torch.Tensor
    targets: torch.Tensor
    target_mask: torch.Tensor
    patient_ids: tuple[str, ...]


@dataclass(frozen=True)
class PredictionBatch:
    prefix: torch.Tensor
    zero_h: torch.Tensor
    reliability: torch.Tensor
    event_patient_index: torch.Tensor
    fine_patient: torch.Tensor
    patient_ids: tuple[str, ...]


def _smoke_scope(patient_folds: torch.Tensor) -> torch.Tensor:
    selected = []
    for fold in OUTER_FOLDS:
        matches = torch.nonzero(patient_folds == fold, as_tuple=False).flatten()
        if matches.numel() < 2:
            raise ValueError("smoke scope requires two patients per outer fold")
        selected.extend(matches[:2].tolist())
    return torch.tensor(sorted(selected), dtype=torch.long)


def _load_inputs(args: argparse.Namespace) -> V11BInputs:
    if _file_sha(PROTOCOL_PATH) != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("v11-B protocol changed after it was frozen")
    union = load_public_development_union(
        args.union_directory,
        expected_manifest_sha256=EXPECTED_PUBLIC_DEVELOPMENT_UNION_MANIFEST_SHA256,
    )
    fine_manifest = _load_json_manifest(
        args.fine_directory / "manifest.json", expected_sha=EXPECTED_FINE_MANIFEST_SHA256
    )
    prefix_manifest = _load_json_manifest(
        args.prefix_directory / "manifest.json",
        expected_sha=EXPECTED_PREFIX_MANIFEST_SHA256,
    )
    _require_target_free_cache(fine_manifest, label="v11-B fine evidence")
    _require_target_free_cache(prefix_manifest, label="v11-B LaBraM prefix")
    event_ids = tuple(event.event_id for event in union.events)
    for label, manifest in (("fine", fine_manifest), ("prefix", prefix_manifest)):
        if tuple(str(value) for value in manifest.get("event_ids", ())) != event_ids:
            raise ValueError(f"v11-B {label} event order differs from union")
    fine_file = args.fine_directory / str(fine_manifest["tensor_file"])
    prefix_file = args.prefix_directory / str(prefix_manifest["tensor_file"])
    if _file_sha(fine_file) != EXPECTED_FINE_TENSOR_FILE_SHA256 or (
        _file_sha(prefix_file) != EXPECTED_PREFIX_TENSOR_FILE_SHA256
    ):
        raise ValueError("v11-B frozen input tensor SHA changed")
    fine_payload = load_file(str(fine_file), device="cpu")
    fine_all = fine_payload["features"].detach().float().contiguous()
    prefix_payload = load_file(str(prefix_file), device="cpu")
    prefix_all = prefix_payload["prefix_tokens"].detach().float().contiguous()
    if tuple(fine_all.shape) != (988, 19, 20) or tuple(prefix_all.shape) != (
        988,
        15,
        77,
        200,
    ) or tuple(fine_manifest["feature_names"]) != FINE_TEMPORAL_FEATURE_NAMES:
        raise ValueError("v11-B frozen input shape/vocabulary changed")

    target = load_verified_deepsoz_target_v2_artifact(
        args.target_directory,
        args.source_csv,
        args.split_csv,
        expected_target_artifact_sha256=EXPECTED_TARGET_ARTIFACT_SHA256,
        expected_summary_artifact_sha256=EXPECTED_TARGET_SUMMARY_SHA256,
        expected_readme_artifact_sha256=EXPECTED_TARGET_README_SHA256,
        expected_source_input_sha256=EXPECTED_SOURCE_SHA256,
        expected_split_input_sha256=EXPECTED_SPLIT_SHA256,
    )
    if target.receipt.receipt_sha256 != EXPECTED_TARGET_RECEIPT_SHA256 or (
        target.receipt.policy_sha256 != TARGET_V2_POLICY_SHA256
    ):
        raise ValueError("v11-B target receipt/policy changed")
    target_batch = target.registry.target_batch(union.patient_ids, require_eligible=True)
    targets_all = target_batch.values.cpu()
    mask_all = target_batch.mask.cpu()
    complete = _complete_candidate_label_rows(mask_all)
    excluded = [
        union.patient_ids[index]
        for index in torch.nonzero(~complete, as_tuple=False).flatten().tolist()
    ]
    if excluded != [EXCLUDED_PATIENT]:
        raise ValueError(f"v11-B incomplete reference roster changed: {excluded}")
    eligible_patients = torch.nonzero(complete, as_tuple=False).flatten()
    eligible_folds = torch.tensor(union.patient_folds, dtype=torch.long).index_select(
        0, eligible_patients
    )
    if args.smoke:
        scope_local = _smoke_scope(eligible_folds)
        formal = False
    else:
        scope_local = torch.arange(PRIMARY_PATIENT_COUNT, dtype=torch.long)
        formal = True
    scope_original = eligible_patients.index_select(0, scope_local)
    selected_patient = torch.zeros(len(union.patient_ids), dtype=torch.bool)
    selected_patient[scope_original] = True
    event_patient_original = torch.tensor(union.event_patient_index, dtype=torch.long)
    event_keep = selected_patient[event_patient_original]
    prefix = prefix_all[event_keep]
    fine_event = fine_all[event_keep]
    old_to_new = torch.full((len(union.patient_ids),), -1, dtype=torch.long)
    old_to_new[scope_original] = torch.arange(scope_original.numel())
    event_patient_index = old_to_new[event_patient_original[event_keep]]
    artifact_index = FINE_TEMPORAL_FEATURE_NAMES.index("artifact_burden_0_12s")
    reliability = (1.0 - fine_event[:, :, artifact_index]).clamp(0.0, 1.0)
    fine_pool = robust_pool_complete_patient_bags(
        fine_event,
        event_patient_index,
        int(scope_original.numel()),
        reliability,
    )
    patient_ids = tuple(union.patient_ids[index] for index in scope_original.tolist())
    targets = targets_all.index_select(0, scope_original)
    target_mask = mask_all.index_select(0, scope_original)
    folds = torch.tensor(union.patient_folds, dtype=torch.long).index_select(
        0, scope_original
    )
    if formal and (
        len(patient_ids) != PRIMARY_PATIENT_COUNT
        or prefix.shape[0] != PRIMARY_EVENT_COUNT
        or tuple(torch.bincount(folds, minlength=5).tolist()) != (20, 21, 20, 21, 19)
        or tuple(
            torch.zeros(5, dtype=torch.long)
            .scatter_add_(0, folds, fine_pool.event_counts)
            .tolist()
        )
        != (197, 198, 197, 198, 194)
    ):
        raise ValueError("v11-B formal 101/984 fold scope changed")
    return V11BInputs(
        prefix=prefix,
        fine_event=fine_event,
        reliability=reliability,
        event_patient_index=event_patient_index,
        fine_patient=fine_pool.features,
        patient_ids=patient_ids,
        targets=targets,
        target_mask=target_mask,
        patient_folds=folds,
        event_counts=fine_pool.event_counts,
        formal_scope=formal,
        target_free_union_manifest_sha256=union.manifest_sha256,
        target_receipt_sha256=target.receipt.receipt_sha256,
        target_artifact_sha256=target.receipt.target_artifact_sha256,
    )


def _seeded_suffix(
    args: argparse.Namespace, *, seed: int, device: torch.device
) -> OfficialLaBraMMinimalPEFTSuffix:
    fork_devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(seed)
        suffix = OfficialLaBraMMinimalPEFTSuffix(
            modeling_path=args.modeling_path,
            checkpoint_path=args.checkpoint_path,
            expected_sha256=AUDITED_LABRAM_BASE_SHA256,
            expected_modeling_sha256=AUDITED_LABRAM_MODELING_SHA256,
        )
    return suffix.to(device)


def _qkv_original_state(
    suffix: OfficialLaBraMMinimalPEFTSuffix,
) -> dict[str, torch.Tensor]:
    return {
        f"block{block}.qkv_original": suffix.backbone.blocks[
            block
        ].attn.qkv.parametrizations.weight.original.detach().cpu().clone()
        for block in LABRAM_PEFT_BLOCKS
    }


def _collect_suffix_h(
    suffix: OfficialLaBraMMinimalPEFTSuffix,
    prefix: torch.Tensor,
    *,
    device: torch.device,
    event_microbatch: int,
) -> torch.Tensor:
    rows = []
    suffix.eval()
    with torch.no_grad():
        for indices in _chunks(prefix.shape[0], event_microbatch):
            node = suffix_node_tokens(suffix, prefix.index_select(0, indices).to(device))
            rows.append(differentiable_suffix_phase_contrasts(node).cpu())
    result = torch.cat(rows, dim=0).contiguous()
    if tuple(result.shape) != (prefix.shape[0], 19, 600) or not torch.isfinite(
        result
    ).all():
        raise RuntimeError("v11-B suffix H collection is incomplete")
    return result


def _subset_events(
    event_patient_index: torch.Tensor,
    patient_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    patient_selected = torch.zeros(
        int(event_patient_index.max()) + 1, dtype=torch.bool
    )
    patient_selected[patient_indices] = True
    event_indices = torch.nonzero(
        patient_selected[event_patient_index], as_tuple=False
    ).flatten()
    old_to_new = torch.full((len(patient_selected),), -1, dtype=torch.long)
    old_to_new[patient_indices] = torch.arange(len(patient_indices))
    return event_indices, old_to_new[event_patient_index[event_indices]]


def _train_batch(
    inputs: V11BInputs,
    patient_indices: torch.Tensor,
    zero_h_local: torch.Tensor,
) -> TrainBatch:
    event_indices, local_event_patient = _subset_events(
        inputs.event_patient_index, patient_indices
    )
    if tuple(zero_h_local.shape) != (len(event_indices), 19, 600):
        raise ValueError("train-local zero-LoRA H does not align with complete events")
    return TrainBatch(
        prefix=inputs.prefix.index_select(0, event_indices),
        zero_h=zero_h_local,
        reliability=inputs.reliability.index_select(0, event_indices),
        event_patient_index=local_event_patient,
        fine_patient=inputs.fine_patient.index_select(0, patient_indices),
        targets=inputs.targets.index_select(0, patient_indices),
        target_mask=inputs.target_mask.index_select(0, patient_indices),
        patient_ids=tuple(inputs.patient_ids[index] for index in patient_indices.tolist()),
    )


def _prediction_batch(
    inputs: V11BInputs,
    patient_indices: torch.Tensor,
    zero_h_local: torch.Tensor,
) -> PredictionBatch:
    event_indices, local_event_patient = _subset_events(
        inputs.event_patient_index, patient_indices
    )
    if tuple(zero_h_local.shape) != (len(event_indices), 19, 600):
        raise ValueError("held-local zero-LoRA H does not align with complete events")
    return PredictionBatch(
        prefix=inputs.prefix.index_select(0, event_indices),
        zero_h=zero_h_local,
        reliability=inputs.reliability.index_select(0, event_indices),
        event_patient_index=local_event_patient,
        fine_patient=inputs.fine_patient.index_select(0, patient_indices),
        patient_ids=tuple(inputs.patient_ids[index] for index in patient_indices.tolist()),
    )


def _patient_logits_from_h(
    event_h: torch.Tensor,
    event_patient_index: torch.Tensor,
    reliability: torch.Tensor,
    fine_patient: torch.Tensor,
    transform: FoldFeatureTransform,
    reasoner: SharedPositiveSetReasoner,
    *,
    device: torch.device,
) -> torch.Tensor:
    pooled = differentiable_pool_complete_patient_bags(
        event_h.to(device),
        event_patient_index.to(device),
        fine_patient.shape[0],
        reliability.to(device),
    )
    transformed = apply_fold_transform_differentiable(
        pooled, fine_patient.to(device), transform
    )
    return reasoner(transformed).logits


def _fit_fold_initialization(
    batch: TrainBatch,
) -> tuple[FoldFeatureTransform, Mapping[str, torch.Tensor], Mapping[str, object]]:
    patient_h = robust_pool_complete_patient_bags(
        batch.zero_h,
        batch.event_patient_index,
        len(batch.patient_ids),
        batch.reliability,
    ).features
    local_indices = tuple(range(len(batch.patient_ids)))
    transform = fit_fold_transform(patient_h, batch.fine_patient, local_indices)
    fitted = _fit_reasoner(
        transform.apply(patient_h, batch.fine_patient),
        batch.targets,
        batch.target_mask,
        local_indices,
        use_h=True,
        use_fine=True,
        l2=WARM_START_L2,
    )
    if "h_weight" not in fitted.state or not bool(
        torch.count_nonzero(fitted.state["h_weight"])
    ):
        raise RuntimeError("v11-B fold warm start has a zero H gradient bridge")
    return transform, fitted.state, fitted.diagnostics


def _gradient_payload(
    suffix: OfficialLaBraMMinimalPEFTSuffix,
) -> dict[str, bool]:
    result = {}
    for block in LABRAM_PEFT_BLOCKS:
        adapter = suffix._lora(block)
        for factor in ("A", "B"):
            gradient = getattr(adapter, f"lora_{factor}").grad
            result[f"block{block}_lora_{factor}_finite"] = bool(
                gradient is not None and torch.isfinite(gradient).all()
            )
            result[f"block{block}_lora_{factor}_nonzero"] = bool(
                gradient is not None and torch.count_nonzero(gradient) > 0
            )
    return result


def _train_matched(
    batch: TrainBatch,
    transform: FoldFeatureTransform,
    reasoner: SharedPositiveSetReasoner,
    *,
    epochs: int,
    device: torch.device,
) -> tuple[SharedPositiveSetReasoner, dict[str, object]]:
    optimizer = torch.optim.AdamW(
        reasoner.parameters(), lr=HEAD_LR, weight_decay=WEIGHT_DECAY
    )
    h = batch.zero_h.to(device)
    event_patient = batch.event_patient_index.to(device)
    reliability = batch.reliability.to(device)
    fine = batch.fine_patient.to(device)
    targets = batch.targets.to(device)
    target_mask = batch.target_mask.to(device)
    curve = []
    reasoner.train()
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        pooled = differentiable_pool_complete_patient_bags(
            h, event_patient, len(batch.patient_ids), reliability
        )
        logits = reasoner(
            apply_fold_transform_differentiable(pooled, fine, transform)
        ).logits
        loss = positive_set_mass_loss(logits, targets, target_mask)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            tuple(reasoner.parameters()), MAX_GRAD_NORM
        )
        if not torch.isfinite(grad_norm):
            raise RuntimeError("v11-B matched head gradient became non-finite")
        optimizer.step()
        curve.append(
            {
                "epoch": epoch + 1,
                "positive_set_mass_loss": float(loss.detach()),
                "gradient_norm_before_clip": float(grad_norm.detach()),
            }
        )
    optimizer.zero_grad(set_to_none=True)
    reasoner.eval()
    return reasoner, {
        "epochs": epochs,
        "curve": curve,
        "train_patient_count": len(batch.patient_ids),
        "train_event_count": int(batch.prefix.shape[0]),
        "trainable_parameter_count": reasoner.n_trainable_parameters,
    }


def _train_peft(
    batch: TrainBatch,
    transform: FoldFeatureTransform,
    reasoner: SharedPositiveSetReasoner,
    suffix: OfficialLaBraMMinimalPEFTSuffix,
    initial_h: torch.Tensor,
    *,
    epochs: int,
    device: torch.device,
    event_microbatch: int,
) -> tuple[SharedPositiveSetReasoner, OfficialLaBraMMinimalPEFTSuffix, dict[str, object]]:
    parameter_groups = [
        {
            "params": tuple(reasoner.parameters()),
            "lr": HEAD_LR,
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": tuple(
                parameter for parameter in suffix.parameters() if parameter.requires_grad
            ),
            "lr": LORA_LR,
            "weight_decay": WEIGHT_DECAY,
        },
    ]
    if suffix.n_trainable_parameters != LABRAM_PEFT_TRAINABLE_PARAMETERS:
        raise RuntimeError("v11-B suffix must expose exactly 6,400 trainable parameters")
    total_trainable = sum(
        parameter.numel() for group in parameter_groups for parameter in group["params"]
    )
    if total_trainable != 6436:
        raise RuntimeError("v11-B PEFT must expose exactly 6,436 trainable parameters")
    optimizer = torch.optim.AdamW(parameter_groups)
    event_patient = batch.event_patient_index.to(device)
    reliability = batch.reliability.to(device)
    fine = batch.fine_patient.to(device)
    targets = batch.targets.to(device)
    target_mask = batch.target_mask.to(device)
    original_qkv = _qkv_original_state(suffix)
    curve = []
    first_backward = None
    second_backward = None
    maximum_replay_error = 0.0
    cached_first = initial_h
    for epoch in range(epochs):
        reasoner.train()
        suffix.train()
        first_h = cached_first if epoch == 0 else _collect_suffix_h(
            suffix,
            batch.prefix,
            device=device,
            event_microbatch=event_microbatch,
        )
        cached_first = None
        optimizer.zero_grad(set_to_none=True)
        objective = patient_loss_and_h_upstream(
            first_h.to(device),
            event_patient,
            reliability,
            fine,
            transform,
            reasoner,
            targets,
            target_mask,
        )
        replay_error = 0.0
        for indices in _chunks(batch.prefix.shape[0], event_microbatch):
            node = suffix_node_tokens(
                suffix, batch.prefix.index_select(0, indices).to(device)
            )
            replay_h = differentiable_suffix_phase_contrasts(node)
            reference = first_h.index_select(0, indices).to(device)
            replay_error = max(
                replay_error, float((replay_h.detach() - reference).abs().max())
            )
            replay_h.backward(
                objective.event_h_upstream.index_select(
                    0, indices.to(objective.event_h_upstream.device)
                )
            )
        maximum_replay_error = max(maximum_replay_error, replay_error)
        if replay_error > 1e-6:
            raise RuntimeError("v11-B two-pass H replay became stochastic")
        gradient = _gradient_payload(suffix)
        if epoch == 0:
            first_backward = gradient
        elif epoch == 1:
            second_backward = gradient
        parameters = [
            parameter for group in parameter_groups for parameter in group["params"]
        ]
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, MAX_GRAD_NORM)
        if not torch.isfinite(grad_norm):
            raise RuntimeError("v11-B PEFT gradient became non-finite")
        frozen_with_grad = tuple(
            name
            for name, parameter in suffix.named_parameters()
            if not parameter.requires_grad and parameter.grad is not None
        )
        if frozen_with_grad:
            raise RuntimeError("v11-B frozen foundation parameter received a gradient")
        optimizer.step()
        curve.append(
            {
                "epoch": epoch + 1,
                "positive_set_mass_loss": float(objective.loss),
                "gradient_norm_before_clip": float(grad_norm.detach()),
                "two_pass_replay_max_abs_error": replay_error,
            }
        )
    optimizer.zero_grad(set_to_none=True)
    current_qkv = _qkv_original_state(suffix)
    adapter_before = suffix.lora_state_dict()
    adapter_hash = _state_sha(adapter_before)
    suffix.load_lora_state_dict(adapter_before)
    adapter_replay_hash = _state_sha(suffix.lora_state_dict())
    implementation_checks = {
        "total_trainable_6436": total_trainable == 6436,
        "two_pass_replay_le_1e_6": maximum_replay_error <= 1e-6,
        "original_qkv_unchanged": all(
            torch.equal(original_qkv[name], current_qkv[name]) for name in original_qkv
        ),
        "adapter_only_state_self_replay_exact": adapter_hash == adapter_replay_hash,
        "first_backward_B_finite_nonzero": bool(
            first_backward
            and all(
                first_backward[f"block{block}_lora_B_finite"]
                and first_backward[f"block{block}_lora_B_nonzero"]
                for block in LABRAM_PEFT_BLOCKS
            )
        ),
        "second_backward_A_B_finite_nonzero": bool(
            epochs < 2
            or (
                second_backward
                and all(
                    second_backward[f"block{block}_lora_{factor}_finite"]
                    and second_backward[f"block{block}_lora_{factor}_nonzero"]
                    for block in LABRAM_PEFT_BLOCKS
                    for factor in ("A", "B")
                )
            )
        ),
    }
    if not all(implementation_checks.values()):
        failed = [name for name, passed in implementation_checks.items() if not passed]
        raise RuntimeError(f"v11-B implementation validity gate failed: {failed}")
    reasoner.eval()
    suffix.eval()
    return reasoner, suffix, {
        "epochs": epochs,
        "curve": curve,
        "train_patient_count": len(batch.patient_ids),
        "train_event_count": int(batch.prefix.shape[0]),
        "trainable_parameter_count": total_trainable,
        "first_backward": first_backward,
        "second_backward": second_backward,
        "implementation_checks": implementation_checks,
        "maximum_two_pass_replay_error": maximum_replay_error,
        "original_qkv_sha256": _state_sha(original_qkv),
        "adapter_state_sha256": adapter_hash,
    }


def _predict_outer_held_no_grad(
    batch: PredictionBatch,
    transform: FoldFeatureTransform,
    reasoner: SharedPositiveSetReasoner,
    suffix: OfficialLaBraMMinimalPEFTSuffix | None,
    *,
    device: torch.device,
    event_microbatch: int,
) -> torch.Tensor:
    reasoner.eval()
    if suffix is None:
        event_h = batch.zero_h
    else:
        suffix.eval()
        event_h = _collect_suffix_h(
            suffix,
            batch.prefix,
            device=device,
            event_microbatch=event_microbatch,
        )
    with torch.no_grad():
        logits = _patient_logits_from_h(
            event_h,
            batch.event_patient_index,
            batch.reliability,
            batch.fine_patient,
            transform,
            reasoner,
            device=device,
        )
    return logits.detach().cpu().contiguous()


def _checkpoint_state(
    transform: FoldFeatureTransform,
    matched_head: SharedPositiveSetReasoner,
    peft_head: SharedPositiveSetReasoner,
    suffix: OfficialLaBraMMinimalPEFTSuffix,
    *,
    outer_fold: int,
) -> dict[str, torch.Tensor]:
    prefix = f"outer{outer_fold}"
    state = {
        f"{prefix}.{name}": value.detach().cpu().clone()
        for name, value in _transform_state(transform).items()
    }
    for candidate, model in ((MATCHED_FROZEN, matched_head), (PEFT, peft_head)):
        for name, value in model.state_dict().items():
            state[f"{prefix}.{candidate}.head.{name}"] = value.detach().cpu().clone()
    for name, value in suffix.lora_state_dict().items():
        state[f"{prefix}.peft.adapter.{name}"] = value.detach().cpu().clone()
    forbidden = ("backbone", "original", "checkpoint", "optimizer", "raw", "target")
    if any(any(token in key.lower() for token in forbidden) for key in state):
        raise RuntimeError("v11-B checkpoint contains a forbidden foundation/data payload")
    adapter_values = sum(
        value.numel() for key, value in state.items() if ".peft.adapter." in key
    )
    if adapter_values != LABRAM_PEFT_TRAINABLE_PARAMETERS:
        raise RuntimeError("v11-B checkpoint must contain exactly 6,400 adapter values")
    return state


def _assess_peft_support(
    metrics: Mapping[str, Mapping[str, object]],
    fold_strict: Mapping[str, Sequence[float]],
    paired: Mapping[str, Mapping[str, object]],
    implementation_valid: bool,
) -> tuple[bool, dict[str, bool]]:
    peft = metrics[PEFT]
    matched = metrics[MATCHED_FROZEN]
    checks = {
        "strict_nonlower": peft["top1"]["strict_accuracy"]
        >= matched["top1"]["strict_accuracy"],
        "relaxed_nonlower": peft["top1"]["relaxed_accuracy"]
        >= matched["top1"]["relaxed_accuracy"],
        "macro_ap_positive": peft["ranking"]["macro_average_precision"]
        > matched["ranking"]["macro_average_precision"],
        "far_error_nonincreasing": peft["far_error_count"]
        <= matched["far_error_count"],
        "four_of_five_fold_strict_nonlower": sum(
            left >= right
            for left, right in zip(fold_strict[PEFT], fold_strict[MATCHED_FROZEN])
        )
        >= 4,
        "strict_bootstrap_noninferior": paired["strict"]["ci95"][0]
        >= -NONINFERIORITY_MARGIN,
        "relaxed_bootstrap_noninferior": paired["relaxed"]["ci95"][0]
        >= -NONINFERIORITY_MARGIN,
        "implementation_valid": implementation_valid,
    }
    checks["scientific_strict_increment_supported"] = paired["strict"]["ci95"][0] > 0
    engineering = all(
        value
        for name, value in checks.items()
        if name != "scientific_strict_increment_supported"
    )
    return engineering, {name: bool(value) for name, value in checks.items()}


def _source_hashes() -> dict[str, str]:
    paths = {
        "runner": Path(__file__).resolve(),
        "v11_input_runner": ROOT
        / "scripts/run_labram_fine_temporal_nested_oof_v11.py",
        "v11_1_metrics_runner": ROOT
        / "scripts/run_labram_fine_temporal_nested_oof_v11_1.py",
        "v11b_bridge": ROOT / "src/soz/v11b_peft.py",
        "deepsoz_target_v2": ROOT / "src/soz/data/deepsoz_target_v2.py",
        "deepsoz_target_v1": ROOT / "src/soz/data/deepsoz.py",
        "geometry": ROOT / "src/soz/geometry.py",
        "metrics": ROOT / "src/soz/metrics.py",
        "fine_temporal_evidence": ROOT / "src/soz/fine_temporal_evidence.py",
        "development_union": ROOT / "src/soz/v11_development_union.py",
        "labram_wrapper": ROOT / "src/soz/models/labram.py",
        "peft_suffix": ROOT / "src/soz/models/labram_peft.py",
        "peft_recovery": ROOT / "src/soz/labram_peft_recovery.py",
        "v11_reasoner": ROOT / "src/soz/v11_reasoner.py",
    }
    return {name: _file_sha(path) for name, path in paths.items()}


def run(
    args: argparse.Namespace,
) -> tuple[Mapping[str, object], Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]]:
    started = time.monotonic()
    inputs = _load_inputs(args)
    source_hashes = _source_hashes()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal v11-B requires an available CUDA device")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    zero_qkv_sha: str | None = None

    patients = len(inputs.patient_ids)
    oof = {
        MATCHED_FROZEN: torch.full((patients, 19), torch.nan),
        PEFT: torch.full((patients, 19), torch.nan),
    }
    fold_strict = {MATCHED_FROZEN: [], PEFT: []}
    fold_results = []
    checkpoint: dict[str, torch.Tensor] = {
        "config.candidate_mask": V11_CANDIDATE_MASK.clone()
    }
    all_implementation_valid = True

    for outer_fold in OUTER_FOLDS:
        train_indices = torch.nonzero(
            inputs.patient_folds != outer_fold, as_tuple=False
        ).flatten()
        held_indices = torch.nonzero(
            inputs.patient_folds == outer_fold, as_tuple=False
        ).flatten()
        train_event_indices, _ = _subset_events(
            inputs.event_patient_index, train_indices
        )
        train_prefix = inputs.prefix.index_select(0, train_event_indices)
        seed = BASE_SEED + outer_fold
        suffix = _seeded_suffix(args, seed=seed, device=device)
        zero_adapter_state = suffix.lora_state_dict()
        if any(
            torch.count_nonzero(value) != 0
            for name, value in zero_adapter_state.items()
            if name.endswith("lora_B")
        ):
            raise RuntimeError(
                "v11-B zero adapter must initialize every LoRA-B to zero"
            )
        fold_qkv_sha = _state_sha(_qkv_original_state(suffix))
        if zero_qkv_sha is None:
            zero_qkv_sha = fold_qkv_sha
        elif fold_qkv_sha != zero_qkv_sha:
            raise RuntimeError("v11-B original qkv state changed across folds")
        initial_peft_h = _collect_suffix_h(
            suffix,
            train_prefix,
            device=device,
            event_microbatch=args.event_microbatch,
        )
        repeated_initial_h = _collect_suffix_h(
            suffix,
            train_prefix,
            device=device,
            event_microbatch=args.event_microbatch,
        )
        zero_h_error = float((initial_peft_h - repeated_initial_h).abs().max())
        if zero_h_error > 1e-6:
            difference = (initial_peft_h - repeated_initial_h).abs()
            flat_index = int(difference.argmax())
            location = []
            remainder = flat_index
            for size in reversed(difference.shape):
                location.append(remainder % int(size))
                remainder //= int(size)
            location.reverse()
            raise RuntimeError(
                "v11-B zero-LoRA final-suffix H parity failed: "
                f"max_abs_error={zero_h_error:.9g}, index={tuple(location)}"
            )
        train = _train_batch(inputs, train_indices, initial_peft_h)
        transform, initial_state, warm_diagnostics = _fit_fold_initialization(train)
        matched_head, peft_head = clone_reasoner_pair(
            initial_state, device=device
        )
        initial_head_hash = _state_sha(initial_state)
        with torch.no_grad():
            initial_matched_logits = _patient_logits_from_h(
                train.zero_h,
                train.event_patient_index,
                train.reliability,
                train.fine_patient,
                transform,
                matched_head,
                device=device,
            )
            initial_peft_logits = _patient_logits_from_h(
                initial_peft_h,
                train.event_patient_index,
                train.reliability,
                train.fine_patient,
                transform,
                peft_head,
                device=device,
            )
        step0_error = float((initial_matched_logits - initial_peft_logits).abs().max())
        if step0_error > 1e-6:
            raise RuntimeError("v11-B matched/PEFT step-0 logits differ")

        matched_head, matched_fit = _train_matched(
            train,
            transform,
            matched_head,
            epochs=args.epochs,
            device=device,
        )
        peft_head, suffix, peft_fit = _train_peft(
            train,
            transform,
            peft_head,
            suffix,
            initial_peft_h,
            epochs=args.epochs,
            device=device,
            event_microbatch=args.event_microbatch,
        )
        fit_state_before_held = _state_sha(
            _checkpoint_state(
                transform,
                matched_head,
                peft_head,
                suffix,
                outer_fold=outer_fold,
            )
        )
        # Only after both candidates have completed every optimizer step may
        # the held prefix enter a forward path.  A fresh zero-LoRA suffix
        # supplies the matched-frozen held carrier; no held target is passed.
        held_event_indices, _ = _subset_events(
            inputs.event_patient_index, held_indices
        )
        held_prefix = inputs.prefix.index_select(0, held_event_indices)
        held_zero_suffix = _seeded_suffix(args, seed=seed, device=device)
        held_zero_h = _collect_suffix_h(
            held_zero_suffix,
            held_prefix,
            device=device,
            event_microbatch=args.event_microbatch,
        )
        if _state_sha(_qkv_original_state(held_zero_suffix)) != zero_qkv_sha:
            raise RuntimeError("v11-B held zero-suffix qkv state changed")
        del held_zero_suffix
        torch.cuda.empty_cache()
        held = _prediction_batch(inputs, held_indices, held_zero_h)
        matched_logits = _predict_outer_held_no_grad(
            held,
            transform,
            matched_head,
            None,
            device=device,
            event_microbatch=args.event_microbatch,
        )
        peft_logits = _predict_outer_held_no_grad(
            held,
            transform,
            peft_head,
            suffix,
            device=device,
            event_microbatch=args.event_microbatch,
        )
        fold_state = _checkpoint_state(
            transform,
            matched_head,
            peft_head,
            suffix,
            outer_fold=outer_fold,
        )
        if _state_sha(fold_state) != fit_state_before_held:
            raise RuntimeError("v11-B held prediction mutated the trained state")
        checkpoint.update(fold_state)
        oof[MATCHED_FROZEN].index_copy_(0, held_indices, matched_logits)
        oof[PEFT].index_copy_(0, held_indices, peft_logits)
        held_targets = inputs.targets.index_select(0, held_indices)
        held_mask = inputs.target_mask.index_select(0, held_indices)
        held_metrics = {
            MATCHED_FROZEN: _evaluate(matched_logits, held_targets, held_mask),
            PEFT: _evaluate(peft_logits, held_targets, held_mask),
        }
        for candidate in (MATCHED_FROZEN, PEFT):
            fold_strict[candidate].append(
                held_metrics[candidate]["top1"]["strict_accuracy"]
            )
        implementation_valid = all(peft_fit["implementation_checks"].values())
        all_implementation_valid &= implementation_valid
        fold_results.append(
            {
                "outer_fold": outer_fold,
                "train_patient_count": len(train.patient_ids),
                "train_event_count": int(train.prefix.shape[0]),
                "held_patient_count": len(held.patient_ids),
                "held_event_count": int(held.prefix.shape[0]),
                "train_patient_ids": list(train.patient_ids),
                "held_patient_ids": list(held.patient_ids),
                "fold_local_warm_start": dict(warm_diagnostics),
                "initial_head_state_sha256_both_candidates": initial_head_hash,
                "zero_lora_h_max_abs_error": zero_h_error,
                "step0_patient_logit_max_abs_error": step0_error,
                "matched_fit": matched_fit,
                "peft_fit": peft_fit,
                "held_metrics": held_metrics,
                "state_sha256_before_and_after_held": fit_state_before_held,
            }
        )
        print(
            json.dumps(
                {
                    "outer_fold": outer_fold,
                    "status": "complete",
                    "matched_strict": held_metrics[MATCHED_FROZEN]["top1"][
                        "strict_accuracy"
                    ],
                    "peft_strict": held_metrics[PEFT]["top1"]["strict_accuracy"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del matched_head, peft_head, suffix
        torch.cuda.empty_cache()

    if any(not torch.isfinite(value).all() for value in oof.values()):
        raise RuntimeError("v11-B OOF prediction matrix is incomplete")
    metrics = {
        name: _evaluate(value, inputs.targets, inputs.target_mask)
        for name, value in oof.items()
    }
    absolute = {
        name: _absolute_bootstrap(value, inputs.targets, inputs.target_mask)
        for name, value in oof.items()
    }
    paired = _paired_bootstrap(
        oof[PEFT], oof[MATCHED_FROZEN], inputs.targets, inputs.target_mask
    )
    engineering_support, decision_checks = _assess_peft_support(
        metrics, fold_strict, paired, all_implementation_valid
    )

    if _file_sha(V11_1_REFERENCE / "manifest.json") != EXPECTED_V11_1_MANIFEST_SHA256 or (
        _file_sha(V11_1_REFERENCE / "oof_predictions.safetensors")
        != EXPECTED_V11_1_OOF_SHA256
    ):
        raise ValueError("v11-B pinned v11.1 descriptive comparator changed")
    v11_1_manifest = json.loads(
        (V11_1_REFERENCE / "manifest.json").read_text(encoding="utf-8")
    )
    reference_ids = tuple(str(value) for value in v11_1_manifest["patient_ids"])
    reference_index = {patient: index for index, patient in enumerate(reference_ids)}
    if len(reference_ids) != PRIMARY_PATIENT_COUNT or not set(
        inputs.patient_ids
    ).issubset(reference_index):
        raise ValueError("v11-B v11.1 comparator patient roster changed")
    reference_rows = torch.tensor(
        [reference_index[patient] for patient in inputs.patient_ids], dtype=torch.long
    )
    if inputs.formal_scope and reference_rows.tolist() != list(
        range(PRIMARY_PATIENT_COUNT)
    ):
        raise ValueError("formal v11-B/v11.1 patient order changed")
    v11_1_payload = load_file(
        str(V11_1_REFERENCE / "oof_predictions.safetensors"), device="cpu"
    )
    reference_targets = v11_1_payload["targets"].index_select(0, reference_rows)
    reference_mask = v11_1_payload["target_mask"].index_select(0, reference_rows)
    if not torch.equal(reference_targets, inputs.targets) or not torch.equal(
        reference_mask, inputs.target_mask
    ):
        raise ValueError("v11-B v11.1 comparator target carrier changed")
    v11_1_logits = v11_1_payload[
        "oof.full_frozen_labram_plus_fine"
    ].index_select(0, reference_rows)
    descriptive_v11_1 = {
        "not_capacity_matched": True,
        "metrics": _evaluate(v11_1_logits, inputs.targets, inputs.target_mask),
        "paired_peft_minus_v11_1": _paired_bootstrap(
            oof[PEFT], v11_1_logits, inputs.targets, inputs.target_mask
        ),
    }

    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    if _source_hashes() != source_hashes:
        raise RuntimeError("v11-B source files changed during execution")
    formal = inputs.formal_scope and args.epochs == FORMAL_EPOCHS and (
        args.event_microbatch == EVENT_MICROBATCH
    )
    if not formal and not args.smoke:
        raise RuntimeError("non-formal v11-B configuration must be explicitly smoke-only")
    status = "completed_formal_developmental_oof" if formal else "completed_smoke_only"
    decision = (
        "PEFT_ENGINEERING_SUPPORT"
        if formal and engineering_support
        else (
            "PEFT_NOT_SUPPORTED_STOP_NO_MORE_SCAN"
            if formal
            else "SMOKE_ONLY_NO_SCIENTIFIC_DECISION"
        )
    )
    manifest = {
        "schema_version": SCHEMA,
        "status": status,
        "decision": decision,
        "formal_scope": formal,
        "claim_boundary": {
            "pretraining_exposed_developmental_capacity_diagnostic": True,
            "public_confirmation": False,
            "external_validation": False,
            "private_used": False,
            "clinical_deployment_allowed": False,
        },
        "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "source_file_sha256": source_hashes,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "safetensors": safetensors.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
        },
        "foundation": {
            "backbone": "official_pretrained_LaBraM_Base_not_replaced",
            "checkpoint_sha256": AUDITED_LABRAM_BASE_SHA256,
            "modeling_sha256": AUDITED_LABRAM_MODELING_SHA256,
            "trainable_blocks": list(LABRAM_PEFT_BLOCKS),
            "trainable_scope": "attention_qkv_lora_r4_alpha8_only",
            "foundation_trainable_parameters": LABRAM_PEFT_TRAINABLE_PARAMETERS,
            "foundation_weights_serialized": False,
            "zero_original_qkv_sha256": zero_qkv_sha,
        },
        "training": {
            "base_seed": BASE_SEED,
            "fold_seed_rule": "base_seed_plus_outer_fold",
            "fold_seeds": _fold_seed_manifest(),
            "epochs": args.epochs,
            "head_learning_rate": HEAD_LR,
            "lora_learning_rate": LORA_LR,
            "weight_decay": WEIGHT_DECAY,
            "max_gradient_norm": MAX_GRAD_NORM,
            "event_microbatch": args.event_microbatch,
            "amp": False,
            "early_stopping": False,
            "warm_start_l2": WARM_START_L2,
            "loss": "patient_equal_positive_set_mass",
            "formal_epoch_budget": FORMAL_EPOCHS,
        },
        "patient_count": len(inputs.patient_ids),
        "event_count": int(inputs.prefix.shape[0]),
        "patient_ids": list(inputs.patient_ids),
        "event_counts": inputs.event_counts.tolist(),
        "patient_folds": inputs.patient_folds.tolist(),
        "excluded_partial_reference_patient": EXCLUDED_PATIENT,
        "fixed_candidate_mask": V11_CANDIDATE_MASK.tolist(),
        "prediction_score_contract": {
            "carrier_channels": 19,
            "candidate_count": int(V11_CANDIDATE_MASK.sum().item()),
            "excluded_carrier_channels": ["PZ"],
            "oof_scores_are_candidate_masked": True,
            "excluded_score": "torch.finfo(dtype).min",
            "all_oof_scores_finite": True,
            "official_metric_replay_compatible": True,
            "checkpoint_inference_requires_candidate_mask": True,
        },
        "fold_results": fold_results,
        "metrics": metrics,
        "absolute_patient_bootstrap": absolute,
        "paired_peft_minus_matched_frozen": paired,
        "fold_strict": fold_strict,
        "implementation_valid_all_folds": all_implementation_valid,
        "decision_checks": decision_checks,
        "engineering_support": bool(formal and engineering_support),
        "scientific_strict_increment_supported": bool(
            formal and decision_checks["scientific_strict_increment_supported"]
        ),
        "v11_1_block9_descriptive_comparison": descriptive_v11_1,
        "resource_usage": {
            "wall_time_seconds": time.monotonic() - started,
            "cuda_peak_allocated_bytes": int(peak_allocated),
            "cuda_peak_reserved_bytes": int(peak_reserved),
        },
        "lineage": {
            "union_manifest_sha256": inputs.target_free_union_manifest_sha256,
            "fine_manifest_sha256": EXPECTED_FINE_MANIFEST_SHA256,
            "fine_tensor_sha256": EXPECTED_FINE_TENSOR_FILE_SHA256,
            "prefix_manifest_sha256": EXPECTED_PREFIX_MANIFEST_SHA256,
            "prefix_tensor_sha256": EXPECTED_PREFIX_TENSOR_FILE_SHA256,
            "target_receipt_sha256": inputs.target_receipt_sha256,
            "target_artifact_sha256": inputs.target_artifact_sha256,
            "v11_1_manifest_sha256": EXPECTED_V11_1_MANIFEST_SHA256,
            "v11_1_oof_sha256": EXPECTED_V11_1_OOF_SHA256,
        },
        "access_receipt": {
            "patient_258_excluded_before_all_fit": True,
            "held_targets_not_passed_to_prediction": True,
            "held_prediction_no_grad_and_state_immutable": True,
            "private_eeg_loaded": False,
            "private_target_values_loaded": False,
            "private_forward_count": 0,
            "llm_used_as_soz_predictor": False,
        },
    }
    tensors = {
        f"oof.{name}": _masked_oof_scores_for_publish(value)
        for name, value in oof.items()
    }
    tensors.update(
        {
            "targets": inputs.targets,
            "target_mask": inputs.target_mask,
            "patient_folds": inputs.patient_folds,
            "patient_event_counts": inputs.event_counts,
            "config.candidate_mask": V11_CANDIDATE_MASK.clone(),
        }
    )
    return manifest, tensors, checkpoint


def _publish(
    output_directory: Path,
    manifest: Mapping[str, object],
    tensors: Mapping[str, torch.Tensor],
    checkpoint: Mapping[str, torch.Tensor],
) -> Path:
    if not torch.equal(checkpoint["config.candidate_mask"], V11_CANDIDATE_MASK):
        raise ValueError("v11-B checkpoint lost the fixed candidate mask")
    target = Path(os.path.abspath(output_directory))
    if target.exists():
        raise FileExistsError(target)
    if not target.parent.is_dir():
        raise FileNotFoundError(target.parent)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        prediction_path = staging / "oof_predictions.safetensors"
        checkpoint_path = staging / "fold_adapter_states.safetensors"
        save_file(dict(tensors), str(prediction_path))
        save_file(dict(checkpoint), str(checkpoint_path))
        completed = dict(manifest)
        completed["files"] = {
            prediction_path.name: {
                "sha256": _file_sha(prediction_path),
                "size_bytes": prediction_path.stat().st_size,
            },
            checkpoint_path.name: {
                "sha256": _file_sha(checkpoint_path),
                "size_bytes": checkpoint_path.stat().st_size,
            },
        }
        (staging / "manifest.json").write_bytes(_canonical_bytes(completed, newline=True))
        os.replace(staging, target)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return target


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--union-directory", type=Path, default=DEFAULT_UNION)
    parser.add_argument("--fine-directory", type=Path, default=DEFAULT_FINE)
    parser.add_argument("--prefix-directory", type=Path, default=DEFAULT_PREFIX)
    parser.add_argument("--target-directory", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--modeling-path", type=Path, default=DEFAULT_MODELING)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=FORMAL_EPOCHS)
    parser.add_argument("--event-microbatch", type=int, default=EVENT_MICROBATCH)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.event_microbatch < 1:
        parser.error("epochs and event-microbatch must be positive")
    if not args.smoke and (
        args.epochs != FORMAL_EPOCHS or args.event_microbatch != EVENT_MICROBATCH
    ):
        parser.error("formal v11-B locks epochs=20 and event-microbatch=4")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    manifest, tensors, checkpoint = run(args)
    output = args.output_directory
    if args.smoke and output == DEFAULT_OUTPUT:
        output = ROOT / "outputs/labram_fine_temporal_peft_oof_v11_b_smoke_r3"
    path = _publish(output, manifest, tensors, checkpoint)
    completed = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "status": completed["status"],
                "decision": completed["decision"],
                "path": str(path),
                "manifest_sha256": _file_sha(path / "manifest.json"),
                "matched_strict": completed["metrics"][MATCHED_FROZEN]["top1"][
                    "strict_accuracy"
                ],
                "peft_strict": completed["metrics"][PEFT]["top1"][
                    "strict_accuracy"
                ],
                "private_used": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
