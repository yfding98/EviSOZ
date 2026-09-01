#!/usr/bin/env python3
"""Run the fixed LaBraM time-resolved global-gate developmental OOF v12."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
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
    _load_json_manifest,
    _require_target_free_cache,
    _state_sha,
)
from scripts.run_labram_fine_temporal_nested_oof_v11_1 import (  # noqa: E402
    _absolute_bootstrap,
    _complete_candidate_label_rows,
    _evaluate,
    _event_consistency,
    _event_count_strata,
    _paired_bootstrap,
)
from src.soz.data.deepsoz_target_v2 import (  # noqa: E402
    TARGET_V2_POLICY_SHA256,
    load_verified_deepsoz_target_v2_artifact,
)
from src.soz.fine_temporal_evidence import FINE_TEMPORAL_FEATURE_NAMES  # noqa: E402
from src.soz.time_resolved_localizer_v12 import (  # noqa: E402
    TimeResolvedEventOutputV12,
    TimeResolvedNodeLocalizerV12,
    V12PatientAggregation,
    V12TemporalMasks,
    V12_CANDIDATE_MASK,
    V12_GATE_FLOOR,
    V12_NODE_PCA_DIM,
    V12_N_SECONDS,
    V12_PREICTAL_SECONDS,
    V12_RELIABILITY_FLOOR,
    V12_TIME_RESOLVED_SCHEMA,
    baseline_difference_node_time,
    fit_node_feature_transform,
    jeffreys_reference_prior_logits,
    positive_set_mass_loss,
    restore_prefix_node_time,
    robust_aggregate_patient_logits,
)
from src.soz.v11_development_union import (  # noqa: E402
    EXPECTED_PUBLIC_DEVELOPMENT_UNION_MANIFEST_SHA256,
    load_public_development_union,
)


PROTOCOL_PATH = (
    ROOT
    / "research/02_method/"
    "labram_time_resolved_global_gate_recovery_protocol_v12_20260811_zh.md"
)
EXPECTED_PROTOCOL_SHA256 = (
    "9dc78999542a033ec8e75e0b742574e6a0f9df3cc31e52f221933c284d318402"
)
DEFAULT_OUTPUT = ROOT / "outputs/labram_time_resolved_global_gate_oof_v12_20260811"
DEFAULT_V11_1_REFERENCE = (
    ROOT / "outputs/labram_fine_temporal_nested_oof_v11_1_20260811_r2"
)
EXPECTED_V11_1_MANIFEST_SHA256 = (
    "f399678e5756ae30cbe5f9f87d9d8bb5b220b16015e1b2a0417110f20e70195c"
)
EXPECTED_V11_1_OOF_SHA256 = (
    "6443680b18b53b0c552b9634e7c9e2547284c9d08cccd5cd99c35b9e1a27ac08"
)

SCHEMA = "soz_labram_time_resolved_global_gate_oof_v12"
PRIMARY_PATIENT_COUNT = 101
PRIMARY_EVENT_COUNT = 984
EXCLUDED_PARTIAL_REFERENCE_PATIENT = "258"
EXPECTED_FOLD_PATIENT_COUNTS = (20, 21, 20, 21, 19)
EXPECTED_FOLD_EVENT_COUNTS = (197, 198, 197, 198, 194)
UNIFORM = "uniform_time_matched_control"
LEARNED = "learned_positive_derivative_global_gate"
V11_FULL = "v11_1_full_frozen_labram_plus_fine"
V11_FROZEN = "v11_1_frozen_labram_only"
L2 = 0.20
MACRO_AP_MINIMUM_GAIN = 1.0e-8
STEP0_PARITY_ATOL = 5.0e-6
LBFGS_LR = 1.0
LBFGS_MAX_ITER = 100
LBFGS_TOLERANCE_GRAD = 1.0e-7
LBFGS_TOLERANCE_CHANGE = 1.0e-9
LBFGS_LINE_SEARCH = "strong_wolfe"


@dataclass(frozen=True)
class V12Inputs:
    node_time: torch.Tensor
    temporal_masks: V12TemporalMasks
    duration_seconds: torch.Tensor
    reliability: torch.Tensor
    event_patient_index: torch.Tensor
    event_ids: tuple[str, ...]
    patient_ids: tuple[str, ...]
    targets: torch.Tensor
    target_mask: torch.Tensor
    patient_folds: torch.Tensor
    event_counts: torch.Tensor
    union_manifest_sha256: str
    target_receipt_sha256: str
    target_artifact_sha256: str

    def __post_init__(self) -> None:
        events = int(self.node_time.shape[0])
        patients = len(self.patient_ids)
        if tuple(self.node_time.shape) != (events, 19, V12_N_SECONDS, 200):
            raise ValueError("v12 node-time carrier must be [E,19,60,200]")
        if self.temporal_masks.n_events != events:
            raise ValueError("v12 temporal masks do not align with events")
        if tuple(self.duration_seconds.shape) != (events,) or not torch.isfinite(
            self.duration_seconds
        ).all():
            raise ValueError("v12 event durations must be finite [E]")
        if tuple(self.reliability.shape) != (events, 19):
            raise ValueError("v12 artifact reliability must be [E,19]")
        if self.reliability.requires_grad or not torch.isfinite(
            self.reliability
        ).all():
            raise ValueError("v12 artifact reliability must be finite and detached")
        if torch.any(
            (self.reliability < V12_RELIABILITY_FLOOR)
            | (self.reliability > 1)
        ):
            raise ValueError("v12 artifact reliability must lie in [0.1,1]")
        if tuple(self.event_patient_index.shape) != (events,) or (
            self.event_patient_index.dtype != torch.long
        ):
            raise TypeError("v12 event-patient index must be long [E]")
        if len(self.event_ids) != events or len(set(self.event_ids)) != events:
            raise ValueError("v12 event IDs must be unique and complete")
        if tuple(self.targets.shape) != (patients, 19) or tuple(
            self.target_mask.shape
        ) != (patients, 19):
            raise ValueError("v12 target carrier must be [P,19]")
        if not torch.equal(
            self.target_mask,
            V12_CANDIDATE_MASK.view(1, -1).expand_as(self.target_mask),
        ):
            raise ValueError("v12 requires the fixed 18-candidate target mask")
        if tuple(self.patient_folds.shape) != (patients,) or tuple(
            self.event_counts.shape
        ) != (patients,):
            raise ValueError("v12 patient fold/count carriers must be [P]")
        if int(self.event_counts.sum()) != events:
            raise ValueError("v12 event counts do not cover complete patient bags")


@dataclass(frozen=True)
class PinnedV11Reference:
    full: torch.Tensor
    frozen: torch.Tensor
    manifest_sha256: str
    predictions_sha256: str


@dataclass(frozen=True)
class FitResult:
    diagnostics: Mapping[str, object]
    state: Mapping[str, torch.Tensor]


def _masked_scores_for_publish(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[1] != 19 or not logits.is_floating_point():
        raise ValueError("published v12 logits must be floating point [P,19]")
    if not torch.isfinite(logits).all():
        raise ValueError("published v12 logits must be finite before candidate masking")
    sentinel = torch.finfo(logits.dtype).min
    return logits.detach().cpu().masked_fill(~V12_CANDIDATE_MASK, sentinel).contiguous()


def _build_temporal_masks(
    global_t0_seconds: Sequence[float],
    global_stop_seconds: Sequence[float],
) -> tuple[V12TemporalMasks, torch.Tensor]:
    """Keep only 1-s tokens wholly contained in ``[t0, stop)``."""

    if len(global_t0_seconds) != len(global_stop_seconds) or not global_t0_seconds:
        raise ValueError("v12 t0/stop vectors must be aligned and non-empty")
    t0 = torch.tensor(tuple(global_t0_seconds), dtype=torch.float64)
    stop = torch.tensor(tuple(global_stop_seconds), dtype=torch.float64)
    duration = stop - t0
    if not torch.isfinite(duration).all() or torch.any(duration <= 0):
        raise ValueError("v12 every public event must have stop strictly after t0")

    events = int(duration.numel())
    baseline = torch.zeros((events, V12_N_SECONDS), dtype=torch.bool)
    baseline[:, :V12_PREICTAL_SECONDS] = True
    ictal = torch.zeros_like(baseline)
    relative_second = torch.arange(
        V12_N_SECONDS - V12_PREICTAL_SECONDS, dtype=torch.long
    )
    # A cached token covers [t,t+1).  Requiring t+1 <= duration excludes the
    # partially post-stop final token.  K=floor(D), capped at 48, is therefore
    # deliberately conservative rather than a ceil/overlap discretization.
    valid_count = torch.floor(duration).long().clamp(
        max=V12_N_SECONDS - V12_PREICTAL_SECONDS
    )
    if torch.any(valid_count < 1):
        raise ValueError("v12 every event needs at least one full ictal second")
    ictal[:, V12_PREICTAL_SECONDS:] = relative_second.unsqueeze(0) < valid_count.unsqueeze(
        1
    )
    masks = V12TemporalMasks(
        preictal_baseline_mask=baseline,
        ictal_valid_mask=ictal,
    )
    actual_count = ictal[:, V12_PREICTAL_SECONDS:].sum(dim=1)
    if not torch.equal(actual_count, valid_count):
        raise RuntimeError("v12 true-stop mask no longer follows floor(duration)")
    return masks, duration.float().contiguous()


def _subset_masks(masks: V12TemporalMasks, indices: torch.Tensor) -> V12TemporalMasks:
    return V12TemporalMasks(
        preictal_baseline_mask=masks.preictal_baseline_mask.index_select(0, indices),
        ictal_valid_mask=masks.ictal_valid_mask.index_select(0, indices),
    )


def _subset_events(
    event_patient_index: torch.Tensor,
    patient_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if patient_indices.ndim != 1 or patient_indices.dtype != torch.long or (
        patient_indices.numel() < 1
    ):
        raise TypeError("patient indices must be non-empty long [K]")
    n_patients = int(event_patient_index.max()) + 1
    selected = torch.zeros(n_patients, dtype=torch.bool)
    selected[patient_indices] = True
    event_indices = torch.nonzero(
        selected[event_patient_index], as_tuple=False
    ).flatten()
    old_to_new = torch.full((n_patients,), -1, dtype=torch.long)
    old_to_new[patient_indices] = torch.arange(patient_indices.numel())
    local = old_to_new[event_patient_index.index_select(0, event_indices)]
    if local.min().item() != 0 or local.max().item() != patient_indices.numel() - 1:
        raise RuntimeError("v12 event subsetting lost a complete patient bag")
    return event_indices, local


def _model_state(model: TimeResolvedNodeLocalizerV12) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _fixed18_log_prob(logits: torch.Tensor, *, level: str) -> torch.Tensor:
    """Normalize only the evaluable 18 candidates and keep PZ a finite carrier."""

    if logits.ndim != 2 or logits.shape[1] != 19 or not logits.is_floating_point():
        raise ValueError(f"v12 {level} logits must be floating point [N,19]")
    if not torch.isfinite(logits).all():
        raise ValueError(f"v12 {level} logits must be finite before normalization")
    candidate_mask = V12_CANDIDATE_MASK.to(device=logits.device)
    result = torch.zeros_like(logits)
    result[:, candidate_mask] = torch.log_softmax(
        logits[:, candidate_mask], dim=1
    )
    if not torch.isfinite(result).all() or bool(
        torch.count_nonzero(result[:, ~candidate_mask])
    ):
        raise RuntimeError(f"v12 {level} log-probability carrier is invalid")
    mass = result[:, candidate_mask].exp().sum(dim=1)
    if not torch.allclose(
        mass,
        torch.ones_like(mass),
        atol=1.0e-6,
        rtol=1.0e-6,
    ):
        raise RuntimeError(f"v12 {level} fixed-18 mass is not normalized")
    return result.contiguous()


def _trainable_parameters(
    model: TimeResolvedNodeLocalizerV12,
) -> tuple[torch.nn.Parameter, ...]:
    values = tuple(value for value in model.parameters() if value.requires_grad)
    if not values:
        raise RuntimeError("v12 localizer has no trainable parameters")
    return values


def _patient_logits(
    model: TimeResolvedNodeLocalizerV12,
    z: torch.Tensor,
    masks: V12TemporalMasks,
    event_patient_index: torch.Tensor,
    n_patients: int,
    reliability: torch.Tensor,
) -> tuple[
    TimeResolvedEventOutputV12,
    torch.Tensor,
    V12PatientAggregation,
    torch.Tensor,
]:
    event = model(z, masks)
    event_log_prob = _fixed18_log_prob(event.event_logits, level="event")
    patient = robust_aggregate_patient_logits(
        event_log_prob,
        event_patient_index,
        n_patients,
        reliability,
    )
    patient_log_prob = _fixed18_log_prob(patient.logits, level="patient")
    return event, event_log_prob, patient, patient_log_prob


def _fit_localizer(
    model: TimeResolvedNodeLocalizerV12,
    z: torch.Tensor,
    masks: V12TemporalMasks,
    event_patient_index: torch.Tensor,
    reliability: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
) -> FitResult:
    patients = int(targets.shape[0])
    parameters = _trainable_parameters(model)
    optimizer = torch.optim.LBFGS(
        parameters,
        lr=LBFGS_LR,
        max_iter=LBFGS_MAX_ITER,
        tolerance_grad=LBFGS_TOLERANCE_GRAD,
        tolerance_change=LBFGS_TOLERANCE_CHANGE,
        line_search_fn=LBFGS_LINE_SEARCH,
    )
    closure_calls = 0
    first_total: float | None = None

    def objective() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, _, _, logits = _patient_logits(
            model,
            z,
            masks,
            event_patient_index,
            patients,
            reliability,
        )
        set_loss = positive_set_mass_loss(logits, targets, target_mask)
        penalty = sum(parameter.square().sum() for parameter in parameters)
        return set_loss + L2 * penalty, set_loss, penalty

    def closure() -> torch.Tensor:
        nonlocal closure_calls, first_total
        optimizer.zero_grad(set_to_none=True)
        total, _, _ = objective()
        if not torch.isfinite(total):
            raise RuntimeError("v12 LBFGS objective became non-finite")
        total.backward()
        closure_calls += 1
        if first_total is None:
            first_total = float(total.detach())
        return total

    model.train()
    optimizer.step(closure)
    optimizer.zero_grad(set_to_none=True)
    final_total_tensor, final_set_tensor, final_penalty_tensor = objective()
    if not torch.isfinite(final_total_tensor):
        raise RuntimeError("v12 final objective became non-finite")
    final_total_tensor.backward()
    gradient_terms = [
        parameter.grad.detach().square().sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    final_gradient_norm = float(
        torch.sqrt(sum(gradient_terms)) if gradient_terms else torch.tensor(0.0)
    )
    final_total = float(final_total_tensor.detach())
    if first_total is None:
        raise RuntimeError("v12 LBFGS did not call its closure")
    if final_total > first_total + 1.0e-6:
        raise RuntimeError("v12 final objective is worse than initialization")
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    optimizer_state = next(iter(optimizer.state.values()), {})
    return FitResult(
        diagnostics={
            "gate_mode": model.gate_mode,
            "l2": L2,
            "train_patient_count": patients,
            "train_event_count": int(z.shape[0]),
            "trainable_parameter_count": model.n_trainable_parameters,
            "closure_calls": closure_calls,
            "first_total_loss": first_total,
            "final_total_loss": final_total,
            "final_set_mass_loss": float(final_set_tensor.detach()),
            "final_l2_penalty_unweighted": float(final_penalty_tensor.detach()),
            "final_gradient_norm": final_gradient_norm,
            "optimizer_iterations": int(optimizer_state.get("n_iter", 0)),
            "optimizer_function_evaluations": int(
                optimizer_state.get("func_evals", closure_calls)
            ),
        },
        state=_model_state(model),
    )


def _matched_learned_initialization(
    uniform: TimeResolvedNodeLocalizerV12,
    prior: torch.Tensor,
) -> TimeResolvedNodeLocalizerV12:
    learned = TimeResolvedNodeLocalizerV12(prior, gate_mode="learned")
    with torch.no_grad():
        learned.node_weight.copy_(uniform.node_weight)
        learned.gate_weight.zero_()
    if not torch.equal(learned.node_weight, uniform.node_weight) or bool(
        torch.count_nonzero(learned.gate_weight)
    ):
        raise RuntimeError("v12 learned gate lost its matched uniform initialization")
    return learned


def _step0_parity(
    uniform: TimeResolvedNodeLocalizerV12,
    learned: TimeResolvedNodeLocalizerV12,
    z: torch.Tensor,
    masks: V12TemporalMasks,
    event_patient_index: torch.Tensor,
    reliability: torch.Tensor,
    n_patients: int,
) -> dict[str, float]:
    uniform.eval()
    learned.eval()
    with torch.no_grad():
        (
            uniform_event,
            uniform_event_log_prob,
            uniform_aggregation,
            uniform_patient,
        ) = _patient_logits(
            uniform,
            z,
            masks,
            event_patient_index,
            n_patients,
            reliability,
        )
        (
            learned_event,
            learned_event_log_prob,
            learned_aggregation,
            learned_patient,
        ) = _patient_logits(
            learned,
            z,
            masks,
            event_patient_index,
            n_patients,
            reliability,
        )
    errors = {
        "event_logit_max_abs_error": float(
            (uniform_event.event_logits - learned_event.event_logits).abs().max()
        ),
        "event_log_probability_max_abs_error": float(
            (uniform_event_log_prob - learned_event_log_prob).abs().max()
        ),
        "patient_logit_max_abs_error": float(
            (uniform_patient - learned_patient).abs().max()
        ),
        "pooled_log_probability_max_abs_error": float(
            (uniform_aggregation.logits - learned_aggregation.logits).abs().max()
        ),
        "gate_weight_max_abs_error": float(
            (uniform_event.gate_weights - learned_event.gate_weights).abs().max()
        ),
    }
    # The learned softplus-normalized zero gate and the explicit uniform
    # division are mathematically identical but can differ by a few float32
    # ulps after two fixed-18 log-softmax operations and patient pooling.
    if any(value > STEP0_PARITY_ATOL for value in errors.values()):
        raise RuntimeError(f"v12 matched step-0 parity failed: {errors}")
    return errors


def _gate_audit(output: TimeResolvedEventOutputV12) -> dict[str, float]:
    weights = output.gate_weights.detach()
    uniform = output.uniform_gate_weights.detach()
    effective_seconds = weights.square().sum(dim=1).reciprocal()
    peak_relative_seconds = weights.argmax(dim=1).float() - V12_PREICTAL_SECONDS
    return {
        "mean_l1_distance_from_uniform": float((weights - uniform).abs().sum(dim=1).mean()),
        "mean_effective_gate_seconds": float(effective_seconds.mean()),
        "min_effective_gate_seconds": float(effective_seconds.min()),
        "mean_peak_relative_second": float(peak_relative_seconds.mean()),
        "gate_sum_max_abs_error": float((weights.sum(dim=1) - 1.0).abs().max()),
        "post_stop_mass_max": float(
            weights.masked_select(~output.ictal_valid_mask).abs().max()
            if bool((~output.ictal_valid_mask).any())
            else 0.0
        ),
    }


def _load_inputs(args: argparse.Namespace) -> V12Inputs:
    union = load_public_development_union(
        args.union_directory,
        expected_manifest_sha256=EXPECTED_PUBLIC_DEVELOPMENT_UNION_MANIFEST_SHA256,
    )
    fine_manifest = _load_json_manifest(
        args.fine_directory / "manifest.json",
        expected_sha=EXPECTED_FINE_MANIFEST_SHA256,
    )
    prefix_manifest = _load_json_manifest(
        args.prefix_directory / "manifest.json",
        expected_sha=EXPECTED_PREFIX_MANIFEST_SHA256,
    )
    _require_target_free_cache(fine_manifest, label="v12 artifact reliability")
    _require_target_free_cache(prefix_manifest, label="v12 LaBraM prefix")
    union_event_ids = tuple(event.event_id for event in union.events)
    for label, manifest in (("fine", fine_manifest), ("prefix", prefix_manifest)):
        if tuple(str(value) for value in manifest.get("event_ids", ())) != union_event_ids:
            raise ValueError(f"v12 {label} event order differs from frozen union")
    fine_file = args.fine_directory / str(fine_manifest["tensor_file"])
    prefix_file = args.prefix_directory / str(prefix_manifest["tensor_file"])
    if _file_sha(fine_file) != EXPECTED_FINE_TENSOR_FILE_SHA256 or (
        _file_sha(prefix_file) != EXPECTED_PREFIX_TENSOR_FILE_SHA256
    ):
        raise ValueError("v12 frozen evidence tensor SHA changed")

    fine_payload = load_file(str(fine_file), device="cpu")
    fine_all = fine_payload["features"].detach().float().contiguous()
    prefix_payload = load_file(str(prefix_file), device="cpu")
    prefix_all = prefix_payload["prefix_tokens"].detach().float().contiguous()
    if tuple(fine_all.shape) != (988, 19, 20) or tuple(prefix_all.shape) != (
        988,
        15,
        77,
        200,
    ) or tuple(fine_manifest.get("feature_names", ())) != FINE_TEMPORAL_FEATURE_NAMES:
        raise ValueError("v12 frozen input shape/vocabulary changed")

    masks_all, duration_all = _build_temporal_masks(
        [event.global_t0_sec for event in union.events],
        [event.global_stop_sec for event in union.events],
    )

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
        raise ValueError("v12 verified target receipt/policy changed")
    target_batch = target.registry.target_batch(union.patient_ids, require_eligible=True)
    targets_all = target_batch.values.cpu()
    target_mask_all = target_batch.mask.cpu()
    complete = _complete_candidate_label_rows(target_mask_all)
    excluded = [
        union.patient_ids[index]
        for index in torch.nonzero(~complete, as_tuple=False).flatten().tolist()
    ]
    if excluded != [EXCLUDED_PARTIAL_REFERENCE_PATIENT]:
        raise ValueError(f"v12 incomplete reference roster changed: {excluded}")
    selected_patients = torch.nonzero(complete, as_tuple=False).flatten()
    if selected_patients.numel() != PRIMARY_PATIENT_COUNT:
        raise ValueError("v12 fixed 101-patient scope changed")

    event_patient_all = torch.tensor(union.event_patient_index, dtype=torch.long)
    event_keep = complete[event_patient_all]
    selected_event_rows = torch.nonzero(event_keep, as_tuple=False).flatten()
    if selected_event_rows.numel() != PRIMARY_EVENT_COUNT:
        raise ValueError("v12 fixed 984-event scope changed")
    old_to_new = torch.full((len(union.patient_ids),), -1, dtype=torch.long)
    old_to_new[selected_patients] = torch.arange(PRIMARY_PATIENT_COUNT)
    event_patient_index = old_to_new[event_patient_all[event_keep]]
    event_counts = torch.bincount(
        event_patient_index, minlength=PRIMARY_PATIENT_COUNT
    )

    # The fine branch contributes only this target-free artifact reliability.
    # No other fine feature enters the v12 localization scorer or transform.
    artifact_index = FINE_TEMPORAL_FEATURE_NAMES.index("artifact_burden_0_12s")
    reliability = (1.0 - fine_all[event_keep, :, artifact_index]).clamp(
        V12_RELIABILITY_FLOOR, 1.0
    )
    selected_prefix = prefix_all[event_keep]
    del fine_all, fine_payload, prefix_all, prefix_payload
    raw_node_time = restore_prefix_node_time(selected_prefix)
    del selected_prefix

    temporal_masks = _subset_masks(masks_all, selected_event_rows)
    duration_seconds = duration_all.index_select(0, selected_event_rows)
    node_time = baseline_difference_node_time(raw_node_time, temporal_masks)
    del raw_node_time
    patient_ids = tuple(
        union.patient_ids[index] for index in selected_patients.tolist()
    )
    event_ids = tuple(union_event_ids[index] for index in selected_event_rows.tolist())
    targets = targets_all.index_select(0, selected_patients)
    target_mask = target_mask_all.index_select(0, selected_patients)
    patient_folds = torch.tensor(union.patient_folds, dtype=torch.long).index_select(
        0, selected_patients
    )
    fold_patient_counts = tuple(
        torch.bincount(patient_folds, minlength=len(OUTER_FOLDS)).tolist()
    )
    fold_event_counts = tuple(
        torch.zeros(len(OUTER_FOLDS), dtype=torch.long)
        .scatter_add_(0, patient_folds, event_counts)
        .tolist()
    )
    if fold_patient_counts != EXPECTED_FOLD_PATIENT_COUNTS or (
        fold_event_counts != EXPECTED_FOLD_EVENT_COUNTS
    ):
        raise ValueError("v12 patient folds differ from v11.1 r2")
    if not (((targets == 1) & target_mask).any(dim=1)).all():
        raise ValueError("v12 every patient requires a DeepSOZ positive set")
    return V12Inputs(
        node_time=node_time,
        temporal_masks=temporal_masks,
        duration_seconds=duration_seconds,
        reliability=reliability.detach().contiguous(),
        event_patient_index=event_patient_index,
        event_ids=event_ids,
        patient_ids=patient_ids,
        targets=targets,
        target_mask=target_mask,
        patient_folds=patient_folds,
        event_counts=event_counts,
        union_manifest_sha256=union.manifest_sha256,
        target_receipt_sha256=target.receipt.receipt_sha256,
        target_artifact_sha256=target.receipt.target_artifact_sha256,
    )


def _load_pinned_reference(
    directory: Path,
    inputs: V12Inputs,
) -> PinnedV11Reference:
    manifest_path = directory / "manifest.json"
    prediction_path = directory / "oof_predictions.safetensors"
    if _file_sha(manifest_path) != EXPECTED_V11_1_MANIFEST_SHA256 or (
        _file_sha(prediction_path) != EXPECTED_V11_1_OOF_SHA256
    ):
        raise ValueError("v12 pinned v11.1 r2 comparator changed")
    manifest = _load_json_manifest(
        manifest_path, expected_sha=EXPECTED_V11_1_MANIFEST_SHA256
    )
    if tuple(str(value) for value in manifest.get("patient_ids", ())) != (
        inputs.patient_ids
    ):
        raise ValueError("v12/v11.1 patient IDs or order changed")
    if manifest.get("primary_patient_count") != PRIMARY_PATIENT_COUNT or (
        manifest.get("primary_event_count") != PRIMARY_EVENT_COUNT
    ):
        raise ValueError("v12 pinned v11.1 scope changed")
    payload = load_file(str(prediction_path), device="cpu")
    required = {
        "targets",
        "target_mask",
        "patient_folds",
        "patient_event_counts",
        "config.candidate_mask",
        "oof.full_frozen_labram_plus_fine",
        "oof.frozen_labram_only",
    }
    if not required.issubset(payload):
        raise ValueError("v12 pinned v11.1 prediction carrier is incomplete")
    checks = {
        "targets": torch.equal(payload["targets"], inputs.targets),
        "target_mask": torch.equal(payload["target_mask"], inputs.target_mask),
        "patient_folds": torch.equal(
            payload["patient_folds"], inputs.patient_folds
        ),
        "patient_event_counts": torch.equal(
            payload["patient_event_counts"], inputs.event_counts
        ),
        "candidate_mask": torch.equal(
            payload["config.candidate_mask"], V12_CANDIDATE_MASK
        ),
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"v12 pinned v11.1 carrier mismatch: {failed}")
    full = payload["oof.full_frozen_labram_plus_fine"].detach().float().contiguous()
    frozen = payload["oof.frozen_labram_only"].detach().float().contiguous()
    if tuple(full.shape) != (PRIMARY_PATIENT_COUNT, 19) or tuple(frozen.shape) != (
        PRIMARY_PATIENT_COUNT,
        19,
    ) or not torch.isfinite(full).all() or not torch.isfinite(frozen).all():
        raise ValueError("v12 pinned v11.1 logits are invalid")
    return PinnedV11Reference(
        full=full,
        frozen=frozen,
        manifest_sha256=EXPECTED_V11_1_MANIFEST_SHA256,
        predictions_sha256=EXPECTED_V11_1_OOF_SHA256,
    )


def _source_hashes() -> dict[str, str]:
    paths = {
        "runner_v12": Path(__file__).resolve(),
        "localizer_v12": ROOT / "src/soz/time_resolved_localizer_v12.py",
        "runner_v11": ROOT / "scripts/run_labram_fine_temporal_nested_oof_v11.py",
        "metrics_v11_1": ROOT
        / "scripts/run_labram_fine_temporal_nested_oof_v11_1.py",
        "development_union": ROOT / "src/soz/v11_development_union.py",
        "fine_temporal_evidence": ROOT / "src/soz/fine_temporal_evidence.py",
        "target_loader": ROOT / "src/soz/data/deepsoz_target_v2.py",
        "metrics": ROOT / "src/soz/metrics.py",
    }
    return {name: _file_sha(path) for name, path in paths.items()}


def _fold_strict(
    logits: torch.Tensor,
    inputs: V12Inputs,
) -> list[float]:
    values = []
    for fold in OUTER_FOLDS:
        indices = torch.nonzero(
            inputs.patient_folds == fold, as_tuple=False
        ).flatten()
        values.append(
            float(
                _evaluate(
                    logits.index_select(0, indices),
                    inputs.targets.index_select(0, indices),
                    inputs.target_mask.index_select(0, indices),
                )["top1"]["strict_accuracy"]
            )
        )
    return values


def _assess_go(
    metrics: Mapping[str, Mapping[str, object]],
    fold_strict: Mapping[str, Sequence[float]],
) -> tuple[bool, dict[str, bool]]:
    learned = metrics[LEARNED]
    reference = metrics[V11_FULL]
    learned_folds = tuple(float(value) for value in fold_strict[LEARNED])
    reference_folds = tuple(float(value) for value in fold_strict[V11_FULL])
    if len(learned_folds) != 5 or len(reference_folds) != 5:
        raise ValueError("v12 GO requires the same five outer folds as v11.1")
    checks = {
        "strict_nonlower_than_v11_1_full": learned["top1"]["strict_accuracy"]
        >= reference["top1"]["strict_accuracy"],
        "far_error_nonincreasing_vs_v11_1_full": learned["far_error_count"]
        <= reference["far_error_count"],
        "macro_ap_positive_vs_v11_1_full": learned["ranking"][
            "macro_average_precision"
        ]
        - reference["ranking"]["macro_average_precision"]
        > MACRO_AP_MINIMUM_GAIN,
        "four_of_five_fold_strict_nonlower_vs_v11_1_full": sum(
            left >= right for left, right in zip(learned_folds, reference_folds)
        )
        >= 4,
    }
    return all(checks.values()), {name: bool(value) for name, value in checks.items()}


def _assess_gate_mechanism_support(
    metrics: Mapping[str, Mapping[str, object]],
) -> tuple[bool, dict[str, bool]]:
    """Separate gate attribution from time-resolved representation recovery."""

    learned = metrics[LEARNED]
    uniform = metrics[UNIFORM]
    checks = {
        "strict_nonlower_than_uniform": learned["top1"]["strict_accuracy"]
        >= uniform["top1"]["strict_accuracy"],
        "far_error_nonincreasing_vs_uniform": learned["far_error_count"]
        <= uniform["far_error_count"],
        "macro_ap_positive_vs_uniform": learned["ranking"][
            "macro_average_precision"
        ]
        - uniform["ranking"]["macro_average_precision"]
        > MACRO_AP_MINIMUM_GAIN,
    }
    return all(checks.values()), {name: bool(value) for name, value in checks.items()}


def run(
    args: argparse.Namespace,
) -> tuple[Mapping[str, object], Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]]:
    if not EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("v12 protocol SHA has not been frozen")
    if _file_sha(PROTOCOL_PATH) != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("v12 protocol changed after freeze")
    started = time.monotonic()
    source_hashes_before = _source_hashes()
    inputs = _load_inputs(args)
    reference = _load_pinned_reference(args.v11_1_reference_directory, inputs)

    oof = {
        UNIFORM: torch.full((PRIMARY_PATIENT_COUNT, 19), torch.nan),
        LEARNED: torch.full((PRIMARY_PATIENT_COUNT, 19), torch.nan),
    }
    patient_audit_oof = {
        "uniform.dispersion": torch.full((PRIMARY_PATIENT_COUNT, 19), torch.nan),
        "uniform.reliability_sum": torch.full((PRIMARY_PATIENT_COUNT, 19), torch.nan),
        "learned.dispersion": torch.full((PRIMARY_PATIENT_COUNT, 19), torch.nan),
        "learned.reliability_sum": torch.full((PRIMARY_PATIENT_COUNT, 19), torch.nan),
    }
    event_oof = {
        "uniform.event_logits": torch.full((PRIMARY_EVENT_COUNT, 19), torch.nan),
        "uniform.event_log_prob": torch.full((PRIMARY_EVENT_COUNT, 19), torch.nan),
        "uniform.evidence_logits": torch.full((PRIMARY_EVENT_COUNT, 19), torch.nan),
        "learned.event_logits": torch.full((PRIMARY_EVENT_COUNT, 19), torch.nan),
        "learned.event_log_prob": torch.full((PRIMARY_EVENT_COUNT, 19), torch.nan),
        "learned.evidence_logits": torch.full((PRIMARY_EVENT_COUNT, 19), torch.nan),
        "learned.node_time_logits": torch.full(
            (PRIMARY_EVENT_COUNT, 19, V12_N_SECONDS), torch.nan
        ),
        "learned.global_trajectory": torch.full(
            (PRIMARY_EVENT_COUNT, V12_N_SECONDS), torch.nan
        ),
        "learned.onset_derivative": torch.full(
            (PRIMARY_EVENT_COUNT, V12_N_SECONDS), torch.nan
        ),
        "learned.gate_weights": torch.full(
            (PRIMARY_EVENT_COUNT, V12_N_SECONDS), torch.nan
        ),
        "uniform.gate_weights": torch.full(
            (PRIMARY_EVENT_COUNT, V12_N_SECONDS), torch.nan
        ),
        "learned.baseline_score": torch.full((PRIMARY_EVENT_COUNT,), torch.nan),
    }
    fold_results: list[dict[str, object]] = []
    fold_strict = {UNIFORM: [], LEARNED: []}
    outer_states: dict[str, torch.Tensor] = {
        "config.candidate_mask": V12_CANDIDATE_MASK.clone(),
        "config.l2": torch.tensor(L2, dtype=torch.float32),
    }

    for outer_fold in OUTER_FOLDS:
        train_patients = torch.nonzero(
            inputs.patient_folds != outer_fold, as_tuple=False
        ).flatten()
        held_patients = torch.nonzero(
            inputs.patient_folds == outer_fold, as_tuple=False
        ).flatten()
        train_events, train_event_patient = _subset_events(
            inputs.event_patient_index, train_patients
        )
        held_events, held_event_patient = _subset_events(
            inputs.event_patient_index, held_patients
        )
        transform = fit_node_feature_transform(
            inputs.node_time,
            inputs.event_patient_index,
            train_patients.tolist(),
            inputs.temporal_masks,
        )
        for name, value in transform.tensor_state().items():
            outer_states[f"outer{outer_fold}.transform.{name}"] = value

        train_masks = _subset_masks(inputs.temporal_masks, train_events)
        z_train = transform.apply(
            inputs.node_time.index_select(0, train_events), train_masks
        )
        train_reliability = inputs.reliability.index_select(0, train_events)
        train_targets = inputs.targets.index_select(0, train_patients)
        train_target_mask = inputs.target_mask.index_select(0, train_patients)
        prior = jeffreys_reference_prior_logits(train_targets, train_target_mask)
        if bool(torch.count_nonzero(prior[~V12_CANDIDATE_MASK])):
            raise RuntimeError("v12 PZ reference prior must remain finite zero")

        uniform = TimeResolvedNodeLocalizerV12(prior, gate_mode="uniform")
        if uniform.n_trainable_parameters != 16:
            raise RuntimeError("v12 uniform control must expose exactly 16 parameters")
        uniform_fit = _fit_localizer(
            uniform,
            z_train,
            train_masks,
            train_event_patient,
            train_reliability,
            train_targets,
            train_target_mask,
        )
        learned = _matched_learned_initialization(uniform, prior)
        if learned.n_trainable_parameters != 32:
            raise RuntimeError("v12 learned gate must expose exactly 32 parameters")
        step0 = _step0_parity(
            uniform,
            learned,
            z_train,
            train_masks,
            train_event_patient,
            train_reliability,
            int(train_patients.numel()),
        )
        learned_fit = _fit_localizer(
            learned,
            z_train,
            train_masks,
            train_event_patient,
            train_reliability,
            train_targets,
            train_target_mask,
        )
        for candidate, fit in ((UNIFORM, uniform_fit), (LEARNED, learned_fit)):
            for name, value in fit.state.items():
                outer_states[f"outer{outer_fold}.{candidate}.{name}"] = value
        fold_prefix = f"outer{outer_fold}."
        fold_state = {
            key.removeprefix(fold_prefix): value
            for key, value in outer_states.items()
            if key.startswith(fold_prefix)
        }
        state_sha_before_held = _state_sha(fold_state)

        # Held signals enter the transformed/model path only after both
        # candidates are completely fitted; held DeepSOZ targets are never
        # passed to either forward path.
        del z_train
        held_masks = _subset_masks(inputs.temporal_masks, held_events)
        z_held = transform.apply(
            inputs.node_time.index_select(0, held_events), held_masks
        )
        held_reliability = inputs.reliability.index_select(0, held_events)
        uniform.eval()
        learned.eval()
        with torch.no_grad():
            (
                uniform_event,
                uniform_event_log_prob,
                uniform_aggregation,
                uniform_patient,
            ) = _patient_logits(
                uniform,
                z_held,
                held_masks,
                held_event_patient,
                int(held_patients.numel()),
                held_reliability,
            )
            (
                learned_event,
                learned_event_log_prob,
                learned_aggregation,
                learned_patient,
            ) = _patient_logits(
                learned,
                z_held,
                held_masks,
                held_event_patient,
                int(held_patients.numel()),
                held_reliability,
            )
        if not torch.allclose(
            uniform_event.gate_weights,
            learned_event.uniform_gate_weights,
            atol=0.0,
            rtol=0.0,
        ):
            raise RuntimeError("v12 held uniform control changed between arms")
        if _state_sha(
            {
                **{
                    f"transform.{name}": value
                    for name, value in transform.tensor_state().items()
                },
                **{
                    f"{UNIFORM}.{name}": value
                    for name, value in _model_state(uniform).items()
                },
                **{
                    f"{LEARNED}.{name}": value
                    for name, value in _model_state(learned).items()
                },
            }
        ) != state_sha_before_held:
            raise RuntimeError("v12 held prediction mutated fold state")

        oof[UNIFORM].index_copy_(0, held_patients, uniform_patient.cpu())
        oof[LEARNED].index_copy_(0, held_patients, learned_patient.cpu())
        patient_audit_oof["uniform.dispersion"].index_copy_(
            0, held_patients, uniform_aggregation.dispersion.cpu()
        )
        patient_audit_oof["uniform.reliability_sum"].index_copy_(
            0, held_patients, uniform_aggregation.reliability_sum.cpu()
        )
        patient_audit_oof["learned.dispersion"].index_copy_(
            0, held_patients, learned_aggregation.dispersion.cpu()
        )
        patient_audit_oof["learned.reliability_sum"].index_copy_(
            0, held_patients, learned_aggregation.reliability_sum.cpu()
        )
        event_values = {
            "uniform.event_logits": uniform_event.event_logits,
            "uniform.event_log_prob": uniform_event_log_prob,
            "uniform.evidence_logits": uniform_event.evidence_logits,
            "learned.event_logits": learned_event.event_logits,
            "learned.event_log_prob": learned_event_log_prob,
            "learned.evidence_logits": learned_event.evidence_logits,
            "learned.node_time_logits": learned_event.node_time_logits,
            "learned.global_trajectory": learned_event.global_trajectory,
            "learned.onset_derivative": learned_event.onset_derivative,
            "learned.gate_weights": learned_event.gate_weights,
            "uniform.gate_weights": uniform_event.gate_weights,
            "learned.baseline_score": learned_event.baseline_score,
        }
        for name, value in event_values.items():
            event_oof[name].index_copy_(0, held_events, value.detach().cpu())

        held_targets = inputs.targets.index_select(0, held_patients)
        held_target_mask = inputs.target_mask.index_select(0, held_patients)
        held_metrics = {
            UNIFORM: _evaluate(uniform_patient, held_targets, held_target_mask),
            LEARNED: _evaluate(learned_patient, held_targets, held_target_mask),
        }
        for candidate in (UNIFORM, LEARNED):
            fold_strict[candidate].append(
                float(held_metrics[candidate]["top1"]["strict_accuracy"])
            )
        fold_results.append(
            {
                "outer_fold": outer_fold,
                "train_patient_count": int(train_patients.numel()),
                "held_patient_count": int(held_patients.numel()),
                "train_event_count": int(train_events.numel()),
                "held_event_count": int(held_events.numel()),
                "train_patient_ids": [inputs.patient_ids[i] for i in train_patients],
                "held_patient_ids": [inputs.patient_ids[i] for i in held_patients],
                "transform_train_event_count": transform.train_event_count,
                "uniform_fit": dict(uniform_fit.diagnostics),
                "learned_fit": dict(learned_fit.diagnostics),
                "matched_step0": step0,
                "held_metrics": held_metrics,
                "held_gate_audit": _gate_audit(learned_event),
                "state_sha256_before_and_after_held": state_sha_before_held,
            }
        )
        print(
            json.dumps(
                {
                    "outer_fold": outer_fold,
                    "status": "complete",
                    "uniform_strict": held_metrics[UNIFORM]["top1"][
                        "strict_accuracy"
                    ],
                    "learned_strict": held_metrics[LEARNED]["top1"][
                        "strict_accuracy"
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if any(not torch.isfinite(value).all() for value in oof.values()):
        raise RuntimeError("v12 patient OOF matrices are incomplete")
    if any(not torch.isfinite(value).all() for value in patient_audit_oof.values()):
        raise RuntimeError("v12 patient aggregation OOF audit matrices are incomplete")
    if any(not torch.isfinite(value).all() for value in event_oof.values()):
        raise RuntimeError("v12 event/gate OOF audit matrices are incomplete")

    all_logits = {
        UNIFORM: oof[UNIFORM],
        LEARNED: oof[LEARNED],
        V11_FULL: reference.full,
        V11_FROZEN: reference.frozen,
    }
    metrics = {
        name: _evaluate(value, inputs.targets, inputs.target_mask)
        for name, value in all_logits.items()
    }
    absolute = {
        name: _absolute_bootstrap(value, inputs.targets, inputs.target_mask)
        for name, value in all_logits.items()
    }
    paired = {
        baseline: _paired_bootstrap(
            oof[LEARNED],
            all_logits[baseline],
            inputs.targets,
            inputs.target_mask,
        )
        for baseline in (UNIFORM, V11_FULL, V11_FROZEN)
    }
    fold_strict[V11_FULL] = _fold_strict(reference.full, inputs)
    fold_strict[V11_FROZEN] = _fold_strict(reference.frozen, inputs)
    recovery_go, recovery_go_checks = _assess_go(metrics, fold_strict)
    gate_support, gate_support_checks = _assess_gate_mechanism_support(metrics)

    source_hashes_after = _source_hashes()
    if source_hashes_after != source_hashes_before:
        raise RuntimeError("v12 source files changed during execution")
    learned_gate = event_oof["learned.gate_weights"]
    valid_mask = inputs.temporal_masks.ictal_valid_mask
    manifest = {
        "schema_version": SCHEMA,
        "status": "completed_internal_developmental_patient_oof",
        "decision": (
            "RECOVERY_GO_GATE_MECHANISM_SUPPORTED_FREEZE_SEPARATE_REFIT_PROTOCOL"
            if recovery_go and gate_support
            else (
                "RECOVERY_GO_GATE_MECHANISM_NOT_SUPPORTED_DO_NOT_ATTRIBUTE_GAIN_TO_GATE"
                if recovery_go
                else "NO_GO_TIME_RESOLVED_RECOVERY_NOT_SUPPORTED"
            )
        ),
        "claim_boundary": {
            "internal_public_developmental_oof_only": True,
            "pretraining_exposed_downstream_label_patient_held_out_oof": True,
            "held_signal_claim_allowed": False,
            "public_confirmation": False,
            "external_validation": False,
            "deployment_checkpoint_created": False,
            "private_used": False,
            "clinical_deployment_allowed": False,
        },
        "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "source_file_sha256": source_hashes_after,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "safetensors": safetensors.__version__,
            "torch_num_threads": torch.get_num_threads(),
        },
        "foundation": {
            "backbone": "official_pretrained_LaBraM_Base_not_replaced",
            "trained_from_scratch": False,
            "foundation_trainable_parameters": 0,
            "frozen_carrier": "block9_one_second_node_tokens",
            "frozen_blocks": list(range(10)),
        },
        "method": {
            "schema": V12_TIME_RESOLVED_SCHEMA,
            "raw_token_shape_before_baseline_difference": [
                PRIMARY_EVENT_COUNT,
                19,
                60,
                200,
            ],
            "baseline_differenced_token_shape": [PRIMARY_EVENT_COUNT, 19, 60, 200],
            "transformed_token_shape": [PRIMARY_EVENT_COUNT, 19, 60, V12_NODE_PCA_DIM],
            "relative_time_axis_seconds": [-12, 48],
            "preictal_baseline_seconds": 12,
            "ictal_mask_rule": "full_one_second_token_contained_in_[global_t0,global_stop)_K=floor(duration)_capped_at_48s",
            "baseline_difference": "each_ictal_node_token_minus_same_channel_12s_preictal_mean",
            "node_scorer": "channel_shared_linear_16d_no_bias",
            "global_gate": "sigmoid_candidate_mean_trajectory_positive_first_difference_softplus_plus_floor",
            "gate_floor": V12_GATE_FLOOR,
            "artifact_reliability": "clip_one_minus_fine_artifact_burden_0_12s_to_[0.1,1]",
            "artifact_reliability_floor": V12_RELIABILITY_FLOOR,
            "patient_aggregation": "complete_event_bag_reliability_weighted_10_90_winsorized_mean_if_ge3",
            "winsor_bounds_materialized": False,
            "winsor_bounds_replay": "recompute_from_saved_event_log_prob_per_patient_channel_with_fixed_0.10_0.90_quantiles",
            "event_probability_contract": "fixed18_event_log_softmax_PZ_finite_zero_then_pool_then_fixed18_patient_log_softmax",
            "target_semantics": "DeepSOZ_clinician_reference_positive_set",
            "not_claimed": [
                "cortical_seizure_onset_time",
                "seizure_onset_zone_onset_time",
                "propagation_ground_truth",
            ],
        },
        "training": {
            "optimizer": "full_batch_deterministic_LBFGS",
            "lr": LBFGS_LR,
            "max_iter": LBFGS_MAX_ITER,
            "tolerance_grad": LBFGS_TOLERANCE_GRAD,
            "tolerance_change": LBFGS_TOLERANCE_CHANGE,
            "line_search": LBFGS_LINE_SEARCH,
            "l2": L2,
            "hyperparameter_scan": False,
            "early_stopping": False,
            "loss": "patient_equal_positive_set_mass_plus_fixed_l2",
            "learned_initialization": "copy_fitted_uniform_node_weight_and_zero_gate_weight",
            "matched_step0_float32_atol": STEP0_PARITY_ATOL,
        },
        "primary_patient_count": PRIMARY_PATIENT_COUNT,
        "primary_event_count": PRIMARY_EVENT_COUNT,
        "signal_carrier_channel_count": 19,
        "fixed_output_candidate_count": int(V12_CANDIDATE_MASK.sum()),
        "fixed_candidate_mask": V12_CANDIDATE_MASK.tolist(),
        "excluded_partial_reference_patients": [
            EXCLUDED_PARTIAL_REFERENCE_PATIENT
        ],
        "patient_ids": list(inputs.patient_ids),
        "event_ids": list(inputs.event_ids),
        "patient_folds": inputs.patient_folds.tolist(),
        "event_counts": inputs.event_counts.tolist(),
        "event_duration_seconds_summary": {
            "minimum": float(inputs.duration_seconds.min()),
            "median": float(inputs.duration_seconds.median()),
            "maximum": float(inputs.duration_seconds.max()),
            "capped_at_48_count": int((inputs.duration_seconds >= 48.0).sum()),
        },
        "valid_ictal_token_count_summary": {
            "minimum": int(valid_mask.sum(dim=1).min()),
            "median": int(valid_mask.sum(dim=1).median()),
            "maximum": int(valid_mask.sum(dim=1).max()),
        },
        "fold_results": fold_results,
        "metrics": metrics,
        "absolute_patient_bootstrap": absolute,
        "paired_learned_minus_comparators": paired,
        "fold_strict": fold_strict,
        "event_count_strata": _event_count_strata(
            all_logits,
            inputs.targets,
            inputs.target_mask,
            inputs.event_counts,
        ),
        "event_to_patient_consistency": {
            UNIFORM: _event_consistency(
                oof[UNIFORM],
                event_oof["uniform.event_log_prob"],
                inputs.event_patient_index,
                inputs.patient_ids,
            ),
            LEARNED: _event_consistency(
                oof[LEARNED],
                event_oof["learned.event_log_prob"],
                inputs.event_patient_index,
                inputs.patient_ids,
            ),
        },
        "recovery_go_checks_vs_v11_1_full": recovery_go_checks,
        "recovery_go": recovery_go,
        "recovery_decision": (
            "V12_RECOVERY_GO" if recovery_go else "V12_RECOVERY_NO_GO_STOP"
        ),
        "gate_mechanism_support_checks_vs_uniform": gate_support_checks,
        "gate_mechanism_support": gate_support,
        "gate_mechanism_decision": (
            "V12_GATE_MECHANISM_SUPPORT"
            if gate_support
            else "V12_GATE_MECHANISM_NOT_SUPPORTED"
        ),
        "scientific_strict_increment_supported": bool(
            paired[V11_FULL]["strict"]["ci95"][0] > 0.0
        ),
        "global_gate_oof_audit": {
            "mean_l1_distance_from_uniform": float(
                (
                    learned_gate - event_oof["uniform.gate_weights"]
                ).abs().sum(dim=1).mean()
            ),
            "gate_sum_max_abs_error": float(
                (learned_gate.sum(dim=1) - 1.0).abs().max()
            ),
            "post_stop_mass_max": float(
                learned_gate.masked_select(~valid_mask).abs().max()
                if bool((~valid_mask).any())
                else 0.0
            ),
        },
        "lineage": {
            "union_manifest_sha256": inputs.union_manifest_sha256,
            "fine_manifest_sha256": EXPECTED_FINE_MANIFEST_SHA256,
            "fine_tensor_sha256": EXPECTED_FINE_TENSOR_FILE_SHA256,
            "prefix_manifest_sha256": EXPECTED_PREFIX_MANIFEST_SHA256,
            "prefix_tensor_sha256": EXPECTED_PREFIX_TENSOR_FILE_SHA256,
            "target_receipt_sha256": inputs.target_receipt_sha256,
            "target_artifact_sha256": inputs.target_artifact_sha256,
            "v11_1_manifest_sha256": reference.manifest_sha256,
            "v11_1_oof_sha256": reference.predictions_sha256,
        },
        "resource_usage": {"wall_time_seconds": time.monotonic() - started},
        "access_receipt": {
            "patient_258_excluded_before_all_fit": True,
            "outer_transform_fit_on_train_patient_events_only": True,
            "held_targets_not_passed_to_prediction": True,
            "fine_features_used_only_for_target_free_artifact_reliability": True,
            "tusz_global_stop_used_only_for_true_stop_temporal_mask": True,
            "private_eeg_loaded": False,
            "private_target_values_loaded": False,
            "private_forward_count": 0,
            "llm_used_as_soz_predictor": False,
        },
    }
    tensors = {
        f"oof.{UNIFORM}": _masked_scores_for_publish(oof[UNIFORM]),
        f"oof.{LEARNED}": _masked_scores_for_publish(oof[LEARNED]),
        f"reference.{V11_FULL}": reference.full,
        f"reference.{V11_FROZEN}": reference.frozen,
        **{f"event_oof.{name}": value for name, value in event_oof.items()},
        **{
            f"patient_oof.{name}": value
            for name, value in patient_audit_oof.items()
        },
        "targets": inputs.targets,
        "target_mask": inputs.target_mask,
        "patient_folds": inputs.patient_folds,
        "patient_event_counts": inputs.event_counts,
        "event_patient_index": inputs.event_patient_index,
        "event_duration_seconds": inputs.duration_seconds,
        "event_artifact_reliability": inputs.reliability,
        "time.preictal_baseline_mask": inputs.temporal_masks.preictal_baseline_mask,
        "time.ictal_valid_mask": inputs.temporal_masks.ictal_valid_mask,
        "config.candidate_mask": V12_CANDIDATE_MASK.clone(),
    }
    return manifest, tensors, outer_states


def _publish(
    output_directory: Path,
    manifest: Mapping[str, object],
    tensors: Mapping[str, torch.Tensor],
    outer_states: Mapping[str, torch.Tensor],
) -> Path:
    if not torch.equal(
        outer_states.get("config.candidate_mask"), V12_CANDIDATE_MASK
    ):
        raise ValueError("v12 outer states lost the fixed candidate mask")
    target = Path(os.path.abspath(output_directory))
    if target.exists():
        raise FileExistsError(target)
    if not target.parent.is_dir():
        raise FileNotFoundError(target.parent)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        prediction_path = staging / "oof_predictions.safetensors"
        state_path = staging / "outer_fold_states.safetensors"
        save_file(dict(tensors), str(prediction_path))
        save_file(dict(outer_states), str(state_path))
        completed = dict(manifest)
        completed["files"] = {
            prediction_path.name: {
                "sha256": _file_sha(prediction_path),
                "size_bytes": prediction_path.stat().st_size,
            },
            state_path.name: {
                "sha256": _file_sha(state_path),
                "size_bytes": state_path.stat().st_size,
            },
        }
        (staging / "manifest.json").write_bytes(
            _canonical_bytes(completed, newline=True)
        )
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
    parser.add_argument(
        "--v11-1-reference-directory",
        type=Path,
        default=DEFAULT_V11_1_REFERENCE,
    )
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    manifest, tensors, outer_states = run(args)
    path = _publish(args.output_directory, manifest, tensors, outer_states)
    completed = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    learned = completed["metrics"][LEARNED]
    print(
        json.dumps(
            {
                "status": completed["status"],
                "decision": completed["decision"],
                "path": str(path),
                "manifest_sha256": _file_sha(path / "manifest.json"),
                "learned_strict": learned["top1"]["strict_accuracy"],
                "learned_relaxed": learned["top1"]["relaxed_accuracy"],
                "learned_macro_ap": learned["ranking"][
                    "macro_average_precision"
                ],
                "learned_far_errors": learned["far_error_count"],
                "private_used": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
