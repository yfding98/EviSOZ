#!/usr/bin/env python3
"""Run frozen-LaBraM DeepSOZ-style temporal-attention developmental OOF.

This runner is intentionally a single fixed experiment.  It trains a
source-native seizure detector inside each outer training fold, freezes that
detector, and then compares detection-attention against a matched uniform-time
control for patient-level SOZ localization.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Mapping, Sequence

import numpy as np
from safetensors.torch import load_file, save_file
from sklearn.metrics import average_precision_score, roc_auc_score
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_labram_fine_temporal_nested_oof_v11 import (  # noqa: E402
    DEFAULT_PREFIX,
    DEFAULT_SOURCE,
    DEFAULT_SPLIT,
    DEFAULT_TARGET,
    DEFAULT_UNION,
    EXPECTED_SOURCE_SHA256,
    EXPECTED_SPLIT_SHA256,
    EXPECTED_TARGET_ARTIFACT_SHA256,
    EXPECTED_TARGET_README_SHA256,
    EXPECTED_TARGET_RECEIPT_SHA256,
    EXPECTED_TARGET_SUMMARY_SHA256,
    OUTER_FOLDS,
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
from src.soz.labram_deepsoz_temporal_attention import (  # noqa: E402
    DETECTOR_HIDDEN,
    FrozenLaBraMTemporalDetector,
    N_SECONDS,
    SharedChannelSOZHead,
    aggregate_patient_probabilities,
    build_detection_targets,
    masked_patient_bce_l1,
    pool_event_features,
    weighted_detection_loss,
)
from src.soz.time_resolved_localizer_v12 import restore_prefix_node_time  # noqa: E402
from src.soz.v11_development_union import (  # noqa: E402
    EXPECTED_PUBLIC_DEVELOPMENT_UNION_MANIFEST_SHA256,
    load_public_development_union,
)
from src.soz.v11_reasoner import (  # noqa: E402
    V11_CANDIDATE_MASK,
    jeffreys_reference_prior_logits,
)


SCHEMA = "soz_labram_deepsoz_temporal_attention_oof_v1"
PROTOCOL_PATH = (
    ROOT
    / "research/02_method/"
    "labram_deepsoz_temporal_attention_recovery_protocol_v1_20260812_zh.md"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/labram_deepsoz_temporal_attention_oof_v1_20260812"
)
DEFAULT_V11_REFERENCE = (
    ROOT / "outputs/labram_fine_temporal_nested_oof_v11_1_20260811_r2"
)

PRIMARY_PATIENT_COUNT = 101
PRIMARY_EVENT_COUNT = 984
EXCLUDED_PARTIAL_REFERENCE_PATIENT = "258"
EXPECTED_FOLD_PATIENT_COUNTS = (20, 21, 20, 21, 19)
EXPECTED_FOLD_EVENT_COUNTS = (197, 198, 197, 198, 194)

LEARNED = "deepsoz_detection_attention"
UNIFORM = "matched_uniform_time"
V11_FULL = "v11_1_full_frozen_labram_plus_fine"
V11_FROZEN = "v11_1_frozen_labram_only"
PREVALENCE = "v11_1_prevalence_only"

SEED = 20260812
DETECTION_EPOCHS = 30
LOCALIZATION_EPOCHS = 30
EVENT_BATCH_SIZE = 16
DETECTION_TRAIN_CROP_SECONDS = 48
DETECTION_TRAIN_MAX_START = N_SECONDS - DETECTION_TRAIN_CROP_SECONDS
DETECTION_LR = 1.0e-4
LOCALIZATION_LR = 1.0e-4
DETECTION_DROPOUT = 0.15
GRADIENT_CLIP = 1.0
L1_WEIGHT = 0.1


@dataclass(frozen=True)
class Inputs:
    node_time: torch.Tensor
    detection_targets: torch.Tensor
    event_patient_index: torch.Tensor
    patient_ids: tuple[str, ...]
    event_ids: tuple[str, ...]
    targets: torch.Tensor
    target_mask: torch.Tensor
    patient_folds: torch.Tensor
    event_counts: torch.Tensor
    durations: torch.Tensor

    def __post_init__(self) -> None:
        if tuple(self.node_time.shape) != (PRIMARY_EVENT_COUNT, 19, 60, 200):
            raise ValueError("LaBraM node-time carrier must be [984,19,60,200]")
        if self.node_time.requires_grad or not torch.isfinite(self.node_time).all():
            raise ValueError("LaBraM node-time carrier must be finite and detached")
        if tuple(self.detection_targets.shape) != (PRIMARY_EVENT_COUNT, 60) or (
            self.detection_targets.dtype != torch.long
        ):
            raise TypeError("detection targets must be long [984,60]")
        if tuple(self.event_patient_index.shape) != (PRIMARY_EVENT_COUNT,) or (
            self.event_patient_index.dtype != torch.long
        ):
            raise TypeError("event-patient index must be long [984]")
        if len(self.patient_ids) != PRIMARY_PATIENT_COUNT or len(set(self.patient_ids)) != (
            PRIMARY_PATIENT_COUNT
        ):
            raise ValueError("patient roster must contain 101 unique patients")
        if len(self.event_ids) != PRIMARY_EVENT_COUNT or len(set(self.event_ids)) != (
            PRIMARY_EVENT_COUNT
        ):
            raise ValueError("event roster must contain 984 unique events")
        for value, name in (
            (self.targets, "targets"),
            (self.target_mask, "target mask"),
        ):
            if tuple(value.shape) != (PRIMARY_PATIENT_COUNT, 19):
                raise ValueError(f"{name} must have shape [101,19]")
        if self.target_mask.dtype != torch.bool or not torch.equal(
            self.target_mask,
            V11_CANDIDATE_MASK.unsqueeze(0).expand_as(self.target_mask),
        ):
            raise ValueError("primary target mask must be the fixed 18-candidate mask")
        if tuple(self.patient_folds.shape) != (PRIMARY_PATIENT_COUNT,) or (
            self.patient_folds.dtype != torch.long
        ):
            raise TypeError("patient folds must be long [101]")
        if tuple(self.event_counts.shape) != (PRIMARY_PATIENT_COUNT,) or (
            self.event_counts.dtype != torch.long
        ):
            raise TypeError("event counts must be long [101]")
        if int(self.event_counts.sum()) != PRIMARY_EVENT_COUNT:
            raise ValueError("event counts must cover all 984 events")
        if tuple(self.durations.shape) != (PRIMARY_EVENT_COUNT,) or not torch.isfinite(
            self.durations
        ).all():
            raise ValueError("durations must be finite [984]")


@dataclass(frozen=True)
class Comparator:
    full: torch.Tensor
    frozen: torch.Tensor
    prevalence: torch.Tensor


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _read_json(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return payload


def _load_inputs(args: argparse.Namespace) -> Inputs:
    union = load_public_development_union(
        args.union_directory,
        expected_manifest_sha256=EXPECTED_PUBLIC_DEVELOPMENT_UNION_MANIFEST_SHA256,
    )
    prefix_manifest = _read_json(args.prefix_directory / "manifest.json")
    if prefix_manifest.get("event_count") != 988 or (
        prefix_manifest.get("patient_count") != 102
    ):
        raise ValueError("frozen LaBraM prefix scope changed")
    union_event_ids = tuple(event.event_id for event in union.events)
    if tuple(str(value) for value in prefix_manifest.get("event_ids", ())) != (
        union_event_ids
    ):
        raise ValueError("prefix event order differs from the public union")
    access = prefix_manifest.get("access_receipt")
    if not isinstance(access, Mapping) or any(
        access.get(field) is not False
        for field in (
            "deepsoz_target_values_loaded",
            "private_eeg_loaded",
            "private_target_values_loaded",
            "historical_prediction_artifacts_loaded",
        )
    ):
        raise ValueError("prefix is not a target/private-free frozen carrier")

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
        raise ValueError("verified DeepSOZ target receipt changed")
    batch = target.registry.target_batch(union.patient_ids, require_eligible=True)
    targets_all = batch.values.cpu()
    mask_all = batch.mask.cpu()
    complete = _complete_candidate_label_rows(mask_all)
    excluded = [
        union.patient_ids[index]
        for index in torch.nonzero(~complete, as_tuple=False).flatten().tolist()
    ]
    if excluded != [EXCLUDED_PARTIAL_REFERENCE_PATIENT]:
        raise ValueError(f"unexpected incomplete target roster: {excluded}")
    selected_patients = torch.nonzero(complete, as_tuple=False).flatten()

    event_patient_all = torch.tensor(union.event_patient_index, dtype=torch.long)
    event_keep = complete[event_patient_all]
    selected_events = torch.nonzero(event_keep, as_tuple=False).flatten()
    if selected_patients.numel() != PRIMARY_PATIENT_COUNT or (
        selected_events.numel() != PRIMARY_EVENT_COUNT
    ):
        raise ValueError("fixed 101-patient/984-event scope changed")
    old_to_new = torch.full((len(union.patient_ids),), -1, dtype=torch.long)
    old_to_new[selected_patients] = torch.arange(PRIMARY_PATIENT_COUNT)
    event_patient_index = old_to_new[event_patient_all[event_keep]]
    event_counts = torch.bincount(
        event_patient_index, minlength=PRIMARY_PATIENT_COUNT
    )

    prefix_file = args.prefix_directory / str(prefix_manifest["tensor_file"])
    prefix_payload = load_file(str(prefix_file), device="cpu")
    prefix_all = prefix_payload["prefix_tokens"].detach()
    if tuple(prefix_all.shape) != (988, 15, 77, 200):
        raise ValueError("frozen LaBraM prefix tensor shape changed")
    selected_prefix = prefix_all.index_select(0, selected_events).contiguous()
    del prefix_all, prefix_payload
    node_time = restore_prefix_node_time(selected_prefix)
    del selected_prefix

    durations_all = torch.tensor(
        [event.global_stop_sec - event.global_t0_sec for event in union.events],
        dtype=torch.float32,
    )
    durations = durations_all.index_select(0, selected_events)
    detection_targets = build_detection_targets(durations)
    patient_ids = tuple(
        union.patient_ids[index] for index in selected_patients.tolist()
    )
    event_ids = tuple(union_event_ids[index] for index in selected_events.tolist())
    targets = targets_all.index_select(0, selected_patients)
    target_mask = mask_all.index_select(0, selected_patients)
    patient_folds = torch.tensor(union.patient_folds, dtype=torch.long).index_select(
        0, selected_patients
    )
    patient_fold_counts = tuple(
        torch.bincount(patient_folds, minlength=5).tolist()
    )
    event_fold_counts = tuple(
        torch.zeros(5, dtype=torch.long)
        .scatter_add_(0, patient_folds, event_counts)
        .tolist()
    )
    if patient_fold_counts != EXPECTED_FOLD_PATIENT_COUNTS or (
        event_fold_counts != EXPECTED_FOLD_EVENT_COUNTS
    ):
        raise ValueError("outer patient folds differ from the frozen v11.1 scope")
    if not (((targets == 1) & target_mask).any(dim=1)).all():
        raise ValueError("every patient must have an observed DeepSOZ positive")
    return Inputs(
        node_time=node_time,
        detection_targets=detection_targets,
        event_patient_index=event_patient_index,
        patient_ids=patient_ids,
        event_ids=event_ids,
        targets=targets,
        target_mask=target_mask,
        patient_folds=patient_folds,
        event_counts=event_counts,
        durations=durations,
    )


def _load_comparator(directory: Path, inputs: Inputs) -> Comparator:
    path = directory / "oof_predictions.safetensors"
    payload = load_file(str(path), device="cpu")
    required = {
        "targets",
        "target_mask",
        "patient_folds",
        "patient_event_counts",
        "config.candidate_mask",
        "oof.full_frozen_labram_plus_fine",
        "oof.frozen_labram_only",
        "oof.prevalence_only",
    }
    if not required <= set(payload):
        raise ValueError("v11.1 comparator carrier is incomplete")
    checks = {
        "targets": torch.equal(payload["targets"], inputs.targets),
        "target_mask": torch.equal(payload["target_mask"], inputs.target_mask),
        "patient_folds": torch.equal(payload["patient_folds"], inputs.patient_folds),
        "event_counts": torch.equal(
            payload["patient_event_counts"], inputs.event_counts
        ),
        "candidate_mask": torch.equal(
            payload["config.candidate_mask"], V11_CANDIDATE_MASK
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"v11.1 comparator mismatch: {failed}")
    return Comparator(
        full=payload["oof.full_frozen_labram_plus_fine"].float().contiguous(),
        frozen=payload["oof.frozen_labram_only"].float().contiguous(),
        prevalence=payload["oof.prevalence_only"].float().contiguous(),
    )


def _patient_event_rows(
    event_patient_index: torch.Tensor,
    patient_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if patient_indices.ndim != 1 or patient_indices.dtype != torch.long or (
        patient_indices.numel() < 1
    ):
        raise TypeError("patient_indices must be non-empty long [P]")
    selected = torch.zeros(
        int(event_patient_index.max()) + 1,
        dtype=torch.bool,
    )
    selected[patient_indices] = True
    event_rows = torch.nonzero(
        selected[event_patient_index], as_tuple=False
    ).flatten()
    old_to_new = torch.full((selected.numel(),), -1, dtype=torch.long)
    old_to_new[patient_indices] = torch.arange(patient_indices.numel())
    local_patient = old_to_new[event_patient_index.index_select(0, event_rows)]
    if int(local_patient.min()) != 0 or int(local_patient.max()) != (
        patient_indices.numel() - 1
    ):
        raise RuntimeError("patient event subsetting lost a complete event bag")
    return event_rows, local_patient


def _train_detector(
    inputs: Inputs,
    train_events: torch.Tensor,
    *,
    device: torch.device,
    seed: int,
) -> tuple[FrozenLaBraMTemporalDetector, list[Mapping[str, float]]]:
    _set_seed(seed)
    model = FrozenLaBraMTemporalDetector(
        hidden_dim=DETECTOR_HIDDEN,
        dropout=DETECTION_DROPOUT,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=DETECTION_LR)
    generator = torch.Generator().manual_seed(seed)
    history: list[Mapping[str, float]] = []
    for epoch in range(DETECTION_EPOCHS):
        order = train_events.index_select(
            0, torch.randperm(train_events.numel(), generator=generator)
        )
        weighted_loss = 0.0
        seen = 0
        model.train()
        for start in range(0, order.numel(), EVENT_BATCH_SIZE):
            rows = order[start : start + EVENT_BATCH_SIZE]
            crop_starts = torch.randint(
                0,
                DETECTION_TRAIN_MAX_START + 1,
                (rows.numel(),),
                generator=generator,
            )
            x = torch.stack(
                [
                    inputs.node_time[int(row), :, int(crop) : int(crop) + DETECTION_TRAIN_CROP_SECONDS]
                    for row, crop in zip(rows, crop_starts)
                ]
            ).to(device)
            y = torch.stack(
                [
                    inputs.detection_targets[
                        int(row), int(crop) : int(crop) + DETECTION_TRAIN_CROP_SECONDS
                    ]
                    for row, crop in zip(rows, crop_starts)
                ]
            ).to(device)
            output = model(x)
            loss = weighted_detection_loss(output.detection_logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
            n = int(rows.numel())
            weighted_loss += float(loss.detach()) * n
            seen += n
        history.append(
            {
                "epoch": float(epoch + 1),
                "weighted_cross_entropy": weighted_loss / max(seen, 1),
            }
        )
    model.eval()
    return model, history


@torch.no_grad()
def _materialize_fold_features(
    model: FrozenLaBraMTemporalDetector,
    node_time: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    events = int(node_time.shape[0])
    learned = torch.empty((events, 19, 200), dtype=torch.float32)
    uniform = torch.empty_like(learned)
    detection_probability = torch.empty((events, N_SECONDS), dtype=torch.float32)
    attention = torch.empty_like(detection_probability)
    model.eval()
    for start in range(0, events, EVENT_BATCH_SIZE):
        stop = min(start + EVENT_BATCH_SIZE, events)
        x = node_time[start:stop].to(device)
        output = model(x)
        learned[start:stop] = pool_event_features(
            output.normalized_node_time, output.attention
        ).cpu()
        uniform[start:stop] = output.normalized_node_time.mean(dim=2).cpu()
        detection_probability[start:stop] = torch.softmax(
            output.detection_logits, dim=-1
        )[..., 1].cpu()
        attention[start:stop] = output.attention.cpu()
    return learned, uniform, detection_probability, attention


@torch.no_grad()
def _shortcut_detection_probabilities(
    model: FrozenLaBraMTemporalDetector,
    held_node_time: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Audit absolute-position shortcuts without fitting another model."""

    events = int(held_node_time.shape[0])
    zero_probability = torch.empty((events, N_SECONDS), dtype=torch.float32)
    reversed_probability = torch.empty_like(zero_probability)
    model.eval()
    for start in range(0, events, EVENT_BATCH_SIZE):
        stop = min(start + EVENT_BATCH_SIZE, events)
        x = held_node_time[start:stop].to(device)
        zero_output = model(torch.zeros_like(x))
        reversed_output = model(x.flip(dims=(2,)))
        zero_probability[start:stop] = torch.softmax(
            zero_output.detection_logits, dim=-1
        )[..., 1].cpu()
        reversed_probability[start:stop] = torch.softmax(
            reversed_output.detection_logits, dim=-1
        )[..., 1].cpu()
    return zero_probability, reversed_probability


def _train_localizer(
    event_features: torch.Tensor,
    inputs: Inputs,
    train_patients: torch.Tensor,
    prior: torch.Tensor,
    *,
    device: torch.device,
    seed: int,
) -> tuple[SharedChannelSOZHead, list[Mapping[str, float]]]:
    _set_seed(seed)
    head = SharedChannelSOZHead(prior).to(device)
    if head.n_trainable_parameters != 200:
        raise RuntimeError("shared channel head must expose exactly 200 parameters")
    optimizer = torch.optim.Adam(head.parameters(), lr=LOCALIZATION_LR)
    generator = torch.Generator().manual_seed(seed)
    history: list[Mapping[str, float]] = []
    patient_event_rows = {
        int(patient): torch.nonzero(
            inputs.event_patient_index == int(patient), as_tuple=False
        ).flatten()
        for patient in train_patients.tolist()
    }
    for epoch in range(LOCALIZATION_EPOCHS):
        order = train_patients.index_select(
            0, torch.randperm(train_patients.numel(), generator=generator)
        )
        total_sum = bce_sum = sparsity_sum = 0.0
        head.train()
        for patient_tensor in order:
            patient = int(patient_tensor)
            rows = patient_event_rows[patient]
            features = event_features.index_select(0, rows).to(device)
            event_logits = head(features)
            probability = torch.sigmoid(event_logits).mean(dim=0, keepdim=True)
            target = inputs.targets[patient : patient + 1].to(device)
            mask = inputs.target_mask[patient : patient + 1].to(device)
            total, bce, sparsity = masked_patient_bce_l1(
                probability,
                target,
                mask,
                l1_weight=L1_WEIGHT,
            )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), GRADIENT_CLIP)
            optimizer.step()
            total_sum += float(total.detach())
            bce_sum += float(bce.detach())
            sparsity_sum += float(sparsity.detach())
        denominator = max(int(train_patients.numel()), 1)
        history.append(
            {
                "epoch": float(epoch + 1),
                "total": total_sum / denominator,
                "benchmark_bce": bce_sum / denominator,
                "probability_l1": sparsity_sum / denominator,
            }
        )
    head.eval()
    return head, history


@torch.no_grad()
def _predict_event_logits(
    head: SharedChannelSOZHead,
    features: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    output = torch.empty((features.shape[0], 19), dtype=torch.float32)
    for start in range(0, features.shape[0], 128):
        stop = min(start + 128, features.shape[0])
        output[start:stop] = head(features[start:stop].to(device)).cpu()
    return output


def _probability_to_logit(probability: torch.Tensor) -> torch.Tensor:
    if not torch.isfinite(probability).all() or bool(
        ((probability < 0) | (probability > 1)).any()
    ):
        raise ValueError("patient probability must be finite in [0,1]")
    return torch.logit(probability.clamp(1.0e-6, 1.0 - 1.0e-6))


def _detection_metrics(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    attention: torch.Tensor,
) -> Mapping[str, float]:
    if tuple(probabilities.shape) != tuple(targets.shape) or (
        tuple(attention.shape) != tuple(targets.shape)
    ):
        raise ValueError("detection probability/target/attention must align")
    p = probabilities.clamp(1.0e-6, 1.0 - 1.0e-6)
    y = targets.float()
    flat_y = y.reshape(-1).numpy()
    flat_p = p.reshape(-1).numpy()
    peak = attention.argmax(dim=1)
    peak_in_ictal = y.gather(1, peak.unsqueeze(1)).squeeze(1)
    entropy = -(attention.clamp_min(1.0e-12) * attention.clamp_min(1.0e-12).log()).sum(
        dim=1
    ) / math.log(N_SECONDS)
    return {
        "bce": float(-(y * p.log() + (1.0 - y) * (1.0 - p).log()).mean()),
        "brier": float((p - y).square().mean()),
        "auroc": float(roc_auc_score(flat_y, flat_p)),
        "average_precision": float(average_precision_score(flat_y, flat_p)),
        "accuracy_at_0_5": float(((p >= 0.5) == y.bool()).float().mean()),
        "attention_mass_inside_tusz_ictal_interval": float((attention * y).sum(1).mean()),
        "attention_peak_inside_tusz_ictal_interval_rate": float(peak_in_ictal.mean()),
        "normalized_attention_entropy": float(entropy.mean()),
    }


def _probability_only_detection_metrics(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
) -> Mapping[str, float]:
    if tuple(probabilities.shape) != tuple(targets.shape):
        raise ValueError("shortcut detection probability and target must align")
    p = probabilities.clamp(1.0e-6, 1.0 - 1.0e-6)
    y = targets.float()
    flat_y = y.reshape(-1).numpy()
    flat_p = p.reshape(-1).numpy()
    return {
        "bce": float(-(y * p.log() + (1.0 - y) * (1.0 - p).log()).mean()),
        "brier": float((p - y).square().mean()),
        "auroc": float(roc_auc_score(flat_y, flat_p)),
        "average_precision": float(average_precision_score(flat_y, flat_p)),
        "accuracy_at_0_5": float(((p >= 0.5) == y.bool()).float().mean()),
    }


def _fold_strict(logits: torch.Tensor, inputs: Inputs) -> list[float]:
    result = []
    for fold in OUTER_FOLDS:
        rows = torch.nonzero(inputs.patient_folds == fold, as_tuple=False).flatten()
        result.append(
            float(
                _evaluate(
                    logits.index_select(0, rows),
                    inputs.targets.index_select(0, rows),
                    inputs.target_mask.index_select(0, rows),
                )["top1"]["strict_accuracy"]
            )
        )
    return result


def _promotion_gate(
    metrics: Mapping[str, Mapping[str, object]],
    detection: Mapping[str, float],
    zero_input: Mapping[str, float],
    time_reversal: Mapping[str, float],
) -> tuple[bool, Mapping[str, bool]]:
    learned = metrics[LEARNED]
    uniform = metrics[UNIFORM]
    checks = {
        "held_detection_auroc_ge_0_80": detection["auroc"] >= 0.80,
        "auroc_gain_over_zero_input_ge_0_02": detection["auroc"]
        - zero_input["auroc"]
        >= 0.02,
        "auroc_gain_over_time_reversal_ge_0_02": detection["auroc"]
        - time_reversal["auroc"]
        >= 0.02,
        "strict_count_gt_52": round(
            learned["top1"]["strict_accuracy"] * PRIMARY_PATIENT_COUNT
        )
        > 52,
        "one_hop_count_ge_78": round(
            learned["top1"]["relaxed_accuracy"] * PRIMARY_PATIENT_COUNT
        )
        >= 78,
        "macro_ap_gt_v11_1_full": learned["ranking"]["macro_average_precision"]
        > 0.53142035,
        "far_error_count_le_23": learned["far_error_count"] <= 23,
        "strict_nonlower_than_uniform": learned["top1"]["strict_accuracy"]
        >= uniform["top1"]["strict_accuracy"],
        "macro_ap_nonlower_than_uniform": learned["ranking"][
            "macro_average_precision"
        ]
        >= uniform["ranking"]["macro_average_precision"],
    }
    return all(checks.values()), {name: bool(value) for name, value in checks.items()}


def _state_to_cpu(module: torch.nn.Module) -> Mapping[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def run(args: argparse.Namespace) -> Mapping[str, object]:
    if not PROTOCOL_PATH.is_file():
        raise FileNotFoundError(PROTOCOL_PATH)
    started = time.monotonic()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    inputs = _load_inputs(args)
    comparator = _load_comparator(args.v11_reference_directory, inputs)

    oof_patient = {
        LEARNED: torch.full((PRIMARY_PATIENT_COUNT, 19), torch.nan),
        UNIFORM: torch.full((PRIMARY_PATIENT_COUNT, 19), torch.nan),
    }
    oof_event = {
        LEARNED: torch.full((PRIMARY_EVENT_COUNT, 19), torch.nan),
        UNIFORM: torch.full((PRIMARY_EVENT_COUNT, 19), torch.nan),
    }
    detection_probability = torch.full((PRIMARY_EVENT_COUNT, N_SECONDS), torch.nan)
    zero_input_detection_probability = torch.full_like(
        detection_probability, torch.nan
    )
    time_reversal_detection_probability = torch.full_like(
        detection_probability, torch.nan
    )
    temporal_attention = torch.full_like(detection_probability, torch.nan)
    fold_results: list[Mapping[str, object]] = []
    fold_states: dict[str, torch.Tensor] = {}

    for outer_fold in OUTER_FOLDS:
        train_patients = torch.nonzero(
            inputs.patient_folds != outer_fold, as_tuple=False
        ).flatten()
        held_patients = torch.nonzero(
            inputs.patient_folds == outer_fold, as_tuple=False
        ).flatten()
        train_events, _ = _patient_event_rows(
            inputs.event_patient_index, train_patients
        )
        held_events, held_event_patient = _patient_event_rows(
            inputs.event_patient_index, held_patients
        )
        detector, detection_history = _train_detector(
            inputs,
            train_events,
            device=device,
            seed=SEED + outer_fold,
        )
        for parameter in detector.parameters():
            parameter.requires_grad_(False)
        if any(parameter.requires_grad for parameter in detector.parameters()):
            raise RuntimeError("detector must be frozen before SOZ localization")

        learned_features, uniform_features, fold_detection, fold_attention = (
            _materialize_fold_features(detector, inputs.node_time, device=device)
        )
        zero_probability, reversed_probability = _shortcut_detection_probabilities(
            detector,
            inputs.node_time.index_select(0, held_events),
            device=device,
        )
        train_targets = inputs.targets.index_select(0, train_patients)
        train_mask = inputs.target_mask.index_select(0, train_patients)
        prior = jeffreys_reference_prior_logits(train_targets, train_mask)
        learned_head, learned_history = _train_localizer(
            learned_features,
            inputs,
            train_patients,
            prior,
            device=device,
            seed=SEED + 100 + outer_fold,
        )
        uniform_head, uniform_history = _train_localizer(
            uniform_features,
            inputs,
            train_patients,
            prior,
            device=device,
            seed=SEED + 100 + outer_fold,
        )
        learned_event_logits = _predict_event_logits(
            learned_head, learned_features, device=device
        )
        uniform_event_logits = _predict_event_logits(
            uniform_head, uniform_features, device=device
        )
        learned_patient_probability = aggregate_patient_probabilities(
            learned_event_logits.index_select(0, held_events),
            held_event_patient,
            int(held_patients.numel()),
        )
        uniform_patient_probability = aggregate_patient_probabilities(
            uniform_event_logits.index_select(0, held_events),
            held_event_patient,
            int(held_patients.numel()),
        )
        learned_patient_logits = _probability_to_logit(learned_patient_probability)
        uniform_patient_logits = _probability_to_logit(uniform_patient_probability)

        oof_patient[LEARNED].index_copy_(0, held_patients, learned_patient_logits)
        oof_patient[UNIFORM].index_copy_(0, held_patients, uniform_patient_logits)
        oof_event[LEARNED].index_copy_(
            0, held_events, learned_event_logits.index_select(0, held_events)
        )
        oof_event[UNIFORM].index_copy_(
            0, held_events, uniform_event_logits.index_select(0, held_events)
        )
        detection_probability.index_copy_(
            0, held_events, fold_detection.index_select(0, held_events)
        )
        zero_input_detection_probability.index_copy_(
            0, held_events, zero_probability
        )
        time_reversal_detection_probability.index_copy_(
            0, held_events, reversed_probability
        )
        temporal_attention.index_copy_(
            0, held_events, fold_attention.index_select(0, held_events)
        )

        held_targets = inputs.targets.index_select(0, held_patients)
        held_mask = inputs.target_mask.index_select(0, held_patients)
        held_metrics = {
            LEARNED: _evaluate(learned_patient_logits, held_targets, held_mask),
            UNIFORM: _evaluate(uniform_patient_logits, held_targets, held_mask),
        }
        held_detection_metrics = _detection_metrics(
            fold_detection.index_select(0, held_events),
            inputs.detection_targets.index_select(0, held_events),
            fold_attention.index_select(0, held_events),
        )
        held_zero_metrics = _probability_only_detection_metrics(
            zero_probability,
            inputs.detection_targets.index_select(0, held_events),
        )
        held_reversal_metrics = _probability_only_detection_metrics(
            reversed_probability,
            inputs.detection_targets.index_select(0, held_events),
        )
        for name, value in _state_to_cpu(detector).items():
            fold_states[f"outer{outer_fold}.detector.{name}"] = value
        for arm, head in ((LEARNED, learned_head), (UNIFORM, uniform_head)):
            for name, value in _state_to_cpu(head).items():
                fold_states[f"outer{outer_fold}.{arm}.{name}"] = value
        fold_states[f"outer{outer_fold}.prior"] = prior.detach().cpu()
        fold_results.append(
            {
                "outer_fold": outer_fold,
                "train_patient_count": int(train_patients.numel()),
                "held_patient_count": int(held_patients.numel()),
                "train_event_count": int(train_events.numel()),
                "held_event_count": int(held_events.numel()),
                "train_patient_ids": [inputs.patient_ids[i] for i in train_patients],
                "held_patient_ids": [inputs.patient_ids[i] for i in held_patients],
                "detector_train_final": detection_history[-1],
                "learned_localizer_train_final": learned_history[-1],
                "uniform_localizer_train_final": uniform_history[-1],
                "held_detection_metrics": held_detection_metrics,
                "held_zero_input_detection_metrics": held_zero_metrics,
                "held_time_reversal_detection_metrics": held_reversal_metrics,
                "held_soz_metrics": held_metrics,
            }
        )
        print(
            json.dumps(
                {
                    "outer_fold": outer_fold,
                    "status": "complete",
                    "detector_auroc": held_detection_metrics["auroc"],
                    "learned_strict": held_metrics[LEARNED]["top1"][
                        "strict_accuracy"
                    ],
                    "uniform_strict": held_metrics[UNIFORM]["top1"][
                        "strict_accuracy"
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del (
            detector,
            learned_head,
            uniform_head,
            learned_features,
            uniform_features,
            fold_detection,
            fold_attention,
            zero_probability,
            reversed_probability,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    tensors_to_check = (
        *oof_patient.values(),
        *oof_event.values(),
        detection_probability,
        zero_input_detection_probability,
        time_reversal_detection_probability,
        temporal_attention,
    )
    if any(not torch.isfinite(value).all() for value in tensors_to_check):
        raise RuntimeError("OOF prediction carrier is incomplete")
    if not torch.allclose(
        temporal_attention.sum(dim=1),
        torch.ones(PRIMARY_EVENT_COUNT),
        atol=1.0e-6,
        rtol=1.0e-6,
    ):
        raise RuntimeError("OOF temporal attention no longer sums to one")

    all_logits = {
        LEARNED: oof_patient[LEARNED],
        UNIFORM: oof_patient[UNIFORM],
        V11_FULL: comparator.full,
        V11_FROZEN: comparator.frozen,
        PREVALENCE: comparator.prevalence,
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
            oof_patient[LEARNED],
            all_logits[baseline],
            inputs.targets,
            inputs.target_mask,
        )
        for baseline in (UNIFORM, V11_FULL, V11_FROZEN, PREVALENCE)
    }
    detection_metrics = _detection_metrics(
        detection_probability,
        inputs.detection_targets,
        temporal_attention,
    )
    zero_input_metrics = _probability_only_detection_metrics(
        zero_input_detection_probability,
        inputs.detection_targets,
    )
    time_reversal_metrics = _probability_only_detection_metrics(
        time_reversal_detection_probability,
        inputs.detection_targets,
    )
    qualified, gate_checks = _promotion_gate(
        metrics,
        detection_metrics,
        zero_input_metrics,
        time_reversal_metrics,
    )
    fold_strict = {
        name: _fold_strict(logits, inputs) for name, logits in all_logits.items()
    }

    output = args.output_directory
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is non-empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    prediction_tensors = {
        "oof.patient.deepsoz_detection_attention": oof_patient[LEARNED],
        "oof.patient.matched_uniform_time": oof_patient[UNIFORM],
        "oof.event.deepsoz_detection_attention": oof_event[LEARNED],
        "oof.event.matched_uniform_time": oof_event[UNIFORM],
        "oof.detection_probability": detection_probability,
        "oof.zero_input_detection_probability": zero_input_detection_probability,
        "oof.time_reversal_detection_probability": time_reversal_detection_probability,
        "oof.temporal_attention": temporal_attention,
        "detection_targets": inputs.detection_targets,
        "targets": inputs.targets,
        "target_mask": inputs.target_mask,
        "patient_folds": inputs.patient_folds,
        "patient_event_counts": inputs.event_counts,
        "event_patient_index": inputs.event_patient_index,
        "config.candidate_mask": V11_CANDIDATE_MASK,
    }
    save_file(
        {name: value.detach().cpu().contiguous() for name, value in prediction_tensors.items()},
        str(output / "oof_predictions.safetensors"),
    )
    save_file(
        {name: value.detach().cpu().contiguous() for name, value in fold_states.items()},
        str(output / "outer_fold_states.safetensors"),
    )
    manifest: Mapping[str, object] = {
        "schema_version": SCHEMA,
        "status": "completed_internal_developmental_patient_oof",
        "decision": (
            "PROMOTE_AS_PUBLIC_DEVELOPMENT_ANCHOR_ONLY"
            if qualified
            else "STOP_PREDECLARED_PROMOTION_GATE_FAILED"
        ),
        "claim_boundary": {
            "internal_developmental_oof_only": True,
            "fresh_public_test": False,
            "external_validation": False,
            "private_used": False,
            "clinical_deployment_allowed": False,
            "published_deepsoz_reproduction": False,
        },
        "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)),
        "foundation": {
            "backbone": "official_pretrained_LaBraM_Base",
            "replaced": False,
            "trained_from_scratch": False,
            "foundation_trainable_parameters": 0,
            "carrier": "frozen_block9_node_tokens",
        },
        "method": {
            "input_shape": [PRIMARY_EVENT_COUNT, 19, 60, 200],
            "time_axis_seconds": [-12, 48],
            "temporal_resolution_seconds": 1,
            "detector": "non_affine_LayerNorm_channel_mean_BiLSTM100x2_two_class",
            "attention": "class_softmax_seizure_probability_normalized_over_60_seconds",
            "channel_head": "shared_linear_200_to_1_no_bias_plus_outer_train_Jeffreys_prior",
            "event_pool": "attention_weighted_frozen_node_token",
            "patient_pool": "equal_mean_of_all_event_probabilities",
            "target": "DeepSOZ_patient_electrode_benchmark_membership",
            "not_claimed": [
                "cortical_onset_time",
                "SOZ_onset_time",
                "propagation_ground_truth",
            ],
        },
        "training": {
            "seed_policy": "20260812_plus_outer_fold_no_seed_scan",
            "detector_epochs": DETECTION_EPOCHS,
            "localization_epochs": LOCALIZATION_EPOCHS,
            "event_batch_size": EVENT_BATCH_SIZE,
            "detector_train_random_crop_seconds": DETECTION_TRAIN_CROP_SECONDS,
            "detector_train_random_crop_start_inclusive": [
                0,
                DETECTION_TRAIN_MAX_START,
            ],
            "detector_held_inference_seconds": N_SECONDS,
            "detector_lr": DETECTION_LR,
            "localization_lr": LOCALIZATION_LR,
            "detector_dropout": DETECTION_DROPOUT,
            "detector_hidden_per_direction": DETECTOR_HIDDEN,
            "detection_loss": "weighted_cross_entropy_weights_0.2_0.8",
            "localization_loss": "patient_masked_BCE_plus_0.1_probability_L1",
            "early_stopping": False,
            "hyperparameter_scan": False,
            "detector_frozen_during_localization": True,
        },
        "primary_patient_count": PRIMARY_PATIENT_COUNT,
        "primary_event_count": PRIMARY_EVENT_COUNT,
        "excluded_partial_reference_patient": EXCLUDED_PARTIAL_REFERENCE_PATIENT,
        "patient_ids": list(inputs.patient_ids),
        "event_ids": list(inputs.event_ids),
        "patient_folds": inputs.patient_folds.tolist(),
        "event_counts": inputs.event_counts.tolist(),
        "duration_seconds": {
            "minimum": float(inputs.durations.min()),
            "median": float(inputs.durations.median()),
            "maximum": float(inputs.durations.max()),
            "capped_at_48_count": int((inputs.durations >= 48.0).sum()),
        },
        "fold_results": fold_results,
        "detection_oof_metrics": detection_metrics,
        "zero_input_detection_oof_metrics": zero_input_metrics,
        "time_reversal_detection_oof_metrics": time_reversal_metrics,
        "soz_metrics": metrics,
        "absolute_patient_bootstrap": absolute,
        "paired_learned_minus_comparators": paired,
        "fold_strict": fold_strict,
        "promotion_gate": {
            "qualified": qualified,
            "checks": gate_checks,
        },
        "elapsed_seconds": time.monotonic() - started,
        "files": {
            "predictions": "oof_predictions.safetensors",
            "fold_states": "outer_fold_states.safetensors",
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--union-directory", type=Path, default=DEFAULT_UNION)
    parser.add_argument("--prefix-directory", type=Path, default=DEFAULT_PREFIX)
    parser.add_argument("--target-directory", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument(
        "--v11-reference-directory", type=Path, default=DEFAULT_V11_REFERENCE
    )
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(args)
    summary = {
        "decision": result["decision"],
        "detection_auroc": result["detection_oof_metrics"]["auroc"],
        "learned_strict": result["soz_metrics"][LEARNED]["top1"][
            "strict_accuracy"
        ],
        "learned_relaxed": result["soz_metrics"][LEARNED]["top1"][
            "relaxed_accuracy"
        ],
        "learned_macro_ap": result["soz_metrics"][LEARNED]["ranking"][
            "macro_average_precision"
        ],
        "uniform_strict": result["soz_metrics"][UNIFORM]["top1"][
            "strict_accuracy"
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
