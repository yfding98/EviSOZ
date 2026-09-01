#!/usr/bin/env python3
"""Train the heterogeneous SOZ pretraining model."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from soz_pre.constants import HEMISPHERE_CLASSES, REGION_NAMES, REGION_TO_CHANNELS, TCP_CHANNELS  # noqa: E402
from soz_pre.dataset import SOURCE_NAME, UnifiedSOZDataset  # noqa: E402
from soz_pre.model import EEGNetSOZNet, SOZPreNet  # noqa: E402
from soz_pre.vepiset import VEPiSetSOZPreDataset  # noqa: E402


DEFAULT_PREPROCESSED = "outputs/soz_pre/preprocessed"
DEFAULT_OUTPUT = "outputs/soz_pre/region_soz_run"

REGION_CHANNEL_INDICES: Tuple[Tuple[int, ...], ...] = tuple(
    tuple(TCP_CHANNELS.index(channel) for channel in REGION_TO_CHANNELS[name])
    for name in REGION_NAMES
)


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def snapshot_model_state(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Clone a stable CPU copy so later optimizer steps cannot mutate best state."""

    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def parse_list(value: object) -> List[str]:
    text = str(value or "").strip()
    if not text:
        return []
    return [item.strip() for item in text.replace(";", ",").split(",") if item.strip()]


def masked_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    sample_weight: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if pos_weight is not None:
        pos_weight = pos_weight.to(logits.device)
    loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction="none")
    weight = mask.float() * sample_weight.float().view(-1, *([1] * (loss.ndim - 1))).clamp_min(0.0)
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)


def hemisphere_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, sample_weight: torch.Tensor) -> torch.Tensor:
    valid = (mask.float() > 0.5) & (targets >= 0)
    if not valid.any():
        return logits.sum() * 0.0
    loss = F.cross_entropy(logits[valid], targets[valid], reduction="none")
    weights = sample_weight[valid].float().clamp_min(0.0)
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def compute_pos_weight(labels: np.ndarray, masks: np.ndarray, max_weight: float = 5.0) -> torch.Tensor:
    valid = np.asarray(masks, dtype=np.float32) > 0.5
    positives = (np.asarray(labels, dtype=np.float32) > 0.5) & valid
    pos = positives.sum(axis=0).astype(np.float32)
    total = valid.sum(axis=0).astype(np.float32)
    neg = np.maximum(total - pos, 0.0)
    weights = np.ones_like(pos, dtype=np.float32)
    seen = pos > 0
    weights[seen] = neg[seen] / np.maximum(pos[seen], 1.0)
    weights = np.clip(weights, 1.0 / max(float(max_weight), 1.0), max(float(max_weight), 1.0))
    return torch.as_tensor(weights, dtype=torch.float32)


def channel_logits_to_region_logits(channel_logits: torch.Tensor) -> torch.Tensor:
    return channel_logits_to_region_logits_with_indices(channel_logits, REGION_CHANNEL_INDICES)


def channel_logits_to_region_logits_with_indices(
    channel_logits: torch.Tensor,
    region_channel_indices: Sequence[Sequence[int]],
) -> torch.Tensor:
    pooled = []
    for indices in region_channel_indices:
        idx = torch.as_tensor(indices, dtype=torch.long, device=channel_logits.device)
        pooled.append(channel_logits.index_select(1, idx).max(dim=1).values)
    return torch.stack(pooled, dim=1)


def channel_probs_to_region_probs(
    channel_probs: np.ndarray,
    region_channel_indices: Sequence[Sequence[int]] = REGION_CHANNEL_INDICES,
) -> np.ndarray:
    values = []
    for indices in region_channel_indices:
        values.append(np.max(channel_probs[:, list(indices)], axis=1))
    return np.stack(values, axis=1).astype(np.float32)


def blend_region_probs(
    region_probs: np.ndarray,
    channel_probs: np.ndarray,
    blend: float,
    region_channel_indices: Sequence[Sequence[int]] = REGION_CHANNEL_INDICES,
) -> np.ndarray:
    alpha = float(np.clip(blend, 0.0, 1.0))
    if alpha <= 0.0:
        return region_probs
    pooled = channel_probs_to_region_probs(channel_probs, region_channel_indices)
    return ((1.0 - alpha) * region_probs + alpha * pooled).astype(np.float32)


def dataset_region_names(dataset: UnifiedSOZDataset) -> Tuple[str, ...]:
    return tuple(getattr(dataset, "region_names", REGION_NAMES))


def dataset_channel_names(dataset: UnifiedSOZDataset) -> Tuple[str, ...]:
    return tuple(getattr(dataset, "channel_names", TCP_CHANNELS))


def dataset_region_channel_indices(dataset: UnifiedSOZDataset) -> Tuple[Tuple[int, ...], ...]:
    return tuple(
        tuple(int(idx) for idx in indices)
        for indices in getattr(dataset, "region_channel_indices", REGION_CHANNEL_INDICES)
    )


def source_scaled_sample_weight(batch: Dict[str, torch.Tensor], base_weight: torch.Tensor, args) -> torch.Tensor:
    """Downweight weak public spatial supervision without touching seizure loss."""

    source_id = batch.get("source_id")
    if source_id is None:
        return base_weight
    source_id = source_id.to(base_weight.device)
    scale = torch.ones_like(base_weight, dtype=torch.float32)
    scale = torch.where(
        source_id == 0,
        scale * float(getattr(args, "tusz_spatial_weight_scale", 1.0)),
        scale,
    )
    scale = torch.where(
        source_id == 1,
        scale * float(getattr(args, "private_spatial_weight_scale", 1.0)),
        scale,
    )
    scale = torch.where(
        source_id >= 2,
        scale * float(getattr(args, "other_spatial_weight_scale", 1.0)),
        scale,
    )
    return base_weight.float() * scale.clamp_min(0.0)


def masked_pairwise_ranking_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    sample_weight: torch.Tensor,
    margin: float = 0.0,
) -> torch.Tensor:
    """Rank positive SOZ labels above valid negatives for top-k localization."""

    losses: List[torch.Tensor] = []
    weights: List[torch.Tensor] = []
    valid_mask = mask.float() > 0.5
    positive_mask = (targets.float() > 0.5) & valid_mask
    negative_mask = (targets.float() <= 0.5) & valid_mask
    for row_idx in range(logits.shape[0]):
        pos = logits[row_idx][positive_mask[row_idx]]
        neg = logits[row_idx][negative_mask[row_idx]]
        weight = sample_weight[row_idx].float().clamp_min(0.0)
        if pos.numel() == 0 or neg.numel() == 0 or float(weight.detach().cpu()) <= 0.0:
            continue
        pair_loss = F.softplus(neg.unsqueeze(1) - pos.unsqueeze(0) + float(margin)).mean()
        losses.append(pair_loss)
        weights.append(weight)
    if not losses:
        return logits.sum() * 0.0
    stacked = torch.stack(losses)
    weight_tensor = torch.stack(weights)
    return (stacked * weight_tensor).sum() / weight_tensor.sum().clamp_min(1.0)


def empty_metrics(prefix: str = "") -> Dict[str, float]:
    return {
        f"{prefix}accuracy": 0.0,
        f"{prefix}precision": 0.0,
        f"{prefix}recall": 0.0,
        f"{prefix}f1": 0.0,
        f"{prefix}support_positive": 0.0,
        f"{prefix}support_negative": 0.0,
    }


def binary_metrics(probs: np.ndarray, targets: np.ndarray, masks: np.ndarray, threshold: float, prefix: str = "") -> Dict[str, float]:
    valid = np.asarray(masks) > 0.5
    if valid.sum() == 0:
        return empty_metrics(prefix)
    pred = (np.asarray(probs) >= float(threshold)) & valid
    truth = (np.asarray(targets) >= 0.5) & valid
    tp = float(np.logical_and(pred, truth).sum())
    fp = float(np.logical_and(pred, ~truth & valid).sum())
    tn = float(np.logical_and(~pred & valid, ~truth).sum())
    fn = float(np.logical_and(~pred & valid, truth).sum())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    accuracy = (tp + tn) / max(tp + fp + tn + fn, 1.0)
    return {
        f"{prefix}accuracy": accuracy,
        f"{prefix}precision": precision,
        f"{prefix}recall": recall,
        f"{prefix}f1": f1,
        f"{prefix}support_positive": tp + fn,
        f"{prefix}support_negative": tn + fp,
    }


def macro_binary_metrics(probs: np.ndarray, targets: np.ndarray, masks: np.ndarray, threshold: float, prefix: str = "") -> Dict[str, float]:
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []
    accuracies: List[float] = []
    for col_idx in range(np.asarray(probs).shape[1]):
        valid = np.asarray(masks)[:, col_idx] > 0.5
        if not valid.any():
            continue
        pred = np.asarray(probs)[valid, col_idx] >= float(threshold)
        truth = np.asarray(targets)[valid, col_idx] >= 0.5
        tp = float(np.logical_and(pred, truth).sum())
        fp = float(np.logical_and(pred, ~truth).sum())
        tn = float(np.logical_and(~pred, ~truth).sum())
        fn = float(np.logical_and(~pred, truth).sum())
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
        accuracies.append((tp + tn) / max(tp + fp + tn + fn, 1.0))
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return {
        f"{prefix}macro_accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
        f"{prefix}macro_precision": float(np.mean(precisions)) if precisions else 0.0,
        f"{prefix}macro_recall": float(np.mean(recalls)) if recalls else 0.0,
        f"{prefix}macro_f1": float(np.mean(f1s)) if f1s else 0.0,
    }


def sample_multilabel_metrics(probs: np.ndarray, targets: np.ndarray, masks: np.ndarray, threshold: float, prefix: str = "") -> Dict[str, float]:
    jaccards: List[float] = []
    exact_matches: List[float] = []
    empty_preds: List[float] = []
    for score, target, mask in zip(probs, targets, masks):
        valid = mask > 0.5
        positives = (target > 0.5) & valid
        if not valid.any() or not positives.any():
            continue
        pred = (score >= float(threshold)) & valid
        union = np.logical_or(pred, positives)
        inter = np.logical_and(pred, positives)
        jaccards.append(float(inter.sum() / max(union.sum(), 1)))
        exact_matches.append(float(np.array_equal(pred[valid], positives[valid])))
        empty_preds.append(float(pred.sum() == 0))
    return {
        f"{prefix}sample_jaccard": float(np.mean(jaccards)) if jaccards else 0.0,
        f"{prefix}exact_match": float(np.mean(exact_matches)) if exact_matches else 0.0,
        f"{prefix}empty_prediction_rate": float(np.mean(empty_preds)) if empty_preds else 0.0,
    }


def top1_multilabel_hit(probs: np.ndarray, targets: np.ndarray, masks: np.ndarray) -> float:
    hits: List[float] = []
    for score, target, mask in zip(probs, targets, masks):
        valid = np.where(mask > 0.5)[0]
        positives = valid[target[valid] > 0.5]
        if len(valid) == 0 or len(positives) == 0:
            continue
        top = valid[np.argmax(score[valid])]
        hits.append(float(top in set(positives.tolist())))
    return float(np.mean(hits)) if hits else 0.0


def topk_channel_hit(probs: np.ndarray, targets: np.ndarray, masks: np.ndarray) -> float:
    hits: List[float] = []
    for score, target, mask in zip(probs, targets, masks):
        valid = np.where(mask > 0.5)[0]
        positives = valid[target[valid] > 0.5]
        if len(valid) == 0 or len(positives) == 0:
            continue
        ranked = valid[np.argsort(score[valid])[::-1]]
        topk = set(ranked[: len(positives)].tolist())
        hits.append(len(topk & set(positives.tolist())) / float(len(positives)))
    return float(np.mean(hits)) if hits else 0.0


def hemisphere_accuracy(logits: np.ndarray, targets: np.ndarray, masks: np.ndarray) -> float:
    valid = (masks > 0.5) & (targets >= 0)
    if not valid.any():
        return 0.0
    pred = np.argmax(logits[valid], axis=1)
    return float((pred == targets[valid]).mean())


def _patient_rank_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    patient_ids: Sequence[str],
) -> Dict[str, float]:
    groups: Dict[str, List[int]] = defaultdict(list)
    for idx, patient_id in enumerate(patient_ids):
        groups[str(patient_id or f"sample_{idx}")].append(idx)

    top1_hits: List[float] = []
    top2_hits: List[float] = []
    topk_recalls: List[float] = []
    for indices in groups.values():
        n_labels = probs.shape[1]
        score_sum = np.zeros(n_labels, dtype=np.float64)
        score_count = np.zeros(n_labels, dtype=np.float64)
        target_union = np.zeros(n_labels, dtype=bool)
        valid_union = np.zeros(n_labels, dtype=bool)
        for idx in indices:
            valid = masks[idx] > 0.5
            positive = (targets[idx] > 0.5) & valid
            if not valid.any() or not positive.any():
                continue
            score = np.asarray(probs[idx], dtype=np.float64)
            max_score = float(score[valid].max())
            if max_score <= 1e-12:
                max_score = 1.0
            score_sum[valid] += score[valid] / max_score
            score_count[valid] += 1.0
            target_union |= positive
            valid_union |= valid
        positives = np.where(target_union & valid_union)[0]
        valid_indices = np.where(valid_union & (score_count > 0))[0]
        if len(positives) == 0 or len(valid_indices) == 0:
            continue
        aggregate = np.zeros(n_labels, dtype=np.float64)
        aggregate[valid_indices] = score_sum[valid_indices] / np.maximum(score_count[valid_indices], 1.0)
        ranked = valid_indices[np.argsort(aggregate[valid_indices])[::-1]]
        positive_set = set(positives.tolist())
        top1_hits.append(float(int(ranked[0]) in positive_set))
        top2_hits.append(float(bool(set(ranked[: min(2, len(ranked))].tolist()) & positive_set)))
        topk = set(ranked[: max(1, len(positive_set))].tolist())
        topk_recalls.append(float(len(topk & positive_set) / max(len(positive_set), 1)))

    return {
        "top1_hit": float(np.mean(top1_hits)) if top1_hits else 0.0,
        "top2_hit": float(np.mean(top2_hits)) if top2_hits else 0.0,
        "topk_hit": float(np.mean(topk_recalls)) if topk_recalls else 0.0,
        "n_patients": float(len(top1_hits)),
    }


def _patient_threshold_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    patient_ids: Sequence[str],
    threshold: float,
) -> Dict[str, float]:
    groups: Dict[str, List[int]] = defaultdict(list)
    for idx, patient_id in enumerate(patient_ids):
        groups[str(patient_id or f"sample_{idx}")].append(idx)

    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []
    jaccards: List[float] = []
    empty_preds: List[float] = []
    for indices in groups.values():
        n_labels = probs.shape[1]
        score_sum = np.zeros(n_labels, dtype=np.float64)
        score_count = np.zeros(n_labels, dtype=np.float64)
        target_union = np.zeros(n_labels, dtype=bool)
        valid_union = np.zeros(n_labels, dtype=bool)
        for idx in indices:
            valid = masks[idx] > 0.5
            positive = (targets[idx] > 0.5) & valid
            if not valid.any() or not positive.any():
                continue
            score = np.asarray(probs[idx], dtype=np.float64)
            max_score = float(score[valid].max())
            if max_score <= 1e-12:
                max_score = 1.0
            score_sum[valid] += score[valid] / max_score
            score_count[valid] += 1.0
            target_union |= positive
            valid_union |= valid
        positives = target_union & valid_union
        valid_indices = valid_union & (score_count > 0)
        if not positives.any() or not valid_indices.any():
            continue
        aggregate = np.zeros(n_labels, dtype=np.float64)
        aggregate[valid_indices] = score_sum[valid_indices] / np.maximum(score_count[valid_indices], 1.0)
        pred = (aggregate >= float(threshold)) & valid_indices
        tp = float(np.logical_and(pred, positives).sum())
        fp = float(np.logical_and(pred, ~positives & valid_indices).sum())
        fn = float(np.logical_and(~pred & valid_indices, positives).sum())
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
        union = float(np.logical_or(pred, positives).sum())
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        jaccards.append(tp / max(union, 1.0))
        empty_preds.append(float(pred.sum() == 0))
    return {
        "threshold_precision": float(np.mean(precisions)) if precisions else 0.0,
        "threshold_recall": float(np.mean(recalls)) if recalls else 0.0,
        "threshold_f1": float(np.mean(f1s)) if f1s else 0.0,
        "threshold_jaccard": float(np.mean(jaccards)) if jaccards else 0.0,
        "threshold_empty_rate": float(np.mean(empty_preds)) if empty_preds else 0.0,
    }


def patient_aggregate_metrics(
    pred: Dict[str, np.ndarray],
    dataset: UnifiedSOZDataset,
    *,
    prefix: str = "patient_",
    source: str = "",
    threshold: float = 0.5,
) -> Dict[str, float]:
    indices: List[int] = []
    patient_ids: List[str] = []
    source_filter = str(source or "").lower()
    for idx, meta in enumerate(dataset.segment_meta):
        row_source = str(meta.get("source", "")).lower()
        if source_filter and row_source != source_filter:
            continue
        indices.append(idx)
        patient_ids.append(str(meta.get("base_patient_id") or meta.get("patient_id") or f"sample_{idx}"))
    if not indices:
        return {
            f"{prefix}region_top1_hit": 0.0,
            f"{prefix}region_top2_hit": 0.0,
            f"{prefix}region_topk_hit": 0.0,
            f"{prefix}region_threshold_precision": 0.0,
            f"{prefix}region_threshold_recall": 0.0,
            f"{prefix}region_threshold_f1": 0.0,
            f"{prefix}region_threshold_jaccard": 0.0,
            f"{prefix}region_threshold_empty_rate": 0.0,
            f"{prefix}channel_top1_hit": 0.0,
            f"{prefix}channel_top2_hit": 0.0,
            f"{prefix}channel_topk_hit": 0.0,
            f"{prefix}n_patients": 0.0,
        }
    idx_arr = np.asarray(indices, dtype=int)
    region = _patient_rank_metrics(
        pred["region_probs"][idx_arr],
        dataset.region_labels_np[idx_arr],
        dataset.region_masks_np[idx_arr],
        patient_ids,
    )
    channel = _patient_rank_metrics(
        pred["channel_probs"][idx_arr],
        dataset.channel_labels_np[idx_arr],
        dataset.channel_masks_np[idx_arr],
        patient_ids,
    )
    region_threshold = _patient_threshold_metrics(
        pred["region_probs"][idx_arr],
        dataset.region_labels_np[idx_arr],
        dataset.region_masks_np[idx_arr],
        patient_ids,
        threshold,
    )
    return {
        f"{prefix}region_top1_hit": region["top1_hit"],
        f"{prefix}region_top2_hit": region["top2_hit"],
        f"{prefix}region_topk_hit": region["topk_hit"],
        f"{prefix}region_threshold_precision": region_threshold["threshold_precision"],
        f"{prefix}region_threshold_recall": region_threshold["threshold_recall"],
        f"{prefix}region_threshold_f1": region_threshold["threshold_f1"],
        f"{prefix}region_threshold_jaccard": region_threshold["threshold_jaccard"],
        f"{prefix}region_threshold_empty_rate": region_threshold["threshold_empty_rate"],
        f"{prefix}channel_top1_hit": channel["top1_hit"],
        f"{prefix}channel_top2_hit": channel["top2_hit"],
        f"{prefix}channel_topk_hit": channel["topk_hit"],
        f"{prefix}n_patients": max(region["n_patients"], channel["n_patients"]),
    }


def collect_predictions(
    model: SOZPreNet,
    dataset: UnifiedSOZDataset,
    batch_size: int,
    device: torch.device,
    num_workers: int,
    region_pool_blend: float = 0.0,
) -> Dict[str, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.eval()
    region_channel_indices = dataset_region_channel_indices(dataset)
    out: Dict[str, List[np.ndarray]] = {
        "channel_probs": [],
        "region_probs": [],
        "propagation_probs": [],
        "seizure_probs": [],
        "hemisphere_logits": [],
        "attention": [],
    }
    with torch.no_grad():
        for batch in loader:
            y = model(batch["x"].to(device), batch["input_mask"].to(device))
            channel_probs = torch.sigmoid(y["channel_logits"]).cpu().numpy()
            region_probs = torch.sigmoid(y["region_logits"]).cpu().numpy()
            out["channel_probs"].append(channel_probs)
            out["region_probs"].append(
                blend_region_probs(region_probs, channel_probs, region_pool_blend, region_channel_indices)
            )
            out["propagation_probs"].append(torch.sigmoid(y["propagation_logits"]).cpu().numpy())
            out["seizure_probs"].append(torch.sigmoid(y["seizure_logits"]).cpu().numpy())
            out["hemisphere_logits"].append(y["hemisphere_logits"].cpu().numpy())
            out["attention"].append(y["attention"].cpu().numpy())
    return {key: np.concatenate(values, axis=0) for key, values in out.items()}


def evaluate_from_predictions(pred: Dict[str, np.ndarray], dataset: UnifiedSOZDataset, threshold: float) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    metrics.update(binary_metrics(pred["channel_probs"], dataset.channel_labels_np, dataset.channel_masks_np, threshold, prefix="channel_"))
    metrics.update(binary_metrics(pred["region_probs"], dataset.region_labels_np, dataset.region_masks_np, threshold, prefix="region_"))
    metrics.update(macro_binary_metrics(pred["region_probs"], dataset.region_labels_np, dataset.region_masks_np, threshold, prefix="region_"))
    metrics.update(sample_multilabel_metrics(pred["region_probs"], dataset.region_labels_np, dataset.region_masks_np, threshold, prefix="region_"))
    metrics.update(macro_binary_metrics(pred["channel_probs"], dataset.channel_labels_np, dataset.channel_masks_np, threshold, prefix="channel_"))
    metrics.update(sample_multilabel_metrics(pred["channel_probs"], dataset.channel_labels_np, dataset.channel_masks_np, threshold, prefix="channel_"))
    metrics.update(binary_metrics(pred["propagation_probs"], dataset.propagation_labels_np, dataset.propagation_masks_np, threshold, prefix="propagation_"))
    metrics.update(binary_metrics(pred["seizure_probs"], dataset.seizure_y_np, dataset.seizure_mask_np, threshold, prefix="seizure_"))
    metrics["region_top1_hit"] = top1_multilabel_hit(pred["region_probs"], dataset.region_labels_np, dataset.region_masks_np)
    metrics["channel_topk_hit"] = topk_channel_hit(pred["channel_probs"], dataset.channel_labels_np, dataset.channel_masks_np)
    metrics["hemisphere_accuracy"] = hemisphere_accuracy(pred["hemisphere_logits"], dataset.hemisphere_labels_np, dataset.hemisphere_masks_np)
    metrics.update(patient_aggregate_metrics(pred, dataset, prefix="patient_", threshold=threshold))
    for source in ("private", "tusz"):
        if any(str(meta.get("source", "")).lower() == source for meta in dataset.segment_meta):
            metrics.update(patient_aggregate_metrics(pred, dataset, prefix=f"{source}_patient_", source=source, threshold=threshold))
    return metrics


def run_epoch(
    model: SOZPreNet,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    args,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {
        "loss": 0.0,
        "channel_loss": 0.0,
        "region_loss": 0.0,
        "propagation_loss": 0.0,
        "seizure_loss": 0.0,
        "hemisphere_loss": 0.0,
        "channel_rank_loss": 0.0,
        "region_rank_loss": 0.0,
        "channel_region_loss": 0.0,
    }
    n_batches = 0
    for batch in loader:
        x = batch["x"].to(device)
        input_mask = batch["input_mask"].to(device)
        sample_weight = batch["sample_weight"].to(device)
        spatial_sample_weight = source_scaled_sample_weight(batch, sample_weight, args)
        with torch.set_grad_enabled(training):
            out = model(x, input_mask)
            channel_y = batch["channel_y"].to(device)
            channel_mask = batch["channel_mask"].to(device)
            region_y = batch["region_y"].to(device)
            region_mask = batch["region_mask"].to(device)
            channel_pos_weight_values = getattr(args, "_channel_pos_weight_values", None)
            region_pos_weight_values = getattr(args, "_region_pos_weight_values", None)
            channel_pos_weight = torch.as_tensor(channel_pos_weight_values, dtype=torch.float32, device=device) if channel_pos_weight_values else None
            region_pos_weight = torch.as_tensor(region_pos_weight_values, dtype=torch.float32, device=device) if region_pos_weight_values else None
            channel_loss = masked_bce(out["channel_logits"], channel_y, channel_mask, spatial_sample_weight, pos_weight=channel_pos_weight)
            region_loss = masked_bce(out["region_logits"], region_y, region_mask, spatial_sample_weight, pos_weight=region_pos_weight)
            region_channel_indices = getattr(loader.dataset, "region_channel_indices", REGION_CHANNEL_INDICES)
            channel_region_logits = channel_logits_to_region_logits_with_indices(
                out["channel_logits"],
                region_channel_indices,
            )
            channel_region_loss = masked_bce(channel_region_logits, region_y, region_mask, spatial_sample_weight, pos_weight=region_pos_weight)
            propagation_loss = masked_bce(out["propagation_logits"], batch["propagation_y"].to(device), batch["propagation_mask"].to(device), spatial_sample_weight)
            seizure_loss = masked_bce(out["seizure_logits"], batch["seizure_y"].to(device), batch["seizure_mask"].to(device), torch.ones_like(sample_weight))
            hemi_loss = hemisphere_loss(out["hemisphere_logits"], batch["hemisphere_y"].to(device), batch["hemisphere_mask"].to(device), spatial_sample_weight)
            channel_rank_loss = masked_pairwise_ranking_loss(
                out["channel_logits"],
                channel_y,
                channel_mask,
                spatial_sample_weight,
                margin=float(getattr(args, "ranking_margin", 0.0)),
            )
            region_rank_loss = masked_pairwise_ranking_loss(
                out["region_logits"],
                region_y,
                region_mask,
                spatial_sample_weight,
                margin=float(getattr(args, "ranking_margin", 0.0)),
            )
            loss = (
                float(args.channel_loss_weight) * channel_loss
                + float(args.region_loss_weight) * region_loss
                + float(args.propagation_loss_weight) * propagation_loss
                + float(args.seizure_loss_weight) * seizure_loss
                + float(args.hemisphere_loss_weight) * hemi_loss
                + float(getattr(args, "channel_ranking_loss_weight", 0.0)) * channel_rank_loss
                + float(getattr(args, "region_ranking_loss_weight", 0.0)) * region_rank_loss
                + float(getattr(args, "channel_region_loss_weight", 0.0)) * channel_region_loss
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
                optimizer.step()
        totals["loss"] += float(loss.detach().cpu())
        totals["channel_loss"] += float(channel_loss.detach().cpu())
        totals["region_loss"] += float(region_loss.detach().cpu())
        totals["propagation_loss"] += float(propagation_loss.detach().cpu())
        totals["seizure_loss"] += float(seizure_loss.detach().cpu())
        totals["hemisphere_loss"] += float(hemi_loss.detach().cpu())
        totals["channel_rank_loss"] += float(channel_rank_loss.detach().cpu())
        totals["region_rank_loss"] += float(region_rank_loss.detach().cpu())
        totals["channel_region_loss"] += float(channel_region_loss.detach().cpu())
        n_batches += 1
    return {key: value / max(n_batches, 1) for key, value in totals.items()}


def save_history(path: Path, rows: Sequence[Dict[str, float]]) -> None:
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def save_predictions(path: Path, dataset: UnifiedSOZDataset, pred: Dict[str, np.ndarray]) -> None:
    region_names = dataset_region_names(dataset)
    channel_names = dataset_channel_names(dataset)
    fields = [
        "sample_index", "source", "split", "patient_id", "base_patient_id", "event_id",
        "sample_id", "sample_role", "edf_path", "true_regions", "pred_top_region",
        "hemisphere_true", "hemisphere_pred", "true_channels", "pred_top_channels",
    ]
    fields += [f"prob_region_{name}" for name in region_names]
    fields += [f"label_region_{name}" for name in region_names]
    fields += [f"prob_channel_{name.replace('-', '_')}" for name in channel_names]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for idx, meta in enumerate(dataset.segment_meta):
            region_probs = pred["region_probs"][idx]
            region_mask = dataset.region_masks_np[idx] > 0.5
            valid_region_idx = np.where(region_mask)[0]
            top_region = ""
            if len(valid_region_idx):
                top_region = region_names[int(valid_region_idx[np.argmax(region_probs[valid_region_idx])])]
            hemi_pred = HEMISPHERE_CLASSES[int(np.argmax(pred["hemisphere_logits"][idx]))]
            hemi_true_idx = dataset.hemisphere_labels_np[idx]
            hemi_true = HEMISPHERE_CLASSES[int(hemi_true_idx)] if 0 <= hemi_true_idx < len(HEMISPHERE_CLASSES) else ""
            channel_probs = pred["channel_probs"][idx]
            valid_channels = np.where(dataset.channel_masks_np[idx] > 0.5)[0]
            true_channels = [channel_names[i] for i in valid_channels if dataset.channel_labels_np[idx, i] > 0.5]
            top_n = max(1, len(true_channels))
            ranked = valid_channels[np.argsort(channel_probs[valid_channels])[::-1]] if len(valid_channels) else np.asarray([], dtype=int)
            row = {
                "sample_index": idx,
                "source": meta.get("source", ""),
                "split": meta.get("split", ""),
                "patient_id": meta.get("patient_id", ""),
                "base_patient_id": meta.get("base_patient_id", ""),
                "event_id": meta.get("event_id", ""),
                "sample_id": meta.get("sample_id", ""),
                "sample_role": meta.get("sample_role", ""),
                "edf_path": meta.get("edf_path", ""),
                "true_regions": ";".join(region_names[i] for i in range(len(region_names)) if dataset.region_labels_np[idx, i] > 0.5),
                "pred_top_region": top_region,
                "hemisphere_true": hemi_true,
                "hemisphere_pred": hemi_pred,
                "true_channels": ";".join(true_channels),
                "pred_top_channels": ";".join(channel_names[int(i)] for i in ranked[:top_n]),
            }
            for ridx, name in enumerate(region_names):
                row[f"prob_region_{name}"] = float(region_probs[ridx])
                row[f"label_region_{name}"] = int(dataset.region_labels_np[idx, ridx] > 0.5)
            for cidx, name in enumerate(channel_names):
                row[f"prob_channel_{name.replace('-', '_')}"] = float(channel_probs[cidx])
            writer.writerow(row)


def _aggregate_patient_prediction(
    probs: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    indices: Sequence[int],
    names: Sequence[str],
    threshold: float = 0.5,
    min_predictions: int = 1,
    max_predictions: int = 3,
) -> Tuple[List[str], List[str], Dict[str, float]]:
    n_labels = len(names)
    score_sum = np.zeros(n_labels, dtype=np.float64)
    score_count = np.zeros(n_labels, dtype=np.float64)
    target_union = np.zeros(n_labels, dtype=bool)
    valid_union = np.zeros(n_labels, dtype=bool)
    for idx in indices:
        valid = masks[idx] > 0.5
        positive = (targets[idx] > 0.5) & valid
        if not valid.any() or not positive.any():
            continue
        score = np.asarray(probs[idx], dtype=np.float64)
        max_score = float(score[valid].max())
        if max_score <= 1e-12:
            max_score = 1.0
        score_sum[valid] += score[valid] / max_score
        score_count[valid] += 1.0
        target_union |= positive
        valid_union |= valid
    valid_indices = np.where(valid_union & (score_count > 0))[0]
    aggregate = np.zeros(n_labels, dtype=np.float64)
    aggregate[valid_indices] = score_sum[valid_indices] / np.maximum(score_count[valid_indices], 1.0)
    ranked = valid_indices[np.argsort(aggregate[valid_indices])[::-1]] if len(valid_indices) else np.asarray([], dtype=int)
    true_names = [names[int(idx)] for idx in np.where(target_union & valid_union)[0]]
    thresholded = [int(idx) for idx in ranked if aggregate[int(idx)] >= float(threshold)]
    min_n = max(0, int(min_predictions))
    max_n = max(min_n, int(max_predictions))
    if len(thresholded) < min_n:
        thresholded = [int(idx) for idx in ranked[:min_n]]
    thresholded = thresholded[:max_n]
    top_names = [names[int(idx)] for idx in thresholded]
    scores = {names[int(idx)]: float(aggregate[int(idx)]) for idx in ranked}
    return true_names, top_names, scores


def save_patient_predictions(path: Path, dataset: UnifiedSOZDataset, pred: Dict[str, np.ndarray], args) -> None:
    region_names = dataset_region_names(dataset)
    channel_names = dataset_channel_names(dataset)
    groups: Dict[str, List[int]] = defaultdict(list)
    patient_sources: Dict[str, str] = {}
    for idx, meta in enumerate(dataset.segment_meta):
        patient = str(meta.get("base_patient_id") or meta.get("patient_id") or f"sample_{idx}")
        groups[patient].append(idx)
        patient_sources.setdefault(patient, str(meta.get("source", "")))

    fields = [
        "patient_id", "source", "n_samples",
        "true_regions", "pred_regions",
        "true_channels", "pred_channels",
        "region_scores_json", "channel_scores_json",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for patient, indices in sorted(groups.items()):
            true_regions, pred_regions, region_scores = _aggregate_patient_prediction(
                pred["region_probs"],
                dataset.region_labels_np,
                dataset.region_masks_np,
                indices,
                region_names,
                threshold=float(getattr(args, "patient_region_threshold", getattr(args, "threshold", 0.5))),
                min_predictions=int(getattr(args, "patient_min_regions", 1)),
                max_predictions=int(getattr(args, "patient_max_regions", 3)),
            )
            true_channels, pred_channels, channel_scores = _aggregate_patient_prediction(
                pred["channel_probs"],
                dataset.channel_labels_np,
                dataset.channel_masks_np,
                indices,
                channel_names,
                threshold=float(getattr(args, "patient_channel_threshold", getattr(args, "threshold", 0.5))),
                min_predictions=int(getattr(args, "patient_min_channels", 1)),
                max_predictions=int(getattr(args, "patient_max_channels", 8)),
            )
            writer.writerow({
                "patient_id": patient,
                "source": patient_sources.get(patient, ""),
                "n_samples": len(indices),
                "true_regions": ";".join(true_regions),
                "pred_regions": ";".join(pred_regions),
                "true_channels": ";".join(true_channels),
                "pred_channels": ";".join(pred_channels),
                "region_scores_json": json.dumps(region_scores, ensure_ascii=False, sort_keys=True),
                "channel_scores_json": json.dumps(channel_scores, ensure_ascii=False, sort_keys=True),
            })


def build_model_from_dataset(dataset: UnifiedSOZDataset, args) -> torch.nn.Module:
    model_name = str(getattr(args, "model", "deepsoz") or "deepsoz").lower()
    if model_name in {"deepsoz", "sozpre", "sozprenet"}:
        return SOZPreNet(
            n_input_channels=dataset.n_input_channels,
            n_label_channels=dataset.n_label_channels,
            window_samples=dataset.window_samples,
            n_regions=dataset.n_regions,
            n_hemisphere_classes=dataset.n_hemisphere_classes,
            d_model=args.d_model,
            nhead=args.nhead,
            transformer_layers=args.transformer_layers,
            dim_feedforward=args.dim_feedforward,
            lstm_hidden_dim=args.lstm_hidden_dim,
            dropout=args.dropout,
            attention_temperature=args.attention_temperature,
        )
    if model_name == "eegnet":
        return EEGNetSOZNet(
            n_input_channels=dataset.n_input_channels,
            n_label_channels=dataset.n_label_channels,
            window_samples=dataset.window_samples,
            n_windows=dataset.n_windows,
            n_regions=dataset.n_regions,
            n_hemisphere_classes=dataset.n_hemisphere_classes,
            temporal_filters=args.eegnet_temporal_filters,
            depth_multiplier=args.eegnet_depth_multiplier,
            pointwise_filters=args.eegnet_pointwise_filters,
            kernel_length=args.eegnet_kernel_length,
            separable_kernel_length=args.eegnet_separable_kernel_length,
            pool1=args.eegnet_pool1,
            pool2=args.eegnet_pool2,
            dropout=args.dropout,
            attention_temperature=args.attention_temperature,
        )
    raise ValueError(f"Unknown --model {model_name!r}; expected deepsoz or eegnet")


def build_source_balanced_sampler(dataset: UnifiedSOZDataset, args) -> Tuple[Optional[WeightedRandomSampler], Dict[str, object]]:
    mode = str(getattr(args, "source_balance", "none") or "none").lower()
    source_ids = np.asarray(getattr(dataset, "source_ids_np", []), dtype=np.int64)
    counts = Counter(source_ids.tolist())
    summary = {
        "mode": mode,
        "source_counts": {SOURCE_NAME.get(int(key), str(key)): int(value) for key, value in counts.items()},
        "enabled": False,
    }
    if mode == "none" or len(counts) <= 1:
        return None, summary
    if mode != "source":
        raise ValueError("--source_balance must be none or source")

    weights = np.asarray([1.0 / max(counts[int(source_id)], 1) for source_id in source_ids], dtype=np.float64)
    weights = weights / max(float(weights.mean()), 1e-12)
    cap = max(float(getattr(args, "sampler_weight_cap", 10.0)), 1.0)
    weights = np.clip(weights, 1.0 / cap, cap)
    weights = weights / max(float(weights.mean()), 1e-12)
    summary.update({
        "enabled": True,
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "weight_mean": float(weights.mean()),
        "num_samples": int(len(weights)),
    })
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )
    return sampler, summary


def train_once(train_ds: UnifiedSOZDataset, val_ds: UnifiedSOZDataset, output_dir: Path, args) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model_from_dataset(train_ds, args).to(device)
    if str(getattr(args, "region_pos_weight_mode", "none")).lower() == "balanced":
        args._region_pos_weight_values = compute_pos_weight(
            train_ds.region_labels_np,
            train_ds.region_masks_np,
            max_weight=float(getattr(args, "max_pos_weight", 5.0)),
        ).tolist()
    else:
        args._region_pos_weight_values = None
    if str(getattr(args, "channel_pos_weight_mode", "none")).lower() == "balanced":
        args._channel_pos_weight_values = compute_pos_weight(
            train_ds.channel_labels_np,
            train_ds.channel_masks_np,
            max_weight=float(getattr(args, "max_pos_weight", 5.0)),
        ).tolist()
    else:
        args._channel_pos_weight_values = None
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=device)
        model.load_state_dict(checkpoint.get("model_state", checkpoint), strict=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    sampler, sampler_summary = build_source_balanced_sampler(train_ds, args)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=int(args.num_workers),
    )
    val_loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))
    (output_dir / "sampler_summary.json").write_text(json.dumps(sampler_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    history: List[Dict[str, float]] = []
    best_metric = -1e9
    best_payload: Dict[str, object] = {}
    for epoch in range(1, int(args.epochs) + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, args)
        val_loss = run_epoch(model, val_loader, None, device, args)
        pred = collect_predictions(
            model,
            val_ds,
            int(args.batch_size),
            device,
            int(args.num_workers),
            region_pool_blend=float(getattr(args, "region_pool_blend", 0.0)),
        )
        val_metrics = evaluate_from_predictions(pred, val_ds, threshold=float(args.threshold))
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_loss.items()}, **{f"val_{k}": v for k, v in val_loss.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        metric_value = float(row.get(f"val_{args.selection_metric}", row.get("val_region_top1_hit", 0.0)))
        if metric_value > best_metric:
            best_metric = metric_value
            best_payload = {
                "epoch": epoch,
                "model_state": snapshot_model_state(model),
                "metrics": dict(row),
                "config": dict(vars(args)),
                "sampler_summary": dict(sampler_summary),
                "n_input_channels": train_ds.n_input_channels,
                "window_samples": train_ds.window_samples,
            }
            torch.save(best_payload, output_dir / "best_model.pt")
        print(json.dumps({"epoch": epoch, "selection": metric_value, "best": best_metric, "val_region_top1_hit": row.get("val_region_top1_hit", 0.0), "val_loss": row.get("val_loss", 0.0)}, ensure_ascii=False))
    save_history(output_dir / "history.csv", history)
    if best_payload:
        model.load_state_dict(best_payload["model_state"])
    final_pred = collect_predictions(
        model,
        val_ds,
        int(args.batch_size),
        device,
        int(args.num_workers),
        region_pool_blend=float(getattr(args, "region_pool_blend", 0.0)),
    )
    final_metrics = evaluate_from_predictions(final_pred, val_ds, threshold=float(args.threshold))
    final_metrics.update({"best_epoch": int(best_payload.get("epoch", 0)), "best_selection": float(best_metric)})
    save_predictions(output_dir / "val_predictions.csv", val_ds, final_pred)
    save_patient_predictions(output_dir / "val_patient_predictions.csv", val_ds, final_pred, args)
    (output_dir / "val_metrics.json").write_text(json.dumps(final_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")
    return {"metrics": final_metrics, "history": history}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train heterogeneous SOZ region/channel model")
    parser.add_argument("--dataset_format", choices=["unified", "vepiset"], default="unified")
    parser.add_argument("--preprocessed_dir", default=DEFAULT_PREPROCESSED)
    parser.add_argument("--vepiset_root", default="/mnt/hd1/dyf/dataset/vepiset-dataset/opensource-dataset")
    parser.add_argument("--vepiset_val_split", type=float, default=0.15)
    parser.add_argument("--vepiset_test_split", type=float, default=0.15)
    parser.add_argument("--vepiset_max_samples_per_class", type=int, default=0)
    parser.add_argument("--vepiset_max_non_ied_samples", type=int, default=0)
    parser.add_argument("--vepiset_target_samples", type=int, default=2000)
    parser.add_argument("--no_vepiset_normalize", action="store_true")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--train_splits", default="train,private")
    parser.add_argument("--val_splits", default="dev")
    parser.add_argument("--train_sources", default="")
    parser.add_argument("--val_sources", default="")
    parser.add_argument("--exclude_patients", default="")
    parser.add_argument("--val_patients", default="")
    parser.add_argument("--init_checkpoint", default="")
    parser.add_argument("--model", choices=["deepsoz", "eegnet"], default="deepsoz")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--transformer_layers", type=int, default=2)
    parser.add_argument("--dim_feedforward", type=int, default=128)
    parser.add_argument("--lstm_hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--attention_temperature", type=float, default=1.0)
    parser.add_argument("--eegnet_temporal_filters", type=int, default=16)
    parser.add_argument("--eegnet_depth_multiplier", type=int, default=2)
    parser.add_argument("--eegnet_pointwise_filters", type=int, default=32)
    parser.add_argument("--eegnet_kernel_length", type=int, default=64)
    parser.add_argument("--eegnet_separable_kernel_length", type=int, default=16)
    parser.add_argument("--eegnet_pool1", type=int, default=4)
    parser.add_argument("--eegnet_pool2", type=int, default=8)
    parser.add_argument("--channel_loss_weight", type=float, default=1.0)
    parser.add_argument("--region_loss_weight", type=float, default=1.5)
    parser.add_argument("--propagation_loss_weight", type=float, default=0.5)
    parser.add_argument("--seizure_loss_weight", type=float, default=0.5)
    parser.add_argument("--hemisphere_loss_weight", type=float, default=0.7)
    parser.add_argument("--channel_ranking_loss_weight", type=float, default=0.2)
    parser.add_argument("--region_ranking_loss_weight", type=float, default=0.1)
    parser.add_argument("--channel_region_loss_weight", type=float, default=0.3)
    parser.add_argument("--ranking_margin", type=float, default=0.0)
    parser.add_argument("--region_pool_blend", type=float, default=0.5)
    parser.add_argument("--region_pos_weight_mode", choices=["none", "balanced"], default="balanced")
    parser.add_argument("--channel_pos_weight_mode", choices=["none", "balanced"], default="none")
    parser.add_argument("--max_pos_weight", type=float, default=5.0)
    parser.add_argument("--tusz_spatial_weight_scale", type=float, default=0.25)
    parser.add_argument("--private_spatial_weight_scale", type=float, default=1.0)
    parser.add_argument("--other_spatial_weight_scale", type=float, default=1.0)
    parser.add_argument("--source_balance", choices=["none", "source"], default="source")
    parser.add_argument("--sampler_weight_cap", type=float, default=10.0)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--patient_region_threshold", type=float, default=0.5)
    parser.add_argument("--patient_channel_threshold", type=float, default=0.5)
    parser.add_argument("--patient_min_regions", type=int, default=1)
    parser.add_argument("--patient_max_regions", type=int, default=3)
    parser.add_argument("--patient_min_channels", type=int, default=1)
    parser.add_argument("--patient_max_channels", type=int, default=8)
    parser.add_argument("--selection_metric", default="region_macro_f1")
    parser.add_argument("--device", default="")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.dataset_format == "vepiset":
        common = dict(
            root=args.vepiset_root,
            val_ratio=float(args.vepiset_val_split),
            test_ratio=float(args.vepiset_test_split),
            seed=int(args.seed),
            max_samples_per_class=int(args.vepiset_max_samples_per_class),
            max_non_ied_samples=int(args.vepiset_max_non_ied_samples),
            target_samples=int(args.vepiset_target_samples),
            normalize=not bool(args.no_vepiset_normalize),
        )
        train_ds = VEPiSetSOZPreDataset(split="train", **common)
        val_ds = VEPiSetSOZPreDataset(split="val", **common)
        print(json.dumps({
            "dataset_format": "vepiset",
            "train": train_ds.split_meta,
            "val": val_ds.split_meta,
        }, indent=2, ensure_ascii=False))
    else:
        train_ds = UnifiedSOZDataset(
            args.preprocessed_dir,
            splits=parse_list(args.train_splits),
            sources=parse_list(args.train_sources) or None,
            exclude_patients=parse_list(args.exclude_patients) or None,
        )
        val_ds = UnifiedSOZDataset(
            args.preprocessed_dir,
            splits=parse_list(args.val_splits),
            sources=parse_list(args.val_sources) or None,
            include_patients=parse_list(args.val_patients) or None,
        )
    result = train_once(train_ds, val_ds, Path(args.output_dir), args)
    print(json.dumps(result["metrics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
