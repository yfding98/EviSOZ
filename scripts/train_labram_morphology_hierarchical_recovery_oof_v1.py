#!/usr/bin/env python3
"""Train one development-only TUEV morphology recovery OOF fold.

No official TUEV evaluation path or threshold argument exists in this entry
point. It consumes only the source-train files bound by the passed CPU
preflight bundle. Run ``--preflight-only`` while another job owns the GPU.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import re
import sys
from typing import Mapping, Sequence


_REQUIRED_CUBLAS_WORKSPACE = ":4096:8"
_observed_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
if _observed_workspace is None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = _REQUIRED_CUBLAS_WORKSPACE
elif _observed_workspace != _REQUIRED_CUBLAS_WORKSPACE:
    raise RuntimeError(
        "Morphology recovery requires CUBLAS_WORKSPACE_CONFIG=':4096:8'"
    )

import numpy as np  # noqa: E402
import torch  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soz.morphology_recovery import (  # noqa: E402
    HierarchicalMorphologyEvidenceHead,
    audit_morphology_recovery_source,
    fit_hierarchical_morphology_weights,
    hierarchical_morphology_group_balanced_loss,
    load_morphology_recovery_preflight,
)
from src.soz.morphology_recovery_oof import (  # noqa: E402
    morphology_recovery_head_state_sha256,
    morphology_recovery_training_config,
    save_morphology_recovery_oof_run,
)


_SHA_RE = re.compile(r"[0-9a-f]{64}")


def _sha256(value: str) -> str:
    text = str(value).strip().lower()
    if not _SHA_RE.fullmatch(text):
        raise argparse.ArgumentTypeError("expected a lowercase SHA256")
    return text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument(
        "--selection", required=True, choices=tuple(f"fold{fold}" for fold in range(5))
    )
    parser.add_argument(
        "--parity-directory",
        type=Path,
        default=Path("outputs/preprocessing_parity_formal_v1_20260809"),
    )
    parser.add_argument(
        "--preflight-bundle",
        type=Path,
        default=Path(
            "outputs/labram_morphology_hierarchical_recovery_preflight_v1_20260810"
        ),
    )
    parser.add_argument(
        "--expected-preflight-receipt-sha256", type=_sha256, required=True
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path(
            "outputs/labram_morphology_hierarchical_recovery_oof_v1_20260810"
        ),
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def _progress(selection: str, stage: str, **values: object) -> None:
    print(
        json.dumps(
            {
                "selection": selection,
                "stage": stage,
                "development_only": True,
                "formal_promotion": False,
                "official_tuev_eval_used": False,
                **values,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _as_tensor(values: object, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    array = np.array(values, copy=True, order="C")
    tensor = torch.from_numpy(array)
    return tensor if dtype is None else tensor.to(dtype=dtype)


def _single_slot_probabilities(output):
    """Remove only the frozen singleton morphology time dimension."""

    if tuple(output.ce6_logits.shape[1:]) != (20, 1, 6) or tuple(
        output.auxiliary_logits.shape[1:]
    ) != (20, 1, 3):
        raise ValueError(
            "Morphology recovery inference requires [B,20,1,6]/[B,20,1,3] logits"
        )
    ce6 = output.ce6_logits.softmax(dim=-1).squeeze(2)
    auxiliary = output.auxiliary_logits.sigmoid().squeeze(2)
    if tuple(ce6.shape[1:]) != (20, 6) or tuple(auxiliary.shape[1:]) != (20, 3):
        raise RuntimeError("Morphology recovery singleton-time collapse failed")
    return ce6, auxiliary


def _weighted_average_precision(
    targets: np.ndarray, scores: np.ndarray, weights: np.ndarray
) -> float:
    targets = np.asarray(targets, dtype=np.bool_)
    scores = np.asarray(scores, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    positive = weights[targets].sum()
    if positive <= 0:
        raise ValueError("Average precision requires positive weighted support")
    order = np.argsort(-scores, kind="stable")
    ordered_targets = targets[order]
    ordered_weights = weights[order]
    true_mass = np.cumsum(ordered_weights * ordered_targets)
    total_mass = np.cumsum(ordered_weights)
    precision = true_mass / np.maximum(total_mass, np.finfo(np.float64).eps)
    return float(
        np.sum(precision * ordered_weights * ordered_targets, dtype=np.float64)
        / positive
    )


def _weighted_auroc(
    targets: np.ndarray, scores: np.ndarray, weights: np.ndarray
) -> float:
    targets = np.asarray(targets, dtype=np.bool_)
    scores = np.asarray(scores, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    positive = weights[targets].sum()
    negative = weights[~targets].sum()
    if positive <= 0 or negative <= 0:
        raise ValueError("AUROC requires positive and negative weighted support")
    order = np.argsort(scores, kind="stable")
    ordered_scores = scores[order]
    ordered_targets = targets[order]
    ordered_weights = weights[order]
    numerator = 0.0
    cumulative_negative = 0.0
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and ordered_scores[stop] == ordered_scores[start]:
            stop += 1
        group_targets = ordered_targets[start:stop]
        group_weights = ordered_weights[start:stop]
        group_positive = group_weights[group_targets].sum()
        group_negative = group_weights[~group_targets].sum()
        numerator += group_positive * (cumulative_negative + 0.5 * group_negative)
        cumulative_negative += group_negative
        start = stop
    return float(numerator / (positive * negative))


def _held_metrics(
    *,
    held_indices: Sequence[int],
    held_groups: Sequence[str],
    group_indices: Mapping[str, Sequence[int]],
    labels: np.ndarray,
    masks: np.ndarray,
    weights: np.ndarray,
    ce6_probabilities: np.ndarray,
    auxiliary_probabilities: np.ndarray,
) -> dict[str, object]:
    indices = np.asarray(held_indices, dtype=np.int64)
    index_row = {int(index): row for row, index in enumerate(indices.tolist())}
    held_labels = np.asarray(labels[indices])
    held_mask = np.asarray(masks[indices])
    held_weights = np.asarray(weights[indices], dtype=np.float64)
    observed_labels = held_labels[held_mask]
    observed_weights = held_weights[held_mask]
    observed_ce6 = ce6_probabilities[held_mask]
    true_probability = observed_ce6[np.arange(len(observed_labels)), observed_labels]
    ce6_nll = float(
        np.average(-np.log(np.maximum(true_probability, 1e-12)), weights=observed_weights)
    )
    one_hot = np.eye(6, dtype=np.float64)[observed_labels]
    ce6_brier = float(
        np.average(np.square(observed_ce6 - one_hot).sum(axis=-1), weights=observed_weights)
    )
    predictions = observed_ce6.argmax(axis=-1)
    class_ap = []
    for class_index, name in enumerate(("SPSW", "GPED", "PLED", "EYEM", "ARTF", "BCKG")):
        truth = observed_labels == class_index
        class_ap.append(
            {
                "class": name,
                "support": float(observed_weights[truth].sum()),
                "average_precision": _weighted_average_precision(
                    truth, observed_ce6[:, class_index], observed_weights
                ),
            }
        )

    local_truth = np.isin(observed_labels, (0, 2))
    local_score = observed_ce6[:, 0] + observed_ce6[:, 2]
    local_positive = float(observed_weights[local_truth].sum())
    local_negative = float(observed_weights[~local_truth].sum())
    local_nll = float(
        np.average(
            -(
                local_truth * np.log(np.maximum(local_score, 1e-12))
                + (~local_truth) * np.log(np.maximum(1.0 - local_score, 1e-12))
            ),
            weights=observed_weights,
        )
    )
    role_truths = (
        local_truth,
        np.isin(observed_labels, (3, 4)),
        observed_labels == 1,
    )
    observed_auxiliary = auxiliary_probabilities[held_mask]
    auxiliary_ap = {
        role: _weighted_average_precision(
            truth, observed_auxiliary[:, role_index], observed_weights
        )
        for role_index, (role, truth) in enumerate(
            zip(("localizing", "artifact", "generalized"), role_truths)
        )
    }

    group_balanced = []
    group_local_ap = []
    for group in held_groups:
        source_indices = list(group_indices[group])
        rows = [index_row[index] for index in source_indices]
        group_mask = np.asarray(masks[source_indices])
        group_labels = np.asarray(labels[source_indices])[group_mask]
        group_weights = np.asarray(weights[source_indices], dtype=np.float64)[group_mask]
        group_ce6 = ce6_probabilities[rows][group_mask]
        group_prediction = group_ce6.argmax(axis=-1)
        recalls = []
        for class_index in range(6):
            truth = group_labels == class_index
            if truth.any():
                recalls.append(
                    float(
                        group_weights[truth & (group_prediction == class_index)].sum()
                        / group_weights[truth].sum()
                    )
                )
        if recalls:
            group_balanced.append(sum(recalls) / len(recalls))
        group_local_truth = np.isin(group_labels, (0, 2))
        if group_local_truth.any() and (~group_local_truth).any():
            group_local_ap.append(
                _weighted_average_precision(
                    group_local_truth,
                    group_ce6[:, 0] + group_ce6[:, 2],
                    group_weights,
                )
            )
    if not group_balanced or not group_local_ap:
        raise RuntimeError("Held morphology fold lacks group-level evaluable support")
    return {
        "held_item_count": len(held_indices),
        "held_group_count": len(held_groups),
        "held_observed_cell_count": int(held_mask.sum()),
        "held_ce6_nll": ce6_nll,
        "held_ce6_brier": ce6_brier,
        "held_ce6_class_average_precision": class_ap,
        "held_ce6_macro_average_precision": float(
            np.mean([row["average_precision"] for row in class_ap])
        ),
        "held_group_macro_balanced_accuracy": float(np.mean(group_balanced)),
        "held_localizing_positive_mass": local_positive,
        "held_localizing_negative_mass": local_negative,
        "held_localizing_average_precision": _weighted_average_precision(
            local_truth, local_score, observed_weights
        ),
        "held_localizing_auroc": _weighted_auroc(
            local_truth, local_score, observed_weights
        ),
        "held_localizing_brier": float(
            np.average(np.square(local_score - local_truth), weights=observed_weights)
        ),
        "held_localizing_nll": local_nll,
        "held_group_macro_localizing_average_precision": float(np.mean(group_local_ap)),
        "held_group_macro_localizing_ap_group_count": len(group_local_ap),
        "held_auxiliary_role_average_precision": auxiliary_ap,
        "thresholds_evaluated": 0,
    }


def _load_and_reaudit(args):
    receipt, preflight_payload = load_morphology_recovery_preflight(
        args.preflight_bundle,
        expected_receipt_sha256=args.expected_preflight_receipt_sha256,
        verify_source_files=True,
    )
    parity = args.parity_directory.resolve(strict=True)
    expected_paths = {
        "run_plan": parity / "run-plan.json",
        "tokens": parity / "arrays" / "tuev_tokens_C-CAR19.npy",
        "labels": parity / "arrays" / "tuev_labels.npy",
        "mask": parity / "arrays" / "tuev_mask.npy",
        "weights": parity / "arrays" / "tuev_weights.npy",
    }
    sources = preflight_payload["source_files"]
    for name, expected in expected_paths.items():
        if Path(str(sources[name]["path"])).resolve(strict=True) != expected.resolve(strict=True):
            raise ValueError(f"Preflight {name} is bound to another parity directory")
    run_plan = json.loads(expected_paths["run_plan"].read_text(encoding="utf-8"))
    arrays = {
        "tokens": np.load(expected_paths["tokens"], mmap_mode="r"),
        "labels": np.load(expected_paths["labels"], mmap_mode="r"),
        "mask": np.load(expected_paths["mask"], mmap_mode="r"),
        "weights": np.load(expected_paths["weights"], mmap_mode="r"),
    }
    current = audit_morphology_recovery_source(
        run_plan=run_plan,
        tokens=arrays["tokens"],
        labels=arrays["labels"],
        source_target_mask=arrays["mask"],
        overlap_component_weights=arrays["weights"],
    )
    if current.canonical_payload != receipt.canonical_payload:
        raise ValueError("Morphology recovery source changed after preflight")
    return receipt, preflight_payload, run_plan, arrays


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    fold = int(args.selection[-1])
    config = morphology_recovery_training_config(fold)
    _progress(args.selection, "load_and_reaudit_source_train_preflight")
    receipt, preflight_payload, run_plan, arrays = _load_and_reaudit(args)
    items = run_plan["tuev_items"]
    group_indices: dict[str, list[int]] = {}
    group_fold: dict[str, int] = {}
    for item in items:
        group = str(item["parent_group_id"])
        group_indices.setdefault(group, []).append(int(item["index"]))
        previous = group_fold.setdefault(group, int(item["fold"]))
        if previous != int(item["fold"]):
            raise ValueError("One morphology group crosses folds after preflight")
    for values in group_indices.values():
        values.sort()
    fit_groups = tuple(sorted(group for group, value in group_fold.items() if value != fold))
    held_groups = tuple(sorted(group for group, value in group_fold.items() if value == fold))
    fit_indices = tuple(index for group in fit_groups for index in group_indices[group])
    held_indices = tuple(sorted(index for group in held_groups for index in group_indices[group]))
    _progress(
        args.selection,
        "selection_preflight_passed",
        fit_group_count=len(fit_groups),
        held_group_count=len(held_groups),
        fit_item_count=len(fit_indices),
        held_item_count=len(held_indices),
        observed_cell_count=receipt.observed_cell_count,
        unknown_cell_count=receipt.unknown_cell_count,
    )
    if args.preflight_only:
        return 0

    planned_output = args.output_directory.resolve() / args.selection
    if os.path.lexists(planned_output):
        raise FileExistsError(
            f"Morphology recovery output already exists: {planned_output}"
        )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    labels = arrays["labels"]
    masks = arrays["mask"]
    weights = arrays["weights"]
    fit_labels = _as_tensor(labels[list(fit_indices)], dtype=torch.long).unsqueeze(-1)
    fit_masks = _as_tensor(masks[list(fit_indices)]).bool().unsqueeze(-1)
    fit_weights = _as_tensor(weights[list(fit_indices)]).float().unsqueeze(-1)
    fitted = fit_hierarchical_morphology_weights(
        fit_labels, fit_masks, fit_weights, cap=10.0
    )

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    head = HierarchicalMorphologyEvidenceHead(token_dim=200, hidden_dim=128)
    initial_state_sha = morphology_recovery_head_state_sha256(head)
    head.to(device)
    ce6_class_weights = fitted.ce6_class_weights.to(device)
    auxiliary_pos_weights = fitted.auxiliary_pos_weights.to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    epoch_losses = []
    generator = random.Random(seed)
    try:
        torch.use_deterministic_algorithms(True)
        for epoch in range(int(config["fixed_epochs"])):
            order = list(fit_groups)
            generator.shuffle(order)
            head.train()
            group_losses = []
            for group in order:
                indices = group_indices[group]
                denominator = float(
                    np.asarray(weights[indices], dtype=np.float64)[
                        np.asarray(masks[indices], dtype=np.bool_)
                    ].sum()
                )
                if denominator <= 0:
                    raise RuntimeError("Morphology fit group has no effective target mass")
                optimizer.zero_grad(set_to_none=True)
                group_loss = 0.0
                microbatch = int(config["crop_microbatch_size"])
                for start in range(0, len(indices), microbatch):
                    selection = indices[start : start + microbatch]
                    batch_tokens = _as_tensor(arrays["tokens"][selection]).float().to(device)
                    batch_labels = _as_tensor(labels[selection], dtype=torch.long).unsqueeze(-1).to(device)
                    batch_masks = _as_tensor(masks[selection]).bool().unsqueeze(-1).to(device)
                    batch_weights = _as_tensor(weights[selection]).float().unsqueeze(-1).to(device)
                    output = head(batch_tokens)
                    micro_loss = hierarchical_morphology_group_balanced_loss(
                        output,
                        batch_labels,
                        batch_masks,
                        batch_weights,
                        torch.zeros(len(selection), dtype=torch.long, device=device),
                        ce6_class_weights=ce6_class_weights,
                        auxiliary_pos_weights=auxiliary_pos_weights,
                    )
                    micro_mass = float(batch_weights[batch_masks].sum().detach().cpu())
                    scaled = micro_loss * (micro_mass / denominator)
                    scaled.backward()
                    group_loss += float(scaled.detach().cpu())
                torch.nn.utils.clip_grad_norm_(
                    head.parameters(), float(config["gradient_clip_norm"])
                )
                optimizer.step()
                group_losses.append(group_loss)
            mean_loss = float(sum(group_losses) / len(group_losses))
            if not math.isfinite(mean_loss):
                raise RuntimeError("Morphology recovery produced a non-finite epoch loss")
            epoch_losses.append(mean_loss)
            _progress(
                args.selection,
                "train_epoch",
                epoch=epoch,
                mean_group_loss=mean_loss,
            )
    finally:
        torch.use_deterministic_algorithms(previous_determinism)

    head.eval()
    ce6_rows = []
    auxiliary_rows = []
    with torch.inference_mode():
        for start in range(0, len(held_indices), 128):
            selection = held_indices[start : start + 128]
            batch = _as_tensor(arrays["tokens"][list(selection)]).float().to(device)
            output = head(batch)
            ce6, auxiliary = _single_slot_probabilities(output)
            ce6_rows.append(ce6.cpu())
            auxiliary_rows.append(auxiliary.cpu())
    ce6_probabilities = torch.cat(ce6_rows)
    auxiliary_probabilities = torch.cat(auxiliary_rows)
    metrics = _held_metrics(
        held_indices=held_indices,
        held_groups=held_groups,
        group_indices=group_indices,
        labels=labels,
        masks=masks,
        weights=weights,
        ce6_probabilities=ce6_probabilities.numpy(),
        auxiliary_probabilities=auxiliary_probabilities.numpy(),
    )
    head.cpu()
    output_root = args.output_directory.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    source_files_sha = {
        name: str(row["sha256"])
        for name, row in preflight_payload["source_files"].items()
    }
    saved = save_morphology_recovery_oof_run(
        output_root / args.selection,
        selection=args.selection,
        head=head,
        held_item_indices=torch.tensor(held_indices, dtype=torch.long),
        ce6_probabilities=ce6_probabilities,
        auxiliary_probabilities=auxiliary_probabilities,
        fit_group_ids=fit_groups,
        held_group_ids=held_groups,
        preflight_receipt_sha256=receipt.receipt_sha256,
        source_plan_sha256=receipt.source_plan_sha256,
        source_files_sha256=source_files_sha,
        ce6_class_weights=tuple(float(value) for value in fitted.ce6_class_weights),
        auxiliary_pos_weights=tuple(float(value) for value in fitted.auxiliary_pos_weights),
        epoch_group_mean_losses=tuple(epoch_losses),
        metrics=metrics,
        initial_head_state_sha256=initial_state_sha,
    )
    _progress(
        args.selection,
        "atomic_artifact_saved",
        output=str(saved.path),
        manifest_file_sha256=saved.manifest_file_sha256,
        held_localizing_average_precision=metrics[
            "held_localizing_average_precision"
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
