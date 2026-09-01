#!/usr/bin/env python3
"""Run the fixed LaBraM V-directed endpoint-router source-train OOF screen."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_labram_temporal_mil_nested_oof_v1 import (  # noqa: E402
    _canonical_bytes,
    _file_sha256,
    _indices_for_folds,
    _load_source_train,
    _metrics,
    _subset,
    _tensor_state_sha256,
)
from src.soz.aggregation import aggregate_patient_logits  # noqa: E402
from src.soz.geometry import (  # noqa: E402
    CHANNEL_INDEX,
    STANDARD_19,
    TCP_20_EDGES,
)
from src.soz.metrics import DEEPSOZ_STANDARD19_NEIGHBORS  # noqa: E402
from src.soz.safe_anchor_h_recovery import (  # noqa: E402
    within_tcp_edge_direction_metrics,
)
from src.soz.temporal_mil_recovery import (  # noqa: E402
    TemporalMILPatientBatch,
    jeffreys_channel_prior_logits,
    temporal_mil_objective,
)
from src.soz.v_directed_endpoint_recovery import (  # noqa: E402
    V_DIRECTED_ENDPOINT_RECOVERY_SCHEMA,
    VDirectedEndpointTemporalMILReasoner,
)


PROTOCOL_PATH = (
    ROOT
    / "research/02_method/"
    "labram_v_directed_endpoint_router_protocol_v5_20260811_zh.md"
)
MODULE_PATH = ROOT / "src/soz/v_directed_endpoint_recovery.py"
TEMPORAL_ANCHOR_PATH = (
    ROOT / "outputs/labram_temporal_mil_nested_oof_v1_20260810"
)
TEMPORAL_ANCHOR_MANIFEST_SHA256 = (
    "58cbfcc3d25e8ff4b13ab93e388e8aa5691e1c8fc9dc515ec2e8b51b226c9811"
)
TEMPORAL_ANCHOR_PREDICTION_SHA256 = (
    "9373dc6bf269002c812ae26ca6ea8365b7518d3396037c4fc5b3a67603e1211d"
)

OUTER_FOLDS = tuple(range(5))
EPOCHS = 100
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 1e-2
MAX_GRAD_NORM = 1.0
BOOTSTRAP_REPLICATES = 2000
BASE_SEED = 20260811


def _seeded_model(
    prior: torch.Tensor,
    *,
    seed: int,
    device: torch.device,
) -> VDirectedEndpointTemporalMILReasoner:
    fork_devices: list[int] = []
    if device.type == "cuda":
        fork_devices = [device.index if device.index is not None else 0]
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(seed)
        model = VDirectedEndpointTemporalMILReasoner(prior).to(device)
    return model


def _fit_router(
    train: TemporalMILPatientBatch,
    *,
    seed: int,
    device: torch.device,
) -> tuple[VDirectedEndpointTemporalMILReasoner, dict[str, object]]:
    prior = jeffreys_channel_prior_logits(train).detach().cpu()
    model = _seeded_model(prior, seed=seed, device=device)
    parameter_count = model.n_trainable_parameters
    if parameter_count >= 500:
        raise RuntimeError("router violated the frozen capacity gate")
    batch = train.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    first: dict[str, float] | None = None
    last: dict[str, float] | None = None
    for _ in range(EPOCHS):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(batch.evidence)
        loss = temporal_mil_objective(
            output.event_logits,
            batch,
            neighbor_weight=0.0,
        )
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        row = {
            "total": float(loss.total.detach().cpu()),
            "exact_set_mass": float(loss.exact_set_mass.detach().cpu()),
            "pairwise": float(loss.pairwise.detach().cpu()),
            "bce": float(loss.bce.detach().cpu()),
            "consistency": float(loss.consistency.detach().cpu()),
        }
        if first is None:
            first = row
        last = row
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    model.eval()
    model.requires_grad_(False)
    assert first is not None and last is not None
    return model, {
        "seed": seed,
        "epochs": EPOCHS,
        "first_epoch": first,
        "final_epoch": last,
        "parameter_count": parameter_count,
        "endpoint_scale": float(
            torch.nn.functional.softplus(model.raw_endpoint_scale).detach().cpu()
        ),
    }


def _predict_router(
    model: VDirectedEndpointTemporalMILReasoner,
    batch: TemporalMILPatientBatch,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, object]]:
    moved = batch.to(device)
    model.eval()
    with torch.no_grad():
        output = model(moved.evidence)
        patient_logits = aggregate_patient_logits(
            output.event_logits,
            moved.event_patient_index,
        ).logits

    edge_valid = moved.evidence.ictal_mask & moved.evidence.phase_mask.unsqueeze(1)
    endpoint_valid_count = output.endpoint_valid.sum(dim=-1)
    probability = output.endpoint_probability
    entropy = -(
        probability.clamp_min(1e-8) * probability.clamp_min(1e-8).log()
    ).sum(dim=-1)
    selected_entropy = entropy[edge_valid]
    selected_direction = (probability[..., 0] - probability[..., 1]).abs()[edge_valid]
    return patient_logits.detach().cpu(), {
        "event_count": batch.evidence.batch_size,
        "valid_edge_tile_count": int(edge_valid.sum().item()),
        "both_v_endpoints_valid_count": int(
            (edge_valid & (endpoint_valid_count == 2)).sum().item()
        ),
        "one_v_endpoint_valid_count": int(
            (edge_valid & (endpoint_valid_count == 1)).sum().item()
        ),
        "neither_v_endpoint_valid_count": int(
            (edge_valid & (endpoint_valid_count == 0)).sum().item()
        ),
        "mean_endpoint_entropy": (
            float(selected_entropy.mean().detach().cpu())
            if selected_entropy.numel()
            else None
        ),
        "mean_absolute_endpoint_probability_difference": (
            float(selected_direction.mean().detach().cpu())
            if selected_direction.numel()
            else None
        ),
        "endpoint_scale": float(output.endpoint_scale.detach().cpu()),
    }


def _load_temporal_anchor(
    full: TemporalMILPatientBatch,
    patient_folds: Sequence[int],
) -> tuple[torch.Tensor, dict[str, object]]:
    manifest_path = TEMPORAL_ANCHOR_PATH / "manifest.json"
    prediction_path = TEMPORAL_ANCHOR_PATH / "oof_predictions.safetensors"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("temporal anchor manifest is unavailable")
    if not prediction_path.is_file() or prediction_path.is_symlink():
        raise ValueError("temporal anchor predictions are unavailable")
    raw = manifest_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != TEMPORAL_ANCHOR_MANIFEST_SHA256:
        raise ValueError("temporal anchor manifest changed")
    if _file_sha256(prediction_path) != TEMPORAL_ANCHOR_PREDICTION_SHA256:
        raise ValueError("temporal anchor predictions changed")
    manifest = json.loads(raw)
    if tuple(manifest.get("patient_ids", ())) != full.patient_ids:
        raise ValueError("temporal anchor patient roster differs")
    if tuple(manifest.get("patient_folds", ())) != tuple(patient_folds):
        raise ValueError("temporal anchor patient folds differ")
    if manifest.get("private_used") is not False or manifest.get(
        "source_eval_used"
    ) is not False:
        raise ValueError("temporal anchor violates the development boundary")
    try:
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("safetensors is required for anchor loading") from exc
    tensors = load_file(str(prediction_path), device="cpu")
    anchor = tensors.get("temporal_mil_exact")
    if anchor is None or tuple(anchor.shape) != (len(full.patient_ids), 19) or (
        anchor.dtype != torch.float32
    ):
        raise ValueError("temporal exact anchor tensor schema changed")
    if not torch.equal(tensors.get("targets"), full.targets.cpu()) or not torch.equal(
        tensors.get("target_mask"), full.target_mask.cpu()
    ):
        raise ValueError("temporal anchor target carrier differs")
    expected_folds = torch.tensor(patient_folds, dtype=torch.int64)
    if not torch.equal(tensors.get("patient_folds"), expected_folds):
        raise ValueError("temporal anchor fold tensor differs")
    return anchor.contiguous(), manifest


def _paired_patient_bootstrap(
    candidate: torch.Tensor,
    anchor: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, object]:
    rows: list[torch.Tensor] = []
    for patient in range(targets.shape[0]):
        index = slice(patient, patient + 1)
        candidate_metrics = _metrics(candidate[index], targets[index], mask[index])
        anchor_metrics = _metrics(anchor[index], targets[index], mask[index])
        rows.append(
            torch.tensor(
                [
                    candidate_metrics["top1"]["strict_accuracy"]
                    - anchor_metrics["top1"]["strict_accuracy"],
                    candidate_metrics["top1"]["relaxed_accuracy"]
                    - anchor_metrics["top1"]["relaxed_accuracy"],
                    candidate_metrics["ranking"]["macro_average_precision"]
                    - anchor_metrics["ranking"]["macro_average_precision"],
                    candidate_metrics["ranking"]["mean_reciprocal_rank"]
                    - anchor_metrics["ranking"]["mean_reciprocal_rank"],
                    candidate_metrics["ranking"]["hit_at_k"][3]
                    - anchor_metrics["ranking"]["hit_at_k"][3],
                    candidate_metrics["ranking"]["hit_at_k"][5]
                    - anchor_metrics["ranking"]["hit_at_k"][5],
                ],
                dtype=torch.float64,
            )
        )
    row_deltas = torch.stack(rows)
    generator = torch.Generator().manual_seed(BASE_SEED)
    indices = torch.randint(
        0,
        targets.shape[0],
        (BOOTSTRAP_REPLICATES, targets.shape[0]),
        generator=generator,
    )
    samples = row_deltas[indices].mean(dim=1)
    point = row_deltas.mean(dim=0)
    names = (
        "strict_top1",
        "relaxed_top1",
        "macro_ap",
        "mrr",
        "hit_at_3",
        "hit_at_5",
    )
    return {
        name: {
            "delta": float(point[column]),
            "ci95": [
                float(torch.quantile(samples[:, column], 0.025)),
                float(torch.quantile(samples[:, column], 0.975)),
            ],
            "patient_improved_count": int((row_deltas[:, column] > 0).sum()),
            "patient_worsened_count": int((row_deltas[:, column] < 0).sum()),
            "patient_equal_count": int((row_deltas[:, column] == 0).sum()),
        }
        for column, name in enumerate(names)
    }


def _tcp_neighbour_set(positive_indices: torch.Tensor) -> set[int]:
    positives = {int(index) for index in positive_indices.tolist()}
    neighbours: set[int] = set()
    for left, right in TCP_20_EDGES:
        left_index = CHANNEL_INDEX[left]
        right_index = CHANNEL_INDEX[right]
        if left_index in positives:
            neighbours.add(right_index)
        if right_index in positives:
            neighbours.add(left_index)
    return neighbours - positives


def _top1_states(
    scores: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[list[str], int]:
    states: list[str] = []
    tie_count = 0
    for patient in range(scores.shape[0]):
        evaluable = mask[patient]
        row = scores[patient].masked_fill(~evaluable, -torch.inf)
        top_set = torch.nonzero(row == row.max(), as_tuple=False).flatten()
        if top_set.numel() != 1:
            states.append("tie")
            tie_count += 1
            continue
        predicted = int(top_set.item())
        positive = torch.nonzero(
            (targets[patient] == 1) & evaluable,
            as_tuple=False,
        ).flatten()
        positives = {int(index) for index in positive.tolist()}
        if predicted in positives:
            states.append("exact")
            continue
        if predicted in _tcp_neighbour_set(positive):
            states.append("tcp_neighbour_only")
            continue
        official: set[int] = set()
        if len(positives) <= 4:
            for index in positives:
                official.update(DEEPSOZ_STANDARD19_NEIGHBORS[index])
            official = {
                index for index in official if bool(evaluable[index])
            } - positives
        if predicted in official:
            states.append("official_non_tcp_neighbour_only")
        else:
            states.append("far")
    return states, tie_count


def _transition_diagnostic(
    candidate: torch.Tensor,
    anchor: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, object]:
    candidate_state, candidate_ties = _top1_states(candidate, targets, mask)
    anchor_state, anchor_ties = _top1_states(anchor, targets, mask)
    transition_counts: dict[str, int] = {}
    for before, after in zip(anchor_state, candidate_state):
        key = f"{before}->{after}"
        transition_counts[key] = transition_counts.get(key, 0) + 1
    rescue = transition_counts.get("tcp_neighbour_only->exact", 0)
    loss = transition_counts.get("exact->tcp_neighbour_only", 0)
    return {
        "anchor_state_counts": {
            state: anchor_state.count(state) for state in sorted(set(anchor_state))
        },
        "candidate_state_counts": {
            state: candidate_state.count(state)
            for state in sorted(set(candidate_state))
        },
        "transitions": dict(sorted(transition_counts.items())),
        "tcp_to_exact_rescue_count": rescue,
        "exact_to_tcp_loss_count": loss,
        "rescue_exceeds_loss": rescue > loss,
        "anchor_far_count": anchor_state.count("far"),
        "candidate_far_count": candidate_state.count("far"),
        "far_errors_nonincreasing": (
            candidate_state.count("far") <= anchor_state.count("far")
        ),
        "anchor_top_tie_patient_count": anchor_ties,
        "candidate_top_tie_patient_count": candidate_ties,
    }


def _direction_payload(report) -> dict[str, object]:
    return {
        "patient_macro_accuracy": report.patient_macro_accuracy,
        "eligible_patient_count": report.eligible_patient_count,
        "informative_pair_count": report.informative_pair_count,
        "semantics": (
            "fixed TCP-20 edge with exactly one observed positive endpoint; "
            "patient-macro; ties=0.5; diagnostic only"
        ),
    }


def _run(
    full: TemporalMILPatientBatch,
    patient_folds: tuple[int, ...],
    anchor: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[
    dict[str, object],
    dict[str, torch.Tensor],
    VDirectedEndpointTemporalMILReasoner,
]:
    patients = len(full.patient_ids)
    router_oof = torch.full((patients, 19), torch.nan, dtype=torch.float32)
    outer_rows: list[dict[str, object]] = []
    directionally_nonlower_folds = 0

    for outer_fold in OUTER_FOLDS:
        train_indices = _indices_for_folds(
            patient_folds,
            tuple(fold for fold in OUTER_FOLDS if fold != outer_fold),
        )
        held_indices = _indices_for_folds(patient_folds, (outer_fold,))
        train = _subset(full, train_indices)
        held = _subset(full, held_indices)
        model, fit = _fit_router(
            train,
            seed=BASE_SEED + outer_fold * 1000,
            device=device,
        )
        logits, routing = _predict_router(model, held, device=device)
        router_oof[list(held_indices)] = logits

        held_index = torch.tensor(held_indices, dtype=torch.long)
        anchor_held = anchor.index_select(0, held_index)
        candidate_direction = within_tcp_edge_direction_metrics(
            logits,
            held.targets,
            held.target_mask,
        )
        anchor_direction = within_tcp_edge_direction_metrics(
            anchor_held,
            held.targets,
            held.target_mask,
        )
        direction_nonlower = (
            candidate_direction.patient_macro_accuracy
            >= anchor_direction.patient_macro_accuracy
        )
        directionally_nonlower_folds += int(direction_nonlower)
        outer_rows.append(
            {
                "outer_fold": outer_fold,
                "train_patient_count": len(train_indices),
                "held_patient_count": len(held_indices),
                "fit": fit,
                "routing": routing,
                "candidate_metrics": _metrics(
                    logits,
                    held.targets,
                    held.target_mask,
                ),
                "anchor_metrics": _metrics(
                    anchor_held,
                    held.targets,
                    held.target_mask,
                ),
                "candidate_within_tcp_direction": _direction_payload(
                    candidate_direction
                ),
                "anchor_within_tcp_direction": _direction_payload(anchor_direction),
                "direction_nonlower": direction_nonlower,
            }
        )
        print(
            json.dumps(
                {
                    "stage": "outer_complete",
                    "outer_fold": outer_fold,
                    "candidate_strict": outer_rows[-1]["candidate_metrics"]["top1"][
                        "strict_accuracy"
                    ],
                    "candidate_direction": candidate_direction.patient_macro_accuracy,
                    "anchor_direction": anchor_direction.patient_macro_accuracy,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if not torch.isfinite(router_oof).all():
        raise RuntimeError("router OOF left a patient prediction unfilled")

    candidate_metrics = _metrics(router_oof, full.targets, full.target_mask)
    anchor_metrics = _metrics(anchor, full.targets, full.target_mask)
    candidate_direction = within_tcp_edge_direction_metrics(
        router_oof,
        full.targets,
        full.target_mask,
    )
    anchor_direction = within_tcp_edge_direction_metrics(
        anchor,
        full.targets,
        full.target_mask,
    )
    transitions = _transition_diagnostic(
        router_oof,
        anchor,
        full.targets,
        full.target_mask,
    )

    strict_nonlower = candidate_metrics["top1"]["strict_accuracy"] >= (
        anchor_metrics["top1"]["strict_accuracy"]
    )
    ap_nonlower = candidate_metrics["ranking"]["macro_average_precision"] >= (
        anchor_metrics["ranking"]["macro_average_precision"]
    )
    direction_higher = candidate_direction.patient_macro_accuracy > (
        anchor_direction.patient_macro_accuracy
    )
    fold_consistency = directionally_nonlower_folds >= 3
    gate_checks = {
        "strict_top1_nonlower": strict_nonlower,
        "macro_ap_nonlower": ap_nonlower,
        "within_tcp_direction_strictly_higher": direction_higher,
        "tcp_to_exact_rescues_exceed_exact_to_tcp_losses": transitions[
            "rescue_exceeds_loss"
        ],
        "far_errors_nonincreasing": transitions["far_errors_nonincreasing"],
        "direction_nonlower_in_at_least_3_of_5_folds": fold_consistency,
    }
    gate_pass = all(gate_checks.values())

    final_model, final_fit = _fit_router(
        full,
        seed=BASE_SEED + 99999,
        device=device,
    )
    final_model = final_model.cpu()
    result = {
        "screen_kind": (
            "post_hoc_source_train_patient_oof_mechanism_recovery;"
            "architecture_fixed_before_this_run_but_informed_by_prior_oof_errors"
        ),
        "metrics": {
            "temporal_mil_exact_anchor": anchor_metrics,
            "v_directed_endpoint_router": candidate_metrics,
        },
        "candidate_minus_anchor_paired_patient_bootstrap": (
            _paired_patient_bootstrap(
                router_oof,
                anchor,
                full.targets,
                full.target_mask,
            )
        ),
        "within_tcp_direction": {
            "anchor": _direction_payload(anchor_direction),
            "candidate": _direction_payload(candidate_direction),
            "candidate_minus_anchor": (
                candidate_direction.patient_macro_accuracy
                - anchor_direction.patient_macro_accuracy
            ),
        },
        "top1_transition_diagnostic": transitions,
        "outer_folds": outer_rows,
        "directionally_nonlower_fold_count": directionally_nonlower_folds,
        "frozen_go_no_go_gate": {
            "checks": gate_checks,
            "pass": gate_pass,
            "status": (
                "go_candidate_for_later_locked_validation"
                if gate_pass
                else "no_go_keep_temporal_mil_exact"
            ),
            "interpretation": (
                "post-hoc source-train development point-estimate gate; not a "
                "statistical noninferiority or external-generalization claim"
            ),
        },
        "final_full_source_train_fit": final_fit,
    }
    tensors = {
        "v_directed_endpoint_router": router_oof.contiguous(),
        "temporal_mil_exact_anchor": anchor.contiguous(),
        "targets": full.targets.detach().cpu().contiguous(),
        "target_mask": full.target_mask.detach().cpu().contiguous(),
        "patient_folds": torch.tensor(patient_folds, dtype=torch.int64),
    }
    return result, tensors, final_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.preflight_only and args.output_directory is None:
        raise ValueError("full OOF run requires --output-directory")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    full, patient_folds, lineage = _load_source_train()
    anchor, anchor_manifest = _load_temporal_anchor(full, patient_folds)
    fold_counts = {
        str(fold): sum(value == fold for value in patient_folds)
        for fold in OUTER_FOLDS
    }
    preflight = {
        "status": "ready_fixed_source_train_patient_oof",
        "schema_version": V_DIRECTED_ENDPOINT_RECOVERY_SCHEMA,
        "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)),
        "protocol_sha256": _file_sha256(PROTOCOL_PATH),
        "device": str(device),
        "patient_count": len(full.patient_ids),
        "event_count": full.evidence.batch_size,
        "fold_counts": fold_counts,
        "lineage": {
            **lineage,
            "temporal_anchor_manifest_sha256": TEMPORAL_ANCHOR_MANIFEST_SHA256,
            "temporal_anchor_prediction_sha256": (
                TEMPORAL_ANCHOR_PREDICTION_SHA256
            ),
            "runner_sha256": _file_sha256(Path(__file__).resolve()),
            "router_module_sha256": _file_sha256(MODULE_PATH),
        },
        "config": {
            "outer_folds": list(OUTER_FOLDS),
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "max_grad_norm": MAX_GRAD_NORM,
            "base_seed": BASE_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "candidate_count": 1,
            "inner_architecture_selection": False,
            "neighbor_training_auxiliary": False,
            "patient_event_pooling": "equal_event_logit_mean",
        },
        "foundation_backbone": "official_pretrained_LaBraM_frozen_not_replaced",
        "foundation_trainable_parameter_count": 0,
        "source_dev_forward_count": 0,
        "source_dev_target_values_reachable": False,
        "source_eval_used": False,
        "private_used": False,
        "formal_promotion": False,
    }
    if args.preflight_only:
        print(json.dumps(preflight, sort_keys=True), flush=True)
        return 0

    output = Path(os.path.abspath(args.output_directory))
    if output.name in {"", ".", ".."} or os.path.lexists(output):
        raise FileExistsError(f"output already exists or is invalid: {output}")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise ValueError("output parent must be a regular directory")
    for source in (
        PROTOCOL_PATH,
        MODULE_PATH,
        TEMPORAL_ANCHOR_PATH,
    ):
        resolved = source.resolve(strict=True)
        if output == resolved or output in resolved.parents or resolved in output.parents:
            raise ValueError("output path overlaps an input")

    result, tensors, final_model = _run(
        full,
        patient_folds,
        anchor,
        device=device,
    )
    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("safetensors is required for OOF publication") from exc
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    published = False
    try:
        prediction_path = temporary / "oof_predictions.safetensors"
        save_file(tensors, str(prediction_path))
        final_state = {
            name: value.detach().cpu().contiguous()
            for name, value in final_model.state_dict().items()
        }
        checkpoint_path = temporary / "final_checkpoint.safetensors"
        save_file(final_state, str(checkpoint_path))
        manifest = {
            **preflight,
            "status": "completed_post_hoc_source_train_patient_oof",
            "patient_ids": list(full.patient_ids),
            "patient_folds": list(patient_folds),
            "result": result,
            "files": {
                "oof_predictions.safetensors": {
                    "sha256": _file_sha256(prediction_path),
                    "size_bytes": prediction_path.stat().st_size,
                },
                "final_checkpoint.safetensors": {
                    "sha256": _file_sha256(checkpoint_path),
                    "size_bytes": checkpoint_path.stat().st_size,
                    "state_sha256": _tensor_state_sha256(final_state),
                },
            },
            "scientific_boundary": {
                "foundation_backbone": "official_pretrained_LaBraM_frozen",
                "foundation_replaced": False,
                "foundation_trainable_parameter_count": 0,
                "router_is_discriminative_not_physiological_endpoint_posterior": True,
                "ictal_semantics": (
                    "retrospective_scalp_visible_involvement_not_soz"
                ),
                "evolution_semantics": (
                    "observable_descriptors_not_propagation_or_origin_truth"
                ),
                "attention_semantics": (
                    "discriminative_temporal_weight_not_onset_or_propagation"
                ),
                "post_hoc_after_prior_source_train_oof_error_analysis": True,
                "source_dev_used": False,
                "source_eval_used": False,
                "private_used": False,
                "formal_promotion": False,
            },
            "anchor_reported_final_candidate": anchor_manifest.get("result", {}).get(
                "final_candidate"
            ),
        }
        raw = _canonical_bytes(manifest)
        (temporary / "manifest.json").write_bytes(raw)
        os.rename(temporary, output)
        published = True
        print(
            json.dumps(
                {
                    "status": manifest["status"],
                    "output_directory": str(output),
                    "metrics": result["metrics"],
                    "within_tcp_direction": result["within_tcp_direction"],
                    "top1_transition_diagnostic": result[
                        "top1_transition_diagnostic"
                    ],
                    "frozen_go_no_go_gate": result["frozen_go_no_go_gate"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
