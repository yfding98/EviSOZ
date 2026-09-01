#!/usr/bin/env python3
"""Run the single frozen LaBraM anchor-constrained endpoint reranker v9."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_labram_endpoint_aligned_peft_oof_v8 import (  # noqa: E402
    DEFAULT_PREFIX_CACHE,
    DEFAULT_PREFIX_CACHE_MANIFEST_SHA256,
    DEFAULT_SOURCE_TRAIN_IV,
    DEFAULT_SOURCE_TRAIN_IV_MANIFEST_SHA256,
    DEFAULT_TARGET_SCOPE,
    TEMPORAL_ANCHOR,
    _load_access_audit,
    _load_fixed_comparators,
    _load_inputs,
)
from scripts.run_labram_temporal_mil_nested_oof_v1 import (  # noqa: E402
    _file_sha256,
    _metrics,
    _tensor_state_sha256,
)
from scripts.run_labram_v_directed_endpoint_oof_v5 import (  # noqa: E402
    _paired_patient_bootstrap,
    _top1_states,
    _transition_diagnostic,
)
from src.soz.anchor_constrained_endpoint_reranker import (  # noqa: E402
    ANCHOR_CONSTRAINED_ENDPOINT_RERANKER_SCHEMA,
    ENDPOINT_FLIP_LOGIT_MARGIN,
    ENDPOINT_L2_WEIGHT,
    ENDPOINT_LBFGS_MAX_ITER,
    AnchorConstrainedEndpointReranker,
    anchor_constrained_endpoint_objective,
    apply_fixed_selective_endpoint_rerank,
    build_deepsoz_exact_endpoint_training_pairs,
    propose_anchor_adjacent_endpoint,
)
from src.soz.anchor_endpoint_features import (  # noqa: E402
    ENDPOINT_NODE_FEATURE_DIM,
    H_FEATURE_SLICE,
    I_FEATURE_SLICE,
    Q_FEATURE_SLICE,
    V_FEATURE_SLICE,
    endpoint_adjacency_edges,
    fit_fold_endpoint_features,
)
from src.soz.development_reasoner_training_v1_1 import (  # noqa: E402
    FROZEN_SOURCE_TRAIN_TARGET_SCOPE_RECEIPT_SHA256,
)
from src.soz.geometry import (  # noqa: E402
    CHANNEL_INDEX,
    N_STANDARD_CHANNELS,
    TCP_20_EDGES,
)
from src.soz.safe_anchor_h_recovery import (  # noqa: E402
    within_tcp_edge_direction_metrics,
)


SCHEMA_VERSION = "soz_labram_anchor_constrained_endpoint_reranker_oof_v9"
PROTOCOL_PATH = (
    ROOT
    / "research/02_method/"
    "labram_anchor_constrained_endpoint_reranker_protocol_v9_20260811_zh.md"
)
FEATURE_MODULE_PATH = ROOT / "src/soz/anchor_endpoint_features.py"
RERANKER_MODULE_PATH = ROOT / "src/soz/anchor_constrained_endpoint_reranker.py"
RUNNER_PATH = Path(__file__).resolve()
DEFAULT_OUTPUT = (
    ROOT / "outputs/labram_anchor_constrained_endpoint_reranker_oof_v9_20260811"
)
OUTER_FOLDS = tuple(range(5))
BASE_SEED = 20260811
BOOTSTRAP_REPLICATES = 2000
EXPECTED_ANCHOR_STRICT_HITS = 42
EXPECTED_ANCHOR_RELAXED_HITS = 55
EXPECTED_ANCHOR_MACRO_AP = 0.6328
EXPECTED_ANCHOR_FAR = 10
EXPECTED_HIT_ATOL = 1e-5


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _scope_sha256(values: Sequence[object]) -> str:
    return hashlib.sha256(_canonical_bytes(list(values))).hexdigest()


def _patient_indices_for_fold(
    patient_folds: Sequence[int], fold: int, *, held: bool
) -> tuple[int, ...]:
    return tuple(
        index
        for index, value in enumerate(patient_folds)
        if (int(value) == int(fold)) is held
    )


def _subset_rows(value: torch.Tensor, indices: Sequence[int]) -> torch.Tensor:
    return value.index_select(0, torch.tensor(tuple(indices), dtype=torch.long))


def _fit_reranker(
    node_features: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    patient_ids: Sequence[str],
    *,
    seed: int,
) -> tuple[AnchorConstrainedEndpointReranker, dict[str, object]]:
    torch.manual_seed(int(seed))
    model = AnchorConstrainedEndpointReranker()
    with torch.no_grad():
        model.endpoint_utility.weight.zero_()
    batch = build_deepsoz_exact_endpoint_training_pairs(
        node_features,
        targets,
        target_mask,
        patient_ids,
    )
    if model.n_trainable_parameters != ENDPOINT_NODE_FEATURE_DIM:
        raise RuntimeError("v9 reranker parameter count changed")
    initial = anchor_constrained_endpoint_objective(model, batch)
    closure_count = 0
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=ENDPOINT_LBFGS_MAX_ITER,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        nonlocal closure_count
        closure_count += 1
        optimizer.zero_grad(set_to_none=True)
        objective = anchor_constrained_endpoint_objective(model, batch)
        objective.total.backward()
        return objective.total

    optimizer.step(closure)
    optimizer.zero_grad(set_to_none=True)
    final = anchor_constrained_endpoint_objective(model, batch)
    if not torch.isfinite(final.total) or float(final.total) > float(initial.total) + 1e-6:
        raise RuntimeError("v9 deterministic optimization did not reduce its objective")
    with torch.no_grad():
        original = model(batch.endpoint_features)
        swapped = model(batch.endpoint_features.flip(1))
    if not torch.allclose(original, -swapped, atol=1e-6, rtol=1e-6):
        raise RuntimeError("v9 endpoint-swap antisymmetry failed after fitting")
    pair_patients = int(torch.unique(batch.pair_patient_index).numel())
    fit = {
        "seed": int(seed),
        "optimizer": "full_batch_LBFGS_strong_wolfe",
        "max_iter": ENDPOINT_LBFGS_MAX_ITER,
        "closure_count": closure_count,
        "trainable_parameter_count": model.n_trainable_parameters,
        "pair_count": batch.pair_count,
        "informative_patient_count": pair_patients,
        "benchmark_complement_assumption": (
            "zero_endpoint_is_DeepSOZ_benchmark_complement_not_clinically_verified_non_SOZ"
        ),
        "initial": {
            "total": float(initial.total.detach()),
            "bradley_terry": float(initial.bradley_terry.detach()),
            "l2_penalty": float(initial.l2_penalty.detach()),
        },
        "final": {
            "total": float(final.total.detach()),
            "bradley_terry": float(final.bradley_terry.detach()),
            "l2_penalty": float(final.l2_penalty.detach()),
        },
        "weight_l2": float(model.endpoint_utility.weight.detach().norm()),
        "endpoint_swap_max_abs_error": float((original + swapped).abs().max()),
    }
    model.eval()
    model.requires_grad_(False)
    return model, fit


def _node_group_contributions(
    model: AnchorConstrainedEndpointReranker,
    node_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    weight = model.endpoint_utility.weight.detach().squeeze(0)
    h = (node_features[..., H_FEATURE_SLICE] * weight[H_FEATURE_SLICE]).sum(dim=-1)
    v = (node_features[..., V_FEATURE_SLICE] * weight[V_FEATURE_SLICE]).sum(dim=-1)
    norms = {
        "h": float(weight[H_FEATURE_SLICE].norm()),
        "v": float(weight[V_FEATURE_SLICE].norm()),
        "i": float(weight[I_FEATURE_SLICE].norm()),
        "q": float(weight[Q_FEATURE_SLICE].norm()),
    }
    return h, v, norms


def _proposal_with_patient_gap(
    model: AnchorConstrainedEndpointReranker,
    node_features: torch.Tensor,
    anchor_scores: torch.Tensor,
    evaluable_mask: torch.Tensor,
):
    h, v, group_norms = _node_group_contributions(model, node_features)
    initial = propose_anchor_adjacent_endpoint(
        model,
        node_features,
        anchor_scores,
        evaluable_mask,
        h,
        v,
        torch.zeros(anchor_scores.shape[0], dtype=anchor_scores.dtype),
    )
    masked = anchor_scores.masked_fill(~evaluable_mask, 0.0)
    count = evaluable_mask.sum(dim=1).clamp_min(1).to(anchor_scores.dtype)
    mean = masked.sum(dim=1) / count
    variance = (
        ((anchor_scores - mean.unsqueeze(1)).square() * evaluable_mask).sum(dim=1)
        / count
    )
    scale = variance.clamp_min(1e-8).sqrt()
    gap = torch.full_like(scale, 2.0)
    available = initial.candidate_available
    if bool(available.any()):
        rows = torch.nonzero(available, as_tuple=False).flatten()
        anchor = initial.anchor_index.index_select(0, rows)
        candidate = initial.candidate_index.index_select(0, rows)
        gap[rows] = (
            anchor_scores[rows, anchor] - anchor_scores[rows, candidate]
        ) / scale.index_select(0, rows)
    proposal = propose_anchor_adjacent_endpoint(
        model,
        node_features,
        anchor_scores,
        evaluable_mask,
        h,
        v,
        gap,
    )
    if not torch.equal(initial.anchor_index, proposal.anchor_index) or not torch.equal(
        initial.candidate_index, proposal.candidate_index
    ):
        raise RuntimeError("anchor-gap calculation changed the fixed proposal")
    return proposal, gap, group_norms


def _proposal_diagnostics(proposal, gap: torch.Tensor) -> dict[str, object]:
    tcp = {
        tuple(sorted((CHANNEL_INDEX[left], CHANNEL_INDEX[right])))
        for left, right in TCP_20_EDGES
    }
    applied_tcp = 0
    applied_official_only = 0
    for patient in torch.nonzero(proposal.eligible, as_tuple=False).flatten().tolist():
        edge = tuple(
            sorted(
                (
                    int(proposal.anchor_index[patient]),
                    int(proposal.candidate_index[patient]),
                )
            )
        )
        if edge in tcp:
            applied_tcp += 1
        else:
            applied_official_only += 1
    available_gap = gap[proposal.candidate_available]
    available_confidence = proposal.candidate_confidence[
        proposal.candidate_available
    ]
    return {
        "candidate_available_count": int(proposal.candidate_available.sum()),
        "margin_pass_count": int(proposal.margin_pass.sum()),
        "h_direction_pass_count": int(proposal.h_direction_pass.sum()),
        "v_direction_pass_count": int(proposal.v_direction_pass.sum()),
        "anchor_gap_pass_count": int(proposal.anchor_gap_pass.sum()),
        "eligible_flip_count": int(proposal.eligible.sum()),
        "eligible_tcp_count": applied_tcp,
        "eligible_official_only_count": applied_official_only,
        "candidate_confidence_mean": (
            float(available_confidence.mean()) if available_confidence.numel() else None
        ),
        "candidate_gap_z_mean": (
            float(available_gap.mean()) if available_gap.numel() else None
        ),
        "candidate_gap_z_max": (
            float(available_gap.max()) if available_gap.numel() else None
        ),
    }


def _flip_outcomes(
    proposal,
    applied: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
) -> dict[str, int]:
    beneficial = harmful = neutral = unobserved = 0
    for patient in torch.nonzero(applied, as_tuple=False).flatten().tolist():
        anchor = int(proposal.anchor_index[patient])
        candidate = int(proposal.candidate_index[patient])
        if not bool(target_mask[patient, anchor] and target_mask[patient, candidate]):
            unobserved += 1
            continue
        anchor_positive = bool(targets[patient, anchor] == 1)
        candidate_positive = bool(targets[patient, candidate] == 1)
        if candidate_positive and not anchor_positive:
            beneficial += 1
        elif anchor_positive and not candidate_positive:
            harmful += 1
        else:
            neutral += 1
    return {
        "applied": int(applied.sum()),
        "beneficial": beneficial,
        "harmful": harmful,
        "neutral": neutral,
        "unobserved": unobserved,
        "net_exact": beneficial - harmful,
    }


def _transition_summary(
    candidate: torch.Tensor,
    anchor: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, object]:
    result = _transition_diagnostic(candidate, anchor, targets, mask)
    transitions = result["transitions"]
    neighbour_to_exact = transitions.get("tcp_neighbour_only->exact", 0) + transitions.get(
        "official_non_tcp_neighbour_only->exact", 0
    )
    exact_to_neighbour = transitions.get("exact->tcp_neighbour_only", 0) + transitions.get(
        "exact->official_non_tcp_neighbour_only", 0
    )
    result.update(
        {
            "all_neighbour_to_exact_rescue_count": neighbour_to_exact,
            "exact_to_any_neighbour_loss_count": exact_to_neighbour,
            "all_neighbour_rescues_exceed_losses": neighbour_to_exact
            > exact_to_neighbour,
            "far_to_exact_rescue_count": transitions.get("far->exact", 0),
            "exact_to_far_loss_count": transitions.get("exact->far", 0),
        }
    )
    return result


def _direction_payload(scores, targets, mask) -> dict[str, object]:
    report = within_tcp_edge_direction_metrics(scores, targets, mask)
    return {
        "patient_macro_accuracy": report.patient_macro_accuracy,
        "eligible_patient_count": report.eligible_patient_count,
        "informative_pair_count": report.informative_pair_count,
        "semantics": (
            "TCP edge with exactly one observed DeepSOZ benchmark-positive endpoint; diagnostic only"
        ),
    }


def _fold_coefficient_stability(weights: Sequence[torch.Tensor]) -> dict[str, object]:
    matrix = torch.stack(tuple(value.float() for value in weights))
    normalized = matrix / matrix.norm(dim=1, keepdim=True).clamp_min(1e-8)
    cosine = normalized @ normalized.transpose(0, 1)
    off_diagonal = cosine[~torch.eye(len(weights), dtype=torch.bool)]
    sign = torch.sign(matrix)
    nonzero = sign != 0
    unanimous = ((sign == sign[:1]) | ~nonzero).all(dim=0) & nonzero.any(dim=0)
    return {
        "pairwise_cosine_mean": float(off_diagonal.mean()),
        "pairwise_cosine_min": float(off_diagonal.min()),
        "unanimous_nonzero_sign_fraction": float(unanimous.float().mean()),
        "semantics": "descriptive coefficient stability; not feature causality",
    }


def _run_oof(full, patient_folds, cache_tokens, anchor):
    patient_count = len(full.patient_ids)
    candidate_oof = torch.full_like(anchor, torch.nan)
    applied_oof = torch.zeros(patient_count, dtype=torch.bool)
    candidate_index_oof = torch.full((patient_count,), -1, dtype=torch.long)
    pair_margin_oof = torch.zeros(patient_count)
    gap_oof = torch.zeros(patient_count)
    fold_rows: list[dict[str, object]] = []
    fold_weights: list[torch.Tensor] = []
    fold_nonlower = 0
    fold_strict_delta: list[float] = []
    start = time.monotonic()

    for fold in OUTER_FOLDS:
        train_indices = _patient_indices_for_fold(patient_folds, fold, held=False)
        held_indices = _patient_indices_for_fold(patient_folds, fold, held=True)
        features, feature_state = fit_fold_endpoint_features(
            cache_tokens,
            full.evidence,
            full.event_patient_index,
            patient_count,
            train_indices,
        )
        train_features = _subset_rows(features, train_indices)
        train_targets = _subset_rows(full.targets, train_indices)
        train_mask = _subset_rows(full.target_mask, train_indices)
        model, fit = _fit_reranker(
            train_features,
            train_targets,
            train_mask,
            tuple(full.patient_ids[index] for index in train_indices),
            seed=BASE_SEED + fold * 1000,
        )
        held_features = _subset_rows(features, held_indices)
        held_anchor = _subset_rows(anchor, held_indices)
        held_targets = _subset_rows(full.targets, held_indices)
        held_mask = _subset_rows(full.target_mask, held_indices)
        proposal, gap, group_norms = _proposal_with_patient_gap(
            model, held_features, held_anchor, held_mask
        )
        reranked = apply_fixed_selective_endpoint_rerank(
            held_anchor, held_mask, proposal
        )
        candidate_oof[list(held_indices)] = reranked.scores
        applied_oof[list(held_indices)] = reranked.applied
        candidate_index_oof[list(held_indices)] = proposal.candidate_index
        pair_margin_oof[list(held_indices)] = proposal.candidate_minus_anchor_logit
        gap_oof[list(held_indices)] = gap

        candidate_metrics = _metrics(reranked.scores, held_targets, held_mask)
        anchor_metrics = _metrics(held_anchor, held_targets, held_mask)
        candidate_hits = float(candidate_metrics["top1"]["strict_accuracy"]) * len(
            held_indices
        )
        anchor_hits = float(anchor_metrics["top1"]["strict_accuracy"]) * len(
            held_indices
        )
        delta = candidate_hits - anchor_hits
        fold_strict_delta.append(delta)
        fold_nonlower += int(delta >= -EXPECTED_HIT_ATOL)
        fold_rows.append(
            {
                "outer_fold": fold,
                "train_patient_count": len(train_indices),
                "held_patient_count": len(held_indices),
                "train_patient_roster_sha256": _scope_sha256(
                    tuple(full.patient_ids[index] for index in train_indices)
                ),
                "fit": fit,
                "feature_state": {
                    "h_pca_components": int(feature_state.h_components.shape[0]),
                    "node_feature_dim": ENDPOINT_NODE_FEATURE_DIM,
                    "feature_scale_min": float(feature_state.feature_scale.min()),
                    "feature_scale_max": float(feature_state.feature_scale.max()),
                },
                "weight_group_l2": group_norms,
                "proposal": _proposal_diagnostics(proposal, gap),
                "flip_outcomes": _flip_outcomes(
                    proposal, reranked.applied, held_targets, held_mask
                ),
                "candidate_metrics": candidate_metrics,
                "anchor_metrics": anchor_metrics,
                "strict_hit_delta": delta,
            }
        )
        fold_weights.append(model.endpoint_utility.weight.detach().cpu().squeeze(0))
        print(
            json.dumps(
                {
                    "stage": "outer_complete",
                    "fold": fold,
                    "strict_hit_delta": delta,
                    "applied": reranked.applied_count,
                    "candidate_strict": candidate_metrics["top1"]["strict_accuracy"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if not torch.isfinite(candidate_oof).all():
        raise RuntimeError("v9 OOF left patient predictions unfilled")
    candidate_metrics = _metrics(candidate_oof, full.targets, full.target_mask)
    anchor_metrics = _metrics(anchor, full.targets, full.target_mask)
    transitions = _transition_summary(
        candidate_oof, anchor, full.targets, full.target_mask
    )
    strict_hits = float(candidate_metrics["top1"]["strict_accuracy"]) * patient_count
    relaxed_hits = float(candidate_metrics["top1"]["relaxed_accuracy"]) * patient_count
    anchor_strict_hits = float(anchor_metrics["top1"]["strict_accuracy"]) * patient_count
    anchor_relaxed_hits = float(anchor_metrics["top1"]["relaxed_accuracy"]) * patient_count
    if abs(anchor_strict_hits - EXPECTED_ANCHOR_STRICT_HITS) > EXPECTED_HIT_ATOL or abs(
        anchor_relaxed_hits - EXPECTED_ANCHOR_RELAXED_HITS
    ) > EXPECTED_HIT_ATOL:
        raise RuntimeError("frozen temporal anchor counts changed")
    gate_checks = {
        "strict_top1_strictly_above_42_of_65": strict_hits
        > EXPECTED_ANCHOR_STRICT_HITS + EXPECTED_HIT_ATOL,
        "relaxed_top1_at_least_55_of_65": relaxed_hits + EXPECTED_HIT_ATOL
        >= EXPECTED_ANCHOR_RELAXED_HITS,
        "macro_ap_at_least_anchor_0_6328": float(
            candidate_metrics["ranking"]["macro_average_precision"]
        )
        >= EXPECTED_ANCHOR_MACRO_AP,
        "neighbour_to_exact_rescues_exceed_exact_to_neighbour_losses": transitions[
            "all_neighbour_rescues_exceed_losses"
        ],
        "far_errors_at_most_10": int(transitions["candidate_far_count"])
        <= EXPECTED_ANCHOR_FAR,
        "strict_nonlower_in_at_least_4_of_5_folds": fold_nonlower >= 4,
        "no_fold_loses_more_than_one_strict_patient": min(fold_strict_delta)
        >= -1.0 - EXPECTED_HIT_ATOL,
        "complete_65_patient_oof": patient_count == 65
        and int(torch.isfinite(candidate_oof).all(dim=1).sum()) == 65,
    }
    go = all(gate_checks.values())
    aggregate_flip_outcomes = {
        key: sum(int(row["flip_outcomes"][key]) for row in fold_rows)
        for key in (
            "applied",
            "beneficial",
            "harmful",
            "neutral",
            "unobserved",
            "net_exact",
        )
    }
    result = {
        "screen_kind": (
            "single_frozen_post_v8_source_train_patient_oof_mechanism_recovery;"
            "same_65_patients_previously_used_for_development"
        ),
        "metrics": {
            "temporal_mil_exact_anchor": anchor_metrics,
            "anchor_constrained_endpoint_reranker": candidate_metrics,
        },
        "paired_patient_bootstrap": _paired_patient_bootstrap(
            candidate_oof, anchor, full.targets, full.target_mask
        ),
        "within_tcp_direction": {
            "anchor": _direction_payload(anchor, full.targets, full.target_mask),
            "candidate": _direction_payload(
                candidate_oof, full.targets, full.target_mask
            ),
        },
        "top1_transition_diagnostic": transitions,
        "flip_outcomes": {
            "applied_count": int(applied_oof.sum()),
            "patient_coverage": float(applied_oof.float().mean()),
            **aggregate_flip_outcomes,
        },
        "fold_coefficient_stability": _fold_coefficient_stability(fold_weights),
        "outer_folds": fold_rows,
        "fold_strict_nonlower_count": fold_nonlower,
        "fold_strict_hit_deltas": fold_strict_delta,
        "frozen_go_no_go_gate": {
            "checks": gate_checks,
            "pass": go,
            "status": (
                "go_candidate_for_later_locked_validation"
                if go
                else "no_go_keep_temporal_mil_exact"
            ),
            "interpretation": (
                "exploratory source-train OOF gate after repeated development; not confirmatory"
            ),
        },
        "elapsed_sec": time.monotonic() - start,
    }
    tensors = {
        "anchor_constrained_endpoint_reranker": candidate_oof.contiguous(),
        "temporal_mil_exact_anchor": anchor.contiguous(),
        "targets": full.targets.detach().cpu().contiguous(),
        "target_mask": full.target_mask.detach().cpu().contiguous(),
        "patient_folds": torch.tensor(patient_folds, dtype=torch.int64),
        "flip_applied": applied_oof.contiguous(),
        "proposed_candidate_index": candidate_index_oof.contiguous(),
        "candidate_minus_anchor_pair_margin": pair_margin_oof.contiguous(),
        "anchor_candidate_gap_z": gap_oof.contiguous(),
    }

    final_features, final_feature_state = fit_fold_endpoint_features(
        cache_tokens,
        full.evidence,
        full.event_patient_index,
        patient_count,
        tuple(range(patient_count)),
    )
    final_model, final_fit = _fit_reranker(
        final_features,
        full.targets,
        full.target_mask,
        full.patient_ids,
        seed=BASE_SEED + 99999,
    )
    final_state = {
        "endpoint_utility.weight": final_model.endpoint_utility.weight.detach()
        .cpu()
        .contiguous(),
        "h_center": final_feature_state.h_center.cpu().contiguous(),
        "h_components": final_feature_state.h_components.cpu().contiguous(),
        "feature_mean": final_feature_state.feature_mean.cpu().contiguous(),
        "feature_scale": final_feature_state.feature_scale.cpu().contiguous(),
    }
    result["final_full_source_train_fit"] = final_fit
    return result, tensors, final_state


def _graph_payload() -> dict[str, object]:
    graph = set(endpoint_adjacency_edges())
    tcp = {
        tuple(sorted((CHANNEL_INDEX[left], CHANNEL_INDEX[right])))
        for left, right in TCP_20_EDGES
    }
    return {
        "combined_edge_count": len(graph),
        "tcp_edge_count": len(graph & tcp),
        "official_one_hop_only_edge_count": len(graph - tcp),
        "semantics": "metric-informed safety candidate prior; not anatomy or propagation graph",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    access_audit = _load_access_audit()
    cache, full, patient_folds, event_ids, lineage = _load_inputs(
        prefix_cache_path=DEFAULT_PREFIX_CACHE,
        expected_prefix_manifest_sha256=DEFAULT_PREFIX_CACHE_MANIFEST_SHA256,
        source_train_iv_path=DEFAULT_SOURCE_TRAIN_IV,
        expected_source_train_iv_manifest_sha256=(
            DEFAULT_SOURCE_TRAIN_IV_MANIFEST_SHA256
        ),
        target_scope_path=DEFAULT_TARGET_SCOPE,
        expected_target_receipt_sha256=(
            FROZEN_SOURCE_TRAIN_TARGET_SCOPE_RECEIPT_SHA256
        ),
        require_full_scope=True,
    )
    comparators, comparator_receipt = _load_fixed_comparators(full, patient_folds)
    anchor = comparators[TEMPORAL_ANCHOR]
    preflight = {
        "status": "ready_single_frozen_source_train_patient_oof",
        "schema_version": SCHEMA_VERSION,
        "reranker_schema_version": ANCHOR_CONSTRAINED_ENDPOINT_RERANKER_SCHEMA,
        "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)),
        "protocol_sha256": _file_sha256(PROTOCOL_PATH),
        "patient_count": len(full.patient_ids),
        "event_count": full.evidence.batch_size,
        "fold_counts": {
            str(fold): sum(value == fold for value in patient_folds)
            for fold in OUTER_FOLDS
        },
        "event_order_sha256": _scope_sha256(event_ids),
        "lineage": {
            **lineage,
            "feature_module_sha256": _file_sha256(FEATURE_MODULE_PATH),
            "reranker_module_sha256": _file_sha256(RERANKER_MODULE_PATH),
            "runner_sha256": _file_sha256(RUNNER_PATH),
            "comparator": comparator_receipt,
            "access_audit": access_audit,
        },
        "config": {
            "outer_folds": list(OUTER_FOLDS),
            "candidate_count": 1,
            "node_feature_dim": ENDPOINT_NODE_FEATURE_DIM,
            "h_pca_components": 8,
            "reranker_trainable_parameters": ENDPOINT_NODE_FEATURE_DIM,
            "l2_weight": ENDPOINT_L2_WEIGHT,
            "lbfgs_max_iter": ENDPOINT_LBFGS_MAX_ITER,
            "flip_logit_margin": ENDPOINT_FLIP_LOGIT_MARGIN,
            "flip_conditional_bt_score": 0.75,
            "anchor_gap_z_max": 1.0,
            "threshold_scan": False,
            "channel_identity_feature": False,
            "foundation_trainable_parameter_count": 0,
        },
        "candidate_graph": _graph_payload(),
        "foundation_backbone": "official_pretrained_LaBraM_Base_not_replaced_frozen",
        "source_dev_forward_count": 0,
        "source_eval_forward_count": 0,
        "private_forward_count": 0,
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
        FEATURE_MODULE_PATH,
        RERANKER_MODULE_PATH,
        DEFAULT_PREFIX_CACHE,
        DEFAULT_SOURCE_TRAIN_IV,
        DEFAULT_TARGET_SCOPE,
    ):
        resolved = source.resolve(strict=True)
        if output == resolved or output in resolved.parents or resolved in output.parents:
            raise ValueError("output path overlaps an immutable input")

    result, tensors, final_state = _run_oof(
        full, patient_folds, cache.tokens, anchor
    )
    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("safetensors is required") from exc
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    published = False
    try:
        prediction_path = temporary / "oof_predictions.safetensors"
        checkpoint_path = temporary / "final_checkpoint.safetensors"
        save_file(tensors, str(prediction_path))
        save_file(final_state, str(checkpoint_path))
        manifest = {
            **preflight,
            "status": "completed_exploratory_source_train_patient_oof",
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
                "foundation_replaced": False,
                "foundation_trainable_parameter_count": 0,
                "labram_feature": "frozen_blocks_0_to_9_prefix_not_multilevel_mix",
                "deepsoz_zero_semantics": (
                    "benchmark_complement_not_clinically_verified_negative"
                ),
                "candidate_graph_semantics": (
                    "metric_informed_local_safety_prior_not_propagation"
                ),
                "h_v_agreement_semantics": (
                    "correlated_feature_group_agreement_not_independent_confirmation"
                ),
                "ictal_semantics": (
                    "retrospective_scalp_visible_involvement_not_soz"
                ),
                "source_dev_used": False,
                "source_eval_used": False,
                "private_used": False,
                "same_source_train_patients_reused_after_v1_to_v8": True,
                "formal_promotion": False,
            },
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
                    "manifest_sha256": hashlib.sha256(raw).hexdigest(),
                    "metrics": result["metrics"],
                    "transitions": result["top1_transition_diagnostic"],
                    "decision": result["frozen_go_no_go_gate"],
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
