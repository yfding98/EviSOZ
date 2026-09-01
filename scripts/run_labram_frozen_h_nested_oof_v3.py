#!/usr/bin/env python3
"""Run frozen node-indexed LaBraM-H nested source-train OOF recovery."""

from __future__ import annotations

import argparse
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

from scripts.materialize_tusz_ictal_token_cache import (  # noqa: E402
    load_formal_token_corpus,
)
from scripts.run_labram_global_i_v_nested_oof_v2 import (  # noqa: E402
    COMPARATOR_PATH,
    _load_comparators,
)
from scripts.run_labram_temporal_mil_nested_oof_v1 import (  # noqa: E402
    BASE_SEED,
    BOOTSTRAP_REPLICATES,
    MAX_GRAD_NORM,
    OUTER_FOLDS,
    WEIGHT_DECAY,
    _canonical_bytes,
    _file_sha256,
    _indices_for_folds,
    _load_source_train,
    _metrics,
    _paired_bootstrap,
    _tensor_state_sha256,
)
from src.soz.data.tusz_training import (  # noqa: E402
    load_tusz_ictal_training_manifest,
)
from src.soz.development_reasoner_v1_1 import (  # noqa: E402
    FROZEN_OOF_PROTOCOL_ARTIFACT_SHA256,
    FROZEN_OOF_PROTOCOL_RECEIPT_SHA256,
    FROZEN_V1_1_CAPABILITY_MANIFEST_SHA256,
    load_development_iv_evidence_capability_v1_1,
)
from src.soz.frozen_h_crosswalk import (  # noqa: E402
    load_source_train_frozen_h_crosswalk,
)
from src.soz.frozen_h_recovery import (  # noqa: E402
    FROZEN_H_CANDIDATES,
    FROZEN_H_RECOVERY_SCHEMA,
    FrozenHCandidate,
    FrozenHNodeLocalizer,
    FrozenHPatientBatch,
    fit_frozen_h_standardization,
    frozen_h_objective,
    subset_frozen_h_patient_batch,
)
from src.soz.global_i_v_recovery import (  # noqa: E402
    aggregate_patient_event_probabilities,
    target_free_event_aq_weight,
)
from src.soz.ictal_native_eval import (  # noqa: E402
    load_bound_deepsoz_signal_preflight_artifact,
)
from src.soz.ictal_recovery_evidence_v1_2 import (  # noqa: E402
    load_target_free_ictal_oof_protocol,
)
from src.soz.preprocessing_parity import (  # noqa: E402
    load_preprocessing_selection_capability,
)
from src.soz.temporal_mil_recovery import (  # noqa: E402
    jeffreys_channel_prior_logits,
)


PROTOCOL_PATH = (
    ROOT
    / "research/02_method/"
    "labram_node_native_frozen_recovery_protocol_v3_20260810_zh.md"
)
CAPABILITY_PATH = ROOT / "outputs/labram_iv_development_candidate_capability_v1_1_20260810"
SIGNAL_PATH = ROOT / "outputs/deepsoz_signal_preflight_v2_20260809_current"
OOF_PROTOCOL_PATH = ROOT / "outputs/ictal_concept_oof_protocol_v2_20260808"
MASTER_MANIFEST_PATH = ROOT / "outputs/tusz_ictal_master_manifest_v4_1_20260809_current_preflight"
PREPROCESSING_PATH = ROOT / "outputs/preprocessing_parity_formal_v2_1_20260809/selection-capability"
TOKEN_CORPUS_PATH = ROOT / "outputs/tusz_ictal_token_corpus_formal_v4_20260809/master"
CROSSWALK_PATH = ROOT / "outputs/labram_frozen_h_source_train_crosswalk_v1_20260810"
V2_RESULT_PATH = ROOT / "outputs/labram_global_i_v_nested_oof_v2_20260810"

SIGNAL_ARTIFACT_SHA256 = "a2fdf45dd122e39ec6e73b3a3edafa1264669875fd2d8cd2b9cb7e8313d1ee66"
SIGNAL_RECEIPT_SHA256 = "10128ad30d2163838222d0b4a27d9889a767276a9b697812e3cf568a3d9fd446"
MASTER_BUNDLE_SHA256 = "73e821d08805c3a7e8ae75011dd98fe10c388d7291c74881286438e91cacc35f"
MASTER_SOURCE_SHA256 = "d5329b9231ecea7aaae6e126f5cd7a17a51f21b950025b32369592379acf8cb8"
PREPROCESSING_ARTIFACT_SHA256 = "b4aa73bff2800f12186085976a5655db6882a38232d775d11234efa387171485"
PREPROCESSING_PROTOCOL_SHA256 = "9a75dd2f3293d4d944380c0d82dcfca6a95e332f3b999e32e52b15d89622a196"
TOKEN_INDEX_SHA256 = "a7d672e3228cdc71fafb46e910033f6a5302a9e2e0a5f5716f7f4c8292ecfc26"
CROSSWALK_MANIFEST_SHA256 = "f5a0b40e7d9ecc48ffb2f10a76128da4e110b791db47ac09ace54495bd2d797b"
CROSSWALK_RECEIPT_SHA256 = "4eec735065d93f761c1e17753977fe1f0e633d1fdbb6c6888f0af4eb78f6bbee"

EPOCHS = 100
LEARNING_RATE = 3e-3
COMPLEXITY_ORDER = {name: index for index, name in enumerate(FROZEN_H_CANDIDATES)}


def _scope_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256(_canonical_bytes(list(values))).hexdigest()


def _seeded_model(
    prior: torch.Tensor,
    stats,
    *,
    candidate: FrozenHCandidate,
    seed: int,
    device: torch.device,
) -> FrozenHNodeLocalizer:
    fork_devices: list[int] = []
    if device.type == "cuda":
        fork_devices = [device.index if device.index is not None else 0]
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(seed)
        model = FrozenHNodeLocalizer(
            prior,
            stats,
            candidate=candidate,
        )
    return model.to(device)


def _common_initialization_receipt(
    prior: torch.Tensor,
    stats,
    *,
    seed: int,
) -> dict[str, str]:
    models = {
        candidate: _seeded_model(
            prior,
            stats,
            candidate=candidate,
            seed=seed,
            device=torch.device("cpu"),
        )
        for candidate in FROZEN_H_CANDIDATES
    }
    h_states = {
        name: model.h_scorer.weight.detach().clone() for name, model in models.items()
    }
    if not all(torch.equal(h_states[FROZEN_H_CANDIDATES[0]], value) for value in h_states.values()):
        raise RuntimeError("frozen-H candidates do not share H initialization")
    v_names = ("frozen_h_v_uniform", "frozen_h_v_global_i")
    first_v = dict(models[v_names[0]].v_scorer.named_parameters())
    second_v = dict(models[v_names[1]].v_scorer.named_parameters())
    if set(first_v) != set(second_v) or any(
        not torch.equal(first_v[name], second_v[name]) for name in first_v
    ):
        raise RuntimeError("H+V candidates do not share V initialization")
    return {
        "h_initialization_sha256": _tensor_state_sha256(
            {"h_scorer.weight": h_states[FROZEN_H_CANDIDATES[0]]}
        ),
        "v_initialization_sha256": _tensor_state_sha256(
            {name: value.detach() for name, value in first_v.items()}
        ),
    }


def _fit(
    train: FrozenHPatientBatch,
    *,
    candidate: FrozenHCandidate,
    seed: int,
    device: torch.device,
) -> tuple[FrozenHNodeLocalizer, dict[str, object]]:
    # Both operations are deliberately inside each fit and see only the
    # corresponding inner/outer training patient scope.
    stats = fit_frozen_h_standardization(train)
    prior = jeffreys_channel_prior_logits(train.base).detach().cpu()
    initialization = _common_initialization_receipt(prior, stats, seed=seed)
    model = _seeded_model(
        prior,
        stats,
        candidate=candidate,
        seed=seed,
        device=device,
    )
    parameter_count = model.n_trainable_parameters
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
        output = model(batch.node_tokens, batch.base.evidence)
        objective = frozen_h_objective(output.event_probabilities, batch)
        objective.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        row = {
            "total": float(objective.total.detach().cpu()),
            "exact_set_mass": float(objective.exact_set_mass.detach().cpu()),
            "pairwise": float(objective.pairwise.detach().cpu()),
        }
        if first is None:
            first = row
        last = row
    optimizer.zero_grad(set_to_none=True)
    del optimizer, batch
    model.eval()
    model.requires_grad_(False)
    assert first is not None and last is not None
    preprocessing_state = {
        "h_mean": stats.mean.detach().cpu(),
        "h_scale": stats.scale.detach().cpu(),
        "channel_prior": prior.detach().cpu(),
    }
    return model, {
        "candidate": candidate,
        "seed": seed,
        "epochs": EPOCHS,
        "trainable_parameter_count": parameter_count,
        "fit_scope_patient_count": len(train.base.patient_ids),
        "fit_scope_patient_roster_sha256": _scope_sha256(train.base.patient_ids),
        "fold_local_preprocessing_state_sha256": _tensor_state_sha256(
            preprocessing_state
        ),
        "first_epoch": first,
        "final_epoch": last,
        **initialization,
    }


def _predict(
    model: FrozenHNodeLocalizer,
    batch: FrozenHPatientBatch,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    moved = batch.to(device)
    model.eval()
    with torch.no_grad():
        output = model(moved.node_tokens, moved.base.evidence)
        aggregation = aggregate_patient_event_probabilities(
            output.event_probabilities,
            moved.base.event_patient_index,
            mode="equal_probability_mean",
        )
    scores = aggregation.ranking_logits.detach().cpu()
    probabilities = aggregation.probabilities.detach().cpu()
    event_probability = output.event_probabilities.detach().cpu()
    h_contribution = output.h_contribution.detach().cpu()
    v_contribution = output.v_contribution.detach().cpu()
    prior_only = output.prior_only_event.detach().cpu()
    h_weights = output.h_temporal_weights.detach().cpu()
    raw_aq = target_free_event_aq_weight(batch.base.evidence).detach().cpu()
    event_index = batch.base.event_patient_index.detach().cpu()
    uncertainty = []
    for patient in range(len(batch.base.patient_ids)):
        selected = event_index == patient
        patient_probability = probabilities[patient]
        ordered = torch.sort(patient_probability, descending=True).values
        patient_top = int(torch.argmax(patient_probability).item())
        event_top = torch.argmax(event_probability[selected], dim=1)
        attention = h_weights[selected]
        attention_entropy = -(
            attention.clamp_min(1e-8) * attention.clamp_min(1e-8).log()
        ).sum(dim=-1)
        uncertainty.append(
            {
                "patient_id": batch.base.patient_ids[patient],
                "event_count": int(selected.sum().item()),
                "prior_only_event_count": int(prior_only[selected].sum().item()),
                "top1_probability_margin": float((ordered[0] - ordered[1]).item()),
                "ranking_entropy": float(
                    (-(patient_probability * patient_probability.clamp_min(1e-8).log()).sum()).item()
                ),
                "event_top1_disagreement_rate": float(
                    (event_top != patient_top).float().mean().item()
                ),
                "mean_channel_event_probability_std": float(
                    event_probability[selected].std(dim=0, unbiased=False).mean().item()
                ),
                "mean_h_abs_contribution": float(h_contribution[selected].abs().mean().item()),
                "mean_v_abs_contribution": float(v_contribution[selected].abs().mean().item()),
                "aq_diagnostic_min": float(raw_aq[selected].min().item()),
                "aq_diagnostic_max": float(raw_aq[selected].max().item()),
                "mean_h_temporal_weight_entropy": float(attention_entropy.mean().item()),
            }
        )
    del moved
    return scores, probabilities, {"patients": uncertainty}


def _candidate_key(metrics: Mapping[str, object], name: str) -> tuple[float, ...]:
    return (
        float(metrics["top1"]["strict_accuracy"]),
        float(metrics["ranking"]["macro_average_precision"]),
        float(metrics["ranking"]["mean_reciprocal_rank"]),
        float(-COMPLEXITY_ORDER[name]),
    )


def _subset(full: FrozenHPatientBatch, indices: Sequence[int]) -> FrozenHPatientBatch:
    return subset_frozen_h_patient_batch(full, indices)


def _load_frozen_h_source_train() -> tuple[
    FrozenHPatientBatch,
    tuple[int, ...],
    dict[str, object],
]:
    base, patient_folds, base_lineage = _load_source_train()
    signal = load_bound_deepsoz_signal_preflight_artifact(
        SIGNAL_PATH,
        expected_artifact_sha256=SIGNAL_ARTIFACT_SHA256,
        expected_receipt_sha256=SIGNAL_RECEIPT_SHA256,
    )
    protocol = load_target_free_ictal_oof_protocol(
        OOF_PROTOCOL_PATH,
        expected_artifact_sha256=FROZEN_OOF_PROTOCOL_ARTIFACT_SHA256,
        expected_protocol_receipt_sha256=FROZEN_OOF_PROTOCOL_RECEIPT_SHA256,
    )
    capability = load_development_iv_evidence_capability_v1_1(
        CAPABILITY_PATH,
        signal,
        protocol,
        expected_manifest_sha256=FROZEN_V1_1_CAPABILITY_MANIFEST_SHA256,
    )
    master = load_tusz_ictal_training_manifest(
        MASTER_MANIFEST_PATH,
        expected_bundle_manifest_sha256=MASTER_BUNDLE_SHA256,
        expected_source_manifest_sha256=MASTER_SOURCE_SHA256,
    )
    preprocessing = load_preprocessing_selection_capability(
        PREPROCESSING_PATH,
        expected_artifact_sha256=PREPROCESSING_ARTIFACT_SHA256,
        expected_protocol_receipt_sha256=PREPROCESSING_PROTOCOL_SHA256,
    )
    token_corpus = load_formal_token_corpus(
        TOKEN_CORPUS_PATH,
        expected_index_sha256=TOKEN_INDEX_SHA256,
        preprocessing_selection=preprocessing,
    )
    crosswalk = load_source_train_frozen_h_crosswalk(
        CROSSWALK_PATH,
        capability=capability,
        signal=signal,
        protocol=protocol,
        master_manifest=master,
        token_corpus=token_corpus,
        expected_manifest_sha256=CROSSWALK_MANIFEST_SHA256,
        expected_receipt_sha256=CROSSWALK_RECEIPT_SHA256,
    )
    expected_patient_by_event = tuple(
        base.patient_ids[int(index)] for index in base.event_patient_index.tolist()
    )
    expected_fold_by_event = tuple(
        patient_folds[int(index)] for index in base.event_patient_index.tolist()
    )
    if crosswalk.patient_ids_by_event != expected_patient_by_event:
        raise RuntimeError("frozen-H patient/event order differs from target join")
    if crosswalk.oof_folds != expected_fold_by_event:
        raise RuntimeError("frozen-H event folds differ from patient folds")
    if crosswalk.patient_ids != base.patient_ids:
        raise RuntimeError("frozen-H patient roster differs from target join")
    tokens = crosswalk.materialize_tokens()
    full = FrozenHPatientBatch(base=base, node_tokens=tokens)
    lineage = {
        **base_lineage,
        "crosswalk_manifest_sha256": crosswalk.manifest_sha256,
        "crosswalk_receipt_sha256": crosswalk.receipt_sha256,
        "crosswalk_event_order_sha256": crosswalk.receipt["event_order_sha256"],
        "crosswalk_token_binding_roster_sha256": crosswalk.receipt[
            "token_binding_roster_sha256"
        ],
        "formal_token_corpus_index_sha256": token_corpus.index_sha256,
        "foundation_feature_receipt_sha256": crosswalk.receipt[
            "foundation_feature_receipt_sha256"
        ],
        "foundation_checkpoint_sha256": crosswalk.receipt[
            "foundation_checkpoint_sha256"
        ],
        "foundation_modeling_sha256": crosswalk.receipt[
            "foundation_modeling_sha256"
        ],
        "raw_replay_verified": crosswalk.receipt["raw_replay_verified"],
    }
    return full, patient_folds, lineage


def _load_all_comparators(
    full: FrozenHPatientBatch,
    patient_folds: tuple[int, ...],
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    comparators, receipt = _load_comparators(full.base, patient_folds)
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError("safetensors is required for comparator loading") from exc
    v2_prediction_path = V2_RESULT_PATH / "oof_predictions.safetensors"
    tensors = load_file(str(v2_prediction_path), device="cpu")
    name = "global_i_v_equal_probability_mean"
    if name not in tensors or tuple(tensors[name].shape) != (len(full.base.patient_ids), 19):
        raise RuntimeError("v2 V-only comparator is missing")
    if not torch.equal(tensors["targets"], full.base.targets.cpu()) or not torch.equal(
        tensors["target_mask"], full.base.target_mask.cpu()
    ):
        raise RuntimeError("v2 V-only comparator targets differ from source-train")
    comparators["v2_v_only_equal"] = tensors[name].float().contiguous()
    return comparators, {
        **receipt,
        "v2_artifact_path": str(V2_RESULT_PATH.relative_to(ROOT)),
        "v2_prediction_file_sha256": _file_sha256(v2_prediction_path),
    }


def _run(
    full: FrozenHPatientBatch,
    patient_folds: tuple[int, ...],
    comparators: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
) -> tuple[
    dict[str, object],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    FrozenHNodeLocalizer,
]:
    patients = len(full.base.patient_ids)
    predictions = {
        "selected": torch.full((patients, 19), torch.nan),
        **{
            name: torch.full((patients, 19), torch.nan)
            for name in FROZEN_H_CANDIDATES
        },
        **{name: value.clone() for name, value in comparators.items()},
    }
    probabilities = {
        "selected": torch.full((patients, 19), torch.nan),
        **{
            name: torch.full((patients, 19), torch.nan)
            for name in FROZEN_H_CANDIDATES
        },
    }
    selection_counts = {name: 0 for name in FROZEN_H_CANDIDATES}
    outer_rows = []
    all_uncertainty: list[dict[str, object]] = []

    for outer_fold in OUTER_FOLDS:
        outer_train_folds = tuple(fold for fold in OUTER_FOLDS if fold != outer_fold)
        inner_predictions = {
            name: torch.full((patients, 19), torch.nan)
            for name in FROZEN_H_CANDIDATES
        }
        inner_rows = []
        for inner_fold in outer_train_folds:
            train_folds = tuple(fold for fold in outer_train_folds if fold != inner_fold)
            train_indices = _indices_for_folds(patient_folds, train_folds)
            held_indices = _indices_for_folds(patient_folds, (inner_fold,))
            train = _subset(full, train_indices)
            held = _subset(full, held_indices)
            seed = BASE_SEED + 40000 + outer_fold * 100 + inner_fold * 10
            fit_rows = []
            for name in FROZEN_H_CANDIDATES:
                model, fit = _fit(
                    train,
                    candidate=name,
                    seed=seed,
                    device=device,
                )
                scores, _, _ = _predict(model, held, device=device)
                inner_predictions[name][list(held_indices)] = scores
                fit_rows.append(fit)
                del model
                print(
                    json.dumps(
                        {
                            "stage": "inner_complete",
                            "outer_fold": outer_fold,
                            "inner_fold": inner_fold,
                            "candidate": name,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            preprocessing_hashes = {
                str(row["fold_local_preprocessing_state_sha256"]) for row in fit_rows
            }
            h_initializations = {str(row["h_initialization_sha256"]) for row in fit_rows}
            v_initializations = {
                str(row["v_initialization_sha256"])
                for row in fit_rows
                if row["candidate"] != "frozen_h_uniform"
            }
            if len(preprocessing_hashes) != 1 or len(h_initializations) != 1 or len(v_initializations) != 1:
                raise RuntimeError("same-fold candidate preprocessing/init fairness failed")
            inner_rows.append(
                {
                    "inner_fold": inner_fold,
                    "train_patient_count": len(train_indices),
                    "held_patient_count": len(held_indices),
                    "same_seed": seed,
                    "candidate_fits": fit_rows,
                }
            )

        outer_train_indices = _indices_for_folds(patient_folds, outer_train_folds)
        index = torch.tensor(outer_train_indices, dtype=torch.long)
        inner_metrics = {
            name: _metrics(
                scores.index_select(0, index),
                full.base.targets.index_select(0, index),
                full.base.target_mask.index_select(0, index),
            )
            for name, scores in inner_predictions.items()
        }
        selected = max(
            FROZEN_H_CANDIDATES,
            key=lambda name: _candidate_key(inner_metrics[name], name),
        )
        selection_counts[selected] += 1

        outer_train = _subset(full, outer_train_indices)
        held_indices = _indices_for_folds(patient_folds, (outer_fold,))
        held = _subset(full, held_indices)
        seed = BASE_SEED + 50000 + outer_fold * 1000
        outer_candidates = {}
        fit_rows = []
        for name in FROZEN_H_CANDIDATES:
            model, fit = _fit(
                outer_train,
                candidate=name,
                seed=seed,
                device=device,
            )
            scores, patient_probability, uncertainty = _predict(
                model, held, device=device
            )
            predictions[name][list(held_indices)] = scores
            probabilities[name][list(held_indices)] = patient_probability
            if name == selected:
                predictions["selected"][list(held_indices)] = scores
                probabilities["selected"][list(held_indices)] = patient_probability
                all_uncertainty.extend(uncertainty["patients"])
            outer_candidates[name] = {
                "fit": fit,
                "held_metrics": _metrics(
                    scores, held.base.targets, held.base.target_mask
                ),
            }
            fit_rows.append(fit)
            del model
        if len({str(row["fold_local_preprocessing_state_sha256"]) for row in fit_rows}) != 1:
            raise RuntimeError("outer candidates do not share fold-local preprocessing")
        outer_rows.append(
            {
                "outer_fold": outer_fold,
                "outer_train_patient_count": len(outer_train_indices),
                "held_patient_count": len(held_indices),
                "same_seed": seed,
                "selected_candidate": selected,
                "inner_candidate_metrics": inner_metrics,
                "inner_fits": inner_rows,
                "outer_candidates": outer_candidates,
            }
        )
        print(
            json.dumps(
                {
                    "stage": "outer_complete",
                    "outer_fold": outer_fold,
                    "selected_candidate": selected,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if any(not torch.isfinite(value).all() for value in predictions.values()) or any(
        not torch.isfinite(value).all() for value in probabilities.values()
    ):
        raise RuntimeError("frozen-H nested OOF left predictions unfilled")
    metrics = {
        name: _metrics(value, full.base.targets, full.base.target_mask)
        for name, value in predictions.items()
    }
    majority = max(selection_counts.values())
    tied = [name for name, count in selection_counts.items() if count == majority]
    final_candidate = min(tied, key=lambda name: COMPLEXITY_ORDER[name])
    selection_basis = (
        "outer_inner_selection_majority"
        if len(tied) == 1
        else "majority_tie_then_preregistered_lower_complexity_no_outer_label_tiebreak"
    )
    final_model, final_fit = _fit(
        full,
        candidate=final_candidate,
        seed=BASE_SEED + 159999,
        device=device,
    )
    final_model = final_model.cpu()
    result = {
        "outer_folds": outer_rows,
        "selection_counts": selection_counts,
        "selected_nested_oof_metrics": metrics["selected"],
        "all_candidate_oof_metrics": metrics,
        "selected_vs_phase_paired_patient_bootstrap": _paired_bootstrap(
            predictions["selected"], predictions["phase_baseline"],
            full.base.targets, full.base.target_mask,
        ),
        "selected_vs_temporal_exact_paired_patient_bootstrap": _paired_bootstrap(
            predictions["selected"], predictions["temporal_mil_exact"],
            full.base.targets, full.base.target_mask,
        ),
        "selected_vs_v2_v_only_equal_paired_patient_bootstrap": _paired_bootstrap(
            predictions["selected"], predictions["v2_v_only_equal"],
            full.base.targets, full.base.target_mask,
        ),
        "final_candidate": final_candidate,
        "final_selection_basis": selection_basis,
        "final_fit": final_fit,
        "uncertainty": {
            "semantics": (
                "ranking_entropy_event_dispersion_H_V_contribution_and_quality_"
                "diagnostics;not_bayesian_posterior_uncertainty"
            ),
            "patients": sorted(all_uncertainty, key=lambda row: row["patient_id"]),
        },
    }
    return result, predictions, probabilities, final_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.preflight_only and args.output_directory is None:
        raise ValueError("full run requires --output-directory")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    full, patient_folds, lineage = _load_frozen_h_source_train()
    comparators, comparator_receipt = _load_all_comparators(full, patient_folds)
    preflight = {
        "status": "ready_frozen_h_nested_source_train_only",
        "schema_version": FROZEN_H_RECOVERY_SCHEMA,
        "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)),
        "protocol_sha256": _file_sha256(PROTOCOL_PATH),
        "device": str(device),
        "patient_count": len(full.base.patient_ids),
        "event_count": full.base.evidence.batch_size,
        "node_token_shape": list(full.node_tokens.shape),
        "fold_counts": {
            str(fold): sum(value == fold for value in patient_folds)
            for fold in OUTER_FOLDS
        },
        "candidates": list(FROZEN_H_CANDIDATES),
        "patient_pooling": "equal_event_probability_mean",
        "lineage": lineage,
        "comparator_receipt": comparator_receipt,
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
        CROSSWALK_PATH,
        TOKEN_CORPUS_PATH,
        COMPARATOR_PATH,
        V2_RESULT_PATH,
    ):
        resolved = source.resolve(strict=True)
        if output == resolved or output in resolved.parents or resolved in output.parents:
            raise ValueError("output path overlaps immutable input")

    result, predictions, probabilities, final_model = _run(
        full, patient_folds, comparators, device=device
    )
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError("safetensors is required for publication") from exc
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    published = False
    try:
        prediction_tensors = {
            **{name: value.contiguous() for name, value in predictions.items()},
            **{
                f"probability__{name}": value.contiguous()
                for name, value in probabilities.items()
            },
            "targets": full.base.targets.detach().cpu().contiguous(),
            "target_mask": full.base.target_mask.detach().cpu().contiguous(),
            "patient_folds": torch.tensor(patient_folds, dtype=torch.int64),
        }
        prediction_path = temporary / "oof_predictions.safetensors"
        save_file(prediction_tensors, str(prediction_path))
        final_state = {
            name: value.detach().cpu().contiguous()
            for name, value in final_model.state_dict().items()
        }
        checkpoint_path = temporary / "final_checkpoint.safetensors"
        save_file(final_state, str(checkpoint_path))
        manifest = {
            **preflight,
            "status": "completed_development_only",
            "config": {
                "outer_folds": list(OUTER_FOLDS),
                "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "max_grad_norm": MAX_GRAD_NORM,
                "base_seed": BASE_SEED,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "same_seed_within_fold_candidates": True,
                "fold_local_h_standardization": True,
                "fold_local_channel_prior": True,
                "objective": "exact_positive_probability_mass_plus_0.25_pairwise",
            },
            "result": result,
            "patient_ids": list(full.base.patient_ids),
            "patient_folds": list(patient_folds),
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
                "foundation_backbone": "official_LaBraM_frozen_not_replaced_not_trained_from_scratch",
                "foundation_latent": "matched_direct_node_indexed_contextual_token_not_named_concept",
                "ictal_path": "channel_agnostic_global_temporal_gate_only",
                "ictal_semantics": "retrospective_scalp_visible_involvement_not_soz",
                "attention_semantics": "discriminative_temporal_weight_not_onset_or_propagation",
                "position_embedding_shortcut_control_pending_before_promotion": True,
                "source_dev_used": False,
                "source_eval_used": False,
                "private_used": False,
                "formal_promotion": False,
            },
        }
        raw = _canonical_bytes(manifest)
        manifest_path = temporary / "manifest.json"
        manifest_path.write_bytes(raw)
        os.rename(temporary, output)
        published = True
        print(
            json.dumps(
                {
                    "status": "completed_development_only",
                    "output_directory": str(output),
                    "manifest_sha256": hashlib.sha256(raw).hexdigest(),
                    "final_candidate": result["final_candidate"],
                    "selection_counts": result["selection_counts"],
                    "selected_nested_oof_metrics": result["selected_nested_oof_metrics"],
                    "selected_vs_temporal_exact": result[
                        "selected_vs_temporal_exact_paired_patient_bootstrap"
                    ],
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
