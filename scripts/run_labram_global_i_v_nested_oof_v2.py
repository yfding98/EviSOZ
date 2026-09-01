#!/usr/bin/env python3
"""Run the frozen global-I gate / V-only localizer nested source-train OOF."""

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
    _selection_key,
    _subset,
    _tensor_state_sha256,
)
from src.soz.global_i_v_recovery import (  # noqa: E402
    GLOBAL_I_V_RECOVERY_SCHEMA,
    GlobalITemporalGateVLocalizer,
    PatientPoolingMode,
    aggregate_patient_event_probabilities,
    global_i_v_objective,
    target_free_event_aq_weight,
)
from src.soz.temporal_mil_recovery import (  # noqa: E402
    TemporalMILPatientBatch,
    jeffreys_channel_prior_logits,
)


PROTOCOL_PATH = (
    ROOT
    / "research/02_method/"
    "labram_global_i_v_localizer_recovery_protocol_v2_20260810_zh.md"
)
COMPARATOR_PATH = ROOT / "outputs/labram_temporal_mil_nested_oof_v1_20260810"
CANDIDATES: Mapping[str, PatientPoolingMode] = {
    "global_i_v_equal_probability_mean": "equal_probability_mean",
    "global_i_v_aq_probability_mean": "aq_probability_mean",
}
EPOCHS = 100
LEARNING_RATE = 3e-3


def _seeded_model(
    prior: torch.Tensor,
    *,
    seed: int,
    device: torch.device,
) -> GlobalITemporalGateVLocalizer:
    fork_devices: list[int] = []
    if device.type == "cuda":
        fork_devices = [device.index if device.index is not None else 0]
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(seed)
        return GlobalITemporalGateVLocalizer(prior).to(device)


def _fit(
    train: TemporalMILPatientBatch,
    *,
    mode: PatientPoolingMode,
    seed: int,
    device: torch.device,
) -> tuple[GlobalITemporalGateVLocalizer, dict[str, object]]:
    prior = jeffreys_channel_prior_logits(train).detach().cpu()
    model = _seeded_model(prior, seed=seed, device=device)
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
        output = model(batch.evidence)
        objective = global_i_v_objective(
            output.event_probabilities,
            batch,
            mode=mode,
        )
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
    del optimizer
    model.eval()
    model.requires_grad_(False)
    assert first is not None and last is not None
    return model, {
        "seed": seed,
        "epochs": EPOCHS,
        "first_epoch": first,
        "final_epoch": last,
        "trainable_parameter_count": parameter_count,
    }


def _predict(
    model: GlobalITemporalGateVLocalizer,
    batch: TemporalMILPatientBatch,
    *,
    mode: PatientPoolingMode,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    moved = batch.to(device)
    model.eval()
    with torch.no_grad():
        output = model(moved.evidence)
        aq = (
            target_free_event_aq_weight(moved.evidence)
            if mode == "aq_probability_mean"
            else None
        )
        aggregation = aggregate_patient_event_probabilities(
            output.event_probabilities,
            moved.event_patient_index,
            mode=mode,
            aq_event_weight=aq,
        )

    scores = aggregation.ranking_logits.detach().cpu()
    probabilities = aggregation.probabilities.detach().cpu()
    event_probability = output.event_probabilities.detach().cpu()
    event_weights = aggregation.event_normalized_weights.detach().cpu()
    temporal_weights = output.temporal_weights.detach().cpu()
    raw_aq = target_free_event_aq_weight(batch.evidence).detach().cpu()
    uncertainty = []
    event_index = batch.event_patient_index.detach().cpu()
    for patient in range(len(batch.patient_ids)):
        selected = event_index == patient
        patient_probability = probabilities[patient]
        ordered = torch.sort(patient_probability, descending=True).values
        patient_top = int(torch.argmax(patient_probability).item())
        event_top = torch.argmax(event_probability[selected], dim=1)
        attention = temporal_weights[selected]
        attention_entropy = -(
            attention.clamp_min(1e-8) * attention.clamp_min(1e-8).log()
        ).sum(dim=-1)
        normalized = event_weights[selected]
        uncertainty.append(
            {
                "patient_id": batch.patient_ids[patient],
                "event_count": int(selected.sum().item()),
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
                "effective_event_count": float(
                    (1.0 / normalized.square().sum().clamp_min(1e-8)).item()
                ),
                "aq_weight_min": float(raw_aq[selected].min().item()),
                "aq_weight_max": float(raw_aq[selected].max().item()),
                "mean_temporal_weight_entropy": float(attention_entropy.mean().item()),
            }
        )
    return scores, probabilities, {"patients": uncertainty}


def _load_comparators(
    full: TemporalMILPatientBatch,
    patient_folds: tuple[int, ...],
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    manifest_path = COMPARATOR_PATH / "manifest.json"
    tensor_path = COMPARATOR_PATH / "oof_predictions.safetensors"
    if not manifest_path.is_file() or not tensor_path.is_file():
        raise FileNotFoundError("frozen temporal-MIL comparator artifact is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if tuple(manifest.get("patient_ids", ())) != full.patient_ids:
        raise RuntimeError("comparator patient roster differs from source-train")
    if tuple(int(value) for value in manifest.get("patient_folds", ())) != patient_folds:
        raise RuntimeError("comparator patient folds differ from source-train")
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError("safetensors is required for comparator loading") from exc
    tensors = load_file(str(tensor_path), device="cpu")
    required = ("phase_baseline", "temporal_mil_exact", "targets", "target_mask")
    if any(name not in tensors for name in required):
        raise RuntimeError("comparator prediction payload is incomplete")
    if not torch.equal(tensors["targets"], full.targets.cpu()) or not torch.equal(
        tensors["target_mask"], full.target_mask.cpu()
    ):
        raise RuntimeError("comparator targets differ from current source-train")
    comparators = {
        "phase_baseline": tensors["phase_baseline"].float().contiguous(),
        "temporal_mil_exact": tensors["temporal_mil_exact"].float().contiguous(),
    }
    if any(tuple(value.shape) != (len(full.patient_ids), 19) for value in comparators.values()):
        raise RuntimeError("comparator logits require shape [P,19]")
    return comparators, {
        "artifact_path": str(COMPARATOR_PATH.relative_to(ROOT)),
        "manifest_sha256": _file_sha256(manifest_path),
        "prediction_file_sha256": _file_sha256(tensor_path),
        "forward_recomputed": False,
    }


def _run(
    full: TemporalMILPatientBatch,
    patient_folds: tuple[int, ...],
    comparators: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
) -> tuple[
    dict[str, object],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    GlobalITemporalGateVLocalizer,
]:
    n_patients = len(full.patient_ids)
    predictions = {
        "selected": torch.full((n_patients, 19), torch.nan),
        **{
            name: torch.full((n_patients, 19), torch.nan)
            for name in CANDIDATES
        },
        **{name: value.clone() for name, value in comparators.items()},
    }
    probabilities = {
        "selected": torch.full((n_patients, 19), torch.nan),
        **{
            name: torch.full((n_patients, 19), torch.nan)
            for name in CANDIDATES
        },
    }
    outer_rows = []
    selection_counts = {name: 0 for name in CANDIDATES}
    all_uncertainty: list[dict[str, object]] = []

    for outer_fold in OUTER_FOLDS:
        outer_train_folds = tuple(fold for fold in OUTER_FOLDS if fold != outer_fold)
        inner_predictions = {
            name: torch.full((n_patients, 19), torch.nan) for name in CANDIDATES
        }
        inner_rows = []
        for inner_fold in outer_train_folds:
            inner_train_folds = tuple(
                fold for fold in outer_train_folds if fold != inner_fold
            )
            train_indices = _indices_for_folds(patient_folds, inner_train_folds)
            validation_indices = _indices_for_folds(patient_folds, (inner_fold,))
            train = _subset(full, train_indices)
            validation = _subset(full, validation_indices)
            for candidate_index, (name, mode) in enumerate(CANDIDATES.items()):
                seed = BASE_SEED + 20000 + outer_fold * 100 + inner_fold * 10 + candidate_index
                model, fit = _fit(train, mode=mode, seed=seed, device=device)
                scores, _, _ = _predict(model, validation, mode=mode, device=device)
                inner_predictions[name][list(validation_indices)] = scores
                inner_rows.append(
                    {
                        "inner_fold": inner_fold,
                        "candidate": name,
                        "train_patient_count": len(train_indices),
                        "validation_patient_count": len(validation_indices),
                        "fit": fit,
                    }
                )
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

        outer_train_indices = _indices_for_folds(patient_folds, outer_train_folds)
        index = torch.tensor(outer_train_indices, dtype=torch.long)
        inner_metrics = {
            name: _metrics(
                value.index_select(0, index),
                full.targets.index_select(0, index),
                full.target_mask.index_select(0, index),
            )
            for name, value in inner_predictions.items()
        }
        selected = max(
            CANDIDATES,
            key=lambda name: _selection_key(
                inner_metrics[name],
                conservative=name == "global_i_v_equal_probability_mean",
            ),
        )
        selection_counts[selected] += 1

        outer_train = _subset(full, outer_train_indices)
        held_indices = _indices_for_folds(patient_folds, (outer_fold,))
        held = _subset(full, held_indices)
        outer_candidate_rows = {}
        for candidate_index, (name, mode) in enumerate(CANDIDATES.items()):
            seed = BASE_SEED + 30000 + outer_fold * 1000 + candidate_index
            model, fit = _fit(outer_train, mode=mode, seed=seed, device=device)
            scores, patient_probability, uncertainty = _predict(
                model, held, mode=mode, device=device
            )
            predictions[name][list(held_indices)] = scores
            probabilities[name][list(held_indices)] = patient_probability
            if name == selected:
                predictions["selected"][list(held_indices)] = scores
                probabilities["selected"][list(held_indices)] = patient_probability
                all_uncertainty.extend(uncertainty["patients"])
            outer_candidate_rows[name] = {
                "fit": fit,
                "held_metrics": _metrics(scores, held.targets, held.target_mask),
            }
        outer_rows.append(
            {
                "outer_fold": outer_fold,
                "outer_train_patient_count": len(outer_train_indices),
                "held_patient_count": len(held_indices),
                "selected_candidate": selected,
                "inner_candidate_metrics": inner_metrics,
                "inner_fits": inner_rows,
                "outer_candidates": outer_candidate_rows,
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
        raise RuntimeError("nested OOF left a patient prediction unfilled")
    candidate_metrics = {
        name: _metrics(value, full.targets, full.target_mask)
        for name, value in predictions.items()
    }
    majority = max(selection_counts.values())
    tied = [name for name, count in selection_counts.items() if count == majority]
    if len(tied) == 1:
        final_candidate = tied[0]
        final_selection_basis = "outer_inner_selection_majority"
    else:
        final_candidate = max(
            tied,
            key=lambda name: _selection_key(
                candidate_metrics[name],
                conservative=name == "global_i_v_equal_probability_mean",
            ),
        )
        final_selection_basis = "majority_tie_then_full_oof_strict_ap_mrr"

    final_model, final_fit = _fit(
        full,
        mode=CANDIDATES[final_candidate],
        seed=BASE_SEED + 99999 + 20000,
        device=device,
    )
    final_model = final_model.cpu()
    result = {
        "outer_folds": outer_rows,
        "selection_counts": selection_counts,
        "selected_nested_oof_metrics": candidate_metrics["selected"],
        "all_candidate_oof_metrics": candidate_metrics,
        "selected_vs_phase_paired_patient_bootstrap": _paired_bootstrap(
            predictions["selected"],
            predictions["phase_baseline"],
            full.targets,
            full.target_mask,
        ),
        "selected_vs_temporal_exact_paired_patient_bootstrap": _paired_bootstrap(
            predictions["selected"],
            predictions["temporal_mil_exact"],
            full.targets,
            full.target_mask,
        ),
        "final_candidate": final_candidate,
        "final_selection_basis": final_selection_basis,
        "final_fit": final_fit,
        "uncertainty": {
            "semantics": (
                "ranking_entropy_event_dispersion_effective_event_count_and_"
                "temporal_weight_entropy;not_bayesian_posterior_uncertainty"
            ),
            "patients": sorted(all_uncertainty, key=lambda row: row["patient_id"]),
        },
    }
    return result, predictions, probabilities, final_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
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

    full, patient_folds, lineage = _load_source_train()
    comparators, comparator_receipt = _load_comparators(full, patient_folds)
    fold_counts = {
        str(fold): sum(value == fold for value in patient_folds) for fold in OUTER_FOLDS
    }
    preflight = {
        "status": "ready_nested_source_train_only",
        "schema_version": GLOBAL_I_V_RECOVERY_SCHEMA,
        "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)),
        "protocol_sha256": _file_sha256(PROTOCOL_PATH),
        "device": str(device),
        "patient_count": len(full.patient_ids),
        "event_count": full.evidence.batch_size,
        "fold_counts": fold_counts,
        "lineage": lineage,
        "comparator_receipt": comparator_receipt,
        "candidates": dict(CANDIDATES),
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
    for source in (PROTOCOL_PATH, COMPARATOR_PATH):
        resolved = source.resolve(strict=True)
        if output == resolved or output in resolved.parents or resolved in output.parents:
            raise ValueError("output path overlaps an input path")

    result, predictions, probabilities, final_model = _run(
        full,
        patient_folds,
        comparators,
        device=device,
    )
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError("safetensors is required for recovery publication") from exc
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    published = False
    try:
        prediction_tensors = {
            **{name: value.contiguous() for name, value in predictions.items()},
            **{
                f"probability__{name}": value.contiguous()
                for name, value in probabilities.items()
            },
            "targets": full.targets.detach().cpu().contiguous(),
            "target_mask": full.target_mask.detach().cpu().contiguous(),
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
                "objective": "exact_positive_probability_mass_plus_0.25_pairwise",
                "checkpoint_selection": (
                    "final_fixed_epoch_only_after_nested_oof_pooling_selection"
                ),
            },
            "result": result,
            "patient_ids": list(full.patient_ids),
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
                "foundation_backbone": "LaBraM_not_replaced_not_trained_from_scratch",
                "ictal_path": "channel_agnostic_global_temporal_gate_only",
                "event_dependent_channel_localizer": "V_physical_channel_descriptors_only",
                "ictal_semantics": "retrospective_scalp_visible_involvement_not_soz",
                "evolution_semantics": "observable_descriptors_not_propagation_truth",
                "attention_semantics": "discriminative_temporal_weight_not_onset_or_propagation",
                "source_dev_reused_for_selection": False,
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
                    "selected_nested_oof_metrics": result["selected_nested_oof_metrics"],
                    "selected_vs_phase": result[
                        "selected_vs_phase_paired_patient_bootstrap"
                    ],
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
