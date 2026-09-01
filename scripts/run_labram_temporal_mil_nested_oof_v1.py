#!/usr/bin/env python3
"""Run the frozen LaBraM I+V temporal-MIL nested patient-OOF recovery trial."""

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

from src.soz.aggregation import aggregate_patient_logits  # noqa: E402
from src.soz.development_reasoner import (  # noqa: E402
    DevelopmentIVAdditiveReasoner,
)
from src.soz.development_reasoner_training_v1_1 import (  # noqa: E402
    FROZEN_SOURCE_TRAIN_TARGET_SCOPE_RECEIPT_SHA256,
    join_development_iv_split_targets_v1_1,
)
from src.soz.development_reasoner_v1_1 import (  # noqa: E402
    FROZEN_OOF_PROTOCOL_ARTIFACT_SHA256,
    FROZEN_OOF_PROTOCOL_RECEIPT_SHA256,
    FROZEN_SIGNAL_PREFLIGHT_ARTIFACT_SHA256,
    FROZEN_SIGNAL_PREFLIGHT_RECEIPT_SHA256,
    FROZEN_V1_1_CAPABILITY_MANIFEST_SHA256,
    load_development_iv_evidence_capability_v1_1,
)
from src.soz.development_target_scope_v1_1 import (  # noqa: E402
    load_development_target_scope_v1_1,
)
from src.soz.ictal_native_eval import (  # noqa: E402
    load_bound_deepsoz_signal_preflight_artifact,
)
from src.soz.ictal_recovery_evidence_v1_2 import (  # noqa: E402
    load_target_free_ictal_oof_protocol,
)
from src.soz.losses import PatientLevelSOZObjective  # noqa: E402
from src.soz.metrics import (  # noqa: E402
    deepsoz_style_top1_metrics,
    patient_localization_metrics,
)
from src.soz.temporal_mil_recovery import (  # noqa: E402
    TEMPORAL_MIL_RECOVERY_SCHEMA,
    TemporalMILEvidenceReasoner,
    TemporalMILPatientBatch,
    jeffreys_channel_prior_logits,
    subset_patient_batch,
    temporal_mil_objective,
)


PROTOCOL_PATH = (
    ROOT
    / "research/02_method/labram_temporal_mil_recovery_protocol_v1_20260810_zh.md"
)
CAPABILITY_PATH = (
    ROOT / "outputs/labram_iv_development_candidate_capability_v1_1_20260810"
)
OOF_PROTOCOL_PATH = ROOT / "outputs/ictal_concept_oof_protocol_v2_20260808"
SIGNAL_PREFLIGHT_PATH = (
    ROOT / "outputs/deepsoz_signal_preflight_v2_20260809_current"
)
TRAIN_TARGET_PATH = (
    ROOT / "outputs/development_target_scope_v1_1_final_20260810/train"
)

TEMPORAL_CANDIDATES: Mapping[str, float] = {
    "temporal_mil_exact": 0.0,
    "temporal_mil_neighbor_aux": 0.05,
}
OUTER_FOLDS = tuple(range(5))
TEMPORAL_EPOCHS = 100
TEMPORAL_LEARNING_RATE = 3e-3
BASELINE_EPOCHS = 20
BASELINE_LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
MAX_GRAD_NORM = 1.0
BOOTSTRAP_REPLICATES = 2000
BASE_SEED = 20260810


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        metadata = f"{name}|{tuple(tensor.shape)}|{tensor.dtype}".encode("ascii")
        digest.update(len(metadata).to_bytes(4, "little"))
        digest.update(metadata)
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def _seeded_model(
    prior: torch.Tensor,
    *,
    seed: int,
    device: torch.device,
) -> TemporalMILEvidenceReasoner:
    fork_devices: list[int] = []
    if device.type == "cuda":
        fork_devices = [device.index if device.index is not None else 0]
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(seed)
        return TemporalMILEvidenceReasoner(prior).to(device)


def _fit_temporal(
    train: TemporalMILPatientBatch,
    *,
    neighbor_weight: float,
    seed: int,
    device: torch.device,
) -> tuple[TemporalMILEvidenceReasoner, dict[str, object]]:
    prior = jeffreys_channel_prior_logits(train).detach().cpu()
    model = _seeded_model(prior, seed=seed, device=device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    batch = train.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TEMPORAL_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    first: dict[str, float] | None = None
    last: dict[str, float] | None = None
    for _ in range(TEMPORAL_EPOCHS):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(batch.evidence)
        loss = temporal_mil_objective(
            output.event_logits, batch, neighbor_weight=neighbor_weight
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
            "neighbor_auxiliary": float(loss.neighbor_auxiliary.detach().cpu()),
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
        "epochs": TEMPORAL_EPOCHS,
        "first_epoch": first,
        "final_epoch": last,
        "parameter_count": parameter_count,
    }


def _fit_phase_baseline(
    train: TemporalMILPatientBatch,
    *,
    seed: int,
    device: torch.device,
) -> tuple[DevelopmentIVAdditiveReasoner, dict[str, object]]:
    fork_devices: list[int] = []
    if device.type == "cuda":
        fork_devices = [device.index if device.index is not None else 0]
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(seed)
        model = DevelopmentIVAdditiveReasoner(hidden_dim=16).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=BASELINE_LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    objective = PatientLevelSOZObjective(
        ranking_weight=0.25, ranking_margin=0.0, require_positive=True
    )
    patient_batches = []
    for patient_index in range(len(train.patient_ids)):
        patient_batches.append(
            subset_patient_batch(
                train.evidence,
                train.event_patient_index,
                train.patient_ids,
                train.targets,
                train.target_mask,
                (patient_index,),
            ).to(device)
        )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    first_mean: float | None = None
    last_mean: float | None = None
    for _ in range(BASELINE_EPOCHS):
        losses = []
        order = torch.randperm(len(patient_batches), generator=generator).tolist()
        model.train()
        for patient_index in order:
            batch = patient_batches[patient_index]
            optimizer.zero_grad(set_to_none=True)
            event_logits = model(batch.evidence).event_logits
            patient_logits = aggregate_patient_logits(
                event_logits, batch.event_patient_index
            ).logits
            loss = objective(patient_logits, batch.targets, batch.target_mask)
            loss.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            losses.append(float(loss.total.detach().cpu()))
        mean = sum(losses) / len(losses)
        if first_mean is None:
            first_mean = mean
        last_mean = mean
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    model.eval()
    model.requires_grad_(False)
    assert first_mean is not None and last_mean is not None
    return model, {
        "seed": seed,
        "epochs": BASELINE_EPOCHS,
        "first_epoch_mean_total": first_mean,
        "final_epoch_mean_total": last_mean,
        "parameter_count": sum(value.numel() for value in model.parameters()),
    }


def _predict_temporal(
    model: TemporalMILEvidenceReasoner,
    batch: TemporalMILPatientBatch,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, object]]:
    moved = batch.to(device)
    model.eval()
    with torch.no_grad():
        output = model(moved.evidence)
        aggregation = aggregate_patient_logits(
            output.event_logits, moved.event_patient_index
        )
    patient_logits = aggregation.logits.detach().cpu()
    event_logits = output.event_logits.detach().cpu()
    temporal_weights = output.temporal_weights.detach().cpu()
    uncertainty = []
    for patient in range(len(batch.patient_ids)):
        event_rows = event_logits[batch.event_patient_index.cpu() == patient]
        observed = batch.target_mask[patient].cpu()
        patient_row = patient_logits[patient][observed]
        probability = torch.softmax(patient_row, dim=0)
        ordered = torch.sort(patient_row, descending=True).values
        patient_top = int(torch.argmax(patient_row).item())
        event_top = torch.argmax(event_rows[:, observed], dim=1)
        attention = temporal_weights[batch.event_patient_index.cpu() == patient]
        attention_entropy = -(
            attention.clamp_min(1e-8) * attention.clamp_min(1e-8).log()
        ).sum(dim=-1)
        uncertainty.append(
            {
                "patient_id": batch.patient_ids[patient],
                "event_count": int(event_rows.shape[0]),
                "top1_margin": float((ordered[0] - ordered[1]).item()),
                "ranking_entropy": float(
                    (-(probability * probability.clamp_min(1e-8).log()).sum()).item()
                ),
                "event_top1_disagreement_rate": float(
                    (event_top != patient_top).float().mean().item()
                ),
                "mean_channel_event_logit_std": float(
                    event_rows[:, observed].std(dim=0, unbiased=False).mean().item()
                ),
                "mean_temporal_weight_entropy": float(attention_entropy.mean().item()),
            }
        )
    return patient_logits, {"patients": uncertainty}


def _predict_phase(
    model: DevelopmentIVAdditiveReasoner,
    batch: TemporalMILPatientBatch,
    *,
    device: torch.device,
) -> torch.Tensor:
    moved = batch.to(device)
    model.eval()
    with torch.no_grad():
        event_logits = model(moved.evidence).event_logits
        logits = aggregate_patient_logits(
            event_logits, moved.event_patient_index
        ).logits
    return logits.detach().cpu()


def _metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, object]:
    ranking = asdict(
        patient_localization_metrics(logits, targets, mask, k_values=(1, 3, 5))
    )
    top1 = asdict(deepsoz_style_top1_metrics(logits, targets, mask))
    top1["spread_top1_rate"] = None
    return {"ranking": ranking, "top1": top1}


def _selection_key(metrics: Mapping[str, object], *, conservative: bool) -> tuple:
    top1 = metrics["top1"]
    ranking = metrics["ranking"]
    return (
        float(top1["strict_accuracy"]),
        float(ranking["macro_average_precision"]),
        float(ranking["mean_reciprocal_rank"]),
        int(conservative),
    )


def _metric_vector(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    report = _metrics(logits, targets, mask)
    return torch.tensor(
        [
            report["top1"]["strict_accuracy"],
            report["top1"]["relaxed_accuracy"],
            report["ranking"]["macro_average_precision"],
            report["ranking"]["mean_reciprocal_rank"],
        ],
        dtype=torch.float64,
    )


def _paired_bootstrap(
    candidate: torch.Tensor,
    baseline: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, object]:
    # All four endpoints are patient-macro means.  Compute each patient's
    # paired contribution once, then resample those rows.  This is exactly the
    # same patient-cluster bootstrap as repeatedly re-evaluating the metric,
    # while avoiding millions of redundant tie-aware ranking operations.
    patient_deltas = []
    for patient in range(targets.shape[0]):
        row = slice(patient, patient + 1)
        patient_deltas.append(
            _metric_vector(candidate[row], targets[row], mask[row])
            - _metric_vector(baseline[row], targets[row], mask[row])
        )
    row_deltas = torch.stack(patient_deltas)
    generator = torch.Generator().manual_seed(BASE_SEED)
    indices = torch.randint(
        0,
        targets.shape[0],
        (BOOTSTRAP_REPLICATES, targets.shape[0]),
        generator=generator,
    )
    samples = row_deltas[indices].mean(dim=1)
    point = row_deltas.mean(dim=0)
    names = ("strict_top1", "relaxed_top1", "macro_ap", "mrr")
    return {
        name: {
            "delta": float(point[index]),
            "ci95": [
                float(torch.quantile(samples[:, index], 0.025)),
                float(torch.quantile(samples[:, index], 0.975)),
            ],
        }
        for index, name in enumerate(names)
    }


def _load_source_train() -> tuple[
    TemporalMILPatientBatch,
    tuple[int, ...],
    dict[str, object],
]:
    protocol = load_target_free_ictal_oof_protocol(
        OOF_PROTOCOL_PATH,
        expected_artifact_sha256=FROZEN_OOF_PROTOCOL_ARTIFACT_SHA256,
        expected_protocol_receipt_sha256=FROZEN_OOF_PROTOCOL_RECEIPT_SHA256,
    )
    signal = load_bound_deepsoz_signal_preflight_artifact(
        SIGNAL_PREFLIGHT_PATH,
        expected_artifact_sha256=FROZEN_SIGNAL_PREFLIGHT_ARTIFACT_SHA256,
        expected_receipt_sha256=FROZEN_SIGNAL_PREFLIGHT_RECEIPT_SHA256,
    )
    capability = load_development_iv_evidence_capability_v1_1(
        CAPABILITY_PATH,
        signal,
        protocol,
        expected_manifest_sha256=FROZEN_V1_1_CAPABILITY_MANIFEST_SHA256,
    )
    target = load_development_target_scope_v1_1(
        TRAIN_TARGET_PATH,
        expected_model_split="source_train",
        expected_receipt_file_sha256=(
            FROZEN_SOURCE_TRAIN_TARGET_SCOPE_RECEIPT_SHA256
        ),
    )
    joined = join_development_iv_split_targets_v1_1(capability, target)
    if joined.model_split != "source_train":
        raise RuntimeError("nested OOF received a non-train split")
    full = joined.dataset.full_batch()
    source = capability.capability.base.capability.source_train
    if tuple(full.event_ids) != tuple(source.event_ids) or (
        tuple(full.patient_ids) != tuple(source.patient_ids)
    ):
        raise RuntimeError("source-train evidence/target event identity drifted")
    folds_by_patient: dict[str, set[int | None]] = {}
    for patient_id, fold in zip(source.patient_ids_by_event, source.oof_folds):
        folds_by_patient.setdefault(patient_id, set()).add(fold)
    if any(len(values) != 1 for values in folds_by_patient.values()):
        raise RuntimeError("a source-train patient crosses upstream OOF folds")
    patient_folds = tuple(
        int(next(iter(folds_by_patient[patient_id])))
        for patient_id in full.patient_ids
    )
    if set(patient_folds) != set(OUTER_FOLDS):
        raise RuntimeError("source-train does not cover the frozen five folds")
    complete = TemporalMILPatientBatch(
        evidence=full.evidence,
        event_patient_index=full.event_patient_index,
        patient_ids=full.patient_ids,
        targets=full.targets,
        target_mask=full.target_mask,
    )
    lineage = {
        "split_dataset_receipt_sha256": joined.receipt.receipt_sha256,
        "capability_manifest_sha256": FROZEN_V1_1_CAPABILITY_MANIFEST_SHA256,
        "oof_protocol_artifact_sha256": FROZEN_OOF_PROTOCOL_ARTIFACT_SHA256,
        "oof_protocol_receipt_sha256": FROZEN_OOF_PROTOCOL_RECEIPT_SHA256,
        "signal_preflight_artifact_sha256": FROZEN_SIGNAL_PREFLIGHT_ARTIFACT_SHA256,
        "signal_preflight_receipt_sha256": FROZEN_SIGNAL_PREFLIGHT_RECEIPT_SHA256,
        "train_target_scope_receipt_sha256": (
            FROZEN_SOURCE_TRAIN_TARGET_SCOPE_RECEIPT_SHA256
        ),
        "source_train_patient_count": len(full.patient_ids),
        "source_train_event_count": full.evidence.batch_size,
    }
    return complete, patient_folds, lineage


def _indices_for_folds(patient_folds: Sequence[int], folds: Sequence[int]) -> tuple[int, ...]:
    accepted = set(int(value) for value in folds)
    return tuple(
        index for index, fold in enumerate(patient_folds) if fold in accepted
    )


def _subset(full: TemporalMILPatientBatch, indices: Sequence[int]) -> TemporalMILPatientBatch:
    return subset_patient_batch(
        full.evidence,
        full.event_patient_index,
        full.patient_ids,
        full.targets,
        full.target_mask,
        indices,
    )


def _run(
    full: TemporalMILPatientBatch,
    patient_folds: tuple[int, ...],
    *,
    device: torch.device,
) -> tuple[dict[str, object], dict[str, torch.Tensor], TemporalMILEvidenceReasoner]:
    n_patients = len(full.patient_ids)
    predictions = {
        "selected": torch.full((n_patients, 19), torch.nan),
        "temporal_mil_exact": torch.full((n_patients, 19), torch.nan),
        "temporal_mil_neighbor_aux": torch.full((n_patients, 19), torch.nan),
        "phase_baseline": torch.full((n_patients, 19), torch.nan),
        "fold_local_prevalence": torch.full((n_patients, 19), torch.nan),
    }
    outer_rows = []
    selection_counts = {name: 0 for name in TEMPORAL_CANDIDATES}
    all_uncertainty: list[dict[str, object]] = []

    for outer_fold in OUTER_FOLDS:
        outer_train_folds = tuple(fold for fold in OUTER_FOLDS if fold != outer_fold)
        inner_predictions = {
            name: torch.full((n_patients, 19), torch.nan)
            for name in TEMPORAL_CANDIDATES
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
            for candidate_index, (name, neighbor_weight) in enumerate(
                TEMPORAL_CANDIDATES.items()
            ):
                seed = (
                    BASE_SEED
                    + outer_fold * 100
                    + inner_fold * 10
                    + candidate_index
                )
                model, fit = _fit_temporal(
                    train,
                    neighbor_weight=neighbor_weight,
                    seed=seed,
                    device=device,
                )
                logits, _ = _predict_temporal(model, validation, device=device)
                inner_predictions[name][list(validation_indices)] = logits
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
        candidate_inner_metrics = {}
        index_tensor = torch.tensor(outer_train_indices, dtype=torch.long)
        for name in TEMPORAL_CANDIDATES:
            candidate_inner_metrics[name] = _metrics(
                inner_predictions[name].index_select(0, index_tensor),
                full.targets.index_select(0, index_tensor),
                full.target_mask.index_select(0, index_tensor),
            )
        selected = max(
            TEMPORAL_CANDIDATES,
            key=lambda name: _selection_key(
                candidate_inner_metrics[name],
                conservative=name == "temporal_mil_exact",
            ),
        )
        selection_counts[selected] += 1

        outer_train = _subset(full, outer_train_indices)
        held_indices = _indices_for_folds(patient_folds, (outer_fold,))
        held = _subset(full, held_indices)
        outer_candidate_rows = {}
        for candidate_index, (name, neighbor_weight) in enumerate(
            TEMPORAL_CANDIDATES.items()
        ):
            seed = BASE_SEED + outer_fold * 1000 + candidate_index
            model, fit = _fit_temporal(
                outer_train,
                neighbor_weight=neighbor_weight,
                seed=seed,
                device=device,
            )
            logits, uncertainty = _predict_temporal(model, held, device=device)
            predictions[name][list(held_indices)] = logits
            if name == selected:
                predictions["selected"][list(held_indices)] = logits
                all_uncertainty.extend(uncertainty["patients"])
            outer_candidate_rows[name] = {
                "fit": fit,
                "held_metrics": _metrics(logits, held.targets, held.target_mask),
            }

        baseline, baseline_fit = _fit_phase_baseline(
            outer_train,
            seed=BASE_SEED + outer_fold * 1000 + 99,
            device=device,
        )
        baseline_logits = _predict_phase(baseline, held, device=device)
        predictions["phase_baseline"][list(held_indices)] = baseline_logits
        prior = jeffreys_channel_prior_logits(outer_train).unsqueeze(0)
        predictions["fold_local_prevalence"][list(held_indices)] = prior.repeat(
            len(held_indices), 1
        )
        outer_rows.append(
            {
                "outer_fold": outer_fold,
                "outer_train_patient_count": len(outer_train_indices),
                "held_patient_count": len(held_indices),
                "selected_candidate": selected,
                "inner_candidate_metrics": candidate_inner_metrics,
                "inner_fits": inner_rows,
                "outer_candidates": outer_candidate_rows,
                "phase_baseline": {
                    "fit": baseline_fit,
                    "held_metrics": _metrics(
                        baseline_logits, held.targets, held.target_mask
                    ),
                },
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

    if any(not torch.isfinite(value).all() for value in predictions.values()):
        raise RuntimeError("nested OOF left a patient prediction unfilled")
    candidate_metrics = {
        name: _metrics(logits, full.targets, full.target_mask)
        for name, logits in predictions.items()
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
                conservative=name == "temporal_mil_exact",
            ),
        )
        final_selection_basis = "majority_tie_then_full_oof_strict_ap_mrr"

    final_model, final_fit = _fit_temporal(
        full,
        neighbor_weight=TEMPORAL_CANDIDATES[final_candidate],
        seed=BASE_SEED + 99999,
        device=device,
    )
    final_model = final_model.cpu()
    selected_vs_phase = _paired_bootstrap(
        predictions["selected"],
        predictions["phase_baseline"],
        full.targets,
        full.target_mask,
    )
    result = {
        "outer_folds": outer_rows,
        "selection_counts": selection_counts,
        "selected_nested_oof_metrics": candidate_metrics["selected"],
        "all_candidate_oof_metrics": candidate_metrics,
        "selected_vs_phase_paired_patient_bootstrap": selected_vs_phase,
        "final_candidate": final_candidate,
        "final_selection_basis": final_selection_basis,
        "final_fit": final_fit,
        "uncertainty": {
            "semantics": (
                "ranking_margin_entropy_event_dispersion_and_attention_entropy;"
                "not_bayesian_posterior_uncertainty"
            ),
            "patients": sorted(all_uncertainty, key=lambda row: row["patient_id"]),
        },
    }
    return result, predictions, final_model


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
    fold_counts = {
        str(fold): sum(value == fold for value in patient_folds)
        for fold in OUTER_FOLDS
    }
    preflight = {
        "status": "ready_nested_source_train_only",
        "schema_version": TEMPORAL_MIL_RECOVERY_SCHEMA,
        "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)),
        "protocol_sha256": _file_sha256(PROTOCOL_PATH),
        "device": str(device),
        "patient_count": len(full.patient_ids),
        "event_count": full.evidence.batch_size,
        "fold_counts": fold_counts,
        "lineage": lineage,
        "temporal_candidates": dict(TEMPORAL_CANDIDATES),
        "source_dev_forward_count": 0,
        "source_dev_target_values_reachable": False,
        "source_dev_evidence_loaded_by_strict_container_only": True,
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
        CAPABILITY_PATH,
        OOF_PROTOCOL_PATH,
        SIGNAL_PREFLIGHT_PATH,
        TRAIN_TARGET_PATH,
        PROTOCOL_PATH,
    ):
        resolved = source.resolve(strict=True)
        if output == resolved or output in resolved.parents or resolved in output.parents:
            raise ValueError("output path overlaps an input path")

    result, predictions, final_model = _run(
        full, patient_folds, device=device
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
                "temporal_epochs": TEMPORAL_EPOCHS,
                "temporal_learning_rate": TEMPORAL_LEARNING_RATE,
                "baseline_epochs": BASELINE_EPOCHS,
                "baseline_learning_rate": BASELINE_LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "max_grad_norm": MAX_GRAD_NORM,
                "base_seed": BASE_SEED,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "checkpoint_selection": "final_fixed_epoch_only_after_nested_oof_architecture_selection",
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
                "morphology_present": False,
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
                    "selected_nested_oof_metrics": result[
                        "selected_nested_oof_metrics"
                    ],
                    "selected_vs_phase_paired_patient_bootstrap": result[
                        "selected_vs_phase_paired_patient_bootstrap"
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
