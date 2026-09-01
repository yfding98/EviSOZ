#!/usr/bin/env python3
"""Run the source-train-only robust temporal-MIL and matched-direct OOF trial.

The script has no source-dev, source-eval, or private input.  It compares one
promotable intervention (robust complete-event-bag pooling) against an equal-
pooling mechanism control under the same pure positive-set objective, the
historical temporal_mil_exact anchor, matched frozen-LaBraM direct controls,
and a fold-local prevalence prior.
"""

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
import time
from typing import Mapping, Sequence

import torch
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soz.data.deepsoz_target_v2 import TARGET_V2_POLICY_SHA256  # noqa: E402
from src.soz.development_reasoner import DevelopmentIVEvidenceBatch  # noqa: E402
from src.soz.clinical_reporting import LATERALITY_GROUPS  # noqa: E402
from src.soz.development_reasoner_training_v1_1 import (  # noqa: E402
    FROZEN_SOURCE_TRAIN_TARGET_SCOPE_RECEIPT_SHA256,
)
from src.soz.geometry import CHANNEL_INDEX  # noqa: E402
from src.soz.labram_peft_prefix_cache import (  # noqa: E402
    load_labram_peft_prefix_cache,
)
from src.soz.metrics import (  # noqa: E402
    DEEPSOZ_STANDARD19_NEIGHBORS,
    deepsoz_style_top1_metrics,
    patient_localization_metrics,
)
from src.soz.models.baselines import MatchedDirectFrozenTokenHead  # noqa: E402
from src.soz.robust_temporal_mil_candidate import (  # noqa: E402
    ROBUST_TEMPORAL_MIL_SCHEMA,
    CompletePatientBagAggregation,
    aggregate_complete_patient_bags,
    compress_target_free_event_reliability,
    observational_patient_uncertainty,
    positive_set_mil_objective,
    restore_block9_physical_tokens,
)
from src.soz.source_train_iv_capability import (  # noqa: E402
    load_and_join_source_train_iv_target_scope,
)
from src.soz.temporal_mil_recovery import (  # noqa: E402
    TemporalMILEvidenceReasoner,
    TemporalMILPatientBatch,
    jeffreys_channel_prior_logits,
    subset_patient_batch,
)
from src.soz.v11_reasoner import V11_CANDIDATE_MASK  # noqa: E402


SCHEMA = "soz_labram_robust_temporal_mil_direct_oof_v14"
DEFAULT_SOURCE_TRAIN_IV = (
    ROOT / "outputs/labram_iv_source_train_only_capability_v1_20260811"
)
EXPECTED_SOURCE_TRAIN_IV_MANIFEST = (
    "ccd238b17e1da0aa24f2542a314c770900eeed71cbc31282a4acb76dcf957821"
)
DEFAULT_TARGET = ROOT / "outputs/development_target_scope_v1_1_final_20260810/train"
DEFAULT_PREFIX = ROOT / "outputs/labram_peft_prefix_cache_v8_20260811"
EXPECTED_PREFIX_MANIFEST = (
    "82679da220ecbd3c09c01b8badc6a2d610b42bc16cf717ce73ec6ab443c97ff4"
)
DEFAULT_ANCHOR = ROOT / "outputs/labram_temporal_mil_nested_oof_v1_20260810"
DEFAULT_V_SPATIAL_COMPARATOR = (
    ROOT / "outputs/labram_global_i_v_nested_oof_v2_20260810"
)
DEFAULT_OUTPUT = ROOT / "outputs/labram_robust_temporal_mil_direct_oof_v14_20260811"

TEMPORAL_EQUAL = "temporal_equal_positive_set_control"
TEMPORAL_ROBUST = "temporal_robust_positive_set_candidate"
DIRECT_EQUAL = "low_capacity_direct_token_equal_positive_set_control"
DIRECT_ROBUST = "low_capacity_direct_token_robust_positive_set_control"
PREVALENCE = "fold_local_prevalence"
HISTORICAL_ANCHOR = "historical_temporal_mil_exact"
HISTORICAL_PHASE_IV = "historical_phase_I_plus_V_baseline"
HISTORICAL_V_SPATIAL = "historical_V_spatial_with_shared_global_I_gate"
LEARNED_ARMS = (TEMPORAL_EQUAL, TEMPORAL_ROBUST, DIRECT_EQUAL, DIRECT_ROBUST)
OUTER_FOLDS = tuple(range(5))
EPOCHS = 100
LEARNING_RATE = 3.0e-3
WEIGHT_DECAY = 1.0e-2
MAX_GRAD_NORM = 1.0
BASE_SEED = 20260814
BOOTSTRAP_REPLICATES = 2000
EXPECTED_ANCHOR_MANIFEST_SHA256 = (
    "58cbfcc3d25e8ff4b13ab93e388e8aa5691e1c8fc9dc515ec2e8b51b226c9811"
)
EXPECTED_ANCHOR_PREDICTIONS_SHA256 = (
    "9373dc6bf269002c812ae26ca6ea8365b7518d3396037c4fc5b3a67603e1211d"
)
EXPECTED_V_SPATIAL_MANIFEST_SHA256 = (
    "3504626b63260635c81a6abb036a70a34346edb22f7cde6efb7fd08a0b8596fb"
)
EXPECTED_V_SPATIAL_PREDICTIONS_SHA256 = (
    "f61ac28e25bf85264b7a49c8b6d8bfe96fd25fa25899e8fe098a5e5fef1db824"
)

# Locked before this runner opens/summarizes any new OOF result.  This remains
# an exploratory mechanism gate because the same 65 patients have been used
# repeatedly in prior development.
LOCKED_PROMOTION_GATE = {
    "strict_expected_hits_minimum": 45.0,
    "relaxed_expected_hits_minimum": 55.0,
    "macro_average_precision_minimum": 0.6328,
    "strict_expected_hit_gain_over_anchor_minimum": 3.0,
    "far_error_expected_count_must_not_increase_vs_anchor": True,
    "wrong_hemisphere_expected_count_must_not_increase_vs_anchor": True,
    "strict_must_exceed_equal_positive_set_control": True,
    "strict_must_not_be_lower_than_low_capacity_block9_direct_robust": True,
}


def _top_tie_diagnostics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
) -> dict[str, float | int]:
    """Tie-aware far-error and wrong-hemisphere expected counts."""

    left = {CHANNEL_INDEX[name] for name in LATERALITY_GROUPS["left"]}
    right = {CHANNEL_INDEX[name] for name in LATERALITY_GROUPS["right"]}
    far_expected = 0.0
    wrong_expected = 0.0
    hemisphere_eligible = 0
    for patient in range(logits.shape[0]):
        observed = target_mask[patient]
        observed_indices = torch.nonzero(observed, as_tuple=False).flatten()
        row = logits[patient, observed]
        top = observed_indices[row == row.max()].tolist()
        positive = observed & (targets[patient] == 1)
        accepted = positive.clone()
        if int(positive.sum()) <= 4:
            for index in torch.nonzero(positive, as_tuple=False).flatten().tolist():
                accepted[list(DEEPSOZ_STANDARD19_NEIGHBORS[index])] = True
            accepted &= observed
        far_expected += sum(not bool(accepted[index]) for index in top) / len(top)

        positive_indices = set(
            torch.nonzero(positive, as_tuple=False).flatten().tolist()
        )
        positive_left = bool(positive_indices & left)
        positive_right = bool(positive_indices & right)
        if positive_left != positive_right:
            hemisphere_eligible += 1
            wrong_side = right if positive_left else left
            wrong_expected += sum(index in wrong_side for index in top) / len(top)
    return {
        "far_error_expected_count": far_expected,
        "wrong_hemisphere_expected_count": wrong_expected,
        "wrong_hemisphere_eligible_patient_count": hemisphere_eligible,
        "wrong_hemisphere_expected_rate": (
            wrong_expected / hemisphere_eligible if hemisphere_eligible else 0.0
        ),
    }


def _metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
) -> dict[str, object]:
    top1 = asdict(deepsoz_style_top1_metrics(logits, targets, target_mask))
    # Public DeepSOZ provides no spread-electrode reference.  The metric
    # library's numeric default must not be read as observed absence of spread.
    top1["spread_top1_rate"] = None
    result = {
        "ranking": asdict(
            patient_localization_metrics(
                logits, targets, target_mask, k_values=(1, 3, 5)
            )
        ),
        "top1": top1,
    }
    result["diagnostics"] = _top_tie_diagnostics(logits, targets, target_mask)
    return result


def _patient_metric_rows(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    rows = []
    for patient in range(logits.shape[0]):
        report = _metrics(
            logits[patient : patient + 1],
            targets[patient : patient + 1],
            target_mask[patient : patient + 1],
        )
        rows.append(
            torch.tensor(
                (
                    report["top1"]["strict_accuracy"],
                    report["top1"]["relaxed_accuracy"],
                    report["ranking"]["macro_average_precision"],
                    report["ranking"]["mean_reciprocal_rank"],
                ),
                dtype=torch.float64,
            )
        )
    return torch.stack(rows)


def _paired_bootstrap(
    candidate: torch.Tensor,
    baseline: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
) -> dict[str, object]:
    difference = _patient_metric_rows(
        candidate, targets, target_mask
    ) - _patient_metric_rows(baseline, targets, target_mask)
    generator = torch.Generator().manual_seed(BASE_SEED)
    indices = torch.randint(
        0,
        difference.shape[0],
        (BOOTSTRAP_REPLICATES, difference.shape[0]),
        generator=generator,
    )
    samples = difference[indices].mean(dim=1)
    names = ("strict_top1", "relaxed_top1", "macro_ap", "mrr")
    return {
        name: {
            "delta": float(difference[:, column].mean()),
            "ci95": [
                float(torch.quantile(samples[:, column], 0.025)),
                float(torch.quantile(samples[:, column], 0.975)),
            ],
        }
        for column, name in enumerate(names)
    }


def _load_inputs(
    args: argparse.Namespace,
) -> tuple[
    object,
    TemporalMILPatientBatch,
    tuple[int, ...],
    tuple[str, ...],
    Mapping[str, str],
]:
    cache = load_labram_peft_prefix_cache(
        args.prefix_directory,
        expected_manifest_sha256=EXPECTED_PREFIX_MANIFEST,
        require_full_scope=True,
    )
    joined = load_and_join_source_train_iv_target_scope(
        args.source_train_iv,
        args.target_directory,
        expected_capability_manifest_sha256=EXPECTED_SOURCE_TRAIN_IV_MANIFEST,
        expected_target_receipt_file_sha256=(
            FROZEN_SOURCE_TRAIN_TARGET_SCOPE_RECEIPT_SHA256
        ),
    )
    patient = joined.batch
    full = TemporalMILPatientBatch(
        evidence=patient.evidence,
        event_patient_index=patient.event_patient_index,
        patient_ids=patient.patient_ids,
        targets=patient.targets,
        target_mask=patient.target_mask,
    )
    event_ids = tuple(patient.event_ids)
    patient_folds = tuple(int(value) for value in joined.patient_folds)
    patient_by_event = tuple(
        full.patient_ids[int(index)] for index in full.event_patient_index.tolist()
    )
    folds_by_event = tuple(
        patient_folds[int(index)] for index in full.event_patient_index.tolist()
    )
    checks = {
        # The strict loader above hard-requires expected_model_split=source_train;
        # SourceTrainIVTargetJoin deliberately exposes receipts rather than a
        # redundant model_split attribute.
        "source-train evidence receipt": joined.evidence_manifest_sha256
        == EXPECTED_SOURCE_TRAIN_IV_MANIFEST,
        "source-train target receipt": joined.target_receipt_file_sha256
        == FROZEN_SOURCE_TRAIN_TARGET_SCOPE_RECEIPT_SHA256,
        "65 patients": len(full.patient_ids) == 65,
        "582 events": full.evidence.batch_size == 582,
        "five patient folds": set(patient_folds) == set(OUTER_FOLDS),
        "cache event IDs": tuple(cache.event_ids) == event_ids,
        "cache patients": tuple(cache.patient_ids_by_event) == patient_by_event,
        "cache folds": tuple(int(value) for value in cache.oof_folds) == folds_by_event,
        "cache detached": not cache.tokens.requires_grad,
        "cache tensor shape": tuple(cache.tokens.shape) == (582, 15, 77, 200),
        "cache source-train split": cache.manifest.get("model_split")
        == "source_train",
        "cache target-free": cache.manifest.get("deepsoz_target_values_loaded")
        is False,
        "source dev absent": cache.manifest.get("source_dev_used") is False,
        "source eval absent": cache.manifest.get("source_eval_used") is False,
        "private absent": cache.manifest.get("private_used") is False,
        "fixed PZ-only candidate mask": torch.equal(
            full.target_mask,
            V11_CANDIDATE_MASK.view(1, -1).expand_as(full.target_mask),
        ),
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"v14 source-train input boundary failed: {failed}")
    roster_bytes = json.dumps(
        event_ids, ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    lineage = {
        "prefix_cache_manifest_sha256": cache.manifest_sha256,
        "source_train_iv_manifest_sha256": joined.evidence_manifest_sha256,
        "source_train_iv_receipt_sha256": joined.evidence_receipt_sha256,
        "source_train_target_receipt_file_sha256": (
            joined.target_receipt_file_sha256
        ),
        "event_roster_sha256": hashlib.sha256(roster_bytes).hexdigest(),
    }
    return cache, full, patient_folds, event_ids, lineage


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_anchor(
    directory: Path,
    full: TemporalMILPatientBatch,
    patient_folds: Sequence[int],
) -> Mapping[str, torch.Tensor]:
    manifest_path = directory / "manifest.json"
    predictions_path = directory / "oof_predictions.safetensors"
    if _file_sha256(manifest_path) != EXPECTED_ANCHOR_MANIFEST_SHA256 or (
        _file_sha256(predictions_path) != EXPECTED_ANCHOR_PREDICTIONS_SHA256
    ):
        raise ValueError("historical anchor differs from the pinned input")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("patient_ids") != list(full.patient_ids) or manifest.get(
        "patient_folds"
    ) != list(patient_folds):
        raise ValueError("historical anchor patient roster/folds differ")
    if manifest.get("source_eval_used") is not False or manifest.get(
        "private_used"
    ) is not False:
        raise ValueError("historical anchor is outside the source-train boundary")
    tensors = load_file(str(predictions_path), device="cpu")
    if not torch.equal(tensors["targets"], full.targets.cpu()) or not torch.equal(
        tensors["target_mask"], full.target_mask.cpu()
    ):
        raise ValueError("historical anchor target carrier differs")
    return {
        HISTORICAL_ANCHOR: tensors["temporal_mil_exact"].detach().contiguous(),
        HISTORICAL_PHASE_IV: tensors["phase_baseline"].detach().contiguous(),
    }


def _load_v_spatial_comparator(
    directory: Path,
    full: TemporalMILPatientBatch,
    patient_folds: Sequence[int],
) -> torch.Tensor:
    """Load the historical V-localizing/shared-global-I-gate OOF comparator."""

    manifest_path = directory / "manifest.json"
    predictions_path = directory / "oof_predictions.safetensors"
    if _file_sha256(manifest_path) != EXPECTED_V_SPATIAL_MANIFEST_SHA256 or (
        _file_sha256(predictions_path) != EXPECTED_V_SPATIAL_PREDICTIONS_SHA256
    ):
        raise ValueError("historical V-spatial comparator differs from pinned input")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("patient_ids") != list(full.patient_ids) or manifest.get(
        "patient_folds"
    ) != list(patient_folds):
        raise ValueError("historical V-spatial comparator roster/folds differ")
    if manifest.get("source_eval_used") is not False or manifest.get(
        "private_used"
    ) is not False:
        raise ValueError("historical V-spatial comparator is outside source-train")
    tensors = load_file(str(predictions_path), device="cpu")
    if not torch.equal(tensors["targets"], full.targets.cpu()) or not torch.equal(
        tensors["target_mask"], full.target_mask.cpu()
    ) or not torch.equal(
        tensors["patient_folds"], torch.tensor(patient_folds, dtype=torch.long)
    ):
        raise ValueError("historical V-spatial target/fold carrier differs")
    return tensors["global_i_v_equal_probability_mean"].detach().contiguous()


def _patient_indices(patient_folds: Sequence[int], fold: int, *, held: bool) -> tuple[int, ...]:
    return tuple(
        index
        for index, value in enumerate(patient_folds)
        if (value == fold) is held
    )


def _subset(
    full: TemporalMILPatientBatch,
    patient_indices: Sequence[int],
) -> tuple[TemporalMILPatientBatch, torch.Tensor]:
    selected = tuple(int(value) for value in patient_indices)
    patient_selector = torch.zeros(len(full.patient_ids), dtype=torch.bool)
    patient_selector[torch.tensor(selected, dtype=torch.long)] = True
    event_indices = torch.nonzero(
        patient_selector[full.event_patient_index], as_tuple=False
    ).flatten()
    batch = subset_patient_batch(
        full.evidence,
        full.event_patient_index,
        full.patient_ids,
        full.targets,
        full.target_mask,
        selected,
    )
    if event_indices.numel() != batch.evidence.batch_size:
        raise RuntimeError("patient subset did not preserve the complete event bag")
    return batch, event_indices


def _seeded_temporal(
    prior: torch.Tensor, seed: int, device: torch.device
) -> TemporalMILEvidenceReasoner:
    fork_devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(seed)
        return TemporalMILEvidenceReasoner(prior).to(device)


def _seeded_direct(
    prior: torch.Tensor, seed: int, device: torch.device
) -> MatchedDirectFrozenTokenHead:
    fork_devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(seed)
        model = MatchedDirectFrozenTokenHead(token_dim=200)
    with torch.no_grad():
        model.channel_bias.copy_(prior)
    # Use exactly the same fold-local prior contract as temporal-MIL.  The
    # direct comparator trains only evidence weights, not a second prevalence
    # model hidden inside its channel bias.
    model.channel_bias.requires_grad_(False)
    return model.to(device)


def _fit_temporal(
    batch: TemporalMILPatientBatch,
    *,
    pooling: str,
    seed: int,
    device: torch.device,
) -> tuple[TemporalMILEvidenceReasoner, Mapping[str, object]]:
    prior = jeffreys_channel_prior_logits(batch).detach().cpu()
    model = _seeded_temporal(prior, seed, device)
    moved = batch.to(device)
    reliability = compress_target_free_event_reliability(moved.evidence).values
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    first = None
    last = None
    for _ in range(EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        event_logits = model(moved.evidence).event_logits
        objective = positive_set_mil_objective(
            event_logits, moved, reliability, pooling=pooling
        )
        objective.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        value = float(objective.total.detach().cpu())
        first = value if first is None else first
        last = value
    model.eval().requires_grad_(False)
    return model, {
        "seed": seed,
        "epochs": EPOCHS,
        "first_positive_set_mass": first,
        "final_positive_set_mass": last,
        "trainable_parameter_count": model.n_trainable_parameters,
        "pooling": pooling,
        "objective": "pure_patient_positive_set_mass",
    }


def _direct_masks(evidence: DevelopmentIVEvidenceBatch) -> torch.Tensor:
    return evidence.evolution_mask & evidence.phase_mask.unsqueeze(1)


def _fit_direct(
    batch: TemporalMILPatientBatch,
    physical_tokens: torch.Tensor,
    *,
    pooling: str,
    seed: int,
    device: torch.device,
) -> tuple[MatchedDirectFrozenTokenHead, Mapping[str, object]]:
    prior = jeffreys_channel_prior_logits(batch).detach().cpu()
    model = _seeded_direct(prior, seed, device)
    moved = batch.to(device)
    tokens = physical_tokens.to(device=device)
    token_mask = _direct_masks(moved.evidence)
    reliability = compress_target_free_event_reliability(moved.evidence).values
    parameters = tuple(value for value in model.parameters() if value.requires_grad)
    optimizer = torch.optim.AdamW(
        parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    first = None
    last = None
    for _ in range(EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        event_logits = model(tokens, token_mask, moved.evidence.phase_mask).event_logits
        objective = positive_set_mil_objective(
            event_logits, moved, reliability, pooling=pooling
        )
        objective.total.backward()
        torch.nn.utils.clip_grad_norm_(parameters, MAX_GRAD_NORM)
        optimizer.step()
        value = float(objective.total.detach().cpu())
        first = value if first is None else first
        last = value
    model.eval().requires_grad_(False)
    return model, {
        "seed": seed,
        "epochs": EPOCHS,
        "first_positive_set_mass": first,
        "final_positive_set_mass": last,
        "trainable_parameter_count": sum(value.numel() for value in parameters),
        "fixed_fold_local_prior_parameter_count": 19,
        "pooling": pooling,
        "objective": "pure_patient_positive_set_mass",
    }


@torch.no_grad()
def _predict_temporal(
    model: TemporalMILEvidenceReasoner,
    evidence: DevelopmentIVEvidenceBatch,
    event_patient_index: torch.Tensor,
    n_patients: int,
    *,
    pooling: str,
    device: torch.device,
) -> tuple[torch.Tensor, CompletePatientBagAggregation, Mapping[str, torch.Tensor]]:
    moved_evidence = evidence.to(device)
    moved_index = event_patient_index.to(device=device)
    event_logits = model(moved_evidence).event_logits
    reliability = compress_target_free_event_reliability(moved_evidence).values
    aggregation = aggregate_complete_patient_bags(
        event_logits,
        moved_index,
        n_patients,
        reliability,
        pooling=pooling,
    )
    uncertainty = observational_patient_uncertainty(
        event_logits,
        moved_index,
        V11_CANDIDATE_MASK.to(device),
        aggregation,
    )
    return event_logits.cpu(), aggregation, asdict(uncertainty)


@torch.no_grad()
def _predict_direct(
    model: MatchedDirectFrozenTokenHead,
    evidence: DevelopmentIVEvidenceBatch,
    event_patient_index: torch.Tensor,
    n_patients: int,
    physical_tokens: torch.Tensor,
    *,
    pooling: str,
    device: torch.device,
) -> tuple[torch.Tensor, CompletePatientBagAggregation, Mapping[str, torch.Tensor]]:
    moved_evidence = evidence.to(device)
    moved_index = event_patient_index.to(device=device)
    tokens = physical_tokens.to(device=device)
    event_logits = model(
        tokens, _direct_masks(moved_evidence), moved_evidence.phase_mask
    ).event_logits
    reliability = compress_target_free_event_reliability(moved_evidence).values
    aggregation = aggregate_complete_patient_bags(
        event_logits,
        moved_index,
        n_patients,
        reliability,
        pooling=pooling,
    )
    uncertainty = observational_patient_uncertainty(
        event_logits,
        moved_index,
        V11_CANDIDATE_MASK.to(device),
        aggregation,
    )
    return event_logits.cpu(), aggregation, asdict(uncertainty)


def _state(prefix: str, model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        f"{prefix}.{name}": value.detach().cpu().contiguous()
        for name, value in model.state_dict().items()
    }


def _run(
    cache: object,
    full: TemporalMILPatientBatch,
    patient_folds: tuple[int, ...],
    historical: Mapping[str, torch.Tensor],
    input_lineage: Mapping[str, str],
    *,
    device: torch.device,
) -> tuple[dict[str, object], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    physical_tokens = restore_block9_physical_tokens(cache.tokens)
    patients = len(full.patient_ids)
    events = full.evidence.batch_size
    predictions = {
        name: torch.full((patients, 19), torch.nan) for name in LEARNED_ARMS
    }
    predictions[PREVALENCE] = torch.full((patients, 19), torch.nan)
    event_predictions = {
        name: torch.full((events, 19), torch.nan) for name in LEARNED_ARMS
    }
    uncertainty = {
        name: {
            field: torch.full((patients,), torch.nan)
            for field in (
                "normalized_ranking_entropy",
                "top1_margin",
                "mean_event_dispersion",
                "event_top1_disagreement",
                "mean_effective_event_count",
            )
        }
        for name in LEARNED_ARMS
    }
    states: dict[str, torch.Tensor] = {}
    fold_rows = []
    started = time.monotonic()
    for fold in OUTER_FOLDS:
        train_indices = _patient_indices(patient_folds, fold, held=False)
        held_indices = _patient_indices(patient_folds, fold, held=True)
        train, train_events = _subset(full, train_indices)
        held, held_events = _subset(full, held_indices)
        train_tokens = physical_tokens.index_select(0, train_events)
        held_tokens = physical_tokens.index_select(0, held_events)
        seed = BASE_SEED + fold * 1000
        fold_fits: dict[str, object] = {}

        # Equal and robust arms within a family use identical initialization,
        # seed, split, epoch count, optimizer, and pure positive-set objective.
        for arm, pooling in (
            (TEMPORAL_EQUAL, "equal"),
            (TEMPORAL_ROBUST, "quality_winsorized"),
        ):
            model, fit = _fit_temporal(
                train, pooling=pooling, seed=seed, device=device
            )
            event_logits, aggregation, arm_uncertainty = _predict_temporal(
                model,
                held.evidence,
                held.event_patient_index,
                len(held.patient_ids),
                pooling=pooling,
                device=device,
            )
            predictions[arm][list(held_indices)] = aggregation.logits.cpu()
            event_predictions[arm].index_copy_(0, held_events, event_logits)
            for field, value in arm_uncertainty.items():
                uncertainty[arm][field][list(held_indices)] = value.cpu()
            states.update(_state(f"fold{fold}.{arm}", model))
            fold_fits[arm] = fit

        for arm, pooling in (
            (DIRECT_EQUAL, "equal"),
            (DIRECT_ROBUST, "quality_winsorized"),
        ):
            model, fit = _fit_direct(
                train,
                train_tokens,
                pooling=pooling,
                seed=seed,
                device=device,
            )
            event_logits, aggregation, arm_uncertainty = _predict_direct(
                model,
                held.evidence,
                held.event_patient_index,
                len(held.patient_ids),
                held_tokens,
                pooling=pooling,
                device=device,
            )
            predictions[arm][list(held_indices)] = aggregation.logits.cpu()
            event_predictions[arm].index_copy_(0, held_events, event_logits)
            for field, value in arm_uncertainty.items():
                uncertainty[arm][field][list(held_indices)] = value.cpu()
            states.update(_state(f"fold{fold}.{arm}", model))
            fold_fits[arm] = fit

        prior = jeffreys_channel_prior_logits(train).unsqueeze(0)
        predictions[PREVALENCE][list(held_indices)] = prior.expand(
            len(held_indices), -1
        )
        fold_rows.append(
            {
                "fold": fold,
                "train_patient_count": len(train_indices),
                "held_patient_count": len(held_indices),
                "train_event_count": train.evidence.batch_size,
                "held_event_count": held.evidence.batch_size,
                "fits": fold_fits,
                "held_metrics": {
                    arm: _metrics(
                        predictions[arm][list(held_indices)],
                        held.targets,
                        held.target_mask,
                    )
                    for arm in (*LEARNED_ARMS, PREVALENCE)
                },
            }
        )
        print(
            json.dumps(
                {
                    "stage": "outer_fold_complete",
                    "fold": fold,
                    "elapsed_seconds": time.monotonic() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if any(not torch.isfinite(value).all() for value in predictions.values()) or any(
        not torch.isfinite(value).all() for value in event_predictions.values()
    ):
        raise RuntimeError("v14 OOF predictions are incomplete")
    if any(
        not torch.isfinite(value).all()
        for arm in uncertainty.values()
        for value in arm.values()
    ):
        raise RuntimeError("v14 OOF uncertainty is incomplete")

    all_predictions = {**predictions, **historical}
    metrics = {
        name: _metrics(value, full.targets, full.target_mask)
        for name, value in all_predictions.items()
    }
    comparisons = {
        "robust_temporal_minus_equal_positive_set_control": _paired_bootstrap(
            predictions[TEMPORAL_ROBUST],
            predictions[TEMPORAL_EQUAL],
            full.targets,
            full.target_mask,
        ),
        "robust_temporal_minus_historical_exact_anchor": _paired_bootstrap(
            predictions[TEMPORAL_ROBUST],
            historical[HISTORICAL_ANCHOR],
            full.targets,
            full.target_mask,
        ),
        "robust_temporal_minus_low_capacity_block9_direct_robust": _paired_bootstrap(
            predictions[TEMPORAL_ROBUST],
            predictions[DIRECT_ROBUST],
            full.targets,
            full.target_mask,
        ),
        "direct_robust_minus_direct_equal": _paired_bootstrap(
            predictions[DIRECT_ROBUST],
            predictions[DIRECT_EQUAL],
            full.targets,
            full.target_mask,
        ),
    }
    candidate = metrics[TEMPORAL_ROBUST]
    equal = metrics[TEMPORAL_EQUAL]
    anchor_metrics = metrics[HISTORICAL_ANCHOR]
    direct = metrics[DIRECT_ROBUST]
    candidate_count = int(candidate["top1"]["n_samples"])
    candidate_strict_hits = candidate["top1"]["strict_accuracy"] * candidate_count
    candidate_relaxed_hits = candidate["top1"]["relaxed_accuracy"] * candidate_count
    anchor_strict_hits = anchor_metrics["top1"]["strict_accuracy"] * candidate_count
    equal_strict_hits = equal["top1"]["strict_accuracy"] * candidate_count
    direct_strict_hits = direct["top1"]["strict_accuracy"] * candidate_count
    go_checks = {
        "strict_expected_hits_at_least_45": candidate_strict_hits + 1e-6 >= 45.0,
        "relaxed_expected_hits_at_least_55": candidate_relaxed_hits + 1e-6 >= 55.0,
        "macro_ap_at_least_0_6328": candidate["ranking"][
            "macro_average_precision"
        ]
        + 1e-8
        >= 0.6328,
        "strict_expected_gain_over_anchor_at_least_3": candidate_strict_hits
        - anchor_strict_hits
        + 1e-6
        >= 3.0,
        "far_error_nonincreasing_vs_anchor": candidate["diagnostics"][
            "far_error_expected_count"
        ]
        <= anchor_metrics["diagnostics"]["far_error_expected_count"] + 1e-6,
        "wrong_hemisphere_nonincreasing_vs_anchor": candidate["diagnostics"][
            "wrong_hemisphere_expected_count"
        ]
        <= anchor_metrics["diagnostics"]["wrong_hemisphere_expected_count"]
        + 1e-6,
        "strict_exceeds_equal_positive_set_control": candidate_strict_hits
        > equal_strict_hits + 1e-6,
        "strict_nonlower_than_low_capacity_block9_direct_robust": candidate_strict_hits + 1e-6
        >= direct_strict_hits,
    }
    tensors = {
        **{f"patient_logits.{name}": value for name, value in all_predictions.items()},
        **{
            f"event_logits.{name}": value for name, value in event_predictions.items()
        },
        **{
            f"uncertainty.{arm}.{field}": value
            for arm, fields in uncertainty.items()
            for field, value in fields.items()
        },
        "targets": full.targets.cpu(),
        "target_mask": full.target_mask.cpu(),
        "candidate_mask": V11_CANDIDATE_MASK.clone(),
        "patient_folds": torch.tensor(patient_folds, dtype=torch.long),
        "event_patient_index": full.event_patient_index.cpu(),
    }
    result = {
        "schema_version": SCHEMA,
        "method_schema": ROBUST_TEMPORAL_MIL_SCHEMA,
        "status": "completed_exploratory_source_train_patient_oof",
        "decision": "GO_TEMPORAL_MIL_AQ_ROBUST_PAIRING" if all(go_checks.values()) else (
            "NO_GO_TEMPORAL_MIL_AQ_ROBUST_PAIRING_KEEP_HISTORICAL_EXACT_ANCHOR"
        ),
        "patient_count": patients,
        "event_count": events,
        "patient_ids": list(full.patient_ids),
        "patient_folds": list(patient_folds),
        "arms": [
            *LEARNED_ARMS,
            PREVALENCE,
            HISTORICAL_ANCHOR,
            HISTORICAL_PHASE_IV,
            HISTORICAL_V_SPATIAL,
        ],
        "config": {
            "outer_folds": list(OUTER_FOLDS),
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "max_grad_norm": MAX_GRAD_NORM,
            "base_seed": BASE_SEED,
            "objective": "pure_patient_positive_set_mass_no_zero_BCE_no_pairwise",
            "partial_label_denominator_semantics": (
                "all_18_fixed_candidates_compete_in_the_listwise_denominator;"
                "zero_rows_are_not_independent_BCE_negatives_but_unlisted_candidates_"
                "still_reduce_known_positive_set_mass"
            ),
            "robust_pooling": "target_free_reliability_weighted_10_90_winsorized_complete_event_bag",
            "patient_statistical_weight": "one_equal_loss_term_per_patient",
            "candidate_mask": "fixed_standard19_except_PZ_not_patient_target_mask",
        },
        "metrics": metrics,
        "paired_patient_bootstrap": comparisons,
        "go_checks": go_checks,
        "go_all": all(go_checks.values()),
        "locked_exploratory_promotion_gate": LOCKED_PROMOTION_GATE,
        "locked_gate_counts": {
            "candidate_strict_expected_hits": candidate_strict_hits,
            "candidate_relaxed_expected_hits": candidate_relaxed_hits,
            "historical_anchor_strict_expected_hits": anchor_strict_hits,
            "equal_positive_set_control_strict_expected_hits": equal_strict_hits,
            "low_capacity_block9_direct_robust_strict_expected_hits": direct_strict_hits,
        },
        "outer_folds": fold_rows,
        "uncertainty_semantics": {
            "reported": [
                "fixed_carrier_ranking_entropy",
                "top1_margin",
                "within_patient_event_dispersion",
                "event_top1_disagreement",
                "quality_weight_effective_event_count",
            ],
            "epistemic_posterior_estimated": False,
            "target_values_used": False,
            "patient_specific_target_mask_used": False,
            "held_target_values_loaded_for_post_inference_metrics": True,
            "held_inference_API_accepts_targets_or_target_mask": False,
        },
        "scientific_boundary": {
            "development_scope": "public_source_train_only_65_patients",
            "historical_source_train_reuse": True,
            "source_dev_loaded": False,
            "source_eval_loaded": False,
            "private_loaded": False,
            "foundation": "official_pretrained_frozen_LaBraM_block9",
            "foundation_trained_from_scratch": False,
            "target": "clinician_integrated_scalp_electrode_SOZ_reference",
            "not_cortical_SOZ_or_surgical_target": True,
            "temporal_equal_arm_is_mechanism_control_not_promotable": True,
            "region_or_laterality_auxiliary_used": False,
            "historical_phase_baseline_semantics": (
                "DevelopmentIVAdditiveReasoner_uses_both_I_and_V_not_V_only"
            ),
            "historical_V_spatial_comparator_semantics": (
                "spatial_localization_from_V_with_shared_nonspatial_global_I_gate"
            ),
            "direct_control_matching_boundary": (
                "same_block9_carrier_patient_folds_epochs_optimizer_positive_set_"
                "objective_and_pooling;not_exact_capacity_or_temporal_architecture_match"
            ),
            "direct_control_is_full_final_layer_LaBraM": False,
            "direct_control_layer": "frozen_block9_prefix",
            "direct_trainable_parameter_count": 206,
            "temporal_mil_trainable_parameter_count": (
                TemporalMILEvidenceReasoner(torch.zeros(19)).n_trainable_parameters
            ),
            "relation_to_v12": (
                "v12_already_tested_10_90_robust_pooling;v14_increment_is_pairing_"
                "that_pooling_with_existing_temporal_MIL_and_AQ_coverage_reliability"
            ),
        },
        "input_lineage": {
            **dict(input_lineage),
            "historical_anchor_manifest_sha256": EXPECTED_ANCHOR_MANIFEST_SHA256,
            "historical_anchor_predictions_sha256": (
                EXPECTED_ANCHOR_PREDICTIONS_SHA256
            ),
            "historical_V_spatial_manifest_sha256": (
                EXPECTED_V_SPATIAL_MANIFEST_SHA256
            ),
            "historical_V_spatial_predictions_sha256": (
                EXPECTED_V_SPATIAL_PREDICTIONS_SHA256
            ),
        },
        "elapsed_seconds": time.monotonic() - started,
        "target_policy_sha256": TARGET_V2_POLICY_SHA256,
    }
    return result, tensors, states


def _publish(
    output_directory: Path,
    manifest: Mapping[str, object],
    tensors: Mapping[str, torch.Tensor],
    states: Mapping[str, torch.Tensor],
) -> Path:
    target = Path(os.path.abspath(output_directory))
    if target.exists():
        raise FileExistsError(target)
    if not target.parent.is_dir():
        raise FileNotFoundError(target.parent)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        save_file(dict(tensors), str(staging / "oof_predictions.safetensors"))
        save_file(dict(states), str(staging / "outer_fold_states.safetensors"))
        (staging / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        os.replace(staging, target)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return target


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-train-iv", type=Path, default=DEFAULT_SOURCE_TRAIN_IV)
    parser.add_argument("--target-directory", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--prefix-directory", type=Path, default=DEFAULT_PREFIX)
    parser.add_argument("--anchor-directory", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument(
        "--v-spatial-comparator-directory",
        type=Path,
        default=DEFAULT_V_SPATIAL_COMPARATOR,
    )
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def _validate_output_boundary(args: argparse.Namespace) -> None:
    output = Path(os.path.abspath(args.output_directory))
    if output.exists():
        raise FileExistsError(output)
    if not output.parent.is_dir():
        raise FileNotFoundError(output.parent)
    for value in (
        args.source_train_iv,
        args.target_directory,
        args.prefix_directory,
        args.anchor_directory,
        args.v_spatial_comparator_directory,
    ):
        source = Path(value).resolve(strict=True)
        if output == source or output in source.parents or source in output.parents:
            raise ValueError("output directory overlaps an input boundary")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    _validate_output_boundary(args)
    cache, full, patient_folds, event_ids, input_lineage = _load_inputs(args)
    historical = dict(_load_anchor(args.anchor_directory, full, patient_folds))
    historical[HISTORICAL_V_SPATIAL] = _load_v_spatial_comparator(
        args.v_spatial_comparator_directory, full, patient_folds
    )
    preflight = {
        "status": "ready_source_train_only",
        "patient_count": len(full.patient_ids),
        "event_count": full.evidence.batch_size,
        "event_id_count": len(event_ids),
        "fold_counts": {
            str(fold): patient_folds.count(fold) for fold in OUTER_FOLDS
        },
        "device": args.device,
        "source_dev_loaded": False,
        "source_eval_loaded": False,
        "private_loaded": False,
        "locked_exploratory_promotion_gate": LOCKED_PROMOTION_GATE,
        "input_lineage": input_lineage,
    }
    if args.preflight_only:
        print(json.dumps(preflight, sort_keys=True), flush=True)
        return 0
    manifest, tensors, states = _run(
        cache,
        full,
        patient_folds,
        historical,
        input_lineage,
        device=torch.device(args.device),
    )
    output = _publish(args.output_directory, manifest, tensors, states)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "decision": manifest["decision"],
                "output_directory": str(output),
                "candidate_metrics": manifest["metrics"][TEMPORAL_ROBUST],
                "private_used": False,
                "source_eval_used": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
