#!/usr/bin/env python3
"""Run the audited v11.1 fixed-candidate LaBraM developmental nested OOF."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
from typing import Mapping, Sequence

import safetensors
from safetensors.torch import load_file, save_file
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_labram_fine_temporal_nested_oof_v11 import (  # noqa: E402
    ARMS,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    DEFAULT_ANCHOR,
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
    INNER_FOLDS,
    L2_CANDIDATES,
    OUTER_FOLDS,
    _InnerContext,
    _canonical_bytes,
    _complement_dropout_mask,
    _file_sha,
    _fit_reasoner,
    _inner_assignments,
    _load_json_manifest,
    _require_target_free_cache,
    _select_l2,
    _state_sha,
    _transform_state,
)
from src.soz.data.deepsoz_target_v2 import (  # noqa: E402
    TARGET_V2_POLICY_SHA256,
    load_verified_deepsoz_target_v2_artifact,
)
from src.soz.fine_temporal_evidence import FINE_TEMPORAL_FEATURE_NAMES  # noqa: E402
from src.soz.metrics import (  # noqa: E402
    DEEPSOZ_STANDARD19_NEIGHBORS,
    deepsoz_style_top1_metrics,
    patient_localization_metrics,
)
from src.soz.v11_development_union import (  # noqa: E402
    EXPECTED_PUBLIC_DEVELOPMENT_UNION_MANIFEST_SHA256,
    load_public_development_union,
)
from src.soz.v11_reasoner import (  # noqa: E402
    SharedPositiveSetReasoner,
    V11_CANDIDATE_MASK,
    apply_fixed_candidate_mask,
    extract_block9_phase_contrasts,
    fit_fold_transform,
    jeffreys_reference_prior_logits,
    robust_pool_complete_patient_bags,
)


PROTOCOL_PATH = (
    ROOT
    / "research/02_method/"
    "labram_fine_temporal_development_union_protocol_v11_1_20260811_zh.md"
)
EXPECTED_PROTOCOL_SHA256 = (
    "f0b47a2aac7585579c4c5abb95dd0f35824caf98acc3e8db0482318c252bd0a4"
)
DEFAULT_OUTPUT = ROOT / "outputs/labram_fine_temporal_nested_oof_v11_1_20260811"
INVALID_PILOT = ROOT / "outputs/labram_fine_temporal_nested_oof_v11_20260811"
SCHEMA = "soz_labram_fine_temporal_nested_oof_v11_1"
PRIMARY_PATIENT_COUNT = 101
PRIMARY_EVENT_COUNT = 984
EXCLUDED_PARTIAL_REFERENCE_PATIENT = "258"
NONINFERIORITY_MARGIN = 0.05
EXPECTED_ANCHOR_MANIFEST_SHA256 = (
    "58cbfcc3d25e8ff4b13ab93e388e8aa5691e1c8fc9dc515ec2e8b51b226c9811"
)
EXPECTED_ANCHOR_PREDICTIONS_SHA256 = (
    "9373dc6bf269002c812ae26ca6ea8365b7518d3396037c4fc5b3a67603e1211d"
)


def _complete_candidate_label_rows(target_mask: torch.Tensor) -> torch.Tensor:
    """Return rows whose annotation covers the fixed 18-candidate space."""

    if not isinstance(target_mask, torch.Tensor) or target_mask.dtype != torch.bool:
        raise TypeError("target_mask must be a bool tensor")
    if target_mask.ndim != 2 or target_mask.shape[1] != 19:
        raise ValueError("target_mask must have shape [P,19]")
    fixed = V11_CANDIDATE_MASK.to(target_mask.device).expand_as(target_mask)
    return (target_mask == fixed).all(dim=1)


def _require_fixed_rows(target_mask: torch.Tensor) -> None:
    complete = _complete_candidate_label_rows(target_mask)
    if not bool(complete.all()):
        raise ValueError("primary evaluation requires the fixed complete candidate mask")


def _far_error_count(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
) -> float:
    """Tie-aware expected far-error count over the fixed candidate set."""

    _require_fixed_rows(target_mask)
    if logits.ndim != 2 or tuple(logits.shape) != tuple(targets.shape) or (
        tuple(target_mask.shape) != tuple(logits.shape)
    ):
        raise ValueError("far-error inputs must have aligned shape [P,19]")
    if not logits.is_floating_point() or not targets.is_floating_point():
        raise TypeError("far-error logits/targets must be floating point")
    if not torch.isfinite(logits).all() or not torch.isfinite(targets[target_mask]).all():
        raise ValueError("far-error observed inputs must be finite")

    expected_errors: list[torch.Tensor] = []
    for patient in range(logits.shape[0]):
        candidate_indices = torch.nonzero(target_mask[patient], as_tuple=False).flatten()
        candidate_logits = logits[patient, candidate_indices]
        top_value = candidate_logits.max()
        tied = candidate_indices[candidate_logits == top_value]
        positive = target_mask[patient] & (targets[patient] == 1)
        if not bool(positive.any()):
            raise ValueError("far-error metric requires an observed positive")
        accepted = positive.clone()
        if int(positive.sum()) <= 4:
            for index in torch.nonzero(positive, as_tuple=False).flatten().tolist():
                accepted[list(DEEPSOZ_STANDARD19_NEIGHBORS[index])] = True
        accepted &= target_mask[patient]
        expected_errors.append((~accepted[tied]).float().mean())
    return float(torch.stack(expected_errors).sum().cpu())


def _evaluate(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
) -> dict[str, object]:
    _require_fixed_rows(target_mask)
    ranking = asdict(
        patient_localization_metrics(logits, targets, target_mask, k_values=(1, 3, 5))
    )
    # These are calibration scores for membership in the current annotation
    # reference, not calibration against biological non-SOZ truth.
    ranking["reference_membership_brier"] = ranking.pop("brier")
    ranking["reference_membership_nll"] = ranking.pop("nll")
    return {
        "ranking": ranking,
        "top1": asdict(deepsoz_style_top1_metrics(logits, targets, target_mask)),
        "far_error_count": _far_error_count(logits, targets, target_mask),
    }


def _patient_contributions(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    rows = {
        name: []
        for name in (
            "strict",
            "relaxed",
            "macro_ap",
            "mrr",
            "hit_at_3",
            "hit_at_5",
            "far_error",
        )
    }
    for patient in range(logits.shape[0]):
        result = _evaluate(
            logits[patient : patient + 1],
            targets[patient : patient + 1],
            target_mask[patient : patient + 1],
        )
        rows["strict"].append(result["top1"]["strict_accuracy"])
        rows["relaxed"].append(result["top1"]["relaxed_accuracy"])
        rows["macro_ap"].append(result["ranking"]["macro_average_precision"])
        rows["mrr"].append(result["ranking"]["mean_reciprocal_rank"])
        rows["hit_at_3"].append(result["ranking"]["hit_at_k"][3])
        rows["hit_at_5"].append(result["ranking"]["hit_at_k"][5])
        rows["far_error"].append(result["far_error_count"])
    return {
        name: torch.tensor(values, dtype=torch.float64)
        for name, values in rows.items()
    }


def _bootstrap_indices(n_patients: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(BOOTSTRAP_SEED)
    return torch.randint(
        0,
        n_patients,
        (BOOTSTRAP_REPLICATES, n_patients),
        generator=generator,
    )


def _absolute_bootstrap(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
) -> dict[str, object]:
    values = _patient_contributions(logits, targets, target_mask)
    indices = _bootstrap_indices(logits.shape[0])
    result: dict[str, object] = {}
    for name, contribution in values.items():
        samples = contribution[indices].mean(dim=1)
        result[name] = {
            "estimate": float(contribution.mean()),
            "ci95": [
                float(torch.quantile(samples, 0.025)),
                float(torch.quantile(samples, 0.975)),
            ],
        }
    return result


def _paired_bootstrap(
    candidate: torch.Tensor,
    baseline: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
) -> dict[str, object]:
    candidate_rows = _patient_contributions(candidate, targets, target_mask)
    baseline_rows = _patient_contributions(baseline, targets, target_mask)
    indices = _bootstrap_indices(candidate.shape[0])
    result: dict[str, object] = {}
    for name in candidate_rows:
        difference = candidate_rows[name] - baseline_rows[name]
        samples = difference[indices].mean(dim=1)
        result[name] = {
            "delta": float(difference.mean()),
            "ci95": [
                float(torch.quantile(samples, 0.025)),
                float(torch.quantile(samples, 0.975)),
            ],
        }
    return result


def _assess_go(
    metrics: Mapping[str, Mapping[str, object]],
    fold_strict: Mapping[str, Sequence[float]],
    paired: Mapping[str, Mapping[str, Mapping[str, object]]],
    *,
    noninferiority_margin: float = NONINFERIORITY_MARGIN,
) -> tuple[bool, dict[str, bool]]:
    if not math.isfinite(noninferiority_margin) or noninferiority_margin < 0:
        raise ValueError("noninferiority_margin must be finite and non-negative")
    full_name = "full_frozen_labram_plus_fine"
    baselines = ("fine_change_only", "frozen_labram_only")
    full = metrics[full_name]
    checks: dict[str, bool] = {}
    for baseline in baselines:
        reference = metrics[baseline]
        suffix = "fine" if baseline == "fine_change_only" else "labram"
        checks[f"strict_nonlower_than_{suffix}"] = bool(
            full["top1"]["strict_accuracy"]
            >= reference["top1"]["strict_accuracy"]
        )
        checks[f"relaxed_nonlower_than_{suffix}"] = bool(
            full["top1"]["relaxed_accuracy"]
            >= reference["top1"]["relaxed_accuracy"]
        )
        checks[f"far_nonincreasing_vs_{suffix}"] = bool(
            full["far_error_count"] <= reference["far_error_count"]
        )
        checks[f"macro_ap_positive_vs_{suffix}"] = bool(
            full["ranking"]["macro_average_precision"]
            > reference["ranking"]["macro_average_precision"]
        )
        full_folds = tuple(float(value) for value in fold_strict[full_name])
        base_folds = tuple(float(value) for value in fold_strict[baseline])
        if len(full_folds) != 5 or len(base_folds) != 5:
            raise ValueError("GO assessment requires five outer folds per arm")
        checks[f"four_of_five_fold_strict_nonlower_vs_{suffix}"] = bool(
            sum(left >= right for left, right in zip(full_folds, base_folds)) >= 4
        )
        for endpoint in ("strict", "relaxed"):
            lower = float(paired[baseline][endpoint]["ci95"][0])
            checks[f"{endpoint}_bootstrap_noninferior_vs_{suffix}"] = bool(
                lower >= -noninferiority_margin
            )

    superiority = bool(
        float(paired["frozen_labram_only"]["strict"]["ci95"][0]) > 0.0
    )
    checks["superiority_supported"] = superiority
    engineering_checks = {
        key: value for key, value in checks.items() if key != "superiority_supported"
    }
    return all(engineering_checks.values()), checks


def _load_reasoner_from_fit(state: Mapping[str, torch.Tensor]) -> SharedPositiveSetReasoner:
    use_h = "h_weight" in state
    use_fine = "fine_weight" in state
    model = SharedPositiveSetReasoner(
        state["prior_logits"], use_h=use_h, use_fine=use_fine
    )
    model.load_state_dict(dict(state), strict=True)
    model.eval()
    return model


def _event_consistency(
    patient_logits: torch.Tensor,
    event_logits: torch.Tensor,
    event_patient_index: torch.Tensor,
    patient_ids: Sequence[str],
) -> dict[str, object]:
    patient_top = apply_fixed_candidate_mask(patient_logits).argmax(dim=1)
    event_top = apply_fixed_candidate_mask(event_logits).argmax(dim=1)
    rows = []
    for patient, patient_id in enumerate(patient_ids):
        selected = event_top[event_patient_index == patient]
        if selected.numel() < 1:
            raise RuntimeError("event consistency lost a patient bag")
        counts = torch.bincount(selected, minlength=19).float()[V11_CANDIDATE_MASK]
        probabilities = counts / counts.sum()
        nonzero = probabilities > 0
        entropy = float(
            (-(probabilities[nonzero] * probabilities[nonzero].log()).sum())
            / math.log(int(V11_CANDIDATE_MASK.sum()))
        )
        rows.append(
            {
                "patient_id": patient_id,
                "event_count": int(selected.numel()),
                "agreement_with_patient_top1": float(
                    (selected == patient_top[patient]).float().mean()
                ),
                "modal_share": float(counts.max() / counts.sum()),
                "normalized_vote_entropy": entropy,
            }
        )
    return {
        "patient_equal_mean_agreement_with_patient_top1": sum(
            row["agreement_with_patient_top1"] for row in rows
        )
        / len(rows),
        "patient_equal_mean_modal_share": sum(row["modal_share"] for row in rows)
        / len(rows),
        "patient_equal_mean_normalized_vote_entropy": sum(
            row["normalized_vote_entropy"] for row in rows
        )
        / len(rows),
        "patients": rows,
    }


def _event_count_strata(
    oof: Mapping[str, torch.Tensor],
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    event_counts: torch.Tensor,
) -> dict[str, object]:
    strata = {
        "1": event_counts == 1,
        "2": event_counts == 2,
        "3_to_5": (event_counts >= 3) & (event_counts <= 5),
        "ge_6": event_counts >= 6,
    }
    result: dict[str, object] = {}
    for name, selected in strata.items():
        indices = torch.nonzero(selected, as_tuple=False).flatten()
        if indices.numel() == 0:
            result[name] = {"patient_count": 0, "metrics": None}
            continue
        result[name] = {
            "patient_count": int(indices.numel()),
            "metrics": {
                arm: _evaluate(
                    logits.index_select(0, indices),
                    targets.index_select(0, indices),
                    target_mask.index_select(0, indices),
                )
                for arm, logits in oof.items()
            },
        }
    return result


def _artifact_selective_coverage(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    quality: torch.Tensor,
) -> dict[str, object]:
    if tuple(quality.shape) != (logits.shape[0],) or not torch.isfinite(quality).all():
        raise ValueError("artifact quality must be finite [P]")
    order = torch.argsort(quality, descending=True, stable=True)
    result: dict[str, object] = {}
    for coverage in (1.0, 0.9, 0.8):
        retained = max(1, math.ceil(coverage * logits.shape[0]))
        indices = order[:retained]
        result[f"coverage_{int(coverage * 100)}"] = {
            "retained_patient_count": retained,
            "abstained_patient_count": logits.shape[0] - retained,
            "minimum_retained_quality": float(quality[indices].min()),
            "metrics": _evaluate(
                logits.index_select(0, indices),
                targets.index_select(0, indices),
                target_mask.index_select(0, indices),
            ),
        }
    return result


def _source_hashes() -> dict[str, str]:
    paths = {
        "runner_v11_1": Path(__file__).resolve(),
        "runner_shared_v11": ROOT / "scripts/run_labram_fine_temporal_nested_oof_v11.py",
        "reasoner": ROOT / "src/soz/v11_reasoner.py",
        "metrics": ROOT / "src/soz/metrics.py",
        "target_loader": ROOT / "src/soz/data/deepsoz_target_v2.py",
        "fine_evidence": ROOT / "src/soz/fine_temporal_evidence.py",
    }
    return {name: _file_sha(path) for name, path in paths.items()}


def run(
    args: argparse.Namespace,
) -> tuple[
    Mapping[str, object],
    Mapping[str, torch.Tensor],
    Mapping[str, torch.Tensor],
    Mapping[str, torch.Tensor],
]:
    if _file_sha(PROTOCOL_PATH) != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("v11.1 correction protocol changed after it was frozen")
    source_hashes_before_target = _source_hashes()
    union = load_public_development_union(
        args.union_directory,
        expected_manifest_sha256=EXPECTED_PUBLIC_DEVELOPMENT_UNION_MANIFEST_SHA256,
    )
    fine_manifest = _load_json_manifest(
        args.fine_directory / "manifest.json", expected_sha=EXPECTED_FINE_MANIFEST_SHA256
    )
    prefix_manifest = _load_json_manifest(
        args.prefix_directory / "manifest.json",
        expected_sha=EXPECTED_PREFIX_MANIFEST_SHA256,
    )
    _require_target_free_cache(fine_manifest, label="fine evidence")
    _require_target_free_cache(prefix_manifest, label="LaBraM prefix")
    event_ids = tuple(event.event_id for event in union.events)
    for label, manifest in (("fine", fine_manifest), ("prefix", prefix_manifest)):
        if tuple(str(value) for value in manifest.get("event_ids", ())) != event_ids:
            raise ValueError(f"{label} event order differs from the frozen union")
    fine_file = args.fine_directory / str(fine_manifest["tensor_file"])
    prefix_file = args.prefix_directory / str(prefix_manifest["tensor_file"])
    if _file_sha(fine_file) != EXPECTED_FINE_TENSOR_FILE_SHA256 or (
        _file_sha(prefix_file) != EXPECTED_PREFIX_TENSOR_FILE_SHA256
    ):
        raise ValueError("v11.1 evidence tensor SHA mismatch")

    fine_payload = load_file(str(fine_file), device="cpu")
    fine_event_all = fine_payload["features"].detach()
    if tuple(fine_event_all.shape) != (988, 19, 20) or (
        tuple(fine_manifest["feature_names"]) != FINE_TEMPORAL_FEATURE_NAMES
    ):
        raise ValueError("v11.1 fine feature tensor/vocabulary changed")
    prefix_payload = load_file(str(prefix_file), device="cpu")
    prefix = prefix_payload["prefix_tokens"].detach()
    if tuple(prefix.shape) != (988, 15, 77, 200):
        raise ValueError("v11.1 LaBraM prefix tensor changed")
    h_event_all = extract_block9_phase_contrasts(prefix)
    del prefix, prefix_payload

    event_patient_index_all = torch.tensor(union.event_patient_index, dtype=torch.long)
    artifact_index = FINE_TEMPORAL_FEATURE_NAMES.index("artifact_burden_0_12s")
    reliability = (1.0 - fine_event_all[:, :, artifact_index]).clamp(0.0, 1.0)
    h_pool_all = robust_pool_complete_patient_bags(
        h_event_all, event_patient_index_all, len(union.patient_ids), reliability
    )
    fine_pool_all = robust_pool_complete_patient_bags(
        fine_event_all, event_patient_index_all, len(union.patient_ids), reliability
    )
    if not torch.equal(h_pool_all.event_counts, fine_pool_all.event_counts):
        raise RuntimeError("H/fine patient bags disagree")

    # First target-value read.  Candidate geometry, folds, features, source
    # hashes, exclusion rule, model arms, and hyperparameters are already fixed.
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
        raise ValueError("verified target receipt/policy changed")
    batch = target.registry.target_batch(union.patient_ids, require_eligible=True)
    targets_all = batch.values.cpu()
    target_mask_all = batch.mask.cpu()
    complete = _complete_candidate_label_rows(target_mask_all)
    excluded = [
        union.patient_ids[index]
        for index in torch.nonzero(~complete, as_tuple=False).flatten().tolist()
    ]
    if excluded != [EXCLUDED_PARTIAL_REFERENCE_PATIENT]:
        raise ValueError(f"unexpected incomplete candidate-label roster: {excluded}")
    selected_original = torch.nonzero(complete, as_tuple=False).flatten()
    if selected_original.numel() != PRIMARY_PATIENT_COUNT:
        raise ValueError("v11.1 primary complete-case patient count changed")

    targets = targets_all.index_select(0, selected_original)
    target_mask = target_mask_all.index_select(0, selected_original)
    _require_fixed_rows(target_mask)
    if not (((targets == 1) & target_mask).any(dim=1)).all():
        raise ValueError("v11.1 requires a fixed-head positive per patient")
    patient_ids = tuple(union.patient_ids[index] for index in selected_original.tolist())
    patient_folds = torch.tensor(union.patient_folds, dtype=torch.long).index_select(
        0, selected_original
    )
    h_patient = h_pool_all.features.index_select(0, selected_original).cpu()
    fine_patient = fine_pool_all.features.index_select(0, selected_original).cpu()
    event_counts = h_pool_all.event_counts.index_select(0, selected_original).cpu()
    if int(event_counts.sum()) != PRIMARY_EVENT_COUNT:
        raise ValueError("v11.1 primary complete-case event count changed")

    old_to_new = torch.full((len(union.patient_ids),), -1, dtype=torch.long)
    old_to_new[selected_original] = torch.arange(PRIMARY_PATIENT_COUNT)
    eligible_event = complete[event_patient_index_all]
    event_patient_index = old_to_new[event_patient_index_all[eligible_event]]
    h_event = h_event_all[eligible_event]
    fine_event = fine_event_all[eligible_event]
    if tuple(h_event.shape) != (PRIMARY_EVENT_COUNT, 19, 600) or (
        tuple(fine_event.shape) != (PRIMARY_EVENT_COUNT, 19, 20)
    ):
        raise RuntimeError("v11.1 eligible event carrier shape drifted")
    del h_event_all, fine_event_all, fine_payload, reliability

    oof = {
        "prevalence_only": torch.full((PRIMARY_PATIENT_COUNT, 19), torch.nan),
        **{
            name: torch.full((PRIMARY_PATIENT_COUNT, 19), torch.nan)
            for name in ARMS
        },
    }
    complement_oof = torch.full((PRIMARY_PATIENT_COUNT, 19), torch.nan)
    event_oof_full = torch.full((PRIMARY_EVENT_COUNT, 19), torch.nan)
    fold_results = []
    fold_strict = {name: [] for name in oof}
    selected_l2_by_arm = {name: [] for name in ARMS}
    outer_states: dict[str, torch.Tensor] = {}

    for outer_fold in OUTER_FOLDS:
        held = tuple(
            torch.nonzero(patient_folds == outer_fold, as_tuple=False).flatten().tolist()
        )
        train = tuple(
            torch.nonzero(patient_folds != outer_fold, as_tuple=False).flatten().tolist()
        )
        if not held or not train:
            raise RuntimeError("v11.1 outer fold lost its train/held partition")
        transform = fit_fold_transform(h_patient, fine_patient, train)
        transformed = transform.apply(h_patient, fine_patient)
        for name, value in _transform_state(transform).items():
            outer_states[f"outer{outer_fold}.{name}"] = value
        train_tensor = torch.tensor(train, dtype=torch.long)
        held_tensor = torch.tensor(held, dtype=torch.long)
        prior = jeffreys_reference_prior_logits(
            targets.index_select(0, train_tensor),
            target_mask.index_select(0, train_tensor),
        )
        oof["prevalence_only"].index_copy_(
            0, held_tensor, prior.expand(len(held), -1)
        )

        inner_assignment = _inner_assignments(
            train,
            patient_ids=patient_ids,
            event_counts=event_counts,
            outer_fold=outer_fold,
        )
        inner_contexts = []
        inner_receipts = []
        for inner_fold in INNER_FOLDS:
            inner_held = tuple(
                index for index in train if inner_assignment[index] == inner_fold
            )
            inner_train = tuple(
                index for index in train if inner_assignment[index] != inner_fold
            )
            inner_transform = fit_fold_transform(h_patient, fine_patient, inner_train)
            inner_contexts.append(
                _InnerContext(
                    fold=inner_fold,
                    train_indices=inner_train,
                    held_indices=inner_held,
                    transformed=inner_transform.apply(h_patient, fine_patient),
                )
            )
            inner_receipts.append(
                {
                    "inner_fold": inner_fold,
                    "train_patient_ids": [patient_ids[index] for index in inner_train],
                    "held_patient_ids": [patient_ids[index] for index in inner_held],
                }
            )

        arm_rows = {}
        full_fit = None
        for arm, (use_h, use_fine) in ARMS.items():
            selected_l2, selection = _select_l2(
                inner_contexts,
                targets,
                target_mask,
                use_h=use_h,
                use_fine=use_fine,
            )
            selected_l2_by_arm[arm].append(selected_l2)
            fitted = _fit_reasoner(
                transformed,
                targets,
                target_mask,
                train,
                use_h=use_h,
                use_fine=use_fine,
                l2=selected_l2,
            )
            oof[arm].index_copy_(
                0, held_tensor, fitted.logits.index_select(0, held_tensor)
            )
            held_metrics = _evaluate(
                fitted.logits.index_select(0, held_tensor),
                targets.index_select(0, held_tensor),
                target_mask.index_select(0, held_tensor),
            )
            fold_strict[arm].append(held_metrics["top1"]["strict_accuracy"])
            for name, value in fitted.state.items():
                outer_states[f"outer{outer_fold}.{arm}.{name}"] = value
            arm_rows[arm] = {
                "selected_l2": selected_l2,
                "inner_selection": selection,
                "fit": dict(fitted.diagnostics),
                "held_metrics": held_metrics,
            }
            if arm == "full_frozen_labram_plus_fine":
                full_fit = fitted

        if full_fit is None:
            raise RuntimeError("v11.1 full fit was not materialized")
        event_transformed = transform.apply(h_event, fine_event)
        held_event_indices = torch.nonzero(
            torch.isin(event_patient_index, held_tensor), as_tuple=False
        ).flatten()
        full_model = _load_reasoner_from_fit(full_fit.state)
        with torch.no_grad():
            held_event_logits = full_model(
                event_transformed.index_select(held_event_indices)
            ).logits.cpu()
        event_oof_full.index_copy_(0, held_event_indices, held_event_logits)

        prevalence_metrics = _evaluate(
            oof["prevalence_only"].index_select(0, held_tensor),
            targets.index_select(0, held_tensor),
            target_mask.index_select(0, held_tensor),
        )
        fold_strict["prevalence_only"].append(
            prevalence_metrics["top1"]["strict_accuracy"]
        )

        full_l2 = arm_rows["full_frozen_labram_plus_fine"]["selected_l2"]
        sensitivity_mask = _complement_dropout_mask(
            targets,
            target_mask,
            patient_ids,
            train,
            outer_fold=outer_fold,
        )
        sensitivity_fit = _fit_reasoner(
            transformed,
            targets,
            sensitivity_mask,
            train,
            use_h=True,
            use_fine=True,
            l2=full_l2,
            allow_candidate_subset=True,
        )
        for name, value in sensitivity_fit.state.items():
            outer_states[
                f"outer{outer_fold}.candidate_membership_mask_perturbation.{name}"
            ] = value
        complement_oof.index_copy_(
            0,
            held_tensor,
            sensitivity_fit.logits.index_select(0, held_tensor),
        )
        train_zero = (
            target_mask.index_select(0, train_tensor)
            & (targets.index_select(0, train_tensor) == 0)
        )
        train_dropped = train_zero & ~sensitivity_mask.index_select(0, train_tensor)
        fold_results.append(
            {
                "outer_fold": outer_fold,
                "train_patient_count": len(train),
                "held_patient_count": len(held),
                "train_event_count": int(event_counts[train_tensor].sum()),
                "held_event_count": int(event_counts[held_tensor].sum()),
                "train_patient_ids": [patient_ids[index] for index in train],
                "held_patient_ids": [patient_ids[index] for index in held],
                "inner_folds": inner_receipts,
                "prevalence_held_metrics": prevalence_metrics,
                "arms": arm_rows,
                "candidate_membership_mask_perturbation": {
                    "drop_fraction_of_outer_train_reference_complements": float(
                        train_dropped.sum() / train_zero.sum().clamp_min(1)
                    ),
                    "fit": dict(sensitivity_fit.diagnostics),
                },
            }
        )
        print(
            json.dumps(
                {
                    "outer_fold": outer_fold,
                    "held_patients": len(held),
                    "full_strict": arm_rows["full_frozen_labram_plus_fine"][
                        "held_metrics"
                    ]["top1"]["strict_accuracy"],
                    "full_l2": full_l2,
                    "status": "complete",
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if any(not torch.isfinite(value).all() for value in (*oof.values(), complement_oof)):
        raise RuntimeError("v11.1 OOF prediction matrix is incomplete")
    if not torch.isfinite(event_oof_full).all():
        raise RuntimeError("v11.1 event-level OOF matrix is incomplete")
    metrics = {name: _evaluate(value, targets, target_mask) for name, value in oof.items()}
    absolute_bootstrap = {
        name: _absolute_bootstrap(value, targets, target_mask)
        for name, value in oof.items()
    }
    full_name = "full_frozen_labram_plus_fine"
    paired = {
        name: _paired_bootstrap(oof[full_name], oof[name], targets, target_mask)
        for name in ("fine_change_only", "frozen_labram_only", "prevalence_only")
    }
    engineering_go, go_checks = _assess_go(
        metrics,
        fold_strict,
        paired,
        noninferiority_margin=NONINFERIORITY_MARGIN,
    )
    complement_metrics = _evaluate(complement_oof, targets, target_mask)
    complement_agreement = float(
        (
            apply_fixed_candidate_mask(oof[full_name]).argmax(dim=1)
            == apply_fixed_candidate_mask(complement_oof).argmax(dim=1)
        )
        .float()
        .mean()
    )

    l2_counts = Counter(selected_l2_by_arm[full_name])
    final_l2 = max(
        L2_CANDIDATES,
        key=lambda value: (l2_counts[value], -abs(math.log(value / 0.05))),
    )
    all_indices = tuple(range(PRIMARY_PATIENT_COUNT))
    final_transform = fit_fold_transform(h_patient, fine_patient, all_indices)
    final_fit = _fit_reasoner(
        final_transform.apply(h_patient, fine_patient),
        targets,
        target_mask,
        all_indices,
        use_h=True,
        use_fine=True,
        l2=final_l2,
    )
    final_state = {
        **_transform_state(final_transform),
        **{f"reasoner.{name}": value for name, value in final_fit.state.items()},
        "config.l2": torch.tensor(final_l2, dtype=torch.float32),
        "config.candidate_mask": V11_CANDIDATE_MASK.clone(),
    }

    anchor_manifest_path = args.anchor_directory / "manifest.json"
    anchor_prediction_path = args.anchor_directory / "oof_predictions.safetensors"
    if _file_sha(anchor_manifest_path) != EXPECTED_ANCHOR_MANIFEST_SHA256 or (
        _file_sha(anchor_prediction_path) != EXPECTED_ANCHOR_PREDICTIONS_SHA256
    ):
        raise ValueError("v11.1 pinned developmental anchor changed")
    anchor_manifest = json.loads(anchor_manifest_path.read_text(encoding="utf-8"))
    anchor_ids = tuple(str(value) for value in anchor_manifest["patient_ids"])
    if len(anchor_ids) != 65 or len(set(anchor_ids)) != 65 or (
        not set(anchor_ids).issubset(patient_ids)
    ):
        raise ValueError("v11.1 anchor patient roster is invalid")
    patient_index = {patient: index for index, patient in enumerate(patient_ids)}
    anchor_rows = torch.tensor([patient_index[patient] for patient in anchor_ids])
    anchor_payload = load_file(str(anchor_prediction_path), device="cpu")
    anchor_logits = anchor_payload["temporal_mil_exact"]
    if tuple(anchor_logits.shape) != (65, 19) or not torch.isfinite(anchor_logits).all():
        raise ValueError("v11.1 anchor OOF tensor is invalid")
    anchor_targets = targets.index_select(0, anchor_rows)
    anchor_mask = target_mask.index_select(0, anchor_rows)
    v11_1_anchor_rows = oof[full_name].index_select(0, anchor_rows)
    anchor_comparison = {
        "scope": "original_65_only_training_cohorts_differ_developmental_only",
        "patient_count": 65,
        "anchor_metrics": _evaluate(anchor_logits, anchor_targets, anchor_mask),
        "v11_1_metrics": _evaluate(v11_1_anchor_rows, anchor_targets, anchor_mask),
        "paired_v11_1_minus_anchor": _paired_bootstrap(
            v11_1_anchor_rows, anchor_logits, anchor_targets, anchor_mask
        ),
    }

    quality = 1.0 - fine_patient[:, :, artifact_index].mean(dim=1)
    source_hashes_after = _source_hashes()
    if source_hashes_after != source_hashes_before_target:
        raise RuntimeError("v11.1 source files changed during execution")
    full_metrics = metrics[full_name]
    oof_tensors = {
        **{f"oof.{name}": value for name, value in oof.items()},
        "oof.candidate_membership_mask_perturbation_full": complement_oof,
        "oof.event_full": event_oof_full,
        "targets": targets,
        "target_mask": target_mask,
        "config.candidate_mask": V11_CANDIDATE_MASK.clone(),
        "patient_folds": patient_folds,
        "patient_event_counts": event_counts,
        "patient_artifact_quality": quality,
    }
    manifest = {
        "schema_version": SCHEMA,
        "reporting_revision": "r2_add_secondary_fold_state_and_all_primary_bootstrap_endpoints",
        "status": "completed_internal_developmental_nested_oof",
        "decision": (
            "ENGINEERING_GO_separate_fold_local_peft_capacity_diagnostic"
            if engineering_go
            else "NO_GO_fine_temporal_recovery_not_supported"
        ),
        "invalid_predecessor": {
            "path": str(INVALID_PILOT.relative_to(ROOT)),
            "status": "invalid_pilot_target_mask_oracle_do_not_use_for_claims",
        },
        "claim_boundary": {
            "public_confirmation": False,
            "external_validation": False,
            "pretraining_exposed_downstream_label_oof": True,
            "held_signal_or_zero_shot_claim_allowed": False,
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
            "foundation_trainable_parameters_v11_1": 0,
            "foundation_pretraining_exposure_possible": True,
        },
        "signal_carrier_channel_count": 19,
        "fixed_output_candidate_count": 18,
        "fixed_candidate_mask": V11_CANDIDATE_MASK.tolist(),
        "target_free_union_patient_count": 102,
        "target_free_union_event_count": 988,
        "primary_patient_count": PRIMARY_PATIENT_COUNT,
        "primary_event_count": PRIMARY_EVENT_COUNT,
        "excluded_partial_reference_patients": [EXCLUDED_PARTIAL_REFERENCE_PATIENT],
        "patient_ids": list(patient_ids),
        "event_counts": event_counts.tolist(),
        "outer_folds": list(OUTER_FOLDS),
        "inner_fold_count": len(INNER_FOLDS),
        "l2_candidates": list(L2_CANDIDATES),
        "selected_l2_by_arm": selected_l2_by_arm,
        "fold_results": fold_results,
        "metrics": metrics,
        "absolute_patient_bootstrap": absolute_bootstrap,
        "paired_full_minus_baselines": paired,
        "engineering_noninferiority_margin": NONINFERIORITY_MARGIN,
        "go_checks": go_checks,
        "engineering_go_all": engineering_go,
        "scientific_increment_supported": go_checks["superiority_supported"],
        "candidate_membership_mask_perturbation_sensitivity": {
            "metrics": complement_metrics,
            "top1_agreement_with_primary": complement_agreement,
            "not_a_pu_or_missing_label_solution": True,
        },
        "event_count_strata": _event_count_strata(
            oof, targets, target_mask, event_counts
        ),
        "event_to_patient_consistency": _event_consistency(
            oof[full_name], event_oof_full, event_patient_index, patient_ids
        ),
        "artifact_quality_selective_coverage": _artifact_selective_coverage(
            oof[full_name], targets, target_mask, quality
        ),
        "goal_thresholds_descriptive_only": {
            "strict_top1_ge_0_80": full_metrics["top1"]["strict_accuracy"] >= 0.80,
            "relaxed_top1_ge_0_85": full_metrics["top1"]["relaxed_accuracy"] >= 0.85,
            "not_used_for_model_selection": True,
        },
        "anchor_comparison": anchor_comparison,
        "development_refit_non_deployable": {
            "selected_l2_by_outer_mode": final_l2,
            "outer_selected_l2_counts": {
                str(key): value for key, value in l2_counts.items()
            },
            "fit": dict(final_fit.diagnostics),
            "state_sha256": _state_sha(final_state),
            "foundation_weights_serialized": False,
            "clinical_deployment_allowed": False,
        },
        "lineage": {
            "union_manifest_sha256": union.manifest_sha256,
            "fine_manifest_sha256": EXPECTED_FINE_MANIFEST_SHA256,
            "fine_tensor_file_sha256": EXPECTED_FINE_TENSOR_FILE_SHA256,
            "prefix_manifest_sha256": EXPECTED_PREFIX_MANIFEST_SHA256,
            "prefix_tensor_file_sha256": EXPECTED_PREFIX_TENSOR_FILE_SHA256,
            "target_artifact_sha256": target.receipt.target_artifact_sha256,
            "target_receipt_sha256": target.receipt.receipt_sha256,
            "target_policy_sha256": target.receipt.policy_sha256,
            "anchor_manifest_sha256": EXPECTED_ANCHOR_MANIFEST_SHA256,
            "anchor_predictions_sha256": EXPECTED_ANCHOR_PREDICTIONS_SHA256,
        },
        "access_receipt": {
            "target_values_loaded_only_after_protocol_folds_features_and_code_hashes_frozen": True,
            "patient_specific_target_mask_used_for_prediction": False,
            "private_eeg_loaded": False,
            "private_target_values_loaded": False,
            "private_forward_count": 0,
            "llm_used_as_soz_predictor": False,
        },
    }
    return manifest, oof_tensors, final_state, outer_states


def _publish(
    output_directory: Path,
    manifest: Mapping[str, object],
    oof_tensors: Mapping[str, torch.Tensor],
    final_state: Mapping[str, torch.Tensor],
    outer_states: Mapping[str, torch.Tensor] | None = None,
) -> Path:
    candidate_mask = final_state.get("config.candidate_mask")
    if not isinstance(candidate_mask, torch.Tensor) or not torch.equal(
        candidate_mask.cpu(), V11_CANDIDATE_MASK
    ):
        raise ValueError("final checkpoint must contain the fixed config.candidate_mask")
    target = Path(os.path.abspath(output_directory))
    if target.exists():
        raise FileExistsError(target)
    if not target.parent.is_dir():
        raise FileNotFoundError(target.parent)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        oof_path = staging / "oof_predictions.safetensors"
        final_path = staging / "final_checkpoint.safetensors"
        save_file(dict(oof_tensors), str(oof_path))
        save_file(dict(final_state), str(final_path))
        files = {
            "oof_predictions.safetensors": {
                "sha256": _file_sha(oof_path),
                "size_bytes": oof_path.stat().st_size,
            },
            "final_checkpoint.safetensors": {
                "sha256": _file_sha(final_path),
                "size_bytes": final_path.stat().st_size,
            },
        }
        if outer_states is not None:
            outer_path = staging / "outer_fold_states.safetensors"
            save_file(dict(outer_states), str(outer_path))
            files["outer_fold_states.safetensors"] = {
                "sha256": _file_sha(outer_path),
                "size_bytes": outer_path.stat().st_size,
            }
        completed = dict(manifest)
        completed["files"] = files
        (staging / "manifest.json").write_bytes(_canonical_bytes(completed, newline=True))
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
    parser.add_argument("--anchor-directory", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    args = parse_args(argv)
    manifest, oof, final_state, outer_states = run(args)
    path = _publish(args.output_directory, manifest, oof, final_state, outer_states)
    completed = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    full = completed["metrics"]["full_frozen_labram_plus_fine"]
    print(
        json.dumps(
            {
                "status": completed["status"],
                "decision": completed["decision"],
                "path": str(path),
                "manifest_sha256": _file_sha(path / "manifest.json"),
                "strict_top1": full["top1"]["strict_accuracy"],
                "relaxed_top1": full["top1"]["relaxed_accuracy"],
                "macro_ap": full["ranking"]["macro_average_precision"],
                "scientific_increment_supported": completed[
                    "scientific_increment_supported"
                ],
                "private_used": False,
                "public_confirmation": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
