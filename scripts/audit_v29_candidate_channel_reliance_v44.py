#!/usr/bin/env python3
"""Candidate-specific frozen channel-content reliance audit for v29.

For every public patient and private target-blind event, the script performs
two prespecified representation interventions without fitting or selecting:

1. comprehensiveness: replace the originally predicted Top-1 channel's H and
   D carrier content by the within-unit channel mean;
2. sufficiency: retain only the originally predicted Top-1 channel's H and D
   content and replace all other channels by the within-unit channel mean.

It also removes each of the 18 candidate channels exhaustively, so the Top-1
removal effect is compared with all alternative single-channel removals rather
than a hand-picked control.  Interventions are made on cached LaBraM carriers,
not raw EEG.  The result can support candidate-channel representation reliance
but cannot establish waveform-time causality or a clinical explanation.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Mapping, Sequence

import numpy as np
import torch
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_labram_rank1_direct_token_oof_v28 as v28  # noqa: E402
from scripts.audit_labram_v29_h_carrier_stress_v43 import (  # noqa: E402
    _private_h_fold_probability,
    _public_h_probability,
)
from scripts.audit_labram_v29_token_stress_v38 import (  # noqa: E402
    _probability,
    _probability_logits,
    _stability,
    _state_for_fold,
)
from scripts.audit_private_frozen_publication_v36 import (  # noqa: E402
    BOOTSTRAP_SEED,
    _event_rows,
    _paired as _private_paired,
    _read_csv,
    _summary as _private_summary,
)
from scripts.predict_private_labram_portable_equal_v29 import (  # noqa: E402
    _direct_probability as _private_direct_probability,
)
from scripts.run_labram_fine_temporal_nested_oof_v11_1 import (  # noqa: E402
    _evaluate,
    _paired_bootstrap,
)
from src.soz.geometry import STANDARD_19  # noqa: E402
from src.soz.v11_reasoner import V11_CANDIDATE_MASK  # noqa: E402


SCHEMA = "trustworthy_soz_v29_candidate_channel_reliance_v44"
DEFAULT_V16 = (
    ROOT / "outputs/labram_identity_recovery_closed_replay_v16_replay_20260815"
)
DEFAULT_V28 = ROOT / "outputs/labram_rank1_direct_token_oof_v28_20260815"
DEFAULT_V29 = ROOT / "outputs/labram_portable_equal_ensemble_public_oof_v29_20260815"
DEFAULT_PRIVATE_PREDICTION = (
    ROOT / "outputs/labram_portable_equal_private_target_blind_v29_20260815"
)
DEFAULT_PRIVATE_H = ROOT / "outputs/labram_private_target_blind_evidence_v18_20260814"
DEFAULT_PRIVATE_PHASE = ROOT / "outputs/private_target_blind_rank1_phase_v29_20260815"
DEFAULT_PRIVATE_TARGET = (
    ROOT / "outputs/labram_private_zero_adaptation_bundle_v18_20260814/target_ledger.csv"
)
DEFAULT_OUTPUT = ROOT / "outputs/trustworthy_soz_v29_candidate_channel_reliance_v44_20260816"
BOOTSTRAP_REPLICATES = 10_000
CANDIDATE_INDICES = torch.nonzero(
    V11_CANDIDATE_MASK, as_tuple=False
).flatten().long()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _replace_one_with_channel_mean(
    features: torch.Tensor, channel: int
) -> torch.Tensor:
    if channel not in CANDIDATE_INDICES.tolist():
        raise ValueError("intervened channel must be in the frozen C18 candidate set")
    result = features.clone()
    result[:, channel] = features.mean(dim=1)
    return result.contiguous()


def _retain_selected_only(
    features: torch.Tensor, selected_channels: torch.Tensor
) -> torch.Tensor:
    if selected_channels.dtype != torch.long or tuple(selected_channels.shape) != (
        len(features),
    ):
        raise TypeError("selected_channels must be long [N]")
    if not bool(V11_CANDIDATE_MASK.index_select(0, selected_channels).all()):
        raise ValueError("every retained channel must be in C18")
    result = features.mean(dim=1, keepdim=True).expand_as(features).clone()
    rows = torch.arange(len(features), dtype=torch.long)
    result[rows, selected_channels] = features[rows, selected_channels]
    return result.contiguous()


def _public_d_probability(
    features: torch.Tensor,
    event_patient_index: torch.Tensor,
    stable: object,
    states: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    bag = v28.PatientBag(
        phase_features=features,
        event_patient_index=event_patient_index,
        targets=stable.targets,
        target_mask=stable.target_mask,
        patient_ids=stable.patient_ids,
    )
    logits = torch.full((len(stable.patient_ids), 19), torch.nan)
    with torch.inference_mode():
        for fold in range(5):
            held_indices = tuple(
                torch.nonzero(stable.patient_folds == fold, as_tuple=False)
                .flatten()
                .tolist()
            )
            held = v28._subset_bag(bag, held_indices)
            state = _state_for_fold(states, fold)
            model = v28.RankOneDirectTokenHead(state["prior_logits"])
            model.load_state_dict(state, strict=True)
            model.eval().requires_grad_(False)
            event_logits = model(held.phase_features)
            patient_logits = v28._aggregate_equal(
                event_logits, held.event_patient_index, len(held.patient_ids)
            )
            logits[list(held_indices)] = patient_logits
    if not torch.isfinite(logits).all():
        raise RuntimeError("public D intervention replay is incomplete")
    return _probability(logits, stable.target_mask)


def _selected_intervention(
    removal_probabilities: torch.Tensor,
    original_probability: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if removal_probabilities.ndim != 3 or tuple(
        removal_probabilities.shape[1:]
    ) != (len(CANDIDATE_INDICES), 19):
        raise ValueError("removal probabilities must be [N,18,19]")
    top1 = original_probability.masked_fill(
        ~V11_CANDIDATE_MASK, -torch.inf
    ).argmax(dim=1)
    candidate_position = torch.full((19,), -1, dtype=torch.long)
    candidate_position[CANDIDATE_INDICES] = torch.arange(len(CANDIDATE_INDICES))
    selected_position = candidate_position.index_select(0, top1)
    if bool((selected_position < 0).any()):
        raise RuntimeError("original Top-1 fell outside C18")
    rows = torch.arange(len(top1), dtype=torch.long)
    selected_probability = removal_probabilities[rows, selected_position]
    original_top1_probability = original_probability[rows, top1]
    removed_original_top1_probability = removal_probabilities.gather(
        2, top1.view(-1, 1, 1).expand(-1, len(CANDIDATE_INDICES), 1)
    ).squeeze(2)
    effects = original_top1_probability.unsqueeze(1) - removed_original_top1_probability
    selected_effect = effects[rows, selected_position]
    return selected_probability, top1, effects, selected_effect


def _cluster_bootstrap_mean(
    values: torch.Tensor, cluster_ids: Sequence[str], *, seed: int
) -> dict[str, object]:
    if values.ndim != 1 or len(values) != len(cluster_ids):
        raise ValueError("bootstrap values/clusters differ")
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, cluster in zip(values.tolist(), cluster_ids):
        grouped[str(cluster)].append(float(value))
    cluster_values = np.asarray(
        [np.mean(grouped[key]) for key in sorted(grouped)], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    sampled = rng.integers(
        0,
        len(cluster_values),
        size=(BOOTSTRAP_REPLICATES, len(cluster_values)),
    )
    bootstrap = cluster_values[sampled].mean(axis=1)
    return {
        "unit_count": len(values),
        "cluster_count": len(cluster_values),
        "cluster_equal_mean": float(cluster_values.mean()),
        "cluster_bootstrap_ci95": [
            float(value) for value in np.quantile(bootstrap, (0.025, 0.975))
        ],
    }


def _effect_summary(
    *,
    effects: torch.Tensor,
    selected_effect: torch.Tensor,
    top1: torch.Tensor,
    cluster_ids: Sequence[str],
    seed: int,
) -> dict[str, object]:
    candidate_position = torch.full((19,), -1, dtype=torch.long)
    candidate_position[CANDIDATE_INDICES] = torch.arange(len(CANDIDATE_INDICES))
    selected_position = candidate_position.index_select(0, top1)
    rows = torch.arange(len(top1), dtype=torch.long)
    other_sum = effects.sum(dim=1) - effects[rows, selected_position]
    other_mean = other_sum / (len(CANDIDATE_INDICES) - 1)
    contrast = selected_effect - other_mean
    rank = 1 + (effects > selected_effect.unsqueeze(1)).sum(dim=1)
    return {
        "original_top1_probability_drop": {
            "mean": float(selected_effect.mean()),
            "median": float(selected_effect.median()),
            "fraction_positive": float((selected_effect > 0).float().mean()),
        },
        "top1_removal_effect_vs_other_single_channel_removals": {
            "mean_effect_difference": float(contrast.mean()),
            "fraction_top1_removal_is_largest": float((rank == 1).float().mean()),
            "median_effect_rank_of_18": float(rank.float().median()),
            "cluster_interval": _cluster_bootstrap_mean(
                contrast, cluster_ids, seed=seed
            ),
        },
    }


def _public_metrics(
    probability: torch.Tensor, stable: object
) -> Mapping[str, object]:
    return _evaluate(
        _probability_logits(probability, stable.target_mask),
        stable.targets,
        stable.target_mask,
    )


def _public_summary_row(
    arm: str,
    metrics: Mapping[str, object],
    stability: Mapping[str, float],
) -> dict[str, object]:
    return {
        "dataset": "public_consumed_development",
        "arm": arm,
        "units": 102,
        "clusters": 102,
        "strict": metrics["top1"]["strict_accuracy"],
        "neighborhood4": metrics["top1"]["relaxed_accuracy"],
        "hit_at_5": metrics["ranking"]["hit_at_k"][5],
        "far_count": metrics["far_error_count"],
        "contralateral_far_count": "",
        **stability,
    }


def _private_summary_row(
    arm: str,
    summary: Mapping[str, object],
    stability: Mapping[str, float],
) -> dict[str, object]:
    return {
        "dataset": "private_post_open_transport",
        "arm": arm,
        "units": summary["event_count"],
        "clusters": summary["patient_count"],
        "strict": summary["event_micro"]["strict"],
        "neighborhood4": summary["event_micro"]["relaxed"],
        "hit_at_5": summary["event_micro"]["hit_at_5"],
        "far_count": summary["endpoint_counts"]["far"],
        "contralateral_far_count": summary["endpoint_counts"][
            "contralateral_far"
        ],
        **stability,
    }


def run(
    *,
    v16_directory: Path,
    v28_directory: Path,
    v29_directory: Path,
    private_prediction_directory: Path,
    private_h_directory: Path,
    private_phase_directory: Path,
    private_target_path: Path,
) -> tuple[dict[str, object], dict[str, torch.Tensor], list[dict[str, object]]]:
    loader_args = v28.build_parser().parse_args(["--device", "cpu"])
    stable = v28.v17._load_stable_development(loader_args)
    public_prefix, event_patient_index = v28._load_stable_prefix(loader_args, stable)
    public_d_features = v28.extract_rank1_phase_features(public_prefix)
    del public_prefix

    h_state_path = (
        v16_directory / "outer_fold_states.safetensors"
    ).resolve(strict=True)
    h_states = load_file(str(h_state_path), device="cpu")
    v28_state_path = (v28_directory / "model_and_oof.safetensors").resolve(
        strict=True
    )
    d_states = load_file(str(v28_state_path), device="cpu")
    v29_path = (v29_directory / "oof_predictions.safetensors").resolve(strict=True)
    v29 = load_file(str(v29_path), device="cpu")
    public_original_h = _public_h_probability(
        stable.h_patient, h_states, stable.patient_folds
    )
    public_original_d = _public_d_probability(
        public_d_features, event_patient_index, stable, d_states
    )
    public_original = v29["oof.portable_equal_ensemble_probability"].float()
    public_identity = 0.5 * public_original_h + 0.5 * public_original_d
    public_identity_difference = float((public_identity - public_original).abs().max())
    if public_identity_difference > 1e-5:
        raise ValueError(f"public H/D identity replay drifted: {public_identity_difference}")

    public_removed_rows = []
    for channel in CANDIDATE_INDICES.tolist():
        h_probability = _public_h_probability(
            _replace_one_with_channel_mean(stable.h_patient, channel),
            h_states,
            stable.patient_folds,
        )
        d_probability = _public_d_probability(
            _replace_one_with_channel_mean(public_d_features, channel),
            event_patient_index,
            stable,
            d_states,
        )
        public_removed_rows.append(0.5 * h_probability + 0.5 * d_probability)
    public_removed = torch.stack(public_removed_rows, dim=1).contiguous()
    (
        public_top1_removed,
        public_top1,
        public_effects,
        public_selected_effect,
    ) = _selected_intervention(public_removed, public_original)
    public_event_top1 = public_top1.index_select(0, event_patient_index)
    public_sufficient_h = _public_h_probability(
        _retain_selected_only(stable.h_patient, public_top1),
        h_states,
        stable.patient_folds,
    )
    public_sufficient_d = _public_d_probability(
        _retain_selected_only(public_d_features, public_event_top1),
        event_patient_index,
        stable,
        d_states,
    )
    public_sufficient = (0.5 * public_sufficient_h + 0.5 * public_sufficient_d).contiguous()

    private_manifest_path = (
        private_prediction_directory / "manifest.json"
    ).resolve(strict=True)
    private_manifest = json.loads(private_manifest_path.read_text(encoding="utf-8"))
    private_events = private_manifest.get("events")
    if not isinstance(private_events, list) or len(private_events) != 88:
        raise ValueError("private v29 event roster changed")
    private_prediction_path = (
        private_prediction_directory / "predictions.safetensors"
    ).resolve(strict=True)
    private_prediction = load_file(str(private_prediction_path), device="cpu")
    private_original = private_prediction[
        "private_portable_equal_probability"
    ].float()

    private_h_manifest_path = (private_h_directory / "manifest.json").resolve(
        strict=True
    )
    private_h_manifest = json.loads(private_h_manifest_path.read_text(encoding="utf-8"))
    private_h_path = (
        private_h_directory / str(private_h_manifest["tensor_file"])
    ).resolve(strict=True)
    private_h = load_file(str(private_h_path), device="cpu")["h_event"].float()
    private_phase_manifest_path = (
        private_phase_directory / "manifest.json"
    ).resolve(strict=True)
    private_phase_manifest = json.loads(
        private_phase_manifest_path.read_text(encoding="utf-8")
    )
    private_phase_path = (
        private_phase_directory / str(private_phase_manifest["tensor_file"])
    ).resolve(strict=True)
    private_phase = load_file(str(private_phase_path), device="cpu")[
        "phase_features"
    ].float()
    if tuple(private_h.shape) != (88, 19, 600) or tuple(
        private_phase.shape
    ) != (88, 19, 5, 200):
        raise ValueError("private H/D carrier shape changed")
    event_ids = [str(row["event_id"]) for row in private_events]
    if event_ids != [str(row["event_id"]) for row in private_h_manifest["events"]] or (
        event_ids
        != [str(row["event_id"]) for row in private_phase_manifest["events"]]
    ):
        raise ValueError("private H/D/prediction event identity differs")

    private_identity_h_fold = _private_h_fold_probability(private_h, h_states)
    private_identity_d_fold = torch.stack(
        [
            _private_direct_probability(private_phase, d_states, fold)
            for fold in range(5)
        ],
        dim=1,
    )
    private_identity = (0.5 * private_identity_h_fold + 0.5 * private_identity_d_fold).mean(
        dim=1
    )
    private_identity_difference = float((private_identity - private_original).abs().max())
    if private_identity_difference > 1e-6:
        raise ValueError(f"private H/D identity replay drifted: {private_identity_difference}")

    private_removed_rows = []
    for channel in CANDIDATE_INDICES.tolist():
        h_fold = _private_h_fold_probability(
            _replace_one_with_channel_mean(private_h, channel), h_states
        )
        d_fold = torch.stack(
            [
                _private_direct_probability(
                    _replace_one_with_channel_mean(private_phase, channel),
                    d_states,
                    fold,
                )
                for fold in range(5)
            ],
            dim=1,
        )
        private_removed_rows.append((0.5 * h_fold + 0.5 * d_fold).mean(dim=1))
    private_removed = torch.stack(private_removed_rows, dim=1).contiguous()
    (
        private_top1_removed,
        private_top1,
        private_effects,
        private_selected_effect,
    ) = _selected_intervention(private_removed, private_original)
    private_sufficient_h = _private_h_fold_probability(
        _retain_selected_only(private_h, private_top1), h_states
    )
    private_sufficient_d = torch.stack(
        [
            _private_direct_probability(
                _retain_selected_only(private_phase, private_top1), d_states, fold
            )
            for fold in range(5)
        ],
        dim=1,
    )
    private_sufficient = (
        0.5 * private_sufficient_h + 0.5 * private_sufficient_d
    ).mean(dim=1).contiguous()

    public_mask = stable.target_mask
    private_mask = V11_CANDIDATE_MASK.unsqueeze(0).expand(len(private_h), -1)
    public_original_metrics = _public_metrics(public_original, stable)
    public_removed_metrics = _public_metrics(public_top1_removed, stable)
    public_sufficient_metrics = _public_metrics(public_sufficient, stable)
    public_removed_stability = _stability(
        public_original, public_top1_removed, public_mask
    )
    public_sufficient_stability = _stability(
        public_original, public_sufficient, public_mask
    )

    private_target_rows = _read_csv(private_target_path)
    private_original_rows, private_flow = _event_rows(
        scores=private_original,
        events=private_events,
        target_rows=private_target_rows,
    )
    private_removed_event_rows, removed_flow = _event_rows(
        scores=private_top1_removed,
        events=private_events,
        target_rows=private_target_rows,
    )
    private_sufficient_event_rows, sufficient_flow = _event_rows(
        scores=private_sufficient,
        events=private_events,
        target_rows=private_target_rows,
    )
    if private_flow != removed_flow or private_flow != sufficient_flow:
        raise RuntimeError("private intervention changed the evaluation cohort")
    private_original_summary = _private_summary(
        private_original_rows, seed=BOOTSTRAP_SEED
    )
    private_removed_summary = _private_summary(
        private_removed_event_rows, seed=BOOTSTRAP_SEED + 1000
    )
    private_sufficient_summary = _private_summary(
        private_sufficient_event_rows, seed=BOOTSTRAP_SEED + 2000
    )
    private_removed_stability = _stability(
        private_original, private_top1_removed, private_mask
    )
    private_sufficient_stability = _stability(
        private_original, private_sufficient, private_mask
    )

    public_effect_summary = _effect_summary(
        effects=public_effects,
        selected_effect=public_selected_effect,
        top1=public_top1,
        cluster_ids=stable.patient_ids,
        seed=20260816,
    )
    private_cluster_ids = [str(row["patient_id"]) for row in private_events]
    private_effect_summary = _effect_summary(
        effects=private_effects,
        selected_effect=private_selected_effect,
        top1=private_top1,
        cluster_ids=private_cluster_ids,
        seed=20260817,
    )

    result: dict[str, object] = {
        "schema_version": SCHEMA,
        "status": "completed_candidate_specific_cached_carrier_reliance_audit",
        "analysis_role": {
            "public": "posthoc_consumed_public_development_audit",
            "private": "post_open_descriptive_transport_audit",
        },
        "intervention": {
            "replacement": "within_unit_all_19_channel_mean_at_cached_carrier_level",
            "comprehensiveness": "replace_original_top1_H_and_D_content",
            "sufficiency": "retain_only_original_top1_H_and_D_content",
            "exhaustive_control": "remove_each_frozen_C18_candidate_once",
            "candidate_selection_uses_target": False,
        },
        "identity_replay_max_absolute_probability_difference": {
            "public": public_identity_difference,
            "private": private_identity_difference,
        },
        "public": {
            "patient_count": len(stable.patient_ids),
            "event_count_before_patient_pooling": int(stable.event_counts.sum()),
            "original_metrics": public_original_metrics,
            "top1_content_removed_metrics": public_removed_metrics,
            "top1_content_sufficient_metrics": public_sufficient_metrics,
            "top1_content_removed_stability": public_removed_stability,
            "top1_content_sufficient_stability": public_sufficient_stability,
            "candidate_channel_effect": public_effect_summary,
            "paired_top1_removed_minus_original": _paired_bootstrap(
                _probability_logits(public_top1_removed, public_mask),
                _probability_logits(public_original, public_mask),
                stable.targets,
                public_mask,
            ),
            "paired_top1_sufficient_minus_original": _paired_bootstrap(
                _probability_logits(public_sufficient, public_mask),
                _probability_logits(public_original, public_mask),
                stable.targets,
                public_mask,
            ),
        },
        "private": {
            "target_blind_event_count": len(private_events),
            "evaluable_event_count": private_flow["primary_intersection_events"],
            "evaluable_patient_clusters": private_flow["primary_intersection_patients"],
            "cohort_flow": private_flow,
            "original_summary": private_original_summary,
            "top1_content_removed_summary": private_removed_summary,
            "top1_content_sufficient_summary": private_sufficient_summary,
            "top1_content_removed_stability_all_88": private_removed_stability,
            "top1_content_sufficient_stability_all_88": private_sufficient_stability,
            "candidate_channel_effect_all_88": private_effect_summary,
            "paired_top1_removed_minus_original": _private_paired(
                private_removed_event_rows,
                private_original_rows,
                seed=BOOTSTRAP_SEED + 30_000,
            ),
            "paired_top1_sufficient_minus_original": _private_paired(
                private_sufficient_event_rows,
                private_original_rows,
                seed=BOOTSTRAP_SEED + 40_000,
            ),
        },
        "source_files": {
            "h_fold_states": str(h_state_path.relative_to(ROOT)),
            "h_fold_states_sha256": _sha256(h_state_path),
            "d_model_and_oof": str(v28_state_path.relative_to(ROOT)),
            "d_model_and_oof_sha256": _sha256(v28_state_path),
            "public_v29": str(v29_path.relative_to(ROOT)),
            "public_v29_sha256": _sha256(v29_path),
            "private_prediction_manifest": str(
                private_manifest_path.relative_to(ROOT)
            ),
            "private_prediction_manifest_sha256": _sha256(private_manifest_path),
            "private_prediction_tensor": str(
                private_prediction_path.relative_to(ROOT)
            ),
            "private_prediction_tensor_sha256": _sha256(private_prediction_path),
            "private_H_manifest": str(private_h_manifest_path.relative_to(ROOT)),
            "private_H_manifest_sha256": _sha256(private_h_manifest_path),
            "private_H_tensor": str(private_h_path.relative_to(ROOT)),
            "private_H_tensor_sha256": _sha256(private_h_path),
            "private_D_manifest": str(private_phase_manifest_path.relative_to(ROOT)),
            "private_D_manifest_sha256": _sha256(private_phase_manifest_path),
            "private_D_tensor": str(private_phase_path.relative_to(ROOT)),
            "private_D_tensor_sha256": _sha256(private_phase_path),
            "private_target_ledger": str(private_target_path.resolve().relative_to(ROOT)),
            "private_target_ledger_sha256": _sha256(private_target_path.resolve()),
        },
        "access_receipt": {
            "raw_EEG_loaded": False,
            "cached_foundation_H_and_D_carriers_loaded": True,
            "foundation_forward_performed": False,
            "model_training_performed": False,
            "model_or_intervention_selected_from_outcomes": False,
            "fusion_weight_or_threshold_changed": False,
            "report_text_changed": False,
            "existing_public_targets_loaded_for_frozen_evaluation": True,
            "previously_opened_private_targets_loaded_for_descriptive_evaluation": True,
        },
        "interpretation_boundary": {
            "model_aligned_level": "cached_candidate_channel_representation_reliance",
            "raw_EEG_intervention": False,
            "specific_waveform_interval_faithfulness_validated": False,
            "clinical_morphology_onset_or_propagation_concept_validated": False,
            "attention_used_as_explanation": False,
            "public_confirmatory_inference": False,
            "private_fresh_external_validation": False,
            "allowed_claim": (
                "the frozen v29 candidate ranking shows the reported reliance on "
                "the originally selected channel's cached H/D representation"
            ),
        },
    }

    rows = [
        _public_summary_row(
            "original_v29",
            public_original_metrics,
            {"top1_retention": 1.0, "top3_jaccard": 1.0, "mean_absolute_probability_shift": 0.0},
        ),
        _public_summary_row(
            "top1_content_removed",
            public_removed_metrics,
            public_removed_stability,
        ),
        _public_summary_row(
            "top1_content_sufficient",
            public_sufficient_metrics,
            public_sufficient_stability,
        ),
        _private_summary_row(
            "original_v29",
            private_original_summary,
            {"top1_retention": 1.0, "top3_jaccard": 1.0, "mean_absolute_probability_shift": 0.0},
        ),
        _private_summary_row(
            "top1_content_removed",
            private_removed_summary,
            private_removed_stability,
        ),
        _private_summary_row(
            "top1_content_sufficient",
            private_sufficient_summary,
            private_sufficient_stability,
        ),
    ]
    tensors = {
        "public.original": public_original.contiguous(),
        "public.each_candidate_removed": public_removed.contiguous(),
        "public.top1_removed": public_top1_removed.contiguous(),
        "public.top1_sufficient": public_sufficient.contiguous(),
        "public.original_top1": public_top1.contiguous(),
        "public.original_top1_removal_effects": public_effects.contiguous(),
        "private.original": private_original.contiguous(),
        "private.each_candidate_removed": private_removed.contiguous(),
        "private.top1_removed": private_top1_removed.contiguous(),
        "private.top1_sufficient": private_sufficient.contiguous(),
        "private.original_top1": private_top1.contiguous(),
        "private.original_top1_removal_effects": private_effects.contiguous(),
        "candidate_indices": CANDIDATE_INDICES.contiguous(),
        "candidate_mask": V11_CANDIDATE_MASK.contiguous(),
    }
    return result, tensors, rows


def publish(
    *,
    output: Path,
    result: Mapping[str, object],
    tensors: Mapping[str, torch.Tensor],
    rows: Sequence[Mapping[str, object]],
) -> Path:
    target = output.resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        (staging / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        save_file(dict(tensors), str(staging / "intervention_predictions.safetensors"))
        with (staging / "intervention_summary.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(staging, target)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--v16", type=Path, default=DEFAULT_V16)
    parser.add_argument("--v28", type=Path, default=DEFAULT_V28)
    parser.add_argument("--v29", type=Path, default=DEFAULT_V29)
    parser.add_argument(
        "--private-prediction", type=Path, default=DEFAULT_PRIVATE_PREDICTION
    )
    parser.add_argument("--private-h", type=Path, default=DEFAULT_PRIVATE_H)
    parser.add_argument("--private-phase", type=Path, default=DEFAULT_PRIVATE_PHASE)
    parser.add_argument("--private-target", type=Path, default=DEFAULT_PRIVATE_TARGET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result, tensors, rows = run(
        v16_directory=args.v16,
        v28_directory=args.v28,
        v29_directory=args.v29,
        private_prediction_directory=args.private_prediction,
        private_h_directory=args.private_h,
        private_phase_directory=args.private_phase,
        private_target_path=args.private_target,
    )
    output = publish(output=args.output, result=result, tensors=tensors, rows=rows)
    print(
        json.dumps(
            {
                "output": str(output),
                "public_top1_removal_retention": result["public"][
                    "top1_content_removed_stability"
                ]["top1_retention"],
                "private_top1_removal_retention": result["private"][
                    "top1_content_removed_stability_all_88"
                ]["top1_retention"],
                "training_performed": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
