#!/usr/bin/env python3
"""Audit v29 event-bag sensitivity and within-patient seizure consistency.

The frozen v29 H/D heads, fold assignments, candidate mask, fusion weight and
targets are replayed without training or selection.  Public patient bags are
subsampled at prespecified event budgets (1, 2, 4 and 8 seizures, or all
available seizures).  H is re-pooled with the original reliability-weighted
winsorized rule and D is re-pooled by the original equal-logit mean.  The
private cohort remains event-level; no patient consensus target is fabricated.

All results are post-hoc audits on consumed public development data or opened
private transport data.  Subsampling quantiles describe draw variability, not
confirmatory confidence intervals.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
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
    _public_h_probability,
)
from scripts.audit_labram_v29_token_stress_v38 import (  # noqa: E402
    _probability,
    _probability_logits,
    _stability,
    _state_for_fold,
)
from scripts.audit_trustworthy_soz_candidate_v21 import (  # noqa: E402
    _fold_h_only_probability,
)
from scripts.audit_v29_candidate_channel_reliance_v44 import (  # noqa: E402
    _public_d_probability,
)
from scripts.run_labram_fine_temporal_nested_oof_v11_1 import (  # noqa: E402
    _evaluate,
)
from src.soz.v11_reasoner import (  # noqa: E402
    V11_CANDIDATE_MASK,
    extract_block9_phase_contrasts,
    robust_pool_complete_patient_bags,
)


SCHEMA = "trustworthy_soz_v29_patient_bag_event_consistency_v46"
DEFAULT_V16 = (
    ROOT / "outputs/labram_identity_recovery_closed_replay_v16_replay_20260815"
)
DEFAULT_V28 = ROOT / "outputs/labram_rank1_direct_token_oof_v28_20260815"
DEFAULT_V29 = ROOT / "outputs/labram_portable_equal_ensemble_public_oof_v29_20260815"
DEFAULT_PRIVATE = (
    ROOT / "outputs/labram_portable_equal_private_target_blind_v29_20260815"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/trustworthy_soz_v29_patient_bag_event_consistency_v46_20260816"
)
SUBSAMPLE_SIZES = (1, 2, 4, 8)
REPEATS = 100
SEED = 20260816


def _load_public_event_carriers(
    args: argparse.Namespace,
    stable: object,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    prefix, event_patient_index = v28._load_stable_prefix(args, stable)
    h_event = extract_block9_phase_contrasts(prefix).float().cpu().contiguous()
    d_event = v28.extract_rank1_phase_features(prefix)
    del prefix

    fine = v28.v17.identity_v16._load_identity_cache(
        args.stable_fine_directory,
        expected_manifest_sha256=args.expected_stable_fine_manifest_sha256,
        expected_tensor_sha256=args.expected_stable_fine_tensor_sha256,
        tensor_key="features",
        tensor_tail_shape=(19, 20),
        union=stable.union,
        legacy_directory=args.legacy_fine_directory,
        expected_legacy_manifest_sha256=(
            v28.v17.identity_v16.EXPECTED_LEGACY_FINE_MANIFEST_SHA256
        ),
        expected_legacy_tensor_sha256=(
            v28.v17.identity_v16.EXPECTED_LEGACY_FINE_TENSOR_SHA256
        ),
        label="stable fine evidence identity-v12 for v46 bag audit",
    )
    patient_index = {value: index for index, value in enumerate(stable.patient_ids)}
    selected_rows = [
        row
        for row, event in enumerate(stable.union.events)
        if event.patient_id in patient_index
    ]
    selected_event_ids = tuple(stable.union.events[row].event_id for row in selected_rows)
    if selected_event_ids != stable.stable_event_ids:
        raise RuntimeError("public stable event identity/order drifted")
    fine_event = fine.tensor.index_select(
        0, torch.tensor(selected_rows, dtype=torch.long)
    ).float()
    artifact_index = v28.v17.FINE_TEMPORAL_FEATURE_NAMES.index(
        "artifact_burden_0_12s"
    )
    reliability = (1.0 - fine_event[:, :, artifact_index]).clamp(0.0, 1.0)
    if tuple(reliability.shape) != (len(h_event), 19):
        raise RuntimeError("event reliability shape drifted")
    return h_event, d_event, reliability.contiguous(), event_patient_index


def _public_d_event_probability(
    features: torch.Tensor,
    event_folds: torch.Tensor,
    states: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    output = torch.full((len(features), 19), torch.nan, dtype=torch.float32)
    with torch.inference_mode():
        for fold in range(5):
            held = torch.nonzero(event_folds == fold, as_tuple=False).flatten()
            state = _state_for_fold(states, fold)
            model = v28.RankOneDirectTokenHead(state["prior_logits"])
            model.load_state_dict(state, strict=True)
            model.eval().requires_grad_(False)
            logits = model(features.index_select(0, held))
            mask = V11_CANDIDATE_MASK.unsqueeze(0).expand(len(held), -1)
            output[held] = _probability(logits, mask)
    if not torch.isfinite(output).all():
        raise RuntimeError("event-level D replay is incomplete")
    return output.contiguous()


def _selected_event_rows(
    event_patient_index: torch.Tensor,
    *,
    max_events: int,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    rows: list[torch.Tensor] = []
    for patient in range(int(event_patient_index.max()) + 1):
        available = torch.nonzero(
            event_patient_index == patient, as_tuple=False
        ).flatten()
        keep = min(max_events, len(available))
        permutation = torch.randperm(len(available), generator=generator)[:keep]
        rows.append(available.index_select(0, permutation))
    selected = torch.cat(rows).sort().values
    if torch.unique(event_patient_index.index_select(0, selected)).numel() != int(
        event_patient_index.max()
    ) + 1:
        raise RuntimeError("subsampling removed a patient")
    return selected


def _replay_selected_bag(
    *,
    selected: torch.Tensor,
    h_event: torch.Tensor,
    d_event: torch.Tensor,
    reliability: torch.Tensor,
    event_patient_index: torch.Tensor,
    stable: object,
    h_states: Mapping[str, torch.Tensor],
    d_states: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    selected_patient_index = event_patient_index.index_select(0, selected)
    h_patient = robust_pool_complete_patient_bags(
        h_event.index_select(0, selected),
        selected_patient_index,
        len(stable.patient_ids),
        reliability.index_select(0, selected),
    ).features
    h_probability = _public_h_probability(
        h_patient, h_states, stable.patient_folds
    )
    d_probability = _public_d_probability(
        d_event.index_select(0, selected),
        selected_patient_index,
        stable,
        d_states,
    )
    return (0.5 * h_probability + 0.5 * d_probability).contiguous()


def _metric_row(
    probability: torch.Tensor,
    stable: object,
    original: torch.Tensor,
) -> dict[str, float]:
    metrics = _evaluate(
        _probability_logits(probability, stable.target_mask),
        stable.targets,
        stable.target_mask,
    )
    stability = _stability(original, probability, stable.target_mask)
    return {
        "strict": float(metrics["top1"]["strict_accuracy"]),
        "neighborhood4": float(metrics["top1"]["relaxed_accuracy"]),
        "macro_average_precision": float(
            metrics["ranking"]["macro_average_precision"]
        ),
        "hit_at_5": float(metrics["ranking"]["hit_at_k"][5]),
        "far_count": float(metrics["far_error_count"]),
        **stability,
    }


def _draw_summary(rows: Sequence[Mapping[str, float]]) -> dict[str, object]:
    result: dict[str, object] = {"draws": len(rows)}
    for name in rows[0]:
        values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        result[name] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "draw_quantile_2_5_97_5": [
                float(value) for value in np.quantile(values, (0.025, 0.975))
            ],
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }
    return result


def _topk(probability: torch.Tensor, k: int) -> torch.Tensor:
    return torch.topk(
        probability.masked_fill(~V11_CANDIDATE_MASK, -torch.inf),
        k=k,
        dim=1,
    ).indices


def _consistency(
    *,
    probability: torch.Tensor,
    event_patient_index: torch.Tensor,
    patient_ids: Sequence[str],
    patient_probability: torch.Tensor | None,
    dataset: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    top1 = _topk(probability, 1).squeeze(1)
    top3 = _topk(probability, 3)
    patient_top1 = (
        None if patient_probability is None else _topk(patient_probability, 1).squeeze(1)
    )
    rows: list[dict[str, object]] = []
    for patient, patient_id in enumerate(patient_ids):
        selected_rows = torch.nonzero(
            event_patient_index == patient, as_tuple=False
        ).flatten()
        selected = top1.index_select(0, selected_rows)
        counts = torch.bincount(selected, minlength=19).float()[V11_CANDIDATE_MASK]
        vote = counts / counts.sum()
        nonzero = vote > 0
        entropy = float(
            -(vote[nonzero] * vote[nonzero].log()).sum()
            / math.log(int(V11_CANDIDATE_MASK.sum()))
        )
        pair_agreements: list[float] = []
        top3_jaccards: list[float] = []
        for left in range(len(selected_rows)):
            for right in range(left + 1, len(selected_rows)):
                pair_agreements.append(float(selected[left] == selected[right]))
                lhs = set(top3[selected_rows[left]].tolist())
                rhs = set(top3[selected_rows[right]].tolist())
                top3_jaccards.append(len(lhs & rhs) / len(lhs | rhs))
        row: dict[str, object] = {
            "dataset": dataset,
            "unit_id": f"{dataset.split('_', 1)[0].upper()}-{patient:03d}",
            "event_count": len(selected_rows),
            "modal_share": float(counts.max() / counts.sum()),
            "normalized_vote_entropy": entropy,
            "pairwise_top1_agreement": (
                None if not pair_agreements else float(np.mean(pair_agreements))
            ),
            "pairwise_top3_jaccard": (
                None if not top3_jaccards else float(np.mean(top3_jaccards))
            ),
        }
        if patient_top1 is not None:
            row["agreement_with_full_patient_top1"] = float(
                (selected == patient_top1[patient]).float().mean()
            )
        rows.append(row)

    def aggregate(selected_rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
        fields = (
            "modal_share",
            "normalized_vote_entropy",
            "pairwise_top1_agreement",
            "pairwise_top3_jaccard",
            "agreement_with_full_patient_top1",
        )
        summary: dict[str, object] = {
            "patient_count": len(selected_rows),
            "event_count": sum(int(row["event_count"]) for row in selected_rows),
        }
        for field in fields:
            values = [
                float(row[field])
                for row in selected_rows
                if row.get(field) is not None
            ]
            if values:
                summary[f"patient_equal_mean_{field}"] = float(np.mean(values))
        return summary

    summary = {
        "all_patients": aggregate(rows),
        "multi_event_patients": aggregate(
            [row for row in rows if int(row["event_count"]) >= 2]
        ),
        "single_event_patient_count": sum(
            int(row["event_count"]) == 1 for row in rows
        ),
    }
    return summary, rows


def run(
    *,
    v16_directory: Path,
    v28_directory: Path,
    v29_directory: Path,
    private_directory: Path,
    repeats: int,
) -> tuple[
    dict[str, object],
    dict[str, torch.Tensor],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    if repeats < 10:
        raise ValueError("bag audit requires at least 10 fixed draws")
    loader_args = v28.build_parser().parse_args(["--device", "cpu"])
    stable = v28.v17._load_stable_development(loader_args)
    h_event, d_event, reliability, event_patient_index = _load_public_event_carriers(
        loader_args, stable
    )
    full_h_pool = robust_pool_complete_patient_bags(
        h_event,
        event_patient_index,
        len(stable.patient_ids),
        reliability,
    ).features
    h_pool_difference = float((full_h_pool - stable.h_patient).abs().max())
    if h_pool_difference > 1e-6:
        raise ValueError(f"full H bag replay drifted: {h_pool_difference}")

    h_state_path = (v16_directory / "outer_fold_states.safetensors").resolve(
        strict=True
    )
    d_state_path = (v28_directory / "model_and_oof.safetensors").resolve(
        strict=True
    )
    v29_path = (v29_directory / "oof_predictions.safetensors").resolve(strict=True)
    h_states = load_file(str(h_state_path), device="cpu")
    d_states = load_file(str(d_state_path), device="cpu")
    v29 = load_file(str(v29_path), device="cpu")
    original = v29["oof.portable_equal_ensemble_probability"].float()
    full_replay = _replay_selected_bag(
        selected=torch.arange(len(h_event), dtype=torch.long),
        h_event=h_event,
        d_event=d_event,
        reliability=reliability,
        event_patient_index=event_patient_index,
        stable=stable,
        h_states=h_states,
        d_states=d_states,
    )
    identity_difference = float((full_replay - original).abs().max())
    if identity_difference > 1e-5:
        raise ValueError(f"v29 full-bag replay drifted: {identity_difference}")

    draw_rows: list[dict[str, object]] = []
    tensors: dict[str, torch.Tensor] = {
        "public.original_probability": original.contiguous(),
        "public.event_patient_index": event_patient_index.contiguous(),
        "public.patient_event_counts": stable.event_counts.contiguous(),
    }
    bag_summary: dict[str, object] = {
        "all": {
            "selected_events": len(h_event),
            "metrics": _metric_row(original, stable, original),
        }
    }
    for max_events in SUBSAMPLE_SIZES:
        probabilities: list[torch.Tensor] = []
        metric_rows: list[dict[str, float]] = []
        selected_counts: list[int] = []
        for repeat in range(repeats):
            selected = _selected_event_rows(
                event_patient_index,
                max_events=max_events,
                seed=SEED + 100_000 * max_events + repeat,
            )
            probability = _replay_selected_bag(
                selected=selected,
                h_event=h_event,
                d_event=d_event,
                reliability=reliability,
                event_patient_index=event_patient_index,
                stable=stable,
                h_states=h_states,
                d_states=d_states,
            )
            metrics = _metric_row(probability, stable, original)
            probabilities.append(probability)
            metric_rows.append(metrics)
            selected_counts.append(len(selected))
            draw_rows.append(
                {
                    "max_events_per_patient": max_events,
                    "repeat": repeat,
                    "selected_events": len(selected),
                    **metrics,
                }
            )
        tensors[f"public.max_{max_events}_events_probability"] = torch.stack(
            probabilities
        ).contiguous()
        bag_summary[str(max_events)] = {
            "repeats": repeats,
            "selected_events_per_draw": selected_counts[0],
            "selection_rule": "uniform_without_replacement_within_each_patient",
            "metrics_and_stability": _draw_summary(metric_rows),
        }

    event_folds = stable.patient_folds.index_select(0, event_patient_index)
    h_event_probability = _public_h_probability(h_event, h_states, event_folds)
    d_event_probability = _public_d_event_probability(d_event, event_folds, d_states)
    public_event_probability = (
        0.5 * h_event_probability + 0.5 * d_event_probability
    ).contiguous()
    public_consistency, public_patient_rows = _consistency(
        probability=public_event_probability,
        event_patient_index=event_patient_index,
        patient_ids=stable.patient_ids,
        patient_probability=original,
        dataset="public_consumed_development",
    )
    tensors["public.event_probability"] = public_event_probability

    private_manifest_path = (private_directory / "manifest.json").resolve(strict=True)
    private_tensor_path = (private_directory / "predictions.safetensors").resolve(
        strict=True
    )
    private_manifest = json.loads(private_manifest_path.read_text(encoding="utf-8"))
    private_events = private_manifest.get("events")
    if not isinstance(private_events, list) or len(private_events) != 88:
        raise ValueError("private event roster changed")
    private_payload = load_file(str(private_tensor_path), device="cpu")
    private_probability = private_payload["private_portable_equal_probability"].float()
    private_patient_ids = tuple(sorted({str(row["patient_id"]) for row in private_events}))
    private_patient_index = {value: index for index, value in enumerate(private_patient_ids)}
    private_event_patient_index = torch.tensor(
        [private_patient_index[str(row["patient_id"])] for row in private_events],
        dtype=torch.long,
    )
    private_consistency, private_patient_rows = _consistency(
        probability=private_probability,
        event_patient_index=private_event_patient_index,
        patient_ids=private_patient_ids,
        patient_probability=None,
        dataset="private_post_open_target_blind",
    )
    tensors["private.event_probability"] = private_probability.contiguous()
    tensors["private.event_patient_index"] = private_event_patient_index.contiguous()

    result: dict[str, object] = {
        "schema_version": SCHEMA,
        "status": "completed_frozen_v29_patient_bag_and_event_consistency_audit",
        "analysis_role": {
            "public": "posthoc_consumed_development_bag_sensitivity",
            "private": "post_open_target_blind_within_patient_prediction_consistency",
        },
        "public": {
            "patient_count": len(stable.patient_ids),
            "event_count": len(h_event),
            "event_count_distribution": {
                "minimum": int(stable.event_counts.min()),
                "median": float(stable.event_counts.float().median()),
                "maximum": int(stable.event_counts.max()),
                "patients_with_one_event": int((stable.event_counts == 1).sum()),
                "patients_with_at_least_two_events": int((stable.event_counts >= 2).sum()),
            },
            "bag_subsampling": bag_summary,
            "event_to_full_patient_prediction_consistency": public_consistency,
        },
        "private": {
            "target_blind_event_count": len(private_events),
            "patient_count": len(private_patient_ids),
            "event_prediction_consistency": private_consistency,
            "patient_consensus_target_created": False,
            "patient_bag_performance_computed": False,
        },
        "identity_replay": {
            "H_full_bag_max_absolute_difference": h_pool_difference,
            "v29_full_probability_max_absolute_difference": identity_difference,
        },
        "source_files": {
            "H_fold_states": str(h_state_path.relative_to(ROOT)),
            "D_fold_states": str(d_state_path.relative_to(ROOT)),
            "public_v29": str(v29_path.relative_to(ROOT)),
            "private_manifest": str(private_manifest_path.relative_to(ROOT)),
            "private_prediction": str(private_tensor_path.relative_to(ROOT)),
        },
        "access_receipt": {
            "cached_foundation_carriers_loaded": True,
            "raw_EEG_loaded": False,
            "foundation_forward_performed": False,
            "model_training_performed": False,
            "target_or_private_outcome_used_for_subsampling": False,
            "model_threshold_fusion_or_report_changed": False,
            "public_targets_loaded_for_frozen_metrics": True,
            "private_targets_loaded": False,
        },
        "interpretation_boundary": {
            "draw_quantiles_are_confidence_intervals": False,
            "single_event_public_reference_is_event_specific_gold": False,
            "private_patient_consensus_gold_inferred": False,
            "bag_curve_used_for_model_selection": False,
            "allowed_claim": (
                "the frozen v29 ranker's sensitivity to the number of observed "
                "seizures and within-patient prediction consistency are quantified"
            ),
        },
    }
    patient_rows = public_patient_rows + private_patient_rows
    return result, tensors, draw_rows, patient_rows


def publish(
    *,
    output: Path,
    result: Mapping[str, object],
    tensors: Mapping[str, torch.Tensor],
    draw_rows: Sequence[Mapping[str, object]],
    patient_rows: Sequence[Mapping[str, object]],
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
        save_file(dict(tensors), str(staging / "bag_and_event_predictions.safetensors"))
        with (staging / "bag_subsample_draws.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(draw_rows[0]))
            writer.writeheader()
            writer.writerows(draw_rows)
        fields = sorted({key for row in patient_rows for key in row})
        with (staging / "hashed_patient_consistency.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(patient_rows)
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
    parser.add_argument("--private", type=Path, default=DEFAULT_PRIVATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result, tensors, draws, patients = run(
        v16_directory=args.v16,
        v28_directory=args.v28,
        v29_directory=args.v29,
        private_directory=args.private,
        repeats=args.repeats,
    )
    output = publish(
        output=args.output,
        result=result,
        tensors=tensors,
        draw_rows=draws,
        patient_rows=patients,
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "status": result["status"],
                "public_event_count": result["public"]["event_count"],
                "private_event_count": result["private"]["target_blind_event_count"],
                "training_performed": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
