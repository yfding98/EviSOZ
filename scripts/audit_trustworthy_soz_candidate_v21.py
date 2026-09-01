#!/usr/bin/env python3
"""Audit the v21 target-blind evidence-family gate and frozen fallback.

This is a post-private-open exploratory audit.  It fits no model and selects no
threshold from private outcomes.  The fine-family transport decision uses only
source and target feature distributions; private reference values are opened
after that decision solely to describe the already-frozen fallback.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
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

from scripts.evaluate_private_labram_zero_adaptation_v18 import (  # noqa: E402
    STANDARD_19,
    V11_CANDIDATE_MASK,
    _aggregate,
    _as_json_list,
    _rank_metrics,
    _read_csv,
)


DEFAULT_PROTOCOL = ROOT / "configs/trustworthy_soz_candidate_v21.json"
DEFAULT_STATES = (
    ROOT
    / "outputs/labram_identity_recovery_closed_replay_v16_20260812/"
    "outer_fold_states.safetensors"
)
DEFAULT_V16 = (
    ROOT / "outputs/labram_identity_recovery_closed_replay_v16_20260812"
)
DEFAULT_V17 = ROOT / "outputs/labram_masked_variable_auxiliary_oof_v17_20260812"
DEFAULT_PUBLIC_FINE = (
    ROOT / "outputs/public_development_fine_evidence_identity_v12_20260812"
)
DEFAULT_PRIVATE_EVIDENCE = (
    ROOT / "outputs/labram_private_target_blind_evidence_v18_20260814"
)
DEFAULT_PRIVATE_BUNDLE = (
    ROOT / "outputs/labram_private_zero_adaptation_bundle_v18_20260814"
)
DEFAULT_PRIVATE_V18 = (
    ROOT / "outputs/labram_private_zero_adaptation_v18_20260814"
)
DEFAULT_DEEPSOZ = ROOT / "outputs/deepsoz_official_local_oof_full.json"
DEFAULT_OUTPUT = ROOT / "outputs/trustworthy_soz_candidate_v21_20260815"
N_FOLDS = 5
H_ONLY_ARM = "frozen_labram_only"


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _fold_h_only_probability(
    h: torch.Tensor,
    states: Mapping[str, torch.Tensor],
    fold: int,
) -> torch.Tensor:
    prefix = f"outer{fold}."
    arm = prefix + H_ONLY_ARM + "."
    if not torch.equal(states[arm + "candidate_mask"], V11_CANDIDATE_MASK):
        raise ValueError("v21 H-only candidate mask drifted")
    transformed = torch.matmul(
        (h - states[prefix + "transform.h_center"])
        / states[prefix + "transform.h_scale"]
        - states[prefix + "transform.h_pca_mean"],
        states[prefix + "transform.h_components"],
    )
    logits = states[arm + "prior_logits"].expand(h.shape[0], -1).clone()
    logits += torch.einsum("ecd,d->ec", transformed, states[arm + "h_weight"])
    logits = logits.masked_fill(~V11_CANDIDATE_MASK, -torch.inf)
    probability = torch.softmax(logits, dim=1)
    if not torch.isfinite(probability).all() or not torch.allclose(
        probability.sum(dim=1), torch.ones(h.shape[0]), atol=1e-6, rtol=0
    ):
        raise RuntimeError("v21 H-only probability contract failed")
    return probability


def _patient_statistic(values: torch.Tensor) -> dict[str, float]:
    absolute = values.abs().reshape(-1)
    return {
        "median_abs_z": float(absolute.median()),
        "fraction_abs_z_gt_5": float((absolute > 5.0).float().mean()),
        "fraction_abs_z_gt_10": float((absolute > 10.0).float().mean()),
    }


def _source_patient_statistics(
    features: torch.Tensor,
    events: Sequence[Mapping[str, object]],
    states: Mapping[str, torch.Tensor],
) -> dict[str, dict[str, float]]:
    by_patient: dict[str, list[torch.Tensor]] = defaultdict(list)
    patient_fold: dict[str, int] = {}
    for index, event in enumerate(events):
        patient = str(event["patient_id"])
        fold = int(event["outer_fold"])
        previous = patient_fold.setdefault(patient, fold)
        if previous != fold:
            raise ValueError("one public patient crosses v21 outer folds")
        prefix = f"outer{fold}.transform."
        standardized = (
            features[index] - states[prefix + "fine_center"]
        ) / states[prefix + "fine_scale"]
        by_patient[patient].append(standardized)
    return {
        patient: _patient_statistic(torch.stack(rows))
        for patient, rows in sorted(by_patient.items())
    }


def _target_patient_statistics(
    features: torch.Tensor,
    events: Sequence[Mapping[str, object]],
    states: Mapping[str, torch.Tensor],
) -> dict[str, dict[str, float]]:
    indices: dict[str, list[int]] = defaultdict(list)
    for index, event in enumerate(events):
        indices[str(event["patient_id"])].append(index)
    result: dict[str, dict[str, float]] = {}
    for patient, rows in sorted(indices.items()):
        selected = features[torch.tensor(rows, dtype=torch.long)]
        fold_statistics = []
        for fold in range(N_FOLDS):
            prefix = f"outer{fold}.transform."
            standardized = (
                selected - states[prefix + "fine_center"]
            ) / states[prefix + "fine_scale"]
            fold_statistics.append(_patient_statistic(standardized))
        result[patient] = {
            key: float(np.median([row[key] for row in fold_statistics]))
            for key in fold_statistics[0]
        }
    return result


def _decide_transport(
    source: Mapping[str, Mapping[str, float]],
    target: Mapping[str, Mapping[str, float]],
    quantile: float,
) -> dict[str, object]:
    if not 0.5 < quantile < 1.0:
        raise ValueError("source patient quantile must be in (0.5,1)")
    statistic_names = tuple(next(iter(source.values())).keys())
    thresholds = {
        name: float(np.quantile([row[name] for row in source.values()], quantile))
        for name in statistic_names
    }
    target_medians = {
        name: float(np.median([row[name] for row in target.values()]))
        for name in statistic_names
    }
    checks = {
        name: target_medians[name] <= thresholds[name] for name in statistic_names
    }
    passed = all(checks.values())
    return {
        "source_patient_count": len(source),
        "target_patient_count": len(target),
        "source_patient_quantile": quantile,
        "source_thresholds": thresholds,
        "target_domain_patient_medians": target_medians,
        "checks": checks,
        "passed": passed,
        "selected_arm": (
            "masked_variable_auxiliary_full_v17" if passed else H_ONLY_ARM
        ),
        "target_values_used_for_decision": False,
    }


def _private_metrics(
    probability: torch.Tensor,
    events: Sequence[Mapping[str, object]],
    target_rows: Sequence[Mapping[str, str]],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    event_index = {str(row["event_id"]): index for index, row in enumerate(events)}
    selected = [
        row
        for row in target_rows
        if row["primary_reference_preeligible"] == "1"
        and row["event_id"] in event_index
    ]
    if len(selected) != 51 or len({row["patient_id"] for row in selected}) != 23:
        raise RuntimeError("v21 private primary denominator drifted")
    rows: list[dict[str, object]] = []
    for target in selected:
        index = event_index[target["event_id"]]
        positive = _as_json_list(target["candidate_positive_electrodes"])
        spread = _as_json_list(target["known_spread_electrodes"])
        score = probability[index]
        rows.append(
            {
                "event_id": target["event_id"],
                "patient_id": target["patient_id"],
                "top1": STANDARD_19[int(score.argmax())],
                **_rank_metrics(score, positive, spread),
            }
        )
    metric_names = (
        "exact",
        "relaxed_neighbor4",
        "hit_at_3",
        "hit_at_5",
        "mrr",
        "positive_recall_at_3",
        "positive_recall_at_5",
        "laterality_agreement",
        "bucket_agreement",
        "significant_over_spread_pairwise",
    )
    return _aggregate(rows, metric_names), rows


def run(args: argparse.Namespace) -> dict[str, object]:
    protocol = _read_json(args.protocol)
    if protocol.get("schema_version") != "trustworthy_soz_candidate_protocol_v21":
        raise ValueError("wrong v21 protocol schema")
    gate_contract = protocol.get("evidence_family_transport_gate")
    if not isinstance(gate_contract, Mapping):
        raise TypeError("v21 protocol lacks evidence family gate")
    quantile = float(gate_contract["source_patient_quantile"])

    states = load_file(str(args.states.resolve(strict=True)))
    public_manifest = _read_json(args.public_fine / "manifest.json")
    public_tensors = load_file(
        str(args.public_fine / str(public_manifest["tensor_file"]))
    )
    public_features = public_tensors["features"].float().contiguous()
    public_events = public_manifest["events"]
    if not isinstance(public_events, list) or len(public_events) != len(public_features):
        raise ValueError("v21 public fine evidence roster drifted")

    private_manifest = _read_json(args.private_evidence / "manifest.json")
    private_tensors = load_file(
        str(args.private_evidence / str(private_manifest["tensor_file"]))
    )
    private_features = private_tensors["fine_event"].float().contiguous()
    private_h = private_tensors["h_event"].float().contiguous()
    private_events = private_manifest["events"]
    if not isinstance(private_events, list) or len(private_events) != len(private_h):
        raise ValueError("v21 private evidence roster drifted")

    source_statistics = _source_patient_statistics(
        public_features, public_events, states
    )
    target_statistics = _target_patient_statistics(
        private_features, private_events, states
    )
    gate = _decide_transport(source_statistics, target_statistics, quantile)

    fold_probability = torch.stack(
        [
            _fold_h_only_probability(private_h, states, fold)
            for fold in range(N_FOLDS)
        ],
        dim=1,
    ).contiguous()
    h_only_probability = fold_probability.mean(dim=1).contiguous()

    # The gate above is complete before any private target value is loaded.
    target_rows = _read_csv(args.private_bundle / "target_ledger.csv")
    private_metrics, evaluation_rows = _private_metrics(
        h_only_probability, private_events, target_rows
    )

    v16 = _read_json(args.v16 / "manifest.json")
    v17 = _read_json(args.v17 / "manifest.json")
    deepsoz = _read_json(args.deepsoz)
    private_v18 = _read_json(args.private_v18 / "metrics.json")
    public_h_only = v16["absolute_patient_bootstrap_all_102"][H_ONLY_ARM]
    public_full = v17["primary_comparison"]["candidate_metrics"]
    local_deepsoz = deepsoz["held_out_ensemble_metrics"]
    private_full = private_v18["evaluations"]["primary"]["event"]

    payload = {
        "schema_version": "trustworthy_soz_candidate_result_v21",
        "status": "post_open_exploratory_domain_qualified_result_complete",
        "paper_objective": protocol["paper_objective"],
        "unit_correction": {
            "deepsoz_manifest": "652_records_124_patients",
            "public_primary": "102_patients_patient_level_reference",
            "public_core_signal_events": 988,
            "private_primary": "51_events_23_patients_event_level_reference",
            "events_are_not_independent_soz_labels": True,
        },
        "target_blind_evidence_family_gate": gate,
        "public_development": {
            "local_published_deepsoz_weight_transfer": local_deepsoz,
            "frozen_h_only_v16": public_h_only,
            "active_in_domain_full_v17": public_full,
        },
        "private_exploratory": {
            "previous_full_v18": private_full,
            "domain_gate_selected_arm": gate["selected_arm"],
            "h_only_fallback": private_metrics,
        },
        "claim_audit": {
            "public_neighborhood4_exceeds_paper_point_estimate": (
                float(public_full["top1"]["relaxed_accuracy"])
                > float(protocol["performance_claim"]["public_target_point_estimate"])
            ),
            "private_h_only_event_micro_relaxed_exceeds_0_70": (
                float(private_metrics["event_micro"]["relaxed_neighbor4"])
                > float(protocol["performance_claim"]["private_exploratory_relaxed_floor"])
            ),
            "strict_top1_exceeds_0_70": False,
            "same_metric_and_unit_across_public_private": False,
            "significantly_superior_to_deepsoz_proven": False,
            "private_external_confirmation": False,
            "clinical_patient_level_abstention_calibrated": False,
        },
        "access_receipt": {
            "model_training_performed": False,
            "private_target_values_used_for_family_gate": False,
            "private_target_values_opened_after_family_gate": True,
            "private_used_for_parameter_fitting": False,
            "private_raw_eeg_reopened": False,
            "private_target_blind_cached_evidence_reused": True,
        },
        "limitations": [
            "post_private_open_exploratory_revision_not_untouched_validation",
            "public_102_patients_are_repeatedly_used_development_benchmark",
            "public_patient_level_and_private_event_level_references_are_not_exchangeable",
            "neighborhood4_is_a_relaxed_sensitivity_not_strict_channel_accuracy",
            "patient_level_selective_abstention_threshold_remains_uncalibrated",
        ],
    }

    target = args.output.resolve()
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        save_file(
            {
                "private_h_only_fold_probability": fold_probability,
                "private_h_only_probability": h_only_probability,
            },
            str(staging / "predictions.safetensors"),
        )
        (staging / "result.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        with (staging / "evaluation_rows.jsonl").open("w", encoding="utf-8") as stream:
            for row in evaluation_rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n")
        os.rename(staging, target)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--states", type=Path, default=DEFAULT_STATES)
    parser.add_argument("--v16", type=Path, default=DEFAULT_V16)
    parser.add_argument("--v17", type=Path, default=DEFAULT_V17)
    parser.add_argument("--public-fine", type=Path, default=DEFAULT_PUBLIC_FINE)
    parser.add_argument("--private-evidence", type=Path, default=DEFAULT_PRIVATE_EVIDENCE)
    parser.add_argument("--private-bundle", type=Path, default=DEFAULT_PRIVATE_BUNDLE)
    parser.add_argument("--private-v18", type=Path, default=DEFAULT_PRIVATE_V18)
    parser.add_argument("--deepsoz", type=Path, default=DEFAULT_DEEPSOZ)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "fine_family_transport_passed": result[
                    "target_blind_evidence_family_gate"
                ]["passed"],
                "selected_arm": result["target_blind_evidence_family_gate"][
                    "selected_arm"
                ],
                "public_relaxed": result["public_development"][
                    "active_in_domain_full_v17"
                ]["top1"]["relaxed_accuracy"],
                "private_fallback_relaxed": result["private_exploratory"][
                    "h_only_fallback"
                ]["event_micro"]["relaxed_neighbor4"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
