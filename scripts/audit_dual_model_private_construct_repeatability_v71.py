#!/usr/bin/env python3
"""Audit two frozen private rankers for construct sensitivity and repeatability.

This is a read-only post-open method audit.  It compares the unchanged v29 and
Raw200-Shallow private probability tensors under the same documented-reference
perturbation and target-blind within-patient repeatability definitions.  It
does not train, tune, select, calibrate, aggregate, or replace either model.
No unlabeled electrode or private spread electrode is promoted to a positive.
"""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Mapping, Sequence

import numpy as np
import torch
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import audit_v29_reference_set_perturbation_v67 as v67  # noqa: E402
from scripts.audit_v29_patient_bag_event_consistency_v46 import (  # noqa: E402
    _consistency,
)
from src.soz.geometry import CHANNEL_INDEX, STANDARD_19  # noqa: E402
from src.soz.v11_reasoner import V11_CANDIDATE_MASK  # noqa: E402


SCHEMA = "trustworthy_soz_dual_model_private_construct_repeatability_v71"
DEFAULT_V29 = ROOT / "outputs/labram_portable_equal_private_target_blind_v29_20260815"
DEFAULT_RAW200 = ROOT / "outputs/trustworthy_soz_raw200_shallow_baseline_v60_20260816"
DEFAULT_PRIVATE_AUDIT = ROOT / "outputs/trustworthy_soz_private_frozen_publication_v36_20260816"
DEFAULT_OUTPUT = ROOT / "outputs/trustworthy_soz_dual_model_private_construct_repeatability_v71_20260816"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260871
CANDIDATES = tuple(channel for channel in STANDARD_19 if channel != "PZ")


def _load_predictions(
    v29_directory: Path,
    raw200_directory: Path,
) -> tuple[list[dict[str, object]], dict[str, torch.Tensor]]:
    v29_manifest = json.loads(
        (v29_directory / "manifest.json").resolve(strict=True).read_text(encoding="utf-8")
    )
    raw_manifest = json.loads(
        (raw200_directory / "manifest.json").resolve(strict=True).read_text(encoding="utf-8")
    )
    v29_events = v29_manifest.get("events")
    raw_events = raw_manifest.get("private", {}).get("events")
    if not isinstance(v29_events, list) or not isinstance(raw_events, list):
        raise ValueError("private event manifests are missing")
    if len(v29_events) != 88 or len(raw_events) != 88:
        raise ValueError("frozen private roster changed")
    v29_identity = [
        (str(row["event_id"]), str(row["patient_id"])) for row in v29_events
    ]
    raw_identity = [
        (str(row["event_id"]), str(row["patient_id"])) for row in raw_events
    ]
    if v29_identity != raw_identity:
        raise ValueError("v29 and Raw200 private event order differs")

    v29 = load_file(
        str((v29_directory / "predictions.safetensors").resolve(strict=True)),
        device="cpu",
    )["private_portable_equal_probability"].float()
    raw_payload = load_file(
        str((raw200_directory / "raw200_shallow_predictions.safetensors").resolve(strict=True)),
        device="cpu",
    )
    raw = raw_payload["private.probability"].float()
    candidate_mask = raw_payload["candidate_mask"].bool()
    if tuple(v29.shape) != (88, 19) or tuple(raw.shape) != (88, 19):
        raise ValueError("private probability tensor shape changed")
    if not torch.equal(candidate_mask, V11_CANDIDATE_MASK):
        raise ValueError("candidate mask differs across frozen models")
    if not torch.isfinite(v29).all() or not torch.isfinite(raw).all():
        raise ValueError("nonfinite private probability")
    return [dict(row) for row in v29_events], {"v29": v29, "raw200": raw}


def _reference_rows(
    *,
    events: Sequence[Mapping[str, object]],
    probabilities: Mapping[str, torch.Tensor],
    private_audit_directory: Path,
) -> dict[str, list[dict[str, object]]]:
    event_index = {str(row["event_id"]): index for index, row in enumerate(events)}
    output: dict[str, list[dict[str, object]]] = {name: [] for name in probabilities}
    with (private_audit_directory / "private_event_error_audit.csv").resolve(
        strict=True
    ).open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            event_id = str(raw["unit_id"])
            positives = [
                CHANNEL_INDEX[str(value)] for value in ast.literal_eval(raw["positive_channels"])
            ]
            spread = [
                CHANNEL_INDEX[str(value)]
                for value in ast.literal_eval(raw["known_spread_channels"])
            ]
            for model, probability in probabilities.items():
                row = v67._unit_row(
                    dataset="private_post_open_transport",
                    unit_id=event_id,
                    patient_id=str(raw["patient_id"]),
                    probability=probability[event_index[event_id]],
                    evaluable_indices=[CHANNEL_INDEX[value] for value in CANDIDATES],
                    positive_indices=positives,
                    spread_indices=spread,
                )
                row["model"] = model
                output[model].append(row)
    for model, rows in output.items():
        if len(rows) != 51 or len({str(row["patient_id"]) for row in rows}) != 23:
            raise ValueError(f"{model} private evaluable roster changed")
    return output


def _paired_patient_delta(
    left: Sequence[Mapping[str, object]],
    right: Sequence[Mapping[str, object]],
    metric: str,
    *,
    seed: int,
) -> dict[str, object]:
    right_by_id = {str(row["unit_id"]): row for row in right}
    bags: dict[str, list[float]] = defaultdict(list)
    for row in left:
        other = right_by_id[str(row["unit_id"])]
        bags[str(row["patient_id"])].append(float(row[metric]) - float(other[metric]))
    patient_delta = np.asarray(
        [np.mean(bags[patient]) for patient in sorted(bags)], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    sampled = rng.integers(
        0,
        len(patient_delta),
        size=(BOOTSTRAP_REPLICATES, len(patient_delta)),
    )
    bootstrap = patient_delta[sampled].mean(axis=1)
    return {
        "patients": len(patient_delta),
        "patient_equal_delta_v29_minus_raw200": float(patient_delta.mean()),
        "patient_cluster_bootstrap_ci95": [
            float(value) for value in np.quantile(bootstrap, (0.025, 0.975))
        ],
    }


def _repeatability(
    *,
    events: Sequence[Mapping[str, object]],
    probabilities: Mapping[str, torch.Tensor],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    patient_ids = tuple(sorted({str(row["patient_id"]) for row in events}))
    patient_index = {value: index for index, value in enumerate(patient_ids)}
    event_patient_index = torch.tensor(
        [patient_index[str(row["patient_id"])] for row in events], dtype=torch.long
    )
    summaries: dict[str, object] = {}
    all_rows: dict[str, list[dict[str, object]]] = {}
    published_rows: list[dict[str, object]] = []
    for model, probability in probabilities.items():
        summary, rows = _consistency(
            probability=probability,
            event_patient_index=event_patient_index,
            patient_ids=patient_ids,
            patient_probability=None,
            dataset=f"private_{model}_target_blind",
        )
        for patient_id, row in zip(patient_ids, rows, strict=True):
            row["patient_id"] = patient_id
            row["model"] = model
        summaries[model] = summary
        all_rows[model] = rows
        published_rows.extend(rows)

    raw_by_patient = {str(row["patient_id"]): row for row in all_rows["raw200"]}
    paired: dict[str, object] = {}
    for offset, metric in enumerate(
        (
            "modal_share",
            "normalized_vote_entropy",
            "pairwise_top1_agreement",
            "pairwise_top3_jaccard",
        )
    ):
        deltas = []
        for row in all_rows["v29"]:
            other = raw_by_patient[str(row["patient_id"])]
            if row.get(metric) is None or other.get(metric) is None:
                continue
            deltas.append(float(row[metric]) - float(other[metric]))
        values = np.asarray(deltas, dtype=np.float64)
        rng = np.random.default_rng(BOOTSTRAP_SEED + 100 + offset)
        sampled = rng.integers(
            0, len(values), size=(BOOTSTRAP_REPLICATES, len(values))
        )
        bootstrap = values[sampled].mean(axis=1)
        paired[metric] = {
            "patients": len(values),
            "patient_equal_delta_v29_minus_raw200": float(values.mean()),
            "patient_bootstrap_ci95": [
                float(value) for value in np.quantile(bootstrap, (0.025, 0.975))
            ],
        }
    return {"models": summaries, "paired": paired}, published_rows


def _cross_model_concordance(
    *,
    events: Sequence[Mapping[str, object]],
    probabilities: Mapping[str, torch.Tensor],
    reference_rows: Mapping[str, Sequence[Mapping[str, object]]],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    mask = V11_CANDIDATE_MASK.unsqueeze(0).expand(len(events), -1)
    rankings = {
        name: torch.argsort(
            probability.masked_fill(~mask, -torch.inf),
            dim=1,
            descending=True,
            stable=True,
        )
        for name, probability in probabilities.items()
    }
    v29_reference = {str(row["unit_id"]): row for row in reference_rows["v29"]}
    raw_reference = {str(row["unit_id"]): row for row in reference_rows["raw200"]}
    rows: list[dict[str, object]] = []
    top1_agreement = []
    top3_jaccard = []
    per_patient_top1: dict[str, list[float]] = defaultdict(list)
    evaluated_counts = {"both_strict": 0, "v29_only_strict": 0, "raw200_only_strict": 0, "neither_strict": 0}
    for index, event in enumerate(events):
        event_id = str(event["event_id"])
        patient_id = str(event["patient_id"])
        left = rankings["v29"][index]
        right = rankings["raw200"][index]
        agreement = float(left[0] == right[0])
        left_top3 = set(left[:3].tolist())
        right_top3 = set(right[:3].tolist())
        jaccard = len(left_top3 & right_top3) / len(left_top3 | right_top3)
        top1_agreement.append(agreement)
        top3_jaccard.append(jaccard)
        per_patient_top1[patient_id].append(agreement)
        row: dict[str, object] = {
            "event_id": event_id,
            "patient_id": patient_id,
            "v29_top1": STANDARD_19[int(left[0])],
            "raw200_top1": STANDARD_19[int(right[0])],
            "top1_agreement": agreement,
            "top3_jaccard": jaccard,
            "reference_evaluable": event_id in v29_reference,
            "v29_strict": None,
            "raw200_strict": None,
        }
        if event_id in v29_reference:
            v29_strict = int(float(v29_reference[event_id]["original_set_strict"]))
            raw_strict = int(float(raw_reference[event_id]["original_set_strict"]))
            row["v29_strict"] = v29_strict
            row["raw200_strict"] = raw_strict
            category = (
                "both_strict"
                if v29_strict and raw_strict
                else "v29_only_strict"
                if v29_strict
                else "raw200_only_strict"
                if raw_strict
                else "neither_strict"
            )
            evaluated_counts[category] += 1
        rows.append(row)
    return {
        "all_88_event_micro_top1_agreement": float(np.mean(top1_agreement)),
        "all_88_patient_equal_top1_agreement": float(
            np.mean([np.mean(values) for values in per_patient_top1.values()])
        ),
        "all_88_event_micro_top3_jaccard": float(np.mean(top3_jaccard)),
        "evaluated_51_strict_overlap_counts": evaluated_counts,
    }, rows


def run(
    *,
    v29_directory: Path,
    raw200_directory: Path,
    private_audit_directory: Path,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    events, probabilities = _load_predictions(v29_directory, raw200_directory)
    reference_rows = _reference_rows(
        events=events,
        probabilities=probabilities,
        private_audit_directory=private_audit_directory,
    )
    if sum(float(row["original_set_strict"]) for row in reference_rows["v29"]) != 25:
        raise RuntimeError("v29 strict endpoint did not replay")
    if sum(float(row["original_set_strict"]) for row in reference_rows["raw200"]) != 21:
        raise RuntimeError("Raw200 strict endpoint did not replay")

    reference_summary = {
        model: {
            "summary": v67._summary(rows, seed=BOOTSTRAP_SEED + index * 20_000),
            "positive_set_size_strata": v67._size_strata(
                rows, seed=BOOTSTRAP_SEED + 10_000 + index * 20_000
            ),
        }
        for index, (model, rows) in enumerate(reference_rows.items())
    }
    paired_reference = {
        metric: _paired_patient_delta(
            reference_rows["v29"],
            reference_rows["raw200"],
            metric,
            seed=BOOTSTRAP_SEED + 50_000 + offset,
        )
        for offset, metric in enumerate(
            (
                "original_set_strict",
                "documented_singleton_uniform_top1",
                "documented_singleton_uniform_hit_at_3",
                "documented_singleton_uniform_hit_at_5",
                "documented_singleton_mean_rank",
                "set_cardinality_gain_top1",
            )
        )
    }
    repeatability, repeatability_rows = _repeatability(
        events=events, probabilities=probabilities
    )
    concordance, concordance_rows = _cross_model_concordance(
        events=events,
        probabilities=probabilities,
        reference_rows=reference_rows,
    )

    result: dict[str, object] = {
        "schema_version": SCHEMA,
        "status": "completed_frozen_dual_model_private_construct_repeatability_audit",
        "analysis_role": "post_open_read_only_private_method_audit",
        "cohort": {
            "target_blind_prediction_events": 88,
            "target_blind_prediction_patients": 31,
            "reference_evaluable_events": 51,
            "reference_evaluable_patient_clusters": 23,
        },
        "models": {
            "v29": "frozen_LaBraM_H_D_probability_mean",
            "raw200": "frozen_3425_parameter_full_bandwidth_comparator",
        },
        "reference_construct": {
            "models": reference_summary,
            "paired_v29_minus_raw200": paired_reference,
        },
        "target_blind_repeatability": repeatability,
        "cross_model_concordance": concordance,
        "audit_contract": {
            "complete_documented_positive_set_remains_formal": True,
            "unlabeled_electrode_added": False,
            "known_spread_added_to_positive": False,
            "patient_consensus_target_inferred": False,
            "model_trained_tuned_selected_calibrated_or_aggregated": False,
            "formal_v29_or_raw200_prediction_changed": False,
        },
        "access_receipt": {
            "frozen_private_probability_tensors_loaded": True,
            "private_reference_loaded_for_read_only_construct_audit": True,
            "private_reference_loaded_for_repeatability": False,
            "raw_EEG_loaded": False,
            "foundation_forward_performed": False,
            "model_training_or_selection_performed": False,
        },
        "interpretation_boundary": {
            "dual_model_replication_proves_model_agnostic_generality": False,
            "reference_singleton_functional_is_accuracy": False,
            "repeatability_identifies_true_patient_SOZ": False,
            "paired_post_open_interval_is_confirmatory": False,
            "allowed_claim": (
                "the same documented-reference and target-blind repeatability audits "
                "were applied without adaptation to two frozen models on the private cohort"
            ),
        },
        "bootstrap": {
            "unit": "patient_cluster",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
        },
        "files": {
            "reference_rows": "reference_rows.csv",
            "repeatability_rows": "repeatability_rows.csv",
            "cross_model_rows": "cross_model_rows.csv",
        },
    }
    flat_reference_rows = reference_rows["v29"] + reference_rows["raw200"]
    return result, flat_reference_rows, repeatability_rows, concordance_rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def publish(
    *,
    output: Path,
    result: Mapping[str, object],
    reference_rows: Sequence[Mapping[str, object]],
    repeatability_rows: Sequence[Mapping[str, object]],
    concordance_rows: Sequence[Mapping[str, object]],
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
        _write_csv(staging / "reference_rows.csv", reference_rows)
        _write_csv(staging / "repeatability_rows.csv", repeatability_rows)
        _write_csv(staging / "cross_model_rows.csv", concordance_rows)
        os.replace(staging, target)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--v29-directory", type=Path, default=DEFAULT_V29)
    parser.add_argument("--raw200-directory", type=Path, default=DEFAULT_RAW200)
    parser.add_argument("--private-audit-directory", type=Path, default=DEFAULT_PRIVATE_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result, reference_rows, repeatability_rows, concordance_rows = run(
        v29_directory=args.v29_directory,
        raw200_directory=args.raw200_directory,
        private_audit_directory=args.private_audit_directory,
    )
    output = publish(
        output=args.output,
        result=result,
        reference_rows=reference_rows,
        repeatability_rows=repeatability_rows,
        concordance_rows=concordance_rows,
    )
    models = result["reference_construct"]["models"]
    repeatability = result["target_blind_repeatability"]["models"]
    print(
        json.dumps(
            {
                "output": str(output),
                "v29_singleton_top1": models["v29"]["summary"]["documented_singleton_uniform_top1"]["unit_micro"],
                "raw200_singleton_top1": models["raw200"]["summary"]["documented_singleton_uniform_top1"]["unit_micro"],
                "v29_repeatability_top1": repeatability["v29"]["multi_event_patients"]["patient_equal_mean_pairwise_top1_agreement"],
                "raw200_repeatability_top1": repeatability["raw200"]["multi_event_patients"]["patient_equal_mean_pairwise_top1_agreement"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
