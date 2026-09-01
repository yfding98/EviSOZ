#!/usr/bin/env python3
"""Evaluate a frozen private common17 prediction against post-freeze labels.

The prediction directory is fully validated and content-hashed before the
doctor-label release is opened.  FZ/PZ are mapped to CZ on the target side
only.  Significant electrodes are the hard endpoint; spread electrodes remain
an independent soft endpoint and never become hard positives.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
from safetensors.torch import load_file
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.sota_soz.train_common17_oracle_event_oof_v1 import (  # noqa: E402
    induced_common17_neighbors,
)
from src.soz.geometry import CHANNEL_INDEX, STANDARD_19  # noqa: E402


SCHEMA = "private_common17_literal_postfreeze_evaluation_v1"
PREDICTION_SCHEMA = "private_common17_literal_public_fold_ensemble_v1"
COMMON17 = tuple(channel for channel in STANDARD_19 if channel not in {"FZ", "PZ"})
DEFAULT_PREDICTION = ROOT / "outputs/private_common17_literal_zero_shot_v1_20260825"
DEFAULT_LABELS = ROOT / "outputs/private_clinical_eeg_doctor_labels_postfreeze_v2_3_20260820.json"
DEFAULT_OUTPUT = ROOT / "outputs/private_common17_literal_postfreeze_evaluation_v1_20260825"
METRICS = ("exact_top1", "deepsoz_N2", "deepsoz_N4", "hit_at_3", "hit_at_5", "mrr")
SOFT_METRICS = ("exact_top1", "hit_at_3", "hit_at_5", "mrr")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve(strict=True).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _map_target(values: Sequence[object]) -> tuple[set[str], set[str], set[str]]:
    raw = {str(value).strip().upper() for value in values if str(value).strip()}
    standard = raw & set(STANDARD_19)
    mapped = set(standard)
    if mapped & {"FZ", "PZ"}:
        mapped.add("CZ")
    mapped -= {"FZ", "PZ"}
    outside = raw - set(STANDARD_19)
    if not mapped <= set(COMMON17):
        raise RuntimeError("target mapping left the common17 space")
    return mapped, standard, outside


def _wilson(success: float, denominator: int) -> dict[str, float | str] | None:
    if denominator < 1:
        return None
    z = 1.959963984540054
    p = float(success) / denominator
    scale = 1.0 + z * z / denominator
    center = (p + z * z / (2.0 * denominator)) / scale
    radius = z * math.sqrt(p * (1.0 - p) / denominator + z * z / (4 * denominator**2)) / scale
    return {
        "lower": max(0.0, center - radius),
        "upper": min(1.0, center + radius),
        "method": "wilson_score_two_sided_95_percent",
    }


def _metric_atoms(
    probability: torch.Tensor,
    hard: set[str],
    soft_spread: set[str],
    pre_mapping_hard_count: int,
    graph: Sequence[Sequence[int]],
) -> tuple[dict[str, float], list[str]]:
    if tuple(probability.shape) != (17,) or not torch.isfinite(probability).all():
        raise ValueError("metric probability must be finite common17")
    hard_indices = {COMMON17.index(value) for value in hard}
    soft_indices = {COMMON17.index(value) for value in soft_spread}
    if not hard_indices:
        raise ValueError("hard metric requires at least one positive")
    order = torch.argsort(probability, descending=True, stable=True).tolist()
    ranking = [COMMON17[index] for index in order]
    top = float(probability.max())
    tied = [index for index in range(17) if float(probability[index]) == top]
    exact = sum(index in hard_indices for index in tied) / len(tied)
    relaxed: dict[int, float] = {}
    for gate in (2, 4):
        acceptable = set(hard_indices)
        if pre_mapping_hard_count <= gate:
            for index in hard_indices:
                acceptable.update(graph[index])
            acceptable -= soft_indices
            acceptable |= hard_indices
        relaxed[gate] = sum(index in acceptable for index in tied) / len(tied)
    first = min(order.index(index) for index in hard_indices)
    return {
        "exact_top1": float(exact),
        "deepsoz_N2": float(relaxed[2]),
        "deepsoz_N4": float(relaxed[4]),
        "hit_at_3": float(any(index in hard_indices for index in order[:3])),
        "hit_at_5": float(any(index in hard_indices for index in order[:5])),
        "mrr": 1.0 / (first + 1.0),
    }, ranking


def _soft_atoms(probability: torch.Tensor, soft: set[str]) -> dict[str, float]:
    indices = {COMMON17.index(value) for value in soft}
    if not indices:
        raise ValueError("soft metric requires at least one positive")
    order = torch.argsort(probability, descending=True, stable=True).tolist()
    top = float(probability.max())
    tied = [index for index in range(17) if float(probability[index]) == top]
    first = min(order.index(index) for index in indices)
    return {
        "exact_top1": sum(index in indices for index in tied) / len(tied),
        "hit_at_3": float(any(index in indices for index in order[:3])),
        "hit_at_5": float(any(index in indices for index in order[:5])),
        "mrr": 1.0 / (first + 1.0),
    }


def _cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    metrics: Sequence[str],
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["patient_pseudonym"])].append(row)
    patients = sorted(grouped)
    if not patients:
        return {"replicates": replicates, "patient_clusters": 0, "metrics": {}}
    patient_sum = {
        metric: np.asarray(
            [sum(float(row["atoms"][metric]) for row in grouped[patient]) for patient in patients],
            dtype=np.float64,
        )
        for metric in metrics
    }
    patient_count = np.asarray([len(grouped[patient]) for patient in patients], dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(patients), size=(replicates, len(patients)))
    output: dict[str, Any] = {}
    denominator = patient_count[samples].sum(axis=1)
    for metric in metrics:
        micro = patient_sum[metric][samples].sum(axis=1) / denominator
        macro = (patient_sum[metric] / patient_count)[samples].mean(axis=1)
        output[metric] = {
            "record_micro_ci95": [float(np.quantile(micro, 0.025)), float(np.quantile(micro, 0.975))],
            "patient_macro_ci95": [float(np.quantile(macro, 0.025)), float(np.quantile(macro, 0.975))],
        }
    return {
        "method": "patient_cluster_nonparametric_bootstrap_two_sided_95_percent",
        "seed": seed,
        "replicates": replicates,
        "patient_clusters": len(patients),
        "metrics": output,
    }


def _summary(
    rows: Sequence[Mapping[str, Any]],
    metrics: Sequence[str],
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    if not rows:
        return {"denominator": 0}
    per_patient: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        per_patient[str(row["patient_pseudonym"])].append(row)
    result: dict[str, Any] = {
        "denominator": len(rows),
        "patient_denominator": len(per_patient),
        "metrics": {},
    }
    for metric in metrics:
        values = [float(row["atoms"][metric]) for row in rows]
        mean = float(np.mean(values))
        patient_macro = float(
            np.mean(
                [
                    np.mean([float(row["atoms"][metric]) for row in patient_rows])
                    for patient_rows in per_patient.values()
                ]
            )
        )
        payload: dict[str, Any] = {
            "value": mean,
            "sum": float(sum(values)),
            "patient_macro": patient_macro,
        }
        if all(value in (0.0, 1.0) for value in values):
            payload["correct_count"] = int(sum(values))
            payload["wilson_95_ci"] = _wilson(sum(values), len(values))
        result["metrics"][metric] = payload
    result["patient_cluster_bootstrap_ci95"] = _cluster_bootstrap(
        rows,
        metrics,
        seed=seed,
        replicates=replicates,
    )
    return result


def _forced_conditional(
    rows: Sequence[Mapping[str, Any]],
    metrics: Sequence[str],
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    conditional = [row for row in rows if row["prediction_available"]]
    return {
        "forced_full_GT_denominator": _summary(rows, metrics, seed=seed, replicates=replicates),
        "conditional_on_prediction": _summary(
            conditional,
            metrics,
            seed=seed + 1,
            replicates=replicates,
        ),
        "missing_prediction_count": len(rows) - len(conditional),
    }


def run(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    prediction_root = args.predictions.resolve(strict=True)
    prediction_manifest_path = prediction_root / "manifest.json"
    prediction_receipt_path = prediction_root / "receipt.json"
    prediction = _read_json(prediction_manifest_path)
    receipt = _read_json(prediction_receipt_path)
    if prediction.get("schema_version") != PREDICTION_SCHEMA:
        raise ValueError("wrong private common17 prediction schema")
    if prediction.get("status") != "completed_target_blind_private_common17_external_fold_ensemble":
        raise ValueError("private common17 prediction is not frozen and complete")
    access = prediction.get("access_receipt", {})
    if (
        access.get("private_target_values_loaded") is not False
        or access.get("doctor_label_release_loaded") is not False
        or access.get("private_training_or_parameter_fitting_performed") is not False
    ):
        raise RuntimeError("private prediction violated the target-blind boundary")
    tensor_path = prediction_root / str(prediction["tensor_file"])
    prediction_hashes = {
        "manifest_sha256": _sha256(prediction_manifest_path),
        "receipt_sha256": _sha256(prediction_receipt_path),
        "tensor_sha256": _sha256(tensor_path),
    }
    if prediction_hashes["tensor_sha256"] != prediction.get("tensor_sha256"):
        raise RuntimeError("prediction tensor hash mismatch")
    if prediction_hashes["manifest_sha256"] != receipt.get("manifest_sha256"):
        raise RuntimeError("prediction manifest hash mismatch")
    if prediction_hashes["tensor_sha256"] != receipt.get("tensor_sha256"):
        raise RuntimeError("prediction receipt tensor hash mismatch")
    records = prediction.get("records")
    patients = prediction.get("patients")
    if not isinstance(records, list) or len(records) != 141:
        raise ValueError("prediction record roster changed")
    if not isinstance(patients, list) or len(patients) != 45:
        raise ValueError("prediction patient roster changed")
    tensors = load_file(str(tensor_path), device="cpu")
    record_probability = tensors["record_probability"].float()
    record_mask = tensors["record_prediction_mask"].bool()
    patient_probability = tensors["patient_probability"].float()
    patient_mask = tensors["patient_prediction_mask"].bool()
    if tuple(record_probability.shape) != (141, 17) or tuple(record_mask.shape) != (141,):
        raise ValueError("record prediction tensor shape changed")
    if tuple(patient_probability.shape) != (45, 17) or tuple(patient_mask.shape) != (45,):
        raise ValueError("patient prediction tensor shape changed")
    prediction_snapshot = {
        "prediction_roster_sha256": prediction["prediction_roster_sha256"],
        "prediction_hashes": prediction_hashes,
        "record_ids": [str(row["recording_id"]) for row in records],
        "patient_ids": [str(row["patient_pseudonym"]) for row in patients],
        "record_prediction_mask": record_mask.tolist(),
        "patient_prediction_mask": patient_mask.tolist(),
    }
    prediction_snapshot_sha256 = _canonical_sha256(prediction_snapshot)

    # Target access begins only after the immutable prediction snapshot above.
    label_path = args.doctor_label_release.resolve(strict=True)
    labels = _read_json(label_path)
    label_records = labels.get("records")
    if not isinstance(label_records, list) or len(label_records) != 141:
        raise ValueError("doctor-label release no longer contains 141 records")
    label_by_id = {str(row["recording_id"]): row for row in label_records}
    if set(label_by_id) != set(prediction_snapshot["record_ids"]):
        raise ValueError("prediction and doctor-label record rosters differ")

    common_indices = torch.tensor([CHANNEL_INDEX[value] for value in COMMON17], dtype=torch.long)
    graph = induced_common17_neighbors(common_indices)
    record_eval_rows: list[dict[str, Any]] = []
    hard_rows: list[dict[str, Any]] = []
    soft_rows: list[dict[str, Any]] = []
    patient_hard_raw: dict[str, set[str]] = defaultdict(set)
    patient_hard: dict[str, set[str]] = defaultdict(set)
    patient_soft_raw: dict[str, set[str]] = defaultdict(set)
    patient_outside: dict[str, set[str]] = defaultdict(set)
    patient_excluded_count: Counter[str] = Counter()
    hard_spread_overlap_record_count = 0
    for index, prediction_row in enumerate(records):
        recording_id = str(prediction_row["recording_id"])
        patient = str(prediction_row["patient_pseudonym"])
        label_row = label_by_id[recording_id]
        eligible = [
            row
            for row in label_row.get("doctor_labels", [])
            if row.get("evaluation_eligible") is True
        ]
        hard_projected: set[str] = set()
        hard_standard: set[str] = set()
        hard_outside: set[str] = set()
        spread_projected: set[str] = set()
        spread_standard: set[str] = set()
        spread_outside: set[str] = set()
        excluded_significant_tokens = 0
        excluded_spread_tokens = 0
        for row in eligible:
            channel = row.get("physician_channel_reference", {})
            mapped, standard, outside = _map_target(channel.get("significant_electrodes", []))
            hard_projected |= mapped
            hard_standard |= standard
            hard_outside |= outside
            mapped, standard, outside = _map_target(channel.get("spread_electrodes", []))
            spread_projected |= mapped
            spread_standard |= standard
            spread_outside |= outside
            excluded_significant_tokens += int(channel.get("excluded_out_of_scope_significant_token_count", 0))
            excluded_spread_tokens += int(channel.get("excluded_out_of_scope_spread_token_count", 0))
        overlap = hard_projected & spread_projected
        if overlap:
            hard_spread_overlap_record_count += 1
        soft_projected = spread_projected - hard_projected
        available = bool(record_mask[index])
        probability = record_probability[index]
        ranking = (
            [COMMON17[value] for value in torch.argsort(probability, descending=True, stable=True).tolist()]
            if available
            else []
        )
        row_out: dict[str, Any] = {
            "recording_id": recording_id,
            "patient_pseudonym": patient,
            "prediction_available": available,
            "prediction_status": prediction_row["prediction_status"],
            "ranked_electrodes": ranking,
            "eligible_doctor_label_count": len(eligible),
            "hard_significant_common17": sorted(hard_projected),
            "soft_spread_common17": sorted(soft_projected),
            "pre_mapping_hard_standard19_count": len(hard_standard),
            "pre_mapping_spread_standard19_count": len(spread_standard),
            "projected_model_outside_significant_electrodes": sorted(hard_outside),
            "projected_model_outside_spread_electrodes": sorted(spread_outside),
            "excluded_out_of_scope_significant_token_count": excluded_significant_tokens,
            "excluded_out_of_scope_spread_token_count": excluded_spread_tokens,
            "hard_spread_overlap_mapped_common17": sorted(overlap),
        }
        if hard_projected:
            atoms, ranking_check = _metric_atoms(
                probability,
                hard_projected,
                soft_projected,
                len(hard_standard),
                graph,
            ) if available else ({metric: 0.0 for metric in METRICS}, [])
            if available and ranking_check != ranking:
                raise RuntimeError("record ranking implementations disagree")
            hard_metric_row = {
                "recording_id": recording_id,
                "patient_pseudonym": patient,
                "prediction_available": available,
                "atoms": atoms,
            }
            hard_rows.append(hard_metric_row)
            row_out["hard_metric_atoms"] = atoms
            patient_hard_raw[patient] |= hard_standard
            patient_hard[patient] |= hard_projected
            patient_outside[patient] |= hard_outside
            patient_excluded_count[patient] += excluded_significant_tokens
        else:
            row_out["hard_metric_atoms"] = None
        if soft_projected:
            atoms = _soft_atoms(probability, soft_projected) if available else {
                metric: 0.0 for metric in SOFT_METRICS
            }
            soft_rows.append(
                {
                    "recording_id": recording_id,
                    "patient_pseudonym": patient,
                    "prediction_available": available,
                    "atoms": atoms,
                }
            )
            row_out["soft_metric_atoms"] = atoms
            patient_soft_raw[patient] |= spread_standard
        else:
            row_out["soft_metric_atoms"] = None
        record_eval_rows.append(row_out)

    if len(hard_rows) != 92 or len({row["patient_pseudonym"] for row in hard_rows}) != 34:
        raise RuntimeError("private common17 hard GT denominator drifted")
    if len(soft_rows) != 97:
        raise RuntimeError("private common17 soft-spread denominator drifted")
    patient_axis = {str(row["patient_pseudonym"]): int(row["patient_index"]) for row in patients}
    patient_eval_rows: list[dict[str, Any]] = []
    patient_metric_rows: list[dict[str, Any]] = []
    for patient in sorted(patient_hard):
        index = patient_axis[patient]
        hard = patient_hard[patient]
        soft, _, _ = _map_target(patient_soft_raw.get(patient, set()))
        soft -= hard
        available = bool(patient_mask[index])
        probability = patient_probability[index]
        atoms, ranking = _metric_atoms(
            probability,
            hard,
            soft,
            len(patient_hard_raw[patient]),
            graph,
        ) if available else ({metric: 0.0 for metric in METRICS}, [])
        metric_row = {
            "recording_id": f"PATIENT::{patient}",
            "patient_pseudonym": patient,
            "prediction_available": available,
            "atoms": atoms,
        }
        patient_metric_rows.append(metric_row)
        patient_eval_rows.append(
            {
                "patient_pseudonym": patient,
                "prediction_available": available,
                "ranked_electrodes": ranking,
                "hard_significant_common17_union": sorted(hard),
                "soft_spread_common17_union": sorted(soft),
                "pre_mapping_hard_standard19_count": len(patient_hard_raw[patient]),
                "projected_model_outside_significant_electrodes": sorted(patient_outside[patient]),
                "excluded_out_of_scope_significant_token_count": patient_excluded_count[patient],
                "metric_atoms": atoms,
            }
        )

    record_metrics = _forced_conditional(
        hard_rows,
        METRICS,
        seed=args.bootstrap_seed,
        replicates=args.bootstrap_replicates,
    )
    record_metrics["deepsoz_gate_eligibility"] = {
        "N2_records": sum(row["pre_mapping_hard_standard19_count"] <= 2 for row in record_eval_rows if row["hard_metric_atoms"] is not None),
        "N4_records": sum(row["pre_mapping_hard_standard19_count"] <= 4 for row in record_eval_rows if row["hard_metric_atoms"] is not None),
        "gate_definition": "mapping_before_distinct_standard19_hard_positive_count",
    }
    patient_metrics = _forced_conditional(
        patient_metric_rows,
        METRICS,
        seed=args.bootstrap_seed + 10,
        replicates=args.bootstrap_replicates,
    )
    patient_metrics["deepsoz_gate_eligibility"] = {
        "N2_patients": sum(row["pre_mapping_hard_standard19_count"] <= 2 for row in patient_eval_rows),
        "N4_patients": sum(row["pre_mapping_hard_standard19_count"] <= 4 for row in patient_eval_rows),
    }
    soft_metrics = _forced_conditional(
        soft_rows,
        SOFT_METRICS,
        seed=args.bootstrap_seed + 20,
        replicates=args.bootstrap_replicates,
    )

    top1_distribution = Counter()
    for index, available in enumerate(record_mask.tolist()):
        if available:
            top = int(torch.argmax(record_probability[index]))
            top1_distribution[COMMON17[top]] += 1
    partial_outside = sum(
        row["hard_metric_atoms"] is not None
        and (
            row["excluded_out_of_scope_significant_token_count"] > 0
            or bool(row["projected_model_outside_significant_electrodes"])
        )
        for row in record_eval_rows
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "completed_prediction_first_postfreeze_private_common17_evaluation",
        "evaluation_unit": {
            "primary": "one_unique_long_EEG_recording_after_multi_event_model_pooling",
            "secondary": "one_patient_after_pooling_all_EEG_only_events_across_records",
            "event_accuracy_reported": False,
            "event_reason": "doctor record/patient labels are not independent event-level labels",
        },
        "prediction_first_gate": {
            "prediction_manifest_tensor_and_receipt_validated_before_label_open": True,
            "prediction_snapshot_sha256": prediction_snapshot_sha256,
            "prediction_hashes": prediction_hashes,
            "prediction_roster_sha256": prediction["prediction_roster_sha256"],
            "threshold_weight_window_or_ranking_changed_after_label_open": False,
        },
        "coverage": {
            "records": 141,
            "patients": 45,
            "records_with_common17_prediction": int(record_mask.sum()),
            "records_without_prediction": int((~record_mask).sum()),
            "patients_with_common17_prediction": int(patient_mask.sum()),
            "EEG_only_detector_events": int(tensors["event_probability"].shape[0]),
            "hard_GT_records": len(hard_rows),
            "hard_GT_patients": len(patient_metric_rows),
            "hard_GT_records_with_prediction": sum(row["prediction_available"] for row in hard_rows),
            "soft_spread_records": len(soft_rows),
            "soft_spread_records_with_prediction": sum(row["prediction_available"] for row in soft_rows),
        },
        "target_contract": {
            "GT_only_FZ_or_PZ_to_CZ": True,
            "prediction_side_score_mapping": False,
            "hard_endpoint": "union_of_evaluation_eligible_significant_electrodes_within_common17_after_mapping",
            "soft_endpoint": "mapped_spread_minus_mapped_hard_scored_separately",
            "unlisted_electrodes_are_negative": False,
            "hard_spread_overlap_records_hard_priority": hard_spread_overlap_record_count,
            "partial_out_of_model_scope_hard_GT_record_count": partial_outside,
            "partial_scope_warning": "metric is membership in known common17 significant electrodes, not complete clinical SOZ accuracy",
        },
        "metrics": {
            "record_level_hard_significant": record_metrics,
            "patient_level_hard_significant_union": patient_metrics,
            "record_level_soft_spread_separate_endpoint": soft_metrics,
        },
        "diagnostics": {
            "predicted_record_top1_distribution": dict(sorted(top1_distribution.items())),
            "common17_channels": list(COMMON17),
        },
        "metric_contract": {
            "accuracy_equals_exact_top1": True,
            "N2_N4_are_not_top2_or_top4": True,
            "N2_N4_graph": "DeepSOZ_STANDARD19_one_hop_induced_by_deleting_FZ_PZ_nodes_and_edges_no_bridge",
            "known_soft_spread_removed_from_neighbor_acceptable_set": True,
            "all_hard_GT_rows_remain_in_N2_N4_denominator": True,
            "MRR_uses_full_common17_ranking": True,
            "missing_prediction_for_applicable_GT_scores_zero_in_forced_denominator": True,
        },
        "access_receipt": {
            "prediction_frozen_before_doctor_label_release_open": True,
            "doctor_label_release_opened_for_evaluation": True,
            "raw_Excel_or_EDF_annotations_opened": False,
            "doctor_labels_used_for_training_calibration_threshold_window_or_model_selection": False,
            "private_result_is_fresh_confirmatory_external_test": False,
            "private_result_role": "post_open_retrospective_external_transfer_evaluation",
        },
        "lineage": {
            "prediction_manifest_sha256": prediction_hashes["manifest_sha256"],
            "prediction_tensor_sha256": prediction_hashes["tensor_sha256"],
            "doctor_label_release_sha256": _sha256(label_path),
        },
    }
    payload["content_sha256"] = _canonical_sha256(payload)
    return payload, record_eval_rows, patient_eval_rows


def publish(
    output: Path,
    payload: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    patients: Sequence[Mapping[str, Any]],
) -> Path:
    target = output.resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        result_path = staging / "result.json"
        result_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        with (staging / "record_rows.jsonl").open("w", encoding="utf-8") as handle:
            for row in records:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        with (staging / "patient_rows.jsonl").open("w", encoding="utf-8") as handle:
            for row in patients:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        receipt = {
            "schema_version": f"{SCHEMA}_receipt",
            "status": payload["status"],
            "content_sha256": payload["content_sha256"],
            "result_sha256": _sha256(result_path),
            "record_rows_sha256": _sha256(staging / "record_rows.jsonl"),
            "patient_rows_sha256": _sha256(staging / "patient_rows.jsonl"),
        }
        (staging / "receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, target)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTION)
    parser.add_argument("--doctor-label-release", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-seed", type=int, default=20260825)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.bootstrap_replicates < 1_000:
        raise ValueError("bootstrap-replicates must be at least 1000")
    payload, records, patients = run(args)
    output = publish(args.output, payload, records, patients)
    hard = payload["metrics"]["record_level_hard_significant"]
    patient = payload["metrics"]["patient_level_hard_significant_union"]
    print(
        json.dumps(
            {
                "output": str(output),
                "record_forced_exact": hard["forced_full_GT_denominator"]["metrics"]["exact_top1"]["value"],
                "record_forced_N4": hard["forced_full_GT_denominator"]["metrics"]["deepsoz_N4"]["value"],
                "patient_forced_exact": patient["forced_full_GT_denominator"]["metrics"]["exact_top1"]["value"],
                "patient_forced_N4": patient["forced_full_GT_denominator"]["metrics"]["deepsoz_N4"]["value"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
