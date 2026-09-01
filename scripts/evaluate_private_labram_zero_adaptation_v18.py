#!/usr/bin/env python3
"""Run frozen v16 five-fold inference and one-shot private v18 evaluation."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import sys
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soz.fine_temporal_evidence import (  # noqa: E402
    FINE_STRIDE_SECONDS,
    FINE_SUSTAINED_WINDOWS,
    FINE_TEMPORAL_FEATURE_NAMES,
    FINE_WINDOW_SECONDS,
)
from src.soz.geometry import (  # noqa: E402
    CHANNEL_INDEX,
    STANDARD_19,
    TCP_20_EDGES,
)
from src.soz.metrics import DEEPSOZ_STANDARD19_NEIGHBORS  # noqa: E402
from src.soz.v11_reasoner import (  # noqa: E402
    V11_CANDIDATE_MASK,
    robust_pool_complete_patient_bags,
)


SCHEMA = "soz_labram_private_zero_adaptation_evaluation_v18"
PREDICTION_SCHEMA = "soz_labram_private_target_blind_predictions_v18"
DEFAULT_BUNDLE = ROOT / "outputs/labram_private_zero_adaptation_bundle_v18_20260814"
DEFAULT_EVIDENCE = ROOT / "outputs/labram_private_target_blind_evidence_v18_20260814"
DEFAULT_STATES = (
    ROOT
    / "outputs/labram_identity_recovery_closed_replay_v16_20260812/"
    "outer_fold_states.safetensors"
)
DEFAULT_OUTPUT = ROOT / "outputs/labram_private_zero_adaptation_v18_20260814"
ARM = "full_frozen_labram_plus_fine"
N_FOLDS = 5
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260814

BUCKETS: dict[str, tuple[str, ...]] = {
    "left_frontal_scalp": ("FP1", "F3"),
    "right_frontal_scalp": ("FP2", "F4"),
    "left_temporal_chain": ("F7", "T7", "P7"),
    "right_temporal_chain": ("F8", "T8", "P8"),
    "left_parietal_scalp": ("P3",),
    "right_parietal_scalp": ("P4",),
    "occipital_scalp": ("O1", "O2"),
    "central_midline_sensor_composite": ("FZ", "CZ", "PZ", "C3", "C4"),
}
CHANNEL_BUCKET = {
    channel: bucket for bucket, channels in BUCKETS.items() for channel in channels
}
LEFT = {"FP1", "F7", "F3", "T7", "C3", "P7", "P3", "O1"}
RIGHT = {"FP2", "F8", "F4", "T8", "C4", "P8", "P4", "O2"}
MIDLINE = {"FZ", "CZ", "PZ"}


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _read_targets_after_target_blind_outputs(
    bundle: Path, output: Path
) -> list[dict[str, str]]:
    """Fail closed unless prediction and target-blind report files exist.

    This check does not prove operating-system-level access order, but it makes
    the required application-level ordering executable and independently
    testable: the target ledger cannot be opened through the evaluator before
    both target-blind products have been fully closed on disk.
    """

    required = (
        output / "predictions.safetensors",
        output / "prediction_manifest.json",
        output / "structured_reports.jsonl",
    )
    if any(not path.is_file() or path.stat().st_size == 0 for path in required):
        raise RuntimeError(
            "private target ledger cannot open before target-blind outputs"
        )
    return _read_csv(bundle / "target_ledger.csv")


def _as_json_list(value: str) -> tuple[str, ...]:
    parsed = json.loads(value)
    if not isinstance(parsed, list) or any(not isinstance(item, str) for item in parsed):
        raise ValueError("target ledger list field is invalid")
    return tuple(parsed)


def _fold_logits(
    h: torch.Tensor,
    fine: torch.Tensor,
    states: Mapping[str, torch.Tensor],
    fold: int,
) -> torch.Tensor:
    prefix = f"outer{fold}."
    h_center = states[prefix + "transform.h_center"]
    h_scale = states[prefix + "transform.h_scale"]
    h_mean = states[prefix + "transform.h_pca_mean"]
    components = states[prefix + "transform.h_components"]
    fine_center = states[prefix + "transform.fine_center"]
    fine_scale = states[prefix + "transform.fine_scale"]
    arm = prefix + ARM + "."
    if not torch.equal(states[arm + "candidate_mask"], V11_CANDIDATE_MASK):
        raise ValueError("v16 fold candidate mask drifted")
    h_transformed = torch.matmul((h - h_center) / h_scale - h_mean, components)
    fine_transformed = (fine - fine_center) / fine_scale
    logits = states[arm + "prior_logits"].expand(h.shape[0], -1).clone()
    logits += torch.einsum("pcd,d->pc", h_transformed, states[arm + "h_weight"])
    logits += torch.einsum(
        "pcd,d->pc", fine_transformed, states[arm + "fine_weight"]
    )
    if not torch.isfinite(logits).all():
        raise RuntimeError("private fold logits are non-finite")
    return logits


def _fold_probabilities(
    h: torch.Tensor,
    fine: torch.Tensor,
    states: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    rows = []
    for fold in range(N_FOLDS):
        logits = _fold_logits(h, fine, states, fold)
        logits = logits.masked_fill(~V11_CANDIDATE_MASK, -torch.inf)
        probabilities = torch.softmax(logits, dim=1)
        if not torch.isfinite(probabilities).all() or not torch.allclose(
            probabilities.sum(dim=1), torch.ones(h.shape[0]), atol=1e-6, rtol=0
        ):
            raise RuntimeError("private fold probability contract failed")
        rows.append(probabilities)
    return torch.stack(rows, dim=1).contiguous()


def _uncertainty(probabilities: torch.Tensor) -> list[dict[str, object]]:
    mean = probabilities.mean(dim=1)
    candidate = mean[:, V11_CANDIDATE_MASK]
    top2 = candidate.topk(2, dim=1).values
    entropy = -(candidate * candidate.clamp_min(1e-12).log()).sum(dim=1) / math.log(18)
    fold_top = probabilities.argmax(dim=2)
    rows = []
    for index in range(mean.shape[0]):
        votes = torch.bincount(fold_top[index], minlength=19).float()
        votes = votes[V11_CANDIDATE_MASK] / N_FOLDS
        nonzero = votes > 0
        vote_entropy = float(
            (-(votes[nonzero] * votes[nonzero].log()).sum() / math.log(N_FOLDS))
            if int(nonzero.sum()) > 1
            else 0.0
        )
        rows.append(
            {
                "normalized_predictive_entropy": float(entropy[index]),
                "top1_top2_margin": float(top2[index, 0] - top2[index, 1]),
                "fold_top1_vote_entropy": vote_entropy,
                "fold_top1_unique_count": int(nonzero.sum()),
            }
        )
    return rows


def _side(channel: str) -> str:
    if channel in LEFT:
        return "left"
    if channel in RIGHT:
        return "right"
    if channel in MIDLINE:
        return "midline"
    return "indeterminate"


def _band(frequency: float) -> str:
    if frequency < 4.0:
        return "delta"
    if frequency < 8.0:
        return "theta"
    if frequency < 13.0:
        return "alpha"
    if frequency < 30.0:
        return "beta"
    return "high-frequency"


def _target_blind_report(
    event_id: str,
    patient_id: str,
    probability: torch.Tensor,
    uncertainty: Mapping[str, object],
    evidence: Mapping[str, torch.Tensor],
    index: int,
) -> dict[str, object]:
    ranking_indices = torch.argsort(probability, descending=True).tolist()
    ranking = [
        {"channel": STANDARD_19[channel], "probability": float(probability[channel])}
        for channel in ranking_indices[:5]
        if bool(V11_CANDIDATE_MASK[channel])
    ]
    top = ranking[0]["channel"]
    edge_detected = evidence["bipolar_change_detected"][index]
    edge_latency = evidence["bipolar_change_latency_sec"][index]
    detected_edges = torch.nonzero(edge_detected, as_tuple=False).flatten().tolist()
    earliest_edges: list[int] = []
    first_latency: float | None = None
    if detected_edges:
        first_latency = min(float(edge_latency[edge]) for edge in detected_edges)
        earliest_edges = [
            edge for edge in detected_edges if abs(float(edge_latency[edge]) - first_latency) <= 0.25
        ]
    edge_names = [f"{TCP_20_EDGES[edge][0]}-{TCP_20_EDGES[edge][1]}" for edge in earliest_edges]
    rhythm = None
    if earliest_edges and first_latency is not None:
        centers = evidence["window_center_sec"]
        window = (centers >= first_latency) & (centers < first_latency + 1.5)
        endpoint_indices = sorted(
            {
                CHANNEL_INDEX[channel]
                for edge in earliest_edges
                for channel in TCP_20_EDGES[edge]
            }
        )
        values = evidence["dominant_frequency_hz"][index][endpoint_indices][:, window]
        if values.numel() > 0:
            frequency = float(values.median())
            rhythm = {"candidate_band": _band(frequency), "median_frequency_hz": frequency}

    node_detected = evidence["node_change_detected"][index]
    node_latency = evidence["node_change_latency_sec"][index]
    later = []
    if first_latency is not None:
        for channel_index in torch.nonzero(node_detected, as_tuple=False).flatten().tolist():
            delay = float(node_latency[channel_index]) - first_latency
            if delay >= 1.0:
                later.append(
                    {
                        "channel": STANDARD_19[channel_index],
                        "delay_sec": delay,
                        "bucket": CHANNEL_BUCKET[STANDARD_19[channel_index]],
                    }
                )
        later.sort(key=lambda row: (row["delay_sec"], row["channel"]))

    clauses = []
    if first_latency is not None:
        edges = "、".join(edge_names) if edge_names else "若干双极导联"
        interval_end = (
            first_latency
            + FINE_WINDOW_SECONDS
            + (FINE_SUSTAINED_WINDOWS - 1) * FINE_STRIDE_SECONDS
        )
        clauses.append(
            f"固定算法在相对临床事件锚点{first_latency:.2f}--{interval_end:.2f}秒，"
            f"于{edges}检测到持续变化候选"
            f"（连续{FINE_SUSTAINED_WINDOWS}个重叠{FINE_WINDOW_SECONDS:g}秒窗）"
        )
        if rhythm is not None:
            clauses.append(
                f"该候选时段主频中位数约{rhythm['median_frequency_hz']:.1f} Hz，"
                f"属于{rhythm['candidate_band']}频段候选"
            )
    else:
        clauses.append("固定算法未检出满足冻结阈值的持续双极变化候选")
    if later:
        first_later = later[0]
        clauses.append(
            f"约{first_later['delay_sec']:.2f}秒后在{first_later['channel']}出现后续头皮可见变化候选；"
            "该时间差不解释为传播真值"
        )
    clauses.append(
        f"五折模型平均后C18临床参考候选排名首位为{top}，归于"
        f"{CHANNEL_BUCKET[top]}，侧别为{_side(top)}"
    )
    clauses.append(
        "私有EDF未记录原始共同参考；结果仅在19导共享未记录参考并经CAR19消除的假设下成立，"
        "无法评价跨蒙太奇一致性"
    )
    clauses.append("熵、分数间隔和五折分歧均为未校准指标，不表示错误概率")
    clauses.append(
        "本流水线未通过临床伪差亚型资格门，因此不作伪差亚型判断。"
        "结果是头皮电极clinical-reference候选，不等同皮层SOZ；缺少侵入式电生理验证，"
        "不能作为独立手术靶点，需医生复核"
    )
    return {
        "event_id": event_id,
        "patient_id": patient_id,
        "top5": ranking,
        "laterality": _side(top),
        "scalp_sensor_bucket": CHANNEL_BUCKET[top],
        "earliest_algorithmic_bipolar_change_candidate": {
            "latency_sec": first_latency,
            "interval_sec": (
                None
                if first_latency is None
                else [
                    first_latency,
                    first_latency
                    + FINE_WINDOW_SECONDS
                    + (FINE_SUSTAINED_WINDOWS - 1) * FINE_STRIDE_SECONDS,
                ]
            ),
            "edges": edge_names,
            "not_physical_electrode_onset": True,
        },
        "rhythm_candidate": rhythm,
        "later_visible_candidates": later[:5],
        "uncertainty": {
            **dict(uncertainty),
            "calibration_status": "uncalibrated_indicators_not_error_probability",
        },
        "artifact_subtype_status": "unavailable_not_qualified",
        "reference_status": "unlabeled_common_reference_assumption_then_CAR19",
        "report_text_zh": "。".join(clauses) + "。",
    }


def _rank_metrics(
    probability: torch.Tensor,
    positives: Sequence[str],
    spread: Sequence[str],
) -> dict[str, float]:
    positive_indices = {CHANNEL_INDEX[channel] for channel in positives}
    spread_indices = {CHANNEL_INDEX[channel] for channel in spread}
    ranking = [
        index
        for index in torch.argsort(probability, descending=True).tolist()
        if bool(V11_CANDIDATE_MASK[index])
    ]
    top = ranking[0]
    exact = float(top in positive_indices)
    acceptable = set(positive_indices)
    if len(positive_indices) <= 4:
        for index in positive_indices:
            acceptable.update(DEEPSOZ_STANDARD19_NEIGHBORS[index])
    acceptable.difference_update(spread_indices - positive_indices)
    relaxed = float(top in acceptable)
    first_positive_rank = min(ranking.index(index) + 1 for index in positive_indices)
    result = {
        "exact": exact,
        "relaxed_neighbor4": relaxed,
        "hit_at_3": float(bool(set(ranking[:3]) & positive_indices)),
        "hit_at_5": float(bool(set(ranking[:5]) & positive_indices)),
        "mrr": 1.0 / first_positive_rank,
        "positive_recall_at_3": len(set(ranking[:3]) & positive_indices) / len(positive_indices),
        "positive_recall_at_5": len(set(ranking[:5]) & positive_indices) / len(positive_indices),
        "laterality_agreement": float(
            _side(STANDARD_19[top]) in {_side(STANDARD_19[index]) for index in positive_indices}
        ),
        "bucket_agreement": float(
            CHANNEL_BUCKET[STANDARD_19[top]]
            in {CHANNEL_BUCKET[STANDARD_19[index]] for index in positive_indices}
        ),
    }
    comparisons = [
        1.0 if probability[p] > probability[s] else 0.5 if probability[p] == probability[s] else 0.0
        for p in positive_indices
        for s in spread_indices - positive_indices
    ]
    result["significant_over_spread_pairwise"] = (
        float(np.mean(comparisons)) if comparisons else math.nan
    )
    return result


def _aggregate(
    rows: Sequence[Mapping[str, object]], metric_names: Sequence[str]
) -> dict[str, object]:
    patients: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        patients[str(row["patient_id"])].append(row)
    output: dict[str, object] = {
        "event_count": len(rows),
        "patient_count": len(patients),
        "event_micro": {},
        "patient_macro": {},
        "patient_cluster_bootstrap_ci95": {},
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    patient_ids = sorted(patients)
    for metric in metric_names:
        event_values = np.asarray(
            [float(row[metric]) for row in rows if math.isfinite(float(row[metric]))],
            dtype=np.float64,
        )
        patient_values = {
            patient: np.asarray(
                [
                    float(row[metric])
                    for row in patients[patient]
                    if math.isfinite(float(row[metric]))
                ],
                dtype=np.float64,
            )
            for patient in patient_ids
        }
        patient_values = {key: value for key, value in patient_values.items() if value.size}
        if event_values.size == 0 or not patient_values:
            output["event_micro"][metric] = None
            output["patient_macro"][metric] = None
            output["patient_cluster_bootstrap_ci95"][metric] = None
            continue
        patient_means = np.asarray([value.mean() for value in patient_values.values()])
        output["event_micro"][metric] = float(event_values.mean())
        output["patient_macro"][metric] = float(patient_means.mean())
        available = list(patient_values)
        bootstrap = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
        for replicate in range(BOOTSTRAP_REPLICATES):
            sampled = rng.integers(0, len(available), size=len(available))
            bootstrap[replicate] = np.mean(
                [patient_values[available[index]].mean() for index in sampled]
            )
        output["patient_cluster_bootstrap_ci95"][metric] = [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ]
    return output


def run(bundle: Path, evidence_dir: Path, state_path: Path, output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    evidence_manifest = _read_json(evidence_dir / "manifest.json")
    if evidence_manifest.get("schema_version") != "soz_private_target_blind_labram_evidence_v18":
        raise ValueError("private evidence is not the full v18 artifact")
    access = evidence_manifest.get("access_receipt")
    if not isinstance(access, Mapping) or access.get("target_ledger_opened") is not False:
        raise ValueError("private evidence target firewall failed")
    evidence = load_file(str(evidence_dir / str(evidence_manifest["tensor_file"])))
    h_event = evidence["h_event"]
    fine_event = evidence["fine_event"]
    events = evidence_manifest["events"]
    if not isinstance(events, list) or len(events) != h_event.shape[0]:
        raise ValueError("private evidence event identity mismatch")
    event_ids = [str(row["event_id"]) for row in events]
    event_patients = [str(row["patient_id"]) for row in events]
    patient_ids = sorted(set(event_patients))
    patient_index = {patient: index for index, patient in enumerate(patient_ids)}
    event_patient_index = torch.tensor(
        [patient_index[patient] for patient in event_patients], dtype=torch.long
    )
    artifact_index = FINE_TEMPORAL_FEATURE_NAMES.index("artifact_burden_0_12s")
    reliability = (1.0 - fine_event[:, :, artifact_index]).clamp(0.0, 1.0)
    h_patient = robust_pool_complete_patient_bags(
        h_event, event_patient_index, len(patient_ids), reliability
    ).features
    fine_patient = robust_pool_complete_patient_bags(
        fine_event, event_patient_index, len(patient_ids), reliability
    ).features
    states = load_file(str(state_path))
    event_fold_probability = _fold_probabilities(h_event, fine_event, states)
    patient_fold_probability = _fold_probabilities(h_patient, fine_patient, states)
    event_probability = event_fold_probability.mean(dim=1)
    patient_probability = patient_fold_probability.mean(dim=1)
    event_uncertainty = _uncertainty(event_fold_probability)
    patient_uncertainty = _uncertainty(patient_fold_probability)

    output.mkdir(parents=True)
    prediction_tensors = {
        "event_fold_probability": event_fold_probability,
        "event_probability": event_probability,
        "patient_fold_probability": patient_fold_probability,
        "patient_probability": patient_probability,
        "event_patient_index": event_patient_index,
    }
    save_file(prediction_tensors, str(output / "predictions.safetensors"))
    prediction_manifest = {
        "schema_version": PREDICTION_SCHEMA,
        "inference_arm": ARM,
        "fold_ensemble": "equal_mean_of_five_fold_C18_probabilities",
        "event_ids": event_ids,
        "event_patient_ids": event_patients,
        "patient_ids": patient_ids,
        "event_uncertainty": event_uncertainty,
        "patient_uncertainty": patient_uncertainty,
        "access_receipt": {
            "private_evidence_loaded": True,
            "private_target_ledger_loaded_before_predictions": False,
            "private_target_values_used_for_prediction": False,
            "private_model_selection_performed": False,
            "private_calibration_performed": False,
            "training_performed": False,
        },
    }
    (output / "prediction_manifest.json").write_text(
        json.dumps(prediction_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    reports = [
        _target_blind_report(
            event_ids[index],
            event_patients[index],
            event_probability[index],
            event_uncertainty[index],
            evidence,
            index,
        )
        for index in range(len(event_ids))
    ]
    with (output / "structured_reports.jsonl").open("w", encoding="utf-8") as stream:
        for report in reports:
            stream.write(json.dumps(report, ensure_ascii=False) + "\n")

    # First target-ledger read occurs only after predictions and target-blind
    # reports are fully materialized above.
    targets = _read_targets_after_target_blind_outputs(bundle, output)
    target_by_event = {row["event_id"]: row for row in targets}
    if len(target_by_event) != len(targets):
        raise ValueError("private target ledger contains duplicate event IDs")
    evidence_index = {event_id: index for index, event_id in enumerate(event_ids)}
    for event_id, index in evidence_index.items():
        target = target_by_event.get(event_id)
        if target is None or target["patient_id"] != event_patients[index]:
            raise ValueError("private target/evidence event-patient linkage drifted")
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
    evaluations: dict[str, object] = {}
    per_event_rows: list[dict[str, object]] = []
    for cohort, field, expected_events, expected_patients in (
        ("primary", "primary_reference_preeligible", 51, 23),
        ("expanded_anchor_sensitivity", "expanded_reference_preeligible", 60, 24),
    ):
        selected_targets = [
            row for row in targets if row[field] == "1" and row["event_id"] in evidence_index
        ]
        selected_patient_count = len({row["patient_id"] for row in selected_targets})
        if len(selected_targets) != expected_events or selected_patient_count != expected_patients:
            raise RuntimeError(
                f"private {cohort} denominator drifted: "
                f"{len(selected_targets)}/{selected_patient_count}"
            )
        mode_rows: dict[str, list[dict[str, object]]] = {"event": [], "patient_bag": []}
        for target in selected_targets:
            event_index = evidence_index[target["event_id"]]
            patient = target["patient_id"]
            positive = _as_json_list(target["candidate_positive_electrodes"])
            spread = _as_json_list(target["known_spread_electrodes"])
            for mode, probability in (
                ("event", event_probability[event_index]),
                ("patient_bag", patient_probability[patient_index[patient]]),
            ):
                metrics = _rank_metrics(probability, positive, spread)
                row = {
                    "cohort": cohort,
                    "mode": mode,
                    "event_id": target["event_id"],
                    "patient_id": patient,
                    "positive_set": list(positive),
                    "spread_set": list(spread),
                    "top1": STANDARD_19[int(probability.argmax())],
                    **metrics,
                }
                mode_rows[mode].append(row)
                per_event_rows.append(row)
        evaluations[cohort] = {
            mode: _aggregate(rows, metric_names) for mode, rows in mode_rows.items()
        }

    with (output / "evaluation_rows.jsonl").open("w", encoding="utf-8") as stream:
        for row in per_event_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    primary_event = evaluations["primary"]["event"]
    result: dict[str, object] = {
        "schema_version": SCHEMA,
        "model": {
            "backbone": "official_pretrained_LaBraM_Base",
            "foundation_trained_from_scratch": False,
            "inference_arm": ARM,
            "fold_ensemble": "equal_probability_mean_5",
            "candidate_space": [
                channel for channel, allowed in zip(STANDARD_19, V11_CANDIDATE_MASK) if bool(allowed)
            ],
        },
        "denominators": {
            "source_events": 123,
            "source_patients": 43,
            "target_blind_time_supported_events": 94,
            "target_blind_signal_eligible_events": len(event_ids),
            "primary_events": 51,
            "primary_patients": 23,
            "expanded_events": 60,
            "expanded_patients": 24,
        },
        "evaluations": evaluations,
        "goal_audit": {
            "primary_exact_event_micro_target": 0.80,
            "primary_exact_event_micro_achieved": (
                primary_event["event_micro"]["exact"] >= 0.80
            ),
            "primary_relaxed_event_micro_target": 0.85,
            "primary_relaxed_event_micro_achieved": (
                primary_event["event_micro"]["relaxed_neighbor4"] >= 0.85
            ),
            "primary_exact_patient_macro_target": 0.80,
            "primary_exact_patient_macro_achieved": (
                primary_event["patient_macro"]["exact"] >= 0.80
            ),
            "primary_relaxed_patient_macro_target": 0.85,
            "primary_relaxed_patient_macro_achieved": (
                primary_event["patient_macro"]["relaxed_neighbor4"] >= 0.85
            ),
            "targets_are_goals_not_selection_rules": True,
        },
        "reporting": {
            "structured_report_count": len(reports),
            "llm_used_as_soz_predictor": False,
            "artifact_subtype_reported": False,
            "cortical_soz_claimed": False,
            "surgical_target_claimed": False,
        },
        "claim_boundary": {
            "private_used_for_training": False,
            "private_used_for_model_selection": False,
            "private_used_for_calibration": False,
            "private_historically_reused": True,
            "external_confirmatory_validation": False,
            "retrospective_zero_adaptation_transfer": True,
        },
    }
    (output / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--states", type=Path, default=DEFAULT_STATES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.bundle, args.evidence, args.states, args.output)
    primary = result["evaluations"]["primary"]["event"]
    print(
        json.dumps(
            {
                "denominators": result["denominators"],
                "primary_event_micro": primary["event_micro"],
                "primary_patient_macro": primary["patient_macro"],
                "goal_audit": result["goal_audit"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
