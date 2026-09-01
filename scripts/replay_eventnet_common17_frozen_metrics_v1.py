#!/usr/bin/env python3
"""Independently replay EventNet common17 metrics from frozen prediction payloads.

This script does not import torch, load a checkpoint, read EDF samples, or run
inference.  It reconstructs the threshold decoder and ordered one-to-one event
matching from prediction peaks plus the already-frozen manifest reference
sidecar, then writes a content-addressed replay receipt.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "outputs/eventnet_common17_streaming_v1_20260824/manifest.json"
DEFAULT_PREDICTIONS = PROJECT_ROOT / "outputs/eventnet_common17_streaming_bf16_contract_v2_20260824/source_dev_full_global_v3/predictions"
DEFAULT_FROZEN_METRICS = PROJECT_ROOT / "outputs/eventnet_common17_streaming_bf16_contract_v2_20260824/source_dev_full_global_v3/metrics.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/eventnet_common17_user_requested_metric_replay_v1_20260825/receipt.json"
EXPECTED_MANIFEST_SHA256 = "15d5d4bb115ba38b4168a1a4035d990a80f63543c194341e381f19fc53112c62"
EXPECTED_CHECKPOINT_SHA256 = "3057f991e07fb6aa73d18a6ef2f6798c6598e54f7f6c5249c3958ff50fbc8ee0"
EXPECTED_PREDICTION_SCHEMA = "eventnet_common17_dev_prediction_global_posterior_runtime_v3"
COMMON17 = (
    "FP1", "F3", "C3", "P3", "O1", "F7", "T7", "P7", "CZ",
    "FP2", "F4", "C4", "P4", "O2", "F8", "T8", "P8",
)
TARGET_FS_HZ = 256.0
MAXIMUM_DURATION_SECONDS = 300.0
THRESHOLD = 0.02
TOLERANCES = (1.0, 3.0, 5.0, 10.0)


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_address(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    result["receipt_sha256"] = hashlib.sha256(canonical_bytes(result)).hexdigest()
    return result


def decode_peaks(
    peaks: Sequence[Mapping[str, object]],
    *,
    recording_duration_seconds: float,
    threshold: float = THRESHOLD,
) -> list[dict[str, float]]:
    events: list[dict[str, float]] = []
    for peak in peaks:
        probability = float(peak["center_probability"])
        if probability < threshold:
            continue
        duration = min(
            MAXIMUM_DURATION_SECONDS,
            max(1.0 / TARGET_FS_HZ, float(peak["duration_fraction"]) * MAXIMUM_DURATION_SECONDS),
        )
        center = float(peak["center_seconds"])
        if not all(math.isfinite(value) for value in (probability, duration, center)):
            raise ValueError("prediction peak contains a non-finite value")
        start = max(0.0, center - duration / 2.0)
        stop = min(recording_duration_seconds, center + duration / 2.0)
        if stop > start:
            events.append({"start_seconds": start, "stop_seconds": stop})
    merged: list[dict[str, float]] = []
    for event in sorted(events, key=lambda row: (row["start_seconds"], row["stop_seconds"])):
        if not merged or event["start_seconds"] > merged[-1]["stop_seconds"]:
            merged.append(dict(event))
        else:
            merged[-1]["stop_seconds"] = max(merged[-1]["stop_seconds"], event["stop_seconds"])
    return merged


def iou(reference: Mapping[str, float], prediction: Mapping[str, float]) -> float:
    intersection = max(
        0.0,
        min(float(reference["stop_seconds"]), float(prediction["stop_seconds"]))
        - max(float(reference["start_seconds"]), float(prediction["start_seconds"])),
    )
    if intersection <= 0:
        return 0.0
    union = max(float(reference["stop_seconds"]), float(prediction["stop_seconds"])) - min(
        float(reference["start_seconds"]), float(prediction["start_seconds"])
    )
    return intersection / union


def ordered_match(
    references: Sequence[Mapping[str, float]], predictions: Sequence[Mapping[str, float]]
) -> list[tuple[int, int, float]]:
    """Fresh DP: maximize matches, then IoU, then minimize onset error."""

    n_ref, n_pred = len(references), len(predictions)
    scores = [[(0, 0.0, 0.0) for _ in range(n_pred + 1)] for _ in range(n_ref + 1)]
    parents: list[list[tuple[int, int, str] | None]] = [
        [None for _ in range(n_pred + 1)] for _ in range(n_ref + 1)
    ]
    for ref_index in range(1, n_ref + 1):
        parents[ref_index][0] = (ref_index - 1, 0, "skip_reference")
    for pred_index in range(1, n_pred + 1):
        parents[0][pred_index] = (0, pred_index - 1, "skip_prediction")
    action_priority = {"skip_reference": 0, "skip_prediction": 1, "match": 2}
    for ref_index in range(1, n_ref + 1):
        for pred_index in range(1, n_pred + 1):
            candidates = [
                (scores[ref_index - 1][pred_index], (ref_index - 1, pred_index, "skip_reference")),
                (scores[ref_index][pred_index - 1], (ref_index, pred_index - 1, "skip_prediction")),
            ]
            overlap = iou(references[ref_index - 1], predictions[pred_index - 1])
            if overlap > 0:
                previous = scores[ref_index - 1][pred_index - 1]
                onset_error = abs(
                    float(predictions[pred_index - 1]["start_seconds"])
                    - float(references[ref_index - 1]["start_seconds"])
                )
                candidates.append(
                    (
                        (previous[0] + 1, previous[1] + overlap, previous[2] - onset_error),
                        (ref_index - 1, pred_index - 1, "match"),
                    )
                )
            best_score, best_parent = max(
                candidates, key=lambda item: (item[0], action_priority[item[1][2]])
            )
            scores[ref_index][pred_index] = best_score
            parents[ref_index][pred_index] = best_parent
    matches: list[tuple[int, int, float]] = []
    ref_index, pred_index = n_ref, n_pred
    while ref_index or pred_index:
        parent = parents[ref_index][pred_index]
        if parent is None:
            raise RuntimeError("event matching backtrace is incomplete")
        previous_ref, previous_pred, action = parent
        if action == "match":
            matches.append(
                (ref_index - 1, pred_index - 1, iou(references[ref_index - 1], predictions[pred_index - 1]))
            )
        ref_index, pred_index = previous_ref, previous_pred
    matches.reverse()
    return matches


def replay(
    *,
    manifest_path: Path,
    prediction_dir: Path,
    frozen_metrics_path: Path,
    expected_checkpoint_sha256: str = EXPECTED_CHECKPOINT_SHA256,
    threshold: float = THRESHOLD,
) -> dict[str, Any]:
    if len(expected_checkpoint_sha256) != 64:
        raise ValueError("expected checkpoint SHA-256 is invalid")
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("replay threshold must be finite and in [0, 1]")
    if file_sha256(manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("manifest SHA-256 does not match the frozen common17 source")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = [row for row in manifest["records"] if row["model_split"] == "source_dev"]
    if len(records) != 1821:
        raise ValueError(f"expected 1821 source-dev records, got {len(records)}")
    record_by_id = {str(row["analysis_identity_id"]): row for row in records}
    if len(record_by_id) != len(records):
        raise ValueError("manifest analysis identities are not unique")
    payload_by_id: dict[str, dict[str, Any]] = {}
    inventory_rows: list[dict[str, str]] = []
    for path in sorted(prediction_dir.glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        identity = str(payload.get("analysis_identity_id"))
        if identity not in record_by_id or identity in payload_by_id:
            raise ValueError(f"prediction identity is extra or duplicated: {identity}")
        if payload.get("schema_version") != EXPECTED_PREDICTION_SCHEMA:
            raise ValueError(f"prediction schema mismatch: {identity}")
        if payload.get("checkpoint_file_sha256") != expected_checkpoint_sha256:
            raise ValueError(f"checkpoint binding mismatch: {identity}")
        if tuple(payload.get("common17_channel_order", [])) != COMMON17:
            raise ValueError(f"common17 channel binding mismatch: {identity}")
        if payload.get("FZ_or_PZ_model_axis_present") is not False:
            raise ValueError(f"FZ/PZ axis leaked into prediction: {identity}")
        if float(payload.get("minimum_peak_threshold")) > threshold:
            raise ValueError(f"payload peak floor exceeds replay threshold: {identity}")
        payload_by_id[identity] = payload
        inventory_rows.append({"analysis_identity_id": identity, "file_sha256": file_sha256(path)})
    if set(payload_by_id) != set(record_by_id):
        missing = sorted(set(record_by_id) - set(payload_by_id))
        raise ValueError(f"prediction inventory incomplete; first missing={missing[:1]}")

    reference_count = prediction_count = matched_count = 0
    total_seconds = 0.0
    hit_counts = {tolerance: 0 for tolerance in TOLERANCES}
    onset_errors: list[float] = []
    seizure_record_count = seizure_record_hit_count = 0
    patient_parts: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    runtime_totals = {
        "model_inference_seconds": 0.0,
        "EEG_IO_and_resample_seconds": 0.0,
        "end_to_end_pipeline_seconds": 0.0,
    }
    for identity in sorted(record_by_id):
        record = record_by_id[identity]
        payload = payload_by_id[identity]
        duration = float(Fraction(*record["recording_duration_seconds_fraction"]))
        references = sorted(
            [
                {"start_seconds": float(row["start_seconds"]), "stop_seconds": float(row["stop_seconds"])}
                for row in record["seizure_events"]
            ],
            key=lambda row: (row["start_seconds"], row["stop_seconds"]),
        )
        predictions = decode_peaks(
            payload["peaks"],
            recording_duration_seconds=duration,
            threshold=threshold,
        )
        matches = ordered_match(references, predictions)
        reference_count += len(references)
        prediction_count += len(predictions)
        matched_count += len(matches)
        total_seconds += duration
        patient_parts[str(record["patient_id"])].append((len(references), len(predictions), len(matches)))
        if references:
            seizure_record_count += 1
            if matches:
                seizure_record_hit_count += 1
        for reference_index, prediction_index, _ in matches:
            error = predictions[prediction_index]["start_seconds"] - references[reference_index]["start_seconds"]
            onset_errors.append(error)
            for tolerance in TOLERANCES:
                if abs(error) <= tolerance + 1e-12:
                    hit_counts[tolerance] += 1
        runtime = payload["runtime"]
        for key in runtime_totals:
            value = float(runtime[key])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid runtime field {key}: {identity}")
            runtime_totals[key] += value

    false_alarm_count = prediction_count - matched_count
    sensitivity = matched_count / reference_count
    precision = matched_count / prediction_count
    f1 = 2.0 * sensitivity * precision / (sensitivity + precision)
    false_alarms_per_24h = false_alarm_count / (total_seconds / 86400.0)
    patient_sensitivities: list[float] = []
    patient_precisions: list[float] = []
    patient_f1s: list[float] = []
    patient_fa24: list[float] = []
    duration_by_patient: dict[str, float] = defaultdict(float)
    for record in records:
        duration_by_patient[str(record["patient_id"])] += float(Fraction(*record["recording_duration_seconds_fraction"]))
    for patient_id, parts in patient_parts.items():
        refs = sum(part[0] for part in parts)
        preds = sum(part[1] for part in parts)
        matched = sum(part[2] for part in parts)
        if refs:
            patient_sensitivities.append(matched / refs)
        if preds:
            patient_precision = matched / preds
            patient_precisions.append(patient_precision)
            if refs:
                patient_sensitivity = matched / refs
                patient_f1s.append(
                    0.0
                    if patient_precision + patient_sensitivity == 0
                    else 2 * patient_precision * patient_sensitivity / (patient_precision + patient_sensitivity)
                )
        patient_fa24.append((preds - matched) / (duration_by_patient[patient_id] / 86400.0))

    replay_metrics = {
        "threshold": threshold,
        "recording_count": len(records),
        "patient_count": len(patient_parts),
        "recording_hours": total_seconds / 3600.0,
        "reference_event_count": reference_count,
        "predicted_alarm_count": prediction_count,
        "matched_event_count": matched_count,
        "false_alarm_count": false_alarm_count,
        "event_sensitivity": sensitivity,
        "event_precision": precision,
        "event_f1": f1,
        "false_alarms_per_24h": false_alarms_per_24h,
        "seizure_record_recall": {
            "hit_count": seizure_record_hit_count,
            "denominator": seizure_record_count,
            "rate": seizure_record_hit_count / seizure_record_count,
        },
        "onset_absolute_hit_rate": {
            f"{tolerance:g}s": {
                "hit_count": hit_counts[tolerance],
                "reference_event_denominator": reference_count,
                "rate": hit_counts[tolerance] / reference_count,
            }
            for tolerance in TOLERANCES
        },
        "onset_absolute_error_median_matched_only_seconds": statistics.median(abs(value) for value in onset_errors),
        "patient_macro": {
            "event_sensitivity_evaluable_patients_only": sum(patient_sensitivities) / len(patient_sensitivities),
            "event_precision_evaluable_patients_only": sum(patient_precisions) / len(patient_precisions),
            "event_f1_evaluable_patients_only": sum(patient_f1s) / len(patient_f1s),
            "false_alarms_per_24h": sum(patient_fa24) / len(patient_fa24),
        },
        "runtime_from_frozen_payloads": {
            **runtime_totals,
            "warm_model_inference_RTF": runtime_totals["model_inference_seconds"] / total_seconds,
            "end_to_end_RTF": runtime_totals["end_to_end_pipeline_seconds"] / total_seconds,
        },
    }

    frozen = json.loads(frozen_metrics_path.read_text(encoding="utf-8"))
    frozen_pooled = frozen["best_source_dev_diagnostic_operating_point"]["pooled"]
    frozen_macro = frozen["best_source_dev_diagnostic_operating_point"]["patient_macro"]
    frozen_seizure_record_recall = frozen_pooled["seizure_record_recall"]
    replay_seizure_record_recall = replay_metrics["seizure_record_recall"]
    frozen_onset_latency = frozen_pooled["onset_latency_seconds"]
    replay_onset_coverage = matched_count / reference_count
    comparisons = {
        "threshold": frozen["best_source_dev_diagnostic_operating_point"]["center_threshold"] == threshold,
        "recording_count": frozen_pooled["recording_count"] == len(records),
        "reference_event_count": frozen_pooled["reference_event_count"] == reference_count,
        "predicted_alarm_count": frozen_pooled["predicted_alarm_count"] == prediction_count,
        "matched_event_count": frozen_pooled["matched_event_count"] == matched_count,
        "false_alarm_count": frozen_pooled["false_alarm_count"] == false_alarm_count,
        "event_sensitivity": abs(frozen_pooled["event_sensitivity"] - sensitivity) <= 1e-15,
        "event_precision": abs(frozen_pooled["event_precision"] - precision) <= 1e-15,
        "event_f1": abs(frozen_pooled["event_f1"] - f1) <= 1e-15,
        "false_alarms_per_24h": abs(frozen_pooled["alarm_false_alarms_per_24h"] - false_alarms_per_24h) <= 1e-12,
        "onset_hit_at_1s": frozen_pooled["onset_absolute_hit_rate"]["1s"]["hit_count"] == hit_counts[1.0],
        "onset_hit_at_3s": frozen_pooled["onset_absolute_hit_rate"]["3s"]
        == replay_metrics["onset_absolute_hit_rate"]["3s"],
        "onset_hit_at_5s": frozen_pooled["onset_absolute_hit_rate"]["5s"]["hit_count"] == hit_counts[5.0],
        "onset_hit_at_10s": frozen_pooled["onset_absolute_hit_rate"]["10s"]["hit_count"] == hit_counts[10.0],
        "seizure_record_recall_hit_count": frozen_seizure_record_recall[
            "seizure_recording_hit_count"
        ]
        == replay_seizure_record_recall["hit_count"],
        "seizure_record_recall_denominator": frozen_seizure_record_recall[
            "seizure_recording_denominator"
        ]
        == replay_seizure_record_recall["denominator"],
        "seizure_record_recall_rate": abs(
            frozen_seizure_record_recall["rate"]
            - replay_seizure_record_recall["rate"]
        )
        <= 1e-15,
        "onset_absolute_error_median_matched_only_seconds": abs(
            frozen_onset_latency["absolute_median_matched_only"]
            - replay_metrics["onset_absolute_error_median_matched_only_seconds"]
        )
        <= 1e-12,
        "onset_matched_event_denominator": frozen_onset_latency[
            "matched_event_denominator"
        ]
        == matched_count,
        "onset_reference_event_denominator": frozen_onset_latency[
            "reference_event_denominator"
        ]
        == reference_count,
        "onset_matched_reference_coverage": abs(
            frozen_onset_latency["matched_reference_coverage"]
            - replay_onset_coverage
        )
        <= 1e-15,
        "patient_macro_event_sensitivity": abs(
            frozen_macro["event_sensitivity"]
            - replay_metrics["patient_macro"]["event_sensitivity_evaluable_patients_only"]
        ) <= 1e-15,
        "patient_macro_event_precision": abs(
            frozen_macro["event_precision"]
            - replay_metrics["patient_macro"]["event_precision_evaluable_patients_only"]
        ) <= 1e-15,
        "patient_macro_event_f1": abs(
            frozen_macro["event_f1"]
            - replay_metrics["patient_macro"]["event_f1_evaluable_patients_only"]
        ) <= 1e-15,
        "patient_macro_false_alarms_per_24h": abs(
            frozen_macro["alarm_false_alarms_per_24h"]
            - replay_metrics["patient_macro"]["false_alarms_per_24h"]
        ) <= 1e-12,
        "end_to_end_RTF": abs(
            frozen["runtime"]["end_to_end_EEG_IO_resample_inference_decode_RTF"]
            - replay_metrics["runtime_from_frozen_payloads"]["end_to_end_RTF"]
        ) <= 1e-15,
    }
    if not all(comparisons.values()):
        raise RuntimeError(f"independent metric replay differs from frozen metrics: {comparisons}")
    inventory_sha256 = hashlib.sha256(canonical_bytes(inventory_rows)).hexdigest()
    return content_address(
        {
            "schema_version": "eventnet_common17_frozen_metric_replay_v1",
            "status": "pass_independent_decoder_matching_metric_replay",
            "replay_performed_inference": False,
            "replay_loaded_model_or_checkpoint": False,
            "replay_read_EEG_samples": False,
            "reference_values_used_for_scoring_only": True,
            "manifest_path": str(manifest_path.resolve()),
            "manifest_file_sha256": file_sha256(manifest_path),
            "prediction_directory": str(prediction_dir.resolve()),
            "prediction_file_count": len(inventory_rows),
            "prediction_inventory_sha256": inventory_sha256,
            "checkpoint_file_sha256_from_payloads": expected_checkpoint_sha256,
            "frozen_metrics_path": str(frozen_metrics_path.resolve()),
            "frozen_metrics_file_sha256": file_sha256(frozen_metrics_path),
            "decoder_contract": {
                "center_threshold": threshold,
                "target_fs_hz": TARGET_FS_HZ,
                "maximum_duration_seconds": MAXIMUM_DURATION_SECONDS,
                "merge_overlapping_decoded_events": True,
            },
            "matching_contract": "ordered_one_to_one_max_match_then_max_IoU_then_min_absolute_onset_error",
            "metrics": replay_metrics,
            "exact_frozen_metric_comparisons": comparisons,
            "receipt_sha256": "CONTENT-ADDRESS-PENDING",
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--frozen-metrics", type=Path, default=DEFAULT_FROZEN_METRICS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--expected-checkpoint-sha256", default=EXPECTED_CHECKPOINT_SHA256
    )
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    args = parser.parse_args()
    receipt = replay(
        manifest_path=args.manifest.resolve(strict=True),
        prediction_dir=args.predictions.resolve(strict=True),
        frozen_metrics_path=args.frozen_metrics.resolve(strict=True),
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        threshold=args.threshold,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_bytes(receipt) + b"\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
