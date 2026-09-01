#!/usr/bin/env python3
"""Audit irreversible 60 s NMS loss in frozen common17 EN17 predictions.

This is a CPU-only, post-prediction audit.  It reads gzip prediction payloads
and the frozen manifest reference sidecar for evaluation, but never loads EEG,
a checkpoint, torch, EDF annotations, clinical text, or spreadsheets.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording.continuous_detection_benchmark import (  # noqa: E402
    _aggregate_metrics,
    _ordered_event_matching,
)


DEFAULT_RUN = (
    ROOT
    / "outputs/eventnet_common17_streaming_bf16_common17_20epoch_exploratory_v1_20260825"
)
DEFAULT_MANIFEST = (
    ROOT / "outputs/eventnet_common17_streaming_v1_20260824/manifest.json"
)
DEFAULT_PREDICTIONS = DEFAULT_RUN / "source_dev_full_global_v3/predictions"
DEFAULT_METRICS = DEFAULT_RUN / "source_dev_full_global_v3/metrics.json"
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/eventnet_common17_20epoch_60s_nms_information_loss_audit_v1_20260825/receipt.json"
)
LEGACY_SCHEMA = "eventnet_common17_dev_prediction_global_posterior_runtime_v3"
REPLAYABLE_SCHEMA = "eventnet_common17_dev_prediction_replayable_pre_nms_runtime_v4"
TARGET_FS_HZ = 256.0
MAXIMUM_DURATION_SECONDS = 300.0
TOLERANCES = (1.0, 3.0, 5.0, 10.0)
PENDING = "CONTENT-ADDRESS-PENDING"


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_address(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["receipt_sha256"] = PENDING
    result["receipt_sha256"] = hashlib.sha256(canonical_bytes(result)).hexdigest()
    return result


def decode_legacy_peaks(
    peaks: Sequence[Mapping[str, object]],
    *,
    recording_duration_seconds: float,
    threshold: float,
) -> list[dict[str, float]]:
    events: list[dict[str, float]] = []
    for peak in peaks:
        probability = float(peak["center_probability"])
        if probability < threshold:
            continue
        duration = min(
            MAXIMUM_DURATION_SECONDS,
            max(
                1.0 / TARGET_FS_HZ,
                float(peak["duration_fraction"]) * MAXIMUM_DURATION_SECONDS,
            ),
        )
        center = float(peak["center_seconds"])
        if not all(math.isfinite(value) for value in (probability, duration, center)):
            raise ValueError("legacy peak contains a non-finite value")
        start = max(0.0, center - duration / 2.0)
        stop = min(recording_duration_seconds, center + duration / 2.0)
        if stop > start:
            events.append({"start_seconds": start, "stop_seconds": stop})
    merged: list[dict[str, float]] = []
    for event in sorted(events, key=lambda row: (row["start_seconds"], row["stop_seconds"])):
        if not merged or event["start_seconds"] > merged[-1]["stop_seconds"]:
            merged.append(dict(event))
        else:
            merged[-1]["stop_seconds"] = max(
                merged[-1]["stop_seconds"], event["stop_seconds"]
            )
    return merged


def reference_events(record: Mapping[str, Any]) -> list[dict[str, float]]:
    return sorted(
        [
            {
                "start_seconds": float(row["start_seconds"]),
                "stop_seconds": float(row["stop_seconds"]),
            }
            for row in record["seizure_events"]
        ],
        key=lambda row: (row["start_seconds"], row["stop_seconds"]),
    )


def close_reference_structure(
    references: Sequence[Mapping[str, float]],
) -> tuple[set[int], list[tuple[int, int]], int]:
    close_indices: set[int] = set()
    close_center_pairs: list[tuple[int, int]] = []
    close_onset_pair_count = 0
    for index in range(len(references) - 1):
        left = references[index]
        right = references[index + 1]
        onset_gap = float(right["start_seconds"]) - float(left["start_seconds"])
        left_center = 0.5 * (
            float(left["start_seconds"]) + float(left["stop_seconds"])
        )
        right_center = 0.5 * (
            float(right["start_seconds"]) + float(right["stop_seconds"])
        )
        if onset_gap < 60.0:
            close_onset_pair_count += 1
        if right_center - left_center < 60.0:
            close_indices.update((index, index + 1))
            close_center_pairs.append((index, index + 1))
    return close_indices, close_center_pairs, close_onset_pair_count


def nms_noninjective_proof() -> dict[str, Any]:
    def suppress(rows: Sequence[Mapping[str, float]]) -> list[dict[str, float]]:
        kept: list[dict[str, float]] = []
        for row in sorted(
            rows,
            key=lambda value: (-value["center_probability"], value["center_seconds"]),
        ):
            if all(
                abs(row["center_seconds"] - previous["center_seconds"]) >= 60.0
                for previous in kept
            ):
                kept.append(dict(row))
        return sorted(kept, key=lambda value: value["center_seconds"])

    high = {"center_seconds": 40.0, "center_probability": 0.9}
    suppressed = {"center_seconds": 20.0, "center_probability": 0.7}
    one_candidate = [high]
    two_candidates = [suppressed, high]
    first = suppress(one_candidate)
    second = suppress(two_candidates)
    return {
        "input_candidate_counts": [len(one_candidate), len(two_candidates)],
        "input_candidate_gap_seconds": 20.0,
        "serialized_outputs_identical": first == second,
        "serialized_output": first,
        "interpretation": (
            "two_distinct_pre_nms_candidate_sets_map_to_the_same_legacy_payload;"
            "the_suppressed_candidate_cannot_be_reconstructed"
        ),
    }


def evaluate_threshold(
    *,
    threshold: float,
    records: Sequence[Mapping[str, Any]],
    payload_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    close_reference_count = 0
    close_matched_count = 0
    close_hits = {tolerance: 0 for tolerance in TOLERANCES}
    close_pair_count = 0
    close_pair_both_matched = 0
    close_pair_both_hits = {tolerance: 0 for tolerance in TOLERANCES}
    for record in records:
        identity = str(record["analysis_identity_id"])
        duration = float(Fraction(*record["recording_duration_seconds_fraction"]))
        references = reference_events(record)
        predictions = decode_legacy_peaks(
            payload_by_id[identity]["peaks"],
            recording_duration_seconds=duration,
            threshold=threshold,
        )
        rows.append(
            {
                "duration_seconds": duration,
                "reference_events": references,
                "predicted_events": predictions,
            }
        )
        close_indices, close_pairs, _ = close_reference_structure(references)
        matches = _ordered_event_matching(references, predictions)
        prediction_by_reference = {
            reference_index: prediction_index
            for reference_index, prediction_index, _iou in matches
        }
        close_reference_count += len(close_indices)
        close_matched_count += sum(
            index in prediction_by_reference for index in close_indices
        )
        for reference_index in close_indices:
            prediction_index = prediction_by_reference.get(reference_index)
            if prediction_index is None:
                continue
            error = (
                float(predictions[prediction_index]["start_seconds"])
                - float(references[reference_index]["start_seconds"])
            )
            for tolerance in TOLERANCES:
                if abs(error) <= tolerance + 1e-12:
                    close_hits[tolerance] += 1
        close_pair_count += len(close_pairs)
        for left, right in close_pairs:
            if left not in prediction_by_reference or right not in prediction_by_reference:
                continue
            close_pair_both_matched += 1
            left_error = (
                float(predictions[prediction_by_reference[left]]["start_seconds"])
                - float(references[left]["start_seconds"])
            )
            right_error = (
                float(predictions[prediction_by_reference[right]]["start_seconds"])
                - float(references[right]["start_seconds"])
            )
            for tolerance in TOLERANCES:
                if abs(left_error) <= tolerance + 1e-12 and abs(right_error) <= tolerance + 1e-12:
                    close_pair_both_hits[tolerance] += 1
    pooled = _aggregate_metrics(rows, tolerances=TOLERANCES)
    return {
        "threshold": threshold,
        "overall": {
            "event_sensitivity": pooled["event_sensitivity"],
            "event_precision": pooled["event_precision"],
            "event_f1": pooled["event_f1"],
            "false_alarms_per_24h": pooled["alarm_false_alarms_per_24h"],
            "onset_hit_at_1s": pooled["onset_absolute_hit_rate"]["1s"]["rate"],
            "onset_hit_at_5s": pooled["onset_absolute_hit_rate"]["5s"]["rate"],
            "onset_hit_at_10s": pooled["onset_absolute_hit_rate"]["10s"]["rate"],
        },
        "adjacent_center_gap_lt60s_event_subgroup": {
            "reference_event_count": close_reference_count,
            "matched_event_count": close_matched_count,
            "event_sensitivity": (
                None
                if close_reference_count == 0
                else close_matched_count / close_reference_count
            ),
            "onset_hit_at_1s": (
                None if close_reference_count == 0 else close_hits[1.0] / close_reference_count
            ),
            "onset_hit_at_5s": (
                None if close_reference_count == 0 else close_hits[5.0] / close_reference_count
            ),
            "onset_hit_at_10s": (
                None if close_reference_count == 0 else close_hits[10.0] / close_reference_count
            ),
        },
        "adjacent_center_gap_lt60s_pair_recovery": {
            "pair_count": close_pair_count,
            "both_members_matched_count": close_pair_both_matched,
            "both_members_matched_rate": (
                None if close_pair_count == 0 else close_pair_both_matched / close_pair_count
            ),
            "both_members_onset_hit_at_1s_count": close_pair_both_hits[1.0],
            "both_members_onset_hit_at_5s_count": close_pair_both_hits[5.0],
            "both_members_onset_hit_at_10s_count": close_pair_both_hits[10.0],
        },
    }


def audit(
    *,
    manifest_path: Path,
    prediction_dir: Path,
    frozen_metrics_path: Path,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frozen_metrics = json.loads(frozen_metrics_path.read_text(encoding="utf-8"))
    records = [row for row in manifest["records"] if row["model_split"] == "source_dev"]
    record_by_id = {str(row["analysis_identity_id"]): row for row in records}
    if len(records) != 1821 or len(record_by_id) != len(records):
        raise ValueError("source-dev manifest is not the frozen 1,821-record roster")
    payload_by_id: dict[str, dict[str, Any]] = {}
    inventory_rows: list[dict[str, str]] = []
    schema_values: set[str] = set()
    checkpoint_values: set[str] = set()
    distance_values: set[float] = set()
    floor_values: set[float] = set()
    smoothing_values: set[int] = set()
    total_peak_count = 0
    zero_peak_records = 0
    one_peak_records = 0
    multi_peak_records = 0
    interpeak_gaps: list[float] = []
    forbidden_replay_fields = {
        "center_posterior",
        "duration_posterior",
        "pre_nms_candidate_cache",
        "pre_nms_peak_candidates",
        "raw_local_maxima",
        "suppressed_peaks",
    }
    replay_field_presence = {key: 0 for key in sorted(forbidden_replay_fields)}
    for path in sorted(prediction_dir.glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        identity = str(payload.get("analysis_identity_id"))
        if identity not in record_by_id or identity in payload_by_id:
            raise ValueError(f"prediction identity is extra or duplicated: {identity}")
        payload_by_id[identity] = payload
        inventory_rows.append(
            {"analysis_identity_id": identity, "file_sha256": file_sha256(path)}
        )
        schema_values.add(str(payload.get("schema_version")))
        checkpoint_values.add(str(payload.get("checkpoint_file_sha256")))
        distance_values.add(float(payload.get("minimum_peak_distance_seconds")))
        floor_values.add(float(payload.get("minimum_peak_threshold")))
        smoothing_values.add(int(payload.get("smoothing_sigma_samples")))
        for key in replay_field_presence:
            replay_field_presence[key] += int(key in payload)
        peaks = payload.get("peaks")
        if not isinstance(peaks, list):
            raise ValueError(f"prediction peaks are missing: {identity}")
        centers = [float(row["center_seconds"]) for row in peaks]
        if centers != sorted(centers) or any(not math.isfinite(value) for value in centers):
            raise ValueError(f"prediction peak centers are malformed: {identity}")
        gaps = [right - left for left, right in zip(centers, centers[1:])]
        interpeak_gaps.extend(gaps)
        total_peak_count += len(peaks)
        zero_peak_records += int(len(peaks) == 0)
        one_peak_records += int(len(peaks) == 1)
        multi_peak_records += int(len(peaks) > 1)
    if set(payload_by_id) != set(record_by_id):
        raise ValueError("prediction inventory does not close the source-dev roster")
    if schema_values != {LEGACY_SCHEMA}:
        raise ValueError(f"unexpected prediction schemas: {sorted(schema_values)}")
    if distance_values != {60.0} or floor_values != {0.001} or smoothing_values != {100}:
        raise ValueError("legacy prediction decoder contract drifted")

    adjacent_pair_count = 0
    center_gap_lt60_pair_count = 0
    onset_gap_lt60_pair_count = 0
    center_gap_lt60_event_count = 0
    center_gap_lt60_record_count = 0
    for record in records:
        references = reference_events(record)
        adjacent_pair_count += max(0, len(references) - 1)
        close_indices, close_pairs, close_onset_count = close_reference_structure(references)
        center_gap_lt60_pair_count += len(close_pairs)
        onset_gap_lt60_pair_count += close_onset_count
        center_gap_lt60_event_count += len(close_indices)
        center_gap_lt60_record_count += int(bool(close_pairs))

    thresholds = [float(row["center_threshold"]) for row in frozen_metrics["metric_grid"]]
    threshold_grid = [
        evaluate_threshold(
            threshold=threshold,
            records=records,
            payload_by_id=payload_by_id,
        )
        for threshold in thresholds
    ]
    best_threshold = float(
        frozen_metrics["best_source_dev_diagnostic_operating_point"]["center_threshold"]
    )
    selected = next(row for row in threshold_grid if row["threshold"] == best_threshold)
    frozen_selected = frozen_metrics["best_source_dev_diagnostic_operating_point"]["pooled"]
    exact_replay = {
        "event_sensitivity": selected["overall"]["event_sensitivity"]
        == frozen_selected["event_sensitivity"],
        "event_precision": selected["overall"]["event_precision"]
        == frozen_selected["event_precision"],
        "event_f1": selected["overall"]["event_f1"] == frozen_selected["event_f1"],
        "false_alarms_per_24h": selected["overall"]["false_alarms_per_24h"]
        == frozen_selected["alarm_false_alarms_per_24h"],
        "onset_hit_at_10s": selected["overall"]["onset_hit_at_10s"]
        == frozen_selected["onset_absolute_hit_rate"]["10s"]["rate"],
    }
    if not all(exact_replay.values()):
        raise RuntimeError("CPU replay does not reproduce the frozen selected metrics")

    source_path = ROOT / "src/clinical_eeg_long_recording/eventnet_common17_streaming_v1.py"
    test_path = ROOT / "tests/test_eventnet_common17_streaming_v1.py"
    return content_address(
        {
            "schema_version": "eventnet_common17_legacy_60s_nms_information_loss_audit_v1",
            "status": "legacy_candidates_irreversibly_lost_upstream_repair_implemented",
            "inputs": {
                "manifest_path": str(manifest_path.resolve(strict=True)),
                "manifest_file_sha256": file_sha256(manifest_path),
                "prediction_directory": str(prediction_dir.resolve(strict=True)),
                "prediction_file_count": len(payload_by_id),
                "prediction_inventory_sha256": hashlib.sha256(
                    canonical_bytes(inventory_rows)
                ).hexdigest(),
                "frozen_metrics_path": str(frozen_metrics_path.resolve(strict=True)),
                "frozen_metrics_file_sha256": file_sha256(frozen_metrics_path),
                "checkpoint_file_sha256_values": sorted(checkpoint_values),
            },
            "legacy_payload_audit": {
                "payload_schema_values": sorted(schema_values),
                "minimum_peak_distance_seconds_values": sorted(distance_values),
                "minimum_peak_threshold_values": sorted(floor_values),
                "smoothing_sigma_samples_values": sorted(smoothing_values),
                "total_serialized_post_nms_peak_count": total_peak_count,
                "zero_peak_record_count": zero_peak_records,
                "one_peak_record_count": one_peak_records,
                "multi_peak_record_count": multi_peak_records,
                "minimum_observed_serialized_interpeak_gap_seconds": (
                    None if not interpeak_gaps else min(interpeak_gaps)
                ),
                "serialized_interpeak_gap_lt60_count": sum(
                    value < 60.0 - 1e-12 for value in interpeak_gaps
                ),
                "pre_nms_or_dense_replay_field_presence_counts": replay_field_presence,
                "all_1821_payloads_lack_pre_nms_replay_state": all(
                    value == 0 for value in replay_field_presence.values()
                ),
            },
            "reference_adjacency_evaluation_only": {
                "reference_event_count": sum(
                    len(record["seizure_events"]) for record in records
                ),
                "adjacent_pair_count": adjacent_pair_count,
                "adjacent_onset_gap_lt60_pair_count": onset_gap_lt60_pair_count,
                "adjacent_center_gap_lt60_pair_count": center_gap_lt60_pair_count,
                "events_participating_in_center_gap_lt60_pair_count": (
                    center_gap_lt60_event_count
                ),
                "records_with_center_gap_lt60_pair_count": center_gap_lt60_record_count,
                "reference_used_after_prediction_freeze_for_scoring_only": True,
            },
            "cpu_only_legacy_threshold_grid": threshold_grid,
            "frozen_best_f1_threshold_replay": {
                "threshold": best_threshold,
                "metrics": selected,
                "exact_frozen_comparisons": exact_replay,
            },
            "adaptive_deblending_result": {
                "status": "not_identifiable_from_legacy_post_nms_payloads",
                "event_sensitivity_delta": None,
                "false_alarms_per_24h_delta": None,
                "onset_hit_delta": None,
                "reason": (
                    "all_local_maxima_suppressed_by_the_fixed_60s_nms_were_discarded;"
                    "thresholding_the_kept_peaks_cannot_restore_them"
                ),
                "noninjective_mapping_proof": nms_noninjective_proof(),
            },
            "upstream_repair": {
                "prediction_schema_version": REPLAYABLE_SCHEMA,
                "capture_stage": (
                    "full_record_smoothed_center_posterior_before_distance_nms"
                ),
                "preserved_reference_free_statistics": [
                    "center_sample",
                    "center_probability",
                    "duration_fraction",
                    "left_valley_probability",
                    "right_valley_probability",
                ],
                "supports_cpu_only_replay": [
                    "threshold",
                    "minimum_peak_distance",
                    "adjacent_valley_deblending",
                ],
                "legacy_v3_in_place_overwrite_refused": True,
                "source_path": str(source_path.relative_to(ROOT)),
                "source_file_sha256": file_sha256(source_path),
                "test_path": str(test_path.relative_to(ROOT)),
                "test_file_sha256": file_sha256(test_path),
                "new_full_prediction_materialization_performed": False,
            },
            "firewall": {
                "GPU_used": False,
                "model_or_checkpoint_loaded": False,
                "EEG_samples_or_EDF_opened": False,
                "EDF_annotation_used": False,
                "clinical_text_spreadsheet_video_behavior_used": False,
                "Qwen_touched": False,
                "global_TERM_seiz_used_for_postfreeze_evaluation_only": True,
            },
            "receipt_sha256": PENDING,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--frozen-metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = audit(
        manifest_path=args.manifest.resolve(strict=True),
        prediction_dir=args.predictions.resolve(strict=True),
        frozen_metrics_path=args.frozen_metrics.resolve(strict=True),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_bytes(result) + b"\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
