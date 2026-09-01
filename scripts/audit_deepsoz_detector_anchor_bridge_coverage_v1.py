#!/usr/bin/env python3
"""Audit the frozen detector-to-DeepSOZ bridge without running inference.

The emitted JSONL is an identity/prediction-only connection carrier.  The
receipt keeps a separate, explicitly post-freeze evaluation join so seizure
times can never select, delete, or replace detector anchors.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from fractions import Fraction
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS_PROJECTION = (
    ROOT / "outputs/tusz_complete_detector_roster_v2_20260823/analysis_projection.json"
)
DEFAULT_DETECTOR_MANIFEST = (
    ROOT / "outputs/eventnet_common17_streaming_v1_20260824/manifest.json"
)
DEFAULT_ORACLE_CARRIER = (
    ROOT / "outputs/clinical_eeg_common17_car17_labram_phase_v1_20260824/manifest.json"
)
DEFAULT_EVALUATION = (
    ROOT
    / "outputs/eventnet_common17_streaming_bf16_common17_20epoch_exploratory_v1_20260825"
    / "source_dev_full_global_v3"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/clinical_eeg_deepsoz_detector_anchor_bridge_coverage_v1r2_20260825"
)

SCHEMA = "clinical_eeg_deepsoz_detector_anchor_bridge_coverage_v1"
CONNECTION_SCHEMA = "clinical_eeg_target_blind_detector_bridge_record_v1"
PREDICTION_SCHEMA = "eventnet_common17_dev_prediction_global_posterior_runtime_v3"
PENDING = "CONTENT-ADDRESS-PENDING"


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve(strict=True).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _read_gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path.resolve(strict=True), "rt", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected gzip JSON object: {path}")
    return value


def _decode_alarm_intervals(
    peaks: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
    duration_seconds: float,
    maximum_duration_seconds: float = 300.0,
) -> list[dict[str, float]]:
    events: list[dict[str, float]] = []
    for peak in peaks:
        if float(peak["center_probability"]) < threshold:
            continue
        duration = min(
            maximum_duration_seconds,
            max(1.0 / 256.0, float(peak["duration_fraction"]) * maximum_duration_seconds),
        )
        center = float(peak["center_seconds"])
        start = max(0.0, center - duration / 2.0)
        stop = min(duration_seconds, center + duration / 2.0)
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


def _overlap_match_counts(
    references: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
) -> tuple[int, int, int]:
    """Deterministic one-to-one maximum-overlap matching for aggregate counts."""

    pairs: list[tuple[float, int, int]] = []
    for reference_index, reference in enumerate(references):
        ref_start = float(reference["start_seconds"])
        ref_stop = float(reference["stop_seconds"])
        for prediction_index, prediction in enumerate(predictions):
            pred_start = float(prediction["start_seconds"])
            pred_stop = float(prediction["stop_seconds"])
            overlap = min(ref_stop, pred_stop) - max(ref_start, pred_start)
            if overlap > 0:
                pairs.append((-overlap, reference_index, prediction_index))
    used_reference: set[int] = set()
    used_prediction: set[int] = set()
    for _negative_overlap, reference_index, prediction_index in sorted(pairs):
        if reference_index in used_reference or prediction_index in used_prediction:
            continue
        used_reference.add(reference_index)
        used_prediction.add(prediction_index)
    matched = len(used_reference)
    return matched, len(references) - matched, len(predictions) - matched


def _onset_match_count(
    references: Sequence[float], predictions: Sequence[float], tolerance_seconds: float
) -> int:
    pairs = sorted(
        (abs(reference - prediction), reference_index, prediction_index)
        for reference_index, reference in enumerate(references)
        for prediction_index, prediction in enumerate(predictions)
        if abs(reference - prediction) <= tolerance_seconds
    )
    used_reference: set[int] = set()
    used_prediction: set[int] = set()
    for _distance, reference_index, prediction_index in pairs:
        if reference_index in used_reference or prediction_index in used_prediction:
            continue
        used_reference.add(reference_index)
        used_prediction.add(prediction_index)
    return len(used_reference)


def _content_address(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(value))
    result["receipt_sha256"] = PENDING
    result["receipt_sha256"] = _canonical_sha256(result)
    return result


def run(
    *,
    analysis_projection_path: Path,
    detector_manifest_path: Path,
    oracle_carrier_path: Path,
    evaluation_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    projection = _read_json(analysis_projection_path)
    detector = _read_json(detector_manifest_path)
    carrier = _read_json(oracle_carrier_path)
    metrics_path = evaluation_dir.resolve(strict=True) / "metrics.json"
    prediction_dir = evaluation_dir.resolve(strict=True) / "predictions"
    metrics = _read_json(metrics_path)
    if not prediction_dir.is_dir():
        raise NotADirectoryError(prediction_dir)

    event_rows = carrier.get("scope", {}).get("event_roster")
    if not isinstance(event_rows, list) or len(event_rows) != 1_145:
        raise ValueError("oracle carrier is not the frozen 1,145-event roster")
    patients = {str(row["patient_id"]) for row in event_rows}
    if len(patients) != 102:
        raise ValueError("oracle carrier is not the frozen 102-patient roster")
    patient_by_path: dict[str, str] = {}
    onsets_by_path: dict[str, list[float]] = defaultdict(list)
    for row in event_rows:
        path = str(row["relative_edf_path"])
        patient = str(row["patient_id"])
        previous = patient_by_path.setdefault(path, patient)
        if previous != patient:
            raise RuntimeError("one EDF maps to multiple DeepSOZ patients")
        onsets_by_path[path].append(float(row["global_t0_sec"]))
    carrier_paths = sorted(patient_by_path)
    if len(carrier_paths) != 455:
        raise ValueError("oracle event carrier is not the frozen 455-EDF roster")

    projection_by_path = {str(row["local_edf_path"]): row for row in projection["records"]}
    if len(projection_by_path) != len(projection["records"]):
        raise RuntimeError("analysis projection has duplicate EDF paths")
    missing_projection = sorted(set(carrier_paths) - set(projection_by_path))
    if missing_projection:
        raise RuntimeError("455-EDF carrier is not closed by the identity projection")
    detector_by_path = {str(row["edf_relative_path"]): row for row in detector["records"]}
    if len(detector_by_path) != len(detector["records"]):
        raise RuntimeError("detector manifest has duplicate EDF paths")

    selected_threshold = float(
        metrics["user_facing_best_source_dev_diagnostic_metrics"]["center_threshold"]
    )
    # The payload itself is authoritative for the cache floor; the best grid
    # threshold above is only the source-dev-selected operating point.
    checkpoint_sha = str(metrics["checkpoint_file_sha256"])
    checkpoint_step = int(metrics["checkpoint_global_step"])

    connection_rows: list[dict[str, Any]] = []
    prediction_payloads: dict[str, dict[str, Any]] = {}
    prediction_paths: dict[str, Path] = {}
    for ordinal, path in enumerate(carrier_paths):
        identity_row = projection_by_path[path]
        identity = str(identity_row["analysis_identity_id"])
        detector_row = detector_by_path.get(path)
        prediction_path = prediction_dir / f"{identity}.json.gz"
        prediction = None
        if prediction_path.is_file():
            prediction = _read_gzip_json(prediction_path)
            if (
                prediction.get("schema_version") != PREDICTION_SCHEMA
                or prediction.get("analysis_identity_id") != identity
                or prediction.get("patient_id") != identity_row["local_patient_id"]
                or prediction.get("checkpoint_file_sha256") != checkpoint_sha
                or int(prediction.get("checkpoint_global_step", -1)) != checkpoint_step
                or prediction.get("FZ_or_PZ_model_axis_present") is not False
            ):
                raise RuntimeError(f"prediction identity/contract mismatch: {identity}")
            prediction_payloads[path] = prediction
            prediction_paths[path] = prediction_path
        selected_peak_count = 0
        decoded_anchor_count = 0
        cached_count = 0
        cache_floor = None
        if prediction is not None:
            peaks = prediction["peaks"]
            cached_count = len(peaks)
            cache_floor = float(prediction["minimum_peak_threshold"])
            selected_peak_count = sum(
                float(peak["center_probability"]) >= selected_threshold for peak in peaks
            )
            decoded_anchor_count = len(
                _decode_alarm_intervals(
                    peaks,
                    threshold=selected_threshold,
                    duration_seconds=float(prediction["recording_duration_seconds"]),
                )
            )
        connection_rows.append(
            {
                "schema_version": CONNECTION_SCHEMA,
                "record_ordinal": ordinal,
                "deepsoz_patient_id": patient_by_path[path],
                "edf_relative_path": path,
                "official_split": str(identity_row["official_split"]),
                "analysis_identity_id": identity,
                "source_edf_container_sha256": str(
                    identity_row["source_edf_container_sha256"]
                ),
                "detector_manifest_member": detector_row is not None,
                "detector_model_split": (
                    None if detector_row is None else str(detector_row["model_split"])
                ),
                "frozen_peak_prediction_available": prediction is not None,
                "prediction_representation": (
                    None if prediction is None else "post_NMS_peak_list_only"
                ),
                "dense_posterior_available": False,
                "prediction_file_sha256": (
                    None if prediction is None else _file_sha256(prediction_path)
                ),
                "cached_peak_floor_threshold": cache_floor,
                "cached_peak_count": cached_count,
                "frozen_selected_center_threshold": selected_threshold,
                "selected_peak_count": selected_peak_count,
                "decoded_onset_anchor_count": decoded_anchor_count,
            }
        )

    prediction_splits = Counter(
        row["official_split"]
        for row in connection_rows
        if row["frozen_peak_prediction_available"]
    )
    patient_prediction_coverage = {
        row["deepsoz_patient_id"]
        for row in connection_rows
        if row["frozen_peak_prediction_available"]
    }
    patients_with_candidates = {
        row["deepsoz_patient_id"]
        for row in connection_rows
        if row["decoded_onset_anchor_count"] > 0
    }

    full_reference_count = 0
    alarm_count = 0
    matched_count = 0
    missed_count = 0
    false_alarm_count = 0
    evaluated_seconds = 0.0
    bridge_reference_count = 0
    bridge_onset_hit_10s = 0
    bridge_record_hits: set[str] = set()
    bridge_patient_hits: set[str] = set()
    for path, prediction in prediction_payloads.items():
        detector_row = detector_by_path.get(path)
        if detector_row is None:
            raise RuntimeError("prediction exists outside detector manifest")
        duration = float(prediction["recording_duration_seconds"])
        evaluated_seconds += duration
        alarms = _decode_alarm_intervals(
            prediction["peaks"], threshold=selected_threshold, duration_seconds=duration
        )
        references = detector_row["seizure_events"]
        matched, missed, false = _overlap_match_counts(references, alarms)
        full_reference_count += len(references)
        alarm_count += len(alarms)
        matched_count += matched
        missed_count += missed
        false_alarm_count += false

        bridge_onsets = onsets_by_path[path]
        candidate_onsets = [float(alarm["start_seconds"]) for alarm in alarms]
        hit = _onset_match_count(bridge_onsets, candidate_onsets, 10.0)
        bridge_reference_count += len(bridge_onsets)
        bridge_onset_hit_10s += hit
        if hit:
            bridge_record_hits.add(path)
            bridge_patient_hits.add(patient_by_path[path])

    split_records = Counter(row["official_split"] for row in connection_rows)
    split_events = Counter(
        path.split("/", 1)[0] for path in onsets_by_path for _ in onsets_by_path[path]
    )
    split_patients: dict[str, int] = {}
    for split in ("train", "dev", "eval"):
        split_patients[split] = len(
            {
                row["deepsoz_patient_id"]
                for row in connection_rows
                if row["official_split"] == split
            }
        )
    manifest_members = sum(row["detector_manifest_member"] for row in connection_rows)
    prediction_count = len(prediction_payloads)
    uncovered_bridge_events = len(event_rows) - bridge_reference_count
    current_total_hit = bridge_onset_hit_10s

    connection_payload_sha256 = _canonical_sha256(connection_rows)
    receipt = _content_address(
        {
            "schema_version": SCHEMA,
            "status": "pass_connection_coverage_not_end_to_end_metric",
            "inference_performed": False,
            "SOZ_inference_performed": False,
            "raw_EEG_loaded": False,
            "cohort_boundary": {
                "patients": 102,
                "oracle_event_carrier_events": 1_145,
                "unique_EDF_records": 455,
                "records_by_official_split": dict(sorted(split_records.items())),
                "events_by_official_split": dict(sorted(split_events.items())),
                "patients_by_official_split": split_patients,
                "selection_warning": (
                    "The 455 records are the oracle-event signal-eligible carrier. "
                    "The emitted connection rows are target-blind, but the cohort itself "
                    "was selected using prior oracle-event eligibility and is not an "
                    "unbiased detector false-alarm cohort."
                ),
            },
            "identity_connection": {
                "analysis_projection_matches": len(carrier_paths),
                "analysis_projection_missing": 0,
                "detector_manifest_members": manifest_members,
                "detector_manifest_missing": len(carrier_paths) - manifest_members,
                "source_train_manifest_records": sum(
                    row["detector_model_split"] == "source_train"
                    for row in connection_rows
                ),
                "source_dev_manifest_records": sum(
                    row["detector_model_split"] == "source_dev"
                    for row in connection_rows
                ),
                "source_eval_identity_only_records": sum(
                    row["official_split"] == "eval" for row in connection_rows
                ),
                "connection_manifest_file": "target_blind_connection_manifest.jsonl",
                "connection_manifest_row_count": len(connection_rows),
                "connection_manifest_canonical_sha256": connection_payload_sha256,
                "connection_fields_exclude_seizure_times_and_SOZ_targets": True,
            },
            "frozen_prediction_coverage": {
                "artifact_role": "latest_full_source_dev_exploratory_cache_not_held_out_test",
                "checkpoint_file_sha256": checkpoint_sha,
                "checkpoint_global_step": checkpoint_step,
                "source_dev_selected_center_threshold": selected_threshold,
                "prediction_records": prediction_count,
                "prediction_records_by_official_split": dict(sorted(prediction_splits.items())),
                "record_coverage_fraction": prediction_count / len(carrier_paths),
                "patients_with_any_prediction": len(patient_prediction_coverage),
                "patient_coverage_fraction": len(patient_prediction_coverage) / len(patients),
                "patients_with_selected_candidate": len(patients_with_candidates),
                "cached_peak_count_at_payload_floor": sum(
                    row["cached_peak_count"] for row in connection_rows
                ),
                "selected_peak_count_before_interval_merge": sum(
                    row["selected_peak_count"] for row in connection_rows
                ),
                "decoded_onset_anchor_count_after_interval_merge": sum(
                    row["decoded_onset_anchor_count"] for row in connection_rows
                ),
                "peak_list_available": True,
                "dense_center_or_duration_posterior_available": False,
                "source_train_prediction_cache_available": False,
                "source_eval_prediction_cache_available": False,
            },
            "post_freeze_GT_join_diagnostic": {
                "join_order": (
                    "identity roster -> frozen detector predictions -> threshold decode -> "
                    "GT join; GT never selects or deletes a candidate"
                ),
                "complete_TERM_seiz_reference_scope_for_predicted_records": {
                    "records": prediction_count,
                    "recording_hours": evaluated_seconds / 3600.0,
                    "reference_events": full_reference_count,
                    "decoded_alarms": alarm_count,
                    "one_to_one_overlap_matched_events": matched_count,
                    "missed_reference_events": missed_count,
                    "false_alarms": false_alarm_count,
                    "false_alarms_per_24h": (
                        None
                        if evaluated_seconds == 0
                        else false_alarm_count * 86_400.0 / evaluated_seconds
                    ),
                    "not_full_455_record_detector_metric": True,
                },
                "oracle_carrier_anchor_connection_at_10s": {
                    "full_intent_to_diagnose_reference_events": len(event_rows),
                    "reference_events_in_records_with_cached_predictions": bridge_reference_count,
                    "reference_events_in_records_without_cached_predictions": uncovered_bridge_events,
                    "onset_hits_within_10s_in_cached_records": bridge_onset_hit_10s,
                    "onset_misses_within_10s_in_cached_records": (
                        bridge_reference_count - bridge_onset_hit_10s
                    ),
                    "current_cache_unavailable_or_missed_events": (
                        len(event_rows) - current_total_hit
                    ),
                    "records_with_at_least_one_10s_hit": len(bridge_record_hits),
                    "patients_with_at_least_one_10s_hit": len(bridge_patient_hits),
                    "descriptive_connectivity_only_not_detector_performance": True,
                },
                "SOZ_targets_joined": False,
            },
            "primary_end_to_end_evaluation_contract": {
                "stage_order": [
                    "freeze_455_record_identity_roster",
                    "run_frozen_detector_on_every_record_without_GT",
                    "freeze_all_candidate_anchors_and_payload_hashes",
                    "run_adaptive_Findings_on_every_candidate_including_false_alarms",
                    "aggregate_candidates_to_record_then_patient_prediction",
                    "freeze_predictions",
                    "join_TERM_seiz_and_patient_SOZ_GT_for_evaluation_only",
                ],
                "detector_event_denominator": (
                    "all reference seizures in the predeclared evaluation record roster"
                ),
                "record_denominator": 455,
                "patient_SOZ_denominator": 102,
                "patient_with_no_detector_anchor_policy": (
                    "retain in the 102-patient denominator as end-to-end failure/abstention"
                ),
                "false_candidate_policy": (
                    "process identically through Findings and aggregation; never remove by GT"
                ),
                "oracle_anchor_policy": (
                    "never replace a missed detector anchor in the primary arm; oracle anchors "
                    "are allowed only as a separately labelled localization ceiling"
                ),
                "record_level_SOZ_accuracy_is_primary": False,
                "reason": (
                    "SOZ GT is patient-level; treating 455 records as independent labels "
                    "would pseudo-replicate patients"
                ),
                "source_train_role": "in-sample development only_not_final_detector_test",
                "source_dev_role": "threshold_selection_only_not_held_out_test",
                "source_eval_role": (
                    "potential held-out test after separate execution admission and full cache"
                ),
            },
            "blockers": [
                "253 source-train carrier records have no frozen common17 prediction cache",
                "102 source-eval carrier records have identity projection but no detector-manifest execution/cache",
                "only post-NMS peaks, not dense posteriors, were persisted",
                "the current source-dev operating threshold was selected on source-dev itself",
                "adaptive Findings and record/patient SOZ aggregation have not been executed on detector anchors",
            ],
            "claim_boundary": {
                "102_patient_end_to_end_metric_available": False,
                "current_100_record_post_freeze_join_is_subset_diagnostic_only": True,
                "oracle_onset_SOZ_metrics_must_not_be_relabelled_end_to_end": True,
                "clinical_deployment_allowed": False,
            },
            "lineage": {
                "analysis_projection": {
                    "path": str(analysis_projection_path.resolve()),
                    "sha256": _file_sha256(analysis_projection_path),
                },
                "detector_manifest": {
                    "path": str(detector_manifest_path.resolve()),
                    "sha256": _file_sha256(detector_manifest_path),
                },
                "oracle_carrier_manifest": {
                    "path": str(oracle_carrier_path.resolve()),
                    "sha256": _file_sha256(oracle_carrier_path),
                },
                "detector_metrics": {
                    "path": str(metrics_path),
                    "sha256": _file_sha256(metrics_path),
                },
                "prediction_directory": str(prediction_dir),
                "audit_script": {
                    "path": str(Path(__file__).resolve()),
                    "sha256": _file_sha256(Path(__file__).resolve()),
                },
            },
            "access_receipt": {
                "raw_EEG_loaded": False,
                "EDF_annotations_loaded": False,
                "clinical_text_or_spreadsheet_loaded": False,
                "SOZ_target_tensor_loaded": False,
                "seizure_times_loaded_only_in_post_freeze_evaluation_join": True,
                "detector_or_SOZ_inference_performed": False,
                "GPU_used": False,
            },
            "receipt_sha256": PENDING,
        }
    )
    return receipt, connection_rows


def publish(output: Path, receipt: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> Path:
    target = output.resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    published = False
    try:
        connection_path = staging / "target_blind_connection_manifest.jsonl"
        with connection_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False))
                handle.write("\n")
        (staging / "coverage_receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        os.replace(staging, target)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--analysis-projection", type=Path, default=DEFAULT_ANALYSIS_PROJECTION)
    parser.add_argument("--detector-manifest", type=Path, default=DEFAULT_DETECTOR_MANIFEST)
    parser.add_argument("--oracle-carrier", type=Path, default=DEFAULT_ORACLE_CARRIER)
    parser.add_argument("--evaluation-dir", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    receipt, rows = run(
        analysis_projection_path=args.analysis_projection,
        detector_manifest_path=args.detector_manifest,
        oracle_carrier_path=args.oracle_carrier,
        evaluation_dir=args.evaluation_dir,
    )
    output = publish(args.output, receipt, rows)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "output": str(output),
                "records": receipt["cohort_boundary"]["unique_EDF_records"],
                "prediction_records": receipt["frozen_prediction_coverage"]["prediction_records"],
                "patients_with_predictions": receipt["frozen_prediction_coverage"]["patients_with_any_prediction"],
                "receipt_sha256": receipt["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
