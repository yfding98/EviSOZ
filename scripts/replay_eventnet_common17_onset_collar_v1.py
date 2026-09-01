#!/usr/bin/env python3
"""Replay independent onset-collar metrics from the frozen V6 predictions.

This is a read-only prediction replay.  It never opens an EDF, checkpoint, or
reference sidecar and never mutates the frozen EventNet output directory.  The
only write is a new content-addressed receipt at ``--output``.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from fractions import Fraction
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clinical_eeg_long_recording.continuous_detection_benchmark import (
    CONTINUOUS_BENCHMARK_METHOD_ID,
    aggregate_continuous_detection_metrics,
)
from src.clinical_eeg_long_recording.eventnet_common17_streaming_v1 import (
    COMMON17_CHANNEL_ORDER,
    DEFAULT_MINIMUM_PEAK_DISTANCE_SECONDS,
    DEFAULT_SMOOTHING_SIGMA_SAMPLES,
    PREDICTION_SCHEMA_VERSION,
    _decode_peaks,
    _valid_pre_nms_candidate_cache,
    load_common17_manifest,
)
from src.clinical_eeg_long_recording.onset_collar_scoring_v1 import (
    aggregate_onset_collar_metrics,
)
from src.clinical_eeg_long_recording.st16_common17_source_dev_evaluation_v1 import (
    _official_timescoring_metrics,
)


SCHEMA_VERSION = "eventnet_common17_v6_onset_collar_replay_receipt_v1"
MULTITRACK_SCHEMA_VERSION = (
    "eventnet_common17_v6_strict_onset_collar_szcore_replay_receipt_v2"
)
DEFAULT_THRESHOLD = 0.02
DEFAULT_COLLARS = (
    (1.0, 1.0),
    (3.0, 3.0),
    (5.0, 5.0),
    (10.0, 10.0),
    (30.0, 60.0),
)
EXPECTED_RECORDING_COUNT = 1821
EXPECTED_PATIENT_COUNT = 53
EXPECTED_REFERENCE_EVENT_COUNT = 1074
EXPECTED_CHECKPOINT_SHA256 = (
    "0d2f80ee9b63eaa5cc02dc9c7bff6f39f44f02a3159bcfc2bf4a0c2eef5ff297"
)
_PENDING = "CONTENT-ADDRESS-PENDING"

DEFAULT_MANIFEST = (
    ROOT / "outputs/eventnet_common17_streaming_v1_20260824/manifest.json"
)
DEFAULT_V6_ROOT = (
    ROOT
    / "outputs/clinical_eeg_common17_user_requested_detector_independent_retrain_v2_20260825"
)
DEFAULT_PREDICTION_DIR = DEFAULT_V6_ROOT / "source_dev_full_global_v4/predictions"
DEFAULT_STRICT_METRICS = DEFAULT_V6_ROOT / "source_dev_full_global_v4/metrics.json"
DEFAULT_V6_RECEIPT = DEFAULT_V6_ROOT / "two_level_validation_receipt_v6.json"
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/clinical_eeg_common17_eventnet_onset_collar_replay_v1_20260825/receipt.json"
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _content_address(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(value))
    result["receipt_sha256"] = _PENDING
    result["receipt_sha256"] = hashlib.sha256(_canonical_bytes(result)).hexdigest()
    return result


def _validate_content_address(value: Mapping[str, Any], *, context: str) -> None:
    pending = deepcopy(dict(value))
    supplied = pending.get("receipt_sha256")
    pending["receipt_sha256"] = _PENDING
    expected = hashlib.sha256(_canonical_bytes(pending)).hexdigest()
    if supplied != expected:
        raise ValueError(f"{context} is not content-addressed")


def _strict_summary(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "reference_event_count": metrics["reference_event_count"],
        "predicted_alarm_count": metrics["predicted_alarm_count"],
        "matched_event_count": metrics["matched_event_count"],
        "false_alarm_count": metrics["false_alarm_count"],
        "event_sensitivity": metrics["event_sensitivity"],
        "event_precision": metrics["event_precision"],
        "event_f1": metrics["event_f1"],
        "false_alarms_per_24h": metrics["alarm_false_alarms_per_24h"],
        "onset_absolute_hit_rate": deepcopy(metrics["onset_absolute_hit_rate"]),
        "onset_absolute_error_median_matched_only_seconds": metrics[
            "onset_latency_seconds"
        ]["absolute_median_matched_only"],
    }


def _frozen_strict_summary(metrics_receipt: Mapping[str, Any]) -> dict[str, Any]:
    best = metrics_receipt["best_source_dev_diagnostic_operating_point"]
    if float(best["center_threshold"]) != DEFAULT_THRESHOLD:
        raise ValueError("frozen V6 strict operating point is not threshold 0.02")
    return _strict_summary(best["pooled"])


def _v6_reported_strict_summary(v6_receipt: Mapping[str, Any]) -> dict[str, Any]:
    evaluation = v6_receipt["detection"]["evaluation"]
    return {
        "reference_event_count": evaluation["reference_events"],
        "predicted_alarm_count": evaluation["predicted_alarms"],
        "matched_event_count": evaluation["matched_reference_events"],
        "false_alarm_count": evaluation["false_alarms"],
        "event_sensitivity": evaluation["event_sensitivity"],
        "event_precision": evaluation["event_precision"],
        "event_f1": evaluation["event_f1"],
        "false_alarms_per_24h": evaluation["false_alarms_per_24h"],
        "onset_absolute_hit_rate": {
            f"{seconds:g}s": {
                "rate": evaluation[f"onset_hit_at_{seconds:g}s"]
            }
            for seconds in (1.0, 3.0, 5.0, 10.0)
        },
        "onset_absolute_error_median_matched_only_seconds": evaluation[
            "matched_only_absolute_onset_error_median_seconds"
        ],
    }


def _same_number(left: object, right: object, *, tolerance: float = 1e-12) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    return left == right


def _comparison(
    replayed: Mapping[str, Any], frozen: Mapping[str, Any]
) -> dict[str, bool]:
    fields = (
        "reference_event_count",
        "predicted_alarm_count",
        "matched_event_count",
        "false_alarm_count",
        "event_sensitivity",
        "event_precision",
        "event_f1",
        "false_alarms_per_24h",
        "onset_absolute_error_median_matched_only_seconds",
    )
    result = {
        field: _same_number(replayed[field], frozen[field]) for field in fields
    }
    for seconds in (1.0, 3.0, 5.0, 10.0):
        key = f"{seconds:g}s"
        replayed_row = replayed["onset_absolute_hit_rate"][key]
        frozen_row = frozen["onset_absolute_hit_rate"][key]
        for subfield in ("hit_count", "reference_event_denominator", "rate"):
            if subfield in frozen_row:
                result[f"onset_{key}_{subfield}"] = _same_number(
                    replayed_row[subfield], frozen_row[subfield]
                )
    return result


def replay(
    *,
    manifest_path: Path,
    prediction_dir: Path,
    strict_metrics_path: Path,
    v6_receipt_path: Path,
    output_path: Path,
    threshold: float = DEFAULT_THRESHOLD,
    expected_checkpoint_sha256: str = EXPECTED_CHECKPOINT_SHA256,
    include_szcore: bool = False,
) -> dict[str, Any]:
    """Recompute strict and onset-collar endpoints from frozen JSON payloads."""

    if not math.isfinite(threshold) or threshold != DEFAULT_THRESHOLD:
        raise ValueError("V6 onset-collar replay requires frozen threshold 0.02")
    if (
        len(expected_checkpoint_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_checkpoint_sha256)
    ):
        raise ValueError("expected checkpoint SHA-256 is invalid")

    manifest_source = manifest_path.resolve(strict=True)
    prediction_source = prediction_dir.resolve(strict=True)
    strict_metrics_source = strict_metrics_path.resolve(strict=True)
    v6_receipt_source = v6_receipt_path.resolve(strict=True)
    output = output_path.resolve()
    if not prediction_source.is_dir():
        raise NotADirectoryError(prediction_source)
    if output in {manifest_source, strict_metrics_source, v6_receipt_source}:
        raise PermissionError("onset-collar output must not replace a frozen input")
    try:
        output.relative_to(prediction_source)
    except ValueError:
        pass
    else:
        raise PermissionError(
            "onset-collar output must be outside the frozen prediction directory"
        )

    manifest = load_common17_manifest(manifest_source, require_complete=True)
    strict_metrics_receipt = json.loads(
        strict_metrics_source.read_text(encoding="utf-8")
    )
    v6_receipt = json.loads(v6_receipt_source.read_text(encoding="utf-8"))
    _validate_content_address(strict_metrics_receipt, context="frozen strict metrics")
    records = [
        row for row in manifest["records"] if row["model_split"] == "source_dev"
    ]
    record_by_id = {str(row["analysis_identity_id"]): row for row in records}
    patient_ids = {str(row["patient_id"]) for row in records}
    reference_count = sum(len(row["seizure_events"]) for row in records)
    if (
        len(records) != EXPECTED_RECORDING_COUNT
        or len(record_by_id) != EXPECTED_RECORDING_COUNT
        or len(patient_ids) != EXPECTED_PATIENT_COUNT
        or reference_count != EXPECTED_REFERENCE_EVENT_COUNT
    ):
        raise ValueError("V6 source-dev manifest denominator drifted")
    if (
        strict_metrics_receipt.get("complete_source_dev_denominator") is not True
        or int(strict_metrics_receipt.get("recording_count", -1))
        != EXPECTED_RECORDING_COUNT
        or int(strict_metrics_receipt.get("reference_event_count", -1))
        != EXPECTED_REFERENCE_EVENT_COUNT
        or strict_metrics_receipt.get("checkpoint_file_sha256")
        != expected_checkpoint_sha256
    ):
        raise ValueError("frozen strict metrics binding drifted")
    if (
        v6_receipt.get("schema_version")
        != "clinical_eeg_common17_two_level_user_validation_v6"
        or v6_receipt.get("detection", {})
        .get("training", {})
        .get("checkpoint_file_sha256")
        != expected_checkpoint_sha256
    ):
        raise ValueError("V6 validation receipt binding drifted")

    files = sorted(prediction_source.glob("*.json.gz"))
    expected_names = {f"{identity}.json.gz" for identity in record_by_id}
    actual_names = {path.name for path in files}
    if actual_names != expected_names or len(files) != EXPECTED_RECORDING_COUNT:
        raise ValueError("frozen prediction roster is missing, extra, or duplicated")

    rows: list[dict[str, Any]] = []
    inventory: list[dict[str, str]] = []
    checkpoint_steps: set[int] = set()
    for path in files:
        identity = path.name[: -len(".json.gz")]
        record = record_by_id[identity]
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        minimum_peak_threshold = payload.get("minimum_peak_threshold")
        if (
            payload.get("schema_version") != PREDICTION_SCHEMA_VERSION
            or payload.get("analysis_identity_id") != identity
            or payload.get("checkpoint_file_sha256")
            != expected_checkpoint_sha256
            or payload.get("common17_channel_order")
            != list(COMMON17_CHANNEL_ORDER)
            or payload.get("FZ_or_PZ_model_axis_present") is not False
            or isinstance(minimum_peak_threshold, bool)
            or not isinstance(minimum_peak_threshold, (int, float))
            or not math.isfinite(float(minimum_peak_threshold))
            or not 0.0 < float(minimum_peak_threshold) <= threshold
            or payload.get("smoothing_sigma_samples")
            != DEFAULT_SMOOTHING_SIGMA_SAMPLES
            or payload.get("minimum_peak_distance_seconds")
            != DEFAULT_MINIMUM_PEAK_DISTANCE_SECONDS
            or not _valid_pre_nms_candidate_cache(
                payload.get("pre_nms_candidate_cache")
            )
        ):
            raise ValueError(f"frozen prediction payload binding drifted: {identity}")
        checkpoint_steps.add(int(payload["checkpoint_global_step"]))
        duration = float(Fraction(*record["recording_duration_seconds_fraction"]))
        if not math.isclose(
            float(payload["recording_duration_seconds"]),
            duration,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"prediction/manifest duration drifted: {identity}")
        predicted_events = _decode_peaks(
            payload["peaks"],
            threshold=threshold,
            recording_duration_seconds=duration,
        )
        rows.append(
            {
                "patient_id": str(record["patient_id"]),
                "recording_id": identity,
                "split": "source_dev",
                "duration_seconds": duration,
                "reference_events": deepcopy(record["seizure_events"]),
                "predicted_events": predicted_events,
            }
        )
        inventory.append(
            {
                "analysis_identity_id": identity,
                "prediction_file_sha256": _file_sha256(path),
            }
        )
    if checkpoint_steps != {1752}:
        raise ValueError("frozen prediction global-step roster drifted")

    strict_replay = aggregate_continuous_detection_metrics(
        rows, tolerances_seconds=(1.0, 3.0, 5.0, 10.0)
    )
    onset_collar = aggregate_onset_collar_metrics(rows, collars=DEFAULT_COLLARS)
    szcore = (
        _official_timescoring_metrics(rows, project_root=ROOT)
        if include_szcore
        else None
    )
    replayed_summary = _strict_summary(strict_replay)
    frozen_summary = _frozen_strict_summary(strict_metrics_receipt)
    v6_summary = _v6_reported_strict_summary(v6_receipt)
    frozen_comparison = _comparison(replayed_summary, frozen_summary)
    v6_comparison = _comparison(replayed_summary, v6_summary)
    if not all(frozen_comparison.values()) or not all(v6_comparison.values()):
        raise RuntimeError("strict-overlap control did not reproduce frozen V6 metrics")

    receipt = _content_address(
        {
            "schema_version": (
                MULTITRACK_SCHEMA_VERSION if include_szcore else SCHEMA_VERSION
            ),
            "stage": "read_only_frozen_prediction_replay",
            "status": (
                "pass_strict_control_independent_onset_collars_and_szcore"
                if include_szcore
                else "pass_strict_control_and_independent_onset_collars"
            ),
            "threshold": threshold,
            "input_lineage": {
                "manifest_path": str(manifest_source),
                "manifest_file_sha256": _file_sha256(manifest_source),
                "manifest_receipt_sha256": manifest["receipt_sha256"],
                "prediction_directory": str(prediction_source),
                "prediction_schema_version": PREDICTION_SCHEMA_VERSION,
                "prediction_file_count": len(files),
                "prediction_inventory_sha256": hashlib.sha256(
                    _canonical_bytes(inventory)
                ).hexdigest(),
                "checkpoint_file_sha256_from_payloads": expected_checkpoint_sha256,
                "checkpoint_global_step_from_payloads": 1752,
                "strict_metrics_path": str(strict_metrics_source),
                "strict_metrics_file_sha256": _file_sha256(strict_metrics_source),
                "strict_metrics_receipt_sha256": strict_metrics_receipt[
                    "receipt_sha256"
                ],
                "v6_validation_receipt_path": str(v6_receipt_source),
                "v6_validation_receipt_file_sha256": _file_sha256(
                    v6_receipt_source
                ),
            },
            "denominator": {
                "patient_count": len(patient_ids),
                "recording_count": len(records),
                "recording_hours": onset_collar["total_recording_hours"],
                "reference_event_count": reference_count,
                "predicted_alarm_count": onset_collar["predicted_alarm_count"],
            },
            "strict_overlap_control": {
                "method_id": CONTINUOUS_BENCHMARK_METHOD_ID,
                "replayed_summary": replayed_summary,
                "frozen_metrics_summary": frozen_summary,
                "v6_reported_summary": v6_summary,
                "replayed_equals_frozen_metrics": frozen_comparison,
                "replayed_equals_v6_report": v6_comparison,
                "all_control_checks_passed": True,
            },
            "independent_onset_collar_metrics": onset_collar,
            **(
                {"szcore_compatible_metrics": szcore}
                if szcore is not None
                else {}
            ),
            "scope": {
                "EDF_opened": False,
                "checkpoint_opened": False,
                "reference_sidecar_opened": False,
                "manifest_embedded_reference_events_used_for_scoring": True,
                "frozen_prediction_files_mutated": False,
                "frozen_metrics_mutated": False,
                "new_model_inference_run": False,
                "source_eval_used": False,
                "source_dev_diagnostic_replay_only": True,
                "strict_overlap_and_onset_collar_endpoints_kept_separate": True,
                "szcore_included": include_szcore,
                "szcore_tolerance_is_not_called_onset_accuracy": True,
            },
            "receipt_sha256": _PENDING,
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_bytes(_canonical_bytes(receipt) + b"\n")
    os.replace(temporary, output)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--prediction-dir", type=Path, default=DEFAULT_PREDICTION_DIR)
    parser.add_argument("--strict-metrics", type=Path, default=DEFAULT_STRICT_METRICS)
    parser.add_argument("--v6-receipt", type=Path, default=DEFAULT_V6_RECEIPT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--expected-checkpoint-sha256",
        default=EXPECTED_CHECKPOINT_SHA256,
    )
    parser.add_argument("--include-szcore", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    receipt = replay(
        manifest_path=args.manifest,
        prediction_dir=args.prediction_dir,
        strict_metrics_path=args.strict_metrics,
        v6_receipt_path=args.v6_receipt,
        output_path=args.output,
        threshold=args.threshold,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        include_szcore=args.include_szcore,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
