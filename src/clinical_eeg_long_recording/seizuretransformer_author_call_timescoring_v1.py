"""Independent post-freeze replay of SeizureTransformer's scoring calls.

The released ``time_step_level/eval_test.py`` constructs reference and
hypothesis ``Annotation`` objects at 256 Hz, then calls
``SampleScoring(ref, hyp)`` and ``EventScoring(ref, hyp)`` without optional
arguments.  With the hash-pinned timescoring 0.0.7 authority used by this
project, those calls mean 1 Hz sample scoring and the default event parameters
30/60/0/300/90 respectively.

This module deliberately does not alter the existing unified score or its
256 Hz sample-sensitivity lane.  It validates and freezes the complete
external19 prediction inventory first, opens only the source-dev label sidecar
after that gate, and writes a separate content-addressed receipt.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

from src.clinical_eeg_long_recording.continuous_detection_benchmark import (
    validate_continuous_benchmark_rows,
)
from src.clinical_eeg_long_recording import (
    seizuretransformer_tusz_native18_runner_v1 as runner,
)
from src.lookaroundnet_native18.unified_score import _timescoring_authority


SCHEMA_VERSION: Final[str] = (
    "seizuretransformer_external19_author_call_timescoring_postfreeze_v1"
)
RECEIPT_NAME: Final[str] = "author_call_timescoring_receipt.json"
NATIVE18_SCHEMA_VERSION: Final[str] = (
    "seizuretransformer_native18_author_call_timescoring_postfreeze_v1"
)
NATIVE18_RECEIPT_NAME: Final[str] = "native18_author_call_timescoring_receipt.json"
ANNOTATION_CLOCK_HZ: Final[int] = 256
EXPECTED_SAMPLE_SCORING_HZ: Final[int] = 1
EXPECTED_EVENT_PARAMETERS: Final[dict[str, float]] = {
    "toleranceStart": 30,
    "toleranceEnd": 60,
    "minOverlap": 0,
    "maxEventDuration": 300,
    "minDurationBetweenEvents": 90,
}
_ALLOWED_OUTPUT_FILES: Final[frozenset[str]] = frozenset(
    {"prediction_freeze_gate.json", RECEIPT_NAME}
)


def _safe_rate(numerator: int | float, denominator: int | float) -> float | None:
    return None if denominator <= 0 else float(numerator) / float(denominator)


def _pooled_metrics(
    totals: Mapping[str, int], *, duration_seconds: float, unit: str
) -> dict[str, Any]:
    ref_true = int(totals["ref_true"])
    true_positive = int(totals["tp"])
    false_positive = int(totals["fp"])
    result: dict[str, Any] = {
        "reference_positive_count": ref_true,
        "true_positive_count": true_positive,
        "false_positive_count": false_positive,
        "sensitivity": _safe_rate(true_positive, ref_true),
        "precision": _safe_rate(true_positive, true_positive + false_positive),
        "f1": (
            None
            if ref_true + false_positive == 0
            else 2.0
            * true_positive
            / (2.0 * true_positive + false_positive + ref_true - true_positive)
        ),
        "aggregation_duration_seconds": float(duration_seconds),
        "false_positive_unit": unit,
    }
    if unit == "event":
        result["false_positives_per_24h"] = _safe_rate(
            false_positive, duration_seconds / 86400.0
        )
    elif unit == "sample":
        false_positive_seconds = false_positive / EXPECTED_SAMPLE_SCORING_HZ
        result.update(
            {
                "false_positive_seconds": float(false_positive_seconds),
                "false_positive_time_fraction": _safe_rate(
                    false_positive_seconds, duration_seconds
                ),
                "false_positive_seconds_per_24h": _safe_rate(
                    false_positive_seconds, duration_seconds / 86400.0
                ),
            }
        )
    else:  # pragma: no cover - internal constant only.
        raise AssertionError(unit)
    return result


def author_call_pattern_metrics(
    rows: Sequence[Mapping[str, Any]], *, project_root: str | Path
) -> dict[str, Any]:
    """Replay the released scoring call pattern with pinned timescoring 0.0.7.

    The split-only check intentionally precedes reference-event validation and
    timescoring import.  A source-eval row is therefore rejected before any of
    its reference fields can be interpreted.
    """

    if (
        not isinstance(rows, Sequence)
        or isinstance(rows, (str, bytes))
        or not rows
    ):
        raise TypeError("author-call benchmark rows must be a non-empty sequence")
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"author-call benchmark row {index} must be an object")
        if row.get("split") != "source_dev":
            raise PermissionError(
                "author-call timescoring accepts source_dev only before references"
            )

    validated = validate_continuous_benchmark_rows(rows)
    Annotation, EventScoring, SampleScoring, authority = _timescoring_authority(
        Path(project_root).resolve(strict=True)
    )
    defaults = EventScoring.Parameters()
    observed_defaults = {
        name: getattr(defaults, name) for name in EXPECTED_EVENT_PARAMETERS
    }
    if observed_defaults != EXPECTED_EVENT_PARAMETERS:
        raise PermissionError("timescoring 0.0.7 event defaults drifted")

    event_totals: Counter[str] = Counter()
    sample_totals: Counter[str] = Counter()
    event_num_samples = 0
    sample_num_samples = 0
    patient_ids: set[str] = set()
    for row in validated:
        raw_samples = float(row["duration_seconds"]) * ANNOTATION_CLOCK_HZ
        sample_count = round(raw_samples)
        if (
            sample_count < 1
            or not math.isclose(
                raw_samples, sample_count, rel_tol=0.0, abs_tol=1e-7
            )
        ):
            raise ValueError(
                "author-call recording duration is not on the exact 256 Hz clock"
            )
        references = [
            (float(event["start_seconds"]), float(event["stop_seconds"]))
            for event in row["reference_events"]
        ]
        predictions = [
            (float(event["start_seconds"]), float(event["stop_seconds"]))
            for event in row["predicted_events"]
        ]
        reference = Annotation(references, ANNOTATION_CLOCK_HZ, sample_count)
        hypothesis = Annotation(predictions, ANNOTATION_CLOCK_HZ, sample_count)

        # These omissions are intentional: they are the literal released
        # eval_test.py calls whose defaults are being audited here.
        sample_score = SampleScoring(reference, hypothesis)
        event_score = EventScoring(reference, hypothesis)
        if (
            int(sample_score.fs) != EXPECTED_SAMPLE_SCORING_HZ
            or int(event_score.fs) != 10
            or int(reference.fs) != ANNOTATION_CLOCK_HZ
            or int(hypothesis.fs) != ANNOTATION_CLOCK_HZ
        ):
            raise PermissionError("timescoring author-call rates drifted")
        sample_totals.update(
            {
                "ref_true": int(sample_score.refTrue),
                "tp": int(sample_score.tp),
                "fp": int(sample_score.fp),
            }
        )
        event_totals.update(
            {
                "ref_true": int(event_score.refTrue),
                "tp": int(event_score.tp),
                "fp": int(event_score.fp),
            }
        )
        sample_num_samples += int(sample_score.numSamples)
        event_num_samples += int(event_score.numSamples)
        patient_ids.add(str(row["patient_id"]))

    sample_duration = sample_num_samples / EXPECTED_SAMPLE_SCORING_HZ
    event_duration = event_num_samples / 10.0
    return {
        "authority": {
            **authority,
            "upstream_entrypoint": "time_step_level/eval_test.py",
            "upstream_requirements_declared_timescoring_version": "0.0.6",
            "replay_timescoring_version": "0.0.7",
            "byte_exact_author_environment_reproduction": False,
            "author_call_pattern_replayed": True,
            "annotation_clock_hz": ANNOTATION_CLOCK_HZ,
            "sample_scoring_call": "SampleScoring(reference,hypothesis)",
            "effective_sample_scoring_rate_hz": EXPECTED_SAMPLE_SCORING_HZ,
            "event_scoring_call": "EventScoring(reference,hypothesis)",
            "event_parameters_from_unmodified_defaults": observed_defaults,
            "aggregation": "pooled_tp_fp_refTrue_and_numSamples_across_records",
        },
        "record_count": len(validated),
        "patient_count": len(patient_ids),
        "event_pooled": _pooled_metrics(
            event_totals, duration_seconds=event_duration, unit="event"
        ),
        "sample_1hz_pooled": _pooled_metrics(
            sample_totals, duration_seconds=sample_duration, unit="sample"
        ),
        "event_aggregation_num_samples_at_10hz": event_num_samples,
        "sample_aggregation_num_samples_at_1hz": sample_num_samples,
    }


def _require_independent_output(output: Path) -> None:
    if output.is_symlink():
        raise PermissionError("author-call score output may not be a symlink")
    if output.exists() and not output.is_dir():
        raise PermissionError("author-call score output must be a directory")
    if output.is_dir():
        unexpected = {
            child.name for child in output.iterdir() if child.name not in _ALLOWED_OUTPUT_FILES
        }
        if unexpected:
            raise PermissionError(
                "author-call score requires an independent output directory"
            )


def _require_dev_only_sidecar(path: str | Path) -> Path:
    source = Path(path)
    if source.is_symlink():
        raise PermissionError("source-dev label sidecar may not be a symlink")
    if source.name != "source_dev.labels.jsonl":
        raise PermissionError(
            "author-call score accepts only the isolated source_dev label sidecar"
        )
    return source.resolve(strict=True)


def score_external19_frozen_source_dev(
    *,
    prediction_manifest_path: str | Path,
    reference_sidecar_path: str | Path,
    output_dir: str | Path,
    project_root: str | Path,
) -> dict[str, Any]:
    """Write the independent dual-denominator author-call-pattern receipt."""

    prediction_source = Path(prediction_manifest_path).resolve(strict=True)
    output = Path(output_dir).resolve(strict=False)
    _require_independent_output(output)

    # Prediction-only validation replays the complete manifest, every row,
    # every posterior hash and the decoded events before output or references.
    prediction, roster, prediction_file_sha = (
        runner._validate_external19_frozen_prediction_manifest(prediction_source)
    )
    output.mkdir(parents=True, exist_ok=True)
    gate = runner._materialize_prediction_freeze_gate(
        output_dir=output,
        prediction_manifest=prediction,
        prediction_manifest_file_sha256=prediction_file_sha,
        roster=roster,
    )

    # This is intentionally the first reference-bearing read.
    reference_source = _require_dev_only_sidecar(reference_sidecar_path)
    reference = runner._load_source_dev_reference_manifest(
        reference_source, roster=roster
    )
    reference_by_identity = {
        str(row["analysis_identity_id"]): row for row in reference["records"]
    }
    prediction_by_identity = {
        str(row["analysis_identity_id"]): row
        for row in prediction["prediction_rows"]
    }
    roster_identities = {
        str(row["analysis_identity_id"]) for row in roster["records"]
    }
    if (
        len(roster_identities) != 1821
        or set(reference_by_identity) != roster_identities
        or set(prediction_by_identity) != roster_identities
    ):
        raise PermissionError("author-call source-dev roster binding drifted")

    rows: list[dict[str, Any]] = []
    for identity_row in roster["records"]:
        identity = str(identity_row["analysis_identity_id"])
        reference_row = reference_by_identity[identity]
        prediction_row = prediction_by_identity[identity]
        target_samples = int(identity_row["target_sample_count_256hz"])
        if int(reference_row["target_sample_count_256hz"]) != target_samples:
            raise PermissionError("author-call reference duration drifted")
        rows.append(
            {
                "patient_id": str(identity_row["local_patient_id"]),
                "recording_id": identity,
                "split": "source_dev",
                "duration_seconds": target_samples / ANNOTATION_CLOCK_HZ,
                "reference_events": runner._merge_reference_events_at_native_clock(
                    reference_row
                ),
                "predicted_events": [
                    {
                        "start_seconds": float(event["start_seconds"]),
                        "stop_seconds": float(event["stop_seconds"]),
                    }
                    for event in prediction_row["predicted_events"]
                ],
            }
        )

    complete_status = "external19_prediction_complete"
    signal_rows = [
        row
        for row, prediction_row in zip(
            rows, prediction["prediction_rows"], strict=True
        )
        if prediction_row["status"] == complete_status
    ]
    status_counts = Counter(
        str(row["status"]) for row in prediction["prediction_rows"]
    )
    signal_metrics = author_call_pattern_metrics(
        signal_rows, project_root=project_root
    )
    full_metrics = author_call_pattern_metrics(rows, project_root=project_root)
    receipt = runner._content_address(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "completed_external19_source_dev_author_call_postfreeze_score",
            "claim_boundary": (
                "external19 artifact diagnostic on source-dev; timescoring 0.0.7 "
                "replay of SeizureTransformer's released scoring call pattern; "
                "not paper-native18, not the author's 0.0.6 environment, and not "
                "an independent test estimate"
            ),
            "prediction_freeze_gate": gate,
            "reference_opened_only_after_complete_prediction_replay_and_gate": True,
            "prediction_manifest_path": str(prediction_source),
            "prediction_manifest_file_sha256": prediction_file_sha,
            "prediction_manifest_receipt_sha256": prediction["receipt_sha256"],
            "reference_sidecar_path": str(reference_source),
            "reference_sidecar_file_sha256": runner._file_sha256(reference_source),
            "reference_adapter_receipt_sha256": reference["receipt_sha256"],
            "prediction_status_counts": dict(sorted(status_counts.items())),
            "metric_denominator_lanes": {
                "public_get_data_signal_side_evaluable": {
                    "admission": (
                        "external19 prediction-complete records only; released "
                        "short-record skips and typed missing-axis failures excluded"
                    ),
                    "record_count": len(signal_rows),
                    "excluded_record_count": len(rows) - len(signal_rows),
                    "metrics": signal_metrics,
                },
                "full_intention_to_assess_1821": {
                    "admission": (
                        "all frozen source-dev records; released short-record skips "
                        "and typed failures retained as zero alarms"
                    ),
                    "record_count": len(rows),
                    "zero_alarm_failure_and_skip_count": (
                        status_counts["external19_upstream_skip_below_60s"]
                        + status_counts["external19_typed_technical_failure"]
                    ),
                    "metrics": full_metrics,
                },
            },
            "top_level_metric_alias": "full_intention_to_assess_1821",
            "existing_unified_256hz_sample_lane_modified": False,
            "source_eval_opened": False,
            "clinical_use_authorized": False,
            "receipt_sha256": runner._PENDING,
        }
    )
    return runner._install_or_replay_json(
        output / RECEIPT_NAME,
        receipt,
        context="external19 author-call timescoring receipt",
    )


def _require_native18_independent_output(output: Path) -> None:
    allowed = {"prediction_freeze_gate.json", NATIVE18_RECEIPT_NAME}
    if output.is_symlink():
        raise PermissionError("native18 author-call score output may not be a symlink")
    if output.exists() and not output.is_dir():
        raise PermissionError("native18 author-call score output must be a directory")
    if output.is_dir():
        unexpected = {child.name for child in output.iterdir()} - allowed
        if unexpected:
            raise PermissionError(
                "native18 author-call score requires an independent output directory"
            )


def score_native18_frozen_source_dev(
    *,
    prediction_manifest_path: str | Path,
    reference_sidecar_path: str | Path,
    output_dir: str | Path,
    project_root: str | Path,
) -> dict[str, Any]:
    """Score a completed fixed-epoch-100 native18 source-dev inventory.

    This wrapper is separate from the external19 diagnostic wrapper and emits
    a different schema and filename.  It validates both the frozen prediction
    inventory and the inference-eligible epoch-100 checkpoint before writing
    the prediction gate or opening the development reference sidecar.
    """

    prediction_source = Path(prediction_manifest_path).resolve(strict=True)
    output = Path(output_dir).resolve(strict=False)
    _require_native18_independent_output(output)

    prediction, roster, prediction_file_sha = (
        runner._validate_frozen_prediction_manifest(prediction_source)
    )
    checkpoint_path = Path(str(prediction["checkpoint_path"])).resolve(strict=True)
    model, checkpoint, checkpoint_sha = runner.load_epoch100_native18_checkpoint(
        checkpoint_path, device_name="cpu"
    )
    del model
    status_counts = Counter(
        str(row["status"]) for row in prediction["prediction_rows"]
    )
    if (
        prediction.get("claim_status") != "source_dev_native18_post_epoch100"
        or prediction.get("checkpoint_completed_epoch_count") != 100
        or checkpoint_sha != prediction.get("checkpoint_sha256")
        or checkpoint.get("checkpoint_role")
        != "epoch100_primary_inference_eligible"
        or checkpoint.get("inference_eligible") is not True
        or checkpoint.get("training_complete") is not True
        or checkpoint.get("completed_epoch_count") != 100
        or checkpoint.get("source_dev_opened") is not False
        or checkpoint.get("source_eval_opened") is not False
        or dict(sorted(status_counts.items()))
        != {"prediction_complete": 1606, "upstream_skip_below_60s": 215}
    ):
        raise PermissionError(
            "native18 author-call input is not the fixed epoch100 frozen inventory"
        )
    checkpoint_verification = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_role": checkpoint["checkpoint_role"],
        "training_complete": checkpoint["training_complete"],
        "inference_eligible": checkpoint["inference_eligible"],
        "completed_epoch_count": checkpoint["completed_epoch_count"],
        "source_dev_opened_during_training": checkpoint["source_dev_opened"],
        "source_eval_opened_during_training": checkpoint["source_eval_opened"],
    }
    # The checkpoint also contains model and optimizer tensors.  They are no
    # longer needed after the eligibility gate and should not stay resident
    # while full-record annotation masks are scored on CPU.
    del checkpoint

    output.mkdir(parents=True, exist_ok=True)
    gate = runner._materialize_prediction_freeze_gate(
        output_dir=output,
        prediction_manifest=prediction,
        prediction_manifest_file_sha256=prediction_file_sha,
        roster=roster,
    )

    # This is intentionally the first reference-bearing read.
    reference_source = _require_dev_only_sidecar(reference_sidecar_path)
    reference = runner._load_source_dev_reference_manifest(
        reference_source, roster=roster
    )
    reference_by_identity = {
        str(row["analysis_identity_id"]): row for row in reference["records"]
    }
    prediction_by_identity = {
        str(row["analysis_identity_id"]): row
        for row in prediction["prediction_rows"]
    }
    roster_identities = {
        str(row["analysis_identity_id"]) for row in roster["records"]
    }
    if (
        len(roster_identities) != 1821
        or set(reference_by_identity) != roster_identities
        or set(prediction_by_identity) != roster_identities
    ):
        raise PermissionError("native18 author-call source-dev roster binding drifted")

    rows: list[dict[str, Any]] = []
    for identity_row in roster["records"]:
        identity = str(identity_row["analysis_identity_id"])
        reference_row = reference_by_identity[identity]
        prediction_row = prediction_by_identity[identity]
        target_samples = int(identity_row["target_sample_count_256hz"])
        if int(reference_row["target_sample_count_256hz"]) != target_samples:
            raise PermissionError("native18 author-call reference duration drifted")
        rows.append(
            {
                "patient_id": str(identity_row["local_patient_id"]),
                "recording_id": identity,
                "split": "source_dev",
                "duration_seconds": target_samples / ANNOTATION_CLOCK_HZ,
                "reference_events": runner._merge_reference_events_at_native_clock(
                    reference_row
                ),
                "predicted_events": [
                    {
                        "start_seconds": float(event["start_seconds"]),
                        "stop_seconds": float(event["stop_seconds"]),
                    }
                    for event in prediction_row["predicted_events"]
                ],
            }
        )

    signal_rows = [
        row
        for row, prediction_row in zip(
            rows, prediction["prediction_rows"], strict=True
        )
        if prediction_row["status"] == "prediction_complete"
    ]
    if len(signal_rows) != 1606 or len(rows) != 1821:
        raise PermissionError("native18 author-call denominator drifted")
    signal_metrics = author_call_pattern_metrics(
        signal_rows, project_root=project_root
    )
    full_metrics = author_call_pattern_metrics(rows, project_root=project_root)
    receipt = runner._content_address(
        {
            "schema_version": NATIVE18_SCHEMA_VERSION,
            "status": "completed_native18_source_dev_author_call_postfreeze_score",
            "claim_boundary": (
                "fixed epoch100 TUSZ native18 clean-room architecture and "
                "preprocessing path on source-dev; timescoring 0.0.7 replay of "
                "SeizureTransformer's released scoring call pattern; not the "
                "author's issued checkpoint or proven bit-equivalent training "
                "roster, not the author's 0.0.6 environment, and not an "
                "independent test estimate"
            ),
            "prediction_freeze_gate": gate,
            "reference_opened_only_after_epoch100_prediction_replay_and_gate": True,
            "prediction_manifest_path": str(prediction_source),
            "prediction_manifest_file_sha256": prediction_file_sha,
            "prediction_manifest_receipt_sha256": prediction["receipt_sha256"],
            "epoch100_checkpoint_verification": checkpoint_verification,
            "reference_sidecar_path": str(reference_source),
            "reference_sidecar_file_sha256": runner._file_sha256(reference_source),
            "reference_adapter_receipt_sha256": reference["receipt_sha256"],
            "prediction_status_counts": dict(sorted(status_counts.items())),
            "metric_denominator_lanes": {
                "released_signal_side_evaluable_ge60s": {
                    "admission": (
                        "the 1,606 frozen native18 prediction-complete records at "
                        "least 60 seconds; released short-record skips excluded"
                    ),
                    "record_count": len(signal_rows),
                    "excluded_record_count": len(rows) - len(signal_rows),
                    "metrics": signal_metrics,
                },
                "full_intention_to_assess_native18": {
                    "admission": (
                        "all 1,821 frozen source-dev records; the 215 released "
                        "short-record skips retain references and zero alarms"
                    ),
                    "record_count": len(rows),
                    "zero_alarm_short_record_skip_count": status_counts[
                        "upstream_skip_below_60s"
                    ],
                    "metrics": full_metrics,
                },
            },
            "top_level_metric_alias": "full_intention_to_assess_native18",
            "paper_native18_architecture_and_preprocessing_path": True,
            "paper_training_roster_equivalence_proven": False,
            "paper_training_checkpoint_reproduced": False,
            "source_dev_selected_epoch": False,
            "source_dev_selected_threshold": False,
            "existing_unified_256hz_sample_lane_modified": False,
            "external19_receipt_or_semantics_modified": False,
            "source_eval_opened": False,
            "clinical_use_authorized": False,
            "receipt_sha256": runner._PENDING,
        }
    )
    return runner._install_or_replay_json(
        output / NATIVE18_RECEIPT_NAME,
        receipt,
        context="native18 author-call timescoring receipt",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Post-freeze SeizureTransformer author-call timescoring 0.0.7 replay"
        )
    )
    parser.add_argument(
        "--profile", choices=("external19", "native18"), default="external19"
    )
    parser.add_argument("--prediction-manifest", required=True)
    parser.add_argument("--reference-sidecar", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--project-root", default=str(Path(__file__).resolve().parents[2])
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    scorer = (
        score_native18_frozen_source_dev
        if arguments.profile == "native18"
        else score_external19_frozen_source_dev
    )
    result = scorer(
        prediction_manifest_path=arguments.prediction_manifest,
        reference_sidecar_path=arguments.reference_sidecar,
        output_dir=arguments.output_dir,
        project_root=arguments.project_root,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ANNOTATION_CLOCK_HZ",
    "EXPECTED_SAMPLE_SCORING_HZ",
    "NATIVE18_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "author_call_pattern_metrics",
    "main",
    "score_external19_frozen_source_dev",
    "score_native18_frozen_source_dev",
]
