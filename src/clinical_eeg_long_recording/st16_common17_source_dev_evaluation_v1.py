"""Post-freeze source-development evaluation for the exploratory ST16 model.

The dense prediction inventory is validated and byte-replayed before the TUSZ
TERM reference manifest is opened.  This module deliberately has no
``source_eval`` entry point.  It reports two non-interchangeable event tracks:

* strict, zero-dilation, ordered one-to-one event/onset metrics; and
* the pinned official ``timescoring`` 0.0.7 SzCORE-compatible metrics.

The decoder is intentionally small and auditable: the 256 Hz dense posterior is
averaged into non-overlapping physical one-second bins, thresholded, and runs
shorter than two seconds are discarded.  Threshold selection is performed only
after the prediction inventory is frozen and is explicitly an in-sample
source-development result, not an independent test result.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time
from typing import Any, Final, Mapping, Sequence

import numpy as np

from .continuous_detection_benchmark import (
    aggregate_continuous_detection_metrics,
)
from .eventnet_common17_streaming_v1 import load_common17_manifest
from .tusz_canonical_physical_signal_audit_v1 import (
    validate_tusz_canonical_physical_analysis_projection_v1,
)
from .tusz_complete_detector_roster_v2 import (
    validate_tusz_analysis_identity_projection_v2,
)


SCHEMA_VERSION: Final[str] = "st16_common17_source_dev_evaluation_v1"
TARGET_FS_HZ: Final[int] = 256
DECODER_BIN_SECONDS: Final[float] = 1.0
MINIMUM_EVENT_SECONDS: Final[float] = 2.0
TIMESCORING_COMMIT: Final[str] = "426f8d2b77974641dc9db71884e0812b249ba93b"
TIMESCORING_VERSION: Final[str] = "0.0.7"
_PENDING: Final[str] = "pending"
_PREDICTION_PENDING: Final[str] = "CONTENT-ADDRESS-PENDING"
ST16_MINIMUM_ANALYZABLE_SECONDS: Final[float] = 60.0


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


def _content_address(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(value))
    if result.get("receipt_sha256") != _PENDING:
        raise ValueError("content-addressed payload requires pending receipt")
    result["receipt_sha256"] = _canonical_sha256(
        {**result, "receipt_sha256": _PENDING}
    )
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_content_address(
    value: Mapping[str, Any], *, pending: str, context: str
) -> None:
    supplied = value.get("receipt_sha256")
    replay = deepcopy(dict(value))
    replay["receipt_sha256"] = pending
    if (
        not isinstance(supplied, str)
        or len(supplied) != 64
        or any(character not in "0123456789abcdef" for character in supplied)
        or supplied != _canonical_sha256(replay)
    ):
        raise ValueError(f"{context} is not content-addressed")


def _nonnegative_finite(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{context} must be finite and non-negative")
    return result


def _write_json_atomic(path: Path, value: object, *, replace: bool) -> None:
    target = path.resolve(strict=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not replace and (target.exists() or target.is_symlink()):
        raise FileExistsError(target)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with open(descriptor, "w", encoding="utf-8", closefd=True) as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()


def default_thresholds() -> tuple[float, ...]:
    """Return a frozen source-dev threshold grid containing paper threshold 0.8."""

    values = {0.001, 0.002, 0.005, 0.995, 0.998, 0.999, 0.8}
    values.update(round(index / 100.0, 2) for index in range(1, 100))
    return tuple(sorted(values))


def posterior_to_one_hz_mean(probability: np.ndarray) -> np.ndarray:
    """Average every physical 256 Hz second without dropping the final tail."""

    values = np.asarray(probability)
    if values.ndim != 1 or values.dtype != np.dtype("float32"):
        raise TypeError("dense ST16 probability must be one-dimensional float32")
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("dense ST16 probability must be finite and non-empty")
    if float(values.min()) < 0.0 or float(values.max()) > 1.0:
        raise ValueError("dense ST16 probability lies outside [0,1]")
    starts = np.arange(0, len(values), TARGET_FS_HZ, dtype=np.int64)
    sums = np.add.reduceat(values.astype(np.float64, copy=False), starts)
    stops = np.minimum(starts + TARGET_FS_HZ, len(values))
    counts = stops - starts
    return np.asarray(sums / counts, dtype=np.float32)


def decode_one_hz_runs(
    score: np.ndarray,
    *,
    threshold: float,
    duration_seconds: float,
    minimum_event_seconds: float = MINIMUM_EVENT_SECONDS,
) -> list[dict[str, float]]:
    """Decode contiguous one-second posterior runs with a two-second floor."""

    values = np.asarray(score)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("one-Hz posterior score is invalid")
    if not 0.0 < float(threshold) < 1.0:
        raise ValueError("threshold must lie strictly within (0,1)")
    if duration_seconds <= 0 or minimum_event_seconds <= 0:
        raise ValueError("decoder durations must be positive")
    positive = values >= float(threshold)
    transitions = np.diff(
        np.concatenate(
            (np.asarray([False]), positive, np.asarray([False]))
        ).astype(np.int8)
    )
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1)
    events: list[dict[str, float]] = []
    for start_bin, stop_bin in zip(starts.tolist(), stops.tolist(), strict=True):
        start = float(start_bin) * DECODER_BIN_SECONDS
        stop = min(float(stop_bin) * DECODER_BIN_SECONDS, float(duration_seconds))
        if stop - start + 1e-12 >= minimum_event_seconds:
            events.append({"start_seconds": start, "stop_seconds": stop})
    return events


def _safe_rate(numerator: float, denominator: float) -> float | None:
    return None if denominator <= 0 else float(numerator) / float(denominator)


def _count_metrics(
    *, ref_true: int, true_positive: int, false_positive: int, seconds: float
) -> dict[str, Any]:
    sensitivity = _safe_rate(true_positive, ref_true)
    precision = _safe_rate(true_positive, true_positive + false_positive)
    f1 = (
        None
        if ref_true + false_positive == 0
        else 2.0
        * true_positive
        / (2.0 * true_positive + false_positive + ref_true - true_positive)
    )
    return {
        "reference_positive_count": int(ref_true),
        "true_positive_count": int(true_positive),
        "false_positive_count": int(false_positive),
        "sensitivity": sensitivity,
        "precision": precision,
        "f1": f1,
        "false_positives_per_24h": (
            None if seconds <= 0 else false_positive / (seconds / 86400.0)
        ),
    }


def _timescoring_authority(project_root: Path):
    vendor = (
        project_root
        / "third_party/epilepsy_performance_metrics_426f8d2b"
    ).resolve(strict=True)
    head = (vendor / ".git/HEAD").read_text(encoding="utf-8").strip()
    if head != TIMESCORING_COMMIT:
        raise PermissionError("pinned timescoring checkout commit drifted")
    source = vendor / "src"
    import sys

    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    import timescoring.annotations as annotation_module  # type: ignore
    import timescoring.scoring as scoring_module  # type: ignore
    from timescoring.annotations import Annotation  # type: ignore
    from timescoring.scoring import EventScoring, SampleScoring  # type: ignore

    if (
        Path(str(annotation_module.__file__)).resolve(strict=True)
        != (source / "timescoring/annotations.py").resolve(strict=True)
        or Path(str(scoring_module.__file__)).resolve(strict=True)
        != (source / "timescoring/scoring.py").resolve(strict=True)
    ):
        raise PermissionError("runtime timescoring import does not use pinned source")

    return Annotation, EventScoring, SampleScoring, {
        "repository": "https://github.com/esl-epfl/epilepsy_performance_metrics",
        "commit": TIMESCORING_COMMIT,
        "version": TIMESCORING_VERSION,
        "scoring_source_sha256": _file_sha256(source / "timescoring/scoring.py"),
        "annotation_source_sha256": _file_sha256(
            source / "timescoring/annotations.py"
        ),
        "event_parameters": {
            "toleranceStart": 30,
            "toleranceEnd": 60,
            "minOverlap": 0,
            "maxEventDuration": 300,
            "minDurationBetweenEvents": 90,
        },
        "one_hz_record_tail_policy": "ceil_to_preserve_observed_partial_second",
    }


def _official_timescoring_metrics(
    rows: Sequence[Mapping[str, Any]], *, project_root: Path
) -> dict[str, Any]:
    Annotation, EventScoring, SampleScoring, authority = _timescoring_authority(
        project_root
    )
    parameters = EventScoring.Parameters(
        toleranceStart=30,
        toleranceEnd=60,
        minOverlap=0,
        maxEventDuration=300,
        minDurationBetweenEvents=90,
    )
    event_totals = Counter()
    sample_totals = Counter()
    patient_event_totals: dict[str, Counter[str]] = defaultdict(Counter)
    total_seconds = 0.0
    for row in rows:
        duration = float(row["duration_seconds"])
        annotation_samples = max(1, int(math.ceil(duration)))
        references = [
            (float(event["start_seconds"]), float(event["stop_seconds"]))
            for event in row["reference_events"]
        ]
        predictions = [
            (float(event["start_seconds"]), float(event["stop_seconds"]))
            for event in row["predicted_events"]
        ]
        reference = Annotation(references, 1, annotation_samples)
        hypothesis = Annotation(predictions, 1, annotation_samples)
        event_score = EventScoring(reference, hypothesis, parameters)
        sample_score = SampleScoring(reference, hypothesis, fs=1)
        for name, value in (
            ("ref_true", event_score.refTrue),
            ("tp", event_score.tp),
            ("fp", event_score.fp),
        ):
            event_totals[name] += int(value)
            patient_event_totals[str(row["patient_id"])][name] += int(value)
        for name, value in (
            ("ref_true", sample_score.refTrue),
            ("tp", sample_score.tp),
            ("fp", sample_score.fp),
        ):
            sample_totals[name] += int(value)
        patient_event_totals[str(row["patient_id"])]["seconds"] += annotation_samples
        total_seconds += annotation_samples
    event = _count_metrics(
        ref_true=event_totals["ref_true"],
        true_positive=event_totals["tp"],
        false_positive=event_totals["fp"],
        seconds=total_seconds,
    )
    sample = _count_metrics(
        ref_true=sample_totals["ref_true"],
        true_positive=sample_totals["tp"],
        false_positive=sample_totals["fp"],
        seconds=total_seconds,
    )
    patient_rows = [
        _count_metrics(
            ref_true=totals["ref_true"],
            true_positive=totals["tp"],
            false_positive=totals["fp"],
            seconds=totals["seconds"],
        )
        for totals in patient_event_totals.values()
    ]
    patient_macro = {
        name: (
            None
            if not [row[name] for row in patient_rows if row[name] is not None]
            else float(
                np.mean(
                    [row[name] for row in patient_rows if row[name] is not None]
                )
            )
        )
        for name in ("sensitivity", "precision", "f1", "false_positives_per_24h")
    }
    return {
        "authority": authority,
        "event_pooled": event,
        "event_patient_macro": patient_macro,
        "sample_1hz_pooled": sample,
    }


def _binary_ranking_metrics(target: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(target, dtype=np.bool_)
    values = np.asarray(score, dtype=np.float64)
    if labels.ndim != 1 or values.shape != labels.shape or not len(labels):
        raise ValueError("binary ranking carriers are invalid")
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if not positives or not negatives:
        return {
            "sample_count": len(labels),
            "positive_count": positives,
            "negative_count": negatives,
            "AUROC": None,
            "average_precision": None,
        }
    order = np.argsort(-values, kind="stable")
    sorted_labels = labels[order].astype(np.int64)
    sorted_scores = values[order]
    group_end = np.flatnonzero(
        np.concatenate((sorted_scores[1:] != sorted_scores[:-1], [True]))
    )
    true_positive = np.cumsum(sorted_labels)[group_end]
    false_positive = (group_end + 1) - true_positive
    tpr = np.concatenate(([0.0], true_positive / positives, [1.0]))
    fpr = np.concatenate(([0.0], false_positive / negatives, [1.0]))
    auroc = float(np.trapz(tpr, fpr))
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / positives
    recall_before = np.concatenate(([0.0], recall[:-1]))
    average_precision = float(np.sum((recall - recall_before) * precision))
    return {
        "sample_count": len(labels),
        "positive_count": positives,
        "negative_count": negatives,
        "AUROC": auroc,
        "average_precision": average_precision,
    }


def _validate_prediction_manifest(
    path: Path,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    source = path.resolve(strict=True)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version")
        != "st16_common17_source_dev_dense_prediction_inventory_v1"
        or payload.get("provider_id")
        != "st16_common17_exploratory_nonpromotable_v1"
        or payload.get("claim_status") != "exploratory_nonpromotable"
        or payload.get("split") != "source_dev"
        or payload.get("source_eval_opened") is not False
        or payload.get("reference_annotation_or_target_opened") is not False
        or payload.get("complete_prediction_inventory") is not True
        or payload.get("source_dev_metrics_authorized_only_after_inventory_freeze")
        is not True
        or payload.get("source_eval_metrics_authorized") is not False
        or payload.get("architecture_promotable") is not False
        or payload.get("maximum_records_smoke_limit") is not None
        or payload.get("pre_threshold_dense_sidecar_for_every_success") is not True
        or payload.get("threshold_morphology_hysteresis_or_NMS_applied")
        is not False
        or payload.get("materialized_record_count")
        != payload.get("full_expected_record_count")
    ):
        raise PermissionError("ST16 source-dev prediction inventory is not frozen")
    _verify_content_address(
        payload,
        pending=_PREDICTION_PENDING,
        context="ST16 prediction manifest",
    )
    checkpoint_sha256 = payload.get("checkpoint_sha256")
    if (
        not isinstance(checkpoint_sha256, str)
        or len(checkpoint_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in checkpoint_sha256
        )
    ):
        raise ValueError("ST16 prediction checkpoint SHA-256 is invalid")
    checkpoint_path = Path(str(payload.get("checkpoint_path"))).resolve(strict=True)
    if _file_sha256(checkpoint_path) != checkpoint_sha256:
        raise ValueError("ST16 prediction checkpoint failed byte replay")
    if (
        isinstance(payload.get("checkpoint_next_epoch"), bool)
        or not isinstance(payload.get("checkpoint_next_epoch"), int)
        or payload["checkpoint_next_epoch"] < 1
        or isinstance(payload.get("inference_batch_size"), bool)
        or not isinstance(payload.get("inference_batch_size"), int)
        or not 1 <= payload["inference_batch_size"] <= 8
    ):
        raise ValueError("ST16 prediction checkpoint/batch contract drifted")
    projection_path = Path(str(payload.get("analysis_projection_path"))).resolve(
        strict=True
    )
    if _file_sha256(projection_path) != payload.get(
        "analysis_projection_file_sha256"
    ):
        raise ValueError("ST16 target-free analysis projection failed byte replay")
    projection = validate_tusz_analysis_identity_projection_v2(
        json.loads(projection_path.read_text(encoding="utf-8"))
    )
    if projection["receipt_sha256"] != payload.get(
        "analysis_projection_receipt_sha256"
    ):
        raise ValueError("ST16 target-free analysis projection receipt drifted")
    projected_source_dev = {
        str(row["analysis_identity_id"]): row
        for row in projection["records"]
        if row["model_split"] == "source_dev"
    }
    rows = payload.get("prediction_rows")
    if not isinstance(rows, list) or len(rows) != payload["full_expected_record_count"]:
        raise ValueError("ST16 prediction roster is incomplete")
    if len(rows) != len(projected_source_dev):
        raise ValueError("ST16 prediction roster does not cover target-free source-dev")
    identities: set[str] = set()
    status_counts = Counter()
    forbidden_fields = {
        "seizure_events",
        "reference_events",
        "reference_csv_bi_sha256",
        "annotation",
        "doctor_text",
        "excel",
    }
    for index, row in enumerate(rows):
        if (
            not isinstance(row, Mapping)
            or forbidden_fields.intersection(row)
            or row.get("schema_version")
            != "st16_common17_source_dev_dense_prediction_row_v1"
            or row.get("provider_id")
            != "st16_common17_exploratory_nonpromotable_v1"
            or row.get("claim_status") != "exploratory_nonpromotable"
            or row.get("prediction_roster_index") != index
            or row.get("model_split") != "source_dev"
            or row.get("source_eval_opened") is not False
            or row.get("reference_annotation_or_target_opened") is not False
            or row.get("threshold_morphology_hysteresis_or_NMS_applied")
            is not False
            or row.get("checkpoint_sha256") != checkpoint_sha256
            or row.get("analysis_projection_receipt_sha256")
            != projection["receipt_sha256"]
            or row.get("status")
            not in {"dense_prediction_complete", "typed_technical_failure"}
        ):
            raise PermissionError("ST16 prediction row lineage drifted")
        _verify_content_address(
            row,
            pending=_PREDICTION_PENDING,
            context=f"ST16 prediction row {index}",
        )
        identity = str(row.get("analysis_identity_id"))
        expected_projection_row = projected_source_dev.get(identity)
        if (
            not identity
            or identity in identities
            or expected_projection_row is None
            or row.get("recording_id")
            != expected_projection_row["local_edf_path"]
            or row.get("patient_id")
            != expected_projection_row["local_patient_id"]
        ):
            raise ValueError("ST16 prediction identity roster is invalid")
        identities.add(identity)
        status_counts[str(row["status"])] += 1
        _nonnegative_finite(
            row.get("wall_seconds"), context=f"prediction row {index} wall time"
        )
        if row["status"] == "dense_prediction_complete":
            sidecar = Path(str(row["dense_probability_path"])).resolve(strict=True)
            if _file_sha256(sidecar) != row.get("dense_probability_sha256"):
                raise ValueError("ST16 dense posterior failed byte replay")
            values = np.load(sidecar, mmap_mode="r", allow_pickle=False)
            if (
                values.ndim != 1
                or len(values) < 1
                or values.dtype != np.dtype("float32")
                or isinstance(row.get("sample_count"), bool)
                or not isinstance(row.get("sample_count"), int)
                or row["sample_count"] != len(values)
                or row.get("sampling_rate_hz") != TARGET_FS_HZ
                or row.get("dense_probability_dtype") != "float32"
                or row.get("pre_threshold_dense_complete") is not True
                or row.get("pre_NMS_candidate_information_complete") is not True
                or row.get("OLA_coverage_receipt", {}).get(
                    "complete_record_posterior_coverage"
                )
                is not True
                or row.get("OLA_coverage_receipt", {}).get(
                    "uncovered_record_sample_count"
                )
                != 0
                or not np.isfinite(values).all()
                or float(values.min()) < 0.0
                or float(values.max()) > 1.0
            ):
                raise ValueError("ST16 dense posterior contract drifted")
        elif (
            not isinstance(row.get("failure_type"), str)
            or not row["failure_type"]
            or not isinstance(row.get("failure_message"), str)
            or row.get("dense_probability_path") is not None
            or row.get("dense_probability_sha256") is not None
            or row.get("pre_threshold_dense_complete") is not False
            or row.get("pre_NMS_candidate_information_complete") is not False
        ):
            raise ValueError("ST16 typed technical failure contract drifted")
    if identities != set(projected_source_dev):
        raise ValueError("ST16 target-free source-dev identity closure failed")
    if (
        status_counts["dense_prediction_complete"]
        != payload.get("dense_prediction_complete_count")
        or status_counts["typed_technical_failure"]
        != payload.get("typed_technical_failure_count")
        or sum(status_counts.values()) != len(rows)
    ):
        raise ValueError("ST16 prediction status accounting drifted")
    return payload, _file_sha256(source), projection


def _materialize_freeze_gate(
    *,
    output_dir: Path,
    prediction_manifest: Mapping[str, Any],
    manifest_sha256: str,
    canonical_projection: Mapping[str, Any],
    canonical_projection_sha256: str,
    roster_accounting: Mapping[str, Any],
) -> dict[str, Any]:
    gate = _content_address(
        {
            "schema_version": "st16_source_dev_post_prediction_freeze_gate_v2",
            "status": "prediction_inventory_frozen_reference_join_now_authorized",
            "prediction_manifest_file_sha256": manifest_sha256,
            "prediction_manifest_receipt_sha256": prediction_manifest["receipt_sha256"],
            "checkpoint_sha256": prediction_manifest["checkpoint_sha256"],
            "prediction_intention_to_evaluate_record_count": prediction_manifest[
                "materialized_record_count"
            ],
            "canonical_metric_record_count": roster_accounting[
                "canonical_metric_record_count"
            ],
            "zero_weight_noncanonical_prediction_count": roster_accounting[
                "zero_weight_noncanonical_prediction_count"
            ],
            "canonical_physical_projection_receipt_sha256": (
                canonical_projection["receipt_sha256"]
            ),
            "canonical_physical_projection_file_sha256": (
                canonical_projection_sha256
            ),
            "source_eval_opened": False,
            "reference_opened_before_this_gate": False,
            "receipt_sha256": _PENDING,
        }
    )
    path = output_dir / "post_prediction_freeze_gate.json"
    if path.is_file() and not path.is_symlink():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != gate:
            raise PermissionError("existing post-prediction freeze gate drifted")
    else:
        _write_json_atomic(path, gate, replace=False)
    return gate


def _canonical_metric_projection(
    *,
    path: Path,
    prediction_manifest: Mapping[str, Any],
    source_projection: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, Any], str]:
    """Validate the target-free 1,832-to-1,821 physical roster projection.

    The concrete counts are data properties, not hard-coded constants: every
    prediction remains in the prediction/intention-to-evaluate inventory, while
    only one validated canonical physical identity receives metric weight.
    """

    source = path.resolve(strict=True)
    physical = validate_tusz_canonical_physical_analysis_projection_v1(
        json.loads(source.read_text(encoding="utf-8"))
    )
    binding = physical["source_binding"]
    if (
        binding["source_analysis_projection_receipt_sha256"]
        != source_projection["receipt_sha256"]
        or binding["source_analysis_projection_receipt_sha256"]
        != prediction_manifest["analysis_projection_receipt_sha256"]
        or physical["projection_inventory"]["analysis_quarantined_identity_count"]
        != 0
        or physical["projection_inventory"][
            "cross_boundary_quarantined_identity_count"
        ]
        != 0
        or physical["projection_inventory"][
            "identical_payload_discordant_clock_quarantined_identity_count"
        ]
        != 0
    ):
        raise PermissionError("canonical physical projection lineage is not admissible")
    source_dev = {
        str(row["analysis_identity_id"]): row
        for row in source_projection["records"]
        if row["model_split"] == "source_dev"
    }
    canonical_dev = {
        str(row["analysis_identity_id"]): row
        for row in physical["records"]
        if row["model_split"] == "source_dev"
    }
    prediction_ids = {
        str(row["analysis_identity_id"])
        for row in prediction_manifest["prediction_rows"]
    }
    if prediction_ids != set(source_dev) or not set(canonical_dev).issubset(
        prediction_ids
    ):
        raise ValueError("prediction/canonical physical source-dev rosters do not close")
    common_identity_fields = {
        "analysis_identity_id",
        "model_split",
        "official_split",
        "local_patient_id",
        "local_edf_path",
        "source_edf_container_sha256",
        "exact_container_equivalence_id",
        "source_official_path_multiplicity",
        "analysis_unit_weight",
    }
    for identity, row in canonical_dev.items():
        source_row = source_dev[identity]
        if any(row[field] != source_row[field] for field in common_identity_fields):
            raise ValueError("canonical physical projection changed source identity facts")
    excluded = sorted(prediction_ids.difference(canonical_dev))
    accounting = {
        "prediction_intention_to_evaluate_record_count": len(prediction_ids),
        "canonical_metric_record_count": len(canonical_dev),
        "zero_weight_noncanonical_prediction_count": len(excluded),
        "zero_weight_noncanonical_prediction_identity_ids": excluded,
        "zero_weight_noncanonical_prediction_roster_sha256": _canonical_sha256(
            excluded
        ),
        "zero_weight_reason": (
            "validated_same_patient_same_split_physical_alias_excluded_by_"
            "canonical_physical_projection"
        ),
        "prediction_failures_are_not_excluded_from_canonical_metric_roster": True,
    }
    return canonical_dev, accounting, physical, _file_sha256(source)


def _reference_rows_after_gate(
    *,
    manifest_path: Path,
    canonical_by_identity: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    payload = load_common17_manifest(manifest_path, require_complete=True)
    target_contract = payload.get("target_contract", {})
    if (
        payload.get("method_id")
        != "canonical_physical_common17_exact_TERM_seiz_v1"
        or payload.get("patient_disjoint_train_dev") is not True
        or target_contract.get("global_TERM_seiz_only") is not True
        or target_contract.get("channel_specific_annotations_used") is not False
        or target_contract.get("EDF_plus_annotations_used") is not False
        or target_contract.get("clinical_text_or_spreadsheet_used") is not False
    ):
        raise PermissionError("common17 TERM reference authority drifted")
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("common17 TERM manifest records are absent")
    selected: dict[str, dict[str, Any]] = {}
    for raw in records:
        if raw.get("model_split") != "source_dev":
            continue
        identity = str(raw.get("analysis_identity_id"))
        if identity in selected:
            raise ValueError("duplicate source-dev TERM identity")
        selected[identity] = dict(raw)
    if set(selected) != set(canonical_by_identity):
        raise ValueError("canonical physical and TERM source-dev identity rosters differ")
    for identity, reference in selected.items():
        canonical = canonical_by_identity[identity]
        if (
            reference.get("patient_id") != canonical["local_patient_id"]
            or reference.get("edf_relative_path") != canonical["local_edf_path"]
            or reference.get("canonical_source_tensor_sha256")
            != canonical["canonical_physical_source_tensor_sha256"]
        ):
            raise ValueError("TERM row changed canonical physical identity facts")
    return selected


def _joined_records(
    *,
    prediction_rows: Sequence[Mapping[str, Any]],
    reference_by_identity: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    records: list[dict[str, Any]] = []
    ranking_targets: list[np.ndarray] = []
    ranking_scores: list[np.ndarray] = []
    for prediction in prediction_rows:
        identity = str(prediction["analysis_identity_id"])
        if identity not in reference_by_identity:
            continue
        reference = reference_by_identity[identity]
        target_samples = int(reference["target_sample_count_256hz"])
        if target_samples < 1:
            raise ValueError("source-dev target sample count is invalid")
        references = [
            {
                "start_seconds": float(event["start_seconds"]),
                "stop_seconds": float(event["stop_seconds"]),
            }
            for event in reference.get("seizure_events", [])
        ]
        duration = target_samples / TARGET_FS_HZ
        previous_stop = 0.0
        for event_index, event in enumerate(references):
            if (
                not math.isfinite(event["start_seconds"])
                or not math.isfinite(event["stop_seconds"])
                or event["start_seconds"] < 0
                or event["stop_seconds"] <= event["start_seconds"]
                or event["stop_seconds"] > duration + 1e-9
                or (event_index and event["start_seconds"] < previous_stop - 1e-9)
            ):
                raise ValueError("source-dev TERM event lies outside physical EEG")
            previous_stop = event["stop_seconds"]
        if prediction["status"] == "dense_prediction_complete":
            dense = np.load(
                Path(str(prediction["dense_probability_path"])),
                mmap_mode="r",
                allow_pickle=False,
            )
            if len(dense) != target_samples:
                raise ValueError("ST16 dense posterior and canonical EEG clock differ")
            score = posterior_to_one_hz_mean(dense)
        else:
            score = np.zeros(max(1, int(math.ceil(duration))), dtype=np.float32)
        annotation_samples = max(1, int(math.ceil(duration)))
        if len(score) != annotation_samples:
            raise ValueError("ST16 one-Hz posterior does not cover the physical tail")
        target = np.zeros(annotation_samples, dtype=np.bool_)
        for event in references:
            start = max(0, int(round(event["start_seconds"])))
            stop = min(annotation_samples, int(round(event["stop_seconds"])))
            target[start:stop] = True
        ranking_targets.append(target)
        ranking_scores.append(np.asarray(score, dtype=np.float32))
        records.append(
            {
                "analysis_identity_id": identity,
                "patient_id": str(reference["patient_id"]),
                "recording_id": str(reference["edf_relative_path"]),
                "duration_seconds": duration,
                "reference_events": references,
                "one_hz_score": score,
                "one_hz_target": target,
                "prediction_status": prediction["status"],
                "prediction_wall_seconds": float(prediction["wall_seconds"]),
                "predefined_minimum_duration_eligible": (
                    duration + 1e-12 >= ST16_MINIMUM_ANALYZABLE_SECONDS
                ),
            }
        )
    if len(records) != len(reference_by_identity) or {
        record["analysis_identity_id"] for record in records
    } != set(reference_by_identity):
        raise ValueError("canonical metric predictions are incomplete")
    return records, np.concatenate(ranking_targets), np.concatenate(ranking_scores)


def _benchmark_rows(
    records: Sequence[Mapping[str, Any]], *, threshold: float
) -> list[dict[str, Any]]:
    return [
        {
            "patient_id": record["patient_id"],
            "recording_id": record["recording_id"],
            "split": "source_dev",
            "duration_seconds": record["duration_seconds"],
            "reference_events": record["reference_events"],
            "predicted_events": decode_one_hz_runs(
                record["one_hz_score"],
                threshold=threshold,
                duration_seconds=float(record["duration_seconds"]),
            ),
        }
        for record in records
    ]


def _record_level_detection_summary(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    seizure_rows = [row for row in rows if row["reference_events"]]

    def has_overlap(row: Mapping[str, Any]) -> bool:
        return any(
            min(reference["stop_seconds"], prediction["stop_seconds"])
            - max(reference["start_seconds"], prediction["start_seconds"])
            > 0
            for reference in row["reference_events"]
            for prediction in row["predicted_events"]
        )

    hit_count = sum(has_overlap(row) for row in seizure_rows)
    return {
        "recording_count": len(rows),
        "zero_alarm_recording_count": sum(not row["predicted_events"] for row in rows),
        "seizure_recording_count": len(seizure_rows),
        "seizure_recording_with_strict_overlap_count": hit_count,
        "seizure_recording_recall": _safe_rate(hit_count, len(seizure_rows)),
    }


def _prediction_execution_efficiency(
    *,
    records: Sequence[Mapping[str, Any]],
    prediction_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize measured row-attempt wall time without inventing GPU timing."""

    canonical_wall = [float(row["prediction_wall_seconds"]) for row in records]
    duration_seconds = sum(float(row["duration_seconds"]) for row in records)
    eligible = [row for row in records if row["predefined_minimum_duration_eligible"]]
    eligible_duration = sum(float(row["duration_seconds"]) for row in eligible)
    eligible_wall = sum(float(row["prediction_wall_seconds"]) for row in eligible)
    successful = [
        row for row in records if row["prediction_status"] == "dense_prediction_complete"
    ]
    successful_duration = sum(float(row["duration_seconds"]) for row in successful)
    successful_wall = sum(float(row["prediction_wall_seconds"]) for row in successful)
    all_inventory_wall = sum(
        float(row["wall_seconds"])
        for row in prediction_manifest["prediction_rows"]
    )
    return {
        "measurement_scope": (
            "per-record warm-like attempt wall time: EDF read, preprocessing, "
            "inference, OLA and sidecar write; checkpoint load excluded"
        ),
        "all_prediction_inventory": {
            "record_count": len(prediction_manifest["prediction_rows"]),
            "attempt_wall_seconds": all_inventory_wall,
            "RTF": None,
            "RTF_reason": (
                "noncanonical physical aliases have zero metric weight and no "
                "independent duration authority in this receipt"
            ),
        },
        "canonical_intention_to_evaluate": {
            "record_count": len(records),
            "EEG_hours": duration_seconds / 3600.0,
            "attempt_wall_seconds": sum(canonical_wall),
            "real_time_factor": _safe_rate(sum(canonical_wall), duration_seconds),
            "wall_seconds_per_EEG_hour": _safe_rate(
                sum(canonical_wall), duration_seconds / 3600.0
            ),
            "record_wall_seconds_median": (
                None if not canonical_wall else float(np.median(canonical_wall))
            ),
            "record_wall_seconds_p95": (
                None
                if not canonical_wall
                else float(np.quantile(canonical_wall, 0.95))
            ),
        },
        "predefined_at_least_60s_stratum": {
            "record_count": len(eligible),
            "EEG_hours": eligible_duration / 3600.0,
            "attempt_wall_seconds": eligible_wall,
            "real_time_factor": _safe_rate(eligible_wall, eligible_duration),
        },
        "dense_prediction_success_descriptive_only": {
            "record_count": len(successful),
            "EEG_hours": successful_duration / 3600.0,
            "attempt_wall_seconds": successful_wall,
            "real_time_factor": _safe_rate(successful_wall, successful_duration),
            "failure_exclusion_bias_warning": True,
        },
        "GPU_active_time_measured": False,
        "peak_GPU_memory_measured_during_source_dev_inference": False,
        "co_resident_service_state_bound_in_prediction_manifest": False,
        "formal_isolated_efficiency_claim_authorized": False,
    }


def _technical_accounting(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    eligible = [row for row in records if row["predefined_minimum_duration_eligible"]]
    below_minimum = [
        row for row in records if not row["predefined_minimum_duration_eligible"]
    ]
    return {
        "canonical_intention_to_evaluate_record_count": len(records),
        "canonical_dense_prediction_complete_count": sum(
            row["prediction_status"] == "dense_prediction_complete" for row in records
        ),
        "canonical_typed_technical_failure_count": sum(
            row["prediction_status"] == "typed_technical_failure" for row in records
        ),
        "predefined_at_least_60s_record_count": len(eligible),
        "predefined_at_least_60s_typed_technical_failure_count": sum(
            row["prediction_status"] == "typed_technical_failure" for row in eligible
        ),
        "predefined_below_60s_record_count": len(below_minimum),
        "predefined_below_60s_reference_event_count": sum(
            len(row["reference_events"]) for row in below_minimum
        ),
        "primary_metrics_retain_all_canonical_failures_as_zero_alarm_and_zero_score": True,
        "at_least_60s_stratum_retains_unexpected_failures_as_zero_alarm_and_zero_score": True,
        "success_only_accuracy_metric_authorized": False,
    }


def _ranking_metrics_for_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not records:
        return {
            "sample_count": 0,
            "positive_count": 0,
            "negative_count": 0,
            "AUROC": None,
            "average_precision": None,
        }
    return _binary_ranking_metrics(
        np.concatenate([np.asarray(row["one_hz_target"]) for row in records]),
        np.concatenate([np.asarray(row["one_hz_score"]) for row in records]),
    )


def _with_complete_patient_macro(
    metrics: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    result = deepcopy(dict(metrics))
    by_patient: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_patient[str(row["patient_id"])].append(row)
    patient_metrics = [
        aggregate_continuous_detection_metrics(by_patient[patient_id])
        for patient_id in sorted(by_patient)
    ]
    for field in ("event_precision", "event_f1"):
        values = [
            float(row[field]) for row in patient_metrics if row[field] is not None
        ]
        result["patient_macro"][f"{field}_macro"] = (
            None if not values else float(np.mean(values))
        )
        result["patient_macro"][f"{field}_evaluable_patient_count"] = len(values)
    return result


def _compact_curve_row(
    *, threshold: float, strict: Mapping[str, Any], official: Mapping[str, Any]
) -> dict[str, Any]:
    sz_event = official["event_pooled"]
    sz_sample = official["sample_1hz_pooled"]
    hours = float(strict["total_recording_hours"])
    return {
        "threshold": threshold,
        "strict_event_sensitivity": strict["event_sensitivity"],
        "strict_event_precision": strict["event_precision"],
        "strict_event_f1": strict["event_f1"],
        "strict_false_alarms_per_24h": strict["alarm_false_alarms_per_24h"],
        "strict_onset_hit_at_1s": strict["onset_absolute_hit_rate"]["1s"]["rate"],
        "strict_onset_hit_at_3s": strict["onset_absolute_hit_rate"]["3s"]["rate"],
        "strict_onset_hit_at_5s": strict["onset_absolute_hit_rate"]["5s"]["rate"],
        "strict_onset_hit_at_10s": strict["onset_absolute_hit_rate"]["10s"]["rate"],
        "candidates_per_recording_hour": (
            None if hours <= 0 else strict["predicted_alarm_count"] / hours
        ),
        "szcore_event_sensitivity": sz_event["sensitivity"],
        "szcore_event_precision": sz_event["precision"],
        "szcore_event_f1": sz_event["f1"],
        "szcore_false_alarms_per_24h": sz_event["false_positives_per_24h"],
        "szcore_sample_sensitivity": sz_sample["sensitivity"],
        "szcore_sample_precision": sz_sample["precision"],
        "szcore_sample_f1": sz_sample["f1"],
    }


def _metric_or_negative(value: object) -> float:
    return -math.inf if value is None else float(value)


def _workload_f1_for_selection(row: Mapping[str, Any]) -> float:
    """Rank the zero-alarm, all-miss endpoint as F1=0 without rewriting it.

    Pooled strict precision and F1 are correctly undefined when there are no
    predicted alarms.  For workload selection only, however, a source-dev set
    with reference events, zero sensitivity, and zero candidates is the
    conventional all-miss F1=0 endpoint.  Treating that ``None`` as negative
    infinity would prefer a gratuitous false alarm whose reported F1 is 0.
    The compact metric row remains unchanged (precision/F1 stay ``None``).
    """

    value = row["strict_event_f1"]
    if value is not None:
        return float(value)
    sensitivity = row["strict_event_sensitivity"]
    candidates = row["candidates_per_recording_hour"]
    if (
        sensitivity is not None
        and float(sensitivity) == 0.0
        and candidates is not None
        and float(candidates) == 0.0
    ):
        return 0.0
    return -math.inf


def _select_best(curve: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    return max(
        curve,
        key=lambda row: (
            _metric_or_negative(row["strict_event_f1"]),
            _metric_or_negative(row["strict_event_sensitivity"]),
            -_metric_or_negative(row["strict_false_alarms_per_24h"]),
            -abs(float(row["threshold"]) - 0.8),
            float(row["threshold"]),
        ),
    )


def _workload_operating_points(
    curve: Sequence[Mapping[str, Any]], *, field: str, budgets: Sequence[float]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for budget in budgets:
        eligible = [
            row
            for row in curve
            if row[field] is not None and float(row[field]) <= float(budget)
        ]
        selected = (
            None
            if not eligible
            else max(
                eligible,
                key=lambda row: (
                    _metric_or_negative(row["strict_event_sensitivity"]),
                    _workload_f1_for_selection(row),
                    -float(row[field]),
                    -abs(float(row["threshold"]) - 0.8),
                    float(row["threshold"]),
                ),
            )
        )
        result.append(
            {
                "budget": float(budget),
                "selection_objective": [
                    "maximum_strict_pooled_event_sensitivity",
                    (
                        "maximum_strict_pooled_event_f1_with_zero_alarm_"
                        "all_miss_none_ranked_as_zero"
                    ),
                    f"minimum_{field}",
                    "closest_to_paper_threshold_0.8",
                    "higher_threshold_final_tiebreak",
                ],
                "selected": None if selected is None else dict(selected),
            }
        )
    return result


def _bind_operating_points(
    points: Sequence[Mapping[str, Any]],
    *,
    family: str,
    decoder_policy_sha256: str,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for point in points:
        row = deepcopy(dict(point))
        selected = row["selected"]
        binding = {
            "family": family,
            "budget": row["budget"],
            "selected_threshold": (
                None if selected is None else selected["threshold"]
            ),
            "decoder_policy_sha256": decoder_policy_sha256,
            "source_split": "source_dev",
            "selection_role": "in_sample_operating_point_calibration",
        }
        row["operating_point_id"] = (
            None
            if selected is None
            else f"ST16-OP-{_canonical_sha256(binding)[:24]}"
        )
        row["binding"] = binding
        result.append(row)
    return result


def evaluate_st16_source_dev(
    *,
    prediction_manifest_path: str | Path,
    canonical_physical_projection_path: str | Path,
    reference_manifest_path: str | Path,
    output_dir: str | Path,
    project_root: str | Path,
    thresholds: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Evaluate a complete frozen source-dev inventory and write a receipt."""

    began = time.perf_counter()
    (
        prediction_manifest,
        prediction_file_sha,
        source_projection,
    ) = _validate_prediction_manifest(Path(prediction_manifest_path))
    (
        canonical_by_identity,
        roster_accounting,
        canonical_projection,
        canonical_projection_file_sha,
    ) = _canonical_metric_projection(
        path=Path(canonical_physical_projection_path),
        prediction_manifest=prediction_manifest,
        source_projection=source_projection,
    )
    output = Path(output_dir).resolve(strict=False)
    output.mkdir(parents=True, exist_ok=True)
    gate = _materialize_freeze_gate(
        output_dir=output,
        prediction_manifest=prediction_manifest,
        manifest_sha256=prediction_file_sha,
        canonical_projection=canonical_projection,
        canonical_projection_sha256=canonical_projection_file_sha,
        roster_accounting=roster_accounting,
    )
    # This is intentionally the first reference-bearing read in this function.
    reference_path = Path(reference_manifest_path).resolve(strict=True)
    reference_by_identity = _reference_rows_after_gate(
        manifest_path=reference_path,
        canonical_by_identity=canonical_by_identity,
    )
    records, ranking_target, ranking_score = _joined_records(
        prediction_rows=prediction_manifest["prediction_rows"],
        reference_by_identity=reference_by_identity,
    )
    threshold_grid = tuple(
        default_thresholds() if thresholds is None else sorted(set(thresholds))
    )
    if (
        not threshold_grid
        or any(not 0.0 < float(value) < 1.0 for value in threshold_grid)
        or 0.8 not in threshold_grid
    ):
        raise ValueError("evaluation threshold grid is invalid")
    curve: list[dict[str, Any]] = []
    for threshold_index, threshold in enumerate(threshold_grid, start=1):
        benchmark_rows = _benchmark_rows(records, threshold=float(threshold))
        strict = aggregate_continuous_detection_metrics(benchmark_rows)
        official = _official_timescoring_metrics(
            benchmark_rows, project_root=Path(project_root).resolve(strict=True)
        )
        curve.append(
            _compact_curve_row(
                threshold=float(threshold), strict=strict, official=official
            )
        )
        if threshold_index % 10 == 0 or threshold_index == len(threshold_grid):
            print(
                json.dumps(
                    {
                        "stage": "st16_source_dev_threshold_progress",
                        "threshold_count_completed": threshold_index,
                        "threshold_count_total": len(threshold_grid),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    selected_row = dict(_select_best(curve))
    selected_threshold = float(selected_row["threshold"])
    selected_benchmark_rows = _benchmark_rows(
        records, threshold=selected_threshold
    )
    selected_strict = _with_complete_patient_macro(
        aggregate_continuous_detection_metrics(selected_benchmark_rows),
        selected_benchmark_rows,
    )
    selected_official = _official_timescoring_metrics(
        selected_benchmark_rows, project_root=Path(project_root).resolve(strict=True)
    )
    fixed_08 = next(row for row in curve if float(row["threshold"]) == 0.8)
    eligible_records = [
        record for record in records if record["predefined_minimum_duration_eligible"]
    ]
    eligible_benchmark_rows = _benchmark_rows(
        eligible_records, threshold=selected_threshold
    )
    eligible_strict = _with_complete_patient_macro(
        aggregate_continuous_detection_metrics(eligible_benchmark_rows),
        eligible_benchmark_rows,
    )
    eligible_official = _official_timescoring_metrics(
        eligible_benchmark_rows, project_root=Path(project_root).resolve(strict=True)
    )
    failure_types = Counter(
        str(row.get("failure_type"))
        for row in prediction_manifest["prediction_rows"]
        if row["status"] == "typed_technical_failure"
    )
    canonical_prediction_by_identity = {
        str(row["analysis_identity_id"]): row
        for row in prediction_manifest["prediction_rows"]
        if str(row["analysis_identity_id"]) in canonical_by_identity
    }
    canonical_failure_types = Counter(
        str(row["failure_type"])
        for row in canonical_prediction_by_identity.values()
        if row["status"] == "typed_technical_failure"
    )
    decoder_contract = {
        "posterior_source": "complete pre-threshold 256 Hz weighted-OLA",
        "aggregation": "nonoverlapping physical 1-second arithmetic mean",
        "partial_final_second_policy": "preserve_and_average_observed_samples",
        "threshold_grid": list(threshold_grid),
        "minimum_event_seconds": MINIMUM_EVENT_SECONDS,
        "gap_merge_before_scoring_seconds": 0,
        "paper_anchor_threshold_reported_separately": 0.8,
        "threshold_or_operating_point_selected_before_prediction_freeze": False,
        "threshold_and_operating_points_fit_only_after_complete_prediction_inventory_freeze": True,
    }
    decoder_policy_sha256 = _canonical_sha256(decoder_contract)
    primary_operating_point_binding = {
        "family": "maximum_strict_pooled_event_f1_primary",
        "selected_threshold": selected_threshold,
        "decoder_policy_sha256": decoder_policy_sha256,
        "source_split": "source_dev",
        "selection_role": "in_sample_operating_point_calibration",
    }
    alarm_operating_points = _bind_operating_points(
        _workload_operating_points(
            curve,
            field="strict_false_alarms_per_24h",
            budgets=(1, 3, 6, 12),
        ),
        family="alarm_false_alarms_per_24h_budget",
        decoder_policy_sha256=decoder_policy_sha256,
    )
    navigation_operating_points = _bind_operating_points(
        _workload_operating_points(
            curve,
            field="candidates_per_recording_hour",
            budgets=(1, 2, 4, 8, 16),
        ),
        family="navigation_candidates_per_recording_hour_budget",
        decoder_policy_sha256=decoder_policy_sha256,
    )
    receipt = _content_address(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "completed_source_dev_postfreeze_exploratory_evaluation",
            "claim_boundary": (
                "source_dev in-sample decoder selection; not independent test, "
                "not source_eval, not clinical deployment evidence"
            ),
            "prediction_freeze_gate": gate,
            "prediction_manifest_path": str(
                Path(prediction_manifest_path).resolve(strict=True)
            ),
            "prediction_manifest_file_sha256": prediction_file_sha,
            "prediction_manifest_receipt_sha256": prediction_manifest[
                "receipt_sha256"
            ],
            "reference_manifest_path": str(reference_path),
            "reference_manifest_file_sha256": _file_sha256(reference_path),
            "reference_opened_only_after_prediction_freeze_gate": True,
            "source_eval_opened": False,
            "canonical_physical_projection_path": str(
                Path(canonical_physical_projection_path).resolve(strict=True)
            ),
            "canonical_physical_projection_file_sha256": (
                canonical_projection_file_sha
            ),
            "canonical_physical_projection_receipt_sha256": (
                canonical_projection["receipt_sha256"]
            ),
            "roster_accounting": roster_accounting,
            "record_count": len(records),
            "record_count_role": "canonical_physical_metric_denominator",
            "patient_count": len({record["patient_id"] for record in records}),
            "prediction_intention_to_evaluate_status_accounting": {
                "record_count": prediction_manifest["materialized_record_count"],
                "dense_prediction_complete_count": prediction_manifest[
                    "dense_prediction_complete_count"
                ],
                "typed_technical_failure_count": prediction_manifest[
                    "typed_technical_failure_count"
                ],
            },
            "dense_prediction_complete_count": prediction_manifest[
                "dense_prediction_complete_count"
            ],
            "typed_technical_failure_count": prediction_manifest[
                "typed_technical_failure_count"
            ],
            "typed_technical_failure_types": dict(sorted(failure_types.items())),
            "canonical_typed_technical_failure_types": dict(
                sorted(canonical_failure_types.items())
            ),
            "technical_accounting": _technical_accounting(records),
            "technical_failures_retained_as_zero_alarm_and_zero_score": True,
            "decoder": {
                **decoder_contract,
                "decoder_policy_sha256": decoder_policy_sha256,
                "evaluation_module_file_sha256": _file_sha256(
                    Path(__file__).resolve(strict=True)
                ),
                "strict_benchmark_module_file_sha256": _file_sha256(
                    Path(__file__).with_name("continuous_detection_benchmark.py")
                ),
            },
            "threshold_selection": {
                "role": "source_dev_in_sample_decoder_calibration",
                "lexicographic_objective": [
                    "maximum_strict_pooled_event_f1",
                    "maximum_strict_pooled_event_sensitivity",
                    "minimum_strict_false_alarms_per_24h",
                    "closest_to_paper_threshold_0.8",
                    "higher_threshold_final_tiebreak",
                ],
                "selected_threshold": selected_threshold,
                "operating_point_id": (
                    "ST16-OP-"
                    + _canonical_sha256(primary_operating_point_binding)[:24]
                ),
                "operating_point_binding": primary_operating_point_binding,
                "selected_compact_metrics": selected_row,
                "fixed_threshold_0.8_compact_metrics": fixed_08,
            },
            "selected_strict_zero_dilation_ordered_one_to_one": selected_strict,
            "selected_strict_record_level_summary": (
                _record_level_detection_summary(selected_benchmark_rows)
            ),
            "selected_official_szcore_compatible_timescoring_0_0_7": (
                selected_official
            ),
            "predefined_at_least_60s_stratum_selected_threshold": {
                "role": (
                    "secondary_predefined_technical_evaluability_stratum; "
                    "threshold inherited from canonical intention-to-evaluate primary"
                ),
                "record_count": len(eligible_records),
                "strict_zero_dilation_ordered_one_to_one": eligible_strict,
                "strict_record_level_summary": _record_level_detection_summary(
                    eligible_benchmark_rows
                ),
                "official_szcore_compatible_timescoring_0_0_7": eligible_official,
                "prethreshold_one_hz_ranking_metrics": _ranking_metrics_for_records(
                    eligible_records
                ),
            },
            "prethreshold_one_hz_ranking_metrics": _binary_ranking_metrics(
                ranking_target, ranking_score
            ),
            "alarm_workload_operating_points": alarm_operating_points,
            "navigation_workload_operating_points": navigation_operating_points,
            "threshold_curve": curve,
            "prediction_execution_efficiency": _prediction_execution_efficiency(
                records=records, prediction_manifest=prediction_manifest
            ),
            "evaluation_wall_seconds": time.perf_counter() - began,
            "clinical_use_authorized": False,
            "architecture_promotable": False,
            "receipt_sha256": _PENDING,
        }
    )
    _write_json_atomic(output / "evaluation_receipt.json", receipt, replace=False)
    return receipt


__all__ = [
    "DECODER_BIN_SECONDS",
    "MINIMUM_EVENT_SECONDS",
    "SCHEMA_VERSION",
    "decode_one_hz_runs",
    "default_thresholds",
    "evaluate_st16_source_dev",
    "posterior_to_one_hz_mean",
]
