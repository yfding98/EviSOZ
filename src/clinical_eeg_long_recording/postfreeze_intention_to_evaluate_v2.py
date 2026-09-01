"""Post-freeze intention-to-evaluate metrics for long-recording EEG reports.

This append-only v2 contract fixes a denominator problem in the legacy
``postfreeze_evaluation`` module without changing any legacy artifact.  The
reference roster is validated independently of model output and therefore
defines every full-coverage denominator before event matching.  A missing
reference remains not available; a model abstention, technical failure, zero
alarm attempt, or unmatched reference event remains in an applicable
denominator with score zero.

The schema deliberately separates:

* one recording-level, typed Excel-summary projection; and
* zero or more event-level physician hard-significant and soft-spread sets.

References never contain predicted event IDs.  Event bindings are produced
after prediction freeze by deterministic one-to-one interval matching.  The
module reads no EDF, annotation, spreadsheet, physician narrative, report
text, or private path.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

from src.clinical_eeg_report.schema import canonicalize_electrode


POSTFREEZE_ITE_REFERENCE_ROSTER_V2_SCHEMA_VERSION = "postfreeze_ite_reference_roster_v2"
POSTFREEZE_ITE_PREDICTION_ATTEMPT_V2_SCHEMA_VERSION = (
    "postfreeze_ite_prediction_attempt_v2"
)
POSTFREEZE_ITE_POLICY_V2_SCHEMA_VERSION = "postfreeze_ite_policy_v2"
POSTFREEZE_ITE_EVALUATION_V2_SCHEMA_VERSION = "postfreeze_ite_evaluation_v2"

HARD_SIGNIFICANT_METRICS = (
    "top1_hit",
    "top3_hit",
    "top5_hit",
    "recall_at_1",
    "recall_at_3",
    "recall_at_5",
    "mrr",
    "average_precision",
)
SOFT_SPREAD_METRICS = (
    "spread_hit_at_1",
    "spread_hit_at_3",
    "spread_hit_at_5",
    "spread_mrr",
    "spread_top1_gain",
    "spread_ndcg_at_1",
    "spread_ndcg_at_3",
    "spread_ndcg_at_5",
)

DEFAULT_POSTFREEZE_ITE_POLICY_V2: Mapping[str, Any] = {
    "schema_version": POSTFREEZE_ITE_POLICY_V2_SCHEMA_VERSION,
    "policy_id": "postfreeze_reference_first_interval_hungarian_v2",
    "minimum_event_interval_iou": 0.10,
    "maximum_onset_distance_seconds": 5.0,
    "soft_spread_gain": 0.35,
    "matching_objective": "maximum_cardinality_then_iou_then_onset_proximity",
    "abstention_on_applicable_reference": "zero_in_full_coverage",
    "technical_failure_on_applicable_reference": "zero_in_full_coverage",
    "unmatched_reference": "zero_in_full_coverage",
    "missing_reference_label": "not_available",
    "selected_summary_replaces_full_coverage": False,
}

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_LATERALITIES = {
    "left",
    "right",
    "bilateral",
    "midline",
    "none",
    "indeterminate",
}
_REGIONS = {
    "frontal",
    "temporal",
    "central",
    "parietal",
    "occipital",
    "frontotemporal",
    "centrotemporal",
    "temporoparietal",
    "posterior",
    "diffuse",
    "midline",
    "unknown",
}
_ONSET_UNCERTAINTY = {"clear", "uncertain_or_unclear", "indeterminate"}
_ATTEMPT_STATUSES = {
    "completed_with_predictions",
    "completed_with_abstentions",
    "completed_zero_alarm",
    "partial_coverage",
    "technical_failure",
}
_SELECTION_STATUSES = {"accepted", "abstained"}
_REFERENCE_COMPLETENESS = {"exhaustive", "positive_only_unknown_complement"}
_STANDARD_19_PLUS_REFERENCES = {
    "FP1",
    "FP2",
    "F7",
    "F8",
    "F3",
    "F4",
    "FZ",
    "C3",
    "C4",
    "CZ",
    "T7",
    "T8",
    "P7",
    "P8",
    "P3",
    "P4",
    "PZ",
    "O1",
    "O2",
    "M1",
    "M2",
}

_REFERENCE_ROSTER_KEYS = {
    "schema_version",
    "reference_roster_id",
    "patient_id",
    "recording_id",
    "recording_summary_reference",
    "event_references",
    "reference_boundary",
}
_REFERENCE_BOUNDARY = {
    "joined_after_prediction_freeze": True,
    "raw_excel_text_included": False,
    "raw_physician_text_included": False,
    "edf_annotations_included": False,
    "source_path_included": False,
    "used_for_training_or_calibration": False,
    "used_for_report_generation": False,
    "used_for_llm": False,
}
_RECORDING_SUMMARY_REFERENCE_KEYS = {
    "reference_id",
    "mapping_status",
    "laterality",
    "regions",
    "onset_uncertainty",
}
_REFERENCE_EVENT_KEYS = {
    "reference_event_id",
    "interval",
    "channel_reference",
}
_CHANNEL_REFERENCE_KEYS = {
    "reference_id",
    "reference_completeness",
    "hard_significant_electrodes",
    "soft_spread_electrodes",
}
_INTERVAL_KEYS = {
    "lower_seconds",
    "upper_seconds",
    "left_censored",
    "right_censored",
}

_PREDICTION_ATTEMPT_KEYS = {
    "schema_version",
    "attempt_id",
    "patient_id",
    "recording_id",
    "prediction_bundle_sha256",
    "attempt_status",
    "recording_summary_prediction",
    "predicted_events",
    "prediction_boundary",
}
_PREDICTION_BOUNDARY = {
    "prediction_frozen_before_reference_join": True,
    "excel_used_for_inference": False,
    "physician_labels_used_for_inference": False,
    "edf_annotations_used_for_inference": False,
    "clinical_text_used_for_inference": False,
}
_RECORDING_SUMMARY_PREDICTION_KEYS = {
    "selection_status",
    "laterality",
    "regions",
    "onset_uncertainty",
}
_PREDICTED_EVENT_KEYS = {
    "predicted_event_id",
    "interval",
    "selection_status",
    "ranked_electrodes",
}


def _strict_object(value: object, keys: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be an object")
    actual = set(value)
    missing = keys - actual
    extra = actual - keys
    if missing:
        raise ValueError(f"{context} missing keys: {sorted(missing)}")
    if extra:
        raise ValueError(f"{context} has unknown keys: {sorted(extra)}")
    return {str(key): deepcopy(item) for key, item in value.items()}


def _identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be an opaque identifier")
    return value


def _sha256_identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _canonical_sha256(value: object) -> str:
    body = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _finite_nonnegative(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{context} must be a finite non-negative number")
    return result


def _ratio(numerator: float, denominator: float) -> float | str:
    if denominator <= 0:
        return "not_available"
    return round(float(numerator) / float(denominator), 12)


def _validate_literal_boundary(
    value: object, expected: Mapping[str, bool], context: str
) -> dict[str, bool]:
    data = _strict_object(value, set(expected), context)
    for key, literal in expected.items():
        if data[key] is not literal:
            raise ValueError(f"{context}.{key} must be {literal}")
    return dict(expected)


def _validate_interval(value: object, context: str) -> dict[str, Any]:
    data = _strict_object(value, _INTERVAL_KEYS, context)
    lower = _finite_nonnegative(data["lower_seconds"], f"{context}.lower_seconds")
    upper = _finite_nonnegative(data["upper_seconds"], f"{context}.upper_seconds")
    if lower > upper:
        raise ValueError(f"{context} lower_seconds exceeds upper_seconds")
    for key in ("left_censored", "right_censored"):
        if type(data[key]) is not bool:
            raise TypeError(f"{context}.{key} must be boolean")
    return {
        "lower_seconds": lower,
        "upper_seconds": upper,
        "left_censored": bool(data["left_censored"]),
        "right_censored": bool(data["right_censored"]),
    }


def _optional_regions(value: object, context: str) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise TypeError(f"{context} must be null or a non-empty list")
    result: list[str] = []
    for raw in value:
        if not isinstance(raw, str) or raw not in _REGIONS:
            raise ValueError(f"{context} contains an unsupported controlled code")
        if raw in result:
            raise ValueError(f"{context} contains duplicate values")
        result.append(raw)
    return sorted(result)


def _optional_electrodes(value: object, context: str) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise TypeError(f"{context} must be null or a non-empty list")
    result: list[str] = []
    for raw in value:
        canonical = canonicalize_electrode(raw)
        if canonical not in _STANDARD_19_PLUS_REFERENCES:
            raise ValueError(f"{context} contains an out-of-scope electrode")
        if canonical in result:
            raise ValueError(f"{context} contains duplicate canonical electrodes")
        result.append(canonical)
    return result


def _validate_recording_summary_reference(
    value: object | None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    context = "recording_summary_reference"
    data = _strict_object(value, _RECORDING_SUMMARY_REFERENCE_KEYS, context)
    data["reference_id"] = _identifier(data["reference_id"], f"{context}.reference_id")
    if data["mapping_status"] not in {"available", "ambiguous", "unmappable"}:
        raise ValueError(f"{context}.mapping_status is invalid")
    laterality = data["laterality"]
    if laterality is not None and laterality not in _LATERALITIES:
        raise ValueError(f"{context}.laterality is invalid")
    uncertainty = data["onset_uncertainty"]
    if uncertainty is not None and uncertainty not in _ONSET_UNCERTAINTY:
        raise ValueError(f"{context}.onset_uncertainty is invalid")
    regions = _optional_regions(data["regions"], f"{context}.regions")
    fields = (laterality, regions, uncertainty)
    if data["mapping_status"] == "available" and all(item is None for item in fields):
        raise ValueError(f"{context} available mapping has no typed field")
    if data["mapping_status"] != "available" and any(
        item is not None for item in fields
    ):
        raise ValueError(
            f"{context} ambiguous/unmappable mapping cannot carry scored fields"
        )
    return {
        "reference_id": data["reference_id"],
        "mapping_status": data["mapping_status"],
        "laterality": laterality,
        "regions": regions,
        "onset_uncertainty": uncertainty,
    }


def _validate_channel_reference(
    value: object | None, context: str
) -> dict[str, Any] | None:
    if value is None:
        return None
    data = _strict_object(value, _CHANNEL_REFERENCE_KEYS, context)
    reference_id = _identifier(data["reference_id"], f"{context}.reference_id")
    completeness = data["reference_completeness"]
    if completeness not in _REFERENCE_COMPLETENESS:
        raise ValueError(f"{context}.reference_completeness is invalid")
    hard = _optional_electrodes(
        data["hard_significant_electrodes"],
        f"{context}.hard_significant_electrodes",
    )
    soft = _optional_electrodes(
        data["soft_spread_electrodes"],
        f"{context}.soft_spread_electrodes",
    )
    if hard is None and soft is None:
        raise ValueError(f"{context} has neither a hard nor soft reference")
    hard_set = set(hard or [])
    resolved_soft = [item for item in soft or [] if item not in hard_set]
    return {
        "reference_id": reference_id,
        "reference_completeness": completeness,
        "hard_significant_electrodes": hard,
        "soft_spread_electrodes": resolved_soft if soft is not None else None,
        "hard_overrides_soft_spread": True,
    }


def validate_postfreeze_ite_reference_roster_v2(value: object) -> dict[str, Any]:
    """Validate a prediction-independent typed reference roster."""

    data = _strict_object(value, _REFERENCE_ROSTER_KEYS, "reference roster v2")
    if data["schema_version"] != POSTFREEZE_ITE_REFERENCE_ROSTER_V2_SCHEMA_VERSION:
        raise ValueError("reference roster v2 schema_version mismatch")
    for key in ("reference_roster_id", "patient_id", "recording_id"):
        data[key] = _identifier(data[key], f"reference roster v2.{key}")
    data["recording_summary_reference"] = _validate_recording_summary_reference(
        data["recording_summary_reference"]
    )
    raw_events = data["event_references"]
    if not isinstance(raw_events, list):
        raise TypeError("reference roster v2.event_references must be a list")
    events: list[dict[str, Any]] = []
    event_ids: set[str] = set()
    channel_reference_ids: set[str] = set()
    for index, raw in enumerate(raw_events):
        context = f"reference roster v2.event_references[{index}]"
        item = _strict_object(raw, _REFERENCE_EVENT_KEYS, context)
        event_id = _identifier(
            item["reference_event_id"], f"{context}.reference_event_id"
        )
        if event_id in event_ids:
            raise ValueError("reference roster v2 contains duplicate reference events")
        event_ids.add(event_id)
        channel_reference = _validate_channel_reference(
            item["channel_reference"], f"{context}.channel_reference"
        )
        if channel_reference is not None:
            reference_id = str(channel_reference["reference_id"])
            if reference_id in channel_reference_ids:
                raise ValueError(
                    "reference roster v2 reuses an event channel reference ID"
                )
            channel_reference_ids.add(reference_id)
        events.append(
            {
                "reference_event_id": event_id,
                "interval": _validate_interval(item["interval"], f"{context}.interval"),
                "channel_reference": channel_reference,
            }
        )
    events.sort(
        key=lambda row: (
            float(row["interval"]["lower_seconds"]),
            float(row["interval"]["upper_seconds"]),
            str(row["reference_event_id"]),
        )
    )
    data["event_references"] = events
    data["reference_boundary"] = _validate_literal_boundary(
        data["reference_boundary"],
        _REFERENCE_BOUNDARY,
        "reference roster v2.reference_boundary",
    )
    return data


def _validate_recording_summary_prediction(
    value: object | None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    context = "recording_summary_prediction"
    data = _strict_object(value, _RECORDING_SUMMARY_PREDICTION_KEYS, context)
    if data["selection_status"] not in _SELECTION_STATUSES:
        raise ValueError(f"{context}.selection_status is invalid")
    laterality = data["laterality"]
    if laterality is not None and laterality not in _LATERALITIES:
        raise ValueError(f"{context}.laterality is invalid")
    uncertainty = data["onset_uncertainty"]
    if uncertainty is not None and uncertainty not in _ONSET_UNCERTAINTY:
        raise ValueError(f"{context}.onset_uncertainty is invalid")
    regions = _optional_regions(data["regions"], f"{context}.regions")
    fields = (laterality, regions, uncertainty)
    if data["selection_status"] == "accepted" and all(item is None for item in fields):
        raise ValueError(f"{context} accepted prediction has no typed field")
    if data["selection_status"] == "abstained" and any(
        item is not None for item in fields
    ):
        raise ValueError(f"{context} abstention cannot carry semantic predictions")
    return {
        "selection_status": data["selection_status"],
        "laterality": laterality,
        "regions": regions,
        "onset_uncertainty": uncertainty,
    }


def validate_postfreeze_ite_prediction_attempt_v2(value: object) -> dict[str, Any]:
    """Validate one frozen, reference-free prediction attempt."""

    data = _strict_object(value, _PREDICTION_ATTEMPT_KEYS, "prediction attempt v2")
    if data["schema_version"] != POSTFREEZE_ITE_PREDICTION_ATTEMPT_V2_SCHEMA_VERSION:
        raise ValueError("prediction attempt v2 schema_version mismatch")
    for key in ("attempt_id", "patient_id", "recording_id"):
        data[key] = _identifier(data[key], f"prediction attempt v2.{key}")
    data["prediction_bundle_sha256"] = _sha256_identifier(
        data["prediction_bundle_sha256"],
        "prediction attempt v2.prediction_bundle_sha256",
    )
    if data["attempt_status"] not in _ATTEMPT_STATUSES:
        raise ValueError("prediction attempt v2.attempt_status is invalid")
    data["recording_summary_prediction"] = _validate_recording_summary_prediction(
        data["recording_summary_prediction"]
    )
    raw_events = data["predicted_events"]
    if not isinstance(raw_events, list):
        raise TypeError("prediction attempt v2.predicted_events must be a list")
    events: list[dict[str, Any]] = []
    event_ids: set[str] = set()
    for index, raw in enumerate(raw_events):
        context = f"prediction attempt v2.predicted_events[{index}]"
        item = _strict_object(raw, _PREDICTED_EVENT_KEYS, context)
        event_id = _identifier(
            item["predicted_event_id"], f"{context}.predicted_event_id"
        )
        if event_id in event_ids:
            raise ValueError(
                "prediction attempt v2 contains duplicate predicted events"
            )
        event_ids.add(event_id)
        selection = item["selection_status"]
        if selection not in _SELECTION_STATUSES:
            raise ValueError(f"{context}.selection_status is invalid")
        raw_ranking = item["ranked_electrodes"]
        if not isinstance(raw_ranking, list):
            raise TypeError(f"{context}.ranked_electrodes must be a list")
        ranking = (
            _optional_electrodes(raw_ranking, f"{context}.ranked_electrodes")
            if raw_ranking
            else []
        )
        if selection == "accepted" and not ranking:
            raise ValueError(f"{context} accepted event requires a non-empty ranking")
        if selection == "abstained" and ranking:
            raise ValueError(f"{context} abstained event cannot carry a ranking")
        events.append(
            {
                "predicted_event_id": event_id,
                "interval": _validate_interval(item["interval"], f"{context}.interval"),
                "selection_status": selection,
                "ranked_electrodes": ranking,
            }
        )
    events.sort(
        key=lambda row: (
            float(row["interval"]["lower_seconds"]),
            float(row["interval"]["upper_seconds"]),
            str(row["predicted_event_id"]),
        )
    )
    status = str(data["attempt_status"])
    accepted_count = sum(item["selection_status"] == "accepted" for item in events)
    if status in {"completed_zero_alarm", "technical_failure"} and events:
        raise ValueError(f"prediction attempt v2 {status} cannot contain events")
    if (
        status == "technical_failure"
        and data["recording_summary_prediction"] is not None
    ):
        raise ValueError(
            "prediction attempt v2 technical_failure cannot carry a recording prediction"
        )
    if status == "completed_with_predictions" and (not events or not accepted_count):
        raise ValueError("completed_with_predictions requires an accepted event")
    if status == "completed_with_abstentions" and (not events or accepted_count):
        raise ValueError("completed_with_abstentions requires only abstained events")
    data["predicted_events"] = events
    data["prediction_boundary"] = _validate_literal_boundary(
        data["prediction_boundary"],
        _PREDICTION_BOUNDARY,
        "prediction attempt v2.prediction_boundary",
    )
    return data


def validate_postfreeze_ite_policy_v2(value: object | None = None) -> dict[str, Any]:
    data = _strict_object(
        deepcopy(DEFAULT_POSTFREEZE_ITE_POLICY_V2 if value is None else value),
        set(DEFAULT_POSTFREEZE_ITE_POLICY_V2),
        "postfreeze ITE policy v2",
    )
    if data["schema_version"] != POSTFREEZE_ITE_POLICY_V2_SCHEMA_VERSION:
        raise ValueError("postfreeze ITE policy v2 schema_version mismatch")
    data["policy_id"] = _identifier(
        data["policy_id"], "postfreeze ITE policy v2.policy_id"
    )
    data["minimum_event_interval_iou"] = _finite_nonnegative(
        data["minimum_event_interval_iou"],
        "postfreeze ITE policy v2.minimum_event_interval_iou",
    )
    if data["minimum_event_interval_iou"] > 1:
        raise ValueError("minimum_event_interval_iou must be in [0, 1]")
    data["maximum_onset_distance_seconds"] = _finite_nonnegative(
        data["maximum_onset_distance_seconds"],
        "postfreeze ITE policy v2.maximum_onset_distance_seconds",
    )
    for key, expected in DEFAULT_POSTFREEZE_ITE_POLICY_V2.items():
        if key in {
            "schema_version",
            "policy_id",
            "minimum_event_interval_iou",
            "maximum_onset_distance_seconds",
        }:
            continue
        if data[key] != expected:
            raise ValueError(f"postfreeze ITE policy v2.{key} is fail-closed")
    return data


def _interval_match_details(
    predicted: Mapping[str, Any],
    reference: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    pred_lower = float(predicted["lower_seconds"])
    pred_upper = float(predicted["upper_seconds"])
    ref_lower = float(reference["lower_seconds"])
    ref_upper = float(reference["upper_seconds"])
    intersection = max(0.0, min(pred_upper, ref_upper) - max(pred_lower, ref_lower))
    union = max(pred_upper, ref_upper) - min(pred_lower, ref_lower)
    if union == 0 and pred_lower == ref_lower:
        iou = 1.0
    else:
        iou = intersection / union if union > 0 else 0.0
    if ref_lower <= pred_lower <= ref_upper:
        onset_distance = 0.0
    else:
        onset_distance = min(abs(pred_lower - ref_lower), abs(pred_lower - ref_upper))
    eligible = (
        iou > 0 and iou >= float(policy["minimum_event_interval_iou"])
    ) or onset_distance <= float(policy["maximum_onset_distance_seconds"])
    onset_proximity = 1.0 / (1.0 + onset_distance)
    return {
        "eligible": eligible,
        "interval_iou": round(iou, 12),
        "onset_distance_seconds": round(onset_distance, 12),
        "secondary_score": round(iou + onset_proximity, 12),
    }


def _maximum_weight_pairs(weights: Sequence[Sequence[float]]) -> list[tuple[int, int]]:
    row_count = len(weights)
    column_count = max((len(row) for row in weights), default=0)
    if row_count == 0 or column_count == 0:
        return []
    size = max(row_count, column_count)
    maximum = max((max(row) for row in weights if row), default=0.0)
    cost = [[maximum for _ in range(size)] for _ in range(size)]
    for row_index, row in enumerate(weights):
        for column_index, value in enumerate(row):
            cost[row_index][column_index] = maximum - float(value)
    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    p = [0] * (size + 1)
    way = [0] * (size + 1)
    for row_index in range(1, size + 1):
        p[0] = row_index
        column0 = 0
        minv = [math.inf] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[column0] = True
            row0 = p[column0]
            delta = math.inf
            column1 = 0
            for column in range(1, size + 1):
                if used[column]:
                    continue
                current = cost[row0 - 1][column - 1] - u[row0] - v[column]
                if current < minv[column]:
                    minv[column] = current
                    way[column] = column0
                if minv[column] < delta:
                    delta = minv[column]
                    column1 = column
            for column in range(size + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    minv[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = way[column0]
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break
    return [
        (p[column] - 1, column - 1)
        for column in range(1, size + 1)
        if p[column] > 0 and p[column] - 1 < row_count and column - 1 < column_count
    ]


def _match_events(
    references: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    comparisons = [
        [
            _interval_match_details(
                prediction["interval"], reference["interval"], policy
            )
            for prediction in predictions
        ]
        for reference in references
    ]
    maximum_pairs = min(len(references), len(predictions))
    cardinality_bonus = 2.0 * (maximum_pairs + 1)
    weights = [
        [
            cardinality_bonus + float(item["secondary_score"])
            if item["eligible"]
            else 0.0
            for item in row
        ]
        for row in comparisons
    ]
    pairs: list[dict[str, Any]] = []
    matched_reference_indices: set[int] = set()
    matched_prediction_indices: set[int] = set()
    for reference_index, prediction_index in _maximum_weight_pairs(weights):
        details = comparisons[reference_index][prediction_index]
        if not details["eligible"]:
            continue
        matched_reference_indices.add(reference_index)
        matched_prediction_indices.add(prediction_index)
        pairs.append(
            {
                "reference_event_id": references[reference_index]["reference_event_id"],
                "predicted_event_id": predictions[prediction_index][
                    "predicted_event_id"
                ],
                "interval_iou": details["interval_iou"],
                "onset_distance_seconds": details["onset_distance_seconds"],
                "binding_source": "deterministic_postfreeze_interval_matching",
            }
        )
    pair_by_reference = {str(item["reference_event_id"]): item for item in pairs}
    unmatched_references: list[dict[str, Any]] = []
    for reference_index, reference in enumerate(references):
        if reference_index in matched_reference_indices:
            continue
        eligible_predictions = [
            str(predictions[index]["predicted_event_id"])
            for index, details in enumerate(comparisons[reference_index])
            if details["eligible"]
        ]
        unmatched_references.append(
            {
                "reference_event_id": reference["reference_event_id"],
                "eligible_predicted_event_ids": eligible_predictions,
                "failure_kind": (
                    "merge_or_matching_competition"
                    if eligible_predictions
                    else "missed_reference_event"
                ),
            }
        )
    unmatched_predictions: list[dict[str, Any]] = []
    for prediction_index, prediction in enumerate(predictions):
        if prediction_index in matched_prediction_indices:
            continue
        eligible_references = [
            str(references[index]["reference_event_id"])
            for index, row in enumerate(comparisons)
            if row[prediction_index]["eligible"]
        ]
        unmatched_predictions.append(
            {
                "predicted_event_id": prediction["predicted_event_id"],
                "eligible_reference_event_ids": eligible_references,
                "failure_kind": (
                    "duplicate_or_fragment_candidate"
                    if eligible_references
                    else "false_alarm_candidate"
                ),
            }
        )
    return {
        "pairs": sorted(pairs, key=lambda row: str(row["reference_event_id"])),
        "pair_by_reference": pair_by_reference,
        "unmatched_references": unmatched_references,
        "unmatched_predictions": unmatched_predictions,
    }


def _hard_scores(
    ranking: Sequence[str], labels: Sequence[str], completeness: str
) -> dict[str, float | None]:
    relevant = set(labels)
    output: dict[str, float | None] = {}
    for k in (1, 3, 5):
        hits = sum(item in relevant for item in ranking[:k])
        output[f"top{k}_hit"] = float(hits > 0)
        output[f"recall_at_{k}"] = (
            float(hits / len(relevant)) if completeness == "exhaustive" else None
        )
    first = next(
        (index for index, item in enumerate(ranking, start=1) if item in relevant),
        None,
    )
    output["mrr"] = float(1.0 / first) if first is not None else 0.0
    if completeness == "exhaustive":
        seen = 0
        precision_sum = 0.0
        for rank, electrode in enumerate(ranking, start=1):
            if electrode in relevant:
                seen += 1
                precision_sum += seen / rank
        output["average_precision"] = float(precision_sum / len(relevant))
    else:
        output["average_precision"] = None
    return output


def _dcg(gains: Sequence[float]) -> float:
    return sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))


def _soft_spread_scores(
    ranking: Sequence[str], labels: Sequence[str], *, soft_spread_gain: float
) -> dict[str, float | None]:
    relevant = set(labels)
    output: dict[str, float | None] = {}
    for k in (1, 3, 5):
        gains = [float(item in relevant) for item in ranking[:k]]
        output[f"spread_hit_at_{k}"] = float(any(gains))
        ideal = _dcg([1.0] * min(k, len(relevant)))
        output[f"spread_ndcg_at_{k}"] = _dcg(gains) / ideal if ideal > 0 else 0.0
    first = next(
        (index for index, item in enumerate(ranking, start=1) if item in relevant),
        None,
    )
    output["spread_mrr"] = float(1.0 / first) if first is not None else 0.0
    output["spread_top1_gain"] = (
        float(soft_spread_gain) if ranking and ranking[0] in relevant else 0.0
    )
    return output


def _metric_family_summary(
    rows: Sequence[Mapping[str, Any]], metrics: Sequence[str]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric in metrics:
        full_values = [
            float(row["full_scores"][metric])
            for row in rows
            if row["full_scores"] is not None and row["full_scores"][metric] is not None
        ]
        selected_values = [
            float(row["selected_scores"][metric])
            for row in rows
            if row["selected_scores"] is not None
            and row["selected_scores"][metric] is not None
        ]
        result[metric] = {
            "full_coverage": {
                "score_sum": round(sum(full_values), 12),
                "reference_defined_denominator": len(full_values),
                "mean": _ratio(sum(full_values), len(full_values)),
            },
            "selected": {
                "score_sum": round(sum(selected_values), 12),
                "selected_denominator": len(selected_values),
                "mean": _ratio(sum(selected_values), len(selected_values)),
                "coverage_over_reference_defined_denominator": _ratio(
                    len(selected_values), len(full_values)
                ),
            },
        }
    return result


def _evaluate_recording_summary(
    reference: Mapping[str, Any] | None,
    prediction: Mapping[str, Any] | None,
    attempt_status: str,
) -> dict[str, Any]:
    mapping_available = (
        reference is not None and reference["mapping_status"] == "available"
    )
    selected = prediction is not None and prediction["selection_status"] == "accepted"
    field_specs = {
        "laterality_exact": "laterality",
        "regions_exact": "regions",
        "regions_jaccard": "regions",
        "onset_uncertainty_exact": "onset_uncertainty",
    }
    fields: dict[str, Any] = {}
    for metric, field in field_specs.items():
        reference_value = reference[field] if mapping_available else None
        reference_applicable = reference_value is not None
        prediction_value = prediction[field] if prediction is not None else None
        prediction_available = selected and prediction_value is not None
        score: float | None = None
        if reference_applicable:
            if not prediction_available:
                score = 0.0
            elif metric == "regions_jaccard":
                left = set(prediction_value)
                right = set(reference_value)
                score = len(left.intersection(right)) / len(left.union(right))
            else:
                score = float(prediction_value == reference_value)
        fields[metric] = {
            "reference_applicable": reference_applicable,
            "reference_value": deepcopy(reference_value),
            "prediction_selected": prediction_available,
            "prediction_value": deepcopy(prediction_value),
            "full_coverage_score": (
                round(score, 12) if score is not None else "not_available"
            ),
            "selected_score": (
                round(score, 12)
                if score is not None and prediction_available
                else "not_available"
            ),
        }
    return {
        "reference_status": (
            "available"
            if mapping_available
            else reference["mapping_status"]
            if reference is not None
            else "missing_label"
        ),
        "prediction_status": (
            prediction["selection_status"] if prediction is not None else attempt_status
        ),
        "comparison_count_per_record": 1 if mapping_available else 0,
        "fields": fields,
    }


def evaluate_postfreeze_intention_to_evaluate_v2(
    reference_roster: object,
    prediction_attempt: object,
    *,
    policy: object | None = None,
) -> dict[str, Any]:
    """Evaluate one frozen recording using reference-first denominators."""

    reference = validate_postfreeze_ite_reference_roster_v2(reference_roster)
    prediction = validate_postfreeze_ite_prediction_attempt_v2(prediction_attempt)
    match_policy = validate_postfreeze_ite_policy_v2(policy)
    for key in ("patient_id", "recording_id"):
        if reference[key] != prediction[key]:
            raise ValueError(f"reference/prediction {key} mismatch")

    matching = _match_events(
        reference["event_references"], prediction["predicted_events"], match_policy
    )
    prediction_by_id = {
        str(item["predicted_event_id"]): item for item in prediction["predicted_events"]
    }
    pair_by_reference = matching.pop("pair_by_reference")
    hard_rows: list[dict[str, Any]] = []
    spread_rows: list[dict[str, Any]] = []
    role_rows: list[dict[str, Any]] = []
    for reference_event in reference["event_references"]:
        reference_event_id = str(reference_event["reference_event_id"])
        pair = pair_by_reference.get(reference_event_id)
        predicted_event = (
            prediction_by_id[str(pair["predicted_event_id"])]
            if pair is not None
            else None
        )
        accepted = (
            predicted_event is not None
            and predicted_event["selection_status"] == "accepted"
        )
        ranking = predicted_event["ranked_electrodes"] if accepted else []
        system_status = (
            "accepted"
            if accepted
            else "abstained"
            if predicted_event is not None
            else "unmatched_reference"
        )
        channel_reference = reference_event["channel_reference"]
        hard_labels = (
            channel_reference["hard_significant_electrodes"]
            if channel_reference is not None
            else None
        )
        soft_labels = (
            channel_reference["soft_spread_electrodes"]
            if channel_reference is not None
            else None
        )
        completeness = (
            channel_reference["reference_completeness"]
            if channel_reference is not None
            else None
        )
        hard_scores = (
            _hard_scores(ranking, hard_labels, str(completeness))
            if hard_labels
            else None
        )
        spread_scores = (
            _soft_spread_scores(
                ranking,
                soft_labels,
                soft_spread_gain=float(match_policy["soft_spread_gain"]),
            )
            if soft_labels
            else None
        )
        hard_rows.append(
            {
                "reference_event_id": reference_event_id,
                "predicted_event_id": (
                    predicted_event["predicted_event_id"]
                    if predicted_event is not None
                    else None
                ),
                "system_output_status": system_status,
                "reference_completeness": completeness or "not_available",
                "hard_significant_electrodes": deepcopy(hard_labels),
                "full_scores": hard_scores,
                "selected_scores": hard_scores if accepted else None,
            }
        )
        spread_rows.append(
            {
                "reference_event_id": reference_event_id,
                "predicted_event_id": (
                    predicted_event["predicted_event_id"]
                    if predicted_event is not None
                    else None
                ),
                "system_output_status": system_status,
                "soft_spread_electrodes": deepcopy(soft_labels),
                "full_scores": spread_scores,
                "selected_scores": spread_scores if accepted else None,
            }
        )
        if hard_labels and soft_labels:
            top1 = ranking[0] if ranking else None
            role_rows.append(
                {
                    "reference_event_id": reference_event_id,
                    "predicted_event_id": (
                        predicted_event["predicted_event_id"]
                        if predicted_event is not None
                        else None
                    ),
                    "selected": accepted,
                    "top1_electrode": top1,
                    "top1_role": (
                        "hard_significant"
                        if top1 in set(hard_labels)
                        else "soft_spread"
                        if top1 in set(soft_labels)
                        else "other"
                        if top1 is not None
                        else "unavailable"
                    ),
                }
            )

    matched_count = len(matching["pairs"])
    reference_event_count = len(reference["event_references"])
    accepted_role_rows = [row for row in role_rows if row["selected"]]
    spread_top1_count = sum(
        row["top1_role"] == "soft_spread" for row in accepted_role_rows
    )
    artifact: dict[str, Any] = {
        "schema_version": POSTFREEZE_ITE_EVALUATION_V2_SCHEMA_VERSION,
        "status": "completed_reference_first_postfreeze_ite_v2",
        "patient_id": reference["patient_id"],
        "recording_id": reference["recording_id"],
        "attempt_id": prediction["attempt_id"],
        "attempt_status": prediction["attempt_status"],
        "reference_roster_sha256": _canonical_sha256(reference),
        "prediction_attempt_sha256": _canonical_sha256(prediction),
        "prediction_bundle_sha256": prediction["prediction_bundle_sha256"],
        "policy": match_policy,
        "policy_sha256": _canonical_sha256(match_policy),
        "event_matching": {
            "reference_event_count": reference_event_count,
            "predicted_event_count": len(prediction["predicted_events"]),
            "matched_event_count": matched_count,
            "intention_to_evaluate_event_sensitivity": _ratio(
                matched_count, reference_event_count
            ),
            "pairs": matching["pairs"],
            "unmatched_references": matching["unmatched_references"],
            "unmatched_predictions": matching["unmatched_predictions"],
            "unmatched_reference_count": len(matching["unmatched_references"]),
            "false_alarm_candidate_count": sum(
                row["failure_kind"] == "false_alarm_candidate"
                for row in matching["unmatched_predictions"]
            ),
            "duplicate_or_fragment_candidate_count": sum(
                row["failure_kind"] == "duplicate_or_fragment_candidate"
                for row in matching["unmatched_predictions"]
            ),
            "reference_denominator_independent_of_prediction": True,
        },
        "recording_summary_evaluation": _evaluate_recording_summary(
            reference["recording_summary_reference"],
            prediction["recording_summary_prediction"],
            str(prediction["attempt_status"]),
        ),
        "hard_significant_evaluation": {
            "interpretation": "hard_physician_significant_known_positive_endpoint",
            "event_rows": hard_rows,
            "summary": _metric_family_summary(hard_rows, HARD_SIGNIFICANT_METRICS),
            "reference_labeled_event_count": sum(
                row["full_scores"] is not None for row in hard_rows
            ),
            "missing_hard_label_event_count": sum(
                row["full_scores"] is None for row in hard_rows
            ),
        },
        "soft_spread_evaluation": {
            "interpretation": "separate_soft_spread_retrieval_never_hard_onset_credit",
            "event_rows": spread_rows,
            "summary": _metric_family_summary(spread_rows, SOFT_SPREAD_METRICS),
            "reference_labeled_event_count": sum(
                row["full_scores"] is not None for row in spread_rows
            ),
            "missing_soft_label_event_count": sum(
                row["full_scores"] is None for row in spread_rows
            ),
        },
        "hard_vs_soft_top1_audit": {
            "event_rows": role_rows,
            "reference_event_count_with_both_roles": len(role_rows),
            "selected_event_count": len(accepted_role_rows),
            "soft_spread_selected_as_top1_count": spread_top1_count,
            "soft_spread_selected_as_top1_rate_among_selected": _ratio(
                spread_top1_count, len(accepted_role_rows)
            ),
            "selected_coverage": _ratio(len(accepted_role_rows), len(role_rows)),
        },
        "denominator_contract": {
            "full_coverage_reference_roster_is_source_independent": True,
            "abstention_scores_zero_when_reference_is_applicable": True,
            "technical_failure_scores_zero_when_reference_is_applicable": True,
            "unmatched_reference_scores_zero_when_reference_is_applicable": True,
            "missing_reference_label_is_not_available_not_zero": True,
            "selected_metrics_are_secondary_and_report_coverage": True,
            "recording_excel_summary_is_never_repeated_per_event": True,
            "event_binding_is_computed_not_caller_asserted": True,
            "hard_significant_and_soft_spread_are_never_merged": True,
        },
        "claim_boundary": {
            "legacy_postfreeze_artifact_modified": False,
            "raw_excel_text_loaded": False,
            "raw_physician_text_loaded": False,
            "edf_annotations_loaded": False,
            "clinical_text_loaded": False,
            "reference_used_for_inference": False,
            "clinical_validation_claimed": False,
        },
    }
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


__all__ = [
    "DEFAULT_POSTFREEZE_ITE_POLICY_V2",
    "HARD_SIGNIFICANT_METRICS",
    "POSTFREEZE_ITE_EVALUATION_V2_SCHEMA_VERSION",
    "POSTFREEZE_ITE_POLICY_V2_SCHEMA_VERSION",
    "POSTFREEZE_ITE_PREDICTION_ATTEMPT_V2_SCHEMA_VERSION",
    "POSTFREEZE_ITE_REFERENCE_ROSTER_V2_SCHEMA_VERSION",
    "SOFT_SPREAD_METRICS",
    "evaluate_postfreeze_intention_to_evaluate_v2",
    "validate_postfreeze_ite_policy_v2",
    "validate_postfreeze_ite_prediction_attempt_v2",
    "validate_postfreeze_ite_reference_roster_v2",
]
