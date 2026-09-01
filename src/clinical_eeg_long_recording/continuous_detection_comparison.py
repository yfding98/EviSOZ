"""Comparable, fail-closed summaries for frozen detector benchmark receipts.

The comparator never sees EEG, reference annotations, or dense posteriors.  It
accepts only content-bound v3 benchmark receipts and refuses to compare models
unless they used the identical reference inventory, recording roster, patient
roster, split, and onset tolerance.  It returns Pareto fronts rather than a
blended score.  A separate row-level entry point performs paired patient
bootstrap with one shared patient draw across every model; neither entry point
authorizes production promotion, a SOTA claim, or paired runtime inference.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import random
from typing import Any, Mapping, Sequence

from .continuous_detection_benchmark import (
    aggregate_continuous_detection_metrics,
    evaluate_patient_level_continuous_detection,
    validate_continuous_benchmark_rows,
    validate_continuous_detection_benchmark_receipt,
)


CONTINUOUS_DETECTOR_COMPARISON_SCHEMA_VERSION = (
    "continuous_detector_benchmark_comparison_v1"
)
CONTINUOUS_DETECTOR_COMPARISON_METHOD_ID = (
    "identical_reference_inventory_accuracy_efficiency_pareto_v1"
)
CONTINUOUS_DETECTOR_PAIRED_COMPARISON_SCHEMA_VERSION = (
    "continuous_detector_paired_patient_comparison_v1"
)
CONTINUOUS_DETECTOR_PAIRED_COMPARISON_METHOD_ID = (
    "paired_patient_bootstrap_benefit_difference_v1"
)

_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_BENEFIT_METRIC_NAMES = (
    "event_sensitivity",
    "patient_macro_event_sensitivity",
    "event_f1",
    "onset_hit_rate",
    "negative_false_alarms_per_recording_hour",
    "negative_onset_absolute_error_median_matched_only_seconds",
)
_PAIRED_MEMBER_FIELDS = {
    "member_id",
    "provider_id",
    "operating_point_id",
    "benchmark_receipt_id",
    "input_rows_sha256",
    "reference_inventory_sha256",
    "prediction_inventory_sha256",
    "recording_roster_sha256",
    "expected_recording_roster_sha256",
    "evaluation_patient_roster_sha256",
    "evaluation_split",
    "point_benefit_metrics",
}
_PAIRED_INTERVAL_FIELDS = {
    "positive_favors",
    "point_benefit_difference",
    "valid_replicates",
    "lower_2_5_percentile",
    "upper_97_5_percentile",
}
_PAIRED_QUALIFICATION_LIMITATIONS = [
    "paired_runtime_inference_requires_repeated_controlled_execution",
    "comparison_receipt_does_not_authorize_production_or_a_sota_claim",
]


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TypeError(f"{context} must be a non-empty trimmed string")
    if len(value) > 512 or any(ord(character) < 32 for character in value):
        raise ValueError(f"{context} is invalid")
    return value


def _sha256(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_CHARACTERS for character in value)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _optional_finite(value: object, context: str) -> float | None:
    return None if value is None else _finite(value, context)


def _normalized_identifier_roster(values: Sequence[str], *, context: str) -> list[str]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError(f"{context} must be a sequence")
    roster = sorted({_identifier(value, context) for value in values})
    if not roster:
        raise ValueError(f"{context} must not be empty")
    return roster


def _paired_member_id(provider_id: str, operating_point_id: str) -> str:
    return (
        "DETPAIRMEM-"
        + _canonical_sha256(
            {
                "provider_id": provider_id,
                "operating_point_id": operating_point_id,
            }
        )[:20]
    )


def _paired_pair_id(
    left_member_id: str,
    right_member_id: str,
    reference_inventory_sha256: str,
) -> str:
    return (
        "DETPAIR-"
        + _canonical_sha256(
            {
                "left": left_member_id,
                "right": right_member_id,
                "reference_inventory_sha256": reference_inventory_sha256,
            }
        )[:20]
    )


def _percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    location = probability * (len(ordered) - 1)
    lower = int(math.floor(location))
    upper = int(math.ceil(location))
    fraction = location - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _front(
    rows: Sequence[Mapping[str, Any]],
    dimensions: Sequence[tuple[str, str]],
) -> list[str]:
    """Return IDs not strictly dominated on all requested dimensions."""

    def dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
        no_worse = True
        strictly_better = False
        for field, direction in dimensions:
            left_value = _finite(left[field], field)
            right_value = _finite(right[field], field)
            if direction == "maximize":
                no_worse &= left_value >= right_value
                strictly_better |= left_value > right_value
            elif direction == "minimize":
                no_worse &= left_value <= right_value
                strictly_better |= left_value < right_value
            else:
                raise ValueError("Pareto direction must be maximize or minimize")
        return no_worse and strictly_better

    return sorted(
        str(row["comparison_member_id"])
        for row in rows
        if not any(
            dominates(other, row)
            for other in rows
            if other["comparison_member_id"] != row["comparison_member_id"]
        )
    )


def compare_continuous_detection_benchmark_receipts(
    receipts: Sequence[Mapping[str, Any]],
    *,
    onset_tolerance_seconds: float = 5.0,
    require_complete_inventory: bool = True,
    require_frozen_operating_points: bool = True,
) -> dict[str, Any]:
    """Compare at least two frozen providers on one identical benchmark."""

    if (
        not isinstance(receipts, Sequence)
        or isinstance(receipts, (str, bytes))
        or len(receipts) < 2
    ):
        raise ValueError("detector comparison requires at least two receipts")
    if (
        type(require_complete_inventory) is not bool
        or type(require_frozen_operating_points) is not bool
    ):
        raise TypeError("comparison requirements must be booleans")
    tolerance = _finite(onset_tolerance_seconds, "onset tolerance")
    if tolerance <= 0:
        raise ValueError("onset tolerance must be positive")
    normalized = [
        validate_continuous_detection_benchmark_receipt(dict(receipt))
        for receipt in receipts
    ]
    unique_members = {
        (str(receipt["provider_id"]), str(receipt["operating_point_id"]))
        for receipt in normalized
    }
    if len(unique_members) != len(normalized):
        raise ValueError("detector comparison members must be unique")
    comparable_fields = (
        "reference_inventory_sha256",
        "recording_roster_sha256",
        "evaluation_patient_roster_sha256",
        "evaluation_split",
    )
    for field in comparable_fields:
        if len({receipt[field] for receipt in normalized}) != 1:
            raise ValueError(f"detector receipts disagree on {field}")
    tolerance_key = f"{tolerance:g}s"
    for receipt in normalized:
        if tolerance not in tuple(
            float(value) for value in receipt["tolerances_seconds"]
        ):
            raise ValueError("detector receipt lacks the requested onset tolerance")
        if require_complete_inventory and receipt["evaluation_inventory_status"] != (
            "verified_complete_expected_recording_inventory"
        ):
            raise ValueError("detector comparison lacks complete inventory proof")
        if (
            require_frozen_operating_points
            and receipt["operating_point_frozen_before_evaluation"] is not True
        ):
            raise ValueError("detector comparison includes an unfrozen operating point")

    members: list[dict[str, Any]] = []
    for receipt in normalized:
        metrics = receipt["metrics"]
        warm = receipt["execution_metrics"]["service_state_metrics"]["warm"]
        warm_wall = warm["wall_seconds_per_eeg_hour"]
        member = {
            "comparison_member_id": "DETCMPMEM-"
            + _canonical_sha256(
                {
                    "benchmark_receipt_id": receipt["benchmark_receipt_id"],
                    "provider_id": receipt["provider_id"],
                    "operating_point_id": receipt["operating_point_id"],
                }
            )[:20],
            "provider_id": receipt["provider_id"],
            "operating_point_id": receipt["operating_point_id"],
            "benchmark_receipt_id": receipt["benchmark_receipt_id"],
            "event_sensitivity": _finite(
                metrics["event_sensitivity"], "event sensitivity"
            ),
            "event_precision": _finite(metrics["event_precision"], "event precision"),
            "event_f1": _finite(metrics["event_f1"], "event F1"),
            "false_alarms_per_recording_hour": _finite(
                metrics["alarm_false_alarms_per_recording_hour"],
                "false alarms per recording hour",
            ),
            "onset_hit_rate": _finite(
                metrics["onset_absolute_hit_rate"][tolerance_key]["rate"],
                "onset hit rate",
            ),
            "onset_absolute_error_median_matched_only_seconds": _finite(
                metrics["onset_latency_seconds"]["absolute_median_matched_only"],
                "onset absolute error median",
            ),
            "patient_macro_event_sensitivity": _finite(
                metrics["patient_macro"]["event_sensitivity_macro"],
                "patient macro event sensitivity",
            ),
            "warm_wall_seconds_per_eeg_hour": (
                None if warm_wall is None else _finite(warm_wall, "warm wall time")
            ),
            "patient_bootstrap_available": receipt["patient_bootstrap"] is not None,
        }
        members.append(member)

    accuracy_dimensions = (
        ("event_sensitivity", "maximize"),
        ("patient_macro_event_sensitivity", "maximize"),
        ("onset_hit_rate", "maximize"),
        ("false_alarms_per_recording_hour", "minimize"),
    )
    accuracy_front = _front(members, accuracy_dimensions)
    efficiency_ready = all(
        member["warm_wall_seconds_per_eeg_hour"] is not None for member in members
    )
    accuracy_efficiency_front = (
        _front(
            members,
            (*accuracy_dimensions, ("warm_wall_seconds_per_eeg_hour", "minimize")),
        )
        if efficiency_ready
        else None
    )
    limitations: list[str] = []
    if not all(member["patient_bootstrap_available"] for member in members):
        limitations.append("one_or_more_patient_bootstrap_intervals_missing")
    if not efficiency_ready:
        limitations.append("one_or_more_warm_execution_receipts_missing")
    limitations.append(
        "aggregate_receipts_do_not_authorize_paired_patient_difference_inference"
    )
    body: dict[str, Any] = {
        "schema_version": CONTINUOUS_DETECTOR_COMPARISON_SCHEMA_VERSION,
        "comparison_receipt_id": "DETECTOR-COMPARISON-PENDING",
        "method_id": CONTINUOUS_DETECTOR_COMPARISON_METHOD_ID,
        "reference_inventory_sha256": normalized[0]["reference_inventory_sha256"],
        "recording_roster_sha256": normalized[0]["recording_roster_sha256"],
        "evaluation_patient_roster_sha256": normalized[0][
            "evaluation_patient_roster_sha256"
        ],
        "evaluation_split": normalized[0]["evaluation_split"],
        "onset_tolerance_seconds": tolerance,
        "members": members,
        "accuracy_pareto_front_member_ids": accuracy_front,
        "accuracy_efficiency_pareto_front_member_ids": accuracy_efficiency_front,
        "blended_accuracy_efficiency_score_used": False,
        "paired_patient_difference_confidence_intervals_available": False,
        "qualification_limitations": limitations,
        "production_promotion_status": "comparison_only_not_a_promotion_receipt",
        "sota_claim_authorized": False,
    }
    body["comparison_receipt_id"] = "DETCMP-" + _canonical_sha256(body)[:24]
    return validate_continuous_detection_comparison_receipt(body)


def _benefit_metrics(
    metrics: Mapping[str, Any], *, tolerance_key: str
) -> dict[str, float | None]:
    """Orient every scalar so a larger difference favours the left model."""

    raw = {
        "event_sensitivity": metrics["event_sensitivity"],
        "patient_macro_event_sensitivity": metrics["patient_macro"][
            "event_sensitivity_macro"
        ],
        "event_f1": metrics["event_f1"],
        "onset_hit_rate": metrics["onset_absolute_hit_rate"][tolerance_key]["rate"],
        "negative_false_alarms_per_recording_hour": (
            None
            if metrics["alarm_false_alarms_per_recording_hour"] is None
            else -float(metrics["alarm_false_alarms_per_recording_hour"])
        ),
        "negative_onset_absolute_error_median_matched_only_seconds": (
            None
            if metrics["onset_latency_seconds"]["absolute_median_matched_only"] is None
            else -float(
                metrics["onset_latency_seconds"]["absolute_median_matched_only"]
            )
        ),
    }
    return {
        name: None if value is None else _finite(value, name)
        for name, value in raw.items()
    }


def compare_continuous_detection_rows_paired(
    members: Sequence[Mapping[str, Any]],
    *,
    development_patient_ids: Sequence[str],
    expected_evaluation_recording_ids: Sequence[str],
    onset_tolerance_seconds: float = 5.0,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 20260821,
) -> dict[str, Any]:
    """Run paired patient bootstrap on frozen predictions for the same records."""

    if (
        not isinstance(members, Sequence)
        or isinstance(members, (str, bytes))
        or len(members) < 2
    ):
        raise ValueError("paired detector comparison requires at least two members")
    if (
        isinstance(bootstrap_replicates, bool)
        or not isinstance(bootstrap_replicates, int)
        or bootstrap_replicates < 1
    ):
        raise ValueError(
            "paired detector comparison needs positive bootstrap replicates"
        )
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int):
        raise TypeError("paired detector bootstrap seed must be an integer")
    tolerance = _finite(onset_tolerance_seconds, "onset tolerance")
    if tolerance <= 0:
        raise ValueError("onset tolerance must be positive")
    tolerance_key = f"{tolerance:g}s"
    development_roster = _normalized_identifier_roster(
        development_patient_ids,
        context="development patient ID",
    )
    expected_recording_roster = _normalized_identifier_roster(
        expected_evaluation_recording_ids,
        context="expected evaluation recording ID",
    )

    normalized_members: list[dict[str, Any]] = []
    rows_by_member: dict[str, list[dict[str, Any]]] = {}
    provider_operating_points: set[tuple[str, str]] = set()
    for raw in members:
        if type(raw) is not dict or set(raw) != {
            "provider_id",
            "operating_point_id",
            "rows",
        }:
            raise ValueError("paired detector member fields are invalid")
        provider_id = _identifier(raw["provider_id"], "provider_id")
        operating_point_id = _identifier(
            raw["operating_point_id"], "operating_point_id"
        )
        provider_operating_point = (provider_id, operating_point_id)
        if provider_operating_point in provider_operating_points:
            raise ValueError(
                "paired detector provider/operating-point pairs must be unique"
            )
        provider_operating_points.add(provider_operating_point)
        validated_rows = validate_continuous_benchmark_rows(raw["rows"])
        receipt = evaluate_patient_level_continuous_detection(
            provider_id=provider_id,
            operating_point_id=operating_point_id,
            rows=validated_rows,
            operating_point_frozen_before_evaluation=True,
            development_patient_ids=development_roster,
            expected_evaluation_recording_ids=expected_recording_roster,
            bootstrap_replicates=0,
        )
        member_id = _paired_member_id(provider_id, operating_point_id)
        if member_id in rows_by_member:
            raise ValueError("paired detector member IDs must be unique")
        rows_by_member[member_id] = validated_rows
        normalized_members.append(
            {
                "member_id": member_id,
                "provider_id": provider_id,
                "operating_point_id": operating_point_id,
                "benchmark_receipt_id": receipt["benchmark_receipt_id"],
                "input_rows_sha256": receipt["input_rows_sha256"],
                "reference_inventory_sha256": receipt["reference_inventory_sha256"],
                "prediction_inventory_sha256": receipt["prediction_inventory_sha256"],
                "recording_roster_sha256": receipt["recording_roster_sha256"],
                "expected_recording_roster_sha256": receipt[
                    "expected_recording_roster_sha256"
                ],
                "evaluation_patient_roster_sha256": receipt[
                    "evaluation_patient_roster_sha256"
                ],
                "evaluation_split": receipt["evaluation_split"],
                "point_benefit_metrics": _benefit_metrics(
                    receipt["metrics"], tolerance_key=tolerance_key
                ),
            }
        )

    comparable_member_fields = (
        "reference_inventory_sha256",
        "recording_roster_sha256",
        "expected_recording_roster_sha256",
        "evaluation_patient_roster_sha256",
        "evaluation_split",
    )
    for field in comparable_member_fields:
        if len({member[field] for member in normalized_members}) != 1:
            raise ValueError(f"paired detector members disagree on {field}")

    first_rows = rows_by_member[normalized_members[0]["member_id"]]
    patients = sorted({str(row["patient_id"]) for row in first_rows})
    if not patients:
        raise ValueError("paired detector comparison has no patients")
    for member in normalized_members[1:]:
        member_patients = sorted(
            {str(row["patient_id"]) for row in rows_by_member[member["member_id"]]}
        )
        if member_patients != patients:
            raise ValueError("paired detector members disagree on patient roster")

    patient_rows_by_member: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for member in normalized_members:
        by_patient: dict[str, list[dict[str, Any]]] = {}
        for row in rows_by_member[member["member_id"]]:
            by_patient.setdefault(str(row["patient_id"]), []).append(row)
        patient_rows_by_member[member["member_id"]] = by_patient

    ordered_members = sorted(normalized_members, key=lambda value: value["member_id"])
    pair_definitions: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for left_index, left in enumerate(ordered_members):
        for right in ordered_members[left_index + 1 :]:
            pair_definitions.append((left, right))
    distributions: dict[tuple[str, str, str], list[float]] = {}
    random_state = random.Random(bootstrap_seed)
    for _ in range(bootstrap_replicates):
        sampled_patients = [
            patients[random_state.randrange(len(patients))] for _ in patients
        ]
        replicate_metrics: dict[str, dict[str, float | None]] = {}
        for member in ordered_members:
            sampled_rows: list[dict[str, Any]] = []
            for draw_index, patient_id in enumerate(sampled_patients):
                for raw_row in patient_rows_by_member[member["member_id"]][patient_id]:
                    row = deepcopy(raw_row)
                    row["patient_id"] = f"BOOT-{draw_index:05d}-{patient_id}"
                    row[
                        "recording_id"
                    ] = f"BOOT-{draw_index:05d}-{raw_row['recording_id']}"
                    sampled_rows.append(row)
            replicate_metrics[member["member_id"]] = _benefit_metrics(
                aggregate_continuous_detection_metrics(sampled_rows),
                tolerance_key=tolerance_key,
            )
        for left, right in pair_definitions:
            left_values = replicate_metrics[left["member_id"]]
            right_values = replicate_metrics[right["member_id"]]
            for metric_name in left_values:
                left_value = left_values[metric_name]
                right_value = right_values[metric_name]
                if left_value is None or right_value is None:
                    continue
                distributions.setdefault(
                    (left["member_id"], right["member_id"], metric_name), []
                ).append(float(left_value) - float(right_value))

    pairwise: list[dict[str, Any]] = []
    for left, right in pair_definitions:
        point_left = left["point_benefit_metrics"]
        point_right = right["point_benefit_metrics"]
        intervals: dict[str, Any] = {}
        for metric_name in point_left:
            values = distributions.get(
                (left["member_id"], right["member_id"], metric_name), []
            )
            point_difference = (
                None
                if point_left[metric_name] is None or point_right[metric_name] is None
                else float(point_left[metric_name]) - float(point_right[metric_name])
            )
            intervals[metric_name] = {
                "positive_favors": "left_member",
                "point_benefit_difference": point_difference,
                "valid_replicates": len(values),
                "lower_2_5_percentile": _percentile(values, 0.025),
                "upper_97_5_percentile": _percentile(values, 0.975),
            }
        pairwise.append(
            {
                "pair_id": _paired_pair_id(
                    left["member_id"],
                    right["member_id"],
                    str(left["reference_inventory_sha256"]),
                ),
                "left_member_id": left["member_id"],
                "right_member_id": right["member_id"],
                "benefit_difference_intervals": intervals,
            }
        )

    body: dict[str, Any] = {
        "schema_version": CONTINUOUS_DETECTOR_PAIRED_COMPARISON_SCHEMA_VERSION,
        "paired_comparison_receipt_id": "DETECTOR-PAIRED-COMPARISON-PENDING",
        "method_id": CONTINUOUS_DETECTOR_PAIRED_COMPARISON_METHOD_ID,
        "reference_inventory_sha256": ordered_members[0]["reference_inventory_sha256"],
        "recording_roster_sha256": ordered_members[0]["recording_roster_sha256"],
        "evaluation_patient_roster_sha256": _canonical_sha256(patients),
        "expected_evaluation_recording_roster_sha256": _canonical_sha256(
            expected_recording_roster
        ),
        "evaluation_split": ordered_members[0]["evaluation_split"],
        "onset_tolerance_seconds": tolerance,
        "members": ordered_members,
        "pairwise_patient_bootstrap": pairwise,
        "bootstrap": {
            "unit": "patient",
            "paired_resampling_draw_shared_by_all_models": True,
            "method": "percentile_bootstrap",
            "seed": bootstrap_seed,
            "requested_replicates": bootstrap_replicates,
            "confidence_level": 0.95,
        },
        "paired_patient_difference_confidence_intervals_available": True,
        "efficiency_difference_inference_available": False,
        "scope_receipt": {
            "predictions_frozen_before_reference_scoring": True,
            "reference_labels_available_to_detector_provider": False,
            "edf_annotations_used": False,
            "excel_or_doctor_labels_used": False,
        },
        "qualification_limitations": list(_PAIRED_QUALIFICATION_LIMITATIONS),
        "production_promotion_status": "paired_comparison_only_not_a_promotion_receipt",
        "sota_claim_authorized": False,
    }
    body["paired_comparison_receipt_id"] = "DETPAIRCMP-" + _canonical_sha256(body)[:24]
    return validate_continuous_detection_paired_comparison_receipt(body)


def validate_continuous_detection_paired_comparison_receipt(
    payload: object,
) -> dict[str, Any]:
    """Recompute the paired receipt's closed-world semantic contract."""

    if type(payload) is not dict:
        raise TypeError("paired detector comparison receipt must be an object")
    data = deepcopy(payload)
    required = {
        "schema_version",
        "paired_comparison_receipt_id",
        "method_id",
        "reference_inventory_sha256",
        "recording_roster_sha256",
        "evaluation_patient_roster_sha256",
        "expected_evaluation_recording_roster_sha256",
        "evaluation_split",
        "onset_tolerance_seconds",
        "members",
        "pairwise_patient_bootstrap",
        "bootstrap",
        "paired_patient_difference_confidence_intervals_available",
        "efficiency_difference_inference_available",
        "scope_receipt",
        "qualification_limitations",
        "production_promotion_status",
        "sota_claim_authorized",
    }
    if set(data) != required:
        raise ValueError("paired detector comparison receipt has invalid fields")
    if (
        data["schema_version"] != CONTINUOUS_DETECTOR_PAIRED_COMPARISON_SCHEMA_VERSION
        or data["method_id"] != CONTINUOUS_DETECTOR_PAIRED_COMPARISON_METHOD_ID
    ):
        raise ValueError("paired detector comparison schema or method drifted")
    for field in (
        "reference_inventory_sha256",
        "recording_roster_sha256",
        "evaluation_patient_roster_sha256",
        "expected_evaluation_recording_roster_sha256",
    ):
        _sha256(data[field], f"paired detector comparison {field}")
    if (
        data["recording_roster_sha256"]
        != data["expected_evaluation_recording_roster_sha256"]
    ):
        raise ValueError(
            "paired detector comparison recording roster lacks complete expected "
            "inventory identity"
        )
    evaluation_split = _identifier(data["evaluation_split"], "evaluation_split")
    tolerance = _finite(data["onset_tolerance_seconds"], "onset tolerance")
    if tolerance <= 0:
        raise ValueError("paired detector comparison onset tolerance must be positive")
    if data["paired_patient_difference_confidence_intervals_available"] is not True:
        raise ValueError("paired detector comparison lost its paired inference flag")
    if data["efficiency_difference_inference_available"] is not False:
        raise ValueError("paired detector comparison overclaimed runtime inference")
    expected_scope = {
        "predictions_frozen_before_reference_scoring": True,
        "reference_labels_available_to_detector_provider": False,
        "edf_annotations_used": False,
        "excel_or_doctor_labels_used": False,
    }
    if data["scope_receipt"] != expected_scope:
        raise ValueError("paired detector comparison violated its input firewall")
    if data["qualification_limitations"] != _PAIRED_QUALIFICATION_LIMITATIONS:
        raise ValueError("paired detector comparison limitations drifted")
    if (
        data["production_promotion_status"]
        != ("paired_comparison_only_not_a_promotion_receipt")
        or data["sota_claim_authorized"] is not False
    ):
        raise ValueError("paired detector comparison exceeded its scientific scope")

    bootstrap = data["bootstrap"]
    bootstrap_fields = {
        "unit",
        "paired_resampling_draw_shared_by_all_models",
        "method",
        "seed",
        "requested_replicates",
        "confidence_level",
    }
    if type(bootstrap) is not dict or set(bootstrap) != bootstrap_fields:
        raise ValueError("paired detector bootstrap fields are invalid")
    if (
        bootstrap["unit"] != "patient"
        or bootstrap["paired_resampling_draw_shared_by_all_models"] is not True
        or bootstrap["method"] != "percentile_bootstrap"
    ):
        raise ValueError("paired detector bootstrap method or unit drifted")
    if isinstance(bootstrap["seed"], bool) or not isinstance(bootstrap["seed"], int):
        raise TypeError("paired detector bootstrap seed must be an integer")
    requested_replicates = bootstrap["requested_replicates"]
    if (
        isinstance(requested_replicates, bool)
        or not isinstance(requested_replicates, int)
        or requested_replicates < 1
    ):
        raise ValueError("paired detector bootstrap replicate count is invalid")
    if _finite(bootstrap["confidence_level"], "bootstrap confidence level") != 0.95:
        raise ValueError("paired detector bootstrap confidence level drifted")

    members = data["members"]
    pairs = data["pairwise_patient_bootstrap"]
    if not isinstance(members, list) or len(members) < 2:
        raise ValueError("paired detector comparison has too few members")
    if any(
        type(member) is not dict or set(member) != _PAIRED_MEMBER_FIELDS
        for member in members
    ):
        raise ValueError("paired detector comparison member fields are invalid")
    if members != sorted(members, key=lambda member: str(member["member_id"])):
        raise ValueError("paired detector comparison members are not canonical")

    member_by_id: dict[str, dict[str, Any]] = {}
    provider_operating_points: set[tuple[str, str]] = set()
    benchmark_receipt_ids: set[str] = set()
    for index, member in enumerate(members):
        context = f"paired detector member {index}"
        provider_id = _identifier(member["provider_id"], f"{context} provider_id")
        operating_point_id = _identifier(
            member["operating_point_id"], f"{context} operating_point_id"
        )
        member_id = _identifier(member["member_id"], f"{context} member_id")
        expected_member_id = _paired_member_id(provider_id, operating_point_id)
        if member_id != expected_member_id:
            raise ValueError("paired detector member ID is inconsistent")
        provider_operating_point = (provider_id, operating_point_id)
        if provider_operating_point in provider_operating_points:
            raise ValueError(
                "paired detector provider/operating-point pair is duplicated"
            )
        provider_operating_points.add(provider_operating_point)
        if member_id in member_by_id:
            raise ValueError("paired detector member ID is duplicated")
        member_by_id[member_id] = member
        benchmark_receipt_id = _identifier(
            member["benchmark_receipt_id"], f"{context} benchmark_receipt_id"
        )
        if benchmark_receipt_id in benchmark_receipt_ids:
            raise ValueError("paired detector benchmark receipt ID is duplicated")
        benchmark_receipt_ids.add(benchmark_receipt_id)
        for field in (
            "input_rows_sha256",
            "reference_inventory_sha256",
            "prediction_inventory_sha256",
            "recording_roster_sha256",
            "expected_recording_roster_sha256",
            "evaluation_patient_roster_sha256",
        ):
            _sha256(member[field], f"{context} {field}")
        if member["reference_inventory_sha256"] != data["reference_inventory_sha256"]:
            raise ValueError("paired detector member reference inventory disagrees")
        if member["recording_roster_sha256"] != data["recording_roster_sha256"]:
            raise ValueError("paired detector member recording roster disagrees")
        if (
            member["expected_recording_roster_sha256"]
            != data["expected_evaluation_recording_roster_sha256"]
        ):
            raise ValueError(
                "paired detector member expected recording roster disagrees"
            )
        if (
            member["evaluation_patient_roster_sha256"]
            != data["evaluation_patient_roster_sha256"]
        ):
            raise ValueError("paired detector member patient roster disagrees")
        if _identifier(member["evaluation_split"], f"{context} evaluation_split") != (
            evaluation_split
        ):
            raise ValueError("paired detector member evaluation split disagrees")

        point_metrics = member["point_benefit_metrics"]
        if type(point_metrics) is not dict or set(point_metrics) != set(
            _BENEFIT_METRIC_NAMES
        ):
            raise ValueError("paired detector point metric fields are invalid")
        for metric_name in _BENEFIT_METRIC_NAMES:
            value = _optional_finite(
                point_metrics[metric_name], f"{context} point metric {metric_name}"
            )
            if value is None:
                continue
            if (
                metric_name
                in {
                    "event_sensitivity",
                    "patient_macro_event_sensitivity",
                    "event_f1",
                    "onset_hit_rate",
                }
                and not 0.0 <= value <= 1.0
            ):
                raise ValueError("paired detector rate benefit metric is out of range")
            if metric_name.startswith("negative_") and value > 0.0:
                raise ValueError(
                    "paired detector negated cost metric must be non-positive"
                )

    expected_pairs = [
        (left["member_id"], right["member_id"])
        for left_index, left in enumerate(members)
        for right in members[left_index + 1 :]
    ]
    if not isinstance(pairs, list) or len(pairs) != len(expected_pairs):
        raise ValueError("paired detector comparison pair count is inconsistent")
    for index, ((expected_left, expected_right), pair) in enumerate(
        zip(expected_pairs, pairs)
    ):
        if type(pair) is not dict or set(pair) != {
            "pair_id",
            "left_member_id",
            "right_member_id",
            "benefit_difference_intervals",
        }:
            raise ValueError("paired detector comparison pair fields are invalid")
        if (
            pair["left_member_id"] != expected_left
            or pair["right_member_id"] != expected_right
        ):
            raise ValueError("paired detector comparison pair closure is inconsistent")
        expected_pair_id = _paired_pair_id(
            expected_left,
            expected_right,
            data["reference_inventory_sha256"],
        )
        if pair["pair_id"] != expected_pair_id:
            raise ValueError("paired detector pair ID is inconsistent")
        intervals = pair["benefit_difference_intervals"]
        if type(intervals) is not dict or set(intervals) != set(_BENEFIT_METRIC_NAMES):
            raise ValueError("paired detector interval metric fields are invalid")
        left_metrics = member_by_id[expected_left]["point_benefit_metrics"]
        right_metrics = member_by_id[expected_right]["point_benefit_metrics"]
        for metric_name in _BENEFIT_METRIC_NAMES:
            interval = intervals[metric_name]
            if type(interval) is not dict or set(interval) != _PAIRED_INTERVAL_FIELDS:
                raise ValueError("paired detector interval fields are invalid")
            if interval["positive_favors"] != "left_member":
                raise ValueError("paired detector benefit direction drifted")
            left_value = left_metrics[metric_name]
            right_value = right_metrics[metric_name]
            expected_difference = (
                None
                if left_value is None or right_value is None
                else float(left_value) - float(right_value)
            )
            point_difference = _optional_finite(
                interval["point_benefit_difference"],
                f"pair {index} {metric_name} point difference",
            )
            if point_difference != expected_difference:
                raise ValueError(
                    "paired detector point benefit difference is inconsistent"
                )
            valid_replicates = interval["valid_replicates"]
            if (
                isinstance(valid_replicates, bool)
                or not isinstance(valid_replicates, int)
                or not 0 <= valid_replicates <= requested_replicates
            ):
                raise ValueError("paired detector valid replicate count is invalid")
            lower = _optional_finite(
                interval["lower_2_5_percentile"],
                f"pair {index} {metric_name} lower percentile",
            )
            upper = _optional_finite(
                interval["upper_97_5_percentile"],
                f"pair {index} {metric_name} upper percentile",
            )
            if point_difference is None and valid_replicates != 0:
                raise ValueError(
                    "paired detector unavailable point metric has valid replicates"
                )
            if valid_replicates == 0:
                if lower is not None or upper is not None:
                    raise ValueError(
                        "paired detector empty bootstrap interval must be null"
                    )
            elif lower is None or upper is None or lower > upper:
                raise ValueError(
                    "paired detector bootstrap interval bounds are invalid"
                )
            if (
                metric_name == "negative_false_alarms_per_recording_hour"
                and valid_replicates != requested_replicates
            ):
                raise ValueError(
                    "paired detector false-alarm bootstrap must use every replicate"
                )

    digest = deepcopy(data)
    digest["paired_comparison_receipt_id"] = "DETECTOR-PAIRED-COMPARISON-PENDING"
    expected_id = "DETPAIRCMP-" + _canonical_sha256(digest)[:24]
    if data["paired_comparison_receipt_id"] != expected_id:
        raise ValueError("paired detector comparison receipt is not content-bound")
    return data


def validate_continuous_detection_comparison_receipt(
    payload: object,
) -> dict[str, Any]:
    """Validate immutable comparison output and its non-promotion boundary."""

    if type(payload) is not dict:
        raise TypeError("detector comparison receipt must be an object")
    data = deepcopy(payload)
    required = {
        "schema_version",
        "comparison_receipt_id",
        "method_id",
        "reference_inventory_sha256",
        "recording_roster_sha256",
        "evaluation_patient_roster_sha256",
        "evaluation_split",
        "onset_tolerance_seconds",
        "members",
        "accuracy_pareto_front_member_ids",
        "accuracy_efficiency_pareto_front_member_ids",
        "blended_accuracy_efficiency_score_used",
        "paired_patient_difference_confidence_intervals_available",
        "qualification_limitations",
        "production_promotion_status",
        "sota_claim_authorized",
    }
    if set(data) != required:
        raise ValueError("detector comparison receipt has invalid fields")
    if (
        data["schema_version"] != CONTINUOUS_DETECTOR_COMPARISON_SCHEMA_VERSION
        or data["method_id"] != CONTINUOUS_DETECTOR_COMPARISON_METHOD_ID
    ):
        raise ValueError("detector comparison schema or method drifted")
    for field in (
        "reference_inventory_sha256",
        "recording_roster_sha256",
        "evaluation_patient_roster_sha256",
    ):
        value = data[field]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"detector comparison {field} is invalid")
    if _finite(data["onset_tolerance_seconds"], "onset tolerance") <= 0:
        raise ValueError("detector comparison onset tolerance must be positive")
    if data["blended_accuracy_efficiency_score_used"] is not False:
        raise ValueError("detector comparison must not hide tradeoffs in one score")
    if data["paired_patient_difference_confidence_intervals_available"] is not False:
        raise ValueError("aggregate detector comparison cannot claim paired inference")
    if (
        data["production_promotion_status"]
        != ("comparison_only_not_a_promotion_receipt")
        or data["sota_claim_authorized"] is not False
    ):
        raise ValueError("detector comparison exceeded its scientific scope")
    members = data["members"]
    if not isinstance(members, list) or len(members) < 2:
        raise ValueError("detector comparison requires at least two members")
    member_fields = {
        "comparison_member_id",
        "provider_id",
        "operating_point_id",
        "benchmark_receipt_id",
        "event_sensitivity",
        "event_precision",
        "event_f1",
        "false_alarms_per_recording_hour",
        "onset_hit_rate",
        "onset_absolute_error_median_matched_only_seconds",
        "patient_macro_event_sensitivity",
        "warm_wall_seconds_per_eeg_hour",
        "patient_bootstrap_available",
    }
    if any(
        type(member) is not dict or set(member) != member_fields for member in members
    ):
        raise ValueError("detector comparison member fields are invalid")
    member_ids = {member.get("comparison_member_id") for member in members}
    if None in member_ids or len(member_ids) != len(members):
        raise ValueError("detector comparison member IDs are invalid")
    if not set(data["accuracy_pareto_front_member_ids"]) <= member_ids:
        raise ValueError("detector accuracy Pareto front references an unknown member")
    accuracy_dimensions = (
        ("event_sensitivity", "maximize"),
        ("patient_macro_event_sensitivity", "maximize"),
        ("onset_hit_rate", "maximize"),
        ("false_alarms_per_recording_hour", "minimize"),
    )
    if data["accuracy_pareto_front_member_ids"] != _front(members, accuracy_dimensions):
        raise ValueError("detector accuracy Pareto front is inconsistent")
    efficiency_front = data["accuracy_efficiency_pareto_front_member_ids"]
    if efficiency_front is not None and not set(efficiency_front) <= member_ids:
        raise ValueError(
            "detector efficiency Pareto front references an unknown member"
        )
    efficiency_ready = all(
        member["warm_wall_seconds_per_eeg_hour"] is not None for member in members
    )
    expected_efficiency_front = (
        _front(
            members,
            (*accuracy_dimensions, ("warm_wall_seconds_per_eeg_hour", "minimize")),
        )
        if efficiency_ready
        else None
    )
    if efficiency_front != expected_efficiency_front:
        raise ValueError("detector accuracy-efficiency Pareto front is inconsistent")
    digest = deepcopy(data)
    digest["comparison_receipt_id"] = "DETECTOR-COMPARISON-PENDING"
    if data["comparison_receipt_id"] != "DETCMP-" + _canonical_sha256(digest)[:24]:
        raise ValueError("detector comparison receipt is not content-bound")
    return data


__all__ = [
    "CONTINUOUS_DETECTOR_COMPARISON_METHOD_ID",
    "CONTINUOUS_DETECTOR_COMPARISON_SCHEMA_VERSION",
    "CONTINUOUS_DETECTOR_PAIRED_COMPARISON_METHOD_ID",
    "CONTINUOUS_DETECTOR_PAIRED_COMPARISON_SCHEMA_VERSION",
    "compare_continuous_detection_benchmark_receipts",
    "compare_continuous_detection_rows_paired",
    "validate_continuous_detection_comparison_receipt",
    "validate_continuous_detection_paired_comparison_receipt",
]
