"""Additive G0a geometry controls and cardinality-first event matching.

This module deliberately does not modify the frozen ``...candidate_roster_v1``
contract.  It consumes that prediction-first, patient-OOF roster and adds two
pieces that are required before a credible G0a experiment can be run:

* reference-free random backgrounds matched to each detector candidate on
  support length, anchor fraction, record-edge stratum, censor signature, and
  EEG/QC opportunity geometry; and
* detector-event matching that maximizes global bipartite cardinality before a
  fixed lexicographic quality preference (IoU, anchor error, stable IDs).

Random backgrounds are selected by bounded SHA-256 rejection sampling.  The
first admissible random draw is used: there is no nearest-neighbour fallback,
grid search, target-aware relocation, or tolerance relaxation.  Exhausting the
declared draw budget raises ``BAIEGG0AGeometrySamplingInfeasible`` and therefore
fails closed.

Only detector proposals participate in the event-sensitivity matching.  A
post-freeze random background that collides with a public event is labelled as
a positive collision for training audit, but never consumes an event or
improves detector recall.  Completed zero-candidate and technical-failure
records remain in the record/event denominator.

Passing this software/lineage contract is *not* evidence that G0a passed, that
the detector is accurate, that the random controls are statistically balanced
on real data, or that any model is clinically valid.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Any, Final, Mapping, Sequence

from .ba_ieg_g0_a1_candidate_roster_v1 import (
    BA_IEG_G0_A1_RECORD_OUTCOMES,
    validate_ba_ieg_g0_a1_prediction_roster_v1,
    validate_ba_ieg_g0_a1_reference_roster_v1,
)


BA_IEG_G0A_GEOMETRY_POLICY_SCHEMA_V2: Final[str] = (
    "ba_ieg_g0a_reference_free_geometry_background_policy_v2"
)
BA_IEG_G0A_GEOMETRY_FREEZE_SCHEMA_V2: Final[str] = (
    "ba_ieg_g0a_reference_free_geometry_background_freeze_v2"
)
BA_IEG_G0A_GLOBAL_MATCH_POLICY_SCHEMA_V2: Final[str] = (
    "ba_ieg_g0a_maximum_cardinality_global_match_policy_v2"
)
BA_IEG_G0A_GLOBAL_JOIN_SCHEMA_V2: Final[str] = (
    "ba_ieg_g0a_maximum_cardinality_postfreeze_target_join_v2"
)
BA_IEG_G0A_ORIGINS_V2: Final[tuple[str, ...]] = (
    "detector_proposal",
    "candidate_geometry_matched_random_background",
)
BA_IEG_G0A_TRAINING_CLASSES_V2: Final[tuple[str, ...]] = (
    "matched_true_event",
    "unmatched_false_candidate",
    "fragmented_or_duplicate_hard_candidate",
    "near_event_hard_candidate",
    "candidate_geometry_matched_random_background",
)

_SHA256_ALPHABET = frozenset("0123456789abcdef")
_EPS = 1e-9


class BAIEGG0AGeometrySamplingInfeasible(RuntimeError):
    """Raised when declared true-random rejection sampling cannot materialize."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _seal(body: Mapping[str, Any], *, id_field: str, prefix: str) -> dict[str, Any]:
    result = deepcopy(dict(body))
    result[id_field] = prefix + "-PENDING"
    result["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    id_source = deepcopy(result)
    result[id_field] = prefix + "-" + _digest(id_source)[:24]
    receipt_source = deepcopy(result)
    receipt_source["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    result["receipt_sha256"] = _digest(receipt_source)
    return result


def _replay_seal(
    value: Mapping[str, Any], *, id_field: str, prefix: str, context: str
) -> None:
    expected = _seal(value, id_field=id_field, prefix=prefix)
    if (
        value.get(id_field) != expected[id_field]
        or value.get("receipt_sha256") != expected["receipt_sha256"]
    ):
        raise ValueError(f"{context} content address does not replay")


def _strict(value: object, fields: set[str], context: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{context} fields drifted")
    return deepcopy(value)


def _identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TypeError(f"{context} must be a non-empty trimmed string")
    return value


def _sha(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or set(value).difference(_SHA256_ALPHABET)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _finite(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _positive_int(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TypeError(f"{context} must be a positive integer")
    return value


def build_ba_ieg_g0a_geometry_background_policy_v2(
    *,
    seed_sha256: str,
    backgrounds_per_detector_candidate: int = 1,
    maximum_random_draws_per_background: int = 4096,
    record_edge_bin_seconds: float = 30.0,
    record_edge_interior_cap_bins: int = 4,
    qc_observed_fraction_tolerance: float = 0.02,
    qc_longest_run_fraction_tolerance: float = 0.02,
    detector_exclusion_margin_seconds: float = 0.0,
) -> dict[str, Any]:
    """Freeze the target-free geometry and bounded random-sampling policy."""

    edge_width = _finite(record_edge_bin_seconds, "record-edge bin seconds")
    observed_tol = _finite(
        qc_observed_fraction_tolerance, "QC observed-fraction tolerance"
    )
    run_tol = _finite(
        qc_longest_run_fraction_tolerance, "QC longest-run tolerance"
    )
    exclusion = _finite(
        detector_exclusion_margin_seconds, "detector exclusion margin"
    )
    if edge_width <= 0 or not 0 <= observed_tol <= 1 or not 0 <= run_tol <= 1:
        raise ValueError("geometry matching tolerances are invalid")
    if exclusion < 0:
        raise ValueError("detector exclusion margin cannot be negative")
    body = {
        "schema_version": BA_IEG_G0A_GEOMETRY_POLICY_SCHEMA_V2,
        "policy_id": "BAIEG-G0A-GEOMETRY-POLICY-PENDING",
        "seed_sha256": _sha(seed_sha256, "random seed"),
        "backgrounds_per_detector_candidate": _positive_int(
            backgrounds_per_detector_candidate, "background count"
        ),
        "maximum_random_draws_per_background": _positive_int(
            maximum_random_draws_per_background, "maximum random draws"
        ),
        "record_edge_bin_seconds": edge_width,
        "record_edge_interior_cap_bins": _positive_int(
            record_edge_interior_cap_bins, "record-edge interior cap"
        ),
        "qc_observed_fraction_tolerance": observed_tol,
        "qc_longest_run_fraction_tolerance": run_tol,
        "detector_exclusion_margin_seconds": exclusion,
        "length_match_rule": "exact_source_candidate_support_seconds",
        "anchor_match_rule": "exact_source_candidate_anchor_fraction",
        "record_edge_match_rule": "exact_capped_left_and_right_distance_bins",
        "censor_match_rule": "exact_record_boundary_reason_signature",
        "qc_opportunity_match_rule": (
            "exact_gap_count_and_endpoint_flags_with_declared_fraction_tolerances"
        ),
        "sampling_rule": (
            "first_accepted_sha256_counter_uniform_start_rejection_draw_"
            "without_grid_nearest_target_or_tolerance_relaxation"
        ),
        "infeasible_rule": "raise_and_fail_closed_after_declared_draw_budget",
        "eligible_prediction_outcomes": ["completed_with_candidates"],
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    result = _seal(body, id_field="policy_id", prefix="BAIEGG0AGEOMPOL")
    validate_ba_ieg_g0a_geometry_background_policy_v2(result)
    return result


def validate_ba_ieg_g0a_geometry_background_policy_v2(
    payload: object,
) -> dict[str, Any]:
    fields = {
        "schema_version", "policy_id", "seed_sha256",
        "backgrounds_per_detector_candidate", "maximum_random_draws_per_background",
        "record_edge_bin_seconds", "record_edge_interior_cap_bins",
        "qc_observed_fraction_tolerance", "qc_longest_run_fraction_tolerance",
        "detector_exclusion_margin_seconds", "length_match_rule",
        "anchor_match_rule", "record_edge_match_rule", "censor_match_rule",
        "qc_opportunity_match_rule", "sampling_rule", "infeasible_rule",
        "eligible_prediction_outcomes", "receipt_sha256",
    }
    data = _strict(payload, fields, "G0a geometry policy")
    if data["schema_version"] != BA_IEG_G0A_GEOMETRY_POLICY_SCHEMA_V2:
        raise ValueError("G0a geometry policy schema drifted")
    _sha(data["seed_sha256"], "random seed")
    _positive_int(data["backgrounds_per_detector_candidate"], "background count")
    _positive_int(data["maximum_random_draws_per_background"], "draw budget")
    if _finite(data["record_edge_bin_seconds"], "edge width") <= 0:
        raise ValueError("record-edge width must be positive")
    _positive_int(data["record_edge_interior_cap_bins"], "edge cap")
    for name in ("qc_observed_fraction_tolerance", "qc_longest_run_fraction_tolerance"):
        if not 0 <= _finite(data[name], name) <= 1:
            raise ValueError(f"{name} is invalid")
    if _finite(data["detector_exclusion_margin_seconds"], "exclusion") < 0:
        raise ValueError("detector exclusion margin is invalid")
    expected_literals = {
        "length_match_rule": "exact_source_candidate_support_seconds",
        "anchor_match_rule": "exact_source_candidate_anchor_fraction",
        "record_edge_match_rule": "exact_capped_left_and_right_distance_bins",
        "censor_match_rule": "exact_record_boundary_reason_signature",
        "qc_opportunity_match_rule": "exact_gap_count_and_endpoint_flags_with_declared_fraction_tolerances",
        "sampling_rule": "first_accepted_sha256_counter_uniform_start_rejection_draw_without_grid_nearest_target_or_tolerance_relaxation",
        "infeasible_rule": "raise_and_fail_closed_after_declared_draw_budget",
        "eligible_prediction_outcomes": ["completed_with_candidates"],
    }
    if any(data[key] != value for key, value in expected_literals.items()):
        raise ValueError("G0a geometry policy semantics drifted")
    _replay_seal(data, id_field="policy_id", prefix="BAIEGG0AGEOMPOL", context="geometry policy")
    return data


_OPPORTUNITY_FIELDS = {
    "patient_uid", "recording_id", "recording_duration_seconds",
    "source_signal_sha256", "opportunity_status", "observed_intervals_seconds",
    "quality_gap_intervals", "record_left_censor_reason_code",
    "record_right_censor_reason_code", "source_eeg_qc_receipt_sha256", "source_kind",
}


def _intervals(value: object, *, duration: float, context: str) -> list[list[float]]:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be a list")
    rows: list[list[float]] = []
    previous_stop = -math.inf
    for index, raw in enumerate(value):
        if not isinstance(raw, list) or len(raw) != 2:
            raise TypeError(f"{context} interval {index} must be [start, stop]")
        start = _finite(raw[0], f"{context} start")
        stop = _finite(raw[1], f"{context} stop")
        if start < 0 or stop <= start or stop > duration or start < previous_stop - _EPS:
            raise ValueError(f"{context} intervals are invalid/overlapping/unsorted")
        rows.append([start, stop])
        previous_stop = stop
    return rows


def _normalize_opportunity(value: object, index: int) -> dict[str, Any]:
    row = _strict(value, _OPPORTUNITY_FIELDS, f"opportunity record {index}")
    duration = _finite(row["recording_duration_seconds"], "opportunity duration")
    if duration <= 0:
        raise ValueError("opportunity duration must be positive")
    status = row["opportunity_status"]
    if status not in {"available", "unavailable"}:
        raise ValueError("opportunity status is unsupported")
    observed = _intervals(
        row["observed_intervals_seconds"], duration=duration, context="observed opportunity"
    )
    if not isinstance(row["quality_gap_intervals"], list):
        raise TypeError("quality gaps must be a list")
    gaps: list[dict[str, Any]] = []
    previous_stop = -math.inf
    for gap_index, raw in enumerate(row["quality_gap_intervals"]):
        gap = _strict(
            raw,
            {"start_offset_seconds", "stop_offset_seconds", "reason_code"},
            f"quality gap {index}:{gap_index}",
        )
        start = _finite(gap["start_offset_seconds"], "quality-gap start")
        stop = _finite(gap["stop_offset_seconds"], "quality-gap stop")
        if start < 0 or stop <= start or stop > duration or start < previous_stop - _EPS:
            raise ValueError("quality gaps are invalid/overlapping/unsorted")
        gaps.append({
            "start_offset_seconds": start,
            "stop_offset_seconds": stop,
            "reason_code": _identifier(gap["reason_code"], "quality-gap reason"),
        })
        previous_stop = stop
    if status == "unavailable":
        if observed or gaps:
            raise ValueError("unavailable opportunity cannot expose intervals")
    else:
        partition = sorted(
            [(a, b) for a, b in observed]
            + [(g["start_offset_seconds"], g["stop_offset_seconds"]) for g in gaps]
        )
        cursor = 0.0
        for start, stop in partition:
            if abs(start - cursor) > _EPS:
                raise ValueError("observed opportunities and QC gaps must partition the recording")
            cursor = stop
        if abs(cursor - duration) > _EPS:
            raise ValueError("observed opportunities and QC gaps must cover the recording")
    if row["source_kind"] != "eeg_signal_qc_only":
        raise ValueError("opportunity source must be EEG/QC only")
    return {
        "patient_uid": _identifier(row["patient_uid"], "opportunity patient UID"),
        "recording_id": _identifier(row["recording_id"], "opportunity recording ID"),
        "recording_duration_seconds": duration,
        "source_signal_sha256": _sha(row["source_signal_sha256"], "source signal"),
        "opportunity_status": status,
        "observed_intervals_seconds": observed,
        "quality_gap_intervals": gaps,
        "record_left_censor_reason_code": _identifier(
            row["record_left_censor_reason_code"], "left censor reason"
        ),
        "record_right_censor_reason_code": _identifier(
            row["record_right_censor_reason_code"], "right censor reason"
        ),
        "source_eeg_qc_receipt_sha256": _sha(
            row["source_eeg_qc_receipt_sha256"], "EEG/QC receipt"
        ),
        "source_kind": "eeg_signal_qc_only",
    }


def _overlap(start: float, stop: float, left: float, right: float) -> float:
    return max(0.0, min(stop, right) - max(start, left))


def _geometry(
    *, start: float, stop: float, anchor: float, record: Mapping[str, Any], policy: Mapping[str, Any]
) -> dict[str, Any]:
    length = stop - start
    duration = float(record["recording_duration_seconds"])
    width = float(policy["record_edge_bin_seconds"])
    cap = int(policy["record_edge_interior_cap_bins"])
    observed_runs = [
        _overlap(start, stop, left, right)
        for left, right in record["observed_intervals_seconds"]
    ]
    observed_seconds = sum(observed_runs)
    gap_count = sum(
        _overlap(start, stop, gap["start_offset_seconds"], gap["stop_offset_seconds"]) > _EPS
        for gap in record["quality_gap_intervals"]
    )
    probe = min(1e-6, length / 4.0)
    def observed_at(point: float) -> bool:
        return any(left - _EPS <= point <= right + _EPS for left, right in record["observed_intervals_seconds"])
    left_touch = start <= _EPS
    right_touch = duration - stop <= _EPS
    return {
        "support_length_seconds": length,
        "anchor_fraction": (anchor - start) / length,
        "record_edge_bins": [
            min(cap, int(math.floor(max(0.0, start) / width))),
            min(cap, int(math.floor(max(0.0, duration - stop) / width))),
        ],
        "censor_signature": [
            "record:" + record["record_left_censor_reason_code"] if left_touch else "none",
            "record:" + record["record_right_censor_reason_code"] if right_touch else "none",
        ],
        "qc_observed_fraction": observed_seconds / length,
        "qc_longest_observed_run_fraction": (max(observed_runs, default=0.0) / length),
        "qc_gap_component_count": gap_count,
        "qc_endpoint_observed_flags": [
            observed_at(min(stop, start + probe)),
            observed_at(max(start, stop - probe)),
        ],
    }


def _geometry_matches(source: Mapping[str, Any], target: Mapping[str, Any], policy: Mapping[str, Any]) -> bool:
    return (
        abs(float(source["support_length_seconds"]) - float(target["support_length_seconds"])) <= _EPS
        and abs(float(source["anchor_fraction"]) - float(target["anchor_fraction"])) <= _EPS
        and source["record_edge_bins"] == target["record_edge_bins"]
        and source["censor_signature"] == target["censor_signature"]
        and source["qc_gap_component_count"] == target["qc_gap_component_count"]
        and source["qc_endpoint_observed_flags"] == target["qc_endpoint_observed_flags"]
        and abs(float(source["qc_observed_fraction"]) - float(target["qc_observed_fraction"]))
        <= float(policy["qc_observed_fraction_tolerance"]) + _EPS
        and abs(float(source["qc_longest_observed_run_fraction"]) - float(target["qc_longest_observed_run_fraction"]))
        <= float(policy["qc_longest_run_fraction_tolerance"]) + _EPS
    )


def _uniform(seed_payload: Mapping[str, Any]) -> float:
    return int.from_bytes(hashlib.sha256(_canonical_bytes(seed_payload)).digest()[:8], "big") / float(1 << 64)


def _intersects_expanded(start: float, stop: float, candidate: Mapping[str, Any], margin: float) -> bool:
    return stop > float(candidate["start_offset_seconds"]) - margin + _EPS and start < float(candidate["stop_offset_seconds"]) + margin - _EPS


def _construct_geometry_freeze(
    source: Mapping[str, Any], opportunities: Sequence[Mapping[str, Any]], policy: Mapping[str, Any]
) -> dict[str, Any]:
    records_by_id = {row["recording_id"]: row for row in source["records"]}
    normalized = [_normalize_opportunity(row, index) for index, row in enumerate(opportunities)]
    normalized.sort(key=lambda row: (row["patient_uid"], row["recording_id"]))
    if len({row["recording_id"] for row in normalized}) != len(normalized) or set(records_by_id) != {row["recording_id"] for row in normalized}:
        raise ValueError("EEG/QC opportunity denominator must exactly equal prediction records")
    opportunity_by_id = {row["recording_id"]: row for row in normalized}
    detector = [deepcopy(row) for row in source["candidates"] if row["origin"] == "detector_proposal"]
    detector.sort(key=lambda row: (row["patient_uid"], row["recording_id"], row["start_offset_seconds"], row["candidate_id"]))
    for recording_id, prediction in records_by_id.items():
        opportunity = opportunity_by_id[recording_id]
        if (
            opportunity["patient_uid"] != prediction["patient_uid"]
            or abs(float(opportunity["recording_duration_seconds"]) - float(prediction["recording_duration_seconds"])) > _EPS
            or opportunity["source_signal_sha256"] != prediction["source_signal_sha256"]
        ):
            raise ValueError("EEG/QC opportunity crosses prediction identity")
        if prediction["outcome"] in policy["eligible_prediction_outcomes"] and opportunity["opportunity_status"] != "available":
            raise BAIEGG0AGeometrySamplingInfeasible(
                f"{recording_id}: EEG/QC opportunity unavailable for eligible candidate sampling"
            )
    detector_by_record: dict[str, list[dict[str, Any]]] = {key: [] for key in records_by_id}
    for row in detector:
        detector_by_record[row["recording_id"]].append(row)
    backgrounds: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for candidate in detector:
        record_id = candidate["recording_id"]
        prediction = records_by_id[record_id]
        if prediction["outcome"] not in policy["eligible_prediction_outcomes"]:
            continue
        opportunity = opportunity_by_id[record_id]
        length = float(candidate["stop_offset_seconds"]) - float(candidate["start_offset_seconds"])
        maximum_start = float(prediction["recording_duration_seconds"]) - length
        source_geometry = _geometry(
            start=float(candidate["start_offset_seconds"]), stop=float(candidate["stop_offset_seconds"]),
            anchor=float(candidate["anchor_offset_seconds"]), record=opportunity, policy=policy,
        )
        for ordinal in range(1, int(policy["backgrounds_per_detector_candidate"]) + 1):
            accepted: tuple[int, float, float, float, dict[str, Any]] | None = None
            for attempt in range(1, int(policy["maximum_random_draws_per_background"]) + 1):
                start = _uniform({
                    "seed_sha256": policy["seed_sha256"],
                    "source_prediction_roster_receipt_sha256": source["receipt_sha256"],
                    "source_candidate_id": candidate["candidate_id"],
                    "ordinal": ordinal,
                    "attempt": attempt,
                }) * maximum_start
                stop = start + length
                if any(
                    _intersects_expanded(start, stop, item, float(policy["detector_exclusion_margin_seconds"]))
                    for item in detector_by_record[record_id]
                ) or any(
                    item["recording_id"] == record_id and _intersects_expanded(start, stop, item, 0.0)
                    for item in backgrounds
                ):
                    continue
                anchor = start + float(source_geometry["anchor_fraction"]) * length
                target_geometry = _geometry(start=start, stop=stop, anchor=anchor, record=opportunity, policy=policy)
                if _geometry_matches(source_geometry, target_geometry, policy):
                    accepted = (attempt, start, stop, anchor, target_geometry)
                    break
            if accepted is None:
                raise BAIEGG0AGeometrySamplingInfeasible(
                    f"{record_id}::{candidate['candidate_id']}::{ordinal}: no admissible true-random geometry match in declared draw budget"
                )
            attempt, start, stop, anchor, target_geometry = accepted
            receipt_body = {
                "schema": "ba_ieg_g0a_geometry_matched_random_candidate_v2",
                "policy_receipt_sha256": policy["receipt_sha256"],
                "source_prediction_roster_receipt_sha256": source["receipt_sha256"],
                "source_candidate_receipt_sha256": candidate["source_candidate_receipt_sha256"],
                "source_eeg_qc_receipt_sha256": opportunity["source_eeg_qc_receipt_sha256"],
                "source_candidate_id": candidate["candidate_id"],
                "ordinal": ordinal,
                "accepted_random_draw_ordinal": attempt,
                "start_offset_seconds": start,
                "stop_offset_seconds": stop,
                "anchor_offset_seconds": anchor,
            }
            receipt = _digest(receipt_body)
            background = {
                "candidate_id": "G0AGEOM-" + receipt[:24],
                "patient_uid": candidate["patient_uid"],
                "recording_id": record_id,
                "origin": "candidate_geometry_matched_random_background",
                "start_offset_seconds": start,
                "stop_offset_seconds": stop,
                "anchor_offset_seconds": anchor,
                "score": None,
                "decision_available_offset_seconds": stop,
                "source_candidate_receipt_sha256": receipt,
                "matched_source_detector_candidate_id": candidate["candidate_id"],
                "random_draw_ordinal": attempt,
            }
            backgrounds.append(background)
            pairs.append({
                "source_detector_candidate_id": candidate["candidate_id"],
                "background_candidate_id": background["candidate_id"],
                "background_replication_ordinal": ordinal,
                "accepted_random_draw_ordinal": attempt,
                "source_geometry": source_geometry,
                "background_geometry": target_geometry,
                "geometry_parity_passed": True,
            })
    backgrounds.sort(key=lambda row: (row["patient_uid"], row["recording_id"], row["start_offset_seconds"], row["candidate_id"]))
    pairs.sort(key=lambda row: (row["source_detector_candidate_id"], row["background_replication_ordinal"]))
    records = [
        {
            "patient_uid": row["patient_uid"],
            "recording_id": row["recording_id"],
            "prediction_outcome": row["outcome"],
            "detector_candidate_count": len(row["detector_candidate_ids"]),
            "geometry_background_count": sum(item["recording_id"] == row["recording_id"] for item in backgrounds),
            "opportunity_status": opportunity_by_id[row["recording_id"]]["opportunity_status"],
        }
        for row in source["records"]
    ]
    records.sort(key=lambda row: (row["patient_uid"], row["recording_id"]))
    return {
        "schema_version": BA_IEG_G0A_GEOMETRY_FREEZE_SCHEMA_V2,
        "freeze_id": "BAIEG-G0A-GEOMETRY-FREEZE-PENDING",
        "source_prediction_roster": deepcopy(source),
        "geometry_background_policy": deepcopy(policy),
        "eeg_qc_opportunity_records": normalized,
        "records": records,
        "detector_candidates": detector,
        "geometry_background_candidates": backgrounds,
        "geometry_pairs": pairs,
        "counts": {
            "patients": len({row["patient_uid"] for row in records}),
            "records": len(records),
            "detector_candidates": len(detector),
            "geometry_background_candidates": len(backgrounds),
            "completed_zero_candidate_records": sum(row["prediction_outcome"] == "completed_zero_candidate" for row in records),
            "technical_failure_records": sum(row["prediction_outcome"] == "technical_failure" for row in records),
        },
        "scope_receipt": {
            "prediction_and_geometry_frozen_before_target_join": True,
            "geometry_sources_are_detector_candidate_record_identity_and_eeg_qc_only": True,
            "true_random_draw_without_nearest_or_relaxation": True,
            "infeasible_sampling_fails_closed": True,
            "complete_zero_partial_failure_denominator_retained": True,
            "public_event_intervals_opened": 0,
            "edf_annotations_opened": 0,
            "channel_or_soz_targets_opened": 0,
            "spreadsheet_doctor_clinical_or_behavior_text_opened": 0,
            "software_contract_is_not_g0a_pass": True,
            "training_authorized": False,
            "g0a_promotion_authorized": False,
        },
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }


def build_ba_ieg_g0a_geometry_background_freeze_v2(
    *, prediction_roster: Mapping[str, Any], eeg_qc_opportunity_records: Sequence[Mapping[str, Any]], geometry_background_policy: Mapping[str, Any]
) -> dict[str, Any]:
    source = validate_ba_ieg_g0_a1_prediction_roster_v1(dict(prediction_roster))
    policy = validate_ba_ieg_g0a_geometry_background_policy_v2(dict(geometry_background_policy))
    body = _construct_geometry_freeze(source, eeg_qc_opportunity_records, policy)
    result = _seal(body, id_field="freeze_id", prefix="BAIEGG0AGEOMFREEZE")
    validate_ba_ieg_g0a_geometry_background_freeze_v2(result)
    return result


def validate_ba_ieg_g0a_geometry_background_freeze_v2(payload: object) -> dict[str, Any]:
    fields = {
        "schema_version", "freeze_id", "source_prediction_roster",
        "geometry_background_policy", "eeg_qc_opportunity_records", "records",
        "detector_candidates", "geometry_background_candidates", "geometry_pairs",
        "counts", "scope_receipt", "receipt_sha256",
    }
    data = _strict(payload, fields, "G0a geometry freeze")
    if data["schema_version"] != BA_IEG_G0A_GEOMETRY_FREEZE_SCHEMA_V2:
        raise ValueError("G0a geometry freeze schema drifted")
    source = validate_ba_ieg_g0_a1_prediction_roster_v1(data["source_prediction_roster"])
    policy = validate_ba_ieg_g0a_geometry_background_policy_v2(data["geometry_background_policy"])
    expected = _construct_geometry_freeze(source, data["eeg_qc_opportunity_records"], policy)
    for field in fields - {"freeze_id", "receipt_sha256"}:
        if data[field] != expected[field]:
            raise ValueError(f"G0a geometry freeze {field} does not replay")
    _replay_seal(data, id_field="freeze_id", prefix="BAIEGG0AGEOMFREEZE", context="geometry freeze")
    return data


def build_ba_ieg_g0a_global_match_policy_v2(
    *, minimum_temporal_iou: float, maximum_anchor_to_onset_seconds: float, near_event_margin_seconds: float
) -> dict[str, Any]:
    iou = _finite(minimum_temporal_iou, "minimum temporal IoU")
    anchor = _finite(maximum_anchor_to_onset_seconds, "anchor tolerance")
    near = _finite(near_event_margin_seconds, "near-event margin")
    if not 0 < iou <= 1 or anchor < 0 or near < 0:
        raise ValueError("global matching thresholds are invalid")
    body = {
        "schema_version": BA_IEG_G0A_GLOBAL_MATCH_POLICY_SCHEMA_V2,
        "policy_id": "BAIEG-G0A-GLOBAL-MATCH-PENDING",
        "minimum_temporal_iou": iou,
        "maximum_anchor_to_onset_seconds": anchor,
        "near_event_margin_seconds": near,
        "positive_edge_rule": "temporal_iou_or_anchor_to_reference_onset",
        "primary_objective": "global_maximum_cardinality_detector_candidate_event_matching",
        "secondary_lexicographic_quality": ["temporal_iou_desc", "anchor_error_asc", "candidate_id_asc", "namespaced_event_id_asc"],
        "detector_score_used": False,
        "random_background_consumes_event_or_detector_recall": False,
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    result = _seal(body, id_field="policy_id", prefix="BAIEGG0AGLOBALMATCH")
    validate_ba_ieg_g0a_global_match_policy_v2(result)
    return result


def validate_ba_ieg_g0a_global_match_policy_v2(payload: object) -> dict[str, Any]:
    fields = {
        "schema_version", "policy_id", "minimum_temporal_iou",
        "maximum_anchor_to_onset_seconds", "near_event_margin_seconds",
        "positive_edge_rule", "primary_objective", "secondary_lexicographic_quality",
        "detector_score_used", "random_background_consumes_event_or_detector_recall",
        "receipt_sha256",
    }
    data = _strict(payload, fields, "G0a global match policy")
    if data["schema_version"] != BA_IEG_G0A_GLOBAL_MATCH_POLICY_SCHEMA_V2:
        raise ValueError("G0a global match policy schema drifted")
    if not 0 < _finite(data["minimum_temporal_iou"], "minimum IoU") <= 1 or _finite(data["maximum_anchor_to_onset_seconds"], "anchor tolerance") < 0 or _finite(data["near_event_margin_seconds"], "near margin") < 0:
        raise ValueError("G0a global match thresholds drifted")
    if (
        data["positive_edge_rule"] != "temporal_iou_or_anchor_to_reference_onset"
        or data["primary_objective"] != "global_maximum_cardinality_detector_candidate_event_matching"
        or data["secondary_lexicographic_quality"] != ["temporal_iou_desc", "anchor_error_asc", "candidate_id_asc", "namespaced_event_id_asc"]
        or data["detector_score_used"] is not False
        or data["random_background_consumes_event_or_detector_recall"] is not False
    ):
        raise ValueError("G0a global match semantics drifted")
    _replay_seal(data, id_field="policy_id", prefix="BAIEGG0AGLOBALMATCH", context="global match policy")
    return data


def _pair_metrics(candidate: Mapping[str, Any], event: Mapping[str, Any]) -> dict[str, float]:
    start, stop = float(candidate["start_offset_seconds"]), float(candidate["stop_offset_seconds"])
    onset, offset = float(event["onset_recording_seconds"]), float(event["offset_recording_seconds"])
    intersection = _overlap(start, stop, onset, offset)
    union = max(stop, offset) - min(start, onset)
    gap = onset - stop if stop < onset else start - offset if offset < start else 0.0
    return {
        "intersection_seconds": intersection,
        "temporal_iou": 0.0 if union <= 0 else intersection / union,
        "anchor_to_onset_seconds": abs(float(candidate["anchor_offset_seconds"]) - onset),
        "interval_gap_seconds": gap,
    }


def _maximum_size(edges: Sequence[dict[str, Any]], blocked_candidates: set[str] | None = None, blocked_events: set[str] | None = None) -> int:
    blocked_candidates = blocked_candidates or set()
    blocked_events = blocked_events or set()
    adjacency: dict[str, list[str]] = {}
    for edge in edges:
        if edge["candidate_key"] not in blocked_candidates and edge["event_key"] not in blocked_events:
            adjacency.setdefault(edge["candidate_key"], []).append(edge["event_key"])
    for values in adjacency.values():
        values.sort()
    event_owner: dict[str, str] = {}
    def augment(candidate_key: str, seen: set[str]) -> bool:
        for event_key in adjacency.get(candidate_key, []):
            if event_key in seen:
                continue
            seen.add(event_key)
            owner = event_owner.get(event_key)
            if owner is None or augment(owner, seen):
                event_owner[event_key] = candidate_key
                return True
        return False
    count = 0
    for candidate_key in sorted(adjacency):
        if augment(candidate_key, set()):
            count += 1
    return count


def _cardinality_first_matching(edges: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(
        edges,
        key=lambda edge: (
            -edge["metrics"]["temporal_iou"],
            edge["metrics"]["anchor_to_onset_seconds"],
            edge["candidate_key"],
            edge["event_key"],
        ),
    )
    target_size = _maximum_size(ordered)
    chosen: list[dict[str, Any]] = []
    used_candidates: set[str] = set()
    used_events: set[str] = set()
    for edge in ordered:
        if edge["candidate_key"] in used_candidates or edge["event_key"] in used_events:
            continue
        next_candidates = used_candidates | {edge["candidate_key"]}
        next_events = used_events | {edge["event_key"]}
        if len(chosen) + 1 + _maximum_size(ordered, next_candidates, next_events) == target_size:
            chosen.append(edge)
            used_candidates = next_candidates
            used_events = next_events
            if len(chosen) == target_size:
                break
    if len(chosen) != target_size:
        raise AssertionError("maximum-cardinality lexicographic matching did not close")
    return chosen


def _construct_join(freeze: Mapping[str, Any], references: Mapping[str, Any], policy: Mapping[str, Any]) -> dict[str, Any]:
    source = freeze["source_prediction_roster"]
    if references["prediction_roster_receipt_sha256"] != source["receipt_sha256"]:
        raise ValueError("reference roster is not bound to the prediction freeze")
    pred_records = {row["recording_id"]: row for row in source["records"]}
    ref_records = {row["recording_id"]: row for row in references["records"]}
    if set(pred_records) != set(ref_records):
        raise ValueError("reference and prediction record denominators differ")
    detector_by_record = {key: [] for key in pred_records}
    random_by_record = {key: [] for key in pred_records}
    for candidate in freeze["detector_candidates"]:
        detector_by_record[candidate["recording_id"]].append(candidate)
    for candidate in freeze["geometry_background_candidates"]:
        random_by_record[candidate["recording_id"]].append(candidate)
    eligible_edges: list[dict[str, Any]] = []
    event_lookup: dict[str, dict[str, Any]] = {}
    for recording_id in sorted(pred_records):
        prediction, reference = pred_records[recording_id], ref_records[recording_id]
        if prediction["patient_uid"] != reference["patient_uid"] or abs(float(prediction["recording_duration_seconds"]) - float(reference["recording_duration_seconds"])) > _EPS:
            raise ValueError("reference crosses patient/record identity")
        evaluable = prediction["outcome"] in {"completed_with_candidates", "completed_zero_candidate"} and reference["reference_coverage_status"] == "complete_recording"
        if not evaluable:
            continue
        for event in reference["seizure_intervals"]:
            event_key = recording_id + "::" + event["public_event_id"]
            if event_key in event_lookup:
                raise ValueError("namespaced public event ID repeats")
            event_lookup[event_key] = event
            for candidate in detector_by_record[recording_id]:
                metrics = _pair_metrics(candidate, event)
                if metrics["temporal_iou"] >= float(policy["minimum_temporal_iou"]) or metrics["anchor_to_onset_seconds"] <= float(policy["maximum_anchor_to_onset_seconds"]):
                    eligible_edges.append({
                        "candidate_key": candidate["candidate_id"],
                        "event_key": event_key,
                        "recording_id": recording_id,
                        "public_event_id": event["public_event_id"],
                        "metrics": metrics,
                    })
    selected = _cardinality_first_matching(eligible_edges)
    match_by_candidate = {edge["candidate_key"]: edge for edge in selected}
    matched_event_keys = {edge["event_key"] for edge in selected}
    targets: list[dict[str, Any]] = []
    denominator: list[dict[str, Any]] = []
    for recording_id in sorted(pred_records):
        prediction, reference = pred_records[recording_id], ref_records[recording_id]
        events = reference["seizure_intervals"]
        evaluable = prediction["outcome"] in {"completed_with_candidates", "completed_zero_candidate"} and reference["reference_coverage_status"] == "complete_recording"
        candidates = detector_by_record[recording_id] + random_by_record[recording_id]
        for candidate in candidates:
            base = deepcopy(candidate)
            if not evaluable:
                targets.append({**base, "target_status": "not_evaluable_prediction_or_reference_coverage", "training_class": None, "relation_role": None, "matched_or_nearest_public_event_id": None, "intersection_seconds": None, "temporal_iou": None, "anchor_to_onset_seconds": None, "interval_gap_seconds": None})
                continue
            selected_edge = match_by_candidate.get(candidate["candidate_id"])
            metrics_by_event = [(event, _pair_metrics(candidate, event)) for event in events]
            positive = [(event, metrics) for event, metrics in metrics_by_event if metrics["temporal_iou"] >= float(policy["minimum_temporal_iou"]) or metrics["anchor_to_onset_seconds"] <= float(policy["maximum_anchor_to_onset_seconds"])]
            overlap_or_positive = [(event, metrics) for event, metrics in metrics_by_event if metrics["intersection_seconds"] > 0 or (event, metrics) in positive]
            if selected_edge is not None:
                event_id = selected_edge["public_event_id"]
                metrics = selected_edge["metrics"]
                training_class = "matched_true_event"
                role = "detector_recall_match"
            elif candidate["origin"] == "candidate_geometry_matched_random_background" and positive:
                event, metrics = min(positive, key=lambda item: (-item[1]["temporal_iou"], item[1]["anchor_to_onset_seconds"], item[0]["public_event_id"]))
                event_id = event["public_event_id"]
                training_class = "matched_true_event"
                role = "background_collision_positive_control_not_detector_recall"
            elif overlap_or_positive:
                event, metrics = min(overlap_or_positive, key=lambda item: (-item[1]["temporal_iou"], item[1]["anchor_to_onset_seconds"], item[0]["public_event_id"]))
                event_id = event["public_event_id"]
                training_class = "fragmented_or_duplicate_hard_candidate"
                role = "unmatched_reference_relation"
            elif metrics_by_event:
                event, metrics = min(metrics_by_event, key=lambda item: (item[1]["interval_gap_seconds"], item[1]["anchor_to_onset_seconds"], item[0]["public_event_id"]))
                if metrics["interval_gap_seconds"] <= float(policy["near_event_margin_seconds"]):
                    event_id, training_class, role = event["public_event_id"], "near_event_hard_candidate", "nearest_reference_relation"
                else:
                    event_id, role = None, "no_reference_relation"
                    training_class = "candidate_geometry_matched_random_background" if candidate["origin"] == "candidate_geometry_matched_random_background" else "unmatched_false_candidate"
            else:
                event_id, role = None, "no_reference_relation"
                metrics = {"intersection_seconds": 0.0, "temporal_iou": 0.0, "anchor_to_onset_seconds": None, "interval_gap_seconds": None}
                training_class = "candidate_geometry_matched_random_background" if candidate["origin"] == "candidate_geometry_matched_random_background" else "unmatched_false_candidate"
            targets.append({**base, "target_status": "evaluable_complete_reference", "training_class": training_class, "relation_role": role, "matched_or_nearest_public_event_id": event_id, **metrics})
        matched_count = sum((recording_id + "::" + event["public_event_id"]) in matched_event_keys for event in events) if evaluable else 0
        denominator.append({
            "patient_uid": prediction["patient_uid"], "recording_id": recording_id,
            "prediction_outcome": prediction["outcome"],
            "reference_coverage_status": reference["reference_coverage_status"],
            "detector_candidate_count": len(detector_by_record[recording_id]),
            "geometry_background_count": len(random_by_record[recording_id]),
            "public_event_count": len(events), "matched_public_event_count": matched_count,
            "missed_public_event_count": len(events) - matched_count if evaluable else None,
            "candidate_target_evaluable": evaluable,
        })
    targets.sort(key=lambda row: (row["patient_uid"], row["recording_id"], row["start_offset_seconds"], row["origin"], row["candidate_id"]))
    denominator.sort(key=lambda row: (row["patient_uid"], row["recording_id"]))
    evaluable_events = sum(row["public_event_count"] for row in denominator if row["candidate_target_evaluable"])
    matched_events = sum(row["matched_public_event_count"] for row in denominator if row["candidate_target_evaluable"])
    return {
        "schema_version": BA_IEG_G0A_GLOBAL_JOIN_SCHEMA_V2,
        "join_id": "BAIEG-G0A-GLOBAL-JOIN-PENDING",
        "geometry_freeze": deepcopy(freeze),
        "reference_roster": deepcopy(references),
        "global_match_policy": deepcopy(policy),
        "selected_detector_event_matches": sorted(selected, key=lambda edge: (edge["recording_id"], edge["candidate_key"], edge["event_key"])),
        "record_denominator": denominator,
        "candidate_targets": targets,
        "counts": {
            "patients": len({row["patient_uid"] for row in denominator}), "records": len(denominator),
            "candidates": len(targets),
            "candidate_training_classes": {name: sum(row["training_class"] == name for row in targets) for name in BA_IEG_G0A_TRAINING_CLASSES_V2},
            "not_evaluable_candidates": sum(row["training_class"] is None for row in targets),
            "all_public_events": sum(row["public_event_count"] for row in denominator),
            "evaluable_public_events": evaluable_events,
            "public_events_on_non_evaluable_records": sum(row["public_event_count"] for row in denominator if not row["candidate_target_evaluable"]),
            "matched_public_events": matched_events, "missed_public_events": evaluable_events - matched_events,
            "random_positive_collisions_excluded_from_detector_recall": sum(row["relation_role"] == "background_collision_positive_control_not_detector_recall" for row in targets),
            "zero_detector_candidate_records": sum(row["prediction_outcome"] == "completed_zero_candidate" for row in denominator),
            "technical_failure_records": sum(row["prediction_outcome"] == "technical_failure" for row in denominator),
        },
        "scope_receipt": {
            "prediction_and_geometry_frozen_before_reference_join": True,
            "maximum_cardinality_precedes_secondary_edge_quality": True,
            "secondary_quality_is_iou_anchor_error_and_stable_ids_only": True,
            "random_collision_cannot_improve_detector_recall": True,
            "zero_candidate_and_failure_event_denominators_retained": True,
            "global_public_event_intervals_only": True,
            "edf_annotations_opened": 0, "channel_or_soz_targets_opened": 0,
            "spreadsheet_doctor_clinical_or_behavior_text_opened": 0,
            "software_contract_is_not_g0a_pass": True,
            "training_authorized": False, "g0a_promotion_authorized": False,
        },
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }


def build_ba_ieg_g0a_global_postfreeze_target_join_v2(
    *, geometry_freeze: Mapping[str, Any], reference_roster: Mapping[str, Any], global_match_policy: Mapping[str, Any]
) -> dict[str, Any]:
    freeze = validate_ba_ieg_g0a_geometry_background_freeze_v2(dict(geometry_freeze))
    references = validate_ba_ieg_g0_a1_reference_roster_v1(dict(reference_roster))
    policy = validate_ba_ieg_g0a_global_match_policy_v2(dict(global_match_policy))
    body = _construct_join(freeze, references, policy)
    result = _seal(body, id_field="join_id", prefix="BAIEGG0AGLOBALJOIN")
    validate_ba_ieg_g0a_global_postfreeze_target_join_v2(result)
    return result


def validate_ba_ieg_g0a_global_postfreeze_target_join_v2(payload: object) -> dict[str, Any]:
    fields = {
        "schema_version", "join_id", "geometry_freeze", "reference_roster",
        "global_match_policy", "selected_detector_event_matches", "record_denominator",
        "candidate_targets", "counts", "scope_receipt", "receipt_sha256",
    }
    data = _strict(payload, fields, "G0a global target join")
    if data["schema_version"] != BA_IEG_G0A_GLOBAL_JOIN_SCHEMA_V2:
        raise ValueError("G0a global target join schema drifted")
    freeze = validate_ba_ieg_g0a_geometry_background_freeze_v2(data["geometry_freeze"])
    references = validate_ba_ieg_g0_a1_reference_roster_v1(data["reference_roster"])
    policy = validate_ba_ieg_g0a_global_match_policy_v2(data["global_match_policy"])
    expected = _construct_join(freeze, references, policy)
    for field in fields - {"join_id", "receipt_sha256"}:
        if data[field] != expected[field]:
            raise ValueError(f"G0a global target join {field} does not replay")
    _replay_seal(data, id_field="join_id", prefix="BAIEGG0AGLOBALJOIN", context="global target join")
    return data


__all__ = [
    "BAIEGG0AGeometrySamplingInfeasible",
    "BA_IEG_G0A_GEOMETRY_POLICY_SCHEMA_V2",
    "BA_IEG_G0A_GEOMETRY_FREEZE_SCHEMA_V2",
    "BA_IEG_G0A_GLOBAL_MATCH_POLICY_SCHEMA_V2",
    "BA_IEG_G0A_GLOBAL_JOIN_SCHEMA_V2",
    "BA_IEG_G0A_ORIGINS_V2",
    "BA_IEG_G0A_TRAINING_CLASSES_V2",
    "build_ba_ieg_g0a_geometry_background_policy_v2",
    "validate_ba_ieg_g0a_geometry_background_policy_v2",
    "build_ba_ieg_g0a_geometry_background_freeze_v2",
    "validate_ba_ieg_g0a_geometry_background_freeze_v2",
    "build_ba_ieg_g0a_global_match_policy_v2",
    "validate_ba_ieg_g0a_global_match_policy_v2",
    "build_ba_ieg_g0a_global_postfreeze_target_join_v2",
    "validate_ba_ieg_g0a_global_postfreeze_target_join_v2",
]
