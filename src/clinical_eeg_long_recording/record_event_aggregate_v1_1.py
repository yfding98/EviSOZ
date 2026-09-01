"""EEG-only recording aggregate over validated event Findings v3 graphs.

The repeatable event card is intentionally not the owner of recording-level
event count, ictal-pattern burden, or inter-event interval facts.  This module
materializes those three quantities exactly once per physical recording while
preserving event deduplication uncertainty and onset/offset censoring.

The aggregate is a public/synthetic shadow contract.  It cannot turn detector
alarms into seizures, create onset/SOZ evidence, read annotations or clinical
labels, invoke Qwen, or promote a report claim.  Every non-empty aggregate is
content-bound to the complete validated ``event_eeg_findings_v3`` payloads.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator

from .event_findings_v3_validation import (
    validate_event_eeg_findings_v3_payload,
)
from .minimum_event_evidence_card_extension_v1_1 import (
    load_minimum_event_evidence_card_extension_registry_v1_1,
)
from .minimum_event_evidence_card_registry_v1 import (
    load_minimum_event_evidence_card_registry_v1,
)


RECORD_EVENT_AGGREGATE_SCHEMA_VERSION_V1_1 = "clinical_eeg_record_event_aggregate_v1_1"
RECORD_EVENT_AGGREGATE_DEDUPLICATION_POLICY_ID_V1_1 = (
    "QUALIFIED-EVENT-FINAL-SUPPORT-COMPONENT-BOUNDS-V1.1"
)
RECORD_EVENT_AGGREGATE_QUERY_IDS_V1_1 = (
    "MEC11-RECORD-DETECTED-QUALIFIED-EVENT-COUNT",
    "MEC11-RECORD-INTER-EVENT-INTERVAL-DISTRIBUTION",
    "MEC11-RECORD-QUALIFIED-ICTAL-PATTERN-BURDEN",
)

_ROOT = Path(__file__).resolve().parents[2]
RECORD_EVENT_AGGREGATE_SCHEMA_PATH_V1_1 = (
    _ROOT / "schemas" / "clinical_eeg_record_event_aggregate_v1_1.schema.json"
)
_INFERENCE_EXCLUSIONS = (
    "edf_annotations_used",
    "excel_used",
    "doctor_labels_used",
    "clinical_text_used",
    "patient_metadata_used",
    "video_used",
    "ecg_emg_eog_used",
    "sleep_staging_used",
    "provocation_used",
)
_SOURCE_FIREWALL = {
    "edf_annotations_used": False,
    "spreadsheet_used": False,
    "doctor_labels_used": False,
    "clinical_reports_used": False,
    "patient_metadata_used": False,
    "video_or_behavior_used": False,
    "sleep_staging_used": False,
    "provocation_used": False,
    "ecg_emg_eog_used": False,
    "qwen_used": False,
}
_AUTHORIZATION = {
    "clinical_correctness_claimed": False,
    "cortical_soz_or_ez_claim_authorized": False,
    "detector_alarm_count_may_masquerade_as_event_count": False,
    "record_aggregate_may_create_onset_support": False,
    "late_spread_may_create_onset_support": False,
    "qwen_authorized": False,
    "report_promotion_authorized": False,
}
_QUALIFIED_OUTCOMES = {
    "qualified_electrographic_event",
    "qualified_electrographic_seizure",
}
_RESOLVED_MERGE_SPLIT_STATES = {"single_event", "split"}
_TOL = 1e-6


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _self_hash(value: Mapping[str, object], field: str) -> str:
    body = deepcopy(dict(value))
    body.pop(field, None)
    return _sha256(body)


def _schema_validator() -> Draft202012Validator:
    schema = json.loads(
        RECORD_EVENT_AGGREGATE_SCHEMA_PATH_V1_1.read_text(encoding="utf-8")
    )
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _schema_errors(value: object) -> list[str]:
    errors = sorted(
        _schema_validator().iter_errors(value), key=lambda item: list(item.path)
    )
    rendered: list[str] = []
    for error in errors[:16]:
        pointer = "/" + "/".join(str(part) for part in error.path)
        rendered.append(f"{pointer}: {error.message}")
    if len(errors) > 16:
        rendered.append(f"... {len(errors) - 16} more error(s)")
    return rendered


def _require_finite_number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _require_explicit_eeg_only_exclusions(
    value: object, *, context: str
) -> dict[str, bool]:
    if type(value) is not dict:
        raise TypeError(f"{context} must be an object")
    if set(value) != set(_INFERENCE_EXCLUSIONS):
        raise ValueError(f"{context} keys are not closed")
    unresolved = sorted(key for key, item in value.items() if item is not False)
    if unresolved:
        raise ValueError(
            "record event aggregate requires every inference exclusion to be "
            f"explicitly false; unresolved fields at {context}: {unresolved}"
        )
    return {key: False for key in _INFERENCE_EXCLUSIONS}


def _validate_record_context(value: object) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError("record_context must be an object")
    expected = {
        "record_id",
        "canonical_signal_sha256",
        "recording_duration_seconds",
        "source_inference_exclusions",
    }
    if set(value) != expected:
        raise ValueError("record_context keys are not closed")
    record_id = value["record_id"]
    signal_hash = value["canonical_signal_sha256"]
    if not isinstance(record_id, str) or not record_id:
        raise TypeError("record_context.record_id must be non-empty")
    if (
        not isinstance(signal_hash, str)
        or len(signal_hash) != 64
        or any(char not in "0123456789abcdef" for char in signal_hash)
    ):
        raise ValueError("record_context canonical signal hash is invalid")
    duration = _require_finite_number(
        value["recording_duration_seconds"],
        "record_context.recording_duration_seconds",
    )
    if duration <= 0.0:
        raise ValueError("recording duration must be positive")
    exclusions = _require_explicit_eeg_only_exclusions(
        value["source_inference_exclusions"],
        context="record_context.source_inference_exclusions",
    )
    return {
        "record_id": record_id,
        "canonical_signal_sha256": signal_hash,
        "recording_duration_seconds": duration,
        "source_inference_exclusions": exclusions,
    }


def _validation_kwargs(
    *,
    trusted_producer_receipts: Mapping[str, Mapping[str, object]] | None,
    trusted_calibration_receipts: Mapping[str, Mapping[str, object]] | None,
    trusted_capability_qualification_receipts: (
        Mapping[str, Mapping[str, object]] | None
    ),
    trusted_sensitivity_receipts: Mapping[str, Mapping[str, object]] | None,
    trusted_term_decision_receipts: Mapping[str, Mapping[str, object]] | None,
    trusted_registry_bindings: Mapping[str, Mapping[str, object]] | None,
) -> dict[str, object]:
    return {
        "trusted_producer_receipts": trusted_producer_receipts,
        "trusted_calibration_receipts": trusted_calibration_receipts,
        "trusted_capability_qualification_receipts": (
            trusted_capability_qualification_receipts
        ),
        "trusted_sensitivity_receipts": trusted_sensitivity_receipts,
        "trusted_term_decision_receipts": trusted_term_decision_receipts,
        "trusted_registry_bindings": trusted_registry_bindings,
    }


def _closed_interval(
    value: object, *, context: str, duration: float
) -> tuple[float, float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
    ):
        raise TypeError(f"{context} must be a two-value interval")
    start = _require_finite_number(value[0], f"{context}[0]")
    stop = _require_finite_number(value[1], f"{context}[1]")
    if start < -_TOL or stop > duration + _TOL or stop < start - _TOL:
        raise ValueError(f"{context} lies outside the recording or is reversed")
    return max(0.0, start), min(duration, stop)


def _interval_union(
    intervals: Sequence[tuple[float, float]],
) -> list[list[float]]:
    ordered = sorted((start, stop) for start, stop in intervals if stop > start + _TOL)
    result: list[list[float]] = []
    for start, stop in ordered:
        if not result or start > result[-1][1] + _TOL:
            result.append([start, stop])
        else:
            result[-1][1] = max(result[-1][1], stop)
    return result


def _union_seconds(intervals: Sequence[Sequence[float]]) -> float:
    return float(sum(float(stop) - float(start) for start, stop in intervals))


def _normalized_event_outcome(source: Mapping[str, Any]) -> str:
    outcome = str(source["event_outcome"]["outcome"])
    if outcome == "no_demonstrable_scalp_ictal_change":
        return "candidate_only"
    return outcome


def _boundary_interval(
    source: Mapping[str, Any], boundary: str
) -> tuple[float, float] | None:
    row = source["window"][boundary]
    if row["status"] != "observed" or row["interval"] is None:
        return None
    interval = row["interval"]
    return float(interval["lower"]), float(interval["upper"])


def _complete_uncensored_boundaries(source: Mapping[str, Any]) -> bool:
    window = source["window"]
    onset = _boundary_interval(source, "onset_boundary")
    offset = _boundary_interval(source, "offset_boundary")
    if (
        onset is None
        or offset is None
        or window["left_censored"]
        or window["right_censored"]
        or window["search_cap_censored"]
    ):
        return False
    return onset[0] <= onset[1] and offset[0] <= offset[1] and onset[0] < offset[1]


def _source_binding(source: Mapping[str, Any]) -> dict[str, Any]:
    outcome = _normalized_event_outcome(source)
    qualification = str(source["event_qualification"]["status"])
    qualified = outcome in _QUALIFIED_OUTCOMES and qualification == outcome
    return {
        "event_id": str(source["event_id"]),
        "source_event_findings_v3_sha256": _sha256(source),
        "normalized_event_outcome": outcome,
        "event_qualification_status": qualification,
        "final_interval": [float(item) for item in source["window"]["final_interval"]],
        "merge_split_status": str(source["window"]["merge_split_status"]),
        "qualified_for_record_aggregate": qualified,
        "complete_uncensored_boundary_support": bool(
            qualified and _complete_uncensored_boundaries(source)
        ),
    }


def _deduplication_bounds(
    qualified: Sequence[Mapping[str, Any]],
) -> tuple[str, int, int]:
    if not qualified:
        return "not_applicable", 0, 0
    intervals = sorted(
        (
            float(row["final_interval"][0]),
            float(row["final_interval"][1]),
        )
        for row in qualified
    )
    nonoverlapping = all(
        intervals[index][0] >= intervals[index - 1][1] - _TOL
        for index in range(1, len(intervals))
    )
    states_resolved = all(
        row["merge_split_status"] in _RESOLVED_MERGE_SPLIT_STATES for row in qualified
    )
    if nonoverlapping and states_resolved:
        return "resolved", len(qualified), len(qualified)

    # Each connected final-support component is certainly separated from the
    # next component, while overlapping candidates within one component may be
    # duplicates or distinct events.  This gives a conservative count bound.
    component_count = 0
    component_stop: float | None = None
    for start, stop in intervals:
        if component_stop is None or start >= component_stop - _TOL:
            component_count += 1
            component_stop = stop
        else:
            component_stop = max(component_stop, stop)
    return "unresolved", component_count, len(qualified)


def _event_roster(bindings: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    qualified = [row for row in bindings if row["qualified_for_record_aggregate"]]
    complete_ids = sorted(
        str(row["event_id"])
        for row in qualified
        if row["complete_uncensored_boundary_support"]
    )
    incomplete_ids = sorted(
        str(row["event_id"])
        for row in qualified
        if not row["complete_uncensored_boundary_support"]
    )
    status, lower, upper = _deduplication_bounds(qualified)
    dedup_body = {
        "binding_domain": "clinical-eeg-record-event-deduplication-v1.1",
        "policy_id": RECORD_EVENT_AGGREGATE_DEDUPLICATION_POLICY_ID_V1_1,
        "qualified_events": [
            {
                "event_id": row["event_id"],
                "final_interval": row["final_interval"],
                "merge_split_status": row["merge_split_status"],
            }
            for row in qualified
        ],
        "status": status,
        "lower": lower,
        "upper": upper,
    }
    outcomes = [str(row["normalized_event_outcome"]) for row in bindings]
    return {
        "detector_candidate_count": len(bindings),
        "raw_qualified_event_count": len(qualified),
        "candidate_only_count": outcomes.count("candidate_only"),
        "uncertain_event_count": outcomes.count("not_possible_to_determine"),
        "not_evaluable_event_count": outcomes.count("obscured_by_artifact"),
        "deduplicated_qualified_count_lower": lower,
        "deduplicated_qualified_count_upper": upper,
        "deduplication_status": status,
        "deduplication_policy_id": (
            RECORD_EVENT_AGGREGATE_DEDUPLICATION_POLICY_ID_V1_1
        ),
        "deduplication_sha256": _sha256(dedup_body),
        "qualified_event_ids": sorted(str(row["event_id"]) for row in qualified),
        "boundary_complete_qualified_event_ids": complete_ids,
        "boundary_incomplete_qualified_event_ids": incomplete_ids,
    }


def _modeled_opportunity(
    sources: Sequence[Mapping[str, Any]], duration: float
) -> dict[str, Any]:
    intervals: list[tuple[float, float]] = []
    for source_index, source in enumerate(sources):
        for interval_index, interval in enumerate(
            source["context"]["queried_intervals"]
        ):
            intervals.append(
                _closed_interval(
                    interval,
                    context=(
                        f"event_findings_v3[{source_index}].context."
                        f"queried_intervals[{interval_index}]"
                    ),
                    duration=duration,
                )
            )
    union = _interval_union(intervals)
    seconds = _union_seconds(union)
    return {
        "queried_interval_union": union,
        "queried_seconds": seconds,
        "recording_fraction": seconds / duration,
    }


def _burden(
    sources_by_id: Mapping[str, Mapping[str, Any]],
    qualified_bindings: Sequence[Mapping[str, Any]],
    duration: float,
) -> dict[str, Any]:
    event_ids = sorted(str(row["event_id"]) for row in qualified_bindings)
    incomplete_ids = sorted(
        str(row["event_id"])
        for row in qualified_bindings
        if not row["complete_uncensored_boundary_support"]
    )
    if not event_ids:
        return {
            "status": "not_evaluable",
            "lower_interval_union": [],
            "upper_interval_union": [],
            "lower_seconds": None,
            "upper_seconds": None,
            "lower_recording_fraction": None,
            "upper_recording_fraction": None,
            "source_event_ids": [],
            "incomplete_event_ids": [],
            "reason_codes": ["no_qualified_event_for_burden"],
        }

    lower_intervals: list[tuple[float, float]] = []
    upper_intervals: list[tuple[float, float]] = []
    for event_id in event_ids:
        source = sources_by_id[event_id]
        if not _complete_uncensored_boundaries(source):
            continue
        onset = _boundary_interval(source, "onset_boundary")
        offset = _boundary_interval(source, "offset_boundary")
        assert onset is not None and offset is not None
        lower_start = max(0.0, onset[1])
        lower_stop = min(duration, offset[0])
        upper_start = max(0.0, onset[0])
        upper_stop = min(duration, offset[1])
        if lower_stop > lower_start + _TOL:
            lower_intervals.append((lower_start, lower_stop))
        if upper_stop > upper_start + _TOL:
            upper_intervals.append((upper_start, upper_stop))

    lower_union = _interval_union(lower_intervals)
    if incomplete_ids:
        # An unobserved/censored qualified boundary supplies no safe finite
        # upper duration.  The recording interval is the conservative ceiling.
        upper_union = [[0.0, duration]]
        status = "limited"
        reasons = ["qualified_event_boundary_incomplete_or_censored"]
    else:
        upper_union = _interval_union(upper_intervals)
        status = "measured"
        reasons = []
    lower_seconds = _union_seconds(lower_union)
    upper_seconds = _union_seconds(upper_union)
    return {
        "status": status,
        "lower_interval_union": lower_union,
        "upper_interval_union": upper_union,
        "lower_seconds": lower_seconds,
        "upper_seconds": upper_seconds,
        "lower_recording_fraction": lower_seconds / duration,
        "upper_recording_fraction": upper_seconds / duration,
        "source_event_ids": event_ids,
        "incomplete_event_ids": incomplete_ids,
        "reason_codes": reasons,
    }


def _empty_interval_summary() -> dict[str, Any]:
    return {
        "count": 0,
        "minimum_lower_seconds": None,
        "median_lower_seconds": None,
        "median_upper_seconds": None,
        "maximum_upper_seconds": None,
    }


def _inter_event_intervals(
    sources_by_id: Mapping[str, Mapping[str, Any]],
    qualified_bindings: Sequence[Mapping[str, Any]],
    *,
    deduplication_status: str,
) -> dict[str, Any]:
    event_ids = [str(row["event_id"]) for row in qualified_bindings]
    if len(event_ids) < 2:
        return {
            "status": "not_evaluable",
            "intervals": [],
            "summary": _empty_interval_summary(),
            "source_event_ids": sorted(event_ids),
            "reason_codes": ["fewer_than_two_qualified_events"],
        }
    if deduplication_status != "resolved":
        return {
            "status": "limited",
            "intervals": [],
            "summary": _empty_interval_summary(),
            "source_event_ids": sorted(event_ids),
            "reason_codes": ["event_deduplication_unresolved"],
        }

    ordered = sorted(
        event_ids,
        key=lambda event_id: (
            (
                _boundary_interval(sources_by_id[event_id], "onset_boundary")
                or (math.inf, math.inf)
            )[0],
            (
                _boundary_interval(sources_by_id[event_id], "onset_boundary")
                or (math.inf, math.inf)
            )[1],
            event_id,
        ),
    )
    rows: list[dict[str, Any]] = []
    missing_pair = False
    for previous_id, next_id in zip(ordered, ordered[1:]):
        previous = sources_by_id[previous_id]
        following = sources_by_id[next_id]
        if not (
            _complete_uncensored_boundaries(previous)
            and _complete_uncensored_boundaries(following)
        ):
            missing_pair = True
            continue
        previous_offset = _boundary_interval(previous, "offset_boundary")
        next_onset = _boundary_interval(following, "onset_boundary")
        assert previous_offset is not None and next_onset is not None
        raw_lower = next_onset[0] - previous_offset[1]
        raw_upper = next_onset[1] - previous_offset[0]
        if raw_lower >= -_TOL:
            relation = "separated"
        elif raw_upper <= _TOL:
            relation = "overlap_certain"
        else:
            relation = "overlap_possible"
        lower = max(0.0, raw_lower)
        upper = max(lower, raw_upper, 0.0)
        rows.append(
            {
                "from_event_id": previous_id,
                "to_event_id": next_id,
                "lower_seconds": lower,
                "upper_seconds": upper,
                "relation_status": relation,
            }
        )

    lower_values = [float(row["lower_seconds"]) for row in rows]
    upper_values = [float(row["upper_seconds"]) for row in rows]
    summary = (
        {
            "count": len(rows),
            "minimum_lower_seconds": min(lower_values),
            "median_lower_seconds": float(median(lower_values)),
            "median_upper_seconds": float(median(upper_values)),
            "maximum_upper_seconds": max(upper_values),
        }
        if rows
        else _empty_interval_summary()
    )
    if missing_pair:
        status = "limited"
        reasons = ["one_or_more_adjacent_event_boundaries_incomplete"]
    else:
        status = "measured"
        reasons = []
    return {
        "status": status,
        "intervals": rows,
        "summary": summary,
        "source_event_ids": sorted(event_ids),
        "reason_codes": reasons,
    }


def _query_result(
    *,
    query_id: str,
    status: str,
    value_pointer: str,
    bindings: Sequence[Mapping[str, Any]],
    reason_codes: Sequence[str],
) -> dict[str, Any]:
    return {
        "query_id": query_id,
        "analysis_unit": "recording",
        "status": status,
        "assertion_level": "measured",
        "value_pointer": value_pointer,
        "source_event_ids": sorted(str(row["event_id"]) for row in bindings),
        "source_event_findings_v3_sha256s": sorted(
            str(row["source_event_findings_v3_sha256"]) for row in bindings
        ),
        "evidence_role_ceiling": "record_aggregation_only",
        "reason_codes": sorted(set(reason_codes)),
        "report_promotion_authorized": False,
    }


def materialize_record_event_aggregate_v1_1(
    event_findings_v3: Sequence[object],
    *,
    record_context: Mapping[str, object] | None = None,
    trusted_producer_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_calibration_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_capability_qualification_receipts: (
        Mapping[str, Mapping[str, object]] | None
    ) = None,
    trusted_sensitivity_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_term_decision_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_registry_bindings: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, Any]:
    """Materialize one deterministic record-level sibling aggregate."""

    if not isinstance(event_findings_v3, Sequence) or isinstance(
        event_findings_v3, (str, bytes)
    ):
        raise TypeError("event_findings_v3 must be an ordered sequence")
    kwargs = _validation_kwargs(
        trusted_producer_receipts=trusted_producer_receipts,
        trusted_calibration_receipts=trusted_calibration_receipts,
        trusted_capability_qualification_receipts=(
            trusted_capability_qualification_receipts
        ),
        trusted_sensitivity_receipts=trusted_sensitivity_receipts,
        trusted_term_decision_receipts=trusted_term_decision_receipts,
        trusted_registry_bindings=trusted_registry_bindings,
    )
    sources = [
        validate_event_eeg_findings_v3_payload(item, **kwargs)
        for item in event_findings_v3
    ]
    event_ids = [str(source["event_id"]) for source in sources]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("event_findings_v3 contains duplicate event IDs")

    if sources:
        first = sources[0]
        record = {
            "record_id": str(first["provenance"]["record_id"]),
            "canonical_signal_sha256": str(
                first["provenance"]["canonical_signal_sha256"]
            ),
            "recording_duration_seconds": float(
                first["coordinates"]["recording_duration_seconds"]
            ),
            "source_inference_exclusions": _require_explicit_eeg_only_exclusions(
                first["provenance"]["inference_exclusions"],
                context=("event_findings_v3[0].provenance.inference_exclusions"),
            ),
        }
        for source_index, source in enumerate(sources[1:], start=1):
            candidate = {
                "record_id": str(source["provenance"]["record_id"]),
                "canonical_signal_sha256": str(
                    source["provenance"]["canonical_signal_sha256"]
                ),
                "recording_duration_seconds": float(
                    source["coordinates"]["recording_duration_seconds"]
                ),
                "source_inference_exclusions": (
                    _require_explicit_eeg_only_exclusions(
                        source["provenance"]["inference_exclusions"],
                        context=(
                            f"event_findings_v3[{source_index}].provenance."
                            "inference_exclusions"
                        ),
                    )
                ),
            }
            if candidate != record:
                raise ValueError("all event graphs must bind the same EEG record")
        if (
            record_context is not None
            and _validate_record_context(record_context) != record
        ):
            raise ValueError("record_context conflicts with embedded event graphs")
    else:
        if record_context is None:
            raise ValueError(
                "zero-event aggregate materialization requires record_context"
            )
        record = _validate_record_context(record_context)

    duration = float(record["recording_duration_seconds"])
    sources = sorted(sources, key=lambda item: str(item["event_id"]))
    sources_by_id = {str(source["event_id"]): source for source in sources}
    bindings = [_source_binding(source) for source in sources]
    roster = _event_roster(bindings)
    qualified = [row for row in bindings if row["qualified_for_record_aggregate"]]
    opportunity = _modeled_opportunity(sources, duration)
    burden = _burden(sources_by_id, qualified, duration)
    intervals = _inter_event_intervals(
        sources_by_id,
        qualified,
        deduplication_status=str(roster["deduplication_status"]),
    )

    base_registry = load_minimum_event_evidence_card_registry_v1()
    extension_registry = load_minimum_event_evidence_card_extension_registry_v1_1()
    registry_binding = {
        "base_registry_id": str(base_registry["registry_id"]),
        "base_registry_sha256": str(base_registry["registry_sha256"]),
        "extension_registry_id": str(extension_registry["registry_id"]),
        "extension_registry_sha256": str(extension_registry["registry_sha256"]),
        "record_destination_id": "R01_RECORD_EVENT_ROSTER_AND_BURDEN",
        "query_ids": list(RECORD_EVENT_AGGREGATE_QUERY_IDS_V1_1),
    }
    owner = {
        "owner_kind": "recording",
        "record_id": str(record["record_id"]),
        "canonical_signal_sha256": str(record["canonical_signal_sha256"]),
        "recording_duration_seconds": duration,
        "event_scoped": False,
        "copied_into_event_cards": False,
    }
    source_roster_sha256 = _sha256(
        {
            "binding_domain": "clinical-eeg-record-event-aggregate-source-roster-v1.1",
            "owner": owner,
            "source_event_bindings": bindings,
        }
    )

    if not qualified:
        count_status = "not_evaluable"
        count_reasons = ["no_qualified_event_without_complete_negative_opportunity"]
    elif roster["deduplication_status"] == "resolved":
        count_status = "present"
        count_reasons = []
    else:
        count_status = "uncertain"
        count_reasons = ["event_deduplication_unresolved"]
    burden_status = {
        "measured": "present",
        "limited": "uncertain",
        "not_evaluable": "not_evaluable",
    }[burden["status"]]
    interval_status = {
        "measured": "present",
        "limited": "uncertain",
        "not_evaluable": "not_evaluable",
    }[intervals["status"]]
    queries = [
        _query_result(
            query_id=RECORD_EVENT_AGGREGATE_QUERY_IDS_V1_1[0],
            status=count_status,
            value_pointer="/event_roster",
            bindings=bindings,
            reason_codes=count_reasons,
        ),
        _query_result(
            query_id=RECORD_EVENT_AGGREGATE_QUERY_IDS_V1_1[1],
            status=interval_status,
            value_pointer="/inter_event_intervals",
            bindings=qualified,
            reason_codes=intervals["reason_codes"],
        ),
        _query_result(
            query_id=RECORD_EVENT_AGGREGATE_QUERY_IDS_V1_1[2],
            status=burden_status,
            value_pointer="/qualified_ictal_pattern_burden",
            bindings=qualified,
            reason_codes=burden["reason_codes"],
        ),
    ]

    if not sources:
        aggregate_status = "not_evaluable"
    elif (
        qualified
        and roster["deduplication_status"] == "resolved"
        and burden["status"] == "measured"
        and intervals["status"] != "limited"
    ):
        aggregate_status = "available"
    else:
        aggregate_status = "limited"
    aggregate: dict[str, Any] = {
        "schema_version": RECORD_EVENT_AGGREGATE_SCHEMA_VERSION_V1_1,
        "aggregate_id": "RECEVTAGG-"
        + _sha256(
            {
                "owner": owner,
                "source_event_roster_sha256": source_roster_sha256,
                "registry_binding": registry_binding,
            }
        )[:24],
        "aggregate_status": aggregate_status,
        "owner": owner,
        "registry_binding": registry_binding,
        "source_event_roster_sha256": source_roster_sha256,
        "source_event_bindings": bindings,
        "event_roster": roster,
        "modeled_opportunity": opportunity,
        "qualified_ictal_pattern_burden": burden,
        "inter_event_intervals": intervals,
        "query_results": queries,
        "source_firewall": deepcopy(_SOURCE_FIREWALL),
        "authorization": deepcopy(_AUTHORIZATION),
        "aggregate_sha256": "0" * 64,
    }
    aggregate["aggregate_sha256"] = _self_hash(aggregate, "aggregate_sha256")
    errors = _schema_errors(aggregate)
    if errors:
        raise ValueError(
            "record event aggregate schema validation failed: " + "; ".join(errors)
        )
    return aggregate


def validate_record_event_aggregate_v1_1(
    value: object,
    source_event_findings_v3: Sequence[object],
    *,
    record_context: Mapping[str, object] | None = None,
    trusted_producer_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_calibration_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_capability_qualification_receipts: (
        Mapping[str, Mapping[str, object]] | None
    ) = None,
    trusted_sensitivity_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_term_decision_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_registry_bindings: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, Any]:
    """Replay sources and reject even a consistently rehashed aggregate drift."""

    if type(value) is not dict:
        raise TypeError("record event aggregate must be an object")
    candidate = deepcopy(value)
    errors = _schema_errors(candidate)
    if errors:
        raise ValueError(
            "record event aggregate schema validation failed: " + "; ".join(errors)
        )
    if candidate["aggregate_sha256"] != _self_hash(candidate, "aggregate_sha256"):
        raise ValueError("record event aggregate self hash drifted")
    expected = materialize_record_event_aggregate_v1_1(
        source_event_findings_v3,
        record_context=record_context,
        trusted_producer_receipts=trusted_producer_receipts,
        trusted_calibration_receipts=trusted_calibration_receipts,
        trusted_capability_qualification_receipts=(
            trusted_capability_qualification_receipts
        ),
        trusted_sensitivity_receipts=trusted_sensitivity_receipts,
        trusted_term_decision_receipts=trusted_term_decision_receipts,
        trusted_registry_bindings=trusted_registry_bindings,
    )
    if candidate != expected:
        raise ValueError(
            "record event aggregate does not replay from source event Findings v3"
        )
    return candidate


__all__ = [
    "RECORD_EVENT_AGGREGATE_DEDUPLICATION_POLICY_ID_V1_1",
    "RECORD_EVENT_AGGREGATE_QUERY_IDS_V1_1",
    "RECORD_EVENT_AGGREGATE_SCHEMA_PATH_V1_1",
    "RECORD_EVENT_AGGREGATE_SCHEMA_VERSION_V1_1",
    "materialize_record_event_aggregate_v1_1",
    "validate_record_event_aggregate_v1_1",
]
