"""Fail-closed validation for ``event_eeg_findings_v3``.

v3 is an additive contract over the frozen v2 event evidence graph.  The v2
projection is validated first, including its raw-sample causal boundary and
EEG-only provenance firewall.  This module then closes the new quantity,
rhythmicity/periodicity, acquisition, differential-hypothesis, and explicit
event-outcome relations.

The extension never creates a second route to scalp-onset localization.
Positive onset claims still require v2 ``onset_eligible`` Findings, whose raw
dependencies are restricted to past-and-present ``onset_causal`` views.
"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from .event_findings_v2_validation import (
    validate_event_eeg_findings_v2_payload,
)


EVENT_EEG_FINDINGS_V3_SCHEMA_VERSION = "event_eeg_findings_v3"
EVENT_FINDINGS_V2_TO_V3_MIGRATOR_ID = (
    "event_eeg_findings_v2_to_v3_fail_closed_v1"
)

_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA_PATH = _ROOT / "schemas" / "clinical_eeg_event_findings_v3.schema.json"
_V2_SCHEMA_PATH = (
    _ROOT / "schemas" / "clinical_eeg_event_findings_v2.schema.json"
)
_TOL = 1e-6
_V3_EXTENSION_KEYS = {
    "occurrence_burden_variability",
    "rhythm_periodicity_qualification",
    "acquisition_capabilities",
    "competing_hypotheses",
    "event_outcome",
    "v3_migration",
}
_REQUIRED_MIGRATION_LOSSES = {
    "v2_rhythm_semantics_not_recoverable",
    "occurrence_burden_not_recorded_in_v2",
    "variability_not_recorded_in_v2",
    "acquisition_capabilities_not_recorded_in_v2",
    "competing_hypotheses_not_recorded_in_v2",
}
_QUALIFIED_OUTCOMES = {
    "qualified_electrographic_event",
    "qualified_electrographic_seizure",
}
_SENSITIVE_TERM_TOKENS = {
    "dc_shift": "dc_shift",
    "direct_current_shift": "dc_shift",
    "high_frequency_oscillation": "high_frequency_oscillation",
    "hfo": "high_frequency_oscillation",
    "fast_ripple": "high_frequency_oscillation",
    "ripple": "high_frequency_oscillation",
    "low_voltage_fast_activity": "low_voltage_fast_activity",
    "lvfa": "low_voltage_fast_activity",
}


@lru_cache(maxsize=1)
def _schema_validator() -> Draft202012Validator:
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    v2_schema = json.loads(_V2_SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    registry = Registry().with_resource(
        str(v2_schema["$id"]), Resource.from_contents(v2_schema)
    )
    return Draft202012Validator(schema, registry=registry)


def _path(error: Any) -> str:
    parts = [str(item) for item in error.absolute_path]
    return ".".join(parts) if parts else "$"


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


def _reject_nonfinite(value: object, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} must be finite")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_nonfinite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nonfinite(item, f"{path}[{index}]")


def _unique(values: Iterable[str], context: str) -> set[str]:
    rows = list(values)
    if len(rows) != len(set(rows)):
        raise ValueError(f"{context} contains duplicate IDs")
    return set(rows)


def _require_refs(values: Iterable[str], available: set[str], context: str) -> None:
    missing = sorted(set(values).difference(available))
    if missing:
        raise ValueError(f"{context} references unknown IDs: {missing}")


def _span(value: Mapping[str, object], context: str) -> tuple[float, float]:
    start = float(value["start"])
    stop = float(value["stop"])
    if stop <= start:
        raise ValueError(f"{context} must have positive duration")
    return start, stop


def _overlap(left: tuple[float, float], right: tuple[float, float]) -> float:
    return max(0.0, min(left[1], right[1]) - max(left[0], right[0]))


def project_event_eeg_findings_v3_to_v2(value: object) -> dict[str, Any]:
    """Return the exact frozen-v2 projection of one v3 wire payload."""

    if type(value) is not dict:
        raise TypeError("event_eeg_findings_v3 payload must be an object")
    result: dict[str, Any] = deepcopy(value)
    for key in _V3_EXTENSION_KEYS:
        result.pop(key, None)
    result["schema_version"] = "event_eeg_findings_v2"
    return result


def _sensitive_feature_class(finding: Mapping[str, Any]) -> str | None:
    if finding["family"] == "high_frequency":
        return "high_frequency_oscillation"
    term_id = str(finding["term"]["term_id"]).lower()
    for token, feature_class in _SENSITIVE_TERM_TOKENS.items():
        if token in term_id:
            return feature_class
    return None


def _finding_has_future_free_causal_evidence(
    payload: Mapping[str, Any], finding: Mapping[str, Any]
) -> bool:
    """Recheck the v2 causal receipt boundary at the v3 hypothesis edge."""

    waveform_map = {
        str(row["waveform_evidence_id"]): row
        for row in payload["waveform_evidence"]
    }
    dependencies: list[Mapping[str, Any]] = []
    for measurement in finding["measurements"]:
        dependency = measurement["source_binding"]["raw_sample_dependency"]
        if dependency is not None:
            dependencies.append(dependency)
    for waveform_id in finding["waveform_evidence_ids"]:
        dependency = waveform_map[str(waveform_id)]["raw_sample_dependency"]
        if dependency is not None:
            dependencies.append(dependency)
    return bool(dependencies) and all(
        dependency["view_role"] == "onset_causal"
        and dependency["dependency_policy"] == "past_and_present_only"
        and not dependency["future_sample_access"]
        and dependency["onset_evidence_authorized"]
        and dependency["onset_support_eligible"]
        for dependency in dependencies
    )


def _expected_incidence(rate_per_minute: float, count: int) -> str:
    if count == 0:
        return "none"
    if rate_per_minute >= 6.0 - _TOL:
        return "abundant"
    if rate_per_minute >= 1.0 - _TOL:
        return "frequent"
    if rate_per_minute >= 1.0 / 60.0 - _TOL:
        return "occasional"
    return "rare"


def _expected_prevalence(proportion: float) -> str:
    if proportion <= _TOL:
        return "none"
    if proportion >= 0.90 - _TOL:
        return "continuous"
    if proportion >= 0.50 - _TOL:
        return "abundant"
    if proportion >= 0.10 - _TOL:
        return "frequent"
    if proportion >= 0.01 - _TOL:
        return "occasional"
    return "rare"


def event_burden_interval_union_sha256_v3(
    *,
    scope_interval: Mapping[str, object],
    evaluable_seconds: float,
    interval_union_policy_id: str,
    interval_union: list[Mapping[str, object]],
) -> str:
    """Content-bind a canonical, de-duplicated burden interval union."""

    return _sha256(
        {
            "binding_domain": "clinical-eeg-event-burden-interval-union-v3",
            "scope_interval": deepcopy(dict(scope_interval)),
            "evaluable_seconds": float(evaluable_seconds),
            "interval_union_policy_id": str(interval_union_policy_id),
            "interval_union": [deepcopy(dict(item)) for item in interval_union],
        }
    )


def event_occurrence_roster_sha256_v3(
    *,
    term_id: str,
    scope_interval: Mapping[str, object],
    evaluable_seconds: float,
    deduplication_policy_id: str,
    deduplicated_occurrences: list[Mapping[str, object]],
) -> str:
    """Content-bind the occurrence roster used for count and rate."""

    return _sha256(
        {
            "binding_domain": "clinical-eeg-event-occurrence-roster-v3",
            "term_id": str(term_id),
            "scope_interval": deepcopy(dict(scope_interval)),
            "evaluable_seconds": float(evaluable_seconds),
            "deduplication_policy_id": str(deduplication_policy_id),
            "deduplicated_occurrences": [
                deepcopy(dict(item)) for item in deduplicated_occurrences
            ],
        }
    )


def _canonical_interval_union(
    rows: list[Mapping[str, object]],
) -> list[dict[str, float]]:
    ordered = sorted(
        (
            {
                "start": float(row["interval"]["start"]),  # type: ignore[index]
                "stop": float(row["interval"]["stop"]),  # type: ignore[index]
                "resolution_seconds": float(
                    row["interval"]["resolution_seconds"]  # type: ignore[index]
                ),
            }
            for row in rows
        ),
        key=lambda item: (item["start"], item["stop"]),
    )
    result: list[dict[str, float]] = []
    for row in ordered:
        if not result or row["start"] > result[-1]["stop"] + _TOL:
            result.append(dict(row))
            continue
        result[-1]["stop"] = max(result[-1]["stop"], row["stop"])
        result[-1]["resolution_seconds"] = max(
            result[-1]["resolution_seconds"], row["resolution_seconds"]
        )
    return result


def _validate_occurrence_burden_variability(
    payload: Mapping[str, Any],
    *,
    finding_map: Mapping[str, Mapping[str, Any]],
    opportunity_map: Mapping[str, Mapping[str, Any]],
    measurement_ids: set[str],
) -> None:
    block = payload["occurrence_burden_variability"]
    status = str(block["status"])
    summaries = block["summaries"]
    reasons = block["reason_codes"]
    if status == "available":
        if reasons or not summaries:
            raise ValueError(
                "available occurrence/burden/variability requires summaries and no reasons"
            )
    elif not reasons:
        raise ValueError(
            "limited/not-evaluable occurrence/burden/variability requires reasons"
        )
    if status == "not_evaluable" and summaries:
        raise ValueError(
            "not-evaluable occurrence/burden/variability cannot carry summaries"
        )

    term_ids = {
        str(row["term"]["term_id"]) for row in payload["findings"]
    } | {
        str(row["term"]["term_id"])
        for row in payload["pattern_candidates"]
    }
    pattern_map = {
        str(row["pattern_candidate_id"]): row
        for row in payload["pattern_candidates"]
    }
    _unique(
        (str(row["summary_id"]) for row in summaries),
        "occurrence_burden_variability.summaries.summary_id",
    )
    for index, summary in enumerate(summaries):
        context = f"occurrence_burden_variability.summaries[{index}]"
        summary_term_id = str(summary["term_id"])
        _require_refs((summary_term_id,), term_ids, f"{context}.term_id")
        pattern_ids = set(str(item) for item in summary["pattern_candidate_ids"])
        _require_refs(pattern_ids, set(pattern_map), f"{context}.pattern_candidate_ids")
        if any(
            str(pattern_map[item]["term"]["term_id"]) != summary_term_id
            for item in pattern_ids
        ):
            raise ValueError(f"{context} composite pattern term does not match summary")
        allowed_evidence = {
            evidence_id
            for evidence_id, finding in finding_map.items()
            if str(finding["term"]["term_id"]) == summary_term_id
        }
        for pattern_id in pattern_ids:
            allowed_evidence.update(
                str(item) for item in pattern_map[pattern_id]["required_atom_ids"]
            )
        scope_start, scope_stop = _span(summary["scope_interval"], f"{context}.scope_interval")
        duration = scope_stop - scope_start
        if scope_start < -_TOL or scope_stop > float(
            payload["coordinates"]["recording_duration_seconds"]
        ) + _TOL:
            raise ValueError(f"{context}.scope_interval lies outside the recording")

        occurrence = summary["occurrence"]
        occurrence_refs = set(
            str(item) for item in occurrence["supporting_evidence_ids"]
        )
        occurrence_opps = set(
            str(item) for item in occurrence["evaluation_opportunity_ids"]
        )
        _require_refs(occurrence_refs, set(finding_map), f"{context}.occurrence")
        _require_refs(
            occurrence_refs,
            allowed_evidence,
            f"{context}.occurrence term-bound evidence",
        )
        _require_refs(occurrence_opps, set(opportunity_map), f"{context}.occurrence")
        occurrence_status = str(occurrence["status"])
        roster = occurrence["deduplicated_occurrences"]
        if occurrence_status == "measured":
            if (
                occurrence["count"] is None
                or occurrence["evaluable_seconds"] is None
                or occurrence["rate_per_minute"] is None
                or occurrence["deduplication_policy_id"] is None
                or occurrence["deduplication_sha256"] is None
                or occurrence["reason_codes"]
                or not occurrence_opps
            ):
                raise ValueError(f"{context}.occurrence measured state is incomplete")
            if any(opportunity_map[item]["status"] != "sufficient" for item in occurrence_opps):
                raise ValueError(f"{context}.occurrence lacks sufficient opportunity")
            count = int(occurrence["count"])
            occurrence_evaluable = float(occurrence["evaluable_seconds"])
            rate = float(occurrence["rate_per_minute"])
            if occurrence_evaluable > duration + _TOL:
                raise ValueError(
                    f"{context}.occurrence evaluable denominator exceeds scope"
                )
            roster_ids = _unique(
                (str(item["occurrence_id"]) for item in roster),
                f"{context}.occurrence.deduplicated_occurrences",
            )
            if count != len(roster_ids):
                raise ValueError(f"{context}.occurrence count differs from roster")
            roster_evidence: set[str] = set()
            previous_roster_key: tuple[float, float, str] | None = None
            previous_roster_stop: float | None = None
            for roster_index, roster_row in enumerate(roster):
                roster_context = (
                    f"{context}.occurrence.deduplicated_occurrences[{roster_index}]"
                )
                roster_start, roster_stop = _span(
                    roster_row["interval"], f"{roster_context}.interval"
                )
                if (
                    roster_start < scope_start - _TOL
                    or roster_stop > scope_stop + _TOL
                ):
                    raise ValueError(f"{roster_context} lies outside summary scope")
                roster_key = (
                    roster_start,
                    roster_stop,
                    str(roster_row["occurrence_id"]),
                )
                if previous_roster_key is not None and roster_key <= previous_roster_key:
                    raise ValueError(f"{context}.occurrence roster is not canonical")
                if (
                    previous_roster_stop is not None
                    and roster_start < previous_roster_stop - _TOL
                ):
                    raise ValueError(
                        f"{context}.occurrence roster contains overlapping duplicates"
                    )
                previous_roster_key = roster_key
                previous_roster_stop = roster_stop
                local_evidence = set(
                    str(item) for item in roster_row["evidence_ids"]
                )
                _require_refs(local_evidence, set(finding_map), roster_context)
                _require_refs(
                    local_evidence,
                    allowed_evidence,
                    f"{roster_context} term-bound evidence",
                )
                if any(
                    finding_map[item]["status"] != "present"
                    for item in local_evidence
                ):
                    raise ValueError(f"{roster_context} requires present evidence")
                roster_evidence.update(local_evidence)
            if not roster_evidence.issubset(occurrence_refs):
                raise ValueError(
                    f"{context}.occurrence supporting evidence omits roster evidence"
                )
            expected_roster_sha = event_occurrence_roster_sha256_v3(
                term_id=str(summary["term_id"]),
                scope_interval=summary["scope_interval"],
                evaluable_seconds=occurrence_evaluable,
                deduplication_policy_id=str(
                    occurrence["deduplication_policy_id"]
                ),
                deduplicated_occurrences=roster,
            )
            if occurrence["deduplication_sha256"] != expected_roster_sha:
                raise ValueError(f"{context}.occurrence roster receipt drifted")
            expected_rate = count * 60.0 / occurrence_evaluable
            if abs(rate - expected_rate) > max(_TOL, expected_rate * 1e-6):
                raise ValueError(f"{context}.occurrence rate/count/scope disagree")
            if occurrence["incidence_category"] != _expected_incidence(rate, count):
                raise ValueError(f"{context}.occurrence incidence category is inconsistent")
            expected_status = "absent_with_opportunity" if count == 0 else "present"
            if not occurrence_refs or any(
                finding_map[item]["status"] != expected_status
                for item in occurrence_refs
            ):
                raise ValueError(
                    f"{context}.occurrence count requires {expected_status} evidence"
                )
            if not {
                str(finding_map[item]["evaluation_opportunity_id"])
                for item in occurrence_refs
            }.issubset(occurrence_opps):
                raise ValueError(
                    f"{context}.occurrence omits an evidence opportunity"
                )
            if count == 0 and any(
                finding_map[item]["sensitivity_receipt_id"] is None
                for item in occurrence_refs
            ):
                raise ValueError(
                    f"{context}.occurrence none requires sensitivity qualification"
                )
        elif occurrence_status == "limited":
            if (
                not occurrence["reason_codes"]
                or occurrence["incidence_category"] != "indeterminate"
                or occurrence["count"] is not None
                or occurrence["evaluable_seconds"] is not None
                or occurrence["rate_per_minute"] is not None
                or roster
                or occurrence["deduplication_policy_id"] is not None
                or occurrence["deduplication_sha256"] is not None
            ):
                raise ValueError(f"{context}.occurrence limited state is over-assertive")
        elif any(
            (
                occurrence["count"] is not None,
                occurrence["evaluable_seconds"] is not None,
                occurrence["rate_per_minute"] is not None,
                occurrence["incidence_category"] != "not_evaluable",
                bool(roster),
                occurrence["deduplication_policy_id"] is not None,
                occurrence["deduplication_sha256"] is not None,
                bool(occurrence_refs),
                bool(occurrence_opps),
                not occurrence["reason_codes"],
            )
        ):
            raise ValueError(
                f"{context}.occurrence not_evaluable is not an absence claim"
            )

        burden = summary["burden"]
        burden_refs = set(str(item) for item in burden["supporting_evidence_ids"])
        burden_opps = set(str(item) for item in burden["evaluation_opportunity_ids"])
        _require_refs(burden_refs, set(finding_map), f"{context}.burden")
        _require_refs(
            burden_refs,
            allowed_evidence,
            f"{context}.burden term-bound evidence",
        )
        _require_refs(burden_opps, set(opportunity_map), f"{context}.burden")
        burden_status = str(burden["status"])
        if burden_status == "measured":
            if (
                burden["observed_seconds"] is None
                or burden["evaluable_seconds"] is None
                or burden["proportion"] is None
                or burden["interval_union_policy_id"] is None
                or burden["interval_union_sha256"] is None
                or burden["reason_codes"]
                or not burden_opps
            ):
                raise ValueError(f"{context}.burden measured state is incomplete")
            if any(opportunity_map[item]["status"] != "sufficient" for item in burden_opps):
                raise ValueError(f"{context}.burden lacks sufficient opportunity")
            observed = float(burden["observed_seconds"])
            evaluable = float(burden["evaluable_seconds"])
            proportion = float(burden["proportion"])
            if (
                occurrence_status == "measured"
                and abs(evaluable - float(occurrence["evaluable_seconds"])) > _TOL
            ):
                raise ValueError(
                    f"{context} occurrence and burden evaluable denominators disagree"
                )
            if (
                evaluable > duration + _TOL
                or observed > evaluable + _TOL
                or abs(proportion - observed / evaluable) > _TOL
            ):
                raise ValueError(
                    f"{context}.burden numerator/denominator/proportion disagree"
                )
            union_seconds = 0.0
            previous_stop: float | None = None
            for union_index, union_row in enumerate(burden["interval_union"]):
                union_start, union_stop = _span(
                    union_row,
                    f"{context}.burden.interval_union[{union_index}]",
                )
                if (
                    union_start < scope_start - _TOL
                    or union_stop > scope_stop + _TOL
                ):
                    raise ValueError(
                        f"{context}.burden interval union leaves its scope"
                    )
                if previous_stop is not None and union_start < previous_stop - _TOL:
                    raise ValueError(
                        f"{context}.burden interval union overlaps or is unsorted"
                    )
                union_seconds += union_stop - union_start
                previous_stop = union_stop
            if abs(union_seconds - observed) > _TOL:
                raise ValueError(
                    f"{context}.burden observed_seconds differs from interval union"
                )
            canonical_union = _canonical_interval_union(roster)
            serialized_union = burden["interval_union"]
            if len(serialized_union) != len(canonical_union) or any(
                abs(float(serialized[key]) - expected[key]) > _TOL
                for serialized, expected in zip(serialized_union, canonical_union)
                for key in ("start", "stop", "resolution_seconds")
            ):
                raise ValueError(
                    f"{context}.burden union is not the canonical occurrence union"
                )
            expected_union_sha = event_burden_interval_union_sha256_v3(
                scope_interval=summary["scope_interval"],
                evaluable_seconds=evaluable,
                interval_union_policy_id=str(burden["interval_union_policy_id"]),
                interval_union=burden["interval_union"],
            )
            if burden["interval_union_sha256"] != expected_union_sha:
                raise ValueError(f"{context}.burden interval-union receipt drifted")
            if burden["prevalence_category"] != _expected_prevalence(proportion):
                raise ValueError(f"{context}.burden prevalence category is inconsistent")
            expected_status = (
                "absent_with_opportunity" if proportion <= _TOL else "present"
            )
            if not burden_refs or any(
                finding_map[item]["status"] != expected_status for item in burden_refs
            ):
                raise ValueError(f"{context}.burden requires {expected_status} evidence")
            if not {
                str(finding_map[item]["evaluation_opportunity_id"])
                for item in burden_refs
            }.issubset(burden_opps):
                raise ValueError(f"{context}.burden omits an evidence opportunity")
            if proportion <= _TOL and any(
                finding_map[item]["sensitivity_receipt_id"] is None
                for item in burden_refs
            ):
                raise ValueError(
                    f"{context}.burden none requires sensitivity qualification"
                )
        elif burden_status == "limited":
            if (
                not burden["reason_codes"]
                or burden["prevalence_category"] != "indeterminate"
                or burden["observed_seconds"] is not None
                or burden["evaluable_seconds"] is not None
                or burden["proportion"] is not None
                or burden["interval_union"]
                or burden["interval_union_policy_id"] is not None
                or burden["interval_union_sha256"] is not None
            ):
                raise ValueError(f"{context}.burden limited state is over-assertive")
        elif any(
            (
                burden["observed_seconds"] is not None,
                burden["evaluable_seconds"] is not None,
                burden["proportion"] is not None,
                burden["prevalence_category"] != "not_evaluable",
                bool(burden["interval_union"]),
                burden["interval_union_policy_id"] is not None,
                burden["interval_union_sha256"] is not None,
                bool(burden_refs),
                bool(burden_opps),
                not burden["reason_codes"],
            )
        ):
            raise ValueError(f"{context}.burden not_evaluable is not an absence claim")

        variability = summary["variability"]
        variability_refs = set(
            str(item) for item in variability["supporting_evidence_ids"]
        )
        _require_refs(variability_refs, set(finding_map), f"{context}.variability")
        _require_refs(
            variability_refs,
            allowed_evidence,
            f"{context}.variability term-bound evidence",
        )
        dimensions = variability["dimensions"]
        dimension_names = _unique(
            (str(row["dimension"]) for row in dimensions),
            f"{context}.variability.dimensions",
        )
        del dimension_names
        variability_status = str(variability["status"])
        if variability_status == "measured":
            if variability["reason_codes"] or not dimensions or not variability_refs:
                raise ValueError(f"{context}.variability measured state is incomplete")
            if any(finding_map[item]["status"] != "present" for item in variability_refs):
                raise ValueError(f"{context}.variability requires present evidence")
        elif variability_status == "limited":
            if not variability["reason_codes"]:
                raise ValueError(f"{context}.variability limited state requires reasons")
        elif dimensions or variability_refs or not variability["reason_codes"]:
            raise ValueError(f"{context}.variability not_evaluable carries evidence")
        for dimension_index, dimension in enumerate(dimensions):
            dim_context = f"{context}.variability.dimensions[{dimension_index}]"
            dim_refs = set(str(item) for item in dimension["supporting_evidence_ids"])
            dim_measurements = set(str(item) for item in dimension["measurement_ids"])
            _require_refs(dim_refs, set(finding_map), dim_context)
            _require_refs(
                dim_refs,
                allowed_evidence,
                f"{dim_context} term-bound evidence",
            )
            _require_refs(dim_measurements, measurement_ids, dim_context)
            dim_status = str(dimension["status"])
            if dim_status in {"stable", "variable"}:
                if dimension["reason_codes"] or not dim_refs:
                    raise ValueError(f"{dim_context} asserted state lacks evidence")
            elif not dimension["reason_codes"]:
                raise ValueError(f"{dim_context} indeterminate state requires reasons")
            if dim_status == "not_evaluable" and (dim_refs or dim_measurements):
                raise ValueError(f"{dim_context} not_evaluable carries evidence")

        component_statuses = {
            occurrence_status,
            burden_status,
            variability_status,
        }
        if "measured" in component_statuses and any(
            pattern_map[item]["status"] != "present" for item in pattern_ids
        ):
            raise ValueError(
                f"{context} measured composite summary requires a present pattern candidate"
            )
        if status == "available" and component_statuses != {"measured"}:
            raise ValueError(f"{context} is incomplete under available parent status")
        if summary["reason_codes"] and status == "available":
            raise ValueError(f"{context} available summary cannot carry reasons")


def _validate_rhythm_periodicity(
    payload: Mapping[str, Any],
    *,
    finding_map: Mapping[str, Mapping[str, Any]],
    opportunity_map: Mapping[str, Mapping[str, Any]],
) -> None:
    block = payload["rhythm_periodicity_qualification"]
    finding_sets: dict[str, set[str]] = {}
    term_sets: dict[str, set[str]] = {}
    decision_sets: dict[str, set[str]] = {}
    capability_ids = {
        str(row["receipt_id"])
        for row in payload["capability_qualification_receipts"]
    }
    decision_ids = {
        str(row["receipt_id"]) for row in payload["term_decision_receipts"]
    }
    for track in ("rhythmicity", "periodicity"):
        gate = block[track]
        context = f"rhythm_periodicity_qualification.{track}"
        findings = set(str(item) for item in gate["finding_ids"])
        terms = set(str(item) for item in gate["term_ids"])
        opportunities = set(
            str(item) for item in gate["evaluation_opportunity_ids"]
        )
        capabilities = set(str(item) for item in gate["capability_receipt_ids"])
        decisions = set(str(item) for item in gate["term_decision_receipt_ids"])
        _require_refs(findings, set(finding_map), context)
        _require_refs(opportunities, set(opportunity_map), context)
        _require_refs(capabilities, capability_ids, context)
        _require_refs(decisions, decision_ids, context)
        if any(finding_map[item]["family"] != "rhythm" for item in findings):
            raise ValueError(f"{context} can reference only v2 rhythm Findings")
        expected_terms = {
            str(finding_map[item]["term"]["term_id"]) for item in findings
        }
        expected_opportunities = {
            str(finding_map[item]["evaluation_opportunity_id"]) for item in findings
        }
        expected_capabilities = {
            str(finding_map[item]["capability_receipt_id"])
            for item in findings
            if finding_map[item]["capability_receipt_id"] is not None
        }
        expected_decisions = {
            str(finding_map[item]["term_decision_receipt_id"])
            for item in findings
            if finding_map[item]["term_decision_receipt_id"] is not None
        }
        if terms != expected_terms or opportunities != expected_opportunities:
            raise ValueError(f"{context} does not exactly bind its Findings")
        if capabilities != expected_capabilities or decisions != expected_decisions:
            raise ValueError(f"{context} clinical receipt projection is incomplete")

        status = str(gate["qualification_status"])
        if status in {"qualified_present", "qualified_absent_with_opportunity"}:
            finding_status = (
                "present"
                if status == "qualified_present"
                else "absent_with_opportunity"
            )
            if (
                not findings
                or gate["reason_codes"]
                or not capabilities
                or not decisions
                or any(
                    finding_map[item]["status"] != finding_status
                    or finding_map[item]["assertion_level"]
                    != "report_eligible_automated"
                    for item in findings
                )
                or any(
                    opportunity_map[item]["status"] != "sufficient"
                    for item in opportunities
                )
            ):
                raise ValueError(f"{context} does not meet its qualified gate")
        elif status == "candidate_only":
            if (
                not findings
                or capabilities
                or decisions
                or not gate["reason_codes"]
                or any(
                    finding_map[item]["status"] not in {"present", "uncertain"}
                    or finding_map[item]["assertion_level"]
                    == "report_eligible_automated"
                    for item in findings
                )
            ):
                raise ValueError(f"{context} candidate gate is inconsistent")
        elif (
            capabilities
            or decisions
            or not gate["reason_codes"]
            or any(finding_map[item]["status"] != "not_evaluable" for item in findings)
        ):
            raise ValueError(f"{context} not_evaluable gate is inconsistent")
        finding_sets[track] = findings
        term_sets[track] = terms
        decision_sets[track] = decisions

    if finding_sets["rhythmicity"].intersection(finding_sets["periodicity"]):
        raise ValueError("rhythmicity and periodicity cannot share a Finding")
    if term_sets["rhythmicity"].intersection(term_sets["periodicity"]):
        raise ValueError("rhythmicity and periodicity cannot silently share a term")
    if decision_sets["rhythmicity"].intersection(decision_sets["periodicity"]):
        raise ValueError("rhythmicity and periodicity require separate decisions")
    native = payload["v3_migration"] is None
    rhythm_findings = {
        evidence_id
        for evidence_id, row in finding_map.items()
        if row["family"] == "rhythm"
    }
    if native and (
        finding_sets["rhythmicity"] | finding_sets["periodicity"]
    ) != rhythm_findings:
        raise ValueError(
            "native v3 must explicitly classify every rhythm Finding as rhythmicity or periodicity"
        )


def _validate_acquisition_capabilities(
    payload: Mapping[str, Any],
    *,
    finding_map: Mapping[str, Mapping[str, Any]],
    opportunity_map: Mapping[str, Mapping[str, Any]],
) -> None:
    block = payload["acquisition_capabilities"]
    capabilities = block["capabilities"]
    status = str(block["status"])
    _unique(
        (str(row["capability_id"]) for row in capabilities),
        "acquisition_capabilities.capability_id",
    )
    term_ids = _unique(
        (str(row["term_id"]) for row in capabilities),
        "acquisition_capabilities.term_id",
    )
    capability_by_term = {str(row["term_id"]): row for row in capabilities}
    if status == "available":
        if not capabilities or block["reason_codes"] or any(
            row["status"] != "evaluable" for row in capabilities
        ):
            raise ValueError("available acquisition capabilities are incomplete")
    elif status == "limited":
        if not capabilities or not block["reason_codes"]:
            raise ValueError("limited acquisition capabilities require rows and reasons")
    elif (
        not block["reason_codes"]
        or any(row["status"] != "not_evaluable" for row in capabilities)
    ):
        raise ValueError("not-evaluable acquisition block cannot imply capability")

    for index, row in enumerate(capabilities):
        context = f"acquisition_capabilities.capabilities[{index}]"
        opportunity_ids = set(
            str(item) for item in row["evaluation_opportunity_ids"]
        )
        _require_refs(opportunity_ids, set(opportunity_map), context)
        row_status = str(row["status"])
        bandwidth = row["effective_bandwidth_hz"]
        sample_rate = row["sample_rate_hz"]
        if bandwidth is not None:
            low, high = (float(item) for item in bandwidth)
            if high <= low:
                raise ValueError(f"{context}.effective_bandwidth_hz is invalid")
            if sample_rate is not None and float(sample_rate) + _TOL < 2.0 * high:
                raise ValueError(f"{context} violates the Nyquist acquisition bound")
        if row_status == "evaluable":
            if (
                not row["source_view_ids"]
                or bandwidth is None
                or sample_rate is None
                or not opportunity_ids
                or row["reason_codes"]
                or any(
                    opportunity_map[item]["status"] != "sufficient"
                    for item in opportunity_ids
                )
            ):
                raise ValueError(f"{context} evaluable state lacks physical opportunity")
            if row["feature_class"] == "dc_shift" and (
                row["coupling"] != "dc" or float(bandwidth[0]) > _TOL
            ):
                raise ValueError(f"{context} DC shift requires DC coupling and zero-Hz access")
            if (
                row["feature_class"] == "high_frequency_oscillation"
                and float(bandwidth[1]) <= 80.0 + _TOL
            ):
                raise ValueError(
                    f"{context} HFO evaluation requires effective bandwidth above 80 Hz"
                )
        elif row_status == "limited":
            if not row["reason_codes"]:
                raise ValueError(f"{context} limited state requires reasons")
        elif (
            not row["reason_codes"]
            or any(
                opportunity_map[item]["status"] == "sufficient"
                for item in opportunity_ids
            )
        ):
            raise ValueError(f"{context} not_evaluable contradicts its opportunity")

        matching = [
            finding
            for finding in finding_map.values()
            if finding["term"]["term_id"] == row["term_id"]
        ]
        if row_status == "not_evaluable" and any(
            finding["status"] != "not_evaluable" for finding in matching
        ):
            raise ValueError(
                f"{context} not_evaluable capability cannot become present or absent"
            )
        if row_status == "limited" and any(
            finding["status"] in {"present", "absent_with_opportunity"}
            for finding in matching
        ):
            raise ValueError(f"{context} limited capability cannot make a hard assertion")

    if payload["v3_migration"] is None:
        for evidence_id, finding in finding_map.items():
            feature_class = _sensitive_feature_class(finding)
            if feature_class is None:
                continue
            term_id = str(finding["term"]["term_id"])
            _require_refs((term_id,), term_ids, f"findings[{evidence_id}] acquisition gate")
            if capability_by_term[term_id]["feature_class"] != feature_class:
                raise ValueError(
                    f"findings[{evidence_id}] acquisition feature class is inconsistent"
                )


def _validate_competing_hypotheses(
    payload: Mapping[str, Any],
    *,
    finding_map: Mapping[str, Mapping[str, Any]],
) -> None:
    block = payload["competing_hypotheses"]
    hypotheses = block["hypotheses"]
    hypothesis_ids = _unique(
        (str(row["hypothesis_id"]) for row in hypotheses),
        "competing_hypotheses.hypothesis_id",
    )
    status = str(block["status"])
    selected = block["selected_hypothesis_id"]
    if status == "available":
        if not hypotheses or selected is None or block["reason_codes"]:
            raise ValueError("available competing hypotheses require a selected candidate")
    elif status == "limited":
        if not block["reason_codes"]:
            raise ValueError("limited competing hypotheses require reasons")
    elif hypotheses or selected is not None or not block["reason_codes"]:
        raise ValueError("not-evaluable competing hypotheses cannot select a candidate")
    if selected is not None:
        _require_refs((str(selected),), hypothesis_ids, "selected_hypothesis_id")

    ranked: list[int] = []
    hypothesis_map: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(hypotheses):
        context = f"competing_hypotheses.hypotheses[{index}]"
        hypothesis_id = str(row["hypothesis_id"])
        hypothesis_map[hypothesis_id] = row
        supporting = set(str(item) for item in row["supporting_evidence_ids"])
        contradictory = set(str(item) for item in row["contradictory_evidence_ids"])
        _require_refs(supporting | contradictory, set(finding_map), context)
        if supporting.intersection(contradictory):
            raise ValueError(f"{context} support and contradiction must be disjoint")
        if any(finding_map[item]["status"] != "present" for item in supporting):
            raise ValueError(f"{context} support requires present signal evidence")
        if any(
            finding_map[item]["status"] in {"uncertain", "not_evaluable"}
            for item in contradictory
        ):
            raise ValueError(f"{context} uncertain evidence cannot contradict")
        disposition = str(row["disposition"])
        if disposition == "supported" and not supporting:
            raise ValueError(f"{context} supported hypothesis lacks support")
        if disposition == "possible" and not supporting and not row["reason_codes"]:
            raise ValueError(f"{context} possible hypothesis lacks evidence/reason")
        if disposition == "disfavored" and not contradictory:
            raise ValueError(f"{context} disfavored hypothesis lacks counterevidence")
        if disposition == "not_evaluable" and (
            supporting or contradictory or not row["reason_codes"]
        ):
            raise ValueError(f"{context} not_evaluable carries evidence")
        if disposition in {"supported", "possible"}:
            if row["rank"] is None:
                raise ValueError(f"{context} ranked disposition lacks rank")
            ranked.append(int(row["rank"]))
        elif row["rank"] is not None:
            raise ValueError(f"{context} non-candidate disposition cannot be ranked")

        onset_eligible = bool(row["onset_claim_eligible"])
        causal_support = any(
            finding_map[item]["status"] == "present"
            and finding_map[item]["intrinsic_evidence_role"] == "onset_eligible"
            and _finding_has_future_free_causal_evidence(payload, finding_map[item])
            for item in supporting
        )
        expected_onset_eligible = bool(
            row["category"] == "cerebral_ictal"
            and disposition == "supported"
            and causal_support
        )
        if onset_eligible != expected_onset_eligible:
            raise ValueError(
                f"{context} onset eligibility is not closed by causal EEG evidence"
            )
    if sorted(ranked) != list(range(1, len(ranked) + 1)):
        raise ValueError("competing hypothesis ranks must be unique and contiguous")
    if selected is not None:
        selected_row = hypothesis_map[str(selected)]
        if (
            selected_row["disposition"] not in {"supported", "possible"}
            or selected_row["rank"] != 1
        ):
            raise ValueError("selected competing hypothesis must be the rank-1 candidate")

    scalp_status = payload["scalp_onset_hypothesis"]["localization_status"]
    if (
        payload["v3_migration"] is None
        and scalp_status in {"ranked_candidates", "phenotype_only"}
    ):
        if selected is None:
            raise ValueError("positive scalp-onset output requires a selected signal hypothesis")
        selected_row = hypothesis_map[str(selected)]
        if (
            selected_row["category"] != "cerebral_ictal"
            or not selected_row["onset_claim_eligible"]
        ):
            raise ValueError(
                "positive scalp-onset output requires a causally supported cerebral-ictal hypothesis"
            )


def _validate_event_outcome(
    payload: Mapping[str, Any],
    *,
    finding_map: Mapping[str, Mapping[str, Any]],
) -> None:
    row = payload["event_outcome"]
    outcome = str(row["outcome"])
    evidence = set(str(item) for item in row["evidence_ids"])
    competing = set(str(item) for item in row["competing_hypothesis_ids"])
    _require_refs(evidence, set(finding_map), "event_outcome.evidence_ids")
    competing_ids = {
        str(item["hypothesis_id"])
        for item in payload["competing_hypotheses"]["hypotheses"]
    }
    _require_refs(competing, competing_ids, "event_outcome.competing_hypothesis_ids")
    artifact_indices = [int(item) for item in row["artifact_interval_indices"]]
    artifacts = payload["quality"]["artifact_intervals"]
    if any(index >= len(artifacts) for index in artifact_indices):
        raise ValueError("event_outcome references an unknown artifact interval index")

    qualification = payload["event_qualification"]
    qualification_status = str(qualification["status"])
    event_support = set(
        str(item) for item in qualification["supporting_evidence_ids"]
    )
    direct_mapping = {
        "qualified_electrographic_seizure": "qualified_electrographic_seizure",
        "qualified_electrographic_event": "qualified_electrographic_event",
        "candidate_only": "unqualified_candidate",
        "no_demonstrable_scalp_ictal_change": "unqualified_candidate",
        "obscured_by_artifact": "not_evaluable",
        "not_possible_to_determine": "not_evaluable",
    }
    if qualification_status != direct_mapping[outcome]:
        raise ValueError("event_outcome contradicts event_qualification")
    if outcome in _QUALIFIED_OUTCOMES:
        if (
            row["reason_codes"]
            or not event_support
            or not event_support.issubset(evidence)
            or any(finding_map[item]["status"] != "present" for item in evidence)
            or (payload["v3_migration"] is None and not competing)
        ):
            raise ValueError("qualified event outcome lacks positive closed evidence")
        if payload["v3_migration"] is None:
            selected = payload["competing_hypotheses"][
                "selected_hypothesis_id"
            ]
            competing_map = {
                str(item["hypothesis_id"]): item
                for item in payload["competing_hypotheses"]["hypotheses"]
            }
            if (
                selected is None
                or str(selected) not in competing
                or competing_map[str(selected)]["category"] != "cerebral_ictal"
                or competing_map[str(selected)]["disposition"] != "supported"
            ):
                raise ValueError(
                    "qualified outcome requires its selected supported cerebral-ictal hypothesis"
                )
    elif outcome == "candidate_only":
        if (
            not row["reason_codes"]
            or any(
                finding_map[item]["status"] == "not_evaluable"
                for item in evidence
            )
            or (
                payload["v3_migration"] is None
                and not any(
                    finding_map[item]["status"] in {"present", "uncertain"}
                    for item in evidence
                )
            )
        ):
            raise ValueError("candidate-only outcome is inconsistent")
    elif outcome == "no_demonstrable_scalp_ictal_change":
        if (
            not row["reason_codes"]
            or not evidence
            or any(
                finding_map[item]["status"] != "absent_with_opportunity"
                for item in evidence
            )
        ):
            raise ValueError(
                "no-demonstrable-change outcome requires qualified negative evidence"
            )
    elif outcome == "obscured_by_artifact":
        final = tuple(float(item) for item in payload["window"]["final_interval"])
        if (
            not row["reason_codes"]
            or not artifact_indices
            or not any(
                _overlap(
                    final,
                    tuple(float(item) for item in artifacts[index]["interval"]),
                )
                > _TOL
                for index in artifact_indices
            )
        ):
            raise ValueError("artifact-obscured outcome lacks overlapping artifact")
    elif not row["reason_codes"]:
        raise ValueError("not-possible-to-determine event outcome requires reasons")
    if outcome != "obscured_by_artifact" and artifact_indices:
        raise ValueError("artifact interval indices are reserved for artifact-obscured outcome")


def _validate_v3_migration(payload: Mapping[str, Any]) -> None:
    receipt = payload["v3_migration"]
    if receipt is None:
        if payload["migration"] is not None:
            raise ValueError("a legacy v2 base requires an explicit v3 migration receipt")
        return
    projection = project_event_eeg_findings_v3_to_v2(payload)
    projection_sha256 = _sha256(projection)
    if (
        receipt["source_payload_sha256"] != projection_sha256
        or receipt["preserved_base_projection_sha256"] != projection_sha256
    ):
        raise ValueError("v3 migration does not preserve the complete v2 projection")
    losses = set(str(item) for item in receipt["loss_codes"])
    missing_losses = sorted(_REQUIRED_MIGRATION_LOSSES.difference(losses))
    if missing_losses:
        raise ValueError(f"v3 migration omits required loss codes: {missing_losses}")

    quantity = payload["occurrence_burden_variability"]
    rhythm = payload["rhythm_periodicity_qualification"]
    acquisition = payload["acquisition_capabilities"]
    competing = payload["competing_hypotheses"]
    if quantity["status"] != "not_evaluable" or quantity["summaries"]:
        raise ValueError("v2 migration cannot manufacture quantity or variability")
    for track in ("rhythmicity", "periodicity"):
        gate = rhythm[track]
        if gate["qualification_status"] != "not_evaluable" or any(
            gate[key]
            for key in (
                "term_ids",
                "finding_ids",
                "evaluation_opportunity_ids",
                "capability_receipt_ids",
                "term_decision_receipt_ids",
            )
        ):
            raise ValueError("v2 rhythm cannot be promoted to rhythmicity/periodicity")
    if acquisition["status"] != "not_evaluable" or any(
        row["status"] != "not_evaluable" for row in acquisition["capabilities"]
    ):
        raise ValueError("v2 migration cannot manufacture acquisition capability")
    if (
        competing["status"] != "not_evaluable"
        or competing["selected_hypothesis_id"] is not None
        or competing["hypotheses"]
    ):
        raise ValueError("v2 migration cannot manufacture competing hypotheses")

    expected_outcome = {
        "qualified_electrographic_seizure": "qualified_electrographic_seizure",
        "qualified_electrographic_event": "qualified_electrographic_event",
        "unqualified_candidate": "candidate_only",
        "not_evaluable": "not_possible_to_determine",
    }[payload["event_qualification"]["status"]]
    if payload["event_outcome"]["outcome"] != expected_outcome:
        raise ValueError("v2 migration changed the explicit event outcome")


def validate_event_eeg_findings_v3_payload(
    value: object,
    *,
    trusted_producer_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_calibration_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_capability_qualification_receipts: (
        Mapping[str, Mapping[str, object]] | None
    ) = None,
    trusted_sensitivity_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_term_decision_receipts: (
        Mapping[str, Mapping[str, object]] | None
    ) = None,
    trusted_registry_bindings: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, Any]:
    """Validate and defensively copy one v3 EEG-only event evidence graph."""

    if type(value) is not dict:
        raise TypeError("event_eeg_findings_v3 payload must be an object")
    candidate: dict[str, Any] = deepcopy(value)
    _reject_nonfinite(candidate)
    errors = sorted(
        _schema_validator().iter_errors(candidate),
        key=lambda item: list(item.path),
    )
    if errors:
        rendered = "; ".join(
            f"{_path(error)}: {error.message}" for error in errors[:8]
        )
        if len(errors) > 8:
            rendered += f"; ... {len(errors) - 8} more error(s)"
        raise ValueError(f"event_eeg_findings_v3 schema validation failed: {rendered}")

    base = project_event_eeg_findings_v3_to_v2(candidate)
    validate_event_eeg_findings_v2_payload(
        base,
        trusted_producer_receipts=trusted_producer_receipts,
        trusted_calibration_receipts=trusted_calibration_receipts,
        trusted_capability_qualification_receipts=(
            trusted_capability_qualification_receipts
        ),
        trusted_sensitivity_receipts=trusted_sensitivity_receipts,
        trusted_term_decision_receipts=trusted_term_decision_receipts,
        trusted_registry_bindings=trusted_registry_bindings,
    )

    finding_map = {
        str(row["evidence_id"]): row for row in candidate["findings"]
    }
    opportunity_map = {
        str(row["evaluation_opportunity_id"]): row
        for row in candidate["evaluation_opportunities"]
    }
    measurement_ids = {
        str(measurement["measurement_id"])
        for finding in candidate["findings"]
        for measurement in finding["measurements"]
    }
    _validate_v3_migration(candidate)
    _validate_occurrence_burden_variability(
        candidate,
        finding_map=finding_map,
        opportunity_map=opportunity_map,
        measurement_ids=measurement_ids,
    )
    _validate_rhythm_periodicity(
        candidate,
        finding_map=finding_map,
        opportunity_map=opportunity_map,
    )
    _validate_acquisition_capabilities(
        candidate,
        finding_map=finding_map,
        opportunity_map=opportunity_map,
    )
    _validate_competing_hypotheses(candidate, finding_map=finding_map)
    _validate_event_outcome(candidate, finding_map=finding_map)
    return candidate


__all__ = [
    "EVENT_EEG_FINDINGS_V3_SCHEMA_VERSION",
    "EVENT_FINDINGS_V2_TO_V3_MIGRATOR_ID",
    "event_burden_interval_union_sha256_v3",
    "event_occurrence_roster_sha256_v3",
    "project_event_eeg_findings_v3_to_v2",
    "validate_event_eeg_findings_v3_payload",
]
