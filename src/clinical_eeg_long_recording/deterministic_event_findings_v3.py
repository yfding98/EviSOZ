"""Native, signal-driven candidate slice for ``event_eeg_findings_v3``.

This module deliberately reuses the replayable native-v2 measurement graph.
It adds only v3 *candidate* heads that can be closed from those measurements:

* a rhythmicity candidate gate over the deterministic autocorrelation and
  spectral-concentration profile;
* an explicitly unavailable periodicity gate (v2 has no inter-discharge
  interval series);
* de-duplicated occurrence and event-window burden for deterministic later-
  involvement candidates when the complete opportunity denominator is known;
* acquisition evaluability facts, including a hard HFO bandwidth gate; and
* a research cerebral-ictal signal hypothesis supported only by v2
  future-free causal onset Findings.

No numeric threshold is promoted to a protected clinical term.  Event
qualification remains ``unqualified_candidate``/``not_evaluable``; all
capability, sensitivity and term-decision receipt arrays remain empty.  The
module performs no I/O and has no route to annotations, spreadsheets,
clinical text, patient metadata, or private labels.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Mapping, Sequence

from .deterministic_event_findings import (
    DEFAULT_DETERMINISTIC_EVENT_FINDINGS_POLICY,
    DeterministicEventFindingsPolicy,
    DeterministicViewInput,
    _canonical_sha256,
)
from .deterministic_event_findings_v2 import (
    DEFAULT_EVENT_FINDINGS_V2_REGISTRY_BINDINGS,
    DETERMINISTIC_EVENT_FINDINGS_V2_METHOD_ID,
    produce_deterministic_event_eeg_findings_v2,
)
from .event_findings_v2_validation import (
    validate_event_eeg_findings_v2_payload,
)
from .event_findings_v3_validation import (
    event_burden_interval_union_sha256_v3,
    event_occurrence_roster_sha256_v3,
    validate_event_eeg_findings_v3_payload,
)


DETERMINISTIC_EVENT_FINDINGS_V3_CANDIDATE_METHOD_ID = (
    "DETERMINISTIC-EVENT-FINDINGS-V3-CANDIDATE"
)
DETERMINISTIC_EVENT_FINDINGS_V3_CANDIDATE_POLICY_ID = (
    "DETERMINISTIC-EVENT-FINDINGS-V3-CANDIDATE-POLICY-V1"
)

_RHYTHMICITY_TERM_ID = "deterministic_event_rhythmicity_profile"
_PERIODICITY_REASON = "inter_discharge_interval_series_not_measured"
_OCCURRENCE_TERM_ID = "deterministic_later_involvement_candidate"
_HFO_TERM_ID = "deterministic_high_frequency_candidate"
_DEDUPLICATION_POLICY_ID = (
    "OVERLAPPING-OR-ADJACENT-WITHIN-RESOLUTION-V1"
)
_INTERVAL_UNION_POLICY_ID = "CANONICAL-DETERMINISTIC-OCCURRENCE-UNION-V1"
_TOL = 1e-6


def _interval(row: Mapping[str, object]) -> tuple[float, float, float]:
    start = float(row["start"])
    stop = float(row["stop"])
    resolution = float(row["resolution_seconds"])
    if stop <= start or resolution <= 0:
        raise ValueError("candidate interval must have positive duration/resolution")
    return start, stop, resolution


def _merge_intervals(
    rows: Sequence[tuple[float, float, float]], *, tolerance: float = 0.0
) -> list[dict[str, float]]:
    ordered = sorted(rows, key=lambda item: (item[0], item[1], item[2]))
    result: list[dict[str, float]] = []
    for start, stop, resolution in ordered:
        if not result or start > result[-1]["stop"] + tolerance + _TOL:
            result.append(
                {
                    "start": float(start),
                    "stop": float(stop),
                    "resolution_seconds": float(resolution),
                }
            )
            continue
        result[-1]["stop"] = max(float(result[-1]["stop"]), float(stop))
        result[-1]["resolution_seconds"] = max(
            float(result[-1]["resolution_seconds"]), float(resolution)
        )
    return result


def _deduplicated_occurrences(
    findings: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[tuple[float, float, float, str]] = []
    for finding in findings:
        span = finding["time_interval"]
        if not isinstance(span, Mapping):
            raise ValueError("present occurrence candidate lacks a physical interval")
        start, stop, resolution = _interval(span)
        rows.append((start, stop, resolution, str(finding["evidence_id"])))
    rows.sort(key=lambda item: (item[0], item[1], item[3]))

    clusters: list[dict[str, Any]] = []
    for start, stop, resolution, evidence_id in rows:
        if (
            not clusters
            or start
            > float(clusters[-1]["interval"]["stop"])
            + max(resolution, float(clusters[-1]["interval"]["resolution_seconds"]))
            + _TOL
        ):
            clusters.append(
                {
                    "interval": {
                        "start": start,
                        "stop": stop,
                        "resolution_seconds": resolution,
                    },
                    "evidence_ids": [evidence_id],
                }
            )
            continue
        cluster = clusters[-1]
        cluster["interval"]["stop"] = max(
            float(cluster["interval"]["stop"]), stop
        )
        cluster["interval"]["resolution_seconds"] = max(
            float(cluster["interval"]["resolution_seconds"]), resolution
        )
        cluster["evidence_ids"].append(evidence_id)

    result: list[dict[str, Any]] = []
    for cluster in clusters:
        evidence_ids = sorted(set(str(item) for item in cluster["evidence_ids"]))
        span = deepcopy(cluster["interval"])
        occurrence_id = "OCC-" + _canonical_sha256(
            {
                "method_id": DETERMINISTIC_EVENT_FINDINGS_V3_CANDIDATE_METHOD_ID,
                "term_id": _OCCURRENCE_TERM_ID,
                "interval": span,
                "evidence_ids": evidence_ids,
                "deduplication_policy_id": _DEDUPLICATION_POLICY_ID,
            }
        )[:24]
        result.append(
            {
                "occurrence_id": occurrence_id,
                "interval": span,
                "evidence_ids": evidence_ids,
            }
        )
    return sorted(
        result,
        key=lambda item: (
            float(item["interval"]["start"]),
            float(item["interval"]["stop"]),
            str(item["occurrence_id"]),
        ),
    )


def _incidence_category(rate_per_minute: float, count: int) -> str:
    if count == 0:
        return "none"
    if rate_per_minute >= 6.0 - _TOL:
        return "abundant"
    if rate_per_minute >= 1.0 - _TOL:
        return "frequent"
    if rate_per_minute >= 1.0 / 60.0 - _TOL:
        return "occasional"
    return "rare"


def _prevalence_category(proportion: float) -> str:
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


def _rhythm_periodicity_block(payload: Mapping[str, Any]) -> dict[str, Any]:
    rhythm_findings = [
        row for row in payload["findings"] if row["family"] == "rhythm"
    ]
    if len(rhythm_findings) != 1:
        raise ValueError("deterministic candidate slice expects one v2 rhythm atom")
    finding = rhythm_findings[0]
    if finding["term"]["term_id"] != _RHYTHMICITY_TERM_ID:
        raise ValueError("unexpected deterministic v2 rhythm term")
    common = {
        "term_ids": [str(finding["term"]["term_id"])],
        "finding_ids": [str(finding["evidence_id"])],
        "evaluation_opportunity_ids": [
            str(finding["evaluation_opportunity_id"])
        ],
        "capability_receipt_ids": [],
        "term_decision_receipt_ids": [],
    }
    if finding["status"] == "not_evaluable":
        rhythmicity = {
            "qualification_status": "not_evaluable",
            **common,
            "reason_codes": ["deterministic_rhythmicity_profile_not_evaluable"],
        }
    elif finding["status"] in {"present", "uncertain"}:
        rhythmicity = {
            "qualification_status": "candidate_only",
            **common,
            "reason_codes": [
                "deterministic_descriptor_candidate_not_clinically_qualified"
            ],
        }
    else:
        raise ValueError("deterministic rhythmicity atom cannot encode absence")
    return {
        "rhythmicity": rhythmicity,
        "periodicity": {
            "qualification_status": "not_evaluable",
            "term_ids": [],
            "finding_ids": [],
            "evaluation_opportunity_ids": [],
            "capability_receipt_ids": [],
            "term_decision_receipt_ids": [],
            "reason_codes": [_PERIODICITY_REASON],
        },
    }


def _limited_quantity_summary(
    *,
    findings: Sequence[Mapping[str, Any]],
    opportunity_ids: Sequence[str],
    scope: Mapping[str, object],
    reason: str,
) -> dict[str, Any]:
    evidence_ids = sorted(str(row["evidence_id"]) for row in findings)
    return {
        "summary_id": "QTY-" + _canonical_sha256(
            [_OCCURRENCE_TERM_ID, evidence_ids, reason]
        )[:24],
        "term_id": _OCCURRENCE_TERM_ID,
        "pattern_candidate_ids": [],
        "scope_interval": deepcopy(dict(scope)),
        "occurrence": {
            "status": "limited",
            "count": None,
            "evaluable_seconds": None,
            "rate_per_minute": None,
            "incidence_category": "indeterminate",
            "deduplicated_occurrences": [],
            "deduplication_policy_id": None,
            "deduplication_sha256": None,
            "supporting_evidence_ids": evidence_ids,
            "evaluation_opportunity_ids": list(opportunity_ids),
            "reason_codes": [reason],
        },
        "burden": {
            "status": "limited",
            "observed_seconds": None,
            "evaluable_seconds": None,
            "proportion": None,
            "prevalence_category": "indeterminate",
            "interval_union": [],
            "interval_union_policy_id": None,
            "interval_union_sha256": None,
            "supporting_evidence_ids": evidence_ids,
            "evaluation_opportunity_ids": list(opportunity_ids),
            "reason_codes": [reason],
        },
        "variability": {
            "status": "not_evaluable",
            "dimensions": [],
            "supporting_evidence_ids": [],
            "reason_codes": ["cross_occurrence_variability_not_measured"],
        },
        "reason_codes": [reason, "cross_occurrence_variability_not_measured"],
    }


def _occurrence_burden_variability_block(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    findings = [
        row
        for row in payload["findings"]
        if row["term"]["term_id"] == _OCCURRENCE_TERM_ID
        and row["status"] == "present"
    ]
    if not findings:
        return {
            "status": "not_evaluable",
            "analysis_scope": "signal_only_event_window",
            "summaries": [],
            "reason_codes": [
                "no_deduplicated_candidate_roster_not_a_qualified_absence"
            ],
        }

    opportunity_map = {
        str(row["evaluation_opportunity_id"]): row
        for row in payload["evaluation_opportunities"]
    }
    opportunity_ids = sorted(
        {
            str(row["evaluation_opportunity_id"])
            for row in findings
        }
    )
    opportunities = [opportunity_map[item] for item in opportunity_ids]
    finding_intervals = [
        _interval(row["time_interval"])
        for row in findings
        if isinstance(row["time_interval"], Mapping)
    ]
    if len(finding_intervals) != len(findings):
        raise ValueError("present deterministic occurrence lacks an interval")
    fallback_scope = {
        "start": min(item[0] for item in finding_intervals),
        "stop": max(item[1] for item in finding_intervals),
        "resolution_seconds": max(item[2] for item in finding_intervals),
    }
    physical_opportunities = all(
        row["status"] == "sufficient"
        and isinstance(row["interval"], Mapping)
        and math.isclose(float(row["usable_fraction"]), 1.0, abs_tol=_TOL)
        for row in opportunities
    )
    if not physical_opportunities:
        summary = _limited_quantity_summary(
            findings=findings,
            opportunity_ids=opportunity_ids,
            scope=fallback_scope,
            reason="complete_evaluable_time_denominator_unavailable",
        )
        return {
            "status": "limited",
            "analysis_scope": "signal_only_event_window",
            "summaries": [summary],
            "reason_codes": ["complete_evaluable_time_denominator_unavailable"],
        }

    opportunity_intervals = [
        _interval(row["interval"]) for row in opportunities
    ]
    opportunity_union = _merge_intervals(opportunity_intervals)
    evaluable_seconds = sum(
        float(row["stop"]) - float(row["start"])
        for row in opportunity_union
    )
    scope = {
        "start": min(float(row["start"]) for row in opportunity_union),
        "stop": max(float(row["stop"]) for row in opportunity_union),
        "resolution_seconds": max(
            float(row["resolution_seconds"]) for row in opportunity_union
        ),
    }
    roster = _deduplicated_occurrences(findings)
    evidence_ids = sorted(
        {
            str(evidence_id)
            for row in roster
            for evidence_id in row["evidence_ids"]
        }
    )
    count = len(roster)
    rate = count * 60.0 / evaluable_seconds
    interval_union = [deepcopy(row["interval"]) for row in roster]
    observed_seconds = sum(
        float(row["stop"]) - float(row["start"])
        for row in interval_union
    )
    proportion = observed_seconds / evaluable_seconds
    occurrence_sha256 = event_occurrence_roster_sha256_v3(
        term_id=_OCCURRENCE_TERM_ID,
        scope_interval=scope,
        evaluable_seconds=evaluable_seconds,
        deduplication_policy_id=_DEDUPLICATION_POLICY_ID,
        deduplicated_occurrences=roster,
    )
    burden_sha256 = event_burden_interval_union_sha256_v3(
        scope_interval=scope,
        evaluable_seconds=evaluable_seconds,
        interval_union_policy_id=_INTERVAL_UNION_POLICY_ID,
        interval_union=interval_union,
    )
    summary = {
        "summary_id": "QTY-" + _canonical_sha256(
            [_OCCURRENCE_TERM_ID, roster, opportunity_ids]
        )[:24],
        "term_id": _OCCURRENCE_TERM_ID,
        "pattern_candidate_ids": [],
        "scope_interval": scope,
        "occurrence": {
            "status": "measured",
            "count": count,
            "evaluable_seconds": evaluable_seconds,
            "rate_per_minute": rate,
            "incidence_category": _incidence_category(rate, count),
            "deduplicated_occurrences": roster,
            "deduplication_policy_id": _DEDUPLICATION_POLICY_ID,
            "deduplication_sha256": occurrence_sha256,
            "supporting_evidence_ids": evidence_ids,
            "evaluation_opportunity_ids": opportunity_ids,
            "reason_codes": [],
        },
        "burden": {
            "status": "measured",
            "observed_seconds": observed_seconds,
            "evaluable_seconds": evaluable_seconds,
            "proportion": proportion,
            "prevalence_category": _prevalence_category(proportion),
            "interval_union": interval_union,
            "interval_union_policy_id": _INTERVAL_UNION_POLICY_ID,
            "interval_union_sha256": burden_sha256,
            "supporting_evidence_ids": evidence_ids,
            "evaluation_opportunity_ids": opportunity_ids,
            "reason_codes": [],
        },
        "variability": {
            "status": "not_evaluable",
            "dimensions": [],
            "supporting_evidence_ids": [],
            "reason_codes": [
                "v2_has_no_replayable_cross_occurrence_variability_measurement"
            ],
        },
        "reason_codes": [
            "v2_has_no_replayable_cross_occurrence_variability_measurement"
        ],
    }
    return {
        "status": "limited",
        "analysis_scope": "signal_only_event_window",
        "summaries": [summary],
        "reason_codes": [
            "cross_occurrence_variability_not_evaluable"
        ],
    }


def _source_acquisition(payload: Mapping[str, Any]) -> tuple[
    list[str], list[float] | None, float
]:
    bindings = [
        measurement["source_binding"]
        for finding in payload["findings"]
        for measurement in finding["measurements"]
    ]
    if not bindings:
        return [], None, float(payload["coordinates"]["model_sample_rate_hz"])
    best = max(
        bindings,
        key=lambda row: float(row["effective_bandwidth_hz"][1]),
    )
    view_ids = sorted({str(row["source_view_id"]) for row in bindings})
    bandwidth = [float(item) for item in best["effective_bandwidth_hz"]]
    return view_ids, bandwidth, float(
        payload["coordinates"]["model_sample_rate_hz"]
    )


def _acquisition_capabilities_block(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    view_ids, bandwidth, sample_rate = _source_acquisition(payload)
    upper = None if bandwidth is None else float(bandwidth[1])
    high_frequency_finding = next(
        row
        for row in payload["findings"]
        if row["term"]["term_id"] == _HFO_TERM_ID
    )
    high_frequency_opportunity = str(
        high_frequency_finding["evaluation_opportunity_id"]
    )

    if (
        upper is None
        or upper <= 80.0 + _TOL
        or sample_rate + _TOL < 2.0 * upper
    ):
        hfo_status = "not_evaluable"
        hfo_reasons = ["effective_bandwidth_does_not_extend_above_80_hz"]
    else:
        hfo_status = "limited"
        hfo_reasons = ["hfo_candidate_producer_not_clinically_qualified"]
    if upper is not None and upper > 13.0 + _TOL:
        lvfa_status = "limited"
        lvfa_reasons = ["term_specific_lvfa_acquisition_rule_not_frozen"]
    else:
        lvfa_status = "not_evaluable"
        lvfa_reasons = ["fast_activity_bandwidth_unavailable"]
    capabilities = [
        {
            "capability_id": "ACQ-" + _canonical_sha256(
                [_HFO_TERM_ID, bandwidth, sample_rate, hfo_status]
            )[:24],
            "term_id": _HFO_TERM_ID,
            "feature_class": "high_frequency_oscillation",
            "status": hfo_status,
            "source_view_ids": view_ids,
            "effective_bandwidth_hz": bandwidth,
            "sample_rate_hz": sample_rate,
            "coupling": "unknown",
            "evaluation_opportunity_ids": [high_frequency_opportunity],
            "reason_codes": hfo_reasons,
        },
        {
            "capability_id": "ACQ-" + _canonical_sha256(
                ["dc_shift", bandwidth, sample_rate]
            )[:24],
            "term_id": "dc_shift",
            "feature_class": "dc_shift",
            "status": "not_evaluable",
            "source_view_ids": view_ids,
            "effective_bandwidth_hz": bandwidth,
            "sample_rate_hz": sample_rate,
            "coupling": "unknown",
            "evaluation_opportunity_ids": [],
            "reason_codes": ["dc_coupling_not_demonstrated_by_v2_measurements"],
        },
        {
            "capability_id": "ACQ-" + _canonical_sha256(
                ["low_voltage_fast_activity", bandwidth, sample_rate, lvfa_status]
            )[:24],
            "term_id": "low_voltage_fast_activity",
            "feature_class": "low_voltage_fast_activity",
            "status": lvfa_status,
            "source_view_ids": view_ids,
            "effective_bandwidth_hz": bandwidth,
            "sample_rate_hz": sample_rate,
            "coupling": "unknown",
            "evaluation_opportunity_ids": [],
            "reason_codes": lvfa_reasons,
        },
    ]
    overall = (
        "limited"
        if any(row["status"] == "limited" for row in capabilities)
        else "not_evaluable"
    )
    return {
        "status": overall,
        "capabilities": capabilities,
        "reason_codes": [
            "term_specific_acquisition_and_clinical_qualification_incomplete"
        ],
    }


def _competing_hypotheses_and_outcome(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    onset_evidence = sorted(
        str(row["evidence_id"])
        for row in payload["findings"]
        if row["status"] == "present"
        and row["intrinsic_evidence_role"] == "onset_eligible"
    )
    if onset_evidence:
        cerebral_id = "H-CEREBRAL-ICTAL-" + _canonical_sha256(
            [payload["event_id"], onset_evidence]
        )[:16]
        uncertain_id = "H-UNCERTAIN-PATTERN-" + _canonical_sha256(
            [payload["event_id"], "alternative"]
        )[:16]
        competing = {
            "status": "available",
            "selected_hypothesis_id": cerebral_id,
            "hypotheses": [
                {
                    "hypothesis_id": cerebral_id,
                    "category": "cerebral_ictal",
                    "term_id": "deterministic_cerebral_ictal_signal_candidate",
                    "disposition": "supported",
                    "rank": 1,
                    "supporting_evidence_ids": onset_evidence,
                    "contradictory_evidence_ids": [],
                    "onset_claim_eligible": True,
                    "reason_codes": [
                        "research_signal_hypothesis_not_clinical_seizure_qualification"
                    ],
                },
                {
                    "hypothesis_id": uncertain_id,
                    "category": "uncertain_pattern",
                    "term_id": "alternative_signal_pattern_unresolved",
                    "disposition": "possible",
                    "rank": 2,
                    "supporting_evidence_ids": [],
                    "contradictory_evidence_ids": [],
                    "onset_claim_eligible": False,
                    "reason_codes": [
                        "alternative_signal_explanations_not_independently_modelled"
                    ],
                },
            ],
            "reason_codes": [],
        }
        competing_ids = [cerebral_id, uncertain_id]
    else:
        uncertain_id = "H-UNCERTAIN-PATTERN-" + _canonical_sha256(
            [payload["event_id"], "no-causal-onset"]
        )[:16]
        competing = {
            "status": "limited",
            "selected_hypothesis_id": uncertain_id,
            "hypotheses": [
                {
                    "hypothesis_id": uncertain_id,
                    "category": "uncertain_pattern",
                    "term_id": "causal_ictal_candidate_not_resolved",
                    "disposition": "possible",
                    "rank": 1,
                    "supporting_evidence_ids": [],
                    "contradictory_evidence_ids": [],
                    "onset_claim_eligible": False,
                    "reason_codes": ["future_free_causal_onset_evidence_unavailable"],
                }
            ],
            "reason_codes": ["future_free_causal_onset_evidence_unavailable"],
        }
        competing_ids = [uncertain_id]

    qualification = payload["event_qualification"]
    status = str(qualification["status"])
    if status == "unqualified_candidate":
        outcome = "candidate_only"
        reasons = ["deterministic_signal_candidate_not_clinically_qualified"]
    elif status == "not_evaluable":
        outcome = "not_possible_to_determine"
        reasons = ["future_free_qualified_event_evidence_unavailable"]
    else:
        raise ValueError(
            "deterministic v3 candidate slice cannot consume a qualified v2 event"
        )
    event_outcome = {
        "outcome": outcome,
        "evidence_ids": [
            str(item) for item in qualification["supporting_evidence_ids"]
        ],
        "competing_hypothesis_ids": competing_ids,
        "artifact_interval_indices": [],
        "reason_codes": reasons,
    }
    return competing, event_outcome


def _append_candidate_provenance(payload: dict[str, Any]) -> None:
    model_ids = list(payload["provenance"]["model_ids"])
    if DETERMINISTIC_EVENT_FINDINGS_V3_CANDIDATE_METHOD_ID not in model_ids:
        model_ids.append(DETERMINISTIC_EVENT_FINDINGS_V3_CANDIDATE_METHOD_ID)
    payload["provenance"]["model_ids"] = model_ids
    limitation_rows = [
        {
            "code": "v3_candidate_heads_not_clinically_qualified",
            "scope": "finding",
            "text_zh": "v3 新增头仅为可重放信号候选，不构成临床节律、周期性或发作资格结论。",
        },
        {
            "code": "periodicity_withheld_without_interval_series",
            "scope": "finding",
            "text_zh": "未记录逐次放电间隔序列，因此不评价周期性。",
        },
        {
            "code": "event_window_quantity_not_record_burden",
            "scope": "finding",
            "text_zh": "候选出现次数与时长占比仅适用于当前事件窗，不代表整段记录负担。",
        },
    ]
    existing = {str(row["code"]) for row in payload["limitations"]}
    payload["limitations"].extend(
        row for row in limitation_rows if row["code"] not in existing
    )


def build_deterministic_event_eeg_findings_v3_candidate_from_v2(
    value: object,
    *,
    trusted_producer_receipts: Mapping[str, Mapping[str, object]],
    trusted_registry_bindings: Mapping[str, Mapping[str, object]],
) -> dict[str, Any]:
    """Extend one trusted native deterministic-v2 ledger into native v3."""

    base = validate_event_eeg_findings_v2_payload(
        value,
        trusted_producer_receipts=trusted_producer_receipts,
        trusted_registry_bindings=trusted_registry_bindings,
    )
    if (
        DETERMINISTIC_EVENT_FINDINGS_V2_METHOD_ID
        not in base["provenance"]["model_ids"]
        or base["migration"] is not None
        or base["capability_qualification_receipts"]
        or base["sensitivity_receipts"]
        or base["term_decision_receipts"]
        or base["event_qualification"]["status"]
        not in {"unqualified_candidate", "not_evaluable"}
    ):
        raise ValueError(
            "native v3 candidate slice requires an unqualified deterministic-v2 ledger"
        )

    result: dict[str, Any] = deepcopy(base)
    result["schema_version"] = "event_eeg_findings_v3"
    _append_candidate_provenance(result)
    result["occurrence_burden_variability"] = (
        _occurrence_burden_variability_block(result)
    )
    result["rhythm_periodicity_qualification"] = (
        _rhythm_periodicity_block(result)
    )
    result["acquisition_capabilities"] = _acquisition_capabilities_block(result)
    competing, outcome = _competing_hypotheses_and_outcome(result)
    result["competing_hypotheses"] = competing
    result["event_outcome"] = outcome
    result["v3_migration"] = None
    return validate_event_eeg_findings_v3_payload(
        result,
        trusted_producer_receipts=trusted_producer_receipts,
        trusted_registry_bindings=trusted_registry_bindings,
    )


def produce_deterministic_event_eeg_findings_v3_candidate(
    *,
    event_id: str,
    adaptive_search_receipt: object,
    adaptive_window_receipt: object,
    canonical_receipt: object,
    views: Sequence[DeterministicViewInput],
    trusted_parent_views: Mapping[str, Mapping[str, object]] | None = None,
    trusted_registry_bindings: Mapping[str, Mapping[str, object]] | None = None,
    policy: DeterministicEventFindingsPolicy = DEFAULT_DETERMINISTIC_EVENT_FINDINGS_POLICY,
) -> dict[str, Any]:
    """Produce a complete native-v3 candidate ledger from replayable EEG."""

    host_registries = (
        deepcopy(DEFAULT_EVENT_FINDINGS_V2_REGISTRY_BINDINGS)
        if trusted_registry_bindings is None
        else trusted_registry_bindings
    )
    base = produce_deterministic_event_eeg_findings_v2(
        event_id=event_id,
        adaptive_search_receipt=adaptive_search_receipt,
        adaptive_window_receipt=adaptive_window_receipt,
        canonical_receipt=canonical_receipt,
        views=views,
        trusted_parent_views=trusted_parent_views,
        trusted_registry_bindings=host_registries,
        policy=policy,
    )
    trusted_producers = {
        str(row["receipt_id"]): deepcopy(row)
        for row in base["producer_receipts"]
    }
    return build_deterministic_event_eeg_findings_v3_candidate_from_v2(
        base,
        trusted_producer_receipts=trusted_producers,
        trusted_registry_bindings=host_registries,
    )


__all__ = [
    "DETERMINISTIC_EVENT_FINDINGS_V3_CANDIDATE_METHOD_ID",
    "DETERMINISTIC_EVENT_FINDINGS_V3_CANDIDATE_POLICY_ID",
    "build_deterministic_event_eeg_findings_v3_candidate_from_v2",
    "produce_deterministic_event_eeg_findings_v3_candidate",
]
