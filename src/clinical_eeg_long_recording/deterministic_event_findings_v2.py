"""Native deterministic ``event_eeg_findings_v2`` producer.

The producer consumes the same immutable canonical EEG, adaptive-search,
adaptive-window and task-view receipts as the executable v1 baseline, but it
builds the v2 evidence graph directly.  It never serializes a v1 payload and
never invokes the v1-to-v2 migrator.

Only replayable measurements and explicitly named research candidates are
emitted.  Clinical terms, negative assertions, event qualification and a
cortical SOZ claim remain closed unless separately qualified by trusted
patient-disjoint receipts (which this deterministic producer does not own).
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .adaptive_event_window import validate_adaptive_event_analysis_window
from .adaptive_search import validate_adaptive_search_receipt
from .canonical_adaptive_binding import (
    validate_canonical_adaptive_binding_against_receipt,
)
from .canonical_signal_views import (
    ONSET_FIR_CLINICAL_ADMISSION_UNQUALIFIED_REASON_CODE,
    ONSET_FIR_RESPONSE_UNQUALIFIED_REASON_CODE,
    recording_seconds_to_canonical_sample_index,
    recording_seconds_to_view_tensor_index,
    validate_canonical_signal_receipt,
    view_tensor_index_to_recording_seconds,
)
from .deterministic_event_findings import (
    DEFAULT_DETERMINISTIC_EVENT_FINDINGS_POLICY,
    DeterministicEventFindingsPolicy,
    DeterministicViewInput,
    _analysis_reference,
    _canonical_sha256,
    _change_scores,
    _family_eligible,
    _feature_grid,
    _first_sustained_changes,
    _identifier,
    _lead_endpoints,
    _prepare_views,
    _quality_invalid_samples,
    _unit_laterality,
    _unit_region,
)
from .event_findings_v2_validation import validate_event_eeg_findings_v2_payload


DETERMINISTIC_EVENT_FINDINGS_V2_METHOD_ID = "DETERMINISTIC-EVENT-FINDINGS-V2"
DETERMINISTIC_EVENT_HYPOTHESIS_V2_METHOD_ID = (
    "DETERMINISTIC-EVENT-HYPOTHESIS-RELATIONS-V2"
)
RAW_SAMPLE_DEPENDENCY_SCHEMA_VERSION = "clinical_eeg_raw_sample_dependency_v1"
_RAW_DEPENDENCY_ID_DOMAIN = "clinical-eeg-raw-sample-dependency-id-v1"
_RAW_DEPENDENCY_DIGEST_DOMAIN = "clinical-eeg-raw-sample-dependency-digest-v1"
_RAW_SUPPORT_COMPONENT_ROLE_ORDER = {
    "baseline_reference": 0,
    "reported_evidence_interval": 1,
    "sustained_confirmation": 2,
}

_FAMILIES = (
    "quality",
    "spectral",
    "rhythm",
    "morphology",
    "evolution",
    "spatial_field",
    "spatial_recruitment",
    "termination_recovery",
    "high_frequency",
)

_TERM_CATALOG: dict[str, tuple[str, str]] = {
    "deterministic_signal_usable_fraction": (
        "quality",
        "RULE-QUALITY-USABLE-FRACTION-V2",
    ),
    "deterministic_background_spectral_profile": (
        "spectral",
        "RULE-BACKGROUND-SPECTRAL-PROFILE-V2",
    ),
    "deterministic_event_spectral_profile": (
        "spectral",
        "RULE-EVENT-SPECTRAL-PROFILE-V2",
    ),
    "deterministic_event_rhythmicity_profile": (
        "rhythm",
        "RULE-EVENT-RHYTHMICITY-PROFILE-V2",
    ),
    "deterministic_morphology_candidate": (
        "morphology",
        "RULE-MORPHOLOGY-WITHHOLD-V2",
    ),
    "deterministic_multifeature_change_point_candidate": (
        "evolution",
        "RULE-MULTIFEATURE-CHANGE-CANDIDATE-V2",
    ),
    "reference_specific_spatial_change_candidate": (
        "spatial_field",
        "RULE-REFERENCE-SPATIAL-CHANGE-CANDIDATE-V2",
    ),
    "deterministic_later_involvement_candidate": (
        "spatial_recruitment",
        "RULE-LATER-INVOLVEMENT-CANDIDATE-V2",
    ),
    "deterministic_recovery_context_profile": (
        "termination_recovery",
        "RULE-RECOVERY-CONTEXT-PROFILE-V2",
    ),
    "deterministic_high_frequency_candidate": (
        "high_frequency",
        "RULE-HIGH-FREQUENCY-WITHHOLD-V2",
    ),
}

_UNIT_CATALOG = {
    "Hz": "frequency_hertz",
    "uV": "amplitude_microvolts",
    "ratio": "dimensionless_ratio",
    "unitless": "dimensionless_index",
    "robust_z": "robust_standardized_change",
    "s": "recording_relative_seconds",
}

DEFAULT_EVENT_FINDINGS_V2_REGISTRY_BINDINGS: dict[str, dict[str, Any]] = {
    "term_registry": {
        "registry_id": "DETERMINISTIC-EEG-ATOMIC-TERM-REGISTRY",
        "version": "2.0.0",
        "registry_sha256": _canonical_sha256(_TERM_CATALOG),
        "trust_status": "host_trusted",
    },
    "unit_registry": {
        "registry_id": "DETERMINISTIC-EEG-UNIT-REGISTRY",
        "version": "2.0.0",
        "registry_sha256": _canonical_sha256(_UNIT_CATALOG),
        "trust_status": "host_trusted",
    },
}


def _producer_policy_sha256(policy: DeterministicEventFindingsPolicy) -> str:
    return _canonical_sha256(
        {
            "method_id": DETERMINISTIC_EVENT_FINDINGS_V2_METHOD_ID,
            "numeric_policy": policy.to_dict(),
            "view_roles": {
                "derivation_source": "validated_signal_view_temporal_evidence_v1",
                "onset_causal": {
                    "future_sample_access": False,
                    "onset_evidence_authorized": True,
                    "dependency_policy": "past_and_present_only",
                    "latest_raw_support_offset_samples_max": 0,
                    "raw_support_end_policy": (
                        "at_or_before_unshifted_evidence_sample_v1"
                    ),
                },
                "context_offline": {
                    "future_sample_access": True,
                    "onset_evidence_authorized": False,
                },
            },
            "negative_assertion_policy": "never_without_sensitivity_receipt",
            "clinical_qualification_policy": "withhold",
        }
    )


def _view_role(view: Any) -> str:
    """Classify a validated view from its effective temporal receipt.

    A spatial-reference transform is instantaneous (`phase_policy=none`) but
    its carrier may be causal or bidirectional.  Looking only at the child
    transform therefore loses the parent dependency and can promote offline
    context into onset evidence.  `validate_signal_view_receipt` derives and
    hash-binds the effective contract through trusted parent receipts, so this
    producer consumes that contract fail closed.
    """

    receipt = view.receipt
    temporal = receipt.get("temporal_evidence")
    if not isinstance(temporal, Mapping):
        return "unknown"
    task_role = str(receipt.get("task_role", ""))
    future = temporal.get("future_sample_access")
    onset_authorized = temporal.get("onset_evidence_authorized")
    dependency = temporal.get("dependency_policy")
    raw_offset = temporal.get("latest_raw_support_offset_samples")
    raw_policy = temporal.get("raw_support_end_policy")

    if onset_authorized is True:
        if (
            future is False
            and dependency == "past_and_present_only"
            and isinstance(raw_offset, int)
            and not isinstance(raw_offset, bool)
            and raw_offset <= 0
            and raw_policy == "at_or_before_unshifted_evidence_sample_v1"
            and task_role in {"onset_causal", "spatial_reference"}
        ):
            return "onset_causal"
        return "unknown"
    if onset_authorized is not False:
        return "unknown"
    if (
        task_role in {"onset_causal", "spatial_reference"}
        and future is False
        and dependency == "past_and_present_only"
        and isinstance(raw_offset, int)
        and not isinstance(raw_offset, bool)
        and raw_offset <= 0
        and raw_policy == "at_or_before_unshifted_evidence_sample_v1"
        and any(
            reason in temporal.get("authorization_reason_codes", ())
            for reason in (
                ONSET_FIR_RESPONSE_UNQUALIFIED_REASON_CODE,
                ONSET_FIR_CLINICAL_ADMISSION_UNQUALIFIED_REASON_CODE,
            )
        )
    ):
        return "onset_causal_unqualified"
    if future is True:
        if (
            dependency == "bidirectional_or_unknown"
            and raw_offset is None
            and raw_policy == "future_dependent_context_not_onset_eligible_v1"
            and task_role
            in {"findings_clinical", "context_offline", "spatial_reference"}
        ):
            return "context_offline"
        return "unknown"
    if (
        future is False
        and dependency == "instantaneous"
        and isinstance(raw_offset, int)
        and not isinstance(raw_offset, bool)
        and raw_offset <= 0
        and raw_policy == "at_or_before_unshifted_evidence_sample_v1"
        and task_role
        in {
            "findings_native",
            "findings_native_morphology",
            "spatial_reference",
        }
    ):
        return "canonical_physical_evidence"
    return "unknown"


def _output_catalog(view: Any) -> dict[str, Mapping[str, Any]]:
    return {str(row["unit_id"]): row for row in view.receipt["output_units"]}


def _unit_is_eligible(view: Any, unit_id: str, family: str) -> bool:
    output = _output_catalog(view)[unit_id]
    return bool(
        _view_role(view) != "unknown"
        and output["observed"]
        and not output["imputed"]
        and output["evidence_eligible"]
        and _family_eligible(output, family)
    )


def _view_bandwidth(view: Any, unit_ids: Sequence[str]) -> list[float]:
    catalog = _output_catalog(view)
    bands = [catalog[item]["effective_bandwidth_hz"] for item in unit_ids]
    lower = max(float(item[0]) for item in bands)
    upper = min(float(item[1]) for item in bands)
    if upper <= lower:
        raise ValueError("selected units have no common effective bandwidth")
    return [lower, upper]


def _mask_component_sha256(view: Any, key: str) -> str:
    return _canonical_sha256(
        {
            "view_receipt_id": view.receipt["view_receipt_id"],
            key: view.receipt["masks"][key],
        }
    )


def _finalize_raw_sample_dependency(body: Mapping[str, Any]) -> dict[str, Any]:
    """Content-address one atom-local raw dependency sidecar."""

    result = deepcopy(dict(body))
    identifier_source = deepcopy(result)
    identifier_source.pop("dependency_id", None)
    identifier_source.pop("dependency_sha256", None)
    result["dependency_id"] = (
        "RAWDEP-"
        + _canonical_sha256(
            {
                "domain": _RAW_DEPENDENCY_ID_DOMAIN,
                "dependency": identifier_source,
            }
        )[:24]
    )
    digest_source = deepcopy(result)
    digest_source["dependency_sha256"] = "CONTENT-ADDRESS-PENDING"
    result["dependency_sha256"] = _canonical_sha256(
        {
            "domain": _RAW_DEPENDENCY_DIGEST_DOMAIN,
            "dependency": digest_source,
        }
    )
    return result


def _term_ref(term_id: str) -> dict[str, str]:
    if term_id not in _TERM_CATALOG:
        raise ValueError(f"unregistered deterministic v2 term: {term_id}")
    return {
        "term_id": term_id,
        "ontology_id": "EEG-SIGNAL-MEASUREMENT-ATOMS-V2",
        "source_id": "DETERMINISTIC-EEG-ATOMIC-TERM-REGISTRY",
        "source_version": "2.0.0",
        "operational_rule_id": _TERM_CATALOG[term_id][1],
    }


def _observed_span(
    interval: tuple[float, float], resolution: float
) -> dict[str, float]:
    return {
        "start": float(interval[0]),
        "stop": float(interval[1]),
        "resolution_seconds": float(resolution),
    }


def _time_interval(lower: float, upper: float, resolution: float) -> dict[str, Any]:
    return {
        "lower": float(lower),
        "median": float((lower + upper) / 2.0),
        "upper": float(upper),
        "resolution_seconds": float(resolution),
        "calibration_status": "uncalibrated",
    }


def _overlap(left: tuple[float, float], right: tuple[float, float]) -> float:
    return max(0.0, min(left[1], right[1]) - max(left[0], right[0]))


def _state_membership(
    interval: tuple[float, float] | None,
    state_spans: Mapping[str, tuple[float, float]],
    *,
    force_zero: bool = False,
) -> dict[str, float]:
    result = {name: 0.0 for name in ("S0", "S1", "S2", "S3")}
    if interval is None or force_zero or not state_spans:
        return result
    for name, span in state_spans.items():
        result[name] = _overlap(interval, span)
    total = sum(result.values())
    if total <= 1e-9:
        return {name: 0.0 for name in result}
    return {name: value / total for name, value in result.items()}


def _state_contract(
    *,
    final_interval: tuple[float, float],
    onset_interval: tuple[float, float] | None,
    offset_anchor: float | None,
    resolution: float,
    receipt_id: str,
) -> tuple[list[dict[str, Any]], dict[str, tuple[float, float]]]:
    if onset_interval is None:
        return [], {}
    start, stop = final_interval
    onset = min(stop, max(start, (onset_interval[0] + onset_interval[1]) / 2.0))
    if offset_anchor is None or offset_anchor <= onset + resolution:
        offset = stop
        has_recovery = False
    else:
        offset = min(stop, max(onset, float(offset_anchor)))
        has_recovery = offset < stop - 1e-9
    early_stop = min(offset, onset + max(2.0, min(10.0, 0.25 * (offset - onset))))
    if early_stop <= onset + 1e-9:
        early_stop = offset

    candidates = (
        ("S0", start, onset),
        ("S1", onset, early_stop),
        ("S2", early_stop, offset),
        ("S3", offset, stop) if has_recovery else ("S3", stop, stop),
    )
    spans: dict[str, tuple[float, float]] = {}
    segments: list[dict[str, Any]] = []
    for name, left, right in candidates:
        if right <= left + 1e-9:
            continue
        span = (float(left), float(right))
        spans[name] = span
        posterior = {key: 0.0 for key in ("S0", "S1", "S2", "S3")}
        posterior[name] = 1.0
        segments.append(
            {
                "segment_id": f"STATE-{name}-{len(segments) + 1}",
                "interval": _observed_span(span, resolution),
                "posterior": posterior,
                "source_receipt_id": receipt_id,
            }
        )
    return segments, spans


def _montage(
    view: Any,
) -> tuple[dict[str, Any], dict[str, bool], dict[str, str], dict[str, str]]:
    input_units: list[dict[str, Any]] = []
    lead_definitions: list[dict[str, str]] = []
    electrode_ids: set[str] = set()
    eligibility: dict[str, bool] = {}
    laterality: dict[str, str] = {}
    region: dict[str, str] = {}
    for unit_id, unit_type, output in zip(
        view.unit_ids, view.unit_types, view.receipt["output_units"]
    ):
        if unit_type not in {"lead", "electrode"}:
            raise ValueError("native v2 producer accepts only electrode or lead units")
        if output["observed"] and not output["imputed"]:
            observation = "observed"
            reasons: list[str] = []
            imputation_id = None
        elif output["imputed"]:
            observation = "imputed"
            reasons = ["task_view_unit_imputed"]
            imputation_id = (
                "IMPUTE-"
                + _canonical_sha256([view.receipt["view_receipt_id"], unit_id])[:20]
            )
        else:
            observation = "missing"
            reasons = ["task_view_unit_not_observed"]
            imputation_id = None
        eligible = bool(
            observation == "observed"
            and output["evidence_eligible"]
            and _view_role(view) != "unknown"
        )
        eligibility[unit_id] = eligible
        unit_laterality = _unit_laterality(unit_id, unit_type)
        unit_region = _unit_region(unit_id, unit_type, unit_laterality)
        laterality[unit_id] = unit_laterality
        region[unit_id] = unit_region
        if unit_type == "lead":
            anode, cathode = _lead_endpoints(unit_id)
            electrode_ids.update((anode, cathode))
            lead_definitions.append(
                {"lead_id": unit_id, "anode": anode, "cathode": cathode}
            )
            wire_type = "bipolar_lead"
        else:
            electrode_ids.add(unit_id)
            wire_type = "electrode"
        input_units.append(
            {
                "unit_id": unit_id,
                "unit_type": wire_type,
                "canonical_name": unit_id,
                "source_name": f"{view.receipt['view_id']}:{unit_id}",
                "observation_status": observation,
                "evidence_eligible": eligible,
                "missing_reason_codes": reasons,
                "imputation_receipt_id": imputation_id,
                "region": unit_region,
                "laterality": unit_laterality,
            }
        )
    reference = _analysis_reference(view)
    perturbations = (
        [reference] if reference in {"common_average", "bipolar", "laplacian"} else []
    )
    return (
        {
            "analysis_reference": reference,
            "input_units": input_units,
            "electrode_ids": sorted(electrode_ids),
            "lead_definitions": lead_definitions,
            "reference_perturbations_evaluated": perturbations,
        },
        eligibility,
        laterality,
        region,
    )


def _quality(
    view: Any,
    *,
    final_interval: tuple[float, float],
    montage_eligibility: Mapping[str, bool],
) -> tuple[list[dict[str, Any]], float, list[dict[str, Any]]]:
    start, stop = view.final_tensor_interval
    invalid = _quality_invalid_samples(view)
    per_unit: list[dict[str, Any]] = []
    fractions: list[float] = []
    for index, unit_id in enumerate(view.unit_ids):
        fraction = max(
            0.0,
            min(1.0, 1.0 - float(np.mean(invalid[index, start:stop]))),
        )
        fractions.append(fraction)
        if fraction >= 1.0 - 1e-9:
            status, reasons = "usable", []
        elif fraction > 0.0:
            status, reasons = "limited", ["quality_mask_partial_coverage"]
        else:
            status, reasons = "unusable", ["no_usable_samples"]
        per_unit.append(
            {
                "unit_id": unit_id,
                "usable_fraction": fraction,
                "status": status,
                "evidence_eligible": bool(
                    montage_eligibility[unit_id] and fraction > 0
                ),
                "reason_codes": reasons,
            }
        )

    artifacts: list[dict[str, Any]] = []
    for row in view.receipt["masks"]["quality_invalid_intervals"]:
        left, right = (int(item) for item in row["tensor_sample_interval"])
        interval = (
            max(
                final_interval[0],
                view_tensor_index_to_recording_seconds(
                    view.receipt, tensor_sample_index=left
                ),
            ),
            min(
                final_interval[1],
                view_tensor_index_to_recording_seconds(
                    view.receipt, tensor_sample_index=right
                ),
            ),
        )
        if interval[1] <= interval[0]:
            continue
        reason_text = " ".join(str(item).lower() for item in row["reason_codes"])
        artifact_type = next(
            (
                name
                for name in ("flat", "clipping", "step", "line_noise")
                if name in reason_text
            ),
            "other",
        )
        artifacts.append(
            {
                "interval": [interval[0], interval[1]],
                "artifact_type": artifact_type,
                "affected_unit_ids": [str(row["unit_id"])],
                "assertion_level": "measured",
            }
        )
    usable = float(sum(fractions) / len(fractions))
    return per_unit, usable, artifacts


class _EvidenceBuilder:
    def __init__(
        self,
        *,
        event_id: str,
        canonical_receipt: Mapping[str, Any],
        policy_sha256: str,
        protection_zone: tuple[float, float],
        protection_zone_id: str,
        state_spans: Mapping[str, tuple[float, float]],
        montage_eligibility: Mapping[str, bool],
        resolution: float,
    ) -> None:
        self.event_id = event_id
        self.canonical = deepcopy(dict(canonical_receipt))
        self.canonical_sha256 = str(self.canonical["source_signal_sha256"])
        self.policy_sha256 = policy_sha256
        self.protection_zone = protection_zone
        self.protection_zone_id = protection_zone_id
        self.state_spans = state_spans
        self.montage_eligibility = montage_eligibility
        self.resolution = resolution
        self.opportunities: list[dict[str, Any]] = []
        self.findings: list[dict[str, Any]] = []
        self.waveforms: list[dict[str, Any]] = []
        self._opportunity_keys: dict[tuple[Any, ...], str] = {}
        self._waveform_keys: dict[tuple[Any, ...], str] = {}
        self._waveform_dependency_by_id: dict[str, str] = {}
        self._measurement_counter = 0

    def raw_sample_dependency(
        self,
        *,
        view: Any,
        unit_ids: Sequence[str],
        interval: tuple[float, float],
        tensor_interval: tuple[int, int],
        support_components: Sequence[tuple[str, tuple[float, float]]] = (),
        decision_available_recording_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Materialize the raw support of one measurement or waveform.

        Intervals are half open.  For a causal onset carrier the latest raw
        support edge is the *unshifted decision-available* edge.  The reported
        Finding interval remains the candidate waveform interval.  FIR group
        delay is retained only as processing latency, while sustained-
        confirmation latency is recorded separately; neither is subtracted
        from an evidence timestamp.  Bidirectional context uses a conservative
        whole-recording raw bound and is explicitly ineligible for onset support.
        """

        role = _view_role(view)
        if role == "unknown":
            raise ValueError("raw dependency requires a qualified task view")
        receipt = view.receipt
        temporal = receipt["temporal_evidence"]
        output_clock = receipt["transform_spec"]["output_clock"]
        output_num = int(output_clock["sampling_rate_numerator"])
        output_den = int(output_clock["sampling_rate_denominator"])
        future = bool(temporal["future_sample_access"])
        onset_authorized = bool(temporal["onset_evidence_authorized"])
        raw_policy = str(temporal["raw_support_end_policy"])
        onset_support = bool(
            role == "onset_causal"
            and not future
            and onset_authorized
            and temporal["dependency_policy"] == "past_and_present_only"
            and temporal["latest_raw_support_offset_samples"] is not None
            and int(temporal["latest_raw_support_offset_samples"]) <= 0
            and raw_policy == "at_or_before_unshifted_evidence_sample_v1"
        )
        if role == "onset_causal" and not onset_support:
            raise ValueError("causal onset view lacks a safe raw-support contract")

        output_catalog = _output_catalog(view)
        canonical_channel_ids = sorted(
            {
                str(channel_id)
                for unit_id in unit_ids
                for channel_id in output_catalog[unit_id][
                    "canonical_source_channel_ids"
                ]
            }
        )
        canonical_catalog = {
            str(row["channel_id"]): row for row in self.canonical["channels"]
        }
        if not canonical_channel_ids or any(
            channel_id not in canonical_catalog
            or not canonical_catalog[channel_id]["observed"]
            for channel_id in canonical_channel_ids
        ):
            raise ValueError("raw dependency lacks observed canonical source channels")

        components = sorted(
            {
                (str(role), float(span[0]), float(span[1]))
                for role, span in (
                    ("reported_evidence_interval", interval),
                    *support_components,
                )
            },
            key=lambda item: (
                _RAW_SUPPORT_COMPONENT_ROLE_ORDER.get(item[0], 99),
                item[1],
                item[2],
            ),
        )
        if any(
            role not in _RAW_SUPPORT_COMPONENT_ROLE_ORDER for role, _, _ in components
        ):
            raise ValueError("raw dependency support component role is unsupported")
        if any(stop <= start for _, start, stop in components):
            raise ValueError("raw dependency support component is empty")
        decision_available = (
            float(interval[1])
            if decision_available_recording_seconds is None
            else float(decision_available_recording_seconds)
        )
        if decision_available < float(interval[1]) - 1e-9:
            raise ValueError("decision availability cannot precede reported evidence")
        confirmation_latency_seconds = max(0.0, decision_available - float(interval[1]))
        confirmation_latency_samples = (
            confirmation_latency_seconds * output_num / output_den
        )
        confirmation_policy = (
            "sustained_observation_no_timestamp_advance_v1"
            if confirmation_latency_seconds > 1e-9
            else "none"
        )
        if future:
            dependency_status = "conservative_future_dependent_recording_bound"
            raw_start_seconds = 0.0
            raw_stop_seconds = float(self.canonical["recording_duration_seconds"])
        elif temporal["dependency_policy"] == "instantaneous":
            dependency_status = "exact_instantaneous"
            raw_start_seconds = min(item[1] for item in components)
            raw_stop_seconds = max(item[2] for item in components)
        else:
            dependency_status = "bounded_past_and_present"
            raw_start_seconds = max(
                0.0,
                min(item[1] for item in components)
                - float(temporal["warm_up_recording_seconds"]),
            )
            # Do not subtract group delay here.  The evidence clock is never
            # advanced; the half-open raw stop remains at/before its unshifted
            # evidence stop.
            raw_stop_seconds = max(item[2] for item in components)
            if onset_support and raw_stop_seconds > decision_available + 1e-9:
                raise ValueError(
                    "causal support component extends after unshifted decision availability"
                )

        raw_intervals: list[dict[str, Any]] = []
        for channel_id in canonical_channel_ids:
            channel = canonical_catalog[channel_id]
            reported_evidence_start = recording_seconds_to_canonical_sample_index(
                self.canonical,
                channel_id=channel_id,
                recording_seconds=float(interval[0]),
                rounding="floor",
            )
            reported_evidence_stop = recording_seconds_to_canonical_sample_index(
                self.canonical,
                channel_id=channel_id,
                recording_seconds=float(interval[1]),
                rounding="ceil",
            )
            decision_available_stop = recording_seconds_to_canonical_sample_index(
                self.canonical,
                channel_id=channel_id,
                recording_seconds=decision_available,
                rounding="ceil",
            )
            raw_start = recording_seconds_to_canonical_sample_index(
                self.canonical,
                channel_id=channel_id,
                recording_seconds=raw_start_seconds,
                rounding="floor",
            )
            raw_stop = recording_seconds_to_canonical_sample_index(
                self.canonical,
                channel_id=channel_id,
                recording_seconds=raw_stop_seconds,
                rounding="ceil",
            )
            raw_intervals.append(
                {
                    "channel_id": channel_id,
                    "sample_rate_numerator": int(channel["sample_rate_numerator"]),
                    "sample_rate_denominator": int(channel["sample_rate_denominator"]),
                    "channel_sample_count": int(channel["sample_count"]),
                    "raw_start_sample": int(raw_start),
                    "raw_stop_sample_exclusive": int(raw_stop),
                    "reported_evidence_start_sample": int(reported_evidence_start),
                    "reported_evidence_stop_sample_exclusive": int(
                        reported_evidence_stop
                    ),
                    "unshifted_decision_available_stop_sample_exclusive": int(
                        decision_available_stop
                    ),
                }
            )

        latency_samples = float(temporal["group_delay_samples"])
        latency_seconds = float(temporal["group_delay_recording_seconds"])
        body = {
            "schema_version": RAW_SAMPLE_DEPENDENCY_SCHEMA_VERSION,
            "dependency_status": dependency_status,
            "canonical_signal_sha256": self.canonical_sha256,
            "source_view_id": str(receipt["view_id"]),
            "view_role": role,
            "evidence_recording_interval": [
                float(interval[0]),
                float(interval[1]),
            ],
            "support_components": [
                {
                    "role": role,
                    "recording_interval": [start, stop],
                }
                for role, start, stop in components
            ],
            "decision_available_recording_seconds": decision_available,
            "confirmation_latency_samples_on_view_clock": (
                confirmation_latency_samples
            ),
            "confirmation_latency_seconds": confirmation_latency_seconds,
            "confirmation_policy": confirmation_policy,
            "view_tensor_sample_interval": [
                int(tensor_interval[0]),
                int(tensor_interval[1]),
            ],
            "view_sampling_rate_numerator": output_num,
            "view_sampling_rate_denominator": output_den,
            "raw_sample_intervals": raw_intervals,
            "dependency_policy": str(temporal["dependency_policy"]),
            "future_sample_access": future,
            "onset_evidence_authorized": onset_authorized,
            "onset_support_eligible": onset_support,
            "processing_latency_samples_on_view_clock": latency_samples,
            "processing_latency_seconds": latency_seconds,
            "processing_latency_policy": str(temporal["delay_correction_policy"]),
            "raw_support_end_policy": raw_policy,
            "receipt_lineage": {
                "canonical_receipt_sha256": str(self.canonical["receipt_sha256"]),
                "source_view_id": str(receipt["view_id"]),
                "source_view_receipt_id": str(receipt["view_receipt_id"]),
                "source_view_receipt_sha256": str(receipt["receipt_sha256"]),
                "source_transform_spec_sha256": str(
                    receipt["transform_spec"]["transform_spec_sha256"]
                ),
                "temporal_evidence_sha256": _canonical_sha256(temporal),
                "parent_view_bindings": deepcopy(receipt["parent_view_bindings"]),
            },
        }
        return _finalize_raw_sample_dependency(body)

    def opportunity(
        self,
        *,
        family: str,
        term_id: str,
        interval: tuple[float, float] | None,
        views: Sequence[Any],
        unit_ids: Sequence[str],
        status: str,
        usable_fraction: float,
        reason_codes: Sequence[str],
    ) -> str:
        key = (
            family,
            term_id,
            interval,
            tuple(str(view.receipt["view_id"]) for view in views),
            tuple(unit_ids),
            status,
            tuple(sorted(set(reason_codes))),
        )
        if key in self._opportunity_keys:
            return self._opportunity_keys[key]
        identifier = "OPP-" + _canonical_sha256(key)[:20]
        self._opportunity_keys[key] = identifier
        if status == "not_evaluable":
            row = {
                "evaluation_opportunity_id": identifier,
                "family": family,
                "term_id": term_id,
                "interval": None,
                "spatial_unit_keys": [],
                "source_view_ids": [],
                "status": status,
                "usable_fraction": 0.0,
                "effective_bandwidth_hz": None,
                "quality_mask_sha256": None,
                "reason_codes": sorted(set(reason_codes)),
            }
        else:
            if interval is None or not views or not unit_ids:
                raise ValueError(
                    "physical opportunity requires interval, views and units"
                )
            lower = max(_view_bandwidth(view, unit_ids)[0] for view in views)
            upper = min(_view_bandwidth(view, unit_ids)[1] for view in views)
            quality_hash = _canonical_sha256(
                [
                    [
                        view.receipt["view_receipt_id"],
                        view.receipt["masks"]["mask_sha256"],
                    ]
                    for view in views
                ]
            )
            row = {
                "evaluation_opportunity_id": identifier,
                "family": family,
                "term_id": term_id,
                "interval": _observed_span(interval, self.resolution),
                "spatial_unit_keys": [
                    f"{'lead' if '-' in unit_id else 'electrode'}:{unit_id}"
                    for unit_id in unit_ids
                ],
                "source_view_ids": [str(view.receipt["view_id"]) for view in views],
                "status": status,
                "usable_fraction": float(usable_fraction),
                "effective_bandwidth_hz": [float(lower), float(upper)],
                "quality_mask_sha256": quality_hash,
                "reason_codes": sorted(set(reason_codes)),
            }
        self.opportunities.append(row)
        return identifier

    def waveform(
        self,
        *,
        view: Any,
        unit_ids: Sequence[str],
        interval: tuple[float, float],
    ) -> str:
        key = (
            view.receipt["view_receipt_id"],
            tuple(unit_ids),
            float(interval[0]),
            float(interval[1]),
        )
        if key in self._waveform_keys:
            return self._waveform_keys[key]
        identifier = "WAVE-" + _canonical_sha256(key)[:20]
        role = _view_role(view)
        start = recording_seconds_to_view_tensor_index(
            view.receipt,
            recording_seconds=interval[0],
            rounding="ceil",
        )
        stop = recording_seconds_to_view_tensor_index(
            view.receipt,
            recording_seconds=interval[1],
            rounding="floor",
        )
        if stop <= start:
            raise ValueError("waveform interval has no complete view samples")
        eligible = bool(
            role != "unknown"
            and all(self.montage_eligibility[item] for item in unit_ids)
        )
        raw_dependency = self.raw_sample_dependency(
            view=view,
            unit_ids=unit_ids,
            interval=interval,
            tensor_interval=(start, stop),
        )
        self.waveforms.append(
            {
                "waveform_evidence_id": identifier,
                "interval": [float(interval[0]), float(interval[1])],
                "unit_ids": list(unit_ids),
                "source_view_id": str(view.receipt["view_id"]),
                "view_role": role,
                "view_receipt_id": str(view.receipt["view_receipt_id"]),
                "view_receipt_sha256": str(view.receipt["receipt_sha256"]),
                "processed_view_sha256": str(view.receipt["processed_view_sha256"]),
                "quality_mask_sha256": str(view.receipt["masks"]["mask_sha256"]),
                "evidence_eligible": eligible,
                "ineligibility_reason_codes": (
                    [] if eligible else ["waveform_source_not_evidence_eligible"]
                ),
                "render_policy": "CANONICAL-VIEW-REPLAY-V2",
                "canonical_signal_sha256": self.canonical_sha256,
                "raw_sample_dependency": raw_dependency,
            }
        )
        self._waveform_dependency_by_id[identifier] = str(
            raw_dependency["dependency_id"]
        )
        self._waveform_keys[key] = identifier
        return identifier

    def measurement(
        self,
        *,
        name_id: str,
        value: float,
        unit_id: str,
        view: Any,
        unit_ids: Sequence[str],
        interval: tuple[float, float],
        evidence_family: str,
        background_reference_ids: Sequence[str] = (),
        baseline_delta: float | None = None,
        raw_support_components: Sequence[tuple[str, tuple[float, float]]] = (),
        decision_available_recording_seconds: float | None = None,
    ) -> dict[str, Any]:
        if unit_id not in _UNIT_CATALOG:
            raise ValueError(f"unregistered deterministic unit: {unit_id}")
        if not math.isfinite(float(value)):
            raise ValueError("deterministic measurement must be finite")
        start = recording_seconds_to_view_tensor_index(
            view.receipt,
            recording_seconds=interval[0],
            rounding="ceil",
        )
        stop = recording_seconds_to_view_tensor_index(
            view.receipt,
            recording_seconds=interval[1],
            rounding="floor",
        )
        if stop <= start:
            raise ValueError("measurement interval has no complete view samples")
        self._measurement_counter += 1
        tolerance = max(1e-12, abs(float(value)) * 1e-12)
        eligible = bool(
            _view_role(view) != "unknown"
            and all(self.montage_eligibility[item] for item in unit_ids)
            and all(_unit_is_eligible(view, item, evidence_family) for item in unit_ids)
        )
        if not eligible:
            raise ValueError("attempted to serialize an ineligible measurement")
        raw_dependency = self.raw_sample_dependency(
            view=view,
            unit_ids=unit_ids,
            interval=interval,
            tensor_interval=(start, stop),
            support_components=raw_support_components,
            decision_available_recording_seconds=(decision_available_recording_seconds),
        )
        row = {
            "measurement_id": f"MEAS-{self._measurement_counter:04d}",
            "name_id": name_id,
            "value": float(value),
            "unit_id": unit_id,
            "unit_registry_status": "registered",
            "baseline_delta": (
                None if baseline_delta is None else float(baseline_delta)
            ),
            "numerical_uncertainty": {
                "status": "deterministic_replay_tolerance",
                "lower": float(value) - tolerance,
                "upper": float(value) + tolerance,
                "coverage": None,
                "calibration_receipt_id": None,
            },
            "producer_type": "deterministic_signal_measurement",
            "source_binding": {
                "canonical_signal_sha256": self.canonical_sha256,
                "source_view_id": str(view.receipt["view_id"]),
                "view_role": _view_role(view),
                "view_receipt_id": str(view.receipt["view_receipt_id"]),
                "view_receipt_sha256": str(view.receipt["receipt_sha256"]),
                "transform_spec_sha256": str(
                    view.receipt["transform_spec"]["transform_spec_sha256"]
                ),
                "processed_view_sha256": str(view.receipt["processed_view_sha256"]),
                "source_unit_ids": list(unit_ids),
                "recording_interval": [float(interval[0]), float(interval[1])],
                "tensor_sample_interval": [int(start), int(stop)],
                "effective_bandwidth_hz": _view_bandwidth(view, unit_ids),
                "reference_type": str(
                    view.receipt["transform_spec"]["reference"]["reference_type"]
                ),
                "evidence_family": evidence_family,
                "quality_mask_sha256": str(view.receipt["masks"]["mask_sha256"]),
                "edge_mask_sha256": _mask_component_sha256(
                    view, "edge_invalid_intervals"
                ),
                "padding_mask_sha256": _mask_component_sha256(
                    view, "padding_intervals"
                ),
                "imputation_mask_sha256": None,
                "evidence_eligible": True,
                "ineligibility_reason_codes": [],
                "background_reference_ids": list(background_reference_ids),
                "method_id": DETERMINISTIC_EVENT_FINDINGS_V2_METHOD_ID,
                "policy_sha256": self.policy_sha256,
                "raw_sample_dependency": raw_dependency,
            },
        }
        return row

    def support(
        self,
        *,
        unit_type: str,
        identifier: str,
        score: float | None = None,
        derived: bool = False,
    ) -> dict[str, Any]:
        eligible = (
            bool(self.montage_eligibility.get(identifier, False))
            if unit_type in {"lead", "electrode"}
            else True
        )
        return {
            "unit_type": unit_type,
            "id": identifier,
            "mapping_status": "derived" if derived else "direct",
            "observation_status": "derived" if derived else "observed",
            "evidence_eligible": eligible,
            "missing_reason_codes": [] if eligible else ["spatial_unit_ineligible"],
            "support_score": None if score is None else float(score),
            "field_observation": None,
        }

    def add_finding(
        self,
        *,
        evidence_id: str,
        family: str,
        term_id: str,
        assertion_level: str,
        status: str,
        role: str,
        temporal_context: str,
        opportunity_id: str,
        interval: tuple[float, float] | None = None,
        spatial_support: Sequence[Mapping[str, Any]] = (),
        measurements: Sequence[Mapping[str, Any]] = (),
        waveform_ids: Sequence[str] = (),
        model_uncertainty: float = 0.0,
    ) -> None:
        if status == "not_evaluable":
            interval = None
            spatial_support = ()
            measurements = ()
            waveform_ids = ()
        dependency_ids = sorted(
            {
                str(dependency["dependency_id"])
                for measurement in measurements
                for dependency in [
                    measurement["source_binding"].get("raw_sample_dependency")
                ]
                if isinstance(dependency, Mapping)
            }
            | {
                self._waveform_dependency_by_id[waveform_id]
                for waveform_id in waveform_ids
            }
        )
        outside = temporal_context == "outside_candidate_protection"
        # A term-level not-evaluable atom has no physical span, but it still
        # belongs to this event's evaluation ledger.  The validator reserves
        # zero ownership for physically outside context, so serialize ledger
        # ownership as complete when no interval exists.
        overlap_fraction = (
            1.0
            if interval is None and not outside
            else 0.0
            if interval is None
            else _overlap(interval, self.protection_zone) / (interval[1] - interval[0])
        )
        if outside:
            ownership_status = "outside_protection"
            owners: list[str] = []
            overlap_fraction = 0.0
        else:
            ownership_status = "event_owned"
            owners = [self.event_id]
        uncertainty = {
            "boundary": 0.5,
            "quality": 0.0,
            "background": 0.0,
            "model": float(model_uncertainty),
            "reference_stability": 1.0 if family == "spatial_field" else 0.5,
            "semantics": "componentwise_descriptive_not_individual_correctness_probability",
        }
        self.findings.append(
            {
                "evidence_id": evidence_id,
                "finding_group_id": None,
                "family": family,
                "term": _term_ref(term_id),
                "assertion_level": assertion_level,
                "status": status,
                "intrinsic_evidence_role": role,
                "signal_temporal_context": temporal_context,
                "ownership": {
                    "owner_event_ids": owners,
                    "event_group_id": None,
                    "protection_zone_id": self.protection_zone_id,
                    "ownership_status": ownership_status,
                    "protection_zone_overlap_fraction": float(overlap_fraction),
                },
                "state_membership": _state_membership(
                    interval,
                    self.state_spans,
                    force_zero=outside or status == "not_evaluable",
                ),
                "time_interval": (
                    None
                    if interval is None
                    else _observed_span(interval, self.resolution)
                ),
                "spatial_support": [dict(item) for item in spatial_support],
                "measurements": [dict(item) for item in measurements],
                "uncertainty": uncertainty,
                "evaluation_opportunity_id": opportunity_id,
                "capability_receipt_id": None,
                "sensitivity_receipt_id": None,
                "term_decision_receipt_id": None,
                "waveform_evidence_ids": list(waveform_ids),
                "raw_sample_dependency_ids": dependency_ids,
            }
        )


def _grid_or_none(
    view: Any | None, policy: DeterministicEventFindingsPolicy
) -> Any | None:
    if view is None:
        return None
    try:
        return _feature_grid(view, policy=policy)
    except ValueError as error:
        if "shorter than one Findings window" not in str(error):
            raise
        return None


def _baseline_mask(
    grid: Any | None, baseline: tuple[float, float] | None
) -> np.ndarray | None:
    if grid is None or baseline is None:
        return None
    return (grid.recording_starts >= baseline[0] - 1e-9) & (
        grid.recording_stops <= baseline[1] + 1e-9
    )


def _first_changes(
    grid: Any | None,
    *,
    baseline: tuple[float, float] | None,
    candidate_stop: float,
    policy: DeterministicEventFindingsPolicy,
) -> tuple[np.ndarray | None, dict[int, int]]:
    mask = _baseline_mask(grid, baseline)
    if (
        grid is None
        or mask is None
        or int(np.count_nonzero(mask)) < policy.minimum_baseline_windows
    ):
        return None, {}
    scores = _change_scores(grid, baseline_mask=mask)
    candidate_start = float(baseline[1])
    candidate_mask = (grid.recording_stops > candidate_start + 1e-9) & (
        grid.recording_starts < candidate_stop - 1e-9
    )
    return scores, _first_sustained_changes(
        scores,
        grid,
        candidate_mask=candidate_mask,
        policy=policy,
    )


def _median(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    return None if finite.size == 0 else float(np.median(finite))


def _provider_receipts(policy_sha256: str) -> list[dict[str, Any]]:
    return [
        {
            "receipt_id": "PROD-DETERMINISTIC-EVENT-FINDINGS-V2",
            "producer_id": DETERMINISTIC_EVENT_FINDINGS_V2_METHOD_ID,
            "producer_type": "event_findings_provider",
            "version": "v2",
            "artifact_sha256": _canonical_sha256(
                [DETERMINISTIC_EVENT_FINDINGS_V2_METHOD_ID, "native-v2"]
            ),
            "policy_sha256": policy_sha256,
            "validation_scope": "none",
            "patient_disjoint": False,
            "frozen_before_inference": True,
        },
        {
            "receipt_id": "PROD-DETERMINISTIC-HYPOTHESIS-RELATIONS-V2",
            "producer_id": DETERMINISTIC_EVENT_HYPOTHESIS_V2_METHOD_ID,
            "producer_type": "hypothesis_relation_builder",
            "version": "v2",
            "artifact_sha256": _canonical_sha256(
                [DETERMINISTIC_EVENT_HYPOTHESIS_V2_METHOD_ID, "native-v2"]
            ),
            "policy_sha256": policy_sha256,
            "validation_scope": "none",
            "patient_disjoint": False,
            "frozen_before_inference": True,
        },
    ]


def produce_deterministic_event_eeg_findings_v2(
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
    """Build a complete native-v2 EEG-only event evidence graph.

    Signal insufficiency (missing causal view, short window, unavailable
    baseline, censoring or no stable spatial change) yields a complete
    not-evaluable/nonlocalizable bundle.  Receipt/hash/cross-record integrity
    failures remain hard errors.
    """

    _identifier(event_id, "event_id")
    if not isinstance(policy, DeterministicEventFindingsPolicy):
        raise TypeError("policy must be DeterministicEventFindingsPolicy")
    search = validate_adaptive_search_receipt(adaptive_search_receipt)
    adaptive = validate_adaptive_event_analysis_window(adaptive_window_receipt)
    canonical = validate_canonical_signal_receipt(canonical_receipt)
    if search["canonical_signal_binding"] is None:
        raise ValueError("adaptive search lacks a canonical signal binding")
    validate_canonical_adaptive_binding_against_receipt(
        search["canonical_signal_binding"], canonical
    )
    if adaptive["source_search_receipt_id"] != search["search_receipt_id"]:
        raise ValueError("adaptive window belongs to a different search receipt")
    if adaptive["source_search_receipt_sha256"] != _canonical_sha256(search):
        raise ValueError("adaptive window source-search hash drifted")
    if search["recording_duration_seconds"] is not None and not math.isclose(
        float(search["recording_duration_seconds"]),
        float(canonical["recording_duration_seconds"]),
        abs_tol=1e-6,
    ):
        raise ValueError("adaptive search and canonical durations differ")

    final_interval = tuple(
        float(item) for item in adaptive["analysis_interval_recording_seconds"]
    )
    prepared = _prepare_views(
        canonical=canonical,
        views=views,
        final_interval=final_interval,
        trusted_parent_views=trusted_parent_views,
    )
    role_views: dict[str, list[Any]] = defaultdict(list)
    for view in prepared:
        role_views[_view_role(view)].append(view)
    causal = role_views["onset_causal"][0] if role_views["onset_causal"] else None
    causal_response_unqualified = causal is None and bool(
        role_views["onset_causal_unqualified"]
    )
    if causal_response_unqualified:
        unqualified_reasons = role_views["onset_causal_unqualified"][0].receipt[
            "temporal_evidence"
        ]["authorization_reason_codes"]
        causal_unavailable_reason = next(
            reason
            for reason in (
                ONSET_FIR_RESPONSE_UNQUALIFIED_REASON_CODE,
                ONSET_FIR_CLINICAL_ADMISSION_UNQUALIFIED_REASON_CODE,
            )
            if reason in unqualified_reasons
        )
    else:
        causal_unavailable_reason = "causal_onset_opportunity_unavailable"
    offline = (
        role_views["context_offline"][0] if role_views["context_offline"] else None
    )
    physical = (
        role_views["canonical_physical_evidence"][0]
        if role_views["canonical_physical_evidence"]
        else None
    )
    quality_view = causal or offline or physical or prepared[0]
    context_view = offline or physical
    resolution = max(policy.step_seconds, 1.0 / quality_view.sampling_rate_hz)
    producer_policy_sha = _producer_policy_sha256(policy)

    montage, montage_eligible, laterality, region = _montage(quality_view)
    per_unit_quality, usable_fraction, artifacts = _quality(
        quality_view,
        final_interval=final_interval,
        montage_eligibility=montage_eligible,
    )

    baseline_raw = adaptive["baseline_context_recording_seconds"]
    baseline = (
        None
        if baseline_raw is None
        else (float(baseline_raw[0]), float(baseline_raw[1]))
    )
    core_raw = adaptive["analysis_core_recording_seconds"]
    # A right-censored adaptive event has a replayable onset boundary but no
    # observed termination, encoded by ``[onset, null]``.  Treating the null
    # endpoint as a float turned a legitimate signal-insufficiency state into
    # a technical failure and prevented the required complete candidate
    # Findings ledger from being emitted.  The observed analysis-envelope end
    # is the only admissible search bound in that case; it is not promoted to
    # a termination estimate.
    core = (
        None
        if core_raw is None
        else (
            float(core_raw[0]),
            None if core_raw[1] is None else float(core_raw[1]),
        )
    )
    candidate_stop = (
        float(core[1])
        if core is not None and core[1] is not None
        else final_interval[1]
    )
    causal_grid = _grid_or_none(causal, policy)
    causal_scores, causal_changes = _first_changes(
        causal_grid,
        baseline=baseline,
        candidate_stop=candidate_stop,
        policy=policy,
    )
    onset_units: list[int] = []
    onset_interval: tuple[float, float] | None = None
    onset_decision_available: float | None = None
    signal_findings_eligible = bool(adaptive["eligibility"]["signal_findings"])
    onset_localization_eligible = bool(
        adaptive["eligibility"]["onset_localization"]
        and not adaptive["censoring"]["left"]
    )
    if (
        signal_findings_eligible
        and onset_localization_eligible
        and causal is not None
        and causal_grid is not None
        and causal_scores is not None
        and causal_changes
    ):
        eligible_changes = [
            index
            for index, window_index in causal_changes.items()
            if causal.unit_types[index] == "lead"
            and montage_eligible[causal.unit_ids[index]]
            and causal_grid.spatial_valid[window_index, index]
        ]
        if eligible_changes:
            earliest = min(
                float(causal_grid.recording_starts[causal_changes[index]])
                for index in eligible_changes
            )
            onset_units = sorted(
                [
                    index
                    for index in eligible_changes
                    if float(causal_grid.recording_starts[causal_changes[index]])
                    <= earliest + policy.near_synchronous_seconds + 1e-9
                ],
                key=lambda index: (
                    -float(causal_scores[causal_changes[index], index]),
                    causal.unit_ids[index],
                ),
            )
            intervals = [
                (
                    float(causal_grid.recording_starts[causal_changes[index]]),
                    float(causal_grid.recording_stops[causal_changes[index]]),
                )
                for index in onset_units
            ]
            onset_interval = (
                min(item[0] for item in intervals),
                max(item[1] for item in intervals),
            )
            # Confirmation changes only when the decision becomes available;
            # it must not move the reported onset candidate or be confused
            # with FIR processing latency.
            onset_decision_available = max(
                float(
                    causal_grid.recording_stops[
                        causal_changes[index] + policy.sustained_change_windows - 1
                    ]
                )
                for index in onset_units
            )

    recovery_raw = adaptive["recovery_context_recording_seconds"]
    recovery = (
        None
        if recovery_raw is None
        else (float(recovery_raw[0]), float(recovery_raw[1]))
    )
    offset_anchor = (
        float(core[1])
        if core is not None
        and core[1] is not None
        and recovery is not None
        and offline is not None
        and adaptive["censoring"]["termination_observed"]
        else None
    )
    state_segments, state_spans = _state_contract(
        final_interval=final_interval,
        onset_interval=onset_interval,
        offset_anchor=offset_anchor,
        resolution=resolution,
        receipt_id=str(adaptive["window_receipt_id"]),
    )
    protection = (
        final_interval
        if onset_interval is None
        else (float(onset_interval[0]), float(final_interval[1]))
    )
    protection_id = (
        "PZONE-"
        + _canonical_sha256([event_id, protection, adaptive["window_receipt_id"]])[:20]
    )

    context_grid = _grid_or_none(context_view, policy)
    context_baseline_mask = _baseline_mask(context_grid, baseline)
    background_available = bool(
        baseline is not None
        and context_view is not None
        and context_grid is not None
        and context_baseline_mask is not None
        and int(np.count_nonzero(context_baseline_mask))
        >= policy.minimum_baseline_windows
        and _overlap(baseline, protection) <= 1e-9
        and any(
            montage_eligible[unit_id]
            and _unit_is_eligible(context_view, unit_id, "spectral")
            for unit_id in context_view.unit_ids
        )
    )
    background_bank_id = (
        "BGBANK-"
        + _canonical_sha256([event_id, baseline, canonical["receipt_sha256"]])[:20]
        if background_available
        else None
    )
    background_selection_id = (
        "BGSELECT-"
        + _canonical_sha256(
            [event_id, search["search_receipt_id"], adaptive["window_receipt_id"]]
        )[:20]
        if background_available
        else None
    )

    builder = _EvidenceBuilder(
        event_id=event_id,
        canonical_receipt=canonical,
        policy_sha256=producer_policy_sha,
        protection_zone=protection,
        protection_zone_id=protection_id,
        state_spans=state_spans,
        montage_eligibility=montage_eligible,
        resolution=resolution,
    )

    eligible_quality_units = [
        unit_id
        for unit_id in quality_view.unit_ids
        if montage_eligible[unit_id]
        and _unit_is_eligible(quality_view, unit_id, "waveform")
    ]
    if eligible_quality_units:
        opportunity = builder.opportunity(
            family="quality",
            term_id="deterministic_signal_usable_fraction",
            interval=final_interval,
            views=[quality_view],
            unit_ids=eligible_quality_units,
            status="sufficient",
            usable_fraction=usable_fraction,
            reason_codes=[],
        )
        measurement = builder.measurement(
            name_id="usable_sample_fraction",
            value=usable_fraction,
            unit_id="ratio",
            view=quality_view,
            unit_ids=eligible_quality_units,
            interval=final_interval,
            evidence_family="waveform",
        )
        waveform = builder.waveform(
            view=quality_view,
            unit_ids=eligible_quality_units,
            interval=final_interval,
        )
        builder.add_finding(
            evidence_id="E-QUALITY-COVERAGE",
            family="quality",
            term_id="deterministic_signal_usable_fraction",
            assertion_level="measured",
            status="present",
            role="early_context",
            temporal_context="pre_candidate",
            opportunity_id=opportunity,
            interval=final_interval,
            measurements=[measurement],
            waveform_ids=[waveform],
        )
    else:
        opportunity = builder.opportunity(
            family="quality",
            term_id="deterministic_signal_usable_fraction",
            interval=None,
            views=[],
            unit_ids=[],
            status="not_evaluable",
            usable_fraction=0.0,
            reason_codes=["no_eligible_physical_task_view"],
        )
        builder.add_finding(
            evidence_id="E-QUALITY-COVERAGE",
            family="quality",
            term_id="deterministic_signal_usable_fraction",
            assertion_level="model_candidate",
            status="not_evaluable",
            role="limitation",
            temporal_context="unknown",
            opportunity_id=opportunity,
            model_uncertainty=1.0,
        )

    if background_available and context_grid is not None and context_view is not None:
        valid_units = [
            index
            for index, unit_id in enumerate(context_view.unit_ids)
            if montage_eligible[unit_id]
            and _unit_is_eligible(context_view, unit_id, "spectral")
            and np.any(context_baseline_mask & context_grid.spectral_valid[:, index])
        ]
        if valid_units:
            unit_ids = [context_view.unit_ids[index] for index in valid_units]
            opportunity = builder.opportunity(
                family="spectral",
                term_id="deterministic_background_spectral_profile",
                interval=baseline,
                views=[context_view],
                unit_ids=unit_ids,
                status="sufficient",
                usable_fraction=usable_fraction,
                reason_codes=[],
            )
            measurements = []
            for ordinal, index in enumerate(valid_units, start=1):
                valid = context_baseline_mask & context_grid.spectral_valid[:, index]
                dominant = _median(context_grid.dominant_frequency_hz[valid, index])
                entropy = _median(context_grid.spectral_entropy[valid, index])
                if dominant is not None:
                    measurements.append(
                        builder.measurement(
                            name_id=f"background_dominant_frequency_{ordinal}",
                            value=dominant,
                            unit_id="Hz",
                            view=context_view,
                            unit_ids=[context_view.unit_ids[index]],
                            interval=baseline,
                            evidence_family="spectral",
                            background_reference_ids=[background_bank_id],
                        )
                    )
                if entropy is not None:
                    measurements.append(
                        builder.measurement(
                            name_id=f"background_spectral_entropy_{ordinal}",
                            value=entropy,
                            unit_id="unitless",
                            view=context_view,
                            unit_ids=[context_view.unit_ids[index]],
                            interval=baseline,
                            evidence_family="spectral",
                            background_reference_ids=[background_bank_id],
                        )
                    )
            waveform = builder.waveform(
                view=context_view, unit_ids=unit_ids, interval=baseline
            )
            builder.add_finding(
                evidence_id="E-BACKGROUND-SPECTRAL",
                family="spectral",
                term_id="deterministic_background_spectral_profile",
                assertion_level="measured",
                status="present",
                role="non_event_context",
                temporal_context="outside_candidate_protection",
                opportunity_id=opportunity,
                interval=baseline,
                spatial_support=[
                    builder.support(
                        unit_type=(
                            "lead"
                            if context_view.unit_types[index] == "lead"
                            else "electrode"
                        ),
                        identifier=context_view.unit_ids[index],
                    )
                    for index in valid_units
                ],
                measurements=measurements,
                waveform_ids=[waveform],
            )

    event_interval = (
        (float(onset_interval[0]), float(offset_anchor or final_interval[1]))
        if onset_interval is not None
        else None
    )
    descriptive_units: list[int] = []
    if (
        event_interval is not None
        and context_grid is not None
        and context_view is not None
    ):
        descriptive_units = [
            index
            for index in onset_units
            if index < len(context_view.unit_ids)
            and montage_eligible[context_view.unit_ids[index]]
            and _unit_is_eligible(
                context_view, context_view.unit_ids[index], "spectral"
            )
        ]
        if not descriptive_units:
            event_mask = (context_grid.recording_starts >= event_interval[0] - 1e-9) & (
                context_grid.recording_stops <= event_interval[1] + 1e-9
            )
            medians = []
            for index in range(len(context_view.unit_ids)):
                value = _median(context_grid.rms_uv[event_mask, index])
                if (
                    value is not None
                    and montage_eligible[context_view.unit_ids[index]]
                    and _unit_is_eligible(
                        context_view,
                        context_view.unit_ids[index],
                        "spectral",
                    )
                ):
                    medians.append((value, index))
            descriptive_units = [
                index
                for _, index in sorted(medians, reverse=True)[
                    : policy.maximum_descriptive_units
                ]
            ]
        descriptive_units = descriptive_units[: policy.maximum_descriptive_units]

    for family, term_id in (
        ("spectral", "deterministic_event_spectral_profile"),
        ("rhythm", "deterministic_event_rhythmicity_profile"),
    ):
        if (
            event_interval is None
            or context_grid is None
            or context_view is None
            or not descriptive_units
        ):
            opportunity = builder.opportunity(
                family=family,
                term_id=term_id,
                interval=None,
                views=[],
                unit_ids=[],
                status="not_evaluable",
                usable_fraction=0.0,
                reason_codes=["offline_event_context_unavailable"],
            )
            builder.add_finding(
                evidence_id=f"E-{family.upper()}-PROFILE",
                family=family,
                term_id=term_id,
                assertion_level="model_candidate",
                status="not_evaluable",
                role="limitation",
                temporal_context="unknown",
                opportunity_id=opportunity,
                model_uncertainty=1.0,
            )
            continue
        unit_ids = [context_view.unit_ids[index] for index in descriptive_units]
        opportunity = builder.opportunity(
            family=family,
            term_id=term_id,
            interval=event_interval,
            views=[context_view],
            unit_ids=unit_ids,
            status="sufficient",
            usable_fraction=usable_fraction,
            reason_codes=[],
        )
        event_mask = (context_grid.recording_starts >= event_interval[0] - 1e-9) & (
            context_grid.recording_stops <= event_interval[1] + 1e-9
        )
        measurements = []
        for ordinal, index in enumerate(descriptive_units, start=1):
            valid = event_mask & context_grid.spectral_valid[:, index]
            if family == "spectral":
                candidates = (
                    (
                        f"event_dominant_frequency_{ordinal}",
                        _median(context_grid.dominant_frequency_hz[valid, index]),
                        "Hz",
                    ),
                    (
                        f"event_spectral_entropy_{ordinal}",
                        _median(context_grid.spectral_entropy[valid, index]),
                        "unitless",
                    ),
                    (
                        f"event_theta_power_ratio_{ordinal}",
                        _median(context_grid.band_ratio[valid, index, 1]),
                        "ratio",
                    ),
                )
            else:
                candidates = (
                    (
                        f"event_rhythmicity_index_{ordinal}",
                        _median(context_grid.rhythmicity_index[valid, index]),
                        "unitless",
                    ),
                    (
                        f"event_spectral_concentration_{ordinal}",
                        _median(context_grid.spectral_concentration[valid, index]),
                        "ratio",
                    ),
                )
            for name, value, unit in candidates:
                if value is None:
                    continue
                measurements.append(
                    builder.measurement(
                        name_id=name,
                        value=value,
                        unit_id=unit,
                        view=context_view,
                        unit_ids=[context_view.unit_ids[index]],
                        interval=event_interval,
                        evidence_family="spectral",
                        background_reference_ids=(
                            [background_bank_id] if background_bank_id else []
                        ),
                    )
                )
        if measurements:
            waveform = builder.waveform(
                view=context_view, unit_ids=unit_ids, interval=event_interval
            )
            builder.add_finding(
                evidence_id=f"E-{family.upper()}-PROFILE",
                family=family,
                term_id=term_id,
                assertion_level="measured",
                status="present",
                role="early_context",
                temporal_context="sustained_candidate",
                opportunity_id=opportunity,
                interval=event_interval,
                spatial_support=[
                    builder.support(
                        unit_type=(
                            "lead"
                            if context_view.unit_types[index] == "lead"
                            else "electrode"
                        ),
                        identifier=context_view.unit_ids[index],
                    )
                    for index in descriptive_units
                ],
                measurements=measurements,
                waveform_ids=[waveform],
            )
        else:
            builder.add_finding(
                evidence_id=f"E-{family.upper()}-PROFILE",
                family=family,
                term_id=term_id,
                assertion_level="model_candidate",
                status="uncertain",
                role="early_context",
                temporal_context="sustained_candidate",
                opportunity_id=opportunity,
                interval=event_interval,
                model_uncertainty=1.0,
            )

    onset_evidence_ids: list[str] = []
    field_evidence_id: str | None = None
    causal_unit_ids = (
        [causal.unit_ids[index] for index in onset_units] if causal is not None else []
    )
    onset_raw_support_components: list[tuple[str, tuple[float, float]]] = []
    if baseline is not None:
        onset_raw_support_components.append(("baseline_reference", baseline))
    if (
        onset_interval is not None
        and onset_decision_available is not None
        and onset_decision_available > onset_interval[1] + 1e-9
    ):
        onset_raw_support_components.append(
            (
                "sustained_confirmation",
                (onset_interval[1], onset_decision_available),
            )
        )
    for family, term_id, evidence_id, source_family in (
        (
            "evolution",
            "deterministic_multifeature_change_point_candidate",
            "E-ONSET-CHANGE",
            "spectral",
        ),
        (
            "spatial_field",
            "reference_specific_spatial_change_candidate",
            "E-ONSET-SPATIAL-FIELD",
            "spatial_field",
        ),
    ):
        if (
            onset_interval is None
            or causal is None
            or causal_grid is None
            or causal_scores is None
            or not onset_units
        ):
            opportunity_status = (
                "not_evaluable"
                if causal is None or baseline is None or not onset_localization_eligible
                else "limited"
            )
            if opportunity_status == "not_evaluable":
                opportunity = builder.opportunity(
                    family=family,
                    term_id=term_id,
                    interval=None,
                    views=[],
                    unit_ids=[],
                    status="not_evaluable",
                    usable_fraction=0.0,
                    reason_codes=[
                        causal_unavailable_reason
                        if causal is None
                        else "onset_boundary_or_baseline_unavailable"
                    ],
                )
                status = "not_evaluable"
                interval = None
            else:
                eligible_ids = [
                    unit_id
                    for unit_id in causal.unit_ids
                    if montage_eligible[unit_id]
                    and _unit_is_eligible(causal, unit_id, source_family)
                ]
                if eligible_ids:
                    opportunity = builder.opportunity(
                        family=family,
                        term_id=term_id,
                        interval=final_interval,
                        views=[causal],
                        unit_ids=eligible_ids,
                        status="limited",
                        usable_fraction=usable_fraction,
                        reason_codes=["stable_sustained_change_not_resolved"],
                    )
                    status = "uncertain"
                    interval = final_interval
                else:
                    opportunity = builder.opportunity(
                        family=family,
                        term_id=term_id,
                        interval=None,
                        views=[],
                        unit_ids=[],
                        status="not_evaluable",
                        usable_fraction=0.0,
                        reason_codes=["no_eligible_causal_onset_units"],
                    )
                    status = "not_evaluable"
                    interval = None
            builder.add_finding(
                evidence_id=evidence_id,
                family=family,
                term_id=term_id,
                assertion_level="model_candidate",
                status=status,
                role="limitation" if status == "not_evaluable" else "early_context",
                temporal_context="unknown"
                if status == "not_evaluable"
                else "pre_candidate",
                opportunity_id=opportunity,
                interval=interval,
                model_uncertainty=1.0,
            )
            continue
        opportunity = builder.opportunity(
            family=family,
            term_id=term_id,
            interval=onset_interval,
            views=[causal],
            unit_ids=causal_unit_ids,
            status="sufficient",
            usable_fraction=usable_fraction,
            reason_codes=[],
        )
        measurements = []
        supports = []
        for ordinal, index in enumerate(onset_units, start=1):
            score = float(causal_scores[causal_changes[index], index])
            unit_id = causal.unit_ids[index]
            measurements.append(
                builder.measurement(
                    name_id=f"onset_multifeature_change_score_{ordinal}",
                    value=score,
                    unit_id="robust_z",
                    view=causal,
                    unit_ids=[unit_id],
                    interval=onset_interval,
                    evidence_family=source_family,
                    background_reference_ids=(
                        [background_bank_id] if background_bank_id else []
                    ),
                    baseline_delta=(score if background_bank_id else None),
                    raw_support_components=onset_raw_support_components,
                    decision_available_recording_seconds=(onset_decision_available),
                )
            )
            supports.append(
                builder.support(unit_type="lead", identifier=unit_id, score=score)
            )
        if family == "spatial_field":
            grouped: dict[tuple[str, str], float] = {}
            for index in onset_units:
                score = float(causal_scores[causal_changes[index], index])
                unit_id = causal.unit_ids[index]
                grouped[("laterality", laterality[unit_id])] = max(
                    grouped.get(("laterality", laterality[unit_id]), -math.inf),
                    score,
                )
                grouped[("region", region[unit_id])] = max(
                    grouped.get(("region", region[unit_id]), -math.inf), score
                )
            supports.extend(
                builder.support(
                    unit_type=unit_type,
                    identifier=identifier,
                    score=score,
                    derived=True,
                )
                for (unit_type, identifier), score in sorted(grouped.items())
            )
            field_evidence_id = evidence_id
        waveform = builder.waveform(
            view=causal, unit_ids=causal_unit_ids, interval=onset_interval
        )
        builder.add_finding(
            evidence_id=evidence_id,
            family=family,
            term_id=term_id,
            assertion_level="measured",
            status="present",
            role="onset_eligible",
            temporal_context="candidate_emergence",
            opportunity_id=opportunity,
            interval=onset_interval,
            spatial_support=supports,
            measurements=measurements,
            waveform_ids=[waveform],
            model_uncertainty=0.5,
        )
        onset_evidence_ids.append(evidence_id)

    offline_grid = _grid_or_none(offline, policy)
    offline_scores, offline_changes = _first_changes(
        offline_grid,
        baseline=baseline,
        candidate_stop=candidate_stop,
        policy=policy,
    )
    late_units: list[int] = []
    if (
        onset_interval is not None
        and offline is not None
        and offline_grid is not None
        and offline_scores is not None
    ):
        onset_names = set(causal_unit_ids)
        source_time = onset_interval[0]
        late_units = sorted(
            [
                index
                for index, window_index in offline_changes.items()
                if offline.unit_ids[index] not in onset_names
                and montage_eligible[offline.unit_ids[index]]
                and offline_grid.spatial_valid[window_index, index]
                and float(offline_grid.recording_starts[window_index]) - source_time
                >= policy.later_involvement_delay_seconds - 1e-9
            ],
            key=lambda index: (
                float(offline_grid.recording_starts[offline_changes[index]]),
                offline.unit_ids[index],
            ),
        )

    recruitment_term = "deterministic_later_involvement_candidate"
    recruitment_opportunity: str
    late_evidence_by_unit: dict[str, str] = {}
    if onset_interval is not None and offline is not None:
        eligible_ids = [
            unit_id
            for unit_id in offline.unit_ids
            if montage_eligible[unit_id]
            and _unit_is_eligible(offline, unit_id, "spatial_field")
        ]
        if eligible_ids:
            recruitment_opportunity = builder.opportunity(
                family="spatial_recruitment",
                term_id=recruitment_term,
                interval=(onset_interval[0], final_interval[1]),
                views=[causal, offline] if causal is not None else [offline],
                unit_ids=eligible_ids,
                status="sufficient",
                usable_fraction=usable_fraction,
                reason_codes=[],
            )
        else:
            recruitment_opportunity = builder.opportunity(
                family="spatial_recruitment",
                term_id=recruitment_term,
                interval=None,
                views=[],
                unit_ids=[],
                status="not_evaluable",
                usable_fraction=0.0,
                reason_codes=["no_eligible_offline_units"],
            )
        for ordinal, index in enumerate(late_units, start=1):
            window_index = offline_changes[index]
            interval = (
                float(offline_grid.recording_starts[window_index]),
                float(offline_grid.recording_stops[window_index]),
            )
            unit_id = offline.unit_ids[index]
            delay = interval[0] - onset_interval[0]
            evidence_id = f"E-LATER-INVOLVEMENT-{ordinal}"
            measurement = builder.measurement(
                name_id=f"later_involvement_delay_{ordinal}",
                value=delay,
                unit_id="s",
                view=offline,
                unit_ids=[unit_id],
                interval=interval,
                evidence_family="spatial_field",
                background_reference_ids=(
                    [background_bank_id] if background_bank_id else []
                ),
            )
            waveform = builder.waveform(
                view=offline, unit_ids=[unit_id], interval=interval
            )
            builder.add_finding(
                evidence_id=evidence_id,
                family="spatial_recruitment",
                term_id=recruitment_term,
                assertion_level="measured",
                status="present",
                role="later_involvement",
                temporal_context="late_involvement",
                opportunity_id=recruitment_opportunity,
                interval=interval,
                spatial_support=[
                    builder.support(
                        unit_type=(
                            "lead"
                            if offline.unit_types[index] == "lead"
                            else "electrode"
                        ),
                        identifier=unit_id,
                        score=float(offline_scores[window_index, index]),
                    )
                ],
                measurements=[measurement],
                waveform_ids=[waveform],
                model_uncertainty=0.5,
            )
            late_evidence_by_unit[unit_id] = evidence_id
        if not late_units and eligible_ids:
            builder.add_finding(
                evidence_id="E-LATER-INVOLVEMENT-UNRESOLVED",
                family="spatial_recruitment",
                term_id=recruitment_term,
                assertion_level="model_candidate",
                status="uncertain",
                role="later_involvement",
                temporal_context="late_involvement",
                opportunity_id=recruitment_opportunity,
                interval=(onset_interval[0], final_interval[1]),
                model_uncertainty=1.0,
            )
        elif not eligible_ids:
            builder.add_finding(
                evidence_id="E-LATER-INVOLVEMENT-UNRESOLVED",
                family="spatial_recruitment",
                term_id=recruitment_term,
                assertion_level="model_candidate",
                status="not_evaluable",
                role="limitation",
                temporal_context="unknown",
                opportunity_id=recruitment_opportunity,
                model_uncertainty=1.0,
            )
    else:
        recruitment_opportunity = builder.opportunity(
            family="spatial_recruitment",
            term_id=recruitment_term,
            interval=None,
            views=[],
            unit_ids=[],
            status="not_evaluable",
            usable_fraction=0.0,
            reason_codes=["causal_onset_or_offline_recruitment_view_unavailable"],
        )
        builder.add_finding(
            evidence_id="E-LATER-INVOLVEMENT-UNRESOLVED",
            family="spatial_recruitment",
            term_id=recruitment_term,
            assertion_level="model_candidate",
            status="not_evaluable",
            role="limitation",
            temporal_context="unknown",
            opportunity_id=recruitment_opportunity,
            model_uncertainty=1.0,
        )

    recovery_term = "deterministic_recovery_context_profile"
    if recovery is not None and offline is not None and offline_grid is not None:
        recovery_mask = (offline_grid.recording_starts >= recovery[0] - 1e-9) & (
            offline_grid.recording_stops <= recovery[1] + 1e-9
        )
        recovery_units = [
            index
            for index, unit_id in enumerate(offline.unit_ids)
            if montage_eligible[unit_id]
            and _unit_is_eligible(offline, unit_id, "amplitude")
            and np.any(recovery_mask & offline_grid.amplitude_valid[:, index])
        ]
    else:
        recovery_mask = None
        recovery_units = []
    if recovery is not None and offline is not None and recovery_units:
        unit_ids = [offline.unit_ids[index] for index in recovery_units]
        opportunity = builder.opportunity(
            family="termination_recovery",
            term_id=recovery_term,
            interval=recovery,
            views=[offline],
            unit_ids=unit_ids,
            status="sufficient",
            usable_fraction=usable_fraction,
            reason_codes=[],
        )
        measurements = []
        for ordinal, index in enumerate(recovery_units, start=1):
            value = _median(offline_grid.rms_uv[recovery_mask, index])
            if value is not None:
                measurements.append(
                    builder.measurement(
                        name_id=f"recovery_rms_amplitude_{ordinal}",
                        value=value,
                        unit_id="uV",
                        view=offline,
                        unit_ids=[offline.unit_ids[index]],
                        interval=recovery,
                        evidence_family="amplitude",
                        background_reference_ids=(
                            [background_bank_id] if background_bank_id else []
                        ),
                    )
                )
        waveform = builder.waveform(view=offline, unit_ids=unit_ids, interval=recovery)
        builder.add_finding(
            evidence_id="E-RECOVERY-CONTEXT",
            family="termination_recovery",
            term_id=recovery_term,
            assertion_level="measured",
            status="present",
            role="early_context",
            temporal_context="return_candidate",
            opportunity_id=opportunity,
            interval=recovery,
            spatial_support=[
                builder.support(
                    unit_type=(
                        "lead" if offline.unit_types[index] == "lead" else "electrode"
                    ),
                    identifier=offline.unit_ids[index],
                )
                for index in recovery_units
            ],
            measurements=measurements,
            waveform_ids=[waveform],
        )
    else:
        opportunity = builder.opportunity(
            family="termination_recovery",
            term_id=recovery_term,
            interval=None,
            views=[],
            unit_ids=[],
            status="not_evaluable",
            usable_fraction=0.0,
            reason_codes=["offline_recovery_context_unavailable"],
        )
        builder.add_finding(
            evidence_id="E-RECOVERY-CONTEXT",
            family="termination_recovery",
            term_id=recovery_term,
            assertion_level="model_candidate",
            status="not_evaluable",
            role="limitation",
            temporal_context="unknown",
            opportunity_id=opportunity,
            model_uncertainty=1.0,
        )

    for family, term_id, evidence_id, reason in (
        (
            "morphology",
            "deterministic_morphology_candidate",
            "E-MORPHOLOGY-UNQUALIFIED",
            "morphology_producer_not_qualified",
        ),
        (
            "high_frequency",
            "deterministic_high_frequency_candidate",
            "E-HIGH-FREQUENCY-UNQUALIFIED",
            "high_frequency_producer_or_bandwidth_not_qualified",
        ),
    ):
        opportunity = builder.opportunity(
            family=family,
            term_id=term_id,
            interval=None,
            views=[],
            unit_ids=[],
            status="not_evaluable",
            usable_fraction=0.0,
            reason_codes=[reason],
        )
        builder.add_finding(
            evidence_id=evidence_id,
            family=family,
            term_id=term_id,
            assertion_level="model_candidate",
            status="not_evaluable",
            role="limitation",
            temporal_context="unknown",
            opportunity_id=opportunity,
            model_uncertainty=1.0,
        )

    family_opportunities: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in builder.opportunities:
        family_opportunities[str(row["family"])].append(row)
    feature_evaluability = []
    for family in _FAMILIES:
        rows = family_opportunities[family]
        statuses = {str(row["status"]) for row in rows}
        if "sufficient" in statuses:
            status, reasons = "available", []
        elif "limited" in statuses:
            status = "limited"
            reasons = sorted(
                {str(reason) for row in rows for reason in row["reason_codes"]}
            ) or ["family_opportunity_limited"]
        else:
            status = "not_evaluable"
            reasons = sorted(
                {str(reason) for row in rows for reason in row["reason_codes"]}
            ) or ["family_opportunity_unavailable"]
        feature_evaluability.append(
            {
                "family": family,
                "status": status,
                "reason_codes": reasons,
                "evaluation_opportunity_ids": [
                    str(row["evaluation_opportunity_id"]) for row in rows
                ],
            }
        )

    finding_by_id = {str(row["evidence_id"]): row for row in builder.findings}
    per_unit_involvement = []
    involvement_by_unit: dict[str, dict[str, Any]] = {}
    onset_name_to_index = (
        {causal.unit_ids[index]: index for index in onset_units}
        if causal is not None
        else {}
    )
    recruitment_opportunity_status = next(
        str(row["status"])
        for row in builder.opportunities
        if row["evaluation_opportunity_id"] == recruitment_opportunity
    )
    for unit_id, unit_type in zip(quality_view.unit_ids, quality_view.unit_types):
        wire_type = "lead" if unit_type == "lead" else "electrode"
        if unit_id in onset_name_to_index and onset_interval is not None:
            index = onset_name_to_index[unit_id]
            window_index = causal_changes[index]
            interval = _time_interval(
                float(causal_grid.recording_starts[window_index]),
                float(causal_grid.recording_stops[window_index]),
                resolution,
            )
            evidence_ids = [
                item
                for item in ("E-ONSET-CHANGE", field_evidence_id)
                if item is not None and item in finding_by_id
            ]
            status = "present"
        elif unit_id in late_evidence_by_unit and offline is not None:
            index = offline.unit_ids.index(unit_id)
            window_index = offline_changes[index]
            interval = _time_interval(
                float(offline_grid.recording_starts[window_index]),
                float(offline_grid.recording_stops[window_index]),
                resolution,
            )
            evidence_ids = [late_evidence_by_unit[unit_id]]
            status = "present"
        elif montage_eligible[unit_id] and recruitment_opportunity_status in {
            "sufficient",
            "limited",
        }:
            interval = None
            evidence_ids = []
            status = "uncertain"
        else:
            interval = None
            evidence_ids = []
            status = "not_evaluable"
        row = {
            "unit_type": wire_type,
            "unit_id": unit_id,
            "status": status,
            "interval": interval,
            "evaluation_opportunity_id": recruitment_opportunity,
            "sensitivity_receipt_id": None,
            "evidence_ids": evidence_ids,
        }
        per_unit_involvement.append(row)
        involvement_by_unit[unit_id] = row

    involvement_order = []
    if onset_units and late_units and causal is not None and offline is not None:
        source_id = causal.unit_ids[onset_units[0]]
        source = involvement_by_unit[source_id]
        for ordinal, target_index in enumerate(late_units, start=1):
            target_id = offline.unit_ids[target_index]
            target = involvement_by_unit[target_id]
            if source["interval"] is None or target["interval"] is None:
                continue
            delay_lower = float(target["interval"]["lower"]) - float(
                source["interval"]["upper"]
            )
            delay_upper = float(target["interval"]["upper"]) - float(
                source["interval"]["lower"]
            )
            relation_status = (
                "precedes" if delay_lower > resolution + 1e-9 else "order_unresolved"
            )
            involvement_order.append(
                {
                    "relation_id": f"INVREL-{ordinal:03d}",
                    "from_type": source["unit_type"],
                    "from_id": source_id,
                    "to_type": target["unit_type"],
                    "to_id": target_id,
                    "delay_interval": _time_interval(
                        delay_lower, delay_upper, resolution
                    ),
                    "relation_status": relation_status,
                    "assertion_level": "model_candidate",
                    "evidence_ids": [late_evidence_by_unit[target_id]],
                }
            )

    producer_receipts = _provider_receipts(producer_policy_sha)
    relation_receipt = producer_receipts[1]
    hypothesis_id = (
        "HYP-" + _canonical_sha256([event_id, adaptive["window_receipt_id"]])[:20]
    )
    candidate_scores: list[dict[str, Any]] = []
    hypothesis_relations: list[dict[str, Any]] = []
    if field_evidence_id is not None and onset_units and causal_scores is not None:
        axis_values: dict[str, dict[str, float]] = defaultdict(dict)
        for index in onset_units[: policy.maximum_ranked_candidates]:
            unit_id = causal.unit_ids[index]
            score = float(causal_scores[causal_changes[index], index])
            axis_values["lead"][unit_id] = score
            axis_values["laterality"][laterality[unit_id]] = max(
                axis_values["laterality"].get(laterality[unit_id], -math.inf),
                score,
            )
            axis_values["region"][region[unit_id]] = max(
                axis_values["region"].get(region[unit_id], -math.inf), score
            )
        onset_lateralities = {
            laterality[causal.unit_ids[index]] for index in onset_units
        }
        phenotype = (
            "bilateral_synchronous_or_rapid_bilateralization_ambiguous"
            if "left" in onset_lateralities and "right" in onset_lateralities
            else "focal"
        )
        axis_values["phenotype"][phenotype] = max(axis_values["lead"].values())
        for axis in ("phenotype", "laterality", "region", "lead"):
            for rank, (candidate_id, score) in enumerate(
                sorted(
                    axis_values[axis].items(),
                    key=lambda item: (-item[1], item[0]),
                ),
                start=1,
            ):
                relation_id = (
                    "HREL-"
                    + _canonical_sha256(
                        [hypothesis_id, axis, candidate_id, field_evidence_id]
                    )[:20]
                )
                hypothesis_relations.append(
                    {
                        "relation_id": relation_id,
                        "hypothesis_id": hypothesis_id,
                        "axis": axis,
                        "candidate_type": axis,
                        "candidate_id": candidate_id,
                        "relation": "supports",
                        "evidence_ids": [field_evidence_id],
                        "producer_receipt_id": relation_receipt["receipt_id"],
                        "policy_sha256": producer_policy_sha,
                    }
                )
                candidate_scores.append(
                    {
                        "rank": rank,
                        "axis": axis,
                        "candidate_type": axis,
                        "candidate_id": candidate_id,
                        "score": score,
                        "score_semantics": "uncalibrated_ranking_score",
                        "calibration_receipt_id": None,
                        "supporting_relation_ids": [relation_id],
                        "contradictory_relation_ids": [],
                    }
                )
        localization_status = "phenotype_only"
        selected_resolution = "phenotype_only"
        hypothesis_reasons = [
            "event_not_clinically_qualified_spatial_scores_research_only"
        ]
        hypothesis_receipt_id = relation_receipt["receipt_id"]
    else:
        phenotype = (
            "scalp_onset_nonlocalizable"
            if causal is not None and onset_localization_eligible
            else "not_evaluable"
        )
        localization_status = (
            "nonlocalizable"
            if phenotype == "scalp_onset_nonlocalizable"
            else "not_evaluable"
        )
        selected_resolution = "none"
        hypothesis_reasons = [
            "stable_causal_spatial_onset_not_resolved"
            if phenotype == "scalp_onset_nonlocalizable"
            else causal_unavailable_reason
        ]
        hypothesis_receipt_id = None

    if onset_evidence_ids:
        event_status = "unqualified_candidate"
        event_reasons = ["deterministic_candidate_not_clinically_qualified"]
        event_support = onset_evidence_ids
    else:
        event_status = "not_evaluable"
        event_reasons = ["qualified_event_evidence_unavailable"]
        if causal_response_unqualified:
            event_reasons.append(causal_unavailable_reason)
        event_support = []

    search_start, search_stop = (
        float(search["envelope_interval_recording_seconds"][0]),
        float(search["envelope_interval_recording_seconds"][1]),
    )
    search_interval = (
        min(search_start, final_interval[0]),
        max(search_stop, final_interval[1]),
    )
    if onset_interval is not None:
        onset_boundary = {
            "status": "observed",
            "interval": _time_interval(
                onset_interval[0], onset_interval[1], resolution
            ),
            "censoring_reason_codes": [],
        }
    elif adaptive["censoring"]["left"]:
        onset_boundary = {
            "status": "censored",
            "interval": None,
            "censoring_reason_codes": ["adaptive_window_left_censored"],
        }
    else:
        onset_boundary = {
            "status": "indeterminate",
            "interval": None,
            "censoring_reason_codes": ["causal_onset_boundary_not_resolved"],
        }
    if offset_anchor is not None:
        half = resolution / 2.0
        offset_boundary = {
            "status": "observed",
            "interval": _time_interval(
                max(final_interval[0], offset_anchor - half),
                min(final_interval[1], offset_anchor + half),
                resolution,
            ),
            "censoring_reason_codes": [],
        }
    elif adaptive["censoring"]["right"]:
        offset_boundary = {
            "status": "censored",
            "interval": None,
            "censoring_reason_codes": ["adaptive_window_right_censored"],
        }
    else:
        offset_boundary = {
            "status": "not_observed",
            "interval": None,
            "censoring_reason_codes": [],
        }

    native_rates = [
        float(row["sample_rate_numerator"]) / float(row["sample_rate_denominator"])
        for row in canonical["channels"]
        if row["observed"]
    ]
    registry_bindings = deepcopy(DEFAULT_EVENT_FINDINGS_V2_REGISTRY_BINDINGS)
    host_registries = (
        registry_bindings
        if trusted_registry_bindings is None
        else trusted_registry_bindings
    )
    payload = {
        "schema_version": "event_eeg_findings_v2",
        "event_id": event_id,
        "provenance": {
            "record_id": _identifier(canonical["recording_id"], "recording_id"),
            "canonical_signal_sha256": canonical["source_signal_sha256"],
            "preprocess_receipt_id": _identifier(
                canonical["canonical_signal_id"], "canonical_signal_id"
            ),
            "model_ids": [
                DETERMINISTIC_EVENT_FINDINGS_V2_METHOD_ID,
                str(search["method_id"]),
                str(search["search_receipt_id"]),
                "SEARCHSHA-" + _canonical_sha256(search),
                str(adaptive["method_id"]),
                str(adaptive["window_receipt_id"]),
                "WINDOWSHA-" + _canonical_sha256(adaptive),
            ],
            "policy_sha256": producer_policy_sha,
            "inference_exclusions": {
                "edf_annotations_used": False,
                "excel_used": False,
                "doctor_labels_used": False,
                "clinical_text_used": False,
                "patient_metadata_used": False,
                "video_used": False,
                "ecg_emg_eog_used": False,
                "sleep_staging_used": False,
                "provocation_used": False,
            },
        },
        "coordinates": {
            "system": "recording_relative_seconds",
            "recording_duration_seconds": float(
                canonical["recording_duration_seconds"]
            ),
            "model_sample_rate_hz": float(quality_view.sampling_rate_hz),
            "native_sample_rate_hz": max(native_rates),
        },
        "registry_bindings": registry_bindings,
        "montage": montage,
        "window": {
            "search_interval": [search_interval[0], search_interval[1]],
            "final_interval": [final_interval[0], final_interval[1]],
            "protection_zone": {
                "protection_zone_id": protection_id,
                "interval": [protection[0], protection[1]],
                "policy_sha256": producer_policy_sha,
            },
            "onset_boundary": onset_boundary,
            "offset_boundary": offset_boundary,
            "state_posterior_status": (
                "limited" if state_segments else "not_evaluable"
            ),
            "state_segments": state_segments,
            "state_path_receipt_id": (
                str(adaptive["window_receipt_id"]) if state_segments else None
            ),
            "left_censored": bool(adaptive["censoring"]["left"]),
            "right_censored": bool(adaptive["censoring"]["right"]),
            "search_cap_censored": bool(
                adaptive["censoring"]["left"] or adaptive["censoring"]["right"]
            ),
            "merge_split_status": "single_event",
        },
        "context": {
            "queried_intervals": [[search_interval[0], search_interval[1]]],
            "local_background_intervals": [list(baseline)]
            if background_available and baseline is not None
            else [],
            "distant_background_intervals": [],
            "background_status": "available" if background_available else "unavailable",
            "background_bank_id": background_bank_id,
            "selection_receipt_id": background_selection_id,
            "selection_scope": "eeg_detector_quality_only",
            "contamination_risk": 0.15 if background_available else 1.0,
        },
        "quality": {
            "usable_fraction": usable_fraction,
            "per_unit": per_unit_quality,
            "artifact_intervals": artifacts,
            "feature_evaluability": feature_evaluability,
        },
        "event_qualification": {
            "status": event_status,
            "qualification_receipt_id": None,
            "supporting_evidence_ids": event_support,
            "reason_codes": event_reasons,
        },
        "producer_receipts": producer_receipts,
        "calibration_receipts": [],
        "capability_qualification_receipts": [],
        "sensitivity_receipts": [],
        "term_decision_receipts": [],
        "evaluation_opportunities": builder.opportunities,
        "findings": builder.findings,
        "scalp_onset_hypothesis": {
            "hypothesis_id": hypothesis_id,
            "layer": "research_ai_hypothesis",
            "claim_boundary": "research_scalp_eeg_onset_candidate_not_cortical_soz",
            "event_id": event_id,
            "localization_status": localization_status,
            "selected_resolution": selected_resolution,
            "phenotype": phenotype,
            "candidate_scores": candidate_scores,
            "per_unit_involvement": per_unit_involvement,
            "involvement_order": involvement_order,
            "reason_codes": hypothesis_reasons,
            "model_receipt_id": hypothesis_receipt_id,
        },
        "hypothesis_evidence_relations": hypothesis_relations,
        "waveform_evidence": builder.waveforms,
        "limitations": [
            {
                "code": "scalp_eeg_onset_candidate_only",
                "scope": "clinical_claim",
                "text_zh": "仅输出头皮 EEG 起始候选，不等同于皮层 SOZ、致痫区或临床诊断。",
            },
            {
                "code": "clinical_terms_not_qualified",
                "scope": "finding",
                "text_zh": "本确定性模块不自动确认 spike、IED、ACNS 演变或电图发作等临床术语。",
            },
            {
                "code": "negative_findings_withheld",
                "scope": "finding",
                "text_zh": "未配置患者独立的灵敏度资格回执，因此不输出 absence 阴性断言。",
            },
            {
                "code": "reference_specific_spatial_candidates",
                "scope": "spatial",
                "text_zh": "空间排序为参考依赖的研究候选；尚未由跨参考场形、极性及稳定性模块确认。",
            },
        ],
        "migration": None,
    }
    trusted_producers = {
        str(row["receipt_id"]): deepcopy(row) for row in producer_receipts
    }
    return validate_event_eeg_findings_v2_payload(
        payload,
        trusted_producer_receipts=trusted_producers,
        trusted_registry_bindings=host_registries,
    )


__all__ = [
    "DEFAULT_EVENT_FINDINGS_V2_REGISTRY_BINDINGS",
    "DETERMINISTIC_EVENT_FINDINGS_V2_METHOD_ID",
    "DETERMINISTIC_EVENT_HYPOTHESIS_V2_METHOD_ID",
    "produce_deterministic_event_eeg_findings_v2",
]
