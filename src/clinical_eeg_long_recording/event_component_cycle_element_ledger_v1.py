"""Replayable native S06 component/cycle/element instance ledgers.

The frozen Findings-v1.2 semantic contract requires explicit instance ledgers
for S06.  A summary frequency, FFT maximum, autocorrelation maximum, or one
aggregate periodicity score is not a substitute for those ledgers.  This
module closes that engineering gap using only two existing signal-native
artifacts:

* ``deterministic_eeg_element_interval_candidate_v1`` supplies individually
  bounded waveform elements and every successive inter-element interval; and
* ``event_morphology_primitive_supervision_v1`` supplies numerical morphology
  measurements for exactly those element intervals from an instantaneous,
  unclipped, physical-volts view.

The resulting instances remain measurements or research candidates.  They do
not qualify a clinical rhythmic/periodic pattern, spike, sharp wave, IED,
seizure, evolution, onset, SOZ, EZ, or report statement.  A bipolar output is
kept as one opaque whole lead and is never split into endpoint facts.

Ordinary signal/QC/bandwidth insufficiency is serialized as ``not_evaluable``
with ``count=None``.  It is never interpreted as zero or absence.  The receipt
embeds and validates its complete typed sources and is deterministically
rebuilt from them, so mutations of a projected interval, measurement, raw
support, censoring state, or authorization fail even if the outer hash is
recomputed.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import re
from typing import Any, Final, Mapping, Sequence

from .canonical_signal_views import validate_canonical_signal_receipt
from .deterministic_event_morphology_primitives_v1 import (
    EVENT_MORPHOLOGY_PRIMITIVE_TARGET_SPECS,
    EventMorphologyPrimitiveQuery,
    EventMorphologyPrimitiveViewInput,
    materialize_event_morphology_primitive_supervision_v1,
    validate_event_morphology_primitive_supervision_v1,
)
from .deterministic_periodicity_candidate import (
    validate_deterministic_periodicity_candidate,
)


EVENT_COMPONENT_CYCLE_ELEMENT_LEDGER_SCHEMA_VERSION: Final[str] = (
    "clinical_eeg_event_component_cycle_element_ledger_v1"
)
EVENT_COMPONENT_CYCLE_ELEMENT_LEDGER_METHOD_ID: Final[str] = (
    "EVENT-COMPONENT-CYCLE-ELEMENT-LEDGER-V1"
)
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_SHA_RE = re.compile(r"^[a-f0-9]{64}$")
_TOL: Final[float] = 1e-9

_FIREWALL: Final[dict[str, bool]] = {
    "eeg_samples_used": True,
    "allowlisted_acquisition_metadata_used": True,
    "edf_annotations_used": False,
    "excel_used": False,
    "doctor_labels_used": False,
    "clinical_text_or_report_used": False,
    "patient_metadata_used": False,
    "video_or_behavior_used": False,
    "sleep_or_activation_labels_used": False,
    "ecg_emg_eog_used": False,
    "qwen_or_other_llm_used": False,
}
_AUTHORIZATION: Final[dict[str, bool | str | list[str]]] = {
    "event_card_slot_id": "S06_RHYTHMICITY_PERIODICITY",
    "projection_scope": "numerical_component_cycle_element_instances_only",
    "whole_bipolar_lead_identity_required": True,
    "bipolar_endpoint_fact_projection_authorized": False,
    "clinical_rhythmic_or_periodic_term_authorized": False,
    "spike_sharp_wave_or_ied_term_authorized": False,
    "evolution_or_seizure_term_authorized": False,
    "negative_clinical_assertion_authorized": False,
    "onset_claim_authorized": False,
    "soz_or_ez_claim_authorized": False,
    "report_text_authorized": False,
    "report_eligible_term_allowlist": [],
}
_CENSORING_POLICY: Final[dict[str, object]] = {
    "policy_id": "S06-ANALYSIS-SUPPORT-CENSORING-POLICY-V1",
    "analysis_support_boundary_guard_source": (
        "source_periodicity_candidate.policy.boundary_guard_seconds"
    ),
    "boundary_touch_semantics": "possible_analysis_support_censoring",
    "no_boundary_touch_semantics": "not_observed_within_support",
    "no_boundary_touch_proves_uncensored": False,
    "record_edge_receipt_available": False,
    "record_edge_censoring_status": "not_evaluable",
}


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _self_hash(value: Mapping[str, object], field: str) -> str:
    body = deepcopy(dict(value))
    body.pop(field, None)
    return _canonical_sha256(body)


def _identifier(value: object, context: str) -> str:
    if type(value) is not str or _ID_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a canonical identifier")
    return value


def _sha(value: object, context: str) -> str:
    if type(value) is not str or _SHA_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _finite(value: object, context: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{context} must be >= {minimum}")
    return result


def _interval(value: Sequence[float], context: str) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{context} must contain two values")
    start = _finite(value[0], f"{context}[0]", minimum=0.0)
    stop = _finite(value[1], f"{context}[1]", minimum=0.0)
    if stop <= start + _TOL:
        raise ValueError(f"{context} must have positive duration")
    return start, stop


def _reject_nonfinite(value: object, context: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{context} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_nonfinite(item, f"{context}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nonfinite(item, f"{context}[{index}]")


def _sorted_reasons(values: Sequence[object]) -> list[str]:
    result = sorted({str(item) for item in values})
    if any(not item or item != item.strip() for item in result):
        raise ValueError("reason codes must be non-empty and trimmed")
    return result


def _candidate_sort_key(value: Mapping[str, Any]) -> tuple[object, ...]:
    interval = _interval(value["requested_recording_interval"], "candidate interval")
    return (
        str(value["analysis_unit_id"]),
        interval[0],
        interval[1],
        str(value["candidate_id"]),
    )


def _validate_candidates(values: Sequence[object]) -> list[dict[str, Any]]:
    if isinstance(values, (str, bytes)):
        raise TypeError("periodicity_candidates must be a sequence")
    result = [validate_deterministic_periodicity_candidate(row) for row in values]
    candidate_ids = [str(row["candidate_id"]) for row in result]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("periodicity candidate IDs must be unique")
    return sorted(result, key=_candidate_sort_key)


def _source_identity(
    *,
    morphology: Mapping[str, Any] | None,
    canonical: Mapping[str, Any] | None,
    analysis_interval_seconds: Sequence[float] | None,
) -> dict[str, Any]:
    if morphology is not None:
        analysis = _interval(
            morphology["analysis_interval_seconds"],
            "morphology analysis_interval_seconds",
        )
        if analysis_interval_seconds is not None and _interval(
            analysis_interval_seconds, "analysis_interval_seconds"
        ) != analysis:
            raise ValueError("S06/morphology analysis interval drifted")
        return {
            "event_id": str(morphology["event_id"]),
            "recording_id": str(morphology["recording_id"]),
            "canonical_signal_id": str(morphology["canonical_signal_id"]),
            "canonical_receipt_sha256": str(
                morphology["canonical_receipt_sha256"]
            ),
            "source_signal_sha256": str(morphology["source_signal_sha256"]),
            "analysis_interval_seconds": list(analysis),
        }
    if canonical is None or analysis_interval_seconds is None:
        raise ValueError(
            "canonical_receipt and analysis_interval_seconds are required when "
            "the morphology receipt is absent"
        )
    analysis = _interval(analysis_interval_seconds, "analysis_interval_seconds")
    if analysis[1] > float(canonical["recording_duration_seconds"]) + _TOL:
        raise ValueError("S06 analysis interval exceeds the recording")
    return {
        "event_id": None,
        "recording_id": str(canonical["recording_id"]),
        "canonical_signal_id": str(canonical["canonical_signal_id"]),
        "canonical_receipt_sha256": str(canonical["receipt_sha256"]),
        "source_signal_sha256": str(canonical["source_signal_sha256"]),
        "analysis_interval_seconds": list(analysis),
    }


def _morphology_rows_by_element_key(
    morphology: Mapping[str, Any] | None,
) -> dict[tuple[str, float, float], Mapping[str, Any]]:
    if morphology is None:
        return {}
    result: dict[tuple[str, float, float], Mapping[str, Any]] = {}
    for row in morphology["rows"]:
        source = row["source_binding"]
        interval = _interval(
            source["requested_recording_interval_seconds"],
            "morphology requested interval",
        )
        key = (str(source["unit_id"]), interval[0], interval[1])
        if key in result:
            raise ValueError("morphology receipt contains duplicate element support")
        result[key] = row
    return result


def _candidate_dependency_by_element(
    candidate: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in candidate["raw_sample_dependencies"]:
        if row["dependency_role"] != "element_waveform":
            continue
        element_id = str(row["element_id"])
        if element_id in result:
            raise ValueError("candidate has duplicate element raw dependencies")
        result[element_id] = row["raw_sample_dependency"]
    return result


def _unit_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    source = row["source_binding"]
    unit_type = str(source["unit_type"])
    unit_id = str(source["unit_id"])
    return {
        "unit_id": unit_id,
        "unit_type": unit_type,
        "identity_scope": (
            "whole_bipolar_or_other_lead"
            if unit_type == "lead"
            else "typed_electrode"
            if unit_type == "electrode"
            else "whole_typed_analysis_unit"
        ),
        "whole_output_unit_identity_preserved": True,
        "bipolar_endpoint_fact_projection_authorized": False,
    }


def _element_censoring(
    candidate: Mapping[str, Any], element: Mapping[str, Any]
) -> dict[str, Any]:
    support = _interval(
        candidate["source_binding"]["actual_recording_interval"],
        "candidate actual interval",
    )
    interval = _interval(element["recording_interval"], "element interval")
    guard = _finite(
        candidate["policy"]["boundary_guard_seconds"],
        "boundary_guard_seconds",
        minimum=0.0,
    )
    left_touch = interval[0] <= support[0] + guard + _TOL
    right_touch = interval[1] >= support[1] - guard - _TOL
    state = lambda touch: (
        "possible_analysis_support_censoring"
        if touch
        else "not_observed_within_support"
    )
    return {
        "analysis_support_interval_seconds": list(support),
        "boundary_guard_seconds": guard,
        "left_boundary_status": state(left_touch),
        "right_boundary_status": state(right_touch),
        "record_edge_status": "not_evaluable",
        "record_edge_receipt_used": False,
        "no_boundary_touch_proves_uncensored": False,
    }


def _component_measurements(
    morphology: Mapping[str, Any], row: Mapping[str, Any]
) -> list[dict[str, Any]]:
    result = []
    mask = row["opportunity"]["target_value_mask"]
    reasons = row["opportunity"]["target_reason_codes"]
    for index, target in enumerate(morphology["target_registry"]):
        result.append(
            {
                "target_name": str(target["target_name"]),
                "unit_id": str(target["unit_id"]),
                "opportunity_family": str(target["opportunity_family"]),
                "value": float(row["values"][index]) if mask[index] else None,
                "value_available": bool(mask[index]),
                "reason_codes": deepcopy(reasons[index]),
            }
        )
    return result


def _dominant_excursion_subinterval(
    row: Mapping[str, Any], measurements: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    by_name = {str(item["target_name"]): item for item in measurements}
    required = (
        "dominant_excursion_latency_seconds",
        "dominant_excursion_rise_half_height_seconds",
        "dominant_excursion_fall_half_height_seconds",
    )
    unavailable = [name for name in required if not by_name[name]["value_available"]]
    if unavailable:
        return {
            "status": "not_evaluable",
            "recording_interval_seconds": None,
            "peak_recording_seconds": None,
            "reason_codes": _sorted_reasons(
                [
                    reason
                    for name in unavailable
                    for reason in by_name[name]["reason_codes"]
                ]
                or ["dominant_excursion_geometry_not_measurable"]
            ),
            "clinical_name_authorized": False,
        }
    source_interval = _interval(
        row["source_binding"]["recording_interval_seconds"],
        "component source interval",
    )
    peak = source_interval[0] + float(
        by_name["dominant_excursion_latency_seconds"]["value"]
    )
    start = peak - float(
        by_name["dominant_excursion_rise_half_height_seconds"]["value"]
    )
    stop = peak + float(
        by_name["dominant_excursion_fall_half_height_seconds"]["value"]
    )
    start = max(source_interval[0], start)
    stop = min(source_interval[1], stop)
    if stop <= start + _TOL:
        return {
            "status": "not_evaluable",
            "recording_interval_seconds": None,
            "peak_recording_seconds": None,
            "reason_codes": ["dominant_excursion_subinterval_degenerate"],
            "clinical_name_authorized": False,
        }
    return {
        "status": "measured",
        "recording_interval_seconds": [start, stop],
        "peak_recording_seconds": peak,
        "reason_codes": [],
        "clinical_name_authorized": False,
    }


def _ledger_block(
    instances: Sequence[Mapping[str, Any]],
    *,
    source_candidates: Sequence[Mapping[str, Any]],
    instance_statuses: Sequence[str] = (),
    empty_reason: str,
) -> dict[str, Any]:
    not_evaluable = [
        row
        for row in source_candidates
        if row["qualification_status"] == "not_evaluable"
    ]
    reasons = _sorted_reasons(
        [
            f"source_candidate_not_evaluable:{reason}"
            for row in not_evaluable
            for reason in row["reason_codes"]
        ]
    )
    if not instances:
        status = "not_evaluable"
        count: int | None = None
        opportunity = "not_evaluable"
        reasons = _sorted_reasons([*reasons, empty_reason])
    else:
        partial_instance = any(item != "measured" for item in instance_statuses)
        status = "partially_measured" if not_evaluable or partial_instance else "measured"
        count = len(instances)
        opportunity = "limited" if status == "partially_measured" else "sufficient"
    body: dict[str, Any] = {
        "status": status,
        "count": count,
        "instances": [deepcopy(dict(row)) for row in instances],
        "opportunity": {
            "status": opportunity,
            "reason_codes": reasons,
            "not_evaluable_is_negative": False,
            "absence_inference_authorized": False,
        },
        "ledger_sha256": "",
    }
    body["ledger_sha256"] = _self_hash(body, "ledger_sha256")
    return body


def _body_from_sources(
    *,
    identity: Mapping[str, Any],
    morphology: Mapping[str, Any] | None,
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    event_id = _identifier(identity["event_id"], "event_id")
    analysis = _interval(
        identity["analysis_interval_seconds"], "analysis_interval_seconds"
    )
    signal_sha = _sha(identity["source_signal_sha256"], "source_signal_sha256")
    for index, candidate in enumerate(candidates):
        if candidate["event_id"] != event_id:
            raise ValueError(f"periodicity_candidates[{index}] crosses event identity")
        if candidate["source_binding"]["canonical_signal_sha256"] != signal_sha:
            raise ValueError(
                f"periodicity_candidates[{index}] crosses canonical signal identity"
            )
        requested = _interval(
            candidate["requested_recording_interval"],
            f"periodicity_candidates[{index}].requested_recording_interval",
        )
        if requested[0] < analysis[0] - _TOL or requested[1] > analysis[1] + _TOL:
            raise ValueError("periodicity candidate lies outside S06 analysis interval")

    candidate_only = [
        row for row in candidates if row["qualification_status"] == "candidate_only"
    ]
    if candidate_only and morphology is None:
        raise ValueError(
            "candidate-only elements require native morphology component rows"
        )
    morphology_rows = _morphology_rows_by_element_key(morphology)
    expected_keys: set[tuple[str, float, float]] = set()
    for candidate in candidate_only:
        for element in candidate["occurrence"]["element_roster"]:
            interval = _interval(element["recording_interval"], "element interval")
            expected_keys.add(
                (str(candidate["analysis_unit_id"]), interval[0], interval[1])
            )
    if morphology is not None and set(morphology_rows) != expected_keys:
        missing = sorted(expected_keys - set(morphology_rows))
        extra = sorted(set(morphology_rows) - expected_keys)
        raise ValueError(
            "native morphology rows do not close the exact S06 element roster: "
            f"missing={missing}, extra={extra}"
        )

    elements: list[dict[str, Any]] = []
    cycles: list[dict[str, Any]] = []
    components: list[dict[str, Any]] = []
    component_statuses: list[str] = []
    for candidate in candidate_only:
        unit_id = str(candidate["analysis_unit_id"])
        dependency_by_element = _candidate_dependency_by_element(candidate)
        source_elements = candidate["occurrence"]["element_roster"]
        source_intervals = candidate["successive_interval_profile"][
            "successive_intervals"
        ]
        source_element_by_id = {
            str(row["element_id"]): row for row in source_elements
        }
        projected_element_by_source_id: dict[str, Mapping[str, Any]] = {}
        for source_element in source_elements:
            source_element_id = str(source_element["element_id"])
            if source_element_id not in dependency_by_element:
                raise ValueError("element ledger lacks a raw-sample dependency")
            element_interval = _interval(
                source_element["recording_interval"], "element interval"
            )
            key = (unit_id, element_interval[0], element_interval[1])
            morphology_row = morphology_rows[key]
            unit = _unit_identity(morphology_row)
            if unit["unit_id"] != unit_id:
                raise ValueError("whole analysis-unit identity drifted")
            dependency = dependency_by_element[source_element_id]
            dependency_sha = _sha(
                dependency["dependency_sha256"], "element dependency_sha256"
            )
            element_body = {
                "source_candidate_id": str(candidate["candidate_id"]),
                "source_candidate_sha256": str(candidate["candidate_sha256"]),
                "source_element_id": source_element_id,
                "assertion_level": "model_candidate",
                "qualification_status": "candidate_only",
                "analysis_unit": unit,
                "ordinal_within_candidate": int(source_element["ordinal"]),
                "recording_interval_seconds": list(element_interval),
                "peak_recording_seconds": float(
                    source_element["peak_recording_seconds"]
                ),
                "duration_seconds": float(source_element["duration_seconds"]),
                "peak_amplitude_uv": float(source_element["peak_amplitude_uv"]),
                "polarity": str(source_element["polarity"]),
                "merged_proposal_count": int(
                    source_element["merged_proposal_count"]
                ),
                "raw_sample_dependency": deepcopy(dependency),
                "raw_sample_dependency_sha256": dependency_sha,
                "censoring": _element_censoring(candidate, source_element),
                "spectral_peak_used": False,
                "autocorrelation_used": False,
                "clinical_term_qualification_authorized": False,
                "onset_support_eligible": False,
                "soz_support_eligible": False,
            }
            element_instance_id = "S06EL-" + _canonical_sha256(element_body)[:24]
            element_instance = {
                "element_instance_id": element_instance_id,
                **element_body,
            }
            elements.append(element_instance)
            projected_element_by_source_id[source_element_id] = element_instance

            measurements = _component_measurements(morphology, morphology_row)
            component_body = {
                "source_candidate_id": str(candidate["candidate_id"]),
                "source_element_id": source_element_id,
                "source_element_instance_id": element_instance_id,
                "source_morphology_row_id": str(morphology_row["row_id"]),
                "source_morphology_row_binding_sha256": str(
                    morphology_row["row_binding_sha256"]
                ),
                "assertion_level": "measured",
                "qualification_status": str(
                    morphology_row["opportunity"]["status"]
                ),
                "analysis_unit": unit,
                "recording_interval_seconds": deepcopy(
                    morphology_row["source_binding"][
                        "recording_interval_seconds"
                    ]
                ),
                "target_measurements": measurements,
                "aggregate_opportunity_reason_codes": deepcopy(
                    morphology_row["opportunity"][
                        "aggregate_opportunity_reason_codes"
                    ]
                ),
                "dominant_excursion_subinterval": (
                    _dominant_excursion_subinterval(morphology_row, measurements)
                ),
                "raw_sample_intervals": deepcopy(
                    morphology_row["source_binding"]["raw_sample_intervals"]
                ),
                "raw_sample_support_sha256": _canonical_sha256(
                    morphology_row["source_binding"]["raw_sample_intervals"]
                ),
                "whole_output_unit_identity_preserved": True,
                "bipolar_endpoint_fact_projection_authorized": False,
                "clinical_name_authorized": False,
                "onset_support_eligible": False,
                "soz_support_eligible": False,
            }
            component_instance_id = "S06COMP-" + _canonical_sha256(
                component_body
            )[:24]
            components.append(
                {
                    "component_instance_id": component_instance_id,
                    **component_body,
                }
            )
            component_statuses.append(
                str(morphology_row["opportunity"]["status"])
            )

        for ordinal, source_cycle in enumerate(source_intervals, start=1):
            from_source_id = str(source_cycle["from_element_id"])
            to_source_id = str(source_cycle["to_element_id"])
            previous = source_element_by_id[from_source_id]
            current = source_element_by_id[to_source_id]
            from_projected = projected_element_by_source_id[from_source_id]
            to_projected = projected_element_by_source_id[to_source_id]
            onset_seconds = float(source_cycle["onset_to_onset_seconds"])
            if onset_seconds <= 0.0:
                raise ValueError("cycle interval must be positive")
            cycle_body = {
                "source_candidate_id": str(candidate["candidate_id"]),
                "source_interval_id": str(source_cycle["interval_id"]),
                "assertion_level": "measured",
                "qualification_status": "measured",
                "analysis_unit": deepcopy(from_projected["analysis_unit"]),
                "ordinal_within_candidate": ordinal,
                "from_element_instance_id": str(
                    from_projected["element_instance_id"]
                ),
                "to_element_instance_id": str(to_projected["element_instance_id"]),
                "recording_interval_seconds": [
                    float(previous["recording_interval"][0]),
                    float(current["recording_interval"][0]),
                ],
                "onset_to_onset_samples": int(
                    source_cycle["onset_to_onset_samples"]
                ),
                "onset_to_onset_seconds": onset_seconds,
                "peak_to_peak_samples": int(source_cycle["peak_to_peak_samples"]),
                "peak_to_peak_seconds": float(
                    source_cycle["peak_to_peak_seconds"]
                ),
                "cycle_rate_hz": 1.0 / onset_seconds,
                "raw_support_references": [
                    {
                        "element_instance_id": str(
                            from_projected["element_instance_id"]
                        ),
                        "raw_sample_dependency_sha256": str(
                            from_projected["raw_sample_dependency_sha256"]
                        ),
                    },
                    {
                        "element_instance_id": str(
                            to_projected["element_instance_id"]
                        ),
                        "raw_sample_dependency_sha256": str(
                            to_projected["raw_sample_dependency_sha256"]
                        ),
                    },
                ],
                "spectral_peak_used": False,
                "autocorrelation_used": False,
                "clinical_term_qualification_authorized": False,
                "onset_support_eligible": False,
                "soz_support_eligible": False,
            }
            cycles.append(
                {
                    "cycle_instance_id": "S06CYCLE-"
                    + _canonical_sha256(cycle_body)[:24],
                    **cycle_body,
                }
            )

    elements.sort(
        key=lambda row: (
            row["analysis_unit"]["unit_id"],
            row["recording_interval_seconds"][0],
            row["element_instance_id"],
        )
    )
    cycles.sort(
        key=lambda row: (
            row["analysis_unit"]["unit_id"],
            row["recording_interval_seconds"][0],
            row["cycle_instance_id"],
        )
    )
    components.sort(
        key=lambda row: (
            row["analysis_unit"]["unit_id"],
            row["recording_interval_seconds"][0],
            row["component_instance_id"],
        )
    )
    source_not_evaluable_reasons = _sorted_reasons(
        [
            f"{row['analysis_unit_id']}:{reason}"
            for row in candidates
            if row["qualification_status"] == "not_evaluable"
            for reason in row["reason_codes"]
        ]
    )
    result = {
        "schema_version": EVENT_COMPONENT_CYCLE_ELEMENT_LEDGER_SCHEMA_VERSION,
        "method_id": EVENT_COMPONENT_CYCLE_ELEMENT_LEDGER_METHOD_ID,
        "event_id": event_id,
        "recording_id": str(identity["recording_id"]),
        "canonical_signal_id": str(identity["canonical_signal_id"]),
        "canonical_receipt_sha256": str(identity["canonical_receipt_sha256"]),
        "source_signal_sha256": signal_sha,
        "analysis_interval_seconds": list(analysis),
        "coordinate_system": "recording_relative_seconds",
        "event_card_slot_id": "S06_RHYTHMICITY_PERIODICITY",
        "source_morphology_primitive_receipt": (
            deepcopy(dict(morphology)) if morphology is not None else None
        ),
        "source_morphology_primitive_receipt_sha256": (
            str(morphology["receipt_sha256"]) if morphology is not None else None
        ),
        "source_periodicity_candidates": [deepcopy(dict(row)) for row in candidates],
        "source_periodicity_candidate_sha256s": [
            str(row["candidate_sha256"]) for row in candidates
        ],
        "source_candidate_opportunity": {
            "candidate_count": len(candidates),
            "candidate_only_count": len(candidate_only),
            "not_evaluable_count": len(candidates) - len(candidate_only),
            "not_evaluable_reason_codes": source_not_evaluable_reasons,
            "not_evaluable_is_negative": False,
        },
        "component_instance_ledger": _ledger_block(
            components,
            source_candidates=candidates,
            instance_statuses=component_statuses,
            empty_reason="no_evaluable_native_component_instances",
        ),
        "cycle_instance_ledger": _ledger_block(
            cycles,
            source_candidates=candidates,
            instance_statuses=["measured"] * len(cycles),
            empty_reason="no_evaluable_successive_cycle_instances",
        ),
        "element_instance_ledger": _ledger_block(
            elements,
            source_candidates=candidates,
            instance_statuses=["measured"] * len(elements),
            empty_reason="no_evaluable_bounded_element_instances",
        ),
        "censoring_policy": deepcopy(_CENSORING_POLICY),
        "firewall": deepcopy(_FIREWALL),
        "authorization": deepcopy(_AUTHORIZATION),
    }
    return result


def materialize_event_component_cycle_element_ledger_v1(
    morphology_primitive_receipt: object | None,
    periodicity_candidates: Sequence[object],
    *,
    canonical_receipt: object | None = None,
    analysis_interval_seconds: Sequence[float] | None = None,
    event_id: str | None = None,
) -> dict[str, Any]:
    """Compose a native S06 ledger from validated signal-derived receipts.

    When every periodicity source is ``not_evaluable``, the morphology receipt
    may be ``None``.  In that case canonical identity, analysis interval and
    ``event_id`` are required; the three ledgers remain empty with null counts.
    """

    morphology = (
        validate_event_morphology_primitive_supervision_v1(
            morphology_primitive_receipt
        )
        if morphology_primitive_receipt is not None
        else None
    )
    canonical = (
        validate_canonical_signal_receipt(canonical_receipt)
        if canonical_receipt is not None
        else None
    )
    candidates = _validate_candidates(periodicity_candidates)
    identity = _source_identity(
        morphology=morphology,
        canonical=canonical,
        analysis_interval_seconds=analysis_interval_seconds,
    )
    resolved_event_id = (
        str(morphology["event_id"])
        if morphology is not None
        else _identifier(event_id, "event_id")
    )
    if event_id is not None and _identifier(event_id, "event_id") != resolved_event_id:
        raise ValueError("explicit event_id crosses morphology event identity")
    identity["event_id"] = resolved_event_id
    result = _body_from_sources(
        identity=identity,
        morphology=morphology,
        candidates=candidates,
    )
    result["receipt_sha256"] = _self_hash(result, "receipt_sha256")
    return validate_event_component_cycle_element_ledger_v1(result)


def _native_view_id_for_unit(
    unit_id: str,
    views: Sequence[EventMorphologyPrimitiveViewInput],
    native_view_id_by_unit: Mapping[str, str] | None,
) -> str:
    if native_view_id_by_unit is not None and unit_id in native_view_id_by_unit:
        return _identifier(native_view_id_by_unit[unit_id], "native view_id")
    candidates: list[str] = []
    for item in views:
        if not isinstance(item, EventMorphologyPrimitiveViewInput):
            raise TypeError("morphology_views contains an invalid view input")
        receipt = item.view_receipt
        if type(receipt) is not dict:
            continue
        if receipt.get("task_role") != "findings_native_morphology":
            continue
        if unit_id in {
            str(row.get("unit_id")) for row in receipt.get("output_units", [])
        }:
            candidates.append(str(receipt.get("view_id")))
    candidates = sorted(set(candidates))
    if len(candidates) != 1:
        raise ValueError(
            f"unit {unit_id!r} requires exactly one native morphology view; "
            f"observed={candidates}"
        )
    return _identifier(candidates[0], "native view_id")


def materialize_event_component_cycle_element_ledger_from_signal_v1(
    *,
    event_id: str,
    canonical_receipt: object,
    morphology_views: Sequence[EventMorphologyPrimitiveViewInput],
    periodicity_candidates: Sequence[object],
    analysis_interval_seconds: Sequence[float],
    native_view_id_by_unit: Mapping[str, str] | None = None,
    trusted_parent_views: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, Any]:
    """Measure exact morphology components for signal-derived element rows."""

    _identifier(event_id, "event_id")
    canonical = validate_canonical_signal_receipt(canonical_receipt)
    analysis = _interval(analysis_interval_seconds, "analysis_interval_seconds")
    candidates = _validate_candidates(periodicity_candidates)
    candidate_only = [
        row for row in candidates if row["qualification_status"] == "candidate_only"
    ]
    if not candidate_only:
        return materialize_event_component_cycle_element_ledger_v1(
            None,
            candidates,
            canonical_receipt=canonical,
            analysis_interval_seconds=analysis,
            event_id=event_id,
        )
    query_by_key: dict[tuple[str, str, float, float], EventMorphologyPrimitiveQuery] = {}
    for candidate in candidate_only:
        unit_id = str(candidate["analysis_unit_id"])
        view_id = _native_view_id_for_unit(
            unit_id,
            morphology_views,
            native_view_id_by_unit,
        )
        for element in candidate["occurrence"]["element_roster"]:
            interval = _interval(element["recording_interval"], "element interval")
            key = (view_id, unit_id, interval[0], interval[1])
            query_by_key[key] = EventMorphologyPrimitiveQuery(
                view_id=view_id,
                unit_id=unit_id,
                recording_interval_seconds=interval,
                query_authority="deterministic_signal_proposal",
            )
    morphology = materialize_event_morphology_primitive_supervision_v1(
        event_id=event_id,
        canonical_receipt=canonical,
        views=morphology_views,
        analysis_interval_seconds=analysis,
        queries=[query_by_key[key] for key in sorted(query_by_key)],
        trusted_parent_views=trusted_parent_views,
    )
    return materialize_event_component_cycle_element_ledger_v1(
        morphology,
        candidates,
        event_id=event_id,
    )


def validate_event_component_cycle_element_ledger_v1(
    value: object,
) -> dict[str, Any]:
    """Replay the complete S06 receipt from its embedded typed sources."""

    if type(value) is not dict:
        raise TypeError("component/cycle/element ledger must be an object")
    candidate = deepcopy(value)
    _reject_nonfinite(candidate)
    required = {
        "schema_version",
        "method_id",
        "event_id",
        "recording_id",
        "canonical_signal_id",
        "canonical_receipt_sha256",
        "source_signal_sha256",
        "analysis_interval_seconds",
        "coordinate_system",
        "event_card_slot_id",
        "source_morphology_primitive_receipt",
        "source_morphology_primitive_receipt_sha256",
        "source_periodicity_candidates",
        "source_periodicity_candidate_sha256s",
        "source_candidate_opportunity",
        "component_instance_ledger",
        "cycle_instance_ledger",
        "element_instance_ledger",
        "censoring_policy",
        "firewall",
        "authorization",
        "receipt_sha256",
    }
    if set(candidate) != required:
        raise ValueError("component/cycle/element ledger fields drifted")
    if (
        candidate["schema_version"]
        != EVENT_COMPONENT_CYCLE_ELEMENT_LEDGER_SCHEMA_VERSION
        or candidate["method_id"]
        != EVENT_COMPONENT_CYCLE_ELEMENT_LEDGER_METHOD_ID
        or candidate["event_card_slot_id"] != "S06_RHYTHMICITY_PERIODICITY"
    ):
        raise ValueError("component/cycle/element ledger identity drifted")
    for field in ("event_id", "recording_id", "canonical_signal_id"):
        _identifier(candidate[field], field)
    for field in (
        "canonical_receipt_sha256",
        "source_signal_sha256",
        "receipt_sha256",
    ):
        _sha(candidate[field], field)
    _interval(candidate["analysis_interval_seconds"], "analysis_interval_seconds")
    if (
        candidate["coordinate_system"] != "recording_relative_seconds"
        or candidate["censoring_policy"] != _CENSORING_POLICY
        or candidate["firewall"] != _FIREWALL
        or candidate["authorization"] != _AUTHORIZATION
    ):
        raise ValueError("S06 coordinate/firewall/authorization policy drifted")
    morphology_payload = candidate["source_morphology_primitive_receipt"]
    if morphology_payload is None:
        if candidate["source_morphology_primitive_receipt_sha256"] is not None:
            raise ValueError("absent morphology source cannot carry a digest")
        morphology = None
    else:
        morphology = validate_event_morphology_primitive_supervision_v1(
            morphology_payload
        )
        if (
            candidate["source_morphology_primitive_receipt_sha256"]
            != morphology["receipt_sha256"]
        ):
            raise ValueError("S06 morphology source binding drifted")
    candidates = _validate_candidates(candidate["source_periodicity_candidates"])
    if candidate["source_periodicity_candidate_sha256s"] != [
        row["candidate_sha256"] for row in candidates
    ]:
        raise ValueError("S06 periodicity candidate roster binding drifted")
    identity = {
        "event_id": candidate["event_id"],
        "recording_id": candidate["recording_id"],
        "canonical_signal_id": candidate["canonical_signal_id"],
        "canonical_receipt_sha256": candidate["canonical_receipt_sha256"],
        "source_signal_sha256": candidate["source_signal_sha256"],
        "analysis_interval_seconds": candidate["analysis_interval_seconds"],
    }
    if morphology is not None:
        for field in (
            "event_id",
            "recording_id",
            "canonical_signal_id",
            "canonical_receipt_sha256",
            "source_signal_sha256",
            "analysis_interval_seconds",
        ):
            if morphology[field] != identity[field]:
                raise ValueError(f"S06 morphology identity field {field} drifted")
    expected = _body_from_sources(
        identity=identity,
        morphology=morphology,
        candidates=candidates,
    )
    actual = deepcopy(candidate)
    actual.pop("receipt_sha256")
    if actual != expected:
        raise ValueError("S06 ledger does not replay exactly from its sources")
    if candidate["receipt_sha256"] != _self_hash(candidate, "receipt_sha256"):
        raise ValueError("S06 ledger self hash drifted")
    return candidate


__all__ = [
    "EVENT_COMPONENT_CYCLE_ELEMENT_LEDGER_METHOD_ID",
    "EVENT_COMPONENT_CYCLE_ELEMENT_LEDGER_SCHEMA_VERSION",
    "materialize_event_component_cycle_element_ledger_from_signal_v1",
    "materialize_event_component_cycle_element_ledger_v1",
    "validate_event_component_cycle_element_ledger_v1",
]
