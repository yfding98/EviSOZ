"""Signal-only bridge for six waveform/rhythm Findings term queries.

This bridge composes two already replayable numerical artifacts:

* ``event_morphology_primitive_supervision_v1`` supplies physically calibrated
  waveform measurements from an unclipped, instantaneous, volts-valued view;
  and
* ``deterministic_eeg_element_interval_candidate_v1`` supplies explicit
  bounded elements and a complete successive-element interval ledger.

The bridge projects those artifacts into six research-query results.  It does
not infer a clinical spike, IED, ACNS periodic/rhythmic pattern, definite
evolution, seizure, onset, SOZ, EZ, or diagnosis.  Physical-amplitude and
course outputs are measurements.  Periodic-element, rhythmic-run, and
sharp-contour outputs are explicitly ``model_candidate`` instances only.

An amplitude course is emitted only from at least two non-overlapping
recording-relative physical-time measurements on the same unit.  A
rhythmicity course is emitted from the time-ordered successive-cycle ledger;
an event-level median is never copied into a trajectory.  Periodic and
rhythmic candidates cannot be constructed from a spectral peak.  Missing or
masked measurements remain ``not_evaluable`` and never encode absence.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
from typing import Any, Final, Mapping, Sequence

import numpy as np

from .deterministic_event_morphology_primitives_v1 import (
    EVENT_MORPHOLOGY_PRIMITIVE_TARGET_SPECS,
    validate_event_morphology_primitive_supervision_v1,
)
from .deterministic_periodicity_candidate import (
    validate_deterministic_periodicity_candidate,
)


EVENT_WAVEFORM_RHYTHM_QUERY_BRIDGE_SCHEMA_VERSION: Final[str] = (
    "clinical_eeg_event_waveform_rhythm_query_bridge_v1"
)
EVENT_WAVEFORM_RHYTHM_QUERY_BRIDGE_METHOD_ID: Final[str] = (
    "EVENT-WAVEFORM-RHYTHM-QUERY-BRIDGE-V1"
)
EVENT_WAVEFORM_RHYTHM_QUERY_BRIDGE_POLICY_ID: Final[str] = (
    "EVENT-WAVEFORM-RHYTHM-QUERY-COMPOSER-POLICY-V1"
)

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_SHA_RE = re.compile(r"^[a-f0-9]{64}$")
_TOL = 1e-9

_QUERY_SPECS: Final[dict[str, tuple[str, str, str]]] = {
    "TQ-EVENT-AMPLITUDE-COURSE": (
        "event_amplitude_course_profile",
        "measurement",
        "evolution",
    ),
    "TQ-EVENT-RHYTHMICITY-COURSE": (
        "event_rhythmicity_course_profile",
        "measurement",
        "rhythm",
    ),
    "TQ-PERIODIC-ELEMENT-INSTANCE": (
        "periodic_element_candidate",
        "instance",
        "rhythm",
    ),
    "TQ-PHYSICAL-AMPLITUDE-PROFILE": (
        "deterministic_event_physical_amplitude_profile",
        "measurement",
        "spectral",
    ),
    "TQ-RHYTHMIC-RUN-INSTANCE": (
        "rhythmic_run_candidate",
        "instance",
        "rhythm",
    ),
    "TQ-SHARP-CONTOURED-ICTAL-COMPONENT-INSTANCE": (
        "ictal_sharp_contoured_component_candidate",
        "instance",
        "morphology",
    ),
}

_AMPLITUDE_TARGETS: Final[tuple[str, ...]] = (
    "rms_uv",
    "peak_to_peak_uv",
    "positive_excursion_uv",
    "negative_excursion_uv",
    "line_length_uv",
)
_SHARP_TARGETS: Final[tuple[str, ...]] = (
    "peak_to_peak_uv",
    "max_rise_slope_uv_per_s",
    "max_fall_slope_uv_per_s",
    "max_abs_curvature_uv_per_s2",
    "dominant_excursion_latency_seconds",
    "dominant_excursion_half_height_width_seconds",
    "dominant_excursion_rise_half_height_seconds",
    "dominant_excursion_fall_half_height_seconds",
)

_AMPLITUDE_COURSE_MINIMUM_SAMPLE_RATE_HZ: Final[float] = 200.0
_AMPLITUDE_COURSE_REQUIRED_BANDWIDTH_HZ: Final[tuple[float, float]] = (0.5, 45.0)
_RHYTHM_MINIMUM_SAMPLE_RATE_HZ: Final[float] = 200.0
_RHYTHM_REQUIRED_BANDWIDTH_HZ: Final[tuple[float, float]] = (0.5, 45.0)
_SHARP_MINIMUM_SAMPLE_RATE_HZ: Final[float] = 256.0
_SHARP_REQUIRED_BANDWIDTH_HZ: Final[tuple[float, float]] = (0.5, 70.0)

_FIREWALL: Final[dict[str, bool]] = {
    "eeg_samples_used": True,
    "edf_annotations_used": False,
    "excel_used": False,
    "doctor_labels_used": False,
    "clinical_text_used": False,
    "patient_metadata_used": False,
    "video_or_behavior_used": False,
    "sleep_or_activation_labels_used": False,
    "ecg_emg_eog_used": False,
    "qwen_or_other_llm_used": False,
}

_AUTHORIZATION: Final[dict[str, bool | str]] = {
    "projection_scope": "measured_and_research_candidate_queries_only",
    "clinical_term_qualification_authorized": False,
    "negative_clinical_assertion_authorized": False,
    "acns_terminology_authorized": False,
    "event_qualification_authorized": False,
    "onset_claim_authorized": False,
    "soz_or_ez_claim_authorized": False,
    "report_text_authorized": False,
}


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _self_hash(value: Mapping[str, object], field: str) -> str:
    body = deepcopy(dict(value))
    body.pop(field, None)
    return _canonical_sha256(body)


def _finite(value: object, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _interval(value: Sequence[float], name: str) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{name} must be a two-item interval")
    start = _finite(value[0], f"{name}[0]", minimum=0.0)
    stop = _finite(value[1], f"{name}[1]", minimum=0.0)
    if stop <= start + _TOL:
        raise ValueError(f"{name} must have positive duration")
    return start, stop


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise ValueError(f"{name} must be a contract-compatible identifier")
    return value


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or not _SHA_RE.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _sorted_reasons(values: Sequence[str]) -> list[str]:
    result = sorted(set(str(item) for item in values))
    if any(not item or item != item.strip() for item in result):
        raise ValueError("reason codes must be non-empty trimmed strings")
    return result


@dataclass(frozen=True)
class EventWaveformRhythmQueryBridgePolicy:
    """Engineering composition thresholds; none is a clinical criterion."""

    minimum_course_points: int = 2
    minimum_rhythmic_run_elements: int = 4
    maximum_adjacent_cycle_ratio: float = 1.35
    maximum_run_robust_cv: float = 0.30
    minimum_sharp_half_height_width_seconds: float = 0.010
    maximum_sharp_half_height_width_seconds: float = 0.200
    minimum_sharp_normalized_slope: float = 0.20
    minimum_sharp_normalized_curvature: float = 0.20

    def __post_init__(self) -> None:
        if (
            type(self.minimum_course_points) is not int
            or self.minimum_course_points < 2
        ):
            raise ValueError("minimum_course_points must be an integer >= 2")
        if (
            type(self.minimum_rhythmic_run_elements) is not int
            or self.minimum_rhythmic_run_elements < 4
        ):
            raise ValueError(
                "minimum_rhythmic_run_elements must be an integer >= 4"
            )
        for name in (
            "maximum_adjacent_cycle_ratio",
            "maximum_run_robust_cv",
            "minimum_sharp_half_height_width_seconds",
            "maximum_sharp_half_height_width_seconds",
            "minimum_sharp_normalized_slope",
            "minimum_sharp_normalized_curvature",
        ):
            _finite(getattr(self, name), name, minimum=0.0)
        if self.maximum_adjacent_cycle_ratio <= 1.0:
            raise ValueError("maximum_adjacent_cycle_ratio must exceed 1")
        if self.maximum_run_robust_cv <= 0.0:
            raise ValueError("maximum_run_robust_cv must be positive")
        if (
            self.maximum_sharp_half_height_width_seconds
            <= self.minimum_sharp_half_height_width_seconds
        ):
            raise ValueError("sharp half-height width bounds are inverted")

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "policy_id": EVENT_WAVEFORM_RHYTHM_QUERY_BRIDGE_POLICY_ID,
            "method_id": EVENT_WAVEFORM_RHYTHM_QUERY_BRIDGE_METHOD_ID,
            "threshold_semantics": "engineering_research_candidate_only",
            "clinical_thresholds_defined": False,
            "periodicity_source": "explicit_successive_element_interval_ledger",
            "single_spectral_peak_authorized": False,
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


DEFAULT_EVENT_WAVEFORM_RHYTHM_QUERY_BRIDGE_POLICY = (
    EventWaveformRhythmQueryBridgePolicy()
)


def _target_catalog(
    sidecar: Mapping[str, Any],
) -> tuple[dict[str, int], dict[str, str]]:
    expected = [
        {
            "target_name": name,
            "unit_id": unit,
            "opportunity_family": family,
            "semantic_level": "numerical_measurement_only",
        }
        for name, unit, family in EVENT_MORPHOLOGY_PRIMITIVE_TARGET_SPECS
    ]
    if sidecar["target_registry"] != expected:
        raise ValueError("morphology target registry drifted")
    return (
        {str(row["target_name"]): index for index, row in enumerate(expected)},
        {str(row["target_name"]): str(row["unit_id"]) for row in expected},
    )


def _row_value(
    row: Mapping[str, Any],
    target_index: Mapping[str, int],
    name: str,
) -> float | None:
    index = target_index[name]
    if not bool(row["opportunity"]["target_value_mask"][index]):
        return None
    return float(row["values"][index])


def _source_artifact_binding(
    *, kind: str, artifact_id: str, artifact_sha256: str
) -> dict[str, str]:
    return {
        "source_kind": kind,
        "source_artifact_id": artifact_id,
        "source_artifact_sha256": artifact_sha256,
    }


def _query_result(
    *,
    term_query_id: str,
    assertion_level: str,
    qualification_status: str,
    opportunity_status: str,
    opportunity_reasons: Sequence[str],
    measurements: Sequence[Mapping[str, Any]] = (),
    trajectories: Sequence[Mapping[str, Any]] = (),
    instances: Sequence[Mapping[str, Any]] = (),
    source_bindings: Sequence[Mapping[str, str]] = (),
    reason_codes: Sequence[str] = (),
) -> dict[str, Any]:
    if term_query_id not in _QUERY_SPECS:
        raise ValueError("unknown waveform/rhythm term query")
    if assertion_level not in {"measured", "model_candidate"}:
        raise ValueError("query assertion level must remain measured/model_candidate")
    if qualification_status not in {"measured", "candidate_only", "not_evaluable"}:
        raise ValueError("query qualification status is unsupported")
    if opportunity_status not in {"sufficient", "limited", "not_evaluable"}:
        raise ValueError("query opportunity status is unsupported")
    term_id, claim_kind, family = _QUERY_SPECS[term_query_id]
    result: dict[str, Any] = {
        "term_query_id": term_query_id,
        "term_id": term_id,
        "claim_kind": claim_kind,
        "family": family,
        "assertion_level": assertion_level,
        "qualification_status": qualification_status,
        "opportunity": {
            "status": opportunity_status,
            "reason_codes": _sorted_reasons(opportunity_reasons),
            "not_evaluable_is_negative": False,
        },
        "measurements": [deepcopy(dict(row)) for row in measurements],
        "trajectories": [deepcopy(dict(row)) for row in trajectories],
        "instances": [deepcopy(dict(row)) for row in instances],
        "source_artifact_bindings": sorted(
            (deepcopy(dict(row)) for row in source_bindings),
            key=lambda row: (
                row["source_kind"],
                row["source_artifact_id"],
                row["source_artifact_sha256"],
            ),
        ),
        "negative_assertion_authorized": False,
        "clinical_term_qualification_authorized": False,
        "report_promotion_authorized": False,
        "onset_support_eligible": False,
        "soz_support_eligible": False,
        "reason_codes": _sorted_reasons(reason_codes),
    }
    result["query_result_sha256"] = _self_hash(result, "query_result_sha256")
    return result


def _amplitude_rows(
    sidecar: Mapping[str, Any],
    *,
    target_index: Mapping[str, int],
    target_units: Mapping[str, str],
) -> tuple[list[dict[str, Any]], list[str]]:
    view_by_id = {
        str(row["view_id"]): row for row in sidecar["view_bindings"]
    }
    rows: list[dict[str, Any]] = []
    capability_reasons: list[str] = []
    for source in sidecar["rows"]:
        values = []
        for name in _AMPLITUDE_TARGETS:
            value = _row_value(source, target_index, name)
            if value is not None:
                if target_units[name] != "uV":
                    raise ValueError("amplitude target is not expressed in microvolts")
                values.append(
                    {"name_id": name, "value": value, "unit_id": "uV"}
                )
        if not values:
            continue
        binding = source["source_binding"]
        view_binding = view_by_id[str(binding["view_id"])]
        bandwidth = [float(value) for value in binding["effective_bandwidth_hz"]]
        row_capability_reasons = []
        if (
            float(view_binding["sampling_rate_hz"]) + _TOL
            < _AMPLITUDE_COURSE_MINIMUM_SAMPLE_RATE_HZ
        ):
            row_capability_reasons.append(
                "sample_rate_below_amplitude_query_minimum"
            )
        if (
            bandwidth[0] > _AMPLITUDE_COURSE_REQUIRED_BANDWIDTH_HZ[0] + _TOL
            or bandwidth[1] + _TOL
            < _AMPLITUDE_COURSE_REQUIRED_BANDWIDTH_HZ[1]
        ):
            row_capability_reasons.append(
                "effective_bandwidth_below_amplitude_query_requirement"
            )
        if row_capability_reasons:
            capability_reasons.extend(row_capability_reasons)
            continue
        if (
            binding["physical_unit"] != "V"
            or binding["dependency_policy"] != "instantaneous"
            or binding["future_sample_access"] is not False
        ):
            raise ValueError("amplitude row lacks instantaneous physical calibration")
        interval = list(binding["recording_interval_seconds"])
        row = {
            "measurement_id": "AMP-"
            + _canonical_sha256(
                {
                    "source_row_id": source["row_id"],
                    "source_row_binding_sha256": source["row_binding_sha256"],
                    "targets": values,
                }
            )[:24],
            "unit_id": str(binding["unit_id"]),
            "unit_type": str(binding["unit_type"]),
            "recording_interval_seconds": interval,
            "recording_time_seconds": float((interval[0] + interval[1]) / 2.0),
            "values": values,
            "calibration_status": (
                "physical_unit_scale_only_from_channel_metadata"
            ),
            "source_physical_unit": "V",
            "sample_rate_hz": float(view_binding["sampling_rate_hz"]),
            "effective_bandwidth_hz": bandwidth,
            "source_row_id": str(source["row_id"]),
            "source_row_binding_sha256": str(source["row_binding_sha256"]),
        }
        rows.append(row)
    return (
        sorted(
            rows,
            key=lambda row: (
                row["unit_id"],
                row["recording_interval_seconds"][0],
                row["recording_interval_seconds"][1],
                row["source_row_id"],
            ),
        ),
        _sorted_reasons(capability_reasons),
    )


def _expected_units(sidecar: Mapping[str, Any]) -> set[str]:
    return {
        str(unit_id)
        for binding in sidecar["view_bindings"]
        for unit_id in binding["output_unit_ids"]
    }


def _value_map(measurement: Mapping[str, Any]) -> dict[str, float]:
    return {
        str(row["name_id"]): float(row["value"])
        for row in measurement["values"]
    }


def _nonoverlapping_course_points(
    measurements: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Select a canonical physical-time trajectory, never duplicate overlap."""

    selected: list[Mapping[str, Any]] = []
    previous_stop: float | None = None
    for row in sorted(
        measurements,
        key=lambda item: (
            float(item["recording_interval_seconds"][0]),
            float(item["recording_interval_seconds"][1]),
            str(item["measurement_id"]),
        ),
    ):
        start, stop = (float(value) for value in row["recording_interval_seconds"])
        if previous_stop is None or start >= previous_stop - _TOL:
            selected.append(row)
            previous_stop = stop
    return selected


def _amplitude_course(
    measurements: Sequence[Mapping[str, Any]],
    *,
    policy: EventWaveformRhythmQueryBridgePolicy,
) -> list[dict[str, Any]]:
    by_unit: dict[str, list[Mapping[str, Any]]] = {}
    for row in measurements:
        values = _value_map(row)
        if "rms_uv" not in values or "peak_to_peak_uv" not in values:
            continue
        by_unit.setdefault(str(row["unit_id"]), []).append(row)

    trajectories: list[dict[str, Any]] = []
    for unit_id in sorted(by_unit):
        selected = _nonoverlapping_course_points(by_unit[unit_id])
        if len(selected) < policy.minimum_course_points:
            continue
        points = []
        for ordinal, row in enumerate(selected, start=1):
            values = _value_map(row)
            points.append(
                {
                    "ordinal": ordinal,
                    "recording_interval_seconds": list(
                        row["recording_interval_seconds"]
                    ),
                    "recording_time_seconds": float(row["recording_time_seconds"]),
                    "rms_uv": values["rms_uv"],
                    "peak_to_peak_uv": values["peak_to_peak_uv"],
                    "source_measurement_id": str(row["measurement_id"]),
                }
            )
        transitions = []
        for previous, current in zip(points, points[1:]):
            delta_time = (
                current["recording_time_seconds"]
                - previous["recording_time_seconds"]
            )
            if delta_time <= _TOL:
                raise ValueError("amplitude trajectory time is not strictly increasing")
            transitions.append(
                {
                    "ordinal": len(transitions) + 1,
                    "recording_interval_seconds": [
                        previous["recording_time_seconds"],
                        current["recording_time_seconds"],
                    ],
                    "delta_rms_uv": current["rms_uv"] - previous["rms_uv"],
                    "rms_slope_uv_per_second": (
                        current["rms_uv"] - previous["rms_uv"]
                    )
                    / delta_time,
                    "delta_peak_to_peak_uv": (
                        current["peak_to_peak_uv"]
                        - previous["peak_to_peak_uv"]
                    ),
                    "peak_to_peak_slope_uv_per_second": (
                        current["peak_to_peak_uv"]
                        - previous["peak_to_peak_uv"]
                    )
                    / delta_time,
                    "from_point_ordinal": previous["ordinal"],
                    "to_point_ordinal": current["ordinal"],
                }
            )
        body = {
            "unit_id": unit_id,
            "coordinate_system": "recording_relative_seconds",
            "trajectory_source": "nonoverlapping_physical_amplitude_measurements",
            "points": points,
            "transition_intervals": transitions,
        }
        body["trajectory_id"] = "AMPCOURSE-" + _canonical_sha256(body)[:24]
        trajectories.append(body)
    return trajectories


def _candidate_sources(
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    return [
        _source_artifact_binding(
            kind="deterministic_element_interval_candidate",
            artifact_id=str(row["candidate_id"]),
            artifact_sha256=str(row["candidate_sha256"]),
        )
        for row in candidates
    ]


def _rhythm_capable_candidates(
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], list[str]]:
    capable = []
    reasons = []
    for candidate in candidates:
        binding = candidate["source_binding"]
        bandwidth = [float(value) for value in binding["effective_bandwidth_hz"]]
        local_reasons = []
        if (
            float(binding["sample_rate_hz"]) + _TOL
            < _RHYTHM_MINIMUM_SAMPLE_RATE_HZ
        ):
            local_reasons.append("sample_rate_below_rhythm_query_minimum")
        if (
            bandwidth[0] > _RHYTHM_REQUIRED_BANDWIDTH_HZ[0] + _TOL
            or bandwidth[1] + _TOL < _RHYTHM_REQUIRED_BANDWIDTH_HZ[1]
        ):
            local_reasons.append(
                "effective_bandwidth_below_rhythm_query_requirement"
            )
        if local_reasons:
            reasons.extend(local_reasons)
        else:
            capable.append(candidate)
    return capable, _sorted_reasons(reasons)


def _element_maps(
    candidate: Mapping[str, Any],
) -> tuple[dict[str, Mapping[str, Any]], list[Mapping[str, Any]]]:
    elements = candidate["occurrence"]["element_roster"]
    intervals = candidate["successive_interval_profile"]["successive_intervals"]
    return {str(row["element_id"]): row for row in elements}, intervals


def _rhythmicity_course(
    candidates: Sequence[Mapping[str, Any]],
    *,
    policy: EventWaveformRhythmQueryBridgePolicy,
) -> list[dict[str, Any]]:
    trajectories: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate["qualification_status"] != "candidate_only":
            continue
        element_by_id, intervals = _element_maps(candidate)
        points = []
        for ordinal, interval_row in enumerate(intervals, start=1):
            previous = element_by_id[str(interval_row["from_element_id"])]
            current = element_by_id[str(interval_row["to_element_id"])]
            start = float(previous["recording_interval"][0])
            stop = float(current["recording_interval"][0])
            cycle_seconds = float(interval_row["onset_to_onset_seconds"])
            if stop <= start + _TOL or cycle_seconds <= 0.0:
                raise ValueError("successive-cycle physical time is invalid")
            points.append(
                {
                    "ordinal": ordinal,
                    "recording_interval_seconds": [start, stop],
                    "recording_time_seconds": (start + stop) / 2.0,
                    "cycle_interval_seconds": cycle_seconds,
                    "cycle_rate_hz": 1.0 / cycle_seconds,
                    "peak_to_peak_interval_seconds": float(
                        interval_row["peak_to_peak_seconds"]
                    ),
                    "source_interval_id": str(interval_row["interval_id"]),
                }
            )
        if len(points) < policy.minimum_course_points:
            continue
        transitions = []
        for previous, current in zip(points, points[1:]):
            delta_time = (
                current["recording_time_seconds"]
                - previous["recording_time_seconds"]
            )
            if delta_time <= _TOL:
                raise ValueError("rhythmicity trajectory time is not increasing")
            transitions.append(
                {
                    "ordinal": len(transitions) + 1,
                    "recording_interval_seconds": [
                        previous["recording_time_seconds"],
                        current["recording_time_seconds"],
                    ],
                    "delta_cycle_rate_hz": (
                        current["cycle_rate_hz"] - previous["cycle_rate_hz"]
                    ),
                    "cycle_rate_slope_hz_per_second": (
                        current["cycle_rate_hz"] - previous["cycle_rate_hz"]
                    )
                    / delta_time,
                    "delta_cycle_interval_seconds": (
                        current["cycle_interval_seconds"]
                        - previous["cycle_interval_seconds"]
                    ),
                    "from_point_ordinal": previous["ordinal"],
                    "to_point_ordinal": current["ordinal"],
                }
            )
        body = {
            "unit_id": str(candidate["analysis_unit_id"]),
            "coordinate_system": "recording_relative_seconds",
            "trajectory_source": "explicit_successive_element_interval_ledger",
            "interval_ledger_sha256": str(
                candidate["successive_interval_profile"]["interval_ledger_sha256"]
            ),
            "spectral_peak_used": False,
            "autocorrelation_used": False,
            "points": points,
            "transition_intervals": transitions,
            "source_candidate_id": str(candidate["candidate_id"]),
        }
        body["trajectory_id"] = "RHYCOURSE-" + _canonical_sha256(body)[:24]
        trajectories.append(body)
    return sorted(trajectories, key=lambda row: (row["unit_id"], row["trajectory_id"]))


def _periodic_element_instances(
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    instances = []
    for candidate in candidates:
        if candidate["qualification_status"] != "candidate_only":
            continue
        ledger_sha = str(
            candidate["successive_interval_profile"]["interval_ledger_sha256"]
        )
        roster_sha = str(candidate["occurrence"]["element_roster_sha256"])
        for element in candidate["occurrence"]["element_roster"]:
            body = {
                "unit_id": str(candidate["analysis_unit_id"]),
                "recording_interval_seconds": list(element["recording_interval"]),
                "peak_recording_seconds": float(element["peak_recording_seconds"]),
                "duration_seconds": float(element["duration_seconds"]),
                "peak_amplitude_uv": float(element["peak_amplitude_uv"]),
                "polarity": str(element["polarity"]),
                "source_element_id": str(element["element_id"]),
                "element_roster_sha256": roster_sha,
                "successive_interval_ledger_sha256": ledger_sha,
                "inference_source": "explicit_successive_element_interval_ledger",
                "spectral_peak_used": False,
                "autocorrelation_used": False,
                "candidate_semantics": (
                    "bounded_element_in_repeated_sequence_research_candidate_only"
                ),
                "source_candidate_id": str(candidate["candidate_id"]),
            }
            body["instance_id"] = "PEREL-" + _canonical_sha256(body)[:24]
            instances.append(body)
    return sorted(
        instances,
        key=lambda row: (
            row["unit_id"],
            row["recording_interval_seconds"][0],
            row["instance_id"],
        ),
    )


def _run_segments(
    intervals: Sequence[Mapping[str, Any]],
    *,
    maximum_adjacent_ratio: float,
) -> list[tuple[int, int]]:
    """Return inclusive interval-index spans with locally continuous cycles."""

    if not intervals:
        return []
    durations = [float(row["onset_to_onset_seconds"]) for row in intervals]
    start = 0
    result = []
    for index in range(1, len(durations)):
        ratio = max(durations[index - 1], durations[index]) / min(
            durations[index - 1], durations[index]
        )
        if ratio > maximum_adjacent_ratio:
            result.append((start, index - 1))
            start = index
    result.append((start, len(durations) - 1))
    return result


def _rhythmic_run_instances(
    candidates: Sequence[Mapping[str, Any]],
    *,
    policy: EventWaveformRhythmQueryBridgePolicy,
) -> list[dict[str, Any]]:
    instances = []
    minimum_intervals = policy.minimum_rhythmic_run_elements - 1
    for candidate in candidates:
        if candidate["qualification_status"] != "candidate_only":
            continue
        element_by_id, intervals = _element_maps(candidate)
        for start_index, stop_index in _run_segments(
            intervals,
            maximum_adjacent_ratio=policy.maximum_adjacent_cycle_ratio,
        ):
            selected = intervals[start_index : stop_index + 1]
            if len(selected) < minimum_intervals:
                continue
            durations = np.asarray(
                [float(row["onset_to_onset_seconds"]) for row in selected],
                dtype=np.float64,
            )
            median = float(np.median(durations))
            mad = float(np.median(np.abs(durations - median)))
            robust_cv = 1.4826 * mad / median
            if robust_cv > policy.maximum_run_robust_cv:
                continue
            first = element_by_id[str(selected[0]["from_element_id"])]
            last = element_by_id[str(selected[-1]["to_element_id"])]
            body = {
                "unit_id": str(candidate["analysis_unit_id"]),
                "recording_interval_seconds": [
                    float(first["recording_interval"][0]),
                    float(last["recording_interval"][1]),
                ],
                "element_count": len(selected) + 1,
                "cycle_interval_count": len(selected),
                "cycle_interval_median_seconds": median,
                "cycle_interval_mad_seconds": mad,
                "cycle_interval_robust_cv": robust_cv,
                "cycle_rate_hz": 1.0 / median,
                "source_element_ids": [
                    str(selected[0]["from_element_id"]),
                    *[str(row["to_element_id"]) for row in selected],
                ],
                "source_interval_ids": [str(row["interval_id"]) for row in selected],
                "successive_interval_ledger_sha256": str(
                    candidate["successive_interval_profile"]["interval_ledger_sha256"]
                ),
                "inference_source": "explicit_successive_element_interval_ledger",
                "spectral_peak_used": False,
                "autocorrelation_used": False,
                "candidate_semantics": "engineering_rhythmic_run_candidate_only",
                "source_candidate_id": str(candidate["candidate_id"]),
            }
            body["instance_id"] = "RHYRUN-" + _canonical_sha256(body)[:24]
            instances.append(body)
    return sorted(
        instances,
        key=lambda row: (
            row["unit_id"],
            row["recording_interval_seconds"][0],
            row["instance_id"],
        ),
    )


def _sharp_instances(
    sidecar: Mapping[str, Any],
    *,
    target_index: Mapping[str, int],
    policy: EventWaveformRhythmQueryBridgePolicy,
) -> tuple[list[dict[str, Any]], int, list[str]]:
    view_by_id = {
        str(row["view_id"]): row for row in sidecar["view_bindings"]
    }
    instances = []
    measurable_rows = 0
    capability_reasons: list[str] = []
    for row in sidecar["rows"]:
        source_binding = row["source_binding"]
        view_binding = view_by_id[str(source_binding["view_id"])]
        bandwidth = [
            float(value) for value in source_binding["effective_bandwidth_hz"]
        ]
        local_capability_reasons = []
        if (
            float(view_binding["sampling_rate_hz"]) + _TOL
            < _SHARP_MINIMUM_SAMPLE_RATE_HZ
        ):
            local_capability_reasons.append("sample_rate_below_sharp_query_minimum")
        if (
            bandwidth[0] > _SHARP_REQUIRED_BANDWIDTH_HZ[0] + _TOL
            or bandwidth[1] + _TOL < _SHARP_REQUIRED_BANDWIDTH_HZ[1]
        ):
            local_capability_reasons.append(
                "effective_bandwidth_below_sharp_query_requirement"
            )
        if local_capability_reasons:
            capability_reasons.extend(local_capability_reasons)
            continue
        values = {
            name: _row_value(row, target_index, name) for name in _SHARP_TARGETS
        }
        if any(value is None for value in values.values()):
            continue
        measurable_rows += 1
        numeric = {name: float(value) for name, value in values.items() if value is not None}
        width = numeric["dominant_excursion_half_height_width_seconds"]
        peak_to_peak = numeric["peak_to_peak_uv"]
        if peak_to_peak <= 0.0 or not (
            policy.minimum_sharp_half_height_width_seconds
            <= width
            <= policy.maximum_sharp_half_height_width_seconds
        ):
            continue
        normalized_slope = (
            min(
                numeric["max_rise_slope_uv_per_s"],
                numeric["max_fall_slope_uv_per_s"],
            )
            * width
            / peak_to_peak
        )
        normalized_curvature = (
            numeric["max_abs_curvature_uv_per_s2"]
            * width
            * width
            / peak_to_peak
        )
        if (
            normalized_slope < policy.minimum_sharp_normalized_slope
            or normalized_curvature < policy.minimum_sharp_normalized_curvature
        ):
            continue
        source_interval = tuple(
            float(value)
            for value in row["source_binding"]["recording_interval_seconds"]
        )
        peak_time = source_interval[0] + numeric[
            "dominant_excursion_latency_seconds"
        ]
        start = max(
            source_interval[0],
            peak_time - numeric["dominant_excursion_rise_half_height_seconds"],
        )
        stop = min(
            source_interval[1],
            peak_time + numeric["dominant_excursion_fall_half_height_seconds"],
        )
        if stop <= start + _TOL:
            continue
        body = {
            "unit_id": str(row["source_binding"]["unit_id"]),
            "recording_interval_seconds": [start, stop],
            "peak_recording_seconds": peak_time,
            "peak_to_peak_uv": peak_to_peak,
            "half_height_width_seconds": width,
            "normalized_slope": normalized_slope,
            "normalized_curvature": normalized_curvature,
            "query_authority": str(row["source_binding"]["query_authority"]),
            "candidate_semantics": (
                "event_context_sharp_contour_research_candidate_only"
            ),
            "clinical_name_authorized": False,
            "source_row_id": str(row["row_id"]),
            "source_row_binding_sha256": str(row["row_binding_sha256"]),
        }
        body["instance_id"] = "SHARPC-" + _canonical_sha256(body)[:24]
        instances.append(body)
    return (
        sorted(
            instances,
            key=lambda row: (
                row["unit_id"],
                row["recording_interval_seconds"][0],
                row["instance_id"],
            ),
        ),
        measurable_rows,
        _sorted_reasons(capability_reasons),
    )


def _not_evaluable_reasons(candidates: Sequence[Mapping[str, Any]]) -> list[str]:
    reasons = [
        str(reason)
        for candidate in candidates
        if candidate["qualification_status"] == "not_evaluable"
        for reason in candidate["reason_codes"]
    ]
    return _sorted_reasons(reasons or ["no_evaluable_element_interval_candidate"])


def _compose_query_results(
    sidecar: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    *,
    policy: EventWaveformRhythmQueryBridgePolicy,
) -> list[dict[str, Any]]:
    target_index, target_units = _target_catalog(sidecar)
    morphology_source = _source_artifact_binding(
        kind="event_morphology_primitive_supervision",
        artifact_id=str(sidecar["event_id"]),
        artifact_sha256=str(sidecar["receipt_sha256"]),
    )
    candidate_sources = _candidate_sources(candidates)

    amplitude, amplitude_capability_reasons = _amplitude_rows(
        sidecar,
        target_index=target_index,
        target_units=target_units,
    )
    measured_units = {str(row["unit_id"]) for row in amplitude}
    expected_units = _expected_units(sidecar)
    if amplitude:
        amplitude_opportunity = (
            "sufficient" if measured_units == expected_units else "limited"
        )
        amplitude_reasons = (
            []
            if amplitude_opportunity == "sufficient"
            else ["physical_units_without_amplitude_measurement_opportunity"]
        )
        amplitude_result = _query_result(
            term_query_id="TQ-PHYSICAL-AMPLITUDE-PROFILE",
            assertion_level="measured",
            qualification_status="measured",
            opportunity_status=amplitude_opportunity,
            opportunity_reasons=amplitude_reasons,
            measurements=amplitude,
            source_bindings=[morphology_source],
            reason_codes=[
                "microvolt_values_replayed_from_instantaneous_physical_view"
            ],
        )
    else:
        amplitude_result = _query_result(
            term_query_id="TQ-PHYSICAL-AMPLITUDE-PROFILE",
            assertion_level="model_candidate",
            qualification_status="not_evaluable",
            opportunity_status="not_evaluable",
            opportunity_reasons=(
                amplitude_capability_reasons
                or ["no_microvolt_amplitude_target_available"]
            ),
            source_bindings=[morphology_source],
            reason_codes=["no_negative_amplitude_assertion"],
        )

    amplitude_trajectories = _amplitude_course(amplitude, policy=policy)
    course_units = {str(row["unit_id"]) for row in amplitude_trajectories}
    if amplitude_trajectories:
        course_opportunity = (
            "sufficient" if course_units == expected_units else "limited"
        )
        course_reasons = (
            []
            if course_opportunity == "sufficient"
            else ["physical_units_without_two_nonoverlapping_course_points"]
        )
        amplitude_course_result = _query_result(
            term_query_id="TQ-EVENT-AMPLITUDE-COURSE",
            assertion_level="measured",
            qualification_status="measured",
            opportunity_status=course_opportunity,
            opportunity_reasons=course_reasons,
            trajectories=amplitude_trajectories,
            source_bindings=[morphology_source],
            reason_codes=[
                "course_contains_recording_relative_points_and_transition_intervals"
            ],
        )
    else:
        amplitude_course_result = _query_result(
            term_query_id="TQ-EVENT-AMPLITUDE-COURSE",
            assertion_level="model_candidate",
            qualification_status="not_evaluable",
            opportunity_status="not_evaluable",
            opportunity_reasons=(
                amplitude_capability_reasons
                or [
                    "fewer_than_two_nonoverlapping_physical_time_points_per_unit"
                ]
            ),
            source_bindings=[morphology_source],
            reason_codes=["single_event_summary_not_promoted_to_course"],
        )

    rhythm_candidates, rhythm_capability_reasons = _rhythm_capable_candidates(
        candidates
    )
    rhythmicity_trajectories = _rhythmicity_course(
        rhythm_candidates, policy=policy
    )
    if rhythmicity_trajectories:
        not_eval_count = sum(
            row["qualification_status"] == "not_evaluable"
            for row in rhythm_candidates
        )
        rhythm_opportunity = (
            "limited"
            if not_eval_count or rhythm_capability_reasons
            else "sufficient"
        )
        rhythm_reasons = (
            [
                *(
                    ["some_unit_element_interval_candidates_not_evaluable"]
                    if not_eval_count
                    else []
                ),
                *rhythm_capability_reasons,
            ]
        )
        rhythmicity_course_result = _query_result(
            term_query_id="TQ-EVENT-RHYTHMICITY-COURSE",
            assertion_level="measured",
            qualification_status="measured",
            opportunity_status=rhythm_opportunity,
            opportunity_reasons=rhythm_reasons,
            trajectories=rhythmicity_trajectories,
            source_bindings=candidate_sources,
            reason_codes=[
                "course_replayed_from_time_ordered_successive_cycle_ledger"
            ],
        )
    else:
        rhythmicity_course_result = _query_result(
            term_query_id="TQ-EVENT-RHYTHMICITY-COURSE",
            assertion_level="model_candidate",
            qualification_status="not_evaluable",
            opportunity_status="not_evaluable",
            opportunity_reasons=(
                rhythm_capability_reasons
                or _not_evaluable_reasons(rhythm_candidates)
            ),
            source_bindings=candidate_sources,
            reason_codes=["event_summary_not_promoted_to_rhythmicity_course"],
        )

    periodic_instances = _periodic_element_instances(rhythm_candidates)
    if periodic_instances:
        periodic_result = _query_result(
            term_query_id="TQ-PERIODIC-ELEMENT-INSTANCE",
            assertion_level="model_candidate",
            qualification_status="candidate_only",
            opportunity_status="sufficient",
            opportunity_reasons=[],
            instances=periodic_instances,
            source_bindings=candidate_sources,
            reason_codes=[
                "explicit_elements_bound_to_complete_successive_interval_ledger",
                "not_an_acns_periodic_discharge_claim",
            ],
        )
    else:
        periodic_result = _query_result(
            term_query_id="TQ-PERIODIC-ELEMENT-INSTANCE",
            assertion_level="model_candidate",
            qualification_status="not_evaluable",
            opportunity_status="not_evaluable",
            opportunity_reasons=(
                rhythm_capability_reasons
                or _not_evaluable_reasons(rhythm_candidates)
            ),
            source_bindings=candidate_sources,
            reason_codes=["no_periodic_element_absence_inference"],
        )

    rhythmic_runs = _rhythmic_run_instances(rhythm_candidates, policy=policy)
    if rhythmic_runs:
        rhythmic_run_result = _query_result(
            term_query_id="TQ-RHYTHMIC-RUN-INSTANCE",
            assertion_level="model_candidate",
            qualification_status="candidate_only",
            opportunity_status="sufficient",
            opportunity_reasons=[],
            instances=rhythmic_runs,
            source_bindings=candidate_sources,
            reason_codes=[
                "engineering_run_rule_bound_to_successive_cycle_ledger",
                "not_an_acns_rhythmic_pattern_claim",
            ],
        )
    else:
        rhythmic_run_result = _query_result(
            term_query_id="TQ-RHYTHMIC-RUN-INSTANCE",
            assertion_level="model_candidate",
            qualification_status="not_evaluable",
            opportunity_status=("limited" if periodic_instances else "not_evaluable"),
            opportunity_reasons=(
                ["engineering_rhythmic_run_rule_not_met_without_sensitivity_receipt"]
                if periodic_instances
                else (
                    rhythm_capability_reasons
                    or _not_evaluable_reasons(rhythm_candidates)
                )
            ),
            source_bindings=candidate_sources,
            reason_codes=["no_rhythmic_run_absence_inference"],
        )

    sharp_instances, sharp_measurable_rows, sharp_capability_reasons = _sharp_instances(
        sidecar,
        target_index=target_index,
        policy=policy,
    )
    if sharp_instances:
        sharp_result = _query_result(
            term_query_id="TQ-SHARP-CONTOURED-ICTAL-COMPONENT-INSTANCE",
            assertion_level="model_candidate",
            qualification_status="candidate_only",
            opportunity_status="sufficient",
            opportunity_reasons=[],
            instances=sharp_instances,
            source_bindings=[morphology_source],
            reason_codes=[
                "engineering_contour_rule_only_no_spike_or_ied_qualification"
            ],
        )
    else:
        sharp_result = _query_result(
            term_query_id="TQ-SHARP-CONTOURED-ICTAL-COMPONENT-INSTANCE",
            assertion_level="model_candidate",
            qualification_status="not_evaluable",
            opportunity_status=("limited" if sharp_measurable_rows else "not_evaluable"),
            opportunity_reasons=(
                ["engineering_sharp_contour_rule_not_met_without_sensitivity_receipt"]
                if sharp_measurable_rows
                else (
                    sharp_capability_reasons
                    or ["sharp_contour_geometry_targets_not_evaluable"]
                )
            ),
            source_bindings=[morphology_source],
            reason_codes=["no_sharp_contour_absence_inference"],
        )

    results = [
        amplitude_course_result,
        rhythmicity_course_result,
        periodic_result,
        amplitude_result,
        rhythmic_run_result,
        sharp_result,
    ]
    return sorted(results, key=lambda row: row["term_query_id"])


def _validated_sources(
    morphology_primitive_receipt: object,
    periodicity_candidates: Sequence[object],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sidecar = validate_event_morphology_primitive_supervision_v1(
        morphology_primitive_receipt
    )
    candidates = [
        validate_deterministic_periodicity_candidate(value)
        for value in periodicity_candidates
    ]
    candidates.sort(
        key=lambda row: (str(row["analysis_unit_id"]), str(row["candidate_id"]))
    )
    candidate_ids = [str(row["candidate_id"]) for row in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("periodicity candidate IDs must be unique")
    units = [str(row["analysis_unit_id"]) for row in candidates]
    if len(units) != len(set(units)):
        raise ValueError("at most one periodicity candidate is allowed per unit")
    analysis = _interval(sidecar["analysis_interval_seconds"], "analysis interval")
    for candidate in candidates:
        if candidate["event_id"] != sidecar["event_id"]:
            raise ValueError("periodicity candidate event identity drifted")
        if (
            candidate["source_binding"]["canonical_signal_sha256"]
            != sidecar["source_signal_sha256"]
        ):
            raise ValueError("periodicity candidate signal identity drifted")
        interval = _interval(
            candidate["requested_recording_interval"],
            "periodicity candidate interval",
        )
        if interval[0] < analysis[0] - _TOL or interval[1] > analysis[1] + _TOL:
            raise ValueError("periodicity candidate lies outside sidecar analysis")
    return sidecar, candidates


def _bundle_body(
    sidecar: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    *,
    policy: EventWaveformRhythmQueryBridgePolicy,
) -> dict[str, Any]:
    results = _compose_query_results(sidecar, candidates, policy=policy)
    return {
        "schema_version": EVENT_WAVEFORM_RHYTHM_QUERY_BRIDGE_SCHEMA_VERSION,
        "method_id": EVENT_WAVEFORM_RHYTHM_QUERY_BRIDGE_METHOD_ID,
        "event_id": str(sidecar["event_id"]),
        "recording_id": str(sidecar["recording_id"]),
        "canonical_signal_id": str(sidecar["canonical_signal_id"]),
        "canonical_receipt_sha256": str(sidecar["canonical_receipt_sha256"]),
        "source_signal_sha256": str(sidecar["source_signal_sha256"]),
        "analysis_interval_seconds": list(sidecar["analysis_interval_seconds"]),
        "coordinate_system": "recording_relative_seconds",
        "policy": policy.to_dict(),
        "policy_sha256": policy.sha256,
        "source_morphology_primitive_receipt": deepcopy(dict(sidecar)),
        "source_morphology_primitive_receipt_sha256": str(sidecar["receipt_sha256"]),
        "source_periodicity_candidates": [deepcopy(dict(row)) for row in candidates],
        "source_periodicity_candidate_sha256s": [
            str(row["candidate_sha256"]) for row in candidates
        ],
        "query_results": results,
        "query_result_roster_sha256": _canonical_sha256(results),
        "firewall": deepcopy(_FIREWALL),
        "authorization": deepcopy(_AUTHORIZATION),
    }


def materialize_event_waveform_rhythm_query_bridge_v1(
    *,
    morphology_primitive_receipt: object,
    periodicity_candidates: Sequence[object],
    policy: EventWaveformRhythmQueryBridgePolicy = (
        DEFAULT_EVENT_WAVEFORM_RHYTHM_QUERY_BRIDGE_POLICY
    ),
) -> dict[str, Any]:
    """Compose the six query results from replayable signal-only artifacts."""

    if not isinstance(policy, EventWaveformRhythmQueryBridgePolicy):
        raise TypeError("policy must be EventWaveformRhythmQueryBridgePolicy")
    sidecar, candidates = _validated_sources(
        morphology_primitive_receipt,
        periodicity_candidates,
    )
    result = _bundle_body(sidecar, candidates, policy=policy)
    result["receipt_sha256"] = _self_hash(result, "receipt_sha256")
    return validate_event_waveform_rhythm_query_bridge_v1(result)


def validate_event_waveform_rhythm_query_bridge_v1(value: object) -> dict[str, Any]:
    """Validate source artifacts, deterministic projection, and content hashes."""

    if type(value) is not dict:
        raise TypeError("waveform/rhythm query bridge must be an object")
    bundle = deepcopy(value)
    expected_keys = {
        "schema_version",
        "method_id",
        "event_id",
        "recording_id",
        "canonical_signal_id",
        "canonical_receipt_sha256",
        "source_signal_sha256",
        "analysis_interval_seconds",
        "coordinate_system",
        "policy",
        "policy_sha256",
        "source_morphology_primitive_receipt",
        "source_morphology_primitive_receipt_sha256",
        "source_periodicity_candidates",
        "source_periodicity_candidate_sha256s",
        "query_results",
        "query_result_roster_sha256",
        "firewall",
        "authorization",
        "receipt_sha256",
    }
    if set(bundle) != expected_keys:
        raise ValueError("waveform/rhythm query bridge keys drifted")
    if (
        bundle["schema_version"]
        != EVENT_WAVEFORM_RHYTHM_QUERY_BRIDGE_SCHEMA_VERSION
        or bundle["method_id"] != EVENT_WAVEFORM_RHYTHM_QUERY_BRIDGE_METHOD_ID
    ):
        raise ValueError("waveform/rhythm query bridge identity drifted")
    _identifier(bundle["event_id"], "event_id")
    _identifier(bundle["recording_id"], "recording_id")
    _identifier(bundle["canonical_signal_id"], "canonical_signal_id")
    _sha(bundle["canonical_receipt_sha256"], "canonical_receipt_sha256")
    _sha(bundle["source_signal_sha256"], "source_signal_sha256")
    _sha(bundle["policy_sha256"], "policy_sha256")
    _sha(
        bundle["source_morphology_primitive_receipt_sha256"],
        "source morphology receipt sha256",
    )
    _sha(bundle["query_result_roster_sha256"], "query result roster sha256")
    _sha(bundle["receipt_sha256"], "receipt_sha256")
    _interval(bundle["analysis_interval_seconds"], "analysis_interval_seconds")
    if bundle["coordinate_system"] != "recording_relative_seconds":
        raise ValueError("query bridge coordinate system drifted")

    policy_data = bundle["policy"]
    if not isinstance(policy_data, Mapping):
        raise TypeError("policy must be an object")
    defaults = asdict(DEFAULT_EVENT_WAVEFORM_RHYTHM_QUERY_BRIDGE_POLICY)
    policy = EventWaveformRhythmQueryBridgePolicy(
        **{name: policy_data[name] for name in defaults}
    )
    if policy_data != policy.to_dict() or bundle["policy_sha256"] != policy.sha256:
        raise ValueError("query bridge policy content/hash drifted")
    if bundle["firewall"] != _FIREWALL or bundle["authorization"] != _AUTHORIZATION:
        raise ValueError("query bridge firewall/authorization drifted")
    if any(
        bool(value)
        for key, value in bundle["firewall"].items()
        if key != "eeg_samples_used"
    ) or bundle["firewall"]["eeg_samples_used"] is not True:
        raise ValueError("query bridge violates the EEG-only firewall")

    source_candidates = bundle["source_periodicity_candidates"]
    if not isinstance(source_candidates, list):
        raise TypeError("source_periodicity_candidates must be an array")
    sidecar, candidates = _validated_sources(
        bundle["source_morphology_primitive_receipt"],
        source_candidates,
    )
    if (
        bundle["source_morphology_primitive_receipt_sha256"]
        != sidecar["receipt_sha256"]
        or bundle["source_periodicity_candidate_sha256s"]
        != [row["candidate_sha256"] for row in candidates]
    ):
        raise ValueError("query bridge source artifact hash roster drifted")
    expected = _bundle_body(sidecar, candidates, policy=policy)
    actual_without_receipt = deepcopy(bundle)
    actual_without_receipt.pop("receipt_sha256")
    if actual_without_receipt != expected:
        raise ValueError("query bridge projection is not replayable from its sources")
    if bundle["receipt_sha256"] != _self_hash(bundle, "receipt_sha256"):
        raise ValueError("query bridge receipt hash mismatch")
    return bundle


def replay_event_waveform_rhythm_query_bridge_v1(
    receipt: object,
    *,
    morphology_primitive_receipt: object,
    periodicity_candidates: Sequence[object],
    policy: EventWaveformRhythmQueryBridgePolicy = (
        DEFAULT_EVENT_WAVEFORM_RHYTHM_QUERY_BRIDGE_POLICY
    ),
) -> dict[str, Any]:
    """Replay against independently supplied source artifacts."""

    validated = validate_event_waveform_rhythm_query_bridge_v1(receipt)
    expected = materialize_event_waveform_rhythm_query_bridge_v1(
        morphology_primitive_receipt=morphology_primitive_receipt,
        periodicity_candidates=periodicity_candidates,
        policy=policy,
    )
    if validated != expected:
        raise ValueError("waveform/rhythm query bridge replay mismatch")
    return expected


__all__ = [
    "DEFAULT_EVENT_WAVEFORM_RHYTHM_QUERY_BRIDGE_POLICY",
    "EVENT_WAVEFORM_RHYTHM_QUERY_BRIDGE_METHOD_ID",
    "EVENT_WAVEFORM_RHYTHM_QUERY_BRIDGE_POLICY_ID",
    "EVENT_WAVEFORM_RHYTHM_QUERY_BRIDGE_SCHEMA_VERSION",
    "EventWaveformRhythmQueryBridgePolicy",
    "materialize_event_waveform_rhythm_query_bridge_v1",
    "replay_event_waveform_rhythm_query_bridge_v1",
    "validate_event_waveform_rhythm_query_bridge_v1",
]
