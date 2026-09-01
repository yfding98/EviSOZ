"""Runtime invariants for ``event_eeg_findings_v1``.

The JSON Schema intentionally describes the portable wire format.  This
module enforces relations that JSON Schema cannot express cleanly: physical
time ordering, probability-mass conservation, evidence/receipt references,
and montage-aware identifier use.  The validator is fail-closed and returns a
deep copy so callers cannot mutate the object that was checked.
"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from jsonschema import Draft202012Validator

from .clinical_term_qualification import (
    PROTECTED_EEG_ONLY_TERMS,
    validate_clinical_eeg_term_qualification,
)
from .event_finding_term_registry import (
    validate_event_finding_term,
)


EVENT_EEG_FINDINGS_SCHEMA_VERSION = "event_eeg_findings_v1"
_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA_PATH = _ROOT / "schemas" / "clinical_eeg_event_findings_v1.schema.json"
_TOL = 1e-6
_TERM_DECISION_BINDING_DOMAIN = "clinical_eeg_event_term_decision_source_v1"

_FEATURE_FAMILIES = {
    "spectral",
    "rhythm",
    "morphology",
    "evolution",
    "spatial_field",
    "recruitment",
    "termination_recovery",
    "high_frequency",
}
_PHENOTYPES = {
    "focal",
    "focal_with_rapid_bilateralization",
    "bilateral_synchronous_or_rapid_bilateralization_ambiguous",
    "generalized_synchronous",
    "scalp_onset_nonlocalizable",
}
_LATERALITIES = {"left", "right", "bilateral", "midline", "indeterminate"}
_RESOLUTION_TO_CANDIDATE_TYPE = {
    "lead": "lead",
    "electrode": "electrode",
    "region": "region",
    "laterality": "laterality",
}
_FINDING_TO_AVAILABILITY_FAMILY = {
    "spectral": "spectral",
    "rhythm": "rhythm",
    "morphology": "morphology",
    "evolution": "evolution",
    "spatial_field": "spatial_field",
    "spatial_recruitment": "recruitment",
    "termination_recovery": "termination_recovery",
    "high_frequency": "high_frequency",
}
_FINDING_TO_SOURCE_EVIDENCE_FAMILIES = {
    "quality": {"waveform"},
    "spectral": {"spectral", "amplitude"},
    "rhythm": {"spectral"},
    "morphology": {"morphology"},
    "evolution": {"spectral", "amplitude", "spatial_field"},
    "spatial_field": {"spatial_field"},
    "spatial_recruitment": {"spatial_field"},
    "termination_recovery": {"spectral", "amplitude"},
    "high_frequency": {"high_frequency"},
}
# Frozen event-level allowlist.  An early, onset-anchored recruitment atom may
# support onset, but late recruitment must carry ``spread_support`` and is
# therefore excluded by the role/phase gates.  Termination and quality are
# never eligible for the spatial onset ranking.
_ONSET_SUPPORT_FAMILIES = frozenset(
    {
        "spectral",
        "rhythm",
        "morphology",
        "evolution",
        "spatial_field",
        "spatial_recruitment",
        "high_frequency",
    }
)
# Spectral, rhythm, morphology and high-frequency atoms may describe the
# earliest observed pattern, but they cannot *by themselves* create a spatial
# SOZ candidate.  Every ranked spatial candidate needs at least one explicit
# field or earliest-involvement anchor.  This keeps a localized rhythm as
# useful context while preventing a non-spatial feature score from being
# silently reinterpreted as source localization.
_SPATIAL_ONSET_ANCHOR_FAMILIES = frozenset(
    {"spatial_field", "spatial_recruitment"}
)
_HIGH_FREQUENCY_MARKERS = {
    "hfo",
    "hfos",
    "high_frequency_activity",
    "high_frequency_oscillation",
    "high_frequency_oscillations",
    "high_frequency_oscillatory",
    "ripple",
    "ripples",
    "fast_ripple",
    "fast_ripples",
    "lvfa",
    "low_amplitude_fast_activity",
    "low_voltage_fast",
    "low_voltage_fast_activity",
}
_SYNCHRONY_TERM_MARKERS = {
    "bilateral_synchronous",
    "generalized_synchronous",
    "simultaneous_bilateral",
    "bilateral_simultaneous",
}


@lru_cache(maxsize=1)
def _schema_validator() -> Draft202012Validator:
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _path(error: Any) -> str:
    parts = [str(item) for item in error.absolute_path]
    return ".".join(parts) if parts else "$"


def _reject_nonfinite(value: object, path: str = "$") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        try:
            finite = math.isfinite(float(value))
        except OverflowError as error:
            raise ValueError(f"{path} must be finite") from error
        if not finite:
            raise ValueError(f"{path} must be finite")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_nonfinite(item, f"{path}.{key}")
        return
    if isinstance(value, Sequence):
        for index, item in enumerate(value):
            _reject_nonfinite(item, f"{path}[{index}]")


def _unique(values: Iterable[str], context: str) -> set[str]:
    materialized = list(values)
    if len(materialized) != len(set(materialized)):
        raise ValueError(f"{context} contains duplicate identifiers")
    return set(materialized)


def _closed_interval(
    interval: Sequence[float],
    context: str,
    *,
    bounds: tuple[float, float] | None = None,
    positive: bool = False,
) -> tuple[float, float]:
    lower, upper = float(interval[0]), float(interval[1])
    if lower > upper + _TOL:
        raise ValueError(f"{context} lower bound exceeds upper bound")
    if positive and upper <= lower + _TOL:
        raise ValueError(f"{context} must have positive duration")
    if bounds is not None:
        bound_lower, bound_upper = bounds
        if lower < bound_lower - _TOL or upper > bound_upper + _TOL:
            raise ValueError(f"{context} lies outside [{bound_lower}, {bound_upper}]")
    return lower, upper


def _observed_span(
    span: Mapping[str, object],
    context: str,
    *,
    bounds: tuple[float, float],
) -> tuple[float, float]:
    start = float(span["start"])
    stop = float(span["stop"])
    if stop <= start + _TOL:
        raise ValueError(f"{context} must have positive duration")
    if start < bounds[0] - _TOL or stop > bounds[1] + _TOL:
        raise ValueError(f"{context} lies outside [{bounds[0]}, {bounds[1]}]")
    return start, stop


def _time_interval(
    interval: Mapping[str, object],
    context: str,
    *,
    bounds: tuple[float, float] | None = None,
    nonnegative: bool = False,
) -> tuple[float, float]:
    lower = float(interval["lower"])
    upper = float(interval["upper"])
    if lower > upper + _TOL:
        raise ValueError(f"{context}.lower exceeds .upper")
    if "median" in interval:
        median = float(interval["median"])
        if median < lower - _TOL or median > upper + _TOL:
            raise ValueError(f"{context}.median lies outside [lower, upper]")
    if nonnegative and lower < -_TOL:
        raise ValueError(f"{context} must be nonnegative")
    if bounds is not None:
        bound_lower, bound_upper = bounds
        if lower < bound_lower - _TOL or upper > bound_upper + _TOL:
            raise ValueError(f"{context} lies outside [{bound_lower}, {bound_upper}]")
    return lower, upper


def _phase_interval(
    phase: Mapping[str, object],
    context: str,
    *,
    bounds: tuple[float, float],
) -> tuple[float, float] | None:
    status = phase["status"]
    interval = phase["interval"]
    if status == "not_observed":
        if interval is not None:
            raise ValueError(f"{context}.interval must be null when status=not_observed")
        return None
    if interval is None:
        raise ValueError(f"{context}.interval is required when status={status}")
    return _observed_span(interval, f"{context}.interval", bounds=bounds)


def _boundary_estimate(
    value: Mapping[str, object],
    context: str,
    *,
    bounds: tuple[float, float],
) -> tuple[float, float] | None:
    status = value["status"]
    interval = value["interval"]
    if status == "not_observed":
        if interval is not None:
            raise ValueError(f"{context}.interval must be null when status=not_observed")
        return None
    if interval is None:
        raise ValueError(f"{context}.interval is required when status={status}")
    return _time_interval(interval, f"{context}.interval", bounds=bounds)


def _onset_boundary_estimate(
    value: Mapping[str, object],
    context: str,
    *,
    bounds: tuple[float, float],
) -> tuple[tuple[float, float] | None, str]:
    """Validate typed onset boundaries while accepting the legacy v1 shape."""

    if "status" not in value:
        return (
            _time_interval(value, context, bounds=bounds),
            "legacy_interval_estimate",
        )

    status = str(value["status"])
    interval = value["interval"]
    if status in {"observed", "interval_estimate"}:
        if interval is None:
            raise ValueError(f"{context}.interval is required when status={status}")
        return (
            _time_interval(
                interval,  # type: ignore[arg-type]
                f"{context}.interval",
                bounds=bounds,
            ),
            status,
        )
    if status in {"not_observed", "indeterminate"}:
        if interval is not None:
            raise ValueError(f"{context}.interval must be null when status={status}")
        return None, status
    if status == "censored":
        if interval is None:
            return None, status
        return (
            _time_interval(
                interval,  # type: ignore[arg-type]
                f"{context}.interval",
                bounds=bounds,
            ),
            status,
        )
    raise ValueError(f"{context}.status is unsupported: {status!r}")


def _interval_list(
    rows: Sequence[Sequence[float]],
    context: str,
    *,
    bounds: tuple[float, float],
) -> list[tuple[float, float]]:
    intervals = [
        _closed_interval(row, f"{context}[{index}]", bounds=bounds, positive=True)
        for index, row in enumerate(rows)
    ]
    if intervals != sorted(intervals):
        raise ValueError(f"{context} must be sorted")
    for previous, current in zip(intervals, intervals[1:]):
        if current[0] < previous[1] - _TOL:
            raise ValueError(f"{context} must be non-overlapping")
    return intervals


def _covered_by(
    interval: tuple[float, float],
    carriers: Sequence[tuple[float, float]],
) -> bool:
    return any(
        interval[0] >= carrier[0] - _TOL and interval[1] <= carrier[1] + _TOL
        for carrier in carriers
    )


def _distribution(
    rows: Sequence[Mapping[str, object]],
    context: str,
    *,
    allowed_names: set[str] | None = None,
) -> set[str]:
    names = _unique((str(row["name"]) for row in rows), f"{context}.name")
    if allowed_names is not None and not names.issubset(allowed_names):
        unknown = sorted(names.difference(allowed_names))
        raise ValueError(f"{context} contains unsupported names: {unknown}")
    if rows:
        total = sum(float(row["score"]) for row in rows)
        if abs(total - 1.0) > _TOL:
            raise ValueError(f"{context} scores must sum to 1; got {total:.12g}")
    return names


def _require_refs(values: Iterable[str], available: set[str], context: str) -> None:
    missing = set(values).difference(available)
    if missing:
        raise ValueError(f"{context} references unknown identifiers: {sorted(missing)}")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _trusted_receipt_registry(
    value: Mapping[str, Mapping[str, object]] | None,
    *,
    name: str,
) -> dict[str, Mapping[str, object]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a host-supplied mapping")
    result: dict[str, Mapping[str, object]] = {}
    for key, raw in value.items():
        if not isinstance(key, str) or not isinstance(raw, Mapping):
            raise TypeError(f"{name} entries are invalid")
        if raw.get("receipt_id") != key:
            raise ValueError(f"{name} key/receipt_id mismatch")
        result[key] = deepcopy(dict(raw))
    return result


def event_term_decision_source_binding_sha256(value: object) -> str:
    """Hash the event evidence ledger without its circular decision references.

    The per-event term receipt is built after candidate Findings exist but
    before its own ID can be inserted into those Findings.  This canonical
    projection removes the embedded decision receipts and normalizes only the
    circular IDs to null; ``finding.term`` and every physical evidence field
    remain bound.
    """

    if type(value) is not dict:
        raise TypeError("event term-decision source must be an event object")
    source = deepcopy(value)
    source.pop("term_decision_receipts", None)
    findings = source.get("findings")
    if not isinstance(findings, list):
        raise ValueError("event term-decision source requires findings")
    for index, finding in enumerate(findings):
        if not isinstance(finding, dict):
            raise TypeError(f"findings[{index}] must be an object")
        finding["term_decision_receipt_id"] = None
    return hashlib.sha256(
        _canonical_json(
            {
                "binding_domain": _TERM_DECISION_BINDING_DOMAIN,
                "event": source,
            }
        ).encode("utf-8")
    ).hexdigest()


def _validate_montage(payload: Mapping[str, Any]) -> dict[str, Any]:
    montage = payload["montage"]
    input_units = montage["input_units"]
    input_unit_ids = _unique(
        (str(row["unit_id"]) for row in input_units),
        "montage.input_units.unit_id",
    )
    electrode_ids = set(str(item) for item in montage["electrode_ids"])
    lead_definitions = montage["lead_definitions"]
    lead_ids = _unique(
        (str(row["lead_id"]) for row in lead_definitions),
        "montage.lead_definitions.lead_id",
    )

    for index, definition in enumerate(lead_definitions):
        anode = str(definition["anode"])
        cathode = str(definition["cathode"])
        if anode == cathode:
            raise ValueError(f"montage.lead_definitions[{index}] has identical endpoints")
        _require_refs(
            (anode, cathode),
            electrode_ids,
            f"montage.lead_definitions[{index}]",
        )

    bipolar_names: set[str] = set()
    electrode_names: set[str] = set()
    region_ids: set[str] = set()
    availability: dict[str, bool] = {}
    unit_laterality: dict[str, str] = {}
    lead_to_input_units: dict[str, set[str]] = {identifier: set() for identifier in lead_ids}
    electrode_to_input_units: dict[str, set[str]] = {
        identifier: set() for identifier in electrode_ids
    }
    region_to_input_units: dict[str, set[str]] = {}
    lead_to_lateralities: dict[str, set[str]] = {
        identifier: set() for identifier in lead_ids
    }
    electrode_to_lateralities: dict[str, set[str]] = {
        identifier: set() for identifier in electrode_ids
    }
    region_to_lateralities: dict[str, set[str]] = {}
    lead_to_regions: dict[str, set[str]] = {
        identifier: set() for identifier in lead_ids
    }
    electrode_to_regions: dict[str, set[str]] = {
        identifier: set() for identifier in electrode_ids
    }
    lead_endpoints = {
        str(row["lead_id"]): {str(row["anode"]), str(row["cathode"])}
        for row in lead_definitions
    }
    for index, unit in enumerate(input_units):
        unit_id = str(unit["unit_id"])
        canonical_name = str(unit["canonical_name"])
        laterality = str(unit["laterality"])
        availability[unit_id] = bool(unit["available"])
        unit_laterality[unit_id] = laterality
        if "region" in unit:
            region = str(unit["region"])
            region_ids.add(region)
            region_to_input_units.setdefault(region, set()).add(unit_id)
            region_to_lateralities.setdefault(region, set()).add(laterality)
        if unit["unit_type"] == "electrode":
            if canonical_name not in electrode_ids:
                raise ValueError(
                    f"montage.input_units[{index}].canonical_name is not a declared electrode"
                )
            electrode_names.add(canonical_name)
            electrode_to_input_units[canonical_name].add(unit_id)
            electrode_to_lateralities[canonical_name].add(laterality)
            if "region" in unit:
                electrode_to_regions[canonical_name].add(str(unit["region"]))
        else:
            if canonical_name not in lead_ids:
                raise ValueError(
                    f"montage.input_units[{index}].canonical_name is not a declared lead"
                )
            bipolar_names.add(canonical_name)
            lead_to_input_units[canonical_name].add(unit_id)
            lead_to_lateralities[canonical_name].add(laterality)
            if "region" in unit:
                lead_to_regions[canonical_name].add(str(unit["region"]))

    if bipolar_names != lead_ids:
        raise ValueError(
            "montage bipolar input units and lead_definitions must have identical canonical IDs"
        )
    if not electrode_names.issubset(electrode_ids):
        raise ValueError("montage electrode input units contain undeclared electrodes")
    _unique(
        (
            f"{row['unit_type']}:{row['canonical_name']}"
            for row in input_units
        ),
        "montage.input_units canonical names",
    )
    if (
        electrode_ids.intersection(lead_ids)
        or electrode_ids.intersection(region_ids)
        or lead_ids.intersection(region_ids)
    ):
        raise ValueError("montage electrode, lead and region namespaces must be disjoint")
    if montage["analysis_reference"] == "bipolar" and not bipolar_names:
        raise ValueError("bipolar analysis_reference requires bipolar input units")

    for lead_id, endpoints in lead_endpoints.items():
        for electrode in endpoints:
            electrode_to_input_units[electrode].update(lead_to_input_units[lead_id])
            electrode_to_lateralities[electrode].update(
                lead_to_lateralities[lead_id]
            )
            electrode_to_regions[electrode].update(lead_to_regions[lead_id])

    return {
        "input_units": input_unit_ids,
        "electrodes": electrode_ids,
        "leads": lead_ids,
        "regions": region_ids,
        "laterality": set(_LATERALITIES),
        "available_input_units": {
            identifier for identifier, available in availability.items() if available
        },
        "unavailable_input_units": {
            identifier for identifier, available in availability.items() if not available
        },
        "lead_to_input_units": lead_to_input_units,
        "electrode_to_input_units": electrode_to_input_units,
        "region_to_input_units": region_to_input_units,
        "lead_endpoints": lead_endpoints,
        "direct_electrodes": electrode_names,
        "lead_to_lateralities": lead_to_lateralities,
        "electrode_to_lateralities": electrode_to_lateralities,
        "region_to_lateralities": region_to_lateralities,
        "lead_to_regions": lead_to_regions,
        "electrode_to_regions": electrode_to_regions,
        "unit_laterality": unit_laterality,
    }


def _validate_quality(payload: Mapping[str, Any], ids: Mapping[str, Any]) -> dict[str, str]:
    quality = payload["quality"]
    per_unit = quality["per_unit"]
    quality_ids = _unique(
        (str(row["unit_id"]) for row in per_unit),
        "quality.per_unit.unit_id",
    )
    if quality_ids != ids["input_units"]:
        missing = sorted(ids["input_units"].difference(quality_ids))
        extra = sorted(quality_ids.difference(ids["input_units"]))
        raise ValueError(
            f"quality.per_unit must cover every input unit exactly once; missing={missing}, extra={extra}"
        )
    for index, row in enumerate(per_unit):
        fraction = float(row["usable_fraction"])
        if row["status"] == "unusable" and abs(fraction) > _TOL:
            raise ValueError(
                f"quality.per_unit[{index}] unusable status requires zero usable_fraction"
            )
        if row["status"] in {"usable", "limited"} and fraction <= _TOL:
            raise ValueError(
                f"quality.per_unit[{index}] {row['status']} status requires positive usable_fraction"
            )
        if str(row["unit_id"]) in ids["unavailable_input_units"]:
            if row["status"] != "unusable" or abs(fraction) > _TOL:
                raise ValueError(
                    f"quality.per_unit[{index}] must mark an unavailable input as unusable with zero usable_fraction"
                )

    mean_usable = sum(float(row["usable_fraction"]) for row in per_unit) / len(per_unit)
    if abs(float(quality["usable_fraction"]) - mean_usable) > _TOL:
        raise ValueError(
            "quality.usable_fraction must equal the mean per-unit usable fraction"
        )

    feature_families = _unique(
        (str(row["family"]) for row in quality["feature_availability"]),
        "quality.feature_availability.family",
    )
    if feature_families != _FEATURE_FAMILIES:
        missing = sorted(_FEATURE_FAMILIES.difference(feature_families))
        extra = sorted(feature_families.difference(_FEATURE_FAMILIES))
        raise ValueError(
            "quality.feature_availability must explicitly cover every family; "
            f"missing={missing}, extra={extra}"
        )
    for index, row in enumerate(quality["feature_availability"]):
        reasons = row["reason_codes"]
        if row["status"] == "available" and reasons:
            raise ValueError(
                f"quality.feature_availability[{index}] available status requires no reason codes"
            )
        if row["status"] != "available" and not reasons:
            raise ValueError(
                f"quality.feature_availability[{index}] limited/not_evaluable requires reason codes"
            )

    search_bounds = tuple(float(item) for item in payload["window"]["search_interval"])
    for index, artifact in enumerate(quality["artifact_intervals"]):
        _closed_interval(
            artifact["interval"],
            f"quality.artifact_intervals[{index}].interval",
            bounds=search_bounds,
            positive=True,
        )
        _require_refs(
            (str(item) for item in artifact["affected_unit_ids"]),
            ids["input_units"],
            f"quality.artifact_intervals[{index}].affected_unit_ids",
        )
    return {
        str(row["family"]): str(row["status"])
        for row in quality["feature_availability"]
    }


def _validate_spatial_id(
    unit_type: str,
    identifier: str,
    ids: Mapping[str, Any],
    context: str,
) -> None:
    key = {
        "lead": "leads",
        "electrode": "electrodes",
        "region": "regions",
        "laterality": "laterality",
    }[unit_type]
    _require_refs((identifier,), ids[key], context)


def _support_input_units(
    support: Mapping[str, object], ids: Mapping[str, Any]
) -> set[str]:
    unit_type = str(support["unit_type"])
    identifier = str(support["id"])
    if unit_type == "lead":
        return set(ids["lead_to_input_units"].get(identifier, set()))
    if unit_type == "electrode":
        return set(ids["electrode_to_input_units"].get(identifier, set()))
    if unit_type == "region":
        return set(ids["region_to_input_units"].get(identifier, set()))
    if unit_type == "laterality":
        if identifier == "bilateral":
            return {
                unit_id
                for unit_id, laterality in ids["unit_laterality"].items()
                if laterality in {"left", "right", "bilateral"}
            }
        return {
            unit_id
            for unit_id, laterality in ids["unit_laterality"].items()
            if laterality == identifier
        }
    return set()


def _validate_measurement_source_binding(
    measurement: Mapping[str, Any],
    finding: Mapping[str, Any],
    *,
    ids: Mapping[str, Any],
    final_bounds: tuple[float, float],
    background_reference_ids: set[str],
    context: str,
) -> None:
    binding = measurement["source_binding"]
    source_units = set(str(item) for item in binding["source_unit_ids"])
    _require_refs(source_units, ids["input_units"], f"{context}.source_unit_ids")
    recording_interval = _closed_interval(
        binding["recording_interval"],
        f"{context}.recording_interval",
        bounds=final_bounds,
        positive=True,
    )
    if finding["time_interval"] is None:
        raise ValueError(f"{context} requires a time-localized parent Finding")
    finding_interval = (
        float(finding["time_interval"]["start"]),
        float(finding["time_interval"]["stop"]),
    )
    if not _covered_by(recording_interval, [finding_interval]):
        raise ValueError(
            f"{context}.recording_interval lies outside its parent Finding interval"
        )
    tensor_start, tensor_stop = (
        int(item) for item in binding["tensor_sample_interval"]
    )
    if tensor_stop <= tensor_start:
        raise ValueError(f"{context}.tensor_sample_interval must have positive length")
    bandwidth = _closed_interval(
        binding["effective_bandwidth_hz"],
        f"{context}.effective_bandwidth_hz",
        positive=True,
    )
    if bandwidth[0] < -_TOL:
        raise ValueError(f"{context}.effective_bandwidth_hz must be nonnegative")
    allowed_source_families = _FINDING_TO_SOURCE_EVIDENCE_FAMILIES[
        str(finding["family"])
    ]
    if binding["evidence_family"] not in allowed_source_families:
        raise ValueError(
            f"{context}.evidence_family is incompatible with Finding family "
            f"{finding['family']!r}"
        )
    background_ids = set(str(item) for item in binding["background_reference_ids"])
    _require_refs(
        background_ids,
        background_reference_ids,
        f"{context}.background_reference_ids",
    )
    if measurement.get("baseline_delta") is not None and not background_ids:
        raise ValueError(
            f"{context}.baseline_delta requires a bound background reference"
        )
    if finding["spatial_support"]:
        supported_units = {
            unit_id
            for support in finding["spatial_support"]
            for unit_id in _support_input_units(support, ids)
        }
        if not source_units.intersection(supported_units):
            raise ValueError(
                f"{context}.source_unit_ids do not cover parent spatial support"
            )


def _requires_high_frequency_gate(finding: Mapping[str, object]) -> bool:
    if finding["family"] == "high_frequency":
        return True
    names = [str(finding["term"])] + [
        str(row["name"]) for row in finding["measurements"]  # type: ignore[index]
    ]
    normalized = [
        name.lower().replace("-", "_").replace(".", "_").replace(":", "_")
        for name in names
    ]
    return any(
        marker in name
        for name in normalized
        for marker in _HIGH_FREQUENCY_MARKERS
    )


def _is_explicit_bilateral_synchrony_finding(
    finding: Mapping[str, Any],
    *,
    waveforms: Mapping[str, Mapping[str, Any]],
    ids: Mapping[str, Any],
) -> bool:
    term = (
        str(finding["term"])
        .lower()
        .replace("-", "_")
        .replace(".", "_")
        .replace(":", "_")
    )
    if finding["family"] != "spatial_field" or not any(
        marker in term for marker in _SYNCHRONY_TERM_MARKERS
    ):
        return False
    if not any(
        support["unit_type"] == "laterality" and support["id"] == "bilateral"
        for support in finding["spatial_support"]
    ):
        return False
    waveform_units = {
        str(unit_id)
        for waveform_id in finding["waveform_evidence_ids"]
        for unit_id in waveforms[str(waveform_id)]["unit_ids"]
    }
    observed_lateralities = {
        ids["unit_laterality"][unit_id]
        for unit_id in waveform_units
        if unit_id in ids["unit_laterality"]
    }
    return {"left", "right"}.issubset(observed_lateralities)


def _candidate_lateralities(
    candidate_type: str, identifier: str, ids: Mapping[str, Any]
) -> set[str]:
    if candidate_type == "laterality":
        return {identifier}
    key = {
        "lead": "lead_to_lateralities",
        "electrode": "electrode_to_lateralities",
        "region": "region_to_lateralities",
    }[candidate_type]
    return set(ids[key].get(identifier, set()))


def _candidate_regions(
    candidate_type: str, identifier: str, ids: Mapping[str, Any]
) -> set[str]:
    if candidate_type == "region":
        return {identifier}
    if candidate_type == "lead":
        return set(ids["lead_to_regions"].get(identifier, set()))
    if candidate_type == "electrode":
        return set(ids["electrode_to_regions"].get(identifier, set()))
    return set()


def _intervals_overlap(
    left: tuple[float, float], right: tuple[float, float]
) -> bool:
    return max(left[0], right[0]) < min(left[1], right[1]) - _TOL


def _validate_finding_phase_time_consistency(
    finding: Mapping[str, Any],
    *,
    context: str,
    phase_bounds: Mapping[str, tuple[float, float] | None],
    onset_bounds: tuple[float, float] | None,
) -> None:
    """Fail closed when a Finding's declared phase and physical time disagree.

    Phase membership is allowed to remain soft, but every non-zero component
    must have positive temporal overlap with the corresponding observed phase.
    Evidence promoted to an onset hypothesis is additionally anchored to the
    inferred onset boundary and cannot be baseline-, recovery-, or late-only
    evidence.
    """

    membership = {
        str(name): float(weight)
        for name, weight in finding["phase_membership"].items()
    }
    interval = finding["time_interval"]
    if interval is None:
        if finding["status"] == "present":
            raise ValueError(f"{context} status=present requires a time_interval")
        if any(weight > _TOL for weight in membership.values()):
            raise ValueError(
                f"{context} non-zero phase_membership requires a time_interval"
            )
        return

    span = (float(interval["start"]), float(interval["stop"]))
    phase_key_to_window_key = {
        "baseline": "baseline",
        "early_ictal": "early_ictal",
        "evolved_ictal": "evolution",
        "recovery": "recovery",
    }
    for phase_key, weight in membership.items():
        if weight <= _TOL:
            continue
        bounds = phase_bounds[phase_key_to_window_key[phase_key]]
        if bounds is None:
            raise ValueError(
                f"{context}.phase_membership.{phase_key} is positive but the "
                "corresponding window phase is not observed"
            )
        if not _intervals_overlap(span, bounds):
            raise ValueError(
                f"{context}.phase_membership.{phase_key} is positive but "
                "time_interval does not overlap that phase"
            )

    if finding["evidence_role"] != "onset_support":
        return
    if finding["family"] not in _ONSET_SUPPORT_FAMILIES:
        raise ValueError(
            f"{context} onset_support family {finding['family']!r} is outside the "
            "frozen onset family allowlist"
        )
    if membership["baseline"] > _TOL or membership["recovery"] > _TOL:
        raise ValueError(
            f"{context} onset_support cannot be baseline or recovery evidence"
        )
    if membership["early_ictal"] <= _TOL:
        raise ValueError(f"{context} onset_support requires early_ictal membership")
    if membership["evolved_ictal"] > membership["early_ictal"] + _TOL:
        raise ValueError(
            f"{context} onset_support cannot be dominated by evolved/late-spread evidence"
        )
    if onset_bounds is None:
        raise ValueError(
            f"{context} onset_support requires an observed/estimated onset boundary; "
            "a detector anchor cannot substitute for onset"
        )
    intersects_onset_boundary = (
        span[0] <= onset_bounds[1] + _TOL
        and span[1] >= onset_bounds[0] - _TOL
    )
    if not intersects_onset_boundary:
        raise ValueError(
            f"{context} onset_support must intersect window.onset_interval"
        )
    recovery = phase_bounds["recovery"]
    if recovery is not None and _intervals_overlap(span, recovery):
        raise ValueError(f"{context} onset_support cannot extend into recovery")


def _validate_finding_signal_support(
    finding: Mapping[str, Any],
    *,
    waveforms: Mapping[str, Mapping[str, Any]],
    ids: Mapping[str, Any],
    artifacts: Sequence[Mapping[str, Any]],
    context: str,
) -> None:
    if finding["status"] == "not_evaluable":
        raise ValueError(f"{context} cannot reference a not_evaluable Finding")
    if finding["time_interval"] is None:
        raise ValueError(f"{context} requires a time-localized Finding")
    if not finding["spatial_support"]:
        raise ValueError(f"{context} requires spatial Finding support")
    if not finding["waveform_evidence_ids"]:
        raise ValueError(f"{context} requires waveform evidence")

    referenced_waveforms = [
        waveforms[str(identifier)] for identifier in finding["waveform_evidence_ids"]
    ]
    waveform_units = {
        str(unit_id)
        for waveform in referenced_waveforms
        for unit_id in waveform["unit_ids"]
    }
    unavailable_units = waveform_units.difference(ids["available_input_units"])
    if unavailable_units:
        raise ValueError(
            f"{context} waveform uses unavailable units: {sorted(unavailable_units)}"
        )
    if not any(
        _support_input_units(support, ids).intersection(waveform_units)
        for support in finding["spatial_support"]
    ):
        raise ValueError(f"{context} waveform units do not cover its spatial support")

    finding_span = (
        float(finding["time_interval"]["start"]),
        float(finding["time_interval"]["stop"]),
    )
    waveform_spans = [
        (float(row["interval"][0]), float(row["interval"][1]))
        for row in referenced_waveforms
    ]
    if not _covered_by(finding_span, waveform_spans):
        raise ValueError(f"{context} Finding interval is not covered by waveform evidence")

    for waveform in referenced_waveforms:
        waveform_span = (
            float(waveform["interval"][0]),
            float(waveform["interval"][1]),
        )
        waveform_unit_ids = set(str(item) for item in waveform["unit_ids"])
        for artifact in artifacts:
            artifact_span = (
                float(artifact["interval"][0]),
                float(artifact["interval"][1]),
            )
            affected = set(str(item) for item in artifact["affected_unit_ids"])
            if (
                _intervals_overlap(waveform_span, artifact_span)
                and waveform_unit_ids.intersection(affected)
            ):
                raise ValueError(f"{context} waveform overlaps an artifact interval")


def _dominant_score(
    rows: Sequence[Mapping[str, object]],
) -> tuple[str, float] | None:
    if not rows:
        return None
    ordered = sorted(
        ((str(row["name"]), float(row["score"])) for row in rows),
        key=lambda item: (-item[1], item[0]),
    )
    if len(ordered) > 1 and abs(ordered[0][1] - ordered[1][1]) <= _TOL:
        return None
    return ordered[0]


def _assert_acyclic(edges: Sequence[tuple[str, str]], context: str) -> None:
    adjacency: dict[str, set[str]] = {}
    indegree: dict[str, int] = {}
    for source, target in edges:
        adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(target, set())
        indegree.setdefault(source, 0)
        indegree[target] = indegree.get(target, 0) + 1
    ready = [node for node, degree in indegree.items() if degree == 0]
    visited = 0
    while ready:
        node = ready.pop()
        visited += 1
        for target in adjacency[node]:
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    if visited != len(indegree):
        raise ValueError(f"{context} contains a directed cycle")


def validate_event_eeg_findings_payload(
    value: object,
    *,
    trusted_capability_qualification_receipts: (
        Mapping[str, Mapping[str, object]] | None
    ) = None,
    trusted_term_decision_receipts: (
        Mapping[str, Mapping[str, object]] | None
    ) = None,
    trusted_qualification_receipts: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, Any]:
    """Validate and canonical-copy one EEG-only variable-length event graph.

    ``trusted_qualification_receipts`` is retained as a compatibility alias
    for the explicitly named capability registry.  It never supplies the
    per-event clinical term decisions.
    """

    if type(value) is not dict:
        raise TypeError("event_eeg_findings_v1 payload must be an object")
    if (
        trusted_capability_qualification_receipts is not None
        and trusted_qualification_receipts is not None
    ):
        raise ValueError(
            "supply only trusted_capability_qualification_receipts or the "
            "legacy trusted_qualification_receipts alias"
        )
    capability_registry_input = (
        trusted_capability_qualification_receipts
        if trusted_capability_qualification_receipts is not None
        else trusted_qualification_receipts
    )
    trusted_capability_receipts = _trusted_receipt_registry(
        capability_registry_input,
        name="trusted_capability_qualification_receipts",
    )
    trusted_term_receipts = _trusted_receipt_registry(
        trusted_term_decision_receipts,
        name="trusted_term_decision_receipts",
    )
    _reject_nonfinite(value)
    errors = sorted(_schema_validator().iter_errors(value), key=lambda item: list(item.path))
    if errors:
        rendered = "; ".join(f"{_path(error)}: {error.message}" for error in errors[:8])
        if len(errors) > 8:
            rendered += f"; ... {len(errors) - 8} more error(s)"
        raise ValueError(f"event_eeg_findings_v1 schema validation failed: {rendered}")

    payload: dict[str, Any] = deepcopy(value)
    duration = float(payload["coordinates"]["recording_duration_seconds"])
    recording_bounds = (0.0, duration)
    window = payload["window"]
    search_bounds = _closed_interval(
        window["search_interval"],
        "window.search_interval",
        bounds=recording_bounds,
        positive=True,
    )
    final_bounds = _closed_interval(
        window["final_interval"],
        "window.final_interval",
        bounds=search_bounds,
        positive=True,
    )
    onset_bounds, onset_status = _onset_boundary_estimate(
        window["onset_interval"],
        "window.onset_interval",
        bounds=final_bounds,
    )
    onset_support_bounds = (
        onset_bounds
        if onset_status in {"observed", "interval_estimate", "legacy_interval_estimate"}
        else None
    )
    phase_bounds = {
        phase_name: _phase_interval(
            window[phase_name], f"window.{phase_name}", bounds=final_bounds
        )
        for phase_name in ("baseline", "early_ictal", "evolution", "recovery")
    }
    offset_bounds = _boundary_estimate(
        window["offset_interval"], "window.offset_interval", bounds=final_bounds
    )
    baseline_bounds = phase_bounds["baseline"]
    early_bounds = phase_bounds["early_ictal"]
    evolution_bounds = phase_bounds["evolution"]
    recovery_bounds = phase_bounds["recovery"]
    if (
        onset_bounds is not None
        and baseline_bounds is not None
        and baseline_bounds[1] > onset_bounds[0] + _TOL
    ):
        raise ValueError("window.baseline must end no later than onset_interval.lower")
    if onset_bounds is not None and early_bounds is not None and (
        early_bounds[0] > onset_bounds[1] + _TOL
        or early_bounds[1] < onset_bounds[0] - _TOL
    ):
        raise ValueError("window.early_ictal must overlap the onset boundary interval")
    if (
        onset_bounds is not None
        and evolution_bounds is not None
        and evolution_bounds[0] < onset_bounds[0] - _TOL
    ):
        raise ValueError("window.evolution cannot start before onset_interval.lower")
    if early_bounds is not None and evolution_bounds is not None and (
        evolution_bounds[0] < early_bounds[0] - _TOL
    ):
        raise ValueError("window.evolution cannot start before early_ictal")
    if (
        onset_bounds is not None
        and offset_bounds is not None
        and offset_bounds[1] < onset_bounds[0] - _TOL
    ):
        raise ValueError("window.offset_interval cannot precede onset_interval")
    if recovery_bounds is not None:
        if offset_bounds is None:
            raise ValueError("window.recovery requires an offset boundary")
        if recovery_bounds[0] < offset_bounds[0] - _TOL:
            raise ValueError("window.recovery cannot start before offset_interval.lower")
    if baseline_bounds is None and not window["left_censored"]:
        raise ValueError("missing baseline requires left_censored=true")
    if onset_status == "censored" and not window["left_censored"]:
        raise ValueError("censored onset requires left_censored=true")
    if offset_bounds is None and not window["right_censored"]:
        raise ValueError("missing offset requires right_censored=true")
    if recovery_bounds is None and not window["right_censored"]:
        raise ValueError("missing recovery requires right_censored=true")
    if window["search_cap_censored"] and not (
        window["left_censored"] or window["right_censored"]
    ):
        raise ValueError("search_cap_censored requires unresolved left or right boundary")

    background_context = payload["context"]
    queried = _interval_list(
        background_context["queried_intervals"],
        "context.queried_intervals",
        bounds=recording_bounds,
    )
    local_background = _interval_list(
        background_context["local_background_intervals"],
        "context.local_background_intervals",
        bounds=recording_bounds,
    )
    distant_background = _interval_list(
        background_context["distant_background_intervals"],
        "context.distant_background_intervals",
        bounds=recording_bounds,
    )
    if not _covered_by(final_bounds, queried):
        raise ValueError("window.final_interval must be covered by a queried interval")
    for name, intervals in (
        ("local_background_intervals", local_background),
        ("distant_background_intervals", distant_background),
    ):
        for index, interval in enumerate(intervals):
            if not _covered_by(interval, queried):
                raise ValueError(f"context.{name}[{index}] is not covered by queried_intervals")
    if onset_bounds is None:
        if local_background:
            raise ValueError(
                "context.local_background_intervals require an observed/estimated onset; "
                "a detector anchor cannot establish pre-onset background"
            )
    elif any(interval[1] > onset_bounds[0] + _TOL for interval in local_background):
        raise ValueError("context.local_background_intervals must end before onset.lower")
    if any(
        max(interval[0], final_bounds[0]) < min(interval[1], final_bounds[1]) - _TOL
        for interval in distant_background
    ):
        raise ValueError("context.distant_background_intervals must not overlap final_interval")
    background_status = str(background_context["background_status"])
    if background_status == "unavailable":
        if local_background or distant_background:
            raise ValueError("background_status=unavailable requires empty background intervals")
        if (
            background_context["background_bank_id"] is not None
            or background_context["selection_receipt_id"] is not None
        ):
            raise ValueError(
                "background_status=unavailable requires null background/selection IDs"
            )
        if float(background_context["contamination_risk"]) < 1.0 - _TOL:
            raise ValueError("unavailable background requires contamination_risk=1")
        limitation_codes = {
            str(row["code"]) for row in payload["limitations"]
        }
        if "background_unavailable" not in limitation_codes:
            raise ValueError(
                "background_status=unavailable requires background_unavailable limitation"
            )
    else:
        if not local_background and not distant_background:
            raise ValueError("available/limited background requires EEG background intervals")
        if (
            background_context["background_bank_id"] is None
            or background_context["selection_receipt_id"] is None
        ):
            raise ValueError(
                "available/limited background requires background/selection IDs"
            )
    background_reference_ids = {
        str(identifier)
        for identifier in (
            background_context["background_bank_id"],
            background_context["selection_receipt_id"],
        )
        if identifier is not None
    }

    ids = _validate_montage(payload)
    feature_status = _validate_quality(payload, ids)

    # Capability receipts answer whether a producer/family/term was validated
    # for this target domain.  They do not decide that the term is present in
    # this event.
    capability_rows = payload["qualification_receipts"]
    capability_receipt_ids = _unique(
        (str(row["receipt_id"]) for row in capability_rows),
        "qualification_receipts.receipt_id",
    )
    capability_receipts = {
        str(row["receipt_id"]): row for row in capability_rows
    }
    for receipt_id, receipt in capability_receipts.items():
        trusted = trusted_capability_receipts.get(receipt_id)
        if trusted is None:
            raise ValueError(
                f"capability qualification receipt {receipt_id!r} is absent from "
                "the host trusted registry"
            )
        if _canonical_json(receipt) != _canonical_json(trusted):
            raise ValueError(
                f"capability qualification receipt {receipt_id!r} differs from "
                "the host trusted registry"
            )

    # Term-decision receipts answer whether protected terminology passed its
    # exact event-level IFCN/ACNS-derived rules.  Revalidate both the embedded
    # row and the independent host registry row; a self-signed receipt is not
    # a trust root even when its content hash happens to be internally valid.
    term_decision_rows = payload["term_decision_receipts"]
    term_decision_receipt_ids = _unique(
        (str(row["receipt_id"]) for row in term_decision_rows),
        "term_decision_receipts.receipt_id",
    )
    term_decision_receipts: dict[str, dict[str, Any]] = {}
    expected_term_source_binding = event_term_decision_source_binding_sha256(
        payload
    )
    for index, raw_receipt in enumerate(term_decision_rows):
        receipt = validate_clinical_eeg_term_qualification(raw_receipt)
        receipt_id = str(receipt["receipt_id"])
        trusted = trusted_term_receipts.get(receipt_id)
        if trusted is None:
            raise ValueError(
                f"term-decision receipt {receipt_id!r} is absent from the host "
                "trusted registry"
            )
        validated_trusted = validate_clinical_eeg_term_qualification(trusted)
        if _canonical_json(receipt) != _canonical_json(validated_trusted):
            raise ValueError(
                f"term-decision receipt {receipt_id!r} differs from the host "
                "trusted registry"
            )
        if receipt["event_id"] != payload["event_id"]:
            raise ValueError(
                f"term_decision_receipts[{index}] belongs to a different event"
            )
        if receipt["source_binding_sha256"] != expected_term_source_binding:
            raise ValueError(
                f"term_decision_receipts[{index}] source binding does not close "
                "the event Findings evidence ledger"
            )
        term_decision_receipts[receipt_id] = receipt

    waveform_rows = payload["waveform_evidence"]
    waveform_ids = _unique(
        (str(row["waveform_evidence_id"]) for row in waveform_rows),
        "waveform_evidence.waveform_evidence_id",
    )
    for index, waveform in enumerate(waveform_rows):
        _closed_interval(
            waveform["interval"],
            f"waveform_evidence[{index}].interval",
            bounds=final_bounds,
            positive=True,
        )
        _require_refs(
            (str(item) for item in waveform["unit_ids"]),
            ids["input_units"],
            f"waveform_evidence[{index}].unit_ids",
        )
        if waveform["signal_sha256"] != payload["provenance"]["signal_sha256"]:
            raise ValueError(
                f"waveform_evidence[{index}].signal_sha256 does not match provenance.signal_sha256"
            )
    waveforms = {
        str(row["waveform_evidence_id"]): row for row in waveform_rows
    }

    finding_rows = payload["findings"]
    evidence_ids = _unique(
        (str(row["evidence_id"]) for row in finding_rows),
        "findings.evidence_id",
    )
    finding_by_id = {str(row["evidence_id"]): row for row in finding_rows}
    used_capability_receipts: set[str] = set()
    used_term_decision_receipts: set[str] = set()
    for index, finding in enumerate(finding_rows):
        context = f"findings[{index}]"
        term = validate_event_finding_term(
            finding["term"],
            family=finding["family"],
            assertion_level=finding["assertion_level"],
            context=context,
        )
        phase_total = sum(float(value) for value in finding["phase_membership"].values())
        if finding["status"] == "not_evaluable" and abs(phase_total) > _TOL:
            raise ValueError(
                f"{context}.phase_membership must be all zero when not_evaluable"
            )
        if finding["status"] != "not_evaluable" and abs(phase_total - 1.0) > _TOL:
            raise ValueError(
                f"{context}.phase_membership must sum to 1; got {phase_total:.12g}"
            )
        if finding["time_interval"] is not None:
            _observed_span(
                finding["time_interval"],
                f"{context}.time_interval",
                bounds=final_bounds,
            )
        _validate_finding_phase_time_consistency(
            finding,
            context=context,
            phase_bounds=phase_bounds,
            onset_bounds=onset_support_bounds,
        )
        support_keys = [
            f"{support['unit_type']}:{support['id']}"
            for support in finding["spatial_support"]
        ]
        _unique(support_keys, f"{context}.spatial_support")
        for support_index, support in enumerate(finding["spatial_support"]):
            _validate_spatial_id(
                str(support["unit_type"]),
                str(support["id"]),
                ids,
                f"{context}.spatial_support[{support_index}].id",
            )
        _unique(
            (str(row["name"]) for row in finding["measurements"]),
            f"{context}.measurements.name",
        )
        for measurement_index, measurement in enumerate(finding["measurements"]):
            _validate_measurement_source_binding(
                measurement,
                finding,
                ids=ids,
                final_bounds=final_bounds,
                background_reference_ids=background_reference_ids,
                context=f"{context}.measurements[{measurement_index}].source_binding",
            )
        _require_refs(
            (str(item) for item in finding["waveform_evidence_ids"]),
            waveform_ids,
            f"{context}.waveform_evidence_ids",
        )

        availability_family = _FINDING_TO_AVAILABILITY_FAMILY.get(
            str(finding["family"])
        )
        if (
            finding["status"] != "not_evaluable"
            and availability_family is not None
            and feature_status[availability_family] == "not_evaluable"
        ):
            raise ValueError(
                f"{context} contradicts quality.feature_availability for {availability_family}"
            )
        if (
            finding["status"] == "not_evaluable"
            and availability_family is not None
            and feature_status[availability_family] == "available"
        ):
            raise ValueError(
                f"{context} not_evaluable status contradicts available feature family "
                f"{availability_family}"
            )
        if (
            finding["status"] != "not_evaluable"
            and _requires_high_frequency_gate(finding)
            and feature_status["high_frequency"] == "not_evaluable"
        ):
            raise ValueError(
                f"{context} HFO/LVFA claim requires an evaluable high_frequency gate"
            )
        if finding["status"] == "not_evaluable" and any(
            (
                finding["time_interval"] is not None,
                bool(finding["spatial_support"]),
                bool(finding["measurements"]),
                bool(finding["waveform_evidence_ids"]),
            )
        ):
            raise ValueError(
                f"{context} not_evaluable assertion must not carry positive evidence"
            )
        if background_status == "unavailable":
            if any(
                measurement.get("baseline_delta") is not None
                for measurement in finding["measurements"]
            ):
                raise ValueError(
                    f"{context} baseline_delta requires an available/limited EEG background"
                )
            if float(finding["uncertainty"]["background"]) < 1.0 - _TOL:
                raise ValueError(
                    f"{context} must carry background uncertainty=1 when background is unavailable"
                )

        capability_receipt_id = finding["qualification_receipt_id"]
        term_decision_receipt_id = finding["term_decision_receipt_id"]
        if finding["assertion_level"] == "clinically_qualified":
            if capability_receipt_id is None:
                raise ValueError(
                    f"{context} requires a capability qualification receipt"
                )
            if term_decision_receipt_id is None:
                raise ValueError(f"{context} requires a per-event term-decision receipt")
            if finding["status"] == "not_evaluable":
                raise ValueError(f"{context} cannot qualify a not_evaluable assertion")
            _require_refs(
                (str(capability_receipt_id),),
                capability_receipt_ids,
                f"{context}.qualification_receipt_id",
            )
            _require_refs(
                (str(term_decision_receipt_id),),
                term_decision_receipt_ids,
                f"{context}.term_decision_receipt_id",
            )
            capability_receipt = capability_receipts[
                str(capability_receipt_id)
            ]
            if finding["family"] not in capability_receipt["qualified_families"]:
                raise ValueError(f"{context} family is outside its qualification receipt")
            if term not in capability_receipt["qualified_terms"]:
                raise ValueError(f"{context} term is outside its qualification receipt")
            decision_receipt = term_decision_receipts[
                str(term_decision_receipt_id)
            ]
            if term not in decision_receipt["qualified_terms"]:
                raise ValueError(
                    f"{context} term is outside its per-event term-decision receipt"
                )
            if decision_receipt["event_id"] != payload["event_id"]:
                raise ValueError(f"{context} term-decision receipt event mismatch")
            if term not in PROTECTED_EEG_ONLY_TERMS:
                raise ValueError(
                    f"{context} clinically qualified term is outside the protected registry"
                )
            if not finding["waveform_evidence_ids"]:
                raise ValueError(f"{context} clinically qualified assertion requires waveform evidence")
            non_laterality_support = [
                support
                for support in finding["spatial_support"]
                if support["unit_type"] != "laterality"
            ]
            if not non_laterality_support:
                raise ValueError(
                    f"{context} clinically qualified assertion requires spatial waveform support"
                )
            referenced_waveforms = [
                waveforms[str(identifier)]
                for identifier in finding["waveform_evidence_ids"]
            ]
            waveform_units = {
                str(unit_id)
                for waveform in referenced_waveforms
                for unit_id in waveform["unit_ids"]
            }
            unavailable_units = waveform_units.difference(
                ids["available_input_units"]
            )
            if unavailable_units:
                raise ValueError(
                    f"{context} clinically qualified waveform uses unavailable units: "
                    f"{sorted(unavailable_units)}"
                )
            for support in non_laterality_support:
                mapped_units = _support_input_units(support, ids)
                if not mapped_units.intersection(waveform_units):
                    raise ValueError(
                        f"{context} waveform units do not cover spatial support "
                        f"{support['unit_type']}:{support['id']}"
                    )
            if finding["time_interval"] is not None:
                finding_span = (
                    float(finding["time_interval"]["start"]),
                    float(finding["time_interval"]["stop"]),
                )
                waveform_spans = [
                    (float(row["interval"][0]), float(row["interval"][1]))
                    for row in referenced_waveforms
                ]
                if not _covered_by(finding_span, waveform_spans):
                    raise ValueError(
                        f"{context} finding interval is not covered by its waveform evidence"
                    )
            for waveform in referenced_waveforms:
                waveform_span = (
                    float(waveform["interval"][0]),
                    float(waveform["interval"][1]),
                )
                waveform_unit_ids = set(str(item) for item in waveform["unit_ids"])
                for artifact in payload["quality"]["artifact_intervals"]:
                    artifact_span = (
                        float(artifact["interval"][0]),
                        float(artifact["interval"][1]),
                    )
                    overlaps = max(waveform_span[0], artifact_span[0]) < min(
                        waveform_span[1], artifact_span[1]
                    ) - _TOL
                    affected = set(str(item) for item in artifact["affected_unit_ids"])
                    if overlaps and waveform_unit_ids.intersection(affected):
                        raise ValueError(
                            f"{context} qualified waveform overlaps an artifact interval"
                        )
            used_capability_receipts.add(str(capability_receipt_id))
            used_term_decision_receipts.add(str(term_decision_receipt_id))
        elif capability_receipt_id is not None or term_decision_receipt_id is not None:
            raise ValueError(
                f"{context} capability and term-decision receipt IDs must both be "
                "null unless assertion_level=clinically_qualified"
            )
        if finding["assertion_level"] == "measured" and not finding["measurements"]:
            raise ValueError(f"{context} measured assertion requires a deterministic measurement")

    unused_capability_receipts = capability_receipt_ids.difference(
        used_capability_receipts
    )
    if unused_capability_receipts:
        raise ValueError(
            "qualification_receipts contains unreferenced capability receipts: "
            f"{sorted(unused_capability_receipts)}"
        )
    unused_term_decisions = term_decision_receipt_ids.difference(
        used_term_decision_receipts
    )
    if unused_term_decisions:
        raise ValueError(
            "term_decision_receipts contains unreferenced per-event decisions: "
            f"{sorted(unused_term_decisions)}"
        )

    spatial = payload["spatial_onset"]
    _distribution(
        spatial["phenotype_scores"],
        "spatial_onset.phenotype_scores",
        allowed_names=_PHENOTYPES,
    )
    _distribution(
        spatial["laterality_scores"],
        "spatial_onset.laterality_scores",
        allowed_names=_LATERALITIES,
    )
    region_names = _distribution(
        spatial["region_scores"], "spatial_onset.region_scores"
    )
    _require_refs(region_names, ids["regions"], "spatial_onset.region_scores.name")

    supporting = set(str(item) for item in spatial["supporting_evidence_ids"])
    contradictory = set(str(item) for item in spatial["contradictory_evidence_ids"])
    _require_refs(supporting, evidence_ids, "spatial_onset.supporting_evidence_ids")
    _require_refs(contradictory, evidence_ids, "spatial_onset.contradictory_evidence_ids")
    if supporting.intersection(contradictory):
        raise ValueError(
            "spatial_onset supporting and contradictory evidence must be disjoint"
        )
    for evidence_id in sorted(supporting):
        finding = finding_by_id[evidence_id]
        if finding["status"] != "present":
            raise ValueError(
                f"spatial_onset supporting evidence {evidence_id!r} must have "
                "status=present"
            )
        if finding["evidence_role"] != "onset_support":
            raise ValueError(
                f"spatial_onset supporting evidence {evidence_id!r} must have "
                "evidence_role=onset_support"
            )
        if finding["family"] not in _ONSET_SUPPORT_FAMILIES:
            raise ValueError(
                f"spatial_onset supporting evidence {evidence_id!r} family is "
                "outside the frozen onset family allowlist"
            )
    for evidence_id in sorted(contradictory):
        finding = finding_by_id[evidence_id]
        if finding["status"] != "present":
            raise ValueError(
                f"spatial_onset contradictory evidence {evidence_id!r} must have "
                "status=present"
            )
        if finding["evidence_role"] != "contradiction":
            raise ValueError(
                f"spatial_onset contradictory evidence {evidence_id!r} must have "
                "evidence_role=contradiction"
            )
    for evidence_id in sorted(supporting | contradictory):
        _validate_finding_signal_support(
            finding_by_id[evidence_id],
            waveforms=waveforms,
            ids=ids,
            artifacts=payload["quality"]["artifact_intervals"],
            context=f"spatial_onset evidence {evidence_id!r}",
        )

    per_unit_keys = [
        f"{row['unit_type']}:{row['unit_id']}"
        for row in spatial["per_unit_intervals"]
    ]
    _unique(per_unit_keys, "spatial_onset.per_unit_intervals")
    per_unit_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for index, row in enumerate(spatial["per_unit_intervals"]):
        unit_type = str(row["unit_type"])
        unit_id = str(row["unit_id"])
        _validate_spatial_id(
            unit_type,
            unit_id,
            ids,
            f"spatial_onset.per_unit_intervals[{index}].unit_id",
        )
        per_unit_by_key[(unit_type, unit_id)] = row
        interval = row["interval"]
        if row["status"] == "not_evaluable":
            if interval is not None:
                raise ValueError(
                    f"spatial_onset.per_unit_intervals[{index}].interval must be null when not_evaluable"
                )
        elif interval is None:
            raise ValueError(
                f"spatial_onset.per_unit_intervals[{index}].interval is required"
            )
        else:
            _time_interval(
                interval,
                f"spatial_onset.per_unit_intervals[{index}].interval",
                bounds=final_bounds,
            )
            mapped_units = _support_input_units(
                {"unit_type": unit_type, "id": unit_id}, ids
            )
            if not mapped_units.intersection(ids["available_input_units"]):
                raise ValueError(
                    f"spatial_onset.per_unit_intervals[{index}] uses no available input unit"
                )
    if onset_support_bounds is None and any(
        row["status"] != "not_evaluable"
        for row in spatial["per_unit_intervals"]
    ):
        raise ValueError(
            "censored/not_observed/indeterminate onset requires all per-unit onset "
            "intervals to be not_evaluable"
        )
    if onset_support_bounds is None and spatial["recruitment_order"]:
        raise ValueError(
            "censored/not_observed/indeterminate onset cannot assert a recruitment order"
        )

    directed_edges: list[tuple[str, str]] = []
    edge_keys: list[str] = []
    unordered_edges: set[frozenset[str]] = set()
    for index, relation in enumerate(spatial["recruitment_order"]):
        from_type = str(relation["from_type"])
        from_id = str(relation["from_id"])
        to_type = str(relation["to_type"])
        to_id = str(relation["to_id"])
        from_key = (from_type, from_id)
        to_key = (to_type, to_id)
        from_node = f"{from_type}:{from_id}"
        to_node = f"{to_type}:{to_id}"
        if from_key == to_key:
            raise ValueError(f"spatial_onset.recruitment_order[{index}] cannot be a self relation")
        _validate_spatial_id(
            from_type,
            from_id,
            ids,
            f"spatial_onset.recruitment_order[{index}].from_id",
        )
        _validate_spatial_id(
            to_type,
            to_id,
            ids,
            f"spatial_onset.recruitment_order[{index}].to_id",
        )
        if from_key not in per_unit_by_key or to_key not in per_unit_by_key:
            raise ValueError(
                f"spatial_onset.recruitment_order[{index}] endpoints require matching "
                "per_unit_intervals"
            )
        source_interval = per_unit_by_key[from_key]["interval"]
        target_interval = per_unit_by_key[to_key]["interval"]
        if source_interval is None or target_interval is None:
            raise ValueError(
                f"spatial_onset.recruitment_order[{index}] endpoints must be evaluable"
            )
        delay_lower, delay_upper = _time_interval(
            relation["delay_interval"],
            f"spatial_onset.recruitment_order[{index}].delay_interval",
        )
        derived_lower = float(target_interval["lower"]) - float(
            source_interval["upper"]
        )
        derived_upper = float(target_interval["upper"]) - float(
            source_interval["lower"]
        )
        if (
            abs(delay_lower - derived_lower) > _TOL
            or abs(delay_upper - derived_upper) > _TOL
        ):
            raise ValueError(
                f"spatial_onset.recruitment_order[{index}].delay_interval is inconsistent "
                "with endpoint onset intervals"
            )
        if (
            "median" in relation["delay_interval"]
            and "median" in source_interval
            and "median" in target_interval
        ):
            expected_median = float(target_interval["median"]) - float(
                source_interval["median"]
            )
            if abs(float(relation["delay_interval"]["median"]) - expected_median) > _TOL:
                raise ValueError(
                    f"spatial_onset.recruitment_order[{index}].delay_interval.median "
                    "is inconsistent with endpoint medians"
                )
        endpoint_resolution = max(
            float(source_interval["resolution_seconds"]),
            float(target_interval["resolution_seconds"]),
        )
        if (
            float(relation["delay_interval"]["resolution_seconds"])
            < endpoint_resolution - _TOL
        ):
            raise ValueError(
                f"spatial_onset.recruitment_order[{index}].delay_interval claims "
                "finer resolution than its endpoints"
            )
        if derived_upper < -endpoint_resolution - _TOL:
            raise ValueError(
                f"spatial_onset.recruitment_order[{index}] direction contradicts onset intervals"
            )
        if derived_lower > endpoint_resolution + _TOL:
            expected_relation = "precedes"
        elif (
            derived_lower >= -endpoint_resolution - _TOL
            and derived_upper <= endpoint_resolution + _TOL
        ):
            expected_relation = "near_synchronous"
        else:
            expected_relation = "order_unresolved"
        if relation["relation_status"] != expected_relation:
            raise ValueError(
                f"spatial_onset.recruitment_order[{index}].relation_status must be "
                f"{expected_relation} from endpoint intervals"
            )
        relation_evidence = set(str(item) for item in relation["evidence_ids"])
        _require_refs(
            relation_evidence,
            evidence_ids,
            f"spatial_onset.recruitment_order[{index}].evidence_ids",
        )
        for evidence_id in relation_evidence:
            finding = finding_by_id[evidence_id]
            if finding["family"] != "spatial_recruitment":
                raise ValueError(
                    f"spatial_onset.recruitment_order[{index}] requires "
                    "spatial_recruitment evidence"
                )
            if (
                finding["status"] != "present"
                or finding["evidence_role"] != "spread_support"
            ):
                raise ValueError(
                    f"spatial_onset.recruitment_order[{index}] requires present "
                    "spread_support evidence"
                )
            _validate_finding_signal_support(
                finding,
                waveforms=waveforms,
                ids=ids,
                artifacts=payload["quality"]["artifact_intervals"],
                context=(
                    f"spatial_onset.recruitment_order[{index}] evidence "
                    f"{evidence_id!r}"
                ),
            )
            supported_nodes = {
                f"{support['unit_type']}:{support['id']}"
                for support in finding["spatial_support"]
            }
            if not {from_node, to_node}.issubset(supported_nodes):
                raise ValueError(
                    f"spatial_onset.recruitment_order[{index}] evidence does not "
                    "spatially support both endpoints"
                )
        edge_keys.append(f"{from_node}->{to_node}")
        unordered = frozenset((from_node, to_node))
        if unordered in unordered_edges:
            raise ValueError(
                "spatial_onset.recruitment_order contains duplicate or reverse edges"
            )
        unordered_edges.add(unordered)
        directed_edges.append((from_node, to_node))
    _unique(edge_keys, "spatial_onset.recruitment_order edges")
    _assert_acyclic(directed_edges, "spatial_onset.recruitment_order")

    top_k = spatial["top_k"]
    ranks = [int(row["rank"]) for row in top_k]
    if ranks != list(range(1, len(top_k) + 1)):
        raise ValueError("spatial_onset.top_k ranks must be contiguous and ordered from 1")
    _unique(
        (f"{row['candidate_type']}:{row['candidate_id']}" for row in top_k),
        "spatial_onset.top_k candidates",
    )
    allowed_resolution = str(spatial["allowed_resolution"])
    localization_status = str(spatial["localization_status"])
    expected_candidate_type = _RESOLUTION_TO_CANDIDATE_TYPE.get(allowed_resolution)
    if onset_support_bounds is None and (
        localization_status != "nonlocalizable"
        or allowed_resolution != "none"
        or top_k
        or supporting
    ):
        raise ValueError(
            "censored/not_observed/indeterminate onset cannot support spatial onset "
            "localization; a detector anchor cannot substitute for onset"
        )
    if allowed_resolution == "none" and top_k:
        raise ValueError(
            f"spatial_onset.top_k must be empty when allowed_resolution={allowed_resolution}"
        )
    if allowed_resolution != "none" and not top_k:
        raise ValueError(
            "spatial_onset.top_k is required when a spatial resolution is allowed"
        )
    if localization_status == "ranked_candidates" and allowed_resolution == "none":
        raise ValueError(
            "localization_status=ranked_candidates requires a spatial resolution"
        )
    if localization_status in {"phenotype_only", "nonlocalizable"} and (
        allowed_resolution != "none" or top_k
    ):
        raise ValueError(
            f"localization_status={localization_status} requires allowed_resolution=none "
            "and empty top_k"
        )
    if allowed_resolution != "none" and not spatial["laterality_scores"]:
        raise ValueError("spatial localization requires laterality_scores")
    if allowed_resolution in {"lead", "electrode", "region"} and not spatial["region_scores"]:
        raise ValueError("lead/electrode/region localization requires region_scores")

    semantics = {str(row["score_semantics"]) for row in top_k}
    if len(semantics) > 1:
        raise ValueError("spatial_onset.top_k must use one score_semantics")
    scores = [float(row["score"]) for row in top_k]
    if any(current > previous + _TOL for previous, current in zip(scores, scores[1:])):
        raise ValueError("spatial_onset.top_k scores must be non-increasing by rank")

    top_k_supporting: set[str] = set()
    for index, candidate in enumerate(top_k):
        candidate_type = str(candidate["candidate_type"])
        if candidate_type != expected_candidate_type:
            raise ValueError(
                f"spatial_onset.top_k[{index}] is finer/different than allowed_resolution"
            )
        _validate_spatial_id(
            candidate_type,
            str(candidate["candidate_id"]),
            ids,
            f"spatial_onset.top_k[{index}].candidate_id",
        )
        if candidate_type != "laterality":
            candidate_unit = per_unit_by_key.get(
                (candidate_type, str(candidate["candidate_id"]))
            )
            if candidate_unit is None or candidate_unit["interval"] is None:
                raise ValueError(
                    f"spatial_onset.top_k[{index}] requires an evaluable matching "
                    "per_unit_interval"
                )
        if candidate["score_semantics"] == "source_dev_calibrated_probability":
            score = float(candidate["score"])
            if score < -_TOL or score > 1.0 + _TOL:
                raise ValueError(
                    f"spatial_onset.top_k[{index}].score must be in [0,1] when calibrated"
                )
        candidate_evidence = set(
            str(item) for item in candidate["supporting_evidence_ids"]
        )
        _require_refs(
            candidate_evidence,
            evidence_ids,
            f"spatial_onset.top_k[{index}].supporting_evidence_ids",
        )
        if not candidate_evidence.issubset(supporting):
            raise ValueError(
                f"spatial_onset.top_k[{index}] evidence is not closed by global "
                "supporting_evidence_ids"
            )
        candidate_id = str(candidate["candidate_id"])
        for evidence_id in candidate_evidence:
            matching_support = [
                support
                for support in finding_by_id[evidence_id]["spatial_support"]
                if support["unit_type"] == candidate_type
                and support["id"] == candidate_id
            ]
            if not matching_support:
                raise ValueError(
                    f"spatial_onset.top_k[{index}] candidate is not spatially supported "
                    f"by Finding {evidence_id!r}"
                )
        spatial_anchor_evidence = [
            evidence_id
            for evidence_id in candidate_evidence
            if finding_by_id[evidence_id]["family"]
            in _SPATIAL_ONSET_ANCHOR_FAMILIES
            and any(
                support["unit_type"] == candidate_type
                and support["id"] == candidate_id
                for support in finding_by_id[evidence_id]["spatial_support"]
            )
        ]
        if not spatial_anchor_evidence:
            raise ValueError(
                f"spatial_onset.top_k[{index}] requires an explicit spatial-field "
                "or earliest-involvement anchor; non-spatial Findings cannot "
                "independently create a ranked SOZ candidate"
            )
        if candidate_type == "electrode":
            matching_electrode_support = [
                support
                for evidence_id in candidate_evidence
                for support in finding_by_id[evidence_id]["spatial_support"]
                if support["unit_type"] == "electrode"
                and support["id"] == candidate_id
                and support["mapping_status"] == "field_qualified"
            ]
            if not matching_electrode_support:
                raise ValueError(
                    f"spatial_onset.top_k[{index}] electrode requires explicit "
                    "field_qualified electrode support"
                )
            if (
                payload["montage"]["analysis_reference"] == "bipolar"
                and candidate_id not in ids["direct_electrodes"]
            ):
                perturbations = set(
                    str(item)
                    for item in payload["montage"].get(
                        "reference_perturbations_evaluated", []
                    )
                )
                if len(perturbations) < 2:
                    raise ValueError(
                        f"spatial_onset.top_k[{index}] bipolar-derived electrode "
                        "requires at least two reference perturbations"
                    )
                incident_leads = {
                    lead_id
                    for lead_id, endpoints in ids["lead_endpoints"].items()
                    if candidate_id in endpoints
                }
                supported_incident_leads = {
                    str(support["id"])
                    for evidence_id in candidate_evidence
                    for support in finding_by_id[evidence_id]["spatial_support"]
                    if support["unit_type"] == "lead"
                    and str(support["id"]) in incident_leads
                }
                if len(supported_incident_leads) < 2:
                    raise ValueError(
                        f"spatial_onset.top_k[{index}] bipolar-derived electrode "
                        "requires support from at least two incident leads"
                    )
        top_k_supporting.update(candidate_evidence)

    if top_k and top_k_supporting != supporting:
        missing = sorted(supporting.difference(top_k_supporting))
        extra = sorted(top_k_supporting.difference(supporting))
        raise ValueError(
            "spatial_onset.top_k supporting evidence must exactly close the global "
            f"supporting set; missing={missing}, extra={extra}"
        )
    if not supporting and localization_status != "nonlocalizable":
        raise ValueError(
            "only localization_status=nonlocalizable may have no supporting evidence"
        )

    phenotype = _dominant_score(spatial["phenotype_scores"])
    phenotype_scores = {
        str(row["name"]): float(row["score"])
        for row in spatial["phenotype_scores"]
    }
    if localization_status == "nonlocalizable" and (
        phenotype is None or phenotype[0] != "scalp_onset_nonlocalizable"
    ):
        raise ValueError(
            "localization_status=nonlocalizable requires dominant scalp_onset_nonlocalizable phenotype"
        )
    if (
        phenotype is not None
        and phenotype[0] == "scalp_onset_nonlocalizable"
        and phenotype[1] >= 0.5 - _TOL
        and localization_status != "nonlocalizable"
    ):
        raise ValueError(
            "dominant scalp_onset_nonlocalizable phenotype requires nonlocalizable status"
        )

    dominant_laterality = _dominant_score(spatial["laterality_scores"])
    dominant_region = _dominant_score(spatial["region_scores"])
    generalized_score = phenotype_scores.get("generalized_synchronous", 0.0)
    if generalized_score >= 0.5 - _TOL:
        if allowed_resolution not in {"none", "laterality"}:
            raise ValueError(
                "high generalized_synchronous phenotype cannot produce focal lead/electrode/region Top-1"
            )
        if top_k and str(top_k[0]["candidate_id"]) in {"left", "right"}:
            raise ValueError(
                "high generalized_synchronous phenotype cannot produce unilateral Top-1"
            )
        if (
            dominant_laterality is not None
            and dominant_laterality[0] in {"left", "right"}
            and dominant_laterality[1] >= 0.5 - _TOL
        ):
            raise ValueError(
                "high generalized_synchronous phenotype cannot have dominant unilateral laterality"
            )
        if not any(
            _is_explicit_bilateral_synchrony_finding(
                finding_by_id[evidence_id],
                waveforms=waveforms,
                ids=ids,
            )
            for evidence_id in supporting
        ):
            raise ValueError(
                "generalized_synchronous phenotype requires positive bilateral synchrony evidence"
            )

    if top_k:
        top_candidate_type = str(top_k[0]["candidate_type"])
        top_candidate_id = str(top_k[0]["candidate_id"])
        if (
            dominant_laterality is not None
            and dominant_laterality[1] >= 0.5 - _TOL
            and dominant_laterality[0] != "indeterminate"
            and dominant_laterality[0]
            not in _candidate_lateralities(
                top_candidate_type, top_candidate_id, ids
            )
        ):
            raise ValueError(
                "spatial_onset Top-1 conflicts with dominant laterality_scores"
            )
        if (
            dominant_region is not None
            and dominant_region[1] >= 0.5 - _TOL
            and top_candidate_type in {"lead", "electrode", "region"}
            and dominant_region[0]
            not in _candidate_regions(top_candidate_type, top_candidate_id, ids)
        ):
            raise ValueError(
                "spatial_onset Top-1 conflicts with dominant region_scores"
            )
    if (
        dominant_laterality is not None
        and dominant_region is not None
        and dominant_laterality[1] >= 0.5 - _TOL
        and dominant_region[1] >= 0.5 - _TOL
        and dominant_laterality[0] not in {"indeterminate"}
        and dominant_laterality[0]
        not in ids["region_to_lateralities"].get(dominant_region[0], set())
    ):
        raise ValueError(
            "spatial_onset dominant region_scores conflict with laterality_scores"
        )

    limitation_codes = [str(row["code"]) for row in payload["limitations"]]
    _unique(limitation_codes, "limitations.code")
    return payload


__all__ = [
    "EVENT_EEG_FINDINGS_SCHEMA_VERSION",
    "event_term_decision_source_binding_sha256",
    "validate_event_eeg_findings_payload",
]
