"""Fail-closed validation for the uniform EEG-only event Findings v2 overlay.

The repository's full ``event_eeg_findings_v2`` evidence graph remains the
authoritative low-level wire.  This module validates an additive clinical
profile in which every event category is represented by the same observation
shape.  A row is either a replayable numerical measurement, an explicitly
named research inference, or a typed ``not_evaluable`` result.

The validator intentionally owns no EEG feature extraction and no clinical
term classifier.  It prevents unavailable or non-EEG concepts from being
promoted merely because a JSON producer emitted a plausible-looking value.
"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from jsonschema import Draft202012Validator


EVENT_FINDINGS_EEG_ONLY_V2_SCHEMA_VERSION = (
    "clinical_eeg_event_findings_eeg_only_v2"
)
EVENT_FINDINGS_EEG_ONLY_V2_REGISTRY_ID = (
    "CLINICAL-EEG-EVENT-FINDINGS-EEG-ONLY-V2-REGISTRY"
)

_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA_PATH = (
    _ROOT / "schemas" / "clinical_eeg_event_findings_eeg_only_v2.schema.json"
)
_REGISTRY_PATH = (
    _ROOT
    / "configs"
    / "clinical_eeg_event_findings_eeg_only_v2_registry.json"
)
_TOL = 1e-8

STANDARD19 = (
    "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8",
    "T7", "C3", "CZ", "C4", "T8",
    "P7", "P3", "PZ", "P4", "P8", "O1", "O2",
)
COMMON17 = tuple(channel for channel in STANDARD19 if channel not in {"FZ", "PZ"})

REQUIRED_CATEGORIES = (
    "background_baseline",
    "waveform_morphology",
    "spectrum",
    "rhythm",
    "temporal_evolution",
    "spatial_onset_propagation",
    "amplitude",
    "synchrony_connectivity",
    "recovery",
    "artifact_quality",
    "reference_stability",
)


@lru_cache(maxsize=1)
def _schema_validator() -> Draft202012Validator:
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _schema_error_path(error: Any) -> str:
    parts = [str(item) for item in error.absolute_path]
    return ".".join(parts) if parts else "$"


def _reject_nonfinite(value: object, path: str = "$") -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
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
    rows = list(values)
    if len(rows) != len(set(rows)):
        raise ValueError(f"{context} contains duplicate identifiers")
    return set(rows)


def _interval(
    value: Sequence[object],
    context: str,
    *,
    bounds: tuple[float, float] | None = None,
) -> tuple[float, float]:
    start, stop = float(value[0]), float(value[1])
    if stop <= start + _TOL:
        raise ValueError(f"{context} must be a positive interval")
    if bounds is not None and (
        start < bounds[0] - _TOL or stop > bounds[1] + _TOL
    ):
        raise ValueError(f"{context} lies outside {bounds}")
    return start, stop


def _covers(carrier: tuple[float, float], target: tuple[float, float]) -> bool:
    return carrier[0] <= target[0] + _TOL and carrier[1] >= target[1] - _TOL


@lru_cache(maxsize=1)
def load_event_findings_eeg_only_v2_registry() -> dict[str, Any]:
    registry = json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
    if registry.get("schema_version") != (
        "clinical_eeg_event_findings_eeg_only_v2_registry"
    ):
        raise ValueError("EEG-only Findings v2 registry schema version drifted")
    if registry.get("registry_id") != EVENT_FINDINGS_EEG_ONLY_V2_REGISTRY_ID:
        raise ValueError("EEG-only Findings v2 registry ID drifted")
    if tuple(registry.get("required_categories", ())) != REQUIRED_CATEGORIES:
        raise ValueError("EEG-only Findings v2 category roster drifted")

    capabilities = registry.get("capabilities")
    if not isinstance(capabilities, list) or not capabilities:
        raise ValueError("EEG-only Findings v2 registry has no capabilities")
    metric_ids = _unique(
        (str(row.get("metric_id", "")) for row in capabilities),
        "capability metric roster",
    )
    if "" in metric_ids:
        raise ValueError("capability metric IDs must be non-empty")

    allowed_maturities = {
        "replayable_measurement",
        "research_proxy",
        "unavailable_unqualified",
        "forbidden_non_eeg",
    }
    allowed_levels = {
        "direct_measurement",
        "algorithmic_inference",
        "not_evaluable",
    }
    for row in capabilities:
        context = f"capability[{row['metric_id']}]"
        if row.get("category") not in REQUIRED_CATEGORIES:
            raise ValueError(f"{context} has an unknown category")
        maturity = row.get("maturity")
        if maturity not in allowed_maturities:
            raise ValueError(f"{context} has an unknown maturity")
        levels = row.get("allowed_evidence_levels")
        if not isinstance(levels, list) or any(level not in allowed_levels for level in levels):
            raise ValueError(f"{context} has invalid evidence levels")
        if maturity == "forbidden_non_eeg" and levels:
            raise ValueError(f"{context} must expose no evidence level")
        if maturity == "unavailable_unqualified" and levels != ["not_evaluable"]:
            raise ValueError(f"{context} may only be not_evaluable")
        units = row.get("allowed_units")
        if not isinstance(units, list):
            raise ValueError(f"{context} has invalid allowed units")
        paths = row.get("implementation_paths")
        if not isinstance(paths, list):
            raise ValueError(f"{context} has invalid implementation paths")
        if maturity in {"replayable_measurement", "research_proxy"} and not paths:
            raise ValueError(f"{context} lacks an implementation binding")
        for relative in paths:
            path = _ROOT / str(relative)
            if not path.is_file():
                raise ValueError(f"{context} implementation does not exist: {relative}")
    return registry


def _capability_index() -> dict[str, Mapping[str, Any]]:
    return {
        str(row["metric_id"]): row
        for row in load_event_findings_eeg_only_v2_registry()["capabilities"]
    }


def _validate_signal_contract(value: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    space = str(value["electrode_space"])
    expected = COMMON17 if space == "common17" else STANDARD19
    observed = tuple(str(item) for item in value["observed_electrodes"])
    if observed != expected:
        raise ValueError(
            f"{space} observed_electrodes must equal its canonical ordered roster"
        )
    missing = tuple(str(item) for item in value["missing_electrodes"])
    expected_missing = ("FZ", "PZ") if space == "common17" else ()
    if missing != expected_missing:
        raise ValueError(f"{space} missing_electrodes must equal {expected_missing}")
    if value["imputed_electrodes"]:
        raise ValueError("primary EEG-only Findings cannot use imputed electrodes")

    leads = value["derived_leads"]
    lead_ids = _unique((str(row["lead_id"]) for row in leads), "derived lead roster")
    observed_set = set(observed)
    for index, lead in enumerate(leads):
        anode = str(lead["anode"])
        cathode = str(lead["cathode"])
        if anode == cathode:
            raise ValueError(f"derived_leads[{index}] has identical endpoints")
        if anode not in observed_set or cathode not in observed_set:
            raise ValueError(f"derived_leads[{index}] uses an unobserved endpoint")

    sampling_rate = float(value["sampling_rate_hz"])
    highpass = value.get("highpass_hz")
    lowpass = value.get("lowpass_hz")
    if highpass is not None and float(highpass) < 0:
        raise ValueError("highpass_hz cannot be negative")
    if lowpass is not None:
        if float(lowpass) <= 0 or float(lowpass) >= sampling_rate / 2 + _TOL:
            raise ValueError("lowpass_hz must lie below Nyquist")
        if highpass is not None and float(highpass) >= float(lowpass):
            raise ValueError("highpass_hz must be below lowpass_hz")
    return observed_set, lead_ids


def _validate_measurement(value: Mapping[str, Any], context: str) -> None:
    value_type = str(value["value_type"])
    raw = value["value"]
    labels = list(value["dimension_labels"])
    unit = str(value["unit_id"])
    if value_type == "none":
        if raw is not None or labels or unit != "not_applicable":
            raise ValueError(f"{context} none measurement must be empty")
    elif value_type == "scalar":
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or labels:
            raise ValueError(f"{context} scalar measurement is malformed")
        if unit == "not_applicable":
            raise ValueError(f"{context} scalar requires a physical/engineering unit")
    elif value_type == "range":
        if not isinstance(raw, list) or len(raw) != 2 or labels:
            raise ValueError(f"{context} range measurement is malformed")
        if float(raw[1]) < float(raw[0]) - _TOL:
            raise ValueError(f"{context} range is decreasing")
        if unit == "not_applicable":
            raise ValueError(f"{context} range requires a unit")
    elif value_type == "vector":
        if not isinstance(raw, list) or len(raw) != len(labels) or not labels:
            raise ValueError(f"{context} vector labels must align with values")
        if unit == "not_applicable":
            raise ValueError(f"{context} vector requires a unit")
    elif value_type == "categorical":
        if not isinstance(raw, str) or labels or unit != "not_applicable":
            raise ValueError(f"{context} categorical measurement is malformed")


def _validate_confidence(value: Mapping[str, Any], context: str) -> None:
    semantics = str(value["semantics"])
    score = value["score"]
    receipt = value["calibration_receipt_id"]
    if semantics == "not_available":
        if score is not None or receipt is not None:
            raise ValueError(f"{context} unavailable confidence must be empty")
    elif semantics == "calibrated_probability":
        if score is None or receipt is None:
            raise ValueError(f"{context} calibrated probability requires a receipt")
    else:
        if score is None or receipt is not None:
            raise ValueError(
                f"{context} non-probability confidence requires a score and no calibrator"
            )


def _validate_evidence_level(
    row: Mapping[str, Any],
    capability: Mapping[str, Any],
    *,
    context: str,
) -> None:
    level = str(row["evidence_level"])
    assertion = str(row["assertion_status"])
    measurement = row["measurement"]
    quality = row["quality"]
    confidence = row["confidence"]
    source = row["source_binding"]
    evidence_interval = row["temporal_support"]["evidence_interval_seconds"]
    waveform_ids = row["waveform_evidence_ids"]
    spatial = row["spatial_support"]
    reasons = row["reason_codes"]

    if level not in capability["allowed_evidence_levels"]:
        raise ValueError(f"{context} evidence level is not registry-authorized")
    if source["maturity"] != capability["maturity"]:
        raise ValueError(f"{context} source maturity disagrees with registry")

    if level == "direct_measurement":
        if capability["maturity"] != "replayable_measurement":
            raise ValueError(f"{context} direct measurement lacks replayable maturity")
        if assertion != "observed":
            raise ValueError(f"{context} direct measurement must be observed")
        if evidence_interval is None or measurement["value_type"] == "none":
            raise ValueError(f"{context} direct measurement lacks measured support")
        if not waveform_ids or not source["source_evidence_ids"]:
            raise ValueError(f"{context} direct measurement lacks evidence binding")
        if quality["status"] not in {"passed", "limited"}:
            raise ValueError(f"{context} direct measurement failed its QC gate")
        if confidence["semantics"] not in {
            "measurement_repeatability",
            "not_available",
        }:
            raise ValueError(f"{context} direct measurement has inference confidence")
        if row["surface_policy"] != "numeric_measurement_only":
            raise ValueError(f"{context} direct measurement must remain numeric")
    elif level == "algorithmic_inference":
        if capability["maturity"] != "research_proxy":
            raise ValueError(f"{context} inference lacks research-proxy maturity")
        if assertion not in {
            "candidate_present",
            "candidate_absent_with_opportunity",
            "uncertain",
        }:
            raise ValueError(f"{context} inference has an invalid assertion")
        if evidence_interval is None or measurement["value_type"] == "none":
            raise ValueError(f"{context} inference lacks numerical support")
        if not waveform_ids or not source["source_evidence_ids"]:
            raise ValueError(f"{context} inference lacks evidence binding")
        if quality["status"] not in {"passed", "limited"}:
            raise ValueError(f"{context} inference failed its QC gate")
        if confidence["semantics"] not in {
            "uncalibrated_score",
            "calibrated_probability",
        }:
            raise ValueError(f"{context} inference requires explicit score semantics")
        if row["surface_policy"] != "research_candidate_only":
            raise ValueError(f"{context} inference must remain a research candidate")
        if (
            assertion == "candidate_absent_with_opportunity"
            and source["sensitivity_receipt_id"] is None
        ):
            raise ValueError(f"{context} negative inference lacks sensitivity receipt")
    else:
        if assertion != "not_evaluable":
            raise ValueError(f"{context} not_evaluable row has a non-null assertion")
        if evidence_interval is not None:
            raise ValueError(f"{context} not_evaluable row has an evidence interval")
        if measurement["value_type"] != "none" or measurement["value"] is not None:
            raise ValueError(f"{context} not_evaluable row has a value")
        if (
            measurement["baseline_relation"] != "not_applicable"
            or measurement["baseline_interval_seconds"] is not None
        ):
            raise ValueError(f"{context} not_evaluable row has baseline semantics")
        if waveform_ids or source["source_evidence_ids"]:
            raise ValueError(f"{context} not_evaluable row cites positive evidence")
        if quality["status"] not in {"failed", "not_evaluable"}:
            raise ValueError(f"{context} not_evaluable row has passing QC")
        if confidence["semantics"] != "not_available":
            raise ValueError(f"{context} not_evaluable row has confidence")
        if not reasons and not quality["reason_codes"]:
            raise ValueError(f"{context} not_evaluable row lacks a typed reason")
        if any(
            spatial[key]
            for key in ("electrodes", "derived_lead_ids", "regions")
        ):
            raise ValueError(f"{context} not_evaluable row has spatial evidence")
        if spatial["laterality"] != "not_applicable" or spatial["spatial_scope"] != "none":
            raise ValueError(f"{context} not_evaluable row has spatial semantics")
        if row["surface_policy"] not in {"technical_limitation_only", "withhold"}:
            raise ValueError(f"{context} not_evaluable row has an invalid surface policy")

    # Check the metric-specific unit after the row semantics.  In particular,
    # an unavailable capability carrying a scalar must fail as fabricated
    # evidence, rather than being reported only as a unit mismatch.
    if (
        level != "not_evaluable"
        and measurement["unit_id"] not in capability["allowed_units"]
    ):
        raise ValueError(f"{context} unit is not authorized for the metric")


def validate_event_findings_eeg_only_v2(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and return a defensive copy of one event profile."""

    result = deepcopy(dict(payload))
    _reject_nonfinite(result)

    # Preserve the more informative frozen-roster error even when removing a
    # category also violates the schema's defensive ``minItems`` constraint.
    # Malformed rows are left to JSON Schema instead of being dereferenced
    # here.
    observations = result.get("observations")
    if (
        isinstance(observations, list)
        and observations
        and all(isinstance(row, Mapping) and isinstance(row.get("category"), str) for row in observations)
    ):
        expected_categories = set(REQUIRED_CATEGORIES)
        observed_categories = {str(row["category"]) for row in observations}
        if observed_categories != expected_categories:
            missing = sorted(expected_categories - observed_categories)
            extra = sorted(observed_categories - expected_categories)
            raise ValueError(
                "observation category coverage mismatch: "
                f"missing={missing}, extra={extra}"
            )

    errors = sorted(
        _schema_validator().iter_errors(result),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        error = errors[0]
        raise ValueError(f"schema validation failed at {_schema_error_path(error)}: {error.message}")

    if result["schema_version"] != EVENT_FINDINGS_EEG_ONLY_V2_SCHEMA_VERSION:
        raise ValueError("event Findings EEG-only v2 schema version drifted")
    registry = load_event_findings_eeg_only_v2_registry()
    capabilities = _capability_index()
    if result["registry_id"] != registry["registry_id"]:
        raise ValueError("event Findings registry binding drifted")

    observed_electrodes, lead_ids = _validate_signal_contract(result["signal_contract"])
    recording_duration = float(result["analysis_window"]["recording_duration_seconds"])
    recording_bounds = (0.0, recording_duration)
    analysis_interval = _interval(
        result["analysis_window"]["query_interval_seconds"],
        "analysis_window.query_interval_seconds",
        bounds=recording_bounds,
    )
    anchor = float(result["analysis_window"]["navigation_anchor_seconds"])
    if anchor < -_TOL or anchor > recording_duration + _TOL:
        raise ValueError("navigation anchor lies outside the recording")

    producer_ids = _unique(
        (str(row["producer_id"]) for row in result["provenance"]["producers"]),
        "producer roster",
    )
    signal_sha = str(result["provenance"]["canonical_signal_sha256"])

    waveform_ids = _unique(
        (str(row["waveform_evidence_id"]) for row in result["waveform_evidence"]),
        "waveform evidence roster",
    )
    declared_references = set(result["signal_contract"]["analysis_references"])
    waveforms: dict[str, tuple[Mapping[str, Any], tuple[float, float]]] = {}
    for index, waveform in enumerate(result["waveform_evidence"]):
        context = f"waveform_evidence[{index}]"
        waveform_id = str(waveform["waveform_evidence_id"])
        interval = _interval(
            waveform["interval_seconds"],
            f"{context}.interval_seconds",
            bounds=recording_bounds,
        )
        if waveform["canonical_signal_sha256"] != signal_sha:
            raise ValueError(f"{context} canonical signal binding drifted")
        if abs(float(waveform["sampling_rate_hz"]) - float(result["signal_contract"]["sampling_rate_hz"])) > _TOL:
            raise ValueError(f"{context} sampling rate drifted")
        if not set(waveform["electrodes"]).issubset(observed_electrodes):
            raise ValueError(f"{context} uses unobserved electrodes")
        if not set(waveform["derived_lead_ids"]).issubset(lead_ids):
            raise ValueError(f"{context} uses unknown derived leads")
        if waveform["reference"] not in declared_references:
            raise ValueError(f"{context} uses an undeclared analysis reference")
        waveforms[waveform_id] = (waveform, interval)
    if set(waveforms) != waveform_ids:
        raise RuntimeError("waveform evidence index drifted")

    expected_categories = set(REQUIRED_CATEGORIES)
    if tuple(result["category_coverage"]) != REQUIRED_CATEGORIES:
        raise ValueError("category_coverage must use the frozen complete order")
    observation_ids = _unique(
        (str(row["observation_id"]) for row in result["observations"]),
        "observation roster",
    )
    if not observation_ids:
        raise ValueError("event profile has no observations")
    observed_categories = {str(row["category"]) for row in result["observations"]}
    if observed_categories != expected_categories:
        missing = sorted(expected_categories - observed_categories)
        extra = sorted(observed_categories - expected_categories)
        raise ValueError(f"observation category coverage mismatch: missing={missing}, extra={extra}")

    for index, row in enumerate(result["observations"]):
        context = f"observations[{index}]"
        metric_id = str(row["metric_id"])
        capability = capabilities.get(metric_id)
        if capability is None:
            raise ValueError(f"{context} metric is absent from the capability registry")
        if capability["maturity"] == "forbidden_non_eeg":
            raise ValueError(f"{context} metric is forbidden in EEG-only Findings")
        if row["category"] != capability["category"]:
            raise ValueError(f"{context} category disagrees with registry")
        if row["source_binding"]["producer_id"] not in producer_ids:
            raise ValueError(f"{context} references an undeclared producer")

        query_interval = _interval(
            row["temporal_support"]["query_interval_seconds"],
            f"{context}.temporal_support.query_interval_seconds",
            bounds=analysis_interval,
        )
        evidence_raw = row["temporal_support"]["evidence_interval_seconds"]
        evidence_interval = None
        if evidence_raw is not None:
            evidence_interval = _interval(
                evidence_raw,
                f"{context}.temporal_support.evidence_interval_seconds",
                bounds=query_interval,
            )
        baseline_raw = row["measurement"]["baseline_interval_seconds"]
        if baseline_raw is not None:
            _interval(
                baseline_raw,
                f"{context}.measurement.baseline_interval_seconds",
                bounds=analysis_interval,
            )

        spatial = row["spatial_support"]
        if not set(spatial["electrodes"]).issubset(observed_electrodes):
            raise ValueError(f"{context} uses unobserved electrodes")
        if not set(spatial["derived_lead_ids"]).issubset(lead_ids):
            raise ValueError(f"{context} uses unknown derived leads")

        _validate_measurement(row["measurement"], f"{context}.measurement")
        _validate_confidence(row["confidence"], f"{context}.confidence")
        _validate_evidence_level(row, capability, context=context)

        referenced_waveforms = set(str(item) for item in row["waveform_evidence_ids"])
        unknown_waveforms = referenced_waveforms - waveform_ids
        if unknown_waveforms:
            raise ValueError(f"{context} references unknown waveforms: {sorted(unknown_waveforms)}")
        if evidence_interval is not None:
            carriers = [waveforms[item] for item in referenced_waveforms]
            covering_carriers = [
                (waveform, interval)
                for waveform, interval in carriers
                if _covers(interval, evidence_interval)
            ]
            if not covering_carriers:
                raise ValueError(f"{context} waveform evidence does not cover its interval")
            spatial_electrodes = set(spatial["electrodes"])
            spatial_leads = set(spatial["derived_lead_ids"])
            if spatial_electrodes or spatial_leads:
                if not any(
                    spatial_electrodes.issubset(set(waveform["electrodes"]))
                    and spatial_leads.issubset(set(waveform["derived_lead_ids"]))
                    for waveform, _ in covering_carriers
                ):
                    raise ValueError(
                        f"{context} no single time-covering waveform evidence "
                        "contains its spatial support"
                    )

    evaluation = result["evaluation_binding"]
    if evaluation["reference_join_status"] == "not_joined":
        if (
            evaluation["join_stage"] != "none"
            or evaluation["reference_artifact_sha256"] is not None
            or evaluation["szcore_event_match_id"] is not None
        ):
            raise ValueError("unjoined evaluation binding contains reference data")
    else:
        if (
            evaluation["join_stage"] != "post_prediction_freeze"
            or evaluation["reference_artifact_sha256"] is None
            or evaluation["szcore_event_match_id"] is None
        ):
            raise ValueError("postfreeze evaluation binding is incomplete")

    return result


__all__ = [
    "COMMON17",
    "STANDARD19",
    "REQUIRED_CATEGORIES",
    "EVENT_FINDINGS_EEG_ONLY_V2_SCHEMA_VERSION",
    "EVENT_FINDINGS_EEG_ONLY_V2_REGISTRY_ID",
    "load_event_findings_eeg_only_v2_registry",
    "validate_event_findings_eeg_only_v2",
]
