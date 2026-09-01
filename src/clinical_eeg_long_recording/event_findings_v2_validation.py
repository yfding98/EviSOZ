"""Fail-closed validation for ``event_eeg_findings_v2``.

The v2 wire contract separates signal observability, evaluation opportunity,
atomic Finding status, clinical-term qualification, and target-relative
hypothesis relations.  JSON Schema validates the portable shape; this module
closes physical time, source-view, montage, receipt, and evidence relations.

``unknown`` is accepted only for an explicit lossy v1 migration.  It never
authorizes a report-eligible term, a qualified event, or a scalp-onset
hypothesis.  This is intentional: a compatibility adapter must not invent
causal filtering, imputation provenance, S0--S3 posteriors, or data-source
firewall facts that v1 did not record.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from jsonschema import Draft202012Validator

from .event_finding_term_registry import validate_event_finding_term_context


EVENT_EEG_FINDINGS_V2_SCHEMA_VERSION = "event_eeg_findings_v2"
_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA_PATH = _ROOT / "schemas" / "clinical_eeg_event_findings_v2.schema.json"
_TOL = 1e-6
_RAW_SAMPLE_DEPENDENCY_SCHEMA_VERSION = "clinical_eeg_raw_sample_dependency_v1"
_RAW_DEPENDENCY_ID_DOMAIN = "clinical-eeg-raw-sample-dependency-id-v1"
_RAW_DEPENDENCY_DIGEST_DOMAIN = "clinical-eeg-raw-sample-dependency-digest-v1"
_RAW_SUPPORT_COMPONENT_ROLE_ORDER = {
    "baseline_reference": 0,
    "reported_evidence_interval": 1,
    "sustained_confirmation": 2,
}

_FAMILIES = {
    "quality",
    "spectral",
    "rhythm",
    "morphology",
    "evolution",
    "spatial_field",
    "spatial_recruitment",
    "termination_recovery",
    "high_frequency",
}
_ONSET_FAMILIES = {
    "spectral",
    "rhythm",
    "morphology",
    "evolution",
    "spatial_field",
    "spatial_recruitment",
    "high_frequency",
}
_SPATIAL_ANCHOR_FAMILIES = {"spatial_field", "spatial_recruitment"}
_PHENOTYPES = {
    "focal",
    "focal_with_rapid_bilateralization",
    "bilateral_synchronous_or_rapid_bilateralization_ambiguous",
    "generalized_synchronous",
    "scalp_onset_nonlocalizable",
    "not_evaluable",
}
_LATERALITIES = {"left", "right", "bilateral", "midline", "indeterminate"}
_PROTECTED_TERMS = {
    "spike",
    "sharp_wave",
    "interictal_epileptiform_discharge",
    "definite_evolution",
    "electrographic_seizure",
}
_IFCN_CRITERIA = {
    "di_or_triphasic_sharp_or_spiky_morphology",
    "duration_differs_from_background",
    "waveform_asymmetry",
    "slow_after_wave",
    "surrounding_background_disruption",
    "physiologic_scalp_field",
}
_AXIS_TO_TYPE = {
    "phenotype": "phenotype",
    "laterality": "laterality",
    "region": "region",
    "lead": "lead",
    "electrode": "electrode",
}
_FINDING_TO_SOURCE_FAMILIES = {
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


@lru_cache(maxsize=1)
def _schema_validator() -> Draft202012Validator:
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


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
    if value is None or isinstance(value, (str, bool)):
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
    rows = list(values)
    if len(rows) != len(set(rows)):
        raise ValueError(f"{context} contains duplicate identifiers")
    return set(rows)


def _require_refs(values: Iterable[str], available: set[str], context: str) -> None:
    missing = set(values).difference(available)
    if missing:
        raise ValueError(f"{context} references unknown identifiers: {sorted(missing)}")


def _closed_interval(
    value: Sequence[object],
    context: str,
    *,
    bounds: tuple[float, float] | None = None,
    positive: bool = True,
) -> tuple[float, float]:
    lower, upper = float(value[0]), float(value[1])
    if lower > upper + _TOL or (positive and upper <= lower + _TOL):
        raise ValueError(f"{context} is not a valid positive interval")
    if bounds is not None and (
        lower < bounds[0] - _TOL or upper > bounds[1] + _TOL
    ):
        raise ValueError(f"{context} lies outside [{bounds[0]}, {bounds[1]}]")
    return lower, upper


def _span(
    value: Mapping[str, object],
    context: str,
    *,
    bounds: tuple[float, float] | None = None,
) -> tuple[float, float]:
    return _closed_interval(
        (value["start"], value["stop"]), context, bounds=bounds, positive=True
    )


def _time_interval(
    value: Mapping[str, object],
    context: str,
    *,
    bounds: tuple[float, float] | None = None,
) -> tuple[float, float]:
    lower, upper = float(value["lower"]), float(value["upper"])
    if lower > upper + _TOL:
        raise ValueError(f"{context}.lower exceeds .upper")
    if "median" in value:
        median = float(value["median"])
        if median < lower - _TOL or median > upper + _TOL:
            raise ValueError(f"{context}.median lies outside [lower, upper]")
    if bounds is not None and (
        lower < bounds[0] - _TOL or upper > bounds[1] + _TOL
    ):
        raise ValueError(f"{context} lies outside [{bounds[0]}, {bounds[1]}]")
    return lower, upper


def _overlap(left: tuple[float, float], right: tuple[float, float]) -> float:
    return max(0.0, min(left[1], right[1]) - max(left[0], right[0]))


def _covered_by(
    interval: tuple[float, float], carriers: Sequence[tuple[float, float]]
) -> bool:
    return any(
        interval[0] >= carrier[0] - _TOL
        and interval[1] <= carrier[1] + _TOL
        for carrier in carriers
    )


def _sorted_nonoverlapping(
    rows: Sequence[Sequence[object]],
    context: str,
    *,
    bounds: tuple[float, float],
) -> list[tuple[float, float]]:
    result = [
        _closed_interval(row, f"{context}[{index}]", bounds=bounds)
        for index, row in enumerate(rows)
    ]
    if result != sorted(result):
        raise ValueError(f"{context} must be sorted")
    for previous, current in zip(result, result[1:]):
        if current[0] < previous[1] - _TOL:
            raise ValueError(f"{context} must be non-overlapping")
    return result


def _validate_raw_sample_dependency(
    value: Mapping[str, object],
    *,
    context: str,
    canonical_signal_sha256: str,
    source_view_id: str,
    view_role: str,
    evidence_interval: tuple[float, float],
    view_tensor_interval: tuple[int, int] | None,
    view_receipt_id: str | None,
    view_receipt_sha256: str | None,
    transform_spec_sha256: str | None,
) -> str:
    """Validate one atom-local raw support sidecar fail closed."""

    dependency = dict(value)
    if dependency["schema_version"] != _RAW_SAMPLE_DEPENDENCY_SCHEMA_VERSION:
        raise ValueError(f"{context} has an unsupported schema version")
    identifier_source = deepcopy(dependency)
    identifier_source.pop("dependency_id", None)
    identifier_source.pop("dependency_sha256", None)
    expected_id = "RAWDEP-" + _sha256(
        {
            "domain": _RAW_DEPENDENCY_ID_DOMAIN,
            "dependency": identifier_source,
        }
    )[:24]
    if dependency["dependency_id"] != expected_id:
        raise ValueError(f"{context}.dependency_id does not bind its content")
    digest_source = deepcopy(dependency)
    digest_source["dependency_sha256"] = "CONTENT-ADDRESS-PENDING"
    expected_digest = _sha256(
        {
            "domain": _RAW_DEPENDENCY_DIGEST_DOMAIN,
            "dependency": digest_source,
        }
    )
    if dependency["dependency_sha256"] != expected_digest:
        raise ValueError(f"{context}.dependency_sha256 does not bind its content")

    if dependency["canonical_signal_sha256"] != canonical_signal_sha256:
        raise ValueError(f"{context} canonical signal mismatch")
    if dependency["source_view_id"] != source_view_id:
        raise ValueError(f"{context} source view mismatch")
    if dependency["view_role"] != view_role:
        raise ValueError(f"{context} view role mismatch")
    dependency_interval = _closed_interval(
        dependency["evidence_recording_interval"],
        f"{context}.evidence_recording_interval",
    )
    if any(
        abs(left - right) > _TOL
        for left, right in zip(dependency_interval, evidence_interval)
    ):
        raise ValueError(f"{context} evidence interval drifted")
    support_components = [
        (
            str(item["role"]),
            _closed_interval(
                item["recording_interval"],
                f"{context}.support_components[{index}].recording_interval",
            ),
        )
        for index, item in enumerate(dependency["support_components"])
    ]
    expected_component_order = sorted(
        support_components,
        key=lambda item: (
            _RAW_SUPPORT_COMPONENT_ROLE_ORDER[item[0]],
            item[1][0],
            item[1][1],
        ),
    )
    if support_components != expected_component_order or len(support_components) != len(
        set(support_components)
    ):
        raise ValueError(f"{context} support components must use frozen order and be unique")
    reported_components = [
        item[1]
        for item in support_components
        if item[0] == "reported_evidence_interval"
    ]
    if len(reported_components) != 1 or any(
        abs(left - right) > _TOL
        for left, right in zip(reported_components[0], evidence_interval)
    ):
        raise ValueError(
            f"{context} requires exactly one reported-evidence support component"
        )
    decision_available = float(dependency["decision_available_recording_seconds"])
    if decision_available < evidence_interval[1] - _TOL:
        raise ValueError(f"{context} decision availability precedes reported evidence")
    dependency_tensor_interval = tuple(
        int(item) for item in dependency["view_tensor_sample_interval"]
    )
    if dependency_tensor_interval[1] <= dependency_tensor_interval[0]:
        raise ValueError(f"{context} view tensor interval is empty")
    if (
        view_tensor_interval is not None
        and dependency_tensor_interval != view_tensor_interval
    ):
        raise ValueError(f"{context} view tensor interval drifted")

    numerator = int(dependency["view_sampling_rate_numerator"])
    denominator = int(dependency["view_sampling_rate_denominator"])
    latency_samples = float(
        dependency["processing_latency_samples_on_view_clock"]
    )
    latency_seconds = float(dependency["processing_latency_seconds"])
    if abs(latency_seconds - latency_samples * denominator / numerator) > _TOL:
        raise ValueError(f"{context} processing latency disagrees with view clock")
    latency_policy = str(dependency["processing_latency_policy"])
    if latency_seconds > _TOL:
        if latency_policy != (
            "report_constant_processing_latency_no_timestamp_advance_v1"
        ):
            raise ValueError(f"{context} positive latency must not advance timestamps")
    elif latency_policy != "none":
        raise ValueError(f"{context} zero processing latency requires policy=none")
    confirmation_latency = float(dependency["confirmation_latency_seconds"])
    confirmation_samples = float(
        dependency["confirmation_latency_samples_on_view_clock"]
    )
    if abs(confirmation_latency - (decision_available - evidence_interval[1])) > _TOL:
        raise ValueError(f"{context} confirmation latency disagrees with availability")
    if abs(confirmation_latency - confirmation_samples * denominator / numerator) > _TOL:
        raise ValueError(f"{context} confirmation latency disagrees with view clock")
    confirmation_policy = str(dependency["confirmation_policy"])
    if confirmation_latency > _TOL:
        if confirmation_policy != "sustained_observation_no_timestamp_advance_v1":
            raise ValueError(f"{context} sustained confirmation cannot advance onset time")
        if not any(role == "sustained_confirmation" for role, _ in support_components):
            raise ValueError(f"{context} confirmation latency lacks a support component")
    elif confirmation_policy != "none":
        raise ValueError(f"{context} zero confirmation latency requires policy=none")

    lineage = dependency["receipt_lineage"]
    if lineage["source_view_id"] != source_view_id:
        raise ValueError(f"{context} receipt lineage source view mismatch")
    if view_receipt_id is not None and lineage["source_view_receipt_id"] != view_receipt_id:
        raise ValueError(f"{context} source view receipt ID drifted")
    if (
        view_receipt_sha256 is not None
        and lineage["source_view_receipt_sha256"] != view_receipt_sha256
    ):
        raise ValueError(f"{context} source view receipt hash drifted")
    if (
        transform_spec_sha256 is not None
        and lineage["source_transform_spec_sha256"] != transform_spec_sha256
    ):
        raise ValueError(f"{context} transform receipt hash drifted")
    parent_ids = [
        str(row["view_id"]) for row in lineage["parent_view_bindings"]
    ]
    if len(parent_ids) != len(set(parent_ids)):
        raise ValueError(f"{context} receipt lineage repeats a parent view")

    raw_rows = dependency["raw_sample_intervals"]
    channel_ids = [str(row["channel_id"]) for row in raw_rows]
    if len(channel_ids) != len(set(channel_ids)):
        raise ValueError(f"{context} repeats a canonical raw channel")
    for index, row in enumerate(raw_rows):
        raw_start = int(row["raw_start_sample"])
        raw_stop = int(row["raw_stop_sample_exclusive"])
        evidence_start = int(row["reported_evidence_start_sample"])
        evidence_stop = int(row["reported_evidence_stop_sample_exclusive"])
        decision_stop = int(
            row["unshifted_decision_available_stop_sample_exclusive"]
        )
        sample_count = int(row["channel_sample_count"])
        if not (0 <= raw_start < raw_stop <= sample_count):
            raise ValueError(f"{context}.raw_sample_intervals[{index}] raw bounds are invalid")
        if not (0 <= evidence_start < evidence_stop <= sample_count):
            raise ValueError(
                f"{context}.raw_sample_intervals[{index}] evidence bounds are invalid"
            )
        if not (evidence_stop <= decision_stop <= sample_count):
            raise ValueError(
                f"{context}.raw_sample_intervals[{index}] decision-availability bound is invalid"
            )
        raw_rate = int(row["sample_rate_numerator"]) / int(
            row["sample_rate_denominator"]
        )
        sample_period = 1.0 / raw_rate
        raw_start_seconds = raw_start / raw_rate
        raw_stop_seconds = raw_stop / raw_rate
        evidence_start_seconds = evidence_start / raw_rate
        evidence_stop_seconds = evidence_stop / raw_rate
        decision_stop_seconds = decision_stop / raw_rate
        if raw_start_seconds > min(item[1][0] for item in support_components) + _TOL:
            raise ValueError(f"{context} raw start omits a support component")
        if raw_stop_seconds < max(item[1][1] for item in support_components) - _TOL:
            raise ValueError(f"{context} raw stop omits a support component")
        if not (
            evidence_start_seconds <= evidence_interval[0] + _TOL
            and evidence_start_seconds
            > evidence_interval[0] - sample_period - _TOL
            and evidence_stop_seconds >= evidence_interval[1] - _TOL
            and evidence_stop_seconds
            < evidence_interval[1] + sample_period + _TOL
        ):
            raise ValueError(f"{context} reported evidence sample mapping drifted")
        if not (
            decision_stop_seconds >= decision_available - _TOL
            and decision_stop_seconds < decision_available + sample_period + _TOL
        ):
            raise ValueError(f"{context} decision-available sample mapping drifted")

    future = bool(dependency["future_sample_access"])
    onset_authorized = bool(dependency["onset_evidence_authorized"])
    onset_support = bool(dependency["onset_support_eligible"])
    policy = str(dependency["dependency_policy"])
    raw_end_policy = str(dependency["raw_support_end_policy"])
    status = str(dependency["dependency_status"])
    causal_contract = bool(
        view_role == "onset_causal"
        and not future
        and onset_authorized
        and policy == "past_and_present_only"
        and raw_end_policy == "at_or_before_unshifted_evidence_sample_v1"
    )
    if onset_support != causal_contract:
        raise ValueError(f"{context} onset-support eligibility is inconsistent")
    if future:
        if (
            onset_authorized
            or onset_support
            or policy != "bidirectional_or_unknown"
            or raw_end_policy
            != "future_dependent_context_not_onset_eligible_v1"
            or status != "conservative_future_dependent_recording_bound"
        ):
            raise ValueError(f"{context} future-dependent evidence can only be context")
    elif view_role == "onset_causal":
        if not causal_contract or status != "bounded_past_and_present":
            raise ValueError(f"{context} causal onset raw dependency is unsafe")
        if any(item[1][1] > decision_available + _TOL for item in support_components):
            raise ValueError(
                f"{context} causal support component extends after unshifted decision availability"
            )
        for index, row in enumerate(raw_rows):
            if int(row["raw_stop_sample_exclusive"]) > int(
                row["unshifted_decision_available_stop_sample_exclusive"]
            ):
                raise ValueError(
                    f"{context}.raw_sample_intervals[{index}] raw stop is later than unshifted decision-available time"
                )
    elif view_role == "canonical_physical_evidence":
        if (
            future
            or onset_authorized
            or onset_support
            or policy != "instantaneous"
            or status != "exact_instantaneous"
            or raw_end_policy != "at_or_before_unshifted_evidence_sample_v1"
        ):
            raise ValueError(f"{context} instantaneous physical dependency is inconsistent")
    elif view_role == "context_offline":
        if not future:
            raise ValueError(f"{context} offline context must remain future-dependent")
    else:
        raise ValueError(f"{context} unsupported view role for raw dependency")
    return str(dependency["dependency_id"])


def _trusted_registry(
    value: Mapping[str, Mapping[str, object]] | None,
    *,
    name: str,
) -> dict[str, Mapping[str, object]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a host-supplied mapping")
    result: dict[str, Mapping[str, object]] = {}
    for key, row in value.items():
        if not isinstance(key, str) or not isinstance(row, Mapping):
            raise TypeError(f"{name} entries are invalid")
        if row.get("receipt_id") != key:
            raise ValueError(f"{name} key/receipt_id mismatch")
        result[key] = deepcopy(dict(row))
    return result


def _validate_embedded_receipts(
    rows: Sequence[Mapping[str, object]],
    trusted: Mapping[str, Mapping[str, object]],
    *,
    context: str,
) -> dict[str, Mapping[str, object]]:
    identifiers = _unique(
        (str(row["receipt_id"]) for row in rows), f"{context}.receipt_id"
    )
    result = {str(row["receipt_id"]): row for row in rows}
    for receipt_id in identifiers:
        host = trusted.get(receipt_id)
        if host is None:
            raise ValueError(
                f"{context} receipt {receipt_id!r} is absent from the host trusted registry"
            )
        if _canonical_json(result[receipt_id]) != _canonical_json(host):
            raise ValueError(
                f"{context} receipt {receipt_id!r} differs from the host trusted registry"
            )
    return result


def pattern_term_registry_sha256_v1(value: Mapping[str, object]) -> str:
    """Content-bind one explicit composite-pattern vocabulary."""

    terms = value.get("terms")
    if not isinstance(terms, list):
        raise TypeError("pattern term registry terms must be an array")
    return _sha256(
        {
            "binding_domain": "clinical-eeg-composite-pattern-term-registry-v1",
            "registry_id": value.get("registry_id"),
            "version": value.get("version"),
            "terms": deepcopy(terms),
        }
    )


def event_term_decision_source_binding_sha256_v2(value: object) -> str:
    """Bind a term decision to the complete v2 event ledger without cycles."""

    if type(value) is not dict:
        raise TypeError("event term-decision source must be an event object")
    source = deepcopy(value)
    source["term_decision_receipts"] = []
    for finding in source.get("findings", []):
        if not isinstance(finding, dict):
            raise TypeError("event term-decision source contains a non-object Finding")
        finding["term_decision_receipt_id"] = None
    pattern_candidates = source.get("pattern_candidates")
    if pattern_candidates == []:
        # Preserve the pre-extension digest for old v2 ledgers whose only
        # upgrade is the fail-closed empty/null pattern default.
        source.pop("pattern_candidates", None)
        for finding in source.get("findings", []):
            finding.pop("pattern_instance_id", None)
        registry_bindings = source.get("registry_bindings")
        if (
            isinstance(registry_bindings, dict)
            and registry_bindings.get("pattern_term_registry") is None
        ):
            registry_bindings.pop("pattern_term_registry", None)
    elif isinstance(pattern_candidates, list):
        for pattern in pattern_candidates:
            if not isinstance(pattern, dict):
                raise TypeError(
                    "event term-decision source contains a non-object pattern candidate"
                )
            pattern["qualification_rule_receipt_id"] = None
    event_qualification = source.get("event_qualification")
    if isinstance(event_qualification, dict):
        receipt_id = event_qualification.get("qualification_receipt_id")
        if receipt_id is not None:
            event_qualification["qualification_receipt_id"] = None
    return _sha256(
        {
            "binding_domain": "clinical_eeg_event_term_decision_source_v2",
            "event": source,
        }
    )


def _validate_boundary(
    row: Mapping[str, object],
    context: str,
    *,
    bounds: tuple[float, float],
) -> tuple[float, float] | None:
    status = str(row["status"])
    interval = row["interval"]
    reasons = row["censoring_reason_codes"]
    if status == "observed":
        if interval is None:
            raise ValueError(f"{context}.interval is required when status=observed")
        if reasons:
            raise ValueError(f"{context} observed status cannot carry censoring reasons")
        return _time_interval(interval, f"{context}.interval", bounds=bounds)  # type: ignore[arg-type]
    if status == "censored":
        if not reasons:
            raise ValueError(f"{context} censored status requires reason codes")
        if interval is None:
            return None
        return _time_interval(interval, f"{context}.interval", bounds=bounds)  # type: ignore[arg-type]
    if interval is not None:
        raise ValueError(f"{context}.interval must be null when status={status}")
    if status == "indeterminate" and not reasons:
        raise ValueError(f"{context} indeterminate status requires reason codes")
    return None


def _validate_montage(payload: Mapping[str, Any]) -> dict[str, Any]:
    montage = payload["montage"]
    input_units = montage["input_units"]
    unit_ids = _unique(
        (str(row["unit_id"]) for row in input_units),
        "montage.input_units.unit_id",
    )
    electrodes = set(str(item) for item in montage["electrode_ids"])
    definitions = montage["lead_definitions"]
    leads = _unique(
        (str(row["lead_id"]) for row in definitions),
        "montage.lead_definitions.lead_id",
    )
    if electrodes.intersection(leads):
        raise ValueError("montage lead and electrode namespaces must be disjoint")
    lead_endpoints: dict[str, set[str]] = {}
    for index, definition in enumerate(definitions):
        anode = str(definition["anode"])
        cathode = str(definition["cathode"])
        if anode == cathode:
            raise ValueError(f"montage.lead_definitions[{index}] has identical endpoints")
        _require_refs((anode, cathode), electrodes, f"montage.lead_definitions[{index}]")
        lead_endpoints[str(definition["lead_id"])] = {anode, cathode}

    regions: set[str] = set()
    observed_units: set[str] = set()
    eligible_units: set[str] = set()
    canonical_to_units: dict[tuple[str, str], set[str]] = defaultdict(set)
    region_to_units: dict[str, set[str]] = defaultdict(set)
    laterality_to_units: dict[str, set[str]] = defaultdict(set)
    for index, row in enumerate(input_units):
        unit_id = str(row["unit_id"])
        unit_type = str(row["unit_type"])
        canonical = str(row["canonical_name"])
        observation = str(row["observation_status"])
        eligible = bool(row["evidence_eligible"])
        reasons = row["missing_reason_codes"]
        imputation = row["imputation_receipt_id"]
        if unit_type == "electrode" and canonical not in electrodes:
            raise ValueError(f"montage.input_units[{index}] has unknown electrode")
        if unit_type == "bipolar_lead" and canonical not in leads:
            raise ValueError(f"montage.input_units[{index}] has unknown bipolar lead")
        if observation == "observed":
            if reasons or imputation is not None:
                raise ValueError(
                    f"montage.input_units[{index}] observed input cannot carry missing/imputation metadata"
                )
            observed_units.add(unit_id)
        elif observation == "imputed":
            if not reasons or imputation is None:
                raise ValueError(
                    f"montage.input_units[{index}] imputed input requires missing reasons and receipt"
                )
        else:
            if not reasons:
                raise ValueError(
                    f"montage.input_units[{index}] {observation} input requires reason codes"
                )
            if imputation is not None:
                raise ValueError(
                    f"montage.input_units[{index}] non-imputed input cannot carry an imputation receipt"
                )
        if eligible and observation != "observed":
            raise ValueError(
                f"montage.input_units[{index}] only directly observed inputs can be evidence eligible"
            )
        if eligible:
            eligible_units.add(unit_id)
        canonical_to_units[(unit_type, canonical)].add(unit_id)
        region = row.get("region")
        if region is not None:
            region_id = str(region)
            regions.add(region_id)
            region_to_units[region_id].add(unit_id)
        laterality_to_units[str(row["laterality"])].add(unit_id)

    bipolar_names = {
        canonical
        for (unit_type, canonical), units in canonical_to_units.items()
        if unit_type == "bipolar_lead" and units
    }
    if bipolar_names != leads:
        raise ValueError(
            "montage bipolar input units and lead_definitions must have identical canonical IDs"
        )
    if montage["analysis_reference"] == "bipolar" and not leads:
        raise ValueError("bipolar analysis_reference requires bipolar input units")

    lead_to_units = {
        lead: set(canonical_to_units.get(("bipolar_lead", lead), set()))
        for lead in leads
    }
    electrode_to_units = {
        electrode: set(canonical_to_units.get(("electrode", electrode), set()))
        for electrode in electrodes
    }
    for lead, endpoints in lead_endpoints.items():
        for electrode in endpoints:
            electrode_to_units[electrode].update(lead_to_units[lead])
    return {
        "input_units": unit_ids,
        "observed_units": observed_units,
        "eligible_units": eligible_units,
        "electrodes": electrodes,
        "leads": leads,
        "regions": regions,
        "lateralities": set(_LATERALITIES),
        "lead_to_units": lead_to_units,
        "electrode_to_units": electrode_to_units,
        "region_to_units": dict(region_to_units),
        "laterality_to_units": dict(laterality_to_units),
    }


def _spatial_units(
    unit_type: str, identifier: str, ids: Mapping[str, Any]
) -> set[str]:
    if unit_type == "lead":
        return set(ids["lead_to_units"].get(identifier, set()))
    if unit_type == "electrode":
        return set(ids["electrode_to_units"].get(identifier, set()))
    if unit_type == "region":
        return set(ids["region_to_units"].get(identifier, set()))
    if unit_type == "laterality":
        if identifier == "bilateral":
            return set().union(
                ids["laterality_to_units"].get("left", set()),
                ids["laterality_to_units"].get("right", set()),
                ids["laterality_to_units"].get("bilateral", set()),
            )
        return set(ids["laterality_to_units"].get(identifier, set()))
    return set()


def _validate_spatial_identifier(
    unit_type: str, identifier: str, ids: Mapping[str, Any], context: str
) -> None:
    allowed = {
        "lead": ids["leads"],
        "electrode": ids["electrodes"],
        "region": ids["regions"],
        "laterality": ids["lateralities"],
    }[unit_type]
    _require_refs((identifier,), allowed, context)


def _assert_acyclic(edges: Sequence[tuple[str, str]], context: str) -> None:
    adjacency: dict[str, set[str]] = defaultdict(set)
    indegree: dict[str, int] = defaultdict(int)
    nodes: set[str] = set()
    for source, target in edges:
        nodes.update((source, target))
        if target not in adjacency[source]:
            adjacency[source].add(target)
            indegree[target] += 1
        indegree.setdefault(source, 0)
    ready = [node for node in nodes if indegree[node] == 0]
    visited = 0
    while ready:
        node = ready.pop()
        visited += 1
        for target in adjacency[node]:
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    if visited != len(nodes):
        raise ValueError(f"{context} contains a directed cycle")


def validate_event_eeg_findings_v2_payload(
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
    """Validate and defensively copy one v2 EEG-only event evidence graph."""

    if type(value) is not dict:
        raise TypeError("event_eeg_findings_v2 payload must be an object")
    candidate_value: dict[str, Any] = deepcopy(value)
    if "pattern_candidates" not in candidate_value:
        # Additive compatibility for ledgers produced before the composite
        # pattern contract existed.  A mixed payload that already uses a
        # pattern_instance_id is not legacy and must fail rather than being
        # silently rewritten.
        candidate_findings = candidate_value.get("findings")
        if isinstance(candidate_findings, list) and any(
            isinstance(row, Mapping) and "pattern_instance_id" in row
            for row in candidate_findings
        ):
            raise ValueError(
                "pattern_candidates is required when Findings declare pattern_instance_id"
            )
        candidate_value["pattern_candidates"] = []
        if isinstance(candidate_findings, list):
            for row in candidate_findings:
                if isinstance(row, dict):
                    row["pattern_instance_id"] = None
    _reject_nonfinite(candidate_value)
    errors = sorted(
        _schema_validator().iter_errors(candidate_value),
        key=lambda item: list(item.path),
    )
    if errors:
        rendered = "; ".join(
            f"{_path(error)}: {error.message}" for error in errors[:8]
        )
        if len(errors) > 8:
            rendered += f"; ... {len(errors) - 8} more error(s)"
        raise ValueError(f"event_eeg_findings_v2 schema validation failed: {rendered}")
    payload: dict[str, Any] = candidate_value
    migrated = payload["migration"] is not None

    # Registry trust is explicit.  A legacy-unverified binding is accepted
    # only inside the lossy migration envelope and later forces all clinical
    # and hypothesis outputs closed.
    registry_host = trusted_registry_bindings or {}
    if not isinstance(registry_host, Mapping):
        raise TypeError("trusted_registry_bindings must be a host-supplied mapping")
    for name, binding in payload["registry_bindings"].items():
        if binding is None:
            continue
        if binding["trust_status"] == "host_trusted":
            trusted = registry_host.get(name)
            if trusted is None or _canonical_json(binding) != _canonical_json(trusted):
                raise ValueError(f"registry binding {name!r} is not host trusted")
        elif not migrated:
            raise ValueError("legacy-unverified registries require an explicit migration receipt")

    exclusions = payload["provenance"]["inference_exclusions"]
    has_unknown_scope = any(value == "unknown" for value in exclusions.values())
    if has_unknown_scope and not migrated:
        raise ValueError("native v2 inference exclusions must all be explicitly false")

    duration = float(payload["coordinates"]["recording_duration_seconds"])
    recording_bounds = (0.0, duration)
    window = payload["window"]
    search_bounds = _closed_interval(
        window["search_interval"], "window.search_interval", bounds=recording_bounds
    )
    final_bounds = _closed_interval(
        window["final_interval"], "window.final_interval", bounds=search_bounds
    )
    protection = window["protection_zone"]
    protection_bounds = _closed_interval(
        protection["interval"], "window.protection_zone.interval", bounds=search_bounds
    )
    onset_bounds = _validate_boundary(
        window["onset_boundary"], "window.onset_boundary", bounds=final_bounds
    )
    offset_bounds = _validate_boundary(
        window["offset_boundary"], "window.offset_boundary", bounds=final_bounds
    )
    if onset_bounds is not None and offset_bounds is not None:
        if offset_bounds[1] < onset_bounds[0] - _TOL:
            raise ValueError("window.offset_boundary cannot precede onset_boundary")
    if window["onset_boundary"]["status"] == "censored" and not window["left_censored"]:
        raise ValueError("censored onset requires left_censored=true")
    if window["offset_boundary"]["status"] == "censored" and not window["right_censored"]:
        raise ValueError("censored offset requires right_censored=true")
    if window["search_cap_censored"] and not (
        window["left_censored"] or window["right_censored"]
    ):
        raise ValueError("search_cap_censored requires a censored side")

    segment_ids = _unique(
        (str(row["segment_id"]) for row in window["state_segments"]),
        "window.state_segments.segment_id",
    )
    del segment_ids
    state_spans: list[tuple[float, float]] = []
    for index, segment in enumerate(window["state_segments"]):
        span = _span(
            segment["interval"],
            f"window.state_segments[{index}].interval",
            bounds=final_bounds,
        )
        posterior_total = sum(float(item) for item in segment["posterior"].values())
        if abs(posterior_total - 1.0) > _TOL:
            raise ValueError(
                f"window.state_segments[{index}].posterior must sum to 1"
            )
        state_spans.append(span)
    if state_spans != sorted(state_spans):
        raise ValueError("window.state_segments must be sorted")
    for previous, current in zip(state_spans, state_spans[1:]):
        if abs(previous[1] - current[0]) > _TOL:
            raise ValueError("window.state_segments must tile without gaps or overlap")
    if window["state_posterior_status"] == "not_evaluable":
        if state_spans or window["state_path_receipt_id"] is not None:
            raise ValueError(
                "not-evaluable state posterior requires no segments or path receipt"
            )
    else:
        if not state_spans or window["state_path_receipt_id"] is None:
            raise ValueError(
                "available/limited state posterior requires segments and a path receipt"
            )
        if (
            abs(state_spans[0][0] - final_bounds[0]) > _TOL
            or abs(state_spans[-1][1] - final_bounds[1]) > _TOL
        ):
            raise ValueError("window.state_segments must exactly tile final_interval")

    context = payload["context"]
    queried = _sorted_nonoverlapping(
        context["queried_intervals"],
        "context.queried_intervals",
        bounds=recording_bounds,
    )
    local_background = _sorted_nonoverlapping(
        context["local_background_intervals"],
        "context.local_background_intervals",
        bounds=recording_bounds,
    )
    distant_background = _sorted_nonoverlapping(
        context["distant_background_intervals"],
        "context.distant_background_intervals",
        bounds=recording_bounds,
    )
    if not _covered_by(final_bounds, queried):
        raise ValueError("window.final_interval must be covered by queried_intervals")
    for name, rows in (
        ("local_background_intervals", local_background),
        ("distant_background_intervals", distant_background),
    ):
        for index, interval in enumerate(rows):
            if not _covered_by(interval, queried):
                raise ValueError(f"context.{name}[{index}] is not queried")
            if _overlap(interval, protection_bounds) > _TOL:
                raise ValueError(
                    f"context.{name}[{index}] overlaps the event protection zone"
                )
    if context["background_status"] == "unavailable":
        if local_background or distant_background:
            raise ValueError("unavailable background requires empty intervals")
        if context["background_bank_id"] is not None or context["selection_receipt_id"] is not None:
            raise ValueError("unavailable background requires null background receipts")
    elif context["background_status"] != "unknown":
        if not local_background and not distant_background:
            raise ValueError("available/limited background requires EEG intervals")
        if context["background_bank_id"] is None or context["selection_receipt_id"] is None:
            raise ValueError("available/limited background requires EEG-only receipts")
    elif not migrated:
        raise ValueError("background_status=unknown is migration-only")
    background_ids = {
        str(item)
        for item in (context["background_bank_id"], context["selection_receipt_id"])
        if item is not None
    }

    ids = _validate_montage(payload)
    canonical_signal = str(payload["provenance"]["canonical_signal_sha256"])

    producer_receipts = _validate_embedded_receipts(
        payload["producer_receipts"],
        _trusted_registry(trusted_producer_receipts, name="trusted_producer_receipts"),
        context="producer_receipts",
    )
    calibration_receipts = _validate_embedded_receipts(
        payload["calibration_receipts"],
        _trusted_registry(
            trusted_calibration_receipts, name="trusted_calibration_receipts"
        ),
        context="calibration_receipts",
    )
    capability_receipts = _validate_embedded_receipts(
        payload["capability_qualification_receipts"],
        _trusted_registry(
            trusted_capability_qualification_receipts,
            name="trusted_capability_qualification_receipts",
        ),
        context="capability_qualification_receipts",
    )
    sensitivity_receipts_map = _validate_embedded_receipts(
        payload["sensitivity_receipts"],
        _trusted_registry(
            trusted_sensitivity_receipts, name="trusted_sensitivity_receipts"
        ),
        context="sensitivity_receipts",
    )
    term_receipts = _validate_embedded_receipts(
        payload["term_decision_receipts"],
        _trusted_registry(
            trusted_term_decision_receipts,
            name="trusted_term_decision_receipts",
        ),
        context="term_decision_receipts",
    )
    for context_name, receipt_map, metric, lower_metric in (
        ("capability", capability_receipts, "precision", "precision_lower_bound"),
        ("sensitivity", sensitivity_receipts_map, "sensitivity", "sensitivity_lower_bound"),
    ):
        for receipt_id, receipt in receipt_map.items():
            if float(receipt[lower_metric]) > float(receipt[metric]) + _TOL:
                raise ValueError(
                    f"{context_name} receipt {receipt_id!r} lower bound exceeds point estimate"
                )
            cohort = receipt["qualification_cohort"]
            if int(cohort["event_count"]) < int(cohort["patient_count"]):
                raise ValueError(
                    f"{context_name} receipt {receipt_id!r} event_count is below patient_count"
                )
            if int(cohort["positive_reference_count"]) > int(cohort["event_count"]):
                raise ValueError(
                    f"{context_name} receipt {receipt_id!r} positive count exceeds events"
                )

    opportunities = payload["evaluation_opportunities"]
    opportunity_ids = _unique(
        (str(row["evaluation_opportunity_id"]) for row in opportunities),
        "evaluation_opportunities.evaluation_opportunity_id",
    )
    opportunity_map = {
        str(row["evaluation_opportunity_id"]): row for row in opportunities
    }
    for index, opportunity in enumerate(opportunities):
        interval = opportunity["interval"]
        if interval is not None:
            _span(
                interval,
                f"evaluation_opportunities[{index}].interval",
                bounds=recording_bounds,
            )
        if opportunity["status"] == "sufficient":
            if opportunity["reason_codes"]:
                raise ValueError(
                    f"evaluation_opportunities[{index}] sufficient status cannot carry reasons"
                )
            if (
                interval is None
                or not opportunity["source_view_ids"]
                or opportunity["effective_bandwidth_hz"] is None
                or opportunity["quality_mask_sha256"] is None
            ):
                raise ValueError(
                    f"evaluation_opportunities[{index}] sufficient status lacks physical opportunity"
                )
        else:
            if not opportunity["reason_codes"]:
                raise ValueError(
                    f"evaluation_opportunities[{index}] limited/not_evaluable requires reasons"
                )
        if opportunity["effective_bandwidth_hz"] is not None:
            bandwidth = _closed_interval(
                opportunity["effective_bandwidth_hz"],
                f"evaluation_opportunities[{index}].effective_bandwidth_hz",
            )
            if bandwidth[0] < -_TOL:
                raise ValueError("effective bandwidth must be nonnegative")

    quality = payload["quality"]
    quality_unit_ids = _unique(
        (str(row["unit_id"]) for row in quality["per_unit"]),
        "quality.per_unit.unit_id",
    )
    if quality_unit_ids != ids["input_units"]:
        raise ValueError("quality.per_unit must cover every input unit exactly once")
    input_by_id = {
        str(row["unit_id"]): row for row in payload["montage"]["input_units"]
    }
    for index, row in enumerate(quality["per_unit"]):
        unit_id = str(row["unit_id"])
        source = input_by_id[unit_id]
        if row["evidence_eligible"] and not source["evidence_eligible"]:
            raise ValueError(
                f"quality.per_unit[{index}] cannot restore evidence eligibility"
            )
        if source["observation_status"] != "observed":
            if row["status"] not in {"not_observed", "unknown"} or float(row["usable_fraction"]) > _TOL:
                raise ValueError(
                    f"quality.per_unit[{index}] unobserved input must have zero usable fraction"
                )
        if row["status"] in {"unusable", "not_observed"} and float(row["usable_fraction"]) > _TOL:
            raise ValueError(
                f"quality.per_unit[{index}] status requires zero usable fraction"
            )
        if row["status"] in {"limited", "unusable", "not_observed", "unknown"} and not row["reason_codes"]:
            raise ValueError(f"quality.per_unit[{index}] status requires reason codes")
    mean_usable = sum(float(row["usable_fraction"]) for row in quality["per_unit"]) / len(quality["per_unit"])
    if abs(float(quality["usable_fraction"]) - mean_usable) > _TOL:
        raise ValueError("quality.usable_fraction must equal mean per-unit usable fraction")
    feature_families = _unique(
        (str(row["family"]) for row in quality["feature_evaluability"]),
        "quality.feature_evaluability.family",
    )
    if feature_families != _FAMILIES:
        raise ValueError("quality.feature_evaluability must cover every v2 family")
    feature_status: dict[str, str] = {}
    for index, row in enumerate(quality["feature_evaluability"]):
        family = str(row["family"])
        feature_status[family] = str(row["status"])
        refs = set(str(item) for item in row["evaluation_opportunity_ids"])
        _require_refs(refs, opportunity_ids, f"quality.feature_evaluability[{index}]")
        if any(opportunity_map[item]["family"] != family for item in refs):
            raise ValueError(
                f"quality.feature_evaluability[{index}] references a different family"
            )
        if row["status"] == "available":
            if row["reason_codes"] or not refs:
                raise ValueError(
                    f"quality.feature_evaluability[{index}] available status requires opportunities and no reasons"
                )
        elif not row["reason_codes"]:
            raise ValueError(
                f"quality.feature_evaluability[{index}] limited/not_evaluable requires reasons"
            )
    for index, artifact in enumerate(quality["artifact_intervals"]):
        _closed_interval(
            artifact["interval"],
            f"quality.artifact_intervals[{index}].interval",
            bounds=search_bounds,
        )
        _require_refs(
            (str(item) for item in artifact["affected_unit_ids"]),
            ids["input_units"],
            f"quality.artifact_intervals[{index}].affected_unit_ids",
        )

    waveform_ids = _unique(
        (str(row["waveform_evidence_id"]) for row in payload["waveform_evidence"]),
        "waveform_evidence.waveform_evidence_id",
    )
    waveforms = {
        str(row["waveform_evidence_id"]): row for row in payload["waveform_evidence"]
    }
    waveform_dependency_ids: dict[str, str] = {}
    for index, waveform in enumerate(payload["waveform_evidence"]):
        waveform_interval = _closed_interval(
            waveform["interval"],
            f"waveform_evidence[{index}].interval",
            bounds=recording_bounds,
        )
        _require_refs(
            (str(item) for item in waveform["unit_ids"]),
            ids["input_units"],
            f"waveform_evidence[{index}].unit_ids",
        )
        if waveform["canonical_signal_sha256"] != canonical_signal:
            raise ValueError(
                f"waveform_evidence[{index}] canonical signal mismatch"
            )
        waveform_units = set(str(item) for item in waveform["unit_ids"])
        eligible = (
            waveform["view_role"] not in {"detector_navigation", "unknown"}
            and waveform_units.issubset(ids["eligible_units"])
            and waveform["view_receipt_id"] is not None
            and waveform["view_receipt_sha256"] is not None
            and waveform["processed_view_sha256"] is not None
            and waveform["quality_mask_sha256"] is not None
        )
        if bool(waveform["evidence_eligible"]) != eligible:
            raise ValueError(
                f"waveform_evidence[{index}] evidence eligibility is inconsistent"
            )
        if eligible and waveform["ineligibility_reason_codes"]:
            raise ValueError(
                f"waveform_evidence[{index}] eligible waveform cannot carry ineligibility reasons"
            )
        if not eligible and not waveform["ineligibility_reason_codes"]:
            raise ValueError(
                f"waveform_evidence[{index}] ineligible waveform requires reason codes"
            )
        raw_dependency = waveform["raw_sample_dependency"]
        if raw_dependency is None:
            if not migrated:
                raise ValueError(
                    f"waveform_evidence[{index}] native v2 waveform requires raw sample dependency"
                )
        else:
            if migrated:
                raise ValueError(
                    f"waveform_evidence[{index}] migration cannot reconstruct raw sample dependency"
                )
            waveform_dependency_ids[str(waveform["waveform_evidence_id"])] = (
                _validate_raw_sample_dependency(
                    raw_dependency,
                    context=f"waveform_evidence[{index}].raw_sample_dependency",
                    canonical_signal_sha256=canonical_signal,
                    source_view_id=str(waveform["source_view_id"]),
                    view_role=str(waveform["view_role"]),
                    evidence_interval=waveform_interval,
                    view_tensor_interval=None,
                    view_receipt_id=(
                        None
                        if waveform["view_receipt_id"] is None
                        else str(waveform["view_receipt_id"])
                    ),
                    view_receipt_sha256=(
                        None
                        if waveform["view_receipt_sha256"] is None
                        else str(waveform["view_receipt_sha256"])
                    ),
                    transform_spec_sha256=None,
                )
            )

    findings = payload["findings"]
    evidence_ids = _unique(
        (str(row["evidence_id"]) for row in findings), "findings.evidence_id"
    )
    finding_map = {str(row["evidence_id"]): row for row in findings}
    measurement_ids: set[str] = set()
    used_capabilities: set[str] = set()
    used_sensitivity: set[str] = set()
    used_decisions: set[str] = set()
    for index, finding in enumerate(findings):
        context_name = f"findings[{index}]"
        family = str(finding["family"])
        term_id = str(finding["term"]["term_id"])
        status = str(finding["status"])
        role = str(finding["intrinsic_evidence_role"])
        validate_event_finding_term_context(
            term_id,
            intrinsic_evidence_role=role,
            context=context_name,
        )
        temporal_context = str(finding["signal_temporal_context"])
        opportunity_id = str(finding["evaluation_opportunity_id"])
        _require_refs((opportunity_id,), opportunity_ids, f"{context_name}.evaluation_opportunity_id")
        opportunity = opportunity_map[opportunity_id]
        if opportunity["family"] != family or opportunity["term_id"] != term_id:
            raise ValueError(f"{context_name} opportunity family/term mismatch")
        if feature_status[family] == "not_evaluable" and status != "not_evaluable":
            raise ValueError(f"{context_name} contradicts family not_evaluable status")

        finding_interval: tuple[float, float] | None = None
        if finding["time_interval"] is not None:
            finding_interval = _span(
                finding["time_interval"], f"{context_name}.time_interval", bounds=recording_bounds
            )
            if opportunity["interval"] is not None:
                opportunity_span = _span(
                    opportunity["interval"],
                    f"{context_name}.opportunity.interval",
                    bounds=recording_bounds,
                )
                if not _covered_by(finding_interval, [opportunity_span]):
                    raise ValueError(f"{context_name} lies outside its evaluation opportunity")
        if status == "present" and finding_interval is None:
            raise ValueError(f"{context_name} present status requires a time interval")
        if status == "absent_with_opportunity":
            sensitivity_id = finding["sensitivity_receipt_id"]
            if opportunity["status"] != "sufficient" or sensitivity_id is None:
                raise ValueError(
                    f"{context_name} absent_with_opportunity requires a sufficient opportunity and sensitivity receipt"
                )
            _require_refs((str(sensitivity_id),), set(sensitivity_receipts_map), f"{context_name}.sensitivity_receipt_id")
            receipt = sensitivity_receipts_map[str(sensitivity_id)]
            if receipt["qualified_family"] != family or receipt["qualified_term_id"] != term_id:
                raise ValueError(f"{context_name} sensitivity receipt family/term mismatch")
            used_sensitivity.add(str(sensitivity_id))
        elif finding["sensitivity_receipt_id"] is not None:
            raise ValueError(
                f"{context_name}.sensitivity_receipt_id is reserved for absent_with_opportunity"
            )
        if status == "not_evaluable":
            if opportunity["status"] != "not_evaluable":
                raise ValueError(f"{context_name} not_evaluable requires a not-evaluable opportunity")
            if any(
                (
                    finding["time_interval"] is not None,
                    bool(finding["spatial_support"]),
                    bool(finding["measurements"]),
                    bool(finding["waveform_evidence_ids"]),
                    bool(finding["raw_sample_dependency_ids"]),
                )
            ):
                raise ValueError(f"{context_name} not_evaluable cannot carry positive evidence")

        membership = {key: float(item) for key, item in finding["state_membership"].items()}
        membership_total = sum(membership.values())
        outside = temporal_context == "outside_candidate_protection"
        if window["state_posterior_status"] == "not_evaluable" or outside or status == "not_evaluable":
            if membership_total > _TOL:
                raise ValueError(f"{context_name} cannot claim unavailable computational-state membership")
        elif finding_interval is not None and abs(membership_total - 1.0) > _TOL:
            raise ValueError(f"{context_name}.state_membership must sum to 1")

        ownership = finding["ownership"]
        if ownership["protection_zone_id"] != protection["protection_zone_id"]:
            raise ValueError(f"{context_name} protection-zone ownership mismatch")
        owners = set(str(item) for item in ownership["owner_event_ids"])
        overlap_fraction = float(ownership["protection_zone_overlap_fraction"])
        if ownership["ownership_status"] in {"event_owned", "shared_overlap"}:
            if payload["event_id"] not in owners or overlap_fraction <= _TOL:
                raise ValueError(f"{context_name} event ownership is inconsistent")
        elif ownership["ownership_status"] == "outside_protection":
            if owners or overlap_fraction > _TOL or temporal_context != "outside_candidate_protection":
                raise ValueError(f"{context_name} outside ownership is inconsistent")
        elif not migrated:
            raise ValueError(f"{context_name} ownership_status=unknown is migration-only")
        if finding_interval is not None and ownership["ownership_status"] != "unknown":
            expected_overlap = _overlap(finding_interval, protection_bounds) / (
                finding_interval[1] - finding_interval[0]
            )
            if abs(expected_overlap - overlap_fraction) > _TOL:
                raise ValueError(f"{context_name} protection-zone overlap fraction drifted")

        if role == "onset_eligible":
            if (
                status != "present"
                or family not in _ONSET_FAMILIES
                or temporal_context != "candidate_emergence"
                or onset_bounds is None
                or window["onset_boundary"]["status"] != "observed"
                or finding_interval is None
                or _overlap(finding_interval, onset_bounds) <= _TOL
            ):
                raise ValueError(f"{context_name} fails the onset-eligible time/status gate")
            if window["state_posterior_status"] != "not_evaluable" and membership["S1"] <= _TOL:
                raise ValueError(f"{context_name} onset_eligible requires S1 membership")
        if role == "later_involvement" and temporal_context not in {
            "sustained_candidate",
            "late_involvement",
            "unknown",
        }:
            raise ValueError(f"{context_name} later involvement has incompatible temporal context")
        if role == "non_event_context" and temporal_context != "outside_candidate_protection":
            raise ValueError(f"{context_name} non-event context must be outside protection")

        support_keys: set[str] = set()
        for support_index, support in enumerate(finding["spatial_support"]):
            unit_type = str(support["unit_type"])
            identifier = str(support["id"])
            _validate_spatial_identifier(
                unit_type,
                identifier,
                ids,
                f"{context_name}.spatial_support[{support_index}]",
            )
            key = f"{unit_type}:{identifier}"
            if key in support_keys:
                raise ValueError(f"{context_name}.spatial_support contains duplicate keys")
            support_keys.add(key)
            mapped_units = _spatial_units(unit_type, identifier, ids)
            eligible = (
                support["observation_status"] in {"observed", "derived"}
                and bool(mapped_units.intersection(ids["eligible_units"]))
                and support["mapping_status"] != "candidate_only"
            )
            if bool(support["evidence_eligible"]) != eligible:
                raise ValueError(f"{context_name} spatial support eligibility is inconsistent")
            if eligible and support["missing_reason_codes"]:
                raise ValueError(f"{context_name} eligible spatial support cannot carry missing reasons")
            if not eligible and not support["missing_reason_codes"]:
                raise ValueError(f"{context_name} ineligible spatial support requires reasons")
            if support["mapping_status"] == "field_qualified" and support["field_observation"] is None:
                raise ValueError(f"{context_name} field_qualified support requires a field observation")
            if support["field_observation"] is not None:
                _require_refs(
                    (str(item) for item in support["field_observation"]["source_unit_ids"]),
                    ids["input_units"],
                    f"{context_name}.field_observation.source_unit_ids",
                )

        local_measurement_ids: set[str] = set()
        local_dependency_ids: set[str] = set()
        measurement_view_roles: set[str] = set()
        causal_binding = False
        for measurement_index, measurement in enumerate(finding["measurements"]):
            measurement_id = str(measurement["measurement_id"])
            if measurement_id in measurement_ids:
                raise ValueError("measurement_id must be globally unique within an event")
            measurement_ids.add(measurement_id)
            local_measurement_ids.add(measurement_id)
            uncertainty = measurement["numerical_uncertainty"]
            lower, upper = uncertainty["lower"], uncertainty["upper"]
            if uncertainty["status"] in {"not_estimated", "legacy_unknown"}:
                if lower is not None or upper is not None or uncertainty["coverage"] is not None or uncertainty["calibration_receipt_id"] is not None:
                    raise ValueError(f"{context_name} unknown numerical uncertainty must be null-valued")
            else:
                if lower is None or upper is None or float(lower) > float(upper) + _TOL:
                    raise ValueError(f"{context_name} numerical uncertainty interval is invalid")
                if not (float(lower) - _TOL <= float(measurement["value"]) <= float(upper) + _TOL):
                    raise ValueError(f"{context_name} measurement value lies outside uncertainty interval")
                if uncertainty["status"] == "calibrated_interval":
                    if uncertainty["coverage"] is None or uncertainty["calibration_receipt_id"] is None:
                        raise ValueError(f"{context_name} calibrated uncertainty lacks receipt/coverage")
                    _require_refs((str(uncertainty["calibration_receipt_id"]),), set(calibration_receipts), f"{context_name}.numerical_uncertainty")
            if measurement["unit_registry_status"] == "unknown" and not migrated:
                raise ValueError(f"{context_name} native v2 measurements require registered units")
            binding = measurement["source_binding"]
            if binding["canonical_signal_sha256"] != canonical_signal:
                raise ValueError(f"{context_name} measurement canonical signal mismatch")
            source_units = set(str(item) for item in binding["source_unit_ids"])
            _require_refs(source_units, ids["input_units"], f"{context_name}.source_unit_ids")
            recording_interval = _closed_interval(
                binding["recording_interval"],
                f"{context_name}.measurements[{measurement_index}].recording_interval",
                bounds=recording_bounds,
            )
            if finding_interval is None or not _covered_by(recording_interval, [finding_interval]):
                raise ValueError(f"{context_name} measurement lies outside its Finding")
            tensor_start, tensor_stop = (int(item) for item in binding["tensor_sample_interval"])
            if tensor_stop <= tensor_start:
                raise ValueError(f"{context_name} tensor sample interval is empty")
            bandwidth = _closed_interval(
                binding["effective_bandwidth_hz"],
                f"{context_name}.measurements[{measurement_index}].effective_bandwidth_hz",
            )
            if bandwidth[0] < -_TOL:
                raise ValueError("measurement bandwidth must be nonnegative")
            if binding["evidence_family"] not in _FINDING_TO_SOURCE_FAMILIES[family]:
                raise ValueError(f"{context_name} source evidence family is incompatible")
            _require_refs(
                (str(item) for item in binding["background_reference_ids"]),
                background_ids,
                f"{context_name}.background_reference_ids",
            )
            if measurement["baseline_delta"] is not None and not binding["background_reference_ids"]:
                raise ValueError(f"{context_name} baseline delta lacks background reference")
            eligible = (
                binding["view_role"] not in {"detector_navigation", "unknown"}
                and source_units.issubset(ids["eligible_units"])
                and binding["imputation_mask_sha256"] is None
            )
            if bool(binding["evidence_eligible"]) != eligible:
                raise ValueError(f"{context_name} measurement evidence eligibility is inconsistent")
            if eligible and binding["ineligibility_reason_codes"]:
                raise ValueError(f"{context_name} eligible binding cannot carry ineligibility reasons")
            if not eligible and not binding["ineligibility_reason_codes"]:
                raise ValueError(f"{context_name} ineligible binding requires reason codes")
            measurement_view_roles.add(str(binding["view_role"]))
            raw_dependency = binding["raw_sample_dependency"]
            if raw_dependency is None:
                if not migrated:
                    raise ValueError(
                        f"{context_name} native v2 measurement requires raw sample dependency"
                    )
            else:
                if migrated:
                    raise ValueError(
                        f"{context_name} migration cannot reconstruct raw sample dependency"
                    )
                local_dependency_ids.add(
                    _validate_raw_sample_dependency(
                        raw_dependency,
                        context=(
                            f"{context_name}.measurements[{measurement_index}]"
                            ".source_binding.raw_sample_dependency"
                        ),
                        canonical_signal_sha256=canonical_signal,
                        source_view_id=str(binding["source_view_id"]),
                        view_role=str(binding["view_role"]),
                        evidence_interval=recording_interval,
                        view_tensor_interval=(tensor_start, tensor_stop),
                        view_receipt_id=str(binding["view_receipt_id"]),
                        view_receipt_sha256=str(binding["view_receipt_sha256"]),
                        transform_spec_sha256=str(
                            binding["transform_spec_sha256"]
                        ),
                    )
                )
            causal_binding = causal_binding or (eligible and binding["view_role"] == "onset_causal")

        if finding["assertion_level"] == "measured" and not finding["measurements"]:
            raise ValueError(f"{context_name} measured assertion requires measurements")
        _require_refs(
            (str(item) for item in finding["waveform_evidence_ids"]),
            waveform_ids,
            f"{context_name}.waveform_evidence_ids",
        )
        causal_waveform = any(
            waveforms[str(item)]["evidence_eligible"]
            and waveforms[str(item)]["view_role"] == "onset_causal"
            for item in finding["waveform_evidence_ids"]
        )
        waveform_view_roles = {
            str(waveforms[str(item)]["view_role"])
            for item in finding["waveform_evidence_ids"]
        }
        local_dependency_ids.update(
            waveform_dependency_ids[str(item)]
            for item in finding["waveform_evidence_ids"]
            if str(item) in waveform_dependency_ids
        )
        serialized_dependency_ids = [
            str(item) for item in finding["raw_sample_dependency_ids"]
        ]
        if serialized_dependency_ids != sorted(local_dependency_ids):
            raise ValueError(
                f"{context_name}.raw_sample_dependency_ids must exactly enumerate measurement and waveform dependencies"
            )
        if role == "onset_eligible" and not (causal_binding or causal_waveform):
            raise ValueError(f"{context_name} onset_eligible requires causal-view evidence")
        if role == "onset_eligible":
            if not local_dependency_ids:
                raise ValueError(
                    f"{context_name} onset-positive atom requires raw sample dependencies"
                )
            if any(
                item != "onset_causal"
                for item in measurement_view_roles | waveform_view_roles
            ):
                raise ValueError(
                    f"{context_name} onset-positive evidence must be entirely causal"
                )

        capability_id = finding["capability_receipt_id"]
        decision_id = finding["term_decision_receipt_id"]
        if finding["assertion_level"] == "report_eligible_automated":
            if has_unknown_scope or migrated:
                raise ValueError(f"{context_name} migrated/unknown scope cannot authorize report text")
            if opportunity["status"] != "sufficient":
                raise ValueError(
                    f"{context_name} report eligibility requires a sufficient evaluation opportunity"
                )
            if capability_id is None or decision_id is None:
                raise ValueError(f"{context_name} report eligibility requires capability and term-decision receipts")
            _require_refs((str(capability_id),), set(capability_receipts), f"{context_name}.capability_receipt_id")
            _require_refs((str(decision_id),), set(term_receipts), f"{context_name}.term_decision_receipt_id")
            capability = capability_receipts[str(capability_id)]
            decision = term_receipts[str(decision_id)]
            if family not in capability["qualified_families"] or term_id not in capability["qualified_term_ids"]:
                raise ValueError(f"{context_name} is outside its capability receipt")
            if (
                decision["event_id"] != payload["event_id"]
                or decision["term_id"] != term_id
                or decision["asserted_status"] != status
                or decision["decision"] != "qualified"
                or decision["capability_receipt_id"] != capability_id
            ):
                raise ValueError(f"{context_name} term-decision receipt does not authorize this assertion")
            if decision["source_binding_sha256"] != event_term_decision_source_binding_sha256_v2(payload):
                raise ValueError(f"{context_name} term-decision source binding drifted")
            if status == "absent_with_opportunity" and decision["sensitivity_receipt_id"] != finding["sensitivity_receipt_id"]:
                raise ValueError(f"{context_name} term-decision sensitivity receipt mismatch")
            if status not in {"present", "absent_with_opportunity"}:
                raise ValueError(f"{context_name} report eligibility requires a qualified four-state assertion")
            if not finding["waveform_evidence_ids"]:
                raise ValueError(f"{context_name} report-eligible assertion requires waveform evidence")
            used_capabilities.add(str(capability_id))
            used_decisions.add(str(decision_id))
        elif capability_id is not None or decision_id is not None:
            raise ValueError(f"{context_name} lower assertion levels cannot carry clinical receipts")
        if term_id in _PROTECTED_TERMS and finding["assertion_level"] != "report_eligible_automated":
            # A legacy migration may retain the term only as an explicitly
            # non-surface candidate.  Native v2 must use the qualification path.
            if not migrated:
                raise ValueError(f"{context_name} protected term requires report eligibility")

    pattern_candidates = payload["pattern_candidates"]
    pattern_term_registry = payload["registry_bindings"].get(
        "pattern_term_registry"
    )
    pattern_term_map: dict[str, Mapping[str, Any]] = {}
    if pattern_candidates:
        if not isinstance(pattern_term_registry, Mapping):
            raise ValueError(
                "pattern candidates require an explicit pattern_term_registry"
            )
        terms = list(pattern_term_registry["terms"])
        term_ids = _unique(
            (str(row["term_id"]) for row in terms),
            "registry_bindings.pattern_term_registry.terms.term_id",
        )
        del term_ids
        if terms != sorted(terms, key=lambda row: str(row["term_id"])):
            raise ValueError("pattern term registry entries must use frozen term order")
        if pattern_term_registry["registry_sha256"] != pattern_term_registry_sha256_v1(
            pattern_term_registry
        ):
            raise ValueError("pattern term registry digest does not bind its content")
        pattern_term_map = {str(row["term_id"]): row for row in terms}
    elif pattern_term_registry is not None:
        raise ValueError(
            "pattern_term_registry must be absent or null when there are no pattern candidates"
        )

    pattern_candidate_ids = _unique(
        (str(row["pattern_candidate_id"]) for row in pattern_candidates),
        "pattern_candidates.pattern_candidate_id",
    )
    # One physical pattern instance may retain multiple competing composite
    # interpretations.  Candidate IDs are unique assertions; instance IDs are
    # deliberately reusable grouping keys.
    pattern_instance_ids = {
        str(row["pattern_instance_id"]) for row in pattern_candidates
    }
    if pattern_candidate_ids.intersection(evidence_ids):
        raise ValueError("pattern candidate IDs must not collide with atomic evidence IDs")

    required_atoms_by_instance: dict[str, set[str]] = {}
    for index, pattern in enumerate(pattern_candidates):
        context_name = f"pattern_candidates[{index}]"
        instance_id = str(pattern["pattern_instance_id"])
        if pattern["event_id"] != payload["event_id"]:
            raise ValueError(f"{context_name} belongs to a different event")

        required_atom_ids = set(str(item) for item in pattern["required_atom_ids"])
        counterevidence_ids = set(
            str(item) for item in pattern["counterevidence_ids"]
        )
        _require_refs(
            required_atom_ids,
            evidence_ids,
            f"{context_name}.required_atom_ids",
        )
        _require_refs(
            counterevidence_ids,
            evidence_ids,
            f"{context_name}.counterevidence_ids",
        )
        if required_atom_ids.intersection(counterevidence_ids):
            raise ValueError(
                f"{context_name} required atoms and counterevidence must be disjoint"
            )
        if any(
            finding_map[item]["pattern_instance_id"] != instance_id
            for item in required_atom_ids
        ):
            raise ValueError(
                f"{context_name} required atoms must carry the same pattern_instance_id"
            )
        required_atoms_by_instance.setdefault(instance_id, set()).update(
            required_atom_ids
        )

        counter_statuses = {
            str(finding_map[item]["status"]) for item in counterevidence_ids
        }
        if counter_statuses.intersection({"uncertain", "not_evaluable"}):
            raise ValueError(
                f"{context_name} uncertain/not-evaluable atoms are not counterevidence"
            )

        required_statuses = {
            str(finding_map[item]["status"]) for item in required_atom_ids
        }
        pattern_status = str(pattern["status"])
        if pattern_status == "present" and required_statuses != {"present"}:
            raise ValueError(
                f"{context_name} present pattern requires all required atoms present"
            )
        if pattern_status == "absent_with_opportunity" and (
            "absent_with_opportunity" not in required_statuses
            or not required_statuses.issubset(
                {"present", "absent_with_opportunity"}
            )
        ):
            raise ValueError(
                f"{context_name} absent pattern requires qualified atomic absence"
            )
        if pattern_status in {"uncertain", "not_evaluable"} and not pattern[
            "reason_codes"
        ]:
            raise ValueError(
                f"{context_name} uncertain/not-evaluable pattern requires reasons"
            )

        required_roles = {
            str(finding_map[item]["intrinsic_evidence_role"])
            for item in required_atom_ids
        }
        course_roles = {"early_context", "later_involvement"}
        if required_roles == {"onset_eligible"}:
            expected_scope = "onset_causal_only"
        elif required_roles and required_roles.issubset(course_roles):
            expected_scope = "event_course_only"
        elif (
            "onset_eligible" in required_roles
            and required_roles.intersection(course_roles)
            and required_roles.issubset(course_roles | {"onset_eligible"})
        ):
            expected_scope = "mixed_onset_causal_and_event_course"
        elif required_roles == {"non_event_context"}:
            expected_scope = "non_event_context_only"
        else:
            expected_scope = "not_evaluable"
        if pattern["source_domain_scope"] != expected_scope:
            raise ValueError(
                f"{context_name} source_domain_scope disagrees with its required atoms"
            )

        receipt_id = pattern["qualification_rule_receipt_id"]
        term_id = str(pattern["term"]["term_id"])
        registered_term = pattern_term_map.get(term_id)
        if registered_term is None:
            raise ValueError(
                f"{context_name}.term is absent from pattern_term_registry"
            )
        term_ref = pattern["term"]
        if (
            term_ref["source_id"] != pattern_term_registry["registry_id"]
            or term_ref["source_version"] != pattern_term_registry["version"]
            or term_ref["ontology_id"] != registered_term["ontology_id"]
            or term_ref["operational_rule_id"]
            != registered_term["operational_rule_id"]
        ):
            raise ValueError(
                f"{context_name}.term does not match its pattern registry binding"
            )
        if pattern["assertion_level"] == "report_eligible_automated":
            if (
                has_unknown_scope
                or migrated
                or pattern_status != "present"
                or receipt_id is None
                or expected_scope
                in {"non_event_context_only", "not_evaluable", "legacy_unknown"}
            ):
                raise ValueError(
                    f"{context_name} report eligibility requires a present EEG event pattern and qualification-rule receipt"
                )
            _require_refs(
                (str(receipt_id),),
                set(term_receipts),
                f"{context_name}.qualification_rule_receipt_id",
            )
            receipt = term_receipts[str(receipt_id)]
            receipt_atom_ids = {
                str(item)
                for criterion in receipt["criterion_results"]
                for item in criterion["evidence_ids"]
            }
            if receipt_atom_ids != required_atom_ids:
                raise ValueError(
                    f"{context_name} qualification rule must bind exactly the required atoms"
                )
            if {
                str(item) for item in receipt["counterevidence_ids"]
            } != counterevidence_ids:
                raise ValueError(
                    f"{context_name} qualification-rule counterevidence drifted"
                )
            if (
                receipt["event_id"] != payload["event_id"]
                or receipt["term_id"] != term_id
                or receipt["asserted_status"] != pattern_status
                or receipt["decision"] != "qualified"
            ):
                raise ValueError(
                    f"{context_name} qualification-rule receipt does not authorize this pattern"
                )
            if receipt["source_binding_sha256"] != event_term_decision_source_binding_sha256_v2(payload):
                raise ValueError(
                    f"{context_name} qualification-rule source binding drifted"
                )
            capability_id = str(receipt["capability_receipt_id"])
            _require_refs(
                (capability_id,),
                set(capability_receipts),
                f"{context_name}.qualification_rule_receipt.capability_receipt_id",
            )
            capability = capability_receipts[capability_id]
            required_families = {
                str(finding_map[item]["family"]) for item in required_atom_ids
            }
            if (
                term_id not in capability["qualified_term_ids"]
                or not required_families.issubset(
                    set(str(item) for item in capability["qualified_families"])
                )
            ):
                raise ValueError(
                    f"{context_name} is outside its qualification capability"
                )
            if not any(
                finding_map[item]["waveform_evidence_ids"]
                for item in required_atom_ids
            ):
                raise ValueError(
                    f"{context_name} report-eligible pattern requires atomic waveform evidence"
                )
            used_capabilities.add(capability_id)
            used_decisions.add(str(receipt_id))
        elif receipt_id is not None:
            raise ValueError(
                f"{context_name} model candidates cannot carry qualification-rule receipts"
            )
        if (
            term_id in _PROTECTED_TERMS
            and pattern["assertion_level"] != "report_eligible_automated"
            and not migrated
        ):
            raise ValueError(
                f"{context_name} protected term requires report eligibility"
            )

    for evidence_id, finding in finding_map.items():
        instance_id = finding["pattern_instance_id"]
        if instance_id is None:
            continue
        instance_id = str(instance_id)
        if (
            instance_id not in pattern_instance_ids
            or evidence_id not in required_atoms_by_instance.get(instance_id, set())
        ):
            raise ValueError(
                f"Finding {evidence_id!r} has a dangling pattern_instance_id"
            )

    if set(capability_receipts).difference(used_capabilities):
        raise ValueError("capability_qualification_receipts contains unreferenced receipts")
    if set(sensitivity_receipts_map).difference(used_sensitivity):
        raise ValueError("sensitivity_receipts contains unreferenced receipts")
    if set(term_receipts).difference(used_decisions):
        raise ValueError("term_decision_receipts contains unreferenced receipts")

    # Validate term-decision criteria after the complete Finding/measurement
    # catalogs exist.  IFCN morphology decisions must reference six separate
    # four-state atoms; legacy host booleans are not accepted by this schema.
    for receipt_id, receipt in term_receipts.items():
        criterion_ids = _unique(
            (str(row["criterion_id"]) for row in receipt["criterion_results"]),
            f"term_decision_receipts[{receipt_id}].criterion_id",
        )
        if receipt["term_id"] in {"spike", "sharp_wave", "interictal_epileptiform_discharge"} and criterion_ids != _IFCN_CRITERIA:
            raise ValueError(f"term decision {receipt_id!r} requires the exact six IFCN atomic criteria")
        for criterion in receipt["criterion_results"]:
            refs = set(str(item) for item in criterion["evidence_ids"])
            _require_refs(refs, evidence_ids, f"term_decision_receipts[{receipt_id}].criterion_results")
            _require_refs(
                (str(item) for item in criterion["measurement_ids"]),
                measurement_ids,
                f"term_decision_receipts[{receipt_id}].measurement_ids",
            )
            _require_refs(
                (str(item) for item in criterion["waveform_evidence_ids"]),
                waveform_ids,
                f"term_decision_receipts[{receipt_id}].waveform_evidence_ids",
            )
            _require_refs(
                (str(criterion["evaluation_opportunity_id"]),),
                opportunity_ids,
                f"term_decision_receipts[{receipt_id}].evaluation_opportunity_id",
            )
            if refs and any(finding_map[item]["status"] != criterion["status"] for item in refs):
                raise ValueError(f"term decision {receipt_id!r} criterion status disagrees with atomic Findings")
        _require_refs(
            (str(item) for item in receipt["counterevidence_ids"]),
            evidence_ids,
            f"term_decision_receipts[{receipt_id}].counterevidence_ids",
        )

    event_qualification = payload["event_qualification"]
    event_support = set(str(item) for item in event_qualification["supporting_evidence_ids"])
    _require_refs(event_support, evidence_ids, "event_qualification.supporting_evidence_ids")
    if any(finding_map[item]["status"] != "present" for item in event_support):
        raise ValueError("event qualification requires present evidence")
    qualification_receipt_id = event_qualification["qualification_receipt_id"]
    if event_qualification["status"] in {
        "qualified_electrographic_event",
        "qualified_electrographic_seizure",
    }:
        if has_unknown_scope or migrated or qualification_receipt_id is None or not event_support:
            raise ValueError("qualified event requires complete EEG-only scope, receipt, and evidence")
        if str(qualification_receipt_id) not in term_receipts and str(qualification_receipt_id) not in capability_receipts:
            raise ValueError("event qualification receipt is not trusted")
        if event_qualification["status"] == "qualified_electrographic_seizure":
            receipt = term_receipts.get(str(qualification_receipt_id))
            if receipt is None or receipt["term_id"] != "electrographic_seizure" or receipt["decision"] != "qualified":
                raise ValueError("qualified electrographic seizure requires its exact term decision")
    elif qualification_receipt_id is not None:
        raise ValueError("unqualified/not-evaluable event cannot carry a qualification receipt")
    if event_qualification["status"] == "not_evaluable" and not event_qualification["reason_codes"]:
        raise ValueError("not-evaluable event qualification requires reason codes")

    hypothesis = payload["scalp_onset_hypothesis"]
    if hypothesis["event_id"] != payload["event_id"]:
        raise ValueError("scalp_onset_hypothesis belongs to a different event")
    relation_ids = _unique(
        (str(row["relation_id"]) for row in payload["hypothesis_evidence_relations"]),
        "hypothesis_evidence_relations.relation_id",
    )
    relation_map = {
        str(row["relation_id"]): row for row in payload["hypothesis_evidence_relations"]
    }
    score_targets: set[tuple[str, str]] = set()
    scores_by_axis: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for index, score in enumerate(hypothesis["candidate_scores"]):
        axis = str(score["axis"])
        if score["candidate_type"] != _AXIS_TO_TYPE[axis]:
            raise ValueError(f"candidate_scores[{index}] axis/type mismatch")
        candidate_id = str(score["candidate_id"])
        if axis == "phenotype":
            _require_refs((candidate_id,), _PHENOTYPES, f"candidate_scores[{index}]")
        elif axis == "laterality":
            _require_refs((candidate_id,), _LATERALITIES, f"candidate_scores[{index}]")
        else:
            _validate_spatial_identifier(axis, candidate_id, ids, f"candidate_scores[{index}]")
        key = (axis, candidate_id)
        if key in score_targets:
            raise ValueError("candidate_scores contains duplicate axis/candidate entries")
        score_targets.add(key)
        scores_by_axis[axis].append(score)
        if score["score_semantics"] == "patient_disjoint_calibrated_probability":
            if not 0.0 <= float(score["score"]) <= 1.0 or score["calibration_receipt_id"] is None:
                raise ValueError("calibrated candidate score lacks probability/receipt")
            _require_refs((str(score["calibration_receipt_id"]),), set(calibration_receipts), f"candidate_scores[{index}]")
        elif score["calibration_receipt_id"] is not None:
            raise ValueError("uncalibrated candidate score cannot cite a calibration receipt")
    for axis, rows in scores_by_axis.items():
        ranks = [int(row["rank"]) for row in rows]
        if ranks != list(range(1, len(rows) + 1)):
            raise ValueError(f"candidate_scores axis {axis!r} ranks must be contiguous")
        values = [float(row["score"]) for row in rows]
        if any(current > previous + _TOL for previous, current in zip(values, values[1:])):
            raise ValueError(f"candidate_scores axis {axis!r} must be non-increasing")

    for index, relation in enumerate(payload["hypothesis_evidence_relations"]):
        if relation["hypothesis_id"] != hypothesis["hypothesis_id"]:
            raise ValueError(f"hypothesis_evidence_relations[{index}] hypothesis mismatch")
        axis = str(relation["axis"])
        candidate_id = str(relation["candidate_id"])
        if relation["candidate_type"] != _AXIS_TO_TYPE[axis] or (axis, candidate_id) not in score_targets:
            raise ValueError(f"hypothesis_evidence_relations[{index}] target is absent from candidate scores")
        refs = set(str(item) for item in relation["evidence_ids"])
        _require_refs(refs, evidence_ids, f"hypothesis_evidence_relations[{index}].evidence_ids")
        _require_refs((str(relation["producer_receipt_id"]),), set(producer_receipts), f"hypothesis_evidence_relations[{index}].producer_receipt_id")
        if relation["relation"] == "supports":
            for evidence_id in refs:
                finding = finding_map[evidence_id]
                if finding["status"] != "present":
                    raise ValueError("only present Findings may support a hypothesis")
                role = finding["intrinsic_evidence_role"]
                if role == "non_event_context":
                    raise ValueError("non-event context cannot positively support scalp onset")
                if role == "later_involvement":
                    if axis != "phenotype" or candidate_id not in {
                        "focal_with_rapid_bilateralization",
                        "bilateral_synchronous_or_rapid_bilateralization_ambiguous",
                    }:
                        raise ValueError("later involvement cannot support onset spatial axes")
                if role == "limitation" and not (
                    axis == "phenotype" and candidate_id == "scalp_onset_nonlocalizable"
                ):
                    raise ValueError("limitation evidence can support only nonlocalizable phenotype")
        else:
            if any(finding_map[item]["status"] in {"uncertain", "not_evaluable"} for item in refs):
                raise ValueError("uncertain/not-evaluable Findings cannot explicitly contradict a hypothesis")

    used_relation_ids: set[str] = set()
    for index, score in enumerate(hypothesis["candidate_scores"]):
        expected = (str(score["axis"]), str(score["candidate_id"]))
        supporting = set(str(item) for item in score["supporting_relation_ids"])
        contradictory = set(str(item) for item in score["contradictory_relation_ids"])
        _require_refs(supporting | contradictory, relation_ids, f"candidate_scores[{index}] relation IDs")
        if supporting.intersection(contradictory):
            raise ValueError("candidate score relation sets must be disjoint")
        for relation_id in supporting:
            relation = relation_map[relation_id]
            if relation["relation"] != "supports" or (relation["axis"], relation["candidate_id"]) != expected:
                raise ValueError("candidate supporting relation does not target the candidate")
        for relation_id in contradictory:
            relation = relation_map[relation_id]
            if relation["relation"] != "contradicts" or (relation["axis"], relation["candidate_id"]) != expected:
                raise ValueError("candidate contradictory relation does not target the candidate")
        used_relation_ids.update(supporting | contradictory)
        if score["axis"] in {"laterality", "region", "lead", "electrode"} and supporting:
            anchor = False
            for relation_id in supporting:
                for evidence_id in relation_map[relation_id]["evidence_ids"]:
                    finding = finding_map[str(evidence_id)]
                    if (
                        finding["family"] in _SPATIAL_ANCHOR_FAMILIES
                        and finding["intrinsic_evidence_role"] == "onset_eligible"
                        and finding["status"] == "present"
                        and any(
                            support["unit_type"] == score["candidate_type"]
                            and support["id"] == score["candidate_id"]
                            and support["evidence_eligible"]
                            for support in finding["spatial_support"]
                        )
                    ):
                        anchor = True
            if not anchor:
                raise ValueError(
                    "each supported spatial candidate requires a causal spatial-field or earliest-involvement anchor"
                )
    if used_relation_ids != relation_ids:
        raise ValueError("hypothesis_evidence_relations contains unreferenced relations")

    if hypothesis["model_receipt_id"] is not None:
        _require_refs((str(hypothesis["model_receipt_id"]),), set(producer_receipts), "scalp_onset_hypothesis.model_receipt_id")
    if hypothesis["localization_status"] == "ranked_candidates":
        if event_qualification["status"] not in {
            "qualified_electrographic_event",
            "qualified_electrographic_seizure",
        }:
            raise ValueError("ranked scalp-onset candidates require a qualified EEG event")
        if hypothesis["selected_resolution"] in {"none", "phenotype_only"}:
            raise ValueError("ranked candidates require a spatial resolution")
        selected_rows = scores_by_axis.get(str(hypothesis["selected_resolution"]), [])
        if not selected_rows or not all(row["supporting_relation_ids"] for row in selected_rows):
            raise ValueError("ranked candidates require supported selected-axis entries")
    elif hypothesis["localization_status"] in {"nonlocalizable", "not_evaluable"}:
        if hypothesis["selected_resolution"] != "none":
            raise ValueError("nonlocalizable/not-evaluable hypothesis requires resolution=none")
        if not hypothesis["reason_codes"]:
            raise ValueError("nonlocalizable/not-evaluable hypothesis requires reasons")
        if any(
            score["axis"] in {"laterality", "region", "lead", "electrode"}
            and score["supporting_relation_ids"]
            for score in hypothesis["candidate_scores"]
        ):
            raise ValueError("nonlocalizable hypothesis cannot retain supported spatial candidates")
    elif hypothesis["selected_resolution"] != "phenotype_only":
        raise ValueError("phenotype_only localization requires selected_resolution=phenotype_only")
    if hypothesis["phenotype"] == "generalized_synchronous":
        if hypothesis["selected_resolution"] in {"lead", "electrode", "region"}:
            raise ValueError("generalized synchronous phenotype cannot false-localize focally")
        bilateral_evidence = False
        for score in scores_by_axis.get("phenotype", []):
            if score["candidate_id"] != "generalized_synchronous":
                continue
            for relation_id in score["supporting_relation_ids"]:
                for evidence_id in relation_map[relation_id]["evidence_ids"]:
                    finding = finding_map[str(evidence_id)]
                    bilateral_evidence = bilateral_evidence or any(
                        support["unit_type"] == "laterality"
                        and support["id"] == "bilateral"
                        and support["evidence_eligible"]
                        for support in finding["spatial_support"]
                    )
        if not bilateral_evidence:
            raise ValueError("generalized synchronous phenotype requires bilateral signal evidence")

    per_unit_keys: set[tuple[str, str]] = set()
    per_unit_map: dict[tuple[str, str], Mapping[str, Any]] = {}
    for index, row in enumerate(hypothesis["per_unit_involvement"]):
        key = (str(row["unit_type"]), str(row["unit_id"]))
        if key in per_unit_keys:
            raise ValueError("per_unit_involvement contains duplicate keys")
        per_unit_keys.add(key)
        per_unit_map[key] = row
        _validate_spatial_identifier(key[0], key[1], ids, f"per_unit_involvement[{index}]")
        _require_refs((str(row["evaluation_opportunity_id"]),), opportunity_ids, f"per_unit_involvement[{index}].evaluation_opportunity_id")
        _require_refs((str(item) for item in row["evidence_ids"]), evidence_ids, f"per_unit_involvement[{index}].evidence_ids")
        if row["status"] == "present":
            if row["interval"] is None:
                raise ValueError("present per-unit involvement requires an interval")
            _time_interval(row["interval"], f"per_unit_involvement[{index}].interval", bounds=final_bounds)
        elif row["status"] == "absent_with_opportunity":
            if row["interval"] is not None or row["sensitivity_receipt_id"] is None:
                raise ValueError("absent per-unit involvement requires sensitivity and no onset interval")
            _require_refs((str(row["sensitivity_receipt_id"]),), set(sensitivity_receipts_map), f"per_unit_involvement[{index}].sensitivity_receipt_id")
        elif row["status"] == "not_evaluable" and (row["interval"] is not None or row["evidence_ids"]):
            raise ValueError("not-evaluable per-unit involvement cannot carry positive evidence")

    relation_keys: set[frozenset[str]] = set()
    directed: list[tuple[str, str]] = []
    involvement_relation_ids: set[str] = set()
    for index, row in enumerate(hypothesis["involvement_order"]):
        relation_id = str(row["relation_id"])
        if relation_id in involvement_relation_ids:
            raise ValueError("involvement_order contains duplicate relation IDs")
        involvement_relation_ids.add(relation_id)
        source_key = (str(row["from_type"]), str(row["from_id"]))
        target_key = (str(row["to_type"]), str(row["to_id"]))
        if source_key == target_key or source_key not in per_unit_map or target_key not in per_unit_map:
            raise ValueError("involvement_order endpoints are invalid")
        source_interval = per_unit_map[source_key]["interval"]
        target_interval = per_unit_map[target_key]["interval"]
        if source_interval is None or target_interval is None:
            raise ValueError("involvement_order endpoints require intervals")
        delay_lower, delay_upper = _time_interval(row["delay_interval"], f"involvement_order[{index}].delay_interval")
        expected_lower = float(target_interval["lower"]) - float(source_interval["upper"])
        expected_upper = float(target_interval["upper"]) - float(source_interval["lower"])
        if abs(delay_lower - expected_lower) > _TOL or abs(delay_upper - expected_upper) > _TOL:
            raise ValueError("involvement delay is inconsistent with endpoint intervals")
        source_node = f"{source_key[0]}:{source_key[1]}"
        target_node = f"{target_key[0]}:{target_key[1]}"
        unordered = frozenset((source_node, target_node))
        if unordered in relation_keys:
            raise ValueError("involvement_order contains duplicate/reverse pairs")
        relation_keys.add(unordered)
        if row["relation_status"] == "precedes":
            directed.append((source_node, target_node))
        elif source_node > target_node:
            raise ValueError("near-synchronous/unresolved pairs must use canonical endpoint order")
        _require_refs((str(item) for item in row["evidence_ids"]), evidence_ids, f"involvement_order[{index}].evidence_ids")
    _assert_acyclic(directed, "scalp_onset_hypothesis.involvement_order")

    # A migration envelope is fail-closed by construction.
    if migrated:
        migration = payload["migration"]
        if not migration["loss_codes"]:
            raise ValueError("lossy migration must enumerate loss codes")
        if (
            any(
                binding is not None
                and binding["trust_status"] != "legacy_unverified"
                for binding in payload["registry_bindings"].values()
            )
            or event_qualification["status"] != "not_evaluable"
            or hypothesis["localization_status"] != "not_evaluable"
            or payload["hypothesis_evidence_relations"]
            or payload["producer_receipts"]
            or payload["calibration_receipts"]
            or payload["capability_qualification_receipts"]
            or payload["sensitivity_receipts"]
            or payload["term_decision_receipts"]
            or payload["pattern_candidates"]
            or any(
                finding["pattern_instance_id"] is not None
                for finding in payload["findings"]
            )
        ):
            raise ValueError("legacy migration must remain fully fail-closed")

    limitation_codes = _unique(
        (str(row["code"]) for row in payload["limitations"]), "limitations.code"
    )
    if migrated and "legacy_v1_migration_loss" not in limitation_codes:
        raise ValueError("legacy migration requires a migration-loss limitation")
    return payload


__all__ = [
    "EVENT_EEG_FINDINGS_V2_SCHEMA_VERSION",
    "event_term_decision_source_binding_sha256_v2",
    "pattern_term_registry_sha256_v1",
    "validate_event_eeg_findings_v2_payload",
]
