"""Fail-closed validation for the record-level EEG SOZ claim graph.

The wire schema separates direct EEG Findings observations from research AI
hypotheses.  This module enforces the cross-object invariants that JSON Schema
cannot express: host-trusted receipts, evidence/event/mode closure,
hierarchical localization, claim relations, epistemic scope, and complete
sentence-plan coverage.  It does not generate prose or train any model.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from jsonschema import Draft202012Validator

from .clinical_term_qualification import (
    PROTECTED_EEG_ONLY_TERMS,
    validate_clinical_eeg_term_qualification,
)
from .event_finding_term_registry import validate_event_finding_term


MULTIEVENT_SOZ_REPORT_SCHEMA_VERSION = "clinical_eeg_multievent_soz_report_v1"
_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA_PATH = _ROOT / "schemas" / "clinical_eeg_multievent_soz_report_v1.schema.json"
_TOL = 1e-6
_DISCORDANT_EVENT_BACKOFF_REASON_CODE = (
    "discordant_cross_event_onset_evidence_without_mode_identifiability"
)

_PHENOTYPES = {
    "focal",
    "focal_with_rapid_bilateralization",
    "bilateral_synchronous_or_rapid_bilateralization_ambiguous",
    "generalized_synchronous",
    "multiple_scalp_onset_modes",
    "scalp_onset_nonlocalizable",
}
_LATERALITIES = {"left", "right", "bilateral", "midline", "indeterminate"}
_FORBIDDEN_SOURCES = {
    "edf_annotations",
    "excel_onset_fields",
    "doctor_labels",
    "clinical_text",
    "patient_metadata",
    "video",
    "ecg_emg_eog",
    "sleep_staging",
    "provocation",
}
_FORBIDDEN_CODE_MARKERS = {
    "annotation",
    "excel",
    "doctor_label",
    "physician_label",
    "clinical_text",
    "patient_history",
    "patient_name",
    "medication",
    "video",
    "ecg",
    "emg",
    "eog",
    "sleep_stage",
    "provocation",
}
_OBSERVATION_PREDICATES = {
    "event_detected",
    "earliest_sustained_change_maximal_at",
    "rhythm_or_morphology_observed",
    "evolves_in_frequency",
    "evolves_in_amplitude",
    "precedes_recruitment_of",
    "near_synchronous_with",
    "recruits_to",
    "terminates_at",
    "recovers_after",
    "artifact_limits_interpretation",
    "bilateral_synchronous_evolution_observed",
    "no_stable_focal_lead_observed",
    "record_signal_technically_limited",
}
_TEMPORAL_RELATION_PREDICATES = {
    "precedes_recruitment_of",
    "near_synchronous_with",
    "recruits_to",
}
_EVENT_INFERENCE_PREDICATES = {
    "event_has_onset_phenotype",
    "event_supports_soz_candidate",
}
_MODE_INFERENCE_PREDICATES = {
    "mode_repeats_onset_pattern",
    "mode_supports_soz_candidate",
}
_RECORD_INFERENCE_PREDICATES = {
    "record_primary_soz_hypothesis",
    "record_alternative_soz_hypothesis",
    "record_has_multiple_onset_modes",
    "record_has_generalized_synchronous_onset",
    "record_onset_nonlocalizable",
    "record_technical_limited",
}
_RELATION_PREDICATES = {"supports_claim", "contradicts_claim"}

_PREDICATE_SURFACE_FRAMES = {
    "event_detected": {"event_detected_v1"},
    "earliest_sustained_change_maximal_at": {
        "event_onset_maximal_at_v1",
        "event_competing_onset_fields_v1",
    },
    "rhythm_or_morphology_observed": {"event_rhythm_morphology_v1"},
    "evolves_in_frequency": {"event_evolution_v1"},
    "evolves_in_amplitude": {"event_evolution_v1"},
    "precedes_recruitment_of": {"event_onset_then_recruitment_v2"},
    "near_synchronous_with": {"event_near_synchronous_v1"},
    "recruits_to": {"event_onset_then_recruitment_v2"},
    "terminates_at": {"event_termination_v1"},
    "recovers_after": {"event_recovery_v1"},
    "artifact_limits_interpretation": {
        "event_limitation_v1",
        "record_nonlocalizable_v1",
    },
    "bilateral_synchronous_evolution_observed": {"record_generalized_synchronous_v1"},
    "no_stable_focal_lead_observed": {"record_nonlocalizable_v1"},
    "record_signal_technically_limited": {"record_technical_limited_v1"},
    "event_has_onset_phenotype": {"event_hypothesis_v1"},
    "event_supports_soz_candidate": {"event_hypothesis_v1"},
    "mode_repeats_onset_pattern": {
        "mode_recurrence_v1",
        "mode_recurrence_with_counterevidence_v1",
    },
    "mode_supports_soz_candidate": {
        "mode_recurrence_v1",
        "mode_recurrence_with_counterevidence_v1",
    },
    "record_primary_soz_hypothesis": {
        "record_primary_focal_hypothesis_v1",
        "record_primary_hypothesis_with_counterevidence_v1",
    },
    "record_alternative_soz_hypothesis": {"record_alternative_hypothesis_v1"},
    "record_has_multiple_onset_modes": {"record_multiple_modes_v1"},
    "record_has_generalized_synchronous_onset": {"record_generalized_synchronous_v1"},
    "record_onset_nonlocalizable": {"record_nonlocalizable_v1"},
    "record_technical_limited": {"record_technical_limited_v1"},
    "supports_claim": {
        "event_hypothesis_v1",
        "mode_recurrence_v1",
        "mode_recurrence_with_counterevidence_v1",
        "record_primary_focal_hypothesis_v1",
        "record_primary_hypothesis_with_counterevidence_v1",
        "record_alternative_hypothesis_v1",
        "record_multiple_modes_v1",
        "record_generalized_synchronous_v1",
        "record_nonlocalizable_v1",
    },
    "contradicts_claim": {
        "event_hypothesis_v1",
        "mode_recurrence_with_counterevidence_v1",
        "record_primary_hypothesis_with_counterevidence_v1",
        "record_alternative_hypothesis_v1",
        "record_nonlocalizable_v1",
    },
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
    rows = list(values)
    if len(rows) != len(set(rows)):
        raise ValueError(f"{context} contains duplicate identifiers")
    return set(rows)


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
    for key, raw in value.items():
        if not isinstance(key, str) or not isinstance(raw, Mapping):
            raise TypeError(f"{name} entries are invalid")
        if raw.get("receipt_id") != key:
            raise ValueError(f"{name} key/receipt_id mismatch")
        result[key] = deepcopy(dict(raw))
    return result


def _validate_host_receipts(
    rows: Sequence[Mapping[str, Any]],
    trusted: Mapping[str, Mapping[str, object]],
    *,
    context: str,
) -> dict[str, Mapping[str, Any]]:
    identifiers = _unique(
        (str(row["receipt_id"]) for row in rows), f"{context}.receipt_id"
    )
    by_id = {str(row["receipt_id"]): row for row in rows}
    for receipt_id in identifiers:
        trusted_row = trusted.get(receipt_id)
        if trusted_row is None:
            raise ValueError(
                f"{context} {receipt_id!r} is absent from the host trusted registry"
            )
        if _canonical_json(by_id[receipt_id]) != _canonical_json(trusted_row):
            raise ValueError(
                f"{context} {receipt_id!r} differs from the host trusted registry"
            )
    return by_id


def _interval(
    row: Mapping[str, object],
    context: str,
    *,
    bounds: tuple[float, float] | None = None,
) -> tuple[float, float]:
    lower = float(row["lower"])
    upper = float(row["upper"])
    if lower > upper + _TOL:
        raise ValueError(f"{context}.lower exceeds .upper")
    if bounds is not None and (lower < bounds[0] - _TOL or upper > bounds[1] + _TOL):
        raise ValueError(f"{context} lies outside [{bounds[0]}, {bounds[1]}]")
    return lower, upper


def _check_code(value: str | None, context: str) -> None:
    if value is None:
        return
    normalized = value.lower().replace("-", "_").replace(".", "_")
    if any(marker in normalized for marker in _FORBIDDEN_CODE_MARKERS):
        raise ValueError(f"{context} contains a forbidden non-EEG source marker")


def _score_axis(
    raw: Mapping[str, Any] | None,
    *,
    axis_name: str,
    allowed_candidates: set[str],
    calibrations: Mapping[str, Mapping[str, Any]],
    categorical: bool,
) -> tuple[str, str] | None:
    if raw is None:
        return None
    entries = raw["entries"]
    ranks = [int(row["rank"]) for row in entries]
    if ranks != list(range(1, len(entries) + 1)):
        raise ValueError(f"{axis_name}_scores ranks must be contiguous from 1")
    candidate_ids = _unique(
        (str(row["candidate_id"]) for row in entries),
        f"{axis_name}_scores candidates",
    )
    _require_refs(candidate_ids, allowed_candidates, f"{axis_name}_scores")
    scores = [float(row["score"]) for row in entries]
    if any(current > previous + _TOL for previous, current in zip(scores, scores[1:])):
        raise ValueError(f"{axis_name}_scores must be non-increasing by rank")
    prediction_set = set(str(item) for item in raw["prediction_set"])
    _require_refs(prediction_set, candidate_ids, f"{axis_name}_scores.prediction_set")
    semantics = str(raw["score_semantics"])
    calibration_id = raw["calibration_receipt_id"]
    if semantics == "uncalibrated_ranking_score":
        if calibration_id is not None or prediction_set:
            raise ValueError(
                f"uncalibrated {axis_name} scores cannot carry calibration or prediction_set"
            )
    else:
        if calibration_id is None:
            raise ValueError(
                f"calibrated {axis_name} probability requires a calibration receipt"
            )
        _require_refs(
            (str(calibration_id),),
            set(calibrations),
            f"{axis_name}_scores.calibration_receipt_id",
        )
        calibration = calibrations[str(calibration_id)]
        if axis_name not in calibration["calibrated_outputs"]:
            raise ValueError(f"calibration receipt does not cover {axis_name} output")
        if any(score < -_TOL or score > 1.0 + _TOL for score in scores):
            raise ValueError(f"calibrated {axis_name} scores must lie in [0,1]")
        if categorical and abs(sum(scores) - 1.0) > _TOL:
            raise ValueError(f"calibrated categorical {axis_name} scores must sum to 1")
        if not prediction_set:
            raise ValueError(f"calibrated {axis_name} output requires a prediction_set")
    return str(entries[0]["candidate_id"]), semantics


def _assert_acyclic(edges: Sequence[tuple[str, str]], context: str) -> None:
    adjacency: dict[str, set[str]] = defaultdict(set)
    indegree: dict[str, int] = {}
    for source, target in edges:
        if source == target:
            raise ValueError(f"{context} contains a self relation")
        adjacency[source].add(target)
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


def _validate_claim_time(
    value: Mapping[str, Any],
    *,
    context: str,
    recording_bounds: tuple[float, float],
    event_bounds: tuple[float, float] | None,
) -> None:
    kind = str(value["kind"])
    lower = value["lower"]
    upper = value["upper"]
    if kind == "none":
        if (
            value["timebase"] != "not_applicable"
            or lower is not None
            or upper is not None
            or value["left_censored"]
            or value["right_censored"]
        ):
            raise ValueError(f"{context} kind=none must not carry temporal values")
        return
    if lower is None or upper is None:
        raise ValueError(f"{context} requires lower and upper")
    interval = (float(lower), float(upper))
    if interval[0] > interval[1] + _TOL:
        raise ValueError(f"{context}.lower exceeds .upper")
    if kind == "recording_interval":
        if value["timebase"] != "recording_relative_seconds":
            raise ValueError(f"{context} recording interval has wrong timebase")
        bounds = event_bounds if event_bounds is not None else recording_bounds
        if interval[0] < bounds[0] - _TOL or interval[1] > bounds[1] + _TOL:
            raise ValueError(f"{context} lies outside its EEG interval")
    elif value["timebase"] != "relative_delay_seconds":
        raise ValueError(f"{context} delay interval has wrong timebase")


def validate_multievent_soz_report_payload(
    value: object,
    *,
    trusted_producer_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_calibration_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_capability_qualification_receipts: (
        Mapping[str, Mapping[str, object]] | None
    ) = None,
    trusted_term_decision_receipts: (Mapping[str, Mapping[str, object]] | None) = None,
) -> dict[str, Any]:
    """Validate and defensively copy one EEG-only record hypothesis graph."""

    if type(value) is not dict:
        raise TypeError(
            "clinical_eeg_multievent_soz_report_v1 payload must be an object"
        )
    _reject_nonfinite(value)
    errors = sorted(
        _schema_validator().iter_errors(value), key=lambda item: list(item.path)
    )
    if errors:
        rendered = "; ".join(f"{_path(error)}: {error.message}" for error in errors[:8])
        if len(errors) > 8:
            rendered += f"; ... {len(errors) - 8} more error(s)"
        raise ValueError(
            f"clinical_eeg_multievent_soz_report_v1 schema validation failed: {rendered}"
        )
    payload: dict[str, Any] = deepcopy(value)

    trusted_producers = _trusted_registry(
        trusted_producer_receipts, name="trusted_producer_receipts"
    )
    trusted_calibrations = _trusted_registry(
        trusted_calibration_receipts, name="trusted_calibration_receipts"
    )
    trusted_capabilities = _trusted_registry(
        trusted_capability_qualification_receipts,
        name="trusted_capability_qualification_receipts",
    )
    trusted_term_decisions = _trusted_registry(
        trusted_term_decision_receipts,
        name="trusted_term_decision_receipts",
    )
    producer_by_id = _validate_host_receipts(
        payload["producer_receipts"],
        trusted_producers,
        context="producer receipt",
    )
    calibration_by_id = _validate_host_receipts(
        payload["calibration_receipts"],
        trusted_calibrations,
        context="calibration receipt",
    )
    capability_by_id = _validate_host_receipts(
        payload["capability_qualification_receipts"],
        trusted_capabilities,
        context="capability qualification receipt",
    )
    term_decision_by_id = _validate_host_receipts(
        payload["term_decision_receipts"],
        trusted_term_decisions,
        context="term-decision receipt",
    )
    for receipt_id, receipt in list(term_decision_by_id.items()):
        validated_receipt = validate_clinical_eeg_term_qualification(receipt)
        validated_trusted = validate_clinical_eeg_term_qualification(
            trusted_term_decisions[receipt_id]
        )
        if _canonical_json(validated_receipt) != _canonical_json(validated_trusted):
            raise ValueError(
                f"term-decision receipt {receipt_id!r} differs from the host "
                "trusted registry after rule validation"
            )
        term_decision_by_id[receipt_id] = validated_receipt
    for receipt_id, receipt in producer_by_id.items():
        disjoint_scope = receipt["validation_scope"] in {
            "source_dev_patient_disjoint",
            "external_patient_disjoint",
        }
        if bool(receipt["patient_disjoint"]) != disjoint_scope:
            raise ValueError(
                f"producer receipt {receipt_id!r} has inconsistent patient-disjoint scope"
            )
        if (
            receipt["producer_type"]
            in {
                "hierarchical_mil_hypothesis_model",
                "risk_controller",
            }
            and not disjoint_scope
        ):
            raise ValueError(
                f"producer receipt {receipt_id!r} requires patient-disjoint validation"
            )

    if set(payload["report_policy"]["forbidden_sources"]) != _FORBIDDEN_SOURCES:
        raise ValueError(
            "report_policy.forbidden_sources must enumerate the complete firewall"
        )
    if "scalp_eeg_signal" not in payload["provenance"]["input_sources"] or (
        "eeg_derived_findings" not in payload["provenance"]["input_sources"]
    ):
        raise ValueError(
            "provenance must bind scalp EEG signal and EEG-derived Findings"
        )

    recording_bounds = (
        0.0,
        float(payload["provenance"]["recording_duration_seconds"]),
    )
    electrode_ids = set(str(item) for item in payload["ontology"]["electrode_ids"])
    region_rows = payload["ontology"]["regions"]
    region_ids = _unique(
        (str(row["region_id"]) for row in region_rows), "ontology.regions.region_id"
    )
    electrode_to_region: dict[str, str] = {}
    region_to_laterality: dict[str, str] = {}
    for index, region in enumerate(region_rows):
        region_id = str(region["region_id"])
        region_to_laterality[region_id] = str(region["laterality"])
        members = set(str(item) for item in region["electrode_ids"])
        _require_refs(
            members, electrode_ids, f"ontology.regions[{index}].electrode_ids"
        )
        for electrode in members:
            if electrode in electrode_to_region:
                raise ValueError(
                    f"ontology electrode {electrode!r} belongs to multiple regions"
                )
            electrode_to_region[electrode] = region_id
    if set(electrode_to_region) != electrode_ids:
        raise ValueError("ontology.regions must cover every electrode exactly once")

    event_rows = payload["events"]
    event_ids = _unique((str(row["event_id"]) for row in event_rows), "events.event_id")
    event_by_id = {str(row["event_id"]): row for row in event_rows}
    event_term_source_bindings = {
        str(row["event_id"]): str(row["term_decision_source_binding_sha256"])
        for row in event_rows
    }
    event_bounds: dict[str, tuple[float, float]] = {}
    usable_event_ids: set[str] = set()
    for index, event in enumerate(event_rows):
        event_id = str(event["event_id"])
        bounds = _interval(
            event["analysis_interval"],
            f"events[{index}].analysis_interval",
            bounds=recording_bounds,
        )
        _interval(
            event["onset_interval"],
            f"events[{index}].onset_interval",
            bounds=bounds,
        )
        event_bounds[event_id] = bounds
        for code in event["limitation_codes"]:
            _check_code(str(code), f"events[{index}].limitation_codes")
        if event["usable_for_hypothesis"]:
            usable_event_ids.add(event_id)
            if event["mode_id"] is None or not event["finding_evidence_ids"]:
                raise ValueError(
                    f"events[{index}] usable event requires a mode and EEG Findings evidence"
                )
        elif event["mode_id"] is not None:
            raise ValueError(
                f"events[{index}] unusable event cannot be assigned to a mode"
            )

    evidence_rows = payload["evidence_catalog"]
    evidence_ids = _unique(
        (str(row["evidence_id"]) for row in evidence_rows),
        "evidence_catalog.evidence_id",
    )
    finding_ids = _unique(
        (str(row["finding_id"]) for row in evidence_rows),
        "evidence_catalog.finding_id",
    )
    evidence_by_id = {str(row["evidence_id"]): row for row in evidence_rows}
    evidence_by_event: dict[str, set[str]] = defaultdict(set)
    used_capability_receipt_ids: set[str] = set()
    used_term_decision_receipt_ids: set[str] = set()
    for index, evidence in enumerate(evidence_rows):
        event_id = str(evidence["event_id"])
        _require_refs((event_id,), event_ids, f"evidence_catalog[{index}].event_id")
        evidence_by_event[event_id].add(str(evidence["evidence_id"]))
        producer_id = str(evidence["producer_receipt_id"])
        _require_refs((producer_id,), set(producer_by_id), f"evidence_catalog[{index}]")
        if producer_by_id[producer_id]["producer_type"] != "event_findings_provider":
            raise ValueError(
                f"evidence_catalog[{index}] requires event_findings_provider"
            )
        term = validate_event_finding_term(
            evidence["term"],
            family=evidence["family"],
            assertion_level=evidence["assertion_level"],
            context=f"evidence_catalog[{index}]",
        )
        if evidence["status"] == "not_evaluable":
            if evidence["waveform_evidence_ids"]:
                raise ValueError(
                    f"evidence_catalog[{index}] not_evaluable evidence cannot carry waveform support"
                )
        elif not evidence["waveform_evidence_ids"]:
            raise ValueError(
                f"evidence_catalog[{index}] evaluable evidence requires waveform support"
            )
        if evidence["assertion_level"] == "clinically_qualified":
            capability_receipt_id = evidence["qualification_receipt_id"]
            term_decision_receipt_id = evidence["term_decision_receipt_id"]
            if capability_receipt_id is None:
                raise ValueError(
                    f"evidence_catalog[{index}] clinically qualified evidence "
                    "requires a capability receipt"
                )
            if term_decision_receipt_id is None:
                raise ValueError(
                    f"evidence_catalog[{index}] clinically qualified evidence "
                    "requires a term-decision receipt"
                )
            _require_refs(
                (str(capability_receipt_id),),
                set(capability_by_id),
                f"evidence_catalog[{index}].qualification_receipt_id",
            )
            _require_refs(
                (str(term_decision_receipt_id),),
                set(term_decision_by_id),
                f"evidence_catalog[{index}].term_decision_receipt_id",
            )
            capability = capability_by_id[str(capability_receipt_id)]
            if evidence["family"] not in capability["qualified_families"]:
                raise ValueError(
                    f"evidence_catalog[{index}] family is outside its capability receipt"
                )
            if term not in capability["qualified_terms"]:
                raise ValueError(
                    f"evidence_catalog[{index}] term is outside its capability receipt"
                )
            decision = term_decision_by_id[str(term_decision_receipt_id)]
            if decision["event_id"] != event_id:
                raise ValueError(
                    f"evidence_catalog[{index}] term-decision receipt event mismatch"
                )
            if (
                decision["source_binding_sha256"]
                != event_term_source_bindings[event_id]
            ):
                raise ValueError(
                    f"evidence_catalog[{index}] term-decision source binding mismatch"
                )
            if term not in decision["qualified_terms"]:
                raise ValueError(
                    f"evidence_catalog[{index}] term is outside its per-event decision"
                )
            if term not in PROTECTED_EEG_ONLY_TERMS:
                raise ValueError(
                    f"evidence_catalog[{index}] term is outside the protected registry"
                )
            used_capability_receipt_ids.add(str(capability_receipt_id))
            used_term_decision_receipt_ids.add(str(term_decision_receipt_id))
        elif (
            evidence["qualification_receipt_id"] is not None
            or evidence["term_decision_receipt_id"] is not None
        ):
            raise ValueError(
                f"evidence_catalog[{index}] non-qualified evidence cannot carry "
                "capability or term-decision qualification"
            )
    unused_capabilities = set(capability_by_id).difference(used_capability_receipt_ids)
    if unused_capabilities:
        raise ValueError(
            "capability_qualification_receipts contains unreferenced receipts: "
            f"{sorted(unused_capabilities)}"
        )
    unused_term_decisions = set(term_decision_by_id).difference(
        used_term_decision_receipt_ids
    )
    if unused_term_decisions:
        raise ValueError(
            "term_decision_receipts contains unreferenced receipts: "
            f"{sorted(unused_term_decisions)}"
        )
    for index, event in enumerate(event_rows):
        declared = set(str(item) for item in event["finding_evidence_ids"])
        expected = evidence_by_event.get(str(event["event_id"]), set())
        if declared != expected:
            raise ValueError(
                f"events[{index}].finding_evidence_ids must exactly close its evidence catalog"
            )

    mode_rows = payload["modes"]
    mode_ids = _unique((str(row["mode_id"]) for row in mode_rows), "modes.mode_id")
    mode_by_id = {str(row["mode_id"]): row for row in mode_rows}
    assigned_events: set[str] = set()
    for index, mode in enumerate(mode_rows):
        members = set(str(item) for item in mode["event_ids"])
        _require_refs(members, usable_event_ids, f"modes[{index}].event_ids")
        if assigned_events.intersection(members):
            raise ValueError("usable events cannot belong to more than one mode")
        assigned_events.update(members)
        if int(mode["event_count"]) != len(members):
            raise ValueError(f"modes[{index}].event_count does not match event_ids")
        if int(mode["total_usable_event_count"]) != len(usable_event_ids):
            raise ValueError(
                f"modes[{index}].total_usable_event_count does not match the record"
            )
        for event_id in members:
            if event_by_id[event_id]["mode_id"] != mode["mode_id"]:
                raise ValueError(f"modes[{index}] conflicts with event.mode_id")
    if assigned_events != usable_event_ids:
        raise ValueError("modes must partition every usable event exactly once")

    hypothesis_rows = payload["hypotheses"]
    hypothesis_ids = _unique(
        (str(row["hypothesis_id"]) for row in hypothesis_rows),
        "hypotheses.hypothesis_id",
    )
    hypothesis_by_id = {str(row["hypothesis_id"]): row for row in hypothesis_rows}
    event_hypotheses: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    mode_hypotheses: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    record_hypotheses: list[Mapping[str, Any]] = []
    for index, hypothesis in enumerate(hypothesis_rows):
        context = f"hypotheses[{index}]"
        scope = str(hypothesis["scope"])
        role = str(hypothesis["role"])
        event_id = hypothesis["event_id"]
        mode_id = hypothesis["mode_id"]
        if scope == "event":
            if event_id is None or mode_id is None or role != "event_specific":
                raise ValueError(f"{context} event scope has inconsistent IDs or role")
            _require_refs((str(event_id),), usable_event_ids, f"{context}.event_id")
            _require_refs((str(mode_id),), mode_ids, f"{context}.mode_id")
            if event_by_id[str(event_id)]["mode_id"] != mode_id:
                raise ValueError(f"{context} event/mode attribution mismatch")
            event_hypotheses[str(event_id)].append(hypothesis)
        elif scope == "mode":
            if event_id is not None or mode_id is None or role != "mode_specific":
                raise ValueError(f"{context} mode scope has inconsistent IDs or role")
            _require_refs((str(mode_id),), mode_ids, f"{context}.mode_id")
            mode_hypotheses[str(mode_id)].append(hypothesis)
        else:
            if (
                event_id is not None
                or mode_id is not None
                or role not in {"primary", "alternative"}
            ):
                raise ValueError(f"{context} record scope has inconsistent IDs or role")
            record_hypotheses.append(hypothesis)

        model_receipt_id = str(hypothesis["model_receipt_id"])
        _require_refs(
            (model_receipt_id,), set(producer_by_id), f"{context}.model_receipt_id"
        )
        if (
            producer_by_id[model_receipt_id]["producer_type"]
            != "hierarchical_mil_hypothesis_model"
        ):
            raise ValueError(f"{context} requires hierarchical_mil_hypothesis_model")

        supporting = set(str(item) for item in hypothesis["supporting_evidence_ids"])
        contradictory = set(
            str(item) for item in hypothesis["contradictory_evidence_ids"]
        )
        _require_refs(supporting | contradictory, evidence_ids, f"{context} evidence")
        if supporting.intersection(contradictory):
            raise ValueError(
                f"{context} supporting and contradictory evidence must be disjoint"
            )
        if any(
            evidence_by_id[item]["status"] != "present"
            or evidence_by_id[item]["evidence_role"] != "onset_support"
            for item in supporting
        ):
            raise ValueError(
                f"{context} SOZ supporting evidence must be "
                "status=present and evidence_role=onset_support"
            )
        if any(
            evidence_by_id[item]["evidence_role"] != "contradiction"
            for item in contradictory
        ):
            raise ValueError(
                f"{context} contradictory evidence requires "
                "evidence_role=contradiction"
            )
        if any(
            evidence_by_id[item]["status"] == "not_evaluable"
            for item in supporting | contradictory
        ):
            raise ValueError(f"{context} cannot use not_evaluable evidence")
        support_events = {str(evidence_by_id[item]["event_id"]) for item in supporting}
        contradiction_events = {
            str(evidence_by_id[item]["event_id"]) for item in contradictory
        }
        if support_events != set(
            str(item) for item in hypothesis["supporting_event_ids"]
        ):
            raise ValueError(
                f"{context}.supporting_event_ids do not close supporting evidence"
            )
        if contradiction_events != set(
            str(item) for item in hypothesis["contradictory_event_ids"]
        ):
            raise ValueError(
                f"{context}.contradictory_event_ids do not close contradictory evidence"
            )
        if scope == "event" and not (support_events | contradiction_events).issubset(
            {str(event_id)}
        ):
            raise ValueError(
                f"{context} event hypothesis uses evidence from another event"
            )
        if scope == "mode":
            members = set(str(item) for item in mode_by_id[str(mode_id)]["event_ids"])
            if not (support_events | contradiction_events).issubset(members):
                raise ValueError(
                    f"{context} mode hypothesis uses evidence from another mode"
                )

        phenotype = hypothesis["phenotype"]
        resolution = str(hypothesis["selected_resolution"])
        if resolution in {"electrode", "region"} and not any(
            evidence_by_id[item]["status"] == "present"
            and evidence_by_id[item]["evidence_role"] == "onset_support"
            for item in supporting
        ):
            raise ValueError(
                f"{context} electrode/region resolution requires present onset support"
            )
        for code in hypothesis["reason_codes"]:
            _check_code(str(code), f"{context}.reason_codes")
        if resolution == "technical_limited":
            if phenotype is not None or hypothesis["phenotype_scores"] is not None:
                raise ValueError(
                    f"{context} technical hypothesis cannot assert a phenotype"
                )
            if supporting or contradictory:
                raise ValueError(
                    f"{context} technical hypothesis cannot assert physiological evidence"
                )
        else:
            if phenotype is None or not supporting:
                raise ValueError(
                    f"{context} analyzable hypothesis requires phenotype and support"
                )

        tops: dict[str, tuple[str, str] | None] = {
            "phenotype": _score_axis(
                hypothesis["phenotype_scores"],
                axis_name="phenotype",
                allowed_candidates=_PHENOTYPES,
                calibrations=calibration_by_id,
                categorical=True,
            ),
            "laterality": _score_axis(
                hypothesis["laterality_scores"],
                axis_name="laterality",
                allowed_candidates=_LATERALITIES,
                calibrations=calibration_by_id,
                categorical=True,
            ),
            "region": _score_axis(
                hypothesis["region_scores"],
                axis_name="region",
                allowed_candidates=region_ids,
                calibrations=calibration_by_id,
                categorical=True,
            ),
            "channel": _score_axis(
                hypothesis["channel_scores"],
                axis_name="channel",
                allowed_candidates=electrode_ids,
                calibrations=calibration_by_id,
                categorical=False,
            ),
        }
        semantics = {item[1] for item in tops.values() if item is not None}
        if len(semantics) > 1:
            raise ValueError(f"{context} cannot mix calibrated and uncalibrated axes")
        if phenotype is not None and (
            tops["phenotype"] is None or tops["phenotype"][0] != phenotype
        ):
            raise ValueError(f"{context} phenotype must be rank-1 phenotype_scores")
        required_axes = {
            "electrode": {"phenotype", "laterality", "region", "channel"},
            "region": {"phenotype", "laterality", "region"},
            "laterality": {"phenotype", "laterality"},
            "multiple_modes": {"phenotype"},
            "phenotype_only": {"phenotype"},
            "technical_limited": set(),
        }[resolution]
        present_axes = {name for name, item in tops.items() if item is not None}
        if present_axes != required_axes:
            raise ValueError(
                f"{context} axes do not match risk-selected resolution {resolution}"
            )
        if resolution in {"electrode", "region"}:
            top_region = tops["region"][0]
            top_laterality = tops["laterality"][0]
            if region_to_laterality[top_region] != top_laterality:
                raise ValueError(f"{context} top region conflicts with top laterality")
        if resolution == "electrode":
            top_channel = tops["channel"][0]
            if electrode_to_region[top_channel] != tops["region"][0]:
                raise ValueError(f"{context} top channel conflicts with top region")

        if phenotype == "generalized_synchronous":
            if resolution not in {"laterality", "phenotype_only"}:
                raise ValueError(f"{context} generalized phenotype cannot be focalized")
            if tops["laterality"] is not None and tops["laterality"][0] != "bilateral":
                raise ValueError(
                    f"{context} generalized phenotype requires bilateral laterality"
                )
        if phenotype == "multiple_scalp_onset_modes" and resolution != "multiple_modes":
            raise ValueError(
                f"{context} multiple-mode phenotype requires multiple_modes resolution"
            )
        if phenotype == "scalp_onset_nonlocalizable":
            if (
                resolution not in {"laterality", "phenotype_only"}
                or not hypothesis["reason_codes"]
            ):
                raise ValueError(
                    f"{context} nonlocalizable phenotype requires reason codes and coarse resolution"
                )

        risk = hypothesis["risk_control"]
        if risk["selected_resolution"] != resolution:
            raise ValueError(f"{context}.risk_control selected_resolution mismatch")
        for code in risk["finer_resolution_rejected_reason_codes"]:
            _check_code(str(code), f"{context}.risk_control reason codes")
        if risk["status"] == "passed":
            policy_id = risk["policy_receipt_id"]
            if policy_id is None:
                raise ValueError(f"{context} passed risk control requires a receipt")
            _require_refs(
                (str(policy_id),), set(producer_by_id), f"{context}.risk_control"
            )
            receipt = producer_by_id[str(policy_id)]
            if (
                receipt["producer_type"] != "risk_controller"
                or not receipt["patient_disjoint"]
            ):
                raise ValueError(
                    f"{context} risk control requires trusted patient-disjoint controller"
                )
            estimated = risk["estimated_conditional_risk"]
            limit = risk["risk_limit"]
            if (
                estimated is None
                or limit is None
                or risk["risk_semantics"]
                != "patient_disjoint_conditional_error_estimate"
            ):
                raise ValueError(
                    f"{context} passed risk control requires risk and limit"
                )
            if not (0 <= float(estimated) <= float(limit) <= 1.0):
                raise ValueError(f"{context} estimated risk exceeds its frozen limit")
            if (
                resolution != "electrode"
                and not risk["finer_resolution_rejected_reason_codes"]
            ):
                raise ValueError(
                    f"{context} coarser resolution requires rejected-finer reasons"
                )
        elif risk["status"] == "backoff_no_risk_calibration":
            if (
                risk["policy_receipt_id"] is not None
                or risk["estimated_conditional_risk"] is not None
                or risk["risk_limit"] is not None
                or risk["risk_semantics"] != "not_available"
                or resolution not in {"phenotype_only", "multiple_modes"}
                or not risk["finer_resolution_rejected_reason_codes"]
            ):
                raise ValueError(f"{context} invalid no-risk-calibration backoff")
        else:
            if (
                resolution != "technical_limited"
                or risk["policy_receipt_id"] is not None
                or risk["estimated_conditional_risk"] is not None
                or risk["risk_limit"] is not None
                or risk["risk_semantics"] != "not_available"
            ):
                raise ValueError(f"{context} invalid technical risk-control state")

    if payload["analysis_status"] == "analyzable":
        if not usable_event_ids:
            raise ValueError("analyzable record requires at least one usable event")
        primary = [row for row in record_hypotheses if row["role"] == "primary"]
        if len(primary) != 1 or primary[0]["phenotype"] is None:
            raise ValueError(
                "analyzable record requires exactly one forced primary phenotype"
            )
        if any(
            row["selected_resolution"] == "technical_limited" for row in hypothesis_rows
        ):
            raise ValueError(
                "analyzable record cannot contain technical-only hypotheses"
            )
    else:
        primary = [row for row in record_hypotheses if row["role"] == "primary"]
        if (
            usable_event_ids
            or len(primary) != 1
            or primary[0]["selected_resolution"] != "technical_limited"
        ):
            raise ValueError(
                "technical/detector-miss record requires one technical primary hypothesis"
            )
    if set(event_hypotheses) != usable_event_ids or any(
        len(rows) != 1 for rows in event_hypotheses.values()
    ):
        raise ValueError("every usable event requires exactly one event hypothesis")
    if set(mode_hypotheses) != mode_ids or any(
        len([row for row in rows if row["role"] == "mode_specific"]) != 1
        for rows in mode_hypotheses.values()
    ):
        raise ValueError("every mode requires exactly one mode hypothesis")

    for index, mode in enumerate(mode_rows):
        _require_refs(
            mode["hypothesis_ids"], hypothesis_ids, f"modes[{index}].hypothesis_ids"
        )
        expected_ids = {
            str(row["hypothesis_id"]) for row in mode_hypotheses[str(mode["mode_id"])]
        }
        if set(mode["hypothesis_ids"]) != expected_ids:
            raise ValueError(
                f"modes[{index}].hypothesis_ids do not close mode hypotheses"
            )
        primary_id = str(mode["primary_hypothesis_id"])
        if primary_id not in expected_ids:
            raise ValueError(f"modes[{index}].primary_hypothesis_id is not mode-scoped")
        primary_hypothesis = hypothesis_by_id[primary_id]
        for field in (
            "supporting_event_ids",
            "contradictory_event_ids",
            "supporting_evidence_ids",
            "contradictory_evidence_ids",
        ):
            if set(mode[field]) != set(primary_hypothesis[field]):
                raise ValueError(
                    f"modes[{index}].{field} must match its primary hypothesis"
                )

    claim_rows = payload["claims"]
    claim_ids = _unique((str(row["claim_id"]) for row in claim_rows), "claims.claim_id")
    claim_by_id = {str(row["claim_id"]): row for row in claim_rows}
    entity_universes = {
        "eeg_record": {str(payload["record_id"])},
        "eeg_event": event_ids,
        "mode": mode_ids,
        "finding": finding_ids,
        "evidence": evidence_ids,
        "claim": claim_ids,
        "hypothesis": hypothesis_ids,
        "phenotype": _PHENOTYPES,
        "laterality": _LATERALITIES,
        "region": region_ids,
        "electrode": electrode_ids,
    }
    core_claims_by_hypothesis: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    relation_claims: list[Mapping[str, Any]] = []
    for index, claim in enumerate(claim_rows):
        context = f"claims[{index}]"
        producer_id = str(claim["producer_receipt_id"])
        _require_refs(
            (producer_id,), set(producer_by_id), f"{context}.producer_receipt_id"
        )
        if (
            producer_by_id[producer_id]["producer_type"]
            != "deterministic_claim_builder"
        ):
            raise ValueError(
                f"{context} claims must come from deterministic_claim_builder"
            )
        entities = [claim["subject"]] + list(claim["object_or_value"]["entities"])
        _unique(
            (
                f"{row['type']}:{row['id']}"
                for row in claim["object_or_value"]["entities"]
            ),
            f"{context}.object_or_value.entities",
        )
        for entity in entities:
            _require_refs(
                (str(entity["id"]),),
                entity_universes[str(entity["type"])],
                f"{context} entity {entity['type']}",
            )
        _unique(
            (str(row["name"]) for row in claim["object_or_value"]["measurements"]),
            f"{context}.object_or_value.measurements",
        )
        _check_code(claim["object_or_value"]["code"], f"{context}.object_or_value.code")
        allowed_frames = set(str(item) for item in claim["allowed_surface_frames"])
        if not allowed_frames.issubset(
            _PREDICATE_SURFACE_FRAMES[str(claim["predicate"])]
        ):
            raise ValueError(f"{context} contains an unauthorized surface frame")
        evidence = set(str(item) for item in claim["evidence_ids"])
        _require_refs(evidence, evidence_ids, f"{context}.evidence_ids")
        if claim["polarity"] == "affirmed" and claim["negation_scope"] != "none":
            raise ValueError(f"{context} affirmed claim must have negation_scope=none")
        if claim["polarity"] == "negated" and claim["negation_scope"] == "none":
            raise ValueError(
                f"{context} negated claim requires explicit negation scope"
            )
        claim_event = claim["event_id"]
        claim_mode = claim["mode_id"]
        if claim_event is not None:
            _require_refs((str(claim_event),), event_ids, f"{context}.event_id")
        if claim_mode is not None:
            _require_refs((str(claim_mode),), mode_ids, f"{context}.mode_id")
        _validate_claim_time(
            claim["time"],
            context=f"{context}.time",
            recording_bounds=recording_bounds,
            event_bounds=event_bounds.get(str(claim_event))
            if claim_event is not None
            else None,
        )

        layer = str(claim["layer"])
        kind = str(claim["claim_kind"])
        predicate = str(claim["predicate"])
        epistemic = str(claim["epistemic_status"])
        if layer == "eeg_findings_observation":
            if kind != "observation" or predicate not in _OBSERVATION_PREDICATES:
                raise ValueError(
                    f"{context} observation layer cannot carry inference claims"
                )
            if epistemic not in {
                "measured",
                "model_candidate",
                "clinically_qualified",
                "not_evaluable",
            }:
                raise ValueError(f"{context} observation has invalid epistemic status")
            if claim["hypothesis_id"] is not None:
                raise ValueError(f"{context} observation cannot bind a hypothesis")
            if evidence:
                evidence_events = {
                    str(evidence_by_id[item]["event_id"]) for item in evidence
                }
                if len(evidence_events) != 1 or claim_event not in evidence_events:
                    raise ValueError(
                        f"{context} event attribution conflicts with evidence"
                    )
                evidence_findings = {
                    str(evidence_by_id[item]["finding_id"]) for item in evidence
                }
                if claim["subject"]["type"] != "finding" or evidence_findings != {
                    str(claim["subject"]["id"])
                }:
                    raise ValueError(
                        f"{context} finding subject does not close its evidence terms"
                    )
                expected_mode = event_by_id[str(claim_event)]["mode_id"]
                if claim_mode != expected_mode:
                    raise ValueError(f"{context} mode attribution conflicts with event")
            elif predicate != "record_signal_technically_limited":
                raise ValueError(f"{context} EEG observation requires evidence")
            if epistemic == "clinically_qualified":
                capability_ids = {
                    evidence_by_id[item]["qualification_receipt_id"]
                    for item in evidence
                }
                term_decision_ids = {
                    evidence_by_id[item]["term_decision_receipt_id"]
                    for item in evidence
                }
                if (
                    None in capability_ids
                    or len(capability_ids) != 1
                    or claim["qualification_receipt_id"] not in capability_ids
                    or None in term_decision_ids
                    or len(term_decision_ids) != 1
                    or claim.get("term_decision_receipt_id") not in term_decision_ids
                    or any(
                        evidence_by_id[item]["assertion_level"]
                        != "clinically_qualified"
                        for item in evidence
                    )
                ):
                    raise ValueError(
                        f"{context} qualification does not match source Findings"
                    )
            else:
                if (
                    claim["qualification_receipt_id"] is not None
                    or claim.get("term_decision_receipt_id") is not None
                ):
                    raise ValueError(
                        f"{context} non-qualified claim cannot carry capability or "
                        "term-decision qualification"
                    )
                if (
                    evidence
                    and epistemic != "not_evaluable"
                    and any(
                        evidence_by_id[item]["assertion_level"] != epistemic
                        for item in evidence
                    )
                ):
                    raise ValueError(
                        f"{context} epistemic status does not match source Findings"
                    )
                if epistemic == "not_evaluable" and any(
                    evidence_by_id[item]["status"] != "not_evaluable"
                    for item in evidence
                ):
                    raise ValueError(
                        f"{context} not_evaluable status conflicts with evidence"
                    )
            if (
                predicate == "earliest_sustained_change_maximal_at"
                and claim["time"]["kind"] != "recording_interval"
            ):
                raise ValueError(
                    f"{context} onset observation requires recording-relative time"
                )
            if predicate in _TEMPORAL_RELATION_PREDICATES:
                if (
                    claim["subject"]["type"] != "finding"
                    or len(claim["object_or_value"]["entities"]) != 1
                    or claim["object_or_value"]["entities"][0]["type"] != "finding"
                    or claim["time"]["kind"] != "delay_interval"
                ):
                    raise ValueError(
                        f"{context} temporal relation has malformed endpoints/time"
                    )
        else:
            if epistemic not in {
                "research_ai_hypothesis",
                "risk_controlled_hypothesis",
                "technical_limited",
            }:
                raise ValueError(
                    f"{context} research layer cannot masquerade as observation"
                )
            if (
                claim["qualification_receipt_id"] is not None
                or claim.get("term_decision_receipt_id") is not None
            ):
                raise ValueError(
                    f"{context} research claim cannot carry clinical qualification"
                )
            if kind == "evidence_relation":
                if (
                    predicate not in _RELATION_PREDICATES
                    or claim["hypothesis_id"] is not None
                ):
                    raise ValueError(f"{context} malformed evidence relation")
                if (
                    claim["subject"]["type"] != "claim"
                    or len(claim["object_or_value"]["entities"]) != 1
                    or claim["object_or_value"]["entities"][0]["type"] != "claim"
                    or claim["object_or_value"]["measurements"]
                    or claim["object_or_value"]["code"] is not None
                    or claim["time"]["kind"] != "none"
                    or not evidence
                    or claim["polarity"] != "affirmed"
                ):
                    raise ValueError(f"{context} evidence relation is not claim-closed")
                if (
                    claim["supporting_relation_claim_ids"]
                    or claim["contradictory_relation_claim_ids"]
                ):
                    raise ValueError(
                        f"{context} relation claim cannot declare incoming relations"
                    )
                relation_claims.append(claim)
            else:
                hypothesis_id = claim["hypothesis_id"]
                if hypothesis_id is None:
                    raise ValueError(
                        f"{context} inference claim requires hypothesis_id"
                    )
                _require_refs(
                    (str(hypothesis_id),), hypothesis_ids, f"{context}.hypothesis_id"
                )
                hypothesis = hypothesis_by_id[str(hypothesis_id)]
                expected_kind = {
                    "event": "event_inference",
                    "mode": "mode_inference",
                    "record": "record_hypothesis",
                }[str(hypothesis["scope"])]
                allowed_predicates = {
                    "event": _EVENT_INFERENCE_PREDICATES,
                    "mode": _MODE_INFERENCE_PREDICATES,
                    "record": _RECORD_INFERENCE_PREDICATES,
                }[str(hypothesis["scope"])]
                if kind != expected_kind or predicate not in allowed_predicates:
                    raise ValueError(
                        f"{context} claim kind/predicate conflicts with hypothesis scope"
                    )
                if (
                    claim_event != hypothesis["event_id"]
                    or claim_mode != hypothesis["mode_id"]
                ):
                    raise ValueError(
                        f"{context} event/mode attribution conflicts with hypothesis"
                    )
                expected_subject = {
                    "event": ("eeg_event", str(hypothesis["event_id"])),
                    "mode": ("mode", str(hypothesis["mode_id"])),
                    "record": ("eeg_record", str(payload["record_id"])),
                }[str(hypothesis["scope"])]
                if (
                    claim["subject"]["type"],
                    str(claim["subject"]["id"]),
                ) != expected_subject:
                    raise ValueError(
                        f"{context} subject conflicts with hypothesis scope"
                    )
                if evidence != set(hypothesis["supporting_evidence_ids"]):
                    raise ValueError(
                        f"{context} core claim evidence must match hypothesis support"
                    )
                expected_epistemic = {
                    "passed": "risk_controlled_hypothesis",
                    "backoff_no_risk_calibration": "research_ai_hypothesis",
                    "not_applicable_technical": "technical_limited",
                }[str(hypothesis["risk_control"]["status"])]
                if epistemic != expected_epistemic:
                    raise ValueError(
                        f"{context} epistemic status exceeds hypothesis risk state"
                    )
                if claim["time"]["kind"] != "none":
                    raise ValueError(
                        f"{context} core hypothesis claim must not invent a time"
                    )
                core_claims_by_hypothesis[str(hypothesis_id)].append(claim)

    if set(core_claims_by_hypothesis) != hypothesis_ids or any(
        len(rows) != 1 for rows in core_claims_by_hypothesis.values()
    ):
        raise ValueError("every hypothesis requires exactly one core inference claim")
    for hypothesis_id, hypothesis in hypothesis_by_id.items():
        core = core_claims_by_hypothesis[hypothesis_id][0]
        if core["claim_id"] != hypothesis["core_claim_id"]:
            raise ValueError(f"hypothesis {hypothesis_id!r} core_claim_id mismatch")

    support_edges_by_target: dict[str, set[str]] = defaultdict(set)
    contradiction_edges_by_target: dict[str, set[str]] = defaultdict(set)
    relation_graph_edges: list[tuple[str, str]] = []
    for relation in relation_claims:
        source_id = str(relation["subject"]["id"])
        target_id = str(relation["object_or_value"]["entities"][0]["id"])
        source = claim_by_id[source_id]
        target = claim_by_id[target_id]
        if (
            target["hypothesis_id"] is None
            or target["claim_kind"] == "evidence_relation"
        ):
            raise ValueError("evidence relations must target a core hypothesis claim")
        if source["claim_kind"] == "evidence_relation":
            raise ValueError("evidence relations cannot use another relation as source")
        relation_evidence = set(str(item) for item in relation["evidence_ids"])
        if not relation_evidence.issubset(
            set(str(item) for item in source["evidence_ids"])
        ):
            raise ValueError(
                "evidence relation carries evidence absent from its source claim"
            )
        if (
            relation["event_id"] != source["event_id"]
            or relation["mode_id"] != source["mode_id"]
        ):
            raise ValueError(
                "evidence relation event/mode attribution must match its source"
            )
        if relation["predicate"] == "supports_claim":
            support_edges_by_target[target_id].add(str(relation["claim_id"]))
        else:
            contradiction_edges_by_target[target_id].add(str(relation["claim_id"]))
        relation_graph_edges.append((source_id, target_id))
    _assert_acyclic(relation_graph_edges, "claim evidence-relation graph")

    for claim_id, claim in claim_by_id.items():
        declared_support = set(
            str(item) for item in claim["supporting_relation_claim_ids"]
        )
        declared_contradiction = set(
            str(item) for item in claim["contradictory_relation_claim_ids"]
        )
        _require_refs(
            declared_support | declared_contradiction,
            claim_ids,
            f"claim {claim_id!r} relations",
        )
        if declared_support != support_edges_by_target.get(claim_id, set()):
            raise ValueError(
                f"claim {claim_id!r} supporting relation set is not closed"
            )
        if declared_contradiction != contradiction_edges_by_target.get(claim_id, set()):
            raise ValueError(
                f"claim {claim_id!r} contradictory relation set is not closed"
            )

    for hypothesis_id, hypothesis in hypothesis_by_id.items():
        core = core_claims_by_hypothesis[hypothesis_id][0]
        support_union = {
            evidence_id
            for relation_id in core["supporting_relation_claim_ids"]
            for evidence_id in claim_by_id[str(relation_id)]["evidence_ids"]
        }
        contradiction_union = {
            evidence_id
            for relation_id in core["contradictory_relation_claim_ids"]
            for evidence_id in claim_by_id[str(relation_id)]["evidence_ids"]
        }
        if support_union != set(hypothesis["supporting_evidence_ids"]):
            raise ValueError(
                f"hypothesis {hypothesis_id!r} support relations do not close evidence"
            )
        if contradiction_union != set(hypothesis["contradictory_evidence_ids"]):
            raise ValueError(
                f"hypothesis {hypothesis_id!r} contradiction relations do not close evidence"
            )

    def supporting_leaf_claims(core_claim_id: str) -> list[Mapping[str, Any]]:
        leaves: list[Mapping[str, Any]] = []
        pending = [core_claim_id]
        seen: set[str] = set()
        while pending:
            target_id = pending.pop()
            if target_id in seen:
                continue
            seen.add(target_id)
            for relation_id in claim_by_id[target_id]["supporting_relation_claim_ids"]:
                source_id = str(claim_by_id[str(relation_id)]["subject"]["id"])
                source = claim_by_id[source_id]
                if source["layer"] == "eeg_findings_observation":
                    leaves.append(source)
                else:
                    pending.append(source_id)
        return leaves

    primary_record = next(row for row in record_hypotheses if row["role"] == "primary")
    primary_core_id = str(primary_record["core_claim_id"])
    primary_leaves = supporting_leaf_claims(primary_core_id)
    if primary_record["phenotype"] == "generalized_synchronous" and not any(
        claim["predicate"] == "bilateral_synchronous_evolution_observed"
        for claim in primary_leaves
    ):
        raise ValueError(
            "generalized primary phenotype lacks positive bilateral synchrony evidence"
        )
    if primary_record["phenotype"] == "scalp_onset_nonlocalizable":
        limitation_supported = any(
            claim["predicate"]
            in {"no_stable_focal_lead_observed", "artifact_limits_interpretation"}
            for claim in primary_leaves
        )
        discordant_event_backoff = (
            _DISCORDANT_EVENT_BACKOFF_REASON_CODE
            in set(str(item) for item in primary_record["reason_codes"])
            and len(mode_ids) >= 2
            and len(set(str(item) for item in primary_record["supporting_event_ids"]))
            >= 2
        )
        if not limitation_supported and not discordant_event_backoff:
            raise ValueError(
                "nonlocalizable primary phenotype lacks limitation or typed "
                "cross-event discordance evidence"
            )
    if primary_record["phenotype"] == "multiple_scalp_onset_modes":
        supporting_modes = {
            str(claim["mode_id"])
            for claim in primary_leaves
            if claim["mode_id"] is not None
        }
        if len(mode_ids) < 2 or len(supporting_modes) < 2:
            raise ValueError(
                "multiple-mode phenotype requires evidence from at least two modes"
            )

    plan = payload["sentence_plan"]
    planner_receipt_id = str(plan["planner_receipt_id"])
    _require_refs(
        (planner_receipt_id,), set(producer_by_id), "sentence_plan.planner_receipt_id"
    )
    expected_planner_type = {
        "deterministic_claim_plan": "deterministic_sentence_planner",
        "qwen_sentence_plan_only": "qwen_sentence_planner",
    }[str(plan["planner_mode"])]
    actual_planner_type = str(producer_by_id[planner_receipt_id]["producer_type"])
    if actual_planner_type != expected_planner_type:
        raise ValueError(
            "sentence_plan planner_mode requires producer_type="
            f"{expected_planner_type!r}, got {actual_planner_type!r}"
        )
    expected_qwen_role = {
        "deterministic_claim_plan": "not_used",
        "qwen_sentence_plan_only": "sentence_plan_only",
    }[str(plan["planner_mode"])]
    if payload["report_policy"]["qwen_role"] != expected_qwen_role:
        raise ValueError(
            "report_policy.qwen_role conflicts with sentence_plan.planner_mode"
        )
    mandatory_claims = {
        str(claim["claim_id"]) for claim in claim_rows if claim["mandatory_for_report"]
    }
    if set(str(item) for item in plan["required_claim_ids"]) != mandatory_claims:
        raise ValueError("sentence_plan.required_claim_ids must equal mandatory claims")
    sentence_ids = _unique(
        (str(row["sentence_id"]) for row in plan["sentences"]),
        "sentence_plan.sentences.sentence_id",
    )
    if not sentence_ids:
        raise ValueError("sentence plan must contain at least one sentence")
    coverage: Counter[str] = Counter()
    for index, sentence in enumerate(plan["sentences"]):
        context = f"sentence_plan.sentences[{index}]"
        sentence_claim_ids = [str(item) for item in sentence["claim_ids"]]
        _require_refs(sentence_claim_ids, claim_ids, f"{context}.claim_ids")
        if set(sentence_claim_ids) != set(
            str(item) for item in sentence["claim_order"]
        ):
            raise ValueError(
                f"{context}.claim_order must be an exact permutation of claim_ids"
            )
        for claim_id in sentence_claim_ids:
            coverage[claim_id] += 1
        template_id = str(sentence["template_id"])
        if any(
            template_id not in claim_by_id[claim_id]["allowed_surface_frames"]
            for claim_id in sentence_claim_ids
        ):
            raise ValueError(f"{context}.template_id is not authorized by every claim")
        predicates = {
            str(claim_by_id[item]["predicate"]) for item in sentence_claim_ids
        }
        connectors = set(str(item) for item in sentence["connector_ids"])
        if "then_after_interval" in connectors and not predicates.intersection(
            _TEMPORAL_RELATION_PREDICATES
        ):
            raise ValueError(f"{context} temporal connector lacks a relation claim")
        if "supports" in connectors and "supports_claim" not in predicates:
            raise ValueError(f"{context} supports connector lacks supports_claim")
        if (
            "however_counterevidence" in connectors
            and "contradicts_claim" not in predicates
        ):
            raise ValueError(
                f"{context} counterevidence connector lacks contradicts_claim"
            )
        if "mode_recurrence" in connectors and not predicates.intersection(
            _MODE_INFERENCE_PREDICATES
        ):
            raise ValueError(f"{context} mode connector lacks mode inference")
        nonnull_events = {
            str(claim_by_id[item]["event_id"])
            for item in sentence_claim_ids
            if claim_by_id[item]["event_id"] is not None
        }
        nonnull_modes = {
            str(claim_by_id[item]["mode_id"])
            for item in sentence_claim_ids
            if claim_by_id[item]["mode_id"] is not None
        }
        if sentence["section_id"] == "ictal_findings" and (
            len(nonnull_events) != 1 or len(nonnull_modes) > 1
        ):
            raise ValueError(f"{context} mixes event/mode attribution")
        if sentence["section_id"] == "cross_event_summary" and len(nonnull_modes) > 1:
            raise ValueError(f"{context} mixes distinct modes")
    if any(
        count != 1
        for claim_id, count in coverage.items()
        if claim_id in mandatory_claims
    ):
        raise ValueError("every mandatory claim must be covered exactly once")
    missing_mandatory = mandatory_claims.difference(coverage)
    if missing_mandatory:
        raise ValueError(
            f"sentence plan omits mandatory claims: {sorted(missing_mandatory)}"
        )
    if any(count > 1 for count in coverage.values()):
        raise ValueError("sentence plan cannot duplicate a claim across sentences")

    return payload


__all__ = [
    "MULTIEVENT_SOZ_REPORT_SCHEMA_VERSION",
    "validate_multievent_soz_report_payload",
]
