"""Fail-closed Findings-v1 bridge for three-axis evolution and relative return.

The four outputs in this module close executable *measurement/candidate*
interfaces for the Findings-v1 core queries below without opening a clinical
or report route:

* ``TQ-EVOLUTION-FREQUENCY``;
* ``TQ-EVOLUTION-MORPHOLOGY``;
* ``TQ-EVOLUTION-LOCATION``; and
* ``TQ-POST-EVENT-RETURN-COMPARABLE-BACKGROUND``.

The axes deliberately do not share a positive rule.  Frequency reuses the
conservative ACNS-derived frequency candidate.  Morphology uses an
axis-agnostic change-point proposal only to choose a boundary and then
re-measures amplitude-invariant native-waveform primitives.  Location uses
only interval-valued spatial involvement from the permission-locked
multi-reference field receipt.  Relative return requires an exact technically
comparable, calibrated, matched within-record context comparison and fails
closed when the event is right-censored.

Every result remains ``model_candidate`` (or an unevaluable candidate).  In
particular, amplitude change, spatial distribution change, a generic change
point, and post-event similarity cannot by themselves become ACNS definite
evolution, seizure onset, SOZ, cortical source, recovery, normal background,
or report text.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import math
import re
from typing import Any, Final, Mapping, Sequence

from .acns_frequency_evolution_candidate import (
    ACNSFrequencyEvolutionCandidate,
)
from .ba_ieg_multireference_field import (
    validate_ba_ieg_multireference_field_result,
)
from .deterministic_event_morphology_primitives_v1 import (
    EVENT_MORPHOLOGY_PRIMITIVE_TARGET_NAMES,
    validate_event_morphology_primitive_supervision_v1,
)
from .event_baseline_context_comparability import (
    validate_event_baseline_context_comparability_receipt,
)
from .event_findings_v3_validation import (
    validate_event_eeg_findings_v3_payload,
)


EVENT_EVOLUTION_RECOVERY_QUERY_BRIDGE_SCHEMA_VERSION: Final[str] = (
    "clinical_eeg_event_evolution_recovery_query_bridge_v1"
)
EVENT_EVOLUTION_RECOVERY_QUERY_BRIDGE_ID: Final[str] = (
    "EEG-ONLY-EVENT-EVOLUTION-RECOVERY-QUERY-BRIDGE-V1"
)

TQ_EVOLUTION_FREQUENCY: Final[str] = "TQ-EVOLUTION-FREQUENCY"
TQ_EVOLUTION_MORPHOLOGY: Final[str] = "TQ-EVOLUTION-MORPHOLOGY"
TQ_EVOLUTION_LOCATION: Final[str] = "TQ-EVOLUTION-LOCATION"
TQ_RETURN_COMPARABLE_BACKGROUND: Final[str] = (
    "TQ-POST-EVENT-RETURN-COMPARABLE-BACKGROUND"
)

_QUERY_AXIS = {
    TQ_EVOLUTION_FREQUENCY: "frequency",
    TQ_EVOLUTION_MORPHOLOGY: "morphology",
    TQ_EVOLUTION_LOCATION: "location",
    TQ_RETURN_COMPARABLE_BACKGROUND: "return_to_matched_context",
}
_QUERY_FAMILY = {
    TQ_EVOLUTION_FREQUENCY: "evolution",
    TQ_EVOLUTION_MORPHOLOGY: "evolution",
    TQ_EVOLUTION_LOCATION: "evolution",
    TQ_RETURN_COMPARABLE_BACKGROUND: "termination_recovery",
}
_QUERY_TERM = {
    TQ_EVOLUTION_FREQUENCY: "acns_derived_frequency_evolution_candidate",
    TQ_EVOLUTION_MORPHOLOGY: "sequential_morphology_change_course_candidate",
    TQ_EVOLUTION_LOCATION: "ordered_spatial_involvement_course_candidate",
    TQ_RETURN_COMPARABLE_BACKGROUND: "return_to_comparable_background_candidate",
}
_ALLOWED_STATUSES = frozenset({"present", "uncertain", "not_evaluable"})
_ALLOWED_OPPORTUNITY = frozenset({"sufficient", "limited", "not_evaluable"})
_ALLOWED_TEMPORAL_ROLES = frozenset(
    {"onset_causal", "context_offline", "morphology_native"}
)
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_SHA = re.compile(r"^[0-9a-f]{64}$")
_TOL = 1e-9

# Deliberately gain- and time-scale-normalized morphology summaries.  Raw RMS,
# peak-to-peak, excursions, line length, slope and curvature are never compared
# as change evidence.  Peak-to-peak/width may only normalize slope/curvature;
# therefore a pure gain or uniform time-dilation counterfactual cannot become
# morphology evolution.
_MORPHOLOGY_FEATURE_IDS: Final[tuple[str, ...]] = (
    "rise_share_of_half_height_width",
    "fall_share_of_half_height_width",
    "dominant_excursion_asymmetry_ratio",
    "gain_scale_normalized_rise_slope",
    "gain_scale_normalized_fall_slope",
    "gain_scale_normalized_curvature",
)
_MORPHOLOGY_RAW_TARGETS = {
    "rise_share_of_half_height_width": (
        "dominant_excursion_rise_half_height_seconds",
        "dominant_excursion_half_height_width_seconds",
    ),
    "fall_share_of_half_height_width": (
        "dominant_excursion_fall_half_height_seconds",
        "dominant_excursion_half_height_width_seconds",
    ),
    "dominant_excursion_asymmetry_ratio": (
        "dominant_excursion_asymmetry_ratio",
    ),
    "gain_scale_normalized_rise_slope": (
        "max_rise_slope_uv_per_s",
        "dominant_excursion_half_height_width_seconds",
        "peak_to_peak_uv",
    ),
    "gain_scale_normalized_fall_slope": (
        "max_fall_slope_uv_per_s",
        "dominant_excursion_half_height_width_seconds",
        "peak_to_peak_uv",
    ),
    "gain_scale_normalized_curvature": (
        "max_abs_curvature_uv_per_s2",
        "dominant_excursion_half_height_width_seconds",
        "peak_to_peak_uv",
    ),
}
_AMPLITUDE_TARGETS = frozenset(
    {
        "rms_uv",
        "peak_to_peak_uv",
        "positive_excursion_uv",
        "negative_excursion_uv",
        "line_length_uv",
        "max_rise_slope_uv_per_s",
        "max_fall_slope_uv_per_s",
        "max_abs_curvature_uv_per_s2",
    }
)

_SCOPE_RECEIPT = {
    "eeg_signal_only": True,
    "edf_annotations_used": False,
    "spreadsheet_used": False,
    "doctor_labels_used": False,
    "clinical_text_used": False,
    "patient_metadata_used": False,
    "video_used": False,
    "sleep_staging_used": False,
    "provocation_used": False,
    "report_route_connected": False,
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


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value) or len(value) > 256:
        raise ValueError(f"{name} must be a safe non-empty identifier")
    return value


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _finite(value: object, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum - _TOL:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _interval(
    value: Sequence[object],
    name: str,
    *,
    allow_point: bool = False,
) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{name} must contain two values")
    lower = _finite(value[0], f"{name}[0]", minimum=0.0)
    upper = _finite(value[1], f"{name}[1]", minimum=0.0)
    if upper < lower - _TOL or (not allow_point and upper <= lower + _TOL):
        raise ValueError(f"{name} must be ordered")
    return lower, upper


def _sorted_hashes(values: Sequence[object], name: str) -> list[str]:
    result = sorted({_sha256(item, name) for item in values})
    return result


def _interval_payload(
    interval: tuple[float, float] | None,
    *,
    semantics: str,
    right_censored: bool = False,
) -> dict[str, object] | None:
    if interval is None:
        return None
    return {
        "lower_seconds": interval[0],
        "upper_seconds": interval[1],
        "coordinate_system": "recording_relative_seconds",
        "interval_semantics": semantics,
        "right_censored": bool(right_censored),
    }


def _term_guard(
    query_id: str,
    *,
    frequency_rule_used: bool = False,
    location_distribution_measurement_used: bool = False,
) -> dict[str, object]:
    return {
        "controlled_candidate_term": _QUERY_TERM[query_id],
        "assertion_ceiling": "model_candidate",
        "clinical_term_qualified": False,
        "negative_clinical_assertion_authorized": False,
        "acns_definite_evolution_qualified": False,
        "frequency_rule_used": bool(frequency_rule_used),
        "acns_exact_terminology_used": query_id == TQ_EVOLUTION_FREQUENCY,
        "per_state_three_cycle_opportunity_closed": (
            query_id == TQ_EVOLUTION_FREQUENCY
        ),
        "project_specific_course_candidate": query_id
        in {TQ_EVOLUTION_MORPHOLOGY, TQ_EVOLUTION_LOCATION},
        "amplitude_measurements_used": False,
        "amplitude_scale_used_for_invariance_normalization_only": (
            query_id == TQ_EVOLUTION_MORPHOLOGY
        ),
        "location_distribution_measurement_used": bool(
            location_distribution_measurement_used
        ),
        "amplitude_or_distribution_used_as_acns_definite_evolution": False,
        "background_normality_authorized": False,
        "recovery_or_normalization_authorized": False,
        "onset_support_authorized": False,
        "soz_support_authorized": False,
        "report_promotion_authorized": False,
    }


def _finalize_ledger(
    *,
    query_id: str,
    event_id: str,
    status: str,
    interval: Mapping[str, object] | None,
    opportunity: Mapping[str, object],
    uncertainty: Mapping[str, object],
    temporal_evidence: Mapping[str, object],
    state_sequence: Sequence[Mapping[str, object]],
    transition_instances: Sequence[Mapping[str, object]],
    lineage: Mapping[str, object],
    term_guard: Mapping[str, object],
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": EVENT_EVOLUTION_RECOVERY_QUERY_BRIDGE_SCHEMA_VERSION,
        "bridge_id": EVENT_EVOLUTION_RECOVERY_QUERY_BRIDGE_ID,
        "query_id": query_id,
        "event_id": event_id,
        "family": _QUERY_FAMILY[query_id],
        "axis": _QUERY_AXIS[query_id],
        "assertion_level": "model_candidate",
        "status": status,
        "interval": None if interval is None else deepcopy(dict(interval)),
        "opportunity": deepcopy(dict(opportunity)),
        "uncertainty": deepcopy(dict(uncertainty)),
        "temporal_evidence": deepcopy(dict(temporal_evidence)),
        "state_sequence": [deepcopy(dict(row)) for row in state_sequence],
        "transition_instances": [
            deepcopy(dict(row)) for row in transition_instances
        ],
        "lineage": deepcopy(dict(lineage)),
        "term_guard": deepcopy(dict(term_guard)),
        "scope_receipt": deepcopy(_SCOPE_RECEIPT),
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    body["receipt_sha256"] = _canonical_sha256(body)
    return validate_event_evolution_recovery_query_ledger_v1(body)


def validate_event_evolution_recovery_query_ledger_v1(
    value: object,
) -> dict[str, Any]:
    """Validate the common candidate ledger and its evidence firewalls."""

    required = {
        "schema_version",
        "bridge_id",
        "query_id",
        "event_id",
        "family",
        "axis",
        "assertion_level",
        "status",
        "interval",
        "opportunity",
        "uncertainty",
        "temporal_evidence",
        "state_sequence",
        "transition_instances",
        "lineage",
        "term_guard",
        "scope_receipt",
        "receipt_sha256",
    }
    if type(value) is not dict or set(value) != required:
        raise ValueError("evolution/recovery query ledger fields drifted")
    data = deepcopy(value)
    if data["schema_version"] != EVENT_EVOLUTION_RECOVERY_QUERY_BRIDGE_SCHEMA_VERSION:
        raise ValueError("evolution/recovery bridge schema drifted")
    if data["bridge_id"] != EVENT_EVOLUTION_RECOVERY_QUERY_BRIDGE_ID:
        raise ValueError("evolution/recovery bridge ID drifted")
    query_id = str(data["query_id"])
    if query_id not in _QUERY_AXIS:
        raise ValueError("unsupported Findings-v1 query")
    _identifier(data["event_id"], "event_id")
    if data["axis"] != _QUERY_AXIS[query_id] or data["family"] != _QUERY_FAMILY[query_id]:
        raise ValueError("query axis/family separation drifted")
    if data["assertion_level"] != "model_candidate":
        raise ValueError("query bridge may emit model candidates only")
    if data["status"] not in _ALLOWED_STATUSES:
        raise ValueError("query candidate status is invalid")

    opportunity = data["opportunity"]
    if not isinstance(opportunity, Mapping) or opportunity.get("status") not in _ALLOWED_OPPORTUNITY:
        raise ValueError("query opportunity is invalid")
    reasons = opportunity.get("reason_codes")
    if not isinstance(reasons, list) or reasons != sorted(set(reasons)):
        raise ValueError("opportunity reason codes must be sorted and unique")
    if data["status"] == "present" and opportunity["status"] != "sufficient":
        raise ValueError("present candidate requires sufficient opportunity")
    transitions = data["transition_instances"]
    states = data["state_sequence"]
    if not isinstance(transitions, list) or not isinstance(states, list):
        raise TypeError("states and transitions must be arrays")
    if data["status"] == "present" and not transitions:
        raise ValueError("present candidate requires a transition instance")
    if any(not isinstance(row, Mapping) for row in (*states, *transitions)):
        raise TypeError("state/transition rows must be mappings")

    interval = data["interval"]
    if interval is not None:
        if not isinstance(interval, Mapping):
            raise TypeError("query interval must be an object or null")
        _interval(
            (interval.get("lower_seconds"), interval.get("upper_seconds")),
            "query interval",
            allow_point=True,
        )
        if interval.get("coordinate_system") != "recording_relative_seconds":
            raise ValueError("query interval must use physical recording time")
        if type(interval.get("right_censored")) is not bool:
            raise TypeError("query interval censoring flag must be boolean")

    temporal = data["temporal_evidence"]
    if not isinstance(temporal, Mapping):
        raise TypeError("temporal_evidence must be an object")
    roles = temporal.get("source_temporal_roles")
    if not isinstance(roles, list) or any(role not in _ALLOWED_TEMPORAL_ROLES for role in roles):
        raise ValueError("source temporal roles are invalid")
    for forbidden in (
        "positive_onset_support_authorized",
        "positive_soz_support_authorized",
        "offline_or_future_context_creates_onset",
    ):
        if temporal.get(forbidden) is not False:
            raise ValueError(f"temporal evidence firewall weakened: {forbidden}")

    lineage = data["lineage"]
    if not isinstance(lineage, Mapping):
        raise TypeError("lineage must be an object")
    source_receipts = lineage.get("source_receipt_sha256s")
    if not isinstance(source_receipts, list) or not source_receipts:
        raise ValueError("query ledger requires source receipt lineage")
    if source_receipts != _sorted_hashes(source_receipts, "source receipt"):
        raise ValueError("source receipt lineage must be sorted and unique")
    for key in ("source_binding_sha256s", "raw_dependency_sha256s"):
        hashes = lineage.get(key)
        if not isinstance(hashes, list) or hashes != _sorted_hashes(hashes, key):
            raise ValueError(f"{key} must be sorted SHA-256 values")
    if lineage.get("eeg_signal_only") is not True:
        raise ValueError("query lineage left EEG-only scope")

    guard = data["term_guard"]
    if not isinstance(guard, Mapping):
        raise TypeError("term_guard must be an object")
    required_false = (
        "clinical_term_qualified",
        "negative_clinical_assertion_authorized",
        "acns_definite_evolution_qualified",
        "amplitude_measurements_used",
        "amplitude_or_distribution_used_as_acns_definite_evolution",
        "background_normality_authorized",
        "recovery_or_normalization_authorized",
        "onset_support_authorized",
        "soz_support_authorized",
        "report_promotion_authorized",
    )
    if any(guard.get(key) is not False for key in required_false):
        raise ValueError("clinical/report term guard was weakened")
    if guard.get("assertion_ceiling") != "model_candidate":
        raise ValueError("assertion ceiling drifted")
    if guard.get("controlled_candidate_term") != _QUERY_TERM[query_id]:
        raise ValueError("controlled candidate term drifted")
    if query_id == TQ_EVOLUTION_FREQUENCY:
        if (
            guard.get("acns_exact_terminology_used") is not True
            or guard.get("per_state_three_cycle_opportunity_closed") is not True
            or guard.get("project_specific_course_candidate") is not False
        ):
            raise ValueError("frequency ACNS-derived rule guard drifted")
    elif query_id in {TQ_EVOLUTION_MORPHOLOGY, TQ_EVOLUTION_LOCATION}:
        if (
            guard.get("acns_exact_terminology_used") is not False
            or guard.get("per_state_three_cycle_opportunity_closed") is not False
            or guard.get("project_specific_course_candidate") is not True
        ):
            raise ValueError("project-specific course-candidate guard drifted")
    elif (
        guard.get("acns_exact_terminology_used") is not False
        or guard.get("per_state_three_cycle_opportunity_closed") is not False
        or guard.get("project_specific_course_candidate") is not False
    ):
        raise ValueError("relative-return terminology guard drifted")
    if data["scope_receipt"] != _SCOPE_RECEIPT:
        raise ValueError("EEG-only scope receipt drifted")

    # Axis-specific leakage guards.  Morphology may use only the explicitly
    # amplitude-invariant feature vocabulary.  Location never exports source
    # model scores, so a distribution transition cannot be misread as ACNS
    # definite evolution or cross-reference probability fusion.
    if query_id == TQ_EVOLUTION_MORPHOLOGY:
        for transition in transitions:
            if transition.get("acns_definite_evolution_step_qualified") is not False:
                raise ValueError("morphology step was promoted to ACNS definite evolution")
            changes = transition.get("feature_changes", [])
            if any(row.get("feature_id") not in _MORPHOLOGY_FEATURE_IDS for row in changes):
                raise ValueError("morphology transition used an unregistered feature")
            serialized = json.dumps(changes, sort_keys=True)
            if any(name in serialized for name in _AMPLITUDE_TARGETS):
                raise ValueError("amplitude leaked into morphology evolution")
    if query_id == TQ_EVOLUTION_LOCATION:
        serialized = json.dumps(transitions, sort_keys=True)
        if "onset_association_score" in serialized or "amplitude" in serialized:
            raise ValueError("score/amplitude leaked into location evolution")
        if any(
            row.get("acns_definite_evolution_step_qualified") is not False
            for row in transitions
        ):
            raise ValueError("location step was promoted to ACNS definite evolution")
    if query_id == TQ_EVOLUTION_FREQUENCY and any(
        row.get("acns_definite_evolution_step_qualified") is not False
        for row in transitions
    ):
        raise ValueError("frequency step was promoted to ACNS definite evolution")
    if query_id == TQ_RETURN_COMPARABLE_BACKGROUND and any(
        row.get("clinical_return_or_recovery_qualified") is not False
        for row in transitions
    ):
        raise ValueError("relative-return candidate became a clinical recovery claim")

    _sha256(data["receipt_sha256"], "receipt_sha256")
    expected = _canonical_sha256(
        {**data, "receipt_sha256": "CONTENT-ADDRESS-PENDING"}
    )
    if data["receipt_sha256"] != expected:
        raise ValueError("query ledger content hash drifted")
    return data


@dataclass(frozen=True)
class EventChangePointProposal:
    """Axis-agnostic, signal-only transition-boundary proposal."""

    proposal_id: str
    event_id: str
    source_signal_sha256: str
    change_interval_seconds: tuple[float, float]
    proposal_status: str
    source_receipt_sha256: str
    source_evidence_id: str
    source_temporal_role: str
    future_sample_access: bool
    boundary_resolution_seconds: float
    source_binding_sha256s: tuple[str, ...]
    raw_dependency_sha256s: tuple[str, ...]
    proposal_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("proposal_id", "event_id", "source_evidence_id"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        for name in ("source_signal_sha256", "source_receipt_sha256"):
            _sha256(getattr(self, name), name)
        object.__setattr__(
            self,
            "change_interval_seconds",
            _interval(self.change_interval_seconds, "change_interval_seconds"),
        )
        if self.proposal_status not in {"present", "uncertain", "not_evaluable"}:
            raise ValueError("change-point proposal status is invalid")
        if self.source_temporal_role not in {"onset_causal", "context_offline"}:
            raise ValueError("change-point temporal role is invalid")
        if type(self.future_sample_access) is not bool:
            raise TypeError("future_sample_access must be boolean")
        if self.source_temporal_role == "onset_causal" and self.future_sample_access:
            raise ValueError("onset-causal change point cannot use future samples")
        if self.source_temporal_role == "context_offline" and not self.future_sample_access:
            raise ValueError("offline change point must disclose future access")
        object.__setattr__(
            self,
            "boundary_resolution_seconds",
            _finite(
                self.boundary_resolution_seconds,
                "boundary_resolution_seconds",
                minimum=_TOL,
            ),
        )
        bindings = tuple(
            _sorted_hashes(self.source_binding_sha256s, "source binding")
        )
        if not bindings:
            raise ValueError("change point needs a source binding")
        raw = tuple(_sorted_hashes(self.raw_dependency_sha256s, "raw dependency"))
        object.__setattr__(self, "source_binding_sha256s", bindings)
        object.__setattr__(self, "raw_dependency_sha256s", raw)
        object.__setattr__(self, "proposal_sha256", _canonical_sha256(self._body()))

    def _body(self) -> dict[str, object]:
        return {
            "proposal_id": self.proposal_id,
            "event_id": self.event_id,
            "source_signal_sha256": self.source_signal_sha256,
            "change_interval_seconds": list(self.change_interval_seconds),
            "proposal_status": self.proposal_status,
            "proposal_scope": "axis_agnostic_transition_boundary_only",
            "source_receipt_sha256": self.source_receipt_sha256,
            "source_evidence_id": self.source_evidence_id,
            "source_temporal_role": self.source_temporal_role,
            "future_sample_access": self.future_sample_access,
            "boundary_resolution_seconds": self.boundary_resolution_seconds,
            "source_binding_sha256s": list(self.source_binding_sha256s),
            "raw_dependency_sha256s": list(self.raw_dependency_sha256s),
            "clinical_term_authorized": False,
            "axis_claim_authorized": False,
            "report_text_authorized": False,
        }

    def to_dict(self) -> dict[str, object]:
        result = self._body()
        result["proposal_sha256"] = self.proposal_sha256
        return result


def extract_axis_agnostic_change_point_proposal_from_v3(
    event_findings_v3: object,
    *,
    evidence_id: str = "E-ONSET-CHANGE",
    **trusted_validation_context: Any,
) -> EventChangePointProposal:
    """Bind the registered multifeature change candidate as a boundary only."""

    payload = validate_event_eeg_findings_v3_payload(
        event_findings_v3, **trusted_validation_context
    )
    evidence_id = _identifier(evidence_id, "evidence_id")
    matches = [
        row for row in payload["findings"] if row["evidence_id"] == evidence_id
    ]
    if len(matches) != 1:
        raise ValueError("change-point evidence ID must identify exactly one Finding")
    finding = matches[0]
    if (
        finding["family"] != "evolution"
        or finding["term"]["term_id"]
        != "deterministic_multifeature_change_point_candidate"
    ):
        raise ValueError("source Finding is not the registered change-point candidate")
    span = finding["time_interval"]
    if not isinstance(span, Mapping):
        raise ValueError("change-point Finding has no physical interval")
    status = str(finding["status"])
    proposal_status = "present" if status == "present" else "uncertain"

    roles: set[str] = set()
    future_flags: list[bool] = []
    binding_hashes: list[str] = []
    raw_hashes: list[str] = []
    for measurement in finding["measurements"]:
        binding = measurement["source_binding"]
        roles.add(str(binding["view_role"]))
        binding_hashes.append(_canonical_sha256(binding))
        raw = binding.get("raw_sample_dependency")
        if isinstance(raw, Mapping):
            future_flags.append(bool(raw["future_sample_access"]))
            raw_hashes.append(str(raw["dependency_sha256"]))
    if len(roles) != 1 or next(iter(roles)) not in _ALLOWED_TEMPORAL_ROLES:
        raise ValueError("change-point source must have one supported temporal role")
    if not binding_hashes:
        raise ValueError("change-point Finding has no replayable measurements")
    role = next(iter(roles))
    future_access = any(future_flags)
    source_receipt_sha256 = _canonical_sha256(payload)
    return EventChangePointProposal(
        proposal_id="CHGPT-" + _canonical_sha256(
            {"source_receipt_sha256": source_receipt_sha256, "evidence_id": evidence_id}
        )[:24],
        event_id=str(payload["event_id"]),
        source_signal_sha256=str(payload["provenance"]["canonical_signal_sha256"]),
        change_interval_seconds=(float(span["start"]), float(span["stop"])),
        proposal_status=proposal_status,
        source_receipt_sha256=source_receipt_sha256,
        source_evidence_id=evidence_id,
        source_temporal_role=role,
        future_sample_access=future_access,
        boundary_resolution_seconds=float(span["resolution_seconds"]),
        source_binding_sha256s=tuple(binding_hashes),
        raw_dependency_sha256s=tuple(raw_hashes),
    )


def compose_frequency_evolution_query_ledger_v1(
    candidate: ACNSFrequencyEvolutionCandidate,
) -> dict[str, Any]:
    """Bridge a validated ACNS-derived frequency candidate into one query ledger."""

    if not isinstance(candidate, ACNSFrequencyEvolutionCandidate):
        raise TypeError("candidate must be ACNSFrequencyEvolutionCandidate")
    payload = candidate.to_dict()
    if candidate.receipt_sha256 != _canonical_sha256(candidate.to_dict(False)):
        raise ValueError("frequency candidate content changed after registration")
    states = [deepcopy(dict(row)) for row in payload["states"]]
    transitions = [deepcopy(dict(row)) for row in payload["transitions"]]
    roles = sorted({str(row["source_temporal_role"]) for row in states})
    future_access = any(bool(row["future_sample_access"]) for row in states)
    evaluable = [row for row in states if row["evaluation_opportunity"]]

    selected_indices = list(payload["selected_transition_indices"])
    selected_transitions: list[dict[str, object]] = []
    for index in selected_indices:
        transition = transitions[index]
        left = states[index]
        right = states[index + 1]
        selected_transitions.append(
            {
                "transition_id": f"FREQ-TRANSITION-{index:04d}",
                "axis": "frequency",
                "from_state_id": left["state_id"],
                "to_state_id": right["state_id"],
                "from_state_interval_seconds": left["recording_interval_seconds"],
                "to_state_interval_seconds": right["recording_interval_seconds"],
                "change_interval_seconds": [
                    left["recording_interval_seconds"][1],
                    right["recording_interval_seconds"][0],
                ],
                "from_frequency_uncertainty_interval_hz": left[
                    "frequency_uncertainty_interval_hz"
                ],
                "to_frequency_uncertainty_interval_hz": right[
                    "frequency_uncertainty_interval_hz"
                ],
                "direction": transition["direction"],
                "conservative_frequency_step_hz": transition[
                    "conservative_frequency_step_hz"
                ],
                "rule_passed": bool(transition["eligible_change"]),
                "acns_definite_evolution_step_qualified": False,
                "reason_codes": list(transition["reason_codes"]),
            }
        )

    if payload["status"] == "present":
        selected_ids = set(payload["selected_state_ids"])
        selected_states = [row for row in states if row["state_id"] in selected_ids]
        course = (
            min(row["recording_interval_seconds"][0] for row in selected_states),
            max(row["recording_interval_seconds"][1] for row in selected_states),
        )
        opportunity_status = "sufficient"
        reasons: list[str] = []
    elif payload["status"] == "not_evaluable":
        course = None
        opportunity_status = "not_evaluable"
        reasons = list(payload["reason_codes"])
    else:
        course = (
            min(row["recording_interval_seconds"][0] for row in states),
            max(row["recording_interval_seconds"][1] for row in states),
        )
        opportunity_status = "limited"
        reasons = list(payload["reason_codes"])

    return _finalize_ledger(
        query_id=TQ_EVOLUTION_FREQUENCY,
        event_id=candidate.event_id,
        status=str(payload["status"]),
        interval=_interval_payload(course, semantics="selected_frequency_state_course"),
        opportunity={
            "status": opportunity_status,
            "evaluable_state_count": len(evaluable),
            "total_state_count": len(states),
            "selected_transition_count": len(selected_transitions),
            "reason_codes": sorted(set(reasons)),
        },
        uncertainty={
            "status": "interval_propagated",
            "frequency_uncertainty_propagated": True,
            "boundary_uncertainty_semantics": (
                "interstate_physical_interval_and_state_frequency_intervals"
            ),
            "calibration_status": "target_domain_not_qualified",
            "reason_codes": sorted(set(payload["reason_codes"])),
        },
        temporal_evidence={
            "intrinsic_evidence_role": "course_only",
            "source_temporal_roles": roles,
            "future_sample_access_present": future_access,
            "causal_evidence_role": "may_anchor_observed_course_only",
            "offline_evidence_role": "may_describe_course_not_create_onset",
            "positive_onset_support_authorized": False,
            "positive_soz_support_authorized": False,
            "offline_or_future_context_creates_onset": False,
        },
        state_sequence=states,
        transition_instances=selected_transitions,
        lineage={
            "source_receipt_sha256s": sorted(
                {candidate.receipt_sha256, candidate.policy_sha256}
            ),
            "source_binding_sha256s": [candidate.source_binding_sha256],
            "raw_dependency_sha256s": list(candidate.raw_dependency_sha256s),
            "source_state_measurement_sha256s": sorted(
                str(row["measurement_sha256"]) for row in states
            ),
            "eeg_signal_only": True,
        },
        term_guard=_term_guard(
            TQ_EVOLUTION_FREQUENCY, frequency_rule_used=True
        ),
    )


@dataclass(frozen=True)
class MorphologyEvolutionPolicy:
    """Engineering gate on amplitude-invariant primitive changes."""

    minimum_measured_features_per_pair: int = 4
    minimum_changed_features_per_pair: int = 2
    minimum_symmetric_relative_effect: float = 0.20

    def __post_init__(self) -> None:
        if (
            isinstance(self.minimum_measured_features_per_pair, bool)
            or not isinstance(self.minimum_measured_features_per_pair, int)
            or self.minimum_measured_features_per_pair < 2
        ):
            raise ValueError("minimum_measured_features_per_pair must be >= 2")
        if (
            isinstance(self.minimum_changed_features_per_pair, bool)
            or not isinstance(self.minimum_changed_features_per_pair, int)
            or self.minimum_changed_features_per_pair < 1
            or self.minimum_changed_features_per_pair
            > self.minimum_measured_features_per_pair
        ):
            raise ValueError("changed-feature requirement is invalid")
        effect = _finite(
            self.minimum_symmetric_relative_effect,
            "minimum_symmetric_relative_effect",
            minimum=_TOL,
        )
        if effect > 1.0 + _TOL:
            raise ValueError("minimum_symmetric_relative_effect must be <= 1")
        object.__setattr__(self, "minimum_symmetric_relative_effect", effect)

    def to_dict(self) -> dict[str, object]:
        return {
            "policy_id": "AMPLITUDE-INVARIANT-MORPHOLOGY-TRANSITION-GATE-V1",
            "minimum_measured_features_per_pair": self.minimum_measured_features_per_pair,
            "minimum_changed_features_per_pair": self.minimum_changed_features_per_pair,
            "minimum_symmetric_relative_effect": self.minimum_symmetric_relative_effect,
            "eligible_feature_ids": list(_MORPHOLOGY_FEATURE_IDS),
            "amplitude_targets_explicitly_excluded": sorted(_AMPLITUDE_TARGETS),
            "clinical_threshold_interpretation": False,
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


DEFAULT_MORPHOLOGY_EVOLUTION_POLICY = MorphologyEvolutionPolicy()


def _morphology_feature_values(row: Mapping[str, Any]) -> dict[str, float | None]:
    names = EVENT_MORPHOLOGY_PRIMITIVE_TARGET_NAMES
    values = dict(zip(names, row["values"]))
    masks = dict(zip(names, row["opportunity"]["target_value_mask"]))

    def measured(*targets: str) -> bool:
        return all(bool(masks[name]) for name in targets)

    result: dict[str, float | None] = {}
    for feature_id, targets in _MORPHOLOGY_RAW_TARGETS.items():
        if not measured(*targets):
            result[feature_id] = None
            continue
        width = float(values["dominant_excursion_half_height_width_seconds"])
        peak_to_peak = float(values["peak_to_peak_uv"])
        if (
            feature_id
            in {
                "rise_share_of_half_height_width",
                "fall_share_of_half_height_width",
                "gain_scale_normalized_rise_slope",
                "gain_scale_normalized_fall_slope",
                "gain_scale_normalized_curvature",
            }
            and width <= _TOL
        ):
            result[feature_id] = None
            continue
        if (
            feature_id
            in {
                "gain_scale_normalized_rise_slope",
                "gain_scale_normalized_fall_slope",
                "gain_scale_normalized_curvature",
            }
            and peak_to_peak <= _TOL
        ):
            result[feature_id] = None
            continue
        if feature_id == "rise_share_of_half_height_width":
            result[feature_id] = float(
                values["dominant_excursion_rise_half_height_seconds"]
            ) / width
        elif feature_id == "fall_share_of_half_height_width":
            result[feature_id] = float(
                values["dominant_excursion_fall_half_height_seconds"]
            ) / width
        elif feature_id == "gain_scale_normalized_rise_slope":
            result[feature_id] = (
                float(values["max_rise_slope_uv_per_s"])
                * width
                / peak_to_peak
            )
        elif feature_id == "gain_scale_normalized_fall_slope":
            result[feature_id] = (
                float(values["max_fall_slope_uv_per_s"])
                * width
                / peak_to_peak
            )
        elif feature_id == "gain_scale_normalized_curvature":
            result[feature_id] = (
                float(values["max_abs_curvature_uv_per_s2"])
                * width
                * width
                / peak_to_peak
            )
        else:
            result[feature_id] = float(values[targets[0]])
    return result


def _relative_effect(feature_id: str, left: float, right: float) -> float:
    if feature_id == "dominant_excursion_asymmetry_ratio":
        floor = 0.10
    else:
        floor = 0.05
    return min(1.0, abs(right - left) / max(abs(left), abs(right), floor))


def compose_morphology_evolution_query_ledger_v1(
    morphology_receipt: object,
    *,
    change_points: Sequence[EventChangePointProposal] | None = None,
    change_point: EventChangePointProposal | None = None,
    policy: MorphologyEvolutionPolicy = DEFAULT_MORPHOLOGY_EVOLUTION_POLICY,
) -> dict[str, Any]:
    """Re-measure sequential boundaries using normalized waveform shape.

    One change may be retained as an uncertain change candidate.  An explicit
    empty proposal roster is retained as a typed ``not_evaluable`` result; it
    is never converted to absence and cannot abort the enclosing Findings
    closure.  The frozen project-specific course candidate requires at least two sequential,
    rule-passing changes over three states for the same view/unit, but remains
    explicitly below ACNS exact terminology because per-state cycle support is
    not present in the morphology sidecar.
    ``change_point`` is a backwards-compatible single-proposal shorthand and
    therefore cannot by itself yield ``present``.
    """

    receipt = validate_event_morphology_primitive_supervision_v1(
        morphology_receipt
    )
    if not isinstance(policy, MorphologyEvolutionPolicy):
        raise TypeError("policy must be MorphologyEvolutionPolicy")
    if change_points is not None and change_point is not None:
        raise ValueError("supply change_points or change_point, not both")
    if change_points is None:
        if change_point is None:
            raise ValueError("at least one change-point proposal is required")
        points = (change_point,)
    else:
        points = tuple(change_points)
    if any(not isinstance(item, EventChangePointProposal) for item in points):
        raise TypeError(
            "change_points must be an EventChangePointProposal sequence"
        )
    if not points:
        reasons = [
            "no_absence_inference_from_empty_change_point_roster",
            "no_morphology_change_point_proposal",
        ]
        return _finalize_ledger(
            query_id=TQ_EVOLUTION_MORPHOLOGY,
            event_id=str(receipt["event_id"]),
            status="not_evaluable",
            interval=None,
            opportunity={
                "status": "not_evaluable",
                "same_view_unit_pair_count": 0,
                "evaluable_pair_count": 0,
                "proposal_count": 0,
                "maximum_sequential_change_count": 0,
                "minimum_required_sequential_change_count": 2,
                "qualifying_view_unit_chain_keys": [],
                "amplitude_invariant_feature_count": len(
                    _MORPHOLOGY_FEATURE_IDS
                ),
                "reason_codes": reasons,
            },
            uncertainty={
                "status": "change_point_proposal_roster_empty",
                "change_intervals_seconds": [],
                "boundary_resolution_seconds": [],
                "proposal_statuses": [],
                "measurement_uncertainty": (
                    "not_evaluable_without_change_point_proposal"
                ),
                "calibration_status": "not_clinically_qualified",
                "reason_codes": reasons,
            },
            temporal_evidence={
                "intrinsic_evidence_role": "course_only",
                "source_temporal_roles": ["morphology_native"],
                "future_sample_access_present": False,
                "causal_evidence_role": "none_without_change_point_proposal",
                "offline_evidence_role": (
                    "no_course_inference_from_empty_change_point_roster"
                ),
                "positive_onset_support_authorized": False,
                "positive_soz_support_authorized": False,
                "offline_or_future_context_creates_onset": False,
            },
            state_sequence=[],
            transition_instances=[],
            lineage={
                "source_receipt_sha256s": sorted(
                    {str(receipt["receipt_sha256"]), policy.sha256}
                ),
                "source_binding_sha256s": sorted(
                    {
                        str(row["source_binding_sha256"])
                        for row in receipt["rows"]
                    }
                ),
                "raw_dependency_sha256s": [],
                "source_row_binding_sha256s": sorted(
                    {str(row["row_binding_sha256"]) for row in receipt["rows"]}
                ),
                "change_point_proposal_ids": [],
                "eeg_signal_only": True,
            },
            term_guard=_term_guard(TQ_EVOLUTION_MORPHOLOGY),
        )
    if len({item.proposal_id for item in points}) != len(points):
        raise ValueError("change-point proposal IDs must be unique")
    points = tuple(
        sorted(
            points,
            key=lambda item: (
                item.change_interval_seconds[0],
                item.change_interval_seconds[1],
                item.proposal_id,
            ),
        )
    )
    event_ids = {item.event_id for item in points}
    source_signals = {item.source_signal_sha256 for item in points}
    if event_ids != {str(receipt["event_id"])}:
        raise ValueError(
            "morphology receipt and change points belong to different events"
        )
    if source_signals != {str(receipt["source_signal_sha256"])}:
        raise ValueError(
            "morphology receipt and change points cross canonical signals"
        )
    analysis = _interval(receipt["analysis_interval_seconds"], "analysis interval")
    for index, point in enumerate(points):
        lower, upper = point.change_interval_seconds
        if lower < analysis[0] - _TOL or upper > analysis[1] + _TOL:
            raise ValueError("change point lies outside morphology analysis interval")
        if (
            index > 0
            and lower < points[index - 1].change_interval_seconds[1] - _TOL
        ):
            raise ValueError("change-point intervals must not overlap")

    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in receipt["rows"]:
        source = row["source_binding"]
        key = (str(source["view_id"]), str(source["unit_id"]))
        groups.setdefault(key, []).append(row)

    pair_rows: list[
        tuple[
            int,
            EventChangePointProposal,
            tuple[str, str],
            Mapping[str, Any],
            Mapping[str, Any],
        ]
    ] = []
    for proposal_index, point in enumerate(points):
        proposal_lower, proposal_upper = point.change_interval_seconds
        for key, rows in groups.items():
            left = [
                row
                for row in rows
                if float(row["source_binding"]["recording_interval_seconds"][1])
                <= proposal_lower + _TOL
            ]
            right = [
                row
                for row in rows
                if float(row["source_binding"]["recording_interval_seconds"][0])
                >= proposal_upper - _TOL
            ]
            if left and right:
                pair_rows.append(
                    (
                        proposal_index,
                        point,
                        key,
                        max(
                            left,
                            key=lambda row: (
                                float(
                                    row["source_binding"][
                                        "recording_interval_seconds"
                                    ][1]
                                ),
                                str(row["row_id"]),
                            ),
                        ),
                        min(
                            right,
                            key=lambda row: (
                                float(
                                    row["source_binding"][
                                        "recording_interval_seconds"
                                    ][0]
                                ),
                                str(row["row_id"]),
                            ),
                        ),
                    )
                )

    transitions: list[dict[str, object]] = []
    evaluable_pair_count = 0
    state_sequence: list[dict[str, object]] = []
    chain_entries: dict[
        tuple[str, str], list[tuple[int, str, str, bool]]
    ] = {}
    ordered_pairs = sorted(
        pair_rows,
        key=lambda item: (item[0], item[2][0], item[2][1], str(item[3]["row_id"])),
    )
    for ordinal, (proposal_index, point, key, left, right) in enumerate(
        ordered_pairs
    ):
        left_values = _morphology_feature_values(left)
        right_values = _morphology_feature_values(right)
        changes: list[dict[str, object]] = []
        measured_count = 0
        changed_count = 0
        for feature_id in _MORPHOLOGY_FEATURE_IDS:
            left_value = left_values[feature_id]
            right_value = right_values[feature_id]
            if left_value is None or right_value is None:
                changes.append(
                    {
                        "feature_id": feature_id,
                        "left_value": left_value,
                        "right_value": right_value,
                        "symmetric_relative_effect": None,
                        "changed_under_policy": False,
                        "evaluation_opportunity": False,
                    }
                )
                continue
            measured_count += 1
            effect = _relative_effect(feature_id, left_value, right_value)
            changed = effect + _TOL >= policy.minimum_symmetric_relative_effect
            changed_count += int(changed)
            changes.append(
                {
                    "feature_id": feature_id,
                    "left_value": left_value,
                    "right_value": right_value,
                    "symmetric_relative_effect": effect,
                    "changed_under_policy": changed,
                    "evaluation_opportunity": True,
                }
            )
        opportunity = measured_count >= policy.minimum_measured_features_per_pair
        rule_passed = bool(
            opportunity
            and changed_count >= policy.minimum_changed_features_per_pair
            and point.proposal_status == "present"
        )
        evaluable_pair_count += int(opportunity)
        chain_entries.setdefault(key, []).append(
            (
                proposal_index,
                str(left["row_id"]),
                str(right["row_id"]),
                rule_passed,
            )
        )
        left_interval = list(left["source_binding"]["recording_interval_seconds"])
        right_interval = list(right["source_binding"]["recording_interval_seconds"])
        state_sequence.extend(
            [
                {
                    "state_id": str(left["row_id"]),
                    "axis": "morphology",
                    "recording_interval_seconds": left_interval,
                    "view_id": left["source_binding"]["view_id"],
                    "unit_id": left["source_binding"]["unit_id"],
                    "row_binding_sha256": left["row_binding_sha256"],
                },
                {
                    "state_id": str(right["row_id"]),
                    "axis": "morphology",
                    "recording_interval_seconds": right_interval,
                    "view_id": right["source_binding"]["view_id"],
                    "unit_id": right["source_binding"]["unit_id"],
                    "row_binding_sha256": right["row_binding_sha256"],
                },
            ]
        )
        reason_codes: list[str] = []
        if not opportunity:
            reason_codes.append("insufficient_measured_amplitude_invariant_features")
        if opportunity and changed_count < policy.minimum_changed_features_per_pair:
            reason_codes.append("morphology_change_rule_not_met")
        if point.proposal_status != "present":
            reason_codes.append("change_point_proposal_not_present")
        transitions.append(
            {
                "transition_id": f"MORPH-TRANSITION-{ordinal:04d}",
                "axis": "morphology",
                "proposal_id": point.proposal_id,
                "proposal_index": proposal_index,
                "from_state_id": left["row_id"],
                "to_state_id": right["row_id"],
                "from_state_interval_seconds": left_interval,
                "to_state_interval_seconds": right_interval,
                "change_interval_seconds": list(point.change_interval_seconds),
                "feature_changes": changes,
                "measured_feature_count": measured_count,
                "changed_feature_count": changed_count,
                "rule_passed": rule_passed,
                "acns_definite_evolution_step_qualified": False,
                "reason_codes": sorted(set(reason_codes)),
            }
        )

    # Count only contiguous, shared-intermediate-state transition chains.
    maximum_sequential_change_count = 0
    qualifying_chain_keys: list[str] = []
    for key, entries in sorted(chain_entries.items()):
        entries.sort(key=lambda item: item[0])
        current = 0
        maximum = 0
        previous: tuple[int, str, str, bool] | None = None
        for entry in entries:
            if not entry[3]:
                current = 0
            elif (
                previous is not None
                and previous[3]
                and entry[0] == previous[0] + 1
                and entry[1] == previous[2]
            ):
                current += 1
            else:
                current = 1
            maximum = max(maximum, current)
            previous = entry
        maximum_sequential_change_count = max(
            maximum_sequential_change_count, maximum
        )
        if maximum >= 2:
            qualifying_chain_keys.append(f"{key[0]}::{key[1]}")

    # De-duplicate state references while retaining physical-time order.
    state_by_id = {str(row["state_id"]): row for row in state_sequence}
    states = sorted(
        state_by_id.values(),
        key=lambda row: (
            row["recording_interval_seconds"],
            row["view_id"],
            row["unit_id"],
            row["state_id"],
        ),
    )
    if maximum_sequential_change_count >= 2:
        status = "present"
        opportunity_status = "sufficient"
        reasons: list[str] = []
    elif not pair_rows or evaluable_pair_count == 0:
        status = "not_evaluable"
        opportunity_status = "not_evaluable"
        reasons = [
            "no_evaluable_same_view_unit_states_flanking_change_points"
        ]
    else:
        status = "uncertain"
        opportunity_status = "sufficient"
        reasons = [
            "morphology_candidate_rule_not_met_or_boundary_uncertain",
            "fewer_than_two_sequential_morphology_changes",
            "no_sensitivity_receipt_for_absent_morphology_evolution",
        ]

    source_binding_hashes = [
        str(row["source_binding_sha256"])
        for _, _, _, left, right in pair_rows
        for row in (left, right)
    ] + [
        digest for point in points for digest in point.source_binding_sha256s
    ]
    raw_hashes = [
        digest for point in points for digest in point.raw_dependency_sha256s
    ]
    course = (
        points[0].change_interval_seconds[0],
        points[-1].change_interval_seconds[1],
    )
    return _finalize_ledger(
        query_id=TQ_EVOLUTION_MORPHOLOGY,
        event_id=points[0].event_id,
        status=status,
        interval=_interval_payload(
            course,
            semantics=(
                "sequential_axis_agnostic_boundaries_remeasured_for_morphology"
            ),
        ),
        opportunity={
            "status": opportunity_status,
            "same_view_unit_pair_count": len(pair_rows),
            "evaluable_pair_count": evaluable_pair_count,
            "proposal_count": len(points),
            "maximum_sequential_change_count": maximum_sequential_change_count,
            "minimum_required_sequential_change_count": 2,
            "qualifying_view_unit_chain_keys": qualifying_chain_keys,
            "amplitude_invariant_feature_count": len(_MORPHOLOGY_FEATURE_IDS),
            "reason_codes": sorted(set(reasons)),
        },
        uncertainty={
            "status": "ordered_boundary_intervals_plus_deterministic_measurement",
            "change_intervals_seconds": [
                list(point.change_interval_seconds) for point in points
            ],
            "boundary_resolution_seconds": [
                point.boundary_resolution_seconds for point in points
            ],
            "proposal_statuses": [point.proposal_status for point in points],
            "measurement_uncertainty": (
                "deterministic_point_measurements_no_sampling_uncertainty_model"
            ),
            "calibration_status": "not_clinically_qualified",
            "reason_codes": sorted(set(reasons)),
        },
        temporal_evidence={
            "intrinsic_evidence_role": "course_only",
            "source_temporal_roles": sorted(
                {
                    "morphology_native",
                    *(point.source_temporal_role for point in points),
                }
            ),
            "future_sample_access_present": any(
                point.future_sample_access for point in points
            ),
            "causal_evidence_role": "boundary_proposal_and_native_shape_remeasurement",
            "offline_evidence_role": "may_describe_course_not_create_onset",
            "positive_onset_support_authorized": False,
            "positive_soz_support_authorized": False,
            "offline_or_future_context_creates_onset": False,
        },
        state_sequence=states,
        transition_instances=transitions,
        lineage={
            "source_receipt_sha256s": sorted(
                {
                    str(receipt["receipt_sha256"]),
                    policy.sha256,
                    *(point.source_receipt_sha256 for point in points),
                    *(point.proposal_sha256 for point in points),
                }
            ),
            "source_binding_sha256s": _sorted_hashes(
                source_binding_hashes, "source binding"
            ),
            "raw_dependency_sha256s": _sorted_hashes(
                raw_hashes, "raw dependency"
            ),
            "source_row_binding_sha256s": sorted(
                {
                    str(row["row_binding_sha256"])
                    for _, _, _, left, right in pair_rows
                    for row in (left, right)
                }
            ),
            "change_point_proposal_ids": [point.proposal_id for point in points],
            "eeg_signal_only": True,
        },
        term_guard=_term_guard(TQ_EVOLUTION_MORPHOLOGY),
    )


def _overlap_clusters(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            float(row["onset_interval_envelope_seconds"][0]),
            float(row["onset_interval_envelope_seconds"][1]),
            str(row["spatial_key"]),
        ),
    )
    clusters: list[dict[str, Any]] = []
    for row in ordered:
        lower, upper = _interval(
            row["onset_interval_envelope_seconds"],
            "location onset envelope",
        )
        if not clusters or lower > float(clusters[-1]["upper_seconds"]) + _TOL:
            clusters.append(
                {
                    "lower_seconds": lower,
                    "upper_seconds": upper,
                    "rows": [row],
                }
            )
        else:
            clusters[-1]["upper_seconds"] = max(
                float(clusters[-1]["upper_seconds"]), upper
            )
            clusters[-1]["rows"].append(row)
    return clusters


def compose_location_evolution_query_ledger_v1(
    multireference_field_result: object,
) -> dict[str, Any]:
    """Compose interval-ordered spatial involvement without score fusion."""

    field_result = validate_ba_ieg_multireference_field_result(
        multireference_field_result
    )
    resolution = str(field_result["selected_resolution"])
    ranking = list(field_result["selected_candidate_ranking"])
    usable_rows = [
        row for row in ranking if row["onset_interval_envelope_seconds"] is not None
    ]
    clusters = _overlap_clusters(usable_rows) if usable_rows else []
    states: list[dict[str, object]] = []
    transitions: list[dict[str, object]] = []
    active: list[str] = []
    for index, cluster in enumerate(clusters):
        newly = sorted(str(row["spatial_key"]) for row in cluster["rows"])
        active = sorted(set(active).union(newly))
        source_ids = sorted(
            {
                str(source_id)
                for row in cluster["rows"]
                for source_id in row["source_candidate_ids"]
            }
        )
        states.append(
            {
                "state_id": f"LOCATION-STATE-{index:04d}",
                "axis": "location",
                "recording_interval_seconds": [
                    cluster["lower_seconds"],
                    cluster["upper_seconds"],
                ],
                "resolution": resolution,
                "newly_involved_spatial_keys": newly,
                "active_spatial_keys": list(active),
                "source_candidate_ids": source_ids,
                "within_state_order_resolved": len(newly) == 1,
            }
        )
        if index > 0:
            previous = states[index - 1]
            transitions.append(
                {
                    "transition_id": f"LOCATION-TRANSITION-{index - 1:04d}",
                    "axis": "location",
                    "from_state_id": previous["state_id"],
                    "to_state_id": states[index]["state_id"],
                    "change_interval_seconds": [
                        cluster["lower_seconds"],
                        cluster["upper_seconds"],
                    ],
                    "resolution": resolution,
                    "from_active_spatial_keys": previous["active_spatial_keys"],
                    "to_active_spatial_keys": states[index]["active_spatial_keys"],
                    "newly_involved_spatial_keys": newly,
                    "source_candidate_ids": source_ids,
                    "rule_passed": True,
                    "acns_definite_evolution_step_qualified": False,
                    "reason_codes": [],
                }
            )

    if resolution == "phenotype_only" or not usable_rows:
        status = "not_evaluable"
        opportunity_status = "not_evaluable"
        reasons = ["no_reference_stable_spatial_resolution_with_timed_units"]
        course = None
    elif len(clusters) < 3:
        status = "uncertain"
        opportunity_status = "limited"
        reasons = [
            "fewer_than_two_sequential_location_changes",
            "no_sensitivity_receipt_for_absent_location_evolution",
        ]
        course = (
            float(clusters[0]["lower_seconds"]),
            float(clusters[0]["upper_seconds"]),
        )
    else:
        status = "present"
        opportunity_status = "sufficient"
        reasons = []
        course = (
            float(clusters[0]["lower_seconds"]),
            float(clusters[-1]["upper_seconds"]),
        )

    binding_map = {
        str(row["candidate_id"]): row
        for row in field_result["source_candidate_bindings"]
    }
    used_ids = sorted(
        {
            str(source_id)
            for row in usable_rows
            for source_id in row["source_candidate_ids"]
        }
    )
    used_bindings = [binding_map[source_id] for source_id in used_ids]
    roles = sorted({str(row["temporal_role"]) for row in used_bindings})
    future_access = any(bool(row["future_sample_access"]) for row in used_bindings)
    source_receipts = {
        str(field_result["receipt_sha256"]),
        str(field_result["policy_sha256"]),
        str(field_result["source_field_head_receipt_sha256"]),
        str(field_result["source_input_batch_sha256"]),
        str(field_result["canonical_receipt_sha256"]),
        str(field_result["adaptive_window_receipt_sha256"]),
    }
    source_binding_hashes = [
        _canonical_sha256(
            {
                "candidate_sha256": row["candidate_sha256"],
                "view_receipt_sha256": row["view_receipt_sha256"],
                "view_transform_sha256": row["view_transform_sha256"],
                "temporal_evidence_sha256": row["temporal_evidence_sha256"],
            }
        )
        for row in used_bindings
    ]
    return _finalize_ledger(
        query_id=TQ_EVOLUTION_LOCATION,
        event_id=str(field_result["event_id"]),
        status=status,
        interval=_interval_payload(course, semantics="ordered_spatial_involvement_course"),
        opportunity={
            "status": opportunity_status,
            "selected_resolution": resolution,
            "timed_spatial_key_count": len(usable_rows),
            "interval_distinguishable_state_count": len(clusters),
            "sequential_location_change_count": max(0, len(clusters) - 1),
            "minimum_required_sequential_change_count": 2,
            "reason_codes": sorted(set(reasons)),
        },
        uncertainty={
            "status": "interval_partial_order",
            "overlapping_intervals_collapsed_into_same_unordered_state": True,
            "within_state_ambiguous_key_groups": [
                state["newly_involved_spatial_keys"]
                for state in states
                if not state["within_state_order_resolved"]
            ],
            "cross_reference_raw_score_fusion_used": False,
            "calibration_status": "research_reference_stability_gate",
            "reason_codes": sorted(set(reasons)),
        },
        temporal_evidence={
            "intrinsic_evidence_role": "later_involvement_course_only",
            "source_temporal_roles": roles,
            "future_sample_access_present": future_access,
            "causal_evidence_role": "interval_ordered_scalp_involvement",
            "offline_evidence_role": "may_describe_later_course_not_create_onset",
            "positive_onset_support_authorized": False,
            "positive_soz_support_authorized": False,
            "offline_or_future_context_creates_onset": False,
        },
        state_sequence=states,
        transition_instances=transitions,
        lineage={
            "source_receipt_sha256s": sorted(source_receipts),
            "source_binding_sha256s": _sorted_hashes(
                source_binding_hashes, "source binding"
            ),
            "raw_dependency_sha256s": [],
            "source_candidate_sha256s": sorted(
                str(row["candidate_sha256"]) for row in used_bindings
            ),
            "bipolar_endpoint_attribution_performed": False,
            "eeg_signal_only": True,
        },
        term_guard=_term_guard(
            TQ_EVOLUTION_LOCATION,
            location_distribution_measurement_used=True,
        ),
    )


def compose_return_to_comparable_background_query_ledger_v1(
    baseline_context_receipt: object,
) -> dict[str, Any]:
    """Compose a relative-return candidate; right censoring always fails closed."""

    receipt = validate_event_baseline_context_comparability_receipt(
        baseline_context_receipt
    )
    event_id = str(receipt["event_binding"]["event_id"])
    segments = {
        str(row["context_id"]): row for row in receipt["context_segments"]
    }
    comparisons = [
        row
        for row in receipt["comparisons"]
        if row["purpose"] == "post_event_return_to_reference"
    ]
    right_censored = bool(receipt["window_binding"]["right_censored"])
    policy_locked = receipt["policy"]["calibration_status"] in {
        "source_dev_locked",
        "split_conformal_locked",
    }

    transition_rows: list[dict[str, object]] = []
    matched: list[Mapping[str, Any]] = []
    for row in comparisons:
        target = segments[str(row["target_context_id"])]
        references = [segments[str(item)] for item in row["reference_context_ids"]]
        technically_comparable = (
            row["technical_comparability"]["status"] == "comparable"
        )
        similarity_matched = row["similarity"]["status"] == "matched"
        similarity_locked = row["similarity"]["calibration_status"] in {
            "source_dev_locked",
            "split_conformal_locked",
        }
        target_eligible = bool(
            target["eligibility"]["post_event_return_target"]
        )
        rule_passed = bool(
            not right_censored
            and policy_locked
            and similarity_locked
            and technically_comparable
            and target_eligible
            and similarity_matched
        )
        reason_codes: list[str] = []
        if right_censored:
            reason_codes.append("event_right_censored")
        if not target_eligible:
            reason_codes.extend(
                target["eligibility"]["post_event_return_target_reason_codes"]
            )
        if not technically_comparable:
            reason_codes.extend(row["technical_comparability"]["reason_codes"])
        if not policy_locked or not similarity_locked:
            reason_codes.append("comparison_or_policy_not_calibration_locked")
        if not similarity_matched:
            reason_codes.append("post_event_context_not_decisively_matched")
        transition = {
            "transition_id": str(row["comparison_id"]),
            "axis": "return_to_matched_context",
            "target_context_id": target["context_id"],
            "target_interval_seconds": target["interval_recording_seconds"],
            "reference_context_ids": [item["context_id"] for item in references],
            "reference_intervals_seconds": [
                item["interval_recording_seconds"] for item in references
            ],
            "technical_comparability_status": row[
                "technical_comparability"
            ]["status"],
            "similarity_status": row["similarity"]["status"],
            "similarity_score": row["similarity"]["score"],
            "similarity_threshold": row["similarity"]["threshold"],
            "similarity_uncertainty_margin": row["similarity"][
                "uncertainty_margin"
            ],
            "right_censored": right_censored,
            "matched_context_required": True,
            "rule_passed": rule_passed,
            "clinical_return_or_recovery_qualified": False,
            "upstream_report_permission_authorized": False,
            "reason_codes": sorted(set(reason_codes)),
        }
        transition_rows.append(transition)
        if rule_passed:
            matched.append(row)

    target_intervals = [
        tuple(float(value) for value in segments[str(row["target_context_id"])]["interval_recording_seconds"])
        for row in comparisons
    ]
    if right_censored:
        status = "not_evaluable"
        opportunity_status = "not_evaluable"
        reasons = ["event_right_censored"]
    elif matched:
        status = "present"
        opportunity_status = "sufficient"
        reasons = []
    elif comparisons:
        status = "uncertain"
        opportunity_status = "limited"
        reasons = [
            "no_decisive_matched_technically_comparable_post_event_context",
            "no_sensitivity_receipt_for_absent_return_assertion",
        ]
    else:
        status = "not_evaluable"
        opportunity_status = "not_evaluable"
        reasons = ["no_post_event_return_comparison"]
    selected_interval = None
    if matched:
        selected = min(
            matched,
            key=lambda row: (
                segments[str(row["target_context_id"])]["interval_recording_seconds"],
                str(row["comparison_id"]),
            ),
        )
        selected_interval = tuple(
            float(value)
            for value in segments[str(selected["target_context_id"])][
                "interval_recording_seconds"
            ]
        )
    elif target_intervals:
        selected_interval = min(target_intervals)

    involved_context_ids = sorted(
        {
            str(row["target_context_id"])
            for row in comparisons
        }
        | {
            str(context_id)
            for row in comparisons
            for context_id in row["reference_context_ids"]
        }
    )
    source_bindings = [segments[context_id]["source_binding"] for context_id in involved_context_ids]
    roles = ["context_offline"] if comparisons else []
    future_access = any(bool(row["future_sample_access"]) for row in source_bindings)
    state_sequence = [
        {
            "state_id": context_id,
            "axis": "return_to_matched_context",
            "context_role": segments[context_id]["role"],
            "recording_interval_seconds": segments[context_id][
                "interval_recording_seconds"
            ],
            "quality_status": segments[context_id]["quality"][
                "qualification_status"
            ],
            "contamination_status": segments[context_id]["contamination"]["status"],
        }
        for context_id in involved_context_ids
    ]
    return _finalize_ledger(
        query_id=TQ_RETURN_COMPARABLE_BACKGROUND,
        event_id=event_id,
        status=status,
        interval=_interval_payload(
            selected_interval,
            semantics=(
                "observed_post_event_candidate_interval_not_normative_recovery"
            ),
            right_censored=right_censored,
        ),
        opportunity={
            "status": opportunity_status,
            "post_event_comparison_count": len(comparisons),
            "matched_comparable_comparison_count": len(matched),
            "right_censored": right_censored,
            "matched_context_required": True,
            "reason_codes": sorted(set(reasons)),
        },
        uncertainty={
            "status": "matched_context_similarity_with_censoring",
            "right_censored": right_censored,
            "offset_boundary_status": receipt["window_binding"][
                "offset_boundary_status"
            ],
            "observed_post_event_candidate_intervals_seconds": [
                list(value) for value in sorted(target_intervals)
            ],
            "similarity_is_normative_background_assessment": False,
            "calibration_status": receipt["policy"]["calibration_status"],
            "reason_codes": sorted(set(reasons)),
        },
        temporal_evidence={
            "intrinsic_evidence_role": "post_event_relative_context_only",
            "source_temporal_roles": roles,
            "future_sample_access_present": future_access,
            "causal_evidence_role": "none_for_positive_onset",
            "offline_evidence_role": "matched_within_record_return_candidate_only",
            "positive_onset_support_authorized": False,
            "positive_soz_support_authorized": False,
            "offline_or_future_context_creates_onset": False,
        },
        state_sequence=state_sequence,
        transition_instances=transition_rows,
        lineage={
            "source_receipt_sha256s": [str(receipt["receipt_sha256"])],
            "source_binding_sha256s": sorted(
                {_canonical_sha256(row) for row in source_bindings}
            ),
            "raw_dependency_sha256s": [],
            "source_comparison_ids": sorted(
                str(row["comparison_id"]) for row in comparisons
            ),
            "source_context_ids": involved_context_ids,
            "eeg_signal_only": True,
        },
        term_guard=_term_guard(TQ_RETURN_COMPARABLE_BACKGROUND),
    )


def _replay_query_ledger_v1(
    receipt: object,
    *,
    expected: Mapping[str, Any],
    query_id: str,
) -> dict[str, Any]:
    """Compare one ledger with an independently recomposed source projection."""

    validated = validate_event_evolution_recovery_query_ledger_v1(receipt)
    if validated["query_id"] != query_id:
        raise ValueError("evolution/recovery replay query mismatch")
    recomposed = validate_event_evolution_recovery_query_ledger_v1(expected)
    if recomposed["query_id"] != query_id or validated != recomposed:
        raise ValueError("evolution/recovery query ledger source replay mismatch")
    return recomposed


def replay_frequency_evolution_query_ledger_v1(
    receipt: object,
    *,
    candidate: ACNSFrequencyEvolutionCandidate,
) -> dict[str, Any]:
    """Replay the frequency ledger from an independently supplied candidate."""

    return _replay_query_ledger_v1(
        receipt,
        expected=compose_frequency_evolution_query_ledger_v1(candidate),
        query_id=TQ_EVOLUTION_FREQUENCY,
    )


def replay_morphology_evolution_query_ledger_v1(
    receipt: object,
    *,
    morphology_receipt: object,
    change_points: Sequence[EventChangePointProposal] | None = None,
    change_point: EventChangePointProposal | None = None,
    policy: MorphologyEvolutionPolicy = DEFAULT_MORPHOLOGY_EVOLUTION_POLICY,
) -> dict[str, Any]:
    """Replay the morphology ledger from independent sidecar/proposal inputs."""

    return _replay_query_ledger_v1(
        receipt,
        expected=compose_morphology_evolution_query_ledger_v1(
            morphology_receipt,
            change_points=change_points,
            change_point=change_point,
            policy=policy,
        ),
        query_id=TQ_EVOLUTION_MORPHOLOGY,
    )


def replay_location_evolution_query_ledger_v1(
    receipt: object,
    *,
    multireference_field_result: object,
) -> dict[str, Any]:
    """Replay the location ledger from an independent field result."""

    return _replay_query_ledger_v1(
        receipt,
        expected=compose_location_evolution_query_ledger_v1(
            multireference_field_result
        ),
        query_id=TQ_EVOLUTION_LOCATION,
    )


def replay_return_to_comparable_background_query_ledger_v1(
    receipt: object,
    *,
    baseline_context_receipt: object,
) -> dict[str, Any]:
    """Replay the relative-return ledger from an independent context receipt."""

    return _replay_query_ledger_v1(
        receipt,
        expected=compose_return_to_comparable_background_query_ledger_v1(
            baseline_context_receipt
        ),
        query_id=TQ_RETURN_COMPARABLE_BACKGROUND,
    )


__all__ = [
    "DEFAULT_MORPHOLOGY_EVOLUTION_POLICY",
    "EVENT_EVOLUTION_RECOVERY_QUERY_BRIDGE_ID",
    "EVENT_EVOLUTION_RECOVERY_QUERY_BRIDGE_SCHEMA_VERSION",
    "EventChangePointProposal",
    "MorphologyEvolutionPolicy",
    "TQ_EVOLUTION_FREQUENCY",
    "TQ_EVOLUTION_LOCATION",
    "TQ_EVOLUTION_MORPHOLOGY",
    "TQ_RETURN_COMPARABLE_BACKGROUND",
    "compose_frequency_evolution_query_ledger_v1",
    "compose_location_evolution_query_ledger_v1",
    "compose_morphology_evolution_query_ledger_v1",
    "compose_return_to_comparable_background_query_ledger_v1",
    "extract_axis_agnostic_change_point_proposal_from_v3",
    "replay_frequency_evolution_query_ledger_v1",
    "replay_location_evolution_query_ledger_v1",
    "replay_morphology_evolution_query_ledger_v1",
    "replay_return_to_comparable_background_query_ledger_v1",
    "validate_event_evolution_recovery_query_ledger_v1",
]
