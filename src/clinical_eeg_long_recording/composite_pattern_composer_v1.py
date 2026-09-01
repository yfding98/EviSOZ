"""EEG-only atomic-to-composite pattern candidate composition.

This target-free stage consumes an already materialized ``event_eeg_findings_v2``
ledger.  It never reads annotations, spreadsheets, doctor labels, clinical text,
or patient targets.  Its outputs remain ``model_candidate`` assertions; clinical
surface eligibility still requires the independent capability and term-decision
receipt path enforced by the v2 validator.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from .event_findings_v2_validation import pattern_term_registry_sha256_v1


COMPOSITE_PATTERN_COMPOSER_V1_METHOD_ID = (
    "eeg_only_atomic_composite_pattern_composer_v1"
)
COMPOSITE_PATTERN_ONTOLOGY_ID = "EEG-COMPOSITE-PATTERN-CANDIDATES-V1"
COMPOSITE_PATTERN_TERM_REGISTRY_ID = (
    "DETERMINISTIC-EEG-COMPOSITE-PATTERN-TERM-REGISTRY"
)
COMPOSITE_PATTERN_TERM_REGISTRY_VERSION = "1.0.0"

_PATTERN_RULES: dict[str, str] = {
    "repetitive_spike_like_pattern_candidate": (
        "RULE-REPETITIVE-SPIKE-LIKE-CANDIDATE-V1"
    ),
    "rhythmic_delta_theta_pattern_candidate": (
        "RULE-RHYTHMIC-DELTA-THETA-CANDIDATE-V1"
    ),
    "widespread_near_synchronous_pattern_candidate": (
        "RULE-WIDESPREAD-NEAR-SYNCHRONOUS-CANDIDATE-V1"
    ),
}

_PATTERN_TERM_ENTRIES = [
    {
        "term_id": term_id,
        "ontology_id": COMPOSITE_PATTERN_ONTOLOGY_ID,
        "operational_rule_id": _PATTERN_RULES[term_id],
    }
    for term_id in sorted(_PATTERN_RULES)
]

DEFAULT_COMPOSITE_PATTERN_TERM_REGISTRY_BINDING: dict[str, Any] = {
    "registry_id": COMPOSITE_PATTERN_TERM_REGISTRY_ID,
    "version": COMPOSITE_PATTERN_TERM_REGISTRY_VERSION,
    "registry_sha256": "0" * 64,
    "trust_status": "host_trusted",
    "terms": deepcopy(_PATTERN_TERM_ENTRIES),
}
DEFAULT_COMPOSITE_PATTERN_TERM_REGISTRY_BINDING["registry_sha256"] = (
    pattern_term_registry_sha256_v1(
        DEFAULT_COMPOSITE_PATTERN_TERM_REGISTRY_BINDING
    )
)


@dataclass(frozen=True)
class CompositePatternComposerPolicy:
    """Frozen target-free gates for research-level pattern candidates."""

    delta_theta_lower_hz: float = 1.0
    delta_theta_upper_hz: float = 8.0
    minimum_rhythmicity_index: float = 0.60
    minimum_spectral_concentration: float = 0.40
    minimum_widespread_units: int = 4

    def __post_init__(self) -> None:
        numeric = (
            self.delta_theta_lower_hz,
            self.delta_theta_upper_hz,
            self.minimum_rhythmicity_index,
            self.minimum_spectral_concentration,
        )
        if not all(math.isfinite(float(item)) for item in numeric):
            raise ValueError("composite pattern policy values must be finite")
        if not 0.0 < self.delta_theta_lower_hz < self.delta_theta_upper_hz:
            raise ValueError("delta/theta frequency bounds are invalid")
        if not 0.0 <= self.minimum_rhythmicity_index <= 1.0:
            raise ValueError("minimum_rhythmicity_index must lie in [0, 1]")
        if not 0.0 <= self.minimum_spectral_concentration <= 1.0:
            raise ValueError("minimum_spectral_concentration must lie in [0, 1]")
        if (
            isinstance(self.minimum_widespread_units, bool)
            or not isinstance(self.minimum_widespread_units, int)
            or self.minimum_widespread_units < 2
        ):
            raise ValueError("minimum_widespread_units must be an integer >= 2")

    def to_dict(self) -> dict[str, object]:
        return {
            "delta_theta_lower_hz": float(self.delta_theta_lower_hz),
            "delta_theta_upper_hz": float(self.delta_theta_upper_hz),
            "minimum_rhythmicity_index": float(
                self.minimum_rhythmicity_index
            ),
            "minimum_spectral_concentration": float(
                self.minimum_spectral_concentration
            ),
            "minimum_widespread_units": int(self.minimum_widespread_units),
        }


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _pattern_term_ref(term_id: str) -> dict[str, str]:
    if term_id not in _PATTERN_RULES:
        raise ValueError(f"unknown composite pattern term: {term_id}")
    return {
        "term_id": term_id,
        "ontology_id": COMPOSITE_PATTERN_ONTOLOGY_ID,
        "source_id": COMPOSITE_PATTERN_TERM_REGISTRY_ID,
        "source_version": COMPOSITE_PATTERN_TERM_REGISTRY_VERSION,
        "operational_rule_id": _PATTERN_RULES[term_id],
    }


def _source_domain_scope(rows: Sequence[Mapping[str, Any]]) -> str:
    roles = {str(row["intrinsic_evidence_role"]) for row in rows}
    course_roles = {"early_context", "later_involvement"}
    if roles == {"onset_eligible"}:
        return "onset_causal_only"
    if roles and roles.issubset(course_roles):
        return "event_course_only"
    if (
        "onset_eligible" in roles
        and roles.intersection(course_roles)
        and roles.issubset(course_roles | {"onset_eligible"})
    ):
        return "mixed_onset_causal_and_event_course"
    if roles == {"non_event_context"}:
        return "non_event_context_only"
    return "not_evaluable"


def _measurement_values_by_unit(
    finding: Mapping[str, Any], name_prefix: str
) -> dict[str, float]:
    values: dict[str, float] = {}
    for measurement in finding["measurements"]:
        if not str(measurement["name_id"]).startswith(name_prefix):
            continue
        value = float(measurement["value"])
        for unit_id in measurement["source_binding"]["source_unit_ids"]:
            values[str(unit_id)] = value
    return values


def _identifier(prefix: str, value: object) -> str:
    return f"{prefix}-{_canonical_sha256(value)[:24]}"


def _append_pattern(
    *,
    result: dict[str, Any],
    finding_map: Mapping[str, dict[str, Any]],
    term_id: str,
    required_atom_ids: Sequence[str],
    counterevidence_ids: Sequence[str],
    status: str,
    reason_codes: Sequence[str],
    policy_sha256: str,
) -> None:
    required = sorted(set(str(item) for item in required_atom_ids))
    counterevidence = sorted(set(str(item) for item in counterevidence_ids))
    if not required:
        raise ValueError("a composite pattern requires at least one atomic Finding")
    if set(required).intersection(counterevidence):
        raise ValueError("composite required atoms and counterevidence must be disjoint")
    missing = set(required + counterevidence).difference(finding_map)
    if missing:
        raise ValueError(f"composite pattern references missing atoms: {sorted(missing)}")

    instance_id = _identifier(
        "PATINST",
        {
            "event_id": result["event_id"],
            "term_id": term_id,
            "required_atom_ids": required,
        },
    )
    for evidence_id in required:
        existing = finding_map[evidence_id]["pattern_instance_id"]
        if existing not in {None, instance_id}:
            raise ValueError(
                f"atomic Finding {evidence_id!r} already belongs to another pattern instance"
            )
        finding_map[evidence_id]["pattern_instance_id"] = instance_id

    candidate_id = _identifier(
        "PATCAND",
        {
            "pattern_instance_id": instance_id,
            "term_id": term_id,
            "status": status,
            "counterevidence_ids": counterevidence,
            "policy_sha256": policy_sha256,
        },
    )
    required_rows = [finding_map[item] for item in required]
    result["pattern_candidates"].append(
        {
            "pattern_candidate_id": candidate_id,
            "pattern_instance_id": instance_id,
            "event_id": result["event_id"],
            "term": _pattern_term_ref(term_id),
            "assertion_level": "model_candidate",
            "status": status,
            "source_domain_scope": _source_domain_scope(required_rows),
            "required_atom_ids": required,
            "counterevidence_ids": counterevidence,
            "qualification_rule_receipt_id": None,
            "reason_codes": sorted(set(str(item) for item in reason_codes)),
        }
    )


def _compose_rhythmic_delta_theta(
    *,
    result: dict[str, Any],
    finding_map: Mapping[str, dict[str, Any]],
    policy: CompositePatternComposerPolicy,
    policy_sha256: str,
) -> None:
    spectral = next(
        (
            row
            for row in result["findings"]
            if row["term"]["term_id"] == "deterministic_event_spectral_profile"
        ),
        None,
    )
    rhythm = next(
        (
            row
            for row in result["findings"]
            if row["term"]["term_id"] == "deterministic_event_rhythmicity_profile"
        ),
        None,
    )
    if spectral is None or rhythm is None:
        return
    required = [str(spectral["evidence_id"]), str(rhythm["evidence_id"])]
    statuses = {str(spectral["status"]), str(rhythm["status"])}
    if "not_evaluable" in statuses:
        _append_pattern(
            result=result,
            finding_map=finding_map,
            term_id="rhythmic_delta_theta_pattern_candidate",
            required_atom_ids=required,
            counterevidence_ids=[],
            status="not_evaluable",
            reason_codes=["rhythmic_delta_theta_atoms_not_evaluable"],
            policy_sha256=policy_sha256,
        )
        return
    if statuses != {"present"}:
        _append_pattern(
            result=result,
            finding_map=finding_map,
            term_id="rhythmic_delta_theta_pattern_candidate",
            required_atom_ids=required,
            counterevidence_ids=[],
            status="uncertain",
            reason_codes=["rhythmic_delta_theta_atoms_incomplete"],
            policy_sha256=policy_sha256,
        )
        return

    frequencies = _measurement_values_by_unit(
        spectral, "event_dominant_frequency_"
    )
    rhythmicity = _measurement_values_by_unit(
        rhythm, "event_rhythmicity_index_"
    )
    concentration = _measurement_values_by_unit(
        rhythm, "event_spectral_concentration_"
    )
    shared_units = set(frequencies).intersection(rhythmicity, concentration)
    qualifying_units = {
        unit_id
        for unit_id in shared_units
        if policy.delta_theta_lower_hz
        <= frequencies[unit_id]
        <= policy.delta_theta_upper_hz
        and rhythmicity[unit_id] >= policy.minimum_rhythmicity_index
        and concentration[unit_id] >= policy.minimum_spectral_concentration
    }
    if not qualifying_units:
        return
    _append_pattern(
        result=result,
        finding_map=finding_map,
        term_id="rhythmic_delta_theta_pattern_candidate",
        required_atom_ids=required,
        counterevidence_ids=[],
        status="present",
        reason_codes=["target_free_candidate_not_clinically_qualified"],
        policy_sha256=policy_sha256,
    )


def _compose_repetitive_spike_like(
    *,
    result: dict[str, Any],
    finding_map: Mapping[str, dict[str, Any]],
    policy_sha256: str,
) -> None:
    morphology = next(
        (
            row
            for row in result["findings"]
            if row["term"]["term_id"] == "deterministic_morphology_candidate"
        ),
        None,
    )
    if morphology is None:
        return
    status = str(morphology["status"])
    if status == "not_evaluable":
        pattern_status = "not_evaluable"
        reasons = ["morphology_atom_not_evaluable_no_spike_like_claim"]
    else:
        pattern_status = "uncertain"
        reasons = ["spike_like_pattern_requires_independent_ifcn_qualification"]
    _append_pattern(
        result=result,
        finding_map=finding_map,
        term_id="repetitive_spike_like_pattern_candidate",
        required_atom_ids=[str(morphology["evidence_id"])],
        counterevidence_ids=[],
        status=pattern_status,
        reason_codes=reasons,
        policy_sha256=policy_sha256,
    )


def _compose_widespread_near_synchronous(
    *,
    result: dict[str, Any],
    finding_map: Mapping[str, dict[str, Any]],
    policy: CompositePatternComposerPolicy,
    policy_sha256: str,
) -> None:
    onset_fields = [
        row
        for row in result["findings"]
        if row["family"] == "spatial_field"
        and row["term"]["term_id"]
        == "reference_specific_spatial_change_candidate"
    ]
    if not onset_fields:
        return
    required = [str(row["evidence_id"]) for row in onset_fields]
    if any(row["status"] == "not_evaluable" for row in onset_fields):
        _append_pattern(
            result=result,
            finding_map=finding_map,
            term_id="widespread_near_synchronous_pattern_candidate",
            required_atom_ids=required,
            counterevidence_ids=[],
            status="not_evaluable",
            reason_codes=["causal_spatial_field_not_evaluable"],
            policy_sha256=policy_sha256,
        )
        return

    hypothesis = result["scalp_onset_hypothesis"]
    onset_boundary = result["window"]["onset_boundary"]
    near_onset_rows: list[Mapping[str, Any]] = []
    if onset_boundary["status"] == "observed":
        onset = onset_boundary["interval"]
        for row in hypothesis["per_unit_involvement"]:
            interval = row["interval"]
            if row["status"] != "present" or interval is None:
                continue
            if (
                float(interval["lower"]) <= float(onset["upper"])
                and float(interval["upper"]) >= float(onset["lower"])
            ):
                near_onset_rows.append(row)

    near_onset_evidence = {
        str(evidence_id)
        for row in near_onset_rows
        for evidence_id in row["evidence_ids"]
        if str(evidence_id) in finding_map
        and finding_map[str(evidence_id)]["status"] == "present"
    }
    bilateral_anchor = any(
        support["unit_type"] == "laterality"
        and support["id"] == "bilateral"
        and support["evidence_eligible"]
        for row in onset_fields
        for support in row["spatial_support"]
    )
    widespread_supported = (
        len(near_onset_rows) >= policy.minimum_widespread_units
        or (bilateral_anchor and len(near_onset_rows) >= 2)
    )
    sequential_counterevidence = {
        str(evidence_id)
        for relation in hypothesis["involvement_order"]
        if relation["relation_status"] == "precedes"
        for evidence_id in relation["evidence_ids"]
        if str(evidence_id) in finding_map
        and finding_map[str(evidence_id)]["status"] == "present"
    }

    if widespread_supported and not sequential_counterevidence:
        required = sorted(set(required).union(near_onset_evidence))
        status = "present"
        reasons = ["target_free_candidate_not_clinically_qualified"]
        counterevidence: list[str] = []
    else:
        status = "uncertain"
        counterevidence = sorted(sequential_counterevidence.difference(required))
        reasons = [
            "sequential_later_involvement_counterevidence"
            if counterevidence
            else "widespread_near_synchronous_support_incomplete"
        ]
    _append_pattern(
        result=result,
        finding_map=finding_map,
        term_id="widespread_near_synchronous_pattern_candidate",
        required_atom_ids=required,
        counterevidence_ids=counterevidence,
        status=status,
        reason_codes=reasons,
        policy_sha256=policy_sha256,
    )


def compose_eeg_only_composite_pattern_candidates_v1(
    value: Mapping[str, object],
    *,
    policy: CompositePatternComposerPolicy | None = None,
) -> dict[str, Any]:
    """Compose deterministic, target-free research pattern candidates.

    The function deliberately runs before any clinical term-decision receipt is
    issued.  Adding a pattern changes the event ledger digest; post-qualification
    composition would therefore invalidate trusted receipts and is rejected.
    """

    if not isinstance(value, Mapping):
        raise TypeError("event Findings payload must be a mapping")
    result: dict[str, Any] = deepcopy(dict(value))
    if result.get("schema_version") != "event_eeg_findings_v2":
        raise ValueError("composite composer requires event_eeg_findings_v2")
    if result.get("migration") is not None:
        raise ValueError("lossy migrated Findings cannot create composite patterns")
    if "pattern_candidates" not in result or any(
        "pattern_instance_id" not in finding for finding in result["findings"]
    ):
        raise ValueError(
            "validate the pre-extension v2 ledger before composite composition"
        )
    exclusions = result["provenance"]["inference_exclusions"]
    if any(item is not False for item in exclusions.values()):
        raise ValueError("composite pattern composition requires explicit EEG-only scope")
    if result.get("pattern_candidates"):
        raise ValueError("composite composer requires an uncomposed atomic ledger")
    if result["registry_bindings"].get("pattern_term_registry") is not None:
        raise ValueError("uncomposed ledger cannot carry a pattern term registry")
    if any(
        finding.get("pattern_instance_id") is not None
        for finding in result["findings"]
    ):
        raise ValueError("uncomposed atomic Findings cannot carry pattern instances")
    if result["term_decision_receipts"] or result["event_qualification"].get(
        "qualification_receipt_id"
    ) is not None:
        raise ValueError("compose patterns before issuing content-bound term receipts")

    active_policy = policy or CompositePatternComposerPolicy()
    policy_sha256 = _canonical_sha256(
        {
            "method_id": COMPOSITE_PATTERN_COMPOSER_V1_METHOD_ID,
            "policy": active_policy.to_dict(),
            "source_scope": "eeg_only_target_free_no_annotations_no_labels",
            "assertion_ceiling": "model_candidate",
        }
    )
    result["pattern_candidates"] = []
    finding_map = {
        str(row["evidence_id"]): row for row in result["findings"]
    }
    _compose_rhythmic_delta_theta(
        result=result,
        finding_map=finding_map,
        policy=active_policy,
        policy_sha256=policy_sha256,
    )
    _compose_repetitive_spike_like(
        result=result,
        finding_map=finding_map,
        policy_sha256=policy_sha256,
    )
    _compose_widespread_near_synchronous(
        result=result,
        finding_map=finding_map,
        policy=active_policy,
        policy_sha256=policy_sha256,
    )

    if result["pattern_candidates"]:
        result["registry_bindings"]["pattern_term_registry"] = deepcopy(
            DEFAULT_COMPOSITE_PATTERN_TERM_REGISTRY_BINDING
        )
        additions = [
            COMPOSITE_PATTERN_COMPOSER_V1_METHOD_ID,
            "PATTERNPOLICY-" + policy_sha256,
        ]
        for identifier in additions:
            if identifier not in result["provenance"]["model_ids"]:
                result["provenance"]["model_ids"].append(identifier)
    return result


__all__ = [
    "COMPOSITE_PATTERN_COMPOSER_V1_METHOD_ID",
    "COMPOSITE_PATTERN_ONTOLOGY_ID",
    "COMPOSITE_PATTERN_TERM_REGISTRY_ID",
    "COMPOSITE_PATTERN_TERM_REGISTRY_VERSION",
    "CompositePatternComposerPolicy",
    "DEFAULT_COMPOSITE_PATTERN_TERM_REGISTRY_BINDING",
    "compose_eeg_only_composite_pattern_candidates_v1",
]
