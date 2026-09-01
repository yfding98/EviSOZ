"""Exact EEG-only qualification rules for protected clinical EEG terms.

This ledger deliberately sits after signal measurements and before report
rendering.  It never discovers a spike, onset, seizure, or SOZ.  Instead it
checks whether host-supplied, signal-bound measurements satisfy frozen
operational definitions.  The IFCN morphology ledger applies only to an
interictal transient and is never reused as ictal onset/evolution evidence.

Primary sources encoded here:

* Kane et al., IFCN glossary revision 2017, DOI 10.1016/j.cnp.2017.07.002;
* Kural et al., clinical validation 2020, DOI 10.1212/WNL.0000000000009439;
* Hirsch et al., ACNS terminology 2021, DOI 10.1097/WNP.0000000000000806.

The historic four-of-six IFCN suggestion is recorded as a candidate flag, not
as a sufficient clinical term gate.  The five-of-six Kural operating point is
the higher-specificity gate (reported specificity 95.65%, sensitivity 81.48%)
and still requires a frozen target-domain qualification receipt, physiologic
scalp field, interictal context, and exclusion of artifact/benign variants.

ACNS 2021 electrographic-seizure criterion A is represented separately from
this project's deliberately stricter promotion gate.  The source definition
accepts either epileptiform discharges or sharply contoured discharges at an
average rate greater than 2.5 Hz for at least 10 seconds; a sharply contoured
discharge is not rejected merely because its main component exceeds 200 ms.
The additional interictal-ED qualification receipt remains a project safety
gate and is never described as part of the complete ACNS source definition.
"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator


CLINICAL_EEG_TERM_QUALIFICATION_SCHEMA_VERSION = (
    "clinical_eeg_term_qualification_v1"
)
IFCN_FEATURE_NAMES = (
    "di_or_triphasic_sharp_or_spiky_morphology",
    "duration_differs_from_background",
    "waveform_asymmetry",
    "slow_after_wave",
    "surrounding_background_disruption",
    "physiologic_scalp_field",
)
PROTECTED_EEG_ONLY_TERMS = frozenset(
    {
        "spike",
        "sharp_wave",
        "interictal_epileptiform_discharge",
        "definite_evolution",
        "electrographic_seizure",
    }
)
FORBIDDEN_EEG_ONLY_TERMS = frozenset(
    {
        "electroclinical_seizure",
        "clinical_semiology",
        "impaired_awareness",
        "behavioral_arrest",
        "motor_manifestation",
        "autonomic_manifestation",
        "clinical_response_to_antiseizure_medication",
    }
)

_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA_PATH = _ROOT / "schemas" / "clinical_eeg_term_qualification_v1.schema.json"
_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_QUALIFIED_TERM_ORDER = (
    "spike",
    "sharp_wave",
    "interictal_epileptiform_discharge",
    "definite_evolution",
    "electrographic_seizure",
)
_ACNS_CRITERION_A_DISCHARGE_TYPES = frozenset(
    {"epileptiform_discharge", "sharply_contoured_discharge"}
)
_SCOPE_RECEIPT = {
    "eeg_signal_only": True,
    "edf_annotations_used": False,
    "spreadsheet_used": False,
    "doctor_labels_used": False,
    "clinical_text_used": False,
    "video_used": False,
    "clinical_correlate_used": False,
    "awareness_or_behavior_used": False,
    "parenteral_medication_response_used": False,
    "semiology_claims_allowed": False,
    "electroclinical_seizure_claims_allowed": False,
}
CLINICAL_TERM_QUALIFICATION_POLICY = {
    "schema_version": "clinical_eeg_term_qualification_policy_v1",
    "spike_main_component_duration_ms": [20.0, 70.0],
    "spike_upper_bound_inclusive": False,
    "sharp_wave_main_component_duration_ms": [70.0, 200.0],
    "sharp_wave_upper_bound_inclusive": True,
    "historically_suggested_ifcn_candidate_count": 4,
    "kural_clinical_implementation_count": 5,
    "kural_five_of_six_specificity_percent": 95.65,
    "kural_five_of_six_sensitivity_percent": 81.48,
    "physiologic_scalp_field_required": True,
    "target_domain_qualification_required": True,
    "interictal_context_required_for_ied": True,
    "ifcn_features_may_support_ictal_onset_or_evolution": False,
    "acns_evolution_minimum_sequential_changes": 2,
    "acns_frequency_minimum_step_hz": 0.5,
    "acns_minimum_cycles_per_state": 3,
    "acns_location_minimum_standard_electrodes": 2,
    "acns_maximum_unchanged_gap_seconds_exclusive": 300.0,
    "amplitude_only_counts_as_evolution": False,
    "acns_esz_criterion_a_minimum_rate_hz_exclusive": 2.5,
    "acns_esz_criterion_a_eligible_discharge_types": [
        "epileptiform_discharge",
        "sharply_contoured_discharge",
    ],
    "acns_esz_criterion_a_sharply_contoured_discharge_200_ms_upper_bound_applied": False,
    "project_esz_criterion_a_requires_interictal_ed_qualification_receipt": True,
    "acns_esz_minimum_duration_seconds": 10.0,
    "electroclinical_terms_allowed_from_eeg_only": False,
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


CLINICAL_TERM_QUALIFICATION_POLICY_SHA256 = _canonical_sha256(
    CLINICAL_TERM_QUALIFICATION_POLICY
)


@lru_cache(maxsize=1)
def _schema_validator() -> Draft202012Validator:
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _path(error: Any) -> str:
    parts = [str(item) for item in error.absolute_path]
    return ".".join(parts) if parts else "$"


def _finite(value: object, context: str, *, nullable: bool = False) -> float | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _sha256(value: object, context: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_CHARACTERS for character in value)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{context} must be a non-empty trimmed identifier")
    return value


def _ifcn_feature_map(value: Mapping[str, object]) -> dict[str, bool]:
    if not isinstance(value, Mapping) or set(value) != set(IFCN_FEATURE_NAMES):
        raise ValueError("IFCN feature ledger must contain exactly the frozen six features")
    if any(type(value[name]) is not bool for name in IFCN_FEATURE_NAMES):
        raise TypeError("every IFCN feature must be boolean")
    return {name: bool(value[name]) for name in IFCN_FEATURE_NAMES}


def _waveform_duration_matches(candidate_type: str, duration_ms: float | None) -> bool:
    if duration_ms is None:
        return False
    if candidate_type == "spike":
        return 20.0 <= duration_ms < 70.0
    if candidate_type == "sharp_wave":
        return 70.0 <= duration_ms <= 200.0
    return False


def _evolution_gate(value: Mapping[str, object]) -> bool:
    category = str(value["category"])
    changes = int(value["unequivocal_sequential_changes"])
    cycles = int(value["minimum_cycles_per_state"])
    locations = int(value["distinct_standard_electrode_locations"])
    gap = _finite(
        value["maximum_unchanged_gap_seconds"],
        "maximum_unchanged_gap_seconds",
        nullable=True,
    )
    if (
        category == "none"
        or bool(value["amplitude_only_change"])
        or changes < 2
        or cycles < 3
        or gap is None
        or gap >= 300.0
    ):
        return False
    if category == "frequency":
        step = _finite(
            value["minimum_frequency_step_hz"],
            "minimum_frequency_step_hz",
            nullable=True,
        )
        return bool(value["frequency_changes_same_direction"]) and step is not None and step >= 0.5
    if category == "morphology":
        return True
    if category == "location":
        return locations >= 2
    return False


def _expected_qualified_terms(payload: Mapping[str, object]) -> list[str]:
    morphology = payload["morphology"]
    evolution = payload["evolution"]
    seizure = payload["electrographic_seizure"]
    terms: set[str] = set()
    promoted = morphology["promoted_term"]
    if promoted is not None:
        terms.add(str(promoted))
    if morphology["ied_promotion_gate_passed"]:
        terms.add("interictal_epileptiform_discharge")
    if evolution["definite_evolution_gate_passed"]:
        terms.add("definite_evolution")
    if seizure["esz_gate_passed"]:
        terms.add("electrographic_seizure")
    return [term for term in _QUALIFIED_TERM_ORDER if term in terms]


def validate_clinical_eeg_term_qualification(payload: object) -> dict[str, Any]:
    """Validate schema, exact thresholds, EEG-only scope, and content hashes."""

    if type(payload) is not dict:
        raise TypeError("clinical EEG term qualification must be an object")
    data = deepcopy(payload)
    errors = sorted(_schema_validator().iter_errors(data), key=lambda item: list(item.path))
    if errors:
        rendered = "; ".join(f"{_path(error)}: {error.message}" for error in errors[:8])
        raise ValueError(f"clinical EEG term qualification schema failed: {rendered}")
    if data["policy_sha256"] != CLINICAL_TERM_QUALIFICATION_POLICY_SHA256:
        raise ValueError("clinical EEG term qualification policy hash drifted")
    _sha256(data["source_binding_sha256"], "source_binding_sha256")
    morphology = data["morphology"]
    features = _ifcn_feature_map(morphology["ifcn_features"])
    count = sum(features.values())
    if morphology["ifcn_criteria_met_count"] != count:
        raise ValueError("IFCN criteria count disagrees with the six-feature ledger")
    if morphology["historically_suggested_four_of_six_candidate"] is not (count >= 4):
        raise ValueError("historic four-of-six candidate flag is inconsistent")
    if morphology["kural_five_of_six_high_specificity_gate"] is not (count >= 5):
        raise ValueError("Kural five-of-six gate is inconsistent")
    target_receipt = _sha256(
        morphology["target_domain_qualification_receipt_sha256"],
        "target_domain_qualification_receipt_sha256",
        nullable=True,
    )
    target_qualified = bool(morphology["target_domain_qualification_passed"])
    if target_qualified != (target_receipt is not None):
        raise ValueError("target-domain qualification must bind exactly one receipt hash")
    duration = _finite(
        morphology["main_component_duration_ms"],
        "main_component_duration_ms",
        nullable=True,
    )
    waveform_gate = (
        _waveform_duration_matches(str(morphology["candidate_type"]), duration)
        and bool(morphology["clearly_distinguished_from_background"])
        and bool(morphology["pointed_peak_at_clinical_display_scale"])
        and bool(morphology["artifact_or_benign_variant_excluded"])
        and target_qualified
    )
    if morphology["waveform_term_gate_passed"] is not waveform_gate:
        raise ValueError("spike/sharp-wave gate is inconsistent with its measurements")
    expected_promoted = morphology["candidate_type"] if waveform_gate else None
    if morphology["promoted_term"] != expected_promoted:
        raise ValueError("promoted spike/sharp-wave term is inconsistent")
    ied_gate = (
        count >= 5
        and features["physiologic_scalp_field"]
        and bool(morphology["artifact_or_benign_variant_excluded"])
        and target_qualified
        and bool(morphology["interictal_context_confirmed"])
    )
    if morphology["ied_promotion_gate_passed"] is not ied_gate:
        raise ValueError("IED promotion gate is inconsistent")
    if morphology["ifcn_features_used_for_ictal_onset_or_evolution"] is not False:
        raise ValueError("IFCN IED features cannot qualify ictal onset/evolution")

    evolution = data["evolution"]
    evolution_gate = _evolution_gate(evolution)
    if evolution["definite_evolution_gate_passed"] is not evolution_gate:
        raise ValueError("ACNS definite-evolution gate is inconsistent")
    seizure = data["electrographic_seizure"]
    criterion_a_discharge_type = str(seizure["criterion_a_acns_discharge_type"])
    criterion_a_discharge_duration_ms = _finite(
        seizure["criterion_a_acns_discharge_main_component_duration_ms"],
        "criterion_a_acns_discharge_main_component_duration_ms",
        nullable=True,
    )
    if (
        criterion_a_discharge_duration_ms is not None
        and criterion_a_discharge_duration_ms <= 0.0
    ):
        raise ValueError(
            "criterion A discharge main-component duration must be positive"
        )
    criterion_a_source_morphology = (
        criterion_a_discharge_type in _ACNS_CRITERION_A_DISCHARGE_TYPES
    )
    if seizure["criterion_a_acns_source_morphology_gate_passed"] is not (
        criterion_a_source_morphology
    ):
        raise ValueError("ACNS ESz criterion A source morphology gate is inconsistent")
    criterion_a_project_receipt = _sha256(
        seizure[
            "criterion_a_project_interictal_ed_qualification_receipt_sha256"
        ],
        "criterion_a_project_interictal_ed_qualification_receipt_sha256",
        nullable=True,
    )
    criterion_a_project_safety_gate = criterion_a_project_receipt is not None
    if seizure["criterion_a_project_interictal_ed_safety_gate_passed"] is not (
        criterion_a_project_safety_gate
    ):
        raise ValueError(
            "project criterion A interictal-ED safety gate must bind one receipt"
        )
    duration_seconds = _finite(
        seizure["pattern_duration_seconds"], "pattern_duration_seconds"
    )
    rate = _finite(
        seizure["criterion_a_discharge_rate_hz"],
        "criterion_a_discharge_rate_hz",
        nullable=True,
    )
    criterion_a_acns_source = (
        duration_seconds >= 10.0
        and criterion_a_source_morphology
        and rate is not None
        and rate > 2.5
    )
    if seizure["criterion_a_acns_source_definition_gate_passed"] is not (
        criterion_a_acns_source
    ):
        raise ValueError("ACNS ESz criterion A source-definition gate is inconsistent")
    criterion_a_project = (
        criterion_a_acns_source and criterion_a_project_safety_gate
    )
    criterion_b = duration_seconds >= 10.0 and evolution_gate
    if seizure["criterion_a_project_conservative_gate_passed"] is not (
        criterion_a_project
    ):
        raise ValueError("project conservative ESz criterion A gate is inconsistent")
    if seizure["criterion_b_definite_evolution"] is not evolution_gate:
        raise ValueError("ESz criterion B does not bind the evolution ledger")
    if seizure["criterion_b_gate_passed"] is not criterion_b:
        raise ValueError("ACNS ESz criterion B gate is inconsistent")
    if seizure["esz_gate_passed"] is not (criterion_a_project or criterion_b):
        raise ValueError("electrographic-seizure gate is inconsistent")
    if seizure["electroclinical_seizure_claimed"] is not False:
        raise ValueError("EEG-only receipt cannot claim an electroclinical seizure")
    if data["qualified_terms"] != _expected_qualified_terms(data):
        raise ValueError("qualified clinical terms do not close their evidence gates")

    _sha256(data["receipt_sha256"], "receipt_sha256")
    digest_source = deepcopy(data)
    digest_source["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["receipt_sha256"] != _canonical_sha256(digest_source):
        raise ValueError("clinical term qualification hash does not bind content")
    id_source = deepcopy(data)
    id_source["receipt_id"] = "CONTENT-ADDRESS-PENDING"
    id_source["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    expected_id = "EEGTERM-" + _canonical_sha256(id_source)[:20]
    if data["receipt_id"] != expected_id:
        raise ValueError("clinical term qualification ID does not bind content")
    return data


def build_clinical_eeg_term_qualification(
    *,
    event_id: str,
    producer_id: str,
    source_binding_sha256: str,
    candidate_type: str,
    main_component_duration_ms: float | None,
    clearly_distinguished_from_background: bool,
    pointed_peak_at_clinical_display_scale: bool,
    ifcn_features: Mapping[str, object],
    artifact_or_benign_variant_excluded: bool,
    target_domain_qualification_receipt_sha256: str | None,
    interictal_context_confirmed: bool,
    evolution_category: str,
    unequivocal_sequential_changes: int,
    frequency_changes_same_direction: bool | None,
    minimum_frequency_step_hz: float | None,
    minimum_cycles_per_state: int,
    distinct_standard_electrode_locations: int,
    maximum_unchanged_gap_seconds: float | None,
    amplitude_only_change: bool,
    pattern_duration_seconds: float,
    criterion_a_epileptiform_discharge_receipt_sha256: str | None = None,
    criterion_a_discharge_rate_hz: float | None = None,
    criterion_a_acns_discharge_type: str | None = None,
    criterion_a_acns_discharge_main_component_duration_ms: float | None = None,
    criterion_a_project_interictal_ed_qualification_receipt_sha256: (
        str | None
    ) = None,
) -> dict[str, Any]:
    """Build a receipt from EEG-only measurements; all gates are derived.

    ``criterion_a_epileptiform_discharge_receipt_sha256`` is a compatibility
    alias for the project-only interictal-ED safety receipt.  It is accepted at
    the API boundary but is not serialized under the misleading historic
    field name.  New callers should supply the explicitly named project field
    and an ACNS discharge type separately.
    """

    event_id = _identifier(event_id, "event_id")
    producer_id = _identifier(producer_id, "producer_id")
    source_binding_sha256 = _sha256(
        source_binding_sha256, "source_binding_sha256"
    )
    if candidate_type not in {"spike", "sharp_wave", "other", "none"}:
        raise ValueError("candidate_type is unsupported")
    features = _ifcn_feature_map(ifcn_features)
    count = sum(features.values())
    target_receipt = _sha256(
        target_domain_qualification_receipt_sha256,
        "target_domain_qualification_receipt_sha256",
        nullable=True,
    )
    target_qualified = target_receipt is not None
    duration_ms = _finite(
        main_component_duration_ms,
        "main_component_duration_ms",
        nullable=True,
    )
    waveform_gate = (
        _waveform_duration_matches(candidate_type, duration_ms)
        and type(clearly_distinguished_from_background) is bool
        and clearly_distinguished_from_background
        and type(pointed_peak_at_clinical_display_scale) is bool
        and pointed_peak_at_clinical_display_scale
        and type(artifact_or_benign_variant_excluded) is bool
        and artifact_or_benign_variant_excluded
        and target_qualified
    )
    ied_gate = (
        count >= 5
        and features["physiologic_scalp_field"]
        and bool(artifact_or_benign_variant_excluded)
        and target_qualified
        and type(interictal_context_confirmed) is bool
        and interictal_context_confirmed
    )
    evolution = {
        "category": evolution_category,
        "unequivocal_sequential_changes": unequivocal_sequential_changes,
        "frequency_changes_same_direction": frequency_changes_same_direction,
        "minimum_frequency_step_hz": minimum_frequency_step_hz,
        "minimum_cycles_per_state": minimum_cycles_per_state,
        "distinct_standard_electrode_locations": distinct_standard_electrode_locations,
        "maximum_unchanged_gap_seconds": maximum_unchanged_gap_seconds,
        "amplitude_only_change": amplitude_only_change,
        "definite_evolution_gate_passed": False,
    }
    evolution["definite_evolution_gate_passed"] = _evolution_gate(evolution)
    pattern_duration = _finite(pattern_duration_seconds, "pattern_duration_seconds")
    discharge_rate = _finite(
        criterion_a_discharge_rate_hz,
        "criterion_a_discharge_rate_hz",
        nullable=True,
    )
    legacy_criterion_a_project_receipt = _sha256(
        criterion_a_epileptiform_discharge_receipt_sha256,
        "criterion_a_epileptiform_discharge_receipt_sha256",
        nullable=True,
    )
    named_criterion_a_project_receipt = _sha256(
        criterion_a_project_interictal_ed_qualification_receipt_sha256,
        "criterion_a_project_interictal_ed_qualification_receipt_sha256",
        nullable=True,
    )
    if (
        legacy_criterion_a_project_receipt is not None
        and named_criterion_a_project_receipt is not None
        and legacy_criterion_a_project_receipt
        != named_criterion_a_project_receipt
    ):
        raise ValueError(
            "legacy and explicitly named criterion A project receipts disagree"
        )
    criterion_a_project_receipt = (
        named_criterion_a_project_receipt
        if named_criterion_a_project_receipt is not None
        else legacy_criterion_a_project_receipt
    )
    if criterion_a_acns_discharge_type is None:
        criterion_a_discharge_type = (
            "epileptiform_discharge"
            if criterion_a_project_receipt is not None
            else "none"
        )
    else:
        criterion_a_discharge_type = str(criterion_a_acns_discharge_type)
    if criterion_a_discharge_type not in {
        *_ACNS_CRITERION_A_DISCHARGE_TYPES,
        "none",
    }:
        raise ValueError("criterion_a_acns_discharge_type is unsupported")
    criterion_a_discharge_duration_ms = _finite(
        criterion_a_acns_discharge_main_component_duration_ms,
        "criterion_a_acns_discharge_main_component_duration_ms",
        nullable=True,
    )
    if (
        criterion_a_discharge_duration_ms is not None
        and criterion_a_discharge_duration_ms <= 0.0
    ):
        raise ValueError(
            "criterion A discharge main-component duration must be positive"
        )
    criterion_a_source_morphology = (
        criterion_a_discharge_type in _ACNS_CRITERION_A_DISCHARGE_TYPES
    )
    criterion_a_acns_source = (
        pattern_duration >= 10.0
        and criterion_a_source_morphology
        and discharge_rate is not None
        and discharge_rate > 2.5
    )
    criterion_a_project_safety_gate = criterion_a_project_receipt is not None
    criterion_a_project = (
        criterion_a_acns_source and criterion_a_project_safety_gate
    )
    criterion_b = pattern_duration >= 10.0 and bool(
        evolution["definite_evolution_gate_passed"]
    )
    morphology = {
        "candidate_type": candidate_type,
        "main_component_duration_ms": duration_ms,
        "clearly_distinguished_from_background": clearly_distinguished_from_background,
        "pointed_peak_at_clinical_display_scale": pointed_peak_at_clinical_display_scale,
        "ifcn_features": features,
        "ifcn_criteria_met_count": count,
        "historically_suggested_candidate_threshold": 4,
        "kural_clinical_implementation_threshold": 5,
        "historically_suggested_four_of_six_candidate": count >= 4,
        "kural_five_of_six_high_specificity_gate": count >= 5,
        "target_domain_qualification_passed": target_qualified,
        "target_domain_qualification_receipt_sha256": target_receipt,
        "interictal_context_confirmed": interictal_context_confirmed,
        "waveform_term_gate_passed": waveform_gate,
        "ied_promotion_gate_passed": ied_gate,
        "ifcn_features_used_for_ictal_onset_or_evolution": False,
        "artifact_or_benign_variant_excluded": artifact_or_benign_variant_excluded,
        "promoted_term": candidate_type if waveform_gate else None,
    }
    seizure = {
        "pattern_duration_seconds": pattern_duration,
        "criterion_a_acns_discharge_type": criterion_a_discharge_type,
        "criterion_a_acns_discharge_main_component_duration_ms": (
            criterion_a_discharge_duration_ms
        ),
        "criterion_a_acns_source_morphology_gate_passed": (
            criterion_a_source_morphology
        ),
        "criterion_a_discharge_rate_hz": discharge_rate,
        "criterion_a_acns_source_definition_gate_passed": (
            criterion_a_acns_source
        ),
        "criterion_a_project_interictal_ed_qualification_receipt_sha256": (
            criterion_a_project_receipt
        ),
        "criterion_a_project_interictal_ed_safety_gate_passed": (
            criterion_a_project_safety_gate
        ),
        "criterion_a_project_conservative_gate_passed": criterion_a_project,
        "criterion_b_definite_evolution": bool(
            evolution["definite_evolution_gate_passed"]
        ),
        "criterion_b_gate_passed": criterion_b,
        "esz_gate_passed": criterion_a_project or criterion_b,
        "ten_second_rule_used": True,
        "electroclinical_seizure_claimed": False,
    }
    body: dict[str, Any] = {
        "schema_version": CLINICAL_EEG_TERM_QUALIFICATION_SCHEMA_VERSION,
        "receipt_id": "CONTENT-ADDRESS-PENDING",
        "event_id": event_id,
        "producer_id": producer_id,
        "morphology": morphology,
        "evolution": evolution,
        "electrographic_seizure": seizure,
        "qualified_terms": [],
        "scope_receipt": deepcopy(_SCOPE_RECEIPT),
        "policy_sha256": CLINICAL_TERM_QUALIFICATION_POLICY_SHA256,
        "source_binding_sha256": source_binding_sha256,
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    body["qualified_terms"] = _expected_qualified_terms(body)
    body["receipt_id"] = "EEGTERM-" + _canonical_sha256(body)[:20]
    body["receipt_sha256"] = _canonical_sha256(body)
    return validate_clinical_eeg_term_qualification(body)


__all__ = [
    "CLINICAL_EEG_TERM_QUALIFICATION_SCHEMA_VERSION",
    "CLINICAL_TERM_QUALIFICATION_POLICY",
    "CLINICAL_TERM_QUALIFICATION_POLICY_SHA256",
    "FORBIDDEN_EEG_ONLY_TERMS",
    "IFCN_FEATURE_NAMES",
    "PROTECTED_EEG_ONLY_TERMS",
    "build_clinical_eeg_term_qualification",
    "validate_clinical_eeg_term_qualification",
]
