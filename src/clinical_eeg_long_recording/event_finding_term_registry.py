"""Closed vocabulary for event-level EEG Finding terms.

The wire contracts deliberately keep ``finding.term`` as a machine-readable
clinical/scientific atom.  A free identifier is not safe at the report
boundary: an unregistered word could otherwise bypass the capability and
per-event decision receipts and later be lexicalized as a clinical fact.

This registry is intentionally small.  Adding a term is a policy change and
requires an explicit family assignment.  Protected clinical terms may only
appear after the per-event ``clinical_eeg_term_qualification_v1`` decision
gate; descriptive measurements and research candidates remain non-clinical.
"""

from __future__ import annotations

from typing import Mapping

from .clinical_term_qualification import PROTECTED_EEG_ONLY_TERMS


# The values are the event Finding families in which the term is meaningful.
# HFO/LVFA markers remain allowed in a spectral candidate because the v1
# event contract separately checks the actual high-frequency bandwidth gate.
EVENT_FINDING_TERM_FAMILIES: Mapping[str, frozenset[str]] = {
    # Protected EEG-only clinical terminology.
    "spike": frozenset({"morphology"}),
    "sharp_wave": frozenset({"morphology"}),
    "interictal_epileptiform_discharge": frozenset({"morphology"}),
    "definite_evolution": frozenset({"evolution"}),
    "electrographic_seizure": frozenset({"evolution"}),
    # Explicit non-clinical candidates / deterministic measurements.
    "theta_band_power_increase": frozenset({"spectral"}),
    "rhythmic_theta_activity": frozenset({"rhythm"}),
    "left_temporal_recruitment_candidate": frozenset({"spatial_recruitment"}),
    "generalized_synchronous_onset": frozenset({"spatial_field"}),
    "bilateral_synchronous_onset_field": frozenset({"spatial_field"}),
    "left_temporal_onset_candidate": frozenset({"spatial_field"}),
    "competing_right_temporal_field_candidate": frozenset({"spatial_field"}),
    "deterministic_signal_usable_fraction": frozenset({"quality"}),
    "deterministic_frequency_rhythm_amplitude_profile": frozenset({"spectral"}),
    "reference_specific_spatial_field_change_candidate": frozenset(
        {"spatial_field"}
    ),
    "deterministic_multivariate_change_point_candidate": frozenset({"evolution"}),
    "reference_specific_spatial_field_measurement": frozenset({"spatial_field"}),
    "deterministic_later_involvement_candidate": frozenset(
        {"spatial_recruitment"}
    ),
    "ictal_sharp_contoured_component_candidate": frozenset({"morphology"}),
    "hfo_rate": frozenset(
        {"spectral", "rhythm", "morphology", "high_frequency"}
    ),
    "high_frequency_activity": frozenset(
        {"spectral", "rhythm", "morphology", "high_frequency"}
    ),
    "high_frequency_oscillation": frozenset(
        {"spectral", "rhythm", "morphology", "high_frequency"}
    ),
    "low_voltage_fast_activity": frozenset(
        {"spectral", "rhythm", "morphology", "high_frequency"}
    ),
}

_INTERICTAL_ONLY_TERMS = frozenset(
    {"spike", "sharp_wave", "interictal_epileptiform_discharge"}
)
_ICTAL_SHARP_COMPONENT_TERMS = frozenset(
    {"ictal_sharp_contoured_component_candidate"}
)


PROTECTED_TERM_SURFACE_ZH: Mapping[str, str] = {
    "spike": "棘波",
    "sharp_wave": "尖波",
    "interictal_epileptiform_discharge": "发作间期癫痫样放电",
    "definite_evolution": "明确演变",
    "electrographic_seizure": "电图发作",
}


def validate_event_finding_term(
    term: object,
    *,
    family: object,
    assertion_level: object,
    context: str,
) -> str:
    """Validate one Finding term against the closed family/level registry."""

    if not isinstance(term, str) or not term or term != term.strip():
        raise ValueError(f"{context}.term must be a non-empty controlled identifier")
    allowed_families = EVENT_FINDING_TERM_FAMILIES.get(term)
    if allowed_families is None:
        raise ValueError(f"{context}.term is absent from the controlled term registry")
    if str(family) not in allowed_families:
        raise ValueError(
            f"{context}.term {term!r} is not registered for family {family!r}"
        )
    protected = term in PROTECTED_EEG_ONLY_TERMS
    qualified = assertion_level == "clinically_qualified"
    if protected != qualified:
        if protected:
            raise ValueError(
                f"{context}.term {term!r} requires clinically_qualified assertion level"
            )
        raise ValueError(
            f"{context}.clinically_qualified term {term!r} is not a protected "
            "per-event decision term"
        )
    return term


def validate_event_finding_term_context(
    term: object,
    *,
    intrinsic_evidence_role: object,
    context: str,
) -> None:
    """Enforce the fail-closed interictal/ictal sharp-term partition."""

    term_id = str(term)
    role = str(intrinsic_evidence_role)
    if term_id in _INTERICTAL_ONLY_TERMS and role != "non_event_context":
        raise ValueError(
            f"{context}.term {term_id!r} is interictal-only and requires "
            "non_event_context"
        )
    if term_id in _ICTAL_SHARP_COMPONENT_TERMS and role != "early_context":
        raise ValueError(
            f"{context}.term {term_id!r} is an ictal sharp-component candidate "
            "and requires early_context"
        )


__all__ = [
    "EVENT_FINDING_TERM_FAMILIES",
    "PROTECTED_TERM_SURFACE_ZH",
    "validate_event_finding_term",
    "validate_event_finding_term_context",
]
