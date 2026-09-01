"""EEG-only event-outcome vocabulary and legacy normalization.

``no_demonstrable_scalp_ictal_change`` is a legacy v3 wire value.  Its wording
presupposes an independently anchored clinical event, which the EEG-only path
does not possess.  The value is retained solely so old payloads remain
auditable; every claim or render route must normalize it to the narrower
statement that the queried detector candidate did not meet the pre-registered
scalp electrographic ictal-pattern qualification gate.
"""

from __future__ import annotations


LEGACY_NO_DEMONSTRABLE_SCALP_ICTAL_CHANGE = "no_demonstrable_scalp_ictal_change"
EEG_ONLY_CANDIDATE_NOT_QUALIFIED_REASON_CODE = "candidate_not_qualified_as_scalp_electrographic_ictal_pattern_within_queried_support"
EEG_ONLY_UNQUALIFIED_CANDIDATE_OUTCOME = "candidate_only"


def normalize_eeg_only_event_outcome(outcome: str) -> str:
    """Return the only EEG-only-safe public code for the legacy outcome."""

    value = str(outcome)
    if value == LEGACY_NO_DEMONSTRABLE_SCALP_ICTAL_CHANGE:
        return EEG_ONLY_UNQUALIFIED_CANDIDATE_OUTCOME
    return value


def event_outcome_uses_deprecated_clinical_anchor_wording(outcome: str) -> bool:
    """Identify a legacy source code that must never reach report surface."""

    return str(outcome) == LEGACY_NO_DEMONSTRABLE_SCALP_ICTAL_CHANGE


__all__ = [
    "EEG_ONLY_CANDIDATE_NOT_QUALIFIED_REASON_CODE",
    "EEG_ONLY_UNQUALIFIED_CANDIDATE_OUTCOME",
    "LEGACY_NO_DEMONSTRABLE_SCALP_ICTAL_CHANGE",
    "event_outcome_uses_deprecated_clinical_anchor_wording",
    "normalize_eeg_only_event_outcome",
]
