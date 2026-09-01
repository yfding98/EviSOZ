"""Signal-bound ACNS-derived frequency-evolution candidate primitive.

This module implements only the frequency branch of the ACNS 2021 definite-
evolution *candidate* rule.  It does not discover seizures, qualify a clinical
term, or support a positive onset/SOZ claim.  Its purpose is to turn replayable
frequency-state measurements into an auditable course-only candidate without
letting amplitude, a single change point, a spectral peak, or offline future
context masquerade as onset evidence.

The encoded source rule is deliberately conservative:

* at least two consecutive, unequivocal frequency changes;
* both changes have the same direction;
* every conservative step is at least 0.5 Hz;
* every participating state contains at least three cycles; and
* an observed unchanged state of 300 seconds or longer breaks the sequence;
  an unobserved interstate gap is never treated as continuity.

These conditions are derived from Hirsch et al., ACNS standardized critical-
care EEG terminology 2021 (DOI 10.1097/WNP.0000000000000806).  The target
population here is long-term epilepsy EEG rather than critical-care EEG, so a
passing result remains ``model_candidate`` and requires an independent,
patient-disjoint target-domain qualification receipt before a report may call
it definite evolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Final, Iterable, Mapping, Sequence


ACNS_FREQUENCY_EVOLUTION_CANDIDATE_SCHEMA_VERSION: Final[str] = (
    "acns_derived_frequency_evolution_candidate_v1"
)
ACNS_FREQUENCY_EVOLUTION_POLICY_ID: Final[str] = (
    "acns_2021_frequency_evolution_candidate_conservative_v1"
)
_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_TOL = 1e-9
_ALLOWED_TEMPORAL_ROLES = frozenset({"onset_causal", "context_offline"})


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{context} must be a non-empty trimmed identifier")
    return value


def _finite(
    value: object,
    context: str,
    *,
    minimum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{context} must be >= {minimum}")
    return result


def _sha256(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_CHARACTERS for character in value)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _sha256s(values: Iterable[object], context: str) -> tuple[str, ...]:
    result = tuple(_sha256(item, context) for item in values)
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{context} must be non-empty and unique")
    return result


@dataclass(frozen=True)
class _FrozenMapping(Mapping[str, object]):
    """Small recursively immutable mapping that remains deepcopy-safe."""

    _items: tuple[tuple[str, object], ...]

    def __getitem__(self, key: str) -> object:
        for candidate, value in self._items:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self):
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __deepcopy__(self, _memo: dict[int, object]) -> _FrozenMapping:
        return self


def _deep_freeze(value: object) -> object:
    """Copy JSON-like state into recursively immutable containers."""

    if isinstance(value, Mapping):
        return _FrozenMapping(
            tuple(
                (str(key), _deep_freeze(item))
                for key, item in value.items()
            )
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _deep_thaw(value: object) -> object:
    """Return a detached JSON-compatible copy of frozen candidate content."""

    if isinstance(value, Mapping):
        return {str(key): _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class ACNSFrequencyEvolutionPolicy:
    """Frozen engineering interpretation of the frequency-evolution rule."""

    minimum_sequential_changes: int = 2
    minimum_frequency_step_hz: float = 0.5
    minimum_cycles_per_state: float = 3.0
    maximum_unchanged_gap_seconds_exclusive: float = 300.0
    minimum_usable_fraction: float = 0.8

    def __post_init__(self) -> None:
        if (
            isinstance(self.minimum_sequential_changes, bool)
            or not isinstance(self.minimum_sequential_changes, int)
            or self.minimum_sequential_changes < 2
        ):
            raise ValueError("minimum_sequential_changes must be an integer >= 2")
        step = _finite(
            self.minimum_frequency_step_hz,
            "minimum_frequency_step_hz",
            minimum=_TOL,
        )
        cycles = _finite(
            self.minimum_cycles_per_state,
            "minimum_cycles_per_state",
            minimum=_TOL,
        )
        gap = _finite(
            self.maximum_unchanged_gap_seconds_exclusive,
            "maximum_unchanged_gap_seconds_exclusive",
            minimum=_TOL,
        )
        usable = _finite(
            self.minimum_usable_fraction,
            "minimum_usable_fraction",
            minimum=0.0,
        )
        if usable > 1.0:
            raise ValueError("minimum_usable_fraction must be <= 1")
        if step < 0.5 - _TOL:
            raise ValueError(
                "ACNS-derived minimum_frequency_step_hz cannot be below 0.5"
            )
        if cycles < 3.0 - _TOL:
            raise ValueError(
                "ACNS-derived minimum_cycles_per_state cannot be below 3"
            )
        if gap > 300.0 + _TOL:
            raise ValueError(
                "ACNS-derived unchanged-state break cannot exceed 300 seconds"
            )
        if usable < 0.8 - _TOL:
            raise ValueError(
                "project quality minimum_usable_fraction cannot be below 0.8"
            )
        object.__setattr__(self, "minimum_frequency_step_hz", step)
        object.__setattr__(self, "minimum_cycles_per_state", cycles)
        object.__setattr__(
            self, "maximum_unchanged_gap_seconds_exclusive", gap
        )
        object.__setattr__(self, "minimum_usable_fraction", usable)

    def to_dict(self) -> dict[str, object]:
        return {
            "policy_id": ACNS_FREQUENCY_EVOLUTION_POLICY_ID,
            "minimum_sequential_changes": self.minimum_sequential_changes,
            "minimum_frequency_step_hz": self.minimum_frequency_step_hz,
            "minimum_cycles_per_state": self.minimum_cycles_per_state,
            "maximum_unchanged_gap_seconds_exclusive": (
                self.maximum_unchanged_gap_seconds_exclusive
            ),
            "minimum_usable_fraction": self.minimum_usable_fraction,
            "unequivocal_step_semantics": (
                "nonoverlapping_frequency_uncertainty_intervals_with_"
                "conservative_edge_separation"
            ),
            "cycle_support_semantics": (
                "longest_contiguous_usable_seconds_times_lower_frequency_bound"
            ),
            "unobserved_interstate_gap_counts_as_continuity": False,
            "amplitude_only_change_counts": False,
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class SignalBoundFrequencyState:
    """One replayable, non-clinical frequency state on physical record time."""

    state_id: str
    recording_interval_seconds: tuple[float, float]
    dominant_frequency_hz: float
    frequency_uncertainty_interval_hz: tuple[float, float]
    effective_bandwidth_hz: tuple[float, float]
    usable_fraction: float
    longest_contiguous_usable_seconds: float
    quality_eligible: bool
    source_view_id: str
    source_temporal_role: str
    future_sample_access: bool
    source_binding_sha256: str
    measurement_sha256: str
    raw_dependency_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "state_id", _identifier(self.state_id, "state_id"))
        if len(self.recording_interval_seconds) != 2:
            raise ValueError("recording_interval_seconds must contain two values")
        start = _finite(
            self.recording_interval_seconds[0],
            "recording_interval_seconds[0]",
            minimum=0.0,
        )
        stop = _finite(
            self.recording_interval_seconds[1],
            "recording_interval_seconds[1]",
            minimum=0.0,
        )
        if stop <= start + _TOL:
            raise ValueError("frequency state interval must have positive duration")
        object.__setattr__(self, "recording_interval_seconds", (start, stop))

        dominant = _finite(
            self.dominant_frequency_hz,
            "dominant_frequency_hz",
            minimum=_TOL,
        )
        if len(self.frequency_uncertainty_interval_hz) != 2:
            raise ValueError(
                "frequency_uncertainty_interval_hz must contain two values"
            )
        lower = _finite(
            self.frequency_uncertainty_interval_hz[0],
            "frequency_uncertainty_interval_hz[0]",
            minimum=_TOL,
        )
        upper = _finite(
            self.frequency_uncertainty_interval_hz[1],
            "frequency_uncertainty_interval_hz[1]",
            minimum=_TOL,
        )
        if upper < lower or dominant < lower - _TOL or dominant > upper + _TOL:
            raise ValueError("dominant frequency must lie inside its uncertainty interval")
        object.__setattr__(self, "dominant_frequency_hz", dominant)
        object.__setattr__(
            self, "frequency_uncertainty_interval_hz", (lower, upper)
        )

        if len(self.effective_bandwidth_hz) != 2:
            raise ValueError("effective_bandwidth_hz must contain two values")
        band_low = _finite(
            self.effective_bandwidth_hz[0],
            "effective_bandwidth_hz[0]",
            minimum=0.0,
        )
        band_high = _finite(
            self.effective_bandwidth_hz[1],
            "effective_bandwidth_hz[1]",
            minimum=_TOL,
        )
        if band_high <= band_low + _TOL:
            raise ValueError("effective bandwidth must be non-empty")
        object.__setattr__(self, "effective_bandwidth_hz", (band_low, band_high))

        usable = _finite(self.usable_fraction, "usable_fraction", minimum=0.0)
        if usable > 1.0:
            raise ValueError("usable_fraction must be <= 1")
        longest_usable = _finite(
            self.longest_contiguous_usable_seconds,
            "longest_contiguous_usable_seconds",
            minimum=0.0,
        )
        total_usable = self.duration_seconds * usable
        if longest_usable > total_usable + _TOL:
            raise ValueError(
                "longest contiguous usable support cannot exceed total usable time"
            )
        if type(self.quality_eligible) is not bool:
            raise TypeError("quality_eligible must be boolean")
        object.__setattr__(self, "usable_fraction", usable)
        object.__setattr__(
            self, "longest_contiguous_usable_seconds", longest_usable
        )
        object.__setattr__(
            self,
            "source_view_id",
            _identifier(self.source_view_id, "source_view_id"),
        )
        if self.source_temporal_role not in _ALLOWED_TEMPORAL_ROLES:
            raise ValueError("frequency state has an unsupported temporal role")
        if type(self.future_sample_access) is not bool:
            raise TypeError("future_sample_access must be boolean")
        if (
            self.source_temporal_role == "onset_causal"
            and self.future_sample_access
        ):
            raise ValueError("onset-causal frequency state cannot access future samples")
        if (
            self.source_temporal_role == "context_offline"
            and not self.future_sample_access
        ):
            raise ValueError("offline context must declare future-sample access")
        object.__setattr__(
            self,
            "source_binding_sha256",
            _sha256(self.source_binding_sha256, "source_binding_sha256"),
        )
        object.__setattr__(
            self,
            "measurement_sha256",
            _sha256(self.measurement_sha256, "measurement_sha256"),
        )
        object.__setattr__(
            self,
            "raw_dependency_sha256s",
            _sha256s(self.raw_dependency_sha256s, "raw_dependency_sha256s"),
        )

    @property
    def duration_seconds(self) -> float:
        return self.recording_interval_seconds[1] - self.recording_interval_seconds[0]

    @property
    def conservative_cycle_count(self) -> float:
        return (
            self.longest_contiguous_usable_seconds
            * self.frequency_uncertainty_interval_hz[0]
        )

    def opportunity_reasons(
        self, policy: ACNSFrequencyEvolutionPolicy
    ) -> tuple[str, ...]:
        reasons: list[str] = []
        lower, upper = self.frequency_uncertainty_interval_hz
        band_low, band_high = self.effective_bandwidth_hz
        if lower < band_low - _TOL or upper > band_high + _TOL:
            reasons.append("frequency_uncertainty_outside_effective_bandwidth")
        if self.usable_fraction + _TOL < policy.minimum_usable_fraction:
            reasons.append("insufficient_usable_fraction")
        if not self.quality_eligible:
            reasons.append("quality_or_artifact_gate_failed")
        if self.conservative_cycle_count + _TOL < policy.minimum_cycles_per_state:
            reasons.append("fewer_than_minimum_conservative_cycles")
        return tuple(sorted(reasons))

    def to_dict(self, policy: ACNSFrequencyEvolutionPolicy) -> dict[str, object]:
        reasons = self.opportunity_reasons(policy)
        return {
            "state_id": self.state_id,
            "recording_interval_seconds": list(self.recording_interval_seconds),
            "duration_seconds": self.duration_seconds,
            "dominant_frequency_hz": self.dominant_frequency_hz,
            "frequency_uncertainty_interval_hz": list(
                self.frequency_uncertainty_interval_hz
            ),
            "effective_bandwidth_hz": list(self.effective_bandwidth_hz),
            "conservative_cycle_count": self.conservative_cycle_count,
            "usable_fraction": self.usable_fraction,
            "longest_contiguous_usable_seconds": (
                self.longest_contiguous_usable_seconds
            ),
            "quality_eligible": self.quality_eligible,
            "evaluation_opportunity": not reasons,
            "opportunity_reason_codes": list(reasons),
            "source_view_id": self.source_view_id,
            "source_temporal_role": self.source_temporal_role,
            "future_sample_access": self.future_sample_access,
            "source_binding_sha256": self.source_binding_sha256,
            "measurement_sha256": self.measurement_sha256,
            "raw_dependency_sha256s": list(self.raw_dependency_sha256s),
        }


def _transition(
    left: SignalBoundFrequencyState,
    right: SignalBoundFrequencyState,
    *,
    policy: ACNSFrequencyEvolutionPolicy,
) -> dict[str, object]:
    gap = right.recording_interval_seconds[0] - left.recording_interval_seconds[1]
    if gap < -_TOL:
        raise ValueError("frequency state intervals must not overlap")
    left_lower, left_upper = left.frequency_uncertainty_interval_hz
    right_lower, right_upper = right.frequency_uncertainty_interval_hz
    upward = right_lower - left_upper
    downward = left_lower - right_upper
    direction = "none"
    conservative_step = 0.0
    if upward >= policy.minimum_frequency_step_hz - _TOL:
        direction = "increase"
        conservative_step = upward
    elif downward >= policy.minimum_frequency_step_hz - _TOL:
        direction = "decrease"
        conservative_step = downward
    reasons: list[str] = []
    if gap > _TOL:
        reasons.append("interstate_gap_has_no_continuity_evidence")
    if gap >= policy.maximum_unchanged_gap_seconds_exclusive - _TOL:
        reasons.append("interstate_gap_reaches_policy_unchanged_break")
    if left.opportunity_reasons(policy) or right.opportunity_reasons(policy):
        reasons.append("state_evaluation_opportunity_failed")
    if direction == "none":
        reasons.append("frequency_change_not_unequivocal_or_below_0_5_hz")
    eligible = not reasons
    return {
        "from_state_id": left.state_id,
        "to_state_id": right.state_id,
        "direction": direction,
        "conservative_frequency_step_hz": conservative_step,
        "interstate_gap_seconds": max(0.0, gap),
        "eligible_change": eligible,
        "reason_codes": sorted(reasons),
    }


def _candidate_runs(
    states: Sequence[SignalBoundFrequencyState],
    transitions: Sequence[Mapping[str, object]],
    *,
    policy: ACNSFrequencyEvolutionPolicy,
) -> list[tuple[int, int]]:
    """Return inclusive transition-index runs satisfying the frozen rule."""

    runs: list[tuple[int, int]] = []
    start = 0
    while start < len(transitions):
        first = transitions[start]
        if not bool(first["eligible_change"]):
            start += 1
            continue
        direction = str(first["direction"])
        stop = start
        while stop + 1 < len(transitions):
            next_transition = transitions[stop + 1]
            shared_state = states[stop + 1]
            if (
                not bool(next_transition["eligible_change"])
                or str(next_transition["direction"]) != direction
                or shared_state.duration_seconds
                >= policy.maximum_unchanged_gap_seconds_exclusive - _TOL
            ):
                break
            stop += 1
        if stop - start + 1 >= policy.minimum_sequential_changes:
            runs.append((start, stop))
        start = max(start + 1, stop + 1)
    return runs


@dataclass(frozen=True)
class ACNSFrequencyEvolutionCandidate:
    """Immutable course-only output with a content-addressed receipt."""

    event_id: str
    source_binding_sha256: str
    policy_sha256: str
    status: str
    reason_codes: tuple[str, ...]
    states: tuple[Mapping[str, object], ...]
    transitions: tuple[Mapping[str, object], ...]
    selected_state_ids: tuple[str, ...]
    selected_transition_indices: tuple[int, ...]
    direction: str | None
    minimum_conservative_frequency_step_hz: float | None
    minimum_conservative_cycles_per_state: float | None
    maximum_selected_unchanged_seconds: float | None
    raw_dependency_sha256s: tuple[str, ...]
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", _identifier(self.event_id, "event_id"))
        object.__setattr__(
            self,
            "source_binding_sha256",
            _sha256(self.source_binding_sha256, "source_binding_sha256"),
        )
        object.__setattr__(
            self,
            "policy_sha256",
            _sha256(self.policy_sha256, "policy_sha256"),
        )
        state_rows = tuple(self.states)
        transition_rows = tuple(self.transitions)
        if not state_rows or not all(isinstance(item, Mapping) for item in state_rows):
            raise ValueError("candidate states must be a non-empty mapping sequence")
        if not all(isinstance(item, Mapping) for item in transition_rows):
            raise ValueError("candidate transitions must be a mapping sequence")
        object.__setattr__(
            self,
            "states",
            tuple(_deep_freeze(item) for item in state_rows),
        )
        object.__setattr__(
            self,
            "transitions",
            tuple(_deep_freeze(item) for item in transition_rows),
        )
        selected_state_ids = tuple(
            _identifier(item, "selected_state_ids")
            for item in self.selected_state_ids
        )
        selected_transition_indices = tuple(self.selected_transition_indices)
        if any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in selected_transition_indices
        ) or len(set(selected_transition_indices)) != len(
            selected_transition_indices
        ):
            raise ValueError(
                "selected_transition_indices must be unique non-negative integers"
            )
        object.__setattr__(self, "selected_state_ids", selected_state_ids)
        object.__setattr__(
            self, "selected_transition_indices", selected_transition_indices
        )
        if self.status not in {"present", "uncertain", "not_evaluable"}:
            raise ValueError("frequency evolution candidate status is unsupported")
        if not self.reason_codes:
            raise ValueError("frequency evolution candidate needs reason codes")
        if tuple(sorted(set(self.reason_codes))) != self.reason_codes:
            raise ValueError("reason_codes must be unique and sorted")
        if self.status == "present":
            if (
                len(self.selected_transition_indices) < 2
                or len(self.selected_state_ids)
                != len(self.selected_transition_indices) + 1
                or self.direction not in {"increase", "decrease"}
                or self.minimum_conservative_frequency_step_hz is None
                or self.minimum_conservative_cycles_per_state is None
                or self.maximum_selected_unchanged_seconds is None
            ):
                raise ValueError("present candidate lacks a complete selected sequence")
        elif (
            self.selected_state_ids
            or self.selected_transition_indices
            or self.direction is not None
            or self.minimum_conservative_frequency_step_hz is not None
            or self.minimum_conservative_cycles_per_state is not None
            or self.maximum_selected_unchanged_seconds is not None
        ):
            raise ValueError("non-present candidate cannot carry a selected sequence")
        object.__setattr__(
            self,
            "raw_dependency_sha256s",
            _sha256s(self.raw_dependency_sha256s, "raw_dependency_sha256s"),
        )
        object.__setattr__(self, "receipt_sha256", _canonical_sha256(self.to_dict(False)))

    def to_dict(self, include_receipt: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": ACNS_FREQUENCY_EVOLUTION_CANDIDATE_SCHEMA_VERSION,
            "event_id": self.event_id,
            "source_binding_sha256": self.source_binding_sha256,
            "policy_id": ACNS_FREQUENCY_EVOLUTION_POLICY_ID,
            "policy_sha256": self.policy_sha256,
            "assertion_level": "model_candidate",
            "status": self.status,
            "controlled_term": "acns_derived_definite_frequency_evolution_candidate",
            "clinical_term_qualified": False,
            "target_domain_qualification_required": True,
            "intrinsic_evidence_role": "course_only",
            "onset_support_eligible": False,
            "soz_support_eligible": False,
            "amplitude_only_change_used": False,
            "reason_codes": list(self.reason_codes),
            "states": _deep_thaw(self.states),
            "transitions": _deep_thaw(self.transitions),
            "selected_state_ids": list(self.selected_state_ids),
            "selected_transition_indices": list(self.selected_transition_indices),
            "direction": self.direction,
            "minimum_conservative_frequency_step_hz": (
                self.minimum_conservative_frequency_step_hz
            ),
            "minimum_conservative_cycles_per_state": (
                self.minimum_conservative_cycles_per_state
            ),
            "maximum_selected_unchanged_seconds": (
                self.maximum_selected_unchanged_seconds
            ),
            "raw_dependency_sha256s": list(self.raw_dependency_sha256s),
            "scope_receipt": {
                "eeg_signal_only": True,
                "edf_annotations_used": False,
                "spreadsheet_used": False,
                "doctor_labels_used": False,
                "clinical_text_used": False,
                "amplitude_only_evolution_allowed": False,
                "offline_or_future_context_may_support_onset": False,
            },
        }
        if include_receipt:
            payload["receipt_sha256"] = self.receipt_sha256
        return payload


def build_acns_frequency_evolution_candidate(
    *,
    event_id: str,
    states: Sequence[SignalBoundFrequencyState],
    policy: ACNSFrequencyEvolutionPolicy | None = None,
) -> ACNSFrequencyEvolutionCandidate:
    """Build a conservative frequency-evolution candidate from measurements."""

    event_id = _identifier(event_id, "event_id")
    if policy is None:
        policy = ACNSFrequencyEvolutionPolicy()
    if not isinstance(policy, ACNSFrequencyEvolutionPolicy):
        raise TypeError("policy must be ACNSFrequencyEvolutionPolicy")
    if not states or not all(isinstance(item, SignalBoundFrequencyState) for item in states):
        raise TypeError("states must be a non-empty SignalBoundFrequencyState sequence")
    ordered = tuple(
        sorted(
            states,
            key=lambda item: (
                item.recording_interval_seconds[0],
                item.recording_interval_seconds[1],
                item.state_id,
            ),
        )
    )
    if len({item.state_id for item in ordered}) != len(ordered):
        raise ValueError("frequency state IDs must be unique")
    bindings = {item.source_binding_sha256 for item in ordered}
    if len(bindings) != 1:
        raise ValueError("frequency states cannot cross canonical source bindings")
    view_signatures = {
        (
            item.source_view_id,
            item.source_temporal_role,
            item.future_sample_access,
        )
        for item in ordered
    }
    if len(view_signatures) != 1:
        raise ValueError(
            "frequency states require one compatible signal view and time role"
        )
    measurement_sha256s = [item.measurement_sha256 for item in ordered]
    if len(measurement_sha256s) != len(set(measurement_sha256s)):
        raise ValueError("frequency states cannot reuse one measurement receipt")
    for left, right in zip(ordered, ordered[1:]):
        if right.recording_interval_seconds[0] < left.recording_interval_seconds[1] - _TOL:
            raise ValueError("frequency state intervals must be non-overlapping")

    transition_rows = tuple(
        _transition(left, right, policy=policy)
        for left, right in zip(ordered, ordered[1:])
    )
    runs = _candidate_runs(ordered, transition_rows, policy=policy)
    state_rows = tuple(item.to_dict(policy) for item in ordered)
    dependency_hashes = tuple(
        sorted(
            {
                dependency
                for item in ordered
                for dependency in item.raw_dependency_sha256s
            }
        )
    )

    if runs:
        # Prefer the longest run, then the earliest physical sequence.
        selected_start, selected_stop = sorted(
            runs,
            key=lambda item: (-(item[1] - item[0] + 1), item[0], item[1]),
        )[0]
        selected_transitions = transition_rows[selected_start : selected_stop + 1]
        selected_states = ordered[selected_start : selected_stop + 2]
        minimum_step = min(
            float(item["conservative_frequency_step_hz"])
            for item in selected_transitions
        )
        minimum_cycles = min(
            item.conservative_cycle_count for item in selected_states
        )
        unchanged_candidates = [
            float(item["interstate_gap_seconds"])
            for item in selected_transitions
        ] + [item.duration_seconds for item in selected_states[1:-1]]
        return ACNSFrequencyEvolutionCandidate(
            event_id=event_id,
            source_binding_sha256=next(iter(bindings)),
            policy_sha256=policy.sha256,
            status="present",
            reason_codes=(
                "acns_derived_frequency_evolution_candidate_rule_passed",
                "target_domain_clinical_qualification_not_supplied",
            ),
            states=state_rows,
            transitions=tuple(dict(item) for item in transition_rows),
            selected_state_ids=tuple(item.state_id for item in selected_states),
            selected_transition_indices=tuple(
                range(selected_start, selected_stop + 1)
            ),
            direction=str(selected_transitions[0]["direction"]),
            minimum_conservative_frequency_step_hz=minimum_step,
            minimum_conservative_cycles_per_state=minimum_cycles,
            maximum_selected_unchanged_seconds=max(unchanged_candidates),
            raw_dependency_sha256s=dependency_hashes,
        )

    evaluable_count = sum(
        not item.opportunity_reasons(policy) for item in ordered
    )
    if evaluable_count == 0:
        status = "not_evaluable"
        reasons = ("no_evaluable_signal_bound_frequency_states",)
    elif evaluable_count < policy.minimum_sequential_changes + 1:
        status = "uncertain"
        reasons = (
            "insufficient_evaluable_states_for_two_sequential_changes",
            "no_sensitivity_receipt_for_absent_evolution_assertion",
        )
    else:
        status = "uncertain"
        reasons = (
            "no_two_same_direction_unequivocal_frequency_changes",
            "no_sensitivity_receipt_for_absent_evolution_assertion",
        )
    return ACNSFrequencyEvolutionCandidate(
        event_id=event_id,
        source_binding_sha256=next(iter(bindings)),
        policy_sha256=policy.sha256,
        status=status,
        reason_codes=tuple(sorted(reasons)),
        states=state_rows,
        transitions=tuple(dict(item) for item in transition_rows),
        selected_state_ids=(),
        selected_transition_indices=(),
        direction=None,
        minimum_conservative_frequency_step_hz=None,
        minimum_conservative_cycles_per_state=None,
        maximum_selected_unchanged_seconds=None,
        raw_dependency_sha256s=dependency_hashes,
    )


__all__ = [
    "ACNS_FREQUENCY_EVOLUTION_CANDIDATE_SCHEMA_VERSION",
    "ACNS_FREQUENCY_EVOLUTION_POLICY_ID",
    "ACNSFrequencyEvolutionCandidate",
    "ACNSFrequencyEvolutionPolicy",
    "SignalBoundFrequencyState",
    "build_acns_frequency_evolution_candidate",
]
