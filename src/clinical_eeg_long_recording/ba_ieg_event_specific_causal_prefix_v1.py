"""Event-specific minimal sufficient causal prefix shadow for BA-IEG.

The frozen K3 arm intentionally uses a fixed three-second typed-onset horizon.
That is easy to audit, but it can truncate slowly evolving seizures and can
retain unnecessary course samples for abrupt seizures.  This additive module
implements the ``H_e`` challenger described by the v1.5 method audit.

For one detected occurrence, a caller performs complete causal recomputation
on a registered sequence of nested physical-time prefixes.  The selector
locks the first prefix for which all EEG-derived closure gates pass for a
registered number of consecutive recomputations.  The selected prefix is
event specific; it is not a length classifier and it never examines a future
suffix that is outside the prefix being assessed.

The module is deliberately fail closed:

* every assessment is bound to final-left-closure, native EEG and full
  recomputation receipts;
* detector score, query geometry, late course state, annotations, labels and
  reports cannot become positive rank evidence;
* prefixes must be nested, use a pre-registered horizon grid, and be supplied
  in physical-time order;
* a seizure that never satisfies the closure gates exposes no positive typed
  onset opportunity; and
* the implementation remains a development shadow until patient-held-out
  comparison against K1/K3/K5 is completed.

This is a software/causal-permission component.  It is not a calibrated
clinical onset detector, a trained SOZ model, or performance evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Final, Mapping, Sequence

import torch
from torch import nn

from .ba_ieg_permission_split_segmental_state_model_v1 import (
    BAIEGCausalTypedUnitTrace,
)
from .ba_ieg_shallow_causal_typed_unit_head_v1 import (
    BAIEGShallowCausalTypedUnitHeadOutput,
    BAIEGShallowCausalTypedUnitOnsetHead,
)


BA_IEG_EVENT_SPECIFIC_CAUSAL_PREFIX_METHOD_ID_V1: Final[str] = (
    "ba_ieg_event_specific_minimal_sufficient_causal_prefix_v1"
)
BA_IEG_EVENT_SPECIFIC_CAUSAL_PREFIX_PRIMARY_ADMITTED_V1: Final[bool] = False
BA_IEG_EVENT_SPECIFIC_CAUSAL_PREFIX_STATUS_LOCKED_V1: Final[str] = (
    "locked_development_shadow"
)
BA_IEG_EVENT_SPECIFIC_CAUSAL_PREFIX_STATUS_NEVER_LOCKED_V1: Final[str] = (
    "never_locked_unresolved_ita_only"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TOL: Final[float] = 1e-8
_ALLOWED_BLOCKERS: Final[frozenset[str]] = frozenset(
    {
        "typed_censor_open",
        "record_edge_open",
        "impassable_qc_gap",
        "neighbor_event_mixing_risk",
        "measurement_opportunity_deficit",
        "reference_family_unavailable",
    }
)
_SOURCE_FIREWALL: Final[dict[str, bool]] = {
    "native_eeg_samples_used": True,
    "allowlisted_acquisition_metadata_used": True,
    "detector_score_used_for_positive_rank": False,
    "query_geometry_used_for_positive_rank": False,
    "late_course_state_used": False,
    "samples_beyond_assessed_prefix_used": False,
    "edf_annotation_used": False,
    "spreadsheet_or_excel_used": False,
    "doctor_label_or_report_used": False,
    "clinical_history_used": False,
    "video_behavior_or_sleep_used": False,
    "ecg_emg_eog_used": False,
    "llm_or_knowledge_base_used": False,
}


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _identifier(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 512
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"{context} must be a non-empty trimmed identifier")
    return value


def _sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _finite(
    value: object,
    context: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    if minimum is not None and result < minimum - _TOL:
        raise ValueError(f"{context} must be >= {minimum}")
    if maximum is not None and result > maximum + _TOL:
        raise ValueError(f"{context} must be <= {maximum}")
    return result


def _interval(value: Sequence[float], context: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise TypeError(f"{context} must contain exactly two values")
    start = _finite(value[0], f"{context}[0]", minimum=0.0)
    stop = _finite(value[1], f"{context}[1]", minimum=0.0)
    if stop <= start + _TOL:
        raise ValueError(f"{context} must have positive duration")
    return start, stop


def _sorted_unique_strings(value: Sequence[str], context: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{context} must be a sequence")
    result = tuple(_identifier(item, context) for item in value)
    if result != tuple(sorted(set(result))):
        raise ValueError(f"{context} must be sorted and unique")
    return result


@dataclass(frozen=True, slots=True)
class BAIEGEventSpecificCausalPrefixPolicyV1:
    """Source-development policy for the unadmitted ``H_e`` shadow."""

    registered_horizons_seconds: tuple[float, ...]
    maximum_boundary_tail_mass: float
    maximum_earliest_field_js_divergence: float
    maximum_reference_instability: float
    minimum_onset_trigger_atom_count: int
    stable_recomputations_required: int
    threshold_registry_receipt_sha256: str
    threshold_selection_split: str = "source_dev"
    primary_admitted: bool = (
        BA_IEG_EVENT_SPECIFIC_CAUSAL_PREFIX_PRIMARY_ADMITTED_V1
    )
    policy_receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        horizons = tuple(
            _finite(value, "registered_horizon", minimum=0.0)
            for value in self.registered_horizons_seconds
        )
        if not horizons or horizons != tuple(sorted(set(horizons))):
            raise ValueError(
                "registered horizons must be a non-empty strictly increasing grid"
            )
        if horizons[0] <= 0.0:
            raise ValueError("registered horizons must be positive")
        object.__setattr__(self, "registered_horizons_seconds", horizons)
        _finite(
            self.maximum_boundary_tail_mass,
            "maximum_boundary_tail_mass",
            minimum=0.0,
            maximum=1.0,
        )
        _finite(
            self.maximum_earliest_field_js_divergence,
            "maximum_earliest_field_js_divergence",
            minimum=0.0,
            maximum=1.0,
        )
        _finite(
            self.maximum_reference_instability,
            "maximum_reference_instability",
            minimum=0.0,
            maximum=1.0,
        )
        if (
            isinstance(self.minimum_onset_trigger_atom_count, bool)
            or not isinstance(self.minimum_onset_trigger_atom_count, int)
            or self.minimum_onset_trigger_atom_count < 1
        ):
            raise ValueError("minimum onset-trigger atom count must be positive")
        if (
            isinstance(self.stable_recomputations_required, bool)
            or not isinstance(self.stable_recomputations_required, int)
            or self.stable_recomputations_required < 2
        ):
            raise ValueError("at least two stable recomputations are required")
        _sha256(
            self.threshold_registry_receipt_sha256,
            "threshold_registry_receipt_sha256",
        )
        if self.threshold_selection_split != "source_dev":
            raise ValueError("H_e thresholds must be selected on source-dev")
        if self.primary_admitted is not False:
            raise ValueError("the event-specific causal prefix is not primary-admitted")
        body = {
            "schema_version": "ba_ieg_event_specific_causal_prefix_policy_v1",
            "method_id": BA_IEG_EVENT_SPECIFIC_CAUSAL_PREFIX_METHOD_ID_V1,
            "registered_horizons_seconds": list(horizons),
            "maximum_boundary_tail_mass": float(
                self.maximum_boundary_tail_mass
            ),
            "maximum_earliest_field_js_divergence": float(
                self.maximum_earliest_field_js_divergence
            ),
            "maximum_reference_instability": float(
                self.maximum_reference_instability
            ),
            "minimum_onset_trigger_atom_count": (
                self.minimum_onset_trigger_atom_count
            ),
            "stable_recomputations_required": self.stable_recomputations_required,
            "threshold_registry_receipt_sha256": (
                self.threshold_registry_receipt_sha256
            ),
            "threshold_selection_split": self.threshold_selection_split,
            "primary_admitted": self.primary_admitted,
            "source_firewall": _SOURCE_FIREWALL,
        }
        object.__setattr__(self, "policy_receipt_sha256", _canonical_sha256(body))


@dataclass(frozen=True, slots=True)
class BAIEGEventSpecificCausalPrefixAssessmentV1:
    """One full causal recomputation on one registered nested prefix."""

    event_id: str
    recording_id: str
    prefix_index: int
    prefix_interval_recording_seconds: tuple[float, float]
    possible_onset_interval_recording_seconds: tuple[float, float]
    horizon_seconds_after_possible_onset: float
    boundary_tail_mass: float
    earliest_field_js_divergence_to_previous: float | None
    reference_instability: float
    onset_trigger_atom_count: int
    unresolved_blocker_codes: tuple[str, ...]
    qc_evaluable: bool
    final_left_closure_receipt_sha256: str
    native_eeg_prefix_receipt_sha256: str
    full_recompute_receipt_sha256: str
    causal_atom_roster_receipt_sha256: str
    reference_family_roster_receipt_sha256: str
    source_firewall: Mapping[str, bool] = field(
        default_factory=lambda: dict(_SOURCE_FIREWALL)
    )
    assessment_receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _identifier(self.event_id, "event_id")
        _identifier(self.recording_id, "recording_id")
        if (
            isinstance(self.prefix_index, bool)
            or not isinstance(self.prefix_index, int)
            or self.prefix_index < 0
        ):
            raise ValueError("prefix_index must be a non-negative integer")
        prefix = _interval(
            self.prefix_interval_recording_seconds,
            "prefix_interval_recording_seconds",
        )
        onset = _interval(
            self.possible_onset_interval_recording_seconds,
            "possible_onset_interval_recording_seconds",
        )
        if onset[0] < prefix[0] - _TOL or onset[1] > prefix[1] + _TOL:
            raise ValueError("possible onset interval must lie inside its prefix")
        horizon = _finite(
            self.horizon_seconds_after_possible_onset,
            "horizon_seconds_after_possible_onset",
            minimum=0.0,
        )
        if abs((prefix[1] - onset[1]) - horizon) > _TOL:
            raise ValueError(
                "prefix stop must equal possible-onset stop plus registered horizon"
            )
        _finite(
            self.boundary_tail_mass,
            "boundary_tail_mass",
            minimum=0.0,
            maximum=1.0,
        )
        if self.earliest_field_js_divergence_to_previous is not None:
            _finite(
                self.earliest_field_js_divergence_to_previous,
                "earliest_field_js_divergence_to_previous",
                minimum=0.0,
                maximum=1.0,
            )
        _finite(
            self.reference_instability,
            "reference_instability",
            minimum=0.0,
            maximum=1.0,
        )
        if (
            isinstance(self.onset_trigger_atom_count, bool)
            or not isinstance(self.onset_trigger_atom_count, int)
            or self.onset_trigger_atom_count < 0
        ):
            raise ValueError("onset_trigger_atom_count must be non-negative")
        blockers = _sorted_unique_strings(
            self.unresolved_blocker_codes, "unresolved_blocker_codes"
        )
        unknown = set(blockers).difference(_ALLOWED_BLOCKERS)
        if unknown:
            raise ValueError(f"unsupported causal-prefix blocker codes: {sorted(unknown)}")
        if type(self.qc_evaluable) is not bool:
            raise TypeError("qc_evaluable must be boolean")
        for name in (
            "final_left_closure_receipt_sha256",
            "native_eeg_prefix_receipt_sha256",
            "full_recompute_receipt_sha256",
            "causal_atom_roster_receipt_sha256",
            "reference_family_roster_receipt_sha256",
        ):
            _sha256(getattr(self, name), name)
        if dict(self.source_firewall) != _SOURCE_FIREWALL:
            raise ValueError("event-specific prefix source firewall drifted")
        object.__setattr__(
            self,
            "source_firewall",
            MappingProxyType(dict(_SOURCE_FIREWALL)),
        )
        body = {
            "schema_version": "ba_ieg_event_specific_causal_prefix_assessment_v1",
            "method_id": BA_IEG_EVENT_SPECIFIC_CAUSAL_PREFIX_METHOD_ID_V1,
            "event_id": self.event_id,
            "recording_id": self.recording_id,
            "prefix_index": self.prefix_index,
            "prefix_interval_recording_seconds": list(prefix),
            "possible_onset_interval_recording_seconds": list(onset),
            "horizon_seconds_after_possible_onset": horizon,
            "boundary_tail_mass": float(self.boundary_tail_mass),
            "earliest_field_js_divergence_to_previous": (
                None
                if self.earliest_field_js_divergence_to_previous is None
                else float(self.earliest_field_js_divergence_to_previous)
            ),
            "reference_instability": float(self.reference_instability),
            "onset_trigger_atom_count": self.onset_trigger_atom_count,
            "unresolved_blocker_codes": list(blockers),
            "qc_evaluable": self.qc_evaluable,
            "final_left_closure_receipt_sha256": (
                self.final_left_closure_receipt_sha256
            ),
            "native_eeg_prefix_receipt_sha256": (
                self.native_eeg_prefix_receipt_sha256
            ),
            "full_recompute_receipt_sha256": self.full_recompute_receipt_sha256,
            "causal_atom_roster_receipt_sha256": (
                self.causal_atom_roster_receipt_sha256
            ),
            "reference_family_roster_receipt_sha256": (
                self.reference_family_roster_receipt_sha256
            ),
            "source_firewall": _SOURCE_FIREWALL,
        }
        object.__setattr__(
            self, "assessment_receipt_sha256", _canonical_sha256(body)
        )


@dataclass(frozen=True, slots=True)
class BAIEGEventSpecificCausalPrefixDecisionV1:
    """Content-bound lock/never-lock result for one event."""

    event_id: str
    recording_id: str
    status: str
    policy_receipt_sha256: str
    assessment_receipt_sha256s: tuple[str, ...]
    selected_prefix_index: int | None
    selected_prefix_interval_recording_seconds: tuple[float, float] | None
    selected_possible_onset_interval_recording_seconds: (
        tuple[float, float] | None
    )
    selected_horizon_seconds: float | None
    reason_codes: tuple[str, ...]
    primary_admitted: bool = (
        BA_IEG_EVENT_SPECIFIC_CAUSAL_PREFIX_PRIMARY_ADMITTED_V1
    )
    positive_typed_rank_authorized: bool = False
    right_course_may_rewrite_locked_prefix_or_rank: bool = False
    decision_receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _identifier(self.event_id, "event_id")
        _identifier(self.recording_id, "recording_id")
        if self.status not in {
            BA_IEG_EVENT_SPECIFIC_CAUSAL_PREFIX_STATUS_LOCKED_V1,
            BA_IEG_EVENT_SPECIFIC_CAUSAL_PREFIX_STATUS_NEVER_LOCKED_V1,
        }:
            raise ValueError("unsupported event-specific prefix decision status")
        _sha256(self.policy_receipt_sha256, "policy_receipt_sha256")
        receipts = tuple(
            _sha256(value, "assessment_receipt_sha256")
            for value in self.assessment_receipt_sha256s
        )
        if not receipts or len(receipts) != len(set(receipts)):
            raise ValueError("assessment receipts must be non-empty and unique")
        reasons = _sorted_unique_strings(self.reason_codes, "reason_codes")
        if not reasons:
            raise ValueError("event-specific prefix decision requires reason codes")
        locked = self.status == BA_IEG_EVENT_SPECIFIC_CAUSAL_PREFIX_STATUS_LOCKED_V1
        selected_fields = (
            self.selected_prefix_index,
            self.selected_prefix_interval_recording_seconds,
            self.selected_possible_onset_interval_recording_seconds,
            self.selected_horizon_seconds,
        )
        if locked and any(value is None for value in selected_fields):
            raise ValueError("locked event-specific prefix lacks selected fields")
        if not locked and any(value is not None for value in selected_fields):
            raise ValueError("never-locked event cannot carry selected prefix fields")
        if locked:
            assert self.selected_prefix_index is not None
            if self.selected_prefix_index < 0:
                raise ValueError("selected prefix index must be non-negative")
            prefix = _interval(
                self.selected_prefix_interval_recording_seconds,  # type: ignore[arg-type]
                "selected_prefix_interval_recording_seconds",
            )
            onset = _interval(
                self.selected_possible_onset_interval_recording_seconds,  # type: ignore[arg-type]
                "selected_possible_onset_interval_recording_seconds",
            )
            horizon = _finite(
                self.selected_horizon_seconds,
                "selected_horizon_seconds",
                minimum=0.0,
            )
        else:
            prefix = None
            onset = None
            horizon = None
        if self.primary_admitted is not False:
            raise ValueError("H_e shadow cannot be marked primary-admitted")
        if self.positive_typed_rank_authorized is not False:
            raise ValueError("H_e shadow cannot authorize positive typed rank")
        if self.right_course_may_rewrite_locked_prefix_or_rank is not False:
            raise ValueError("right-course evidence cannot rewrite the causal lock")
        body = {
            "schema_version": "ba_ieg_event_specific_causal_prefix_decision_v1",
            "method_id": BA_IEG_EVENT_SPECIFIC_CAUSAL_PREFIX_METHOD_ID_V1,
            "event_id": self.event_id,
            "recording_id": self.recording_id,
            "status": self.status,
            "policy_receipt_sha256": self.policy_receipt_sha256,
            "assessment_receipt_sha256s": list(receipts),
            "selected_prefix_index": self.selected_prefix_index,
            "selected_prefix_interval_recording_seconds": (
                None if prefix is None else list(prefix)
            ),
            "selected_possible_onset_interval_recording_seconds": (
                None if onset is None else list(onset)
            ),
            "selected_horizon_seconds": horizon,
            "reason_codes": list(reasons),
            "primary_admitted": self.primary_admitted,
            "positive_typed_rank_authorized": self.positive_typed_rank_authorized,
            "right_course_may_rewrite_locked_prefix_or_rank": (
                self.right_course_may_rewrite_locked_prefix_or_rank
            ),
        }
        object.__setattr__(
            self, "decision_receipt_sha256", _canonical_sha256(body)
        )


def _assessment_failures(
    row: BAIEGEventSpecificCausalPrefixAssessmentV1,
    policy: BAIEGEventSpecificCausalPrefixPolicyV1,
) -> tuple[str, ...]:
    failures: list[str] = []
    if row.boundary_tail_mass > policy.maximum_boundary_tail_mass + _TOL:
        failures.append("boundary_tail_open")
    divergence = row.earliest_field_js_divergence_to_previous
    if divergence is None or (
        divergence > policy.maximum_earliest_field_js_divergence + _TOL
    ):
        failures.append("earliest_field_unstable")
    if row.reference_instability > policy.maximum_reference_instability + _TOL:
        failures.append("reference_unstable")
    if row.onset_trigger_atom_count < policy.minimum_onset_trigger_atom_count:
        failures.append("onset_trigger_atom_deficit")
    if not row.qc_evaluable:
        failures.append("qc_not_evaluable")
    failures.extend(row.unresolved_blocker_codes)
    return tuple(sorted(set(failures)))


def select_ba_ieg_event_specific_causal_prefix_v1(
    assessments: Sequence[BAIEGEventSpecificCausalPrefixAssessmentV1],
    policy: BAIEGEventSpecificCausalPrefixPolicyV1,
) -> BAIEGEventSpecificCausalPrefixDecisionV1:
    """Lock the first event-specific prefix with repeated evidence closure."""

    if not isinstance(policy, BAIEGEventSpecificCausalPrefixPolicyV1):
        raise TypeError("event-specific prefix selection requires a validated policy")
    rows = tuple(assessments)
    if not rows or not all(
        isinstance(row, BAIEGEventSpecificCausalPrefixAssessmentV1) for row in rows
    ):
        raise ValueError("event-specific prefix selection requires assessments")
    event_id = rows[0].event_id
    recording_id = rows[0].recording_id
    closure_receipt = rows[0].final_left_closure_receipt_sha256
    if [row.prefix_index for row in rows] != list(range(len(rows))):
        raise ValueError("prefix assessments must be contiguous and ordered")
    if any(row.event_id != event_id for row in rows):
        raise ValueError("prefix assessments mix event identities")
    if any(row.recording_id != recording_id for row in rows):
        raise ValueError("prefix assessments mix recording identities")
    if any(
        row.final_left_closure_receipt_sha256 != closure_receipt for row in rows
    ):
        raise ValueError("prefix assessments mix final-left-closure receipts")

    prefix_start = rows[0].prefix_interval_recording_seconds[0]
    previous_stop = -math.inf
    previous_onset_start = -math.inf
    for index, row in enumerate(rows):
        start, stop = row.prefix_interval_recording_seconds
        onset_start, _ = row.possible_onset_interval_recording_seconds
        if abs(start - prefix_start) > _TOL or stop <= previous_stop + _TOL:
            raise ValueError("prefix assessments must form a strictly nested sequence")
        if onset_start + _TOL < previous_onset_start:
            raise ValueError("possible-onset start cannot move earlier after left lock")
        if not any(
            abs(row.horizon_seconds_after_possible_onset - registered) <= _TOL
            for registered in policy.registered_horizons_seconds
        ):
            raise ValueError("prefix assessment uses an unregistered horizon")
        if index == 0 and row.earliest_field_js_divergence_to_previous is not None:
            raise ValueError("first prefix cannot have previous-prefix divergence")
        if index > 0 and row.earliest_field_js_divergence_to_previous is None:
            raise ValueError("later prefixes require previous-prefix divergence")
        previous_stop = stop
        previous_onset_start = onset_start

    consecutive = 0
    selected: BAIEGEventSpecificCausalPrefixAssessmentV1 | None = None
    selected_position = -1
    final_failures: tuple[str, ...] = ()
    for position, row in enumerate(rows):
        failures = _assessment_failures(row, policy)
        final_failures = failures
        if failures:
            consecutive = 0
            continue
        consecutive += 1
        if consecutive >= policy.stable_recomputations_required:
            selected = row
            selected_position = position
            break

    if selected is None:
        reasons = tuple(
            sorted(
                set(final_failures)
                | {
                    "no_registered_prefix_satisfied_repeated_evidence_closure",
                    "positive_typed_rank_remains_unresolved",
                }
            )
        )
        return BAIEGEventSpecificCausalPrefixDecisionV1(
            event_id=event_id,
            recording_id=recording_id,
            status=BA_IEG_EVENT_SPECIFIC_CAUSAL_PREFIX_STATUS_NEVER_LOCKED_V1,
            policy_receipt_sha256=policy.policy_receipt_sha256,
            assessment_receipt_sha256s=tuple(
                row.assessment_receipt_sha256 for row in rows
            ),
            selected_prefix_index=None,
            selected_prefix_interval_recording_seconds=None,
            selected_possible_onset_interval_recording_seconds=None,
            selected_horizon_seconds=None,
            reason_codes=reasons,
        )

    return BAIEGEventSpecificCausalPrefixDecisionV1(
        event_id=event_id,
        recording_id=recording_id,
        status=BA_IEG_EVENT_SPECIFIC_CAUSAL_PREFIX_STATUS_LOCKED_V1,
        policy_receipt_sha256=policy.policy_receipt_sha256,
        assessment_receipt_sha256s=tuple(
            row.assessment_receipt_sha256 for row in rows[: selected_position + 1]
        ),
        selected_prefix_index=selected.prefix_index,
        selected_prefix_interval_recording_seconds=(
            selected.prefix_interval_recording_seconds
        ),
        selected_possible_onset_interval_recording_seconds=(
            selected.possible_onset_interval_recording_seconds
        ),
        selected_horizon_seconds=selected.horizon_seconds_after_possible_onset,
        reason_codes=(
            "all_registered_closure_gates_passed",
            "first_repeatedly_stable_prefix_locked",
            "late_course_rewrite_forbidden",
        ),
    )


@dataclass(frozen=True, slots=True)
class BAIEGEventSpecificCausalPrefixGateResultV1:
    """Causal trace after applying one event-specific decision per event."""

    decisions: tuple[BAIEGEventSpecificCausalPrefixDecisionV1, ...]
    locked_event_mask: torch.Tensor
    locked_prefix_interval_seconds: torch.Tensor
    locked_possible_onset_interval_seconds: torch.Tensor
    locked_prefix_group_mask: torch.Tensor
    typed_onset_group_mask: torch.Tensor
    gated_trace: BAIEGCausalTypedUnitTrace
    method_id: str = BA_IEG_EVENT_SPECIFIC_CAUSAL_PREFIX_METHOD_ID_V1
    primary_admitted: bool = (
        BA_IEG_EVENT_SPECIFIC_CAUSAL_PREFIX_PRIMARY_ADMITTED_V1
    )

    def verify(self) -> None:
        self.gated_trace.verify_shapes()
        batch_size, group_count = self.gated_trace.group_mask.shape
        if len(self.decisions) != batch_size:
            raise ValueError("event-specific decision roster does not match batch")
        expected = {
            "locked_event_mask": (self.locked_event_mask, (batch_size,), torch.bool),
            "locked_prefix_interval_seconds": (
                self.locked_prefix_interval_seconds,
                (batch_size, 2),
                None,
            ),
            "locked_possible_onset_interval_seconds": (
                self.locked_possible_onset_interval_seconds,
                (batch_size, 2),
                None,
            ),
            "locked_prefix_group_mask": (
                self.locked_prefix_group_mask,
                (batch_size, group_count),
                torch.bool,
            ),
            "typed_onset_group_mask": (
                self.typed_onset_group_mask,
                (batch_size, group_count),
                torch.bool,
            ),
        }
        for name, (value, shape, dtype) in expected.items():
            if tuple(value.shape) != shape:
                raise ValueError(f"{name} shape drifted")
            if dtype is not None and value.dtype != dtype:
                raise TypeError(f"{name} dtype drifted")
        if self.method_id != BA_IEG_EVENT_SPECIFIC_CAUSAL_PREFIX_METHOD_ID_V1:
            raise ValueError("event-specific causal-prefix method drifted")
        if self.primary_admitted is not False:
            raise ValueError("event-specific causal-prefix shadow was promoted")
        if torch.any(self.typed_onset_group_mask & ~self.locked_prefix_group_mask):
            raise ValueError("typed onset opportunity exceeds its selected prefix")
        if torch.any(
            self.gated_trace.typed_unit_time_mask
            & ~self.typed_onset_group_mask.unsqueeze(-1)
        ):
            raise ValueError("gated typed-unit trace exceeds H_e")
        unlocked = ~self.locked_event_mask
        if torch.any(self.locked_prefix_group_mask[unlocked]) or torch.any(
            self.gated_trace.typed_unit_mask[unlocked]
        ):
            raise ValueError("never-locked event exposes typed onset opportunity")


def build_ba_ieg_event_specific_causal_prefix_gate_v1(
    trace: BAIEGCausalTypedUnitTrace,
    decisions: Sequence[BAIEGEventSpecificCausalPrefixDecisionV1],
) -> BAIEGEventSpecificCausalPrefixGateResultV1:
    """Mask a causal typed-unit trace to independently selected ``H_e``."""

    if not isinstance(trace, BAIEGCausalTypedUnitTrace):
        raise TypeError("event-specific causal-prefix gate requires a causal trace")
    trace.verify_shapes()
    rows = tuple(decisions)
    batch_size, _ = trace.group_mask.shape
    if len(rows) != batch_size or not all(
        isinstance(row, BAIEGEventSpecificCausalPrefixDecisionV1) for row in rows
    ):
        raise ValueError("one validated H_e decision is required per trace event")
    if tuple(row.event_id for row in rows) != tuple(trace.event_ids):
        raise ValueError("H_e decision event order does not match the trace")
    if tuple(row.recording_id for row in rows) != tuple(trace.recording_ids):
        raise ValueError("H_e decision recording order does not match the trace")

    device = trace.group_mask.device
    dtype = trace.group_boundary_bounds_seconds.dtype
    locked = torch.zeros(batch_size, dtype=torch.bool, device=device)
    prefix_intervals = torch.zeros(batch_size, 2, dtype=dtype, device=device)
    onset_intervals = torch.zeros_like(prefix_intervals)
    prefix_mask = torch.zeros_like(trace.group_mask)
    typed_mask = torch.zeros_like(trace.group_mask)

    for batch_index, decision in enumerate(rows):
        if decision.status != BA_IEG_EVENT_SPECIFIC_CAUSAL_PREFIX_STATUS_LOCKED_V1:
            continue
        assert decision.selected_prefix_interval_recording_seconds is not None
        assert (
            decision.selected_possible_onset_interval_recording_seconds is not None
        )
        prefix = decision.selected_prefix_interval_recording_seconds
        onset = decision.selected_possible_onset_interval_recording_seconds
        bounds = trace.group_boundary_bounds_seconds[batch_index]
        valid = trace.group_mask[batch_index]
        prefix_overlap = (
            valid
            & (bounds[:, 1] > prefix[0] + _TOL)
            & (bounds[:, 0] < prefix[1] - _TOL)
        )
        onset_overlap = (
            prefix_overlap
            & (bounds[:, 1] > onset[0] + _TOL)
            & (bounds[:, 0] < prefix[1] - _TOL)
        )
        if not bool(prefix_overlap.any()) or not bool(onset_overlap.any()):
            raise ValueError("selected H_e has no overlap with the causal trace")
        locked[batch_index] = True
        prefix_intervals[batch_index] = torch.tensor(
            prefix, dtype=dtype, device=device
        )
        onset_intervals[batch_index] = torch.tensor(
            onset, dtype=dtype, device=device
        )
        prefix_mask[batch_index] = prefix_overlap
        typed_mask[batch_index] = onset_overlap

    gated_onset_mass = torch.where(
        prefix_mask,
        trace.global_onset_boundary_mass,
        torch.zeros_like(trace.global_onset_boundary_mass),
    )
    gated_no_onset = (
        torch.ones_like(trace.global_no_onset_within_support_mass)
        - trace.global_left_censor_state_mass.sum(dim=1)
        - gated_onset_mass.sum(dim=1)
    ).clamp_min(0.0)
    gated_typed_time_mask = (
        trace.typed_unit_time_mask & typed_mask.unsqueeze(-1)
    )
    gated_trace = replace(
        trace,
        global_onset_boundary_mass=gated_onset_mass,
        global_no_onset_within_support_mass=gated_no_onset,
        typed_unit_time_mask=gated_typed_time_mask,
        typed_unit_mask=gated_typed_time_mask.any(dim=1),
    )
    gated_trace.verify_shapes()
    result = BAIEGEventSpecificCausalPrefixGateResultV1(
        decisions=rows,
        locked_event_mask=locked,
        locked_prefix_interval_seconds=prefix_intervals,
        locked_possible_onset_interval_seconds=onset_intervals,
        locked_prefix_group_mask=prefix_mask,
        typed_onset_group_mask=typed_mask,
        gated_trace=gated_trace,
    )
    result.verify()
    return result


@dataclass(frozen=True, slots=True)
class BAIEGEventSpecificCausalPrefixTypedUnitHeadOutputV1:
    gate: BAIEGEventSpecificCausalPrefixGateResultV1
    typed_unit: BAIEGShallowCausalTypedUnitHeadOutput
    method_id: str = BA_IEG_EVENT_SPECIFIC_CAUSAL_PREFIX_METHOD_ID_V1
    primary_admitted: bool = (
        BA_IEG_EVENT_SPECIFIC_CAUSAL_PREFIX_PRIMARY_ADMITTED_V1
    )


class BAIEGEventSpecificCausalPrefixTypedUnitOnsetHeadV1(nn.Module):
    """Apply an event-specific H_e gate before the frozen shallow rank head."""

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        bottleneck_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.typed_unit_head = BAIEGShallowCausalTypedUnitOnsetHead(
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            dropout=dropout,
        )

    def forward(
        self,
        trace: BAIEGCausalTypedUnitTrace,
        decisions: Sequence[BAIEGEventSpecificCausalPrefixDecisionV1],
    ) -> BAIEGEventSpecificCausalPrefixTypedUnitHeadOutputV1:
        gate = build_ba_ieg_event_specific_causal_prefix_gate_v1(trace, decisions)
        return BAIEGEventSpecificCausalPrefixTypedUnitHeadOutputV1(
            gate=gate,
            typed_unit=self.typed_unit_head(gate.gated_trace),
        )


__all__ = [
    "BA_IEG_EVENT_SPECIFIC_CAUSAL_PREFIX_METHOD_ID_V1",
    "BA_IEG_EVENT_SPECIFIC_CAUSAL_PREFIX_PRIMARY_ADMITTED_V1",
    "BA_IEG_EVENT_SPECIFIC_CAUSAL_PREFIX_STATUS_LOCKED_V1",
    "BA_IEG_EVENT_SPECIFIC_CAUSAL_PREFIX_STATUS_NEVER_LOCKED_V1",
    "BAIEGEventSpecificCausalPrefixAssessmentV1",
    "BAIEGEventSpecificCausalPrefixDecisionV1",
    "BAIEGEventSpecificCausalPrefixGateResultV1",
    "BAIEGEventSpecificCausalPrefixPolicyV1",
    "BAIEGEventSpecificCausalPrefixTypedUnitHeadOutputV1",
    "BAIEGEventSpecificCausalPrefixTypedUnitOnsetHeadV1",
    "build_ba_ieg_event_specific_causal_prefix_gate_v1",
    "select_ba_ieg_event_specific_causal_prefix_v1",
]
