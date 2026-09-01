"""Fail-closed source-only supervision for the BA-IEG segmental model.

The neural forward pass remains target-free.  This module binds public-source
or synthetic boundary targets only after a detector-independent event input
and EEG-only acquisition context have been content-addressed.  It implements
two deliberately different objectives:

* a causal onset interval/censoring NLL over the model's future-free hazard
  distribution; and
* an offline exact constrained-path NLL whose causal start/onset potentials
  are stop-gradient constants.

The latter uses the same exact segmental forward--backward partition for the
unconstrained and target-compatible lattices.  Targets can constrain event
presence, terminal bout class, start/end censoring, the primary observed
onset interval, and the offset interval.  ``unknown``/``not_evaluable`` never
becomes a negative target.  No private, spreadsheet, embedded EDF annotation,
clinical-text, report, or production route exists here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Final, Mapping, Sequence

import torch

from .ba_ieg_permission_split_segmental_state_model_v1 import (
    BA_IEG_SEGMENTAL_CAUSAL_TRANSITION_INDICES,
    BA_IEG_SEGMENTAL_TARGET_AUTHORITIES,
    BA_IEG_SEGMENTAL_TARGET_SCHEMA_VERSION,
    BAIEGPermissionSplitSegmentalStateOutput,
    BAIEGSegmentalBoundaryContext,
)
from .ba_ieg_segmental_dp_kernel_v1 import (
    SEGMENTAL_OFFSET_EDGE_INDICES_V1,
    SEGMENTAL_ONSET_EDGE_INDICES_V1,
    SegmentalPathConstraintsV1,
    SegmentalPotentialsV1,
)
from .ba_ieg_segmental_forward_backward_v1 import (
    build_lognormal_segment_duration_log_scores_v1,
    run_exact_segmental_forward_backward_v1,
)
from .ba_ieg_training_contract import BAIEGCollatedEventBatch


BA_IEG_PERMISSION_SPLIT_SEGMENTAL_SUPERVISION_ID: Final[
    str
] = "ba_ieg_permission_split_segmental_supervision_v1"
BA_IEG_SEGMENTAL_TARGET_BUNDLE_SCHEMA_VERSION: Final[
    str
] = "ba_ieg_segmental_boundary_target_bundle_v1"
BA_IEG_SEGMENTAL_LOSS_CONTRACT_SCHEMA_VERSION: Final[
    str
] = "ba_ieg_permission_split_segmental_loss_contract_v1"
BA_IEG_SEGMENTAL_LATTICE_TARGET_PROJECTION_SCHEMA_VERSION: Final[
    str
] = "ba_ieg_segmental_frozen_lattice_target_projection_v1"
BA_IEG_SEGMENTAL_LATTICE_TARGET_PROJECTION_METHOD_ID: Final[
    str
] = "post_forward_right_closed_physical_support_projection_v1"

_EVENT_STATUSES: Final[frozenset[str]] = frozenset(
    {"present", "absent", "not_evaluable"}
)
_ONSET_STATUSES: Final[frozenset[str]] = frozenset(
    {"observed_interval", "left_censored", "not_observed", "not_evaluable"}
)
_OFFSET_STATUSES: Final[frozenset[str]] = frozenset(
    {"observed_interval", "right_censored", "not_observed", "not_evaluable"}
)
_BOUT_STATUSES: Final[frozenset[str]] = frozenset(
    {"zero_bouts", "single_bout", "two_or_more_bouts", "not_evaluable"}
)
_OPTIMIZATION_ROLE_TO_SPLIT: Final[Mapping[str, str]] = {
    "optimize": "source_train",
    "calibrate": "source_dev",
    "evaluate": "source_eval",
}
_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_TIME_TOLERANCE_SECONDS: Final[float] = 1e-6

_LOSS_POLICY: Final[dict[str, object]] = {
    "schema_version": BA_IEG_SEGMENTAL_LOSS_CONTRACT_SCHEMA_VERSION,
    "causal_objective": "future_free_interval_or_censor_probability_mass_nll",
    "offline_objective": "exact_logZ_minus_target_constrained_logZ",
    "raw_interval_projection_method": (
        BA_IEG_SEGMENTAL_LATTICE_TARGET_PROJECTION_METHOD_ID
    ),
    "raw_interval_is_mutated_or_midpoint_snapped": False,
    "projection_occurs_after_target_free_forward_and_lattice_freeze": True,
    "terminal_offset_without_post_boundary_opportunity": (
        "right_resolution_censor_only_when_frozen_context_authorizes_right_censor"
    ),
    "causal_start_and_onset_potentials_detached_in_offline_objective": True,
    "unknown_or_not_evaluable_is_negative": False,
    "target_may_enter_model_forward": False,
    "patient_equal_after_event_mean": True,
    "causal_weight": 1.0,
    "offline_weight": 1.0,
}

_LATTICE_TARGET_PROJECTION_SCOPE: Final[dict[str, object]] = {
    "projection_stage": "after_target_free_forward_and_lattice_freeze",
    "raw_annotation_interval_mutated": False,
    "nearest_midpoint_snapping_used": False,
    "causal_onset_assignment": (
        "positive_duration_overlap_with_right_closed_causal_support_bin"
    ),
    "offline_offset_assignment": (
        "positive_duration_overlap_with_right_closed_physical_lattice_cell"
    ),
    "quality_gap_or_missing_opportunity_used_as_negative": False,
    "model_logits_or_probabilities_used_for_projection": False,
    "reference_source_opened_by_projection": False,
    "target_available_to_model_forward": False,
    "public_tusz_authority_limited_to_event_and_weak_boundary": True,
    "morphology_rhythm_spatial_or_soz_gold_claimed": False,
}


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


BA_IEG_SEGMENTAL_LOSS_CONTRACT_SHA256: Final[str] = _canonical_sha256(_LOSS_POLICY)


def _identifier(value: object, name: str) -> str:
    text = str(value)
    if not text or text != text.strip():
        raise ValueError(f"{name} must be a non-empty trimmed identifier")
    return text


def _sha256(value: object, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(
        character not in _SHA256_CHARACTERS for character in text
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return text


def _interval_or_none(
    value: Sequence[float] | None,
    *,
    status: str,
    name: str,
) -> tuple[float, float] | None:
    if status == "observed_interval":
        if value is None or isinstance(value, (str, bytes)) or len(value) != 2:
            raise ValueError(f"{name} is required for observed_interval")
        start, stop = float(value[0]), float(value[1])
        if not math.isfinite(start) or not math.isfinite(stop) or stop < start:
            raise ValueError(f"{name} must be a finite closed interval")
        return start, stop
    if value is not None:
        raise ValueError(f"{name} must be None unless status is observed_interval")
    return None


@dataclass(frozen=True)
class BAIEGSegmentalTargetFirewallV1:
    """Explicitly deny every forbidden target or selection route."""

    target_used_as_model_input: bool = False
    target_conditioned_candidate_selection: bool = False
    target_conditioned_window_or_tokenization: bool = False
    embedded_edf_annotation_used: bool = False
    spreadsheet_used: bool = False
    private_doctor_label_used: bool = False
    private_report_or_clinical_text_used: bool = False
    video_or_semiology_used: bool = False
    sleep_activation_or_other_physiology_used: bool = False
    private_source_used: bool = False

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if getattr(self, name) is not False:
                raise ValueError(f"segmental target firewall violates {name}")

    def to_dict(self) -> dict[str, bool]:
        return {name: False for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class BAIEGSegmentalEventTargetV1:
    """One immutable physical-time supervision row for a frozen event."""

    event_id: str
    recording_id: str
    patient_uid: str
    model_split: str
    source_event_receipt_sha256: str
    adaptive_acquisition_receipt_sha256: str
    target_independent_candidate_roster_receipt_sha256: str
    source_reference_receipt_sha256: str
    authority: str
    event_status: str
    onset_status: str
    offset_status: str
    bout_count_status: str
    onset_interval_seconds: tuple[float, float] | None = None
    offset_interval_seconds: tuple[float, float] | None = None
    firewall: BAIEGSegmentalTargetFirewallV1 = field(
        default_factory=BAIEGSegmentalTargetFirewallV1
    )
    schema_version: str = BA_IEG_SEGMENTAL_TARGET_SCHEMA_VERSION
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("event_id", "recording_id", "patient_uid"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        if self.model_split not in {"source_train", "source_dev", "source_eval"}:
            raise ValueError("segmental targets are restricted to public source splits")
        if self.schema_version != BA_IEG_SEGMENTAL_TARGET_SCHEMA_VERSION:
            raise ValueError("segmental target schema drifted")
        if self.authority not in BA_IEG_SEGMENTAL_TARGET_AUTHORITIES:
            raise ValueError("segmental target authority is unsupported")
        if (
            self.authority == "synthetic_signal_injection"
            and self.model_split != "source_train"
        ):
            raise ValueError("synthetic optimization targets belong to source_train")
        if (
            self.authority == "source_development_eeg_expert_atomic_boundary"
            and self.model_split != "source_dev"
        ):
            raise ValueError("expert atomic boundary targets are source_dev-only")
        for name in (
            "source_event_receipt_sha256",
            "adaptive_acquisition_receipt_sha256",
            "target_independent_candidate_roster_receipt_sha256",
            "source_reference_receipt_sha256",
        ):
            _sha256(getattr(self, name), name)
        if self.event_status not in _EVENT_STATUSES:
            raise ValueError("event_status is unsupported")
        if self.onset_status not in _ONSET_STATUSES:
            raise ValueError("onset_status is unsupported")
        if self.offset_status not in _OFFSET_STATUSES:
            raise ValueError("offset_status is unsupported")
        if self.bout_count_status not in _BOUT_STATUSES:
            raise ValueError("bout_count_status is unsupported")
        onset_interval = _interval_or_none(
            self.onset_interval_seconds,
            status=self.onset_status,
            name="onset_interval_seconds",
        )
        offset_interval = _interval_or_none(
            self.offset_interval_seconds,
            status=self.offset_status,
            name="offset_interval_seconds",
        )
        if self.event_status == "absent":
            if (
                self.onset_status != "not_observed"
                or self.offset_status != "not_observed"
                or self.bout_count_status != "zero_bouts"
            ):
                raise ValueError(
                    "absent events require zero bouts and unobserved boundaries"
                )
        elif self.event_status == "not_evaluable":
            if (
                self.onset_status != "not_evaluable"
                or self.offset_status != "not_evaluable"
                or self.bout_count_status != "not_evaluable"
            ):
                raise ValueError(
                    "not-evaluable events cannot smuggle boundary negatives"
                )
        else:
            if self.onset_status not in {
                "observed_interval",
                "left_censored",
                "not_evaluable",
            }:
                raise ValueError("present-event onset status is inconsistent")
            if self.offset_status not in {
                "observed_interval",
                "right_censored",
                "not_evaluable",
            }:
                raise ValueError("present-event offset status is inconsistent")
            if self.bout_count_status == "zero_bouts":
                raise ValueError("present events cannot have zero bouts")
            if self.bout_count_status == "two_or_more_bouts" and (
                self.onset_status == "observed_interval"
                or self.offset_status == "observed_interval"
            ):
                raise ValueError(
                    "v1 interval constraints are primary/terminal single-bout targets; "
                    "multi-bout boundaries must remain not_evaluable"
                )
        if not isinstance(self.firewall, BAIEGSegmentalTargetFirewallV1):
            raise TypeError("firewall must be BAIEGSegmentalTargetFirewallV1")
        object.__setattr__(self, "onset_interval_seconds", onset_interval)
        object.__setattr__(self, "offset_interval_seconds", offset_interval)
        object.__setattr__(self, "receipt_sha256", self._compute_sha256())

    def _compute_sha256(self) -> str:
        return _canonical_sha256(
            {
                "schema_version": self.schema_version,
                "identity": {
                    "event_id": self.event_id,
                    "recording_id": self.recording_id,
                    "patient_uid": self.patient_uid,
                    "model_split": self.model_split,
                },
                "binding": {
                    "source_event_receipt_sha256": self.source_event_receipt_sha256,
                    "adaptive_acquisition_receipt_sha256": self.adaptive_acquisition_receipt_sha256,
                    "target_independent_candidate_roster_receipt_sha256": (
                        self.target_independent_candidate_roster_receipt_sha256
                    ),
                    "source_reference_receipt_sha256": self.source_reference_receipt_sha256,
                },
                "authority": self.authority,
                "event_status": self.event_status,
                "onset": {
                    "status": self.onset_status,
                    "interval_seconds": (
                        list(self.onset_interval_seconds)
                        if self.onset_interval_seconds is not None
                        else None
                    ),
                },
                "offset": {
                    "status": self.offset_status,
                    "interval_seconds": (
                        list(self.offset_interval_seconds)
                        if self.offset_interval_seconds is not None
                        else None
                    ),
                },
                "bout_count_status": self.bout_count_status,
                "firewall": self.firewall.to_dict(),
            }
        )

    def verify_integrity(self) -> None:
        if self.receipt_sha256 != self._compute_sha256():
            raise ValueError("segmental event target changed after registration")


@dataclass(frozen=True)
class BAIEGSegmentalTargetBundleV1:
    """Target tensor analogue kept independent from the model-input hash."""

    source_input_batch_sha256: str
    source_context_receipt_sha256: str
    target_independent_candidate_roster_receipt_sha256: str
    optimization_role: str
    targets: tuple[BAIEGSegmentalEventTargetV1, ...]
    model_split: str = field(init=False)
    receipt_sha256: str = field(init=False)
    schema_version: str = BA_IEG_SEGMENTAL_TARGET_BUNDLE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "source_input_batch_sha256",
            "source_context_receipt_sha256",
            "target_independent_candidate_roster_receipt_sha256",
        ):
            _sha256(getattr(self, name), name)
        if self.schema_version != BA_IEG_SEGMENTAL_TARGET_BUNDLE_SCHEMA_VERSION:
            raise ValueError("segmental target bundle schema drifted")
        if self.optimization_role not in _OPTIMIZATION_ROLE_TO_SPLIT:
            raise ValueError(
                "optimization_role must be optimize, calibrate or evaluate"
            )
        if not self.targets or not all(
            isinstance(target, BAIEGSegmentalEventTargetV1) for target in self.targets
        ):
            raise TypeError("target bundle requires event target rows")
        for target in self.targets:
            target.verify_integrity()
        if len({target.event_id for target in self.targets}) != len(self.targets):
            raise ValueError("target bundle event IDs must be unique")
        splits = {target.model_split for target in self.targets}
        if len(splits) != 1:
            raise ValueError("one target bundle cannot mix source splits")
        model_split = next(iter(splits))
        if _OPTIMIZATION_ROLE_TO_SPLIT[self.optimization_role] != model_split:
            raise ValueError("optimization role disagrees with the source split")
        if any(
            target.target_independent_candidate_roster_receipt_sha256
            != self.target_independent_candidate_roster_receipt_sha256
            for target in self.targets
        ):
            raise ValueError("target rows disagree with the frozen candidate roster")
        object.__setattr__(self, "model_split", model_split)
        object.__setattr__(self, "receipt_sha256", self._compute_sha256())

    def _compute_sha256(self) -> str:
        return _canonical_sha256(
            {
                "schema_version": self.schema_version,
                "source_input_batch_sha256": self.source_input_batch_sha256,
                "source_context_receipt_sha256": self.source_context_receipt_sha256,
                "target_independent_candidate_roster_receipt_sha256": (
                    self.target_independent_candidate_roster_receipt_sha256
                ),
                "optimization_role": self.optimization_role,
                "model_split": self.model_split,
                "event_target_receipt_sha256s": [
                    target.receipt_sha256 for target in self.targets
                ],
                "loss_contract_sha256": BA_IEG_SEGMENTAL_LOSS_CONTRACT_SHA256,
            }
        )

    def verify_integrity(self) -> None:
        for target in self.targets:
            target.verify_integrity()
        if self.receipt_sha256 != self._compute_sha256():
            raise ValueError("segmental target bundle changed after registration")


def build_ba_ieg_segmental_target_bundle_v1(
    batch: BAIEGCollatedEventBatch,
    context: BAIEGSegmentalBoundaryContext,
    targets: Sequence[BAIEGSegmentalEventTargetV1],
    *,
    optimization_role: str,
    target_independent_candidate_roster_receipt_sha256: str,
) -> BAIEGSegmentalTargetBundleV1:
    """Bind already-created public/synthetic targets after input freezing."""

    if not isinstance(batch, BAIEGCollatedEventBatch):
        raise TypeError("target bundle requires BAIEGCollatedEventBatch")
    if not isinstance(context, BAIEGSegmentalBoundaryContext):
        raise TypeError("target bundle requires BAIEGSegmentalBoundaryContext")
    context.verify_integrity()
    if context.source_input_batch_sha256 != batch.input_batch_sha256:
        raise ValueError("target context belongs to another model-input batch")
    rows = tuple(targets)
    if tuple(target.event_id for target in rows) != batch.event_ids:
        raise ValueError("target event order must exactly match the frozen input batch")
    for index, target in enumerate(rows):
        if (
            target.recording_id != batch.recording_ids[index]
            or target.patient_uid != batch.patient_uids[index]
            or target.model_split != batch.model_split
            or target.source_event_receipt_sha256
            != batch.input_event_receipt_sha256s[index]
            or target.adaptive_acquisition_receipt_sha256
            != context.adaptive_acquisition_receipt_sha256s[index]
        ):
            raise ValueError("segmental target identity or source binding drifted")
    return BAIEGSegmentalTargetBundleV1(
        source_input_batch_sha256=batch.input_batch_sha256,
        source_context_receipt_sha256=context.receipt_sha256,
        target_independent_candidate_roster_receipt_sha256=(
            target_independent_candidate_roster_receipt_sha256
        ),
        optimization_role=optimization_role,
        targets=rows,
    )


class BAIEGSegmentalTargetNotRepresentableError(ValueError):
    """The frozen signal lattice has no legal support for a target fact."""


def _json_interval(
    value: tuple[float, float] | None,
) -> list[float] | None:
    return list(value) if value is not None else None


def _support_rows_json(
    rows: tuple[tuple[int, float, float, float], ...],
) -> list[dict[str, object]]:
    return [
        {
            "boundary_index": index,
            "support_interval_seconds": [start, stop],
            "represented_boundary_seconds": boundary,
        }
        for index, start, stop, boundary in rows
    ]


@dataclass(frozen=True)
class BAIEGSegmentalLatticeTargetProjectionV1:
    """A replayable, post-forward projection of raw intervals to a frozen grid.

    The source ``BAIEGSegmentalEventTargetV1`` remains unchanged.  Masks in
    this object describe which already-existing target-free boundary supports
    can represent that raw interval.  They never add a boundary to the model
    lattice and are never accepted by the model forward signature.
    """

    event_id: str
    source_target_receipt_sha256: str
    source_input_batch_sha256: str
    source_context_receipt_sha256: str
    lattice_structure_sha256: str
    raw_onset_interval_seconds: tuple[float, float] | None
    raw_offset_interval_seconds: tuple[float, float] | None
    effective_onset_status: str
    effective_offset_status: str
    onset_projection_status: str
    offset_projection_status: str
    causal_axis_length: int
    lattice_cell_count: int
    causal_onset_candidate_mask: tuple[bool, ...] | None
    offline_onset_boundary_mask: tuple[bool, ...] | None
    offline_offset_boundary_mask: tuple[bool, ...] | None
    onset_selected_support_rows: tuple[tuple[int, float, float, float], ...]
    offset_selected_support_rows: tuple[tuple[int, float, float, float], ...]
    offset_terminal_support_rows: tuple[tuple[int, float, float, float], ...]
    schema_version: str = BA_IEG_SEGMENTAL_LATTICE_TARGET_PROJECTION_SCHEMA_VERSION
    method_id: str = BA_IEG_SEGMENTAL_LATTICE_TARGET_PROJECTION_METHOD_ID
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _identifier(self.event_id, "projection event_id")
        for name in (
            "source_target_receipt_sha256",
            "source_input_batch_sha256",
            "source_context_receipt_sha256",
            "lattice_structure_sha256",
        ):
            _sha256(getattr(self, name), name)
        if (
            self.schema_version
            != BA_IEG_SEGMENTAL_LATTICE_TARGET_PROJECTION_SCHEMA_VERSION
        ):
            raise ValueError("segmental lattice-target projection schema drifted")
        if self.method_id != BA_IEG_SEGMENTAL_LATTICE_TARGET_PROJECTION_METHOD_ID:
            raise ValueError("segmental lattice-target projection method drifted")
        if self.effective_onset_status not in _ONSET_STATUSES:
            raise ValueError("projected onset status is unsupported")
        if self.effective_offset_status not in _OFFSET_STATUSES:
            raise ValueError("projected offset status is unsupported")
        if self.onset_projection_status not in {
            "mapped_to_frozen_causal_support",
            "raw_left_censored",
            "raw_not_observed",
            "raw_not_evaluable",
        }:
            raise ValueError("onset projection disposition is unsupported")
        if self.offset_projection_status not in {
            "mapped_to_frozen_lattice_cell_support",
            "raw_right_censored",
            "resolution_right_censored_on_terminal_support",
            "raw_not_observed",
            "raw_not_evaluable",
        }:
            raise ValueError("offset projection disposition is unsupported")
        if (
            isinstance(self.causal_axis_length, bool)
            or not isinstance(self.causal_axis_length, int)
            or self.causal_axis_length < 1
            or isinstance(self.lattice_cell_count, bool)
            or not isinstance(self.lattice_cell_count, int)
            or self.lattice_cell_count < 1
        ):
            raise ValueError("projected lattice dimensions must be positive integers")
        if self.effective_onset_status == "observed_interval":
            if (
                self.causal_onset_candidate_mask is None
                or self.offline_onset_boundary_mask is None
                or len(self.causal_onset_candidate_mask) != self.causal_axis_length
                or len(self.offline_onset_boundary_mask) != self.lattice_cell_count
                or not any(self.causal_onset_candidate_mask)
                or not any(self.offline_onset_boundary_mask)
                or not self.onset_selected_support_rows
            ):
                raise ValueError("observed onset projection has no frozen support")
        elif (
            self.causal_onset_candidate_mask is not None
            or self.offline_onset_boundary_mask is not None
            or self.onset_selected_support_rows
        ):
            raise ValueError("non-observed onset projection cannot carry edge masks")
        if self.effective_offset_status == "observed_interval":
            if (
                self.offline_offset_boundary_mask is None
                or len(self.offline_offset_boundary_mask) != self.lattice_cell_count
                or not any(self.offline_offset_boundary_mask)
                or not self.offset_selected_support_rows
            ):
                raise ValueError("observed offset projection has no frozen support")
        elif (
            self.offline_offset_boundary_mask is not None
            or self.offset_selected_support_rows
        ):
            raise ValueError("non-observed offset projection cannot carry edge masks")
        if (
            self.offset_projection_status
            == "resolution_right_censored_on_terminal_support"
        ) != bool(self.offset_terminal_support_rows):
            raise ValueError("terminal-support censor trace is inconsistent")
        object.__setattr__(self, "receipt_sha256", self._compute_sha256())

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "method_id": self.method_id,
            "event_id": self.event_id,
            "binding": {
                "source_target_receipt_sha256": self.source_target_receipt_sha256,
                "source_input_batch_sha256": self.source_input_batch_sha256,
                "source_context_receipt_sha256": self.source_context_receipt_sha256,
                "lattice_structure_sha256": self.lattice_structure_sha256,
            },
            "raw_annotation_intervals_seconds": {
                "onset": _json_interval(self.raw_onset_interval_seconds),
                "offset": _json_interval(self.raw_offset_interval_seconds),
            },
            "effective_status": {
                "onset": self.effective_onset_status,
                "offset": self.effective_offset_status,
            },
            "projection_status": {
                "onset": self.onset_projection_status,
                "offset": self.offset_projection_status,
            },
            "axis_lengths": {
                "causal_candidate": self.causal_axis_length,
                "lattice_cell": self.lattice_cell_count,
            },
            "masks": {
                "causal_onset_candidate": (
                    list(self.causal_onset_candidate_mask)
                    if self.causal_onset_candidate_mask is not None
                    else None
                ),
                "offline_onset_boundary": (
                    list(self.offline_onset_boundary_mask)
                    if self.offline_onset_boundary_mask is not None
                    else None
                ),
                "offline_offset_boundary": (
                    list(self.offline_offset_boundary_mask)
                    if self.offline_offset_boundary_mask is not None
                    else None
                ),
            },
            "selected_support": {
                "onset": _support_rows_json(self.onset_selected_support_rows),
                "offset": _support_rows_json(self.offset_selected_support_rows),
                "offset_terminal": _support_rows_json(
                    self.offset_terminal_support_rows
                ),
            },
            "scope_receipt": _LATTICE_TARGET_PROJECTION_SCOPE,
        }

    def _compute_sha256(self) -> str:
        return _canonical_sha256(self._payload())

    def verify_integrity(self) -> None:
        if self.receipt_sha256 != self._compute_sha256():
            raise ValueError("segmental lattice-target projection changed")


def _interval_hits_right_closed_support(
    interval: tuple[float, float], support: tuple[float, float]
) -> bool:
    """Use cell support, never a nearest-center or nearest-edge snap."""

    start, stop = interval
    support_start, support_stop = support
    if stop > start:
        return min(stop, support_stop) - max(start, support_start) > 1e-12
    return (
        support_start + _TIME_TOLERANCE_SECONDS
        < start
        <= support_stop + _TIME_TOLERANCE_SECONDS
    )


def _frozen_lattice_structure(
    output: BAIEGPermissionSplitSegmentalStateOutput,
    event_index: int,
) -> tuple[
    int,
    tuple[tuple[float, float], ...],
    tuple[bool, ...],
    tuple[tuple[bool, ...], ...],
    tuple[float, ...],
    tuple[bool, ...],
    str,
]:
    cells = _active_cell_count(output, event_index)
    bounds = tuple(
        (float(row[0]), float(row[1]))
        for row in output.lattice_cell_bounds_seconds[event_index, :cells]
        .detach()
        .cpu()
    )
    cell_mask = tuple(
        bool(value)
        for value in output.lattice_cell_mask[event_index, :cells].detach().cpu()
    )
    transition_mask = tuple(
        tuple(bool(value) for value in row)
        for row in output.transition_mask[event_index, :cells].detach().cpu()
    )
    causal_times = tuple(
        float(value)
        for value in output.causal_candidate_times_seconds[event_index].detach().cpu()
    )
    causal_mask = tuple(
        bool(value)
        for value in output.causal_candidate_mask[event_index].detach().cpu()
    )
    payload = {
        "event_index": event_index,
        "cell_bounds_seconds": [list(row) for row in bounds],
        "cell_evidence_mask": list(cell_mask),
        "transition_mask": [list(row) for row in transition_mask],
        "causal_candidate_times_seconds": list(causal_times),
        "causal_candidate_mask": list(causal_mask),
    }
    return (
        cells,
        bounds,
        cell_mask,
        transition_mask,
        causal_times,
        causal_mask,
        _canonical_sha256(payload),
    )


def _has_immediate_post_boundary_opportunity(
    cell_index: int,
    bounds: Sequence[tuple[float, float]],
    cell_mask: Sequence[bool],
) -> bool:
    next_index = cell_index + 1
    return (
        next_index < len(bounds)
        and cell_mask[next_index]
        and abs(bounds[cell_index][1] - bounds[next_index][0])
        <= _TIME_TOLERANCE_SECONDS
    )


def _causal_boundary_support_rows(
    *,
    bounds: Sequence[tuple[float, float]],
    cell_mask: Sequence[bool],
    transition_mask: Sequence[Sequence[bool]],
    causal_times: Sequence[float],
    causal_mask: Sequence[bool],
) -> dict[int, tuple[int, float, float, float]]:
    """Return disjoint right-closed support bins for legal causal edges."""

    cell_by_stop: dict[int, int] = {}
    real_candidates: list[tuple[int, float]] = []
    seen_times: list[float] = []
    for candidate_index, time in enumerate(causal_times):
        matching = [
            cell_index
            for cell_index, (_, stop) in enumerate(bounds)
            if abs(stop - time) <= _TIME_TOLERANCE_SECONDS
        ]
        if (
            not matching
            or not causal_mask[candidate_index]
            or not any(
                transition_mask[matching[0]][edge]
                for edge in SEGMENTAL_ONSET_EDGE_INDICES_V1
            )
            or any(
                abs(time - previous) <= _TIME_TOLERANCE_SECONDS
                for previous in seen_times
            )
        ):
            continue
        cell_by_stop[candidate_index] = matching[0]
        real_candidates.append((candidate_index, time))
        seen_times.append(time)
    real_candidates.sort(key=lambda row: row[1])
    result: dict[int, tuple[int, float, float, float]] = {}
    for order, (candidate_index, time) in enumerate(real_candidates):
        cell_index = cell_by_stop[candidate_index]
        if not cell_mask[cell_index] or not _has_immediate_post_boundary_opportunity(
            cell_index, bounds, cell_mask
        ):
            continue
        previous_time = (
            real_candidates[order - 1][1] if order > 0 else bounds[cell_index][0]
        )
        support_start = bounds[cell_index][0]
        cursor = cell_index
        while cursor > 0:
            previous_index = cursor - 1
            if (
                not cell_mask[previous_index]
                or abs(bounds[previous_index][1] - bounds[cursor][0])
                > _TIME_TOLERANCE_SECONDS
                or bounds[previous_index][1] <= previous_time + _TIME_TOLERANCE_SECONDS
            ):
                break
            support_start = bounds[previous_index][0]
            cursor = previous_index
        support_start = max(support_start, previous_time)
        if time - support_start <= _TIME_TOLERANCE_SECONDS:
            continue
        result[candidate_index] = (
            candidate_index,
            support_start,
            time,
            time,
        )
    return result


def project_ba_ieg_segmental_target_to_frozen_lattice_v1(
    target: BAIEGSegmentalEventTargetV1,
    output: BAIEGPermissionSplitSegmentalStateOutput,
    context: BAIEGSegmentalBoundaryContext,
    event_index: int,
) -> BAIEGSegmentalLatticeTargetProjectionV1:
    """Project raw boundary intervals after a target-free lattice is frozen."""

    if not isinstance(target, BAIEGSegmentalEventTargetV1):
        raise TypeError("lattice projection requires a segmental event target")
    if not isinstance(output, BAIEGPermissionSplitSegmentalStateOutput):
        raise TypeError("lattice projection requires a segmental model output")
    if not isinstance(context, BAIEGSegmentalBoundaryContext):
        raise TypeError("lattice projection requires a boundary context")
    if isinstance(event_index, bool) or not isinstance(event_index, int):
        raise TypeError("lattice projection event_index must be an integer")
    target.verify_integrity()
    context.verify_integrity()
    if not 0 <= event_index < len(context.event_ids):
        raise IndexError("lattice projection event_index is outside the batch")
    if (
        output.source_input_batch_sha256 != context.source_input_batch_sha256
        or output.source_context_receipt_sha256 != context.receipt_sha256
    ):
        raise ValueError("lattice projection input/context binding drifted")
    if (
        target.event_id != context.event_ids[event_index]
        or target.source_event_receipt_sha256
        != context.source_event_receipt_sha256s[event_index]
        or target.adaptive_acquisition_receipt_sha256
        != context.adaptive_acquisition_receipt_sha256s[event_index]
    ):
        raise ValueError("lattice projection target/context identity drifted")
    (
        cells,
        bounds,
        cell_mask,
        transition_mask,
        causal_times,
        causal_mask,
        lattice_hash,
    ) = _frozen_lattice_structure(output, event_index)
    causal_support = _causal_boundary_support_rows(
        bounds=bounds,
        cell_mask=cell_mask,
        transition_mask=transition_mask,
        causal_times=causal_times,
        causal_mask=causal_mask,
    )

    causal_onset_mask: tuple[bool, ...] | None = None
    offline_onset_mask: tuple[bool, ...] | None = None
    offline_offset_mask: tuple[bool, ...] | None = None
    onset_rows: tuple[tuple[int, float, float, float], ...] = ()
    offset_rows: tuple[tuple[int, float, float, float], ...] = ()
    terminal_offset_rows: tuple[tuple[int, float, float, float], ...] = ()
    effective_onset = target.onset_status
    effective_offset = target.offset_status

    if target.onset_status == "observed_interval":
        assert target.onset_interval_seconds is not None
        onset_rows = tuple(
            row
            for _, row in sorted(causal_support.items())
            if _interval_hits_right_closed_support(
                target.onset_interval_seconds, (row[1], row[2])
            )
        )
        if not onset_rows:
            if not any(
                active
                and _interval_hits_right_closed_support(
                    target.onset_interval_seconds, bound
                )
                for bound, active in zip(bounds, cell_mask)
            ):
                raise BAIEGSegmentalTargetNotRepresentableError(
                    f"{target.event_id}: observed onset interval has no causal "
                    "candidate: no frozen observed lattice opportunity"
                )
            raise BAIEGSegmentalTargetNotRepresentableError(
                f"{target.event_id}: observed onset interval has no legal "
                "frozen causal support bin"
            )
        candidate_selection = {row[0] for row in onset_rows}
        causal_onset_mask = tuple(
            index in candidate_selection for index in range(len(causal_times))
        )
        selected_times = {row[3] for row in onset_rows}
        offline_onset_mask = tuple(
            active
            and any(
                transition_mask[index][edge] for edge in SEGMENTAL_ONSET_EDGE_INDICES_V1
            )
            and any(
                abs(bounds[index][1] - time) <= _TIME_TOLERANCE_SECONDS
                for time in selected_times
            )
            for index, active in enumerate(cell_mask)
        )
        if not any(offline_onset_mask):
            raise BAIEGSegmentalTargetNotRepresentableError(
                f"{target.event_id}: projected onset has no legal offline edge"
            )
        onset_projection_status = "mapped_to_frozen_causal_support"
    elif target.onset_status == "left_censored":
        if not bool(context.left_censoring_possible[event_index]):
            raise BAIEGSegmentalTargetNotRepresentableError(
                f"{target.event_id}: left-censored onset lacks a frozen left-censor opportunity"
            )
        onset_projection_status = "raw_left_censored"
    elif target.onset_status == "not_observed":
        onset_projection_status = "raw_not_observed"
    else:
        onset_projection_status = "raw_not_evaluable"

    if target.offset_status == "observed_interval":
        assert target.offset_interval_seconds is not None
        eligible: list[tuple[int, float, float, float]] = []
        terminal: list[tuple[int, float, float, float]] = []
        active_overlap = False
        for index, (bound, active) in enumerate(zip(bounds, cell_mask)):
            if not active or not _interval_hits_right_closed_support(
                target.offset_interval_seconds, bound
            ):
                continue
            active_overlap = True
            if not any(
                transition_mask[index][edge]
                for edge in SEGMENTAL_OFFSET_EDGE_INDICES_V1
            ):
                continue
            row = (index, bound[0], bound[1], bound[1])
            if _has_immediate_post_boundary_opportunity(index, bounds, cell_mask):
                eligible.append(row)
            else:
                terminal.append(row)
        if eligible:
            offset_rows = tuple(eligible)
            selected = {row[0] for row in eligible}
            offline_offset_mask = tuple(index in selected for index in range(cells))
            offset_projection_status = "mapped_to_frozen_lattice_cell_support"
        elif terminal and bool(context.right_censoring_possible[event_index]):
            effective_offset = "right_censored"
            terminal_offset_rows = tuple(terminal)
            offset_projection_status = "resolution_right_censored_on_terminal_support"
        elif not active_overlap:
            raise BAIEGSegmentalTargetNotRepresentableError(
                f"{target.event_id}: observed offset interval has no lattice "
                "edge: no frozen observed lattice opportunity"
            )
        else:
            raise BAIEGSegmentalTargetNotRepresentableError(
                f"{target.event_id}: observed offset interval has no legal "
                "lattice edge with post-boundary opportunity"
            )
    elif target.offset_status == "right_censored":
        if not bool(context.right_censoring_possible[event_index]):
            raise BAIEGSegmentalTargetNotRepresentableError(
                f"{target.event_id}: right-censored offset lacks a frozen right-censor opportunity"
            )
        offset_projection_status = "raw_right_censored"
    elif target.offset_status == "not_observed":
        offset_projection_status = "raw_not_observed"
    else:
        offset_projection_status = "raw_not_evaluable"

    return BAIEGSegmentalLatticeTargetProjectionV1(
        event_id=target.event_id,
        source_target_receipt_sha256=target.receipt_sha256,
        source_input_batch_sha256=output.source_input_batch_sha256,
        source_context_receipt_sha256=context.receipt_sha256,
        lattice_structure_sha256=lattice_hash,
        raw_onset_interval_seconds=target.onset_interval_seconds,
        raw_offset_interval_seconds=target.offset_interval_seconds,
        effective_onset_status=effective_onset,
        effective_offset_status=effective_offset,
        onset_projection_status=onset_projection_status,
        offset_projection_status=offset_projection_status,
        causal_axis_length=len(causal_times),
        lattice_cell_count=cells,
        causal_onset_candidate_mask=causal_onset_mask,
        offline_onset_boundary_mask=offline_onset_mask,
        offline_offset_boundary_mask=offline_offset_mask,
        onset_selected_support_rows=onset_rows,
        offset_selected_support_rows=offset_rows,
        offset_terminal_support_rows=terminal_offset_rows,
    )


def validate_ba_ieg_segmental_lattice_target_projection_v1(
    projection: BAIEGSegmentalLatticeTargetProjectionV1,
    *,
    target: BAIEGSegmentalEventTargetV1,
    output: BAIEGPermissionSplitSegmentalStateOutput,
    context: BAIEGSegmentalBoundaryContext,
    event_index: int,
) -> BAIEGSegmentalLatticeTargetProjectionV1:
    """Replay a projection against the unchanged target and frozen lattice."""

    if not isinstance(projection, BAIEGSegmentalLatticeTargetProjectionV1):
        raise TypeError("projection replay requires a lattice-target projection")
    projection.verify_integrity()
    replayed = project_ba_ieg_segmental_target_to_frozen_lattice_v1(
        target, output, context, event_index
    )
    if replayed != projection:
        raise ValueError("segmental lattice-target projection does not replay")
    return projection


@dataclass(frozen=True)
class BAIEGPermissionSplitSegmentalLossOutputV1:
    total_loss: torch.Tensor
    causal_onset_nll: torch.Tensor
    offline_constrained_path_nll: torch.Tensor
    causal_nll_per_event: torch.Tensor
    causal_loss_mask: torch.Tensor
    offline_nll_per_event: torch.Tensor
    offline_loss_mask: torch.Tensor
    total_loss_per_event: torch.Tensor
    total_loss_event_mask: torch.Tensor
    source_input_batch_sha256: str
    source_context_receipt_sha256: str
    target_bundle_receipt_sha256: str
    lattice_target_projection_receipt_sha256s: tuple[str, ...]
    loss_contract_sha256: str = BA_IEG_SEGMENTAL_LOSS_CONTRACT_SHA256
    causal_semantics: str = "future_free_interval_or_censor_nll"
    offline_semantics: str = (
        "exact_constrained_partition_nll_with_causal_start_and_onset_stop_gradient"
    )


def _active_cell_count(
    output: BAIEGPermissionSplitSegmentalStateOutput,
    event_index: int,
) -> int:
    bounds = output.lattice_cell_bounds_seconds[event_index]
    active = bounds[:, 1] > bounds[:, 0]
    count = int(active.sum().detach().cpu())
    if count < 1 or not bool(active[:count].all()) or bool(active[count:].any()):
        raise ValueError("segmental output cell padding is non-canonical")
    return count


def _detached_causal_potentials(
    output: BAIEGPermissionSplitSegmentalStateOutput,
    context: BAIEGSegmentalBoundaryContext,
    event_index: int,
) -> SegmentalPotentialsV1:
    cells = _active_cell_count(output, event_index)
    transition = output.transition_log_scores[event_index, :cells]
    causal_edge_mask = torch.tensor(
        [index in BA_IEG_SEGMENTAL_CAUSAL_TRANSITION_INDICES for index in range(8)],
        dtype=torch.bool,
        device=transition.device,
    ).unsqueeze(0)
    permission_split_transition = torch.where(
        causal_edge_mask,
        transition.detach(),
        transition,
    )
    physical = output.lattice_physical_duration_seconds[event_index, :cells]
    duration_scale = torch.exp(output.duration_scale_log_seconds)
    duration = build_lognormal_segment_duration_log_scores_v1(
        physical_duration=physical,
        duration_location_log_seconds=output.duration_location_log_seconds,
        duration_scale_log_seconds=duration_scale,
        minimum_state_duration_seconds=output.minimum_state_duration_seconds,
    )
    return SegmentalPotentialsV1(
        emission_log_density=output.offline_state_emission_log_prob[
            event_index, :cells
        ],
        opportunity_duration=output.lattice_opportunity_duration_seconds[
            event_index, :cells
        ],
        physical_duration=physical,
        transition_log_scores=permission_split_transition,
        transition_mask=output.transition_mask[event_index, :cells],
        start_log_scores=output.start_state_log_scores[event_index].detach(),
        end_log_scores=output.end_state_log_scores[event_index],
        event_log_score=output.event_presence_log_scores[event_index, 1],
        no_event_log_score=output.event_presence_log_scores[event_index, 0],
        segment_duration_log_scores=duration,
        maximum_segments=min(cells, int(output.full_segment_count_marginals.shape[1])),
        left_censoring_possible=bool(context.left_censoring_possible[event_index]),
        right_censoring_possible=bool(context.right_censoring_possible[event_index]),
    )


def _constraints_for_target(
    target: BAIEGSegmentalEventTargetV1,
    projection: BAIEGSegmentalLatticeTargetProjectionV1,
    potentials: SegmentalPotentialsV1,
) -> SegmentalPathConstraintsV1 | None:
    cells = potentials.cell_count
    starts: tuple[int, ...] | None = None
    ends: tuple[int, ...] | None = None
    terminal_bouts: tuple[int, ...] | None = None
    primary_mask: tuple[bool, ...] | None = None
    offset_mask: tuple[bool, ...] | None = None

    if target.event_status == "absent":
        starts, ends, terminal_bouts = (0,), (0,), (0,)
    elif target.event_status == "present":
        terminal_bouts = {
            "single_bout": (1,),
            "two_or_more_bouts": (2,),
            "not_evaluable": (1, 2),
        }[target.bout_count_status]
        if projection.effective_onset_status == "observed_interval":
            starts = (0,)
            assert projection.offline_onset_boundary_mask is not None
            if len(projection.offline_onset_boundary_mask) != cells:
                raise ValueError("projected onset mask does not align with lattice")
            primary_mask = projection.offline_onset_boundary_mask
            eligible = any(
                primary_mask[index]
                and any(
                    bool(potentials.transition_mask[index, edge])
                    for edge in SEGMENTAL_ONSET_EDGE_INDICES_V1
                )
                for index in range(cells)
            )
            if not eligible:
                raise BAIEGSegmentalTargetNotRepresentableError(
                    f"{target.event_id}: projected onset has no causal lattice edge"
                )
        elif projection.effective_onset_status == "left_censored":
            starts = (1, 2)
        if projection.effective_offset_status == "observed_interval":
            ends = (0, 3)
            assert projection.offline_offset_boundary_mask is not None
            if len(projection.offline_offset_boundary_mask) != cells:
                raise ValueError("projected offset mask does not align with lattice")
            offset_mask = projection.offline_offset_boundary_mask
            eligible = any(
                offset_mask[index]
                and any(
                    bool(potentials.transition_mask[index, edge])
                    for edge in SEGMENTAL_OFFSET_EDGE_INDICES_V1
                )
                for index in range(cells)
            )
            if not eligible:
                raise BAIEGSegmentalTargetNotRepresentableError(
                    f"{target.event_id}: projected offset has no lattice edge"
                )
        elif projection.effective_offset_status == "right_censored":
            ends = (1, 2)
    else:
        return None

    return SegmentalPathConstraintsV1(
        allowed_start_states=starts,
        allowed_end_states=ends,
        allowed_terminal_bout_classes=terminal_bouts,
        allowed_primary_onset_boundary_mask=primary_mask,
        allowed_offset_boundary_mask=offset_mask,
    )


def _causal_target_nll(
    target: BAIEGSegmentalEventTargetV1,
    projection: BAIEGSegmentalLatticeTargetProjectionV1,
    output: BAIEGPermissionSplitSegmentalStateOutput,
    event_index: int,
) -> torch.Tensor | None:
    if (
        target.event_status == "not_evaluable"
        or projection.effective_onset_status == "not_evaluable"
    ):
        return None
    if target.event_status == "absent":
        mass = output.causal_no_onset_within_support_mass[event_index]
    elif projection.effective_onset_status == "left_censored":
        mass = output.causal_left_censor_state_mass[event_index].sum()
        if float(mass.detach().cpu()) <= 0.0:
            raise BAIEGSegmentalTargetNotRepresentableError(
                f"{target.event_id}: left-censored onset is structurally impossible"
            )
    elif projection.effective_onset_status == "observed_interval":
        candidate_mask = output.causal_candidate_mask[event_index]
        assert projection.causal_onset_candidate_mask is not None
        interval_mask = torch.tensor(
            projection.causal_onset_candidate_mask,
            dtype=torch.bool,
            device=candidate_mask.device,
        )
        if tuple(interval_mask.shape) != tuple(candidate_mask.shape):
            raise ValueError("projected causal onset mask does not align")
        selected = candidate_mask & interval_mask
        if not bool(selected.any()):
            raise BAIEGSegmentalTargetNotRepresentableError(
                f"{target.event_id}: projected onset has no causal candidate"
            )
        mass = output.causal_onset_boundary_mass[event_index, selected].sum()
    else:  # guarded by target combination validation
        return None
    return -torch.log(mass.clamp_min(torch.finfo(mass.dtype).tiny))


def _patient_equal_event_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    patient_uids: Sequence[str],
) -> torch.Tensor:
    patient_rows: list[torch.Tensor] = []
    for patient_uid in sorted(set(patient_uids)):
        indices = (
            torch.tensor(
                [uid == patient_uid for uid in patient_uids],
                dtype=torch.bool,
                device=values.device,
            )
            & mask
        )
        if bool(indices.any()):
            patient_rows.append(values[indices].mean())
    if not patient_rows:
        raise ValueError("target bundle contributes no evaluable training objective")
    return torch.stack(patient_rows).mean()


def permission_split_segmental_training_loss_v1(
    batch: BAIEGCollatedEventBatch,
    context: BAIEGSegmentalBoundaryContext,
    output: BAIEGPermissionSplitSegmentalStateOutput,
    target_bundle: BAIEGSegmentalTargetBundleV1,
) -> BAIEGPermissionSplitSegmentalLossOutputV1:
    """Compute the public-source optimization loss with loss-level firewall."""

    if not isinstance(batch, BAIEGCollatedEventBatch):
        raise TypeError("training loss requires BAIEGCollatedEventBatch")
    if not isinstance(context, BAIEGSegmentalBoundaryContext):
        raise TypeError("training loss requires BAIEGSegmentalBoundaryContext")
    if not isinstance(output, BAIEGPermissionSplitSegmentalStateOutput):
        raise TypeError("training loss requires segmental model output")
    if not isinstance(target_bundle, BAIEGSegmentalTargetBundleV1):
        raise TypeError("training loss requires BAIEGSegmentalTargetBundleV1")
    context.verify_integrity()
    target_bundle.verify_integrity()
    if (
        target_bundle.optimization_role != "optimize"
        or target_bundle.model_split != "source_train"
    ):
        raise ValueError(
            "gradient-bearing segmental loss is source_train optimize-only"
        )
    if batch.model_split != "source_train":
        raise ValueError("segmental optimizer batch must be source_train")
    if output.heuristic_phase_posterior_used:
        raise ValueError(
            "heuristic phase-posterior ablation cannot enter main training loss"
        )
    if (
        output.source_input_batch_sha256 != batch.input_batch_sha256
        or context.source_input_batch_sha256 != batch.input_batch_sha256
        or target_bundle.source_input_batch_sha256 != batch.input_batch_sha256
        or output.source_context_receipt_sha256 != context.receipt_sha256
        or target_bundle.source_context_receipt_sha256 != context.receipt_sha256
    ):
        raise ValueError("model, context and target bundle content addresses disagree")
    if tuple(target.event_id for target in target_bundle.targets) != batch.event_ids:
        raise ValueError("target bundle event order drifted")

    event_count = len(batch.event_ids)
    device = output.exact_path_log_partition.device
    dtype = output.exact_path_log_partition.dtype
    causal_rows = torch.zeros(event_count, dtype=dtype, device=device)
    causal_mask = torch.zeros(event_count, dtype=torch.bool, device=device)
    offline_rows = torch.zeros(event_count, dtype=dtype, device=device)
    offline_mask = torch.zeros(event_count, dtype=torch.bool, device=device)
    projections: list[BAIEGSegmentalLatticeTargetProjectionV1] = []

    for event_index, target in enumerate(target_bundle.targets):
        target.verify_integrity()
        projection = project_ba_ieg_segmental_target_to_frozen_lattice_v1(
            target, output, context, event_index
        )
        projection.verify_integrity()
        projections.append(projection)
        causal = _causal_target_nll(target, projection, output, event_index)
        if causal is not None:
            causal_rows[event_index] = causal
            causal_mask[event_index] = True

        potentials = _detached_causal_potentials(output, context, event_index)
        constraints = _constraints_for_target(target, projection, potentials)
        if constraints is None:
            continue
        unconstrained = run_exact_segmental_forward_backward_v1(potentials)
        if not unconstrained.has_finite_support:
            raise BAIEGSegmentalTargetNotRepresentableError(
                f"{target.event_id}: unconstrained offline lattice has no finite path"
            )
        if not torch.allclose(
            unconstrained.exact_log_partition.detach(),
            output.exact_path_log_partition[event_index].detach(),
            atol=2e-5,
            rtol=2e-5,
        ):
            raise ValueError(
                "loss reconstruction drifted from the model's exact partition"
            )
        constrained = run_exact_segmental_forward_backward_v1(
            potentials, constraints=constraints
        )
        if not constrained.has_finite_support:
            raise BAIEGSegmentalTargetNotRepresentableError(
                f"{target.event_id}: target-compatible segmental path set is empty"
            )
        nll = unconstrained.exact_log_partition - constrained.exact_log_partition
        if float(nll.detach().cpu()) < -2e-5:
            raise RuntimeError("constrained partition exceeded the full partition")
        offline_rows[event_index] = nll.clamp_min(0.0)
        offline_mask[event_index] = True

    total_rows = causal_rows + offline_rows
    total_mask = causal_mask | offline_mask
    total_loss = _patient_equal_event_mean(total_rows, total_mask, batch.patient_uids)
    causal_summary = (
        causal_rows[causal_mask].mean()
        if bool(causal_mask.any())
        else total_loss.new_zeros(())
    )
    offline_summary = (
        offline_rows[offline_mask].mean()
        if bool(offline_mask.any())
        else total_loss.new_zeros(())
    )
    return BAIEGPermissionSplitSegmentalLossOutputV1(
        total_loss=total_loss,
        causal_onset_nll=causal_summary,
        offline_constrained_path_nll=offline_summary,
        causal_nll_per_event=causal_rows,
        causal_loss_mask=causal_mask,
        offline_nll_per_event=offline_rows,
        offline_loss_mask=offline_mask,
        total_loss_per_event=total_rows,
        total_loss_event_mask=total_mask,
        source_input_batch_sha256=batch.input_batch_sha256,
        source_context_receipt_sha256=context.receipt_sha256,
        target_bundle_receipt_sha256=target_bundle.receipt_sha256,
        lattice_target_projection_receipt_sha256s=tuple(
            projection.receipt_sha256 for projection in projections
        ),
    )


__all__ = [
    "BA_IEG_PERMISSION_SPLIT_SEGMENTAL_SUPERVISION_ID",
    "BA_IEG_SEGMENTAL_LATTICE_TARGET_PROJECTION_METHOD_ID",
    "BA_IEG_SEGMENTAL_LATTICE_TARGET_PROJECTION_SCHEMA_VERSION",
    "BA_IEG_SEGMENTAL_LOSS_CONTRACT_SHA256",
    "BA_IEG_SEGMENTAL_LOSS_CONTRACT_SCHEMA_VERSION",
    "BA_IEG_SEGMENTAL_TARGET_BUNDLE_SCHEMA_VERSION",
    "BAIEGPermissionSplitSegmentalLossOutputV1",
    "BAIEGSegmentalLatticeTargetProjectionV1",
    "BAIEGSegmentalEventTargetV1",
    "BAIEGSegmentalTargetBundleV1",
    "BAIEGSegmentalTargetFirewallV1",
    "BAIEGSegmentalTargetNotRepresentableError",
    "build_ba_ieg_segmental_target_bundle_v1",
    "permission_split_segmental_training_loss_v1",
    "project_ba_ieg_segmental_target_to_frozen_lattice_v1",
    "validate_ba_ieg_segmental_lattice_target_projection_v1",
]
