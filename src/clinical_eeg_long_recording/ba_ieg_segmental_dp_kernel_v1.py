"""Exact small-grid oracle for BA-IEG segmental potentials.

This module deliberately contains no encoder, label reader, heuristic phase
posterior, or clinical/report logic.  It accepts already materialized scalar
potentials and exhaustively enumerates the finite segmental lattice.  The
enumeration is differentiable with respect to every selected log potential
and is intended as a tiny-grid validation oracle, not as the efficient
production dynamic program.

The four states are ``S0/S1/S2/S3``.  Self persistence is represented by the
duration of a segment; the eight explicit edges are frozen in the same order
as the permission-split BA-IEG model.  A finite ``maximum_segments`` bounds
the recurrent ``S3`` edges.  Emission log density is integrated only over EEG
opportunity time.  Segment-duration potentials are supplied directly by the
caller and may be built from physical-duration prefixes, so a quality gap can
change duration support without emitting state evidence.

All returned masses are exact normalized masses on this finite, uncalibrated
research lattice.  They are not calibrated clinical probabilities and do not
authorize a seizure, onset, SOZ, or report claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Sequence

import torch


SEGMENTAL_STATE_NAMES_V1: Final[tuple[str, ...]] = ("S0", "S1", "S2", "S3")
SEGMENTAL_TRANSITION_NAMES_V1: Final[tuple[str, ...]] = (
    "S0_to_S1",
    "S1_to_S2",
    "S2_to_S3",
    "S0_to_S2_short_event",
    "S1_to_S3_short_return",
    "S3_to_S0_clean_return",
    "S3_to_S1_reentry",
    "S3_to_S2_reentry_short",
)
SEGMENTAL_TRANSITION_EDGES_V1: Final[tuple[tuple[int, int], ...]] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (0, 2),
    (1, 3),
    (3, 0),
    (3, 1),
    (3, 2),
)
SEGMENTAL_DURATION_CENSOR_CLASSES_V1: Final[tuple[str, ...]] = (
    "complete",
    "left",
    "right",
    "both",
)
SEGMENTAL_ONSET_EDGE_INDICES_V1: Final[frozenset[int]] = frozenset(
    {0, 3, 6, 7}
)
SEGMENTAL_S0_ONSET_EDGE_INDICES_V1: Final[frozenset[int]] = frozenset({0, 3})
SEGMENTAL_S3_REENTRY_EDGE_INDICES_V1: Final[frozenset[int]] = frozenset({6, 7})
SEGMENTAL_OFFSET_EDGE_INDICES_V1: Final[frozenset[int]] = frozenset({2, 4})
MAXIMUM_EXACT_ENUMERATION_CELLS_V1: Final[int] = 8


def _canonical_index_subset(
    value: Sequence[int] | None,
    *,
    name: str,
    upper_exclusive: int,
) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be an integer sequence or None")
    result = tuple(value)
    if not result:
        raise ValueError(f"{name} cannot be empty")
    if any(type(item) is not int for item in result):
        raise TypeError(f"{name} must contain only integers")
    if any(item < 0 or item >= upper_exclusive for item in result):
        raise ValueError(f"{name} contains an out-of-range index")
    if tuple(sorted(set(result))) != result:
        raise ValueError(f"{name} must be sorted and unique")
    return result


def _canonical_boolean_vector(
    value: Sequence[bool] | None,
    *,
    name: str,
) -> tuple[bool, ...] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a boolean sequence or None")
    result = tuple(value)
    if not result or any(type(item) is not bool for item in result):
        raise TypeError(f"{name} must be a non-empty boolean sequence")
    return result


@dataclass(frozen=True)
class SegmentalPathConstraintsV1:
    """Immutable optional path-subset constraints for exact supervision.

    Passing ``None`` to either exact decoder preserves the original
    unconstrained lattice.  A constraint only removes complete paths; it
    never changes a potential value.  Bout classes are ``0``, ``1`` and
    ``2+``.  The primary-onset mask applies only to the first observed onset
    edge; an initially active left-censored segment already counts as bout
    one.  The offset mask applies to every explicit ``S1/S2 -> S3`` edge.
    """

    allowed_start_states: tuple[int, ...] | None = None
    allowed_end_states: tuple[int, ...] | None = None
    allowed_terminal_bout_classes: tuple[int, ...] | None = None
    allowed_transition_boundary_mask: tuple[tuple[bool, ...], ...] | None = None
    allowed_primary_onset_boundary_mask: tuple[bool, ...] | None = None
    allowed_offset_boundary_mask: tuple[bool, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "allowed_start_states",
            _canonical_index_subset(
                self.allowed_start_states,
                name="allowed_start_states",
                upper_exclusive=len(SEGMENTAL_STATE_NAMES_V1),
            ),
        )
        object.__setattr__(
            self,
            "allowed_end_states",
            _canonical_index_subset(
                self.allowed_end_states,
                name="allowed_end_states",
                upper_exclusive=len(SEGMENTAL_STATE_NAMES_V1),
            ),
        )
        object.__setattr__(
            self,
            "allowed_terminal_bout_classes",
            _canonical_index_subset(
                self.allowed_terminal_bout_classes,
                name="allowed_terminal_bout_classes",
                upper_exclusive=3,
            ),
        )
        transition = self.allowed_transition_boundary_mask
        if transition is not None:
            if isinstance(transition, (str, bytes)) or not transition:
                raise TypeError(
                    "allowed_transition_boundary_mask must be a non-empty row sequence"
                )
            rows: list[tuple[bool, ...]] = []
            for row_index, row in enumerate(transition):
                canonical = _canonical_boolean_vector(
                    row,
                    name=f"allowed_transition_boundary_mask[{row_index}]",
                )
                assert canonical is not None
                if len(canonical) != len(SEGMENTAL_TRANSITION_NAMES_V1):
                    raise ValueError(
                        "every allowed transition row must have eight edge entries"
                    )
                rows.append(canonical)
            object.__setattr__(
                self, "allowed_transition_boundary_mask", tuple(rows)
            )
        object.__setattr__(
            self,
            "allowed_primary_onset_boundary_mask",
            _canonical_boolean_vector(
                self.allowed_primary_onset_boundary_mask,
                name="allowed_primary_onset_boundary_mask",
            ),
        )
        object.__setattr__(
            self,
            "allowed_offset_boundary_mask",
            _canonical_boolean_vector(
                self.allowed_offset_boundary_mask,
                name="allowed_offset_boundary_mask",
            ),
        )

    def validate_for(self, potentials: "SegmentalPotentialsV1") -> None:
        cells = potentials.cell_count
        for name in (
            "allowed_transition_boundary_mask",
            "allowed_primary_onset_boundary_mask",
            "allowed_offset_boundary_mask",
        ):
            value = getattr(self, name)
            if value is not None and len(value) != cells:
                raise ValueError(f"{name} must have one row per physical cell")

    def allows_start(self, state: int) -> bool:
        return self.allowed_start_states is None or state in self.allowed_start_states

    def allows_end(self, state: int) -> bool:
        return self.allowed_end_states is None or state in self.allowed_end_states

    def allows_terminal_bout(self, bout_class: int) -> bool:
        return (
            self.allowed_terminal_bout_classes is None
            or bout_class in self.allowed_terminal_bout_classes
        )

    def allows_transition(
        self,
        *,
        boundary_index: int,
        edge_index: int,
        current_bout_class: int,
    ) -> bool:
        if (
            self.allowed_transition_boundary_mask is not None
            and not self.allowed_transition_boundary_mask[boundary_index][edge_index]
        ):
            return False
        if (
            edge_index in SEGMENTAL_ONSET_EDGE_INDICES_V1
            and current_bout_class == 0
            and self.allowed_primary_onset_boundary_mask is not None
            and not self.allowed_primary_onset_boundary_mask[boundary_index]
        ):
            return False
        if (
            edge_index in SEGMENTAL_OFFSET_EDGE_INDICES_V1
            and self.allowed_offset_boundary_mask is not None
            and not self.allowed_offset_boundary_mask[boundary_index]
        ):
            return False
        return True


def _require_tensor(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    boolean: bool = False,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if boolean:
        if value.dtype != torch.bool:
            raise TypeError(f"{name} must be boolean")
    elif not value.dtype.is_floating_point:
        raise TypeError(f"{name} must use a floating dtype")
    if dtype is not None and value.dtype != dtype:
        raise TypeError(f"{name} must use dtype {dtype}")
    if device is not None and value.device != device:
        raise ValueError(f"{name} must be on device {device}")
    return value


def _reject_nan_or_positive_infinity(value: torch.Tensor, *, name: str) -> None:
    if torch.isnan(value).any() or torch.isposinf(value).any():
        raise ValueError(f"{name} cannot contain NaN or positive infinity")


@dataclass(frozen=True)
class SegmentalPotentialsV1:
    """Pure potentials for one finite physical-time event lattice.

    ``segment_duration_log_scores[start, end, state, censor_class]`` uses
    inclusive cell indices for ``start`` and ``end``.  Entries below the
    diagonal are structurally invalid and are never inspected.  Censor-class
    order is ``complete, left, right, both``.
    """

    emission_log_density: torch.Tensor
    opportunity_duration: torch.Tensor
    physical_duration: torch.Tensor
    transition_log_scores: torch.Tensor
    transition_mask: torch.Tensor
    start_log_scores: torch.Tensor
    end_log_scores: torch.Tensor
    event_log_score: torch.Tensor
    no_event_log_score: torch.Tensor
    segment_duration_log_scores: torch.Tensor
    maximum_segments: int
    left_censoring_possible: bool
    right_censoring_possible: bool

    def __post_init__(self) -> None:
        if not isinstance(self.emission_log_density, torch.Tensor):
            raise TypeError("emission_log_density must be a torch.Tensor")
        if self.emission_log_density.ndim != 2:
            raise ValueError("emission_log_density must have shape [M,4]")
        cell_count = int(self.emission_log_density.shape[0])
        if tuple(self.emission_log_density.shape) != (cell_count, 4):
            raise ValueError("emission_log_density must have shape [M,4]")
        if cell_count < 1:
            raise ValueError("the segmental lattice must contain at least one cell")
        if (
            isinstance(self.maximum_segments, bool)
            or not isinstance(self.maximum_segments, int)
            or not 1 <= self.maximum_segments <= cell_count
        ):
            raise ValueError("maximum_segments must be an integer in [1,M]")
        if type(self.left_censoring_possible) is not bool or type(
            self.right_censoring_possible
        ) is not bool:
            raise TypeError("left/right censor permissions must be boolean")

        dtype = self.emission_log_density.dtype
        device = self.emission_log_density.device
        if not dtype.is_floating_point:
            raise TypeError("emission_log_density must use a floating dtype")
        floating = {
            "emission_log_density": _require_tensor(
                self.emission_log_density,
                name="emission_log_density",
                shape=(cell_count, 4),
                dtype=dtype,
                device=device,
            ),
            "opportunity_duration": _require_tensor(
                self.opportunity_duration,
                name="opportunity_duration",
                shape=(cell_count,),
                dtype=dtype,
                device=device,
            ),
            "physical_duration": _require_tensor(
                self.physical_duration,
                name="physical_duration",
                shape=(cell_count,),
                dtype=dtype,
                device=device,
            ),
            "transition_log_scores": _require_tensor(
                self.transition_log_scores,
                name="transition_log_scores",
                shape=(cell_count, 8),
                dtype=dtype,
                device=device,
            ),
            "start_log_scores": _require_tensor(
                self.start_log_scores,
                name="start_log_scores",
                shape=(4,),
                dtype=dtype,
                device=device,
            ),
            "end_log_scores": _require_tensor(
                self.end_log_scores,
                name="end_log_scores",
                shape=(4,),
                dtype=dtype,
                device=device,
            ),
            "event_log_score": _require_tensor(
                self.event_log_score,
                name="event_log_score",
                shape=(),
                dtype=dtype,
                device=device,
            ),
            "no_event_log_score": _require_tensor(
                self.no_event_log_score,
                name="no_event_log_score",
                shape=(),
                dtype=dtype,
                device=device,
            ),
        }
        duration = _require_tensor(
            self.segment_duration_log_scores,
            name="segment_duration_log_scores",
            shape=(cell_count, cell_count, 4, 4),
            dtype=dtype,
            device=device,
        )
        _require_tensor(
            self.transition_mask,
            name="transition_mask",
            shape=(cell_count, 8),
            device=device,
            boolean=True,
        )
        for name, value in floating.items():
            _reject_nan_or_positive_infinity(value, name=name)
        valid_duration_entries = duration[
            torch.triu(
                torch.ones(
                    (cell_count, cell_count), dtype=torch.bool, device=device
                )
            )
        ]
        _reject_nan_or_positive_infinity(
            valid_duration_entries, name="valid segment_duration_log_scores"
        )
        if (
            not torch.isfinite(self.opportunity_duration).all()
            or not torch.isfinite(self.physical_duration).all()
            or torch.any(self.opportunity_duration < 0)
            or torch.any(self.physical_duration <= 0)
            or torch.any(
                self.opportunity_duration
                > self.physical_duration
                + 8.0 * torch.finfo(dtype).eps
            )
        ):
            raise ValueError(
                "durations must be finite with 0 <= opportunity <= physical and physical > 0"
            )

    @property
    def cell_count(self) -> int:
        return int(self.emission_log_density.shape[0])


@dataclass(frozen=True)
class ExactSegmentalPathV1:
    """One retained typed path; segment bounds are half-open cell intervals."""

    log_score: torch.Tensor
    posterior_mass: torch.Tensor
    conditional_retained_weight: torch.Tensor
    states: tuple[int, ...]
    segment_cell_bounds: tuple[tuple[int, int], ...]
    segment_physical_duration: torch.Tensor
    transition_boundary_indices: tuple[int, ...]
    transition_edge_indices: tuple[int, ...]
    path_class: str
    event_bout_count: int
    recurrent_cycle_count: int
    left_censored: bool
    right_censored: bool
    has_event: bool


@dataclass(frozen=True)
class ExactSegmentalDPOutputV1:
    """Exact full-posterior marginals plus auditable retained paths."""

    exact_log_partition: torch.Tensor
    state_marginal: torch.Tensor
    transition_marginal: torch.Tensor
    onset_boundary_mass: torch.Tensor
    primary_onset_boundary_mass: torch.Tensor
    secondary_onset_boundary_mass: torch.Tensor
    s0_onset_boundary_mass: torch.Tensor
    s3_reentry_onset_boundary_mass: torch.Tensor
    offset_boundary_mass: torch.Tensor
    event_mass: torch.Tensor
    null_mass: torch.Tensor
    left_censor_mass: torch.Tensor
    right_censor_mass: torch.Tensor
    both_censor_mass: torch.Tensor
    top_paths: tuple[ExactSegmentalPathV1, ...]
    retained_path_mass: torch.Tensor
    residual_path_mass: torch.Tensor
    exact_path_count: int
    has_finite_support: bool
    implementation_semantics: str = "exact_exhaustive_small_grid_enumeration_v1"
    marginal_semantics: str = (
        "exact_full_finite_path_posterior_not_calibrated_clinical_probability"
    )


@dataclass(frozen=True)
class _EnumeratedPathV1:
    score: torch.Tensor
    states: tuple[int, ...]
    segment_cell_bounds: tuple[tuple[int, int], ...]
    transition_boundary_indices: tuple[int, ...]
    transition_edge_indices: tuple[int, ...]
    has_event: bool

    @property
    def left_censored(self) -> bool:
        return self.states[0] in (1, 2)

    @property
    def right_censored(self) -> bool:
        return self.states[-1] in (1, 2)

    @property
    def recurrent_cycle_count(self) -> int:
        return max(0, self.event_bout_count - 1)

    @property
    def event_bout_count(self) -> int:
        initial_active_bout = int(self.states[0] in (1, 2))
        observed_onsets = sum(
            edge in SEGMENTAL_ONSET_EDGE_INDICES_V1
            for edge in self.transition_edge_indices
        )
        return initial_active_bout + observed_onsets

    @property
    def path_class(self) -> str:
        if not self.has_event:
            return "null_no_event"
        if self.left_censored and self.right_censored:
            return "event_left_and_right_censored"
        if self.left_censored:
            return "event_left_censored"
        if self.right_censored:
            return "event_right_censored"
        if self.recurrent_cycle_count:
            return "event_recurrent_complete"
        return "event_complete"


def _censor_class_index(*, left: bool, right: bool) -> int:
    if left and right:
        return 3
    if left:
        return 1
    if right:
        return 2
    return 0


def _possible(score: torch.Tensor) -> bool:
    return bool(torch.isfinite(score.detach()).item())


def _enumerate_paths(
    potentials: SegmentalPotentialsV1,
    constraints: SegmentalPathConstraintsV1 | None = None,
) -> list[_EnumeratedPathV1]:
    if constraints is not None:
        if not isinstance(constraints, SegmentalPathConstraintsV1):
            raise TypeError("constraints must be SegmentalPathConstraintsV1 or None")
        constraints.validate_for(potentials)
    cell_count = potentials.cell_count
    weighted_emission = (
        potentials.emission_log_density
        * potentials.opportunity_duration.unsqueeze(-1)
    )
    emission_prefix = torch.cat(
        (
            torch.zeros(
                (1, 4),
                dtype=weighted_emission.dtype,
                device=weighted_emission.device,
            ),
            torch.cumsum(weighted_emission, dim=0),
        ),
        dim=0,
    )
    allowed_starts = (0, 1, 2) if potentials.left_censoring_possible else (0,)
    if constraints is not None:
        allowed_starts = tuple(
            state for state in allowed_starts if constraints.allows_start(state)
        )
    allowed_ends = (
        (0, 1, 2, 3) if potentials.right_censoring_possible else (0, 3)
    )
    if constraints is not None:
        allowed_ends = tuple(
            state for state in allowed_ends if constraints.allows_end(state)
        )
    outgoing: dict[int, tuple[tuple[int, int], ...]] = {
        state: tuple(
            (edge_index, target)
            for edge_index, (source, target) in enumerate(
                SEGMENTAL_TRANSITION_EDGES_V1
            )
            if source == state
        )
        for state in range(4)
    }
    completed: list[_EnumeratedPathV1] = []

    def segment_score(
        *,
        start_cell: int,
        stop_cell_exclusive: int,
        state: int,
        left_censored: bool,
        right_censored: bool,
    ) -> torch.Tensor:
        emission = (
            emission_prefix[stop_cell_exclusive, state]
            - emission_prefix[start_cell, state]
        )
        censor_index = _censor_class_index(
            left=left_censored, right=right_censored
        )
        duration = potentials.segment_duration_log_scores[
            start_cell, stop_cell_exclusive - 1, state, censor_index
        ]
        return emission + duration

    def visit(
        *,
        state: int,
        segment_start: int,
        initial_state: int,
        prefix_score: torch.Tensor,
        states: tuple[int, ...],
        segment_bounds: tuple[tuple[int, int], ...],
        transition_boundaries: tuple[int, ...],
        transition_edges: tuple[int, ...],
        has_event: bool,
        bout_class: int,
    ) -> None:
        segment_count = len(states)
        first_left_censored = segment_count == 1 and initial_state in (1, 2)

        if state in allowed_ends and (
            constraints is None
            or constraints.allows_terminal_bout(bout_class)
        ):
            last_right_censored = state in (1, 2)
            final_segment = segment_score(
                start_cell=segment_start,
                stop_cell_exclusive=cell_count,
                state=state,
                left_censored=first_left_censored,
                right_censored=last_right_censored,
            )
            terminal_score = (
                prefix_score
                + final_segment
                + potentials.end_log_scores[state]
                + (
                    potentials.event_log_score
                    if has_event
                    else potentials.no_event_log_score
                )
            )
            if _possible(terminal_score):
                completed.append(
                    _EnumeratedPathV1(
                        score=terminal_score,
                        states=states,
                        segment_cell_bounds=segment_bounds
                        + ((segment_start, cell_count),),
                        transition_boundary_indices=transition_boundaries,
                        transition_edge_indices=transition_edges,
                        has_event=has_event,
                    )
                )

        if segment_count >= potentials.maximum_segments:
            return
        for boundary_index in range(segment_start, cell_count - 1):
            current_segment = segment_score(
                start_cell=segment_start,
                stop_cell_exclusive=boundary_index + 1,
                state=state,
                left_censored=first_left_censored,
                right_censored=False,
            )
            if not _possible(prefix_score + current_segment):
                continue
            for edge_index, target in outgoing[state]:
                if not bool(potentials.transition_mask[boundary_index, edge_index]):
                    continue
                if constraints is not None and not constraints.allows_transition(
                    boundary_index=boundary_index,
                    edge_index=edge_index,
                    current_bout_class=bout_class,
                ):
                    continue
                transition_score = potentials.transition_log_scores[
                    boundary_index, edge_index
                ]
                next_prefix = prefix_score + current_segment + transition_score
                if not _possible(next_prefix):
                    continue
                visit(
                    state=target,
                    segment_start=boundary_index + 1,
                    initial_state=initial_state,
                    prefix_score=next_prefix,
                    states=states + (target,),
                    segment_bounds=segment_bounds
                    + ((segment_start, boundary_index + 1),),
                    transition_boundaries=transition_boundaries
                    + (boundary_index,),
                    transition_edges=transition_edges + (edge_index,),
                    has_event=(
                        has_event
                        or edge_index in SEGMENTAL_ONSET_EDGE_INDICES_V1
                    ),
                    bout_class=(
                        min(2, bout_class + 1)
                        if edge_index in SEGMENTAL_ONSET_EDGE_INDICES_V1
                        else bout_class
                    ),
                )

    for start_state in allowed_starts:
        start_score = potentials.start_log_scores[start_state]
        if not _possible(start_score):
            continue
        visit(
            state=start_state,
            segment_start=0,
            initial_state=start_state,
            prefix_score=start_score,
            states=(start_state,),
            segment_bounds=(),
            transition_boundaries=(),
            transition_edges=(),
            has_event=start_state in (1, 2),
            bout_class=int(start_state in (1, 2)),
        )
    return completed


def _canonical_path_order(path: _EnumeratedPathV1) -> tuple[object, ...]:
    return (
        path.transition_boundary_indices,
        path.transition_edge_indices,
        path.states,
        path.segment_cell_bounds,
    )


def _empty_output(potentials: SegmentalPotentialsV1) -> ExactSegmentalDPOutputV1:
    zero = potentials.emission_log_density.new_zeros(())
    return ExactSegmentalDPOutputV1(
        exact_log_partition=potentials.emission_log_density.new_full((), -torch.inf),
        state_marginal=potentials.emission_log_density.new_zeros(
            (potentials.cell_count, 4)
        ),
        transition_marginal=potentials.emission_log_density.new_zeros(
            (potentials.cell_count, 8)
        ),
        onset_boundary_mass=potentials.emission_log_density.new_zeros(
            potentials.cell_count
        ),
        primary_onset_boundary_mass=potentials.emission_log_density.new_zeros(
            potentials.cell_count
        ),
        secondary_onset_boundary_mass=potentials.emission_log_density.new_zeros(
            potentials.cell_count
        ),
        s0_onset_boundary_mass=potentials.emission_log_density.new_zeros(
            potentials.cell_count
        ),
        s3_reentry_onset_boundary_mass=potentials.emission_log_density.new_zeros(
            potentials.cell_count
        ),
        offset_boundary_mass=potentials.emission_log_density.new_zeros(
            potentials.cell_count
        ),
        event_mass=zero,
        null_mass=zero,
        left_censor_mass=zero,
        right_censor_mass=zero,
        both_censor_mass=zero,
        top_paths=(),
        retained_path_mass=zero,
        residual_path_mass=zero,
        exact_path_count=0,
        has_finite_support=False,
    )


def run_exact_segmental_dp_v1(
    potentials: SegmentalPotentialsV1,
    *,
    maximum_paths: int = 8,
    constraints: SegmentalPathConstraintsV1 | None = None,
) -> ExactSegmentalDPOutputV1:
    """Enumerate the exact finite lattice and return full path marginals.

    Despite the public ``dp`` name, v1 is intentionally an exhaustive
    small-grid oracle (``M <= 8``).  It is useful for float64 parity tests and
    for validating a future efficient forward--backward implementation.
    """

    if not isinstance(potentials, SegmentalPotentialsV1):
        raise TypeError("potentials must be SegmentalPotentialsV1")
    if potentials.cell_count > MAXIMUM_EXACT_ENUMERATION_CELLS_V1:
        raise ValueError(
            "the exhaustive v1 oracle is limited to "
            f"{MAXIMUM_EXACT_ENUMERATION_CELLS_V1} cells; use the exact "
            "forward-backward decoder for longer lattices"
        )
    if (
        isinstance(maximum_paths, bool)
        or not isinstance(maximum_paths, int)
        or maximum_paths < 1
    ):
        raise ValueError("maximum_paths must be a positive integer")

    paths = _enumerate_paths(potentials, constraints)
    if not paths:
        return _empty_output(potentials)
    paths = sorted(paths, key=_canonical_path_order)
    scores = torch.stack([path.score for path in paths])
    log_partition = torch.logsumexp(scores, dim=0)
    posterior = torch.exp(scores - log_partition)
    path_count = len(paths)
    cell_count = potentials.cell_count
    dtype = scores.dtype
    device = scores.device

    state_indicator = torch.zeros(
        (path_count, cell_count, 4), dtype=dtype, device=device
    )
    transition_indicator = torch.zeros(
        (path_count, cell_count, 8), dtype=dtype, device=device
    )
    primary_onset_indicator = torch.zeros(
        (path_count, cell_count), dtype=dtype, device=device
    )
    secondary_onset_indicator = torch.zeros(
        (path_count, cell_count), dtype=dtype, device=device
    )
    left_indicator = torch.zeros(path_count, dtype=dtype, device=device)
    right_indicator = torch.zeros(path_count, dtype=dtype, device=device)
    event_indicator = torch.zeros(path_count, dtype=dtype, device=device)
    for path_index, path in enumerate(paths):
        for state, (start, stop) in zip(path.states, path.segment_cell_bounds):
            state_indicator[path_index, start:stop, state] = 1.0
        event_bouts_seen = int(path.states[0] in (1, 2))
        for boundary, edge in zip(
            path.transition_boundary_indices, path.transition_edge_indices
        ):
            transition_indicator[path_index, boundary, edge] = 1.0
            if edge in SEGMENTAL_ONSET_EDGE_INDICES_V1:
                if event_bouts_seen == 0:
                    primary_onset_indicator[path_index, boundary] = 1.0
                else:
                    secondary_onset_indicator[path_index, boundary] = 1.0
                event_bouts_seen += 1
        left_indicator[path_index] = float(path.left_censored)
        right_indicator[path_index] = float(path.right_censored)
        event_indicator[path_index] = float(path.has_event)

    state_marginal = torch.einsum("p,pms->ms", posterior, state_indicator)
    transition_marginal = torch.einsum(
        "p,pme->me", posterior, transition_indicator
    )
    onset_boundary_mass = transition_marginal[
        :, sorted(SEGMENTAL_ONSET_EDGE_INDICES_V1)
    ].sum(dim=-1)
    primary_onset_boundary_mass = torch.einsum(
        "p,pm->m", posterior, primary_onset_indicator
    )
    secondary_onset_boundary_mass = torch.einsum(
        "p,pm->m", posterior, secondary_onset_indicator
    )
    s0_onset_boundary_mass = transition_marginal[
        :, sorted(SEGMENTAL_S0_ONSET_EDGE_INDICES_V1)
    ].sum(dim=-1)
    s3_reentry_onset_boundary_mass = transition_marginal[
        :, sorted(SEGMENTAL_S3_REENTRY_EDGE_INDICES_V1)
    ].sum(dim=-1)
    offset_boundary_mass = transition_marginal[
        :, sorted(SEGMENTAL_OFFSET_EDGE_INDICES_V1)
    ].sum(dim=-1)
    event_mass = torch.sum(posterior * event_indicator)
    null_mass = torch.sum(posterior * (1.0 - event_indicator))
    left_censor_mass = torch.sum(posterior * left_indicator)
    right_censor_mass = torch.sum(posterior * right_indicator)
    both_censor_mass = torch.sum(posterior * left_indicator * right_indicator)

    ranking = torch.argsort(scores, descending=True, stable=True)
    retained_indices = ranking[: min(maximum_paths, path_count)]
    retained_scores = scores[retained_indices]
    retained_conditional = torch.softmax(retained_scores, dim=0)
    retained_mass = posterior[retained_indices].sum().clamp(0.0, 1.0)
    residual_mass = (1.0 - retained_mass).clamp(0.0, 1.0)
    physical_prefix = torch.cat(
        (
            potentials.physical_duration.new_zeros(1),
            torch.cumsum(potentials.physical_duration, dim=0),
        )
    )
    retained: list[ExactSegmentalPathV1] = []
    for retained_position, raw_index in enumerate(retained_indices.tolist()):
        path = paths[int(raw_index)]
        physical_durations = torch.stack(
            [
                physical_prefix[stop] - physical_prefix[start]
                for start, stop in path.segment_cell_bounds
            ]
        )
        retained.append(
            ExactSegmentalPathV1(
                log_score=path.score,
                posterior_mass=posterior[int(raw_index)],
                conditional_retained_weight=retained_conditional[
                    retained_position
                ],
                states=path.states,
                segment_cell_bounds=path.segment_cell_bounds,
                segment_physical_duration=physical_durations,
                transition_boundary_indices=path.transition_boundary_indices,
                transition_edge_indices=path.transition_edge_indices,
                path_class=path.path_class,
                event_bout_count=path.event_bout_count,
                recurrent_cycle_count=path.recurrent_cycle_count,
                left_censored=path.left_censored,
                right_censored=path.right_censored,
                has_event=path.has_event,
            )
        )

    return ExactSegmentalDPOutputV1(
        exact_log_partition=log_partition,
        state_marginal=state_marginal,
        transition_marginal=transition_marginal,
        onset_boundary_mass=onset_boundary_mass,
        primary_onset_boundary_mass=primary_onset_boundary_mass,
        secondary_onset_boundary_mass=secondary_onset_boundary_mass,
        s0_onset_boundary_mass=s0_onset_boundary_mass,
        s3_reentry_onset_boundary_mass=s3_reentry_onset_boundary_mass,
        offset_boundary_mass=offset_boundary_mass,
        event_mass=event_mass,
        null_mass=null_mass,
        left_censor_mass=left_censor_mass,
        right_censor_mass=right_censor_mass,
        both_censor_mass=both_censor_mass,
        top_paths=tuple(retained),
        retained_path_mass=retained_mass,
        residual_path_mass=residual_mass,
        exact_path_count=path_count,
        has_finite_support=True,
    )


__all__ = [
    "ExactSegmentalDPOutputV1",
    "ExactSegmentalPathV1",
    "MAXIMUM_EXACT_ENUMERATION_CELLS_V1",
    "SEGMENTAL_DURATION_CENSOR_CLASSES_V1",
    "SEGMENTAL_OFFSET_EDGE_INDICES_V1",
    "SEGMENTAL_ONSET_EDGE_INDICES_V1",
    "SEGMENTAL_S0_ONSET_EDGE_INDICES_V1",
    "SEGMENTAL_S3_REENTRY_EDGE_INDICES_V1",
    "SEGMENTAL_STATE_NAMES_V1",
    "SEGMENTAL_TRANSITION_EDGES_V1",
    "SEGMENTAL_TRANSITION_NAMES_V1",
    "SegmentalPathConstraintsV1",
    "SegmentalPotentialsV1",
    "run_exact_segmental_dp_v1",
]
