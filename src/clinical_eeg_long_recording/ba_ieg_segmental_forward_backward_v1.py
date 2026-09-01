"""Polynomial exact forward--backward for BA-IEG segmental potentials.

The tiny-grid exhaustive oracle in :mod:`ba_ieg_segmental_dp_kernel_v1`
defines the path semantics.  This module computes the same finite partition
and full marginals without enumerating complete paths.  Its dynamic state is

``(segment_count, current_segment_start, state, bout_class)``

where ``bout_class`` is ``0``, ``1`` or ``2+``.  It is sufficient to separate
null/event paths, primary versus secondary observed onset edges, and recurrent
event mass.  The exact expected number of bouts is recovered from the
left-censored initial-bout mass plus all onset-edge marginals.

The implementation remains a research decoder: it consumes only explicit
potentials, emits uncalibrated structured masses, and authorizes no seizure,
SOZ, or report claim.  Runtime is polynomial rather than exponential, but the
dense duration table still makes the method quadratic in the number of
physical-time cells.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final

import torch

from .ba_ieg_segmental_dp_kernel_v1 import (
    SEGMENTAL_OFFSET_EDGE_INDICES_V1,
    SEGMENTAL_ONSET_EDGE_INDICES_V1,
    SEGMENTAL_S0_ONSET_EDGE_INDICES_V1,
    SEGMENTAL_S3_REENTRY_EDGE_INDICES_V1,
    SEGMENTAL_TRANSITION_EDGES_V1,
    SegmentalPathConstraintsV1,
    SegmentalPotentialsV1,
)


SEGMENTAL_BOUT_CLASSES_V1: Final[tuple[str, ...]] = ("zero", "one", "two_or_more")


@dataclass(frozen=True)
class ExactSegmentalForwardBackwardOutputV1:
    """Exact full-posterior marginals on one finite segmental lattice."""

    exact_log_partition: torch.Tensor
    state_marginal: torch.Tensor
    transition_marginal: torch.Tensor
    start_state_marginal: torch.Tensor
    end_state_marginal: torch.Tensor
    segment_count_marginal: torch.Tensor
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
    recurrent_event_mass: torch.Tensor
    expected_event_bout_count: torch.Tensor
    expected_recurrent_bout_count: torch.Tensor
    has_finite_support: bool
    materialized_forward_state_count: int
    materialized_backward_state_count: int
    implementation_semantics: str = "exact_polynomial_segmental_forward_backward_v1"
    marginal_semantics: str = (
        "exact_full_finite_path_posterior_not_calibrated_clinical_probability"
    )


def _logadd(
    values: dict[object, torch.Tensor], key: object, candidate: torch.Tensor
) -> None:
    # ``logaddexp(finite, logsumexp([-inf, ...]))`` has the right forward
    # value but can retain a NaN autograd branch through the all-impossible
    # term.  Structural impossibility is target-independent and may be tested
    # on a detached scalar, so do not register such branches in the graph.
    if not bool(torch.isfinite(candidate.detach()).item()):
        return
    existing = values.get(key)
    values[key] = candidate if existing is None else torch.logaddexp(existing, candidate)


def _negative_infinity(reference: torch.Tensor) -> torch.Tensor:
    return reference.new_full((), -torch.inf)


def _logsumexp_or_negative_infinity(
    candidates: list[torch.Tensor], reference: torch.Tensor
) -> torch.Tensor:
    finite_candidates = [
        candidate
        for candidate in candidates
        if bool(torch.isfinite(candidate.detach()).item())
    ]
    if not finite_candidates:
        return _negative_infinity(reference)
    return torch.logsumexp(torch.stack(finite_candidates), dim=0)


def _bout_after_edge(bout_class: int, edge_index: int) -> int:
    if edge_index not in SEGMENTAL_ONSET_EDGE_INDICES_V1:
        return bout_class
    return min(2, bout_class + 1)


def _censor_class(*, left: bool, right: bool) -> int:
    if left and right:
        return 3
    if left:
        return 1
    if right:
        return 2
    return 0


def _outgoing_edges() -> tuple[tuple[tuple[int, int], ...], ...]:
    return tuple(
        tuple(
            (edge_index, target)
            for edge_index, (source, target) in enumerate(
                SEGMENTAL_TRANSITION_EDGES_V1
            )
            if source == state
        )
        for state in range(4)
    )


def build_lognormal_segment_duration_log_scores_v1(
    *,
    physical_duration: torch.Tensor,
    duration_location_log_seconds: torch.Tensor,
    duration_scale_log_seconds: torch.Tensor,
    minimum_state_duration_seconds: torch.Tensor,
) -> torch.Tensor:
    """Materialize ``[M,M,4,4]`` direct duration potentials.

    Complete segments use a log-normal log density.  Left-, right-, and
    both-censored classes use the same log survival initializer, matching the
    pre-existing BA-IEG shadow model while keeping all four semantic slots
    explicit.  This initializer is not a fitted censoring model.
    """

    if not isinstance(physical_duration, torch.Tensor) or physical_duration.ndim != 1:
        raise ValueError("physical_duration must be a one-dimensional tensor")
    if not physical_duration.dtype.is_floating_point:
        raise TypeError("physical_duration must use a floating dtype")
    cell_count = int(physical_duration.numel())
    if cell_count < 1 or not torch.isfinite(physical_duration).all() or torch.any(
        physical_duration <= 0
    ):
        raise ValueError("physical_duration must contain positive finite cells")
    dtype = physical_duration.dtype
    device = physical_duration.device
    vectors = (
        ("duration_location_log_seconds", duration_location_log_seconds),
        ("duration_scale_log_seconds", duration_scale_log_seconds),
        ("minimum_state_duration_seconds", minimum_state_duration_seconds),
    )
    for name, value in vectors:
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != (4,)
            or value.dtype != dtype
            or value.device != device
            or not torch.isfinite(value).all()
        ):
            raise ValueError(f"{name} must be a finite [4] tensor on the physical clock")
    if torch.any(duration_scale_log_seconds <= 0) or torch.any(
        minimum_state_duration_seconds < 0
    ):
        raise ValueError("duration scales must be positive and minima non-negative")

    prefix = torch.cat(
        (physical_duration.new_zeros(1), torch.cumsum(physical_duration, dim=0))
    )
    duration = prefix[1:].unsqueeze(0) - prefix[:-1].unsqueeze(1)
    valid_interval = torch.triu(
        torch.ones((cell_count, cell_count), dtype=torch.bool, device=device)
    )
    positive_duration = duration.clamp_min(torch.finfo(dtype).eps).unsqueeze(-1)
    log_duration = torch.log(positive_duration)
    location = duration_location_log_seconds.reshape(1, 1, 4)
    scale = duration_scale_log_seconds.reshape(1, 1, 4)
    z = (log_duration - location) / scale
    complete = (
        -0.5 * z.square()
        - torch.log(scale)
        - log_duration
        - 0.5 * math.log(2.0 * math.pi)
    )
    survival = torch.log(
        (0.5 * torch.erfc(z / math.sqrt(2.0))).clamp_min(
            torch.finfo(dtype).tiny
        )
    )
    table = torch.stack((complete, survival, survival, survival), dim=-1)
    duration_eligible = (
        positive_duration
        + 8.0 * torch.finfo(dtype).eps
        >= minimum_state_duration_seconds.reshape(1, 1, 4)
    )
    valid = valid_interval.unsqueeze(-1) & duration_eligible
    return torch.where(
        valid.unsqueeze(-1),
        table,
        torch.full_like(table, -torch.inf),
    )


def _empty_output(
    potentials: SegmentalPotentialsV1,
    *,
    forward_count: int,
    backward_count: int,
) -> ExactSegmentalForwardBackwardOutputV1:
    zero = potentials.emission_log_density.new_zeros(())
    cells = potentials.cell_count
    return ExactSegmentalForwardBackwardOutputV1(
        exact_log_partition=_negative_infinity(potentials.emission_log_density),
        state_marginal=potentials.emission_log_density.new_zeros((cells, 4)),
        transition_marginal=potentials.emission_log_density.new_zeros((cells, 8)),
        start_state_marginal=potentials.emission_log_density.new_zeros(4),
        end_state_marginal=potentials.emission_log_density.new_zeros(4),
        segment_count_marginal=potentials.emission_log_density.new_zeros(
            potentials.maximum_segments
        ),
        onset_boundary_mass=potentials.emission_log_density.new_zeros(cells),
        primary_onset_boundary_mass=potentials.emission_log_density.new_zeros(cells),
        secondary_onset_boundary_mass=potentials.emission_log_density.new_zeros(cells),
        s0_onset_boundary_mass=potentials.emission_log_density.new_zeros(cells),
        s3_reentry_onset_boundary_mass=potentials.emission_log_density.new_zeros(cells),
        offset_boundary_mass=potentials.emission_log_density.new_zeros(cells),
        event_mass=zero,
        null_mass=zero,
        left_censor_mass=zero,
        right_censor_mass=zero,
        both_censor_mass=zero,
        recurrent_event_mass=zero,
        expected_event_bout_count=zero,
        expected_recurrent_bout_count=zero,
        has_finite_support=False,
        materialized_forward_state_count=forward_count,
        materialized_backward_state_count=backward_count,
    )


def run_exact_segmental_forward_backward_v1(
    potentials: SegmentalPotentialsV1,
    *,
    constraints: SegmentalPathConstraintsV1 | None = None,
) -> ExactSegmentalForwardBackwardOutputV1:
    """Compute the exact finite partition and full forward--backward marginals.

    For ``M`` cells, ``K`` maximum segments, four states, three bout classes,
    and eight fixed edges, time is ``O(K M^2 H E)`` and dynamic-state memory is
    ``O(K M S H)`` where ``H=3``.  The supplied dense duration table itself is
    ``O(M^2 S C)``.
    """

    if not isinstance(potentials, SegmentalPotentialsV1):
        raise TypeError("potentials must be SegmentalPotentialsV1")
    if constraints is not None:
        if not isinstance(constraints, SegmentalPathConstraintsV1):
            raise TypeError("constraints must be SegmentalPathConstraintsV1 or None")
        constraints.validate_for(potentials)
    cell_count = potentials.cell_count
    maximum_segments = potentials.maximum_segments
    weighted_emission = (
        potentials.emission_log_density
        * potentials.opportunity_duration.unsqueeze(-1)
    )
    emission_prefix = torch.cat(
        (
            weighted_emission.new_zeros((1, 4)),
            torch.cumsum(weighted_emission, dim=0),
        ),
        dim=0,
    )
    outgoing = _outgoing_edges()
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

    def transition_allowed(
        *, boundary: int, edge_index: int, bout: int
    ) -> bool:
        return bool(potentials.transition_mask[boundary, edge_index]) and (
            constraints is None
            or constraints.allows_transition(
                boundary_index=boundary,
                edge_index=edge_index,
                current_bout_class=bout,
            )
        )

    def segment_score(
        *,
        start: int,
        stop: int,
        state: int,
        left: bool,
        right: bool,
    ) -> torch.Tensor:
        emission = emission_prefix[stop, state] - emission_prefix[start, state]
        return emission + potentials.segment_duration_log_scores[
            start, stop - 1, state, _censor_class(left=left, right=right)
        ]

    # F[k,a,s,h,l] excludes the current segment and includes the start score,
    # all previous segments, and the transition into state s at cell a.  The
    # initial-left flag remains explicit because both-side censoring can span
    # several observed segments and is not recoverable from the current state.
    forward: dict[tuple[int, int, int, int, bool], torch.Tensor] = {}
    for state in allowed_starts:
        bout = int(state in (1, 2))
        initial_left = state in (1, 2)
        forward[(1, 0, state, bout, initial_left)] = potentials.start_log_scores[
            state
        ]
    for segment_count in range(1, maximum_segments):
        current = [
            (key, score)
            for key, score in forward.items()
            if key[0] == segment_count
        ]
        for (_, start, state, bout, initial_left), prefix_score in current:
            left = segment_count == 1 and state in (1, 2)
            for boundary in range(start, cell_count - 1):
                segment = segment_score(
                    start=start,
                    stop=boundary + 1,
                    state=state,
                    left=left,
                    right=False,
                )
                for edge_index, target in outgoing[state]:
                    if not transition_allowed(
                        boundary=boundary, edge_index=edge_index, bout=bout
                    ):
                        continue
                    next_bout = _bout_after_edge(bout, edge_index)
                    candidate = (
                        prefix_score
                        + segment
                        + potentials.transition_log_scores[boundary, edge_index]
                    )
                    _logadd(
                        forward,
                        (
                            segment_count + 1,
                            boundary + 1,
                            target,
                            next_bout,
                            initial_left,
                        ),
                        candidate,
                    )

    # B[k,a,s,h] includes the current segment, all continuations, end score,
    # and the event/no-event terminal score, but excludes the prefix F score.
    backward: dict[tuple[int, int, int, int], torch.Tensor] = {}
    for segment_count in range(maximum_segments, 0, -1):
        for start in range(segment_count - 1, cell_count):
            for state in range(4):
                left = segment_count == 1 and state in (1, 2)
                for bout in range(3):
                    candidates: list[torch.Tensor] = []
                    if state in allowed_ends and (
                        constraints is None
                        or constraints.allows_terminal_bout(bout)
                    ):
                        right = state in (1, 2)
                        candidates.append(
                            segment_score(
                                start=start,
                                stop=cell_count,
                                state=state,
                                left=left,
                                right=right,
                            )
                            + potentials.end_log_scores[state]
                            + (
                                potentials.event_log_score
                                if bout > 0
                                else potentials.no_event_log_score
                            )
                        )
                    if segment_count < maximum_segments:
                        for boundary in range(start, cell_count - 1):
                            segment = segment_score(
                                start=start,
                                stop=boundary + 1,
                                state=state,
                                left=left,
                                right=False,
                            )
                            for edge_index, target in outgoing[state]:
                                if not transition_allowed(
                                    boundary=boundary,
                                    edge_index=edge_index,
                                    bout=bout,
                                ):
                                    continue
                                next_bout = _bout_after_edge(bout, edge_index)
                                suffix = backward.get(
                                    (
                                        segment_count + 1,
                                        boundary + 1,
                                        target,
                                        next_bout,
                                    )
                                )
                                if suffix is None:
                                    continue
                                candidates.append(
                                    segment
                                    + potentials.transition_log_scores[
                                        boundary, edge_index
                                    ]
                                    + suffix
                                )
                    backward[(segment_count, start, state, bout)] = (
                        _logsumexp_or_negative_infinity(
                            candidates, potentials.emission_log_density
                        )
                    )

    start_log_numerator: dict[int, torch.Tensor] = {}
    partition_candidates: list[torch.Tensor] = []
    for state in allowed_starts:
        bout = int(state in (1, 2))
        suffix = backward[(1, 0, state, bout)]
        score = potentials.start_log_scores[state] + suffix
        start_log_numerator[state] = score
        partition_candidates.append(score)
    log_partition = _logsumexp_or_negative_infinity(
        partition_candidates, potentials.emission_log_density
    )
    if not bool(torch.isfinite(log_partition.detach()).item()):
        return _empty_output(
            potentials,
            forward_count=len(forward),
            backward_count=len(backward),
        )

    segment_log_numerator: dict[tuple[int, int, int], torch.Tensor] = {}
    transition_log_numerator: dict[tuple[int, int], torch.Tensor] = {}
    primary_log_numerator: dict[int, torch.Tensor] = {}
    secondary_log_numerator: dict[int, torch.Tensor] = {}
    end_log_numerator: dict[int, torch.Tensor] = {}
    segment_count_log_numerator: dict[int, torch.Tensor] = {}
    terminal_class_log_numerator: dict[str, torch.Tensor] = {}

    for (
        segment_count,
        start,
        state,
        bout,
        initial_left,
    ), prefix_score in forward.items():
        left = segment_count == 1 and state in (1, 2)
        if state in allowed_ends and (
            constraints is None
            or constraints.allows_terminal_bout(bout)
        ):
            right = state in (1, 2)
            full_score = (
                prefix_score
                + segment_score(
                    start=start,
                    stop=cell_count,
                    state=state,
                    left=left,
                    right=right,
                )
                + potentials.end_log_scores[state]
                + (
                    potentials.event_log_score
                    if bout > 0
                    else potentials.no_event_log_score
                )
            )
            _logadd(
                segment_log_numerator,
                (start, cell_count, state),
                full_score,
            )
            _logadd(end_log_numerator, state, full_score)
            _logadd(segment_count_log_numerator, segment_count, full_score)
            _logadd(
                terminal_class_log_numerator,
                "event" if bout > 0 else "null",
                full_score,
            )
            if right:
                _logadd(terminal_class_log_numerator, "right", full_score)
            if initial_left and right:
                _logadd(terminal_class_log_numerator, "both", full_score)
            if bout == 2:
                _logadd(terminal_class_log_numerator, "recurrent", full_score)

        if segment_count >= maximum_segments:
            continue
        for boundary in range(start, cell_count - 1):
            segment = segment_score(
                start=start,
                stop=boundary + 1,
                state=state,
                left=left,
                right=False,
            )
            for edge_index, target in outgoing[state]:
                if not transition_allowed(
                    boundary=boundary, edge_index=edge_index, bout=bout
                ):
                    continue
                next_bout = _bout_after_edge(bout, edge_index)
                suffix = backward.get(
                    (segment_count + 1, boundary + 1, target, next_bout)
                )
                if suffix is None:
                    continue
                full_score = (
                    prefix_score
                    + segment
                    + potentials.transition_log_scores[boundary, edge_index]
                    + suffix
                )
                _logadd(
                    segment_log_numerator,
                    (start, boundary + 1, state),
                    full_score,
                )
                _logadd(
                    transition_log_numerator,
                    (boundary, edge_index),
                    full_score,
                )
                if edge_index in SEGMENTAL_ONSET_EDGE_INDICES_V1:
                    _logadd(
                        primary_log_numerator
                        if bout == 0
                        else secondary_log_numerator,
                        boundary,
                        full_score,
                    )

    def probability(log_numerator: torch.Tensor | None) -> torch.Tensor:
        if log_numerator is None:
            return potentials.emission_log_density.new_zeros(())
        return torch.exp(log_numerator - log_partition)

    start_state_marginal = torch.stack(
        [probability(start_log_numerator.get(state)) for state in range(4)]
    )
    end_state_marginal = torch.stack(
        [probability(end_log_numerator.get(state)) for state in range(4)]
    )
    segment_count_marginal = torch.stack(
        [
            probability(segment_count_log_numerator.get(count))
            for count in range(1, maximum_segments + 1)
        ]
    )

    # Convert interval-segment masses to per-cell state occupancy with a
    # differentiable difference array, avoiding O(M^3) interval expansion.
    difference = potentials.emission_log_density.new_zeros((cell_count + 1) * 4)
    if segment_log_numerator:
        starts: list[int] = []
        stops: list[int] = []
        masses: list[torch.Tensor] = []
        for (start, stop, state), numerator in segment_log_numerator.items():
            starts.append(start * 4 + state)
            stops.append(stop * 4 + state)
            masses.append(probability(numerator))
        mass_tensor = torch.stack(masses)
        indices = torch.tensor(starts + stops, dtype=torch.long, device=mass_tensor.device)
        updates = torch.cat((mass_tensor, -mass_tensor))
        difference = difference.index_add(0, indices, updates)
    state_marginal = torch.cumsum(
        difference.reshape(cell_count + 1, 4), dim=0
    )[:cell_count]

    transition_flat = potentials.emission_log_density.new_zeros(cell_count * 8)
    if transition_log_numerator:
        indices: list[int] = []
        masses = []
        for (boundary, edge), numerator in transition_log_numerator.items():
            indices.append(boundary * 8 + edge)
            masses.append(probability(numerator))
        transition_flat = transition_flat.index_add(
            0,
            torch.tensor(indices, dtype=torch.long, device=transition_flat.device),
            torch.stack(masses),
        )
    transition_marginal = transition_flat.reshape(cell_count, 8)

    primary_onset_boundary_mass = torch.stack(
        [probability(primary_log_numerator.get(cell)) for cell in range(cell_count)]
    )
    secondary_onset_boundary_mass = torch.stack(
        [probability(secondary_log_numerator.get(cell)) for cell in range(cell_count)]
    )
    onset_boundary_mass = transition_marginal[
        :, sorted(SEGMENTAL_ONSET_EDGE_INDICES_V1)
    ].sum(dim=-1)
    s0_onset_boundary_mass = transition_marginal[
        :, sorted(SEGMENTAL_S0_ONSET_EDGE_INDICES_V1)
    ].sum(dim=-1)
    s3_reentry_onset_boundary_mass = transition_marginal[
        :, sorted(SEGMENTAL_S3_REENTRY_EDGE_INDICES_V1)
    ].sum(dim=-1)
    offset_boundary_mass = transition_marginal[
        :, sorted(SEGMENTAL_OFFSET_EDGE_INDICES_V1)
    ].sum(dim=-1)
    event_mass = probability(terminal_class_log_numerator.get("event"))
    null_mass = probability(terminal_class_log_numerator.get("null"))
    left_censor_mass = start_state_marginal[1:3].sum()
    right_censor_mass = probability(terminal_class_log_numerator.get("right"))
    both_censor_mass = probability(terminal_class_log_numerator.get("both"))
    recurrent_event_mass = probability(
        terminal_class_log_numerator.get("recurrent")
    )
    expected_event_bout_count = left_censor_mass + onset_boundary_mass.sum()
    expected_recurrent_bout_count = (
        expected_event_bout_count - event_mass
    ).clamp_min(0.0)

    return ExactSegmentalForwardBackwardOutputV1(
        exact_log_partition=log_partition,
        state_marginal=state_marginal,
        transition_marginal=transition_marginal,
        start_state_marginal=start_state_marginal,
        end_state_marginal=end_state_marginal,
        segment_count_marginal=segment_count_marginal,
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
        recurrent_event_mass=recurrent_event_mass,
        expected_event_bout_count=expected_event_bout_count,
        expected_recurrent_bout_count=expected_recurrent_bout_count,
        has_finite_support=True,
        materialized_forward_state_count=len(forward),
        materialized_backward_state_count=len(backward),
    )


__all__ = [
    "ExactSegmentalForwardBackwardOutputV1",
    "SEGMENTAL_BOUT_CLASSES_V1",
    "build_lognormal_segment_duration_log_scores_v1",
    "run_exact_segmental_forward_backward_v1",
]
