"""Permission-locked K-path boundary marginalization for BA-IEG events.

This module is a small research primitive between the physical-time event
encoder and later Findings heads.  It enumerates only ordered computational
state paths

``S0 -> S1 -> S2 -> S3``

on recording-relative physical seconds.  A path is represented by the three
ordered transition times ``S0/S1``, ``S1/S2`` and ``S2/S3``.  Token support is
assigned to a state by interval overlap, never by padded row number or an
ordinal position embedding.

The public ``forward`` method accepts a registered ``BAIEGCollatedEventBatch``
and an output carrying the same content-addressed input-batch receipt as the
permission-locked physical-time encoder.  It calls ``positive_onset_inputs()``
internally and never reads ``phase_posterior`` or an offline-context tensor.  In particular,
the deterministic P0 phase posterior is *not* a boundary prior here: that
posterior is informative only on future-dependent context views and therefore
cannot enter this onset-positive primitive.

The returned path weights are a softmax over the retained top-K paths, not a
calibrated posterior over all possible boundaries.  Boundary dispersion is a
retained-path research uncertainty summary, not a clinical confidence
interval.  Nothing in this module qualifies a seizure, cortical SOZ, EZ, or a
reportable clinical term, and the module is not connected to a production
route.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final, Sequence

import torch
from torch import nn

from .ba_ieg_physical_time_encoder import BAIEGPhysicalTimeEncoderOutput
from .ba_ieg_training_contract import (
    BA_IEG_PHASE_STATES,
    BAIEGCollatedEventBatch,
)


BA_IEG_BOUNDARY_PATH_MARGINALIZER_ID: Final[str] = (
    "ba_ieg_permission_locked_physical_time_k_boundary_path_marginalizer_v1"
)
BA_IEG_BOUNDARY_TRANSITIONS: Final[tuple[str, ...]] = (
    "S0_to_S1",
    "S1_to_S2",
    "S2_to_S3",
)


@dataclass(frozen=True)
class BAIEGBoundaryPathMarginalizationOutput:
    """Auditable retained-K state paths and physically pooled event features.

    ``path_weights`` sum to one only across ``path_mask=True`` rows.  They are
    deliberately named weights rather than probabilities.  In unevaluable
    events all masks, weights, uncertainty summaries and embeddings are zero.
    """

    source_input_batch_sha256: str
    candidate_boundary_times_seconds: torch.Tensor
    candidate_transition_logits: torch.Tensor
    candidate_mask: torch.Tensor
    path_boundary_times_seconds: torch.Tensor
    path_log_scores: torch.Tensor
    path_weights: torch.Tensor
    path_mask: torch.Tensor
    boundary_mean_seconds: torch.Tensor
    boundary_standard_deviation_seconds: torch.Tensor
    boundary_retained_support_seconds: torch.Tensor
    normalized_path_entropy: torch.Tensor
    effective_retained_path_count: torch.Tensor
    token_state_marginals: torch.Tensor
    path_state_embeddings: torch.Tensor
    path_state_opportunity_mask: torch.Tensor
    marginal_state_embeddings: torch.Tensor
    marginal_state_opportunity: torch.Tensor
    event_pooled_embeddings: torch.Tensor
    event_evaluable_mask: torch.Tensor


def _select_top_k(
    entries: Sequence[tuple[torch.Tensor, tuple[int, ...]]],
    maximum_paths: int,
) -> list[tuple[torch.Tensor, tuple[int, ...]]]:
    """Select differentiable scores with deterministic stable tie handling."""

    if not entries:
        return []
    scores = torch.stack([entry[0] for entry in entries])
    ordering = torch.argsort(scores, descending=True, stable=True)
    return [entries[int(index)] for index in ordering[:maximum_paths]]


def _top_k_legal_paths(
    candidate_times_seconds: torch.Tensor,
    transition_logits: torch.Tensor,
    *,
    support_start_seconds: float,
    support_stop_seconds: float,
    maximum_paths: int,
    minimum_state_duration_seconds: float,
    tolerance_seconds: float,
) -> list[tuple[torch.Tensor, tuple[int, int, int]]]:
    """K-best dynamic program over strictly ordered physical boundaries.

    Keeping the K best prefixes per last boundary is exact for this additive
    three-transition lattice: all future extensions depend only on the last
    physical boundary and not on an earlier token row identity.
    """

    candidate_count = int(candidate_times_seconds.numel())
    if candidate_count < len(BA_IEG_BOUNDARY_TRANSITIONS):
        return []
    times = [float(value) for value in candidate_times_seconds.detach().cpu()]
    minimum = float(minimum_state_duration_seconds)
    tolerance = float(tolerance_seconds)

    first: list[list[tuple[torch.Tensor, tuple[int, ...]]]] = [
        [] for _ in range(candidate_count)
    ]
    for candidate_index, boundary_time in enumerate(times):
        if boundary_time - support_start_seconds + tolerance >= minimum:
            first[candidate_index] = [
                (transition_logits[candidate_index, 0], (candidate_index,))
            ]

    second: list[list[tuple[torch.Tensor, tuple[int, ...]]]] = [
        [] for _ in range(candidate_count)
    ]
    for candidate_index, boundary_time in enumerate(times):
        prefixes: list[tuple[torch.Tensor, tuple[int, ...]]] = []
        for previous_index in range(candidate_index):
            if boundary_time - times[previous_index] + tolerance < minimum:
                continue
            prefixes.extend(
                (
                    score + transition_logits[candidate_index, 1],
                    path + (candidate_index,),
                )
                for score, path in first[previous_index]
            )
        second[candidate_index] = _select_top_k(prefixes, maximum_paths)

    third: list[list[tuple[torch.Tensor, tuple[int, ...]]]] = [
        [] for _ in range(candidate_count)
    ]
    for candidate_index, boundary_time in enumerate(times):
        if support_stop_seconds - boundary_time + tolerance < minimum:
            continue
        prefixes = []
        for previous_index in range(candidate_index):
            if boundary_time - times[previous_index] + tolerance < minimum:
                continue
            prefixes.extend(
                (
                    score + transition_logits[candidate_index, 2],
                    path + (candidate_index,),
                )
                for score, path in second[previous_index]
            )
        third[candidate_index] = _select_top_k(prefixes, maximum_paths)

    completed = [entry for entries in third for entry in entries]
    selected = _select_top_k(completed, maximum_paths)
    return [
        (score, (int(path[0]), int(path[1]), int(path[2])))
        for score, path in selected
    ]


def _physical_end_time_groups(
    bounds_seconds: torch.Tensor,
    active_mask: torch.Tensor,
    *,
    tolerance_seconds: float,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Group causal tokens by actual support-end time.

    The first tensor in each result is the physical time scalar; the second is
    the token-row index vector.  Row order is only used as a deterministic
    final tie breaker after physical coordinates have been sorted.
    """

    indices = torch.nonzero(active_mask, as_tuple=False).flatten()
    if not bool(indices.numel()):
        return []
    ordered = sorted(
        (int(index) for index in indices.detach().cpu()),
        key=lambda index: (
            float(bounds_seconds[index, 1]),
            float(bounds_seconds[index, 0]),
            index,
        ),
    )
    groups: list[list[int]] = []
    reference_end: float | None = None
    for index in ordered:
        token_end = float(bounds_seconds[index, 1])
        if reference_end is None or abs(token_end - reference_end) > tolerance_seconds:
            groups.append([index])
            reference_end = token_end
        else:
            groups[-1].append(index)
    result: list[tuple[torch.Tensor, torch.Tensor]] = []
    for group in groups:
        group_indices = torch.tensor(
            group, dtype=torch.long, device=bounds_seconds.device
        )
        group_time = bounds_seconds[group_indices, 1].mean()
        result.append((group_time, group_indices))
    return result


class BAIEGKBoundaryPathMarginalizer(nn.Module):
    """Retained-K legal path pooling over onset-positive physical tokens.

    Later recording-relative causal tokens may participate in this
    *retrospective event-level* boundary model, but no preprocessing view with
    future-sample access and no offline phase hint has a computational path.
    A downstream positive channel head must still enforce its own S0/S1 onset
    support rule; this primitive does not emit channel/SOZ logits.
    """

    implementation_id: Final[str] = BA_IEG_BOUNDARY_PATH_MARGINALIZER_ID

    def __init__(
        self,
        *,
        hidden_dim: int,
        maximum_paths: int = 8,
        minimum_state_duration_seconds: float = 0.25,
        path_temperature: float = 1.0,
        time_tolerance_seconds: float = 1e-6,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or maximum_paths <= 0:
            raise ValueError("hidden_dim and maximum_paths must be positive")
        if (
            not math.isfinite(minimum_state_duration_seconds)
            or minimum_state_duration_seconds <= 0.0
            or not math.isfinite(path_temperature)
            or path_temperature <= 0.0
            or not math.isfinite(time_tolerance_seconds)
            or time_tolerance_seconds < 0.0
        ):
            raise ValueError("boundary-path physical/temperature policy is invalid")
        self.hidden_dim = int(hidden_dim)
        self.maximum_paths = int(maximum_paths)
        self.minimum_state_duration_seconds = float(
            minimum_state_duration_seconds
        )
        self.path_temperature = float(path_temperature)
        self.time_tolerance_seconds = float(time_tolerance_seconds)
        self.boundary_norm = nn.LayerNorm(hidden_dim)
        self.boundary_transition_head = nn.Linear(
            hidden_dim, len(BA_IEG_BOUNDARY_TRANSITIONS)
        )
        self.state_pool_gate = nn.Linear(hidden_dim, len(BA_IEG_PHASE_STATES))
        self.event_projection = nn.Sequential(
            nn.Linear(
                len(BA_IEG_PHASE_STATES) * hidden_dim
                + len(BA_IEG_PHASE_STATES),
                hidden_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def _validate_inputs(
        self,
        batch: BAIEGCollatedEventBatch,
        encoder_output: BAIEGPhysicalTimeEncoderOutput,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        if not isinstance(batch, BAIEGCollatedEventBatch):
            raise TypeError("boundary marginalizer requires a registered event batch")
        if not isinstance(encoder_output, BAIEGPhysicalTimeEncoderOutput):
            raise TypeError(
                "boundary marginalizer requires a physical-time encoder output"
            )
        if encoder_output.source_input_batch_sha256 != batch.input_batch_sha256:
            raise ValueError("physical-time encoder output is bound to another batch")
        inputs = batch.positive_onset_inputs()
        if "phase_posterior" in inputs:  # pragma: no cover - contract guard
            raise RuntimeError("offline phase posterior reached boundary marginalization")
        active = inputs["token_row_mask"] & inputs["token_signal_mask"]
        if not torch.equal(active, encoder_output.token_mask):
            raise ValueError("encoder token mask exceeds registered positive-onset rows")
        if torch.any(inputs["token_future_sample_access"] & active):
            raise RuntimeError("future-dependent token reached boundary marginalization")
        embeddings = encoder_output.token_embeddings
        expected_prefix = tuple(active.shape)
        if embeddings.ndim != 3 or tuple(embeddings.shape[:2]) != expected_prefix:
            raise ValueError("encoder token embeddings do not align with the batch")
        if int(embeddings.shape[-1]) != self.hidden_dim:
            raise ValueError("boundary marginalizer hidden dimension drifted")
        if (
            not torch.isfinite(embeddings).all()
            or not torch.isfinite(encoder_output.token_onset_logits).all()
        ):
            raise ValueError("encoder onset output contains non-finite values")
        if torch.any(embeddings[~active] != 0) or torch.any(
            encoder_output.token_onset_logits[~active] != 0
        ):
            raise ValueError("masked encoder rows must remain zero")
        return inputs, active

    def forward(
        self,
        batch: BAIEGCollatedEventBatch,
        encoder_output: BAIEGPhysicalTimeEncoderOutput,
    ) -> BAIEGBoundaryPathMarginalizationOutput:
        inputs, active_mask = self._validate_inputs(batch, encoder_output)
        embeddings = encoder_output.token_embeddings
        bounds = inputs["token_time_bounds_seconds"]
        batch_size, token_count, _ = embeddings.shape
        device = embeddings.device
        dtype = embeddings.dtype

        event_groups: list[list[tuple[torch.Tensor, torch.Tensor]]] = []
        event_candidate_embeddings: list[torch.Tensor] = []
        event_candidate_logits: list[torch.Tensor] = []
        for batch_index in range(batch_size):
            groups = _physical_end_time_groups(
                bounds[batch_index],
                active_mask[batch_index],
                tolerance_seconds=self.time_tolerance_seconds,
            )
            event_groups.append(groups)
            if groups:
                candidate_embeddings = torch.stack(
                    [
                        embeddings[batch_index, indices].mean(dim=0)
                        for _, indices in groups
                    ]
                )
                candidate_logits = self.boundary_transition_head(
                    self.boundary_norm(candidate_embeddings)
                )
            else:
                candidate_embeddings = torch.zeros(
                    (0, self.hidden_dim), dtype=dtype, device=device
                )
                candidate_logits = torch.zeros(
                    (0, len(BA_IEG_BOUNDARY_TRANSITIONS)),
                    dtype=dtype,
                    device=device,
                )
            event_candidate_embeddings.append(candidate_embeddings)
            event_candidate_logits.append(candidate_logits)

        maximum_candidates = max(1, *(len(groups) for groups in event_groups))
        candidate_times = torch.zeros(
            (batch_size, maximum_candidates), dtype=bounds.dtype, device=device
        )
        candidate_logits_padded = torch.zeros(
            (
                batch_size,
                maximum_candidates,
                len(BA_IEG_BOUNDARY_TRANSITIONS),
            ),
            dtype=dtype,
            device=device,
        )
        candidate_mask = torch.zeros(
            (batch_size, maximum_candidates), dtype=torch.bool, device=device
        )
        for batch_index, groups in enumerate(event_groups):
            count = len(groups)
            if not count:
                continue
            candidate_times[batch_index, :count] = torch.stack(
                [time for time, _ in groups]
            )
            candidate_logits_padded[batch_index, :count] = event_candidate_logits[
                batch_index
            ]
            candidate_mask[batch_index, :count] = True

        path_times = torch.zeros(
            (batch_size, self.maximum_paths, len(BA_IEG_BOUNDARY_TRANSITIONS)),
            dtype=bounds.dtype,
            device=device,
        )
        path_scores = torch.zeros(
            (batch_size, self.maximum_paths), dtype=dtype, device=device
        )
        path_weights = torch.zeros_like(path_scores)
        path_mask = torch.zeros_like(path_scores, dtype=torch.bool)
        selected_paths: list[list[tuple[torch.Tensor, tuple[int, int, int]]]] = []

        for batch_index, groups in enumerate(event_groups):
            active = active_mask[batch_index]
            if not bool(active.any()) or not groups:
                selected_paths.append([])
                continue
            support_start = float(bounds[batch_index, active, 0].min())
            support_stop = float(bounds[batch_index, active, 1].max())
            local_times = torch.stack([time for time, _ in groups])
            paths = _top_k_legal_paths(
                local_times,
                event_candidate_logits[batch_index],
                support_start_seconds=support_start,
                support_stop_seconds=support_stop,
                maximum_paths=self.maximum_paths,
                minimum_state_duration_seconds=self.minimum_state_duration_seconds,
                tolerance_seconds=self.time_tolerance_seconds,
            )
            selected_paths.append(paths)
            count = len(paths)
            if not count:
                continue
            local_scores = torch.stack([score for score, _ in paths])
            local_weights = torch.softmax(
                local_scores / self.path_temperature, dim=0
            )
            path_scores[batch_index, :count] = local_scores
            path_weights[batch_index, :count] = local_weights
            path_mask[batch_index, :count] = True
            for path_index, (_, indices) in enumerate(paths):
                path_times[batch_index, path_index] = local_times[
                    torch.tensor(indices, dtype=torch.long, device=device)
                ]

        boundary_mean = torch.zeros(
            (batch_size, len(BA_IEG_BOUNDARY_TRANSITIONS)),
            dtype=bounds.dtype,
            device=device,
        )
        boundary_standard_deviation = torch.zeros_like(boundary_mean)
        boundary_support = torch.zeros(
            (batch_size, len(BA_IEG_BOUNDARY_TRANSITIONS), 2),
            dtype=bounds.dtype,
            device=device,
        )
        normalized_entropy = torch.zeros(batch_size, dtype=dtype, device=device)
        effective_path_count = torch.zeros(batch_size, dtype=dtype, device=device)
        for batch_index in range(batch_size):
            count = int(path_mask[batch_index].sum())
            if not count:
                continue
            weights = path_weights[batch_index, :count]
            times = path_times[batch_index, :count]
            mean = torch.sum(times * weights.unsqueeze(-1), dim=0)
            variance = torch.sum(
                (times - mean).square() * weights.unsqueeze(-1), dim=0
            )
            boundary_mean[batch_index] = mean
            boundary_standard_deviation[batch_index] = torch.sqrt(
                variance.clamp_min(0.0)
            )
            boundary_support[batch_index, :, 0] = times.min(dim=0).values
            boundary_support[batch_index, :, 1] = times.max(dim=0).values
            raw_entropy = -torch.sum(
                weights * torch.log(weights.clamp_min(torch.finfo(dtype).tiny))
            )
            effective_path_count[batch_index] = torch.exp(raw_entropy)
            if count > 1:
                normalized_entropy[batch_index] = raw_entropy / math.log(count)

        path_state_embeddings = torch.zeros(
            (
                batch_size,
                self.maximum_paths,
                len(BA_IEG_PHASE_STATES),
                self.hidden_dim,
            ),
            dtype=dtype,
            device=device,
        )
        path_state_opportunity = torch.zeros(
            (batch_size, self.maximum_paths, len(BA_IEG_PHASE_STATES)),
            dtype=torch.bool,
            device=device,
        )
        path_token_state_fraction = torch.zeros(
            (
                batch_size,
                self.maximum_paths,
                token_count,
                len(BA_IEG_PHASE_STATES),
            ),
            dtype=dtype,
            device=device,
        )
        state_gate_logits = self.state_pool_gate(embeddings)
        for batch_index in range(batch_size):
            active = active_mask[batch_index]
            if not bool(active.any()):
                continue
            token_start = bounds[batch_index, :, 0]
            token_stop = bounds[batch_index, :, 1]
            token_duration = (token_stop - token_start).clamp_min(
                torch.finfo(bounds.dtype).eps
            )
            support_start = token_start[active].min()
            support_stop = token_stop[active].max()
            for path_index in range(int(path_mask[batch_index].sum())):
                boundaries = path_times[batch_index, path_index]
                state_lower = torch.stack(
                    (support_start, boundaries[0], boundaries[1], boundaries[2])
                )
                state_upper = torch.stack(
                    (boundaries[0], boundaries[1], boundaries[2], support_stop)
                )
                for state_index in range(len(BA_IEG_PHASE_STATES)):
                    overlap = (
                        torch.minimum(token_stop, state_upper[state_index])
                        - torch.maximum(token_start, state_lower[state_index])
                    ).clamp_min(0.0)
                    overlap = torch.where(active, overlap, torch.zeros_like(overlap))
                    fraction = overlap / token_duration
                    path_token_state_fraction[
                        batch_index, path_index, :, state_index
                    ] = fraction
                    selected = active & (
                        overlap > self.time_tolerance_seconds
                    )
                    if not bool(selected.any()):
                        continue
                    log_pool_weight = (
                        state_gate_logits[batch_index, selected, state_index]
                        + torch.log(fraction[selected].clamp_min(torch.finfo(dtype).tiny))
                    )
                    pool_weight = torch.softmax(log_pool_weight, dim=0)
                    path_state_embeddings[
                        batch_index, path_index, state_index
                    ] = torch.sum(
                        embeddings[batch_index, selected]
                        * pool_weight.unsqueeze(-1),
                        dim=0,
                    )
                    path_state_opportunity[
                        batch_index, path_index, state_index
                    ] = True

        token_state_marginals = torch.sum(
            path_token_state_fraction * path_weights[:, :, None, None], dim=1
        )
        marginal_state_embeddings = torch.sum(
            path_state_embeddings * path_weights[:, :, None, None], dim=1
        )
        marginal_state_opportunity = torch.sum(
            path_state_opportunity.to(dtype=dtype)
            * path_weights[:, :, None],
            dim=1,
        )
        event_evaluable = path_mask.any(dim=1) & (
            marginal_state_opportunity > 0.0
        ).all(dim=1)
        event_projection_input = torch.cat(
            (
                marginal_state_embeddings.flatten(start_dim=1),
                marginal_state_opportunity,
            ),
            dim=-1,
        )
        event_embeddings = self.event_projection(event_projection_input)
        event_embeddings = torch.where(
            event_evaluable.unsqueeze(-1),
            event_embeddings,
            torch.zeros_like(event_embeddings),
        )

        return BAIEGBoundaryPathMarginalizationOutput(
            source_input_batch_sha256=batch.input_batch_sha256,
            candidate_boundary_times_seconds=candidate_times,
            candidate_transition_logits=candidate_logits_padded,
            candidate_mask=candidate_mask,
            path_boundary_times_seconds=path_times,
            path_log_scores=path_scores,
            path_weights=path_weights,
            path_mask=path_mask,
            boundary_mean_seconds=boundary_mean,
            boundary_standard_deviation_seconds=boundary_standard_deviation,
            boundary_retained_support_seconds=boundary_support,
            normalized_path_entropy=normalized_entropy,
            effective_retained_path_count=effective_path_count,
            token_state_marginals=token_state_marginals,
            path_state_embeddings=path_state_embeddings,
            path_state_opportunity_mask=path_state_opportunity,
            marginal_state_embeddings=marginal_state_embeddings,
            marginal_state_opportunity=marginal_state_opportunity,
            event_pooled_embeddings=event_embeddings,
            event_evaluable_mask=event_evaluable,
        )


__all__ = [
    "BA_IEG_BOUNDARY_PATH_MARGINALIZER_ID",
    "BA_IEG_BOUNDARY_TRANSITIONS",
    "BAIEGBoundaryPathMarginalizationOutput",
    "BAIEGKBoundaryPathMarginalizer",
]
