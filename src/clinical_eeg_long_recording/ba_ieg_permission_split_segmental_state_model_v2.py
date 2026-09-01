"""G1 temporal-forward candidate with lane-restricted clocks and topology.

This additive module deliberately leaves the frozen v1 implementation intact.
It reuses the tested v1 neural/DP primitives, but changes four model-input
permissions in an explicit v2 path:

* boundary/course projections receive only the prediction-frozen,
  support-relative clock (never recording-absolute seconds or support-edge
  flags);
* typed-unit local hidden values are recomputed by a separate signal-only
  projection and never reuse the global boundary projection or prefix hidden;
* offline losses see the causal trace only through the existing detached edge;
* the primary exact decoder permits one event bout only.  Clean return followed
  by another onset and direct S3 re-entry remain available solely as an
  explicitly labelled shadow output.

Recording-absolute times remain in output axes for raw replay and reporting.
They are not concatenated into a trainable projection.  This is a software
candidate, not a trained checkpoint or a G1 admission claim.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, fields, replace
import math
from typing import Final

import torch
from torch import nn
from torch.nn import functional as F

from .ba_ieg_g0_support_relative_shortcut_surface_v1 import (
    BA_IEG_G0_SUPPORT_RELATIVE_TIME_FEATURE_NAMES_V1,
    BAIEGG0SupportRelativeTimeSurfaceV1,
)
from .ba_ieg_permission_split_segmental_state_model_v1 import (
    BA_IEG_CAUSAL_TYPED_UNIT_TRACE_ID,
    BAIEGCausalTypedUnitTrace,
    BAIEGPermissionSplitSegmentalStateModel,
    BAIEGPermissionSplitSegmentalStateOutput,
    BAIEGSegmentalBoundaryContext,
    _GeneralCompletedPath,
    _collated_batch_input_sha256,
    _discrete_hazard_distribution,
    _hierarchical_pool_rows,
    _top_k_general_segmental_paths,
)
from .ba_ieg_segmental_dp_kernel_v1 import SegmentalPotentialsV1
from .ba_ieg_segmental_forward_backward_v1 import (
    build_lognormal_segment_duration_log_scores_v1,
    run_exact_segmental_forward_backward_v1,
)
from .ba_ieg_training_contract import (
    BA_IEG_PHASE_STATES,
    BA_IEG_TOKEN_SCALES,
    BAIEGCollatedEventBatch,
)


BA_IEG_PERMISSION_SPLIT_SEGMENTAL_STATE_MODEL_ID_V2: Final[str] = (
    "ba_ieg_permission_split_stable_clock_single_bout_segmental_state_model_v2"
)
BA_IEG_CAUSAL_TYPED_UNIT_TRACE_ID_V2: Final[str] = (
    "ba_ieg_signal_only_typed_unit_stable_clock_boundary_trace_v2"
)
BA_IEG_REENTRY_SHADOW_MODEL_ID_V2: Final[str] = (
    "ba_ieg_permission_split_reentry_full_topology_shadow_v2"
)

# Indices follow the immutable v1 transition roster.  Primary paths are:
# S0->S1->S2->S3, plus the registered S0->S2 and S1->S3 short paths.
# S3->S0 would make a second bout possible, so it is shadow-only together with
# direct S3->S1/S2 re-entry.
BA_IEG_V2_PRIMARY_TRANSITION_INDICES: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
BA_IEG_V2_SHADOW_ONLY_TRANSITION_INDICES: Final[tuple[int, ...]] = (5, 6, 7)

# The first nine registered G0 features use the stable candidate origin,
# physical duration, observed opportunity and quality-gap overlap.  The final
# two support-edge flags are intentionally excluded from every v2 neural lane.
BA_IEG_V2_BOUNDARY_TIME_FEATURE_NAMES: Final[tuple[str, ...]] = (
    BA_IEG_G0_SUPPORT_RELATIVE_TIME_FEATURE_NAMES_V1[:9]
)

_ACTIVE_TIME_SURFACE: ContextVar[
    BAIEGG0SupportRelativeTimeSurfaceV1 | None
] = ContextVar("ba_ieg_v2_active_time_surface", default=None)


@dataclass(frozen=True)
class BAIEGCausalTypedUnitTraceV2(BAIEGCausalTypedUnitTrace):
    """v2 permission identity over the unchanged, well-tested tensor shape."""

    def verify_shapes(self) -> None:
        if self.implementation_id != BA_IEG_CAUSAL_TYPED_UNIT_TRACE_ID_V2:
            raise ValueError("v2 causal typed-unit trace implementation drifted")
        payload = {field.name: getattr(self, field.name) for field in fields(self)}
        payload["implementation_id"] = BA_IEG_CAUSAL_TYPED_UNIT_TRACE_ID
        BAIEGCausalTypedUnitTrace(**payload).verify_shapes()


@dataclass(frozen=True)
class BAIEGPermissionSplitSegmentalStateOutputV2:
    """Primary single-bout output plus an optional explicit re-entry shadow."""

    primary: BAIEGPermissionSplitSegmentalStateOutput
    source_time_surface_receipt_sha256: str
    source_time_surface_learned_sha256: str
    shadow_reentry: BAIEGPermissionSplitSegmentalStateOutput | None = None
    implementation_id: str = BA_IEG_PERMISSION_SPLIT_SEGMENTAL_STATE_MODEL_ID_V2
    topology_semantics: str = (
        "primary_exact_single_bout_S0_S1_S2_S3_with_registered_short_paths"
    )
    training_status: str = "untrained_component_candidate_not_G1_admitted"

    def __post_init__(self) -> None:
        if self.primary.implementation_id != self.implementation_id:
            raise ValueError("v2 primary implementation identity drifted")
        shadow_indices = list(BA_IEG_V2_SHADOW_ONLY_TRANSITION_INDICES)
        if bool(self.primary.transition_mask[..., shadow_indices].any()):
            raise ValueError("v2 primary transition mask admitted a recurrent bout")
        if bool((self.primary.exact_s3_reentry_onset_boundary_mass > 1e-7).any()):
            raise ValueError("v2 primary posterior admitted S3 re-entry")
        if bool((self.primary.exact_recurrent_event_mass > 1e-7).any()):
            raise ValueError("v2 primary posterior admitted a recurrent event")
        if bool((self.primary.path_recurrent_cycle_count > 0).any()):
            raise ValueError("v2 retained primary paths admitted a recurrent cycle")
        if self.shadow_reentry is not None and (
            self.shadow_reentry.implementation_id
            != BA_IEG_REENTRY_SHADOW_MODEL_ID_V2
        ):
            raise ValueError("v2 re-entry output is not explicitly shadow-labelled")

    def __getattr__(self, name: str) -> object:
        # New consumers get primary semantics by default while retaining the
        # familiar v1 tensor attribute names.
        return getattr(self.primary, name)


def _validate_time_surface(
    batch: BAIEGCollatedEventBatch,
    context: BAIEGSegmentalBoundaryContext,
    time_surface: BAIEGG0SupportRelativeTimeSurfaceV1,
) -> None:
    if not isinstance(time_surface, BAIEGG0SupportRelativeTimeSurfaceV1):
        raise TypeError("v2 temporal forward requires a registered G0 time surface")
    time_surface.verify_integrity()
    if (
        time_surface.source_input_batch_sha256 != batch.input_batch_sha256
        or time_surface.source_context_receipt_sha256 != context.receipt_sha256
        or time_surface.event_ids != batch.event_ids
    ):
        raise ValueError("v2 time surface crosses batch/context identity")
    active = batch.token_row_mask & batch.token_signal_mask
    if not torch.equal(time_surface.token_active_mask, active):
        raise ValueError("v2 time surface active-token roster drifted")
    expected_absolute = torch.where(
        active.unsqueeze(-1),
        batch.token_time_bounds_seconds.detach().to(torch.float64),
        torch.zeros_like(
            batch.token_time_bounds_seconds.detach().to(torch.float64)
        ),
    )
    if not torch.equal(
        time_surface.absolute_token_bounds_recording_seconds_output_only,
        expected_absolute,
    ):
        raise ValueError("v2 time surface absolute output axis drifted from batch")
    if tuple(time_surface.feature_names[:9]) != BA_IEG_V2_BOUNDARY_TIME_FEATURE_NAMES:
        raise ValueError("v2 stable boundary-clock feature roster drifted")


def _single_bout_primary_decode(
    model: "BAIEGPermissionSplitSegmentalStateModelV2",
    output: BAIEGPermissionSplitSegmentalStateOutput,
    context: BAIEGSegmentalBoundaryContext,
) -> BAIEGPermissionSplitSegmentalStateOutput:
    """Replay exact DP/top-K after fail-closing every recurrent-bout edge."""

    shadow_indices = list(BA_IEG_V2_SHADOW_ONLY_TRANSITION_INDICES)
    transition_scores = output.transition_log_scores.clone()
    transition_mask = output.transition_mask.clone()
    transition_scores[..., shadow_indices] = 0.0
    transition_mask[..., shadow_indices] = False

    exact_state = torch.zeros_like(output.full_state_marginals)
    exact_transition = torch.zeros_like(output.full_transition_marginals)
    exact_start = torch.zeros_like(output.full_start_state_marginals)
    exact_end = torch.zeros_like(output.full_end_state_marginals)
    exact_segment_count = torch.zeros_like(output.full_segment_count_marginals)
    exact_onset = torch.zeros_like(output.exact_onset_boundary_mass)
    exact_primary_onset = torch.zeros_like(output.exact_primary_onset_boundary_mass)
    exact_secondary_onset = torch.zeros_like(
        output.exact_secondary_onset_boundary_mass
    )
    exact_s0_onset = torch.zeros_like(output.exact_s0_onset_boundary_mass)
    exact_s3_reentry_onset = torch.zeros_like(
        output.exact_s3_reentry_onset_boundary_mass
    )
    exact_offset = torch.zeros_like(output.exact_offset_boundary_mass)
    exact_event = torch.zeros_like(output.exact_event_mass)
    exact_null = torch.zeros_like(output.exact_null_mass)
    exact_left = torch.zeros_like(output.exact_left_censor_mass)
    exact_right = torch.zeros_like(output.exact_right_censor_mass)
    exact_both = torch.zeros_like(output.exact_both_censor_mass)
    exact_recurrent = torch.zeros_like(output.exact_recurrent_event_mass)
    exact_expected_bout = torch.zeros_like(output.exact_expected_event_bout_count)
    exact_expected_recurrent = torch.zeros_like(
        output.exact_expected_recurrent_bout_count
    )
    log_partition = torch.full_like(output.exact_path_log_partition, -torch.inf)

    path_start = torch.zeros_like(output.path_start_state_index)
    path_end = torch.zeros_like(output.path_end_state_index)
    path_transition_times = torch.zeros_like(output.path_transition_times_seconds)
    path_transition_mask = torch.zeros_like(output.path_transition_mask)
    path_transition_type = torch.full_like(output.path_transition_type_index, -1)
    path_segment_bounds = torch.zeros_like(output.path_segment_bounds_seconds)
    path_segment_state = torch.full_like(output.path_segment_state_index, -1)
    path_segment_mask = torch.zeros_like(output.path_segment_mask)
    path_cycle_count = torch.zeros_like(output.path_recurrent_cycle_count)
    path_scores_out = torch.zeros_like(output.path_log_scores)
    path_posterior_out = torch.zeros_like(output.path_posterior_mass)
    path_weights_out = torch.zeros_like(output.path_weights_conditional_on_retained)
    path_mask_out = torch.zeros_like(output.path_mask)
    retained_mass_out = torch.zeros_like(output.retained_path_mass_fraction)
    retained_state = torch.zeros_like(output.retained_state_marginals)

    duration_location = output.duration_location_log_seconds
    duration_scale = torch.exp(output.duration_scale_log_seconds)
    minimum_duration = output.minimum_state_duration_seconds
    batch_size = int(output.lattice_cell_bounds_seconds.shape[0])
    for batch_index in range(batch_size):
        bounds = output.lattice_cell_bounds_seconds[batch_index]
        cell_count = int((bounds[:, 1] > bounds[:, 0]).sum())
        if cell_count < 1:
            continue
        maximum_segments = min(model.maximum_segments, cell_count)
        physical_duration = output.lattice_physical_duration_seconds[
            batch_index, :cell_count
        ]
        duration_log_scores = build_lognormal_segment_duration_log_scores_v1(
            physical_duration=physical_duration,
            duration_location_log_seconds=duration_location,
            duration_scale_log_seconds=duration_scale,
            minimum_state_duration_seconds=minimum_duration,
        )
        potentials = SegmentalPotentialsV1(
            emission_log_density=output.offline_state_emission_log_prob[
                batch_index, :cell_count
            ],
            opportunity_duration=output.lattice_opportunity_duration_seconds[
                batch_index, :cell_count
            ],
            physical_duration=physical_duration,
            transition_log_scores=transition_scores[batch_index, :cell_count],
            transition_mask=transition_mask[batch_index, :cell_count],
            start_log_scores=output.start_state_log_scores[batch_index],
            end_log_scores=output.end_state_log_scores[batch_index],
            event_log_score=output.event_presence_log_scores[batch_index, 1],
            no_event_log_score=output.event_presence_log_scores[batch_index, 0],
            segment_duration_log_scores=duration_log_scores,
            maximum_segments=maximum_segments,
            left_censoring_possible=bool(
                context.left_censoring_possible[batch_index]
            ),
            right_censoring_possible=bool(
                context.right_censoring_possible[batch_index]
            ),
        )
        posterior = run_exact_segmental_forward_backward_v1(potentials)
        log_partition[batch_index] = posterior.exact_log_partition
        exact_state[batch_index, :cell_count] = posterior.state_marginal
        exact_transition[batch_index, :cell_count] = posterior.transition_marginal
        exact_start[batch_index] = posterior.start_state_marginal
        exact_end[batch_index] = posterior.end_state_marginal
        segment_count = int(posterior.segment_count_marginal.numel())
        exact_segment_count[batch_index, :segment_count] = (
            posterior.segment_count_marginal
        )
        exact_onset[batch_index, :cell_count] = posterior.onset_boundary_mass
        exact_primary_onset[batch_index, :cell_count] = (
            posterior.primary_onset_boundary_mass
        )
        exact_secondary_onset[batch_index, :cell_count] = (
            posterior.secondary_onset_boundary_mass
        )
        exact_s0_onset[batch_index, :cell_count] = posterior.s0_onset_boundary_mass
        exact_s3_reentry_onset[batch_index, :cell_count] = (
            posterior.s3_reentry_onset_boundary_mass
        )
        exact_offset[batch_index, :cell_count] = posterior.offset_boundary_mass
        exact_event[batch_index] = posterior.event_mass
        exact_null[batch_index] = posterior.null_mass
        exact_left[batch_index] = posterior.left_censor_mass
        exact_right[batch_index] = posterior.right_censor_mass
        exact_both[batch_index] = posterior.both_censor_mass
        exact_recurrent[batch_index] = posterior.recurrent_event_mass
        exact_expected_bout[batch_index] = posterior.expected_event_bout_count
        exact_expected_recurrent[batch_index] = (
            posterior.expected_recurrent_bout_count
        )

        paths = _top_k_general_segmental_paths(
            state_emission_log_prob=output.offline_state_emission_log_prob[
                batch_index, :cell_count
            ],
            opportunity_duration_seconds=(
                output.lattice_opportunity_duration_seconds[
                    batch_index, :cell_count
                ]
            ),
            physical_duration_seconds=physical_duration,
            transition_log_scores=transition_scores[batch_index, :cell_count],
            transition_mask=transition_mask[batch_index, :cell_count],
            start_log_scores=output.start_state_log_scores[batch_index],
            end_log_scores=output.end_state_log_scores[batch_index],
            event_log_score=output.event_presence_log_scores[batch_index, 1],
            no_event_log_score=output.event_presence_log_scores[batch_index, 0],
            duration_location=duration_location,
            duration_scale=duration_scale,
            minimum_duration_seconds=minimum_duration,
            left_censoring_possible=bool(
                context.left_censoring_possible[batch_index]
            ),
            right_censoring_possible=bool(
                context.right_censoring_possible[batch_index]
            ),
            maximum_segments=maximum_segments,
            maximum_paths=model.maximum_paths,
        )
        if paths:
            scores = torch.stack([path.score for path in paths])
            weights = torch.softmax(scores, dim=0)
            path_posterior = torch.exp(scores - posterior.exact_log_partition)
            retained_mass_out[batch_index] = path_posterior.sum().clamp(0.0, 1.0)
            count = len(paths)
            path_mask_out[batch_index, :count] = True
            path_scores_out[batch_index, :count] = scores
            path_posterior_out[batch_index, :count] = path_posterior
            path_weights_out[batch_index, :count] = weights
            for path_index, path in enumerate(paths):
                path_start[batch_index, path_index] = path.start_state
                path_end[batch_index, path_index] = path.end_state
                transition_count = len(path.transition_indices)
                if transition_count:
                    boundary_indices = torch.tensor(
                        path.transition_indices,
                        dtype=torch.long,
                        device=bounds.device,
                    )
                    path_transition_times[
                        batch_index, path_index, :transition_count
                    ] = bounds[boundary_indices, 1]
                    path_transition_mask[
                        batch_index, path_index, :transition_count
                    ] = True
                    path_transition_type[
                        batch_index, path_index, :transition_count
                    ] = torch.tensor(
                        path.transition_edge_indices,
                        dtype=torch.long,
                        device=bounds.device,
                    )
                segment_starts = (0,) + tuple(
                    index + 1 for index in path.transition_indices
                )
                segment_ends = tuple(path.transition_indices) + (cell_count - 1,)
                for segment_index, (state, start, end) in enumerate(
                    zip(path.state_sequence, segment_starts, segment_ends)
                ):
                    path_segment_bounds[
                        batch_index, path_index, segment_index, 0
                    ] = bounds[start, 0]
                    path_segment_bounds[
                        batch_index, path_index, segment_index, 1
                    ] = bounds[end, 1]
                    path_segment_state[
                        batch_index, path_index, segment_index
                    ] = state
                    path_segment_mask[
                        batch_index, path_index, segment_index
                    ] = True
                    retained_state[batch_index, start : end + 1, state] += weights[
                        path_index
                    ]

    event_evaluable = (
        torch.isfinite(log_partition)
        & path_mask_out.any(dim=1)
        & output.lattice_cell_mask.any(dim=1)
    )
    return replace(
        output,
        implementation_id=BA_IEG_PERMISSION_SPLIT_SEGMENTAL_STATE_MODEL_ID_V2,
        split_or_reentry_review_logits=torch.zeros_like(
            output.split_or_reentry_review_logits
        ),
        transition_log_scores=transition_scores,
        transition_mask=transition_mask,
        exact_path_log_partition=log_partition,
        full_state_marginals=exact_state,
        full_transition_marginals=exact_transition,
        full_start_state_marginals=exact_start,
        full_end_state_marginals=exact_end,
        full_segment_count_marginals=exact_segment_count,
        exact_onset_boundary_mass=exact_onset,
        exact_primary_onset_boundary_mass=exact_primary_onset,
        exact_secondary_onset_boundary_mass=exact_secondary_onset,
        exact_s0_onset_boundary_mass=exact_s0_onset,
        exact_s3_reentry_onset_boundary_mass=exact_s3_reentry_onset,
        exact_offset_boundary_mass=exact_offset,
        exact_event_mass=exact_event,
        exact_null_mass=exact_null,
        exact_left_censor_mass=exact_left,
        exact_right_censor_mass=exact_right,
        exact_both_censor_mass=exact_both,
        exact_recurrent_event_mass=exact_recurrent,
        exact_expected_event_bout_count=exact_expected_bout,
        exact_expected_recurrent_bout_count=exact_expected_recurrent,
        path_start_state_index=path_start,
        path_end_state_index=path_end,
        path_transition_times_seconds=path_transition_times,
        path_transition_mask=path_transition_mask,
        path_transition_type_index=path_transition_type,
        path_segment_bounds_seconds=path_segment_bounds,
        path_segment_state_index=path_segment_state,
        path_segment_mask=path_segment_mask,
        path_recurrent_cycle_count=path_cycle_count,
        path_log_scores=path_scores_out,
        path_posterior_mass=path_posterior_out,
        path_weights_conditional_on_retained=path_weights_out,
        path_mask=path_mask_out,
        retained_path_mass_fraction=retained_mass_out,
        residual_path_mass_fraction=(1.0 - retained_mass_out).clamp(0.0, 1.0),
        retained_state_marginals=retained_state,
        event_evaluable_mask=event_evaluable,
    )


class BAIEGPermissionSplitSegmentalStateModelV2(
    BAIEGPermissionSplitSegmentalStateModel
):
    """Stable-clock boundary model with a signal-only typed identity lane."""

    implementation_id: Final[str] = BA_IEG_PERMISSION_SPLIT_SEGMENTAL_STATE_MODEL_ID_V2

    def __init__(
        self,
        *,
        feature_dim: int,
        hidden_dim: int = 64,
        maximum_segments: int = 8,
        maximum_paths: int = 8,
        minimum_state_duration_seconds: tuple[float, float, float, float] = (
            0.0,
            0.25,
            0.25,
            0.0,
        ),
        duration_location_seconds: tuple[float, float, float, float] = (
            8.0,
            2.0,
            12.0,
            8.0,
        ),
        duration_scale_initializer: float = 0.75,
        dropout: float = 0.0,
        time_tolerance_seconds: float = 1e-6,
        relative_reference_threshold: float = 0.2,
        maximum_local_footprint: int = 6,
        allow_heuristic_phase_posterior: bool = False,
        emit_reentry_shadow: bool = False,
    ) -> None:
        if type(emit_reentry_shadow) is not bool:
            raise TypeError("emit_reentry_shadow must be boolean")
        super().__init__(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            maximum_segments=maximum_segments,
            maximum_paths=maximum_paths,
            minimum_state_duration_seconds=minimum_state_duration_seconds,
            duration_location_seconds=duration_location_seconds,
            duration_scale_initializer=duration_scale_initializer,
            dropout=dropout,
            time_tolerance_seconds=time_tolerance_seconds,
            relative_reference_threshold=relative_reference_threshold,
            maximum_local_footprint=maximum_local_footprint,
            allow_heuristic_phase_posterior=allow_heuristic_phase_posterior,
        )
        boundary_time_dim = len(BA_IEG_V2_BOUNDARY_TIME_FEATURE_NAMES)
        # Replace only the v2 instance's projections.  Frozen v1 source and
        # construction remain unchanged.
        self.causal_value_projection = nn.Linear(
            2 * feature_dim + boundary_time_dim, hidden_dim
        )
        self.offline_value_projection = nn.Linear(
            2 * feature_dim
            + boundary_time_dim
            + len(BA_IEG_PHASE_STATES)
            + 1,
            hidden_dim,
        )
        self.typed_signal_value_projection = nn.Linear(2 * feature_dim, hidden_dim)
        self.typed_signal_input_norm = nn.LayerNorm(hidden_dim)
        self.emit_reentry_shadow = emit_reentry_shadow

    def _project_token_lanes(
        self, batch: BAIEGCollatedEventBatch
    ) -> tuple[
        dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        time_surface = _ACTIVE_TIME_SURFACE.get()
        if time_surface is None:
            raise RuntimeError("v2 stable-clock surface was not bound to forward")
        causal = batch.onset_causal_inputs()
        full = batch.model_inputs()
        causal_active = causal["token_row_mask"] & causal["token_signal_mask"]
        full_active = full["token_row_mask"] & full["token_signal_mask"]
        if torch.any(causal["token_future_sample_access"] & causal_active):
            raise RuntimeError("future-dependent token reached the v2 causal lane")

        values = batch.token_values
        feature_mask = batch.token_feature_mask
        masked_values = torch.where(feature_mask, values, torch.zeros_like(values))
        # No recording-absolute tensor and no support-edge flag is read here.
        boundary_time = time_surface.learned_time_features[..., :9].to(
            dtype=values.dtype, device=values.device
        )
        scale_index = batch.token_scale_index.clamp(
            min=0, max=len(BA_IEG_TOKEN_SCALES) - 1
        )
        causal_input = torch.cat(
            (masked_values, feature_mask.to(dtype=values.dtype), boundary_time),
            dim=-1,
        )
        causal_hidden = self.causal_value_projection(
            causal_input
        ) + self.causal_scale_embedding(scale_index)
        causal_hidden = torch.where(
            causal_active.unsqueeze(-1),
            self.causal_input_norm(causal_hidden),
            torch.zeros_like(causal_hidden),
        )

        if self.allow_heuristic_phase_posterior:
            phase = torch.where(
                batch.token_phase_context_mask.unsqueeze(-1),
                batch.phase_posterior,
                torch.zeros_like(batch.phase_posterior),
            )
        else:
            phase = torch.zeros(
                (*values.shape[:2], len(BA_IEG_PHASE_STATES)),
                dtype=values.dtype,
                device=values.device,
            )
        offline_input = torch.cat(
            (
                masked_values,
                feature_mask.to(dtype=values.dtype),
                boundary_time,
                phase.to(dtype=values.dtype),
                batch.token_future_sample_access.unsqueeze(-1).to(
                    dtype=values.dtype
                ),
            ),
            dim=-1,
        )
        offline_hidden = self.offline_value_projection(
            offline_input
        ) + self.offline_scale_embedding(scale_index)
        offline_hidden = torch.where(
            full_active.unsqueeze(-1),
            self.offline_input_norm(offline_hidden),
            torch.zeros_like(offline_hidden),
        )
        return full, causal_hidden, offline_hidden, causal_active, full_active

    def _signal_only_typed_trace(
        self,
        batch: BAIEGCollatedEventBatch,
        trace: BAIEGCausalTypedUnitTrace,
    ) -> BAIEGCausalTypedUnitTrace:
        values = batch.token_values
        feature_mask = batch.token_feature_mask
        masked_values = torch.where(feature_mask, values, torch.zeros_like(values))
        typed_input = torch.cat(
            (masked_values, feature_mask.to(dtype=values.dtype)), dim=-1
        )
        typed_token_hidden = self.typed_signal_input_norm(
            self.typed_signal_value_projection(typed_input)
        )
        causal = batch.onset_causal_inputs()
        causal_active = causal["token_row_mask"] & causal["token_signal_mask"]
        typed_token_hidden = torch.where(
            causal_active.unsqueeze(-1),
            typed_token_hidden,
            torch.zeros_like(typed_token_hidden),
        )
        local = torch.zeros_like(trace.typed_unit_local_hidden)
        for batch_index in range(len(batch.event_ids)):
            causal_indices = torch.nonzero(
                causal_active[batch_index], as_tuple=False
            ).flatten().tolist()
            for group_index in torch.nonzero(
                trace.group_mask[batch_index], as_tuple=False
            ).flatten().tolist():
                time = float(trace.group_times_seconds[batch_index, group_index])
                for typed_index in torch.nonzero(
                    trace.typed_unit_time_mask[batch_index, group_index],
                    as_tuple=False,
                ).flatten().tolist():
                    aliases = torch.nonzero(
                        trace.typed_unit_source_analysis_unit_mask[
                            batch_index, typed_index
                        ],
                        as_tuple=False,
                    ).flatten()
                    alias_set = {int(value) for value in aliases.tolist()}
                    selected = [
                        index
                        for index in causal_indices
                        if int(batch.token_unit_index[batch_index, index])
                        in alias_set
                        and bool(batch.token_positive_onset_mask[batch_index, index])
                        and abs(
                            float(
                                batch.token_time_bounds_seconds[
                                    batch_index, index, 1
                                ]
                            )
                            - time
                        )
                        <= self.time_tolerance_seconds
                    ]
                    local[batch_index, group_index, typed_index] = (
                        _hierarchical_pool_rows(
                            typed_token_hidden[batch_index],
                            selected,
                            token_unit_index=batch.token_unit_index[batch_index],
                            token_scale_index=batch.token_scale_index[batch_index],
                            unit_view_index=batch.unit_view_index[batch_index],
                            view_reference_family_index=(
                                batch.view_reference_family_index[batch_index]
                            ),
                            unit_reference_matrix=(
                                batch.unit_reference_matrix[batch_index]
                            ),
                            row_weights=None,
                            relative_reference_threshold=(
                                self.relative_reference_threshold
                            ),
                            maximum_local_footprint=self.maximum_local_footprint,
                        )
                    )
        payload = {field.name: getattr(trace, field.name) for field in fields(trace)}
        payload.update(
            implementation_id=BA_IEG_CAUSAL_TYPED_UNIT_TRACE_ID_V2,
            typed_unit_local_hidden=local,
        )
        result = BAIEGCausalTypedUnitTraceV2(**payload)
        result.verify_shapes()
        return result

    def forward(
        self,
        batch: BAIEGCollatedEventBatch,
        context: BAIEGSegmentalBoundaryContext,
        time_surface: BAIEGG0SupportRelativeTimeSurfaceV1,
    ) -> BAIEGPermissionSplitSegmentalStateOutputV2:
        self._validate_context(batch, context)
        _validate_time_surface(batch, context, time_surface)
        if batch.input_batch_sha256 != _collated_batch_input_sha256(batch):
            raise ValueError("v2 BA-IEG inputs changed after registration")
        token = _ACTIVE_TIME_SURFACE.set(time_surface)
        try:
            full_topology = super().forward(batch, context)
        finally:
            _ACTIVE_TIME_SURFACE.reset(token)

        typed_trace = self._signal_only_typed_trace(
            batch, full_topology.causal_typed_unit_trace
        )
        full_topology = replace(
            full_topology,
            causal_typed_unit_trace=typed_trace,
            implementation_id=(
                BA_IEG_PERMISSION_SPLIT_SEGMENTAL_STATE_MODEL_ID_V2
            ),
        )
        primary = _single_bout_primary_decode(self, full_topology, context)
        shadow = None
        if self.emit_reentry_shadow:
            shadow = replace(
                full_topology,
                implementation_id=BA_IEG_REENTRY_SHADOW_MODEL_ID_V2,
            )
        return BAIEGPermissionSplitSegmentalStateOutputV2(
            primary=primary,
            source_time_surface_receipt_sha256=time_surface.receipt_sha256,
            source_time_surface_learned_sha256=time_surface.learned_surface_sha256,
            shadow_reentry=shadow,
        )


__all__ = [
    "BA_IEG_CAUSAL_TYPED_UNIT_TRACE_ID_V2",
    "BA_IEG_PERMISSION_SPLIT_SEGMENTAL_STATE_MODEL_ID_V2",
    "BA_IEG_REENTRY_SHADOW_MODEL_ID_V2",
    "BA_IEG_V2_BOUNDARY_TIME_FEATURE_NAMES",
    "BA_IEG_V2_PRIMARY_TRANSITION_INDICES",
    "BA_IEG_V2_SHADOW_ONLY_TRANSITION_INDICES",
    "BAIEGCausalTypedUnitTraceV2",
    "BAIEGPermissionSplitSegmentalStateModelV2",
    "BAIEGPermissionSplitSegmentalStateOutputV2",
]
