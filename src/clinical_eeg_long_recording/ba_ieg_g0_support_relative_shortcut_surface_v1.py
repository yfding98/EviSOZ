"""Additive G0 support-relative clock and shortcut-input surface.

The frozen v1 segmental model consumes recording-absolute token coordinates.
That behaviour is intentionally left untouched.  This module defines the
separate v1.3-min input surface required before a new checkpoint may be
trained:

* recording-absolute time is retained only for identity replay and output;
* the local origin is the candidate anchor frozen in a validated source-train
  patient-OOF prediction roster, never the mutable left edge of queried
  support;
* learned time features use stable-origin-relative start/stop/midpoint,
  physical duration, signed observed-opportunity displacement, quality-gap
  overlap, and support edge flags;
* position-only and signal-shuffle controls bind the exact same non-signal
  tensor surface as the main arm; and
* provider/candidate receipts remain lineage, not numeric learned features.

The surface is target-free.  It accepts only a registered BA-IEG batch, its
EEG/QC-derived segmental context, and a prediction-frozen stable-origin
registry.  It does not accept an onset, seizure label, channel target,
annotation, spreadsheet, or clinical text.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Final, Mapping, Sequence

import torch

from .ba_ieg_permission_split_segmental_state_model_v1 import (
    BAIEGSegmentalBoundaryContext,
)
from .ba_ieg_g0_a1_candidate_roster_v1 import (
    BA_IEG_G0_A1_CANDIDATE_ORIGINS,
    validate_ba_ieg_g0_a1_prediction_roster_v1,
)
from .ba_ieg_training_contract import BAIEGCollatedEventBatch


BA_IEG_G0_STABLE_ORIGIN_REGISTRY_SCHEMA_V1: Final[
    str
] = "ba_ieg_g0_patient_oof_candidate_anchor_registry_v1"
BA_IEG_G0_SUPPORT_RELATIVE_TIME_SURFACE_SCHEMA_V1: Final[
    str
] = "ba_ieg_g0_support_relative_time_surface_v1"
BA_IEG_G0_SHORTCUT_SURFACE_SCHEMA_V1: Final[
    str
] = "ba_ieg_g0_exact_nonsignal_shortcut_surface_v1"
BA_IEG_G0_SUPPORT_RELATIVE_TIME_FEATURE_NAMES_V1: Final[tuple[str, ...]] = (
    "asinh_stable_origin_relative_start_seconds_over_60",
    "asinh_stable_origin_relative_stop_seconds_over_60",
    "asinh_stable_origin_relative_midpoint_seconds_over_60",
    "log1p_physical_duration_seconds",
    "asinh_signed_observed_opportunity_displacement_at_start_seconds_over_60",
    "asinh_signed_observed_opportunity_displacement_at_stop_seconds_over_60",
    "asinh_signed_observed_opportunity_displacement_at_midpoint_seconds_over_60",
    "quality_gap_overlap_fraction",
    "quality_gap_overlap_flag",
    "touches_left_support_edge_flag",
    "touches_right_support_edge_flag",
)

_SHA256_ALPHABET: Final[frozenset[str]] = frozenset("0123456789abcdef")
_PROVIDER_LINEAGE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "event_id",
        "patient_uid",
        "recording_id",
        "candidate_id",
        "candidate_origin",
        "provider_id",
        "prediction_roster_id",
        "prediction_roster_receipt_sha256",
        "provider_prediction_receipt_sha256",
        "decoder_policy_receipt_sha256",
        "source_candidate_receipt_sha256",
    }
)

# Everything below is visible to, or structurally controls, the proposed
# v1.3-min causal/offline model while containing no signal value.  The two
# shortcut arms are forbidden from constructing a smaller convenience subset.
_BATCH_NONSIGNAL_TENSORS: Final[tuple[str, ...]] = (
    "token_feature_mask",
    "token_row_mask",
    "token_signal_mask",
    "token_unit_index",
    "token_view_index",
    "token_scale_index",
    "token_family_mask",
    "token_future_sample_access",
    "token_onset_evidence_mask",
    "token_positive_onset_mask",
    "token_phase_context_mask",
    "view_row_mask",
    "view_temporal_role_index",
    "view_dependency_policy_index",
    "view_reference_family_index",
    "view_future_sample_access",
    "view_onset_evidence_authorized",
    "unit_row_mask",
    "unit_view_index",
    "unit_reference_matrix",
    "unit_evidence_mask",
    "unit_family_mask",
    "physical_xyz",
    "physical_xyz_mask",
    "physical_evidence_mask",
)
_TIME_NONSIGNAL_TENSORS: Final[tuple[str, ...]] = (
    "learned_time_features",
    "support_relative_token_bounds_seconds",
    "token_active_mask",
    "support_relative_observed_intervals_seconds",
    "observed_support_mask",
    "support_relative_quality_gap_intervals_seconds",
    "quality_gap_mask",
)
_FORBIDDEN_POSITION_ONLY_INPUT_NAMES: Final[tuple[str, ...]] = (
    "token_values",
    "phase_posterior",
    "deterministic_values",
    "deterministic_value_mask",
    "deterministic_row_mask",
    "deterministic_time_bounds_seconds",
    "detector_score",
    "reference_onset",
    "reference_offset",
    "channel_or_soz_target",
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or set(value).difference(_SHA256_ALPHABET)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TypeError(f"{context} must be a non-empty trimmed string")
    return value


def _tensor_sha256(value: torch.Tensor) -> str:
    if not isinstance(value, torch.Tensor):
        raise TypeError("tensor receipt requires a torch.Tensor")
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _frozen_copy(value: torch.Tensor) -> torch.Tensor:
    result = value.detach().clone()
    result.requires_grad_(False)
    return result


def _clean_intervals(
    support: Sequence[tuple[float, float]],
    gaps: Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    clean: list[tuple[float, float]] = []
    for support_start, support_stop in support:
        cursor = support_start
        for gap_start, gap_stop in gaps:
            if gap_stop <= cursor or gap_start >= support_stop:
                continue
            if gap_start > cursor:
                clean.append((cursor, min(gap_start, support_stop)))
            cursor = max(cursor, min(gap_stop, support_stop))
            if cursor >= support_stop:
                break
        if cursor < support_stop:
            clean.append((cursor, support_stop))
    return [(start, stop) for start, stop in clean if stop > start]


def _signed_opportunity_displacement(
    coordinate: float,
    origin: float,
    clean_intervals: Sequence[tuple[float, float]],
) -> float:
    if coordinate >= origin:
        left, right, sign = origin, coordinate, 1.0
    else:
        left, right, sign = coordinate, origin, -1.0
    return sign * sum(
        max(0.0, min(right, stop) - max(left, start)) for start, stop in clean_intervals
    )


def _gap_overlap(
    start: float,
    stop: float,
    gaps: Sequence[tuple[float, float]],
) -> float:
    return sum(
        max(0.0, min(stop, gap_stop) - max(start, gap_start))
        for gap_start, gap_stop in gaps
    )


@dataclass(frozen=True)
class BAIEGG0StableOriginRegistryV1:
    """Prediction-frozen, target-free local origins for one collated batch.

    The numeric anchors are retained on the recording clock only for replay
    and subtraction.  A model never receives them directly.  Every anchor is
    resolved from a candidate in a validated source-train patient-OOF
    prediction roster, before any reference-event or SOZ target join.
    """

    source_input_batch_sha256: str
    event_ids: tuple[str, ...]
    recording_ids: tuple[str, ...]
    patient_uids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    candidate_origins: tuple[str, ...]
    source_candidate_receipt_sha256s: tuple[str, ...]
    prediction_roster_id: str
    prediction_roster_receipt_sha256: str
    provider_id: str
    provider_prediction_receipt_sha256: str
    decoder_policy_receipt_sha256: str
    stable_origin_recording_seconds_output_only: torch.Tensor
    origin_authority: str = (
        "source_train_patient_oof_prediction_frozen_candidate_anchor_"
        "before_reference_join"
    )
    schema_version: str = BA_IEG_G0_STABLE_ORIGIN_REGISTRY_SCHEMA_V1
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _sha256(self.source_input_batch_sha256, "source batch receipt")
        if self.schema_version != BA_IEG_G0_STABLE_ORIGIN_REGISTRY_SCHEMA_V1:
            raise ValueError("stable-origin registry schema drifted")
        if self.origin_authority != (
            "source_train_patient_oof_prediction_frozen_candidate_anchor_"
            "before_reference_join"
        ):
            raise ValueError("stable-origin authority drifted")
        batch_size = len(self.event_ids)
        aligned = (
            self.recording_ids,
            self.patient_uids,
            self.candidate_ids,
            self.candidate_origins,
            self.source_candidate_receipt_sha256s,
        )
        if batch_size < 1 or any(len(values) != batch_size for values in aligned):
            raise ValueError("stable-origin registry rows do not align")
        if len(set(self.event_ids)) != batch_size:
            raise ValueError("stable-origin registry repeats an event")
        if len(set(self.candidate_ids)) != batch_size:
            raise ValueError(
                "one frozen candidate cannot seed multiple events; create "
                "content-addressed reference-free child candidates first"
            )
        for values, context in (
            (self.event_ids, "event ID"),
            (self.recording_ids, "recording ID"),
            (self.patient_uids, "patient UID"),
            (self.candidate_ids, "candidate ID"),
        ):
            for value in values:
                _identifier(value, context)
        if any(
            origin not in BA_IEG_G0_A1_CANDIDATE_ORIGINS
            for origin in self.candidate_origins
        ):
            raise ValueError("stable origin has an unsupported candidate origin")
        for digest in self.source_candidate_receipt_sha256s:
            _sha256(digest, "source candidate receipt")
        _identifier(self.prediction_roster_id, "prediction roster ID")
        _identifier(self.provider_id, "provider ID")
        for digest, context in (
            (self.prediction_roster_receipt_sha256, "prediction roster receipt"),
            (self.provider_prediction_receipt_sha256, "provider prediction receipt"),
            (self.decoder_policy_receipt_sha256, "decoder policy receipt"),
        ):
            _sha256(digest, context)
        origins = _frozen_copy(self.stable_origin_recording_seconds_output_only)
        if tuple(origins.shape) != (batch_size,) or not origins.is_floating_point():
            raise ValueError("stable-origin tensor shape/dtype does not align")
        if not torch.isfinite(origins).all():
            raise ValueError("stable origins must be finite")
        object.__setattr__(self, "stable_origin_recording_seconds_output_only", origins)
        object.__setattr__(self, "receipt_sha256", self._receipt_hash())

    def _receipt_hash(self) -> str:
        return _canonical_sha256(
            {
                "schema_version": self.schema_version,
                "source_input_batch_sha256": self.source_input_batch_sha256,
                "event_ids": list(self.event_ids),
                "recording_ids": list(self.recording_ids),
                "patient_uids": list(self.patient_uids),
                "candidate_ids": list(self.candidate_ids),
                "candidate_origins": list(self.candidate_origins),
                "source_candidate_receipt_sha256s": list(
                    self.source_candidate_receipt_sha256s
                ),
                "prediction_roster_id": self.prediction_roster_id,
                "prediction_roster_receipt_sha256": (
                    self.prediction_roster_receipt_sha256
                ),
                "provider_id": self.provider_id,
                "provider_prediction_receipt_sha256": (
                    self.provider_prediction_receipt_sha256
                ),
                "decoder_policy_receipt_sha256": (self.decoder_policy_receipt_sha256),
                "origin_authority": self.origin_authority,
                "stable_origin_numeric_model_input": False,
                "reference_or_soz_target_join_opened": False,
                "one_frozen_candidate_seed_at_most_one_model_event": True,
                "parent_candidate_copied_to_multiple_model_events": False,
                "reference_free_child_candidate_split_required_before_multi_event_use": True,
                "reference_free_child_candidate_split_registry_materialized": False,
                "current_registry_scope": "source_train_g0_patient_oof_only",
                "source_dev_eval_private_reference_free_registry_api_materialized": False,
                "stable_origin_tensor_sha256": _tensor_sha256(
                    self.stable_origin_recording_seconds_output_only
                ),
            }
        )

    def verify_integrity(self) -> None:
        if self.receipt_sha256 != self._receipt_hash():
            raise ValueError("stable-origin registry changed after registration")

    def provider_lineage(self) -> list[dict[str, Any]]:
        self.verify_integrity()
        return [
            {
                "event_id": event_id,
                "patient_uid": patient_uid,
                "recording_id": recording_id,
                "candidate_id": candidate_id,
                "candidate_origin": candidate_origin,
                "provider_id": self.provider_id,
                "prediction_roster_id": self.prediction_roster_id,
                "prediction_roster_receipt_sha256": (
                    self.prediction_roster_receipt_sha256
                ),
                "provider_prediction_receipt_sha256": (
                    self.provider_prediction_receipt_sha256
                ),
                "decoder_policy_receipt_sha256": (self.decoder_policy_receipt_sha256),
                "source_candidate_receipt_sha256": candidate_receipt,
            }
            for (
                event_id,
                patient_uid,
                recording_id,
                candidate_id,
                candidate_origin,
                candidate_receipt,
            ) in zip(
                self.event_ids,
                self.patient_uids,
                self.recording_ids,
                self.candidate_ids,
                self.candidate_origins,
                self.source_candidate_receipt_sha256s,
            )
        ]


def build_ba_ieg_g0_stable_origin_registry_v1(
    batch: BAIEGCollatedEventBatch,
    *,
    prediction_roster: Mapping[str, Any],
    candidate_ids_by_event: Sequence[str],
) -> BAIEGG0StableOriginRegistryV1:
    """Resolve local origins only from a frozen patient-OOF prediction roster."""

    if not isinstance(batch, BAIEGCollatedEventBatch):
        raise TypeError("stable origins require a registered collated BA-IEG batch")
    roster = validate_ba_ieg_g0_a1_prediction_roster_v1(dict(prediction_roster))
    if batch.model_split != "source_train" or roster["model_split"] != "source_train":
        raise ValueError("G0 stable origins require source-train patient-OOF inference")
    if len(candidate_ids_by_event) != len(batch.event_ids):
        raise ValueError("candidate-origin bindings must align with every event")
    candidate_ids = tuple(
        _identifier(value, "candidate-origin binding")
        for value in candidate_ids_by_event
    )
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError(
            "one frozen candidate cannot seed multiple events; create "
            "content-addressed reference-free child candidates first"
        )
    candidates = {row["candidate_id"]: row for row in roster["candidates"]}
    selected: list[dict[str, Any]] = []
    origins: list[float] = []
    active = batch.token_row_mask & batch.token_signal_mask
    for index, candidate_id in enumerate(candidate_ids):
        candidate = candidates.get(candidate_id)
        if candidate is None:
            raise ValueError("stable origin candidate is absent from prediction freeze")
        if (
            candidate["patient_uid"] != batch.patient_uids[index]
            or candidate["recording_id"] != batch.recording_ids[index]
        ):
            raise ValueError("stable origin crosses patient/recording identity")
        anchor = float(candidate["anchor_offset_seconds"])
        local_bounds = batch.token_time_bounds_seconds[index, active[index]]
        if not bool(local_bounds.numel()) or not (
            float(local_bounds[:, 0].min()) - 1e-8
            <= anchor
            <= float(local_bounds[:, 1].max()) + 1e-8
        ):
            raise ValueError(
                "frozen candidate anchor lies outside observed event support"
            )
        selected.append(candidate)
        origins.append(anchor)
    return BAIEGG0StableOriginRegistryV1(
        source_input_batch_sha256=batch.input_batch_sha256,
        event_ids=batch.event_ids,
        recording_ids=batch.recording_ids,
        patient_uids=batch.patient_uids,
        candidate_ids=candidate_ids,
        candidate_origins=tuple(str(row["origin"]) for row in selected),
        source_candidate_receipt_sha256s=tuple(
            str(row["source_candidate_receipt_sha256"]) for row in selected
        ),
        prediction_roster_id=str(roster["roster_id"]),
        prediction_roster_receipt_sha256=str(roster["receipt_sha256"]),
        provider_id=str(roster["provider_id"]),
        provider_prediction_receipt_sha256=str(
            roster["provider_prediction_receipt_sha256"]
        ),
        decoder_policy_receipt_sha256=str(roster["decoder_policy_receipt_sha256"]),
        stable_origin_recording_seconds_output_only=torch.tensor(
            origins,
            dtype=torch.float64,
            device=batch.token_time_bounds_seconds.device,
        ),
    )


@dataclass(frozen=True)
class BAIEGG0SupportRelativeTimeSurfaceV1:
    """Registered absolute-output/relative-learned dual clock."""

    source_input_batch_sha256: str
    source_context_receipt_sha256: str
    source_stable_origin_registry_receipt_sha256: str
    event_ids: tuple[str, ...]
    stable_origin_recording_seconds_output_only: torch.Tensor
    absolute_token_bounds_recording_seconds_output_only: torch.Tensor
    support_relative_token_bounds_seconds: torch.Tensor
    learned_time_features: torch.Tensor
    token_active_mask: torch.Tensor
    support_relative_observed_intervals_seconds: torch.Tensor
    observed_support_mask: torch.Tensor
    support_relative_quality_gap_intervals_seconds: torch.Tensor
    quality_gap_mask: torch.Tensor
    left_censor_reason_codes: tuple[str, ...]
    right_censor_reason_codes: tuple[str, ...]
    feature_names: tuple[str, ...] = BA_IEG_G0_SUPPORT_RELATIVE_TIME_FEATURE_NAMES_V1
    schema_version: str = BA_IEG_G0_SUPPORT_RELATIVE_TIME_SURFACE_SCHEMA_V1
    learned_surface_sha256: str = field(init=False)
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _sha256(self.source_input_batch_sha256, "source batch receipt")
        _sha256(self.source_context_receipt_sha256, "source context receipt")
        _sha256(
            self.source_stable_origin_registry_receipt_sha256,
            "source stable-origin registry receipt",
        )
        if self.schema_version != BA_IEG_G0_SUPPORT_RELATIVE_TIME_SURFACE_SCHEMA_V1:
            raise ValueError("support-relative time-surface schema drifted")
        if self.feature_names != BA_IEG_G0_SUPPORT_RELATIVE_TIME_FEATURE_NAMES_V1:
            raise ValueError("support-relative time feature roster drifted")
        if len(self.event_ids) < 1 or len(set(self.event_ids)) != len(self.event_ids):
            raise ValueError("time surface requires unique event IDs")
        tensors = {
            name: _frozen_copy(getattr(self, name))
            for name in (
                "stable_origin_recording_seconds_output_only",
                "absolute_token_bounds_recording_seconds_output_only",
                "support_relative_token_bounds_seconds",
                "learned_time_features",
                "token_active_mask",
                "support_relative_observed_intervals_seconds",
                "observed_support_mask",
                "support_relative_quality_gap_intervals_seconds",
                "quality_gap_mask",
            )
        }
        batch_size = len(self.event_ids)
        token_shape = tuple(tensors["support_relative_token_bounds_seconds"].shape)
        if (
            tuple(tensors["stable_origin_recording_seconds_output_only"].shape)
            != (batch_size,)
            or len(token_shape) != 3
            or token_shape[0] != batch_size
            or token_shape[-1] != 2
            or tuple(
                tensors["absolute_token_bounds_recording_seconds_output_only"].shape
            )
            != token_shape
            or tuple(tensors["token_active_mask"].shape) != token_shape[:2]
            or tuple(tensors["learned_time_features"].shape)
            != (*token_shape[:2], len(self.feature_names))
        ):
            raise ValueError("support-relative token/time tensor shapes do not align")
        if tensors["token_active_mask"].dtype != torch.bool:
            raise TypeError("time-surface active mask must be boolean")
        if (
            tuple(tensors["support_relative_observed_intervals_seconds"].shape[:2])
            != tuple(tensors["observed_support_mask"].shape)
            or tensors["support_relative_observed_intervals_seconds"].shape[-1] != 2
            or tuple(
                tensors["support_relative_quality_gap_intervals_seconds"].shape[:2]
            )
            != tuple(tensors["quality_gap_mask"].shape)
            or tensors["support_relative_quality_gap_intervals_seconds"].shape[-1] != 2
            or tensors["observed_support_mask"].dtype != torch.bool
            or tensors["quality_gap_mask"].dtype != torch.bool
        ):
            raise ValueError("support-relative context tensors do not align")
        if (
            len(self.left_censor_reason_codes) != batch_size
            or len(self.right_censor_reason_codes) != batch_size
        ):
            raise ValueError("time-surface censor metadata does not align")
        active = tensors["token_active_mask"]
        relative = tensors["support_relative_token_bounds_seconds"]
        features = tensors["learned_time_features"]
        if (
            not torch.isfinite(relative[active]).all()
            or not torch.isfinite(features[active]).all()
        ):
            raise ValueError("active support-relative time features must be finite")
        active_relative = relative[active]
        if torch.any(active_relative[:, 1] <= active_relative[:, 0]):
            raise ValueError("active support-relative token duration is invalid")
        if torch.any(relative[~active] != 0) or torch.any(features[~active] != 0):
            raise ValueError("inactive time-surface rows must be exactly zero")
        for name, tensor in tensors.items():
            object.__setattr__(self, name, tensor)
        object.__setattr__(self, "learned_surface_sha256", self._learned_hash())
        object.__setattr__(self, "receipt_sha256", self._receipt_hash())

    def _learned_hash(self) -> str:
        # Deliberately excludes event identity, recording-absolute origin/bounds,
        # and source receipts so a pure global time translation replays exactly.
        return _canonical_sha256(
            {
                "schema": "ba_ieg_g0_translation_invariant_learned_time_surface_v1",
                "feature_names": list(self.feature_names),
                "left_censor_reason_codes": list(self.left_censor_reason_codes),
                "right_censor_reason_codes": list(self.right_censor_reason_codes),
                "tensor_sha256": {
                    name: _tensor_sha256(getattr(self, name))
                    for name in _TIME_NONSIGNAL_TENSORS
                },
            }
        )

    def _receipt_hash(self) -> str:
        return _canonical_sha256(
            {
                "schema_version": self.schema_version,
                "source_input_batch_sha256": self.source_input_batch_sha256,
                "source_context_receipt_sha256": self.source_context_receipt_sha256,
                "source_stable_origin_registry_receipt_sha256": (
                    self.source_stable_origin_registry_receipt_sha256
                ),
                "event_ids": list(self.event_ids),
                "feature_names": list(self.feature_names),
                "absolute_clock_authority": "identity_receipt_and_output_only_not_learned",
                "learned_surface_sha256": self._learned_hash(),
                "tensor_sha256": {
                    "stable_origin_recording_seconds_output_only": _tensor_sha256(
                        self.stable_origin_recording_seconds_output_only
                    ),
                    "absolute_token_bounds_recording_seconds_output_only": _tensor_sha256(
                        self.absolute_token_bounds_recording_seconds_output_only
                    ),
                },
            }
        )

    def verify_integrity(self) -> None:
        if self.learned_surface_sha256 != self._learned_hash():
            raise ValueError("support-relative learned time surface changed")
        if self.receipt_sha256 != self._receipt_hash():
            raise ValueError("support-relative dual-clock receipt changed")


def build_ba_ieg_g0_support_relative_time_surface_v1(
    batch: BAIEGCollatedEventBatch,
    context: BAIEGSegmentalBoundaryContext,
    stable_origin_registry: BAIEGG0StableOriginRegistryV1,
) -> BAIEGG0SupportRelativeTimeSurfaceV1:
    """Project absolute coordinates onto a prediction-frozen local clock."""

    if not isinstance(batch, BAIEGCollatedEventBatch):
        raise TypeError(
            "support-relative time surface requires a collated BA-IEG batch"
        )
    if not isinstance(context, BAIEGSegmentalBoundaryContext):
        raise TypeError("support-relative time surface requires a segmental context")
    if not isinstance(stable_origin_registry, BAIEGG0StableOriginRegistryV1):
        raise TypeError("support-relative time surface requires stable origins")
    context.verify_integrity()
    stable_origin_registry.verify_integrity()
    if (
        context.source_input_batch_sha256 != batch.input_batch_sha256
        or context.event_ids != batch.event_ids
        or context.source_event_receipt_sha256s != batch.input_event_receipt_sha256s
    ):
        raise ValueError("support-relative time surface crosses batch/context identity")
    if (
        stable_origin_registry.source_input_batch_sha256 != batch.input_batch_sha256
        or stable_origin_registry.event_ids != batch.event_ids
        or stable_origin_registry.recording_ids != batch.recording_ids
        or stable_origin_registry.patient_uids != batch.patient_uids
    ):
        raise ValueError("support-relative time surface crosses stable-origin identity")
    active = batch.token_row_mask & batch.token_signal_mask
    batch_size, token_count = active.shape
    device = batch.token_time_bounds_seconds.device
    # float64 keeps physical subtraction stable while the source batch remains
    # the immutable recording-relative output clock.
    absolute = batch.token_time_bounds_seconds.detach().to(torch.float64)
    relative = torch.zeros(
        (batch_size, token_count, 2), dtype=torch.float64, device=device
    )
    learned = torch.zeros(
        (
            batch_size,
            token_count,
            len(BA_IEG_G0_SUPPORT_RELATIVE_TIME_FEATURE_NAMES_V1),
        ),
        dtype=torch.float64,
        device=device,
    )
    support_relative = torch.zeros_like(
        context.observed_support_intervals_seconds, dtype=torch.float64, device=device
    )
    gap_relative = torch.zeros_like(
        context.quality_gap_intervals_seconds, dtype=torch.float64, device=device
    )
    origins = torch.zeros(batch_size, dtype=torch.float64, device=device)
    for batch_index in range(batch_size):
        support_tensor = (
            context.observed_support_intervals_seconds[
                batch_index, context.observed_support_mask[batch_index]
            ]
            .detach()
            .to(torch.float64)
        )
        gap_tensor = (
            context.quality_gap_intervals_seconds[
                batch_index, context.quality_gap_mask[batch_index]
            ]
            .detach()
            .to(torch.float64)
        )
        support_start = float(support_tensor[:, 0].min())
        support_stop = float(support_tensor[:, 1].max())
        origin = float(
            stable_origin_registry.stable_origin_recording_seconds_output_only[
                batch_index
            ]
        )
        if origin < support_start - 1e-8 or origin > support_stop + 1e-8:
            raise ValueError("stable origin lies outside the current support envelope")
        origins[batch_index] = origin
        support_relative[batch_index, context.observed_support_mask[batch_index]] = (
            support_tensor - origin
        )
        if bool(context.quality_gap_mask[batch_index].any()):
            gap_relative[batch_index, context.quality_gap_mask[batch_index]] = (
                gap_tensor - origin
            )
        support_rows = [(float(row[0]), float(row[1])) for row in support_tensor.cpu()]
        gap_rows = [(float(row[0]), float(row[1])) for row in gap_tensor.cpu()]
        clean = _clean_intervals(support_rows, gap_rows)
        for token_index in (
            torch.nonzero(active[batch_index], as_tuple=False).flatten().tolist()
        ):
            start = float(absolute[batch_index, token_index, 0])
            stop = float(absolute[batch_index, token_index, 1])
            midpoint = 0.5 * (start + stop)
            duration = stop - start
            if duration <= 0:
                raise ValueError("active token has non-positive physical duration")
            relative_start = start - origin
            relative_stop = stop - origin
            relative_midpoint = midpoint - origin
            relative[batch_index, token_index] = torch.tensor(
                (relative_start, relative_stop), dtype=torch.float64, device=device
            )
            overlap = _gap_overlap(start, stop, gap_rows)
            learned[batch_index, token_index] = torch.tensor(
                (
                    math.asinh(relative_start / 60.0),
                    math.asinh(relative_stop / 60.0),
                    math.asinh(relative_midpoint / 60.0),
                    math.log1p(duration),
                    math.asinh(
                        _signed_opportunity_displacement(start, origin, clean) / 60.0
                    ),
                    math.asinh(
                        _signed_opportunity_displacement(stop, origin, clean) / 60.0
                    ),
                    math.asinh(
                        _signed_opportunity_displacement(midpoint, origin, clean) / 60.0
                    ),
                    min(1.0, overlap / duration),
                    float(overlap > 1e-12),
                    float(abs(start - support_start) <= 1e-9),
                    float(abs(stop - support_stop) <= 1e-9),
                ),
                dtype=torch.float64,
                device=device,
            )
    absolute_output = torch.where(
        active.unsqueeze(-1), absolute, torch.zeros_like(absolute)
    )
    return BAIEGG0SupportRelativeTimeSurfaceV1(
        source_input_batch_sha256=batch.input_batch_sha256,
        source_context_receipt_sha256=context.receipt_sha256,
        source_stable_origin_registry_receipt_sha256=(
            stable_origin_registry.receipt_sha256
        ),
        event_ids=batch.event_ids,
        stable_origin_recording_seconds_output_only=origins,
        absolute_token_bounds_recording_seconds_output_only=absolute_output,
        support_relative_token_bounds_seconds=relative,
        learned_time_features=learned,
        token_active_mask=active,
        support_relative_observed_intervals_seconds=support_relative,
        observed_support_mask=context.observed_support_mask,
        support_relative_quality_gap_intervals_seconds=gap_relative,
        quality_gap_mask=context.quality_gap_mask,
        left_censor_reason_codes=context.left_censor_reason_codes,
        right_censor_reason_codes=context.right_censor_reason_codes,
    )


def ba_ieg_g0_position_only_nonsignal_inputs_v1(
    batch: BAIEGCollatedEventBatch,
    context: BAIEGSegmentalBoundaryContext,
    time_surface: BAIEGG0SupportRelativeTimeSurfaceV1,
) -> dict[str, torch.Tensor]:
    """Return the exact v1.3 non-signal surface for a dedicated baseline.

    No signal placeholder is included.  A position-only model must consume
    this dictionary directly rather than a zero-valued EEG tensor that could
    silently acquire different masks or metadata.
    """

    if not isinstance(time_surface, BAIEGG0SupportRelativeTimeSurfaceV1):
        raise TypeError("position-only surface requires a registered time surface")
    context.verify_integrity()
    time_surface.verify_integrity()
    if (
        time_surface.source_input_batch_sha256 != batch.input_batch_sha256
        or time_surface.source_context_receipt_sha256 != context.receipt_sha256
    ):
        raise ValueError("position-only surface crosses batch/context identity")
    result = {
        name: _frozen_copy(getattr(batch, name)) for name in _BATCH_NONSIGNAL_TENSORS
    }
    result.update(
        {
            name: _frozen_copy(getattr(time_surface, name))
            for name in _TIME_NONSIGNAL_TENSORS
        }
    )
    result["left_censoring_possible"] = _frozen_copy(context.left_censoring_possible)
    result["right_censoring_possible"] = _frozen_copy(context.right_censoring_possible)
    if set(result).intersection(_FORBIDDEN_POSITION_ONLY_INPUT_NAMES):
        raise RuntimeError(
            "position-only surface acquired a forbidden signal/target field"
        )
    return result


def _normalize_provider_lineage(
    value: object, *, event_id: str, index: int
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _PROVIDER_LINEAGE_FIELDS:
        raise ValueError(f"provider lineage row {index} fields drifted")
    row = deepcopy(value)
    if row["event_id"] != event_id:
        raise ValueError("provider lineage does not align with event order")
    if row["candidate_origin"] not in BA_IEG_G0_A1_CANDIDATE_ORIGINS:
        raise ValueError("provider lineage candidate origin is unsupported")
    return {
        "event_id": _identifier(row["event_id"], "provider event ID"),
        "patient_uid": _identifier(row["patient_uid"], "provider patient UID"),
        "recording_id": _identifier(row["recording_id"], "provider recording ID"),
        "candidate_id": _identifier(row["candidate_id"], "provider candidate ID"),
        "candidate_origin": _identifier(
            row["candidate_origin"], "provider candidate origin"
        ),
        "provider_id": _identifier(row["provider_id"], "provider ID"),
        "prediction_roster_id": _identifier(
            row["prediction_roster_id"], "prediction roster ID"
        ),
        "prediction_roster_receipt_sha256": _sha256(
            row["prediction_roster_receipt_sha256"],
            "prediction roster receipt",
        ),
        "provider_prediction_receipt_sha256": _sha256(
            row["provider_prediction_receipt_sha256"], "provider prediction receipt"
        ),
        "decoder_policy_receipt_sha256": _sha256(
            row["decoder_policy_receipt_sha256"], "decoder policy receipt"
        ),
        "source_candidate_receipt_sha256": _sha256(
            row["source_candidate_receipt_sha256"], "source candidate receipt"
        ),
    }


def _seal_shortcut_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(value))
    result["receipt_id"] = "BAIEG-G0-SHORTCUT-PENDING"
    result["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    id_source = deepcopy(result)
    result["receipt_id"] = "BAIEGG0SHORT-" + _canonical_sha256(id_source)[:24]
    hash_source = deepcopy(result)
    hash_source["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    result["receipt_sha256"] = _canonical_sha256(hash_source)
    return result


def build_ba_ieg_g0_shortcut_surface_contract_v1(
    *,
    batch: BAIEGCollatedEventBatch,
    context: BAIEGSegmentalBoundaryContext,
    stable_origin_registry: BAIEGG0StableOriginRegistryV1,
    time_surface: BAIEGG0SupportRelativeTimeSurfaceV1,
) -> dict[str, Any]:
    """Bind main, position-only, and signal-shuffle non-signal parity."""

    inputs = ba_ieg_g0_position_only_nonsignal_inputs_v1(batch, context, time_surface)
    stable_origin_registry.verify_integrity()
    if (
        stable_origin_registry.source_input_batch_sha256 != batch.input_batch_sha256
        or stable_origin_registry.receipt_sha256
        != time_surface.source_stable_origin_registry_receipt_sha256
    ):
        raise ValueError("shortcut surface crosses stable-origin identity")
    provider_lineage_by_event = stable_origin_registry.provider_lineage()
    provider_lineage = [
        _normalize_provider_lineage(row, event_id=event_id, index=index)
        for index, (row, event_id) in enumerate(
            zip(provider_lineage_by_event, batch.event_ids)
        )
    ]
    tensor_hashes = {
        name: _tensor_sha256(value) for name, value in sorted(inputs.items())
    }
    non_signal_hash = _canonical_sha256(
        {
            "schema": "ba_ieg_g0_v1_3_min_complete_nonsignal_input_surface_v1",
            "tensor_sha256": tensor_hashes,
            "learned_provider_feature_names": [],
        }
    )
    provider_lineage_contract_hash = _canonical_sha256(
        {
            "schema": "ba_ieg_g0_v1_3_min_provider_lineage_only_v1",
            "stable_origin_registry_receipt_sha256": (
                stable_origin_registry.receipt_sha256
            ),
            "provider_lineage": provider_lineage,
            "numeric_provider_feature_names": [],
        }
    )
    body = {
        "schema_version": BA_IEG_G0_SHORTCUT_SURFACE_SCHEMA_V1,
        "receipt_id": "BAIEG-G0-SHORTCUT-PENDING",
        "source_input_batch_sha256": batch.input_batch_sha256,
        "source_context_receipt_sha256": context.receipt_sha256,
        "source_stable_origin_registry_receipt_sha256": (
            stable_origin_registry.receipt_sha256
        ),
        "source_time_surface_receipt_sha256": time_surface.receipt_sha256,
        "translation_invariant_learned_time_surface_sha256": (
            time_surface.learned_surface_sha256
        ),
        "event_ids": list(batch.event_ids),
        "provider_lineage": provider_lineage,
        "provider_lineage_contract_sha256": provider_lineage_contract_hash,
        "learned_provider_feature_names": [],
        "included_nonsignal_tensor_sha256": tensor_hashes,
        "forbidden_position_only_input_names": list(
            _FORBIDDEN_POSITION_ONLY_INPUT_NAMES
        ),
        "main_model_nonsignal_surface_sha256": non_signal_hash,
        "position_only_baseline": {
            "nonsignal_surface_sha256": non_signal_hash,
            "uses_exact_main_model_nonsignal_surface": True,
            "signal_value_input": "absent_dedicated_nonsignal_model",
            "original_feature_opportunity_masks_preserved": True,
        },
        "eeg_shuffled_position_preserved_baseline": {
            "nonsignal_surface_sha256": non_signal_hash,
            "uses_exact_main_model_nonsignal_surface": True,
            "donor_signal_must_be_different_patient_same_split_and_tensor_signature": True,
            "donor_roster_materialized": False,
        },
        "scope_receipt": {
            "old_v1_model_or_receipt_mutated": False,
            "absolute_recording_clock_is_learned_input": False,
            "absolute_recording_clock_retained_for_identity_receipt_and_output": True,
            "support_relative_clock_is_learned_input": True,
            "support_relative_origin_is_prediction_frozen_patient_oof_candidate_anchor": True,
            "mutable_support_left_edge_used_as_time_origin": False,
            "stable_origin_absolute_value_exposed_as_numeric_model_input": False,
            "position_only_receives_same_stable_origin_relative_surface": True,
            "one_frozen_candidate_seed_at_most_one_model_event": True,
            "parent_candidate_copy_as_multi_event_split_allowed": False,
            "reference_free_child_candidate_split_registry_materialized": False,
            "source_dev_eval_private_reference_free_stable_origin_registry_api_materialized": False,
            "training_requires_same_clock_registry_for_frozen_inference_candidates": True,
            "position_only_reuses_every_main_nonsignal_input": True,
            "signal_shuffle_reuses_every_main_nonsignal_input": True,
            "detector_score_or_provider_identity_is_numeric_learned_feature": False,
            "reference_event_or_channel_soz_target_used": False,
            "training_authorized": False,
            "g0_promotion_authorized": False,
        },
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    result = _seal_shortcut_payload(body)
    validate_ba_ieg_g0_shortcut_surface_contract_v1(result)
    return result


def validate_ba_ieg_g0_shortcut_surface_contract_v1(payload: object) -> dict[str, Any]:
    fields = {
        "schema_version",
        "receipt_id",
        "source_input_batch_sha256",
        "source_context_receipt_sha256",
        "source_stable_origin_registry_receipt_sha256",
        "source_time_surface_receipt_sha256",
        "translation_invariant_learned_time_surface_sha256",
        "event_ids",
        "provider_lineage",
        "provider_lineage_contract_sha256",
        "learned_provider_feature_names",
        "included_nonsignal_tensor_sha256",
        "forbidden_position_only_input_names",
        "main_model_nonsignal_surface_sha256",
        "position_only_baseline",
        "eeg_shuffled_position_preserved_baseline",
        "scope_receipt",
        "receipt_sha256",
    }
    if type(payload) is not dict or set(payload) != fields:
        raise ValueError("G0 shortcut-surface contract fields drifted")
    data = deepcopy(payload)
    if data["schema_version"] != BA_IEG_G0_SHORTCUT_SURFACE_SCHEMA_V1:
        raise ValueError("G0 shortcut-surface schema drifted")
    for name in (
        "source_input_batch_sha256",
        "source_context_receipt_sha256",
        "source_stable_origin_registry_receipt_sha256",
        "source_time_surface_receipt_sha256",
        "translation_invariant_learned_time_surface_sha256",
        "main_model_nonsignal_surface_sha256",
        "provider_lineage_contract_sha256",
    ):
        _sha256(data[name], name)
    if data["learned_provider_feature_names"] != []:
        raise ValueError("provider metadata became a learned shortcut feature")
    if data["forbidden_position_only_input_names"] != list(
        _FORBIDDEN_POSITION_ONLY_INPUT_NAMES
    ):
        raise ValueError("position-only forbidden field roster drifted")
    tensor_hashes = data["included_nonsignal_tensor_sha256"]
    if type(tensor_hashes) is not dict or set(tensor_hashes) != set(
        _BATCH_NONSIGNAL_TENSORS + _TIME_NONSIGNAL_TENSORS
    ).union({"left_censoring_possible", "right_censoring_possible"}):
        raise ValueError("shortcut non-signal tensor roster is incomplete")
    for name, digest in tensor_hashes.items():
        _sha256(digest, f"non-signal tensor {name}")
    main_hash = data["main_model_nonsignal_surface_sha256"]
    expected_main_hash = _canonical_sha256(
        {
            "schema": "ba_ieg_g0_v1_3_min_complete_nonsignal_input_surface_v1",
            "tensor_sha256": tensor_hashes,
            "learned_provider_feature_names": [],
        }
    )
    if main_hash != expected_main_hash:
        raise ValueError("main non-signal tensor surface does not replay")
    if data["position_only_baseline"] != {
        "nonsignal_surface_sha256": main_hash,
        "uses_exact_main_model_nonsignal_surface": True,
        "signal_value_input": "absent_dedicated_nonsignal_model",
        "original_feature_opportunity_masks_preserved": True,
    }:
        raise ValueError(
            "position-only baseline does not equal the main non-signal surface"
        )
    if data["eeg_shuffled_position_preserved_baseline"] != {
        "nonsignal_surface_sha256": main_hash,
        "uses_exact_main_model_nonsignal_surface": True,
        "donor_signal_must_be_different_patient_same_split_and_tensor_signature": True,
        "donor_roster_materialized": False,
    }:
        raise ValueError("signal-shuffle non-signal surface drifted")
    expected_scope = {
        "old_v1_model_or_receipt_mutated": False,
        "absolute_recording_clock_is_learned_input": False,
        "absolute_recording_clock_retained_for_identity_receipt_and_output": True,
        "support_relative_clock_is_learned_input": True,
        "support_relative_origin_is_prediction_frozen_patient_oof_candidate_anchor": True,
        "mutable_support_left_edge_used_as_time_origin": False,
        "stable_origin_absolute_value_exposed_as_numeric_model_input": False,
        "position_only_receives_same_stable_origin_relative_surface": True,
        "one_frozen_candidate_seed_at_most_one_model_event": True,
        "parent_candidate_copy_as_multi_event_split_allowed": False,
        "reference_free_child_candidate_split_registry_materialized": False,
        "source_dev_eval_private_reference_free_stable_origin_registry_api_materialized": False,
        "training_requires_same_clock_registry_for_frozen_inference_candidates": True,
        "position_only_reuses_every_main_nonsignal_input": True,
        "signal_shuffle_reuses_every_main_nonsignal_input": True,
        "detector_score_or_provider_identity_is_numeric_learned_feature": False,
        "reference_event_or_channel_soz_target_used": False,
        "training_authorized": False,
        "g0_promotion_authorized": False,
    }
    if data["scope_receipt"] != expected_scope:
        raise ValueError("G0 shortcut-surface authority/firewall drifted")
    if not isinstance(data["event_ids"], list) or not data["event_ids"]:
        raise ValueError("G0 shortcut surface has no event roster")
    if len(data["provider_lineage"]) != len(data["event_ids"]):
        raise ValueError("G0 shortcut provider lineage does not align")
    normalized_lineage = [
        _normalize_provider_lineage(row, event_id=event_id, index=index)
        for index, (row, event_id) in enumerate(
            zip(data["provider_lineage"], data["event_ids"])
        )
    ]
    if len({row["candidate_id"] for row in normalized_lineage}) != len(
        normalized_lineage
    ):
        raise ValueError("shortcut lineage repeats one frozen parent candidate")
    shared_provider = {
        (
            row["provider_id"],
            row["prediction_roster_id"],
            row["prediction_roster_receipt_sha256"],
            row["provider_prediction_receipt_sha256"],
            row["decoder_policy_receipt_sha256"],
        )
        for row in normalized_lineage
    }
    if len(shared_provider) != 1:
        raise ValueError("shortcut lineage mixes provider prediction freezes")
    expected_lineage_hash = _canonical_sha256(
        {
            "schema": "ba_ieg_g0_v1_3_min_provider_lineage_only_v1",
            "stable_origin_registry_receipt_sha256": data[
                "source_stable_origin_registry_receipt_sha256"
            ],
            "provider_lineage": data["provider_lineage"],
            "numeric_provider_feature_names": [],
        }
    )
    if data["provider_lineage_contract_sha256"] != expected_lineage_hash:
        raise ValueError("provider lineage-only contract does not replay")
    expected = _seal_shortcut_payload(data)
    if (
        data["receipt_id"] != expected["receipt_id"]
        or data["receipt_sha256"] != expected["receipt_sha256"]
    ):
        raise ValueError("G0 shortcut-surface content address does not replay")
    return data


__all__ = [
    "BA_IEG_G0_STABLE_ORIGIN_REGISTRY_SCHEMA_V1",
    "BA_IEG_G0_SUPPORT_RELATIVE_TIME_SURFACE_SCHEMA_V1",
    "BA_IEG_G0_SHORTCUT_SURFACE_SCHEMA_V1",
    "BA_IEG_G0_SUPPORT_RELATIVE_TIME_FEATURE_NAMES_V1",
    "BAIEGG0StableOriginRegistryV1",
    "BAIEGG0SupportRelativeTimeSurfaceV1",
    "build_ba_ieg_g0_stable_origin_registry_v1",
    "build_ba_ieg_g0_support_relative_time_surface_v1",
    "ba_ieg_g0_position_only_nonsignal_inputs_v1",
    "build_ba_ieg_g0_shortcut_surface_contract_v1",
    "validate_ba_ieg_g0_shortcut_surface_contract_v1",
]
