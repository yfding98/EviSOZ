"""Permission-locked BA-IEG reference-field candidate materialization.

This module is the narrow bridge between the physical-time BA-IEG onset
encoder and :class:`BAIEGReferenceSpecificFieldCandidate`.  It accepts no EDF
path, annotation, spreadsheet, doctor label, report text or detector tensor.
The caller must supply the immutable event contracts used to build a collated
batch and an encoder output bound to that exact batch.

Only onset-causal, future-free, explicitly onset-authorized analysis units
whose usable signal support is wholly spatial-field eligible are materialized.
Offline views, missing/imputed physical support, QC/family-masked support and
unsupported reference semantics are retained only as typed exclusion
decisions.  A bipolar derivation remains one signed lead and is never split
into endpoint-electrode candidates.

The exported score is an ordinal within-view rank.  It is not a probability,
is not comparable in magnitude across views or reference families, and does
not constitute a clinical SOZ, cortical source, epileptogenic-zone or surgical
target claim.  The result is a research primitive and is not connected to a
Findings or production-report route.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Final, Sequence

import torch

from src.soz.geometry import CHANNEL_INDEX, STANDARD_19

from .ba_ieg_multireference_field import (
    BA_IEG_MULTIREFERENCE_REFERENCE_FAMILIES,
    BAIEGReferenceSpecificFieldCandidate,
)
from .ba_ieg_physical_time_encoder import (
    BA_IEG_PHYSICAL_TIME_ENCODER_ID,
    BAIEGPhysicalTimeEncoderOutput,
)
from .ba_ieg_training_contract import (
    BA_IEG_EVIDENCE_FAMILIES,
    BAIEGCollatedEventBatch,
    BAIEGEventTokens,
    collate_ba_ieg_events,
)


BA_IEG_REFERENCE_FIELD_CANDIDATE_MATERIALIZATION_SCHEMA_VERSION: Final[str] = (
    "ba_ieg_reference_field_candidate_materialization_v1"
)
BA_IEG_REFERENCE_FIELD_CANDIDATE_MATERIALIZER_ID: Final[str] = (
    "ba_ieg_permission_locked_reference_field_candidate_materializer_v1"
)
BA_IEG_REFERENCE_FIELD_SCORE_SEMANTICS: Final[str] = (
    "within_view_ordinal_rank_larger_is_higher_not_probability_v1"
)

_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_TOLERANCE_SECONDS = 1e-6


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(value: object, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in _SHA256_CHARACTERS for character in text):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return text


def _identifier(value: object, name: str) -> str:
    text = str(value)
    if not text or text != text.strip():
        raise ValueError(f"{name} must be a non-empty trimmed identifier")
    return text


def _tensor_sha256(value: torch.Tensor) -> str:
    if not isinstance(value, torch.Tensor):
        raise TypeError("content-addressed field-head values must be tensors")
    tensor = value.detach().cpu().contiguous()
    byte_view = tensor.reshape(-1).view(torch.uint8)
    byte_digest = hashlib.sha256(byte_view.numpy().tobytes()).hexdigest()
    return _canonical_sha256(
        {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "byte_sha256": byte_digest,
        }
    )


def _cpu_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (
        isinstance(left, torch.Tensor)
        and isinstance(right, torch.Tensor)
        and left.dtype == right.dtype
        and tuple(left.shape) == tuple(right.shape)
        and torch.equal(left.detach().cpu(), right.detach().cpu())
    )


def _reference_row_sha256(row: torch.Tensor | Sequence[float]) -> str:
    values = [float(value) for value in row]
    return _canonical_sha256(
        {
            "physical_electrode_ids": list(STANDARD_19),
            "signed_row": values,
        }
    )


def _candidate_integrity(candidate: BAIEGReferenceSpecificFieldCandidate) -> None:
    expected_row = _reference_row_sha256(candidate.signed_reference_row)
    if candidate.reference_row_sha256 != expected_row:
        raise ValueError("reference-field candidate signed row changed after registration")
    body = candidate.to_dict()
    registered = body.pop("candidate_sha256")
    if registered != _canonical_sha256(body):
        raise ValueError("reference-field candidate content changed after registration")


def _union_duration(intervals: Sequence[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    ordered = sorted((float(start), float(stop)) for start, stop in intervals)
    merged_start, merged_stop = ordered[0]
    duration = 0.0
    for start, stop in ordered[1:]:
        if start <= merged_stop + _TOLERANCE_SECONDS:
            merged_stop = max(merged_stop, stop)
        else:
            duration += merged_stop - merged_start
            merged_start, merged_stop = start, stop
    return duration + merged_stop - merged_start


@dataclass(frozen=True)
class BAIEGReferenceFieldUnitDecision:
    """One auditable include/exclude decision for an event analysis unit."""

    event_id: str
    recording_id: str
    batch_event_index: int
    analysis_unit_index: int
    view_index: int
    view_id: str
    analysis_unit_id: str
    reference_family: str
    reference_row_sha256: str
    usable_spatial_support_fraction: float
    candidate_id: str | None
    reason_codes: tuple[str, ...]
    decision_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "event_id",
            "recording_id",
            "view_id",
            "analysis_unit_id",
            "reference_family",
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        for name in ("batch_event_index", "analysis_unit_index", "view_index"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        _sha256(self.reference_row_sha256, "reference_row_sha256")
        fraction = float(self.usable_spatial_support_fraction)
        if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise ValueError("usable_spatial_support_fraction must lie in [0,1]")
        object.__setattr__(self, "usable_spatial_support_fraction", fraction)
        reasons = tuple(sorted(set(str(item) for item in self.reason_codes)))
        if any(not item or item != item.strip() for item in reasons):
            raise ValueError("reason codes must be non-empty trimmed strings")
        candidate_id = self.candidate_id
        if candidate_id is not None:
            candidate_id = _identifier(candidate_id, "candidate_id")
        if (candidate_id is None) == (not reasons):
            raise ValueError(
                "included decisions require one candidate and no reasons; exclusions require reasons"
            )
        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(self, "decision_sha256", _canonical_sha256(self._body()))

    @property
    def included(self) -> bool:
        return self.candidate_id is not None

    def _body(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "recording_id": self.recording_id,
            "batch_event_index": self.batch_event_index,
            "analysis_unit_index": self.analysis_unit_index,
            "view_index": self.view_index,
            "view_id": self.view_id,
            "analysis_unit_id": self.analysis_unit_id,
            "reference_family": self.reference_family,
            "reference_row_sha256": self.reference_row_sha256,
            "usable_spatial_support_fraction": self.usable_spatial_support_fraction,
            "candidate_id": self.candidate_id,
            "reason_codes": list(self.reason_codes),
        }

    def to_dict(self) -> dict[str, Any]:
        result = self._body()
        result["included"] = self.included
        result["decision_sha256"] = self.decision_sha256
        return result

    def verify_integrity(self) -> None:
        if self.decision_sha256 != _canonical_sha256(self._body()):
            raise ValueError("reference-field unit decision changed after registration")


@dataclass(frozen=True)
class BAIEGReferenceFieldCandidateMaterialization:
    """Content-addressed batch result; candidates remain grouped by event IDs."""

    source_input_batch_sha256: str
    source_field_head_receipt_sha256: str
    input_event_receipt_sha256s: tuple[str, ...]
    candidates: tuple[BAIEGReferenceSpecificFieldCandidate, ...]
    unit_decisions: tuple[BAIEGReferenceFieldUnitDecision, ...]
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _sha256(self.source_input_batch_sha256, "source_input_batch_sha256")
        _sha256(
            self.source_field_head_receipt_sha256,
            "source_field_head_receipt_sha256",
        )
        if not self.input_event_receipt_sha256s:
            raise ValueError("candidate materialization requires source event receipts")
        for index, receipt in enumerate(self.input_event_receipt_sha256s):
            _sha256(receipt, f"input_event_receipt_sha256s[{index}]")
        if any(
            not isinstance(item, BAIEGReferenceSpecificFieldCandidate)
            for item in self.candidates
        ):
            raise TypeError("candidates must use BAIEGReferenceSpecificFieldCandidate")
        if not self.unit_decisions or any(
            not isinstance(item, BAIEGReferenceFieldUnitDecision)
            for item in self.unit_decisions
        ):
            raise TypeError("unit_decisions must use BAIEGReferenceFieldUnitDecision")
        self._verify_content()
        object.__setattr__(self, "receipt_sha256", _canonical_sha256(self._body()))

    @property
    def excluded_unit_count(self) -> int:
        return sum(not item.included for item in self.unit_decisions)

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)

    @staticmethod
    def scope_receipt() -> dict[str, object]:
        return {
            "input_scope": "registered_eeg_signal_event_tokens_and_permission_locked_encoder_output_only",
            "edf_annotations_consumed": False,
            "spreadsheets_or_doctor_labels_consumed": False,
            "report_text_consumed": False,
            "detector_tensor_consumed": False,
            "positive_onset_temporal_role": "onset_causal",
            "offline_or_future_positive_support_allowed": False,
            "partial_spatial_field_qc_support_allowed": False,
            "bipolar_endpoint_attribution_allowed": False,
            "score_semantics": BA_IEG_REFERENCE_FIELD_SCORE_SEMANTICS,
            "cross_reference_score_probability_fusion_allowed": False,
            "output_scope": "research_scalp_visible_reference_specific_onset_field_candidate",
            "findings_route_connected": False,
            "production_report_route_connected": False,
        }

    def _verify_content(self) -> None:
        candidate_ids: list[str] = []
        candidate_keys: list[tuple[str, str, str]] = []
        for candidate in self.candidates:
            _candidate_integrity(candidate)
            if candidate.source_input_batch_sha256 != self.source_input_batch_sha256:
                raise ValueError("candidate source batch differs from materialization")
            if (
                candidate.source_field_head_receipt_sha256
                != self.source_field_head_receipt_sha256
            ):
                raise ValueError("candidate field-head receipt differs from materialization")
            candidate_ids.append(candidate.candidate_id)
            candidate_keys.append(
                (candidate.event_id, candidate.view_id, candidate.analysis_unit_id)
            )
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("materialized candidate IDs must be unique")
        if len(candidate_keys) != len(set(candidate_keys)):
            raise ValueError("one event/view cannot duplicate an analysis unit")
        decision_keys: list[tuple[int, int]] = []
        included_ids: list[str] = []
        for decision in self.unit_decisions:
            decision.verify_integrity()
            decision_keys.append(
                (decision.batch_event_index, decision.analysis_unit_index)
            )
            if decision.candidate_id is not None:
                included_ids.append(decision.candidate_id)
        if len(decision_keys) != len(set(decision_keys)):
            raise ValueError("materialization decisions must be unique per event/unit")
        if included_ids != candidate_ids:
            raise ValueError("included decisions and candidate order/identity disagree")

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": (
                BA_IEG_REFERENCE_FIELD_CANDIDATE_MATERIALIZATION_SCHEMA_VERSION
            ),
            "implementation_id": BA_IEG_REFERENCE_FIELD_CANDIDATE_MATERIALIZER_ID,
            "source_input_batch_sha256": self.source_input_batch_sha256,
            "source_field_head_receipt_sha256": (
                self.source_field_head_receipt_sha256
            ),
            "input_event_receipt_sha256s": list(
                self.input_event_receipt_sha256s
            ),
            "score_semantics": BA_IEG_REFERENCE_FIELD_SCORE_SEMANTICS,
            "scope_receipt": self.scope_receipt(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "unit_decisions": [decision.to_dict() for decision in self.unit_decisions],
        }

    def to_dict(self) -> dict[str, Any]:
        result = self._body()
        result["candidate_count"] = self.candidate_count
        result["excluded_unit_count"] = self.excluded_unit_count
        result["receipt_sha256"] = self.receipt_sha256
        return result

    def verify_integrity(self) -> None:
        self._verify_content()
        if self.receipt_sha256 != _canonical_sha256(self._body()):
            raise ValueError("reference-field materialization changed after registration")


def _validate_registered_batch(
    events: tuple[BAIEGEventTokens, ...], batch: BAIEGCollatedEventBatch
) -> None:
    if not events or any(not isinstance(event, BAIEGEventTokens) for event in events):
        raise TypeError("events must be a non-empty BAIEGEventTokens sequence")
    if not isinstance(batch, BAIEGCollatedEventBatch):
        raise TypeError("batch must be a registered BAIEGCollatedEventBatch")
    for event in events:
        event.verify_integrity()
    expected = collate_ba_ieg_events(events)
    if batch.input_batch_sha256 != expected.input_batch_sha256:
        raise ValueError("collated batch is not bound to the supplied event contracts")
    for name in (
        "event_ids",
        "recording_ids",
        "patient_uids",
        "input_event_receipt_sha256s",
        "view_temporal_evidence_sha256s",
    ):
        if getattr(batch, name) != getattr(expected, name):
            raise ValueError(f"collated batch {name} drifted from supplied events")
    expected_inputs = expected.model_inputs()
    actual_inputs = batch.model_inputs()
    if set(actual_inputs) != set(expected_inputs):  # pragma: no cover - contract guard
        raise RuntimeError("BA-IEG model input vocabulary drifted")
    for name, expected_value in expected_inputs.items():
        if not _cpu_equal(actual_inputs[name], expected_value):
            raise ValueError(f"collated batch {name} drifted from supplied events")


def _validate_encoder_output(
    batch: BAIEGCollatedEventBatch,
    encoder_output: BAIEGPhysicalTimeEncoderOutput,
) -> None:
    if not isinstance(encoder_output, BAIEGPhysicalTimeEncoderOutput):
        raise TypeError(
            "candidate materialization requires BAIEGPhysicalTimeEncoderOutput"
        )
    if encoder_output.source_input_batch_sha256 != batch.input_batch_sha256:
        raise ValueError("physical-time encoder output is bound to another batch")
    positive_inputs = batch.positive_onset_inputs()
    active = positive_inputs["token_row_mask"] & positive_inputs["token_signal_mask"]
    if not _cpu_equal(encoder_output.token_mask, active):
        raise ValueError("encoder token mask exceeds registered positive-onset support")
    if bool(
        (batch.token_future_sample_access.detach().cpu() & active.detach().cpu()).any()
    ):
        raise RuntimeError("future-dependent support reached the reference-field head")

    batch_size, token_count = active.shape
    maximum_units = int(batch.unit_row_mask.shape[1])
    token_embeddings = encoder_output.token_embeddings
    if token_embeddings.ndim != 3 or tuple(token_embeddings.shape[:2]) != (
        batch_size,
        token_count,
    ):
        raise ValueError("encoder token embeddings do not align with the batch")
    hidden_dim = int(token_embeddings.shape[-1])
    expected_shapes = {
        "token_onset_logits": (batch_size, token_count),
        "event_embedding": (batch_size, hidden_dim),
        "event_onset_logit": (batch_size,),
        "event_evaluable_mask": (batch_size,),
        "analysis_unit_embeddings": (batch_size, maximum_units, hidden_dim),
        "analysis_unit_onset_logits": (batch_size, maximum_units),
        "analysis_unit_onset_intervals_seconds": (
            batch_size,
            maximum_units,
            2,
        ),
        "analysis_unit_onset_association_rank": (batch_size, maximum_units),
        "analysis_unit_mask": (batch_size, maximum_units),
    }
    for name, shape in expected_shapes.items():
        if tuple(getattr(encoder_output, name).shape) != shape:
            raise ValueError(f"encoder {name} does not align with the batch")
    if encoder_output.token_mask.dtype != torch.bool:
        raise TypeError("encoder token_mask must be boolean")
    if encoder_output.analysis_unit_mask.dtype != torch.bool:
        raise TypeError("encoder analysis_unit_mask must be boolean")
    if encoder_output.event_evaluable_mask.dtype != torch.bool:
        raise TypeError("encoder event_evaluable_mask must be boolean")
    if encoder_output.analysis_unit_onset_association_rank.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise TypeError("encoder analysis-unit ranks must be integer")

    finite_names = (
        "token_embeddings",
        "token_onset_logits",
        "event_embedding",
        "event_onset_logit",
        "analysis_unit_embeddings",
        "analysis_unit_onset_logits",
        "analysis_unit_onset_intervals_seconds",
    )
    if any(not torch.isfinite(getattr(encoder_output, name)).all() for name in finite_names):
        raise ValueError("encoder field-head output contains non-finite values")

    active_cpu = active.detach().cpu()
    token_mask_cpu = encoder_output.token_mask.detach().cpu()
    if bool((encoder_output.token_embeddings.detach().cpu()[~token_mask_cpu] != 0).any()):
        raise ValueError("masked encoder token embeddings must remain zero")
    if bool((encoder_output.token_onset_logits.detach().cpu()[~token_mask_cpu] != 0).any()):
        raise ValueError("masked encoder token logits must remain zero")

    token_units = batch.token_unit_index.detach().cpu()
    unit_rows = batch.unit_row_mask.detach().cpu()
    expected_unit_mask = torch.zeros_like(unit_rows)
    for batch_index in range(batch_size):
        for unit_index in range(maximum_units):
            expected_unit_mask[batch_index, unit_index] = bool(
                (
                    active_cpu[batch_index]
                    & (token_units[batch_index] == unit_index)
                    & unit_rows[batch_index, unit_index]
                ).any()
            )
    unit_mask = encoder_output.analysis_unit_mask.detach().cpu()
    if not torch.equal(unit_mask, expected_unit_mask):
        raise ValueError("encoder analysis-unit mask exceeds positive-onset support")
    if not torch.equal(
        encoder_output.event_evaluable_mask.detach().cpu(),
        expected_unit_mask.any(dim=1),
    ):
        raise ValueError("encoder event evaluability disagrees with analysis-unit support")

    unit_logits = encoder_output.analysis_unit_onset_logits.detach().cpu()
    unit_intervals = (
        encoder_output.analysis_unit_onset_intervals_seconds.detach().cpu()
    )
    unit_ranks = encoder_output.analysis_unit_onset_association_rank.detach().cpu()
    unit_embeddings = encoder_output.analysis_unit_embeddings.detach().cpu()
    if bool((unit_logits[~unit_mask] != 0).any()) or bool(
        (unit_intervals[~unit_mask] != 0).any()
    ) or bool((unit_ranks[~unit_mask] != 0).any()) or bool(
        (unit_embeddings[~unit_mask] != 0).any()
    ):
        raise ValueError("masked analysis-unit field-head outputs must remain zero")

    bounds = batch.token_time_bounds_seconds.detach().cpu()
    for batch_index in range(batch_size):
        active_units = torch.nonzero(unit_mask[batch_index], as_tuple=False).flatten()
        ordered = active_units[
            torch.argsort(
                unit_logits[batch_index, active_units],
                descending=True,
                stable=True,
            )
        ]
        expected_ranks = torch.arange(1, len(ordered) + 1, dtype=unit_ranks.dtype)
        if not torch.equal(unit_ranks[batch_index, ordered], expected_ranks):
            raise ValueError("encoder analysis-unit ranks disagree with its logits")
        for unit_index_tensor in active_units:
            unit_index = int(unit_index_tensor)
            selected = active_cpu[batch_index] & (
                token_units[batch_index] == unit_index
            )
            interval = unit_intervals[batch_index, unit_index]
            lower = float(bounds[batch_index, selected, 0].min())
            upper = float(bounds[batch_index, selected, 1].max())
            if (
                float(interval[1]) <= float(interval[0])
                or float(interval[0]) < lower - _TOLERANCE_SECONDS
                or float(interval[1]) > upper + _TOLERANCE_SECONDS
            ):
                raise ValueError(
                    "encoder onset interval exceeds its positive physical support"
                )


def _field_head_receipt_sha256(
    events: tuple[BAIEGEventTokens, ...],
    batch: BAIEGCollatedEventBatch,
    output: BAIEGPhysicalTimeEncoderOutput,
) -> str:
    return _canonical_sha256(
        {
            "schema": "ba_ieg_permission_locked_reference_field_head_receipt_v1",
            "encoder_implementation_id": BA_IEG_PHYSICAL_TIME_ENCODER_ID,
            "materializer_id": BA_IEG_REFERENCE_FIELD_CANDIDATE_MATERIALIZER_ID,
            "source_input_batch_sha256": batch.input_batch_sha256,
            "event_input_receipt_sha256s": [
                event.input_receipt_sha256 for event in events
            ],
            "score_semantics": BA_IEG_REFERENCE_FIELD_SCORE_SEMANTICS,
            "permission_tensor_sha256": {
                name: _tensor_sha256(getattr(batch, name))
                for name in (
                    "token_positive_onset_mask",
                    "view_future_sample_access",
                    "view_onset_evidence_authorized",
                    "unit_reference_matrix",
                    "unit_evidence_mask",
                    "unit_family_mask",
                    "physical_evidence_mask",
                )
            },
            "field_head_tensor_sha256": {
                name: _tensor_sha256(getattr(output, name))
                for name in (
                    "token_mask",
                    "analysis_unit_onset_logits",
                    "analysis_unit_onset_intervals_seconds",
                    "analysis_unit_onset_association_rank",
                    "analysis_unit_mask",
                )
            },
        }
    )


def _candidate_target(
    event: BAIEGEventTokens, unit_index: int
) -> str | None:
    family = event.reference_families[int(event.unit_view_index[unit_index])]
    if family == "bipolar":
        return None
    target = event.unit_source_ids[unit_index].strip().upper()
    if target not in CHANNEL_INDEX:
        raise ValueError("non-bipolar analysis unit lacks a canonical scalp target")
    return target


def materialize_ba_ieg_reference_field_candidates(
    *,
    events: Sequence[BAIEGEventTokens],
    batch: BAIEGCollatedEventBatch,
    encoder_output: BAIEGPhysicalTimeEncoderOutput,
) -> BAIEGReferenceFieldCandidateMaterialization:
    """Materialize safe reference-specific research candidates for a batch.

    Exclusions are returned as decisions instead of being converted to zero
    scores.  Contract drift or a forged/misaligned encoder output raises; an
    event with no eligible spatial unit returns a valid zero-candidate result.
    """

    event_rows = tuple(events)
    _validate_registered_batch(event_rows, batch)
    _validate_encoder_output(batch, encoder_output)
    field_head_receipt = _field_head_receipt_sha256(
        event_rows, batch, encoder_output
    )

    spatial_family_index = BA_IEG_EVIDENCE_FAMILIES.index("spatial_field")
    positive = batch.token_positive_onset_mask.detach().cpu()
    signal = (batch.token_row_mask & batch.token_signal_mask).detach().cpu()
    token_units = batch.token_unit_index.detach().cpu()
    token_times = batch.token_time_bounds_seconds.detach().cpu()
    unit_output_mask = encoder_output.analysis_unit_mask.detach().cpu()
    unit_output_rank = (
        encoder_output.analysis_unit_onset_association_rank.detach().cpu()
    )
    unit_output_interval = (
        encoder_output.analysis_unit_onset_intervals_seconds.detach().cpu()
    )

    interim: list[dict[str, Any]] = []
    for batch_index, event in enumerate(event_rows):
        unit_count = len(event.unit_ids)
        for unit_index in range(unit_count):
            view_index = int(event.unit_view_index[unit_index])
            view_id = event.view_ids[view_index]
            family = event.reference_families[view_index]
            row = event.unit_reference_matrix[unit_index].detach().cpu()
            reason_codes: list[str] = []

            if event.view_effective_temporal_roles[view_index] != "onset_causal":
                reason_codes.append("view_not_onset_causal")
            if event.view_dependency_policies[view_index] != "past_and_present_only":
                reason_codes.append("view_dependency_not_past_and_present_only")
            if bool(event.view_future_sample_access[view_index]):
                reason_codes.append("future_sample_access_forbidden")
            if not bool(event.view_onset_evidence_authorized[view_index]):
                reason_codes.append("onset_evidence_not_authorized")
            if family not in BA_IEG_MULTIREFERENCE_REFERENCE_FAMILIES:
                reason_codes.append("reference_family_not_supported")
            if not bool(event.unit_evidence_mask[unit_index]):
                reason_codes.append("analysis_unit_not_evidence_eligible")
            if not bool(event.unit_family_mask[unit_index, spatial_family_index]):
                reason_codes.append("spatial_field_family_ineligible")

            nonzero_support = row.abs() > 1e-8
            if not bool(nonzero_support.any()):
                reason_codes.append("reference_row_empty")
            elif bool(
                (
                    nonzero_support
                    & ~event.physical_evidence_mask.detach().cpu()
                ).any()
            ):
                reason_codes.append("reference_support_unobserved_or_imputed")

            unit_signal = signal[batch_index] & (
                token_units[batch_index] == unit_index
            )
            unit_positive = positive[batch_index] & (
                token_units[batch_index] == unit_index
            )
            signal_intervals = [
                tuple(float(value) for value in token_times[batch_index, index])
                for index in torch.nonzero(unit_signal, as_tuple=False).flatten()
            ]
            positive_intervals = [
                tuple(float(value) for value in token_times[batch_index, index])
                for index in torch.nonzero(unit_positive, as_tuple=False).flatten()
            ]
            denominator = _union_duration(signal_intervals)
            numerator = _union_duration(positive_intervals)
            coverage = 0.0 if denominator <= 0.0 else min(1.0, numerator / denominator)
            if not signal_intervals:
                reason_codes.append("onset_causal_signal_support_unavailable")
            if not positive_intervals:
                reason_codes.append("positive_spatial_field_support_unavailable")
            elif coverage < 1.0 - _TOLERANCE_SECONDS:
                reason_codes.append("partial_spatial_field_qc_coverage")
            if not bool(unit_output_mask[batch_index, unit_index]):
                reason_codes.append("field_head_unit_unavailable")

            target: str | None = None
            if not reason_codes:
                try:
                    target = _candidate_target(event, unit_index)
                    # Constructing the candidate later is the final reference
                    # semantic check; this early check produces a typed
                    # exclusion instead of allowing a malformed row through.
                    if family == "bipolar" and event.unit_types[unit_index] != "lead":
                        raise ValueError("bipolar unit must remain a lead")
                    if family == "referential" and event.unit_types[unit_index] != "electrode":
                        raise ValueError("referential unit must remain an electrode")
                    if family in {"common_average", "laplacian"} and event.unit_types[unit_index] != "virtual":
                        raise ValueError("derived target-coordinate unit must remain virtual")
                except (TypeError, ValueError, KeyError, IndexError):
                    reason_codes.append("reference_semantics_invalid")

            candidate_id = (
                f"{event.event_id}::BAIEG-REFERENCE-FIELD::"
                f"V{view_index:02d}::U{unit_index:03d}"
            )
            interim.append(
                {
                    "batch_index": batch_index,
                    "event": event,
                    "unit_index": unit_index,
                    "view_index": view_index,
                    "view_id": view_id,
                    "family": family,
                    "row": row,
                    "coverage": coverage,
                    "target": target,
                    "candidate_id": candidate_id,
                    "reason_codes": tuple(sorted(set(reason_codes))),
                    "source_rank": int(unit_output_rank[batch_index, unit_index]),
                }
            )

    eligible_by_view: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for item in interim:
        if item["reason_codes"]:
            continue
        eligible_by_view.setdefault(
            (int(item["batch_index"]), int(item["view_index"])), []
        ).append(item)
    score_by_candidate: dict[str, float] = {}
    for items in eligible_by_view.values():
        ordered = sorted(
            items,
            key=lambda item: (int(item["source_rank"]), int(item["unit_index"])),
        )
        count = len(ordered)
        for position, item in enumerate(ordered, start=1):
            score_by_candidate[str(item["candidate_id"])] = float(
                count - position + 1
            )

    candidates: list[BAIEGReferenceSpecificFieldCandidate] = []
    decisions: list[BAIEGReferenceFieldUnitDecision] = []
    for item in interim:
        event = item["event"]
        assert isinstance(event, BAIEGEventTokens)  # internal construction guard
        unit_index = int(item["unit_index"])
        view_index = int(item["view_index"])
        reasons = tuple(item["reason_codes"])
        candidate: BAIEGReferenceSpecificFieldCandidate | None = None
        if not reasons:
            try:
                interval_tensor = unit_output_interval[
                    int(item["batch_index"]), unit_index
                ]
                candidate = BAIEGReferenceSpecificFieldCandidate(
                    candidate_id=str(item["candidate_id"]),
                    event_id=event.event_id,
                    recording_id=event.recording_id,
                    analysis_interval_seconds=event.analysis_interval_seconds,
                    canonical_receipt_sha256=event.canonical_receipt_sha256,
                    adaptive_window_receipt_sha256=(
                        event.adaptive_window_receipt_sha256
                    ),
                    source_input_batch_sha256=batch.input_batch_sha256,
                    source_field_head_receipt_sha256=field_head_receipt,
                    view_id=event.view_ids[view_index],
                    view_receipt_sha256=event.view_receipt_sha256s[view_index],
                    view_transform_sha256=(
                        event.view_transform_sha256s[view_index]
                    ),
                    temporal_evidence_sha256=(
                        event.view_temporal_evidence_sha256s[view_index]
                    ),
                    reference_family=event.reference_families[view_index],
                    analysis_unit_id=event.unit_source_ids[unit_index],
                    analysis_unit_type=event.unit_types[unit_index],
                    signed_reference_row=tuple(
                        float(value) for value in item["row"]
                    ),
                    physical_target_electrode_id=item["target"],
                    onset_interval_seconds=tuple(
                        float(value) for value in interval_tensor
                    ),
                    onset_association_score=score_by_candidate[
                        str(item["candidate_id"])
                    ],
                    polarity="indeterminate",
                    coverage_fraction=float(item["coverage"]),
                    observed=True,
                    imputed=False,
                    evidence_eligible=True,
                    quality_pass=True,
                    temporal_role="onset_causal",
                    intrinsic_evidence_role="onset_eligible",
                    future_sample_access=False,
                    onset_evidence_authorized=True,
                )
            except (TypeError, ValueError, KeyError, IndexError):
                reasons = ("reference_semantics_invalid",)
                candidate = None
        if candidate is not None:
            candidates.append(candidate)
        decisions.append(
            BAIEGReferenceFieldUnitDecision(
                event_id=event.event_id,
                recording_id=event.recording_id,
                batch_event_index=int(item["batch_index"]),
                analysis_unit_index=unit_index,
                view_index=view_index,
                view_id=event.view_ids[view_index],
                analysis_unit_id=event.unit_source_ids[unit_index],
                reference_family=event.reference_families[view_index],
                reference_row_sha256=_reference_row_sha256(item["row"]),
                usable_spatial_support_fraction=float(item["coverage"]),
                candidate_id=None if candidate is None else candidate.candidate_id,
                reason_codes=reasons,
            )
        )

    return BAIEGReferenceFieldCandidateMaterialization(
        source_input_batch_sha256=batch.input_batch_sha256,
        source_field_head_receipt_sha256=field_head_receipt,
        input_event_receipt_sha256s=tuple(
            event.input_receipt_sha256 for event in event_rows
        ),
        candidates=tuple(candidates),
        unit_decisions=tuple(decisions),
    )


__all__ = [
    "BA_IEG_REFERENCE_FIELD_CANDIDATE_MATERIALIZATION_SCHEMA_VERSION",
    "BA_IEG_REFERENCE_FIELD_CANDIDATE_MATERIALIZER_ID",
    "BA_IEG_REFERENCE_FIELD_SCORE_SEMANTICS",
    "BAIEGReferenceFieldCandidateMaterialization",
    "BAIEGReferenceFieldUnitDecision",
    "materialize_ba_ieg_reference_field_candidates",
]
