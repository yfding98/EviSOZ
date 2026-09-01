"""EEG-only event->mode->record positive-set MIL reference contract.

This module is the executable contract between event Findings heads and the
record report graph.  It deliberately does not open EEG files and its forward
API accepts no labels, annotations, spreadsheets, clinical text or physician
targets.  Public DeepSOZ positive sets enter only through the separate loss
API; private hard/soft labels enter only through the post-freeze evaluator.

The implementation is a transparent reference model, not a trained clinical
release.  It provides the invariants that a learned replacement must retain:

* onset and later spread use separate tensors and hashes;
* mode discovery consumes only onset-safe phenotype/embedding evidence;
* aliases of one physical occurrence are collapsed exactly once;
* repeated events in one target-free pattern group contribute one mode unit;
* modes, rather than raw event counts, are the record aggregation units;
* multiple modes are preserved instead of averaged into a false focal site;
* probability wording and fine resolution require host-trusted, patient-
  disjoint calibration/risk receipts;
* leave-one-event-out (LOEO) stability is measured, not inferred from count.

Attention/cluster membership is an aggregation mechanism and is never exposed
as physiological evidence.  Report support still has to come from the event
EvidenceGraph and its future-free onset evidence IDs.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

import torch


MODE_AWARE_HIERARCHICAL_MIL_METHOD_ID = (
    "eeg_only_event_mode_record_hierarchical_positive_set_mil_v1"
)
MODE_AWARE_HIERARCHICAL_MIL_FORWARD_SCHEMA_VERSION = (
    "clinical_eeg_mode_aware_hierarchical_mil_forward_v1"
)
MODE_AWARE_HIERARCHICAL_MIL_DECODE_SCHEMA_VERSION = (
    "clinical_eeg_mode_aware_hierarchical_mil_decode_v1"
)
MODE_AWARE_HIERARCHICAL_MIL_LOEO_SCHEMA_VERSION = (
    "clinical_eeg_mode_aware_hierarchical_mil_loeo_v1"
)
# A plain Python mapping is not an authority boundary: the same caller could
# manufacture both receipts and the mapping.  Keep formal report promotion
# closed until a host-only signed registry adapter is connected outside the
# request/payload surface.
MODE_AWARE_HIERARCHICAL_MIL_TRUSTED_REGISTRY_ROUTE_CONNECTED = False

LOCALIZED_PHENOTYPE = "localized_or_lateralized_scalp_visible_onset_pattern"
WIDESPREAD_PHENOTYPE = (
    "widespread_bilateral_near_synchronous_scalp_onset_pattern"
)
NONLOCALIZABLE_PHENOTYPE = "scalp_onset_nonlocalizable"
MULTIPLE_MODE_PHENOTYPE = "multiple_scalp_onset_modes"
EVENT_PHENOTYPES = (
    LOCALIZED_PHENOTYPE,
    WIDESPREAD_PHENOTYPE,
    NONLOCALIZABLE_PHENOTYPE,
)

_SPATIAL_RESOLUTION_ORDER = {
    "phenotype_only": 0,
    "laterality": 1,
    "region": 2,
    "electrode": 3,
}
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_EPS = 1e-12


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be an opaque identifier")
    return value


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    header = f"{tensor.dtype}|{tuple(tensor.shape)}|".encode("ascii")
    return hashlib.sha256(header + tensor.numpy().tobytes()).hexdigest()


def _finite_rate(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0,1]")
    return result


def _positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _validate_vector(value: torch.Tensor, length: int | None, name: str) -> None:
    if not isinstance(value, torch.Tensor) or value.ndim != 1:
        raise TypeError(f"{name} must be a one-dimensional tensor")
    if length is not None and value.shape[0] != length:
        raise ValueError(f"{name} has the wrong length")
    if not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite floating point")


def _unique_identifiers(value: Sequence[str], name: str) -> tuple[str, ...]:
    result = tuple(_identifier(item, name) for item in value)
    if len(result) != len(set(result)):
        raise ValueError(f"{name} contains duplicates")
    return result


@dataclass(frozen=True)
class ModeAwareMILReferenceViewV1:
    """One correlated reference observation of one EEG event."""

    reference_id: str
    producer_sha256: str
    phenotype_logits: torch.Tensor
    onset_channel_logits: torch.Tensor
    spread_channel_logits: torch.Tensor
    onset_safe_mode_embedding: torch.Tensor
    mode_embedding_source_sha256: str
    mode_embedding_permission: str = "future_free_onset_mode_only"
    future_or_spread_samples_used_for_mode_embedding: bool = False
    quality: float = 1.0

    def __post_init__(self) -> None:
        _identifier(self.reference_id, "reference_id")
        _sha256(self.producer_sha256, "producer_sha256")
        _sha256(
            self.mode_embedding_source_sha256,
            "mode_embedding_source_sha256",
        )
        _validate_vector(self.phenotype_logits, len(EVENT_PHENOTYPES), "phenotype_logits")
        _validate_vector(self.onset_channel_logits, None, "onset_channel_logits")
        _validate_vector(self.spread_channel_logits, None, "spread_channel_logits")
        if self.onset_channel_logits.shape != self.spread_channel_logits.shape:
            raise ValueError("onset and spread channel logits must share shape")
        _validate_vector(
            self.onset_safe_mode_embedding,
            None,
            "onset_safe_mode_embedding",
        )
        if self.onset_safe_mode_embedding.numel() < 1:
            raise ValueError("onset_safe_mode_embedding must not be empty")
        if (
            self.mode_embedding_permission != "future_free_onset_mode_only"
            or self.future_or_spread_samples_used_for_mode_embedding is not False
        ):
            raise ValueError(
                "mode embedding must be future-free and cannot contain course/spread"
            )
        object.__setattr__(self, "quality", _finite_rate(self.quality, "quality"))


@dataclass(frozen=True)
class ModeAwareMILEventV1:
    """All target-free observations for one detector-roster event."""

    event_id: str
    source_event_graph_sha256: str
    physical_occurrence_sha256: str
    pattern_group_sha256: str
    reference_views: tuple[ModeAwareMILReferenceViewV1, ...]
    onset_evidence_ids: tuple[str, ...]
    spread_evidence_ids: tuple[str, ...]
    resolution_ceiling: str = "electrode"

    def __post_init__(self) -> None:
        _identifier(self.event_id, "event_id")
        _sha256(self.source_event_graph_sha256, "source_event_graph_sha256")
        _sha256(self.physical_occurrence_sha256, "physical_occurrence_sha256")
        _sha256(self.pattern_group_sha256, "pattern_group_sha256")
        if not self.reference_views or not all(
            isinstance(item, ModeAwareMILReferenceViewV1)
            for item in self.reference_views
        ):
            raise TypeError("reference_views must contain typed views")
        reference_ids = [item.reference_id for item in self.reference_views]
        if len(reference_ids) != len(set(reference_ids)):
            raise ValueError("one event repeats a reference_id")
        _unique_identifiers(self.onset_evidence_ids, "onset_evidence_id")
        _unique_identifiers(self.spread_evidence_ids, "spread_evidence_id")
        if set(self.onset_evidence_ids).intersection(self.spread_evidence_ids):
            raise ValueError("onset and spread evidence IDs must be disjoint")
        if self.resolution_ceiling not in _SPATIAL_RESOLUTION_ORDER:
            raise ValueError("event resolution_ceiling is unsupported")


@dataclass(frozen=True)
class CompleteRecordModeAwareMILBagV1:
    """Complete EEG-only event roster for one long recording."""

    patient_uid: str
    record_id: str
    canonical_signal_sha256: str
    mil_model_artifact_sha256: str
    events: tuple[ModeAwareMILEventV1, ...]
    source_scope: str = "public_source"
    labels_or_external_context_present: bool = False

    def __post_init__(self) -> None:
        _identifier(self.patient_uid, "patient_uid")
        _identifier(self.record_id, "record_id")
        _sha256(self.canonical_signal_sha256, "canonical_signal_sha256")
        _sha256(self.mil_model_artifact_sha256, "mil_model_artifact_sha256")
        if self.source_scope not in {
            "public_source",
            "synthetic",
            "deployment_eeg_only",
        }:
            raise ValueError("record MIL bag has an unsupported source_scope")
        if self.labels_or_external_context_present is not False:
            raise ValueError("labels/external context are forbidden in MIL forward inputs")
        if not self.events or not all(
            isinstance(item, ModeAwareMILEventV1) for item in self.events
        ):
            raise TypeError("record MIL bag requires at least one typed event")
        event_ids = [item.event_id for item in self.events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("complete event roster repeats event_id")


@dataclass(frozen=True)
class ModeAwareMILPolicyV1:
    """Frozen target-free clustering and hierarchy policy."""

    channel_ids: tuple[str, ...]
    channel_to_region: tuple[str, ...]
    region_to_laterality: tuple[tuple[str, str], ...]
    mode_js_threshold: float = 0.12
    reference_js_threshold: float = 0.18
    loeo_js_threshold: float = 0.12
    channel_distance_weight: float = 0.40
    region_distance_weight: float = 0.20
    laterality_distance_weight: float = 0.15
    phenotype_distance_weight: float = 0.15
    onset_mode_embedding_distance_weight: float = 0.10

    def __post_init__(self) -> None:
        channels = _unique_identifiers(self.channel_ids, "channel_id")
        if len(channels) < 2:
            raise ValueError("MIL hierarchy requires at least two channels")
        if len(self.channel_to_region) != len(channels):
            raise ValueError("channel_to_region must align with channel_ids")
        regions_for_channel = tuple(
            _identifier(item, "region_id") for item in self.channel_to_region
        )
        mapping = dict(self.region_to_laterality)
        if len(mapping) != len(self.region_to_laterality):
            raise ValueError("region_to_laterality repeats a region")
        if set(mapping) != set(regions_for_channel):
            raise ValueError("region_to_laterality must cover channel regions exactly")
        for region, laterality in self.region_to_laterality:
            _identifier(region, "region_id")
            _identifier(laterality, "laterality_id")
        object.__setattr__(self, "channel_ids", channels)
        object.__setattr__(self, "channel_to_region", regions_for_channel)
        for name in (
            "mode_js_threshold",
            "reference_js_threshold",
            "loeo_js_threshold",
        ):
            object.__setattr__(self, name, _finite_rate(getattr(self, name), name))
        weights = tuple(
            _finite_rate(getattr(self, name), name)
            for name in (
                "channel_distance_weight",
                "region_distance_weight",
                "laterality_distance_weight",
                "phenotype_distance_weight",
                "onset_mode_embedding_distance_weight",
            )
        )
        if not math.isclose(sum(weights), 1.0, abs_tol=1e-9):
            raise ValueError("mode distance weights must sum to one")

    @property
    def region_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.channel_to_region))

    @property
    def laterality_ids(self) -> tuple[str, ...]:
        mapping = dict(self.region_to_laterality)
        return tuple(dict.fromkeys(mapping[region] for region in self.region_ids))

    @property
    def policy_sha256(self) -> str:
        return _canonical_sha256(
            {
                "method_id": MODE_AWARE_HIERARCHICAL_MIL_METHOD_ID,
                "channel_ids": list(self.channel_ids),
                "channel_to_region": list(self.channel_to_region),
                "region_to_laterality": [list(item) for item in self.region_to_laterality],
                "mode_js_threshold": self.mode_js_threshold,
                "reference_js_threshold": self.reference_js_threshold,
                "loeo_js_threshold": self.loeo_js_threshold,
                "distance_weights": [
                    self.channel_distance_weight,
                    self.region_distance_weight,
                    self.laterality_distance_weight,
                    self.phenotype_distance_weight,
                    self.onset_mode_embedding_distance_weight,
                ],
            }
        )


@dataclass(frozen=True)
class _EventState:
    source_event_graph_sha256: str
    physical_occurrence_sha256: str
    pattern_group_sha256: str
    alias_event_ids: tuple[str, ...]
    phenotype_logits: torch.Tensor
    onset_logits: torch.Tensor
    spread_logits: torch.Tensor
    onset_mode_embedding: torch.Tensor
    quality: float
    maximum_reference_js: float
    resolution_ceiling: str
    onset_evidence_ids: tuple[str, ...]
    spread_evidence_ids: tuple[str, ...]
    content_sha256: str


@dataclass(frozen=True)
class ModeAwareMILForwardV1:
    """Target-free differentiable outputs plus sealed decision identities."""

    schema_version: str
    method_id: str
    patient_uid: str
    record_id: str
    canonical_signal_sha256: str
    policy_sha256: str
    mil_model_artifact_sha256: str
    channel_ids: tuple[str, ...]
    region_ids: tuple[str, ...]
    laterality_ids: tuple[str, ...]
    input_event_count: int
    unique_physical_event_count: int
    pattern_group_count: int
    mode_ids: tuple[str, ...]
    mode_physical_occurrence_sha256s: tuple[tuple[str, ...], ...]
    event_mode_membership: tuple[tuple[str, str, str], ...]
    event_alias_roster: tuple[tuple[str, ...], ...]
    mode_onset_logits: torch.Tensor
    mode_spread_logits: torch.Tensor
    mode_phenotype_logits: torch.Tensor
    record_onset_logits: torch.Tensor
    record_spread_logits: torch.Tensor
    record_phenotype_logits: torch.Tensor
    record_phenotype: str
    record_resolution_ceiling: str
    maximum_reference_js: float
    onset_evidence_ids: tuple[str, ...]
    spread_evidence_ids: tuple[str, ...]
    onset_decision_sha256: str
    spread_decision_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != MODE_AWARE_HIERARCHICAL_MIL_FORWARD_SCHEMA_VERSION:
            raise ValueError("unexpected mode-aware MIL forward schema")
        if self.method_id != MODE_AWARE_HIERARCHICAL_MIL_METHOD_ID:
            raise ValueError("unexpected mode-aware MIL method")
        _sha256(self.mil_model_artifact_sha256, "mil_model_artifact_sha256")
        modes = len(self.mode_ids)
        channels = len(self.channel_ids)
        if modes < 1 or self.unique_physical_event_count < 1:
            raise ValueError("forward output requires events and modes")
        if self.input_event_count < self.unique_physical_event_count:
            raise ValueError("input event count is smaller than deduplicated count")
        expected = {
            "mode_onset_logits": (modes, channels),
            "mode_spread_logits": (modes, channels),
            "mode_phenotype_logits": (modes, len(EVENT_PHENOTYPES)),
            "record_onset_logits": (channels,),
            "record_spread_logits": (channels,),
            "record_phenotype_logits": (len(EVENT_PHENOTYPES),),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if tuple(value.shape) != shape or not torch.isfinite(value).all():
                raise ValueError(f"{name} has invalid shape or values")
        if len(self.mode_physical_occurrence_sha256s) != modes:
            raise ValueError("mode membership does not align with modes")
        if len(self.event_mode_membership) != self.unique_physical_event_count:
            raise ValueError("event->mode membership does not cover unique events")
        if len(self.event_alias_roster) != self.unique_physical_event_count:
            raise ValueError("event alias roster does not cover unique events")
        if self.record_phenotype not in (*EVENT_PHENOTYPES, MULTIPLE_MODE_PHENOTYPE):
            raise ValueError("forward record phenotype is unsupported")
        if self.record_resolution_ceiling not in {
            *_SPATIAL_RESOLUTION_ORDER,
            "multiple_modes",
        }:
            raise ValueError("forward resolution ceiling is unsupported")
        _sha256(self.onset_decision_sha256, "onset_decision_sha256")
        _sha256(self.spread_decision_sha256, "spread_decision_sha256")


def _weighted_mean(values: Sequence[torch.Tensor], weights: Sequence[float]) -> torch.Tensor:
    if not values or len(values) != len(weights):
        raise ValueError("weighted mean requires aligned non-empty values")
    weight = values[0].new_tensor(weights)
    if float(weight.sum().item()) <= 0.0:
        weight = torch.ones_like(weight)
    weight = weight / weight.sum()
    return (torch.stack(tuple(values), dim=0) * weight.reshape((-1,) + (1,) * values[0].ndim)).sum(dim=0)


def _js_divergence(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.detach().to(dtype=torch.float64).clamp_min(_EPS)
    right = right.detach().to(dtype=torch.float64).clamp_min(_EPS)
    left = left / left.sum()
    right = right / right.sum()
    middle = (left + right) / 2.0
    value = 0.5 * (
        (left * (left / middle).log()).sum()
        + (right * (right / middle).log()).sum()
    ) / math.log(2.0)
    return min(1.0, max(0.0, float(value.item())))


def _hierarchy_probabilities(
    channel_probabilities: torch.Tensor,
    policy: ModeAwareMILPolicyV1,
) -> tuple[torch.Tensor, torch.Tensor]:
    region_rows = []
    for region in policy.region_ids:
        indices = [
            index
            for index, candidate_region in enumerate(policy.channel_to_region)
            if candidate_region == region
        ]
        region_rows.append(channel_probabilities[..., indices].sum(dim=-1))
    region = torch.stack(region_rows, dim=-1)
    region_to_laterality = dict(policy.region_to_laterality)
    laterality_rows = []
    for laterality in policy.laterality_ids:
        indices = [
            index
            for index, region_id in enumerate(policy.region_ids)
            if region_to_laterality[region_id] == laterality
        ]
        laterality_rows.append(region[..., indices].sum(dim=-1))
    return region, torch.stack(laterality_rows, dim=-1)


def _resolution_min(left: str, right: str) -> str:
    return min((left, right), key=lambda item: _SPATIAL_RESOLUTION_ORDER[item])


def _event_content_sha256(event: ModeAwareMILEventV1) -> str:
    return _canonical_sha256(
        {
            "physical_occurrence_sha256": event.physical_occurrence_sha256,
            "source_event_graph_sha256": event.source_event_graph_sha256,
            "pattern_group_sha256": event.pattern_group_sha256,
            "resolution_ceiling": event.resolution_ceiling,
            # Physical aliases are permitted only when they carry the exact
            # same evidence permissions.  Unioning different evidence-ID
            # rosters would silently broaden the support of one occurrence.
            "onset_evidence_ids": sorted(event.onset_evidence_ids),
            "spread_evidence_ids": sorted(event.spread_evidence_ids),
            "views": [
                {
                    "reference_id": view.reference_id,
                    "producer_sha256": view.producer_sha256,
                    "phenotype_logits_sha256": _tensor_sha256(view.phenotype_logits),
                    "onset_channel_logits_sha256": _tensor_sha256(
                        view.onset_channel_logits
                    ),
                    "spread_channel_logits_sha256": _tensor_sha256(
                        view.spread_channel_logits
                    ),
                    "onset_safe_mode_embedding_sha256": _tensor_sha256(
                        view.onset_safe_mode_embedding
                    ),
                    "mode_embedding_source_sha256": view.mode_embedding_source_sha256,
                    "mode_embedding_permission": view.mode_embedding_permission,
                    "future_or_spread_samples_used_for_mode_embedding": False,
                    "quality": view.quality,
                }
                for view in sorted(event.reference_views, key=lambda row: row.reference_id)
            ],
        }
    )


def _event_state(
    aliases: Sequence[ModeAwareMILEventV1],
    policy: ModeAwareMILPolicyV1,
) -> _EventState:
    if not aliases:
        raise ValueError("physical event alias group is empty")
    fingerprints = {_event_content_sha256(item) for item in aliases}
    if len(fingerprints) != 1:
        raise ValueError("one physical occurrence has conflicting model evidence")
    representative = min(aliases, key=lambda row: row.event_id)
    if len({item.source_event_graph_sha256 for item in aliases}) != 1:
        raise ValueError("one physical occurrence has conflicting event graph hashes")
    raw_views = sorted(
        representative.reference_views, key=lambda row: row.reference_id
    )
    # A renamed copy of one producer/source observation is correlated evidence,
    # not another independent reference vote.  Exact copies are collapsed;
    # conflicting copies sharing the same producer/source identity are rejected.
    # A genuinely different montage/reference must therefore carry its own
    # source hash.
    observations: dict[tuple[str, str], ModeAwareMILReferenceViewV1] = {}
    for view in raw_views:
        observation_key = (
            view.producer_sha256,
            view.mode_embedding_source_sha256,
        )
        prior = observations.get(observation_key)
        if prior is None:
            observations[observation_key] = view
            continue
        same_payload = (
            torch.equal(prior.phenotype_logits, view.phenotype_logits)
            and torch.equal(prior.onset_channel_logits, view.onset_channel_logits)
            and torch.equal(prior.spread_channel_logits, view.spread_channel_logits)
            and torch.equal(
                prior.onset_safe_mode_embedding,
                view.onset_safe_mode_embedding,
            )
            and prior.mode_embedding_permission == view.mode_embedding_permission
            and prior.future_or_spread_samples_used_for_mode_embedding
            == view.future_or_spread_samples_used_for_mode_embedding
            and math.isclose(prior.quality, view.quality, abs_tol=1e-12)
        )
        if not same_payload:
            raise ValueError(
                "one producer/reference-source observation has conflicting copies"
            )
    views = sorted(observations.values(), key=lambda row: row.reference_id)
    embedding_size = views[0].onset_safe_mode_embedding.numel()
    for view in views:
        if view.onset_channel_logits.numel() != len(policy.channel_ids):
            raise ValueError("event channel logits do not match the frozen ontology")
        if view.onset_safe_mode_embedding.numel() != embedding_size:
            raise ValueError("event onset-safe embeddings have inconsistent dimensions")
    weights = [max(_EPS, view.quality) for view in views]
    phenotype = _weighted_mean([view.phenotype_logits for view in views], weights)
    onset = _weighted_mean([view.onset_channel_logits for view in views], weights)
    spread = _weighted_mean([view.spread_channel_logits for view in views], weights)
    onset_mode_embedding = _weighted_mean(
        [view.onset_safe_mode_embedding for view in views], weights
    )
    reference_probabilities = [
        torch.softmax(view.onset_channel_logits, dim=-1) for view in views
    ]
    maximum_reference_js = max(
        (
            _js_divergence(reference_probabilities[left], reference_probabilities[right])
            for left in range(len(views))
            for right in range(left + 1, len(views))
        ),
        default=0.0,
    )
    ceiling = representative.resolution_ceiling
    if maximum_reference_js > policy.reference_js_threshold:
        ceiling = _resolution_min(ceiling, "laterality")
    return _EventState(
        source_event_graph_sha256=representative.source_event_graph_sha256,
        physical_occurrence_sha256=representative.physical_occurrence_sha256,
        pattern_group_sha256=representative.pattern_group_sha256,
        alias_event_ids=tuple(sorted(item.event_id for item in aliases)),
        phenotype_logits=phenotype,
        onset_logits=onset,
        spread_logits=spread,
        onset_mode_embedding=onset_mode_embedding,
        quality=sum(view.quality for view in views) / len(views),
        maximum_reference_js=maximum_reference_js,
        resolution_ceiling=ceiling,
        onset_evidence_ids=tuple(
            sorted({item for alias in aliases for item in alias.onset_evidence_ids})
        ),
        spread_evidence_ids=tuple(
            sorted({item for alias in aliases for item in alias.spread_evidence_ids})
        ),
        content_sha256=next(iter(fingerprints)),
    )


def _canonical_event_states(
    bag: CompleteRecordModeAwareMILBagV1,
    policy: ModeAwareMILPolicyV1,
) -> list[_EventState]:
    by_physical: dict[str, list[ModeAwareMILEventV1]] = defaultdict(list)
    for event in bag.events:
        by_physical[event.physical_occurrence_sha256].append(event)
    return [
        _event_state(by_physical[key], policy)
        for key in sorted(by_physical)
    ]


def _cosine_distance(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.detach().to(dtype=torch.float64)
    right = right.detach().to(dtype=torch.float64)
    denominator = float(left.norm().item() * right.norm().item())
    if denominator <= _EPS:
        return 0.0 if torch.allclose(left, right) else 1.0
    cosine = float(torch.dot(left, right).item()) / denominator
    return min(1.0, max(0.0, (1.0 - cosine) / 2.0))


def _event_distance(
    left: _EventState,
    right: _EventState,
    policy: ModeAwareMILPolicyV1,
) -> float:
    left_channel = torch.softmax(left.onset_logits, dim=-1)
    right_channel = torch.softmax(right.onset_logits, dim=-1)
    left_region, left_laterality = _hierarchy_probabilities(left_channel, policy)
    right_region, right_laterality = _hierarchy_probabilities(right_channel, policy)
    return (
        policy.channel_distance_weight * _js_divergence(left_channel, right_channel)
        + policy.region_distance_weight * _js_divergence(left_region, right_region)
        + policy.laterality_distance_weight
        * _js_divergence(left_laterality, right_laterality)
        + policy.phenotype_distance_weight
        * _js_divergence(
            torch.softmax(left.phenotype_logits, dim=-1),
            torch.softmax(right.phenotype_logits, dim=-1),
        )
        + policy.onset_mode_embedding_distance_weight
        * _cosine_distance(left.onset_mode_embedding, right.onset_mode_embedding)
    )


def _complete_link_modes(
    states: Sequence[_EventState],
    policy: ModeAwareMILPolicyV1,
) -> list[list[int]]:
    pairwise = {
        (left, right): _event_distance(states[left], states[right], policy)
        for left in range(len(states))
        for right in range(left + 1, len(states))
    }
    clusters: list[list[int]] = [[index] for index in range(len(states))]

    def distance(left: Sequence[int], right: Sequence[int]) -> float:
        return max(
            pairwise[(min(a, b), max(a, b))]
            for a in left
            for b in right
        )

    while len(clusters) > 1:
        candidates: list[tuple[float, tuple[str, ...], int, int]] = []
        for left in range(len(clusters)):
            for right in range(left + 1, len(clusters)):
                value = distance(clusters[left], clusters[right])
                if value <= policy.mode_js_threshold:
                    signature = tuple(
                        sorted(
                            states[index].physical_occurrence_sha256
                            for index in (*clusters[left], *clusters[right])
                        )
                    )
                    candidates.append((value, signature, left, right))
        if not candidates:
            break
        _, _, left, right = min(candidates)
        merged = sorted(clusters[left] + clusters[right])
        clusters = [
            cluster
            for index, cluster in enumerate(clusters)
            if index not in {left, right}
        ]
        clusters.append(merged)
        clusters.sort(
            key=lambda members: tuple(
                states[index].physical_occurrence_sha256 for index in members
            )
        )
    return clusters


def _pattern_equal_aggregate(
    states: Sequence[_EventState],
    indices: Sequence[int],
    field: str,
) -> torch.Tensor:
    by_pattern: dict[str, list[_EventState]] = defaultdict(list)
    for index in indices:
        by_pattern[states[index].pattern_group_sha256].append(states[index])
    pattern_rows: list[torch.Tensor] = []
    for pattern_id in sorted(by_pattern):
        members = by_pattern[pattern_id]
        pattern_rows.append(
            _weighted_mean(
                [getattr(item, field) for item in members],
                [max(_EPS, item.quality) for item in members],
            )
        )
    # One target-free pattern group is one aggregation unit.  Raw seizure
    # recurrence is retained for description but cannot mechanically dominate.
    return torch.stack(pattern_rows, dim=0).mean(dim=0)


def _detached_rows(value: torch.Tensor) -> list[list[float]]:
    rows = value.detach().cpu().to(dtype=torch.float64)
    if rows.ndim == 1:
        rows = rows.unsqueeze(0)
    return [[round(float(item), 12) for item in row] for row in rows.tolist()]


def _decision_seal_payloads(
    *,
    patient_uid: str,
    record_id: str,
    canonical_signal_sha256: str,
    policy_sha256: str,
    mil_model_artifact_sha256: str,
    channel_ids: Sequence[str],
    region_ids: Sequence[str],
    laterality_ids: Sequence[str],
    input_event_count: int,
    unique_physical_event_count: int,
    pattern_group_count: int,
    mode_ids: Sequence[str],
    mode_membership: Sequence[Sequence[str]],
    event_mode_membership: Sequence[Sequence[str]],
    event_alias_roster: Sequence[Sequence[str]],
    mode_onset_logits: torch.Tensor,
    mode_spread_logits: torch.Tensor,
    mode_phenotype_logits: torch.Tensor,
    record_onset_logits: torch.Tensor,
    record_spread_logits: torch.Tensor,
    record_phenotype_logits: torch.Tensor,
    record_phenotype: str,
    record_resolution_ceiling: str,
    maximum_reference_js: float,
    onset_evidence_ids: Sequence[str],
    spread_evidence_ids: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build seals over every identity and tensor consumed by decode.

    Dataclass freezing does not make a ``torch.Tensor`` immutable.  Hashing the
    actual tensors here and replaying this function before decode prevents an
    in-place tensor edit from retaining a stale, apparently valid decision ID.
    """

    common = {
        "schema_version": MODE_AWARE_HIERARCHICAL_MIL_FORWARD_SCHEMA_VERSION,
        "method_id": MODE_AWARE_HIERARCHICAL_MIL_METHOD_ID,
        "patient_uid": patient_uid,
        "record_id": record_id,
        "canonical_signal_sha256": canonical_signal_sha256,
        "policy_sha256": policy_sha256,
        "mil_model_artifact_sha256": mil_model_artifact_sha256,
        "channel_ids": list(channel_ids),
        "region_ids": list(region_ids),
        "laterality_ids": list(laterality_ids),
        "input_event_count": input_event_count,
        "unique_physical_event_count": unique_physical_event_count,
        "pattern_group_count": pattern_group_count,
        "mode_ids": list(mode_ids),
        "mode_membership": [list(item) for item in mode_membership],
        "event_graph_mode_membership": [
            list(item) for item in event_mode_membership
        ],
        "event_alias_roster": [list(item) for item in event_alias_roster],
    }
    onset_payload = {
        **common,
        "mode_onset_logits_sha256": _tensor_sha256(mode_onset_logits),
        "mode_phenotype_logits_sha256": _tensor_sha256(mode_phenotype_logits),
        "record_onset_logits_sha256": _tensor_sha256(record_onset_logits),
        "record_phenotype_logits_sha256": _tensor_sha256(
            record_phenotype_logits
        ),
        "record_phenotype": record_phenotype,
        "record_resolution_ceiling": record_resolution_ceiling,
        "maximum_reference_js": round(float(maximum_reference_js), 12),
        "onset_evidence_ids": list(onset_evidence_ids),
    }
    spread_payload = {
        **common,
        "mode_spread_logits_sha256": _tensor_sha256(mode_spread_logits),
        "record_spread_logits_sha256": _tensor_sha256(record_spread_logits),
        "spread_evidence_ids": list(spread_evidence_ids),
    }
    return onset_payload, spread_payload


def _validate_forward_derivation_and_seals(
    forward: ModeAwareMILForwardV1,
) -> None:
    """Reject public construction, mutation, or inconsistent aggregation."""

    if len(forward.mode_ids) != len(set(forward.mode_ids)):
        raise ValueError("forward mode_ids contain duplicates")
    for mode_id in forward.mode_ids:
        _identifier(mode_id, "forward mode_id")
    _unique_identifiers(forward.channel_ids, "forward channel_id")
    _unique_identifiers(forward.region_ids, "forward region_id")
    _unique_identifiers(forward.laterality_ids, "forward laterality_id")
    if forward.pattern_group_count < 1 or (
        forward.pattern_group_count > forward.unique_physical_event_count
    ):
        raise ValueError("forward pattern_group_count is inconsistent")

    physical_flat = [
        item
        for members in forward.mode_physical_occurrence_sha256s
        for item in members
    ]
    if any(not members for members in forward.mode_physical_occurrence_sha256s):
        raise ValueError("forward contains an empty mode")
    if len(physical_flat) != len(set(physical_flat)) or (
        len(physical_flat) != forward.unique_physical_event_count
    ):
        raise ValueError("forward mode membership is not a physical-event partition")
    for physical_sha in physical_flat:
        _sha256(physical_sha, "forward physical occurrence")

    membership_by_physical: dict[str, tuple[str, str]] = {}
    for physical_sha, graph_sha, mode_id in forward.event_mode_membership:
        _sha256(physical_sha, "forward membership physical occurrence")
        _sha256(graph_sha, "forward membership event graph")
        _identifier(mode_id, "forward membership mode_id")
        if physical_sha in membership_by_physical:
            raise ValueError("forward repeats event->mode membership")
        membership_by_physical[physical_sha] = (graph_sha, mode_id)
    expected_mode_by_physical = {
        physical_sha: mode_id
        for mode_id, members in zip(
            forward.mode_ids,
            forward.mode_physical_occurrence_sha256s,
        )
        for physical_sha in members
    }
    if set(membership_by_physical) != set(expected_mode_by_physical) or any(
        membership_by_physical[physical_sha][1]
        != expected_mode_by_physical[physical_sha]
        for physical_sha in expected_mode_by_physical
    ):
        raise ValueError("forward event->mode rows contradict mode membership")

    aliases = [item for group in forward.event_alias_roster for item in group]
    if any(not group for group in forward.event_alias_roster):
        raise ValueError("forward contains an empty alias group")
    if len(aliases) != len(set(aliases)) or len(aliases) != forward.input_event_count:
        raise ValueError("forward alias roster is not an input-event partition")
    for alias in aliases:
        _identifier(alias, "forward alias event_id")
    _unique_identifiers(forward.onset_evidence_ids, "forward onset_evidence_id")
    _unique_identifiers(forward.spread_evidence_ids, "forward spread_evidence_id")
    if set(forward.onset_evidence_ids).intersection(forward.spread_evidence_ids):
        raise ValueError("forward onset/spread evidence permissions overlap")

    if not torch.allclose(
        forward.record_onset_logits,
        forward.mode_onset_logits.mean(dim=0),
        rtol=0.0,
        atol=1e-7,
    ):
        raise ValueError("record onset logits are not the sealed mode aggregate")
    if not torch.allclose(
        forward.record_spread_logits,
        forward.mode_spread_logits.mean(dim=0),
        rtol=0.0,
        atol=1e-7,
    ):
        raise ValueError("record spread logits are not the sealed mode aggregate")
    if not torch.allclose(
        forward.record_phenotype_logits,
        forward.mode_phenotype_logits.mean(dim=0),
        rtol=0.0,
        atol=1e-7,
    ):
        raise ValueError("record phenotype logits are not the sealed mode aggregate")
    if len(forward.mode_ids) > 1:
        if (
            forward.record_phenotype != MULTIPLE_MODE_PHENOTYPE
            or forward.record_resolution_ceiling != "multiple_modes"
        ):
            raise ValueError("multi-mode forward exposes an inconsistent conclusion")
    else:
        expected_phenotype = EVENT_PHENOTYPES[
            int(torch.argmax(forward.record_phenotype_logits.detach()).item())
        ]
        if forward.record_phenotype != expected_phenotype:
            raise ValueError("single-mode record phenotype contradicts its logits")
        if forward.record_resolution_ceiling == "multiple_modes":
            raise ValueError("single-mode forward has a multiple-mode ceiling")
        if (
            forward.record_phenotype != LOCALIZED_PHENOTYPE
            and forward.record_resolution_ceiling != "phenotype_only"
        ):
            raise ValueError("nonlocalized phenotype exceeds its resolution ceiling")

    onset_payload, spread_payload = _decision_seal_payloads(
        patient_uid=forward.patient_uid,
        record_id=forward.record_id,
        canonical_signal_sha256=forward.canonical_signal_sha256,
        policy_sha256=forward.policy_sha256,
        mil_model_artifact_sha256=forward.mil_model_artifact_sha256,
        channel_ids=forward.channel_ids,
        region_ids=forward.region_ids,
        laterality_ids=forward.laterality_ids,
        input_event_count=forward.input_event_count,
        unique_physical_event_count=forward.unique_physical_event_count,
        pattern_group_count=forward.pattern_group_count,
        mode_ids=forward.mode_ids,
        mode_membership=forward.mode_physical_occurrence_sha256s,
        event_mode_membership=forward.event_mode_membership,
        event_alias_roster=forward.event_alias_roster,
        mode_onset_logits=forward.mode_onset_logits,
        mode_spread_logits=forward.mode_spread_logits,
        mode_phenotype_logits=forward.mode_phenotype_logits,
        record_onset_logits=forward.record_onset_logits,
        record_spread_logits=forward.record_spread_logits,
        record_phenotype_logits=forward.record_phenotype_logits,
        record_phenotype=forward.record_phenotype,
        record_resolution_ceiling=forward.record_resolution_ceiling,
        maximum_reference_js=forward.maximum_reference_js,
        onset_evidence_ids=forward.onset_evidence_ids,
        spread_evidence_ids=forward.spread_evidence_ids,
    )
    if forward.onset_decision_sha256 != _canonical_sha256(onset_payload):
        raise ValueError("forward onset decision seal does not replay")
    if forward.spread_decision_sha256 != _canonical_sha256(spread_payload):
        raise ValueError("forward spread decision seal does not replay")


def forward_mode_aware_hierarchical_mil_v1(
    bag: CompleteRecordModeAwareMILBagV1,
    policy: ModeAwareMILPolicyV1,
) -> ModeAwareMILForwardV1:
    """Run target-free event->mode->record aggregation.

    The non-differentiable complete-link membership is computed from detached
    EEG-only event predictions.  Within a frozen membership, mode and record
    logits remain differentiable for positive-set training.
    """

    if not isinstance(bag, CompleteRecordModeAwareMILBagV1):
        raise TypeError("forward requires CompleteRecordModeAwareMILBagV1")
    if not isinstance(policy, ModeAwareMILPolicyV1):
        raise TypeError("forward requires ModeAwareMILPolicyV1")
    states = _canonical_event_states(bag, policy)
    clusters = _complete_link_modes(states, policy)
    mode_rows: list[tuple[str, list[int]]] = []
    for members in clusters:
        physical = tuple(
            sorted(states[index].physical_occurrence_sha256 for index in members)
        )
        mode_id = f"MODE-{_canonical_sha256({'physical': physical})[:16]}"
        mode_rows.append((mode_id, list(members)))
    mode_rows.sort(key=lambda item: item[0])

    mode_onset = torch.stack(
        [
            _pattern_equal_aggregate(states, members, "onset_logits")
            for _, members in mode_rows
        ],
        dim=0,
    )
    mode_spread = torch.stack(
        [
            _pattern_equal_aggregate(states, members, "spread_logits")
            for _, members in mode_rows
        ],
        dim=0,
    )
    mode_phenotype = torch.stack(
        [
            _pattern_equal_aggregate(states, members, "phenotype_logits")
            for _, members in mode_rows
        ],
        dim=0,
    )
    # Modes are equal record units; raw event prevalence is descriptive only.
    record_onset = mode_onset.mean(dim=0)
    record_spread = mode_spread.mean(dim=0)
    record_phenotype_logits = mode_phenotype.mean(dim=0)
    if len(mode_rows) > 1:
        record_phenotype = MULTIPLE_MODE_PHENOTYPE
        ceiling = "multiple_modes"
    else:
        record_phenotype = EVENT_PHENOTYPES[
            int(torch.argmax(record_phenotype_logits.detach()).item())
        ]
        member_ceilings = [states[index].resolution_ceiling for index in mode_rows[0][1]]
        ceiling = min(
            member_ceilings,
            key=lambda item: _SPATIAL_RESOLUTION_ORDER[item],
        )
        if record_phenotype != LOCALIZED_PHENOTYPE:
            ceiling = _resolution_min(ceiling, "phenotype_only")

    mode_membership = tuple(
        tuple(
            sorted(states[index].physical_occurrence_sha256 for index in members)
        )
        for _, members in mode_rows
    )
    mode_by_state_index = {
        index: mode_id
        for mode_id, members in mode_rows
        for index in members
    }
    event_mode_membership = tuple(
        sorted(
            (
                state.physical_occurrence_sha256,
                state.source_event_graph_sha256,
                mode_by_state_index[index],
            )
            for index, state in enumerate(states)
        )
    )
    event_alias_roster = tuple(
        state.alias_event_ids
        for state in sorted(states, key=lambda row: row.physical_occurrence_sha256)
    )
    maximum_reference_js = max(item.maximum_reference_js for item in states)
    onset_evidence_ids = tuple(
        sorted({item for state in states for item in state.onset_evidence_ids})
    )
    spread_evidence_ids = tuple(
        sorted({item for state in states for item in state.spread_evidence_ids})
    )
    onset_payload, spread_payload = _decision_seal_payloads(
        patient_uid=bag.patient_uid,
        record_id=bag.record_id,
        canonical_signal_sha256=bag.canonical_signal_sha256,
        policy_sha256=policy.policy_sha256,
        mil_model_artifact_sha256=bag.mil_model_artifact_sha256,
        channel_ids=policy.channel_ids,
        region_ids=policy.region_ids,
        laterality_ids=policy.laterality_ids,
        input_event_count=len(bag.events),
        unique_physical_event_count=len(states),
        pattern_group_count=len({item.pattern_group_sha256 for item in states}),
        mode_ids=tuple(item[0] for item in mode_rows),
        mode_membership=mode_membership,
        event_mode_membership=event_mode_membership,
        event_alias_roster=event_alias_roster,
        mode_onset_logits=mode_onset,
        mode_spread_logits=mode_spread,
        mode_phenotype_logits=mode_phenotype,
        record_onset_logits=record_onset,
        record_spread_logits=record_spread,
        record_phenotype_logits=record_phenotype_logits,
        record_phenotype=record_phenotype,
        record_resolution_ceiling=ceiling,
        maximum_reference_js=maximum_reference_js,
        onset_evidence_ids=onset_evidence_ids,
        spread_evidence_ids=spread_evidence_ids,
    )
    result = ModeAwareMILForwardV1(
        schema_version=MODE_AWARE_HIERARCHICAL_MIL_FORWARD_SCHEMA_VERSION,
        method_id=MODE_AWARE_HIERARCHICAL_MIL_METHOD_ID,
        patient_uid=bag.patient_uid,
        record_id=bag.record_id,
        canonical_signal_sha256=bag.canonical_signal_sha256,
        policy_sha256=policy.policy_sha256,
        mil_model_artifact_sha256=bag.mil_model_artifact_sha256,
        channel_ids=policy.channel_ids,
        region_ids=policy.region_ids,
        laterality_ids=policy.laterality_ids,
        input_event_count=len(bag.events),
        unique_physical_event_count=len(states),
        pattern_group_count=len({item.pattern_group_sha256 for item in states}),
        mode_ids=tuple(item[0] for item in mode_rows),
        mode_physical_occurrence_sha256s=mode_membership,
        event_mode_membership=event_mode_membership,
        event_alias_roster=event_alias_roster,
        mode_onset_logits=mode_onset,
        mode_spread_logits=mode_spread,
        mode_phenotype_logits=mode_phenotype,
        record_onset_logits=record_onset,
        record_spread_logits=record_spread,
        record_phenotype_logits=record_phenotype_logits,
        record_phenotype=record_phenotype,
        record_resolution_ceiling=ceiling,
        maximum_reference_js=maximum_reference_js,
        onset_evidence_ids=onset_evidence_ids,
        spread_evidence_ids=spread_evidence_ids,
        onset_decision_sha256=_canonical_sha256(onset_payload),
        spread_decision_sha256=_canonical_sha256(spread_payload),
    )
    _validate_forward_derivation_and_seals(result)
    return result


@dataclass(frozen=True)
class ModeAwareMILPositiveSetTargetV1:
    """Public positive-only supervision; not a forward input."""

    patient_uid: str
    record_ids: tuple[str, ...]
    positive_channel_ids: tuple[str, ...]
    candidate_channel_ids: tuple[str, ...]
    source_reference_sha256: str
    model_split: str = "source_train"
    label_source: str = "deepsoz_public_patient_positive_set_v1"
    private_source: bool = False
    training_only_not_model_input: bool = True

    def __post_init__(self) -> None:
        _identifier(self.patient_uid, "positive-set patient_uid")
        records = _unique_identifiers(self.record_ids, "positive-set record_id")
        positives = _unique_identifiers(
            self.positive_channel_ids, "positive_channel_id"
        )
        candidates = _unique_identifiers(
            self.candidate_channel_ids, "candidate_channel_id"
        )
        if not records or not positives or not candidates:
            raise ValueError("positive-set target fields must be non-empty")
        if not set(positives).issubset(candidates):
            raise ValueError("positive channels must lie in candidate opportunity")
        if self.model_split != "source_train":
            raise ValueError("training loss accepts source_train targets only")
        if self.label_source != "deepsoz_public_patient_positive_set_v1":
            raise ValueError("positive-set training source is unsupported")
        if self.private_source is not False or self.training_only_not_model_input is not True:
            raise ValueError("private or model-input targets are forbidden")
        _sha256(self.source_reference_sha256, "source_reference_sha256")


def mode_aware_positive_set_mass_loss_v1(
    forwards: Sequence[ModeAwareMILForwardV1],
    targets: Sequence[ModeAwareMILPositiveSetTargetV1],
    *,
    smooth_max_temperature: float = 0.25,
) -> torch.Tensor:
    """Patient-equal positive-set MIL over complete record/mode bags.

    Every documented channel is an acceptable positive set member.  Channels
    outside that set are the operational listwise denominator, not asserted
    physician-confirmed negatives.  The target is applied once per patient,
    never copied onto each event.
    """

    if not forwards or not all(isinstance(item, ModeAwareMILForwardV1) for item in forwards):
        raise TypeError("positive-set loss requires non-empty forward outputs")
    for item in forwards:
        _validate_forward_derivation_and_seals(item)
    if not targets or not all(
        isinstance(item, ModeAwareMILPositiveSetTargetV1) for item in targets
    ):
        raise TypeError("positive-set loss requires typed targets")
    target_patient_ids = [item.patient_uid for item in targets]
    if len(target_patient_ids) != len(set(target_patient_ids)):
        raise ValueError("positive-set loss repeats a patient target")
    temperature = _positive_float(smooth_max_temperature, "smooth_max_temperature")
    by_patient: dict[str, list[ModeAwareMILForwardV1]] = defaultdict(list)
    forward_record_keys: set[tuple[str, str]] = set()
    for forward in forwards:
        key = (forward.patient_uid, forward.record_id)
        if key in forward_record_keys:
            raise ValueError("positive-set loss repeats a patient/record forward")
        forward_record_keys.add(key)
        by_patient[forward.patient_uid].append(forward)
    if set(by_patient) != {item.patient_uid for item in targets}:
        raise ValueError("forward and target patient rosters do not match")
    rows: list[torch.Tensor] = []
    for target in targets:
        patient_forwards = by_patient[target.patient_uid]
        if {item.record_id for item in patient_forwards} != set(target.record_ids):
            raise ValueError("positive-set target does not cover the complete record bag")
        channel_ids = patient_forwards[0].channel_ids
        if any(item.channel_ids != channel_ids for item in patient_forwards):
            raise ValueError("one patient bag mixes channel ontologies")
        if len({item.policy_sha256 for item in patient_forwards}) != 1:
            raise ValueError("one patient bag mixes MIL policies")
        if len({item.mil_model_artifact_sha256 for item in patient_forwards}) != 1:
            raise ValueError("one patient bag mixes MIL model artifacts")
        index = {channel: position for position, channel in enumerate(channel_ids)}
        if not set(target.candidate_channel_ids).issubset(index):
            raise ValueError("candidate target contains an unavailable channel")
        candidate_indices = torch.tensor(
            [index[item] for item in target.candidate_channel_ids],
            dtype=torch.long,
            device=patient_forwards[0].mode_onset_logits.device,
        )
        positive_set = set(target.positive_channel_ids)
        positive_positions = torch.tensor(
            [
                position
                for position, channel in enumerate(target.candidate_channel_ids)
                if channel in positive_set
            ],
            dtype=torch.long,
            device=candidate_indices.device,
        )
        log_masses: list[torch.Tensor] = []
        for forward in patient_forwards:
            candidate_logits = forward.mode_onset_logits.index_select(
                dim=1, index=candidate_indices
            )
            log_probabilities = torch.log_softmax(candidate_logits, dim=-1)
            log_masses.extend(
                torch.logsumexp(row.index_select(0, positive_positions), dim=0)
                for row in log_probabilities
            )
        stacked = torch.stack(log_masses)
        # Normalized smooth maximum: duplicating an identical mode cannot make
        # the mass exceed its original value merely by increasing bag length.
        bag_log_mass = temperature * (
            torch.logsumexp(stacked / temperature, dim=0)
            - math.log(len(log_masses))
        )
        rows.append(-bag_log_mass)
    return torch.stack(rows).mean()


@dataclass(frozen=True)
class ModeAwareMILModelReceiptV1:
    """Host-trusted receipt for the frozen event->mode->record model."""

    receipt_id: str
    artifact_sha256: str
    policy_sha256: str
    source_train_manifest_sha256: str
    method_id: str = MODE_AWARE_HIERARCHICAL_MIL_METHOD_ID
    validation_scope: str = "source_dev_patient_disjoint"
    patient_disjoint: bool = True
    frozen_before_inference: bool = True
    private_data_used: bool = False

    def __post_init__(self) -> None:
        _identifier(self.receipt_id, "model receipt_id")
        _sha256(self.artifact_sha256, "model artifact_sha256")
        _sha256(self.policy_sha256, "model policy_sha256")
        _sha256(self.source_train_manifest_sha256, "source_train_manifest_sha256")
        if self.method_id != MODE_AWARE_HIERARCHICAL_MIL_METHOD_ID:
            raise ValueError("model receipt method_id mismatch")
        if self.validation_scope not in {
            "source_dev_patient_disjoint",
            "external_patient_disjoint",
        }:
            raise ValueError("model receipt requires patient-disjoint validation")
        if (
            self.patient_disjoint is not True
            or self.frozen_before_inference is not True
            or self.private_data_used is not False
        ):
            raise ValueError("model receipt must be frozen, patient-disjoint and public")


_HARD_ONSET_PROVENANCE_ROLES = (
    "phenotype_logits",
    "onset_channel_logits",
    "quality_weight",
    "onset_safe_mode_embedding",
    "pattern_group_assignment",
)


@dataclass(frozen=True)
class ModeAwareMILInputProvenanceReceiptV1:
    """Host-registry receipt closing every input path into hard-onset decode.

    The local MIL module cannot inspect an upstream EvidenceGraph.  Therefore a
    formal resolution is fail-closed unless the host supplies this typed,
    artifact-bound receipt after validating event-scoped permission edges and
    constructive spatial receipts for all five hard-onset inputs.
    """

    receipt_id: str
    artifact_sha256: str
    record_id: str
    canonical_signal_sha256: str
    policy_sha256: str
    mil_model_artifact_sha256: str
    forward_onset_decision_sha256: str
    source_event_graph_closure_sha256: str
    closed_roles: tuple[str, ...] = _HARD_ONSET_PROVENANCE_ROLES
    permission_scope: str = "future_free_onset_only"
    event_scoped_evidence_ids: bool = True
    constructive_spatial_receipts_bound: bool = True
    future_or_spread_path_to_hard_onset: bool = False
    host_registry_verified: bool = True

    def __post_init__(self) -> None:
        _identifier(self.receipt_id, "input provenance receipt_id")
        _identifier(self.record_id, "input provenance record_id")
        for name in (
            "artifact_sha256",
            "canonical_signal_sha256",
            "policy_sha256",
            "mil_model_artifact_sha256",
            "forward_onset_decision_sha256",
            "source_event_graph_closure_sha256",
        ):
            _sha256(getattr(self, name), f"input provenance {name}")
        if self.closed_roles != _HARD_ONSET_PROVENANCE_ROLES:
            raise ValueError("input provenance does not close every hard-onset role")
        if self.permission_scope != "future_free_onset_only":
            raise ValueError("hard-onset provenance must be future-free")
        if (
            self.event_scoped_evidence_ids is not True
            or self.constructive_spatial_receipts_bound is not True
            or self.future_or_spread_path_to_hard_onset is not False
            or self.host_registry_verified is not True
        ):
            raise ValueError("input provenance receipt is not host-trusted and closed")


@dataclass(frozen=True)
class ModeAwareMILCalibrationReceiptV1:
    receipt_id: str
    artifact_sha256: str
    source_dev_prediction_sha256: str
    policy_sha256: str
    model_artifact_sha256: str
    channel_temperature: float
    phenotype_temperature: float
    spread_temperature: float
    aps_threshold_by_resolution: tuple[tuple[str, float], ...]
    patient_disjoint: bool = True
    frozen_before_inference: bool = True
    private_data_used: bool = False

    def __post_init__(self) -> None:
        _identifier(self.receipt_id, "calibration receipt_id")
        _sha256(self.artifact_sha256, "calibration artifact_sha256")
        _sha256(
            self.source_dev_prediction_sha256,
            "source_dev_prediction_sha256",
        )
        _sha256(self.policy_sha256, "calibration policy_sha256")
        _sha256(self.model_artifact_sha256, "calibration model_artifact_sha256")
        for name in (
            "channel_temperature",
            "phenotype_temperature",
            "spread_temperature",
        ):
            object.__setattr__(self, name, _positive_float(getattr(self, name), name))
        thresholds = dict(self.aps_threshold_by_resolution)
        if set(thresholds) != set(_SPATIAL_RESOLUTION_ORDER):
            raise ValueError("APS thresholds must cover every spatial resolution")
        for resolution, threshold in self.aps_threshold_by_resolution:
            if resolution not in _SPATIAL_RESOLUTION_ORDER:
                raise ValueError("calibration has an unsupported resolution")
            _finite_rate(threshold, f"APS threshold {resolution}")
        if (
            self.patient_disjoint is not True
            or self.frozen_before_inference is not True
            or self.private_data_used is not False
        ):
            raise ValueError("calibration must be frozen, patient-disjoint and public")


@dataclass(frozen=True)
class ResolutionRiskBoundV1:
    resolution: str
    upper_conditional_risk: float
    risk_limit: float
    maximum_prediction_set_size: int
    minimum_loeo_stability: float

    def __post_init__(self) -> None:
        if self.resolution not in _SPATIAL_RESOLUTION_ORDER:
            raise ValueError("risk bound resolution is unsupported")
        object.__setattr__(
            self,
            "upper_conditional_risk",
            _finite_rate(self.upper_conditional_risk, "upper_conditional_risk"),
        )
        object.__setattr__(self, "risk_limit", _finite_rate(self.risk_limit, "risk_limit"))
        if (
            isinstance(self.maximum_prediction_set_size, bool)
            or not isinstance(self.maximum_prediction_set_size, int)
            or self.maximum_prediction_set_size < 1
        ):
            raise ValueError("maximum_prediction_set_size must be positive")
        object.__setattr__(
            self,
            "minimum_loeo_stability",
            _finite_rate(self.minimum_loeo_stability, "minimum_loeo_stability"),
        )


@dataclass(frozen=True)
class ModeAwareMILResolutionRiskReceiptV1:
    receipt_id: str
    artifact_sha256: str
    source_dev_prediction_sha256: str
    policy_sha256: str
    model_artifact_sha256: str
    calibration_artifact_sha256: str
    calibration_receipt_id: str
    bounds: tuple[ResolutionRiskBoundV1, ...]
    patient_disjoint: bool = True
    frozen_before_inference: bool = True
    private_data_used: bool = False

    def __post_init__(self) -> None:
        _identifier(self.receipt_id, "risk receipt_id")
        _sha256(self.artifact_sha256, "risk artifact_sha256")
        _sha256(self.source_dev_prediction_sha256, "risk source_dev_prediction_sha256")
        _sha256(self.policy_sha256, "risk policy_sha256")
        _sha256(self.model_artifact_sha256, "risk model_artifact_sha256")
        _sha256(
            self.calibration_artifact_sha256,
            "risk calibration_artifact_sha256",
        )
        _identifier(self.calibration_receipt_id, "risk calibration_receipt_id")
        if not self.bounds or not all(
            isinstance(item, ResolutionRiskBoundV1) for item in self.bounds
        ):
            raise TypeError("risk receipt requires typed resolution bounds")
        resolutions = [item.resolution for item in self.bounds]
        if len(resolutions) != len(set(resolutions)):
            raise ValueError("risk receipt repeats a resolution")
        if (
            self.patient_disjoint is not True
            or self.frozen_before_inference is not True
            or self.private_data_used is not False
        ):
            raise ValueError("risk receipt must be frozen, patient-disjoint and public")


def _receipt_plain_value(value: Any) -> Any:
    if is_dataclass(value):
        return {
            field.name: _receipt_plain_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, tuple):
        return [_receipt_plain_value(item) for item in value]
    if isinstance(value, list):
        return [_receipt_plain_value(item) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _receipt_plain_value(item)
            for key, item in value.items()
        }
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError("receipt payload contains a non-canonical value")


def canonical_mode_aware_receipt_sha256_v1(receipt: object) -> str:
    """Canonical host-registry digest for one trusted receipt object."""

    allowed = (
        ModeAwareMILModelReceiptV1,
        ModeAwareMILCalibrationReceiptV1,
        ModeAwareMILResolutionRiskReceiptV1,
        ModeAwareMILInputProvenanceReceiptV1,
    )
    if not isinstance(receipt, allowed):
        raise TypeError("unsupported mode-aware receipt type")
    return _canonical_sha256(
        {
            "receipt_type": type(receipt).__name__,
            "payload": _receipt_plain_value(receipt),
        }
    )


def _host_registry_exact_match(
    host_trusted_receipt_maps: Mapping[str, Mapping[str, Any]] | None,
    *,
    kind: str,
    receipt: object | None,
) -> bool:
    """Require an exact host registry object and its replayed canonical hash."""

    if receipt is None or host_trusted_receipt_maps is None:
        return False
    bucket = host_trusted_receipt_maps.get(kind)
    receipt_id = getattr(receipt, "receipt_id", None)
    if not isinstance(bucket, Mapping) or not isinstance(receipt_id, str):
        return False
    entry = bucket.get(receipt_id)
    if entry is None:
        return False
    if not isinstance(entry, Mapping) or set(entry) != {
        "receipt",
        "canonical_payload_sha256",
    }:
        raise ValueError("host receipt registry entry has an invalid shape")
    registered = entry["receipt"]
    registered_sha = entry["canonical_payload_sha256"]
    if type(registered) is not type(receipt) or registered != receipt:
        raise ValueError("supplied receipt differs from the host registry object")
    replayed_sha = canonical_mode_aware_receipt_sha256_v1(registered)
    if registered_sha != replayed_sha or (
        canonical_mode_aware_receipt_sha256_v1(receipt) != replayed_sha
    ):
        raise ValueError("host receipt canonical payload seal does not replay")
    return True


def _aps_prediction_set(
    identifiers: Sequence[str], probabilities: torch.Tensor, threshold: float
) -> list[str]:
    order = sorted(
        range(len(identifiers)),
        key=lambda index: (-float(probabilities[index].detach().item()), identifiers[index]),
    )
    result: list[str] = []
    cumulative = 0.0
    for index in order:
        result.append(identifiers[index])
        cumulative += float(probabilities[index].detach().item())
        if cumulative + 1e-12 >= threshold:
            break
    return result


def _ranked_axis(
    identifiers: Sequence[str],
    values: torch.Tensor,
    *,
    value_field: str,
) -> list[dict[str, Any]]:
    if value_field not in {"score", "probability"}:
        raise ValueError("ranked axis value_field is unsupported")
    order = sorted(
        range(len(identifiers)),
        key=lambda index: (-float(values[index].detach().item()), identifiers[index]),
    )
    return [
        {
            "rank": rank,
            "candidate_id": identifiers[index],
            value_field: float(values[index].detach().item()),
        }
        for rank, index in enumerate(order, start=1)
    ]


def _canonical_physical_partition(
    groups: Sequence[Sequence[str] | set[str]],
    *,
    removed: str | None = None,
) -> tuple[tuple[str, ...], ...]:
    rows = []
    for group in groups:
        members = set(group)
        if removed is not None:
            members.discard(removed)
        if members:
            rows.append(tuple(sorted(members)))
    return tuple(sorted(rows))


def _loeo_partition_replay_v1(
    full_groups: Sequence[Sequence[str] | set[str]],
    reduced_groups: Sequence[Sequence[str] | set[str]],
    removed: str,
) -> dict[str, Any]:
    """Compare exact projected membership, not merely the number of modes."""

    projected = _canonical_physical_partition(full_groups, removed=removed)
    reduced = _canonical_physical_partition(reduced_groups)
    return {
        "projected_full_partition_sha256": _canonical_sha256(projected),
        "reduced_partition_sha256": _canonical_sha256(reduced),
        "remaining_partition_exact": projected == reduced,
        "unexpected_mode_change": projected != reduced,
    }


def compute_mode_aware_loeo_stability_v1(
    bag: CompleteRecordModeAwareMILBagV1,
    policy: ModeAwareMILPolicyV1,
) -> dict[str, Any]:
    """Recompute the complete model after removing each physical event once."""

    full = forward_mode_aware_hierarchical_mil_v1(bag, policy)
    replay_binding = {
        "method_id": MODE_AWARE_HIERARCHICAL_MIL_METHOD_ID,
        "record_id": bag.record_id,
        "canonical_signal_sha256": bag.canonical_signal_sha256,
        "policy_sha256": policy.policy_sha256,
        "mil_model_artifact_sha256": bag.mil_model_artifact_sha256,
        "full_onset_decision_sha256": full.onset_decision_sha256,
        "event_mode_membership": [
            list(item) for item in full.event_mode_membership
        ],
        "event_alias_roster": [list(item) for item in full.event_alias_roster],
    }
    event_roster_sha256 = _canonical_sha256(replay_binding)
    physical_ids = sorted(
        {event.physical_occurrence_sha256 for event in bag.events}
    )
    if len(physical_ids) == 1:
        payload = {
            "schema_version": MODE_AWARE_HIERARCHICAL_MIL_LOEO_SCHEMA_VERSION,
            "method_id": MODE_AWARE_HIERARCHICAL_MIL_METHOD_ID,
            "record_id": bag.record_id,
            "canonical_signal_sha256": bag.canonical_signal_sha256,
            "policy_sha256": policy.policy_sha256,
            "mil_model_artifact_sha256": bag.mil_model_artifact_sha256,
            "event_roster_sha256": event_roster_sha256,
            "full_onset_decision_sha256": full.onset_decision_sha256,
            "status": "not_evaluable_single_unique_event",
            "unique_event_count": 1,
            "stability_score": None,
            "maximum_channel_js": None,
            "top1_flip_fraction": None,
            "mode_count_change_fraction": None,
            "unexpected_mode_change_count": None,
            "stable_for_resolution": False,
            "rows": [],
        }
        return {**payload, "receipt_sha256": _canonical_sha256(payload)}
    full_probability = torch.softmax(full.record_onset_logits, dim=-1)
    full_top = full.channel_ids[int(torch.argmax(full_probability).item())]
    full_modes = [set(item) for item in full.mode_physical_occurrence_sha256s]
    rows: list[dict[str, Any]] = []
    for removed in physical_ids:
        reduced_bag = CompleteRecordModeAwareMILBagV1(
            patient_uid=bag.patient_uid,
            record_id=bag.record_id,
            canonical_signal_sha256=bag.canonical_signal_sha256,
            mil_model_artifact_sha256=bag.mil_model_artifact_sha256,
            events=tuple(
                event
                for event in bag.events
                if event.physical_occurrence_sha256 != removed
            ),
            source_scope=bag.source_scope,
        )
        reduced = forward_mode_aware_hierarchical_mil_v1(reduced_bag, policy)
        reduced_probability = torch.softmax(reduced.record_onset_logits, dim=-1)
        reduced_top = reduced.channel_ids[int(torch.argmax(reduced_probability).item())]
        containing_mode = next(item for item in full_modes if removed in item)
        expected_mode_loss = len(containing_mode) == 1
        partition_replay = _loeo_partition_replay_v1(
            full_modes,
            [set(item) for item in reduced.mode_physical_occurrence_sha256s],
            removed,
        )
        rows.append(
            {
                "removed_physical_occurrence_sha256": removed,
                "reduced_onset_decision_sha256": reduced.onset_decision_sha256,
                "channel_js": _js_divergence(full_probability, reduced_probability),
                "top1_changed": reduced_top != full_top,
                "full_mode_count": len(full.mode_ids),
                "reduced_mode_count": len(reduced.mode_ids),
                "expected_singleton_mode_loss": expected_mode_loss,
                **partition_replay,
            }
        )
    maximum_js = max(float(row["channel_js"]) for row in rows)
    top_flip = sum(bool(row["top1_changed"]) for row in rows) / len(rows)
    mode_change = sum(
        int(row["full_mode_count"] != row["reduced_mode_count"]) for row in rows
    ) / len(rows)
    unexpected = sum(bool(row["unexpected_mode_change"]) for row in rows)
    stability = max(0.0, 1.0 - sum(float(row["channel_js"]) for row in rows) / len(rows))
    payload = {
        "schema_version": MODE_AWARE_HIERARCHICAL_MIL_LOEO_SCHEMA_VERSION,
        "method_id": MODE_AWARE_HIERARCHICAL_MIL_METHOD_ID,
        "record_id": bag.record_id,
        "canonical_signal_sha256": bag.canonical_signal_sha256,
        "policy_sha256": policy.policy_sha256,
        "mil_model_artifact_sha256": bag.mil_model_artifact_sha256,
        "event_roster_sha256": event_roster_sha256,
        "full_onset_decision_sha256": full.onset_decision_sha256,
        "status": "evaluated",
        "unique_event_count": len(physical_ids),
        "stability_score": stability,
        "maximum_channel_js": maximum_js,
        "top1_flip_fraction": top_flip,
        "mode_count_change_fraction": mode_change,
        "unexpected_mode_change_count": unexpected,
        "stable_for_resolution": (
            maximum_js <= policy.loeo_js_threshold and unexpected == 0
        ),
        "rows": rows,
    }
    return {**payload, "receipt_sha256": _canonical_sha256(payload)}


def decode_mode_aware_hierarchical_mil_v1(
    forward: ModeAwareMILForwardV1,
    policy: ModeAwareMILPolicyV1,
    *,
    model_receipt: ModeAwareMILModelReceiptV1 | None,
    calibration_receipt: ModeAwareMILCalibrationReceiptV1 | None,
    risk_receipt: ModeAwareMILResolutionRiskReceiptV1 | None,
    loeo_receipt: Mapping[str, Any] | None,
    loeo_replay_bag: CompleteRecordModeAwareMILBagV1 | None = None,
    input_provenance_receipt: ModeAwareMILInputProvenanceReceiptV1 | None = None,
    host_trusted_receipt_maps: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Decode calibrated axes and choose the finest risk-controlled resolution."""

    if not isinstance(forward, ModeAwareMILForwardV1):
        raise TypeError("decode requires ModeAwareMILForwardV1")
    _validate_forward_derivation_and_seals(forward)
    if forward.policy_sha256 != policy.policy_sha256:
        raise ValueError("forward/policy hash mismatch")
    if (calibration_receipt is None) is not (risk_receipt is None):
        raise ValueError("calibration and risk receipts must be supplied together")
    if model_receipt is not None and (
        model_receipt.policy_sha256 != forward.policy_sha256
        or model_receipt.artifact_sha256 != forward.mil_model_artifact_sha256
    ):
        raise ValueError("model receipt is not bound to this forward/policy")
    if calibration_receipt is not None and model_receipt is None:
        raise ValueError("formal calibrated decoding requires a trusted model receipt")
    if input_provenance_receipt is not None:
        if not isinstance(
            input_provenance_receipt,
            ModeAwareMILInputProvenanceReceiptV1,
        ):
            raise TypeError("input_provenance_receipt has the wrong type")
        if (
            input_provenance_receipt.record_id != forward.record_id
            or input_provenance_receipt.canonical_signal_sha256
            != forward.canonical_signal_sha256
            or input_provenance_receipt.policy_sha256 != forward.policy_sha256
            or input_provenance_receipt.mil_model_artifact_sha256
            != forward.mil_model_artifact_sha256
            or input_provenance_receipt.forward_onset_decision_sha256
            != forward.onset_decision_sha256
        ):
            raise ValueError("input provenance receipt is not bound to this forward")
    model_host_registered = _host_registry_exact_match(
        host_trusted_receipt_maps,
        kind="model",
        receipt=model_receipt,
    )
    calibration_host_registered = _host_registry_exact_match(
        host_trusted_receipt_maps,
        kind="calibration",
        receipt=calibration_receipt,
    )
    risk_host_registered = _host_registry_exact_match(
        host_trusted_receipt_maps,
        kind="risk",
        receipt=risk_receipt,
    )
    provenance_host_registered = _host_registry_exact_match(
        host_trusted_receipt_maps,
        kind="input_provenance",
        receipt=input_provenance_receipt,
    )
    receipt_chain_exact_registry_match = all(
        (
            model_host_registered,
            calibration_host_registered,
            risk_host_registered,
            provenance_host_registered,
        )
    )
    formal_receipt_chain_host_registered = bool(
        MODE_AWARE_HIERARCHICAL_MIL_TRUSTED_REGISTRY_ROUTE_CONNECTED
        and receipt_chain_exact_registry_match
    )
    input_provenance_authorized = bool(
        MODE_AWARE_HIERARCHICAL_MIL_TRUSTED_REGISTRY_ROUTE_CONNECTED
        and provenance_host_registered
    )
    effective_loeo_receipt: Mapping[str, Any] | None = None
    if loeo_replay_bag is not None:
        if not isinstance(loeo_replay_bag, CompleteRecordModeAwareMILBagV1):
            raise TypeError("loeo_replay_bag must be a complete typed EEG bag")
        replay = forward_mode_aware_hierarchical_mil_v1(loeo_replay_bag, policy)
        if (
            replay.patient_uid != forward.patient_uid
            or replay.record_id != forward.record_id
            or replay.canonical_signal_sha256 != forward.canonical_signal_sha256
            or replay.mil_model_artifact_sha256
            != forward.mil_model_artifact_sha256
            or replay.onset_decision_sha256 != forward.onset_decision_sha256
            or replay.spread_decision_sha256 != forward.spread_decision_sha256
        ):
            raise ValueError("LOEO replay bag does not reproduce the forward output")
        recomputed_loeo = compute_mode_aware_loeo_stability_v1(
            loeo_replay_bag,
            policy,
        )
        if loeo_receipt is not None and dict(loeo_receipt) != recomputed_loeo:
            raise ValueError("caller LOEO receipt does not equal the replayed receipt")
        effective_loeo_receipt = recomputed_loeo
    elif loeo_receipt is not None:
        raise ValueError("caller-sealed LOEO is forbidden without an EEG-bag replay")

    if calibration_receipt is None:
        channel = torch.softmax(forward.record_onset_logits, dim=-1)
        phenotype = torch.softmax(forward.record_phenotype_logits, dim=-1)
        spread = torch.sigmoid(forward.record_spread_logits)
        selected_resolution = "no_record_inference"
        record_inference_status = "research_candidate_only"
        semantics = "uncalibrated_candidate_score_not_report_authorized"
        calibration_id = None
        risk_id = None
        prediction_sets: dict[str, list[str]] = {
            "electrode": [],
            "region": [],
            "laterality": [],
            "phenotype_only": [],
        }
        backoff_reasons = ["patient_disjoint_calibration_and_risk_receipts_missing"]
    else:
        assert risk_receipt is not None
        if (
            calibration_receipt.policy_sha256 != forward.policy_sha256
            or risk_receipt.policy_sha256 != forward.policy_sha256
            or risk_receipt.calibration_receipt_id != calibration_receipt.receipt_id
            or calibration_receipt.model_artifact_sha256
            != model_receipt.artifact_sha256
            or risk_receipt.model_artifact_sha256
            != model_receipt.artifact_sha256
            or risk_receipt.calibration_artifact_sha256
            != calibration_receipt.artifact_sha256
            or risk_receipt.source_dev_prediction_sha256
            != calibration_receipt.source_dev_prediction_sha256
        ):
            raise ValueError(
                "model/calibration/risk receipts do not form one bound chain"
            )
        channel = torch.softmax(
            forward.record_onset_logits / calibration_receipt.channel_temperature,
            dim=-1,
        )
        phenotype = torch.softmax(
            forward.record_phenotype_logits / calibration_receipt.phenotype_temperature,
            dim=-1,
        )
        spread = torch.sigmoid(
            forward.record_spread_logits / calibration_receipt.spread_temperature
        )
        region, laterality = _hierarchy_probabilities(channel, policy)
        thresholds = dict(calibration_receipt.aps_threshold_by_resolution)
        prediction_sets = {
            "electrode": _aps_prediction_set(policy.channel_ids, channel, thresholds["electrode"]),
            "region": _aps_prediction_set(policy.region_ids, region, thresholds["region"]),
            "laterality": _aps_prediction_set(
                policy.laterality_ids, laterality, thresholds["laterality"]
            ),
            "phenotype_only": _aps_prediction_set(
                EVENT_PHENOTYPES, phenotype, thresholds["phenotype_only"]
            ),
        }
        semantics = "calibrated_candidate_score_not_report_authorized"
        record_inference_status = "calibrated_no_risk_controlled_resolution"
        calibration_id = calibration_receipt.receipt_id
        risk_id = risk_receipt.receipt_id
        backoff_reasons: list[str] = []
        loeo_stability = (
            None
            if effective_loeo_receipt is None
            else effective_loeo_receipt.get("stability_score")
        )
        loeo_is_trusted_and_stable = bool(
            effective_loeo_receipt is not None
            and effective_loeo_receipt.get("status") == "evaluated"
            and loeo_stability is not None
            and effective_loeo_receipt.get("stable_for_resolution") is True
        )
        if not input_provenance_authorized:
            backoff_reasons.append("hard_onset_input_provenance_receipt_missing")
        if not formal_receipt_chain_host_registered:
            backoff_reasons.append("host_trusted_receipt_registry_chain_missing")
        if not loeo_is_trusted_and_stable:
            backoff_reasons.append("trusted_loeo_replay_unavailable_or_unstable")
        if forward.record_phenotype == MULTIPLE_MODE_PHENOTYPE:
            if (
                formal_receipt_chain_host_registered
                and loeo_is_trusted_and_stable
            ):
                selected_resolution = "multiple_modes"
                record_inference_status = (
                    "calibrated_mode_specific_hypotheses_only"
                )
                backoff_reasons.append("multiple_onset_modes_preserved")
            else:
                selected_resolution = "no_record_inference"
                backoff_reasons.append("multiple_modes_not_promoted_without_trust")
        else:
            bounds = sorted(
                risk_receipt.bounds,
                key=lambda item: -_SPATIAL_RESOLUTION_ORDER[item.resolution],
            )
            ceiling_value = _SPATIAL_RESOLUTION_ORDER[forward.record_resolution_ceiling]
            selected_resolution = "no_record_inference"
            for bound in bounds:
                if not formal_receipt_chain_host_registered:
                    backoff_reasons.append(
                        f"{bound.resolution}_host_receipt_chain_unverified"
                    )
                    continue
                if _SPATIAL_RESOLUTION_ORDER[bound.resolution] > ceiling_value:
                    backoff_reasons.append(
                        f"{bound.resolution}_above_signal_resolution_ceiling"
                    )
                    continue
                if bound.upper_conditional_risk > bound.risk_limit:
                    backoff_reasons.append(f"{bound.resolution}_risk_above_limit")
                    continue
                if len(prediction_sets[bound.resolution]) > bound.maximum_prediction_set_size:
                    backoff_reasons.append(
                        f"{bound.resolution}_prediction_set_too_large"
                    )
                    continue
                if not input_provenance_authorized:
                    backoff_reasons.append(
                        f"{bound.resolution}_input_provenance_unverified"
                    )
                    continue
                if not loeo_is_trusted_and_stable or (
                    float(loeo_stability) < bound.minimum_loeo_stability
                ):
                    backoff_reasons.append(f"{bound.resolution}_loeo_unstable_or_unavailable")
                    continue
                selected_resolution = bound.resolution
                record_inference_status = "calibrated_research_hypothesis"
                break
            if selected_resolution == "no_record_inference":
                backoff_reasons.append("no_risk_controlled_resolution_passed")
            elif forward.record_phenotype != LOCALIZED_PHENOTYPE:
                backoff_reasons.append("nonlocalized_event_phenotype")

    formal_report_authorized = bool(
        formal_receipt_chain_host_registered
        and selected_resolution
        in {*_SPATIAL_RESOLUTION_ORDER, "multiple_modes"}
        and record_inference_status
        in {
            "calibrated_research_hypothesis",
            "calibrated_mode_specific_hypotheses_only",
        }
    )
    if formal_report_authorized:
        semantics = "patient_disjoint_calibrated_probability"
        value_field = "probability"
    elif calibration_receipt is not None:
        semantics = "calibrated_candidate_score_not_report_authorized"
        value_field = "score"
    else:
        semantics = "uncalibrated_candidate_score_not_report_authorized"
        value_field = "score"
    if not formal_report_authorized:
        prediction_sets = {
            "electrode": [],
            "region": [],
            "laterality": [],
            "phenotype_only": [],
        }

    region, laterality = _hierarchy_probabilities(channel, policy)
    mode_rows = []
    for index, mode_id in enumerate(forward.mode_ids):
        mode_channel = torch.softmax(
            forward.mode_onset_logits[index]
            / (1.0 if calibration_receipt is None else calibration_receipt.channel_temperature),
            dim=-1,
        )
        mode_region, mode_laterality = _hierarchy_probabilities(mode_channel, policy)
        mode_phenotype = torch.softmax(
            forward.mode_phenotype_logits[index]
            / (1.0 if calibration_receipt is None else calibration_receipt.phenotype_temperature),
            dim=-1,
        )
        mode_spread = torch.sigmoid(
            forward.mode_spread_logits[index]
            / (1.0 if calibration_receipt is None else calibration_receipt.spread_temperature)
        )
        mode_rows.append(
            {
                "mode_id": mode_id,
                "mode_role": (
                    "mode_specific_report_support"
                    if formal_report_authorized
                    else "mode_specific_candidate_only"
                ),
                "report_authorized": formal_report_authorized,
                "qwen_or_report_use_authorized": formal_report_authorized,
                "physical_occurrence_sha256s": list(
                    forward.mode_physical_occurrence_sha256s[index]
                ),
                "phenotype_ranking": _ranked_axis(
                    EVENT_PHENOTYPES, mode_phenotype, value_field=value_field
                ),
                "electrode_hard_onset_ranking": _ranked_axis(
                    policy.channel_ids, mode_channel, value_field=value_field
                ),
                "region_hard_onset_ranking": _ranked_axis(
                    policy.region_ids, mode_region, value_field=value_field
                ),
                "laterality_ranking": _ranked_axis(
                    policy.laterality_ids,
                    mode_laterality,
                    value_field=value_field,
                ),
                "electrode_soft_spread_ranking": _ranked_axis(
                    policy.channel_ids, mode_spread, value_field=value_field
                ),
            }
        )
    record_axis_candidate_available = (
        forward.record_phenotype != MULTIPLE_MODE_PHENOTYPE
    )
    record_axis_reportable = bool(
        formal_report_authorized and record_axis_candidate_available
    )
    if record_axis_candidate_available:
        record_electrode_ranking = _ranked_axis(
            policy.channel_ids, channel, value_field=value_field
        )
        record_region_ranking = _ranked_axis(
            policy.region_ids, region, value_field=value_field
        )
        record_laterality_ranking = _ranked_axis(
            policy.laterality_ids, laterality, value_field=value_field
        )
        record_phenotype_ranking = _ranked_axis(
            EVENT_PHENOTYPES, phenotype, value_field=value_field
        )
        record_spread_ranking = _ranked_axis(
            policy.channel_ids, spread, value_field=value_field
        )
    else:
        # Averaging spatially distinct modes can manufacture a scalp site that
        # occurred in no seizure.  Only mode-specific rows remain reportable.
        record_electrode_ranking = []
        record_region_ranking = []
        record_laterality_ranking = []
        record_phenotype_ranking = []
        record_spread_ranking = []
        prediction_sets = {
            "electrode": [],
            "region": [],
            "laterality": [],
            "phenotype_only": [],
        }
    payload: dict[str, Any] = {
        "schema_version": MODE_AWARE_HIERARCHICAL_MIL_DECODE_SCHEMA_VERSION,
        "method_id": MODE_AWARE_HIERARCHICAL_MIL_METHOD_ID,
        "record_id": forward.record_id,
        "onset_decision_sha256": forward.onset_decision_sha256,
        "spread_decision_sha256": forward.spread_decision_sha256,
        "record_inference_status": record_inference_status,
        "formal_report_authorized": formal_report_authorized,
        "qwen_or_report_use_authorized": formal_report_authorized,
        "candidate_rankings_are_nonclinical": not formal_report_authorized,
        "record_phenotype": forward.record_phenotype,
        "selected_resolution": selected_resolution,
        "score_semantics": semantics,
        "model_receipt_id": None if model_receipt is None else model_receipt.receipt_id,
        "calibration_receipt_id": calibration_id,
        "risk_receipt_id": risk_id,
        "input_provenance_receipt_id": (
            None
            if input_provenance_receipt is None
            else input_provenance_receipt.receipt_id
        ),
        "formal_receipt_chain_host_registered": (
            formal_receipt_chain_host_registered
        ),
        "receipt_chain_exact_registry_match": receipt_chain_exact_registry_match,
        "trusted_registry_route_connected": (
            MODE_AWARE_HIERARCHICAL_MIL_TRUSTED_REGISTRY_ROUTE_CONNECTED
        ),
        "hard_onset": {
            "record_axis_reportable": record_axis_reportable,
            "record_axis_candidate_available": record_axis_candidate_available,
            "candidate_only": not formal_report_authorized,
            "qwen_or_report_use_authorized": formal_report_authorized,
            "electrode_ranking": record_electrode_ranking,
            "region_ranking": record_region_ranking,
            "laterality_ranking": record_laterality_ranking,
            "phenotype_ranking": record_phenotype_ranking,
            "prediction_sets": prediction_sets,
            "supporting_evidence_ids": list(forward.onset_evidence_ids),
        },
        "soft_spread": {
            "record_axis_reportable": record_axis_reportable,
            "record_axis_candidate_available": record_axis_candidate_available,
            "candidate_only": not formal_report_authorized,
            "qwen_or_report_use_authorized": formal_report_authorized,
            "electrode_ranking": record_spread_ranking,
            "supporting_evidence_ids": list(forward.spread_evidence_ids),
            "may_support_hard_onset": False,
        },
        "modes": mode_rows,
        "record_mode_decision": {
            "primary_mode_id": forward.mode_ids[0] if len(forward.mode_ids) == 1 else None,
            "alternative_mode_ids": (
                [] if len(forward.mode_ids) == 1 else list(forward.mode_ids)
            ),
            "raw_event_prevalence_used_to_select_primary": False,
            "record_average_spatial_ranking_reportable": record_axis_reportable,
        },
        "source_binding": {
            "canonical_signal_sha256": forward.canonical_signal_sha256,
            "policy_sha256": forward.policy_sha256,
            "mil_model_artifact_sha256": forward.mil_model_artifact_sha256,
            "event_alias_roster": [
                list(item) for item in forward.event_alias_roster
            ],
            "event_to_mode_membership": [
                {
                    "physical_occurrence_sha256": physical_sha,
                    "source_event_graph_sha256": graph_sha,
                    "mode_id": mode_id,
                }
                for physical_sha, graph_sha, mode_id in forward.event_mode_membership
            ],
        },
        "resolution_backoff_reason_codes": sorted(set(backoff_reasons)),
        "loeo_receipt_sha256": (
            None
            if effective_loeo_receipt is None
            else effective_loeo_receipt.get("receipt_sha256")
        ),
        "claim_boundary": {
            "research_scalp_visible_onset_topography_only": True,
            "cortical_soz_or_epileptogenic_zone": False,
            "spread_used_for_onset_or_mode_assignment": False,
            "raw_event_count_used_as_mode_weight": False,
            "multiple_mode_record_average_withheld": (
                forward.record_phenotype == MULTIPLE_MODE_PHENOTYPE
            ),
            "hard_onset_input_provenance_closed": input_provenance_authorized,
            "formal_receipts_equal_host_registry": (
                receipt_chain_exact_registry_match
            ),
            "qwen_or_report_use_authorized": formal_report_authorized,
            "candidate_scores_are_not_clinical_probabilities": (
                not formal_report_authorized
            ),
        },
    }
    payload["decode_sha256"] = _canonical_sha256(payload)
    return payload


@dataclass(frozen=True)
class PostFreezeSpreadEvaluationTargetV1:
    """Optional private/physician spread relevance for evaluation only."""

    record_id: str
    channel_ids: tuple[str, ...]
    relevance: tuple[float, ...]
    source_reference_sha256: str
    purpose: str = "post_freeze_evaluation_only"
    used_for_training_thresholds_or_inference: bool = False

    def __post_init__(self) -> None:
        _identifier(self.record_id, "spread evaluation record_id")
        channels = _unique_identifiers(self.channel_ids, "spread evaluation channel")
        if len(channels) != len(self.relevance):
            raise ValueError("spread relevance must align with channels")
        for index, value in enumerate(self.relevance):
            _finite_rate(value, f"spread relevance[{index}]")
        _sha256(self.source_reference_sha256, "spread source_reference_sha256")
        if (
            self.purpose != "post_freeze_evaluation_only"
            or self.used_for_training_thresholds_or_inference is not False
        ):
            raise ValueError("spread labels are post-freeze evaluation only")


def evaluate_post_freeze_spread_ndcg_v1(
    forward: ModeAwareMILForwardV1,
    target: PostFreezeSpreadEvaluationTargetV1,
) -> dict[str, Any]:
    """Score spread after prediction freeze without mutating onset inference."""

    _validate_forward_derivation_and_seals(forward)
    if forward.record_id != target.record_id or forward.channel_ids != target.channel_ids:
        raise ValueError("spread evaluation target is not bound to the prediction")
    order = sorted(
        range(len(forward.channel_ids)),
        key=lambda index: (
            -float(forward.record_spread_logits[index].detach().item()),
            forward.channel_ids[index],
        ),
    )
    ideal = sorted(target.relevance, reverse=True)

    def dcg(values: Sequence[float]) -> float:
        return sum(
            (2.0**float(value) - 1.0) / math.log2(rank + 1.0)
            for rank, value in enumerate(values, start=1)
        )

    observed = [target.relevance[index] for index in order]
    ideal_dcg = dcg(ideal)
    return {
        "record_id": forward.record_id,
        "prediction_spread_decision_sha256": forward.spread_decision_sha256,
        "prediction_onset_decision_sha256": forward.onset_decision_sha256,
        "source_reference_sha256": target.source_reference_sha256,
        "graded_spread_ndcg": 0.0 if ideal_dcg == 0.0 else dcg(observed) / ideal_dcg,
        "used_for_training_thresholds_or_inference": False,
    }


__all__ = [
    "CompleteRecordModeAwareMILBagV1",
    "EVENT_PHENOTYPES",
    "LOCALIZED_PHENOTYPE",
    "MODE_AWARE_HIERARCHICAL_MIL_DECODE_SCHEMA_VERSION",
    "MODE_AWARE_HIERARCHICAL_MIL_FORWARD_SCHEMA_VERSION",
    "MODE_AWARE_HIERARCHICAL_MIL_LOEO_SCHEMA_VERSION",
    "MODE_AWARE_HIERARCHICAL_MIL_METHOD_ID",
    "MODE_AWARE_HIERARCHICAL_MIL_TRUSTED_REGISTRY_ROUTE_CONNECTED",
    "MULTIPLE_MODE_PHENOTYPE",
    "ModeAwareMILCalibrationReceiptV1",
    "ModeAwareMILEventV1",
    "ModeAwareMILForwardV1",
    "ModeAwareMILInputProvenanceReceiptV1",
    "ModeAwareMILPolicyV1",
    "ModeAwareMILPositiveSetTargetV1",
    "ModeAwareMILReferenceViewV1",
    "ModeAwareMILResolutionRiskReceiptV1",
    "NONLOCALIZABLE_PHENOTYPE",
    "PostFreezeSpreadEvaluationTargetV1",
    "ResolutionRiskBoundV1",
    "WIDESPREAD_PHENOTYPE",
    "canonical_mode_aware_receipt_sha256_v1",
    "compute_mode_aware_loeo_stability_v1",
    "decode_mode_aware_hierarchical_mil_v1",
    "evaluate_post_freeze_spread_ndcg_v1",
    "forward_mode_aware_hierarchical_mil_v1",
    "mode_aware_positive_set_mass_loss_v1",
]
