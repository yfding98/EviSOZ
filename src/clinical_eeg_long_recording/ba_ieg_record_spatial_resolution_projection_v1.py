"""Conservative record-level scalp spatial projection for BA-IEG v1.

This module consumes only the masked, record-local typed-unit logits emitted
by :mod:`ba_ieg_capped_log_mean_exp_event_bag_v1`.  It deterministically
projects physical-electrode and *whole* bipolar-lead candidates onto three
research-only axes: laterality, coarse scalp region, and their admissible
joint identities.

The projection deliberately loses information rather than inventing it:

* a bipolar lead is never split into endpoint-electrode evidence;
* a lead supports laterality only when both endpoints have exactly the same
  frozen laterality (left, right, or midline);
* a lead supports a region only when it first passes that same-side rule and
  both endpoints also have exactly the same frozen coarse scalp region;
* within each projected candidate, physical-electrode evidence takes
  precedence over all mapped bipolar evidence, which is retained only as
  suppressed provenance and never counted as an additional vote; and
* false opportunity masks are absent evidence, not negative evidence.

The input record bag no longer retains occurrence-by-unit membership, so
exact cross-source, same-occurrence deduplication is not identifiable here.
Candidate-local physical precedence is therefore an intentionally stronger,
fail-closed approximation: physical and bipolar scores are never added for
one projected candidate, even if they might have arisen from distinct
occurrences.

All returned scores are uncalibrated scalp-visible onset *ranking* logits.
They are not clinical probabilities, cortical SOZ, epileptogenic-zone, or
surgical-target claims.  In particular, left and right spatial opportunities
never create a bilateral/near-synchronous phenotype; that requires explicit
record-level temporal evidence outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Final

import torch
from torch import nn

from src.soz.geometry import STANDARD_19

from .ba_ieg_capped_log_mean_exp_event_bag_v1 import (
    BA_IEG_CAPPED_LOG_MEAN_EXP_EVENT_BAG_ID,
    BAIEGCappedLogMeanExpEventBagOutput,
)
from .ba_ieg_permission_split_segmental_state_model_v1 import (
    BA_IEG_CAUSAL_TYPED_UNIT_KINDS,
)


BA_IEG_RECORD_SPATIAL_RESOLUTION_PROJECTION_ID: Final[str] = (
    "ba_ieg_record_conservative_spatial_resolution_projection_v1"
)
BA_IEG_RECORD_SPATIAL_RESOLUTION_SCHEMA_VERSION: Final[str] = (
    "ba_ieg_record_spatial_resolution_projection_v1"
)
BA_IEG_RECORD_SPATIAL_LATERALITY_IDS: Final[tuple[str, ...]] = (
    "left",
    "right",
    "midline",
)
BA_IEG_RECORD_SPATIAL_COARSE_REGION_IDS: Final[tuple[str, ...]] = (
    "frontal",
    "temporal",
    "central",
    "parietal",
    "occipital",
)

_LEFT_ELECTRODES: Final[frozenset[str]] = frozenset(
    {"FP1", "F7", "F3", "T7", "C3", "P7", "P3", "O1"}
)
_RIGHT_ELECTRODES: Final[frozenset[str]] = frozenset(
    {"FP2", "F8", "F4", "T8", "C4", "P8", "P4", "O2"}
)
_MIDLINE_ELECTRODES: Final[frozenset[str]] = frozenset({"FZ", "CZ", "PZ"})


def _electrode_laterality(electrode_id: str) -> str:
    if electrode_id in _LEFT_ELECTRODES:
        return "left"
    if electrode_id in _RIGHT_ELECTRODES:
        return "right"
    if electrode_id in _MIDLINE_ELECTRODES:
        return "midline"
    raise ValueError(f"unknown standard-19 electrode: {electrode_id}")


def _electrode_coarse_region(electrode_id: str) -> str:
    # This is a scalp grouping, not a cortical/anatomical source atlas.
    if electrode_id in {"F7", "T7", "P7", "F8", "T8", "P8"}:
        return "temporal"
    if electrode_id.startswith("FP") or electrode_id.startswith("F"):
        return "frontal"
    if electrode_id.startswith("C"):
        return "central"
    if electrode_id.startswith("P"):
        return "parietal"
    if electrode_id.startswith("O"):
        return "occipital"
    raise ValueError(f"standard-19 electrode has no coarse region: {electrode_id}")


_ELECTRODE_LATERALITY: Final[tuple[tuple[str, str], ...]] = tuple(
    (electrode, _electrode_laterality(electrode)) for electrode in STANDARD_19
)
_ELECTRODE_COARSE_REGION: Final[tuple[tuple[str, str], ...]] = tuple(
    (electrode, _electrode_coarse_region(electrode)) for electrode in STANDARD_19
)
_ELECTRODE_JOINT: Final[tuple[tuple[str, str], ...]] = tuple(
    (
        electrode,
        f"{_electrode_laterality(electrode)}_{_electrode_coarse_region(electrode)}",
    )
    for electrode in STANDARD_19
)
BA_IEG_RECORD_SPATIAL_JOINT_REGION_IDS: Final[tuple[str, ...]] = tuple(
    dict.fromkeys(value for _, value in _ELECTRODE_JOINT)
)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class BAIEGRecordSpatialResolutionOntologyV1:
    """Frozen standard-19 scalp projection ontology and policy receipt."""

    schema_version: str = BA_IEG_RECORD_SPATIAL_RESOLUTION_SCHEMA_VERSION
    standard_19: tuple[str, ...] = STANDARD_19
    laterality_ids: tuple[str, ...] = BA_IEG_RECORD_SPATIAL_LATERALITY_IDS
    coarse_region_ids: tuple[str, ...] = BA_IEG_RECORD_SPATIAL_COARSE_REGION_IDS
    joint_region_ids: tuple[str, ...] = BA_IEG_RECORD_SPATIAL_JOINT_REGION_IDS
    electrode_laterality: tuple[tuple[str, str], ...] = _ELECTRODE_LATERALITY
    electrode_coarse_region: tuple[tuple[str, str], ...] = (
        _ELECTRODE_COARSE_REGION
    )
    bipolar_laterality_rule: str = (
        "whole_lead_only_both_endpoints_exact_same_laterality"
    )
    bipolar_region_rule: str = (
        "whole_lead_only_same_laterality_and_exact_same_coarse_scalp_region"
    )
    cross_side_rule: str = "ineligible_for_both_laterality_and_region_axes"
    cross_region_same_side_rule: str = "laterality_only_region_ineligible"
    midline_rule: str = "midline_is_a_distinct_laterality_not_left_or_right"
    candidate_source_precedence: str = (
        "candidate_local_physical_electrode_over_whole_bipolar_lead"
    )
    aggregation: str = "capped_log_mean_exp_clip_12_temperature_1"
    masked_value_semantics: str = "not_evaluable_absent_not_negative"
    phenotype_rule: str = (
        "broad_bilateral_near_synchronous_requires_external_temporal_evidence"
    )
    ontology_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.standard_19 != STANDARD_19:
            raise ValueError("spatial ontology must use the exact STANDARD_19 basis")
        if tuple(item for item, _ in self.electrode_laterality) != STANDARD_19 or (
            tuple(item for item, _ in self.electrode_coarse_region) != STANDARD_19
        ):
            raise ValueError("spatial ontology electrode maps must align to STANDARD_19")
        if set(value for _, value in self.electrode_laterality) - set(
            self.laterality_ids
        ) or set(value for _, value in self.electrode_coarse_region) - set(
            self.coarse_region_ids
        ):
            raise ValueError("spatial ontology contains an undeclared candidate")
        payload = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "ontology_sha256"
        }
        object.__setattr__(self, "ontology_sha256", _canonical_sha256(payload))


BA_IEG_RECORD_SPATIAL_RESOLUTION_ONTOLOGY_V1: Final[
    BAIEGRecordSpatialResolutionOntologyV1
] = BAIEGRecordSpatialResolutionOntologyV1()
BA_IEG_RECORD_SPATIAL_RESOLUTION_ONTOLOGY_SHA256: Final[str] = (
    BA_IEG_RECORD_SPATIAL_RESOLUTION_ONTOLOGY_V1.ontology_sha256
)


@dataclass(frozen=True)
class BAIEGSpatialProjectionSourceProvenanceV1:
    """Identity-only provenance for one selected or suppressed typed unit."""

    typed_unit_index: int
    source_kind: str
    physical_electrode_id: str | None
    whole_bipolar_lead_endpoint_ids: tuple[str, str] | None

    def __post_init__(self) -> None:
        if self.typed_unit_index < 0:
            raise ValueError("typed-unit provenance index must be non-negative")
        if self.source_kind == "physical_electrode":
            if self.physical_electrode_id not in STANDARD_19 or (
                self.whole_bipolar_lead_endpoint_ids is not None
            ):
                raise ValueError("physical provenance must contain one electrode only")
        elif self.source_kind == "whole_bipolar_lead":
            endpoints = self.whole_bipolar_lead_endpoint_ids
            if (
                self.physical_electrode_id is not None
                or endpoints is None
                or len(endpoints) != 2
                or endpoints[0] not in STANDARD_19
                or endpoints[1] not in STANDARD_19
                or endpoints[0] == endpoints[1]
            ):
                raise ValueError("bipolar provenance must retain two whole endpoints")
        else:
            raise ValueError("spatial projection source kind is unsupported")

    @property
    def analysis_unit_id(self) -> str:
        if self.source_kind == "physical_electrode":
            if self.physical_electrode_id is None:  # guarded above
                raise RuntimeError("physical provenance lost its electrode")
            return self.physical_electrode_id
        endpoints = self.whole_bipolar_lead_endpoint_ids
        if endpoints is None:  # guarded above
            raise RuntimeError("bipolar provenance lost its endpoints")
        return f"{endpoints[0]}-{endpoints[1]}"


@dataclass(frozen=True)
class BAIEGSpatialCandidateProvenanceV1:
    candidate_id: str
    opportunity: bool
    selected_source_kind: str
    selected_sources: tuple[BAIEGSpatialProjectionSourceProvenanceV1, ...]
    suppressed_bipolar_sources: tuple[
        BAIEGSpatialProjectionSourceProvenanceV1, ...
    ]
    aggregation: str = "capped_log_mean_exp_clip_12_temperature_1_unique_units"
    masked_value_semantics: str = "not_evaluable_absent_not_negative"

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("spatial candidate provenance needs an identity")
        if self.opportunity != bool(self.selected_sources):
            raise ValueError("candidate opportunity must equal selected-source presence")
        selected_kinds = {item.source_kind for item in self.selected_sources}
        if self.opportunity:
            if selected_kinds != {self.selected_source_kind} or (
                self.selected_source_kind
                not in {"physical_electrode", "whole_bipolar_lead"}
            ):
                raise ValueError("candidate selected source kind is inconsistent")
        elif self.selected_source_kind != "not_evaluable":
            raise ValueError("masked candidate must remain not evaluable")
        if any(
            item.source_kind != "whole_bipolar_lead"
            for item in self.suppressed_bipolar_sources
        ):
            raise ValueError("only bipolar sources may be precedence-suppressed")
        if self.suppressed_bipolar_sources and (
            self.selected_source_kind != "physical_electrode"
        ):
            raise ValueError("bipolar suppression requires physical precedence")


@dataclass(frozen=True)
class BAIEGRecordSpatialProjectionProvenanceV1:
    recording_id: str
    physical_electrode_sources: tuple[
        BAIEGSpatialProjectionSourceProvenanceV1, ...
    ]
    whole_bipolar_lead_sources: tuple[
        BAIEGSpatialProjectionSourceProvenanceV1, ...
    ]
    laterality_candidates: tuple[BAIEGSpatialCandidateProvenanceV1, ...]
    coarse_region_candidates: tuple[BAIEGSpatialCandidateProvenanceV1, ...]
    joint_region_candidates: tuple[BAIEGSpatialCandidateProvenanceV1, ...]
    cross_side_leads_excluded_from_both_axes: tuple[
        BAIEGSpatialProjectionSourceProvenanceV1, ...
    ]
    same_side_cross_region_leads_laterality_only: tuple[
        BAIEGSpatialProjectionSourceProvenanceV1, ...
    ]
    resolution_ceiling: str
    same_occurrence_deduplication_status: str = (
        "occurrence_membership_unavailable_after_record_aggregation_"
        "candidate_local_physical_precedence_fail_closed"
    )
    broad_bilateral_near_synchronous_phenotype_status: str = (
        "not_inferred_requires_external_record_level_temporal_evidence"
    )


@dataclass(frozen=True)
class BAIEGRecordSpatialResolutionProjectionOutputV1:
    source_input_batch_sha256: str
    source_implementation_id: str
    implementation_id: str
    ontology_sha256: str
    recording_ids: tuple[str, ...]
    laterality_candidate_ids: tuple[str, ...]
    laterality_candidate_logits: torch.Tensor
    laterality_candidate_opportunity_mask: torch.Tensor
    laterality_rankings: tuple[tuple[str, ...], ...]
    coarse_region_candidate_ids: tuple[str, ...]
    coarse_region_candidate_logits: torch.Tensor
    coarse_region_candidate_opportunity_mask: torch.Tensor
    coarse_region_rankings: tuple[tuple[str, ...], ...]
    joint_region_candidate_ids: tuple[str, ...]
    joint_region_candidate_logits: torch.Tensor
    joint_region_candidate_opportunity_mask: torch.Tensor
    joint_region_rankings: tuple[tuple[str, ...], ...]
    resolution_ceilings: tuple[str, ...]
    provenance: tuple[BAIEGRecordSpatialProjectionProvenanceV1, ...]
    output_semantics: str = (
        "record_level_uncalibrated_scalp_visible_onset_spatial_rankings_"
        "not_cortical_soz_ez_surgical_target_or_clinical_probability"
    )
    broad_bilateral_near_synchronous_phenotype_status: str = (
        "absent_by_design_requires_external_record_level_temporal_evidence"
    )


@dataclass(frozen=True)
class _TypedSource:
    logit: torch.Tensor
    provenance: BAIEGSpatialProjectionSourceProvenanceV1
    laterality: str
    coarse_region: str


class BAIEGRecordSpatialResolutionProjectionV1(nn.Module):
    """Frozen, deterministic projection of one record-bag output."""

    implementation_id: Final[str] = (
        BA_IEG_RECORD_SPATIAL_RESOLUTION_PROJECTION_ID
    )

    def __init__(
        self,
        *,
        temperature: float = 1.0,
        symmetric_logit_clip: float = 12.0,
    ) -> None:
        super().__init__()
        if temperature != 1.0 or symmetric_logit_clip != 12.0:
            raise ValueError(
                "BA-IEG v1 spatial projection is frozen at tau=1, clip=12"
            )
        self.temperature = float(temperature)
        self.symmetric_logit_clip = float(symmetric_logit_clip)

    def _validate_and_collect(
        self, source: BAIEGCappedLogMeanExpEventBagOutput
    ) -> tuple[tuple[tuple[_TypedSource, ...], tuple[_TypedSource, ...]], ...]:
        if not isinstance(source, BAIEGCappedLogMeanExpEventBagOutput):
            raise TypeError("spatial projection requires a BA-IEG record-bag output")
        if source.implementation_id != BA_IEG_CAPPED_LOG_MEAN_EXP_EVENT_BAG_ID:
            raise ValueError("spatial projection source implementation is unsupported")
        if source.temperature != 1.0 or source.symmetric_event_logit_clip != 12.0:
            raise ValueError("record-bag aggregation policy drifted from BA-IEG v1")
        if len(source.source_input_batch_sha256) != 64:
            raise ValueError("spatial projection requires a source batch SHA-256")
        records = tuple(source.recording_ids)
        if not records or len(set(records)) != len(records) or any(
            not item or item != item.strip() for item in records
        ):
            raise ValueError("spatial projection record identities must be unique")

        typed_logits = source.typed_unit_record_logits
        typed_mask = source.typed_unit_mask
        typed_kind = source.typed_unit_kind_index
        typed_electrode = source.typed_unit_electrode_index
        typed_lead = source.typed_unit_lead_endpoint_index
        physical_logits = source.physical_electrode_record_logits
        physical_mask = source.physical_electrode_mask
        record_count = len(records)
        if typed_logits.ndim != 2 or int(typed_logits.shape[0]) != record_count:
            raise ValueError("typed-unit record logits do not align with records")
        typed_shape = tuple(typed_logits.shape)
        if (
            tuple(typed_mask.shape) != typed_shape
            or tuple(typed_kind.shape) != typed_shape
            or tuple(typed_electrode.shape) != typed_shape
            or tuple(typed_lead.shape) != (*typed_shape, 2)
        ):
            raise ValueError("typed-unit record identity tensors are not aligned")
        if tuple(physical_logits.shape) != (record_count, len(STANDARD_19)) or (
            tuple(physical_mask.shape) != tuple(physical_logits.shape)
        ):
            raise ValueError("physical-electrode record tensors are not aligned")
        if typed_mask.dtype != torch.bool or physical_mask.dtype != torch.bool:
            raise ValueError("spatial opportunity masks must be boolean")
        if not typed_logits.is_floating_point() or not physical_logits.is_floating_point():
            raise ValueError("spatial ranking logits must be floating point")
        if typed_logits.device != physical_logits.device or (
            typed_logits.dtype != physical_logits.dtype
        ):
            raise ValueError("typed and physical record logits must share device/dtype")
        if any(
            tensor.device != typed_logits.device
            for tensor in (
                typed_mask,
                typed_kind,
                typed_electrode,
                typed_lead,
                physical_mask,
            )
        ):
            raise ValueError("record-bag spatial tensors must share one device")
        if not bool(torch.isfinite(typed_logits[typed_mask]).all()) or not bool(
            torch.isfinite(physical_logits[physical_mask]).all()
        ):
            raise ValueError("evaluable spatial ranking logits must be finite")

        electrode_kind = BA_IEG_CAUSAL_TYPED_UNIT_KINDS.index(
            "physical_electrode"
        )
        lead_kind = BA_IEG_CAUSAL_TYPED_UNIT_KINDS.index("bipolar_lead")
        collected: list[tuple[tuple[_TypedSource, ...], tuple[_TypedSource, ...]]] = []
        for record_index in range(record_count):
            physical_sources: list[_TypedSource] = []
            lead_sources: list[_TypedSource] = []
            reconstructed_mask = torch.zeros(
                len(STANDARD_19), dtype=torch.bool, device=typed_logits.device
            )
            reconstructed_logits = torch.zeros(
                len(STANDARD_19), dtype=typed_logits.dtype, device=typed_logits.device
            )
            seen: set[tuple[int, int, int]] = set()
            for typed_index_tensor in torch.nonzero(
                typed_mask[record_index], as_tuple=False
            ).flatten():
                typed_index = int(typed_index_tensor)
                kind = int(typed_kind[record_index, typed_index])
                if kind == electrode_kind:
                    electrode_index = int(
                        typed_electrode[record_index, typed_index]
                    )
                    if electrode_index < 0 or electrode_index >= len(STANDARD_19):
                        raise ValueError("physical typed unit has an invalid electrode")
                    if tuple(
                        int(item)
                        for item in typed_lead[record_index, typed_index].tolist()
                    ) != (-1, -1):
                        raise ValueError("physical typed unit must not carry lead endpoints")
                    key = (kind, electrode_index, -1)
                    if key in seen:
                        raise ValueError("record bag repeats one physical typed unit")
                    seen.add(key)
                    electrode_id = STANDARD_19[electrode_index]
                    provenance = BAIEGSpatialProjectionSourceProvenanceV1(
                        typed_unit_index=typed_index,
                        source_kind="physical_electrode",
                        physical_electrode_id=electrode_id,
                        whole_bipolar_lead_endpoint_ids=None,
                    )
                    physical_sources.append(
                        _TypedSource(
                            logit=typed_logits[record_index, typed_index],
                            provenance=provenance,
                            laterality=_electrode_laterality(electrode_id),
                            coarse_region=_electrode_coarse_region(electrode_id),
                        )
                    )
                    reconstructed_mask[electrode_index] = True
                    reconstructed_logits[electrode_index] = typed_logits[
                        record_index, typed_index
                    ]
                elif kind == lead_kind:
                    if int(typed_electrode[record_index, typed_index]) != -1:
                        raise ValueError("bipolar typed unit must not carry an electrode")
                    first, second = (
                        int(item)
                        for item in typed_lead[record_index, typed_index].tolist()
                    )
                    if (
                        first < 0
                        or second < 0
                        or first >= len(STANDARD_19)
                        or second >= len(STANDARD_19)
                        or first >= second
                    ):
                        raise ValueError("whole bipolar typed unit has invalid endpoints")
                    key = (kind, first, second)
                    if key in seen:
                        raise ValueError("record bag repeats one whole bipolar lead")
                    seen.add(key)
                    endpoint_ids = (STANDARD_19[first], STANDARD_19[second])
                    first_laterality = _electrode_laterality(endpoint_ids[0])
                    second_laterality = _electrode_laterality(endpoint_ids[1])
                    laterality = (
                        first_laterality
                        if first_laterality == second_laterality
                        else "cross_side_ineligible"
                    )
                    first_region = _electrode_coarse_region(endpoint_ids[0])
                    second_region = _electrode_coarse_region(endpoint_ids[1])
                    coarse_region = (
                        first_region
                        if laterality != "cross_side_ineligible"
                        and first_region == second_region
                        else "cross_region_or_side_ineligible"
                    )
                    provenance = BAIEGSpatialProjectionSourceProvenanceV1(
                        typed_unit_index=typed_index,
                        source_kind="whole_bipolar_lead",
                        physical_electrode_id=None,
                        whole_bipolar_lead_endpoint_ids=endpoint_ids,
                    )
                    lead_sources.append(
                        _TypedSource(
                            logit=typed_logits[record_index, typed_index],
                            provenance=provenance,
                            laterality=laterality,
                            coarse_region=coarse_region,
                        )
                    )
                else:
                    raise ValueError("record bag contains an unsupported typed-unit kind")
            if not torch.equal(reconstructed_mask, physical_mask[record_index]) or (
                not torch.equal(
                    reconstructed_logits[reconstructed_mask].detach(),
                    physical_logits[record_index, reconstructed_mask].detach(),
                )
            ):
                raise ValueError(
                    "record physical projection disagrees with typed-unit evidence"
                )
            collected.append((tuple(physical_sources), tuple(lead_sources)))
        return tuple(collected)

    def _aggregate_candidate(
        self,
        *,
        candidate_id: str,
        physical_sources: tuple[_TypedSource, ...],
        lead_sources: tuple[_TypedSource, ...],
        template: torch.Tensor,
    ) -> tuple[torch.Tensor, bool, BAIEGSpatialCandidateProvenanceV1]:
        # Candidate-local physical precedence is deliberately non-additive.
        if physical_sources:
            selected = physical_sources
            suppressed = lead_sources
            selected_kind = "physical_electrode"
        elif lead_sources:
            selected = lead_sources
            suppressed = ()
            selected_kind = "whole_bipolar_lead"
        else:
            selected = ()
            suppressed = ()
            selected_kind = "not_evaluable"
        if selected:
            values = torch.stack(tuple(item.logit for item in selected)).clamp(
                min=-self.symmetric_logit_clip,
                max=self.symmetric_logit_clip,
            ) / self.temperature
            logit = self.temperature * (
                torch.logsumexp(values, dim=0) - math.log(len(selected))
            )
        else:
            # This value is a tensor placeholder only; the opportunity mask is
            # authoritative and downstream code must not interpret it as zero
            # evidence or a negative candidate.
            logit = template.new_zeros(())
        provenance = BAIEGSpatialCandidateProvenanceV1(
            candidate_id=candidate_id,
            opportunity=bool(selected),
            selected_source_kind=selected_kind,
            selected_sources=tuple(item.provenance for item in selected),
            suppressed_bipolar_sources=tuple(
                item.provenance for item in suppressed
            ),
        )
        return logit, bool(selected), provenance

    def _axis(
        self,
        *,
        candidate_ids: tuple[str, ...],
        physical_sources: tuple[_TypedSource, ...],
        lead_sources: tuple[_TypedSource, ...],
        candidate_for_source,
        template: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        tuple[str, ...],
        tuple[BAIEGSpatialCandidateProvenanceV1, ...],
    ]:
        logits: list[torch.Tensor] = []
        opportunities: list[bool] = []
        provenance: list[BAIEGSpatialCandidateProvenanceV1] = []
        for candidate_id in candidate_ids:
            physical = tuple(
                item
                for item in physical_sources
                if candidate_for_source(item) == candidate_id
            )
            leads = tuple(
                item
                for item in lead_sources
                if candidate_for_source(item) == candidate_id
            )
            logit, opportunity, receipt = self._aggregate_candidate(
                candidate_id=candidate_id,
                physical_sources=physical,
                lead_sources=leads,
                template=template,
            )
            logits.append(logit)
            opportunities.append(opportunity)
            provenance.append(receipt)
        logit_tensor = torch.stack(logits)
        opportunity_tensor = torch.tensor(
            opportunities, dtype=torch.bool, device=template.device
        )
        evaluable_indices = torch.nonzero(
            opportunity_tensor, as_tuple=False
        ).flatten()
        if bool(evaluable_indices.numel()):
            ordering = torch.argsort(
                logit_tensor[evaluable_indices], descending=True, stable=True
            )
            ranking = tuple(
                candidate_ids[int(index)]
                for index in evaluable_indices[ordering].tolist()
            )
        else:
            ranking = ()
        return logit_tensor, opportunity_tensor, ranking, tuple(provenance)

    def forward(
        self, source: BAIEGCappedLogMeanExpEventBagOutput
    ) -> BAIEGRecordSpatialResolutionProjectionOutputV1:
        collected = self._validate_and_collect(source)
        laterality_logits: list[torch.Tensor] = []
        laterality_masks: list[torch.Tensor] = []
        laterality_rankings: list[tuple[str, ...]] = []
        region_logits: list[torch.Tensor] = []
        region_masks: list[torch.Tensor] = []
        region_rankings: list[tuple[str, ...]] = []
        joint_logits: list[torch.Tensor] = []
        joint_masks: list[torch.Tensor] = []
        joint_rankings: list[tuple[str, ...]] = []
        resolution_ceilings: list[str] = []
        record_provenance: list[BAIEGRecordSpatialProjectionProvenanceV1] = []

        for record_index, (physical_sources, lead_sources) in enumerate(collected):
            # Use the fixed-width physical tensor as the scalar template so a
            # valid [records, 0] typed-unit opportunity surface also fails
            # closed instead of indexing a synthetic padding unit.
            template = source.physical_electrode_record_logits[record_index, 0]
            (
                record_laterality_logits,
                record_laterality_mask,
                record_laterality_ranking,
                laterality_provenance,
            ) = self._axis(
                candidate_ids=BA_IEG_RECORD_SPATIAL_LATERALITY_IDS,
                physical_sources=physical_sources,
                lead_sources=lead_sources,
                candidate_for_source=lambda item: item.laterality,
                template=template,
            )
            (
                record_region_logits,
                record_region_mask,
                record_region_ranking,
                region_provenance,
            ) = self._axis(
                candidate_ids=BA_IEG_RECORD_SPATIAL_COARSE_REGION_IDS,
                physical_sources=physical_sources,
                lead_sources=lead_sources,
                candidate_for_source=lambda item: item.coarse_region,
                template=template,
            )
            (
                record_joint_logits,
                record_joint_mask,
                record_joint_ranking,
                joint_provenance,
            ) = self._axis(
                candidate_ids=BA_IEG_RECORD_SPATIAL_JOINT_REGION_IDS,
                physical_sources=physical_sources,
                lead_sources=lead_sources,
                candidate_for_source=(
                    lambda item: (
                        f"{item.laterality}_{item.coarse_region}"
                        if item.laterality in BA_IEG_RECORD_SPATIAL_LATERALITY_IDS
                        and item.coarse_region
                        in BA_IEG_RECORD_SPATIAL_COARSE_REGION_IDS
                        else "ineligible"
                    )
                ),
                template=template,
            )
            cross_side = tuple(
                item.provenance
                for item in lead_sources
                if item.laterality == "cross_side_ineligible"
            )
            same_side_cross_region = tuple(
                item.provenance
                for item in lead_sources
                if item.laterality in BA_IEG_RECORD_SPATIAL_LATERALITY_IDS
                and item.coarse_region == "cross_region_or_side_ineligible"
            )
            if physical_sources:
                resolution_ceiling = "physical_electrode_candidate"
            elif bool(record_joint_mask.any()):
                resolution_ceiling = "joint_laterality_coarse_scalp_region"
            elif bool(record_laterality_mask.any()):
                resolution_ceiling = "laterality_only"
            else:
                resolution_ceiling = "not_evaluable"

            laterality_logits.append(record_laterality_logits)
            laterality_masks.append(record_laterality_mask)
            laterality_rankings.append(record_laterality_ranking)
            region_logits.append(record_region_logits)
            region_masks.append(record_region_mask)
            region_rankings.append(record_region_ranking)
            joint_logits.append(record_joint_logits)
            joint_masks.append(record_joint_mask)
            joint_rankings.append(record_joint_ranking)
            resolution_ceilings.append(resolution_ceiling)
            record_provenance.append(
                BAIEGRecordSpatialProjectionProvenanceV1(
                    recording_id=source.recording_ids[record_index],
                    physical_electrode_sources=tuple(
                        item.provenance for item in physical_sources
                    ),
                    whole_bipolar_lead_sources=tuple(
                        item.provenance for item in lead_sources
                    ),
                    laterality_candidates=laterality_provenance,
                    coarse_region_candidates=region_provenance,
                    joint_region_candidates=joint_provenance,
                    cross_side_leads_excluded_from_both_axes=cross_side,
                    same_side_cross_region_leads_laterality_only=(
                        same_side_cross_region
                    ),
                    resolution_ceiling=resolution_ceiling,
                )
            )

        return BAIEGRecordSpatialResolutionProjectionOutputV1(
            source_input_batch_sha256=source.source_input_batch_sha256,
            source_implementation_id=source.implementation_id,
            implementation_id=self.implementation_id,
            ontology_sha256=(
                BA_IEG_RECORD_SPATIAL_RESOLUTION_ONTOLOGY_SHA256
            ),
            recording_ids=source.recording_ids,
            laterality_candidate_ids=BA_IEG_RECORD_SPATIAL_LATERALITY_IDS,
            laterality_candidate_logits=torch.stack(laterality_logits),
            laterality_candidate_opportunity_mask=torch.stack(laterality_masks),
            laterality_rankings=tuple(laterality_rankings),
            coarse_region_candidate_ids=BA_IEG_RECORD_SPATIAL_COARSE_REGION_IDS,
            coarse_region_candidate_logits=torch.stack(region_logits),
            coarse_region_candidate_opportunity_mask=torch.stack(region_masks),
            coarse_region_rankings=tuple(region_rankings),
            joint_region_candidate_ids=BA_IEG_RECORD_SPATIAL_JOINT_REGION_IDS,
            joint_region_candidate_logits=torch.stack(joint_logits),
            joint_region_candidate_opportunity_mask=torch.stack(joint_masks),
            joint_region_rankings=tuple(joint_rankings),
            resolution_ceilings=tuple(resolution_ceilings),
            provenance=tuple(record_provenance),
        )


__all__ = [
    "BA_IEG_RECORD_SPATIAL_RESOLUTION_PROJECTION_ID",
    "BA_IEG_RECORD_SPATIAL_RESOLUTION_SCHEMA_VERSION",
    "BA_IEG_RECORD_SPATIAL_LATERALITY_IDS",
    "BA_IEG_RECORD_SPATIAL_COARSE_REGION_IDS",
    "BA_IEG_RECORD_SPATIAL_JOINT_REGION_IDS",
    "BA_IEG_RECORD_SPATIAL_RESOLUTION_ONTOLOGY_V1",
    "BA_IEG_RECORD_SPATIAL_RESOLUTION_ONTOLOGY_SHA256",
    "BAIEGRecordSpatialResolutionOntologyV1",
    "BAIEGSpatialProjectionSourceProvenanceV1",
    "BAIEGSpatialCandidateProvenanceV1",
    "BAIEGRecordSpatialProjectionProvenanceV1",
    "BAIEGRecordSpatialResolutionProjectionOutputV1",
    "BAIEGRecordSpatialResolutionProjectionV1",
]
