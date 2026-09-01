"""Uncertain-only forced-choice research hypotheses for BA-IEG.

This is deliberately *not* the evidence-grade event bag.  It runs only for a
record with zero qualified occurrences and at least one evaluable uncertain
occurrence.  Exact copies are deduplicated, uncertain event logits receive
equal occurrence weight under the frozen capped log-mean-exp rule, and the
result is exposed only as an uncalibrated low-confidence top-k hypothesis.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final

import torch
from torch import nn

from src.soz.geometry import STANDARD_19

from .ba_ieg_capped_log_mean_exp_event_bag_v1 import (
    BAIEGRecordEventBagManifest,
)
from .ba_ieg_permission_split_segmental_state_model_v1 import (
    BA_IEG_CAUSAL_TYPED_UNIT_KINDS,
)
from .ba_ieg_shallow_causal_typed_unit_head_v1 import (
    BAIEGShallowCausalTypedUnitHeadOutput,
)


BA_IEG_FORCED_CHOICE_RESEARCH_HYPOTHESIS_ID_V1: Final[str] = (
    "ba_ieg_uncertain_event_forced_choice_research_hypothesis_v1"
)
BA_IEG_FORCED_CHOICE_RECORD_STATUSES: Final[tuple[str, ...]] = (
    "ranked_uncalibrated_low_confidence_hypothesis",
    "suppressed_qualified_evidence_available",
    "not_evaluable_no_ranking",
    "uncertain_occurrences_no_typed_unit_hypothesis",
)


@dataclass(frozen=True)
class BAIEGForcedChoiceResearchHypothesisOutputV1:
    source_input_batch_sha256: str
    implementation_id: str
    recording_ids: tuple[str, ...]
    record_hypothesis_statuses: tuple[str, ...]
    physical_electrode_hypothesis_scores: torch.Tensor
    physical_electrode_hypothesis_opportunity_mask: torch.Tensor
    ranked_physical_electrode_indices: torch.Tensor
    ranked_physical_electrode_ids: tuple[tuple[str, ...], ...]
    ranked_hypothesis_scores: torch.Tensor
    ranked_bipolar_lead_endpoint_indices: torch.Tensor
    ranked_bipolar_lead_ids: tuple[tuple[str, ...], ...]
    ranked_bipolar_hypothesis_scores: torch.Tensor
    bipolar_hypothesis_candidate_count: torch.Tensor
    unique_occurrence_count: torch.Tensor
    qualified_unique_occurrence_count: torch.Tensor
    uncertain_unique_occurrence_count: torch.Tensor
    not_evaluable_unique_occurrence_count: torch.Tensor
    copied_event_count: torch.Tensor
    forced_choice_triggered: torch.Tensor
    top_k: int
    temperature: float
    symmetric_event_logit_clip: float
    event_qualification_receipt_sha256s: tuple[str, ...]
    production_qualification_receipts: bool
    component_test_only: bool
    output_semantics: str = (
        "uncalibrated_low_confidence_forced_choice_research_hypothesis_"
        "not_evidence_grade_not_probability_not_clinical_diagnosis"
    )
    evidence_grade_bag_member: bool = False
    outputs_are_probabilities: bool = False
    report_generation_authorized: bool = False
    bipolar_unit_semantics: str = (
        "whole_bipolar_lead_identity_without_endpoint_attribution"
    )
    bipolar_endpoint_attribution_authorized: bool = False


class BAIEGForcedChoiceResearchHypothesisAggregatorV1(nn.Module):
    """Aggregate only uncertain occurrences into an isolated top-k hypothesis."""

    implementation_id: Final[str] = (
        BA_IEG_FORCED_CHOICE_RESEARCH_HYPOTHESIS_ID_V1
    )

    def __init__(
        self,
        *,
        top_k: int = 5,
        temperature: float = 1.0,
        symmetric_event_logit_clip: float = 12.0,
        allow_component_test_qualification: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("forced-choice top-k must be an integer")
        if top_k < 1 or top_k > len(STANDARD_19):
            raise ValueError("forced-choice top-k is outside the electrode roster")
        if temperature != 1.0 or symmetric_event_logit_clip != 12.0:
            raise ValueError("forced-choice v1 is frozen at tau=1 and clip=12")
        self.top_k = top_k
        self.temperature = float(temperature)
        self.symmetric_event_logit_clip = float(symmetric_event_logit_clip)
        if not isinstance(allow_component_test_qualification, bool):
            raise TypeError("component-test qualification gate must be boolean")
        self.allow_component_test_qualification = allow_component_test_qualification

    def forward(
        self,
        event_output: BAIEGShallowCausalTypedUnitHeadOutput,
        manifest: BAIEGRecordEventBagManifest,
    ) -> BAIEGForcedChoiceResearchHypothesisOutputV1:
        if not isinstance(event_output, BAIEGShallowCausalTypedUnitHeadOutput):
            raise TypeError("forced-choice aggregation requires typed-head output")
        if not isinstance(manifest, BAIEGRecordEventBagManifest):
            raise TypeError("forced-choice aggregation requires an event-bag manifest")
        if not manifest.qualification_provider_validated:
            raise ValueError(
                "forced-choice requires content-bound target-free qualification receipts"
            )
        if (
            not manifest.evidence_grade_aggregation_authorized
            and not self.allow_component_test_qualification
        ):
            raise ValueError(
                "non-production qualification receipts cannot enter formal "
                "forced-choice aggregation"
            )
        if event_output.source_input_batch_sha256 != manifest.source_input_batch_sha256:
            raise ValueError("forced-choice manifest is bound to another model batch")
        if (
            event_output.event_ids != manifest.event_ids
            or event_output.recording_ids != manifest.recording_ids
            or event_output.source_event_receipt_sha256s
            != manifest.source_event_receipt_sha256s
            or event_output.identity_roster_sha256
            != manifest.identity_roster_sha256
        ):
            raise ValueError("forced-choice manifest identity/order roster drifted")

        physical_logits = event_output.physical_electrode_event_logits
        physical_mask = event_output.physical_electrode_mask
        typed_logits = event_output.typed_unit_event_logits
        typed_mask = event_output.typed_unit_mask
        typed_kind = event_output.typed_unit_kind_index
        typed_electrode = event_output.typed_unit_electrode_index
        typed_lead = event_output.typed_unit_lead_endpoint_index
        event_count = len(manifest.event_ids)
        if tuple(physical_logits.shape) != (event_count, len(STANDARD_19)):
            raise ValueError("forced-choice physical logits do not align")
        if tuple(physical_mask.shape) != tuple(physical_logits.shape):
            raise ValueError("forced-choice physical opportunity mask does not align")
        if typed_logits.ndim != 2 or int(typed_logits.shape[0]) != event_count:
            raise ValueError("forced-choice typed logits do not align")
        maximum_typed = int(typed_logits.shape[1])
        if (
            tuple(typed_mask.shape) != tuple(typed_logits.shape)
            or tuple(typed_kind.shape) != tuple(typed_logits.shape)
            or tuple(typed_electrode.shape) != tuple(typed_logits.shape)
            or tuple(typed_lead.shape) != (event_count, maximum_typed, 2)
        ):
            raise ValueError("forced-choice typed identity tensors do not align")
        if physical_mask.dtype != torch.bool or typed_mask.dtype != torch.bool:
            raise ValueError("forced-choice opportunity masks must be boolean")
        if (
            not physical_logits.is_floating_point()
            or not typed_logits.is_floating_point()
            or not torch.isfinite(physical_logits).all()
            or not torch.isfinite(typed_logits).all()
        ):
            raise ValueError("forced-choice logits must be finite floating point")

        records = tuple(sorted(set(manifest.recording_ids)))
        record_count = len(records)
        scores = torch.zeros(
            (record_count, len(STANDARD_19)),
            dtype=physical_logits.dtype,
            device=physical_logits.device,
        )
        hypothesis_mask = torch.zeros_like(scores, dtype=torch.bool)
        ranked_indices = torch.full(
            (record_count, self.top_k),
            -1,
            dtype=torch.long,
            device=physical_logits.device,
        )
        ranked_scores = torch.zeros_like(ranked_indices, dtype=physical_logits.dtype)
        ranked_bipolar_endpoints = torch.full(
            (record_count, self.top_k, 2),
            -1,
            dtype=torch.long,
            device=physical_logits.device,
        )
        ranked_bipolar_scores = torch.zeros(
            (record_count, self.top_k),
            dtype=typed_logits.dtype,
            device=typed_logits.device,
        )
        bipolar_candidate_counts = torch.zeros(
            record_count, dtype=torch.long, device=typed_logits.device
        )
        unique_counts = torch.zeros(
            record_count, dtype=torch.long, device=physical_logits.device
        )
        qualified_counts = torch.zeros_like(unique_counts)
        uncertain_counts = torch.zeros_like(unique_counts)
        not_evaluable_counts = torch.zeros_like(unique_counts)
        copied_counts = torch.zeros_like(unique_counts)
        triggered = torch.zeros(
            record_count, dtype=torch.bool, device=physical_logits.device
        )
        statuses: list[str] = []
        ranked_ids: list[tuple[str, ...]] = []
        ranked_bipolar_ids: list[tuple[str, ...]] = []
        bipolar_kind = BA_IEG_CAUSAL_TYPED_UNIT_KINDS.index("bipolar_lead")

        for record_index, recording_id in enumerate(records):
            groups: dict[str, list[int]] = {}
            for event_index, (record, occurrence) in enumerate(
                zip(manifest.recording_ids, manifest.occurrence_equivalence_ids)
            ):
                if record == recording_id:
                    groups.setdefault(occurrence, []).append(event_index)
            representatives: list[int] = []
            for occurrence in sorted(groups):
                indices = sorted(groups[occurrence], key=lambda i: manifest.event_ids[i])
                representative = indices[0]
                for duplicate in indices[1:]:
                    tensors_match = (
                        torch.equal(typed_mask[duplicate], typed_mask[representative])
                        and torch.equal(typed_kind[duplicate], typed_kind[representative])
                        and torch.equal(
                            typed_electrode[duplicate], typed_electrode[representative]
                        )
                        and torch.equal(typed_lead[duplicate], typed_lead[representative])
                        and torch.equal(
                            typed_logits[duplicate].detach(),
                            typed_logits[representative].detach(),
                        )
                        and torch.equal(
                            physical_mask[duplicate], physical_mask[representative]
                        )
                        and torch.equal(
                            physical_logits[duplicate].detach(),
                            physical_logits[representative].detach(),
                        )
                    )
                    if not tensors_match:
                        raise ValueError(
                            "one occurrence equivalence class has non-identical "
                            "forced-choice evidence"
                        )
                representatives.append(representative)
                copied_counts[record_index] += len(indices) - 1

            unique_counts[record_index] = len(representatives)
            qualified = [
                index
                for index in representatives
                if manifest.event_aggregation_statuses[index] == "qualified_ictal"
            ]
            uncertain = [
                index
                for index in representatives
                if manifest.event_aggregation_statuses[index] == "uncertain"
            ]
            not_evaluable = [
                index
                for index in representatives
                if manifest.event_aggregation_statuses[index] == "not_evaluable"
            ]
            qualified_counts[record_index] = len(qualified)
            uncertain_counts[record_index] = len(uncertain)
            not_evaluable_counts[record_index] = len(not_evaluable)

            if qualified:
                statuses.append("suppressed_qualified_evidence_available")
                ranked_ids.append(())
                ranked_bipolar_ids.append(())
                continue
            if not uncertain:
                statuses.append("not_evaluable_no_ranking")
                ranked_ids.append(())
                ranked_bipolar_ids.append(())
                continue

            for electrode_index in range(len(STANDARD_19)):
                values = [
                    physical_logits[event_index, electrode_index]
                    for event_index in uncertain
                    if bool(physical_mask[event_index, electrode_index])
                ]
                if not values:
                    continue
                event_values = torch.stack(values).clamp(
                    min=-self.symmetric_event_logit_clip,
                    max=self.symmetric_event_logit_clip,
                ) / self.temperature
                aggregate = self.temperature * (
                    torch.logsumexp(event_values, dim=0) - math.log(len(values))
                )
                scores[record_index, electrode_index] = aggregate
                hypothesis_mask[record_index, electrode_index] = True

            bipolar_values: dict[tuple[int, int], list[torch.Tensor]] = {}
            for event_index in uncertain:
                event_leads: set[tuple[int, int]] = set()
                for typed_index_tensor in torch.nonzero(
                    typed_mask[event_index], as_tuple=False
                ).flatten():
                    typed_index = int(typed_index_tensor)
                    if int(typed_kind[event_index, typed_index]) != bipolar_kind:
                        continue
                    first, second = (
                        int(value)
                        for value in typed_lead[event_index, typed_index]
                    )
                    if (
                        first < 0
                        or second < 0
                        or first >= len(STANDARD_19)
                        or second >= len(STANDARD_19)
                        or first >= second
                    ):
                        raise ValueError(
                            "forced-choice bipolar lead has invalid whole-lead endpoints"
                        )
                    lead = (first, second)
                    if lead in event_leads:
                        raise ValueError(
                            "forced-choice event repeats one whole bipolar lead"
                        )
                    event_leads.add(lead)
                    bipolar_values.setdefault(lead, []).append(
                        typed_logits[event_index, typed_index]
                    )

            bipolar_aggregates: dict[tuple[int, int], torch.Tensor] = {}
            for lead, values in bipolar_values.items():
                event_values = torch.stack(values).clamp(
                    min=-self.symmetric_event_logit_clip,
                    max=self.symmetric_event_logit_clip,
                ) / self.temperature
                bipolar_aggregates[lead] = self.temperature * (
                    torch.logsumexp(event_values, dim=0) - math.log(len(values))
                )
            bipolar_candidate_counts[record_index] = len(bipolar_aggregates)
            ordered_bipolar = sorted(
                bipolar_aggregates,
                key=lambda lead: (
                    -float(bipolar_aggregates[lead].detach().cpu()),
                    lead,
                ),
            )[: self.top_k]
            for rank, lead in enumerate(ordered_bipolar):
                ranked_bipolar_endpoints[record_index, rank] = torch.tensor(
                    lead, dtype=torch.long, device=typed_logits.device
                )
                ranked_bipolar_scores[record_index, rank] = bipolar_aggregates[lead]
            ranked_bipolar_ids.append(
                tuple(
                    f"{STANDARD_19[first]}-{STANDARD_19[second]}"
                    for first, second in ordered_bipolar
                )
            )

            eligible = torch.nonzero(
                hypothesis_mask[record_index], as_tuple=False
            ).flatten().tolist()
            if not eligible and not ordered_bipolar:
                statuses.append(
                    "uncertain_occurrences_no_typed_unit_hypothesis"
                )
                ranked_ids.append(())
                continue
            ordered = sorted(
                eligible,
                key=lambda index: (
                    -float(scores[record_index, index].detach().cpu()),
                    int(index),
                ),
            )[: self.top_k]
            for rank, electrode_index in enumerate(ordered):
                ranked_indices[record_index, rank] = electrode_index
                ranked_scores[record_index, rank] = scores[
                    record_index, electrode_index
                ]
            statuses.append("ranked_uncalibrated_low_confidence_hypothesis")
            ranked_ids.append(tuple(STANDARD_19[index] for index in ordered))
            triggered[record_index] = True

        if any(status not in BA_IEG_FORCED_CHOICE_RECORD_STATUSES for status in statuses):
            raise RuntimeError("forced-choice emitted an unsupported record status")
        return BAIEGForcedChoiceResearchHypothesisOutputV1(
            source_input_batch_sha256=manifest.source_input_batch_sha256,
            implementation_id=self.implementation_id,
            recording_ids=records,
            record_hypothesis_statuses=tuple(statuses),
            physical_electrode_hypothesis_scores=scores,
            physical_electrode_hypothesis_opportunity_mask=hypothesis_mask,
            ranked_physical_electrode_indices=ranked_indices,
            ranked_physical_electrode_ids=tuple(ranked_ids),
            ranked_hypothesis_scores=ranked_scores,
            ranked_bipolar_lead_endpoint_indices=ranked_bipolar_endpoints,
            ranked_bipolar_lead_ids=tuple(ranked_bipolar_ids),
            ranked_bipolar_hypothesis_scores=ranked_bipolar_scores,
            bipolar_hypothesis_candidate_count=bipolar_candidate_counts,
            unique_occurrence_count=unique_counts,
            qualified_unique_occurrence_count=qualified_counts,
            uncertain_unique_occurrence_count=uncertain_counts,
            not_evaluable_unique_occurrence_count=not_evaluable_counts,
            copied_event_count=copied_counts,
            forced_choice_triggered=triggered,
            top_k=self.top_k,
            temperature=self.temperature,
            symmetric_event_logit_clip=self.symmetric_event_logit_clip,
            event_qualification_receipt_sha256s=(
                manifest.event_qualification_receipt_sha256s
            ),
            production_qualification_receipts=(
                manifest.evidence_grade_aggregation_authorized
            ),
            component_test_only=(
                not manifest.evidence_grade_aggregation_authorized
            ),
        )


__all__ = [
    "BA_IEG_FORCED_CHOICE_RESEARCH_HYPOTHESIS_ID_V1",
    "BA_IEG_FORCED_CHOICE_RECORD_STATUSES",
    "BAIEGForcedChoiceResearchHypothesisOutputV1",
    "BAIEGForcedChoiceResearchHypothesisAggregatorV1",
]
