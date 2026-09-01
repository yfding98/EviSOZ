"""Score-preserving MRSC diagnostics for the frozen v11.1 localizer.

MRSC (multi-reference selective clinical-evidence layer) is an inference-only
diagnostic boundary.  It consumes the already-produced C-CAR19 anchor scores,
the same frozen localizer's paired C-REF19 sensitivity scores, event-level
C-CAR19 scores, and target-free validity/fact masks.  It neither trains nor
reranks the localizer.

The returned ``nonconformity`` is deliberately an *uncalibrated raw*
maximum of the localization-relevant uncertainty components.  It is not a
conformal p-value, an error probability, or an accept/reject threshold.  This
module accepts no threshold and therefore always emits the fail-closed
``selective_threshold_undefined`` abstention reason.  A future threshold may
only be selected outside this module on a new, patient-disjoint clinical
calibration cohort.

Report-fact availability is reported separately and never enters the
nonconformity value.  This prevents a missing prose field from changing an
SOZ ranking or being treated as evidence that the ranking is wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final, Mapping

import torch

from .geometry import STANDARD_19
from .v11_reasoner import V11_CANDIDATE_INDICES


MRSC_SCHEMA: Final[str] = "soz_labram_mrsc_score_preserving_v1"
MRSC_NONCONFORMITY_SEMANTICS: Final[str] = (
    "uncalibrated_max_raw_localization_uncertainty_component_v1"
)
MRSC_USE_POLICY: Final[str] = (
    "uncertainty_review_and_future_external_calibration_only_not_reranking"
)
MRSC_CANDIDATE_CHANNELS: Final[tuple[str, ...]] = tuple(
    STANDARD_19[index] for index in V11_CANDIDATE_INDICES
)
MRSC_N_CANDIDATES: Final[int] = len(MRSC_CANDIDATE_CHANNELS)

# Closed vocabulary mirrors the typed event-report fields.  Values indicate
# only whether a separately qualified producer supplied the field.  They are
# not labels and cannot alter localization uncertainty or scores.
MRSC_REPORT_FACT_FIELDS: Final[tuple[str, ...]] = (
    "sustained_change_interval",
    "first_visible_derivations",
    "rhythm_state",
    "frequency_range_hz",
    "later_visible_delay",
    "later_visible_destination",
    "montage_stability",
    "artifact_assessment",
)

_LOG_TWO = math.log(2.0)


def _validate_score_tensor(
    value: torch.Tensor,
    *,
    name: str,
    shape: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {list(shape)}")
    if value.requires_grad:
        raise ValueError(f"{name} must be detached inference output")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value.detach().clone()


def _validate_quality_mask(
    value: torch.Tensor,
    *,
    event_count: int,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.dtype != torch.bool:
        raise TypeError("event_quality_valid_mask must be a torch.bool tensor")
    expected = (event_count, MRSC_N_CANDIDATES)
    if tuple(value.shape) != expected:
        raise ValueError(
            "event_quality_valid_mask must match event scores with shape "
            f"{list(expected)}"
        )
    if value.requires_grad:
        raise ValueError("event_quality_valid_mask must be detached")
    return value.detach().clone()


def _validate_fact_mask(
    value: Mapping[str, bool],
) -> tuple[tuple[str, bool], ...]:
    if not isinstance(value, Mapping):
        raise TypeError("report_fact_available_mask must be a mapping")
    keys = tuple(value.keys())
    if set(keys) != set(MRSC_REPORT_FACT_FIELDS) or len(keys) != len(
        MRSC_REPORT_FACT_FIELDS
    ):
        raise ValueError(
            "report_fact_available_mask must contain exactly the frozen MRSC "
            "report-fact fields"
        )
    rows: list[tuple[str, bool]] = []
    for name in MRSC_REPORT_FACT_FIELDS:
        available = value[name]
        if type(available) is not bool:
            raise TypeError(f"report fact mask {name!r} must be bool")
        rows.append((name, available))
    return tuple(rows)


def _probabilities(scores: torch.Tensor) -> torch.Tensor:
    # Float64 makes repeated CPU/GPU inputs converge to one stable diagnostic
    # calculation while the original score tensor remains untouched.
    return torch.softmax(scores.detach().to(device="cpu", dtype=torch.float64), dim=-1)


def _normalized_jsd(left: torch.Tensor, right: torch.Tensor) -> float:
    midpoint = 0.5 * (left + right)
    value = 0.5 * torch.sum(left * (torch.log(left) - torch.log(midpoint)))
    value += 0.5 * torch.sum(right * (torch.log(right) - torch.log(midpoint)))
    normalized = float((value / _LOG_TWO).item())
    # Numerical round-off can stray by a few ulps from the analytical range.
    return min(1.0, max(0.0, normalized))


def _stable_rank(scores: torch.Tensor) -> tuple[int, ...]:
    values = scores.detach().to(device="cpu", dtype=torch.float64).tolist()
    return tuple(sorted(range(len(values)), key=lambda index: (-values[index], index)))


def _top_tied(scores: torch.Tensor) -> tuple[int, ...]:
    values = scores.detach().to(device="cpu", dtype=torch.float64)
    maximum = torch.max(values)
    return tuple(torch.nonzero(values == maximum).flatten().tolist())


@dataclass(frozen=True)
class MRSCUncertaintyComponents:
    """Separately named, target-free descriptive uncertainty components."""

    ranking_ambiguity: float
    within_patient_event_dispersion: float | None
    final_score_reference_disagreement: float
    signal_quality_uncertainty: float
    report_fact_unavailability: float

    def __post_init__(self) -> None:
        for name in (
            "ranking_ambiguity",
            "final_score_reference_disagreement",
            "signal_quality_uncertainty",
            "report_fact_unavailability",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must lie in [0,1]")
        dispersion = self.within_patient_event_dispersion
        if dispersion is not None and (
            not isinstance(dispersion, (int, float))
            or not math.isfinite(float(dispersion))
            or not 0.0 <= float(dispersion) <= 1.0
        ):
            raise ValueError("within_patient_event_dispersion must lie in [0,1]")


@dataclass(frozen=True)
class MRSCAssessment:
    """One immutable score-parity and uncertainty assessment.

    ``anchor_scores`` and ``sensitivity_scores`` are detached clones of the
    inputs.  No averaging, masking, voting, swapping, or score correction is
    performed.  ``anchor_top1_*`` is therefore exactly the frozen C-CAR19
    decision (with PyTorch's first-index rule retained for an exact tie).
    """

    anchor_scores: torch.Tensor
    sensitivity_scores: torch.Tensor
    anchor_top1_index: int
    anchor_top1_channel: str
    sensitivity_top1_index: int
    sensitivity_top1_channel: str
    top1_reference_agreement: bool
    top3_reference_jaccard: float
    components: MRSCUncertaintyComponents
    nonconformity: float
    report_fact_available_mask: tuple[tuple[str, bool], ...]
    abstain: bool
    abstention_reason_codes: tuple[str, ...]
    review_reason_codes: tuple[str, ...]
    nonconformity_semantics: str = MRSC_NONCONFORMITY_SEMANTICS
    use_policy: str = MRSC_USE_POLICY
    schema_version: str = MRSC_SCHEMA

    def __post_init__(self) -> None:
        for name in ("anchor_scores", "sensitivity_scores"):
            value = getattr(self, name)
            if (
                not isinstance(value, torch.Tensor)
                or not value.is_floating_point()
                or tuple(value.shape) != (MRSC_N_CANDIDATES,)
                or value.requires_grad
                or not torch.isfinite(value).all()
            ):
                raise ValueError(f"{name} must be detached finite [18]")
        for index_name, channel_name, scores in (
            ("anchor_top1_index", "anchor_top1_channel", self.anchor_scores),
            (
                "sensitivity_top1_index",
                "sensitivity_top1_channel",
                self.sensitivity_scores,
            ),
        ):
            index = getattr(self, index_name)
            if type(index) is not int or not 0 <= index < MRSC_N_CANDIDATES:
                raise ValueError(f"{index_name} is invalid")
            if getattr(self, channel_name) != MRSC_CANDIDATE_CHANNELS[index]:
                raise ValueError(f"{channel_name} does not match its index")
            if index != int(torch.argmax(scores).item()):
                raise ValueError(f"{index_name} does not replay from preserved scores")
        if type(self.top1_reference_agreement) is not bool:
            raise TypeError("top1_reference_agreement must be bool")
        if self.top1_reference_agreement != (
            self.anchor_top1_index == self.sensitivity_top1_index
        ):
            raise ValueError("top1_reference_agreement does not replay")
        for name, value in (
            ("top3_reference_jaccard", self.top3_reference_jaccard),
            ("nonconformity", self.nonconformity),
        ):
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"{name} must lie in [0,1]")
        if not isinstance(self.components, MRSCUncertaintyComponents):
            raise TypeError("components must be MRSCUncertaintyComponents")
        localization_components = [
            self.components.ranking_ambiguity,
            self.components.final_score_reference_disagreement,
            self.components.signal_quality_uncertainty,
        ]
        if self.components.within_patient_event_dispersion is not None:
            localization_components.append(
                self.components.within_patient_event_dispersion
            )
        if not math.isclose(
            float(self.nonconformity),
            max(localization_components),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("nonconformity does not replay from its components")
        if (
            self.report_fact_available_mask
            != tuple(
                (name, dict(self.report_fact_available_mask)[name])
                for name in MRSC_REPORT_FACT_FIELDS
            )
        ):
            raise ValueError("report fact mask is incomplete or out of frozen order")
        for _, available in self.report_fact_available_mask:
            if type(available) is not bool:
                raise TypeError("report fact availability values must be bool")
        if type(self.abstain) is not bool or not self.abstain:
            raise ValueError("Uncalibrated MRSC core must fail closed with abstain=True")
        for name, values in (
            ("abstention_reason_codes", self.abstention_reason_codes),
            ("review_reason_codes", self.review_reason_codes),
        ):
            if (
                not isinstance(values, tuple)
                or len(set(values)) != len(values)
                or any(not isinstance(value, str) or not value for value in values)
            ):
                raise ValueError(f"{name} must contain unique stable tokens")
        if "selective_threshold_undefined" not in self.abstention_reason_codes:
            raise ValueError("Uncalibrated MRSC must declare its missing threshold")
        if self.nonconformity_semantics != MRSC_NONCONFORMITY_SEMANTICS:
            raise ValueError("Unsupported MRSC nonconformity semantics")
        if self.use_policy != MRSC_USE_POLICY:
            raise ValueError("Unsupported MRSC use policy")
        if self.schema_version != MRSC_SCHEMA:
            raise ValueError("Unsupported MRSC schema")


def assess_mrsc_score_preserving(
    car_patient_scores: torch.Tensor,
    ref_patient_scores: torch.Tensor,
    car_event_scores: torch.Tensor,
    event_quality_valid_mask: torch.Tensor,
    report_fact_available_mask: Mapping[str, bool],
) -> MRSCAssessment:
    """Compute target-free MRSC diagnostics without changing anchor scores.

    Shapes are ``[18]`` for each patient score vector, ``[E,18]`` for event
    scores, and ``[E,18]`` for the target-free quality-validity mask.  The
    function intentionally has no target, patient identity, threshold,
    calibrator, model, optimizer, or private-data argument.
    """

    anchor = _validate_score_tensor(
        car_patient_scores,
        name="car_patient_scores",
        shape=(MRSC_N_CANDIDATES,),
    )
    sensitivity = _validate_score_tensor(
        ref_patient_scores,
        name="ref_patient_scores",
        shape=(MRSC_N_CANDIDATES,),
    )
    if not isinstance(car_event_scores, torch.Tensor) or car_event_scores.ndim != 2:
        raise ValueError("car_event_scores must be a rank-2 tensor")
    event_count = int(car_event_scores.shape[0])
    if event_count < 1:
        raise ValueError("car_event_scores must contain a non-empty patient bag")
    events = _validate_score_tensor(
        car_event_scores,
        name="car_event_scores",
        shape=(event_count, MRSC_N_CANDIDATES),
    )
    quality = _validate_quality_mask(
        event_quality_valid_mask,
        event_count=event_count,
    )
    facts = _validate_fact_mask(report_fact_available_mask)

    anchor_probability = _probabilities(anchor)
    sensitivity_probability = _probabilities(sensitivity)
    top_two = torch.topk(anchor_probability, k=2, largest=True, sorted=True).values
    ranking_ambiguity = 1.0 - float((top_two[0] - top_two[1]).item())
    ranking_ambiguity = min(1.0, max(0.0, ranking_ambiguity))

    event_dispersion: float | None
    if event_count == 1:
        event_dispersion = None
    else:
        event_probabilities = _probabilities(events)
        event_dispersion = math.fsum(
            _normalized_jsd(row, anchor_probability) for row in event_probabilities
        ) / event_count

    reference_disagreement = _normalized_jsd(
        anchor_probability,
        sensitivity_probability,
    )
    signal_quality_uncertainty = 1.0 - float(quality.double().mean().item())
    fact_unavailability = 1.0 - (
        sum(int(available) for _, available in facts) / len(facts)
    )
    components = MRSCUncertaintyComponents(
        ranking_ambiguity=ranking_ambiguity,
        within_patient_event_dispersion=event_dispersion,
        final_score_reference_disagreement=reference_disagreement,
        signal_quality_uncertainty=signal_quality_uncertainty,
        report_fact_unavailability=fact_unavailability,
    )

    localization_components = [
        ranking_ambiguity,
        reference_disagreement,
        signal_quality_uncertainty,
    ]
    if event_dispersion is not None:
        localization_components.append(event_dispersion)
    nonconformity = max(localization_components)

    anchor_rank = _stable_rank(anchor)
    sensitivity_rank = _stable_rank(sensitivity)
    anchor_top1 = anchor_rank[0]
    sensitivity_top1 = sensitivity_rank[0]
    top3_anchor = frozenset(anchor_rank[:3])
    top3_sensitivity = frozenset(sensitivity_rank[:3])
    top3_jaccard = len(top3_anchor & top3_sensitivity) / len(
        top3_anchor | top3_sensitivity
    )

    review_reasons: list[str] = []
    if len(_top_tied(anchor)) > 1:
        review_reasons.append("anchor_top_tie")
    if len(_top_tied(sensitivity)) > 1:
        review_reasons.append("sensitivity_top_tie")
    if anchor_top1 != sensitivity_top1:
        review_reasons.append("reference_top1_disagreement")
    if event_count == 1:
        review_reasons.append("event_dispersion_not_estimable")
    if not bool(quality.all()):
        review_reasons.append("signal_quality_coverage_incomplete")
    if any(not available for _, available in facts):
        review_reasons.append("report_fact_coverage_incomplete")

    abstention_reasons = ["selective_threshold_undefined"]
    # The frozen anchor has already aggregated every supplied event.  Once a
    # severe quality producer invalidates any event/candidate cell, silently
    # retaining that score would treat a potentially contaminated input as
    # valid.  MRSC cannot repair or re-aggregate the anchor, so it must abstain
    # while preserving all scores bitwise.
    if not bool(quality.all()):
        abstention_reasons.append("anchor_input_quality_invalid")
    # A candidate with no quality-valid event is the stronger structural
    # deployment failure and receives an additional, specific reason code.
    if bool((quality.any(dim=0) == 0).any()):
        abstention_reasons.append("candidate_quality_unavailable")

    return MRSCAssessment(
        anchor_scores=anchor,
        sensitivity_scores=sensitivity,
        anchor_top1_index=anchor_top1,
        anchor_top1_channel=MRSC_CANDIDATE_CHANNELS[anchor_top1],
        sensitivity_top1_index=sensitivity_top1,
        sensitivity_top1_channel=MRSC_CANDIDATE_CHANNELS[sensitivity_top1],
        top1_reference_agreement=anchor_top1 == sensitivity_top1,
        top3_reference_jaccard=top3_jaccard,
        components=components,
        nonconformity=nonconformity,
        report_fact_available_mask=facts,
        abstain=True,
        abstention_reason_codes=tuple(abstention_reasons),
        review_reason_codes=tuple(review_reasons),
    )


__all__ = [
    "MRSC_CANDIDATE_CHANNELS",
    "MRSC_NONCONFORMITY_SEMANTICS",
    "MRSC_N_CANDIDATES",
    "MRSC_REPORT_FACT_FIELDS",
    "MRSC_SCHEMA",
    "MRSC_USE_POLICY",
    "MRSCAssessment",
    "MRSCUncertaintyComponents",
    "assess_mrsc_score_preserving",
]
