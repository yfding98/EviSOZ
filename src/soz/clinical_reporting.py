"""Deterministic channel-first spatial reports and grounded explanations.

This module is deliberately downstream of the evidence-only reasoner.  It
does not train a region head, consume raw EEG, or reinterpret a bipolar edge
as an SOZ endpoint.  All spatial views are frozen deterministic projections
of the standard-19 channel scores, and every explanation is reconstructed
from :class:`~src.soz.models.reasoner.ReasonerOutput` contributions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
from typing import Mapping, Sequence

import torch

from .geometry import CHANNEL_INDEX, N_STANDARD_CHANNELS, STANDARD_19, TCP_20_EDGES
from .final_score_reference_disagreement import (
    FinalScoreReferenceDisagreementReceipt,
)
from .models.reasoner import PHASE_COMPONENT_NAMES, ReasonerOutput
from .reference_disagreement import (
    REFERENCE_DISAGREEMENT_METRIC_ID,
    REFERENCE_DISAGREEMENT_SCOPE,
    REFERENCE_DISAGREEMENT_USE_POLICY,
    ReferenceDisagreementReceipt,
)


OPERATIONAL_SENSOR_GROUPS: dict[str, tuple[str, ...]] = {
    "left_temporal_chain": ("F7", "T7", "P7"),
    "right_temporal_chain": ("F8", "T8", "P8"),
    "left_parasagittal": ("FP1", "F3", "C3", "P3", "O1"),
    "right_parasagittal": ("FP2", "F4", "C4", "P4", "O2"),
    "midline": ("FZ", "CZ", "PZ"),
}

# SCORE-style scalp-region view.  F7/F8 are explicitly treated as a
# frontotemporal boundary in report wording and routed to the temporal-chain
# policy; P7/P8 are posterior-temporal legacy T5/T6 sites.
CLINICAL_SCALP_REGIONS: dict[str, tuple[str, ...]] = {
    "frontal": ("FP1", "FP2", "F3", "FZ", "F4"),
    "temporal": ("F7", "T7", "P7", "F8", "T8", "P8"),
    "central": ("C3", "CZ", "C4"),
    "parietal": ("P3", "PZ", "P4"),
    "occipital": ("O1", "O2"),
}

LATERALITY_GROUPS: dict[str, tuple[str, ...]] = {
    "left": ("FP1", "F7", "F3", "T7", "C3", "P7", "P3", "O1"),
    "right": ("FP2", "F4", "F8", "T8", "C4", "P4", "P8", "O2"),
    "midline": ("FZ", "CZ", "PZ"),
}

SCORE_SEMANTICS = {
    "uncalibrated_localization_score",
    "public_patient_benchmark_probability",
    "private_patient_calibrator_transformed_score",
}

CLAIM_BOUNDARY = (
    "This is a clinical-reference scalp-electrode SOZ hypothesis; it is not "
    "an invasive cortical SOZ, epileptogenic zone, or treatment target."
)

CLINICAL_REPORT_SCHEMA_V2 = "soz-clinical-report.v2"
EVIDENCE_RECEIPT_SCHEMA_V2 = "soz-event-evidence-receipt.v2"
EVENT_REFERENCE_CONSISTENCY_SCHEMA = (
    "soz-target-free-event-reference-consistency.v1"
)
EVENT_REFERENCE_CONSISTENCY_USE_POLICY = (
    "report_fact_and_abstention_only_not_soz_scoring_or_model_selection"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TIME_COORDINATE_SEMANTICS = {
    "recording_start_seconds",
    "event_window_start_seconds",
}
_RHYTHM_STATES_ZH = {
    "rhythmic": "节律性活动",
    "evolving_rhythmic": "持续演变的节律性活动",
}
_PATIENT_AGGREGATION_METHODS_ZH = {
    "patient_equal_event_mean": "按发作事件等权平均",
}

_ARTIFACT_ZH = {
    "muscle": "肌电",
    "ocular": "眼动",
    "movement": "运动",
    "electrode_transient": "电极瞬态",
    "line_noise": "工频",
}
_MONTAGE_STABILITY_ZH = {
    "consistent": (
        "C-CAR19与C-REF19下该事件的最先检出双极导联及时序一致"
    ),
    "partially_consistent": (
        "C-CAR19与C-REF19下该事件的最先检出双极导联侧别一致，"
        "但精确导联存在差异"
    ),
    "inconsistent": (
        "C-CAR19与C-REF19下该事件的最先检出双极导联不一致"
    ),
    "not_assessed": "尚未完成不同蒙太奇的一致性评估",
}

# Closed vocabulary for a separately validated later-visible region producer.
# Tokens combine the frozen laterality and SCORE-style scalp-region terms;
# arbitrary prose is intentionally rejected at the typed facts boundary.
LATER_VISIBLE_REGIONS_ZH = frozenset(
    f"{laterality}{region}"
    for laterality in ("左", "右", "双侧", "中线", "侧别不确定的")
    for region in ("额区", "颞区", "中央区", "顶区", "枕区", "多区域")
)


def _validate_partition(
    name: str, groups: Mapping[str, Sequence[str]]
) -> None:
    members = tuple(channel for channels in groups.values() for channel in channels)
    if len(members) != N_STANDARD_CHANNELS or set(members) != set(STANDARD_19):
        raise RuntimeError(f"{name} must be an exhaustive disjoint standard-19 partition")


_validate_partition("operational sensor groups", OPERATIONAL_SENSOR_GROUPS)
_validate_partition("clinical scalp regions", CLINICAL_SCALP_REGIONS)
_validate_partition("laterality groups", LATERALITY_GROUPS)


def _channel_vector(
    value: torch.Tensor, *, name: str, dtype: torch.dtype | None = None
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != (
        N_STANDARD_CHANNELS,
    ):
        raise ValueError(f"{name} must have shape [19]")
    if dtype is not None and value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    if value.requires_grad:
        raise ValueError(f"{name} must be detached")
    result = value.detach().cpu().contiguous().clone()
    return result


@dataclass(frozen=True)
class SpatialViewRow:
    name: str
    score: float | None
    top_members: tuple[str, ...]
    evaluable_members: tuple[str, ...]


@dataclass(frozen=True)
class DerivedSpatialReport:
    score_semantics: str
    top_channels: tuple[str, ...]
    top_score: float
    evaluable_channels: tuple[str, ...]
    operational_groups: tuple[SpatialViewRow, ...]
    scalp_regions: tuple[SpatialViewRow, ...]
    lateralities: tuple[SpatialViewRow, ...]
    top_operational_groups: tuple[str, ...]
    top_scalp_regions: tuple[str, ...]
    laterality: str
    claim_boundary: str = CLAIM_BOUNDARY


def _pool_view(
    scores: torch.Tensor,
    evaluable: torch.Tensor,
    groups: Mapping[str, Sequence[str]],
) -> tuple[SpatialViewRow, ...]:
    rows: list[SpatialViewRow] = []
    for name, members in groups.items():
        available = tuple(
            member for member in members if bool(evaluable[CHANNEL_INDEX[member]])
        )
        if not available:
            rows.append(
                SpatialViewRow(
                    name=name,
                    score=None,
                    top_members=(),
                    evaluable_members=(),
                )
            )
            continue
        values = {member: float(scores[CHANNEL_INDEX[member]].item()) for member in available}
        maximum = max(values.values())
        top = tuple(member for member in available if values[member] == maximum)
        rows.append(
            SpatialViewRow(
                name=name,
                score=maximum,
                top_members=top,
                evaluable_members=available,
            )
        )
    return tuple(rows)


def _top_rows(rows: Sequence[SpatialViewRow]) -> tuple[str, ...]:
    observed = tuple(row for row in rows if row.score is not None)
    if not observed:
        raise ValueError("A spatial view contains no evaluable member")
    maximum = max(float(row.score) for row in observed)
    return tuple(row.name for row in observed if row.score == maximum)


def derive_spatial_report(
    channel_scores: torch.Tensor,
    channel_evaluable_mask: torch.Tensor,
    *,
    score_semantics: str,
) -> DerivedSpatialReport:
    """Derive all report views from one masked standard-19 score vector."""

    if score_semantics not in SCORE_SEMANTICS:
        raise ValueError("Unsupported channel-score semantics")
    scores = _channel_vector(channel_scores, name="channel_scores")
    if not scores.is_floating_point() or not torch.isfinite(scores).all():
        raise ValueError("channel_scores must be finite floating point")
    if score_semantics != "uncalibrated_localization_score" and bool(
        ((scores < 0) | (scores > 1)).any()
    ):
        raise ValueError("Probability/transformed score views must lie in [0,1]")
    evaluable = _channel_vector(
        channel_evaluable_mask,
        name="channel_evaluable_mask",
        dtype=torch.bool,
    )
    channels = tuple(
        channel for channel in STANDARD_19 if bool(evaluable[CHANNEL_INDEX[channel]])
    )
    if not channels:
        raise ValueError("At least one standard-19 channel must be evaluable")
    values = {channel: float(scores[CHANNEL_INDEX[channel]].item()) for channel in channels}
    top_score = max(values.values())
    top_channels = tuple(channel for channel in channels if values[channel] == top_score)
    operational = _pool_view(scores, evaluable, OPERATIONAL_SENSOR_GROUPS)
    regions = _pool_view(scores, evaluable, CLINICAL_SCALP_REGIONS)
    lateralities = _pool_view(scores, evaluable, LATERALITY_GROUPS)
    top_lateralities = _top_rows(lateralities)
    if len(top_lateralities) == 1:
        laterality = top_lateralities[0]
    elif set(top_lateralities) == {"left", "right"}:
        laterality = "bilateral"
    else:
        laterality = "indeterminate"
    return DerivedSpatialReport(
        score_semantics=score_semantics,
        top_channels=top_channels,
        top_score=top_score,
        evaluable_channels=channels,
        operational_groups=operational,
        scalp_regions=regions,
        lateralities=lateralities,
        top_operational_groups=_top_rows(operational),
        top_scalp_regions=_top_rows(regions),
        laterality=laterality,
    )


def derive_view_targets(
    channel_targets: torch.Tensor,
    channel_target_mask: torch.Tensor,
    groups: Mapping[str, Sequence[str]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Derive tri-state group targets without manufacturing negatives.

    A group is positive when any observed member is positive.  It is an
    observed negative only when every member is observed and all are zero.
    Otherwise it remains unknown/masked.
    """

    targets = _channel_vector(channel_targets, name="channel_targets")
    mask = _channel_vector(
        channel_target_mask, name="channel_target_mask", dtype=torch.bool
    )
    observed = targets[mask]
    if not targets.is_floating_point() or not torch.isfinite(observed).all():
        raise ValueError("Observed channel targets must be finite floating point")
    if observed.numel() and not torch.all((observed == 0) | (observed == 1)):
        raise ValueError("Observed channel targets must be binary")
    values = torch.zeros(len(groups), dtype=torch.float32)
    output_mask = torch.zeros(len(groups), dtype=torch.bool)
    for group_index, members in enumerate(groups.values()):
        indices = torch.tensor([CHANNEL_INDEX[channel] for channel in members])
        member_mask = mask[indices]
        member_targets = targets[indices]
        if bool(((member_targets == 1) & member_mask).any()):
            values[group_index] = 1.0
            output_mask[group_index] = True
        elif bool(member_mask.all()):
            output_mask[group_index] = True
    return values, output_mask


@dataclass(frozen=True)
class GroundedChannelExplanation:
    channel: str
    score: float
    family_contributions: tuple[tuple[str, float], ...]
    phase_contributions: tuple[tuple[str, float], ...]
    morphology_incident_edges: tuple[tuple[str, float], ...]
    ictal_incident_edges: tuple[tuple[str, float], ...]
    morphology_quality_gate_mean: float
    morphology_specificity_gate: float
    statement: str
    claim_boundary: str = CLAIM_BOUNDARY


@dataclass(frozen=True)
class GroundedPatientChannelExplanation:
    """Equal-event patient explanation on the same scale as patient logits."""

    channel: str
    score: float
    aggregation_event_count: int
    family_contributions: tuple[tuple[str, float], ...]
    phase_contributions: tuple[tuple[str, float], ...]
    morphology_incident_edges: tuple[tuple[str, float], ...]
    ictal_incident_edges: tuple[tuple[str, float], ...]
    morphology_quality_gate_mean: float
    morphology_specificity_gate_mean: float
    statement: str
    claim_boundary: str = CLAIM_BOUNDARY


def _require_nonempty_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _canonical_sha256(payload: object) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _validate_unit_interval(value: float | None, *, name: str) -> None:
    if value is not None and (
        not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0 <= float(value) <= 1
    ):
        raise ValueError(f"{name} must lie in [0,1]")


def _validate_tcp_derivations(values: tuple[str, ...], *, name: str) -> None:
    valid_edges = {f"{left}-{right}" for left, right in TCP_20_EDGES}
    valid_edges |= {f"{right}-{left}" for left, right in TCP_20_EDGES}
    if (
        not isinstance(values, tuple)
        or len(set(values)) != len(values)
        or any(value not in valid_edges for value in values)
    ):
        raise ValueError(f"{name} must contain unique common-TCP derivations")


@dataclass(frozen=True)
class EvidenceProvenanceReceipt:
    """Immutable receipt for event-level, target-free evidence extraction.

    The two target-use declarations apply to construction of the event
    phenotype at inference/report time.  They prevent a SOZ or private label
    from being laundered into an apparently observational onset fact.
    """

    patient_pseudonym: str
    event_pseudonym: str
    signal_artifact_sha256: str
    evidence_artifact_sha256: str
    extractor_model_id: str
    extractor_model_version: str
    time_coordinate_semantics: str
    causal_prefix_safe: bool
    montages: tuple[str, ...]
    evidence_generation_policy: str
    soz_labels_used_for_event_evidence: bool
    private_labels_used_for_event_evidence: bool
    reference_pair_schema_version: str | None = None
    reference_pair_role: str | None = None
    reference_primary_arm_id: str | None = None
    reference_sensitivity_arm_id: str | None = None
    reference_disagreement_metric_id: str | None = None
    reference_disagreement_receipt_sha256: str | None = None
    schema_version: str = EVIDENCE_RECEIPT_SCHEMA_V2

    def __post_init__(self) -> None:
        if self.schema_version != EVIDENCE_RECEIPT_SCHEMA_V2:
            raise ValueError("Unsupported event-evidence receipt schema")
        for name, value in (
            ("patient_pseudonym", self.patient_pseudonym),
            ("event_pseudonym", self.event_pseudonym),
            ("extractor_model_id", self.extractor_model_id),
            ("extractor_model_version", self.extractor_model_version),
            ("evidence_generation_policy", self.evidence_generation_policy),
        ):
            _require_nonempty_text(value, name=name)
        _require_sha256(self.signal_artifact_sha256, name="signal_artifact_sha256")
        _require_sha256(
            self.evidence_artifact_sha256, name="evidence_artifact_sha256"
        )
        if self.time_coordinate_semantics not in _TIME_COORDINATE_SEMANTICS:
            raise ValueError("Unsupported time_coordinate_semantics")
        if type(self.causal_prefix_safe) is not bool:
            raise TypeError("causal_prefix_safe must be bool")
        if (
            not isinstance(self.montages, tuple)
            or not self.montages
            or len(set(self.montages)) != len(self.montages)
        ):
            raise ValueError("montages must be a non-empty unique tuple")
        for montage in self.montages:
            _require_nonempty_text(montage, name="montage")
        for name, value in (
            (
                "soz_labels_used_for_event_evidence",
                self.soz_labels_used_for_event_evidence,
            ),
            (
                "private_labels_used_for_event_evidence",
                self.private_labels_used_for_event_evidence,
            ),
        ):
            if type(value) is not bool:
                raise TypeError(f"{name} must be bool")
            if value:
                raise ValueError(
                    f"{name} must be False for a target-free event phenotype"
                )
        reference_fields = (
            self.reference_pair_schema_version,
            self.reference_pair_role,
            self.reference_primary_arm_id,
            self.reference_sensitivity_arm_id,
            self.reference_disagreement_metric_id,
            self.reference_disagreement_receipt_sha256,
        )
        if any(value is not None for value in reference_fields):
            if any(value is None for value in reference_fields):
                raise ValueError(
                    "Reference-disagreement provenance fields must be all present"
                )
            for name, value in (
                ("reference_pair_schema_version", self.reference_pair_schema_version),
                ("reference_pair_role", self.reference_pair_role),
                ("reference_primary_arm_id", self.reference_primary_arm_id),
                (
                    "reference_sensitivity_arm_id",
                    self.reference_sensitivity_arm_id,
                ),
                (
                    "reference_disagreement_metric_id",
                    self.reference_disagreement_metric_id,
                ),
            ):
                _require_nonempty_text(value, name=name)
            _require_sha256(
                self.reference_disagreement_receipt_sha256,
                name="reference_disagreement_receipt_sha256",
            )
            if self.reference_disagreement_metric_id != REFERENCE_DISAGREEMENT_METRIC_ID:
                raise ValueError("Unsupported reference-disagreement metric")
            if self.reference_primary_arm_id not in self.montages:
                raise ValueError("Primary reference arm is absent from montages")
            if self.reference_sensitivity_arm_id not in self.montages:
                raise ValueError("Sensitivity reference arm is absent from montages")


@dataclass(frozen=True)
class EventReferenceConsistencyReceipt:
    """Target-free receipt for a same-event C-CAR19/C-REF19 comparison.

    The receipt concerns only the event phenotype emitted independently from
    the two reference arms.  It neither reads nor changes a localization
    score.  In particular, ``consistent`` means that the observed bipolar
    derivations and their detection time agree under the frozen rule; it is
    not evidence that a cortical SOZ was reproduced under two montages.
    """

    patient_pseudonym: str
    event_pseudonym: str
    signal_artifact_sha256: str
    primary_evidence_artifact_sha256: str
    sensitivity_evidence_artifact_sha256: str
    primary_arm_id: str
    sensitivity_arm_id: str
    primary_result_status: str
    sensitivity_result_status: str
    temporal_alignment_tolerance_sec: float
    onset_start_delta_sec: float | None
    primary_first_visible_derivations: tuple[str, ...]
    sensitivity_first_visible_derivations: tuple[str, ...]
    montage_stability: str | None
    reason_codes: tuple[str, ...]
    target_labels_used: bool
    private_data_used: bool
    localization_scores_used: bool
    training_performed: bool
    use_policy: str = EVENT_REFERENCE_CONSISTENCY_USE_POLICY
    schema_version: str = EVENT_REFERENCE_CONSISTENCY_SCHEMA

    def __post_init__(self) -> None:
        for name, value in (
            ("patient_pseudonym", self.patient_pseudonym),
            ("event_pseudonym", self.event_pseudonym),
        ):
            _require_nonempty_text(value, name=name)
        for name, value in (
            ("signal_artifact_sha256", self.signal_artifact_sha256),
            (
                "primary_evidence_artifact_sha256",
                self.primary_evidence_artifact_sha256,
            ),
            (
                "sensitivity_evidence_artifact_sha256",
                self.sensitivity_evidence_artifact_sha256,
            ),
        ):
            _require_sha256(value, name=name)
        if self.primary_arm_id != "C-CAR19":
            raise ValueError("Event reference primary arm must be C-CAR19")
        if self.sensitivity_arm_id != "C-REF19":
            raise ValueError("Event reference sensitivity arm must be C-REF19")
        statuses = {"reportable", "abstained"}
        if self.primary_result_status not in statuses:
            raise ValueError("Unsupported primary event-phenotype status")
        if self.sensitivity_result_status not in statuses:
            raise ValueError("Unsupported sensitivity event-phenotype status")
        if (
            isinstance(self.temporal_alignment_tolerance_sec, bool)
            or not isinstance(self.temporal_alignment_tolerance_sec, (int, float))
            or not math.isfinite(float(self.temporal_alignment_tolerance_sec))
            or self.temporal_alignment_tolerance_sec < 0
        ):
            raise ValueError("temporal_alignment_tolerance_sec must be non-negative")
        if self.onset_start_delta_sec is not None and (
            isinstance(self.onset_start_delta_sec, bool)
            or not isinstance(self.onset_start_delta_sec, (int, float))
            or not math.isfinite(float(self.onset_start_delta_sec))
            or self.onset_start_delta_sec < 0
        ):
            raise ValueError("onset_start_delta_sec must be non-negative or None")
        _validate_tcp_derivations(
            self.primary_first_visible_derivations,
            name="primary_first_visible_derivations",
        )
        _validate_tcp_derivations(
            self.sensitivity_first_visible_derivations,
            name="sensitivity_first_visible_derivations",
        )
        both_reportable = (
            self.primary_result_status == "reportable"
            and self.sensitivity_result_status == "reportable"
        )
        if both_reportable:
            if not self.primary_first_visible_derivations:
                raise ValueError("Reportable primary arm needs first-visible derivations")
            if not self.sensitivity_first_visible_derivations:
                raise ValueError(
                    "Reportable sensitivity arm needs first-visible derivations"
                )
            if self.onset_start_delta_sec is None:
                raise ValueError("A paired reportable event needs an onset-time delta")
        else:
            if self.onset_start_delta_sec is not None:
                raise ValueError("An abstained reference arm cannot have an onset delta")
            if (
                self.primary_result_status == "abstained"
                and self.primary_first_visible_derivations
            ):
                raise ValueError("An abstained primary arm cannot carry derivations")
            if (
                self.sensitivity_result_status == "abstained"
                and self.sensitivity_first_visible_derivations
            ):
                raise ValueError("An abstained sensitivity arm cannot carry derivations")
        if self.montage_stability is not None and self.montage_stability not in {
            "consistent",
            "partially_consistent",
            "inconsistent",
        }:
            raise ValueError("Unsupported event reference-consistency state")
        if (
            not isinstance(self.reason_codes, tuple)
            or len(set(self.reason_codes)) != len(self.reason_codes)
            or any(
                not isinstance(value, str)
                or re.fullmatch(r"[a-z][a-z0-9_]*", value) is None
                for value in self.reason_codes
            )
        ):
            raise ValueError("reason_codes must be unique stable tokens")
        if self.montage_stability is None and not self.reason_codes:
            raise ValueError("An unassessed reference pair needs a reason code")
        if self.montage_stability is not None:
            if not both_reportable:
                raise ValueError("Reference stability requires two reportable arms")
            if self.reason_codes:
                raise ValueError("An assessed reference pair cannot have reason codes")
            if self.onset_start_delta_sec is None or (
                self.onset_start_delta_sec
                > self.temporal_alignment_tolerance_sec + 1e-12
            ):
                raise ValueError("Reference stability requires temporally aligned arms")
        for name in (
            "target_labels_used",
            "private_data_used",
            "localization_scores_used",
            "training_performed",
        ):
            value = getattr(self, name)
            if type(value) is not bool:
                raise TypeError(f"{name} must be bool")
            if value:
                raise ValueError(
                    "Event reference consistency must remain target/private/score free"
                )
        if self.use_policy != EVENT_REFERENCE_CONSISTENCY_USE_POLICY:
            raise ValueError("Unsupported event reference-consistency use policy")
        if self.schema_version != EVENT_REFERENCE_CONSISTENCY_SCHEMA:
            raise ValueError("Unsupported event reference-consistency schema")

    @property
    def receipt_sha256(self) -> str:
        return _canonical_sha256(asdict(self))


@dataclass(frozen=True)
class EventScalpPhenotypeEvidence:
    """Explicit event-level scalp-visible facts; never an SOZ target."""

    receipt: EvidenceProvenanceReceipt
    onset_start_sec: float
    onset_end_sec: float
    first_visible_derivations: tuple[str, ...]
    rhythm_state: str | None = None
    frequency_range_hz: tuple[float, float] | None = None
    later_visible_delay_sec: float | None = None
    later_visible_derivations: tuple[str, ...] = ()
    later_visible_region_zh: str | None = None
    montage_stability: str | None = None
    artifact_assessed: bool | None = None
    artifact_types: tuple[str, ...] = ()
    artifact_burden: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.receipt, EvidenceProvenanceReceipt):
            raise TypeError("receipt must be EvidenceProvenanceReceipt")
        for name, value in (
            ("onset_start_sec", self.onset_start_sec),
            ("onset_end_sec", self.onset_end_sec),
        ):
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.onset_start_sec < 0 or self.onset_end_sec < self.onset_start_sec:
            raise ValueError("Onset interval must be non-negative and ordered")
        _validate_tcp_derivations(
            self.first_visible_derivations, name="first_visible_derivations"
        )
        if not self.first_visible_derivations:
            raise ValueError("An event phenotype needs a first-visible derivation")
        if self.rhythm_state is not None and self.rhythm_state not in _RHYTHM_STATES_ZH:
            raise ValueError("Unsupported rhythm_state")
        if self.frequency_range_hz is not None:
            if (
                not isinstance(self.frequency_range_hz, tuple)
                or len(self.frequency_range_hz) != 2
            ):
                raise ValueError("frequency_range_hz must be a two-value tuple")
            lower, upper = self.frequency_range_hz
            if not all(
                isinstance(value, (int, float)) and math.isfinite(float(value))
                for value in (lower, upper)
            ):
                raise ValueError("frequency_range_hz must be finite")
            if lower <= 0 or upper < lower or upper > 45:
                raise ValueError("frequency_range_hz must be ordered within (0,45] Hz")
        _validate_tcp_derivations(
            self.later_visible_derivations, name="later_visible_derivations"
        )
        later_destination = bool(
            self.later_visible_derivations or self.later_visible_region_zh
        )
        if self.later_visible_delay_sec is not None:
            if (
                not isinstance(self.later_visible_delay_sec, (int, float))
                or not math.isfinite(float(self.later_visible_delay_sec))
                or self.later_visible_delay_sec < 0
            ):
                raise ValueError(
                    "later_visible_delay_sec must be finite and non-negative"
                )
            if not later_destination:
                raise ValueError("Later-visible timing requires an explicit destination")
        if self.later_visible_region_zh is not None:
            if self.later_visible_region_zh not in LATER_VISIBLE_REGIONS_ZH:
                raise ValueError(
                    "later_visible_region_zh must use the frozen clinical region "
                    "vocabulary"
                )
        if (
            self.montage_stability is not None
            and self.montage_stability not in _MONTAGE_STABILITY_ZH
        ):
            raise ValueError("Unsupported montage_stability state")
        if self.artifact_assessed is not None and type(self.artifact_assessed) is not bool:
            raise TypeError("artifact_assessed must be bool or None")
        if (
            not isinstance(self.artifact_types, tuple)
            or len(set(self.artifact_types)) != len(self.artifact_types)
            or any(value not in _ARTIFACT_ZH for value in self.artifact_types)
        ):
            raise ValueError("Unsupported or duplicate artifact type")
        if (
            self.artifact_types or self.artifact_burden is not None
        ) and self.artifact_assessed is not True:
            raise ValueError("Artifact facts require artifact_assessed=True")
        _validate_unit_interval(self.artifact_burden, name="artifact_burden")


@dataclass(frozen=True)
class EventScalpPhenotypeAbstention:
    """Target-free event producer abstention without fabricated phenotype facts."""

    receipt: EvidenceProvenanceReceipt
    reason_codes: tuple[str, ...]
    detected_bipolar_edge_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.receipt, EvidenceProvenanceReceipt):
            raise TypeError("receipt must be EvidenceProvenanceReceipt")
        if (
            not isinstance(self.reason_codes, tuple)
            or not self.reason_codes
            or len(set(self.reason_codes)) != len(self.reason_codes)
            or any(
                not isinstance(value, str)
                or re.fullmatch(r"[a-z][a-z0-9_]*", value) is None
                for value in self.reason_codes
            )
        ):
            raise ValueError("reason_codes must be non-empty unique stable tokens")
        if type(self.detected_bipolar_edge_count) is not int or not (
            0 <= self.detected_bipolar_edge_count <= len(TCP_20_EDGES)
        ):
            raise ValueError("detected_bipolar_edge_count is invalid")


@dataclass(frozen=True)
class UncertaintyDecomposition:
    """Optional, separately named uncertainty facts with an abstention receipt."""

    ranking_ambiguity: float | None = None
    within_patient_event_dispersion: float | None = None
    signal_quality_uncertainty: float | None = None
    montage_disagreement: float | None = None
    final_score_reference_disagreement: float | None = None
    epistemic_uncertainty: float | None = None
    abstain: bool = False
    abstention_reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("ranking_ambiguity", self.ranking_ambiguity),
            ("within_patient_event_dispersion", self.within_patient_event_dispersion),
            ("signal_quality_uncertainty", self.signal_quality_uncertainty),
            ("montage_disagreement", self.montage_disagreement),
            (
                "final_score_reference_disagreement",
                self.final_score_reference_disagreement,
            ),
            ("epistemic_uncertainty", self.epistemic_uncertainty),
        ):
            _validate_unit_interval(value, name=name)
        if type(self.abstain) is not bool:
            raise TypeError("abstain must be bool")
        if (
            not isinstance(self.abstention_reason_codes, tuple)
            or len(set(self.abstention_reason_codes))
            != len(self.abstention_reason_codes)
            or any(
                not isinstance(value, str)
                or re.fullmatch(r"[a-z][a-z0-9_]*", value) is None
                for value in self.abstention_reason_codes
            )
        ):
            raise ValueError("abstention_reason_codes must be unique stable tokens")
        if self.abstain and not self.abstention_reason_codes:
            raise ValueError("Abstention requires at least one reason code")
        if self.abstention_reason_codes and not self.abstain:
            raise ValueError("Abstention reason codes require abstain=True")


@dataclass(frozen=True)
class PatientSOZReferenceRanking:
    """Patient-level scalp-electrode ranking with exact aggregation identity."""

    spatial_report: DerivedSpatialReport
    patient_pseudonym: str
    model_id: str
    model_version: str
    model_checkpoint_sha256: str
    aggregation_method: str
    aggregation_event_count: int
    aggregation_event_ids: tuple[str, ...]
    aggregation_receipt_sha256: str
    uncertainty: UncertaintyDecomposition
    ranking_granularity: str = "patient"

    def __post_init__(self) -> None:
        if not isinstance(self.spatial_report, DerivedSpatialReport):
            raise TypeError("spatial_report must be DerivedSpatialReport")
        for name, value in (
            ("patient_pseudonym", self.patient_pseudonym),
            ("model_id", self.model_id),
            ("model_version", self.model_version),
        ):
            _require_nonempty_text(value, name=name)
        _require_sha256(
            self.model_checkpoint_sha256, name="model_checkpoint_sha256"
        )
        _require_sha256(
            self.aggregation_receipt_sha256, name="aggregation_receipt_sha256"
        )
        if self.ranking_granularity != "patient":
            raise ValueError("SOZ-reference ranking_granularity must be patient")
        if self.aggregation_method not in _PATIENT_AGGREGATION_METHODS_ZH:
            raise ValueError("Unsupported patient aggregation_method")
        if type(self.aggregation_event_count) is not int or self.aggregation_event_count < 1:
            raise ValueError("aggregation_event_count must be a positive integer")
        if (
            not isinstance(self.aggregation_event_ids, tuple)
            or not self.aggregation_event_ids
            or len(set(self.aggregation_event_ids)) != len(self.aggregation_event_ids)
        ):
            raise ValueError("aggregation_event_ids must be a non-empty unique tuple")
        for event_id in self.aggregation_event_ids:
            _require_nonempty_text(event_id, name="aggregation_event_id")
        if self.aggregation_event_count != len(self.aggregation_event_ids):
            raise ValueError(
                "aggregation_event_count must match aggregation_event_ids"
            )
        if not isinstance(self.uncertainty, UncertaintyDecomposition):
            raise TypeError("uncertainty must be UncertaintyDecomposition")


@dataclass(frozen=True)
class ClinicalReportFactsV2:
    """Facts-locked v2 boundary between one event and one patient ranking."""

    event_phenotype: EventScalpPhenotypeEvidence | EventScalpPhenotypeAbstention
    patient_ranking: PatientSOZReferenceRanking
    later_visible_region_receipt: "LaterVisibleRegionReceipt | None" = None
    reference_disagreement_receipt: ReferenceDisagreementReceipt | None = None
    final_score_reference_disagreement_receipt: (
        FinalScoreReferenceDisagreementReceipt | None
    ) = None
    event_reference_consistency_receipt: (
        EventReferenceConsistencyReceipt | None
    ) = None
    require_causal_prefix_safe: bool = True
    schema_version: str = CLINICAL_REPORT_SCHEMA_V2

    def __post_init__(self) -> None:
        if self.schema_version != CLINICAL_REPORT_SCHEMA_V2:
            raise ValueError("Unsupported clinical-report schema")
        if not isinstance(
            self.event_phenotype,
            (EventScalpPhenotypeEvidence, EventScalpPhenotypeAbstention),
        ):
            raise TypeError(
                "event_phenotype must be event evidence or an event abstention"
            )
        if not isinstance(self.patient_ranking, PatientSOZReferenceRanking):
            raise TypeError("patient_ranking must be PatientSOZReferenceRanking")
        receipt = self.event_phenotype.receipt
        ranking = self.patient_ranking
        if receipt.patient_pseudonym != ranking.patient_pseudonym:
            raise ValueError("Event receipt and patient ranking refer to different patients")
        if receipt.event_pseudonym not in ranking.aggregation_event_ids:
            raise ValueError(
                "Event receipt is absent from the patient aggregation receipt"
            )
        if type(self.require_causal_prefix_safe) is not bool:
            raise TypeError("require_causal_prefix_safe must be bool")
        if self.require_causal_prefix_safe and not receipt.causal_prefix_safe:
            raise ValueError("Report contract requires causal/prefix-safe event evidence")
        later_region_receipt = self.later_visible_region_receipt
        later_region = (
            self.event_phenotype.later_visible_region_zh
            if isinstance(self.event_phenotype, EventScalpPhenotypeEvidence)
            else None
        )
        if later_region_receipt is None:
            if later_region is not None:
                raise ValueError(
                    "later_visible_region_zh requires a bound typed receipt"
                )
        else:
            # Local import avoids a module-import cycle: the deterministic
            # producer itself consumes this module's frozen scalp ontology.
            from .later_visible_region_producer import (
                LaterVisibleRegionReceipt,
                event_evidence_core_sha256,
            )

            if not isinstance(later_region_receipt, LaterVisibleRegionReceipt):
                raise TypeError(
                    "later_visible_region_receipt must be a "
                    "LaterVisibleRegionReceipt"
                )
            if not isinstance(
                self.event_phenotype, EventScalpPhenotypeEvidence
            ):
                raise ValueError(
                    "An event-phenotype abstention cannot carry a later region"
                )
            expected_later_binding = (
                (later_region_receipt.patient_pseudonym, receipt.patient_pseudonym),
                (later_region_receipt.event_pseudonym, receipt.event_pseudonym),
                (
                    later_region_receipt.evidence_artifact_sha256,
                    receipt.evidence_artifact_sha256,
                ),
                (
                    later_region_receipt.source_event_receipt_sha256,
                    event_evidence_core_sha256(receipt),
                ),
                (
                    later_region_receipt.observed_derivations,
                    self.event_phenotype.later_visible_derivations,
                ),
                (later_region_receipt.later_visible_region_zh, later_region),
            )
            if any(
                actual != expected
                for actual, expected in expected_later_binding
            ):
                raise ValueError(
                    "Later-visible region receipt disagrees with event facts"
                )
        event_reference = self.event_reference_consistency_receipt
        event_stability = (
            self.event_phenotype.montage_stability
            if isinstance(self.event_phenotype, EventScalpPhenotypeEvidence)
            else None
        )
        if event_reference is None:
            if event_stability is not None:
                raise ValueError(
                    "montage_stability requires a bound event-reference receipt"
                )
        else:
            if not isinstance(event_reference, EventReferenceConsistencyReceipt):
                raise TypeError(
                    "event_reference_consistency_receipt must be an "
                    "EventReferenceConsistencyReceipt"
                )
            expected_event_reference = (
                (event_reference.patient_pseudonym, receipt.patient_pseudonym),
                (event_reference.event_pseudonym, receipt.event_pseudonym),
                (
                    event_reference.signal_artifact_sha256,
                    receipt.signal_artifact_sha256,
                ),
                (
                    event_reference.primary_evidence_artifact_sha256,
                    receipt.evidence_artifact_sha256,
                ),
                (event_reference.montage_stability, event_stability),
            )
            if any(actual != expected for actual, expected in expected_event_reference):
                raise ValueError(
                    "Event reference-consistency receipt disagrees with event facts"
                )
            for arm_id in (
                event_reference.primary_arm_id,
                event_reference.sensitivity_arm_id,
            ):
                if arm_id not in receipt.montages:
                    raise ValueError(
                        "Event reference-consistency arm is absent from provenance"
                    )
            expected_primary_status = (
                "reportable"
                if isinstance(self.event_phenotype, EventScalpPhenotypeEvidence)
                else "abstained"
            )
            if event_reference.primary_result_status != expected_primary_status:
                raise ValueError(
                    "Event reference-consistency primary status disagrees with facts"
                )
        reference = self.reference_disagreement_receipt
        reference_digest = receipt.reference_disagreement_receipt_sha256
        if reference is None:
            if reference_digest is not None:
                raise ValueError(
                    "Event provenance has a dangling reference-disagreement receipt"
                )
            if ranking.uncertainty.montage_disagreement is not None:
                raise ValueError(
                    "montage_disagreement requires a bound reference receipt"
                )
        else:
            if not isinstance(reference, ReferenceDisagreementReceipt):
                raise TypeError(
                    "reference_disagreement_receipt must be "
                    "ReferenceDisagreementReceipt"
                )
            if reference.patient_pseudonym != receipt.patient_pseudonym:
                raise ValueError("Reference receipt and event receipt patient mismatch")
            if reference.aggregation_event_ids != ranking.aggregation_event_ids:
                raise ValueError(
                    "Reference receipt and patient ranking aggregation roster mismatch"
                )
            reference_signal_sha = reference.signal_artifact_sha256_for_event(
                receipt.event_pseudonym
            )
            if reference_signal_sha != receipt.signal_artifact_sha256:
                raise ValueError("Reference receipt and event receipt signal mismatch")
            expected_provenance = (
                (
                    receipt.reference_pair_schema_version,
                    reference.reference_pair_schema_version,
                ),
                (receipt.reference_pair_role, reference.reference_pair_role),
                (receipt.reference_primary_arm_id, reference.primary_arm_id),
                (receipt.reference_sensitivity_arm_id, reference.sensitivity_arm_id),
                (receipt.reference_disagreement_metric_id, reference.metric_id),
                (reference_digest, reference.receipt_sha256),
            )
            if any(actual != expected for actual, expected in expected_provenance):
                raise ValueError(
                    "Reference receipt disagrees with event provenance binding"
                )
            disagreement = ranking.uncertainty.montage_disagreement
            if disagreement is None or not math.isclose(
                float(disagreement),
                reference.montage_disagreement,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "Reference receipt disagrees with montage_disagreement"
                )

        final_score_reference = self.final_score_reference_disagreement_receipt
        final_score_disagreement = (
            ranking.uncertainty.final_score_reference_disagreement
        )
        if final_score_reference is None:
            if final_score_disagreement is not None:
                raise ValueError(
                    "final_score_reference_disagreement requires its independent "
                    "final-score receipt"
                )
        else:
            if not isinstance(
                final_score_reference, FinalScoreReferenceDisagreementReceipt
            ):
                raise TypeError(
                    "final_score_reference_disagreement_receipt must be a "
                    "FinalScoreReferenceDisagreementReceipt"
                )
            if final_score_reference.patient_pseudonym != ranking.patient_pseudonym:
                raise ValueError(
                    "Final-score reference receipt and patient ranking mismatch"
                )
            if (
                final_score_reference.aggregation_event_ids
                != ranking.aggregation_event_ids
            ):
                raise ValueError(
                    "Final-score reference receipt and patient ranking aggregation "
                    "roster mismatch"
                )
            if final_score_disagreement is None or not math.isclose(
                float(final_score_disagreement),
                final_score_reference.final_score_reference_disagreement,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "Final-score receipt disagrees with reference-sensitivity fact"
                )


@dataclass(frozen=True)
class TemporalReportEvidence:
    """Typed evidence required for a non-hallucinated event narrative.

    Times and frequency are descriptive scalp-EEG evidence.  They are never
    interpreted as supervised SOZ onset times or propagation ground truth.
    """

    onset_start_sec: float
    onset_end_sec: float
    earliest_derivations: tuple[str, ...]
    frequency_range_hz: tuple[float, float] | None = None
    spread_delay_sec: float | None = None
    later_derivations: tuple[str, ...] = ()
    later_region_zh: str | None = None
    montage_stability: str = "not_assessed"
    artifact_assessed: bool = False
    artifact_types: tuple[str, ...] = ()
    artifact_burden: float | None = None
    uncertainty_score: float | None = None
    abstain: bool = False

    def __post_init__(self) -> None:
        for name, value in (
            ("onset_start_sec", self.onset_start_sec),
            ("onset_end_sec", self.onset_end_sec),
        ):
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.onset_start_sec < 0 or self.onset_end_sec < self.onset_start_sec:
            raise ValueError("Onset interval must be non-negative and ordered")
        valid_edges = {f"{left}-{right}" for left, right in TCP_20_EDGES}
        valid_edges |= {f"{right}-{left}" for left, right in TCP_20_EDGES}
        for name, values in (
            ("earliest_derivations", self.earliest_derivations),
            ("later_derivations", self.later_derivations),
        ):
            if len(set(values)) != len(values) or any(
                value not in valid_edges for value in values
            ):
                raise ValueError(f"{name} must contain unique common-TCP derivations")
        if not self.earliest_derivations:
            raise ValueError("A temporal report requires at least one earliest derivation")
        if self.frequency_range_hz is not None:
            lower, upper = self.frequency_range_hz
            if not all(math.isfinite(float(value)) for value in (lower, upper)):
                raise ValueError("frequency_range_hz must be finite")
            if lower <= 0 or upper < lower or upper > 45:
                raise ValueError("frequency_range_hz must be ordered within (0,45] Hz")
        if self.spread_delay_sec is not None:
            if not math.isfinite(float(self.spread_delay_sec)) or self.spread_delay_sec < 0:
                raise ValueError("spread_delay_sec must be finite and non-negative")
            if not self.later_derivations and not self.later_region_zh:
                raise ValueError("Later-visible evidence needs a derivation or region")
        if self.montage_stability not in _MONTAGE_STABILITY_ZH:
            raise ValueError("Unsupported montage_stability state")
        if any(value not in _ARTIFACT_ZH for value in self.artifact_types):
            raise ValueError("Unsupported artifact type")
        if self.artifact_types and not self.artifact_assessed:
            raise ValueError("Artifact types require artifact_assessed=True")
        for name, value in (
            ("artifact_burden", self.artifact_burden),
            ("uncertainty_score", self.uncertainty_score),
        ):
            if value is not None and (
                not math.isfinite(float(value)) or float(value) < 0 or float(value) > 1
            ):
                raise ValueError(f"{name} must lie in [0,1]")
        if self.artifact_burden is not None and not self.artifact_assessed:
            raise ValueError("artifact_burden requires artifact_assessed=True")


@dataclass(frozen=True)
class GroundedChineseDiagnosticReport:
    text: str
    report_status: str
    onset_phrase: str
    later_visible_phrase: str
    localization_phrase: str
    limitation_phrase: str
    event_phenotype_phrase: str = ""
    patient_ranking_phrase: str = ""
    provenance_phrase: str = ""
    causal_prefix_status: str = ""
    model_identity: str = ""
    event_aggregation_phrase: str = ""
    uncertainty_phrases: tuple[str, ...] = ()
    reference_robustness_phrase: str = ""
    schema_version: str = "legacy_unreceipted_v1"
    migration_status: str = "legacy_requires_v2_migration"
    evidence_receipt: EvidenceProvenanceReceipt | None = None
    aggregation_receipt_sha256: str | None = None
    later_visible_region_receipt_sha256: str | None = None
    reference_disagreement_receipt_sha256: str | None = None
    final_score_reference_disagreement_receipt_sha256: str | None = None
    event_reference_consistency_receipt_sha256: str | None = None


def _frequency_phrase(frequency: tuple[float, float] | None) -> str:
    if frequency is None:
        return ""
    lower, upper = (float(value) for value in frequency)
    bands = (
        (0.5, 4.0, "δ"),
        (4.0, 8.0, "θ"),
        (8.0, 13.0, "α"),
        (13.0, 30.0, "β"),
        (30.0, 45.000001, "低γ"),
    )
    for band_lower, band_upper, label in bands:
        if lower >= band_lower and upper <= band_upper:
            return f"主频位于{label}频段（{lower:.1f}–{upper:.1f} Hz）"
    return f"主频范围为{lower:.1f}–{upper:.1f} Hz"


def _hypothesis_location_zh(report: DerivedSpatialReport) -> str:
    region = report.top_scalp_regions[0] if len(report.top_scalp_regions) == 1 else "multiregional"
    region_zh = {
        "frontal": "额区",
        "temporal": "颞区",
        "central": "中央区",
        "parietal": "顶区",
        "occipital": "枕区",
        "multiregional": "多区域",
    }[region]
    laterality_zh = {
        "left": "左",
        "right": "右",
        "bilateral": "双侧",
        "midline": "中线",
        "indeterminate": "侧别不确定的",
    }[report.laterality]
    return f"{laterality_zh}{region_zh}"


def _event_phenotype_phrase(
    event: EventScalpPhenotypeEvidence,
    *,
    time_coordinate_semantics: str,
) -> str:
    time_origin = {
        "recording_start_seconds": "自记录起",
        "event_window_start_seconds": "自事件窗起",
    }.get(time_coordinate_semantics)
    if time_origin is None:
        raise ValueError("Unsupported event-phenotype time coordinate semantics")
    edges = "及".join(
        value.replace("-", "–") for value in event.first_visible_derivations
    )
    facts = [
        "事件级头皮持续变化候选："
        f"{time_origin}{event.onset_start_sec:.1f}–{event.onset_end_sec:.1f}秒，"
        f"固定算法最先检出的持续变化候选位于{edges}"
    ]
    if event.rhythm_state is not None:
        facts.append(f"表现为{_RHYTHM_STATES_ZH[event.rhythm_state]}")
    frequency = _frequency_phrase(event.frequency_range_hz)
    if frequency:
        facts.append(frequency)
    return "，".join(facts)


def _later_visible_phrase(
    *,
    delay_sec: float | None,
    derivations: tuple[str, ...],
    region_zh: str | None,
) -> str:
    if not derivations and region_zh is None:
        return ""
    destination = region_zh or "、".join(
        value.replace("-", "–") for value in derivations
    )
    timing = f"约{delay_sec:.1f}秒后" if delay_sec is not None else "随后"
    return (
        f"事件内{timing}在{destination}可见后续受累/范围扩展"
        "（仅描述头皮可见范围变化，不作为传播真值）"
    )


def _artifact_phrase(
    *,
    assessed: bool | None,
    artifact_types: tuple[str, ...],
    burden: float | None,
) -> str:
    if assessed is not True:
        return ""
    if not artifact_types:
        return "显式伪迹评估未提示预定义主导伪迹"
    names = "、".join(_ARTIFACT_ZH[value] for value in artifact_types)
    if burden is None:
        return f"显式伪迹评估记录到{names}伪迹"
    return f"显式伪迹评估记录到{names}伪迹，负担评分为{burden:.2f}"


def _montage_phrase(montage_stability: str | None) -> str:
    if montage_stability in (None, "not_assessed"):
        return ""
    return _MONTAGE_STABILITY_ZH[montage_stability]


def _uncertainty_phrases(
    uncertainty: UncertaintyDecomposition,
) -> tuple[str, ...]:
    labels = (
        ("排序歧义", uncertainty.ranking_ambiguity),
        ("患者内事件离散度", uncertainty.within_patient_event_dispersion),
        ("信号质量不确定性", uncertainty.signal_quality_uncertainty),
        ("蒙太奇不一致度", uncertainty.montage_disagreement),
        (
            "最终SOZ候选分数参考敏感性",
            uncertainty.final_score_reference_disagreement,
        ),
        ("模型认知不确定性", uncertainty.epistemic_uncertainty),
    )
    phrases = [f"{label}为{value:.2f}" for label, value in labels if value is not None]
    if uncertainty.abstain:
        reasons = "、".join(uncertainty.abstention_reason_codes)
        phrases.append(f"模型已拒答；原因代码：{reasons}")
    return tuple(phrases)


def _join_report_phrases(phrases: Sequence[str]) -> str:
    return "。".join(phrase.rstrip("。") for phrase in phrases if phrase) + "。"


def _render_v2_report(facts: ClinicalReportFactsV2) -> GroundedChineseDiagnosticReport:
    event = facts.event_phenotype
    receipt = event.receipt
    ranking = facts.patient_ranking
    spatial = ranking.spatial_report
    uncertainty = ranking.uncertainty

    if isinstance(event, EventScalpPhenotypeAbstention):
        reasons = "、".join(event.reason_codes)
        onset = (
            "事件级头皮持续变化候选：固定算法未形成可报告候选；"
            f"原因代码：{reasons}"
        )
        later = ""
        montage = ""
        artifact = ""
    else:
        onset = _event_phenotype_phrase(
            event,
            time_coordinate_semantics=receipt.time_coordinate_semantics,
        )
        later = _later_visible_phrase(
            delay_sec=event.later_visible_delay_sec,
            derivations=event.later_visible_derivations,
            region_zh=event.later_visible_region_zh,
        )
        montage = _montage_phrase(event.montage_stability)
        artifact = _artifact_phrase(
            assessed=event.artifact_assessed,
            artifact_types=event.artifact_types,
            burden=event.artifact_burden,
        )

    channels = "/".join(spatial.top_channels)
    tie = "并列首位候选" if len(spatial.top_channels) > 1 else "首位候选"
    location = _hypothesis_location_zh(spatial)
    aggregation = (
        f"患者级聚合：{_PATIENT_AGGREGATION_METHODS_ZH[ranking.aggregation_method]}，"
        f"共{ranking.aggregation_event_count}个发作事件"
        f"（{','.join(ranking.aggregation_event_ids)}），"
        f"aggregation_receipt_sha256={ranking.aggregation_receipt_sha256}"
    )
    if uncertainty.abstain:
        patient_ranking = (
            "患者级SOZ-reference排序已拒绝形成稳定结论；"
            f"当前{tie}为{channels}，"
            "仅保留供人工复核"
        )
        status = "ai_draft_v2_abstained_requires_clinician_confirmation"
    else:
        patient_ranking = (
            f"患者级SOZ-reference排序{tie}为{channels}；"
            f"头皮电极空间投影视图为{location}，"
            f"最高{spatial.score_semantics}={spatial.top_score:.4f}"
        )
        status = "ai_draft_v2_requires_clinician_confirmation"
    if isinstance(event, EventScalpPhenotypeAbstention):
        if uncertainty.abstain:
            status = (
                "ai_draft_v2_event_phenotype_and_patient_ranking_abstained_"
                "requires_clinician_confirmation"
            )
        else:
            status = (
                "ai_draft_v2_event_phenotype_abstained_"
                "requires_clinician_confirmation"
            )

    model_identity = (
        f"模型身份：{ranking.model_id}@{ranking.model_version}，"
        f"checkpoint_sha256={ranking.model_checkpoint_sha256}"
    )
    provenance = (
        f"事件证据收据：{receipt.schema_version}，event={receipt.event_pseudonym}，"
        f"evidence_sha256={receipt.evidence_artifact_sha256}，"
        "SOZ/private标签未用于事件证据生成"
    )
    later_region_receipt = facts.later_visible_region_receipt
    if later_region_receipt is not None:
        provenance += (
            "，later_visible_region_use=仅描述后续头皮可见范围、不作为传播真值，"
            "later_visible_region_receipt_sha256="
            f"{later_region_receipt.receipt_sha256}"
        )
    event_reference = facts.event_reference_consistency_receipt
    if event_reference is not None:
        provenance += (
            f"，event_reference_pair={event_reference.primary_arm_id}/"
            f"{event_reference.sensitivity_arm_id}，"
            "event_reference_use=仅用于事件表型事实/拒答、不用于SOZ评分，"
            "event_reference_receipt_sha256="
            f"{event_reference.receipt_sha256}"
        )
    reference_boundaries: list[str] = []
    reference = facts.reference_disagreement_receipt
    if reference is not None:
        provenance += (
            f"，reference_pair={reference.primary_arm_id}/{reference.sensitivity_arm_id}，"
            f"reference_metric={reference.metric_id}，"
            f"reference_receipt_sha256={reference.receipt_sha256}"
        )
        reference_boundaries.append(
            "参考稳定性边界：蒙太奇不一致度来自冻结LaBraM block-9的"
            f"19通道节点表征差异，并按{len(reference.aggregation_event_ids)}个"
            "发作事件等权聚合；仅用于参考稳定性与拒答；"
            "不是SOZ概率、定位准确率或预处理选臂依据"
        )
    final_score_reference = facts.final_score_reference_disagreement_receipt
    if final_score_reference is not None:
        provenance += (
            "，final_score_reference_pair="
            f"{final_score_reference.primary_arm_id}/"
            f"{final_score_reference.sensitivity_arm_id}，"
            "final_score_reference_metric="
            f"{final_score_reference.metric_id}，"
            "final_score_reference_receipt_sha256="
            f"{final_score_reference.receipt_sha256}"
        )
        top1_state = (
            "一致"
            if final_score_reference.top1_reference_agreement
            else "不一致"
        )
        reference_boundaries.append(
            "最终分数参考敏感性边界：同一冻结定位器、同一完整患者事件袋和"
            "固定18候选下，C-CAR19/C-REF19患者级最终分数softmax分布的"
            "normalized JSD="
            f"{final_score_reference.final_score_reference_disagreement:.4f}，"
            f"Top-1{top1_state}，Top-3 Jaccard="
            f"{final_score_reference.top3_reference_jaccard:.4f}；"
            "该事实与block-9表征距离相互独立，仅描述参考敏感性，"
            "不修改SOZ分数、排序或阈值"
        )
    reference_robustness = "；".join(reference_boundaries)
    if receipt.causal_prefix_safe:
        causal = "因果/前缀安全状态：证据收据声明为满足"
    else:
        causal = "因果/前缀安全状态：不满足，仅允许回顾性使用"
        status = status.replace("requires", "noncausal_requires", 1)
    uncertainty_rows = _uncertainty_phrases(uncertainty)
    limitation = (
        "患者级排序仅是临床参考头皮电极SOZ假设，不等同于侵入式皮层SOZ、"
        "致痫区或治疗靶点，不得单独作为手术决策依据，"
        "需由医生结合完整临床资料确认"
    )
    text = _join_report_phrases(
        (
            onset,
            later,
            montage,
            artifact,
            aggregation,
            patient_ranking,
            *uncertainty_rows,
            reference_robustness,
            provenance,
            model_identity,
            causal,
            limitation,
        )
    )
    return GroundedChineseDiagnosticReport(
        text=text,
        report_status=status,
        onset_phrase=onset,
        later_visible_phrase=later,
        localization_phrase=patient_ranking,
        limitation_phrase=limitation,
        event_phenotype_phrase=onset,
        patient_ranking_phrase=patient_ranking,
        provenance_phrase=provenance,
        causal_prefix_status=(
            "verified_prefix_safe" if receipt.causal_prefix_safe else "retrospective_only"
        ),
        model_identity=model_identity,
        event_aggregation_phrase=aggregation,
        uncertainty_phrases=uncertainty_rows,
        reference_robustness_phrase=reference_robustness,
        schema_version=CLINICAL_REPORT_SCHEMA_V2,
        migration_status="native_v2",
        evidence_receipt=receipt,
        aggregation_receipt_sha256=ranking.aggregation_receipt_sha256,
        later_visible_region_receipt_sha256=(
            None
            if later_region_receipt is None
            else later_region_receipt.receipt_sha256
        ),
        reference_disagreement_receipt_sha256=(
            None if reference is None else reference.receipt_sha256
        ),
        final_score_reference_disagreement_receipt_sha256=(
            None
            if final_score_reference is None
            else final_score_reference.receipt_sha256
        ),
        event_reference_consistency_receipt_sha256=(
            None if event_reference is None else event_reference.receipt_sha256
        ),
    )


def _render_legacy_report(
    spatial_report: DerivedSpatialReport,
    temporal: TemporalReportEvidence,
) -> GroundedChineseDiagnosticReport:
    """Explicit migration path for the former unreceipted event-level API."""

    edges = "及".join(
        value.replace("-", "–") for value in temporal.earliest_derivations
    )
    onset_parts = [
        "旧版事件级头皮持续变化候选："
        f"在{temporal.onset_start_sec:.1f}–{temporal.onset_end_sec:.1f}秒，"
        f"固定算法最先检出的持续信号变化候选位于{edges}"
    ]
    frequency = _frequency_phrase(temporal.frequency_range_hz)
    if frequency:
        onset_parts.append(frequency)
    onset = "，".join(onset_parts)
    later = _later_visible_phrase(
        delay_sec=temporal.spread_delay_sec,
        derivations=temporal.later_derivations,
        region_zh=temporal.later_region_zh,
    )
    montage = _montage_phrase(temporal.montage_stability)
    artifact = _artifact_phrase(
        assessed=temporal.artifact_assessed,
        artifact_types=temporal.artifact_types,
        burden=temporal.artifact_burden,
    )
    channels = "/".join(spatial_report.top_channels)
    if temporal.abstain:
        localization = (
            f"旧版事件级头皮定位排序已拒答；当前首位候选为{channels}，"
            "不可提升为患者级SOZ-reference结论"
        )
        status = (
            "ai_draft_legacy_unreceipted_abstained_requires_migration_"
            "and_clinician_confirmation"
        )
    else:
        localization = (
            f"旧版事件级头皮定位排序首位候选为{channels}；"
            "该旧接口不具备患者级SOZ-reference语义，"
            "且不得与最早可见导联等同"
        )
        status = (
            "ai_draft_legacy_unreceipted_requires_migration_"
            "and_clinician_confirmation"
        )
    uncertainty_rows = (
        (
            f"旧版未分解不确定性评分为{temporal.uncertainty_score:.2f}；"
            "迁移到v2前不可解释为任一不确定性分量"
        ),
    ) if temporal.uncertainty_score is not None else ()
    provenance = (
        "旧版输入无证据收据；事件ID、模型身份、患者级事件聚合"
        "及标签隔离均未验证"
    )
    causal = "因果/前缀安全状态：旧版接口未验证"
    limitation = (
        "旧版结果只允许作为事件级头皮定位草稿，"
        "不是患者级SOZ-reference排序，"
        "不等同于侵入式皮层SOZ、致痫区或治疗靶点"
    )
    text = _join_report_phrases(
        (
            onset,
            later,
            montage,
            artifact,
            localization,
            *uncertainty_rows,
            provenance,
            causal,
            limitation,
        )
    )
    return GroundedChineseDiagnosticReport(
        text=text,
        report_status=status,
        onset_phrase=onset,
        later_visible_phrase=later,
        localization_phrase=localization,
        limitation_phrase=limitation,
        event_phenotype_phrase=onset,
        patient_ranking_phrase="",
        provenance_phrase=provenance,
        causal_prefix_status="unverified_legacy",
        model_identity="unverified_legacy",
        event_aggregation_phrase="legacy_event_level_not_patient_aggregated",
        uncertainty_phrases=uncertainty_rows,
        schema_version="legacy_unreceipted_v1",
        migration_status="legacy_requires_v2_migration",
        evidence_receipt=None,
        aggregation_receipt_sha256=None,
    )


def render_grounded_chinese_diagnostic_report(
    facts_or_spatial: ClinicalReportFactsV2 | DerivedSpatialReport,
    temporal: TemporalReportEvidence | None = None,
) -> GroundedChineseDiagnosticReport:
    """Render facts-locked v2, or the explicitly marked legacy migration path.

    Native v2 accepts ``ClinicalReportFactsV2`` as its only argument.  The
    former ``(DerivedSpatialReport, TemporalReportEvidence)`` call remains
    executable but cannot silently acquire patient-level semantics.
    """

    if isinstance(facts_or_spatial, ClinicalReportFactsV2):
        if temporal is not None:
            raise TypeError("Native v2 rendering does not accept legacy temporal input")
        return _render_v2_report(facts_or_spatial)
    if not isinstance(facts_or_spatial, DerivedSpatialReport):
        raise TypeError(
            "First argument must be ClinicalReportFactsV2 or DerivedSpatialReport"
        )
    if not isinstance(temporal, TemporalReportEvidence):
        raise TypeError("Legacy rendering requires TemporalReportEvidence")
    return _render_legacy_report(facts_or_spatial, temporal)


def _single_event_output(output: ReasonerOutput) -> None:
    if not isinstance(output, ReasonerOutput):
        raise TypeError("Explanation input must be ReasonerOutput")
    if tuple(output.event_logits.shape) != (1, N_STANDARD_CHANNELS):
        raise ValueError("Grounded event explanation requires one [1,19] output")
    reconstructed = output.reconstructed_logits()
    if not torch.allclose(
        reconstructed,
        output.event_logits,
        atol=1e-6,
        rtol=1e-6,
    ):
        raise ValueError("Reasoner contribution receipt does not reconstruct logits")


def _incident_edge_rows(
    contribution: torch.Tensor, channel_index: int
) -> tuple[tuple[str, float], ...]:
    # contribution is [1,19,20,5].  Nonincident entries are guaranteed zero by
    # the reasoner, but endpoint membership is checked explicitly here.
    values = contribution[0, channel_index].sum(dim=-1).detach().cpu()
    channel = STANDARD_19[channel_index]
    rows = [
        (f"{left}-{right}", float(values[edge_index].item()))
        for edge_index, (left, right) in enumerate(TCP_20_EDGES)
        if channel in (left, right) and float(values[edge_index].item()) != 0.0
    ]
    rows.sort(key=lambda row: (-abs(row[1]), row[0]))
    return tuple(rows)


def explain_reasoner_channel(
    output: ReasonerOutput,
    channel: str,
) -> GroundedChannelExplanation:
    """Create one deterministic, numerically grounded event explanation."""

    _single_event_output(output)
    if channel not in CHANNEL_INDEX:
        raise ValueError("Explanation channel must be a canonical standard-19 name")
    index = CHANNEL_INDEX[channel]
    families = output.family_contributions()
    family_rows = tuple(
        (name, float(value[0, index].detach().cpu().item()))
        for name, value in families.items()
    )
    components = output.component_contributions()
    phase_rows = tuple(
        (name, float(value[0, index].detach().cpu().item()))
        for name, value in components.items()
        if name != "channel_prior"
    )
    morphology_edges = _incident_edge_rows(
        output.morphology_node_edge_phase_contribution, index
    )
    ictal_edges = _incident_edge_rows(
        output.ictal_node_edge_phase_contribution, index
    )
    incident_indices = [
        edge_index
        for edge_index, edge in enumerate(TCP_20_EDGES)
        if channel in edge
    ]
    quality = output.morphology_quality_gate[0, incident_indices].detach().cpu()
    quality_mean = float(quality.mean().item()) if quality.numel() else 0.0
    specificity = float(output.morphology_specificity_gate[0].detach().cpu().item())
    positive_support = sorted(
        ((name, value) for name, value in family_rows if name != "channel_prior"),
        key=lambda row: (-row[1], row[0]),
    )
    leading = tuple(name for name, value in positive_support if value > 0)[:2]
    support_text = (
        ", ".join(leading) if leading else "no positive evidence-family contribution"
    )
    statement = (
        f"{channel} has scalp-reference localization score "
        f"{float(output.event_logits[0, index].detach().cpu().item()):.4f}; "
        f"leading numerical support: {support_text}. "
        "Later-visible changes are evidence descriptors, not a proven propagation path."
    )
    return GroundedChannelExplanation(
        channel=channel,
        score=float(output.event_logits[0, index].detach().cpu().item()),
        family_contributions=family_rows,
        phase_contributions=phase_rows,
        morphology_incident_edges=morphology_edges,
        ictal_incident_edges=ictal_edges,
        morphology_quality_gate_mean=quality_mean,
        morphology_specificity_gate=specificity,
        statement=statement,
    )


def explain_top_reasoner_channels(
    output: ReasonerOutput,
    channel_evaluable_mask: torch.Tensor,
) -> tuple[DerivedSpatialReport, tuple[GroundedChannelExplanation, ...]]:
    """Report and explain every exact top-score tie for one event."""

    _single_event_output(output)
    report = derive_spatial_report(
        output.event_logits[0],
        channel_evaluable_mask,
        score_semantics="uncalibrated_localization_score",
    )
    explanations = tuple(
        explain_reasoner_channel(output, channel) for channel in report.top_channels
    )
    return report, explanations


def _patient_aggregation_mask(
    output: ReasonerOutput,
    aggregation_event_mask: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(output, ReasonerOutput):
        raise TypeError("Patient explanation input must be ReasonerOutput")
    if not isinstance(aggregation_event_mask, torch.Tensor):
        raise TypeError("aggregation_event_mask must be a torch tensor")
    event_count = int(output.event_logits.shape[0])
    if tuple(aggregation_event_mask.shape) != (event_count,):
        raise ValueError("aggregation_event_mask must have shape [E]")
    if aggregation_event_mask.dtype != torch.bool:
        raise TypeError("aggregation_event_mask must be torch.bool")
    if aggregation_event_mask.device != output.event_logits.device:
        raise ValueError("aggregation_event_mask and reasoner output must share a device")
    if not aggregation_event_mask.any().item():
        raise ValueError("Patient explanation requires at least one phase-valid event")
    if not torch.allclose(
        output.reconstructed_logits(),
        output.event_logits,
        atol=1e-6,
        rtol=1e-6,
    ):
        raise ValueError("Reasoner contribution receipt does not reconstruct logits")
    return aggregation_event_mask


def _patient_incident_edge_rows(
    contribution: torch.Tensor,
    aggregation_event_mask: torch.Tensor,
    channel_index: int,
) -> tuple[tuple[str, float], ...]:
    values = (
        contribution[aggregation_event_mask, channel_index]
        .mean(dim=0)
        .sum(dim=-1)
        .detach()
        .cpu()
    )
    channel = STANDARD_19[channel_index]
    rows = [
        (f"{left}-{right}", float(values[edge_index].item()))
        for edge_index, (left, right) in enumerate(TCP_20_EDGES)
        if channel in (left, right) and float(values[edge_index].item()) != 0.0
    ]
    rows.sort(key=lambda row: (-abs(row[1]), row[0]))
    return tuple(rows)


def explain_patient_reasoner_channel(
    output: ReasonerOutput,
    aggregation_event_mask: torch.Tensor,
    channel: str,
) -> GroundedPatientChannelExplanation:
    """Explain one patient logit using the exact equal-event aggregation rule."""

    usable = _patient_aggregation_mask(output, aggregation_event_mask)
    if channel not in CHANNEL_INDEX:
        raise ValueError("Explanation channel must be a canonical standard-19 name")
    index = CHANNEL_INDEX[channel]
    patient_logits = output.event_logits[usable].mean(dim=0)
    families = output.family_contributions()
    family_rows = tuple(
        (
            name,
            float(value[usable, index].mean().detach().cpu().item()),
        )
        for name, value in families.items()
    )
    phase_rows = tuple(
        (
            name,
            float(value[usable, index].mean().detach().cpu().item()),
        )
        for name, value in output.component_contributions().items()
        if name != "channel_prior"
    )
    morphology_edges = _patient_incident_edge_rows(
        output.morphology_node_edge_phase_contribution, usable, index
    )
    ictal_edges = _patient_incident_edge_rows(
        output.ictal_node_edge_phase_contribution, usable, index
    )
    incident_indices = [
        edge_index
        for edge_index, edge in enumerate(TCP_20_EDGES)
        if channel in edge
    ]
    quality = output.morphology_quality_gate[usable][:, incident_indices]
    quality_mean = float(quality.mean().detach().cpu().item()) if quality.numel() else 0.0
    specificity_mean = float(
        output.morphology_specificity_gate[usable].mean().detach().cpu().item()
    )
    support = sorted(
        ((name, value) for name, value in family_rows if name != "channel_prior"),
        key=lambda row: (-row[1], row[0]),
    )
    leading = tuple(name for name, value in support if value > 0)[:2]
    support_text = (
        ", ".join(leading)
        if leading
        else "no positive evidence-family contribution"
    )
    count = int(usable.sum().item())
    score = float(patient_logits[index].detach().cpu().item())
    statement = (
        f"{channel} has patient-level scalp-reference localization score "
        f"{score:.4f}, equally aggregated across {count} phase-valid seizure "
        f"event(s); leading numerical support: {support_text}. "
        "Later-visible changes are evidence descriptors, not a proven propagation path."
    )
    return GroundedPatientChannelExplanation(
        channel=channel,
        score=score,
        aggregation_event_count=count,
        family_contributions=family_rows,
        phase_contributions=phase_rows,
        morphology_incident_edges=morphology_edges,
        ictal_incident_edges=ictal_edges,
        morphology_quality_gate_mean=quality_mean,
        morphology_specificity_gate_mean=specificity_mean,
        statement=statement,
    )


def explain_top_patient_reasoner_channels(
    output: ReasonerOutput,
    aggregation_event_mask: torch.Tensor,
    channel_evaluable_mask: torch.Tensor,
) -> tuple[DerivedSpatialReport, tuple[GroundedPatientChannelExplanation, ...]]:
    """Report and explain patient-level top channels after equal-event pooling."""

    usable = _patient_aggregation_mask(output, aggregation_event_mask)
    patient_logits = output.event_logits[usable].mean(dim=0).detach()
    report = derive_spatial_report(
        patient_logits,
        channel_evaluable_mask,
        score_semantics="uncalibrated_localization_score",
    )
    explanations = tuple(
        explain_patient_reasoner_channel(output, usable, channel)
        for channel in report.top_channels
    )
    return report, explanations


__all__ = [
    "CLAIM_BOUNDARY",
    "CLINICAL_REPORT_SCHEMA_V2",
    "CLINICAL_SCALP_REGIONS",
    "ClinicalReportFactsV2",
    "DerivedSpatialReport",
    "EVIDENCE_RECEIPT_SCHEMA_V2",
    "EvidenceProvenanceReceipt",
    "EventScalpPhenotypeAbstention",
    "EventScalpPhenotypeEvidence",
    "GroundedChannelExplanation",
    "GroundedPatientChannelExplanation",
    "GroundedChineseDiagnosticReport",
    "LATERALITY_GROUPS",
    "LATER_VISIBLE_REGIONS_ZH",
    "OPERATIONAL_SENSOR_GROUPS",
    "SCORE_SEMANTICS",
    "SpatialViewRow",
    "PatientSOZReferenceRanking",
    "ReferenceDisagreementReceipt",
    "TemporalReportEvidence",
    "UncertaintyDecomposition",
    "derive_spatial_report",
    "derive_view_targets",
    "explain_reasoner_channel",
    "explain_patient_reasoner_channel",
    "explain_top_reasoner_channels",
    "explain_top_patient_reasoner_channels",
    "render_grounded_chinese_diagnostic_report",
]
