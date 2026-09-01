"""Permission-locked multi-reference scalp-field evidence primitive.

This module sits downstream of a reference-specific BA-IEG field head and
upstream of any event/record SOZ reasoner.  It does not read EEG files,
annotations, spreadsheets, doctor labels or report text.  Its inputs are
already-bound, reference-specific *research candidates* over one physical
event.  The primitive makes five operations explicit and replayable:

* removal of imputed, unobserved, quality-failed or under-covered units;
* removal of offline/future-dependent and later-involvement candidates from
  the positive onset route;
* rank-only comparison across referential, TCP bipolar, CAR and Laplacian
  views (raw scores are never treated as cross-reference probabilities);
* interval-valued earliest-distinguishable-set construction; and
* an electrode -> region -> laterality -> phenotype-only resolution ladder.

A bipolar row is always one signed analysis edge.  It may contribute a coarse
field laterality/region when both endpoints are known, but it is never split
into endpoint-electrode evidence.  CAR/Laplacian target identities are used
only as scalp analysis coordinates and never as source-localisation claims.

The returned stability decisions are deterministic engineering gates under a
content-addressed policy.  They are not calibrated probabilities, clinical
qualification, cortical SOZ, epileptogenic-zone or surgical-target claims.
Nothing in this module is connected to a production report route.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import itertools
import json
import math
import statistics
from typing import Any, Final, Mapping, Sequence

from src.soz.geometry import CHANNEL_INDEX, STANDARD_19


BA_IEG_MULTIREFERENCE_FIELD_SCHEMA_VERSION: Final[str] = (
    "ba_ieg_multireference_scalp_field_evidence_v1"
)
BA_IEG_MULTIREFERENCE_FIELD_ID: Final[str] = (
    "ba_ieg_permission_locked_rank_multireference_scalp_field_v1"
)
BA_IEG_MULTIREFERENCE_RESOLUTION_LADDER: Final[tuple[str, ...]] = (
    "electrode",
    "region",
    "laterality",
)
BA_IEG_MULTIREFERENCE_REFERENCE_FAMILIES: Final[frozenset[str]] = frozenset(
    {"referential", "bipolar", "common_average", "laplacian"}
)

_TEMPORAL_ROLES = frozenset(
    {"onset_causal", "context_offline", "morphology_native"}
)
_INTRINSIC_ROLES = frozenset(
    {"onset_eligible", "later_involvement", "context_only", "limitation"}
)
_POLARITIES = frozenset(
    {"positive", "negative", "biphasic", "indeterminate"}
)
_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_LEFT_ELECTRODES = frozenset(
    {"FP1", "F7", "F3", "T7", "C3", "P7", "P3", "O1"}
)
_RIGHT_ELECTRODES = frozenset(
    {"FP2", "F8", "F4", "T8", "C4", "P8", "P4", "O2"}
)
_MIDLINE_ELECTRODES = frozenset({"FZ", "CZ", "PZ"})


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _identifier(value: object, name: str) -> str:
    text = str(value)
    if not text or text != text.strip():
        raise ValueError(f"{name} must be a non-empty trimmed identifier")
    return text


def _sha256(value: object, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in _SHA256_CHARACTERS for character in text):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return text


def _finite_interval(
    value: Sequence[float], name: str, *, allow_zero_duration: bool = False
) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{name} must be a two-item interval")
    start, stop = float(value[0]), float(value[1])
    minimum_relation = stop >= start if allow_zero_duration else stop > start
    if not math.isfinite(start) or not math.isfinite(stop) or not minimum_relation:
        raise ValueError(f"{name} must be finite and ordered")
    return start, stop


def _lead_endpoints(unit_id: str) -> tuple[str, str]:
    parts = unit_id.upper().split("-")
    if len(parts) != 2 or any(item not in CHANNEL_INDEX for item in parts):
        raise ValueError(
            "bipolar analysis_unit_id must encode exactly two standard-19 endpoints"
        )
    if parts[0] == parts[1]:
        raise ValueError("bipolar analysis edge cannot repeat one endpoint")
    return parts[0], parts[1]


def _electrode_laterality(electrode: str) -> str:
    if electrode in _LEFT_ELECTRODES:
        return "left"
    if electrode in _RIGHT_ELECTRODES:
        return "right"
    if electrode in _MIDLINE_ELECTRODES:
        return "midline"
    raise ValueError(f"unknown standard-19 electrode: {electrode}")


def _electrode_region(electrode: str) -> str:
    # The frozen scalp-region projection matches the deterministic Findings
    # baseline.  It is deliberately coarse and is not an anatomical source
    # atlas.  Lateral F7/P7 and F8/P8 remain in the temporal scalp group.
    if electrode in {"F7", "T7", "P7", "F8", "T8", "P8"}:
        return "temporal"
    if electrode.startswith("FP") or electrode.startswith("F"):
        return "frontal"
    if electrode.startswith("C"):
        return "central"
    if electrode.startswith("P"):
        return "parietal"
    if electrode.startswith("O"):
        return "occipital"
    return "other"


def _lead_laterality(endpoints: tuple[str, str]) -> str:
    values = {_electrode_laterality(item) for item in endpoints}
    non_midline = values - {"midline"}
    if non_midline == {"left"}:
        return "left"
    if non_midline == {"right"}:
        return "right"
    if not non_midline:
        return "midline"
    if non_midline == {"left", "right"}:
        return "bilateral"
    return "indeterminate"


def _candidate_laterality(candidate: "BAIEGReferenceSpecificFieldCandidate") -> str:
    if candidate.analysis_unit_type == "lead":
        return _lead_laterality(_lead_endpoints(candidate.analysis_unit_id))
    target = candidate.physical_target_electrode_id
    if target is None:  # guarded in the candidate contract
        return "indeterminate"
    return _electrode_laterality(target)


def _candidate_region(candidate: "BAIEGReferenceSpecificFieldCandidate") -> str:
    laterality = _candidate_laterality(candidate)
    if candidate.analysis_unit_type == "lead":
        endpoints = _lead_endpoints(candidate.analysis_unit_id)
        component_regions = {_electrode_region(item) for item in endpoints}
        base = (
            next(iter(component_regions))
            if len(component_regions) == 1
            else "multiregional"
        )
    else:
        target = candidate.physical_target_electrode_id
        if target is None:  # guarded in the candidate contract
            base = "indeterminate"
        else:
            base = _electrode_region(target)
    prefix = laterality if laterality != "indeterminate" else "indeterminate"
    return f"{prefix}_{base}"


@dataclass(frozen=True)
class BAIEGReferenceSpecificFieldCandidate:
    """One reference-specific onset-field candidate over a physical event.

    ``onset_association_score`` is an uncalibrated within-view ranking score.
    Its magnitude is never compared across references.  ``signed_reference_row``
    uses the frozen ``STANDARD_19`` basis.  A target-coordinate row must have
    a strictly dominant positive coefficient at its declared target; a
    bipolar row must remain an equal-magnitude two-endpoint difference.
    """

    candidate_id: str
    event_id: str
    recording_id: str
    analysis_interval_seconds: tuple[float, float]
    canonical_receipt_sha256: str
    adaptive_window_receipt_sha256: str
    source_input_batch_sha256: str
    source_field_head_receipt_sha256: str
    view_id: str
    view_receipt_sha256: str
    view_transform_sha256: str
    temporal_evidence_sha256: str
    reference_family: str
    analysis_unit_id: str
    analysis_unit_type: str
    signed_reference_row: tuple[float, ...]
    physical_target_electrode_id: str | None
    onset_interval_seconds: tuple[float, float] | None
    onset_association_score: float
    polarity: str
    coverage_fraction: float
    observed: bool
    imputed: bool
    evidence_eligible: bool
    quality_pass: bool
    temporal_role: str
    intrinsic_evidence_role: str
    future_sample_access: bool
    onset_evidence_authorized: bool
    reference_row_sha256: str = field(init=False)
    candidate_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "candidate_id",
            "event_id",
            "recording_id",
            "view_id",
            "analysis_unit_id",
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        for name in (
            "canonical_receipt_sha256",
            "adaptive_window_receipt_sha256",
            "source_input_batch_sha256",
            "source_field_head_receipt_sha256",
            "view_receipt_sha256",
            "view_transform_sha256",
            "temporal_evidence_sha256",
        ):
            _sha256(getattr(self, name), name)

        analysis_interval = _finite_interval(
            self.analysis_interval_seconds, "analysis_interval_seconds"
        )
        if analysis_interval[0] < 0.0:
            raise ValueError("recording-relative analysis interval cannot be negative")
        onset_interval = (
            None
            if self.onset_interval_seconds is None
            else _finite_interval(self.onset_interval_seconds, "onset_interval_seconds")
        )
        if onset_interval is not None and (
            onset_interval[0] < analysis_interval[0] - 1e-9
            or onset_interval[1] > analysis_interval[1] + 1e-9
        ):
            raise ValueError("onset candidate interval lies outside the physical event")
        object.__setattr__(self, "analysis_interval_seconds", analysis_interval)
        object.__setattr__(self, "onset_interval_seconds", onset_interval)

        if self.reference_family not in BA_IEG_MULTIREFERENCE_REFERENCE_FAMILIES:
            raise ValueError("multi-reference field family is unsupported")
        if self.analysis_unit_type not in {"electrode", "lead", "virtual"}:
            raise ValueError("analysis_unit_type is unsupported")
        if self.temporal_role not in _TEMPORAL_ROLES:
            raise ValueError("temporal_role is unsupported")
        if self.intrinsic_evidence_role not in _INTRINSIC_ROLES:
            raise ValueError("intrinsic_evidence_role is unsupported")
        if self.polarity not in _POLARITIES:
            raise ValueError("polarity is unsupported")
        for name in (
            "observed",
            "imputed",
            "evidence_eligible",
            "quality_pass",
            "future_sample_access",
            "onset_evidence_authorized",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be boolean")
        if self.observed and self.imputed:
            raise ValueError("an analysis unit cannot be observed and imputed")

        coverage = float(self.coverage_fraction)
        score = float(self.onset_association_score)
        if not math.isfinite(coverage) or not 0.0 <= coverage <= 1.0:
            raise ValueError("coverage_fraction must lie in [0,1]")
        if not math.isfinite(score):
            raise ValueError("onset_association_score must be finite")
        object.__setattr__(self, "coverage_fraction", coverage)
        object.__setattr__(self, "onset_association_score", score)

        if self.temporal_role == "onset_causal":
            if self.future_sample_access or not self.onset_evidence_authorized:
                raise ValueError(
                    "onset-causal candidate must be future-free and onset-authorized"
                )
        elif self.temporal_role == "context_offline":
            if not self.future_sample_access or self.onset_evidence_authorized:
                raise ValueError(
                    "offline-context candidate must declare future access and be onset-ineligible"
                )
        elif self.future_sample_access or self.onset_evidence_authorized:
            raise ValueError(
                "native-morphology candidate must be instantaneous and onset-ineligible"
            )

        row = tuple(float(value) for value in self.signed_reference_row)
        if len(row) != len(STANDARD_19) or any(not math.isfinite(value) for value in row):
            raise ValueError("signed_reference_row must be finite on STANDARD_19")
        if max(abs(value) for value in row) <= 0.0:
            raise ValueError("signed_reference_row cannot be all zero")
        object.__setattr__(self, "signed_reference_row", row)

        target = self.physical_target_electrode_id
        target = None if target is None else str(target).strip().upper()
        if target is not None and target not in CHANNEL_INDEX:
            raise ValueError("physical target must belong to STANDARD_19")
        object.__setattr__(self, "physical_target_electrode_id", target)

        nonzero = [index for index, value in enumerate(row) if abs(value) > 1e-8]
        if self.reference_family == "bipolar":
            if self.analysis_unit_type != "lead" or target is not None:
                raise ValueError(
                    "bipolar evidence must remain a lead with no endpoint target"
                )
            endpoints = _lead_endpoints(self.analysis_unit_id)
            expected = {CHANNEL_INDEX[item] for item in endpoints}
            if set(nonzero) != expected or len(nonzero) != 2:
                raise ValueError("bipolar reference row disagrees with its analysis edge")
            values = [row[index] for index in nonzero]
            if (
                values[0] * values[1] >= 0.0
                or not math.isclose(abs(values[0]), abs(values[1]), rel_tol=1e-6)
                or not math.isclose(sum(values), 0.0, abs_tol=1e-6)
            ):
                raise ValueError("bipolar row must be an equal signed difference")
        else:
            expected_type = "electrode" if self.reference_family == "referential" else "virtual"
            if self.analysis_unit_type != expected_type or target is None:
                raise ValueError(
                    "referential/CAR/Laplacian evidence needs its canonical scalp target"
                )
            target_value = row[CHANNEL_INDEX[target]]
            other_maximum = max(
                (abs(value) for index, value in enumerate(row) if index != CHANNEL_INDEX[target]),
                default=0.0,
            )
            if target_value <= 0.0 or abs(target_value) <= other_maximum + 1e-8:
                raise ValueError(
                    "declared scalp target must have the strictly dominant positive coefficient"
                )
            if self.reference_family in {"common_average", "laplacian"} and not math.isclose(
                sum(row), 0.0, abs_tol=1e-6
            ):
                raise ValueError("CAR/Laplacian rows must reject the common offset")

        reference_row_sha256 = _canonical_sha256(
            {"physical_electrode_ids": list(STANDARD_19), "signed_row": list(row)}
        )
        object.__setattr__(self, "reference_row_sha256", reference_row_sha256)
        object.__setattr__(self, "candidate_sha256", _canonical_sha256(self._body()))

    def _body(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "event_id": self.event_id,
            "recording_id": self.recording_id,
            "analysis_interval_seconds": list(self.analysis_interval_seconds),
            "canonical_receipt_sha256": self.canonical_receipt_sha256,
            "adaptive_window_receipt_sha256": self.adaptive_window_receipt_sha256,
            "source_input_batch_sha256": self.source_input_batch_sha256,
            "source_field_head_receipt_sha256": self.source_field_head_receipt_sha256,
            "view_id": self.view_id,
            "view_receipt_sha256": self.view_receipt_sha256,
            "view_transform_sha256": self.view_transform_sha256,
            "temporal_evidence_sha256": self.temporal_evidence_sha256,
            "reference_family": self.reference_family,
            "analysis_unit_id": self.analysis_unit_id,
            "analysis_unit_type": self.analysis_unit_type,
            "reference_row_sha256": self.reference_row_sha256,
            "physical_target_electrode_id": self.physical_target_electrode_id,
            "onset_interval_seconds": (
                None
                if self.onset_interval_seconds is None
                else list(self.onset_interval_seconds)
            ),
            "onset_association_score": self.onset_association_score,
            "score_semantics": "within_view_uncalibrated_onset_association",
            "polarity": self.polarity,
            "polarity_coordinate": "canonical_signed_analysis_unit_output",
            "coverage_fraction": self.coverage_fraction,
            "observed": self.observed,
            "imputed": self.imputed,
            "evidence_eligible": self.evidence_eligible,
            "quality_pass": self.quality_pass,
            "temporal_role": self.temporal_role,
            "intrinsic_evidence_role": self.intrinsic_evidence_role,
            "future_sample_access": self.future_sample_access,
            "onset_evidence_authorized": self.onset_evidence_authorized,
        }

    def to_dict(self) -> dict[str, Any]:
        result = self._body()
        result["candidate_sha256"] = self.candidate_sha256
        return result


@dataclass(frozen=True)
class BAIEGMultireferenceFieldPolicy:
    """Frozen, non-clinical engineering policy for reference stability."""

    minimum_coverage_fraction: float = 0.80
    earliest_interval_tolerance_seconds: float = 0.25
    minimum_reference_families: int = 2
    top_k: int = 3
    minimum_top_consensus_fraction: float = 2.0 / 3.0
    minimum_mean_top_k_jaccard: float = 0.50
    maximum_rank_reversal_fraction: float = 0.25
    score_tie_tolerance: float = 1e-9
    maximum_rank_reversal_examples: int = 12
    policy_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "minimum_coverage_fraction",
            "minimum_top_consensus_fraction",
            "minimum_mean_top_k_jaccard",
            "maximum_rank_reversal_fraction",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0,1]")
            object.__setattr__(self, name, value)
        for name in ("earliest_interval_tolerance_seconds", "score_tie_tolerance"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        if self.minimum_reference_families < 2:
            raise ValueError("multi-reference stability needs at least two families")
        if self.top_k < 1 or self.maximum_rank_reversal_examples < 1:
            raise ValueError("top_k and reversal example cap must be positive")
        object.__setattr__(self, "policy_sha256", _canonical_sha256(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "minimum_coverage_fraction": self.minimum_coverage_fraction,
            "earliest_interval_tolerance_seconds": self.earliest_interval_tolerance_seconds,
            "minimum_reference_families": self.minimum_reference_families,
            "top_k": self.top_k,
            "minimum_top_consensus_fraction": self.minimum_top_consensus_fraction,
            "minimum_mean_top_k_jaccard": self.minimum_mean_top_k_jaccard,
            "maximum_rank_reversal_fraction": self.maximum_rank_reversal_fraction,
            "score_tie_tolerance": self.score_tie_tolerance,
            "maximum_rank_reversal_examples": self.maximum_rank_reversal_examples,
            "stability_semantics": (
                "deterministic_rank_gate_not_probability_or_clinical_qualification"
            ),
        }


DEFAULT_BA_IEG_MULTIREFERENCE_FIELD_POLICY: Final[
    BAIEGMultireferenceFieldPolicy
] = BAIEGMultireferenceFieldPolicy()


def _eligibility_reasons(
    candidate: BAIEGReferenceSpecificFieldCandidate,
    policy: BAIEGMultireferenceFieldPolicy,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if not candidate.observed:
        reasons.append("analysis_unit_unobserved")
    if candidate.imputed:
        reasons.append("analysis_unit_imputed")
    if not candidate.evidence_eligible:
        reasons.append("spatial_field_family_ineligible")
    if not candidate.quality_pass:
        reasons.append("quality_gate_failed")
    if candidate.coverage_fraction + 1e-12 < policy.minimum_coverage_fraction:
        reasons.append("coverage_below_policy")
    if candidate.temporal_role != "onset_causal":
        reasons.append("temporal_role_not_onset_causal")
    if candidate.future_sample_access:
        reasons.append("future_sample_access_forbidden_for_positive_onset")
    if not candidate.onset_evidence_authorized:
        reasons.append("onset_evidence_not_authorized")
    if candidate.intrinsic_evidence_role != "onset_eligible":
        reasons.append("intrinsic_role_not_onset_eligible")
    if candidate.onset_interval_seconds is None:
        reasons.append("onset_interval_not_available")
    return tuple(reasons)


def _dense_ranks(
    values: Mapping[str, float], *, tolerance: float
) -> dict[str, int]:
    ordered = sorted(values, key=lambda key: (-float(values[key]), key))
    result: dict[str, int] = {}
    previous: float | None = None
    rank = 0
    for key in ordered:
        score = float(values[key])
        if previous is None or abs(score - previous) > tolerance:
            rank += 1
            previous = score
        result[key] = rank
    return result


def _spatial_key(
    candidate: BAIEGReferenceSpecificFieldCandidate, resolution: str
) -> str | None:
    if resolution == "electrode":
        if candidate.analysis_unit_type == "lead":
            return None
        return candidate.physical_target_electrode_id
    if resolution == "region":
        return _candidate_region(candidate)
    if resolution == "laterality":
        return _candidate_laterality(candidate)
    raise ValueError(f"unsupported spatial resolution: {resolution}")


def _physical_identity(
    candidate: BAIEGReferenceSpecificFieldCandidate,
) -> tuple[str, str]:
    if candidate.analysis_unit_type == "lead":
        return "lead", candidate.analysis_unit_id.upper()
    target = candidate.physical_target_electrode_id
    if target is None:  # guarded in candidate validation
        return "virtual", candidate.analysis_unit_id
    return "electrode", target


def _view_candidate_ranks(
    eligible: Sequence[BAIEGReferenceSpecificFieldCandidate],
    policy: BAIEGMultireferenceFieldPolicy,
) -> dict[str, int]:
    grouped: dict[str, dict[str, float]] = defaultdict(dict)
    for candidate in eligible:
        grouped[candidate.view_id][candidate.candidate_id] = (
            candidate.onset_association_score
        )
    result: dict[str, int] = {}
    for scores in grouped.values():
        result.update(_dense_ranks(scores, tolerance=policy.score_tie_tolerance))
    return result


def _rank_reversal_audit(
    family_ranks: Mapping[str, Mapping[str, int]],
    policy: BAIEGMultireferenceFieldPolicy,
) -> tuple[list[dict[str, Any]], int, int]:
    audits: list[dict[str, Any]] = []
    total_comparable = 0
    total_reversals = 0
    for first, second in itertools.combinations(sorted(family_ranks), 2):
        first_ranks = family_ranks[first]
        second_ranks = family_ranks[second]
        common = sorted(set(first_ranks).intersection(second_ranks))
        comparable = 0
        reversals = 0
        examples: list[dict[str, str]] = []
        for left, right in itertools.combinations(common, 2):
            first_delta = first_ranks[left] - first_ranks[right]
            second_delta = second_ranks[left] - second_ranks[right]
            if first_delta == 0 or second_delta == 0:
                continue
            comparable += 1
            if first_delta * second_delta < 0:
                reversals += 1
                if len(examples) < policy.maximum_rank_reversal_examples:
                    examples.append(
                        {
                            "higher_in_first_family": (
                                left if first_delta < 0 else right
                            ),
                            "higher_in_second_family": (
                                left if second_delta < 0 else right
                            ),
                        }
                    )
        total_comparable += comparable
        total_reversals += reversals
        audits.append(
            {
                "first_reference_family": first,
                "second_reference_family": second,
                "shared_spatial_key_count": len(common),
                "comparable_spatial_key_pair_count": comparable,
                "rank_reversal_count": reversals,
                "rank_reversal_fraction": (
                    None if comparable == 0 else reversals / comparable
                ),
                "rank_reversal_examples": examples,
            }
        )
    return audits, total_comparable, total_reversals


def _resolution_projection(
    eligible: Sequence[BAIEGReferenceSpecificFieldCandidate],
    *,
    resolution: str,
    earliest_candidate_ids: frozenset[str],
    policy: BAIEGMultireferenceFieldPolicy,
) -> dict[str, Any]:
    view_ranks = _view_candidate_ranks(eligible, policy)
    by_family_view: dict[
        str, dict[str, list[BAIEGReferenceSpecificFieldCandidate]]
    ] = defaultdict(lambda: defaultdict(list))
    mapped_sources: dict[str, list[BAIEGReferenceSpecificFieldCandidate]] = defaultdict(list)
    for candidate in eligible:
        key = _spatial_key(candidate, resolution)
        if key is None or key == "indeterminate":
            continue
        by_family_view[candidate.reference_family][candidate.view_id].append(candidate)
        mapped_sources[key].append(candidate)

    family_scores: dict[str, dict[str, float]] = {}
    family_source_ids: dict[str, dict[str, list[str]]] = {}
    family_view_ids: dict[str, list[str]] = {}
    for family, view_groups in sorted(by_family_view.items()):
        mapped_views: list[tuple[str, dict[str, int], dict[str, list[str]]]] = []
        for view_id, candidates in sorted(view_groups.items()):
            best_rank: dict[str, int] = {}
            source_ids: dict[str, list[str]] = defaultdict(list)
            for candidate in candidates:
                key = _spatial_key(candidate, resolution)
                if key is None or key == "indeterminate":
                    continue
                rank = view_ranks[candidate.candidate_id]
                best_rank[key] = min(best_rank.get(key, rank), rank)
                source_ids[key].append(candidate.candidate_id)
            if best_rank:
                mapped_views.append((view_id, best_rank, source_ids))
        if not mapped_views:
            continue
        keys = sorted({key for _, ranks, _ in mapped_views for key in ranks})
        family_scores[family] = {
            key: sum(
                0.0 if key not in ranks else 1.0 / ranks[key]
                for _, ranks, _ in mapped_views
            )
            / len(mapped_views)
            for key in keys
        }
        family_source_ids[family] = {
            key: sorted(
                {
                    source_id
                    for _, _, sources in mapped_views
                    for source_id in sources.get(key, [])
                }
            )
            for key in keys
        }
        family_view_ids[family] = [item[0] for item in mapped_views]

    family_ranks = {
        family: _dense_ranks(scores, tolerance=policy.score_tie_tolerance)
        for family, scores in family_scores.items()
    }
    family_rankings = []
    for family in sorted(family_ranks):
        ranks = family_ranks[family]
        family_rankings.append(
            {
                "reference_family": family,
                "view_ids": family_view_ids[family],
                "ranking": [
                    {
                        "spatial_key": key,
                        "dense_rank": ranks[key],
                        "mean_reciprocal_within_view_rank": family_scores[family][key],
                        "source_candidate_ids": family_source_ids[family][key],
                    }
                    for key in sorted(
                        ranks,
                        key=lambda item: (
                            ranks[item],
                            -family_scores[family][item],
                            item,
                        ),
                    )
                ],
            }
        )

    families = sorted(family_ranks)
    top_sets = {
        family: {key for key, rank in family_ranks[family].items() if rank == 1}
        for family in families
    }
    top_support = Counter(key for values in top_sets.values() for key in values)
    consensus_key = None
    top_consensus_fraction = 0.0
    if top_support and families:
        consensus_key = sorted(
            top_support,
            key=lambda key: (
                -top_support[key],
                -sum(family_scores[family].get(key, 0.0) for family in families),
                key,
            ),
        )[0]
        top_consensus_fraction = top_support[consensus_key] / len(families)

    pairwise_top_k: list[dict[str, Any]] = []
    jaccards: list[float] = []
    for first, second in itertools.combinations(families, 2):
        first_set = {
            key
            for key, rank in family_ranks[first].items()
            if rank <= policy.top_k
        }
        second_set = {
            key
            for key, rank in family_ranks[second].items()
            if rank <= policy.top_k
        }
        union = first_set.union(second_set)
        jaccard = (
            1.0 if not union else len(first_set.intersection(second_set)) / len(union)
        )
        jaccards.append(jaccard)
        pairwise_top_k.append(
            {
                "first_reference_family": first,
                "second_reference_family": second,
                "top_k": policy.top_k,
                "intersection": sorted(first_set.intersection(second_set)),
                "union": sorted(union),
                "jaccard": jaccard,
            }
        )
    mean_jaccard = None if not jaccards else sum(jaccards) / len(jaccards)

    reversal_audits, comparable_pairs, reversals = _rank_reversal_audit(
        family_ranks, policy
    )
    reversal_fraction = (
        None if comparable_pairs == 0 else reversals / comparable_pairs
    )

    stability_reasons: list[str] = []
    if len(families) < policy.minimum_reference_families:
        stability_reasons.append("insufficient_independent_reference_families")
    if top_consensus_fraction + 1e-12 < policy.minimum_top_consensus_fraction:
        stability_reasons.append("top_rank_consensus_below_policy")
    if (
        mean_jaccard is not None
        and mean_jaccard + 1e-12 < policy.minimum_mean_top_k_jaccard
    ):
        stability_reasons.append("top_k_field_overlap_below_policy")
    if (
        reversal_fraction is not None
        and reversal_fraction - 1e-12 > policy.maximum_rank_reversal_fraction
    ):
        stability_reasons.append("rank_reversal_above_policy")

    stable = not stability_reasons
    if not family_ranks or len(families) < policy.minimum_reference_families:
        status = "not_evaluable"
    else:
        status = "qualified" if stable else "unstable"

    all_keys = sorted({key for ranks in family_ranks.values() for key in ranks})
    consensus_scores = {
        key: sum(
            0.0
            if key not in family_ranks[family]
            else 1.0 / family_ranks[family][key]
            for family in families
        )
        / len(families)
        for key in all_keys
    } if families else {}
    consensus_ranks = _dense_ranks(
        consensus_scores, tolerance=policy.score_tie_tolerance
    ) if consensus_scores else {}
    ranking: list[dict[str, Any]] = []
    for key in sorted(
        all_keys,
        key=lambda item: (consensus_ranks[item], -consensus_scores[item], item),
    ):
        sources = sorted(
            mapped_sources[key], key=lambda candidate: candidate.candidate_id
        )
        intervals = [
            candidate.onset_interval_seconds
            for candidate in sources
            if candidate.onset_interval_seconds is not None
        ]
        source_types = sorted({candidate.analysis_unit_type for candidate in sources})
        source_units = sorted({candidate.analysis_unit_id for candidate in sources})
        source_ids = sorted({candidate.candidate_id for candidate in sources})
        ranking.append(
            {
                "spatial_key": key,
                "consensus_dense_rank": consensus_ranks[key],
                "mean_reciprocal_family_rank": consensus_scores[key],
                "family_dense_ranks": {
                    family: family_ranks[family].get(key) for family in families
                },
                "top_k_support_reference_families": [
                    family
                    for family in families
                    if family_ranks[family].get(key, policy.top_k + 1) <= policy.top_k
                ],
                "source_candidate_ids": source_ids,
                "source_analysis_unit_ids": source_units,
                "source_analysis_unit_types": source_types,
                "reference_family_count": len(
                    {candidate.reference_family for candidate in sources}
                ),
                "onset_interval_envelope_seconds": (
                    None
                    if not intervals
                    else [
                        min(interval[0] for interval in intervals),
                        max(interval[1] for interval in intervals),
                    ]
                ),
                "earliest_distinguishable": bool(
                    earliest_candidate_ids.intersection(source_ids)
                ),
                "coverage_fraction_median": (
                    None
                    if not sources
                    else statistics.median(
                        candidate.coverage_fraction for candidate in sources
                    )
                ),
            }
        )

    audit_notes: list[str] = []
    if resolution == "electrode" and any(
        candidate.analysis_unit_type == "lead" for candidate in eligible
    ):
        audit_notes.append("bipolar_edges_retained_as_leads_not_electrodes")
    return {
        "resolution": resolution,
        "status": status,
        "candidate_ranking": ranking,
        "eligible_reference_families": families,
        "family_rankings": family_rankings,
        "reference_stability": {
            "stable_under_policy": stable,
            "independent_reference_family_count": len(families),
            "consensus_top_spatial_key": consensus_key,
            "top_rank_consensus_fraction": top_consensus_fraction,
            "mean_pairwise_top_k_jaccard": mean_jaccard,
            "aggregate_comparable_spatial_key_pair_count": comparable_pairs,
            "aggregate_rank_reversal_count": reversals,
            "aggregate_rank_reversal_fraction": reversal_fraction,
            "pairwise_top_k_field_overlap": pairwise_top_k,
            "pairwise_rank_reversal": reversal_audits,
            "score_fusion_semantics": (
                "mean_reciprocal_dense_rank_not_cross_reference_probability"
            ),
        },
        "degradation_reason_codes": stability_reasons,
        "audit_note_codes": audit_notes,
        "bipolar_endpoint_attribution_performed": False,
    }


def _earliest_distinguishable_set(
    eligible: Sequence[BAIEGReferenceSpecificFieldCandidate],
    policy: BAIEGMultireferenceFieldPolicy,
) -> tuple[dict[str, Any], frozenset[str]]:
    intervals = [
        candidate.onset_interval_seconds
        for candidate in eligible
        if candidate.onset_interval_seconds is not None
    ]
    if not intervals:
        return (
            {
                "status": "not_evaluable",
                "selection_rule": "interval_possible_earliest_set_v1",
                "earliest_upper_bound_seconds": None,
                "tolerance_seconds": policy.earliest_interval_tolerance_seconds,
                "members": [],
                "bipolar_endpoint_attribution_performed": False,
            },
            frozenset(),
        )
    earliest_upper = min(interval[1] for interval in intervals)
    cutoff = earliest_upper + policy.earliest_interval_tolerance_seconds
    selected = [
        candidate
        for candidate in eligible
        if candidate.onset_interval_seconds is not None
        and candidate.onset_interval_seconds[0] <= cutoff + 1e-12
    ]
    selected_ids = frozenset(candidate.candidate_id for candidate in selected)
    grouped: dict[
        tuple[str, str], list[BAIEGReferenceSpecificFieldCandidate]
    ] = defaultdict(list)
    for candidate in selected:
        grouped[_physical_identity(candidate)].append(candidate)

    members: list[dict[str, Any]] = []
    for (identity_type, identity), rows in grouped.items():
        candidate_intervals = [
            candidate.onset_interval_seconds
            for candidate in rows
            if candidate.onset_interval_seconds is not None
        ]
        member: dict[str, Any] = {
            "spatial_identity_type": identity_type,
            "spatial_identity": identity,
            "source_candidate_ids": sorted(candidate.candidate_id for candidate in rows),
            "reference_families": sorted({candidate.reference_family for candidate in rows}),
            "source_analysis_unit_ids": sorted(
                {candidate.analysis_unit_id for candidate in rows}
            ),
            "onset_interval_envelope_seconds": [
                min(interval[0] for interval in candidate_intervals),
                max(interval[1] for interval in candidate_intervals),
            ],
            "derived_region": _candidate_region(rows[0]),
            "derived_laterality": _candidate_laterality(rows[0]),
            "physical_target_electrode_id": (
                identity if identity_type == "electrode" else None
            ),
            "bipolar_endpoint_attribution_performed": False,
        }
        members.append(member)
    members.sort(
        key=lambda row: (
            row["onset_interval_envelope_seconds"][0],
            row["spatial_identity_type"],
            row["spatial_identity"],
        )
    )
    return (
        {
            "status": "measured_research_set",
            "selection_rule": (
                "include_interval_if_lower_bound_not_after_global_minimum_upper_bound_plus_tolerance"
            ),
            "earliest_upper_bound_seconds": earliest_upper,
            "tolerance_seconds": policy.earliest_interval_tolerance_seconds,
            "members": members,
            "bipolar_endpoint_attribution_performed": False,
        },
        selected_ids,
    )


def _polarity_audit(
    eligible: Sequence[BAIEGReferenceSpecificFieldCandidate],
) -> dict[str, Any]:
    by_electrode: dict[str, list[BAIEGReferenceSpecificFieldCandidate]] = defaultdict(list)
    bipolar_rows: list[dict[str, Any]] = []
    for candidate in eligible:
        if candidate.analysis_unit_type == "lead":
            bipolar_rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "analysis_unit_type": "lead",
                    "lead_id": candidate.analysis_unit_id,
                    "reference_family": candidate.reference_family,
                    "polarity": candidate.polarity,
                    "polarity_coordinate": "signed_bipolar_analysis_edge",
                    "endpoint_attribution_performed": False,
                }
            )
        elif candidate.physical_target_electrode_id is not None:
            by_electrode[candidate.physical_target_electrode_id].append(candidate)

    rows: list[dict[str, Any]] = []
    comparable_agreements: list[float] = []
    conflict_electrodes: list[str] = []
    for electrode, candidates in sorted(by_electrode.items()):
        by_family: dict[str, set[str]] = defaultdict(set)
        source_ids: dict[str, list[str]] = defaultdict(list)
        for candidate in candidates:
            by_family[candidate.reference_family].add(candidate.polarity)
            source_ids[candidate.reference_family].append(candidate.candidate_id)
        resolved: dict[str, str] = {}
        for family, values in by_family.items():
            comparable = values.intersection({"positive", "negative"})
            resolved[family] = (
                next(iter(comparable))
                if len(comparable) == 1 and len(values) == 1
                else "not_comparable"
            )
        votes = [value for value in resolved.values() if value != "not_comparable"]
        agreement = None
        conflict = False
        if len(votes) >= 2:
            counts = Counter(votes)
            agreement = max(counts.values()) / len(votes)
            conflict = len(counts) > 1
            comparable_agreements.append(agreement)
            if conflict:
                conflict_electrodes.append(electrode)
        rows.append(
            {
                "electrode_id": electrode,
                "polarity_by_reference_family": {
                    family: resolved[family] for family in sorted(resolved)
                },
                "source_candidate_ids_by_reference_family": {
                    family: sorted(source_ids[family]) for family in sorted(source_ids)
                },
                "comparable_reference_family_count": len(votes),
                "polarity_agreement_fraction": agreement,
                "polarity_conflict": conflict,
            }
        )
    return {
        "target_coordinate_polarity": rows,
        "mean_comparable_target_polarity_agreement_fraction": (
            None
            if not comparable_agreements
            else sum(comparable_agreements) / len(comparable_agreements)
        ),
        "polarity_conflict_electrode_ids": conflict_electrodes,
        "bipolar_edge_polarity": sorted(
            bipolar_rows, key=lambda row: row["candidate_id"]
        ),
        "interpretation_limit": (
            "analysis_coordinate_polarity_only_not_cortical_source_polarity"
        ),
        "bipolar_endpoint_attribution_performed": False,
    }


def _candidate_binding(
    candidate: BAIEGReferenceSpecificFieldCandidate,
    reasons: Sequence[str],
) -> dict[str, Any]:
    result = candidate.to_dict()
    # The signed row itself is deliberately not duplicated in the result; its
    # content hash and the candidate hash bind it while preserving analysis
    # unit semantics at every downstream resolution.
    result["positive_onset_support_eligible"] = not reasons
    result["exclusion_reason_codes"] = list(reasons)
    result["bipolar_endpoint_attribution_performed"] = False
    return result


def _family_audits(
    candidates: Sequence[BAIEGReferenceSpecificFieldCandidate],
    reasons_by_id: Mapping[str, Sequence[str]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[BAIEGReferenceSpecificFieldCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.reference_family].append(candidate)
    result: list[dict[str, Any]] = []
    for family, rows in sorted(grouped.items()):
        eligible = [row for row in rows if not reasons_by_id[row.candidate_id]]
        reason_counts = Counter(
            reason
            for row in rows
            for reason in reasons_by_id[row.candidate_id]
        )
        coverage = [row.coverage_fraction for row in eligible]
        result.append(
            {
                "reference_family": family,
                "view_ids": sorted({row.view_id for row in rows}),
                "candidate_count": len(rows),
                "positive_onset_eligible_candidate_count": len(eligible),
                "excluded_candidate_count": len(rows) - len(eligible),
                "positive_onset_eligible_candidate_ids": sorted(
                    row.candidate_id for row in eligible
                ),
                "exclusion_reason_counts": dict(sorted(reason_counts.items())),
                "eligible_coverage_fraction": {
                    "minimum": None if not coverage else min(coverage),
                    "median": None if not coverage else statistics.median(coverage),
                    "maximum": None if not coverage else max(coverage),
                },
                "raw_score_cross_reference_comparison_performed": False,
            }
        )
    return result


_RESULT_KEYS = frozenset(
    {
        "schema_version",
        "implementation_id",
        "event_id",
        "recording_id",
        "analysis_interval_seconds",
        "canonical_receipt_sha256",
        "adaptive_window_receipt_sha256",
        "source_input_batch_sha256",
        "source_field_head_receipt_sha256",
        "policy",
        "policy_sha256",
        "scope_receipt",
        "source_candidate_bindings",
        "eligible_candidate_ids",
        "excluded_candidates",
        "reference_family_audits",
        "earliest_distinguishable_set",
        "polarity_audit",
        "resolution_ladder",
        "selected_resolution",
        "selected_candidate_ranking",
        "degradation_reason_codes",
        "receipt_sha256",
    }
)


def aggregate_ba_ieg_multireference_field(
    candidates: Sequence[BAIEGReferenceSpecificFieldCandidate],
    *,
    policy: BAIEGMultireferenceFieldPolicy = DEFAULT_BA_IEG_MULTIREFERENCE_FIELD_POLICY,
) -> dict[str, Any]:
    """Aggregate one event's reference-specific field candidates fail closed."""

    if not isinstance(policy, BAIEGMultireferenceFieldPolicy):
        raise TypeError("policy must be BAIEGMultireferenceFieldPolicy")
    rows = tuple(candidates)
    if not rows:
        raise ValueError("multi-reference field aggregation needs at least one candidate")
    if any(not isinstance(row, BAIEGReferenceSpecificFieldCandidate) for row in rows):
        raise TypeError("all field candidates must use the registered candidate contract")
    if len({row.candidate_id for row in rows}) != len(rows):
        raise ValueError("reference-specific candidate IDs must be unique")
    if len({(row.view_id, row.analysis_unit_id) for row in rows}) != len(rows):
        raise ValueError("one view cannot duplicate an analysis-unit candidate")

    shared_fields = {
        "event_id": {row.event_id for row in rows},
        "recording_id": {row.recording_id for row in rows},
        "analysis_interval_seconds": {row.analysis_interval_seconds for row in rows},
        "canonical_receipt_sha256": {row.canonical_receipt_sha256 for row in rows},
        "adaptive_window_receipt_sha256": {
            row.adaptive_window_receipt_sha256 for row in rows
        },
        "source_input_batch_sha256": {row.source_input_batch_sha256 for row in rows},
        "source_field_head_receipt_sha256": {
            row.source_field_head_receipt_sha256 for row in rows
        },
    }
    drifted = [name for name, values in shared_fields.items() if len(values) != 1]
    if drifted:
        raise ValueError(
            "reference candidates do not bind one physical event: " + ", ".join(drifted)
        )

    reasons_by_id = {
        row.candidate_id: _eligibility_reasons(row, policy) for row in rows
    }
    eligible = tuple(row for row in rows if not reasons_by_id[row.candidate_id])
    earliest, earliest_ids = _earliest_distinguishable_set(eligible, policy)
    ladder = [
        _resolution_projection(
            eligible,
            resolution=resolution,
            earliest_candidate_ids=earliest_ids,
            policy=policy,
        )
        for resolution in BA_IEG_MULTIREFERENCE_RESOLUTION_LADDER
    ]
    selected_level = next(
        (level for level in ladder if level["status"] == "qualified"), None
    )
    selected_resolution = (
        "phenotype_only" if selected_level is None else selected_level["resolution"]
    )
    degradation: list[str] = []
    for level in ladder:
        if level["resolution"] == selected_resolution:
            break
        degradation.append(f"{level['resolution']}_resolution_not_qualified")
    if selected_level is None:
        degradation.append("no_multireference_spatial_resolution_qualified")

    event_id = next(iter(shared_fields["event_id"]))
    recording_id = next(iter(shared_fields["recording_id"]))
    analysis_interval = next(iter(shared_fields["analysis_interval_seconds"]))
    body: dict[str, Any] = {
        "schema_version": BA_IEG_MULTIREFERENCE_FIELD_SCHEMA_VERSION,
        "implementation_id": BA_IEG_MULTIREFERENCE_FIELD_ID,
        "event_id": event_id,
        "recording_id": recording_id,
        "analysis_interval_seconds": list(analysis_interval),
        "canonical_receipt_sha256": next(
            iter(shared_fields["canonical_receipt_sha256"])
        ),
        "adaptive_window_receipt_sha256": next(
            iter(shared_fields["adaptive_window_receipt_sha256"])
        ),
        "source_input_batch_sha256": next(
            iter(shared_fields["source_input_batch_sha256"])
        ),
        "source_field_head_receipt_sha256": next(
            iter(shared_fields["source_field_head_receipt_sha256"])
        ),
        "policy": policy.to_dict(),
        "policy_sha256": policy.policy_sha256,
        "scope_receipt": {
            "input_scope": "eeg_signal_derived_reference_specific_candidates_only",
            "private_annotations_or_spreadsheets_consumed": False,
            "positive_onset_temporal_role": "onset_causal",
            "offline_or_future_positive_onset_support_allowed": False,
            "later_involvement_positive_onset_support_allowed": False,
            "imputed_or_quality_failed_positive_support_allowed": False,
            "bipolar_endpoint_attribution_allowed": False,
            "raw_score_cross_reference_probability_fusion_allowed": False,
            "output_scope": (
                "research_scalp_visible_onset_field_not_cortical_soz_or_ez"
            ),
            "production_report_route_connected": False,
        },
        "source_candidate_bindings": [
            _candidate_binding(row, reasons_by_id[row.candidate_id])
            for row in sorted(rows, key=lambda item: item.candidate_id)
        ],
        "eligible_candidate_ids": sorted(row.candidate_id for row in eligible),
        "excluded_candidates": [
            {
                "candidate_id": row.candidate_id,
                "candidate_sha256": row.candidate_sha256,
                "reason_codes": list(reasons_by_id[row.candidate_id]),
            }
            for row in sorted(rows, key=lambda item: item.candidate_id)
            if reasons_by_id[row.candidate_id]
        ],
        "reference_family_audits": _family_audits(rows, reasons_by_id),
        "earliest_distinguishable_set": earliest,
        "polarity_audit": _polarity_audit(eligible),
        "resolution_ladder": ladder,
        "selected_resolution": selected_resolution,
        "selected_candidate_ranking": (
            [] if selected_level is None else deepcopy(selected_level["candidate_ranking"])
        ),
        "degradation_reason_codes": degradation,
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    body["receipt_sha256"] = _canonical_sha256(body)
    return validate_ba_ieg_multireference_field_result(body)


def validate_ba_ieg_multireference_field_result(result: object) -> dict[str, Any]:
    """Validate content binding and the principal evidence-firewall invariants."""

    if not isinstance(result, Mapping):
        raise TypeError("multi-reference field result must be a mapping")
    data = deepcopy(dict(result))
    if set(data) != _RESULT_KEYS:
        raise ValueError("multi-reference field result keys drifted")
    if data["schema_version"] != BA_IEG_MULTIREFERENCE_FIELD_SCHEMA_VERSION:
        raise ValueError("multi-reference field schema version drifted")
    if data["implementation_id"] != BA_IEG_MULTIREFERENCE_FIELD_ID:
        raise ValueError("multi-reference field implementation drifted")
    _identifier(data["event_id"], "event_id")
    _identifier(data["recording_id"], "recording_id")
    _finite_interval(data["analysis_interval_seconds"], "analysis_interval_seconds")
    for name in (
        "canonical_receipt_sha256",
        "adaptive_window_receipt_sha256",
        "source_input_batch_sha256",
        "source_field_head_receipt_sha256",
        "policy_sha256",
        "receipt_sha256",
    ):
        _sha256(data[name], name)
    if _canonical_sha256({**data, "receipt_sha256": "CONTENT-ADDRESS-PENDING"}) != data[
        "receipt_sha256"
    ]:
        raise ValueError("multi-reference field result content hash drifted")

    scope = data["scope_receipt"]
    required_false = (
        "private_annotations_or_spreadsheets_consumed",
        "offline_or_future_positive_onset_support_allowed",
        "later_involvement_positive_onset_support_allowed",
        "imputed_or_quality_failed_positive_support_allowed",
        "bipolar_endpoint_attribution_allowed",
        "raw_score_cross_reference_probability_fusion_allowed",
        "production_report_route_connected",
    )
    if not isinstance(scope, Mapping) or any(scope.get(name) is not False for name in required_false):
        raise ValueError("multi-reference field scope firewall drifted")
    if scope.get("positive_onset_temporal_role") != "onset_causal":
        raise ValueError("positive onset temporal permission drifted")

    bindings = data["source_candidate_bindings"]
    if not isinstance(bindings, list) or not bindings:
        raise ValueError("source candidate bindings must be non-empty")
    candidate_ids = [str(row.get("candidate_id")) for row in bindings]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("source candidate bindings contain duplicate IDs")
    binding_by_id = {str(row["candidate_id"]): row for row in bindings}
    eligible = data["eligible_candidate_ids"]
    if (
        not isinstance(eligible, list)
        or eligible != sorted(set(str(item) for item in eligible))
        or not set(eligible).issubset(binding_by_id)
    ):
        raise ValueError("eligible candidate roster is invalid")
    expected_eligible = sorted(
        candidate_id
        for candidate_id, row in binding_by_id.items()
        if row.get("positive_onset_support_eligible") is True
        and row.get("exclusion_reason_codes") == []
    )
    if eligible != expected_eligible:
        raise ValueError("eligible candidate roster disagrees with permission receipts")
    for candidate_id in eligible:
        row = binding_by_id[candidate_id]
        if (
            row.get("observed") is not True
            or row.get("imputed") is not False
            or row.get("evidence_eligible") is not True
            or row.get("quality_pass") is not True
            or row.get("temporal_role") != "onset_causal"
            or row.get("future_sample_access") is not False
            or row.get("onset_evidence_authorized") is not True
            or row.get("intrinsic_evidence_role") != "onset_eligible"
            or row.get("onset_interval_seconds") is None
        ):
            raise ValueError("positive onset roster contains a forbidden candidate")

    excluded = data["excluded_candidates"]
    if not isinstance(excluded, list):
        raise ValueError("excluded candidate ledger must be a list")
    expected_excluded = sorted(set(candidate_ids) - set(eligible))
    actual_excluded = sorted(str(row.get("candidate_id")) for row in excluded)
    if expected_excluded != actual_excluded or any(
        not row.get("reason_codes") for row in excluded
    ):
        raise ValueError("excluded candidate ledger is incomplete")

    earliest = data["earliest_distinguishable_set"]
    if (
        not isinstance(earliest, Mapping)
        or earliest.get("bipolar_endpoint_attribution_performed") is not False
    ):
        raise ValueError("earliest-set bipolar invariant drifted")
    for member in earliest.get("members", []):
        if not set(member.get("source_candidate_ids", [])).issubset(eligible):
            raise ValueError("earliest set uses an excluded candidate")
        if member.get("bipolar_endpoint_attribution_performed") is not False:
            raise ValueError("earliest member attributes a bipolar endpoint")
        if member.get("spatial_identity_type") == "lead" and member.get(
            "physical_target_electrode_id"
        ) is not None:
            raise ValueError("bipolar lead became an endpoint electrode")

    ladder = data["resolution_ladder"]
    if (
        not isinstance(ladder, list)
        or [row.get("resolution") for row in ladder]
        != list(BA_IEG_MULTIREFERENCE_RESOLUTION_LADDER)
    ):
        raise ValueError("spatial resolution ladder drifted")
    for level in ladder:
        if level.get("bipolar_endpoint_attribution_performed") is not False:
            raise ValueError("resolution ladder attributes bipolar endpoints")
        for ranking in level.get("candidate_ranking", []):
            if not set(ranking.get("source_candidate_ids", [])).issubset(eligible):
                raise ValueError("spatial ranking uses an excluded candidate")
            if level["resolution"] == "electrode" and "lead" in ranking.get(
                "source_analysis_unit_types", []
            ):
                raise ValueError("bipolar lead entered electrode resolution")

    first_qualified = next(
        (level for level in ladder if level.get("status") == "qualified"), None
    )
    expected_resolution = (
        "phenotype_only" if first_qualified is None else first_qualified["resolution"]
    )
    expected_ranking = (
        [] if first_qualified is None else first_qualified["candidate_ranking"]
    )
    if data["selected_resolution"] != expected_resolution:
        raise ValueError("selected resolution does not follow the frozen ladder")
    if data["selected_candidate_ranking"] != expected_ranking:
        raise ValueError("selected candidate ranking drifted from its ladder level")
    return data


__all__ = [
    "BA_IEG_MULTIREFERENCE_FIELD_ID",
    "BA_IEG_MULTIREFERENCE_FIELD_SCHEMA_VERSION",
    "BA_IEG_MULTIREFERENCE_REFERENCE_FAMILIES",
    "BA_IEG_MULTIREFERENCE_RESOLUTION_LADDER",
    "BAIEGMultireferenceFieldPolicy",
    "BAIEGReferenceSpecificFieldCandidate",
    "DEFAULT_BA_IEG_MULTIREFERENCE_FIELD_POLICY",
    "aggregate_ba_ieg_multireference_field",
    "validate_ba_ieg_multireference_field_result",
]
