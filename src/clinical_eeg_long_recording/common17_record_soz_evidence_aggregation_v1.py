"""Auditable common-17 event-to-record scalp-onset evidence aggregation.

This module is the record-level numerical core between event Findings and a
claim-locked report graph.  It does not open EEG files, discover events, infer
event boundaries, read labels, or lexicalize a diagnosis.  Its inputs are the
*complete detector event roster* and EEG-derived event evidence only.

The contract deliberately separates two questions that must not share one
softmax:

* where a *localized scalp-visible onset pattern* is ranked on common-17; and
* whether the event pattern is localized, bilateral near-synchronous,
  generalized synchronous, or unresolved.

Consequently, generalized/bilateral/unresolved support is never projected to
``CZ`` (or any other electrode).  ``CZ`` remains one observed midline scalp
electrode.  Every detector-roster event is represented in the evidence ledger;
low-quality events may receive zero numerical weight but are never silently
removed.

Probability language has a second, independent guard.  Event-level
calibration does not make a data-dependent record pool calibrated.  Record
outputs are therefore called ``normalized_support_score`` unless a frozen,
patient-disjoint record calibration object is content-bound through a trusted
registry supplied by the host.  This Python mapping is an executable test
boundary, not a cryptographic host authority by itself.

This is a research scalp-visible onset ranking.  It is not cortical SOZ, the
epileptogenic zone, or a surgical target.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

from .common17_experiment_v1 import COMMON_17


COMMON17_RECORD_SOZ_AGGREGATION_SCHEMA_VERSION = (
    "clinical_eeg_common17_record_soz_evidence_aggregation_v1"
)
COMMON17_RECORD_SOZ_AGGREGATION_METHOD_ID = (
    "complete_roster_reliability_capped_mode_linear_pool_common17_v1"
)

COMMON17_CHANNEL_IDS: tuple[str, ...] = tuple(COMMON_17)
EXPECTED_COMMON17_CHANNEL_IDS: tuple[str, ...] = (
    "FP1",
    "FP2",
    "F7",
    "F3",
    "F4",
    "F8",
    "T7",
    "C3",
    "CZ",
    "C4",
    "T8",
    "P7",
    "P3",
    "P4",
    "P8",
    "O1",
    "O2",
)
if COMMON17_CHANNEL_IDS != EXPECTED_COMMON17_CHANNEL_IDS:
    raise RuntimeError("common-17 aggregation ontology drifted from its frozen basis")

COMMON17_REGION_IDS: tuple[str, ...] = (
    "frontal",
    "temporal",
    "central",
    "parietal",
    "occipital",
)
COMMON17_LATERALITY_IDS: tuple[str, ...] = ("left", "right", "midline")
INDEPENDENT_PATTERN_STATE_IDS: tuple[str, ...] = (
    "localized_scalp_onset",
    "bilateral_near_synchronous",
    "generalized_synchronous",
    "unresolved",
)

COMMON17_CHANNEL_TO_REGION: Mapping[str, str] = {
    "FP1": "frontal",
    "FP2": "frontal",
    "F7": "temporal",
    "F3": "frontal",
    "F4": "frontal",
    "F8": "temporal",
    "T7": "temporal",
    "C3": "central",
    "CZ": "central",
    "C4": "central",
    "T8": "temporal",
    "P7": "temporal",
    "P3": "parietal",
    "P4": "parietal",
    "P8": "temporal",
    "O1": "occipital",
    "O2": "occipital",
}
COMMON17_CHANNEL_TO_LATERALITY: Mapping[str, str] = {
    "FP1": "left",
    "FP2": "right",
    "F7": "left",
    "F3": "left",
    "F4": "right",
    "F8": "right",
    "T7": "left",
    "C3": "left",
    "CZ": "midline",
    "C4": "right",
    "T8": "right",
    "P7": "left",
    "P3": "left",
    "P4": "right",
    "P8": "right",
    "O1": "left",
    "O2": "right",
}

CALIBRATED_PROBABILITY = "calibrated_probability"
UNCALIBRATED_NONNEGATIVE_SCORE = "uncalibrated_nonnegative_score"
NORMALIZED_SUPPORT_SCORE = "normalized_support_score"
NOT_APPLICABLE_NONLOCALIZED = "not_applicable_nonlocalized"
DETERMINISTIC_UNRESOLVED_FALLBACK_MASS = "deterministic_unresolved_fallback_mass"

_INPUT_VALUE_SEMANTICS = {
    CALIBRATED_PROBABILITY,
    UNCALIBRATED_NONNEGATIVE_SCORE,
}
_CHANNEL_INPUT_VALUE_SEMANTICS = {
    *_INPUT_VALUE_SEMANTICS,
    NOT_APPLICABLE_NONLOCALIZED,
}
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_TOL = 1e-9
_ROUND_DIGITS = 12

_INFERENCE_EXCLUSIONS: Mapping[str, bool] = {
    "edf_annotations_used": False,
    "excel_used": False,
    "doctor_labels_used": False,
    "clinical_text_used": False,
    "patient_metadata_used": False,
    "video_used": False,
    "ecg_emg_eog_used": False,
    "sleep_staging_used": False,
    "provocation_used": False,
}


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _self_hash(value: Mapping[str, object], field: str) -> str:
    body = deepcopy(dict(value))
    body.pop(field, None)
    return _canonical_sha256(body)


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be an opaque identifier")
    return value


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _unit_interval(value: object, name: str) -> float:
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


def _normalized_nonnegative_vector(
    values: Sequence[float],
    *,
    expected_length: int,
    semantics: str,
    name: str,
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or len(values) != expected_length:
        raise ValueError(f"{name} must contain exactly {expected_length} values")
    result: list[float] = []
    for index, raw in enumerate(values):
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise TypeError(f"{name}[{index}] must be numeric")
        value = float(raw)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
        result.append(value)
    total = math.fsum(result)
    if total <= 0.0:
        raise ValueError(f"{name} must contain positive mass")
    if semantics == CALIBRATED_PROBABILITY and not math.isclose(
        total, 1.0, rel_tol=0.0, abs_tol=1e-6
    ):
        raise ValueError(f"{name} declared calibrated but does not sum to one")
    return tuple(value / total for value in result)


def _rounded(value: float) -> float:
    return round(float(value), _ROUND_DIGITS)


def _rounded_mapping(ids: Sequence[str], values: Sequence[float]) -> dict[str, float]:
    return {identifier: _rounded(value) for identifier, value in zip(ids, values)}


@dataclass(frozen=True)
class CalibrationBindingV1:
    """Content identity of an event-level calibration receipt."""

    receipt_id: str
    receipt_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.receipt_id, "calibration receipt_id")
        _sha256(self.receipt_sha256, "calibration receipt_sha256")


@dataclass(frozen=True)
class Common17EventQCProfileV1:
    """EEG-derived quality factors used in the reliability weight."""

    signal_valid_fraction: float
    common17_channel_coverage_fraction: float
    artifact_free_fraction: float
    reference_stability: float
    onset_boundary_support: float
    adaptive_support_coverage: float

    def __post_init__(self) -> None:
        for name in (
            "signal_valid_fraction",
            "common17_channel_coverage_fraction",
            "artifact_free_fraction",
            "reference_stability",
            "onset_boundary_support",
            "adaptive_support_coverage",
        ):
            object.__setattr__(self, name, _unit_interval(getattr(self, name), name))

    @property
    def geometric_quality(self) -> float:
        values = (
            self.signal_valid_fraction,
            self.common17_channel_coverage_fraction,
            self.artifact_free_fraction,
            self.reference_stability,
            self.onset_boundary_support,
            self.adaptive_support_coverage,
        )
        if any(value <= 0.0 for value in values):
            return 0.0
        return math.exp(math.fsum(math.log(value) for value in values) / len(values))

    def as_dict(self) -> dict[str, float]:
        return {
            "signal_valid_fraction": _rounded(self.signal_valid_fraction),
            "common17_channel_coverage_fraction": _rounded(
                self.common17_channel_coverage_fraction
            ),
            "artifact_free_fraction": _rounded(self.artifact_free_fraction),
            "reference_stability": _rounded(self.reference_stability),
            "onset_boundary_support": _rounded(self.onset_boundary_support),
            "adaptive_support_coverage": _rounded(self.adaptive_support_coverage),
            "geometric_quality": _rounded(self.geometric_quality),
        }


@dataclass(frozen=True)
class Common17EventSOZEvidenceV1:
    """One event's EEG-only, onset-safe common-17 evidence."""

    event_id: str
    source_event_evidence_sha256: str
    mode_id: str
    channel_values: tuple[float, ...] | None
    channel_value_semantics: str
    state_values: tuple[float, ...]
    state_value_semantics: str
    model_reliability: float
    qc: Common17EventQCProfileV1
    onset_evidence_ids: tuple[str, ...]
    channel_calibration: CalibrationBindingV1 | None = None
    state_calibration: CalibrationBindingV1 | None = None
    labels_or_external_context_present: bool = False

    def __post_init__(self) -> None:
        _identifier(self.event_id, "event_id")
        _identifier(self.mode_id, "mode_id")
        _sha256(self.source_event_evidence_sha256, "source event evidence")
        if self.channel_value_semantics not in _CHANNEL_INPUT_VALUE_SEMANTICS:
            raise ValueError("unsupported channel value semantics")
        if self.state_value_semantics not in _INPUT_VALUE_SEMANTICS:
            raise ValueError("unsupported state value semantics")
        if not isinstance(self.qc, Common17EventQCProfileV1):
            raise TypeError("qc must be Common17EventQCProfileV1")
        object.__setattr__(
            self,
            "model_reliability",
            _unit_interval(self.model_reliability, "model_reliability"),
        )
        state = _normalized_nonnegative_vector(
            self.state_values,
            expected_length=len(INDEPENDENT_PATTERN_STATE_IDS),
            semantics=self.state_value_semantics,
            name="state_values",
        )
        object.__setattr__(self, "state_values", state)

        if self.channel_value_semantics == NOT_APPLICABLE_NONLOCALIZED:
            if self.channel_values is not None or self.channel_calibration is not None:
                raise ValueError("nonlocalized event cannot carry channel values")
            if state[0] > _TOL:
                raise ValueError(
                    "localized state mass requires common-17 channel evidence"
                )
        else:
            if self.channel_values is None:
                raise ValueError("channel values are required for spatial evidence")
            normalized_channel = _normalized_nonnegative_vector(
                self.channel_values,
                expected_length=len(COMMON17_CHANNEL_IDS),
                semantics=self.channel_value_semantics,
                name="channel_values",
            )
            object.__setattr__(self, "channel_values", normalized_channel)

        if self.channel_value_semantics == CALIBRATED_PROBABILITY:
            if not isinstance(self.channel_calibration, CalibrationBindingV1):
                raise ValueError("calibrated channel values require a receipt binding")
        elif self.channel_calibration is not None:
            raise ValueError("uncalibrated channel values cannot carry calibration")
        if self.state_value_semantics == CALIBRATED_PROBABILITY:
            if not isinstance(self.state_calibration, CalibrationBindingV1):
                raise ValueError("calibrated state values require a receipt binding")
        elif self.state_calibration is not None:
            raise ValueError("uncalibrated state values cannot carry calibration")

        if isinstance(self.onset_evidence_ids, (str, bytes)):
            raise TypeError("onset_evidence_ids must be a sequence")
        identifiers = tuple(
            _identifier(item, "onset_evidence_id") for item in self.onset_evidence_ids
        )
        if not identifiers or len(identifiers) != len(set(identifiers)):
            raise ValueError("onset_evidence_ids must be non-empty and unique")
        object.__setattr__(self, "onset_evidence_ids", identifiers)
        if self.labels_or_external_context_present is not False:
            raise ValueError("labels and external clinical context are forbidden")

    @property
    def effective_reliability(self) -> float:
        return self.model_reliability * self.qc.geometric_quality

    @property
    def localized_mass(self) -> float:
        return float(self.state_values[0])

    @property
    def content_sha256(self) -> str:
        return _canonical_sha256(
            {
                "event_id": self.event_id,
                "source_event_evidence_sha256": self.source_event_evidence_sha256,
                "mode_id": self.mode_id,
                "channel_values": self.channel_values,
                "channel_value_semantics": self.channel_value_semantics,
                "channel_calibration": (
                    None
                    if self.channel_calibration is None
                    else {
                        "receipt_id": self.channel_calibration.receipt_id,
                        "receipt_sha256": self.channel_calibration.receipt_sha256,
                    }
                ),
                "state_values": self.state_values,
                "state_value_semantics": self.state_value_semantics,
                "state_calibration": (
                    None
                    if self.state_calibration is None
                    else {
                        "receipt_id": self.state_calibration.receipt_id,
                        "receipt_sha256": self.state_calibration.receipt_sha256,
                    }
                ),
                "model_reliability": self.model_reliability,
                "qc": self.qc.as_dict(),
                "onset_evidence_ids": self.onset_evidence_ids,
                "labels_or_external_context_present": False,
            }
        )


@dataclass(frozen=True)
class Common17CompleteEventRosterV1:
    """Complete, chronologically ordered detector roster for one recording."""

    record_id: str
    canonical_signal_sha256: str
    upstream_model_artifact_sha256: str
    detector_event_roster: tuple[str, ...]
    events: tuple[Common17EventSOZEvidenceV1, ...]
    source_scope: str = "synthetic"
    labels_or_external_context_present: bool = False

    def __post_init__(self) -> None:
        _identifier(self.record_id, "record_id")
        _sha256(self.canonical_signal_sha256, "canonical_signal_sha256")
        _sha256(
            self.upstream_model_artifact_sha256,
            "upstream_model_artifact_sha256",
        )
        if self.source_scope not in {
            "public_source",
            "synthetic",
            "deployment_eeg_only",
        }:
            raise ValueError("unsupported record source_scope")
        if self.labels_or_external_context_present is not False:
            raise ValueError("labels and external clinical context are forbidden")
        if not self.events or not all(
            isinstance(item, Common17EventSOZEvidenceV1) for item in self.events
        ):
            raise TypeError("record aggregation requires at least one typed event")
        roster = tuple(
            _identifier(item, "detector event roster id")
            for item in self.detector_event_roster
        )
        if len(roster) != len(set(roster)):
            raise ValueError("detector event roster contains duplicate IDs")
        event_ids = tuple(item.event_id for item in self.events)
        if event_ids != roster:
            raise ValueError(
                "event evidence must exactly cover the complete detector roster in order"
            )
        hashes = [item.content_sha256 for item in self.events]
        if len(hashes) != len(set(hashes)):
            raise ValueError("record event roster repeats an identical event payload")
        object.__setattr__(self, "detector_event_roster", roster)

    @property
    def detector_event_roster_sha256(self) -> str:
        return _canonical_sha256(
            {
                "record_id": self.record_id,
                "canonical_signal_sha256": self.canonical_signal_sha256,
                "detector_event_roster": self.detector_event_roster,
                "event_content_sha256s": [item.content_sha256 for item in self.events],
            }
        )


@dataclass(frozen=True)
class Common17RecordSOZAggregationPolicyV1:
    """Frozen mode-capped reliability pooling and audit thresholds."""

    mode_state_evidence_cap: float = 1.0
    mode_spatial_evidence_cap: float = 1.0
    low_heterogeneity_js: float = 0.08
    high_heterogeneity_js: float = 0.25
    loeo_stable_max_js: float = 0.12

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "mode_state_evidence_cap",
            _positive_float(self.mode_state_evidence_cap, "mode_state_evidence_cap"),
        )
        object.__setattr__(
            self,
            "mode_spatial_evidence_cap",
            _positive_float(
                self.mode_spatial_evidence_cap, "mode_spatial_evidence_cap"
            ),
        )
        for name in (
            "low_heterogeneity_js",
            "high_heterogeneity_js",
            "loeo_stable_max_js",
        ):
            object.__setattr__(self, name, _unit_interval(getattr(self, name), name))
        if self.low_heterogeneity_js >= self.high_heterogeneity_js:
            raise ValueError("heterogeneity thresholds must be strictly ordered")

    @property
    def policy_sha256(self) -> str:
        return _canonical_sha256(
            {
                "method_id": COMMON17_RECORD_SOZ_AGGREGATION_METHOD_ID,
                "common17_channel_ids": COMMON17_CHANNEL_IDS,
                "channel_to_region": COMMON17_CHANNEL_TO_REGION,
                "channel_to_laterality": COMMON17_CHANNEL_TO_LATERALITY,
                "independent_pattern_state_ids": INDEPENDENT_PATTERN_STATE_IDS,
                "mode_state_evidence_cap": self.mode_state_evidence_cap,
                "mode_spatial_evidence_cap": self.mode_spatial_evidence_cap,
                "low_heterogeneity_js": self.low_heterogeneity_js,
                "high_heterogeneity_js": self.high_heterogeneity_js,
                "loeo_stable_max_js": self.loeo_stable_max_js,
                "event_pool": "reliability_weighted_linear_pool",
                "mode_pool": "reliability_capped_linear_pool",
                "nonlocalized_state_projection_to_channels": "forbidden",
            }
        )


@dataclass(frozen=True)
class Common17RecordCalibrationV1:
    """Frozen record-pool calibrator certified on a patient-disjoint split."""

    receipt_id: str
    artifact_sha256: str
    upstream_model_artifact_sha256: str
    policy_sha256: str
    channel_temperature: float
    state_temperature: float
    validation_scope: str = "source_dev_patient_disjoint"
    method_id: str = COMMON17_RECORD_SOZ_AGGREGATION_METHOD_ID

    def __post_init__(self) -> None:
        _identifier(self.receipt_id, "record calibration receipt_id")
        _sha256(self.artifact_sha256, "record calibration artifact")
        _sha256(
            self.upstream_model_artifact_sha256,
            "record calibration upstream model artifact",
        )
        _sha256(self.policy_sha256, "record calibration policy")
        object.__setattr__(
            self,
            "channel_temperature",
            _positive_float(self.channel_temperature, "channel_temperature"),
        )
        object.__setattr__(
            self,
            "state_temperature",
            _positive_float(self.state_temperature, "state_temperature"),
        )
        if self.validation_scope != "source_dev_patient_disjoint":
            raise ValueError("record calibration must be patient-disjoint")
        if self.method_id != COMMON17_RECORD_SOZ_AGGREGATION_METHOD_ID:
            raise ValueError("record calibration method mismatch")

    @property
    def receipt_payload_sha256(self) -> str:
        return _canonical_sha256(
            {
                "receipt_id": self.receipt_id,
                "artifact_sha256": self.artifact_sha256,
                "upstream_model_artifact_sha256": (self.upstream_model_artifact_sha256),
                "policy_sha256": self.policy_sha256,
                "channel_temperature": self.channel_temperature,
                "state_temperature": self.state_temperature,
                "validation_scope": self.validation_scope,
                "method_id": self.method_id,
            }
        )


@dataclass(frozen=True)
class _PreparedEvent:
    event: Common17EventSOZEvidenceV1
    channel: tuple[float, ...] | None
    state: tuple[float, ...]
    reliability: float
    localized_weight: float


def _verify_calibration_binding(
    binding: CalibrationBindingV1,
    registry: Mapping[str, str] | None,
    *,
    context: str,
) -> None:
    if registry is None:
        raise ValueError(f"{context} requires a trusted event calibration registry")
    trusted = registry.get(binding.receipt_id)
    if trusted != binding.receipt_sha256:
        raise ValueError(f"{context} calibration binding is not host-trusted")


def _prepare_events(
    bag: Common17CompleteEventRosterV1,
    *,
    trusted_event_calibration_registry: Mapping[str, str] | None,
) -> list[_PreparedEvent]:
    result: list[_PreparedEvent] = []
    for event in bag.events:
        if event.channel_value_semantics == CALIBRATED_PROBABILITY:
            assert event.channel_calibration is not None
            _verify_calibration_binding(
                event.channel_calibration,
                trusted_event_calibration_registry,
                context=f"event {event.event_id} channel",
            )
        if event.state_value_semantics == CALIBRATED_PROBABILITY:
            assert event.state_calibration is not None
            _verify_calibration_binding(
                event.state_calibration,
                trusted_event_calibration_registry,
                context=f"event {event.event_id} state",
            )
        reliability = event.effective_reliability
        localized_weight = reliability * event.localized_mass
        result.append(
            _PreparedEvent(
                event=event,
                channel=event.channel_values,
                state=event.state_values,
                reliability=reliability,
                localized_weight=localized_weight,
            )
        )
    return result


def _weighted_linear_pool(
    vectors: Sequence[Sequence[float]], weights: Sequence[float]
) -> tuple[float, ...]:
    if not vectors or len(vectors) != len(weights):
        raise ValueError("weighted linear pool requires aligned non-empty inputs")
    total = math.fsum(weights)
    if total <= 0.0:
        raise ValueError("weighted linear pool requires positive weight")
    width = len(vectors[0])
    if any(len(vector) != width for vector in vectors):
        raise ValueError("weighted linear pool vectors have different widths")
    result = tuple(
        math.fsum(weight * vector[index] for vector, weight in zip(vectors, weights))
        / total
        for index in range(width)
    )
    normalizer = math.fsum(result)
    if normalizer <= 0.0:
        raise RuntimeError("weighted linear pool lost all mass")
    return tuple(value / normalizer for value in result)


def _temperature_scale(
    values: Sequence[float], temperature: float
) -> tuple[float, ...]:
    logits = [math.log(max(float(value), 1e-15)) / temperature for value in values]
    maximum = max(logits)
    exponentials = [math.exp(value - maximum) for value in logits]
    total = math.fsum(exponentials)
    return tuple(value / total for value in exponentials)


def _js_divergence(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("JS divergence requires aligned non-empty vectors")
    value = 0.0
    for first, second in zip(left, right):
        middle = (float(first) + float(second)) / 2.0
        if first > 0.0:
            value += 0.5 * float(first) * math.log2(float(first) / middle)
        if second > 0.0:
            value += 0.5 * float(second) * math.log2(float(second) / middle)
    return min(1.0, max(0.0, value))


def _effective_count(weights: Sequence[float]) -> float:
    total = math.fsum(weights)
    denominator = math.fsum(value * value for value in weights)
    if total <= 0.0 or denominator <= 0.0:
        return 0.0
    return total * total / denominator


def _weighted_pairwise_js(
    vectors_and_weights: Sequence[tuple[str, Sequence[float], float]],
) -> dict[str, Any]:
    pairs: list[tuple[float, float]] = []
    maximum = 0.0
    maximum_pair: tuple[str, str] | None = None
    for left in range(len(vectors_and_weights)):
        left_id, left_vector, left_weight = vectors_and_weights[left]
        if left_weight <= 0.0:
            continue
        for right in range(left + 1, len(vectors_and_weights)):
            right_id, right_vector, right_weight = vectors_and_weights[right]
            if right_weight <= 0.0:
                continue
            divergence = _js_divergence(left_vector, right_vector)
            pair_weight = left_weight * right_weight
            pairs.append((divergence, pair_weight))
            if divergence > maximum:
                maximum = divergence
                maximum_pair = (left_id, right_id)
    if not pairs:
        return {
            "evaluable_pair_count": 0,
            "weighted_mean_js": None,
            "maximum_js": None,
            "maximum_pair": None,
        }
    denominator = math.fsum(weight for _, weight in pairs)
    mean = math.fsum(value * weight for value, weight in pairs) / denominator
    return {
        "evaluable_pair_count": len(pairs),
        "weighted_mean_js": _rounded(mean),
        "maximum_js": _rounded(maximum),
        "maximum_pair": None if maximum_pair is None else list(maximum_pair),
    }


def _mode_entropy(weights: Sequence[float]) -> float:
    positive = [float(value) for value in weights if value > 0.0]
    if len(positive) <= 1:
        return 0.0
    total = math.fsum(positive)
    probabilities = [value / total for value in positive]
    entropy = -math.fsum(value * math.log(value) for value in probabilities)
    return entropy / math.log(len(probabilities))


def _ranked_values(
    identifiers: Sequence[str], values: Sequence[float]
) -> list[dict[str, Any]]:
    order = sorted(
        range(len(identifiers)),
        key=lambda index: (-float(values[index]), identifiers[index]),
    )
    return [
        {
            "rank": rank,
            "candidate_id": identifiers[index],
            "value": _rounded(values[index]),
        }
        for rank, index in enumerate(order, start=1)
    ]


def _project_channel_axis(
    channel: Sequence[float], mapping: Mapping[str, str], target_ids: Sequence[str]
) -> tuple[float, ...]:
    source = dict(zip(COMMON17_CHANNEL_IDS, channel))
    result = tuple(
        math.fsum(
            source[channel_id]
            for channel_id in COMMON17_CHANNEL_IDS
            if mapping[channel_id] == target_id
        )
        for target_id in target_ids
    )
    total = math.fsum(result)
    if not math.isclose(total, 1.0, abs_tol=1e-8):
        raise RuntimeError("common-17 hierarchy projection lost mass")
    return tuple(value / total for value in result)


def _aggregate_core(
    events: Sequence[_PreparedEvent],
    policy: Common17RecordSOZAggregationPolicyV1,
) -> dict[str, Any]:
    if not events:
        raise ValueError("record core requires at least one event")
    by_mode: dict[str, list[_PreparedEvent]] = defaultdict(list)
    for event in events:
        by_mode[event.event.mode_id].append(event)

    mode_rows: list[dict[str, Any]] = []
    contribution: dict[str, dict[str, float]] = {
        event.event.event_id: {} for event in events
    }
    for mode_id in sorted(by_mode):
        members = by_mode[mode_id]
        state_raw_weight = math.fsum(item.reliability for item in members)
        spatial_raw_weight = math.fsum(item.localized_weight for item in members)
        state_weight = min(state_raw_weight, policy.mode_state_evidence_cap)
        spatial_weight = min(spatial_raw_weight, policy.mode_spatial_evidence_cap)
        if state_raw_weight > 0.0:
            state = _weighted_linear_pool(
                [item.state for item in members],
                [item.reliability for item in members],
            )
        else:
            state = (0.0, 0.0, 0.0, 1.0)
        spatial_members = [
            item
            for item in members
            if item.channel is not None and item.localized_weight > 0.0
        ]
        channel = None
        if spatial_members:
            channel = _weighted_linear_pool(
                [item.channel for item in spatial_members if item.channel is not None],
                [item.localized_weight for item in spatial_members],
            )
        for item in members:
            contribution[item.event.event_id] = {
                "state_within_mode_weight": (
                    0.0
                    if state_raw_weight <= 0.0
                    else item.reliability / state_raw_weight
                ),
                "spatial_within_mode_weight": (
                    0.0
                    if spatial_raw_weight <= 0.0
                    else item.localized_weight / spatial_raw_weight
                ),
            }
        mode_rows.append(
            {
                "mode_id": mode_id,
                "event_ids": [item.event.event_id for item in members],
                "event_count": len(members),
                "raw_state_evidence_weight": state_raw_weight,
                "capped_state_evidence_weight": state_weight,
                "raw_spatial_evidence_weight": spatial_raw_weight,
                "capped_spatial_evidence_weight": spatial_weight,
                "state": state,
                "channel": channel,
            }
        )

    total_state_weight = math.fsum(
        row["capped_state_evidence_weight"] for row in mode_rows
    )
    total_spatial_weight = math.fsum(
        row["capped_spatial_evidence_weight"]
        for row in mode_rows
        if row["channel"] is not None
    )
    if total_state_weight > 0.0:
        record_state = _weighted_linear_pool(
            [row["state"] for row in mode_rows],
            [row["capped_state_evidence_weight"] for row in mode_rows],
        )
        state_status = "estimated_from_reliable_event_evidence"
    else:
        record_state = (0.0, 0.0, 0.0, 1.0)
        state_status = "deterministic_unresolved_all_zero_reliability"

    spatial_modes = [
        row
        for row in mode_rows
        if row["channel"] is not None and row["capped_spatial_evidence_weight"] > 0.0
    ]
    record_channel = None
    if spatial_modes:
        record_channel = _weighted_linear_pool(
            [row["channel"] for row in spatial_modes],
            [row["capped_spatial_evidence_weight"] for row in spatial_modes],
        )

    mode_by_id = {row["mode_id"]: row for row in mode_rows}
    for event in events:
        mode = mode_by_id[event.event.mode_id]
        state_record = (
            0.0
            if total_state_weight <= 0.0
            else mode["capped_state_evidence_weight"]
            / total_state_weight
            * contribution[event.event.event_id]["state_within_mode_weight"]
        )
        spatial_record = (
            0.0
            if total_spatial_weight <= 0.0 or mode["channel"] is None
            else mode["capped_spatial_evidence_weight"]
            / total_spatial_weight
            * contribution[event.event.event_id]["spatial_within_mode_weight"]
        )
        contribution[event.event.event_id].update(
            {
                "state_record_contribution_weight": state_record,
                "spatial_record_contribution_weight": spatial_record,
            }
        )

    event_state_heterogeneity = _weighted_pairwise_js(
        [(item.event.event_id, item.state, item.reliability) for item in events]
    )
    event_spatial_heterogeneity = _weighted_pairwise_js(
        [
            (item.event.event_id, item.channel, item.localized_weight)
            for item in events
            if item.channel is not None
        ]
    )
    mode_state_heterogeneity = _weighted_pairwise_js(
        [
            (
                row["mode_id"],
                row["state"],
                row["capped_state_evidence_weight"],
            )
            for row in mode_rows
        ]
    )
    mode_spatial_heterogeneity = _weighted_pairwise_js(
        [
            (
                row["mode_id"],
                row["channel"],
                row["capped_spatial_evidence_weight"],
            )
            for row in mode_rows
            if row["channel"] is not None
        ]
    )
    candidates = [
        value
        for metric in (
            event_state_heterogeneity,
            event_spatial_heterogeneity,
            mode_state_heterogeneity,
            mode_spatial_heterogeneity,
        )
        for value in [metric["maximum_js"]]
        if value is not None
    ]
    if len(events) == 1:
        heterogeneity_class = "single_event_not_estimable"
    elif not candidates:
        heterogeneity_class = "not_estimable_no_reliable_pairs"
    elif max(candidates) <= policy.low_heterogeneity_js:
        heterogeneity_class = "low"
    elif max(candidates) <= policy.high_heterogeneity_js:
        heterogeneity_class = "moderate"
    else:
        heterogeneity_class = "high"

    mode_state_weights = [row["capped_state_evidence_weight"] for row in mode_rows]
    dominant_mode = None
    if any(value > 0.0 for value in mode_state_weights):
        dominant_mode = min(
            mode_rows,
            key=lambda row: (
                -row["capped_state_evidence_weight"],
                row["mode_id"],
            ),
        )["mode_id"]
    return {
        "state": record_state,
        "state_status": state_status,
        "channel": record_channel,
        "total_state_weight": total_state_weight,
        "total_spatial_weight": total_spatial_weight,
        "mode_rows": mode_rows,
        "contribution": contribution,
        "heterogeneity": {
            "classification": heterogeneity_class,
            "event_state": event_state_heterogeneity,
            "event_spatial_conditional_on_localized": event_spatial_heterogeneity,
            "mode_state": mode_state_heterogeneity,
            "mode_spatial_conditional_on_localized": mode_spatial_heterogeneity,
        },
        "mode_diagnostics": {
            "mode_count": len(mode_rows),
            "dominant_mode_id": dominant_mode,
            "normalized_mode_weight_entropy": _rounded(
                _mode_entropy(mode_state_weights)
            ),
            "effective_event_count_by_reliability": _rounded(
                _effective_count([item.reliability for item in events])
            ),
        },
    }


def _apply_record_calibration(
    core: Mapping[str, Any],
    calibration: Common17RecordCalibrationV1 | None,
) -> tuple[tuple[float, ...], tuple[float, ...] | None, str, bool]:
    state = tuple(float(item) for item in core["state"])
    channel_raw = core["channel"]
    channel = (
        None if channel_raw is None else tuple(float(item) for item in channel_raw)
    )
    if core["state_status"] != "estimated_from_reliable_event_evidence":
        return (
            state,
            channel,
            DETERMINISTIC_UNRESOLVED_FALLBACK_MASS,
            False,
        )
    if calibration is None:
        return state, channel, NORMALIZED_SUPPORT_SCORE, False
    calibrated_state = _temperature_scale(state, calibration.state_temperature)
    calibrated_channel = (
        None
        if channel is None
        else _temperature_scale(channel, calibration.channel_temperature)
    )
    return calibrated_state, calibrated_channel, CALIBRATED_PROBABILITY, True


def _argmax_id(ids: Sequence[str], values: Sequence[float] | None) -> str | None:
    if values is None:
        return None
    index = min(
        range(len(ids)), key=lambda position: (-float(values[position]), ids[position])
    )
    return ids[index]


def _loeo(
    events: Sequence[_PreparedEvent],
    policy: Common17RecordSOZAggregationPolicyV1,
    calibration: Common17RecordCalibrationV1 | None,
    full_state: Sequence[float],
    full_channel: Sequence[float] | None,
) -> dict[str, Any]:
    if len(events) == 1:
        return {
            "status": "not_evaluable_single_event",
            "event_count": 1,
            "stability_score": None,
            "maximum_effective_js": None,
            "mean_effective_js": None,
            "top_channel_flip_fraction": None,
            "spatial_estimability_change_fraction": None,
            "stable_under_policy": False,
            "rows": [],
        }
    full_top = _argmax_id(COMMON17_CHANNEL_IDS, full_channel)
    rows: list[dict[str, Any]] = []
    effective_js_values: list[float] = []
    comparable_top_rows = 0
    top_flips = 0
    estimability_changes = 0
    for removed_index, removed in enumerate(events):
        reduced_events = [
            item for index, item in enumerate(events) if index != removed_index
        ]
        reduced_core = _aggregate_core(reduced_events, policy)
        reduced_state, reduced_channel, _, _ = _apply_record_calibration(
            reduced_core, calibration
        )
        state_js = _js_divergence(full_state, reduced_state)
        if (full_channel is None) != (reduced_channel is None):
            channel_js = None
            effective_channel_js = 1.0
            estimability_changed = True
            estimability_changes += 1
        elif full_channel is None:
            channel_js = None
            effective_channel_js = 0.0
            estimability_changed = False
        else:
            assert reduced_channel is not None
            channel_js = _js_divergence(full_channel, reduced_channel)
            effective_channel_js = channel_js
            estimability_changed = False
        reduced_top = _argmax_id(COMMON17_CHANNEL_IDS, reduced_channel)
        top_changed = None
        if full_top is not None and reduced_top is not None:
            comparable_top_rows += 1
            top_changed = full_top != reduced_top
            top_flips += int(top_changed)
        effective = max(state_js, effective_channel_js)
        effective_js_values.append(effective)
        rows.append(
            {
                "removed_event_id": removed.event.event_id,
                "removed_mode_id": removed.event.mode_id,
                "state_js": _rounded(state_js),
                "channel_js_conditional_on_localized": (
                    None if channel_js is None else _rounded(channel_js)
                ),
                "spatial_estimability_changed": estimability_changed,
                "full_top_channel": full_top,
                "reduced_top_channel": reduced_top,
                "top_channel_changed": top_changed,
                "full_mode_count": len({item.event.mode_id for item in events}),
                "reduced_mode_count": len(
                    {item.event.mode_id for item in reduced_events}
                ),
                "effective_axis_js": _rounded(effective),
            }
        )
    maximum = max(effective_js_values)
    mean = math.fsum(effective_js_values) / len(effective_js_values)
    return {
        "status": "evaluated",
        "event_count": len(events),
        "stability_score": _rounded(max(0.0, 1.0 - maximum)),
        "maximum_effective_js": _rounded(maximum),
        "mean_effective_js": _rounded(mean),
        "top_channel_flip_fraction": (
            None
            if comparable_top_rows == 0
            else _rounded(top_flips / comparable_top_rows)
        ),
        "spatial_estimability_change_fraction": _rounded(
            estimability_changes / len(events)
        ),
        "stable_under_policy": maximum <= policy.loeo_stable_max_js,
        "rows": rows,
    }


def _verify_record_calibration(
    calibration: Common17RecordCalibrationV1 | None,
    *,
    bag: Common17CompleteEventRosterV1,
    policy: Common17RecordSOZAggregationPolicyV1,
    trusted_record_calibration_registry: Mapping[str, str] | None,
) -> None:
    if calibration is None:
        return
    if calibration.upstream_model_artifact_sha256 != bag.upstream_model_artifact_sha256:
        raise ValueError("record calibration/upstream model mismatch")
    if calibration.policy_sha256 != policy.policy_sha256:
        raise ValueError("record calibration/policy mismatch")
    if trusted_record_calibration_registry is None or (
        trusted_record_calibration_registry.get(calibration.receipt_id)
        != calibration.receipt_payload_sha256
    ):
        raise ValueError("record calibration receipt is not host-trusted")


def aggregate_common17_record_soz_evidence_v1(
    bag: Common17CompleteEventRosterV1,
    *,
    policy: Common17RecordSOZAggregationPolicyV1 | None = None,
    trusted_event_calibration_registry: Mapping[str, str] | None = None,
    record_calibration: Common17RecordCalibrationV1 | None = None,
    trusted_record_calibration_registry: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Aggregate every detector-roster event into record-level EEG evidence."""

    if not isinstance(bag, Common17CompleteEventRosterV1):
        raise TypeError("bag must be Common17CompleteEventRosterV1")
    if policy is None:
        policy = Common17RecordSOZAggregationPolicyV1()
    if not isinstance(policy, Common17RecordSOZAggregationPolicyV1):
        raise TypeError("policy must be Common17RecordSOZAggregationPolicyV1")
    _verify_record_calibration(
        record_calibration,
        bag=bag,
        policy=policy,
        trusted_record_calibration_registry=trusted_record_calibration_registry,
    )
    prepared = _prepare_events(
        bag,
        trusted_event_calibration_registry=trusted_event_calibration_registry,
    )
    core = _aggregate_core(prepared, policy)
    (
        state,
        channel,
        record_value_semantics,
        probability_authorized,
    ) = _apply_record_calibration(core, record_calibration)
    region = (
        None
        if channel is None
        else _project_channel_axis(
            channel, COMMON17_CHANNEL_TO_REGION, COMMON17_REGION_IDS
        )
    )
    laterality = (
        None
        if channel is None
        else _project_channel_axis(
            channel,
            COMMON17_CHANNEL_TO_LATERALITY,
            COMMON17_LATERALITY_IDS,
        )
    )
    loeo = _loeo(prepared, policy, record_calibration, state, channel)

    mode_rows = []
    for row in core["mode_rows"]:
        mode_rows.append(
            {
                "mode_id": row["mode_id"],
                "event_ids": list(row["event_ids"]),
                "event_count": row["event_count"],
                "raw_state_evidence_weight": _rounded(row["raw_state_evidence_weight"]),
                "capped_state_evidence_weight": _rounded(
                    row["capped_state_evidence_weight"]
                ),
                "raw_spatial_evidence_weight": _rounded(
                    row["raw_spatial_evidence_weight"]
                ),
                "capped_spatial_evidence_weight": _rounded(
                    row["capped_spatial_evidence_weight"]
                ),
                "state_normalized_support": _rounded_mapping(
                    INDEPENDENT_PATTERN_STATE_IDS, row["state"]
                ),
                "channel_normalized_support_conditional_on_localized": (
                    None
                    if row["channel"] is None
                    else _rounded_mapping(COMMON17_CHANNEL_IDS, row["channel"])
                ),
                "mode_outputs_are_not_called_probabilities": True,
            }
        )

    ledger_rows = []
    for item in prepared:
        event = item.event
        weights = core["contribution"][event.event_id]
        ledger_rows.append(
            {
                "event_id": event.event_id,
                "source_event_evidence_sha256": event.source_event_evidence_sha256,
                "event_content_sha256": event.content_sha256,
                "mode_id": event.mode_id,
                "roster_included": True,
                "onset_evidence_ids": list(event.onset_evidence_ids),
                "channel_input_value_semantics": event.channel_value_semantics,
                "channel_calibration_receipt_id": (
                    None
                    if event.channel_calibration is None
                    else event.channel_calibration.receipt_id
                ),
                "state_input_value_semantics": event.state_value_semantics,
                "state_calibration_receipt_id": (
                    None
                    if event.state_calibration is None
                    else event.state_calibration.receipt_id
                ),
                "normalized_channel_input": (
                    None
                    if item.channel is None
                    else _rounded_mapping(COMMON17_CHANNEL_IDS, item.channel)
                ),
                "normalized_independent_state_input": _rounded_mapping(
                    INDEPENDENT_PATTERN_STATE_IDS, item.state
                ),
                "model_reliability": _rounded(event.model_reliability),
                "qc": event.qc.as_dict(),
                "effective_reliability": _rounded(item.reliability),
                "localized_state_mass": _rounded(event.localized_mass),
                "localized_spatial_weight": _rounded(item.localized_weight),
                "included_in_state_pool": item.reliability > 0.0,
                "included_in_spatial_pool": item.localized_weight > 0.0,
                "state_within_mode_weight": _rounded(
                    weights["state_within_mode_weight"]
                ),
                "spatial_within_mode_weight": _rounded(
                    weights["spatial_within_mode_weight"]
                ),
                "state_record_contribution_weight": _rounded(
                    weights["state_record_contribution_weight"]
                ),
                "spatial_record_contribution_weight": _rounded(
                    weights["spatial_record_contribution_weight"]
                ),
                "zero_weight_reason_codes": [
                    reason
                    for condition, reason in (
                        (
                            item.reliability <= 0.0,
                            "zero_eeg_derived_effective_reliability",
                        ),
                        (
                            item.localized_weight <= 0.0,
                            "no_localized_state_mass_for_spatial_pool",
                        ),
                    )
                    if condition
                ],
                "external_context_used": False,
            }
        )

    state_output = {
        "status": core["state_status"],
        "value_semantics": record_value_semantics,
        "probability_language_authorized": probability_authorized,
        "record_calibration_receipt_id": (
            None if record_calibration is None else record_calibration.receipt_id
        ),
        "mass_values": _rounded_mapping(INDEPENDENT_PATTERN_STATE_IDS, state),
        "ranking": _ranked_values(INDEPENDENT_PATTERN_STATE_IDS, state),
        "states_are_independent_from_common17_channel_axis": True,
    }
    if channel is None:
        spatial_output = {
            "status": "not_estimable_no_localized_spatial_evidence",
            "conditioning": "localized_scalp_onset_pattern_only",
            "value_semantics": "not_available",
            "probability_language_authorized": False,
            "record_calibration_receipt_id": None,
            "channel_values": None,
            "channel_ranking": [],
            "region_values": None,
            "region_ranking": [],
            "laterality_values": None,
            "laterality_ranking": [],
            "nonlocalized_states_projected_to_channels": False,
            "cz_is_observed_midline_electrode_not_nonlocalized_bucket": True,
        }
    else:
        assert region is not None and laterality is not None
        spatial_output = {
            "status": "estimated_conditional_on_localized_scalp_onset",
            "conditioning": "localized_scalp_onset_pattern_only",
            "value_semantics": record_value_semantics,
            "probability_language_authorized": probability_authorized,
            "record_calibration_receipt_id": (
                None if record_calibration is None else record_calibration.receipt_id
            ),
            "channel_values": _rounded_mapping(COMMON17_CHANNEL_IDS, channel),
            "channel_ranking": _ranked_values(COMMON17_CHANNEL_IDS, channel),
            "region_values": _rounded_mapping(COMMON17_REGION_IDS, region),
            "region_ranking": _ranked_values(COMMON17_REGION_IDS, region),
            "laterality_values": _rounded_mapping(COMMON17_LATERALITY_IDS, laterality),
            "laterality_ranking": _ranked_values(COMMON17_LATERALITY_IDS, laterality),
            "nonlocalized_states_projected_to_channels": False,
            "cz_is_observed_midline_electrode_not_nonlocalized_bucket": True,
        }

    result: dict[str, Any] = {
        "schema_version": COMMON17_RECORD_SOZ_AGGREGATION_SCHEMA_VERSION,
        "method_id": COMMON17_RECORD_SOZ_AGGREGATION_METHOD_ID,
        "record": {
            "record_id": bag.record_id,
            "canonical_signal_sha256": bag.canonical_signal_sha256,
            "upstream_model_artifact_sha256": bag.upstream_model_artifact_sha256,
            "source_scope": bag.source_scope,
            "source_inference_exclusions": dict(_INFERENCE_EXCLUSIONS),
        },
        "claim_boundary": (
            "research_scalp_visible_ictal_onset_laterality_region_channel_ranking_"
            "not_cortical_soz_ez_or_surgical_target"
        ),
        "common17_ontology": {
            "channel_ids": list(COMMON17_CHANNEL_IDS),
            "excluded_signal_channels": ["FZ", "PZ"],
            "prediction_side_fz_pz_to_cz_mapping_used": False,
            "region_ids": list(COMMON17_REGION_IDS),
            "laterality_ids": list(COMMON17_LATERALITY_IDS),
            "channel_to_region": dict(COMMON17_CHANNEL_TO_REGION),
            "channel_to_laterality": dict(COMMON17_CHANNEL_TO_LATERALITY),
        },
        "policy": {
            "policy_sha256": policy.policy_sha256,
            "mode_state_evidence_cap": policy.mode_state_evidence_cap,
            "mode_spatial_evidence_cap": policy.mode_spatial_evidence_cap,
            "event_pool": "reliability_weighted_linear_pool",
            "mode_pool": "reliability_capped_linear_pool",
            "all_events_enter_before_mode_pooling": True,
        },
        "calibration": {
            "event_calibration_bindings_verified": True,
            "record_calibration_receipt_id": (
                None if record_calibration is None else record_calibration.receipt_id
            ),
            "record_calibration_receipt_sha256": (
                None
                if record_calibration is None
                else record_calibration.receipt_payload_sha256
            ),
            "record_probability_language_authorized": probability_authorized,
            "event_calibration_does_not_imply_record_calibration": True,
        },
        "independent_pattern_state": state_output,
        "spatial_localization": spatial_output,
        "mode_summary": {
            **core["mode_diagnostics"],
            "modes": mode_rows,
        },
        "heterogeneity": core["heterogeneity"],
        "loeo_stability": loeo,
        "evidence_ledger": {
            "detector_event_roster": list(bag.detector_event_roster),
            "detector_event_roster_sha256": bag.detector_event_roster_sha256,
            "input_event_count": len(bag.events),
            "ledger_event_count": len(ledger_rows),
            "all_detector_events_entered": True,
            "excluded_event_ids": [],
            "event_rows": ledger_rows,
        },
        "authorization": {
            "eeg_signal_only": True,
            "late_spread_may_create_onset_support": False,
            "clinical_correctness_claimed": False,
            "cortical_soz_or_ez_claim_authorized": False,
            "report_lexicalization_authorized_by_this_module": False,
        },
    }
    result["aggregation_sha256"] = _self_hash(result, "aggregation_sha256")
    return validate_common17_record_soz_evidence_aggregation_v1(result)


def _require_distribution(
    value: object, expected_ids: Sequence[str], context: str
) -> None:
    if not isinstance(value, Mapping) or set(value) != set(expected_ids):
        raise ValueError(f"{context} has the wrong closed axis")
    values = []
    for identifier in expected_ids:
        raw = value[identifier]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise TypeError(f"{context}.{identifier} must be numeric")
        number = float(raw)
        if not math.isfinite(number) or number < 0.0:
            raise ValueError(f"{context} must be finite and nonnegative")
        values.append(number)
    if not math.isclose(math.fsum(values), 1.0, abs_tol=2e-9):
        raise ValueError(f"{context} must sum to one")


def validate_common17_record_soz_evidence_aggregation_v1(
    value: object,
) -> dict[str, Any]:
    """Validate closure, axis separation, roster completeness, and self-hash."""

    if not isinstance(value, Mapping):
        raise TypeError("common17 record aggregation must be an object")
    data = deepcopy(dict(value))
    if data.get("schema_version") != COMMON17_RECORD_SOZ_AGGREGATION_SCHEMA_VERSION:
        raise ValueError("unexpected common17 record aggregation schema")
    if data.get("method_id") != COMMON17_RECORD_SOZ_AGGREGATION_METHOD_ID:
        raise ValueError("unexpected common17 record aggregation method")
    ontology = data.get("common17_ontology")
    if not isinstance(ontology, Mapping):
        raise TypeError("common17 ontology is missing")
    if ontology.get("channel_ids") != list(COMMON17_CHANNEL_IDS):
        raise ValueError("common17 channel ontology drifted")
    if ontology.get("excluded_signal_channels") != ["FZ", "PZ"] or (
        ontology.get("prediction_side_fz_pz_to_cz_mapping_used") is not False
    ):
        raise ValueError("common17 prediction ontology used a forbidden mapping")

    ledger = data.get("evidence_ledger")
    if not isinstance(ledger, Mapping):
        raise TypeError("evidence ledger is missing")
    roster = ledger.get("detector_event_roster")
    rows = ledger.get("event_rows")
    if not isinstance(roster, list) or not isinstance(rows, list) or not roster:
        raise ValueError("evidence ledger requires a non-empty roster")
    row_ids = [row.get("event_id") for row in rows if isinstance(row, Mapping)]
    if (
        row_ids != roster
        or len(row_ids) != len(rows)
        or len(row_ids) != len(set(row_ids))
        or ledger.get("input_event_count") != len(roster)
        or ledger.get("ledger_event_count") != len(roster)
        or ledger.get("all_detector_events_entered") is not True
        or ledger.get("excluded_event_ids") != []
    ):
        raise ValueError("evidence ledger does not close the detector event roster")
    if any(row.get("roster_included") is not True for row in rows):
        raise ValueError("evidence ledger silently excluded an event")

    state = data.get("independent_pattern_state")
    if not isinstance(state, Mapping):
        raise TypeError("independent pattern state is missing")
    _require_distribution(
        state.get("mass_values"),
        INDEPENDENT_PATTERN_STATE_IDS,
        "independent_pattern_state.mass_values",
    )
    if state.get("states_are_independent_from_common17_channel_axis") is not True:
        raise ValueError("pattern states are not separated from channel localization")
    if state.get("probability_language_authorized") is True:
        if (
            state.get("value_semantics") != CALIBRATED_PROBABILITY
            or state.get("record_calibration_receipt_id") is None
        ):
            raise ValueError("probability wording lacks record calibration")
    elif state.get("value_semantics") == CALIBRATED_PROBABILITY:
        raise ValueError("calibrated probability semantics lack authorization")

    spatial = data.get("spatial_localization")
    if not isinstance(spatial, Mapping):
        raise TypeError("spatial localization is missing")
    if (
        spatial.get("nonlocalized_states_projected_to_channels") is not False
        or spatial.get("cz_is_observed_midline_electrode_not_nonlocalized_bucket")
        is not True
    ):
        raise ValueError("nonlocalized support leaked onto the channel axis")
    if spatial.get("status") == "estimated_conditional_on_localized_scalp_onset":
        _require_distribution(
            spatial.get("channel_values"),
            COMMON17_CHANNEL_IDS,
            "spatial_localization.channel_values",
        )
        _require_distribution(
            spatial.get("region_values"),
            COMMON17_REGION_IDS,
            "spatial_localization.region_values",
        )
        _require_distribution(
            spatial.get("laterality_values"),
            COMMON17_LATERALITY_IDS,
            "spatial_localization.laterality_values",
        )
        if spatial.get("probability_language_authorized") is True:
            if (
                spatial.get("value_semantics") != CALIBRATED_PROBABILITY
                or spatial.get("record_calibration_receipt_id") is None
            ):
                raise ValueError("spatial probability wording lacks calibration")
        elif spatial.get("value_semantics") == CALIBRATED_PROBABILITY:
            raise ValueError("spatial probability semantics lack authorization")
    else:
        if any(
            spatial.get(key) is not None
            for key in ("channel_values", "region_values", "laterality_values")
        ):
            raise ValueError("non-estimable spatial output carries fabricated values")

    loeo = data.get("loeo_stability")
    if not isinstance(loeo, Mapping):
        raise TypeError("LOEO stability is missing")
    if len(roster) == 1:
        if loeo.get("status") != "not_evaluable_single_event" or loeo.get("rows") != []:
            raise ValueError("single-event LOEO must remain non-evaluable")
    else:
        loeo_rows = loeo.get("rows")
        if (
            loeo.get("status") != "evaluated"
            or not isinstance(loeo_rows, list)
            or [row.get("removed_event_id") for row in loeo_rows] != roster
        ):
            raise ValueError("LOEO rows do not cover every event exactly once")

    if data.get("aggregation_sha256") != _self_hash(data, "aggregation_sha256"):
        raise ValueError("common17 record aggregation self-hash mismatch")
    return data


__all__ = [
    "CALIBRATED_PROBABILITY",
    "COMMON17_CHANNEL_IDS",
    "COMMON17_CHANNEL_TO_LATERALITY",
    "COMMON17_CHANNEL_TO_REGION",
    "COMMON17_LATERALITY_IDS",
    "COMMON17_RECORD_SOZ_AGGREGATION_METHOD_ID",
    "COMMON17_RECORD_SOZ_AGGREGATION_SCHEMA_VERSION",
    "COMMON17_REGION_IDS",
    "CalibrationBindingV1",
    "Common17CompleteEventRosterV1",
    "Common17EventQCProfileV1",
    "Common17EventSOZEvidenceV1",
    "Common17RecordCalibrationV1",
    "Common17RecordSOZAggregationPolicyV1",
    "INDEPENDENT_PATTERN_STATE_IDS",
    "NOT_APPLICABLE_NONLOCALIZED",
    "NORMALIZED_SUPPORT_SCORE",
    "UNCALIBRATED_NONNEGATIVE_SCORE",
    "aggregate_common17_record_soz_evidence_v1",
    "validate_common17_record_soz_evidence_aggregation_v1",
]
