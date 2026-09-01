"""Fail-closed event-level baseline/context comparability sidecar.

The event Findings contracts record which signal intervals were inspected, but
an interval being earlier than a detector anchor does not make it a usable
baseline.  This additive sidecar separates four propositions which are often
collapsed in EEG pipelines:

* a context interval was physically observed and measurable;
* it was outside the event protection zone and sufficiently uncontaminated;
* it was technically comparable under the same signal view/reference; and
* a calibrated within-record comparison records an event-contrast or
  return-toward-reference candidate.

The contract never authorizes normative ``normal``/``abnormal`` background
language, onset creation, or SOZ creation.  Until trusted source/quality/
comparison/calibration registries exist, its v1 claim gate authorizes no
report claim at all.  It is deliberately not connected to the private-data
or report routes.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

from .event_findings_v3_validation import (
    validate_event_eeg_findings_v3_payload,
)


EVENT_BASELINE_CONTEXT_COMPARABILITY_SCHEMA_VERSION = (
    "event_baseline_context_comparability_v1"
)
EVENT_BASELINE_CONTEXT_COMPARABILITY_METHOD_ID = (
    "EEG-ONLY-EVENT-BASELINE-CONTEXT-COMPARABILITY-V1"
)

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_HEX = frozenset("0123456789abcdef")
_TOL = 1e-9

_CONTEXT_ROLES = {
    "local_pre_event",
    "distant_pre_event",
    "distant_other_context",
    "event_emergence",
    "post_event_return_candidate",
}
_REFERENCE_ROLES = {"local_pre_event", "distant_pre_event"}
_COMPARISON_PURPOSES = {
    "distant_background_equivalence",
    "event_emergence_contrast",
    "post_event_return_to_reference",
}
_LOCKED_CALIBRATION = {"source_dev_locked", "split_conformal_locked"}
_PERMISSION_KEYS = (
    "context_measurement",
    "within_record_relative_measurement",
    "distant_background_reference",
    "event_emergence_support",
    "return_toward_reference",
    "recovery_support",
    "background_normality_statement",
    "background_abnormality_statement",
    "onset_support",
    "soz_support",
)
_ALWAYS_DENIED = {
    "background_normality_statement": "normative_reference_not_in_scope",
    "background_abnormality_statement": "normative_reference_not_in_scope",
    "onset_support": "context_comparison_cannot_create_onset_evidence",
    "soz_support": "context_comparison_cannot_create_soz_evidence",
}
_V1_UNTRUSTED_REPORT_PERMISSIONS = {
    "distant_background_reference": (
        "v1_trusted_context_comparison_receipts_not_available"
    ),
    "event_emergence_support": (
        "v1_trusted_context_comparison_receipts_not_available"
    ),
    "return_toward_reference": (
        "v1_trusted_context_comparison_receipts_not_available"
    ),
    "recovery_support": (
        "v1_trusted_context_comparison_receipts_not_available"
    ),
}
_V1_CLAIM_AUTHORIZATION_DISABLED_REASON = (
    "v1_baseline_context_sidecar_is_candidate_only_not_report_authorization"
)
_CLAIM_TO_PERMISSION = {
    "quantitative_context_measurement": "context_measurement",
    "within_record_relative_measurement": "within_record_relative_measurement",
    "distant_background_reference": "distant_background_reference",
    "event_emergence_support": "event_emergence_support",
    "return_toward_reference": "return_toward_reference",
    "recovery_support": "recovery_support",
    "background_normality_statement": "background_normality_statement",
    "background_abnormality_statement": "background_abnormality_statement",
    "onset_support": "onset_support",
    "soz_support": "soz_support",
}

_SCOPE_RECEIPT = {
    "eeg_samples_used": True,
    "detector_posterior_used_for_contamination_only": True,
    "edf_annotations_used": False,
    "spreadsheet_used": False,
    "doctor_labels_used": False,
    "clinical_text_used": False,
    "patient_metadata_used": False,
    "video_used": False,
    "sleep_staging_used": False,
    "provocation_used": False,
    "normative_background_reference_used": False,
    "background_normality_authorized": False,
    "background_abnormality_authorized": False,
    "onset_evidence_created": False,
    "soz_evidence_created": False,
    "private_or_report_route_connected": False,
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


def _identifier(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or not _ID.fullmatch(value):
        raise ValueError(f"{context} must be a safe non-empty identifier")
    if len(value) > 256:
        raise ValueError(f"{context} is too long")
    return value


def _sha256(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _finite(value: object, context: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    if minimum is not None and result < minimum - _TOL:
        raise ValueError(f"{context} is below its minimum")
    return result


def _fraction(value: object, context: str) -> float:
    result = _finite(value, context, minimum=0.0)
    if result > 1.0 + _TOL:
        raise ValueError(f"{context} must be in [0,1]")
    return result


def _interval(
    value: Sequence[object], context: str, *, duration: float | None = None
) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or len(value) != 2:
        raise TypeError(f"{context} must be a two-number interval")
    start = _finite(value[0], f"{context}[0]", minimum=0.0)
    stop = _finite(value[1], f"{context}[1]", minimum=0.0)
    if stop <= start + _TOL:
        raise ValueError(f"{context} must have positive duration")
    if duration is not None and stop > duration + _TOL:
        raise ValueError(f"{context} lies outside the recording")
    return start, stop


def _overlap(left: tuple[float, float], right: tuple[float, float]) -> float:
    return max(0.0, min(left[1], right[1]) - max(left[0], right[0]))


def _same_interval(
    left: tuple[float, float], right: tuple[float, float]
) -> bool:
    return math.isclose(left[0], right[0], abs_tol=_TOL) and math.isclose(
        left[1], right[1], abs_tol=_TOL
    )


def _strict_keys(value: object, keys: set[str], context: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{context} must be an object")
    if set(value) != keys:
        raise ValueError(f"{context} has missing or unknown fields")
    return deepcopy(value)


@dataclass(frozen=True)
class BaselineContextSourceBinding:
    """Content-addressed signal-view/reference identity for one interval."""

    canonical_signal_sha256: str
    canonical_receipt_sha256: str
    source_view_id: str
    view_role: str
    view_receipt_id: str
    view_receipt_sha256: str
    transform_spec_sha256: str
    processed_view_sha256: str
    reference_type: str
    source_unit_ids: tuple[str, ...]
    sample_rate_numerator: int
    sample_rate_denominator: int
    effective_bandwidth_hz: tuple[float, float]
    view_recording_interval: tuple[float, float]
    quality_mask_sha256: str
    dependency_policy: str
    future_sample_access: bool

    def __post_init__(self) -> None:
        _sha256(self.canonical_signal_sha256, "canonical_signal_sha256")
        for name in (
            "canonical_receipt_sha256",
            "view_receipt_sha256",
            "transform_spec_sha256",
            "processed_view_sha256",
            "quality_mask_sha256",
        ):
            _sha256(getattr(self, name), name)
        for name in ("source_view_id", "view_receipt_id"):
            _identifier(getattr(self, name), name)
        if not isinstance(self.view_role, str) or not self.view_role:
            raise ValueError("view_role must be non-empty")
        if self.view_role in {"detector_provider", "detector_native"}:
            raise ValueError("detector-native views cannot carry Findings context")
        if not isinstance(self.reference_type, str) or not self.reference_type.strip():
            raise ValueError("reference_type must be non-empty")
        units = tuple(_identifier(item, "source_unit_ids") for item in self.source_unit_ids)
        if not units or len(units) != len(set(units)):
            raise ValueError("source_unit_ids must be non-empty and unique")
        object.__setattr__(self, "source_unit_ids", units)
        if type(self.sample_rate_numerator) is not int or self.sample_rate_numerator < 1:
            raise ValueError("sample_rate_numerator must be a positive integer")
        if type(self.sample_rate_denominator) is not int or self.sample_rate_denominator < 1:
            raise ValueError("sample_rate_denominator must be a positive integer")
        low, high = _interval(self.effective_bandwidth_hz, "effective_bandwidth_hz")
        if high > 0.5 * self.sample_rate_numerator / self.sample_rate_denominator + _TOL:
            raise ValueError("effective bandwidth exceeds Nyquist")
        object.__setattr__(self, "effective_bandwidth_hz", (low, high))
        view_interval = _interval(self.view_recording_interval, "view_recording_interval")
        object.__setattr__(self, "view_recording_interval", view_interval)
        if self.dependency_policy not in {
            "instantaneous",
            "past_and_present_only",
            "bidirectional_or_unknown",
        }:
            raise ValueError("dependency_policy is invalid")
        if type(self.future_sample_access) is not bool:
            raise TypeError("future_sample_access must be boolean")
        if self.future_sample_access and self.dependency_policy != "bidirectional_or_unknown":
            raise ValueError("future-dependent context must declare bidirectional/unknown")

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_signal_sha256": self.canonical_signal_sha256,
            "canonical_receipt_sha256": self.canonical_receipt_sha256,
            "source_view_id": self.source_view_id,
            "view_role": self.view_role,
            "view_receipt_id": self.view_receipt_id,
            "view_receipt_sha256": self.view_receipt_sha256,
            "transform_spec_sha256": self.transform_spec_sha256,
            "processed_view_sha256": self.processed_view_sha256,
            "reference_type": self.reference_type,
            "source_unit_ids": list(self.source_unit_ids),
            "sample_rate_numerator": self.sample_rate_numerator,
            "sample_rate_denominator": self.sample_rate_denominator,
            "effective_bandwidth_hz": list(self.effective_bandwidth_hz),
            "view_recording_interval": list(self.view_recording_interval),
            "quality_mask_sha256": self.quality_mask_sha256,
            "dependency_policy": self.dependency_policy,
            "future_sample_access": self.future_sample_access,
        }

    @classmethod
    def from_dict(cls, value: object) -> "BaselineContextSourceBinding":
        data = _strict_keys(value, set(cls.__dataclass_fields__), "source_binding")
        data["source_unit_ids"] = tuple(data["source_unit_ids"])
        data["effective_bandwidth_hz"] = tuple(data["effective_bandwidth_hz"])
        data["view_recording_interval"] = tuple(data["view_recording_interval"])
        return cls(**data)


@dataclass(frozen=True)
class BaselineContextQuality:
    qualification_status: str
    usable_fraction: float
    artifact_fraction: float
    edge_fraction: float
    padding_fraction: float
    imputation_fraction: float
    missing_unit_fraction: float
    method_id: str
    policy_sha256: str
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.qualification_status not in {
            "qualified",
            "limited",
            "failed",
            "not_evaluable",
        }:
            raise ValueError("quality qualification_status is invalid")
        for name in (
            "usable_fraction",
            "artifact_fraction",
            "edge_fraction",
            "padding_fraction",
            "imputation_fraction",
            "missing_unit_fraction",
        ):
            _fraction(getattr(self, name), name)
        _identifier(self.method_id, "quality method_id")
        _sha256(self.policy_sha256, "quality policy_sha256")
        reasons = tuple(sorted(set(_identifier(item, "quality reason") for item in self.reason_codes)))
        object.__setattr__(self, "reason_codes", reasons)
        if self.qualification_status == "qualified" and reasons:
            raise ValueError("qualified quality cannot carry failure reasons")
        if self.qualification_status != "qualified" and not reasons:
            raise ValueError("non-qualified quality requires reason codes")

    def to_dict(self) -> dict[str, Any]:
        return {
            "qualification_status": self.qualification_status,
            "usable_fraction": float(self.usable_fraction),
            "artifact_fraction": float(self.artifact_fraction),
            "edge_fraction": float(self.edge_fraction),
            "padding_fraction": float(self.padding_fraction),
            "imputation_fraction": float(self.imputation_fraction),
            "missing_unit_fraction": float(self.missing_unit_fraction),
            "method_id": self.method_id,
            "policy_sha256": self.policy_sha256,
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_dict(cls, value: object) -> "BaselineContextQuality":
        data = _strict_keys(value, set(cls.__dataclass_fields__), "quality")
        data["reason_codes"] = tuple(data["reason_codes"])
        return cls(**data)


@dataclass(frozen=True)
class BaselineContextContamination:
    status: str
    detector_posterior_max: float
    candidate_event_overlap_fraction: float
    artifact_candidate_fraction: float
    method_id: str
    policy_sha256: str
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in {"clear", "possible", "contaminated", "not_evaluable"}:
            raise ValueError("contamination status is invalid")
        for name in (
            "detector_posterior_max",
            "candidate_event_overlap_fraction",
            "artifact_candidate_fraction",
        ):
            _fraction(getattr(self, name), name)
        _identifier(self.method_id, "contamination method_id")
        _sha256(self.policy_sha256, "contamination policy_sha256")
        reasons = tuple(sorted(set(_identifier(item, "contamination reason") for item in self.reason_codes)))
        object.__setattr__(self, "reason_codes", reasons)
        if self.status == "clear":
            if reasons:
                raise ValueError("clear contamination status cannot carry reasons")
            if self.candidate_event_overlap_fraction > _TOL:
                raise ValueError("clear context cannot overlap another event candidate")
        elif not reasons:
            raise ValueError("non-clear contamination requires reason codes")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "detector_posterior_max": float(self.detector_posterior_max),
            "candidate_event_overlap_fraction": float(
                self.candidate_event_overlap_fraction
            ),
            "artifact_candidate_fraction": float(self.artifact_candidate_fraction),
            "method_id": self.method_id,
            "policy_sha256": self.policy_sha256,
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_dict(cls, value: object) -> "BaselineContextContamination":
        data = _strict_keys(value, set(cls.__dataclass_fields__), "contamination")
        data["reason_codes"] = tuple(data["reason_codes"])
        return cls(**data)


@dataclass(frozen=True)
class EventContextSegment:
    context_id: str
    role: str
    interval_recording_seconds: tuple[float, float]
    source_binding: BaselineContextSourceBinding
    quality: BaselineContextQuality
    contamination: BaselineContextContamination
    selection_method_id: str
    selection_policy_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.context_id, "context_id")
        if self.role not in _CONTEXT_ROLES:
            raise ValueError("context role is invalid")
        object.__setattr__(
            self,
            "interval_recording_seconds",
            _interval(self.interval_recording_seconds, "context interval"),
        )
        if not isinstance(self.source_binding, BaselineContextSourceBinding):
            raise TypeError("source_binding must be BaselineContextSourceBinding")
        if not isinstance(self.quality, BaselineContextQuality):
            raise TypeError("quality must be BaselineContextQuality")
        if not isinstance(self.contamination, BaselineContextContamination):
            raise TypeError("contamination must be BaselineContextContamination")
        _identifier(self.selection_method_id, "selection_method_id")
        _sha256(self.selection_policy_sha256, "selection_policy_sha256")

    @classmethod
    def from_receipt_row(cls, value: object) -> "EventContextSegment":
        keys = {
            "context_id",
            "role",
            "interval_recording_seconds",
            "source_binding",
            "quality",
            "contamination",
            "selection_method_id",
            "selection_policy_sha256",
            "relation_to_protection",
            "distance_to_protection_seconds",
            "protection_overlap_seconds",
            "eligibility",
        }
        data = _strict_keys(value, keys, "context segment")
        return cls(
            context_id=data["context_id"],
            role=data["role"],
            interval_recording_seconds=tuple(data["interval_recording_seconds"]),
            source_binding=BaselineContextSourceBinding.from_dict(
                data["source_binding"]
            ),
            quality=BaselineContextQuality.from_dict(data["quality"]),
            contamination=BaselineContextContamination.from_dict(
                data["contamination"]
            ),
            selection_method_id=data["selection_method_id"],
            selection_policy_sha256=data["selection_policy_sha256"],
        )


@dataclass(frozen=True)
class ContextSimilarity:
    feature_set_id: str
    feature_set_sha256: str
    method_id: str
    policy_sha256: str
    calibration_status: str
    calibration_receipt_id: str | None
    calibration_receipt_sha256: str | None
    score: float | None
    threshold: float | None
    uncertainty_margin: float | None
    direction: str

    def __post_init__(self) -> None:
        _identifier(self.feature_set_id, "feature_set_id")
        _sha256(self.feature_set_sha256, "feature_set_sha256")
        _identifier(self.method_id, "similarity method_id")
        _sha256(self.policy_sha256, "similarity policy_sha256")
        if self.calibration_status not in _LOCKED_CALIBRATION | {
            "unvalidated",
            "not_available",
        }:
            raise ValueError("similarity calibration_status is invalid")
        if self.direction not in {"higher_is_more_similar", "lower_is_more_similar"}:
            raise ValueError("similarity direction is invalid")
        values = (self.score, self.threshold, self.uncertainty_margin)
        if all(item is None for item in values):
            if self.calibration_status != "not_available":
                raise ValueError("absent similarity values require not_available")
            if self.calibration_receipt_id is not None or self.calibration_receipt_sha256 is not None:
                raise ValueError("not-available similarity cannot cite calibration")
            return
        if any(item is None for item in values):
            raise ValueError("similarity score, threshold and margin are atomic")
        _finite(self.score, "similarity score")
        _finite(self.threshold, "similarity threshold")
        _finite(self.uncertainty_margin, "similarity uncertainty_margin", minimum=0.0)
        if self.calibration_status in _LOCKED_CALIBRATION:
            _identifier(self.calibration_receipt_id, "calibration_receipt_id")
            _sha256(self.calibration_receipt_sha256, "calibration_receipt_sha256")
        elif self.calibration_receipt_id is not None or self.calibration_receipt_sha256 is not None:
            raise ValueError("unvalidated similarity cannot cite a locked calibration")

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_set_id": self.feature_set_id,
            "feature_set_sha256": self.feature_set_sha256,
            "method_id": self.method_id,
            "policy_sha256": self.policy_sha256,
            "calibration_status": self.calibration_status,
            "calibration_receipt_id": self.calibration_receipt_id,
            "calibration_receipt_sha256": self.calibration_receipt_sha256,
            "score": self.score,
            "threshold": self.threshold,
            "uncertainty_margin": self.uncertainty_margin,
            "direction": self.direction,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ContextSimilarity":
        data = _strict_keys(value, set(cls.__dataclass_fields__), "similarity")
        return cls(**data)

    def status(self) -> str:
        if self.score is None:
            return "not_evaluable"
        if self.calibration_status not in _LOCKED_CALIBRATION:
            return "uncertain"
        score = float(self.score)
        threshold = float(self.threshold)
        margin = float(self.uncertainty_margin)
        if self.direction == "higher_is_more_similar":
            if score >= threshold + margin - _TOL:
                return "matched"
            if score <= threshold - margin + _TOL:
                return "not_matched"
        else:
            if score <= threshold - margin + _TOL:
                return "matched"
            if score >= threshold + margin - _TOL:
                return "not_matched"
        return "uncertain"


@dataclass(frozen=True)
class EventContextComparison:
    comparison_id: str
    purpose: str
    target_context_id: str
    reference_context_ids: tuple[str, ...]
    similarity: ContextSimilarity

    def __post_init__(self) -> None:
        _identifier(self.comparison_id, "comparison_id")
        if self.purpose not in _COMPARISON_PURPOSES:
            raise ValueError("comparison purpose is invalid")
        _identifier(self.target_context_id, "target_context_id")
        refs = tuple(_identifier(item, "reference_context_id") for item in self.reference_context_ids)
        if not refs or len(refs) != len(set(refs)) or self.target_context_id in refs:
            raise ValueError("comparison references must be unique and exclude target")
        object.__setattr__(self, "reference_context_ids", refs)
        if not isinstance(self.similarity, ContextSimilarity):
            raise TypeError("similarity must be ContextSimilarity")

    @classmethod
    def from_receipt_row(cls, value: object) -> "EventContextComparison":
        keys = {
            "comparison_id",
            "purpose",
            "target_context_id",
            "reference_context_ids",
            "similarity",
            "technical_comparability",
            "permissions",
        }
        data = _strict_keys(value, keys, "context comparison")
        similarity = deepcopy(data["similarity"])
        if type(similarity) is not dict or similarity.pop("status", None) not in {
            "matched",
            "not_matched",
            "uncertain",
            "not_evaluable",
        }:
            raise ValueError("comparison similarity status is invalid")
        return cls(
            comparison_id=data["comparison_id"],
            purpose=data["purpose"],
            target_context_id=data["target_context_id"],
            reference_context_ids=tuple(data["reference_context_ids"]),
            similarity=ContextSimilarity.from_dict(similarity),
        )


@dataclass(frozen=True)
class BaselineContextComparabilityPolicy:
    local_pre_event_max_gap_seconds: float = 60.0
    minimum_reference_duration_seconds: float = 8.0
    minimum_return_observation_seconds: float = 8.0
    calibration_status: str = "unvalidated"
    calibration_receipt_id: str | None = None
    calibration_receipt_sha256: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "local_pre_event_max_gap_seconds",
            "minimum_reference_duration_seconds",
            "minimum_return_observation_seconds",
        ):
            if _finite(getattr(self, name), name, minimum=0.0) <= _TOL:
                raise ValueError(f"{name} must be positive")
        if self.calibration_status not in _LOCKED_CALIBRATION | {"unvalidated"}:
            raise ValueError("comparability policy calibration status is invalid")
        if self.calibration_status in _LOCKED_CALIBRATION:
            _identifier(self.calibration_receipt_id, "policy calibration_receipt_id")
            _sha256(self.calibration_receipt_sha256, "policy calibration_receipt_sha256")
        elif self.calibration_receipt_id is not None or self.calibration_receipt_sha256 is not None:
            raise ValueError("unvalidated policy cannot cite a locked calibration")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "event_baseline_context_comparability_policy_v1",
            "local_pre_event_max_gap_seconds": float(
                self.local_pre_event_max_gap_seconds
            ),
            "minimum_reference_duration_seconds": float(
                self.minimum_reference_duration_seconds
            ),
            "minimum_return_observation_seconds": float(
                self.minimum_return_observation_seconds
            ),
            "calibration_status": self.calibration_status,
            "calibration_receipt_id": self.calibration_receipt_id,
            "calibration_receipt_sha256": self.calibration_receipt_sha256,
            "exact_view_receipt_required": True,
            "exact_reference_required": True,
            "exact_unit_set_required": True,
            "exact_clock_required": True,
            "exact_bandwidth_required": True,
            "padding_or_imputation_allowed_for_reference": False,
            "distant_reference_requires_local_equivalence": True,
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> "BaselineContextComparabilityPolicy":
        keys = {
            "schema_version",
            "local_pre_event_max_gap_seconds",
            "minimum_reference_duration_seconds",
            "minimum_return_observation_seconds",
            "calibration_status",
            "calibration_receipt_id",
            "calibration_receipt_sha256",
            "exact_view_receipt_required",
            "exact_reference_required",
            "exact_unit_set_required",
            "exact_clock_required",
            "exact_bandwidth_required",
            "padding_or_imputation_allowed_for_reference",
            "distant_reference_requires_local_equivalence",
        }
        data = _strict_keys(value, keys, "comparability policy")
        if data.pop("schema_version") != "event_baseline_context_comparability_policy_v1":
            raise ValueError("comparability policy schema drifted")
        expected = {
            "exact_view_receipt_required": True,
            "exact_reference_required": True,
            "exact_unit_set_required": True,
            "exact_clock_required": True,
            "exact_bandwidth_required": True,
            "padding_or_imputation_allowed_for_reference": False,
            "distant_reference_requires_local_equivalence": True,
        }
        for key, required in expected.items():
            if data.pop(key) is not required:
                raise ValueError(f"comparability policy weakened {key}")
        return cls(**data)


DEFAULT_BASELINE_CONTEXT_COMPARABILITY_POLICY = (
    BaselineContextComparabilityPolicy()
)


def _boundary_event_interval(window: Mapping[str, Any]) -> list[float] | None:
    onset = window["onset_boundary"]
    offset = window["offset_boundary"]
    if onset["status"] != "observed" or offset["status"] != "observed":
        return None
    if onset["interval"] is None or offset["interval"] is None:
        return None
    start = float(onset["interval"]["lower"])
    stop = float(offset["interval"]["upper"])
    return [start, stop] if stop > start else None


def _relation_to_protection(
    interval: tuple[float, float], protection: tuple[float, float]
) -> tuple[str, float, float]:
    overlap = _overlap(interval, protection)
    if interval[1] <= protection[0] + _TOL:
        return "before_protection", max(0.0, protection[0] - interval[1]), overlap
    if interval[0] >= protection[1] - _TOL:
        return "after_protection", max(0.0, interval[0] - protection[1]), overlap
    return "overlaps_protection", 0.0, overlap


def _derive_segment(
    segment: EventContextSegment,
    *,
    event_binding: Mapping[str, Any],
    window_binding: Mapping[str, Any],
    event_context_status: str,
    policy: BaselineContextComparabilityPolicy,
) -> dict[str, Any]:
    duration = float(event_binding["recording_duration_seconds"])
    interval = _interval(
        segment.interval_recording_seconds,
        f"context {segment.context_id}",
        duration=duration,
    )
    view_interval = segment.source_binding.view_recording_interval
    if interval[0] < view_interval[0] - _TOL or interval[1] > view_interval[1] + _TOL:
        raise ValueError(f"context {segment.context_id} lies outside its signal view")
    if segment.source_binding.canonical_signal_sha256 != event_binding[
        "canonical_signal_sha256"
    ]:
        raise ValueError(f"context {segment.context_id} belongs to another signal")
    protection = _interval(
        window_binding["protection_zone_interval"], "protection zone"
    )
    relation, distance, protection_overlap = _relation_to_protection(
        interval, protection
    )
    reasons: list[str] = []
    if segment.quality.qualification_status != "qualified":
        reasons.append("quality_not_qualified")
    if segment.contamination.status != "clear":
        reasons.append("contamination_not_clear")
    if segment.quality.edge_fraction > _TOL:
        reasons.append("edge_samples_present")
    if segment.quality.padding_fraction > _TOL:
        reasons.append("padding_present")
    if segment.quality.imputation_fraction > _TOL:
        reasons.append("imputation_present")
    context_measurement = not reasons

    reference_reasons = list(reasons)
    if segment.role not in _REFERENCE_ROLES:
        reference_reasons.append("not_a_pre_event_reference_role")
    if relation != "before_protection" or protection_overlap > _TOL:
        reference_reasons.append("reference_not_outside_before_protection")
    if interval[1] - interval[0] < policy.minimum_reference_duration_seconds - _TOL:
        reference_reasons.append("reference_duration_insufficient")
    if event_context_status != "available":
        reference_reasons.append("event_findings_background_not_available")
    if (
        segment.role == "local_pre_event"
        and distance > policy.local_pre_event_max_gap_seconds + _TOL
    ):
        reference_reasons.append("local_reference_too_distant")
    if (
        segment.role == "distant_pre_event"
        and distance <= policy.local_pre_event_max_gap_seconds + _TOL
    ):
        reference_reasons.append("distant_reference_is_local_by_policy")

    return_reasons = list(reasons)
    if segment.role != "post_event_return_candidate":
        return_reasons.append("not_a_post_event_return_role")
    if relation != "after_protection" or protection_overlap > _TOL:
        return_reasons.append("return_candidate_not_after_protection")
    if window_binding["offset_boundary_status"] != "observed":
        return_reasons.append("offset_not_observed")
    if bool(window_binding["right_censored"]):
        return_reasons.append("event_right_censored")
    if interval[1] - interval[0] < policy.minimum_return_observation_seconds - _TOL:
        return_reasons.append("return_observation_too_short")
    event_interval = window_binding["event_interval_recording_seconds"]
    if event_interval is None or interval[0] < float(event_interval[1]) - _TOL:
        return_reasons.append("return_candidate_precedes_observed_event_offset")

    event_target_reasons = list(reasons)
    if segment.role != "event_emergence":
        event_target_reasons.append("not_an_event_emergence_role")
    if relation != "overlaps_protection" or protection_overlap <= _TOL:
        event_target_reasons.append("event_target_outside_protection")

    return {
        "context_id": segment.context_id,
        "role": segment.role,
        "interval_recording_seconds": list(interval),
        "source_binding": segment.source_binding.to_dict(),
        "quality": segment.quality.to_dict(),
        "contamination": segment.contamination.to_dict(),
        "selection_method_id": segment.selection_method_id,
        "selection_policy_sha256": segment.selection_policy_sha256,
        "relation_to_protection": relation,
        "distance_to_protection_seconds": float(distance),
        "protection_overlap_seconds": float(protection_overlap),
        "eligibility": {
            "context_measurement": context_measurement,
            "baseline_reference": not reference_reasons,
            "event_emergence_target": not event_target_reasons,
            "post_event_return_target": not return_reasons,
            "context_measurement_reason_codes": sorted(set(reasons)),
            "baseline_reference_reason_codes": sorted(set(reference_reasons)),
            "event_emergence_target_reason_codes": sorted(
                set(event_target_reasons)
            ),
            "post_event_return_target_reason_codes": sorted(set(return_reasons)),
        },
    }


def _technical_comparability(
    comparison: EventContextComparison,
    segment_map: Mapping[str, Mapping[str, Any]],
    *,
    qualified_distant: set[str],
) -> tuple[str, list[str]]:
    target = segment_map[comparison.target_context_id]
    references = [segment_map[item] for item in comparison.reference_context_ids]
    reasons: list[str] = []
    if not target["eligibility"]["context_measurement"]:
        reasons.append("target_context_not_measurable")
    for row in references:
        if not row["eligibility"]["baseline_reference"]:
            reasons.append(f"reference_not_eligible:{row['context_id']}")
        if row["role"] == "distant_pre_event" and comparison.purpose != (
            "distant_background_equivalence"
        ) and row["context_id"] not in qualified_distant:
            reasons.append(f"distant_reference_not_equivalence_qualified:{row['context_id']}")

    source_rows = [target["source_binding"], *(row["source_binding"] for row in references)]
    exact_fields = (
        "canonical_signal_sha256",
        "canonical_receipt_sha256",
        "source_view_id",
        "view_role",
        "view_receipt_id",
        "view_receipt_sha256",
        "transform_spec_sha256",
        "processed_view_sha256",
        "reference_type",
        "source_unit_ids",
        "sample_rate_numerator",
        "sample_rate_denominator",
        "effective_bandwidth_hz",
        "quality_mask_sha256",
    )
    for field in exact_fields:
        if any(row[field] != source_rows[0][field] for row in source_rows[1:]):
            reasons.append(f"source_binding_mismatch:{field}")

    target_role = target["role"]
    reference_roles = {row["role"] for row in references}
    if comparison.purpose == "distant_background_equivalence":
        if target_role != "local_pre_event" or reference_roles != {"distant_pre_event"}:
            reasons.append("invalid_background_equivalence_roles")
    elif comparison.purpose == "event_emergence_contrast":
        if target_role != "event_emergence" or not reference_roles.issubset(
            _REFERENCE_ROLES
        ):
            reasons.append("invalid_event_contrast_roles")
        if not target["eligibility"]["event_emergence_target"]:
            reasons.append("event_emergence_target_not_eligible")
    else:
        if target_role != "post_event_return_candidate" or not reference_roles.issubset(
            _REFERENCE_ROLES
        ):
            reasons.append("invalid_return_comparison_roles")
        if not target["eligibility"]["post_event_return_target"]:
            reasons.append("post_event_return_target_not_eligible")
    return ("comparable" if not reasons else "not_comparable"), sorted(set(reasons))


def _derive_comparison(
    comparison: EventContextComparison,
    segment_map: Mapping[str, Mapping[str, Any]],
    *,
    qualified_distant: set[str],
    policy: BaselineContextComparabilityPolicy,
) -> dict[str, Any]:
    if comparison.target_context_id not in segment_map:
        raise ValueError("comparison target context is unknown")
    missing = sorted(set(comparison.reference_context_ids).difference(segment_map))
    if missing:
        raise ValueError(f"comparison references unknown contexts: {missing}")
    status, reasons = _technical_comparability(
        comparison, segment_map, qualified_distant=qualified_distant
    )
    similarity_status = comparison.similarity.status()
    calibration_locked = bool(
        policy.calibration_status in _LOCKED_CALIBRATION
        and comparison.similarity.calibration_status in _LOCKED_CALIBRATION
    )
    comparable = status == "comparable"
    permissions = {key: False for key in _PERMISSION_KEYS}
    permissions["within_record_relative_measurement"] = comparable and (
        comparison.purpose
        in {"event_emergence_contrast", "post_event_return_to_reference"}
    )
    permissions["distant_background_reference"] = bool(
        comparison.purpose == "distant_background_equivalence"
        and comparable
        and calibration_locked
        and similarity_status == "matched"
    )
    permissions["event_emergence_support"] = bool(
        comparison.purpose == "event_emergence_contrast"
        and comparable
        and calibration_locked
        and similarity_status == "not_matched"
    )
    permissions["return_toward_reference"] = bool(
        comparison.purpose == "post_event_return_to_reference"
        and comparable
        and calibration_locked
        and similarity_status == "matched"
    )
    permissions["recovery_support"] = permissions["return_toward_reference"]
    permission_reasons: list[str] = list(reasons)
    if not calibration_locked:
        permission_reasons.append("comparison_or_policy_not_calibration_locked")
    if similarity_status not in {"matched", "not_matched"}:
        permission_reasons.append("similarity_not_decisive")
    for key, reason in _ALWAYS_DENIED.items():
        permissions[key] = False
        permission_reasons.append(reason)
    # v1 has no trusted registry binding the source view, quality decision,
    # similarity measurements and calibration receipt.  Preserve measurable
    # and technically-comparable *candidate* states, but do not authorize a
    # report conclusion from self-declared metadata.
    for key, reason in _V1_UNTRUSTED_REPORT_PERMISSIONS.items():
        permissions[key] = False
        permission_reasons.append(reason)
    return {
        "comparison_id": comparison.comparison_id,
        "purpose": comparison.purpose,
        "target_context_id": comparison.target_context_id,
        "reference_context_ids": list(comparison.reference_context_ids),
        "similarity": {
            **comparison.similarity.to_dict(),
            "status": similarity_status,
        },
        "technical_comparability": {
            "status": status,
            "reason_codes": reasons,
            "exact_view_receipt": "required",
            "exact_reference": "required",
            "exact_unit_set": "required",
            "exact_clock": "required",
            "exact_bandwidth": "required",
        },
        "permissions": {
            **permissions,
            "reason_codes": sorted(set(permission_reasons)),
        },
    }


def _aggregate_permissions(
    segments: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    measurable = sorted(
        str(row["context_id"])
        for row in segments
        if row["eligibility"]["context_measurement"]
    )
    result["context_measurement"] = {
        "authorized": bool(measurable),
        "context_ids": measurable,
        "comparison_ids": [],
        "reason_codes": [] if measurable else ["no_measurable_context_segment"],
    }
    for permission in _PERMISSION_KEYS[1:]:
        comparison_ids = sorted(
            str(row["comparison_id"])
            for row in comparisons
            if bool(row["permissions"][permission])
        )
        context_ids = sorted(
            {
                str(row["target_context_id"])
                for row in comparisons
                if bool(row["permissions"][permission])
            }
            | {
                str(context_id)
                for row in comparisons
                if bool(row["permissions"][permission])
                for context_id in row["reference_context_ids"]
            }
        )
        denied_reason = _ALWAYS_DENIED.get(permission) or (
            _V1_UNTRUSTED_REPORT_PERMISSIONS.get(permission)
        )
        if denied_reason is not None:
            result[permission] = {
                "authorized": False,
                "context_ids": [],
                "comparison_ids": [],
                "reason_codes": [denied_reason],
            }
        else:
            result[permission] = {
                "authorized": bool(comparison_ids),
                "context_ids": context_ids,
                "comparison_ids": comparison_ids,
                "reason_codes": (
                    [] if comparison_ids else [f"no_authorized_{permission}"]
                ),
            }
    return result


def _event_binding(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "event_id": str(payload["event_id"]),
        "recording_id": str(payload["provenance"]["record_id"]),
        "canonical_signal_sha256": str(
            payload["provenance"]["canonical_signal_sha256"]
        ),
        "recording_duration_seconds": float(
            payload["coordinates"]["recording_duration_seconds"]
        ),
        "event_findings_v3_sha256": _canonical_sha256(payload),
    }


def _window_binding(payload: Mapping[str, Any]) -> dict[str, Any]:
    window = payload["window"]
    protection = window["protection_zone"]
    return {
        "protection_zone_id": str(protection["protection_zone_id"]),
        "protection_zone_interval": [float(item) for item in protection["interval"]],
        "protection_zone_policy_sha256": str(protection["policy_sha256"]),
        "onset_boundary_status": str(window["onset_boundary"]["status"]),
        "offset_boundary_status": str(window["offset_boundary"]["status"]),
        "event_interval_recording_seconds": _boundary_event_interval(window),
        "left_censored": bool(window["left_censored"]),
        "right_censored": bool(window["right_censored"]),
    }


def _context_projection(payload: Mapping[str, Any]) -> dict[str, Any]:
    context = payload["context"]
    return {
        "event_findings_background_status": str(context["background_status"]),
        "local_background_intervals": [
            [float(item) for item in interval]
            for interval in context["local_background_intervals"]
        ],
        "distant_background_intervals": [
            [float(item) for item in interval]
            for interval in context["distant_background_intervals"]
        ],
        "source_contamination_risk": float(context["contamination_risk"]),
        "selection_scope": str(context["selection_scope"]),
    }


def _check_projection_intervals(
    projection: Mapping[str, Any], segments: Sequence[EventContextSegment]
) -> None:
    for role, key in (
        ("local_pre_event", "local_background_intervals"),
        ("distant_pre_event", "distant_background_intervals"),
    ):
        expected = sorted(
            tuple(float(item) for item in interval) for interval in projection[key]
        )
        actual = sorted(
            segment.interval_recording_seconds
            for segment in segments
            if segment.role == role
        )
        if len(expected) != len(actual) or any(
            not _same_interval(left, right) for left, right in zip(expected, actual)
        ):
            raise ValueError(
                f"{role} segments do not exactly bind event Findings context intervals"
            )


def build_event_baseline_context_comparability_receipt(
    event_findings_v3: object,
    *,
    context_segments: Sequence[EventContextSegment],
    comparisons: Sequence[EventContextComparison],
    policy: BaselineContextComparabilityPolicy = (
        DEFAULT_BASELINE_CONTEXT_COMPARABILITY_POLICY
    ),
    trusted_producer_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_calibration_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_capability_qualification_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_sensitivity_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_term_decision_receipts: Mapping[str, Mapping[str, object]] | None = None,
    trusted_registry_bindings: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, Any]:
    """Build a content-addressed sidecar bound to one validated v3 event."""

    payload = validate_event_eeg_findings_v3_payload(
        event_findings_v3,
        trusted_producer_receipts=trusted_producer_receipts,
        trusted_calibration_receipts=trusted_calibration_receipts,
        trusted_capability_qualification_receipts=(
            trusted_capability_qualification_receipts
        ),
        trusted_sensitivity_receipts=trusted_sensitivity_receipts,
        trusted_term_decision_receipts=trusted_term_decision_receipts,
        trusted_registry_bindings=trusted_registry_bindings,
    )
    if not isinstance(policy, BaselineContextComparabilityPolicy):
        raise TypeError("policy must be BaselineContextComparabilityPolicy")
    segment_inputs = list(context_segments)
    if not segment_inputs or any(
        not isinstance(item, EventContextSegment) for item in segment_inputs
    ):
        raise TypeError("context_segments must contain EventContextSegment values")
    if len({item.context_id for item in segment_inputs}) != len(segment_inputs):
        raise ValueError("context segment IDs must be unique")
    comparison_inputs = list(comparisons)
    if any(not isinstance(item, EventContextComparison) for item in comparison_inputs):
        raise TypeError("comparisons must contain EventContextComparison values")
    if len({item.comparison_id for item in comparison_inputs}) != len(comparison_inputs):
        raise ValueError("comparison IDs must be unique")

    event_binding = _event_binding(payload)
    window_binding = _window_binding(payload)
    projection = _context_projection(payload)
    _check_projection_intervals(projection, segment_inputs)
    segment_rows = [
        _derive_segment(
            item,
            event_binding=event_binding,
            window_binding=window_binding,
            event_context_status=projection["event_findings_background_status"],
            policy=policy,
        )
        for item in segment_inputs
    ]
    segment_rows.sort(key=lambda row: (row["interval_recording_seconds"], row["context_id"]))
    segment_map = {str(row["context_id"]): row for row in segment_rows}

    equivalence_inputs = [
        item
        for item in comparison_inputs
        if item.purpose == "distant_background_equivalence"
    ]
    other_inputs = [
        item
        for item in comparison_inputs
        if item.purpose != "distant_background_equivalence"
    ]
    comparison_rows: list[dict[str, Any]] = []
    qualified_distant: set[str] = set()
    for item in equivalence_inputs:
        row = _derive_comparison(
            item, segment_map, qualified_distant=set(), policy=policy
        )
        comparison_rows.append(row)
        if row["permissions"]["distant_background_reference"]:
            qualified_distant.update(
                context_id
                for context_id in row["reference_context_ids"]
                if segment_map[context_id]["role"] == "distant_pre_event"
            )
    for item in other_inputs:
        comparison_rows.append(
            _derive_comparison(
                item,
                segment_map,
                qualified_distant=qualified_distant,
                policy=policy,
            )
        )
    comparison_rows.sort(key=lambda row: str(row["comparison_id"]))
    qualified_local = {
        str(row["context_id"])
        for row in segment_rows
        if row["role"] == "local_pre_event"
        and row["eligibility"]["baseline_reference"]
    }
    qualified_references = sorted(qualified_local | qualified_distant)
    permissions = _aggregate_permissions(segment_rows, comparison_rows)
    core = {
        "schema_version": EVENT_BASELINE_CONTEXT_COMPARABILITY_SCHEMA_VERSION,
        "receipt_id": "CONTENT-ADDRESS-PENDING",
        "method_id": EVENT_BASELINE_CONTEXT_COMPARABILITY_METHOD_ID,
        "event_binding": event_binding,
        "window_binding": window_binding,
        "source_context_projection": projection,
        "context_segments": segment_rows,
        "comparisons": comparison_rows,
        "qualified_reference_context_ids": qualified_references,
        "permissions": permissions,
        "policy": {**policy.to_dict(), "policy_sha256": policy.sha256},
        "scope_receipt": deepcopy(_SCOPE_RECEIPT),
        "limitations": [
            "within_record_signal_comparison_not_normative_background_assessment",
            "context_comparison_cannot_create_onset_or_soz_evidence",
            "recovery_support_means_return_toward_a_qualified_within_record_reference_only",
            "unvalidated_thresholds_fail_closed",
            "v1_context_comparison_metadata_is_candidate_only_not_report_authorization",
        ],
        "receipt_sha256": "CONTENT-ADDRESS-PENDING",
    }
    id_source = deepcopy(core)
    core["receipt_id"] = "BASECTX-" + _canonical_sha256(id_source)[:24]
    digest_source = deepcopy(core)
    core["receipt_sha256"] = _canonical_sha256(digest_source)
    return validate_event_baseline_context_comparability_receipt(core)


def validate_event_baseline_context_comparability_receipt(
    value: object,
) -> dict[str, Any]:
    """Validate a sidecar, recomputing all eligibility and permissions."""

    keys = {
        "schema_version",
        "receipt_id",
        "method_id",
        "event_binding",
        "window_binding",
        "source_context_projection",
        "context_segments",
        "comparisons",
        "qualified_reference_context_ids",
        "permissions",
        "policy",
        "scope_receipt",
        "limitations",
        "receipt_sha256",
    }
    data = _strict_keys(value, keys, "baseline/context comparability receipt")
    if data["schema_version"] != EVENT_BASELINE_CONTEXT_COMPARABILITY_SCHEMA_VERSION:
        raise ValueError("baseline/context comparability schema drifted")
    if data["method_id"] != EVENT_BASELINE_CONTEXT_COMPARABILITY_METHOD_ID:
        raise ValueError("baseline/context comparability method drifted")
    if data["scope_receipt"] != _SCOPE_RECEIPT:
        raise ValueError("baseline/context comparability scope was weakened")
    _identifier(data["receipt_id"], "receipt_id")
    _sha256(data["receipt_sha256"], "receipt_sha256")
    event_binding = _strict_keys(
        data["event_binding"],
        {
            "event_id",
            "recording_id",
            "canonical_signal_sha256",
            "recording_duration_seconds",
            "event_findings_v3_sha256",
        },
        "event_binding",
    )
    _identifier(event_binding["event_id"], "event_id")
    _identifier(event_binding["recording_id"], "recording_id")
    _sha256(event_binding["canonical_signal_sha256"], "canonical_signal_sha256")
    _sha256(event_binding["event_findings_v3_sha256"], "event_findings_v3_sha256")
    duration = _finite(
        event_binding["recording_duration_seconds"],
        "recording_duration_seconds",
        minimum=0.0,
    )
    if duration <= _TOL:
        raise ValueError("recording_duration_seconds must be positive")
    window_binding = _strict_keys(
        data["window_binding"],
        {
            "protection_zone_id",
            "protection_zone_interval",
            "protection_zone_policy_sha256",
            "onset_boundary_status",
            "offset_boundary_status",
            "event_interval_recording_seconds",
            "left_censored",
            "right_censored",
        },
        "window_binding",
    )
    _identifier(window_binding["protection_zone_id"], "protection_zone_id")
    _sha256(window_binding["protection_zone_policy_sha256"], "protection policy")
    _interval(window_binding["protection_zone_interval"], "protection zone", duration=duration)
    if window_binding["event_interval_recording_seconds"] is not None:
        _interval(window_binding["event_interval_recording_seconds"], "event interval", duration=duration)
    if type(window_binding["left_censored"]) is not bool or type(window_binding["right_censored"]) is not bool:
        raise TypeError("window censoring flags must be boolean")
    projection = _strict_keys(
        data["source_context_projection"],
        {
            "event_findings_background_status",
            "local_background_intervals",
            "distant_background_intervals",
            "source_contamination_risk",
            "selection_scope",
        },
        "source_context_projection",
    )
    if projection["event_findings_background_status"] not in {
        "available", "limited", "unavailable", "unknown"
    }:
        raise ValueError("source background status is invalid")
    if projection["selection_scope"] != "eeg_detector_quality_only":
        raise ValueError("source context selection scope drifted")
    _fraction(projection["source_contamination_risk"], "source contamination risk")
    for key in ("local_background_intervals", "distant_background_intervals"):
        for index, interval in enumerate(projection[key]):
            _interval(interval, f"{key}[{index}]", duration=duration)

    policy_row = deepcopy(data["policy"])
    if type(policy_row) is not dict or "policy_sha256" not in policy_row:
        raise ValueError("comparability policy is malformed")
    policy_sha = policy_row.pop("policy_sha256")
    _sha256(policy_sha, "policy_sha256")
    policy = BaselineContextComparabilityPolicy.from_dict(policy_row)
    if policy_sha != policy.sha256:
        raise ValueError("comparability policy hash drifted")

    if not isinstance(data["context_segments"], list) or not data["context_segments"]:
        raise ValueError("receipt requires at least one context segment")
    inputs = [EventContextSegment.from_receipt_row(row) for row in data["context_segments"]]
    if len({item.context_id for item in inputs}) != len(inputs):
        raise ValueError("context segment IDs are duplicated")
    _check_projection_intervals(projection, inputs)
    expected_segments = [
        _derive_segment(
            item,
            event_binding=event_binding,
            window_binding=window_binding,
            event_context_status=projection["event_findings_background_status"],
            policy=policy,
        )
        for item in inputs
    ]
    expected_segments.sort(key=lambda row: (row["interval_recording_seconds"], row["context_id"]))
    if data["context_segments"] != expected_segments:
        raise ValueError("context segment eligibility or ordering was forged")
    segment_map = {str(row["context_id"]): row for row in expected_segments}

    if not isinstance(data["comparisons"], list):
        raise TypeError("comparisons must be an array")
    comparison_inputs = [EventContextComparison.from_receipt_row(row) for row in data["comparisons"]]
    if len({item.comparison_id for item in comparison_inputs}) != len(comparison_inputs):
        raise ValueError("comparison IDs are duplicated")
    equivalence = sorted(
        (item for item in comparison_inputs if item.purpose == "distant_background_equivalence"),
        key=lambda item: item.comparison_id,
    )
    others = sorted(
        (item for item in comparison_inputs if item.purpose != "distant_background_equivalence"),
        key=lambda item: item.comparison_id,
    )
    expected_comparisons: list[dict[str, Any]] = []
    qualified_distant: set[str] = set()
    for item in equivalence:
        row = _derive_comparison(item, segment_map, qualified_distant=set(), policy=policy)
        expected_comparisons.append(row)
        if row["permissions"]["distant_background_reference"]:
            qualified_distant.update(
                context_id
                for context_id in row["reference_context_ids"]
                if segment_map[context_id]["role"] == "distant_pre_event"
            )
    for item in others:
        expected_comparisons.append(
            _derive_comparison(item, segment_map, qualified_distant=qualified_distant, policy=policy)
        )
    expected_comparisons.sort(key=lambda row: str(row["comparison_id"]))
    if data["comparisons"] != expected_comparisons:
        raise ValueError("comparability, similarity status or permissions were forged")
    qualified_local = {
        str(row["context_id"])
        for row in expected_segments
        if row["role"] == "local_pre_event" and row["eligibility"]["baseline_reference"]
    }
    if data["qualified_reference_context_ids"] != sorted(qualified_local | qualified_distant):
        raise ValueError("qualified reference roster drifted")
    expected_permissions = _aggregate_permissions(expected_segments, expected_comparisons)
    if data["permissions"] != expected_permissions:
        raise ValueError("aggregate baseline/context permissions were forged")
    for key, reason in _ALWAYS_DENIED.items():
        permission = data["permissions"].get(key)
        if permission != {
            "authorized": False,
            "context_ids": [],
            "comparison_ids": [],
            "reason_codes": [reason],
        }:
            raise ValueError(f"forbidden permission {key} was enabled")
    for key, reason in _V1_UNTRUSTED_REPORT_PERMISSIONS.items():
        permission = data["permissions"].get(key)
        if permission != {
            "authorized": False,
            "context_ids": [],
            "comparison_ids": [],
            "reason_codes": [reason],
        }:
            raise ValueError(f"untrusted-v1 report permission {key} was enabled")

    digest_source = deepcopy(data)
    digest_source["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["receipt_sha256"] != _canonical_sha256(digest_source):
        raise ValueError("baseline/context receipt hash does not bind its content")
    id_source = deepcopy(data)
    id_source["receipt_id"] = "CONTENT-ADDRESS-PENDING"
    id_source["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    if data["receipt_id"] != "BASECTX-" + _canonical_sha256(id_source)[:24]:
        raise ValueError("baseline/context receipt ID does not bind its content")
    return data


def validate_event_baseline_context_comparability_against_findings(
    event_findings_v3: object,
    receipt: object,
    **trusted_validation_context: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the sidecar's immutable binding to its v3 event graph."""

    payload = validate_event_eeg_findings_v3_payload(
        event_findings_v3, **trusted_validation_context
    )
    sidecar = validate_event_baseline_context_comparability_receipt(receipt)
    expected = _event_binding(payload)
    if sidecar["event_binding"] != expected:
        raise ValueError("baseline/context sidecar belongs to different event Findings")
    expected_window = _window_binding(payload)
    if sidecar["window_binding"] != expected_window:
        raise ValueError("baseline/context sidecar window/protection binding drifted")
    if sidecar["source_context_projection"] != _context_projection(payload):
        raise ValueError("baseline/context sidecar source context projection drifted")
    return payload, sidecar


def validate_baseline_context_claim_authorizations(
    claims: object, receipt: object
) -> list[dict[str, Any]]:
    """Reject report/Finding uses not explicitly authorized by the sidecar.

    This is a future downstream gate, not a route connection.  v1 currently
    has no trusted producer/calibration registries, so *every* non-empty claim
    list fails closed after its candidate bindings are checked.  In
    particular, a measurable interval or technically comparable comparison is
    not report authorization, and normative background claims remain denied
    because this EEG-only contract has no normative cohort.
    """

    sidecar = validate_event_baseline_context_comparability_receipt(receipt)
    if not isinstance(claims, list):
        raise TypeError("baseline/context claims must be an array")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(claims):
        row = _strict_keys(
            raw,
            {"claim_id", "claim_type", "context_ids", "comparison_ids"},
            f"claims[{index}]",
        )
        claim_id = _identifier(row["claim_id"], f"claims[{index}].claim_id")
        if claim_id in seen:
            raise ValueError("claim IDs must be unique")
        seen.add(claim_id)
        claim_type = row["claim_type"]
        if claim_type not in _CLAIM_TO_PERMISSION:
            raise ValueError(f"claims[{index}] has unsupported claim_type")
        context_ids = sorted(set(_identifier(item, "claim context_id") for item in row["context_ids"]))
        comparison_ids = sorted(set(_identifier(item, "claim comparison_id") for item in row["comparison_ids"]))
        permission_name = _CLAIM_TO_PERMISSION[claim_type]
        permission = sidecar["permissions"][permission_name]
        if not permission["authorized"]:
            raise ValueError(
                f"claim {claim_id} is denied: {permission_name}; "
                f"{permission['reason_codes']}"
            )
        if permission_name != "context_measurement":
            if not comparison_ids or not set(comparison_ids).issubset(
                permission["comparison_ids"]
            ):
                raise ValueError(f"claim {claim_id} cites unauthorized comparisons")
            comparison_map = {
                str(item["comparison_id"]): item
                for item in sidecar["comparisons"]
            }
            expected_context_ids = sorted(
                {
                    str(context_id)
                    for comparison_id in comparison_ids
                    for context_id in (
                        comparison_map[comparison_id]["target_context_id"],
                        *comparison_map[comparison_id]["reference_context_ids"],
                    )
                }
            )
            if context_ids != expected_context_ids:
                raise ValueError(
                    f"claim {claim_id} context IDs do not exactly match "
                    "the cited comparison endpoints"
                )
        elif not context_ids or not set(context_ids).issubset(
            permission["context_ids"]
        ):
            raise ValueError(f"claim {claim_id} cites unauthorized context IDs")
        elif comparison_ids:
            raise ValueError("quantitative context measurement cannot cite a comparison")
        normalized.append(
            {
                "claim_id": claim_id,
                "claim_type": claim_type,
                "context_ids": context_ids,
                "comparison_ids": comparison_ids,
            }
        )
    if normalized:
        raise ValueError(_V1_CLAIM_AUTHORIZATION_DISABLED_REASON)
    return normalized


__all__ = [
    "EVENT_BASELINE_CONTEXT_COMPARABILITY_SCHEMA_VERSION",
    "EVENT_BASELINE_CONTEXT_COMPARABILITY_METHOD_ID",
    "BaselineContextSourceBinding",
    "BaselineContextQuality",
    "BaselineContextContamination",
    "EventContextSegment",
    "ContextSimilarity",
    "EventContextComparison",
    "BaselineContextComparabilityPolicy",
    "DEFAULT_BASELINE_CONTEXT_COMPARABILITY_POLICY",
    "build_event_baseline_context_comparability_receipt",
    "validate_event_baseline_context_comparability_receipt",
    "validate_event_baseline_context_comparability_against_findings",
    "validate_baseline_context_claim_authorizations",
]
