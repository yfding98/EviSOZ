"""EEG-only, target-free event qualification for BA-IEG record bags.

The provider in this module is intentionally narrow.  A caller cannot submit
an aggregation status.  It can submit only a content-bound causal global
onset observation, typed-unit opportunity/QC/technical state, and an optional
frozen source-development threshold receipt.  The provider derives the
status and emits a content-addressed per-event receipt.

Without a valid threshold receipt, an otherwise evaluable event is always
``uncertain``.  Technical failure or absent analysis support is
``not_evaluable``.  ``qualified_ictal`` is therefore impossible without a
validated, content-bound source-development threshold receipt.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import torch

from .ba_ieg_shallow_causal_typed_unit_head_v1 import (
    BAIEGShallowCausalTypedUnitHeadOutput,
)


BA_IEG_TARGET_FREE_EVENT_QUALIFICATION_PROVIDER_ID_V1: Final[str] = (
    "ba_ieg_eeg_only_target_free_event_qualification_provider_v1"
)
BA_IEG_SOURCE_DEV_EVENT_QUALIFICATION_THRESHOLD_SCHEMA_V1: Final[str] = (
    "ba_ieg_source_dev_event_qualification_threshold_receipt_v1"
)
BA_IEG_EVENT_QUALIFICATION_OBSERVATION_SCHEMA_V1: Final[str] = (
    "ba_ieg_eeg_only_causal_event_qualification_observation_v1"
)
BA_IEG_TARGET_FREE_EVENT_QUALIFICATION_RECEIPT_SCHEMA_V1: Final[str] = (
    "ba_ieg_target_free_event_qualification_receipt_v1"
)
BA_IEG_SOURCE_DEV_THRESHOLD_REGISTRY_SCHEMA_V1: Final[str] = (
    "ba_ieg_source_dev_event_qualification_trusted_registry_v1"
)
TRUSTED_BA_IEG_PRODUCTION_THRESHOLD_REGISTRY_RECEIPT_SHA256: Final[str] = (
    "c41444f83c53b805662d1626434eeb24a9cc15130a9a51df78fe425bc2fafe1f"
)
BA_IEG_LEGACY_CALLER_STATUS_AUTHORITY: Final[str] = (
    "legacy_caller_status_vector_component_test_only_not_a_provider"
)
BA_IEG_EVENT_QUALIFICATION_STATUSES: Final[tuple[str, ...]] = (
    "qualified_ictal",
    "uncertain",
    "not_evaluable",
)

_SHA256_CHARACTERS: Final[frozenset[str]] = frozenset("0123456789abcdef")
_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_PRODUCTION_THRESHOLD_REGISTRY_PATH: Final[Path] = (
    _ROOT
    / "configs/clinical_eeg_ba_ieg_event_qualification_threshold_registry_v1.json"
)
_CAUSAL_GLOBAL_ONSET_STATES: Final[tuple[str, ...]] = (
    "resolved",
    "unresolved",
    "unavailable",
)
_TYPED_OPPORTUNITY_STATES: Final[tuple[str, ...]] = (
    "available",
    "no_support",
)
_QC_STATES: Final[tuple[str, ...]] = ("pass", "fail")
_TECHNICAL_STATES: Final[tuple[str, ...]] = ("ok", "failure")

_EEG_ONLY_SCOPE: Final[dict[str, bool]] = {
    "eeg_signal_only": True,
    "causal_global_onset_state_used": True,
    "typed_opportunity_qc_technical_state_used": True,
    "source_development_threshold_receipt_only": True,
    "edf_annotations_opened": False,
    "spreadsheet_opened": False,
    "doctor_labels_opened": False,
    "clinical_text_opened": False,
    "seizure_type_opened": False,
    "localization_target_opened": False,
    "private_evaluation_target_opened": False,
}

_FORBIDDEN_INPUT_KEY_FRAGMENTS: Final[tuple[str, ...]] = (
    "annotation",
    "doctor",
    "clinical_text",
    "spreadsheet",
    "excel",
    "ground_truth",
    "localization_target",
    "soz_target",
    "target_channel",
    "seizure_type",
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _content_receipt_sha256(payload: Mapping[str, Any]) -> str:
    candidate = deepcopy(dict(payload))
    candidate["receipt_sha256"] = "CONTENT-ADDRESS-PENDING"
    return _canonical_sha256(candidate)


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_CHARACTERS for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _identifier(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 2048
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"{name} must be a non-empty trimmed identifier")
    return value


def _finite_probability(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite probability")
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or result > 1.0:
        raise ValueError(f"{name} must lie in [0,1]")
    return result


def _exact_keys(
    value: Mapping[str, Any], expected: set[str], name: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    candidate = deepcopy(dict(value))
    _reject_forbidden_input_keys(candidate)
    if set(candidate) != expected:
        missing = sorted(expected - set(candidate))
        extra = sorted(set(candidate) - expected)
        raise ValueError(f"{name} keys drifted; missing={missing}, extra={extra}")
    return candidate


def _reject_forbidden_input_keys(value: object) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            lowered = str(key).lower()
            if key in _EEG_ONLY_SCOPE and nested is _EEG_ONLY_SCOPE[key]:
                # Negative scope declarations are mandatory audit evidence,
                # not forbidden inputs.
                continue
            if any(fragment in lowered for fragment in _FORBIDDEN_INPUT_KEY_FRAGMENTS):
                raise ValueError(
                    "target/annotation/clinical inputs are forbidden in event qualification"
                )
            _reject_forbidden_input_keys(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _reject_forbidden_input_keys(nested)


def _validate_scope(value: Mapping[str, Any], name: str) -> None:
    if dict(value) != _EEG_ONLY_SCOPE:
        raise ValueError(f"{name} EEG-only scope drifted")


def load_ba_ieg_production_threshold_registry_v1() -> dict[str, Any]:
    """Load the fixed production registry; callers cannot inject an allowlist."""

    path = _PRODUCTION_THRESHOLD_REGISTRY_PATH
    if not path.is_file() or path.is_symlink():
        raise ValueError("production threshold registry must be a regular file")
    candidate = _exact_keys(
        json.loads(path.read_text(encoding="utf-8")),
        {
            "schema_version",
            "registry_id",
            "deployment_scope",
            "trusted_threshold_receipt_sha256s",
            "status",
            "receipt_sha256",
        },
        "production threshold registry",
    )
    if (
        candidate["schema_version"]
        != BA_IEG_SOURCE_DEV_THRESHOLD_REGISTRY_SCHEMA_V1
        or candidate["registry_id"]
        != "BAIEG-EVENT-QUAL-PRODUCTION-REGISTRY-V1"
        or candidate["deployment_scope"] != "production"
        or candidate["status"] != "no_production_threshold_preregistered"
    ):
        raise ValueError("production threshold registry authority drifted")
    trusted = candidate["trusted_threshold_receipt_sha256s"]
    if not isinstance(trusted, list) or trusted:
        raise ValueError(
            "v1 production threshold registry is frozen with no trusted threshold"
        )
    if candidate["receipt_sha256"] != (
        TRUSTED_BA_IEG_PRODUCTION_THRESHOLD_REGISTRY_RECEIPT_SHA256
    ):
        raise ValueError("production threshold registry fixed SHA drifted")
    if candidate["receipt_sha256"] != _content_receipt_sha256(candidate):
        raise ValueError("production threshold registry hash does not bind content")
    return candidate


def _component_test_threshold_registry_receipt_sha256(
    threshold_receipt_sha256: str,
) -> str:
    return _canonical_sha256(
        {
            "schema_version": "ba_ieg_component_test_threshold_registry_v1",
            "deployment_scope": "component_test_only_non_production",
            "trusted_threshold_receipt_sha256s": [
                _sha256(threshold_receipt_sha256, "threshold_receipt_sha256")
            ],
            "evidence_grade_authorized": False,
        }
    )


@dataclass(frozen=True)
class BAIEGSourceDevEventQualificationThresholdReceiptV1:
    """Frozen source-development threshold, independent of evaluation targets."""

    source_development_scope_receipt_sha256: str
    model_checkpoint_sha256: str
    threshold_selection_receipt_sha256: str
    causal_global_observed_onset_mass_threshold: float
    schema_version: str = BA_IEG_SOURCE_DEV_EVENT_QUALIFICATION_THRESHOLD_SCHEMA_V1
    provider_id: str = BA_IEG_TARGET_FREE_EVENT_QUALIFICATION_PROVIDER_ID_V1
    selection_split: str = "source_dev"
    metric_id: str = "causal_global_observed_onset_mass"
    frozen: bool = True
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "source_development_scope_receipt_sha256",
            "model_checkpoint_sha256",
            "threshold_selection_receipt_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        threshold = _finite_probability(
            self.causal_global_observed_onset_mass_threshold,
            "causal_global_observed_onset_mass_threshold",
        )
        object.__setattr__(
            self, "causal_global_observed_onset_mass_threshold", threshold
        )
        if (
            self.schema_version
            != BA_IEG_SOURCE_DEV_EVENT_QUALIFICATION_THRESHOLD_SCHEMA_V1
            or self.provider_id
            != BA_IEG_TARGET_FREE_EVENT_QUALIFICATION_PROVIDER_ID_V1
            or self.selection_split != "source_dev"
            or self.metric_id != "causal_global_observed_onset_mass"
            or self.frozen is not True
        ):
            raise ValueError("source-dev threshold receipt authority drifted")
        object.__setattr__(
            self, "receipt_sha256", _content_receipt_sha256(self.to_dict(False))
        )

    def to_dict(self, include_receipt: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "provider_id": self.provider_id,
            "selection_split": self.selection_split,
            "metric_id": self.metric_id,
            "source_development_scope_receipt_sha256": (
                self.source_development_scope_receipt_sha256
            ),
            "model_checkpoint_sha256": self.model_checkpoint_sha256,
            "threshold_selection_receipt_sha256": (
                self.threshold_selection_receipt_sha256
            ),
            "causal_global_observed_onset_mass_threshold": (
                self.causal_global_observed_onset_mass_threshold
            ),
            "frozen": self.frozen,
            "input_scope": deepcopy(_EEG_ONLY_SCOPE),
        }
        if include_receipt:
            result["receipt_sha256"] = self.receipt_sha256
        return result

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "BAIEGSourceDevEventQualificationThresholdReceiptV1":
        expected = {
            "schema_version",
            "provider_id",
            "selection_split",
            "metric_id",
            "source_development_scope_receipt_sha256",
            "model_checkpoint_sha256",
            "threshold_selection_receipt_sha256",
            "causal_global_observed_onset_mass_threshold",
            "frozen",
            "input_scope",
            "receipt_sha256",
        }
        candidate = _exact_keys(value, expected, "source-dev threshold receipt")
        _validate_scope(candidate["input_scope"], "source-dev threshold receipt")
        result = cls(
            source_development_scope_receipt_sha256=candidate[
                "source_development_scope_receipt_sha256"
            ],
            model_checkpoint_sha256=candidate["model_checkpoint_sha256"],
            threshold_selection_receipt_sha256=candidate[
                "threshold_selection_receipt_sha256"
            ],
            causal_global_observed_onset_mass_threshold=candidate[
                "causal_global_observed_onset_mass_threshold"
            ],
            schema_version=candidate["schema_version"],
            provider_id=candidate["provider_id"],
            selection_split=candidate["selection_split"],
            metric_id=candidate["metric_id"],
            frozen=candidate["frozen"],
        )
        if candidate["receipt_sha256"] != result.receipt_sha256:
            raise ValueError("source-dev threshold receipt hash does not bind content")
        return result


@dataclass(frozen=True)
class BAIEGEventQualificationObservationV1:
    """Content-bound EEG-only event state consumed by the provider."""

    source_input_batch_sha256: str
    event_id: str
    recording_id: str
    source_event_receipt_sha256: str
    source_model_state_sha256: str
    technical_state_receipt_sha256: str
    causal_global_onset_state: str
    causal_global_observed_onset_mass: float
    causal_global_left_censored_mass: float
    causal_global_no_onset_mass: float
    causal_analysis_support_count: int
    causal_onset_support_count: int
    typed_opportunity_state: str
    qc_state: str
    technical_state: str
    schema_version: str = BA_IEG_EVENT_QUALIFICATION_OBSERVATION_SCHEMA_V1
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "source_input_batch_sha256",
            "source_event_receipt_sha256",
            "source_model_state_sha256",
            "technical_state_receipt_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        object.__setattr__(self, "event_id", _identifier(self.event_id, "event_id"))
        object.__setattr__(
            self, "recording_id", _identifier(self.recording_id, "recording_id")
        )
        if self.schema_version != BA_IEG_EVENT_QUALIFICATION_OBSERVATION_SCHEMA_V1:
            raise ValueError("event qualification observation schema drifted")
        if self.causal_global_onset_state not in _CAUSAL_GLOBAL_ONSET_STATES:
            raise ValueError("causal global onset state is unsupported")
        if self.typed_opportunity_state not in _TYPED_OPPORTUNITY_STATES:
            raise ValueError("typed opportunity state is unsupported")
        if self.qc_state not in _QC_STATES:
            raise ValueError("QC state is unsupported")
        if self.technical_state not in _TECHNICAL_STATES:
            raise ValueError("technical state is unsupported")
        masses = tuple(
            _finite_probability(getattr(self, name), name)
            for name in (
                "causal_global_observed_onset_mass",
                "causal_global_left_censored_mass",
                "causal_global_no_onset_mass",
            )
        )
        for name, value in zip(
            (
                "causal_global_observed_onset_mass",
                "causal_global_left_censored_mass",
                "causal_global_no_onset_mass",
            ),
            masses,
        ):
            object.__setattr__(self, name, value)
        for name in ("causal_analysis_support_count", "causal_onset_support_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.causal_onset_support_count > self.causal_analysis_support_count:
            raise ValueError("onset support cannot exceed causal analysis support")
        if self.causal_global_onset_state == "unavailable":
            if self.causal_analysis_support_count != 0:
                raise ValueError("unavailable onset state cannot claim causal support")
        else:
            if self.causal_analysis_support_count < 1:
                raise ValueError("evaluable onset state needs causal analysis support")
            if not math.isclose(sum(masses), 1.0, rel_tol=1e-4, abs_tol=1e-4):
                raise ValueError("causal global onset state masses must sum to one")
            resolved = masses[0] > masses[1] + masses[2]
            if resolved != (self.causal_global_onset_state == "resolved"):
                raise ValueError("causal global onset resolved state disagrees with mass")
        object.__setattr__(
            self, "receipt_sha256", _content_receipt_sha256(self.to_dict(False))
        )

    def to_dict(self, include_receipt: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "source_input_batch_sha256": self.source_input_batch_sha256,
            "event_id": self.event_id,
            "recording_id": self.recording_id,
            "source_event_receipt_sha256": self.source_event_receipt_sha256,
            "source_model_state_sha256": self.source_model_state_sha256,
            "technical_state_receipt_sha256": self.technical_state_receipt_sha256,
            "causal_global_onset_state": self.causal_global_onset_state,
            "causal_global_observed_onset_mass": (
                self.causal_global_observed_onset_mass
            ),
            "causal_global_left_censored_mass": (
                self.causal_global_left_censored_mass
            ),
            "causal_global_no_onset_mass": self.causal_global_no_onset_mass,
            "causal_analysis_support_count": self.causal_analysis_support_count,
            "causal_onset_support_count": self.causal_onset_support_count,
            "typed_opportunity_state": self.typed_opportunity_state,
            "qc_state": self.qc_state,
            "technical_state": self.technical_state,
            "input_scope": deepcopy(_EEG_ONLY_SCOPE),
        }
        if include_receipt:
            result["receipt_sha256"] = self.receipt_sha256
        return result

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "BAIEGEventQualificationObservationV1":
        expected = {
            "schema_version",
            "source_input_batch_sha256",
            "event_id",
            "recording_id",
            "source_event_receipt_sha256",
            "source_model_state_sha256",
            "technical_state_receipt_sha256",
            "causal_global_onset_state",
            "causal_global_observed_onset_mass",
            "causal_global_left_censored_mass",
            "causal_global_no_onset_mass",
            "causal_analysis_support_count",
            "causal_onset_support_count",
            "typed_opportunity_state",
            "qc_state",
            "technical_state",
            "input_scope",
            "receipt_sha256",
        }
        candidate = _exact_keys(value, expected, "event qualification observation")
        _validate_scope(candidate["input_scope"], "event qualification observation")
        result = cls(
            **{
                key: candidate[key]
                for key in expected - {"input_scope", "receipt_sha256"}
            }
        )
        if candidate["receipt_sha256"] != result.receipt_sha256:
            raise ValueError("event observation receipt hash does not bind content")
        return result


def _derive_qualification_status(
    observation: BAIEGEventQualificationObservationV1,
    threshold: BAIEGSourceDevEventQualificationThresholdReceiptV1 | None,
) -> tuple[str, str]:
    if observation.technical_state == "failure":
        return "not_evaluable", "technical_failure"
    if observation.qc_state == "fail":
        return "not_evaluable", "qc_failure"
    if observation.causal_global_onset_state == "unavailable":
        return "not_evaluable", "causal_global_onset_support_unavailable"
    if observation.typed_opportunity_state == "no_support":
        return "not_evaluable", "typed_unit_opportunity_unavailable"
    if threshold is None:
        return "uncertain", "source_dev_threshold_receipt_unavailable"
    if observation.causal_global_onset_state != "resolved":
        return "uncertain", "causal_global_onset_unresolved"
    if (
        observation.causal_global_observed_onset_mass
        < threshold.causal_global_observed_onset_mass_threshold
    ):
        return "uncertain", "below_frozen_source_dev_threshold"
    return "qualified_ictal", "meets_frozen_source_dev_threshold"


@dataclass(frozen=True)
class BAIEGTargetFreeEventQualificationReceiptV1:
    """Provider-derived, content-addressed status for exactly one event."""

    observation: BAIEGEventQualificationObservationV1
    occurrence_equivalence_id: str
    threshold_receipt: BAIEGSourceDevEventQualificationThresholdReceiptV1 | None
    threshold_registry_scope: str
    threshold_registry_receipt_sha256: str
    schema_version: str = BA_IEG_TARGET_FREE_EVENT_QUALIFICATION_RECEIPT_SCHEMA_V1
    provider_id: str = BA_IEG_TARGET_FREE_EVENT_QUALIFICATION_PROVIDER_ID_V1
    event_aggregation_status: str = field(init=False)
    disposition_reason: str = field(init=False)
    production_threshold_trusted: bool = field(init=False)
    component_test_only: bool = field(init=False)
    evidence_grade_qualification_authorized: bool = field(init=False)
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.observation, BAIEGEventQualificationObservationV1):
            raise TypeError("qualification receipt needs a typed EEG observation")
        object.__setattr__(
            self,
            "occurrence_equivalence_id",
            _identifier(self.occurrence_equivalence_id, "occurrence_equivalence_id"),
        )
        if self.threshold_receipt is not None and not isinstance(
            self.threshold_receipt,
            BAIEGSourceDevEventQualificationThresholdReceiptV1,
        ):
            raise TypeError("threshold receipt must be typed and content-bound")
        object.__setattr__(
            self,
            "threshold_registry_receipt_sha256",
            _sha256(
                self.threshold_registry_receipt_sha256,
                "threshold_registry_receipt_sha256",
            ),
        )
        if (
            self.schema_version
            != BA_IEG_TARGET_FREE_EVENT_QUALIFICATION_RECEIPT_SCHEMA_V1
            or self.provider_id
            != BA_IEG_TARGET_FREE_EVENT_QUALIFICATION_PROVIDER_ID_V1
        ):
            raise ValueError("event qualification provider authority drifted")
        if self.threshold_registry_scope == "production_preregistered":
            registry = load_ba_ieg_production_threshold_registry_v1()
            if self.threshold_registry_receipt_sha256 != registry["receipt_sha256"]:
                raise ValueError("production threshold registry binding drifted")
            trusted = frozenset(registry["trusted_threshold_receipt_sha256s"])
            if (
                self.threshold_receipt is not None
                and self.threshold_receipt.receipt_sha256 not in trusted
            ):
                raise ValueError(
                    "threshold receipt SHA is not preregistered for production"
                )
            production_authorized = True
            component_test_only = False
        elif self.threshold_registry_scope == (
            "component_test_only_injected_registry"
        ):
            if self.threshold_receipt is None:
                raise ValueError("component-test threshold registry needs a threshold")
            expected_registry_sha256 = (
                _component_test_threshold_registry_receipt_sha256(
                    self.threshold_receipt.receipt_sha256
                )
            )
            if self.threshold_registry_receipt_sha256 != expected_registry_sha256:
                raise ValueError("component-test threshold registry binding drifted")
            production_authorized = False
            component_test_only = True
        else:
            raise ValueError("threshold registry scope is unsupported")
        status, reason = _derive_qualification_status(
            self.observation, self.threshold_receipt
        )
        object.__setattr__(self, "event_aggregation_status", status)
        object.__setattr__(self, "disposition_reason", reason)
        object.__setattr__(
            self,
            "production_threshold_trusted",
            bool(production_authorized and self.threshold_receipt is not None),
        )
        object.__setattr__(self, "component_test_only", component_test_only)
        object.__setattr__(
            self,
            "evidence_grade_qualification_authorized",
            production_authorized,
        )
        object.__setattr__(
            self, "receipt_sha256", _content_receipt_sha256(self.to_dict(False))
        )

    def to_dict(self, include_receipt: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "provider_id": self.provider_id,
            "observation": self.observation.to_dict(),
            "occurrence_equivalence_id": self.occurrence_equivalence_id,
            "threshold_receipt": (
                None
                if self.threshold_receipt is None
                else self.threshold_receipt.to_dict()
            ),
            "threshold_registry_scope": self.threshold_registry_scope,
            "threshold_registry_receipt_sha256": (
                self.threshold_registry_receipt_sha256
            ),
            "event_aggregation_status": self.event_aggregation_status,
            "disposition_reason": self.disposition_reason,
            "production_threshold_trusted": self.production_threshold_trusted,
            "component_test_only": self.component_test_only,
            "evidence_grade_qualification_authorized": (
                self.evidence_grade_qualification_authorized
            ),
            "input_scope": deepcopy(_EEG_ONLY_SCOPE),
        }
        if include_receipt:
            result["receipt_sha256"] = self.receipt_sha256
        return result

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "BAIEGTargetFreeEventQualificationReceiptV1":
        expected = {
            "schema_version",
            "provider_id",
            "observation",
            "occurrence_equivalence_id",
            "threshold_receipt",
            "threshold_registry_scope",
            "threshold_registry_receipt_sha256",
            "event_aggregation_status",
            "disposition_reason",
            "production_threshold_trusted",
            "component_test_only",
            "evidence_grade_qualification_authorized",
            "input_scope",
            "receipt_sha256",
        }
        candidate = _exact_keys(value, expected, "event qualification receipt")
        _validate_scope(candidate["input_scope"], "event qualification receipt")
        observation = BAIEGEventQualificationObservationV1.from_dict(
            candidate["observation"]
        )
        threshold_value = candidate["threshold_receipt"]
        threshold = (
            None
            if threshold_value is None
            else BAIEGSourceDevEventQualificationThresholdReceiptV1.from_dict(
                threshold_value
            )
        )
        result = cls(
            observation=observation,
            occurrence_equivalence_id=candidate["occurrence_equivalence_id"],
            threshold_receipt=threshold,
            threshold_registry_scope=candidate["threshold_registry_scope"],
            threshold_registry_receipt_sha256=candidate[
                "threshold_registry_receipt_sha256"
            ],
            schema_version=candidate["schema_version"],
            provider_id=candidate["provider_id"],
        )
        if candidate["event_aggregation_status"] != result.event_aggregation_status:
            raise ValueError("caller-supplied event qualification status is forbidden")
        if candidate["disposition_reason"] != result.disposition_reason:
            raise ValueError("event qualification disposition reason drifted")
        for name in (
            "production_threshold_trusted",
            "component_test_only",
            "evidence_grade_qualification_authorized",
        ):
            if candidate[name] is not getattr(result, name):
                raise ValueError(f"event qualification {name} drifted")
        if candidate["receipt_sha256"] != result.receipt_sha256:
            raise ValueError("event qualification receipt hash does not bind content")
        return result


class BAIEGTargetFreeEventQualificationProviderV1:
    """Derive event statuses; never accept a caller-provided status vector."""

    provider_id: Final[str] = BA_IEG_TARGET_FREE_EVENT_QUALIFICATION_PROVIDER_ID_V1

    def __init__(
        self,
        threshold_receipt: (
            BAIEGSourceDevEventQualificationThresholdReceiptV1 | None
        ) = None,
    ) -> None:
        if threshold_receipt is not None and not isinstance(
            threshold_receipt,
            BAIEGSourceDevEventQualificationThresholdReceiptV1,
        ):
            raise TypeError("provider threshold must be a typed source-dev receipt")
        registry = load_ba_ieg_production_threshold_registry_v1()
        trusted = frozenset(registry["trusted_threshold_receipt_sha256s"])
        if (
            threshold_receipt is not None
            and threshold_receipt.receipt_sha256 not in trusted
        ):
            raise ValueError(
                "threshold receipt SHA is not preregistered in the fixed "
                "production registry"
            )
        self.threshold_receipt = threshold_receipt
        self.threshold_registry_scope = "production_preregistered"
        self.threshold_registry_receipt_sha256 = registry["receipt_sha256"]

    @classmethod
    def for_component_test_only(
        cls,
        threshold_receipt: BAIEGSourceDevEventQualificationThresholdReceiptV1,
    ) -> "BAIEGTargetFreeEventQualificationProviderV1":
        """Inject one synthetic threshold without granting production trust."""

        if not isinstance(
            threshold_receipt,
            BAIEGSourceDevEventQualificationThresholdReceiptV1,
        ):
            raise TypeError("component-test threshold must be a typed receipt")
        result = object.__new__(cls)
        result.threshold_receipt = threshold_receipt
        result.threshold_registry_scope = (
            "component_test_only_injected_registry"
        )
        result.threshold_registry_receipt_sha256 = (
            _component_test_threshold_registry_receipt_sha256(
                threshold_receipt.receipt_sha256
            )
        )
        return result

    def qualify(
        self,
        observation: BAIEGEventQualificationObservationV1,
        *,
        occurrence_equivalence_id: str,
    ) -> BAIEGTargetFreeEventQualificationReceiptV1:
        if not isinstance(observation, BAIEGEventQualificationObservationV1):
            raise TypeError("provider accepts only a typed EEG-only observation")
        return BAIEGTargetFreeEventQualificationReceiptV1(
            observation=observation,
            occurrence_equivalence_id=occurrence_equivalence_id,
            threshold_receipt=self.threshold_receipt,
            threshold_registry_scope=self.threshold_registry_scope,
            threshold_registry_receipt_sha256=(
                self.threshold_registry_receipt_sha256
            ),
        )

    def qualify_batch(
        self,
        observations: Sequence[BAIEGEventQualificationObservationV1],
        *,
        occurrence_equivalence_ids: Sequence[str],
    ) -> tuple[BAIEGTargetFreeEventQualificationReceiptV1, ...]:
        if len(observations) != len(occurrence_equivalence_ids):
            raise ValueError("observation and occurrence rosters must align")
        return tuple(
            self.qualify(
                observation,
                occurrence_equivalence_id=occurrence_equivalence_id,
            )
            for observation, occurrence_equivalence_id in zip(
                observations, occurrence_equivalence_ids
            )
        )


def _tensor_values(value: torch.Tensor) -> object:
    return value.detach().cpu().tolist()


def build_ba_ieg_event_qualification_observations_from_typed_head_v1(
    event_output: BAIEGShallowCausalTypedUnitHeadOutput,
    *,
    qc_states: Sequence[str],
    technical_states: Sequence[str],
    technical_state_receipt_sha256s: Sequence[str],
) -> tuple[BAIEGEventQualificationObservationV1, ...]:
    """Project a typed-head output onto the provider's strict input surface."""

    if not isinstance(event_output, BAIEGShallowCausalTypedUnitHeadOutput):
        raise TypeError("observation projection requires typed-head output")
    event_count = len(event_output.event_ids)
    if not (
        len(qc_states)
        == len(technical_states)
        == len(technical_state_receipt_sha256s)
        == event_count
    ):
        raise ValueError("QC/technical rosters must align with typed-head events")
    observations: list[BAIEGEventQualificationObservationV1] = []
    for event_index in range(event_count):
        if qc_states[event_index] not in _QC_STATES:
            raise ValueError("QC state is unsupported")
        if technical_states[event_index] not in _TECHNICAL_STATES:
            raise ValueError("technical state is unsupported")
        group_mask = event_output.causal_group_mask[event_index]
        analysis_support_count = int(group_mask.sum().item())
        observed = float(
            event_output.causal_global_onset_boundary_mass[event_index]
            .masked_fill(~group_mask, 0.0)
            .sum()
            .item()
        )
        left_censored = float(
            event_output.causal_global_left_censor_state_mass[event_index]
            .sum()
            .item()
        )
        no_onset = float(
            event_output.causal_global_no_onset_within_support_mass[event_index]
            .item()
        )
        if analysis_support_count == 0:
            causal_state = "unavailable"
            observed = left_censored = no_onset = 0.0
        elif bool(event_output.causal_global_onset_resolved_mask[event_index]):
            causal_state = "resolved"
        else:
            causal_state = "unresolved"
        inventory = event_output.typed_unit_inventory_mask[event_index]
        time_opportunity = event_output.typed_unit_time_mask[event_index]
        if time_opportunity.ndim != 2:
            raise ValueError("typed-unit time opportunity must be group by unit")
        typed_available = bool(
            (inventory & time_opportunity.any(dim=0)).any().item()
        )
        state_payload = {
            "source_input_batch_sha256": event_output.source_input_batch_sha256,
            "event_id": event_output.event_ids[event_index],
            "recording_id": event_output.recording_ids[event_index],
            "source_event_receipt_sha256": (
                event_output.source_event_receipt_sha256s[event_index]
            ),
            "causal_typed_unit_axis_receipt_sha256": (
                event_output.causal_typed_unit_axis_receipt_sha256
            ),
            "causal_group_mask": _tensor_values(group_mask),
            "causal_global_onset_boundary_mass": _tensor_values(
                event_output.causal_global_onset_boundary_mass[event_index]
            ),
            "causal_global_left_censor_state_mass": _tensor_values(
                event_output.causal_global_left_censor_state_mass[event_index]
            ),
            "causal_global_no_onset_within_support_mass": no_onset,
            "causal_global_onset_support_mask": _tensor_values(
                event_output.causal_global_onset_support_mask[event_index]
            ),
            "causal_global_onset_resolved": causal_state == "resolved",
            "typed_unit_inventory_mask": _tensor_values(inventory),
            "typed_unit_time_mask": _tensor_values(time_opportunity),
        }
        observations.append(
            BAIEGEventQualificationObservationV1(
                source_input_batch_sha256=event_output.source_input_batch_sha256,
                event_id=event_output.event_ids[event_index],
                recording_id=event_output.recording_ids[event_index],
                source_event_receipt_sha256=(
                    event_output.source_event_receipt_sha256s[event_index]
                ),
                source_model_state_sha256=_canonical_sha256(state_payload),
                technical_state_receipt_sha256=technical_state_receipt_sha256s[
                    event_index
                ],
                causal_global_onset_state=causal_state,
                causal_global_observed_onset_mass=observed,
                causal_global_left_censored_mass=left_censored,
                causal_global_no_onset_mass=no_onset,
                causal_analysis_support_count=analysis_support_count,
                causal_onset_support_count=int(
                    event_output.causal_global_onset_support_mask[event_index]
                    .sum()
                    .item()
                ),
                typed_opportunity_state=(
                    "available" if typed_available else "no_support"
                ),
                qc_state=qc_states[event_index],
                technical_state=technical_states[event_index],
            )
        )
    return tuple(observations)


__all__ = [
    "BA_IEG_TARGET_FREE_EVENT_QUALIFICATION_PROVIDER_ID_V1",
    "BA_IEG_SOURCE_DEV_EVENT_QUALIFICATION_THRESHOLD_SCHEMA_V1",
    "BA_IEG_EVENT_QUALIFICATION_OBSERVATION_SCHEMA_V1",
    "BA_IEG_TARGET_FREE_EVENT_QUALIFICATION_RECEIPT_SCHEMA_V1",
    "BA_IEG_SOURCE_DEV_THRESHOLD_REGISTRY_SCHEMA_V1",
    "TRUSTED_BA_IEG_PRODUCTION_THRESHOLD_REGISTRY_RECEIPT_SHA256",
    "BA_IEG_LEGACY_CALLER_STATUS_AUTHORITY",
    "BA_IEG_EVENT_QUALIFICATION_STATUSES",
    "BAIEGSourceDevEventQualificationThresholdReceiptV1",
    "BAIEGEventQualificationObservationV1",
    "BAIEGTargetFreeEventQualificationReceiptV1",
    "BAIEGTargetFreeEventQualificationProviderV1",
    "load_ba_ieg_production_threshold_registry_v1",
    "build_ba_ieg_event_qualification_observations_from_typed_head_v1",
]
