"""Closed receipts and an opaque capability for preprocessing-arm selection.

The preprocessing comparison is a *source-train-only* gate.  This module does
not run the comparison and it never consumes EEG or labels.  It defines the
integrity boundary between a completed five-arm formal comparison and a
downstream token producer:

* all five frozen arms must be present and bind the same nested split, raw-QC
  intersection, foundation checkpoint, source roster, and decision policy;
* source-dev, source-eval, private data, DeepSOZ/SOZ targets, and smoke results
  are explicitly forbidden;
* ``O-REF`` is retained as a different-geometry official sanity arm and can
  never authorize a deployment producer; and
* a producer receives an authorization only from a strictly reloaded,
  externally hash-pinned selection bundle.

Numerical noninferiority and arm-selection logic belongs in
``preprocessing_parity_evaluation``.  That evaluator is the only intended
caller of the private decision issuer below.  Keeping the decision object
opaque prevents a mere arm-result claim from becoming a production
capability.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from types import MappingProxyType
from typing import Mapping, Sequence

from .geometry import STANDARD_19


PREPROCESSING_PARITY_PROTOCOL_SCHEMA = "soz_preprocessing_parity_protocol_v1"
PREPROCESSING_ARM_RESULT_SCHEMA = "soz_preprocessing_arm_formal_result_v1"
PREPROCESSING_SELECTION_DECISION_SCHEMA = (
    "soz_preprocessing_selection_decision_v1"
)
PREPROCESSING_SELECTION_ARTIFACT_SCHEMA = (
    "soz_preprocessing_selection_artifact_v1"
)
PREPROCESSING_SELECTION_BUNDLE_RECEIPT_SCHEMA = (
    "soz_preprocessing_selection_bundle_receipt_v1"
)
PREPROCESSING_PRODUCER_AUTHORIZATION_SCHEMA = (
    "soz_preprocessing_producer_authorization_v1"
)
PREPROCESSING_SELECTION_POLICY_SCHEMA = (
    "soz_preprocessing_selection_policy_v1"
)
PREPROCESSING_ARM_SELECTION_METRICS_SCHEMA = (
    "soz_preprocessing_arm_selection_metrics_v2"
)
PREPROCESSING_SELECTION_TRACE_SCHEMA = (
    "soz_preprocessing_selection_trace_v1"
)
PREPROCESSING_SELECTION_NO_GO_SCHEMA = (
    "soz_preprocessing_selection_no_go_v1"
)
PREPROCESSING_NESTED_DEV_RECORD_SCHEMA = (
    "soz_preprocessing_nested_dev_source_record_v1"
)
PREPROCESSING_NESTED_DEV_MANIFEST_SCHEMA = (
    "soz_preprocessing_nested_dev_manifest_v1"
)

PREPROCESSING_SELECTION_FILENAME = "selection.json"
PREPROCESSING_PROTOCOL_FILENAME = "protocol.json"
PREPROCESSING_SELECTION_RECEIPT_FILENAME = "receipt.json"
PREPROCESSING_SELECTION_POLICY_FILENAME = "selection-policy.json"
PREPROCESSING_SELECTION_TRACE_FILENAME = "decision-trace.json"
PREPROCESSING_NESTED_DEV_MANIFEST_FILENAME = "nested-dev-manifest.json"

PREPROCESSING_ARM_IDS = (
    "O-REF",
    "O-CAR19",
    "Z-REF19",
    "Z-CAR19",
    "C-CAR19",
)
DEPLOYABLE_PREPROCESSING_ARM_IDS = PREPROCESSING_ARM_IDS[1:]
PREPROCESSING_PRODUCER_KINDS = ("tuev_morphology", "tusz_ictal")
PREPROCESSING_FORMAL_RUN_TIER = "formal_source_train_nested_dev"
PREPROCESSING_SOURCE_SCOPE = "public_source_train_only"
PREPROCESSING_FORBIDDEN_PARTITIONS = (
    "private",
    "source_dev",
    "source_eval",
)
PREPROCESSING_LABEL_POLICY = (
    "concept_native_labels_only_soz_and_deepsoz_labels_forbidden"
)
PREPROCESSING_RAW_QC_POLICY = (
    "common_pre_filter_raw_qc_intersection_before_any_arm_v1"
)
PREPROCESSING_SPLIT_POLICY = (
    "patient_and_exact_content_component_disjoint_nested_dev_v1"
)
PREPROCESSING_OREF_ROLE = (
    "different_geometry_official_exact_sanity_only_not_deployable"
)
PREPROCESSING_TUEV_PRIMARY_ENDPOINT = (
    "held_patient_content_component_macro_ce6_f1"
)
PREPROCESSING_TUSZ_PRIMARY_ENDPOINT = (
    "held_patient_macro_bce_explicit_native_cells_only_unknown_masked"
)
PREPROCESSING_DEPLOYABLE_TIE_BREAK = (
    "O-CAR19",
    "Z-CAR19",
    "C-CAR19",
    "Z-REF19",
)
PREPROCESSING_TIE_BREAK_RATIONALE = (
    "retrospective_known_window_prefers_official_like_labram_distribution_then_"
    "car19_zero_phase_then_finite_segment_causal_then_reference_dependent_ref19"
)

OFFICIAL_REF23_CHANNELS = (
    "FP1",
    "FP2",
    "F3",
    "F4",
    "C3",
    "C4",
    "P3",
    "P4",
    "O1",
    "O2",
    "F7",
    "F8",
    "T3",
    "T4",
    "T5",
    "T6",
    "A1",
    "A2",
    "FZ",
    "CZ",
    "PZ",
    "T1",
    "T2",
)

LEGACY_FORMAL_V3_TOKEN_SCHEMAS = frozenset(
    {
        "soz_tusz_ictal_token_corpus_index_v3",
    }
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_ARM_ID_RE = re.compile(r"[A-Z0-9-]+")
_MAX_JSON_BYTES = 8 * 1024 * 1024
_DECISION_ISSUER = object()
_CAPABILITY_ISSUER = object()
_PRODUCER_AUTHORIZATION_ISSUER = object()

_ARM_RESULT_FILENAME_BY_ID = MappingProxyType(
    {arm_id: f"arm-{arm_id}.json" for arm_id in PREPROCESSING_ARM_IDS}
)


def _canonical_json_bytes(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Preprocessing-parity payload is not canonical JSON data") from exc
    return (encoded + "\n").encode("utf-8")


def _typed_receipt_sha256(value: object) -> str:
    """Hash a typed payload without the artifact newline."""

    try:
        raw = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("Preprocessing-parity receipt is not JSON data") from exc
    return hashlib.sha256(raw).hexdigest()


def preprocessing_foundation_policy_receipt_sha256(
    *,
    checkpoint_sha256: str,
    modeling_sha256: str,
    position_binding_policy: str,
    record_specific_position_ids: bool,
    token_dim: int,
    input_scale_from_volts: float,
) -> str:
    """Hash the foundation *compatibility policy* used by arm selection.

    Preprocessing parity spans records whose raw headers may use different
    legacy/modern electrode aliases, so it cannot bind one record-specific
    :class:`LaBraMFeatureReceipt`.  The v1 protocol field historically named
    ``foundation_feature_receipt_sha256`` therefore contains this deliberately
    record-invariant policy projection.  Downstream token producers must first
    validate their full typed feature receipt, then compare this projection to
    the selection authorization; the full receipt remains bound separately in
    every token artifact.
    """

    checkpoint = _require_sha256(
        checkpoint_sha256, field="foundation_policy.checkpoint_sha256"
    )
    modeling = _require_sha256(
        modeling_sha256, field="foundation_policy.modeling_sha256"
    )
    if (
        not isinstance(position_binding_policy, str)
        or not position_binding_policy
        or position_binding_policy != position_binding_policy.strip()
    ):
        raise ValueError(
            "foundation_policy.position_binding_policy must be non-empty trimmed text"
        )
    if record_specific_position_ids is not True:
        raise ValueError(
            "Preprocessing parity requires record-specific LaBraM position IDs"
        )
    if isinstance(token_dim, bool) or not isinstance(token_dim, int):
        raise TypeError("foundation_policy.token_dim must be an integer")
    if token_dim < 1:
        raise ValueError("foundation_policy.token_dim must be positive")
    if isinstance(input_scale_from_volts, bool) or not isinstance(
        input_scale_from_volts, (int, float)
    ):
        raise TypeError("foundation_policy.input_scale_from_volts must be numeric")
    scale = float(input_scale_from_volts)
    if not scale > 0 or scale == float("inf"):
        raise ValueError(
            "foundation_policy.input_scale_from_volts must be finite and positive"
        )
    return _typed_receipt_sha256(
        {
            "checkpoint_sha256": checkpoint,
            "modeling_sha256": modeling,
            "position_binding_policy": position_binding_policy,
            "record_specific_position_ids": True,
            "token_dim": token_dim,
            "input_scale_from_volts": scale,
        }
    )


class PreprocessingArmSelectionNoGoError(ValueError):
    """Expected formal gate outcome that must not issue a capability."""

    def __init__(self, message: str, *, receipt: Mapping[str, object]) -> None:
        if not isinstance(message, str) or not message:
            raise ValueError("NO-GO message must be non-empty")
        if not isinstance(receipt, Mapping):
            raise TypeError("NO-GO receipt must be a mapping")
        normalized = json.loads(_canonical_json_bytes(dict(receipt)).decode("utf-8"))
        if normalized.get("schema_version") != PREPROCESSING_SELECTION_NO_GO_SCHEMA:
            raise ValueError("NO-GO receipt has the wrong schema")
        if normalized.get("selection_status") != "NO_GO":
            raise ValueError("NO-GO receipt has the wrong status")
        if normalized.get("downstream_capability_issued") is not False:
            raise ValueError("NO-GO cannot issue a downstream capability")
        self._receipt = normalized
        self.receipt_sha256 = _typed_receipt_sha256(normalized)
        super().__init__(message)

    def to_payload(self) -> dict[str, object]:
        return json.loads(_canonical_json_bytes(self._receipt).decode("utf-8"))


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field is forbidden: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"Non-finite JSON constant is forbidden: {value}")


def _parse_canonical_json(raw: bytes, *, label: str) -> dict[str, object]:
    if not 1 <= len(raw) <= _MAX_JSON_BYTES:
        raise ValueError(f"{label} has an invalid size")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object")
    if _canonical_json_bytes(value) != raw:
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _require_bool(value: object, *, field: str, expected: bool) -> bool:
    if value is not expected:
        raise ValueError(f"{field} must be {expected}")
    return expected


def _normalize_unique_strings(
    values: Sequence[object],
    *,
    field: str,
    expected: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field} must be a sequence of strings")
    normalized = tuple(str(value).strip() for value in values)
    if any(not value for value in normalized):
        raise ValueError(f"{field} cannot contain empty values")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field} cannot contain duplicates")
    if expected is not None and normalized != expected:
        raise ValueError(f"{field} differs from the frozen value")
    return normalized


def _require_arm_id(value: object, *, field: str = "arm_id") -> str:
    if not isinstance(value, str) or not _ARM_ID_RE.fullmatch(value):
        raise ValueError(f"{field} is invalid")
    if value not in PREPROCESSING_ARM_IDS:
        raise ValueError(f"Unknown preprocessing arm: {value!r}")
    return value


@dataclass(frozen=True)
class PreprocessingArmSpec:
    """Frozen scientific identity of one comparison arm."""

    arm_id: str
    channels: tuple[str, ...]
    channel_geometry: str
    event_geometry: str
    filter_family: str
    highpass_hz: float
    lowpass_hz: float
    notch_hz: float | None
    resample_family: str
    output_sfreq_hz: float
    reference: str
    phase: str
    state_scope: str
    deployment_eligible: bool
    role: str
    schema_version: str = "soz_preprocessing_arm_spec_v1"

    def __post_init__(self) -> None:
        _require_arm_id(self.arm_id)
        channels = _normalize_unique_strings(self.channels, field="channels")
        object.__setattr__(self, "channels", channels)
        expected_channels = (
            OFFICIAL_REF23_CHANNELS if self.arm_id == "O-REF" else STANDARD_19
        )
        if channels != expected_channels:
            raise ValueError(f"{self.arm_id} channels differ from the frozen geometry")
        for name in (
            "channel_geometry",
            "event_geometry",
            "filter_family",
            "resample_family",
            "reference",
            "phase",
            "state_scope",
            "role",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        for name in ("highpass_hz", "lowpass_hz", "output_sfreq_hz"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not 0.0 < float(value) < 100_000.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, float(value))
        if self.highpass_hz >= self.lowpass_hz:
            raise ValueError("Arm highpass_hz must be below lowpass_hz")
        if self.notch_hz is not None:
            if isinstance(self.notch_hz, bool) or not isinstance(
                self.notch_hz, (int, float)
            ):
                raise TypeError("notch_hz must be numeric or None")
            if not 0.0 < float(self.notch_hz) < 100_000.0:
                raise ValueError("notch_hz must be finite and positive")
            object.__setattr__(self, "notch_hz", float(self.notch_hz))
        if not isinstance(self.deployment_eligible, bool):
            raise TypeError("deployment_eligible must be bool")
        if self.arm_id == "O-REF":
            if self.deployment_eligible or self.role != PREPROCESSING_OREF_ROLE:
                raise ValueError("O-REF must remain a non-deployable exact sanity arm")
        elif not self.deployment_eligible:
            raise ValueError("Every complete-19 arm must remain deployment-eligible")
        if self.schema_version != "soz_preprocessing_arm_spec_v1":
            raise ValueError("Unsupported preprocessing arm-spec schema")

    @property
    def receipt_sha256(self) -> str:
        return _typed_receipt_sha256(asdict(self))


def _frozen_arm_specs() -> tuple[PreprocessingArmSpec, ...]:
    return (
        PreprocessingArmSpec(
            arm_id="O-REF",
            channels=OFFICIAL_REF23_CHANNELS,
            channel_geometry="official_23_physical_ref_full_record",
            event_geometry="official_centered_5_second_event",
            filter_family="mne_official_zero_phase_fir_0p1_75_then_notch50",
            highpass_hz=0.1,
            lowpass_hz=75.0,
            notch_hz=50.0,
            resample_family="mne_fft_resample_full_record",
            output_sfreq_hz=200.0,
            reference="physical_ref_no_rereference",
            phase="zero_phase",
            state_scope="full_record_before_event_crop",
            deployment_eligible=False,
            role=PREPROCESSING_OREF_ROLE,
        ),
        PreprocessingArmSpec(
            arm_id="O-CAR19",
            channels=STANDARD_19,
            channel_geometry="complete_standard19_deployment_schedule",
            event_geometry="task_specific_frozen_deployment_calls",
            filter_family="mne_official_spectral_family_0p1_75_then_notch50",
            highpass_hz=0.1,
            lowpass_hz=75.0,
            notch_hz=50.0,
            resample_family="mne_fft_resample_then_deployment_calls",
            output_sfreq_hz=200.0,
            reference="car19",
            phase="zero_phase",
            state_scope="full_record_before_deployment_calls",
            deployment_eligible=True,
            role="official_like_deployment_parity_not_official_exact",
        ),
        PreprocessingArmSpec(
            arm_id="Z-REF19",
            channels=STANDARD_19,
            channel_geometry="complete_standard19_deployment_schedule",
            event_geometry="task_specific_frozen_deployment_calls",
            filter_family="zero_phase_bandpass_0p5_45",
            highpass_hz=0.5,
            lowpass_hz=45.0,
            notch_hz=None,
            resample_family="zero_phase_resample_200",
            output_sfreq_hz=200.0,
            reference="physical_ref_no_rereference",
            phase="zero_phase",
            state_scope="full_record_before_deployment_calls",
            deployment_eligible=True,
            role="bandwidth_and_reference_control",
        ),
        PreprocessingArmSpec(
            arm_id="Z-CAR19",
            channels=STANDARD_19,
            channel_geometry="complete_standard19_deployment_schedule",
            event_geometry="task_specific_frozen_deployment_calls",
            filter_family="zero_phase_bandpass_0p5_45",
            highpass_hz=0.5,
            lowpass_hz=45.0,
            notch_hz=None,
            resample_family="zero_phase_resample_200",
            output_sfreq_hz=200.0,
            reference="car19",
            phase="zero_phase",
            state_scope="full_record_before_deployment_calls",
            deployment_eligible=True,
            role="phase_control",
        ),
        PreprocessingArmSpec(
            arm_id="C-CAR19",
            channels=STANDARD_19,
            channel_geometry="complete_standard19_deployment_schedule",
            event_geometry="task_specific_frozen_deployment_calls",
            filter_family="butterworth_order4_forward_sos_0p5_45",
            highpass_hz=0.5,
            lowpass_hz=45.0,
            notch_hz=None,
            resample_family="causal_upfirdn_kaiser5_delay_compensated",
            output_sfreq_hz=200.0,
            reference="car19",
            phase="causal_frequency_dependent_iir_delay_receipted",
            state_scope="finite_segment_zero_state_with_30_second_real_warmup",
            deployment_eligible=True,
            role="current_causal_candidate",
        ),
    )


FROZEN_PREPROCESSING_ARM_SPECS = _frozen_arm_specs()
FROZEN_PREPROCESSING_ARM_SPEC_BY_ID = MappingProxyType(
    {spec.arm_id: spec for spec in FROZEN_PREPROCESSING_ARM_SPECS}
)
FROZEN_PREPROCESSING_ARM_SPECS_SHA256 = _typed_receipt_sha256(
    [asdict(spec) for spec in FROZEN_PREPROCESSING_ARM_SPECS]
)


@dataclass(frozen=True)
class PreprocessingSelectionPolicy:
    """Unique, pre-result rule shared by morphology and ictal parity.

    A single common arm is deliberately required.  Family-specific arm
    selection would make the evidence bottleneck depend on two incompatible
    signal domains and would require a separate cross-family alignment study.
    Under this policy, disagreement that leaves no jointly noninferior arm is
    a formal ``NO-GO`` rather than permission to choose two arms post hoc.
    """

    tuev_primary_endpoint: str = PREPROCESSING_TUEV_PRIMARY_ENDPOINT
    tusz_primary_endpoint: str = PREPROCESSING_TUSZ_PRIMARY_ENDPOINT
    tuev_macro_f1_noninferiority_margin: float = 0.02
    tusz_macro_bce_noninferiority_margin: float = 0.02
    paired_confidence_level: float = 0.95
    maximum_tuev_jitter_macro_f1_drop: float = 0.03
    maximum_tusz_jitter_macro_bce_increase: float = 0.03
    maximum_official_signal_relative_l2_error: float = 1e-6
    maximum_official_token_cosine_distance: float = 1e-5
    maximum_paired_attrition_count: int = 0
    deployable_tie_break: tuple[str, ...] = PREPROCESSING_DEPLOYABLE_TIE_BREAK
    tie_break_rationale: str = PREPROCESSING_TIE_BREAK_RATIONALE
    o_ref_role: str = PREPROCESSING_OREF_ROLE
    common_arm_across_families_required: bool = True
    family_specific_arm_selection_forbidden: bool = True
    source_train_nested_dev_only: bool = True
    source_dev_eval_private_forbidden: bool = True
    soz_labels_forbidden: bool = True
    arm_id_probe_required_patient_disjoint: bool = True
    input_and_token_distribution_analysis_required: bool = True
    schema_version: str = PREPROCESSING_SELECTION_POLICY_SCHEMA

    def __post_init__(self) -> None:
        if self.tuev_primary_endpoint != PREPROCESSING_TUEV_PRIMARY_ENDPOINT:
            raise ValueError("TUEV preprocessing endpoint cannot change")
        if self.tusz_primary_endpoint != PREPROCESSING_TUSZ_PRIMARY_ENDPOINT:
            raise ValueError("TUSZ preprocessing endpoint cannot change")
        fixed_floats = {
            "tuev_macro_f1_noninferiority_margin": 0.02,
            "tusz_macro_bce_noninferiority_margin": 0.02,
            "paired_confidence_level": 0.95,
            "maximum_tuev_jitter_macro_f1_drop": 0.03,
            "maximum_tusz_jitter_macro_bce_increase": 0.03,
            "maximum_official_signal_relative_l2_error": 1e-6,
            "maximum_official_token_cosine_distance": 1e-5,
        }
        for field, expected in fixed_floats.items():
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field} must be numeric")
            if float(value) != expected:
                raise ValueError(f"{field} differs from the preregistered value")
            object.__setattr__(self, field, float(value))
        if (
            isinstance(self.maximum_paired_attrition_count, bool)
            or self.maximum_paired_attrition_count != 0
        ):
            raise ValueError("Paired formal arms cannot have discordant attrition")
        object.__setattr__(
            self,
            "deployable_tie_break",
            _normalize_unique_strings(
                self.deployable_tie_break,
                field="deployable_tie_break",
                expected=PREPROCESSING_DEPLOYABLE_TIE_BREAK,
            ),
        )
        if self.o_ref_role != PREPROCESSING_OREF_ROLE:
            raise ValueError("O-REF role cannot change")
        if self.tie_break_rationale != PREPROCESSING_TIE_BREAK_RATIONALE:
            raise ValueError("Preprocessing tie-break rationale cannot change")
        for field in (
            "common_arm_across_families_required",
            "family_specific_arm_selection_forbidden",
            "source_train_nested_dev_only",
            "source_dev_eval_private_forbidden",
            "soz_labels_forbidden",
            "arm_id_probe_required_patient_disjoint",
            "input_and_token_distribution_analysis_required",
        ):
            _require_bool(getattr(self, field), field=field, expected=True)
        if self.schema_version != PREPROCESSING_SELECTION_POLICY_SCHEMA:
            raise ValueError("Unsupported preprocessing selection-policy schema")

    @property
    def receipt_sha256(self) -> str:
        return _typed_receipt_sha256(asdict(self))


LOCKED_PREPROCESSING_SELECTION_POLICY = PreprocessingSelectionPolicy()
LOCKED_PREPROCESSING_SELECTION_POLICY_RECEIPT_SHA256 = (
    LOCKED_PREPROCESSING_SELECTION_POLICY.receipt_sha256
)


def _normalize_deployable_float_mapping(
    values: Mapping[str, object], *, field: str
) -> Mapping[str, float]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{field} must be a mapping")
    if tuple(sorted(values)) != tuple(sorted(DEPLOYABLE_PREPROCESSING_ARM_IDS)):
        raise ValueError(f"{field} must contain exactly all four deployable arms")
    normalized: dict[str, float] = {}
    for arm_id in DEPLOYABLE_PREPROCESSING_ARM_IDS:
        value = values[arm_id]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{field}[{arm_id}] must be numeric")
        number = float(value)
        if not (-1e9 < number < 1e9):
            raise ValueError(f"{field}[{arm_id}] must be finite")
        normalized[arm_id] = number
    return MappingProxyType(normalized)


@dataclass(frozen=True)
class PreprocessingArmSelectionMetrics:
    """Sufficient paired statistics for the unique cross-family rule.

    Difference conventions are fixed:

    * TUEV values are ``candidate F1 - reference F1`` and use the paired 95%
      lower confidence bound; and
    * TUSZ values are ``candidate BCE - reference BCE`` and use the paired 95%
      upper confidence bound.

    Unknown TUSZ native cells never enter the BCE denominator.
    """

    arm_id: str
    protocol_receipt_sha256: str
    tuev_macro_ce6_f1: float
    tusz_native_macro_bce: float
    tuev_f1_difference_lower95_by_reference: Mapping[str, float]
    tusz_bce_difference_upper95_by_reference: Mapping[str, float]
    paired_denominator_receipt_sha256: str
    tuev_paired_patient_count: int
    tuev_paired_content_component_count: int
    tusz_paired_patient_count: int
    tusz_paired_explicit_cell_count: int
    paired_attrition_count: int
    tuev_jitter_macro_f1_max_drop: float
    tusz_jitter_macro_bce_max_increase: float
    arm_id_probe_balanced_accuracy: float
    arm_id_probe_patient_disjoint: bool
    input_distribution_analysis_complete: bool
    token_distribution_analysis_complete: bool
    concept_endpoints_applicable: bool = True
    official_signal_relative_l2_error: float | None = None
    official_token_cosine_distance: float | None = None
    source_dev_used: bool = False
    source_eval_used: bool = False
    private_data_used: bool = False
    soz_labels_used: bool = False
    schema_version: str = PREPROCESSING_ARM_SELECTION_METRICS_SCHEMA

    def __post_init__(self) -> None:
        arm_id = _require_arm_id(self.arm_id)
        object.__setattr__(self, "arm_id", arm_id)
        object.__setattr__(
            self,
            "protocol_receipt_sha256",
            _require_sha256(
                self.protocol_receipt_sha256, field="protocol_receipt_sha256"
            ),
        )
        object.__setattr__(
            self,
            "paired_denominator_receipt_sha256",
            _require_sha256(
                self.paired_denominator_receipt_sha256,
                field="paired_denominator_receipt_sha256",
            ),
        )
        for field in (
            "tuev_macro_ce6_f1",
            "tusz_native_macro_bce",
            "tuev_jitter_macro_f1_max_drop",
            "tusz_jitter_macro_bce_max_increase",
            "arm_id_probe_balanced_accuracy",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field} must be numeric")
            normalized = float(value)
            if not 0.0 <= normalized <= 1.0:
                raise ValueError(f"{field} must lie in [0,1]")
            object.__setattr__(self, field, normalized)
        object.__setattr__(
            self,
            "tuev_f1_difference_lower95_by_reference",
            _normalize_deployable_float_mapping(
                self.tuev_f1_difference_lower95_by_reference,
                field="tuev_f1_difference_lower95_by_reference",
            ),
        )
        object.__setattr__(
            self,
            "tusz_bce_difference_upper95_by_reference",
            _normalize_deployable_float_mapping(
                self.tusz_bce_difference_upper95_by_reference,
                field="tusz_bce_difference_upper95_by_reference",
            ),
        )
        if not isinstance(self.concept_endpoints_applicable, bool):
            raise TypeError("concept_endpoints_applicable must be bool")
        if arm_id in DEPLOYABLE_PREPROCESSING_ARM_IDS:
            if self.concept_endpoints_applicable is not True:
                raise ValueError("Deployable arms require native concept endpoints")
            if abs(self.tuev_f1_difference_lower95_by_reference[arm_id]) > 1e-12:
                raise ValueError("Self TUEV paired difference must be zero")
            if abs(self.tusz_bce_difference_upper95_by_reference[arm_id]) > 1e-12:
                raise ValueError("Self TUSZ paired difference must be zero")
        elif self.concept_endpoints_applicable is not False:
            raise ValueError("O-REF is a sanity-only geometry, not a concept endpoint arm")
        for field in (
            "tuev_paired_patient_count",
            "tuev_paired_content_component_count",
            "tusz_paired_patient_count",
            "tusz_paired_explicit_cell_count",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        if (
            isinstance(self.paired_attrition_count, bool)
            or not isinstance(self.paired_attrition_count, int)
            or self.paired_attrition_count < 0
        ):
            raise ValueError("paired_attrition_count must be a non-negative integer")
        for field in (
            "arm_id_probe_patient_disjoint",
            "input_distribution_analysis_complete",
            "token_distribution_analysis_complete",
        ):
            _require_bool(getattr(self, field), field=field, expected=True)
        for field in (
            "source_dev_used",
            "source_eval_used",
            "private_data_used",
            "soz_labels_used",
        ):
            _require_bool(getattr(self, field), field=field, expected=False)
        official_values = (
            self.official_signal_relative_l2_error,
            self.official_token_cosine_distance,
        )
        if arm_id == "O-REF":
            if any(
                abs(value) > 1e-12
                for value in (
                    self.tuev_macro_ce6_f1,
                    self.tusz_native_macro_bce,
                    self.tuev_jitter_macro_f1_max_drop,
                    self.tusz_jitter_macro_bce_max_increase,
                    *self.tuev_f1_difference_lower95_by_reference.values(),
                    *self.tusz_bce_difference_upper95_by_reference.values(),
                )
            ):
                raise ValueError(
                    "O-REF concept fields must use the frozen zero not-applicable sentinel"
                )
            if any(value is None for value in official_values):
                raise ValueError("O-REF requires official signal and token sanity errors")
            for field in (
                "official_signal_relative_l2_error",
                "official_token_cosine_distance",
            ):
                value = getattr(self, field)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise TypeError(f"{field} must be numeric for O-REF")
                normalized = float(value)
                if not 0.0 <= normalized < 1e9:
                    raise ValueError(f"{field} must be finite and non-negative")
                object.__setattr__(self, field, normalized)
        elif any(value is not None for value in official_values):
            raise ValueError("Official-exact errors belong only to O-REF")
        if self.schema_version != PREPROCESSING_ARM_SELECTION_METRICS_SCHEMA:
            raise ValueError("Unsupported arm-selection metrics schema")

    def to_payload(self) -> dict[str, object]:
        payload = {
            field_name: getattr(self, field_name)
            for field_name in self.__dataclass_fields__
        }
        payload["tuev_f1_difference_lower95_by_reference"] = dict(
            self.tuev_f1_difference_lower95_by_reference
        )
        payload["tusz_bce_difference_upper95_by_reference"] = dict(
            self.tusz_bce_difference_upper95_by_reference
        )
        return payload

    @property
    def receipt_sha256(self) -> str:
        return _typed_receipt_sha256(self.to_payload())


@dataclass(frozen=True)
class PreprocessingSelectionDecisionTrace:
    protocol_receipt_sha256: str
    selection_policy_receipt_sha256: str
    arm_metrics_by_id: Mapping[str, PreprocessingArmSelectionMetrics]
    arm_metric_receipt_sha256_by_id: Mapping[str, str]
    tuev_reference_arm_id: str
    tusz_reference_arm_id: str
    jointly_noninferior_arm_ids: tuple[str, ...]
    selected_arm_id: str
    tie_break: tuple[str, ...] = PREPROCESSING_DEPLOYABLE_TIE_BREAK
    common_arm_across_families: bool = True
    o_ref_sanity_passed: bool = True
    schema_version: str = PREPROCESSING_SELECTION_TRACE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "protocol_receipt_sha256",
            _require_sha256(
                self.protocol_receipt_sha256, field="protocol_receipt_sha256"
            ),
        )
        if (
            self.selection_policy_receipt_sha256
            != LOCKED_PREPROCESSING_SELECTION_POLICY_RECEIPT_SHA256
        ):
            raise ValueError("Decision trace does not use the locked selection policy")
        metrics = self.arm_metrics_by_id
        if not isinstance(metrics, Mapping) or tuple(sorted(metrics)) != tuple(
            sorted(PREPROCESSING_ARM_IDS)
        ):
            raise ValueError("Decision trace must contain all five arm metrics")
        normalized_metrics: dict[str, PreprocessingArmSelectionMetrics] = {}
        for arm_id in PREPROCESSING_ARM_IDS:
            metric = metrics[arm_id]
            if not isinstance(metric, PreprocessingArmSelectionMetrics):
                raise TypeError(f"arm_metrics_by_id[{arm_id}] has the wrong type")
            if metric.arm_id != arm_id:
                raise ValueError("Decision trace arm metric key disagrees with payload")
            normalized_metrics[arm_id] = metric
        object.__setattr__(
            self, "arm_metrics_by_id", MappingProxyType(normalized_metrics)
        )
        declared_receipts = _normalize_arm_hash_mapping(
            self.arm_metric_receipt_sha256_by_id,
            field="arm_metric_receipt_sha256_by_id",
        )
        actual_receipts = {
            arm_id: normalized_metrics[arm_id].receipt_sha256
            for arm_id in PREPROCESSING_ARM_IDS
        }
        if declared_receipts != actual_receipts:
            raise ValueError("Decision trace metric receipts disagree with metrics")
        object.__setattr__(
            self,
            "arm_metric_receipt_sha256_by_id",
            MappingProxyType(declared_receipts),
        )
        for field in ("tuev_reference_arm_id", "tusz_reference_arm_id"):
            arm_id = _require_arm_id(getattr(self, field), field=field)
            if arm_id not in DEPLOYABLE_PREPROCESSING_ARM_IDS:
                raise ValueError(f"{field} must be a deployable arm")
        jointly = _normalize_unique_strings(
            self.jointly_noninferior_arm_ids,
            field="jointly_noninferior_arm_ids",
        )
        if not jointly or any(
            arm_id not in DEPLOYABLE_PREPROCESSING_ARM_IDS for arm_id in jointly
        ):
            raise ValueError("Decision requires at least one jointly eligible deployable arm")
        object.__setattr__(self, "jointly_noninferior_arm_ids", jointly)
        selected = _require_arm_id(self.selected_arm_id, field="selected_arm_id")
        if selected not in jointly:
            raise ValueError("Selected arm is not jointly noninferior")
        object.__setattr__(self, "selected_arm_id", selected)
        object.__setattr__(
            self,
            "tie_break",
            _normalize_unique_strings(
                self.tie_break,
                field="tie_break",
                expected=PREPROCESSING_DEPLOYABLE_TIE_BREAK,
            ),
        )
        _require_bool(
            self.common_arm_across_families,
            field="common_arm_across_families",
            expected=True,
        )
        _require_bool(
            self.o_ref_sanity_passed, field="o_ref_sanity_passed", expected=True
        )
        if self.schema_version != PREPROCESSING_SELECTION_TRACE_SCHEMA:
            raise ValueError("Unsupported preprocessing decision-trace schema")

    def to_payload(self) -> dict[str, object]:
        return {
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "selection_policy_receipt_sha256": (
                self.selection_policy_receipt_sha256
            ),
            "arm_metrics_by_id": {
                arm_id: self.arm_metrics_by_id[arm_id].to_payload()
                for arm_id in PREPROCESSING_ARM_IDS
            },
            "arm_metric_receipt_sha256_by_id": dict(
                self.arm_metric_receipt_sha256_by_id
            ),
            "tuev_reference_arm_id": self.tuev_reference_arm_id,
            "tusz_reference_arm_id": self.tusz_reference_arm_id,
            "jointly_noninferior_arm_ids": list(self.jointly_noninferior_arm_ids),
            "selected_arm_id": self.selected_arm_id,
            "tie_break": list(self.tie_break),
            "common_arm_across_families": True,
            "o_ref_sanity_passed": True,
            "schema_version": self.schema_version,
        }

    @property
    def receipt_sha256(self) -> str:
        return _typed_receipt_sha256(self.to_payload())


@dataclass(frozen=True)
class PreprocessingNestedDevSourceRecord:
    """One source-train record before the shared raw-QC intersection."""

    dataset_id: str
    task_family: str
    record_id: str
    patient_identity_key: str
    content_component_id: str
    edf_sha256: str
    source_record_receipt_sha256: str
    raw_qc_receipt_sha256: str
    common_raw_qc_eligible: bool
    raw_qc_exclusion_code: str | None
    nested_dev_fold: int | None
    official_partition: str = "train"
    source_train_only: bool = True
    soz_labels_present: bool = False
    schema_version: str = PREPROCESSING_NESTED_DEV_RECORD_SCHEMA

    def __post_init__(self) -> None:
        dataset_to_task = {
            "TUEV": "morphology_ce6",
            "TUSZ": "ictal_native",
        }
        if self.dataset_id not in dataset_to_task:
            raise ValueError("Nested-dev records must come from TUEV or TUSZ")
        if self.task_family != dataset_to_task[self.dataset_id]:
            raise ValueError("Nested-dev dataset and native task family disagree")
        for field in (
            "record_id",
            "patient_identity_key",
            "content_component_id",
        ):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be a non-empty string")
            if value != value.strip():
                raise ValueError(f"{field} cannot contain outer whitespace")
        for field in (
            "edf_sha256",
            "source_record_receipt_sha256",
            "raw_qc_receipt_sha256",
        ):
            object.__setattr__(
                self, field, _require_sha256(getattr(self, field), field=field)
            )
        if not isinstance(self.common_raw_qc_eligible, bool):
            raise TypeError("common_raw_qc_eligible must be bool")
        if self.common_raw_qc_eligible:
            if self.raw_qc_exclusion_code is not None:
                raise ValueError("Eligible records cannot carry a raw-QC exclusion")
            if (
                isinstance(self.nested_dev_fold, bool)
                or not isinstance(self.nested_dev_fold, int)
                or not 0 <= self.nested_dev_fold < 5
            ):
                raise ValueError("Eligible records require a nested-dev fold in [0,4]")
        else:
            if (
                not isinstance(self.raw_qc_exclusion_code, str)
                or not self.raw_qc_exclusion_code.strip()
            ):
                raise ValueError("Excluded records require an explicit raw-QC code")
            if self.nested_dev_fold is not None:
                raise ValueError("Excluded records cannot enter a nested-dev fold")
        if self.official_partition != "train":
            raise ValueError("Only official source-train records may enter parity")
        _require_bool(
            self.source_train_only, field="source_train_only", expected=True
        )
        _require_bool(
            self.soz_labels_present, field="soz_labels_present", expected=False
        )
        if self.schema_version != PREPROCESSING_NESTED_DEV_RECORD_SCHEMA:
            raise ValueError("Unsupported nested-dev source-record schema")

    @property
    def receipt_sha256(self) -> str:
        return _typed_receipt_sha256(asdict(self))


def _nested_record_sort_key(
    record: PreprocessingNestedDevSourceRecord,
) -> tuple[str, str]:
    return record.dataset_id, record.record_id


def _nested_roster_sha256(values: object) -> str:
    return _typed_receipt_sha256(values)


@dataclass(frozen=True)
class PreprocessingParityNestedDevManifest:
    """Closed patient/content-disjoint split and common raw-QC ledger."""

    records: tuple[PreprocessingNestedDevSourceRecord, ...]
    tuev_source_manifest_receipt_sha256: str
    tusz_source_manifest_receipt_sha256: str
    record_roster_sha256: str
    included_record_roster_sha256: str
    excluded_record_roster_sha256: str
    source_patient_roster_sha256: str
    content_component_split_receipt_sha256: str
    raw_qc_intersection_receipt_sha256: str
    record_count: int
    included_record_count: int
    excluded_record_count: int
    included_patient_count: int
    included_content_component_count: int
    fold_record_counts: tuple[int, ...]
    fold_count: int = 5
    split_policy: str = PREPROCESSING_SPLIT_POLICY
    raw_qc_policy: str = PREPROCESSING_RAW_QC_POLICY
    source_scope: str = PREPROCESSING_SOURCE_SCOPE
    forbidden_partitions: tuple[str, ...] = PREPROCESSING_FORBIDDEN_PARTITIONS
    source_dev_used: bool = False
    source_eval_used: bool = False
    private_data_used: bool = False
    soz_labels_used: bool = False
    schema_version: str = PREPROCESSING_NESTED_DEV_MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        records = tuple(self.records)
        if not records or any(
            not isinstance(record, PreprocessingNestedDevSourceRecord)
            for record in records
        ):
            raise ValueError("Nested-dev manifest requires typed source records")
        if records != tuple(sorted(records, key=_nested_record_sort_key)):
            raise ValueError("Nested-dev records must be in canonical dataset/record order")
        record_keys = tuple(
            f"{record.dataset_id}:{record.record_id}" for record in records
        )
        if len(set(record_keys)) != len(record_keys):
            raise ValueError("Nested-dev manifest has duplicate dataset/record IDs")
        object.__setattr__(self, "records", records)
        for field in (
            "tuev_source_manifest_receipt_sha256",
            "tusz_source_manifest_receipt_sha256",
            "record_roster_sha256",
            "included_record_roster_sha256",
            "excluded_record_roster_sha256",
            "source_patient_roster_sha256",
            "content_component_split_receipt_sha256",
            "raw_qc_intersection_receipt_sha256",
        ):
            object.__setattr__(
                self, field, _require_sha256(getattr(self, field), field=field)
            )
        included = tuple(record for record in records if record.common_raw_qc_eligible)
        excluded = tuple(record for record in records if not record.common_raw_qc_eligible)
        if {record.dataset_id for record in included} != {"TUEV", "TUSZ"}:
            raise ValueError("Common raw-QC intersection needs both TUEV and TUSZ")
        patient_fold: dict[str, int] = {}
        component_fold: dict[str, int] = {}
        for record in included:
            assert record.nested_dev_fold is not None
            previous_patient = patient_fold.setdefault(
                record.patient_identity_key, record.nested_dev_fold
            )
            if previous_patient != record.nested_dev_fold:
                raise ValueError("One patient appears in multiple nested-dev folds")
            previous_component = component_fold.setdefault(
                record.content_component_id, record.nested_dev_fold
            )
            if previous_component != record.nested_dev_fold:
                raise ValueError("One exact-content component appears in multiple folds")
        fold_counts = tuple(
            sum(record.nested_dev_fold == fold for record in included)
            for fold in range(5)
        )
        if any(count < 1 for count in fold_counts):
            raise ValueError("Every nested-dev fold must contain eligible records")
        dataset_fold_support = {
            dataset: {
                record.nested_dev_fold
                for record in included
                if record.dataset_id == dataset
            }
            for dataset in ("TUEV", "TUSZ")
        }
        if any(folds != set(range(5)) for folds in dataset_fold_support.values()):
            raise ValueError("Each dataset must contribute eligible records to every fold")

        actual_record_roster = _nested_roster_sha256(record_keys)
        actual_included_roster = _nested_roster_sha256(
            tuple(
                f"{record.dataset_id}:{record.record_id}"
                for record in included
            )
        )
        actual_excluded_roster = _nested_roster_sha256(
            tuple(
                f"{record.dataset_id}:{record.record_id}"
                for record in excluded
            )
        )
        actual_patient_roster = _nested_roster_sha256(tuple(sorted(patient_fold)))
        actual_component_split = _nested_roster_sha256(
            {
                "patient_fold": sorted(patient_fold.items()),
                "content_component_fold": sorted(component_fold.items()),
                "fold_count": 5,
                "policy": PREPROCESSING_SPLIT_POLICY,
            }
        )
        actual_raw_qc = _nested_roster_sha256(
            {
                "policy": PREPROCESSING_RAW_QC_POLICY,
                "records": [
                    {
                        "record_key": f"{record.dataset_id}:{record.record_id}",
                        "raw_qc_receipt_sha256": record.raw_qc_receipt_sha256,
                        "eligible": record.common_raw_qc_eligible,
                        "exclusion_code": record.raw_qc_exclusion_code,
                    }
                    for record in records
                ],
                "included_record_roster_sha256": actual_included_roster,
            }
        )
        expected_hashes = {
            "record_roster_sha256": actual_record_roster,
            "included_record_roster_sha256": actual_included_roster,
            "excluded_record_roster_sha256": actual_excluded_roster,
            "source_patient_roster_sha256": actual_patient_roster,
            "content_component_split_receipt_sha256": actual_component_split,
            "raw_qc_intersection_receipt_sha256": actual_raw_qc,
        }
        for field, expected in expected_hashes.items():
            if getattr(self, field) != expected:
                raise ValueError(f"Nested-dev manifest {field} is inconsistent")
        expected_counts = {
            "record_count": len(records),
            "included_record_count": len(included),
            "excluded_record_count": len(excluded),
            "included_patient_count": len(patient_fold),
            "included_content_component_count": len(component_fold),
        }
        for field, expected in expected_counts.items():
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value != expected:
                raise ValueError(f"Nested-dev manifest {field} is inconsistent")
        if tuple(self.fold_record_counts) != fold_counts:
            raise ValueError("Nested-dev fold record counts are inconsistent")
        object.__setattr__(self, "fold_record_counts", fold_counts)
        if self.fold_count != 5:
            raise ValueError("Nested-dev parity requires exactly five folds")
        if self.split_policy != PREPROCESSING_SPLIT_POLICY:
            raise ValueError("Nested-dev split policy cannot change")
        if self.raw_qc_policy != PREPROCESSING_RAW_QC_POLICY:
            raise ValueError("Nested-dev raw-QC policy cannot change")
        if self.source_scope != PREPROCESSING_SOURCE_SCOPE:
            raise ValueError("Nested-dev manifest must be source-train-only")
        object.__setattr__(
            self,
            "forbidden_partitions",
            _normalize_unique_strings(
                self.forbidden_partitions,
                field="forbidden_partitions",
                expected=PREPROCESSING_FORBIDDEN_PARTITIONS,
            ),
        )
        for field in (
            "source_dev_used",
            "source_eval_used",
            "private_data_used",
            "soz_labels_used",
        ):
            _require_bool(getattr(self, field), field=field, expected=False)
        if self.schema_version != PREPROCESSING_NESTED_DEV_MANIFEST_SCHEMA:
            raise ValueError("Unsupported preprocessing nested-dev manifest schema")

    def to_payload(self) -> dict[str, object]:
        return {
            "records": [asdict(record) for record in self.records],
            "tuev_source_manifest_receipt_sha256": (
                self.tuev_source_manifest_receipt_sha256
            ),
            "tusz_source_manifest_receipt_sha256": (
                self.tusz_source_manifest_receipt_sha256
            ),
            "record_roster_sha256": self.record_roster_sha256,
            "included_record_roster_sha256": self.included_record_roster_sha256,
            "excluded_record_roster_sha256": self.excluded_record_roster_sha256,
            "source_patient_roster_sha256": self.source_patient_roster_sha256,
            "content_component_split_receipt_sha256": (
                self.content_component_split_receipt_sha256
            ),
            "raw_qc_intersection_receipt_sha256": (
                self.raw_qc_intersection_receipt_sha256
            ),
            "record_count": self.record_count,
            "included_record_count": self.included_record_count,
            "excluded_record_count": self.excluded_record_count,
            "included_patient_count": self.included_patient_count,
            "included_content_component_count": (
                self.included_content_component_count
            ),
            "fold_record_counts": list(self.fold_record_counts),
            "fold_count": self.fold_count,
            "split_policy": self.split_policy,
            "raw_qc_policy": self.raw_qc_policy,
            "source_scope": self.source_scope,
            "forbidden_partitions": list(self.forbidden_partitions),
            "source_dev_used": False,
            "source_eval_used": False,
            "private_data_used": False,
            "soz_labels_used": False,
            "schema_version": self.schema_version,
        }

    @property
    def receipt_sha256(self) -> str:
        return _typed_receipt_sha256(self.to_payload())


def build_preprocessing_parity_nested_dev_manifest(
    *,
    records: Sequence[PreprocessingNestedDevSourceRecord],
    tuev_source_manifest_receipt_sha256: str,
    tusz_source_manifest_receipt_sha256: str,
) -> PreprocessingParityNestedDevManifest:
    """Build and self-check the canonical source-train split/QC manifest."""

    ordered = tuple(sorted(records, key=_nested_record_sort_key))
    if not ordered:
        raise ValueError("Nested-dev manifest cannot be empty")
    record_keys = tuple(
        f"{record.dataset_id}:{record.record_id}" for record in ordered
    )
    included = tuple(record for record in ordered if record.common_raw_qc_eligible)
    excluded = tuple(record for record in ordered if not record.common_raw_qc_eligible)
    patient_fold = {
        record.patient_identity_key: record.nested_dev_fold for record in included
    }
    component_fold = {
        record.content_component_id: record.nested_dev_fold for record in included
    }
    included_roster = _nested_roster_sha256(
        tuple(f"{record.dataset_id}:{record.record_id}" for record in included)
    )
    raw_qc_receipt = _nested_roster_sha256(
        {
            "policy": PREPROCESSING_RAW_QC_POLICY,
            "records": [
                {
                    "record_key": f"{record.dataset_id}:{record.record_id}",
                    "raw_qc_receipt_sha256": record.raw_qc_receipt_sha256,
                    "eligible": record.common_raw_qc_eligible,
                    "exclusion_code": record.raw_qc_exclusion_code,
                }
                for record in ordered
            ],
            "included_record_roster_sha256": included_roster,
        }
    )
    return PreprocessingParityNestedDevManifest(
        records=ordered,
        tuev_source_manifest_receipt_sha256=tuev_source_manifest_receipt_sha256,
        tusz_source_manifest_receipt_sha256=tusz_source_manifest_receipt_sha256,
        record_roster_sha256=_nested_roster_sha256(record_keys),
        included_record_roster_sha256=included_roster,
        excluded_record_roster_sha256=_nested_roster_sha256(
            tuple(f"{record.dataset_id}:{record.record_id}" for record in excluded)
        ),
        source_patient_roster_sha256=_nested_roster_sha256(
            tuple(sorted(patient_fold))
        ),
        content_component_split_receipt_sha256=_nested_roster_sha256(
            {
                "patient_fold": sorted(patient_fold.items()),
                "content_component_fold": sorted(component_fold.items()),
                "fold_count": 5,
                "policy": PREPROCESSING_SPLIT_POLICY,
            }
        ),
        raw_qc_intersection_receipt_sha256=raw_qc_receipt,
        record_count=len(ordered),
        included_record_count=len(included),
        excluded_record_count=len(excluded),
        included_patient_count=len(patient_fold),
        included_content_component_count=len(component_fold),
        fold_record_counts=tuple(
            sum(record.nested_dev_fold == fold for record in included)
            for fold in range(5)
        ),
    )


@dataclass(frozen=True)
class PreprocessingParityProtocolReceipt:
    """Common immutable lineage shared by all five formal arm runs."""

    nested_dev_manifest_receipt_sha256: str
    source_patient_roster_sha256: str
    content_component_split_receipt_sha256: str
    raw_qc_intersection_receipt_sha256: str
    foundation_feature_receipt_sha256: str
    tuev_source_manifest_receipt_sha256: str
    tusz_source_manifest_receipt_sha256: str
    head_architecture_receipt_sha256: str
    optimizer_schedule_receipt_sha256: str
    seed_roster_receipt_sha256: str
    evaluation_policy_receipt_sha256: str
    selection_policy_receipt_sha256: str
    arm_specs_sha256: str = FROZEN_PREPROCESSING_ARM_SPECS_SHA256
    arm_ids: tuple[str, ...] = PREPROCESSING_ARM_IDS
    run_tier: str = PREPROCESSING_FORMAL_RUN_TIER
    source_scope: str = PREPROCESSING_SOURCE_SCOPE
    forbidden_partitions: tuple[str, ...] = PREPROCESSING_FORBIDDEN_PARTITIONS
    label_policy: str = PREPROCESSING_LABEL_POLICY
    raw_qc_policy: str = PREPROCESSING_RAW_QC_POLICY
    split_policy: str = PREPROCESSING_SPLIT_POLICY
    soz_labels_used: bool = False
    source_dev_used: bool = False
    source_eval_used: bool = False
    private_data_used: bool = False
    smoke_is_formal: bool = False
    formal_v3_tokens_authorized: bool = False
    schema_version: str = PREPROCESSING_PARITY_PROTOCOL_SCHEMA

    def __post_init__(self) -> None:
        for field in (
            "nested_dev_manifest_receipt_sha256",
            "source_patient_roster_sha256",
            "content_component_split_receipt_sha256",
            "raw_qc_intersection_receipt_sha256",
            "foundation_feature_receipt_sha256",
            "tuev_source_manifest_receipt_sha256",
            "tusz_source_manifest_receipt_sha256",
            "head_architecture_receipt_sha256",
            "optimizer_schedule_receipt_sha256",
            "seed_roster_receipt_sha256",
            "evaluation_policy_receipt_sha256",
            "selection_policy_receipt_sha256",
            "arm_specs_sha256",
        ):
            object.__setattr__(
                self, field, _require_sha256(getattr(self, field), field=field)
            )
        if self.arm_specs_sha256 != FROZEN_PREPROCESSING_ARM_SPECS_SHA256:
            raise ValueError("Protocol arm specs differ from the frozen five-arm design")
        if (
            self.selection_policy_receipt_sha256
            != LOCKED_PREPROCESSING_SELECTION_POLICY_RECEIPT_SHA256
        ):
            raise ValueError("Protocol does not bind the unique locked selection rule")
        object.__setattr__(
            self,
            "arm_ids",
            _normalize_unique_strings(
                self.arm_ids, field="arm_ids", expected=PREPROCESSING_ARM_IDS
            ),
        )
        object.__setattr__(
            self,
            "forbidden_partitions",
            _normalize_unique_strings(
                self.forbidden_partitions,
                field="forbidden_partitions",
                expected=PREPROCESSING_FORBIDDEN_PARTITIONS,
            ),
        )
        fixed_strings = {
            "run_tier": PREPROCESSING_FORMAL_RUN_TIER,
            "source_scope": PREPROCESSING_SOURCE_SCOPE,
            "label_policy": PREPROCESSING_LABEL_POLICY,
            "raw_qc_policy": PREPROCESSING_RAW_QC_POLICY,
            "split_policy": PREPROCESSING_SPLIT_POLICY,
            "schema_version": PREPROCESSING_PARITY_PROTOCOL_SCHEMA,
        }
        for field, expected in fixed_strings.items():
            if getattr(self, field) != expected:
                raise ValueError(f"Protocol {field} differs from the frozen policy")
        for field in (
            "soz_labels_used",
            "source_dev_used",
            "source_eval_used",
            "private_data_used",
            "smoke_is_formal",
            "formal_v3_tokens_authorized",
        ):
            _require_bool(getattr(self, field), field=field, expected=False)

    def require_nested_dev_manifest(
        self, manifest: PreprocessingParityNestedDevManifest
    ) -> None:
        if not isinstance(manifest, PreprocessingParityNestedDevManifest):
            raise TypeError("manifest must be PreprocessingParityNestedDevManifest")
        comparisons = {
            "nested_dev_manifest_receipt_sha256": manifest.receipt_sha256,
            "source_patient_roster_sha256": manifest.source_patient_roster_sha256,
            "content_component_split_receipt_sha256": (
                manifest.content_component_split_receipt_sha256
            ),
            "raw_qc_intersection_receipt_sha256": (
                manifest.raw_qc_intersection_receipt_sha256
            ),
            "tuev_source_manifest_receipt_sha256": (
                manifest.tuev_source_manifest_receipt_sha256
            ),
            "tusz_source_manifest_receipt_sha256": (
                manifest.tusz_source_manifest_receipt_sha256
            ),
        }
        for field, expected in comparisons.items():
            if getattr(self, field) != expected:
                raise ValueError(f"Protocol {field} differs from nested-dev manifest")

    @property
    def receipt_sha256(self) -> str:
        return _typed_receipt_sha256(asdict(self))


@dataclass(frozen=True)
class PreprocessingArmResultReceipt:
    """Receipt-only summary of one complete formal arm execution."""

    arm_id: str
    protocol_receipt_sha256: str
    arm_spec_receipt_sha256: str
    execution_receipt_sha256: str
    paired_attrition_receipt_sha256: str
    input_distribution_receipt_sha256: str
    token_distribution_receipt_sha256: str
    tuev_ce6_fidelity_receipt_sha256: str
    tusz_native_fidelity_receipt_sha256: str
    onset_boundary_jitter_receipt_sha256: str
    arm_id_shortcut_probe_receipt_sha256: str
    metric_payload_receipt_sha256: str
    nested_dev_manifest_receipt_sha256: str
    source_patient_roster_sha256: str
    content_component_split_receipt_sha256: str
    raw_qc_intersection_receipt_sha256: str
    foundation_feature_receipt_sha256: str
    selection_policy_receipt_sha256: str
    formal_complete: bool = True
    common_raw_qc_intersection_used: bool = True
    paired_attrition_complete: bool = True
    tuev_ce6_complete: bool = True
    tusz_native_labels_complete: bool = True
    onset_boundary_jitter_complete: bool = True
    arm_id_shortcut_probe_complete: bool = True
    source_dev_used: bool = False
    source_eval_used: bool = False
    private_data_used: bool = False
    soz_labels_used: bool = False
    legacy_formal_v3_tokens_used: bool = False
    run_tier: str = PREPROCESSING_FORMAL_RUN_TIER
    schema_version: str = PREPROCESSING_ARM_RESULT_SCHEMA

    def __post_init__(self) -> None:
        arm_id = _require_arm_id(self.arm_id)
        object.__setattr__(self, "arm_id", arm_id)
        for field in (
            "protocol_receipt_sha256",
            "arm_spec_receipt_sha256",
            "execution_receipt_sha256",
            "paired_attrition_receipt_sha256",
            "input_distribution_receipt_sha256",
            "token_distribution_receipt_sha256",
            "tuev_ce6_fidelity_receipt_sha256",
            "tusz_native_fidelity_receipt_sha256",
            "onset_boundary_jitter_receipt_sha256",
            "arm_id_shortcut_probe_receipt_sha256",
            "metric_payload_receipt_sha256",
            "nested_dev_manifest_receipt_sha256",
            "source_patient_roster_sha256",
            "content_component_split_receipt_sha256",
            "raw_qc_intersection_receipt_sha256",
            "foundation_feature_receipt_sha256",
            "selection_policy_receipt_sha256",
        ):
            object.__setattr__(
                self, field, _require_sha256(getattr(self, field), field=field)
            )
        expected_spec_sha = FROZEN_PREPROCESSING_ARM_SPEC_BY_ID[
            arm_id
        ].receipt_sha256
        if self.arm_spec_receipt_sha256 != expected_spec_sha:
            raise ValueError(f"{arm_id} result binds the wrong frozen arm spec")
        for field in (
            "formal_complete",
            "common_raw_qc_intersection_used",
            "paired_attrition_complete",
            "tuev_ce6_complete",
            "tusz_native_labels_complete",
            "onset_boundary_jitter_complete",
            "arm_id_shortcut_probe_complete",
        ):
            _require_bool(getattr(self, field), field=field, expected=True)
        for field in (
            "source_dev_used",
            "source_eval_used",
            "private_data_used",
            "soz_labels_used",
            "legacy_formal_v3_tokens_used",
        ):
            _require_bool(getattr(self, field), field=field, expected=False)
        if self.run_tier != PREPROCESSING_FORMAL_RUN_TIER:
            raise ValueError("Smoke/preflight arm results cannot be formal parity")
        if self.schema_version != PREPROCESSING_ARM_RESULT_SCHEMA:
            raise ValueError("Unsupported preprocessing arm-result schema")

    def require_protocol(self, protocol: PreprocessingParityProtocolReceipt) -> None:
        if not isinstance(protocol, PreprocessingParityProtocolReceipt):
            raise TypeError("protocol must be PreprocessingParityProtocolReceipt")
        comparisons = {
            "protocol_receipt_sha256": protocol.receipt_sha256,
            "nested_dev_manifest_receipt_sha256": (
                protocol.nested_dev_manifest_receipt_sha256
            ),
            "source_patient_roster_sha256": protocol.source_patient_roster_sha256,
            "content_component_split_receipt_sha256": (
                protocol.content_component_split_receipt_sha256
            ),
            "raw_qc_intersection_receipt_sha256": (
                protocol.raw_qc_intersection_receipt_sha256
            ),
            "foundation_feature_receipt_sha256": (
                protocol.foundation_feature_receipt_sha256
            ),
            "selection_policy_receipt_sha256": (
                protocol.selection_policy_receipt_sha256
            ),
        }
        for field, expected in comparisons.items():
            if getattr(self, field) != expected:
                raise ValueError(
                    f"{self.arm_id} result {field} differs from the common protocol"
                )

    @property
    def receipt_sha256(self) -> str:
        return _typed_receipt_sha256(asdict(self))


@dataclass(frozen=True, init=False)
class FormalPreprocessingSelectionDecision:
    """Opaque output of the preregistered numerical evaluator."""

    selected_arm_id: str
    protocol_receipt_sha256: str
    selection_policy_receipt_sha256: str
    decision_trace_receipt_sha256: str
    arm_result_receipt_sha256_by_id: Mapping[str, str]
    all_required_noninferiority_passed: bool
    all_required_sanity_and_shortcut_gates_passed: bool
    schema_version: str
    _trace: PreprocessingSelectionDecisionTrace

    def __init__(
        self,
        *,
        _issuer: object,
        selected_arm_id: str,
        protocol_receipt_sha256: str,
        selection_policy_receipt_sha256: str,
        decision_trace_receipt_sha256: str,
        arm_result_receipt_sha256_by_id: Mapping[str, str],
        trace: PreprocessingSelectionDecisionTrace,
    ) -> None:
        if _issuer is not _DECISION_ISSUER:
            raise TypeError(
                "FormalPreprocessingSelectionDecision can only be issued by "
                "the preregistered parity evaluator"
            )
        arm_id = _require_arm_id(selected_arm_id, field="selected_arm_id")
        if arm_id not in DEPLOYABLE_PREPROCESSING_ARM_IDS:
            raise ValueError("O-REF cannot be selected for deployment")
        normalized = _normalize_arm_hash_mapping(
            arm_result_receipt_sha256_by_id,
            field="arm_result_receipt_sha256_by_id",
        )
        if not isinstance(trace, PreprocessingSelectionDecisionTrace):
            raise TypeError("trace must be PreprocessingSelectionDecisionTrace")
        if trace.selected_arm_id != arm_id:
            raise ValueError("Selection decision and trace choose different arms")
        if trace.receipt_sha256 != decision_trace_receipt_sha256:
            raise ValueError("Selection decision trace receipt SHA mismatch")
        object.__setattr__(self, "selected_arm_id", arm_id)
        object.__setattr__(
            self,
            "protocol_receipt_sha256",
            _require_sha256(protocol_receipt_sha256, field="protocol_receipt_sha256"),
        )
        object.__setattr__(
            self,
            "selection_policy_receipt_sha256",
            _require_sha256(
                selection_policy_receipt_sha256,
                field="selection_policy_receipt_sha256",
            ),
        )
        object.__setattr__(
            self,
            "decision_trace_receipt_sha256",
            _require_sha256(
                decision_trace_receipt_sha256,
                field="decision_trace_receipt_sha256",
            ),
        )
        object.__setattr__(
            self,
            "arm_result_receipt_sha256_by_id",
            MappingProxyType(normalized),
        )
        object.__setattr__(self, "all_required_noninferiority_passed", True)
        object.__setattr__(
            self, "all_required_sanity_and_shortcut_gates_passed", True
        )
        object.__setattr__(
            self, "schema_version", PREPROCESSING_SELECTION_DECISION_SCHEMA
        )
        object.__setattr__(self, "_trace", trace)

    def to_payload(self) -> dict[str, object]:
        return {
            "selected_arm_id": self.selected_arm_id,
            "protocol_receipt_sha256": self.protocol_receipt_sha256,
            "selection_policy_receipt_sha256": (
                self.selection_policy_receipt_sha256
            ),
            "decision_trace_receipt_sha256": self.decision_trace_receipt_sha256,
            "arm_result_receipt_sha256_by_id": dict(
                self.arm_result_receipt_sha256_by_id
            ),
            "all_required_noninferiority_passed": True,
            "all_required_sanity_and_shortcut_gates_passed": True,
            "schema_version": self.schema_version,
        }

    @property
    def receipt_sha256(self) -> str:
        return _typed_receipt_sha256(self.to_payload())


def _normalize_arm_hash_mapping(
    values: Mapping[str, object], *, field: str
) -> dict[str, str]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{field} must be a mapping")
    if tuple(sorted(values)) != tuple(sorted(PREPROCESSING_ARM_IDS)):
        raise ValueError(f"{field} must contain exactly the frozen five arms")
    return {
        arm_id: _require_sha256(values[arm_id], field=f"{field}[{arm_id}]")
        for arm_id in PREPROCESSING_ARM_IDS
    }


def _issue_formal_preprocessing_selection_decision(
    *,
    selected_arm_id: str,
    protocol: PreprocessingParityProtocolReceipt,
    arm_results: Mapping[str, PreprocessingArmResultReceipt],
    trace: PreprocessingSelectionDecisionTrace,
) -> FormalPreprocessingSelectionDecision:
    """Private bridge used only after numerical policy evaluation succeeds."""

    normalized_results = _validate_complete_result_set(protocol, arm_results)
    return FormalPreprocessingSelectionDecision(
        _issuer=_DECISION_ISSUER,
        selected_arm_id=selected_arm_id,
        protocol_receipt_sha256=protocol.receipt_sha256,
        selection_policy_receipt_sha256=protocol.selection_policy_receipt_sha256,
        decision_trace_receipt_sha256=trace.receipt_sha256,
        arm_result_receipt_sha256_by_id={
            arm_id: result.receipt_sha256
            for arm_id, result in normalized_results.items()
        },
        trace=trace,
    )


def _validate_complete_result_set(
    protocol: PreprocessingParityProtocolReceipt,
    arm_results: Mapping[str, PreprocessingArmResultReceipt],
) -> dict[str, PreprocessingArmResultReceipt]:
    if not isinstance(protocol, PreprocessingParityProtocolReceipt):
        raise TypeError("protocol must be PreprocessingParityProtocolReceipt")
    if not isinstance(arm_results, Mapping):
        raise TypeError("arm_results must be a mapping")
    if tuple(sorted(arm_results)) != tuple(sorted(PREPROCESSING_ARM_IDS)):
        raise ValueError("Formal parity requires exactly all five frozen arm results")
    normalized: dict[str, PreprocessingArmResultReceipt] = {}
    execution_hashes: set[str] = set()
    metric_hashes: set[str] = set()
    for arm_id in PREPROCESSING_ARM_IDS:
        result = arm_results[arm_id]
        if not isinstance(result, PreprocessingArmResultReceipt):
            raise TypeError(f"arm_results[{arm_id}] has the wrong type")
        if result.arm_id != arm_id:
            raise ValueError(f"arm_results key {arm_id} disagrees with its receipt")
        result.require_protocol(protocol)
        if result.execution_receipt_sha256 in execution_hashes:
            raise ValueError("Distinct arms cannot reuse one execution receipt")
        if result.metric_payload_receipt_sha256 in metric_hashes:
            raise ValueError("Distinct arms cannot reuse one metric payload receipt")
        execution_hashes.add(result.execution_receipt_sha256)
        metric_hashes.add(result.metric_payload_receipt_sha256)
        normalized[arm_id] = result
    return normalized


def _selection_no_go_receipt(
    *,
    reason_code: str,
    protocol: PreprocessingParityProtocolReceipt,
    results: Mapping[str, PreprocessingArmResultReceipt],
    metrics: Mapping[str, PreprocessingArmSelectionMetrics],
    details: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(reason_code, str) or not reason_code:
        raise ValueError("NO-GO reason code must be non-empty")
    return {
        "schema_version": PREPROCESSING_SELECTION_NO_GO_SCHEMA,
        "selection_status": "NO_GO",
        "reason_code": reason_code,
        "protocol_receipt_sha256": protocol.receipt_sha256,
        "selection_policy_receipt_sha256": (
            protocol.selection_policy_receipt_sha256
        ),
        "arm_result_receipt_sha256_by_id": {
            arm_id: results[arm_id].receipt_sha256
            for arm_id in PREPROCESSING_ARM_IDS
        },
        "arm_metric_receipt_sha256_by_id": {
            arm_id: metrics[arm_id].receipt_sha256
            for arm_id in PREPROCESSING_ARM_IDS
        },
        "details": dict(details),
        "jointly_eligible_arm_ids": [],
        "common_arm_across_families_required": True,
        "family_specific_arm_selection_forbidden": True,
        "downstream_capability_issued": False,
    }


def evaluate_preprocessing_arm_selection(
    *,
    protocol: PreprocessingParityProtocolReceipt,
    arm_results: Mapping[str, PreprocessingArmResultReceipt],
    arm_metrics: Mapping[str, PreprocessingArmSelectionMetrics],
) -> FormalPreprocessingSelectionDecision:
    """Apply the one locked cross-family noninferiority and tie-break rule.

    The function has no configurable margins, endpoint names, or tie-break.
    Expected failures of a preregistered numerical gate raise
    :class:`PreprocessingArmSelectionNoGoError`; integrity/contract failures
    remain ordinary hard errors.  Neither can produce a capability.
    """

    results = _validate_complete_result_set(protocol, arm_results)
    if (
        protocol.selection_policy_receipt_sha256
        != LOCKED_PREPROCESSING_SELECTION_POLICY_RECEIPT_SHA256
    ):
        raise ValueError("Protocol does not bind the locked selection policy")
    if not isinstance(arm_metrics, Mapping) or tuple(sorted(arm_metrics)) != tuple(
        sorted(PREPROCESSING_ARM_IDS)
    ):
        raise ValueError("Selection requires exactly all five arm metric payloads")
    metrics: dict[str, PreprocessingArmSelectionMetrics] = {}
    denominators: set[tuple[object, ...]] = set()
    for arm_id in PREPROCESSING_ARM_IDS:
        metric = arm_metrics[arm_id]
        if not isinstance(metric, PreprocessingArmSelectionMetrics):
            raise TypeError(f"arm_metrics[{arm_id}] has the wrong type")
        if metric.arm_id != arm_id:
            raise ValueError(f"arm_metrics key {arm_id} disagrees with its payload")
        if metric.protocol_receipt_sha256 != protocol.receipt_sha256:
            raise ValueError(f"{arm_id} metrics bind the wrong protocol")
        if results[arm_id].metric_payload_receipt_sha256 != metric.receipt_sha256:
            raise ValueError(f"{arm_id} result does not bind its selection metrics")
        if (
            results[arm_id].paired_attrition_receipt_sha256
            != metric.paired_denominator_receipt_sha256
        ):
            raise ValueError(f"{arm_id} denominator receipt is not result-bound")
        if arm_id in DEPLOYABLE_PREPROCESSING_ARM_IDS:
            denominator = (
                metric.paired_denominator_receipt_sha256,
                metric.tuev_paired_patient_count,
                metric.tuev_paired_content_component_count,
                metric.tusz_paired_patient_count,
                metric.tusz_paired_explicit_cell_count,
            )
            denominators.add(denominator)
        metrics[arm_id] = metric
    if len(denominators) != 1:
        raise ValueError(
            "Arm metrics have different paired patient/component/cell denominators"
        )

    attrition_failures = [
        arm_id
        for arm_id in DEPLOYABLE_PREPROCESSING_ARM_IDS
        if metrics[arm_id].paired_attrition_count
        > LOCKED_PREPROCESSING_SELECTION_POLICY.maximum_paired_attrition_count
    ]
    if attrition_failures:
        raise PreprocessingArmSelectionNoGoError(
            "One or more deployable arms failed the paired-attrition gate",
            receipt=_selection_no_go_receipt(
                reason_code="paired_attrition_gate_failed",
                protocol=protocol,
                results=results,
                metrics=metrics,
                details={
                    "failed_arm_ids": attrition_failures,
                    "maximum_paired_attrition_count": (
                        LOCKED_PREPROCESSING_SELECTION_POLICY.maximum_paired_attrition_count
                    ),
                    "paired_attrition_count_by_arm": {
                        arm_id: metrics[arm_id].paired_attrition_count
                        for arm_id in DEPLOYABLE_PREPROCESSING_ARM_IDS
                    },
                },
            ),
        )

    o_ref = metrics["O-REF"]
    assert o_ref.official_signal_relative_l2_error is not None
    assert o_ref.official_token_cosine_distance is not None
    if (
        o_ref.official_signal_relative_l2_error
        > LOCKED_PREPROCESSING_SELECTION_POLICY.maximum_official_signal_relative_l2_error
        or o_ref.official_token_cosine_distance
        > LOCKED_PREPROCESSING_SELECTION_POLICY.maximum_official_token_cosine_distance
    ):
        raise PreprocessingArmSelectionNoGoError(
            "O-REF failed the official exact signal/token sanity gate",
            receipt=_selection_no_go_receipt(
                reason_code="official_reference_sanity_gate_failed",
                protocol=protocol,
                results=results,
                metrics=metrics,
                details={
                    "official_signal_relative_l2_error": (
                        o_ref.official_signal_relative_l2_error
                    ),
                    "maximum_official_signal_relative_l2_error": (
                        LOCKED_PREPROCESSING_SELECTION_POLICY.maximum_official_signal_relative_l2_error
                    ),
                    "official_token_cosine_distance": (
                        o_ref.official_token_cosine_distance
                    ),
                    "maximum_official_token_cosine_distance": (
                        LOCKED_PREPROCESSING_SELECTION_POLICY.maximum_official_token_cosine_distance
                    ),
                },
            ),
        )

    tie_break = LOCKED_PREPROCESSING_SELECTION_POLICY.deployable_tie_break
    best_tuev_value = max(
        metrics[arm_id].tuev_macro_ce6_f1
        for arm_id in DEPLOYABLE_PREPROCESSING_ARM_IDS
    )
    tuev_reference = next(
        arm_id
        for arm_id in tie_break
        if metrics[arm_id].tuev_macro_ce6_f1 == best_tuev_value
    )
    best_tusz_value = min(
        metrics[arm_id].tusz_native_macro_bce
        for arm_id in DEPLOYABLE_PREPROCESSING_ARM_IDS
    )
    tusz_reference = next(
        arm_id
        for arm_id in tie_break
        if metrics[arm_id].tusz_native_macro_bce == best_tusz_value
    )

    eligible: list[str] = []
    gate_details: dict[str, dict[str, object]] = {}
    for arm_id in tie_break:
        metric = metrics[arm_id]
        tuev_point_difference = (
            metric.tuev_macro_ce6_f1
            - metrics[tuev_reference].tuev_macro_ce6_f1
        )
        tuev_lower = metric.tuev_f1_difference_lower95_by_reference[
            tuev_reference
        ]
        if tuev_lower > tuev_point_difference + 1e-12:
            raise ValueError(f"{arm_id} TUEV lower CI excludes its point difference")
        tusz_point_difference = (
            metric.tusz_native_macro_bce
            - metrics[tusz_reference].tusz_native_macro_bce
        )
        tusz_upper = metric.tusz_bce_difference_upper95_by_reference[
            tusz_reference
        ]
        if tusz_upper < tusz_point_difference - 1e-12:
            raise ValueError(f"{arm_id} TUSZ upper CI excludes its point difference")
        tuev_noninferior = (
            tuev_lower
            >= -LOCKED_PREPROCESSING_SELECTION_POLICY.tuev_macro_f1_noninferiority_margin
        )
        tusz_noninferior = (
            tusz_upper
            <= LOCKED_PREPROCESSING_SELECTION_POLICY.tusz_macro_bce_noninferiority_margin
        )
        jitter_passed = (
            metric.tuev_jitter_macro_f1_max_drop
            <= LOCKED_PREPROCESSING_SELECTION_POLICY.maximum_tuev_jitter_macro_f1_drop
            and metric.tusz_jitter_macro_bce_max_increase
            <= LOCKED_PREPROCESSING_SELECTION_POLICY.maximum_tusz_jitter_macro_bce_increase
        )
        gate_details[arm_id] = {
            "tuev_point_difference": tuev_point_difference,
            "tuev_difference_lower95": tuev_lower,
            "tuev_noninferior": tuev_noninferior,
            "tusz_point_difference": tusz_point_difference,
            "tusz_difference_upper95": tusz_upper,
            "tusz_noninferior": tusz_noninferior,
            "tuev_jitter_macro_f1_max_drop": (
                metric.tuev_jitter_macro_f1_max_drop
            ),
            "tusz_jitter_macro_bce_max_increase": (
                metric.tusz_jitter_macro_bce_max_increase
            ),
            "jitter_passed": jitter_passed,
            "jointly_eligible": (
                tuev_noninferior and tusz_noninferior and jitter_passed
            ),
        }
        if tuev_noninferior and tusz_noninferior and jitter_passed:
            eligible.append(arm_id)
    if not eligible:
        jointly_noninferior_before_jitter = [
            arm_id
            for arm_id in tie_break
            if gate_details[arm_id]["tuev_noninferior"]
            and gate_details[arm_id]["tusz_noninferior"]
        ]
        reason_code = (
            "all_jointly_noninferior_arms_failed_jitter_gate"
            if jointly_noninferior_before_jitter
            else "morphology_ictal_noninferiority_conflict"
        )
        message = (
            "All jointly noninferior deployable arms failed the "
            "label-preserving robustness gate"
            if jointly_noninferior_before_jitter
            else (
                "Morphology/ictal preprocessing conflict: no common deployable "
                "arm passes both paired noninferiority rules"
            )
        )
        raise PreprocessingArmSelectionNoGoError(
            message,
            receipt=_selection_no_go_receipt(
                reason_code=reason_code,
                protocol=protocol,
                results=results,
                metrics=metrics,
                details={
                    "tuev_reference_arm_id": tuev_reference,
                    "tusz_reference_arm_id": tusz_reference,
                    "jointly_noninferior_before_jitter_arm_ids": (
                        jointly_noninferior_before_jitter
                    ),
                    "tuev_macro_f1_noninferiority_margin": (
                        LOCKED_PREPROCESSING_SELECTION_POLICY.tuev_macro_f1_noninferiority_margin
                    ),
                    "tusz_macro_bce_noninferiority_margin": (
                        LOCKED_PREPROCESSING_SELECTION_POLICY.tusz_macro_bce_noninferiority_margin
                    ),
                    "maximum_tuev_jitter_macro_f1_drop": (
                        LOCKED_PREPROCESSING_SELECTION_POLICY.maximum_tuev_jitter_macro_f1_drop
                    ),
                    "maximum_tusz_jitter_macro_bce_increase": (
                        LOCKED_PREPROCESSING_SELECTION_POLICY.maximum_tusz_jitter_macro_bce_increase
                    ),
                    "gate_details_by_arm": gate_details,
                },
            ),
        )
    selected = next(arm_id for arm_id in tie_break if arm_id in eligible)
    trace = PreprocessingSelectionDecisionTrace(
        protocol_receipt_sha256=protocol.receipt_sha256,
        selection_policy_receipt_sha256=(
            LOCKED_PREPROCESSING_SELECTION_POLICY_RECEIPT_SHA256
        ),
        arm_metrics_by_id=metrics,
        arm_metric_receipt_sha256_by_id={
            arm_id: metrics[arm_id].receipt_sha256
            for arm_id in PREPROCESSING_ARM_IDS
        },
        tuev_reference_arm_id=tuev_reference,
        tusz_reference_arm_id=tusz_reference,
        jointly_noninferior_arm_ids=tuple(eligible),
        selected_arm_id=selected,
    )
    return _issue_formal_preprocessing_selection_decision(
        selected_arm_id=selected,
        protocol=protocol,
        arm_results=results,
        trace=trace,
    )


_PROTOCOL_FIELDS = frozenset(
    field.name for field in PreprocessingParityProtocolReceipt.__dataclass_fields__.values()
)
_ARM_RESULT_FIELDS = frozenset(
    field.name for field in PreprocessingArmResultReceipt.__dataclass_fields__.values()
)
_NESTED_DEV_RECORD_FIELDS = frozenset(
    field.name
    for field in PreprocessingNestedDevSourceRecord.__dataclass_fields__.values()
)
_NESTED_DEV_MANIFEST_FIELDS = frozenset(
    field.name
    for field in PreprocessingParityNestedDevManifest.__dataclass_fields__.values()
)
_SELECTION_POLICY_FIELDS = frozenset(
    field.name for field in PreprocessingSelectionPolicy.__dataclass_fields__.values()
)
_ARM_SELECTION_METRICS_FIELDS = frozenset(
    field.name
    for field in PreprocessingArmSelectionMetrics.__dataclass_fields__.values()
)
_SELECTION_TRACE_FIELDS = frozenset(
    {
        "protocol_receipt_sha256",
        "selection_policy_receipt_sha256",
        "arm_metrics_by_id",
        "arm_metric_receipt_sha256_by_id",
        "tuev_reference_arm_id",
        "tusz_reference_arm_id",
        "jointly_noninferior_arm_ids",
        "selected_arm_id",
        "tie_break",
        "common_arm_across_families",
        "o_ref_sanity_passed",
        "schema_version",
    }
)
_DECISION_FIELDS = frozenset(
    {
        "selected_arm_id",
        "protocol_receipt_sha256",
        "selection_policy_receipt_sha256",
        "decision_trace_receipt_sha256",
        "arm_result_receipt_sha256_by_id",
        "all_required_noninferiority_passed",
        "all_required_sanity_and_shortcut_gates_passed",
        "schema_version",
    }
)
_SELECTION_FIELDS = frozenset(
    {
        "schema_version",
        "formal_complete",
        "run_tier",
        "source_scope",
        "forbidden_partitions",
        "label_policy",
        "o_ref_role",
        "nested_dev_manifest_file_sha256",
        "protocol_file_sha256",
        "protocol_receipt_sha256",
        "source_patient_roster_sha256",
        "content_component_split_receipt_sha256",
        "raw_qc_intersection_receipt_sha256",
        "foundation_feature_receipt_sha256",
        "selection_policy_receipt_sha256",
        "selection_policy_file_sha256",
        "decision_trace_file_sha256",
        "decision_trace_receipt_sha256",
        "arm_specs_sha256",
        "arm_result_file_sha256_by_id",
        "arm_result_receipt_sha256_by_id",
        "decision",
        "decision_receipt_sha256",
        "selected_arm_id",
        "selected_arm_spec_receipt_sha256",
        "selected_arm_result_receipt_sha256",
        "legacy_formal_v3_tokens_authorized",
        "smoke_is_formal",
    }
)
_BUNDLE_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "selection_artifact_sha256",
        "nested_dev_manifest_file_sha256",
        "protocol_file_sha256",
        "protocol_receipt_sha256",
        "selection_policy_file_sha256",
        "selection_policy_receipt_sha256",
        "decision_trace_file_sha256",
        "decision_trace_receipt_sha256",
        "arm_result_file_sha256_by_id",
        "arm_result_receipt_sha256_by_id",
        "decision_receipt_sha256",
        "selected_arm_id",
        "selected_arm_result_receipt_sha256",
    }
)


def _strict_absolute_directory(path: str | Path, *, output: bool = False) -> Path:
    value = Path(os.path.abspath(path))
    if value.name in {"", ".", ".."}:
        raise ValueError("Preprocessing selection requires a concrete directory")
    if output:
        for component in (value.parent, *value.parent.parents):
            if os.path.lexists(component) and component.is_symlink():
                raise ValueError("Selection output cannot traverse symlinks")
        if not value.parent.is_dir():
            raise FileNotFoundError("Selection output parent does not exist")
        if os.path.lexists(value):
            raise FileExistsError(f"Selection output already exists: {value}")
        return value
    if value.is_symlink() or not value.is_dir() or value.resolve() != value:
        raise ValueError("Selection bundle must be a regular absolute directory")
    return value


def _read_regular_canonical_json(path: Path, *, label: str) -> tuple[bytes, dict[str, object]]:
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise ValueError(f"{label} must be a regular file")
    raw = path.read_bytes()
    return raw, _parse_canonical_json(raw, label=label)


def _protocol_from_payload(payload: Mapping[str, object]) -> PreprocessingParityProtocolReceipt:
    if set(payload) != _PROTOCOL_FIELDS:
        raise ValueError("Preprocessing protocol violates its closed schema")
    normalized = dict(payload)
    for field in ("arm_ids", "forbidden_partitions"):
        value = normalized[field]
        if not isinstance(value, list):
            raise TypeError(f"Protocol {field} must be a JSON array")
        normalized[field] = tuple(value)
    try:
        return PreprocessingParityProtocolReceipt(**normalized)
    except (TypeError, ValueError) as exc:
        raise ValueError("Preprocessing protocol payload is invalid") from exc


def _arm_result_from_payload(payload: Mapping[str, object]) -> PreprocessingArmResultReceipt:
    if set(payload) != _ARM_RESULT_FIELDS:
        raise ValueError("Preprocessing arm result violates its closed schema")
    try:
        return PreprocessingArmResultReceipt(**dict(payload))
    except (TypeError, ValueError) as exc:
        raise ValueError("Preprocessing arm-result payload is invalid") from exc


def _nested_dev_manifest_from_payload(
    payload: Mapping[str, object],
) -> PreprocessingParityNestedDevManifest:
    if set(payload) != _NESTED_DEV_MANIFEST_FIELDS:
        raise ValueError("Nested-dev manifest violates its closed schema")
    raw_records = payload.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("Nested-dev manifest records must be a non-empty array")
    records: list[PreprocessingNestedDevSourceRecord] = []
    for index, raw_record in enumerate(raw_records):
        if not isinstance(raw_record, dict) or set(raw_record) != _NESTED_DEV_RECORD_FIELDS:
            raise ValueError(f"Nested-dev record {index} violates its closed schema")
        try:
            records.append(PreprocessingNestedDevSourceRecord(**raw_record))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Nested-dev record {index} is invalid") from exc
    fold_counts = payload.get("fold_record_counts")
    forbidden = payload.get("forbidden_partitions")
    if not isinstance(fold_counts, list) or not isinstance(forbidden, list):
        raise TypeError("Nested-dev manifest roster fields must be JSON arrays")
    normalized = dict(payload)
    normalized["records"] = tuple(records)
    normalized["fold_record_counts"] = tuple(fold_counts)
    normalized["forbidden_partitions"] = tuple(forbidden)
    try:
        return PreprocessingParityNestedDevManifest(**normalized)
    except (TypeError, ValueError) as exc:
        raise ValueError("Nested-dev manifest payload is invalid") from exc


def _selection_policy_from_payload(
    payload: Mapping[str, object],
) -> PreprocessingSelectionPolicy:
    if set(payload) != _SELECTION_POLICY_FIELDS:
        raise ValueError("Preprocessing selection policy violates its closed schema")
    normalized = dict(payload)
    tie_break = normalized.get("deployable_tie_break")
    if not isinstance(tie_break, list):
        raise TypeError("Selection-policy tie-break must be a JSON array")
    normalized["deployable_tie_break"] = tuple(tie_break)
    try:
        policy = PreprocessingSelectionPolicy(**normalized)
    except (TypeError, ValueError) as exc:
        raise ValueError("Preprocessing selection-policy payload is invalid") from exc
    if asdict(policy) != asdict(LOCKED_PREPROCESSING_SELECTION_POLICY):
        raise ValueError("Preprocessing selection policy differs from the locked rule")
    return policy


def _arm_metrics_from_payload(
    payload: Mapping[str, object],
) -> PreprocessingArmSelectionMetrics:
    if set(payload) != _ARM_SELECTION_METRICS_FIELDS:
        raise ValueError("Preprocessing arm metrics violate their closed schema")
    normalized = dict(payload)
    for field in (
        "tuev_f1_difference_lower95_by_reference",
        "tusz_bce_difference_upper95_by_reference",
    ):
        if not isinstance(normalized.get(field), dict):
            raise TypeError(f"Arm metric {field} must be a JSON object")
    try:
        return PreprocessingArmSelectionMetrics(**normalized)
    except (TypeError, ValueError) as exc:
        raise ValueError("Preprocessing arm-selection metrics are invalid") from exc


def _selection_trace_from_payload(
    payload: Mapping[str, object],
) -> PreprocessingSelectionDecisionTrace:
    if set(payload) != _SELECTION_TRACE_FIELDS:
        raise ValueError("Preprocessing decision trace violates its closed schema")
    raw_metrics = payload.get("arm_metrics_by_id")
    if not isinstance(raw_metrics, dict) or tuple(sorted(raw_metrics)) != tuple(
        sorted(PREPROCESSING_ARM_IDS)
    ):
        raise ValueError("Decision trace must contain exactly five arm metrics")
    metrics: dict[str, PreprocessingArmSelectionMetrics] = {}
    for arm_id in PREPROCESSING_ARM_IDS:
        metric_payload = raw_metrics[arm_id]
        if not isinstance(metric_payload, dict):
            raise TypeError(f"Decision trace metric {arm_id} must be an object")
        metrics[arm_id] = _arm_metrics_from_payload(metric_payload)
    jointly = payload.get("jointly_noninferior_arm_ids")
    tie_break = payload.get("tie_break")
    if not isinstance(jointly, list) or not isinstance(tie_break, list):
        raise TypeError("Decision trace arm rosters must be JSON arrays")
    try:
        return PreprocessingSelectionDecisionTrace(
            protocol_receipt_sha256=str(
                payload.get("protocol_receipt_sha256", "")
            ),
            selection_policy_receipt_sha256=str(
                payload.get("selection_policy_receipt_sha256", "")
            ),
            arm_metrics_by_id=metrics,
            arm_metric_receipt_sha256_by_id=payload.get(
                "arm_metric_receipt_sha256_by_id", {}
            ),
            tuev_reference_arm_id=str(payload.get("tuev_reference_arm_id", "")),
            tusz_reference_arm_id=str(payload.get("tusz_reference_arm_id", "")),
            jointly_noninferior_arm_ids=tuple(jointly),
            selected_arm_id=str(payload.get("selected_arm_id", "")),
            tie_break=tuple(tie_break),
            common_arm_across_families=payload.get(
                "common_arm_across_families", False
            ),
            o_ref_sanity_passed=payload.get("o_ref_sanity_passed", False),
            schema_version=str(payload.get("schema_version", "")),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Preprocessing decision-trace payload is invalid") from exc


def _decision_from_payload(
    payload: Mapping[str, object],
    *,
    protocol: PreprocessingParityProtocolReceipt,
    arm_results: Mapping[str, PreprocessingArmResultReceipt],
    trace: PreprocessingSelectionDecisionTrace,
) -> FormalPreprocessingSelectionDecision:
    if set(payload) != _DECISION_FIELDS:
        raise ValueError("Preprocessing decision violates its closed schema")
    if payload.get("schema_version") != PREPROCESSING_SELECTION_DECISION_SCHEMA:
        raise ValueError("Unsupported preprocessing decision schema")
    _require_bool(
        payload.get("all_required_noninferiority_passed"),
        field="all_required_noninferiority_passed",
        expected=True,
    )
    _require_bool(
        payload.get("all_required_sanity_and_shortcut_gates_passed"),
        field="all_required_sanity_and_shortcut_gates_passed",
        expected=True,
    )
    declared_hashes = _normalize_arm_hash_mapping(
        payload.get("arm_result_receipt_sha256_by_id", {}),
        field="arm_result_receipt_sha256_by_id",
    )
    actual_hashes = {
        arm_id: arm_results[arm_id].receipt_sha256 for arm_id in PREPROCESSING_ARM_IDS
    }
    if declared_hashes != actual_hashes:
        raise ValueError("Decision arm-result receipts differ from the bundle")
    decision = FormalPreprocessingSelectionDecision(
        _issuer=_DECISION_ISSUER,
        selected_arm_id=str(payload.get("selected_arm_id", "")),
        protocol_receipt_sha256=str(payload.get("protocol_receipt_sha256", "")),
        selection_policy_receipt_sha256=str(
            payload.get("selection_policy_receipt_sha256", "")
        ),
        decision_trace_receipt_sha256=str(
            payload.get("decision_trace_receipt_sha256", "")
        ),
        arm_result_receipt_sha256_by_id=declared_hashes,
        trace=trace,
    )
    if decision.protocol_receipt_sha256 != protocol.receipt_sha256:
        raise ValueError("Decision binds the wrong preprocessing protocol")
    if (
        decision.selection_policy_receipt_sha256
        != protocol.selection_policy_receipt_sha256
    ):
        raise ValueError("Decision binds the wrong preregistered selection policy")
    return decision


@dataclass(frozen=True)
class _LoadedSelectionBundle:
    path: Path
    selection_artifact_sha256: str
    bundle_receipt_sha256: str
    nested_dev_manifest_file_sha256: str
    nested_dev_manifest: PreprocessingParityNestedDevManifest
    protocol_file_sha256: str
    protocol: PreprocessingParityProtocolReceipt
    arm_results: Mapping[str, PreprocessingArmResultReceipt]
    arm_result_file_sha256_by_id: Mapping[str, str]
    decision: FormalPreprocessingSelectionDecision
    selected_arm_id: str
    selected_arm_result_receipt_sha256: str


def _read_selection_bundle(
    directory: str | Path,
    *,
    expected_artifact_sha256: str,
    expected_protocol_receipt_sha256: str,
) -> _LoadedSelectionBundle:
    source = _strict_absolute_directory(directory)
    expected_files = {
        PREPROCESSING_SELECTION_FILENAME,
        PREPROCESSING_PROTOCOL_FILENAME,
        PREPROCESSING_SELECTION_RECEIPT_FILENAME,
        PREPROCESSING_SELECTION_POLICY_FILENAME,
        PREPROCESSING_SELECTION_TRACE_FILENAME,
        PREPROCESSING_NESTED_DEV_MANIFEST_FILENAME,
        *_ARM_RESULT_FILENAME_BY_ID.values(),
    }
    if {item.name for item in source.iterdir()} != expected_files:
        raise ValueError("Selection bundle has missing or unknown files")

    protocol_raw, protocol_payload = _read_regular_canonical_json(
        source / PREPROCESSING_PROTOCOL_FILENAME,
        label=PREPROCESSING_PROTOCOL_FILENAME,
    )
    protocol = _protocol_from_payload(protocol_payload)
    if protocol.receipt_sha256 != _require_sha256(
        expected_protocol_receipt_sha256,
        field="expected_protocol_receipt_sha256",
    ):
        raise ValueError("Preprocessing protocol receipt SHA mismatch")
    protocol_file_sha = hashlib.sha256(protocol_raw).hexdigest()

    nested_raw, nested_payload = _read_regular_canonical_json(
        source / PREPROCESSING_NESTED_DEV_MANIFEST_FILENAME,
        label=PREPROCESSING_NESTED_DEV_MANIFEST_FILENAME,
    )
    nested_manifest = _nested_dev_manifest_from_payload(nested_payload)
    nested_file_sha = hashlib.sha256(nested_raw).hexdigest()
    protocol.require_nested_dev_manifest(nested_manifest)

    policy_raw, policy_payload = _read_regular_canonical_json(
        source / PREPROCESSING_SELECTION_POLICY_FILENAME,
        label=PREPROCESSING_SELECTION_POLICY_FILENAME,
    )
    policy = _selection_policy_from_payload(policy_payload)
    policy_file_sha = hashlib.sha256(policy_raw).hexdigest()
    if policy.receipt_sha256 != protocol.selection_policy_receipt_sha256:
        raise ValueError("Selection policy does not match the common protocol")

    arm_results: dict[str, PreprocessingArmResultReceipt] = {}
    arm_file_hashes: dict[str, str] = {}
    for arm_id in PREPROCESSING_ARM_IDS:
        filename = _ARM_RESULT_FILENAME_BY_ID[arm_id]
        raw, payload = _read_regular_canonical_json(source / filename, label=filename)
        result = _arm_result_from_payload(payload)
        if result.arm_id != arm_id:
            raise ValueError(f"{filename} contains the wrong arm")
        arm_results[arm_id] = result
        arm_file_hashes[arm_id] = hashlib.sha256(raw).hexdigest()
    arm_results = _validate_complete_result_set(protocol, arm_results)

    trace_raw, trace_payload = _read_regular_canonical_json(
        source / PREPROCESSING_SELECTION_TRACE_FILENAME,
        label=PREPROCESSING_SELECTION_TRACE_FILENAME,
    )
    trace_file_sha = hashlib.sha256(trace_raw).hexdigest()
    trace = _selection_trace_from_payload(trace_payload)
    if trace.protocol_receipt_sha256 != protocol.receipt_sha256:
        raise ValueError("Decision trace binds the wrong common protocol")
    replayed_decision = evaluate_preprocessing_arm_selection(
        protocol=protocol,
        arm_results=arm_results,
        arm_metrics=trace.arm_metrics_by_id,
    )
    if replayed_decision._trace.to_payload() != trace.to_payload():
        raise ValueError("Decision trace differs from the locked-rule replay")

    selection_raw, selection = _read_regular_canonical_json(
        source / PREPROCESSING_SELECTION_FILENAME,
        label=PREPROCESSING_SELECTION_FILENAME,
    )
    selection_sha = hashlib.sha256(selection_raw).hexdigest()
    if selection_sha != _require_sha256(
        expected_artifact_sha256, field="expected_artifact_sha256"
    ):
        raise ValueError("Preprocessing selection artifact SHA mismatch")
    if set(selection) != _SELECTION_FIELDS:
        raise ValueError("Preprocessing selection artifact violates its closed schema")
    fixed = {
        "schema_version": PREPROCESSING_SELECTION_ARTIFACT_SCHEMA,
        "run_tier": PREPROCESSING_FORMAL_RUN_TIER,
        "source_scope": PREPROCESSING_SOURCE_SCOPE,
        "label_policy": PREPROCESSING_LABEL_POLICY,
        "o_ref_role": PREPROCESSING_OREF_ROLE,
        "nested_dev_manifest_file_sha256": nested_file_sha,
        "protocol_file_sha256": protocol_file_sha,
        "protocol_receipt_sha256": protocol.receipt_sha256,
        "source_patient_roster_sha256": protocol.source_patient_roster_sha256,
        "content_component_split_receipt_sha256": (
            protocol.content_component_split_receipt_sha256
        ),
        "raw_qc_intersection_receipt_sha256": (
            protocol.raw_qc_intersection_receipt_sha256
        ),
        "foundation_feature_receipt_sha256": (
            protocol.foundation_feature_receipt_sha256
        ),
        "selection_policy_receipt_sha256": (
            protocol.selection_policy_receipt_sha256
        ),
        "selection_policy_file_sha256": policy_file_sha,
        "decision_trace_file_sha256": trace_file_sha,
        "decision_trace_receipt_sha256": trace.receipt_sha256,
        "arm_specs_sha256": FROZEN_PREPROCESSING_ARM_SPECS_SHA256,
    }
    for field, expected in fixed.items():
        if selection.get(field) != expected:
            raise ValueError(f"Selection {field} differs from its protocol")
    _require_bool(selection.get("formal_complete"), field="formal_complete", expected=True)
    _require_bool(
        selection.get("legacy_formal_v3_tokens_authorized"),
        field="legacy_formal_v3_tokens_authorized",
        expected=False,
    )
    _require_bool(selection.get("smoke_is_formal"), field="smoke_is_formal", expected=False)
    forbidden = selection.get("forbidden_partitions")
    if not isinstance(forbidden, list) or tuple(forbidden) != PREPROCESSING_FORBIDDEN_PARTITIONS:
        raise ValueError("Selection forbidden partitions differ from the protocol")
    declared_file_hashes = _normalize_arm_hash_mapping(
        selection.get("arm_result_file_sha256_by_id", {}),
        field="arm_result_file_sha256_by_id",
    )
    if declared_file_hashes != arm_file_hashes:
        raise ValueError("Selection arm-result file hashes differ from bundle bytes")
    declared_result_hashes = _normalize_arm_hash_mapping(
        selection.get("arm_result_receipt_sha256_by_id", {}),
        field="arm_result_receipt_sha256_by_id",
    )
    actual_result_hashes = {
        arm_id: arm_results[arm_id].receipt_sha256 for arm_id in PREPROCESSING_ARM_IDS
    }
    if declared_result_hashes != actual_result_hashes:
        raise ValueError("Selection arm-result receipts differ from bundle payloads")
    decision_payload = selection.get("decision")
    if not isinstance(decision_payload, dict):
        raise TypeError("Selection decision must be a JSON object")
    decision = _decision_from_payload(
        decision_payload,
        protocol=protocol,
        arm_results=arm_results,
        trace=trace,
    )
    if decision.to_payload() != replayed_decision.to_payload():
        raise ValueError("Selection decision differs from the locked-rule replay")
    if selection.get("decision_receipt_sha256") != decision.receipt_sha256:
        raise ValueError("Selection decision receipt SHA mismatch")
    selected_arm_id = _require_arm_id(
        selection.get("selected_arm_id"), field="selected_arm_id"
    )
    if selected_arm_id != decision.selected_arm_id:
        raise ValueError("Selection arm differs from the evaluated decision")
    if selected_arm_id not in DEPLOYABLE_PREPROCESSING_ARM_IDS:
        raise ValueError("O-REF cannot authorize deployment")
    selected_result = arm_results[selected_arm_id]
    if (
        selection.get("selected_arm_spec_receipt_sha256")
        != FROZEN_PREPROCESSING_ARM_SPEC_BY_ID[selected_arm_id].receipt_sha256
    ):
        raise ValueError("Selected-arm spec receipt SHA mismatch")
    if (
        selection.get("selected_arm_result_receipt_sha256")
        != selected_result.receipt_sha256
    ):
        raise ValueError("Selected-arm result receipt SHA mismatch")

    receipt_raw, receipt = _read_regular_canonical_json(
        source / PREPROCESSING_SELECTION_RECEIPT_FILENAME,
        label=PREPROCESSING_SELECTION_RECEIPT_FILENAME,
    )
    if set(receipt) != _BUNDLE_RECEIPT_FIELDS:
        raise ValueError("Selection bundle receipt violates its closed schema")
    expected_receipt = {
        "schema_version": PREPROCESSING_SELECTION_BUNDLE_RECEIPT_SCHEMA,
        "selection_artifact_sha256": selection_sha,
        "nested_dev_manifest_file_sha256": nested_file_sha,
        "protocol_file_sha256": protocol_file_sha,
        "protocol_receipt_sha256": protocol.receipt_sha256,
        "selection_policy_file_sha256": policy_file_sha,
        "selection_policy_receipt_sha256": policy.receipt_sha256,
        "decision_trace_file_sha256": trace_file_sha,
        "decision_trace_receipt_sha256": trace.receipt_sha256,
        "arm_result_file_sha256_by_id": arm_file_hashes,
        "arm_result_receipt_sha256_by_id": actual_result_hashes,
        "decision_receipt_sha256": decision.receipt_sha256,
        "selected_arm_id": selected_arm_id,
        "selected_arm_result_receipt_sha256": selected_result.receipt_sha256,
    }
    if receipt != expected_receipt:
        raise ValueError("Selection bundle receipt does not bind its artifacts")
    receipt_sha = hashlib.sha256(receipt_raw).hexdigest()
    return _LoadedSelectionBundle(
        path=source,
        selection_artifact_sha256=selection_sha,
        bundle_receipt_sha256=receipt_sha,
        nested_dev_manifest_file_sha256=nested_file_sha,
        nested_dev_manifest=nested_manifest,
        protocol_file_sha256=protocol_file_sha,
        protocol=protocol,
        arm_results=MappingProxyType(dict(arm_results)),
        arm_result_file_sha256_by_id=MappingProxyType(dict(arm_file_hashes)),
        decision=decision,
        selected_arm_id=selected_arm_id,
        selected_arm_result_receipt_sha256=selected_result.receipt_sha256,
    )


@dataclass(frozen=True, init=False)
class PreprocessingProducerAuthorizationReceipt:
    producer_kind: str
    token_schema_version: str
    selected_arm_id: str
    selected_arm_spec_receipt_sha256: str
    selected_arm_result_receipt_sha256: str
    selection_artifact_sha256: str
    selection_bundle_receipt_sha256: str
    protocol_receipt_sha256: str
    nested_dev_manifest_receipt_sha256: str
    source_patient_roster_sha256: str
    foundation_feature_receipt_sha256: str
    raw_qc_intersection_receipt_sha256: str
    content_component_split_receipt_sha256: str
    schema_version: str

    def __init__(
        self,
        *,
        _issuer: object,
        producer_kind: str,
        token_schema_version: str,
        selected_arm_id: str,
        selected_arm_spec_receipt_sha256: str,
        selected_arm_result_receipt_sha256: str,
        selection_artifact_sha256: str,
        selection_bundle_receipt_sha256: str,
        protocol_receipt_sha256: str,
        nested_dev_manifest_receipt_sha256: str,
        source_patient_roster_sha256: str,
        foundation_feature_receipt_sha256: str,
        raw_qc_intersection_receipt_sha256: str,
        content_component_split_receipt_sha256: str,
    ) -> None:
        if _issuer is not _PRODUCER_AUTHORIZATION_ISSUER:
            raise TypeError(
                "PreprocessingProducerAuthorizationReceipt is issued only by "
                "a verified selection capability"
            )
        if producer_kind not in PREPROCESSING_PRODUCER_KINDS:
            raise ValueError(f"Unsupported preprocessing producer: {producer_kind!r}")
        if not isinstance(token_schema_version, str) or not token_schema_version.strip():
            raise ValueError("token_schema_version must be non-empty")
        if token_schema_version in LEGACY_FORMAL_V3_TOKEN_SCHEMAS:
            raise ValueError(
                "Legacy formal-v3 token corpora are candidate-only and cannot "
                "receive preprocessing authorization"
            )
        arm_id = _require_arm_id(selected_arm_id, field="selected_arm_id")
        if arm_id not in DEPLOYABLE_PREPROCESSING_ARM_IDS:
            raise ValueError("O-REF cannot authorize a producer")
        object.__setattr__(self, "producer_kind", producer_kind)
        object.__setattr__(self, "token_schema_version", token_schema_version)
        object.__setattr__(self, "selected_arm_id", arm_id)
        for field, value in (
            ("selected_arm_spec_receipt_sha256", selected_arm_spec_receipt_sha256),
            ("selected_arm_result_receipt_sha256", selected_arm_result_receipt_sha256),
            ("selection_artifact_sha256", selection_artifact_sha256),
            ("selection_bundle_receipt_sha256", selection_bundle_receipt_sha256),
            ("protocol_receipt_sha256", protocol_receipt_sha256),
            (
                "nested_dev_manifest_receipt_sha256",
                nested_dev_manifest_receipt_sha256,
            ),
            ("source_patient_roster_sha256", source_patient_roster_sha256),
            ("foundation_feature_receipt_sha256", foundation_feature_receipt_sha256),
            ("raw_qc_intersection_receipt_sha256", raw_qc_intersection_receipt_sha256),
            (
                "content_component_split_receipt_sha256",
                content_component_split_receipt_sha256,
            ),
        ):
            object.__setattr__(self, field, _require_sha256(value, field=field))
        object.__setattr__(
            self, "schema_version", PREPROCESSING_PRODUCER_AUTHORIZATION_SCHEMA
        )

    @property
    def receipt_sha256(self) -> str:
        return _typed_receipt_sha256(asdict(self))


@dataclass(frozen=True, init=False)
class AuthorizedPreprocessingSelection:
    """Producer-scoped authorization backed by a live verified capability."""

    receipt: PreprocessingProducerAuthorizationReceipt
    _capability: "VerifiedPreprocessingSelectionCapability"

    def __init__(
        self,
        *,
        _issuer: object,
        receipt: PreprocessingProducerAuthorizationReceipt,
        capability: "VerifiedPreprocessingSelectionCapability",
    ) -> None:
        if _issuer is not _PRODUCER_AUTHORIZATION_ISSUER:
            raise TypeError(
                "AuthorizedPreprocessingSelection can only be issued by a "
                "verified selection capability"
            )
        if not isinstance(receipt, PreprocessingProducerAuthorizationReceipt):
            raise TypeError("receipt has the wrong type")
        if not isinstance(capability, VerifiedPreprocessingSelectionCapability):
            raise TypeError("capability has the wrong type")
        object.__setattr__(self, "receipt", receipt)
        object.__setattr__(self, "_capability", capability)

    @property
    def selected_arm_id(self) -> str:
        return self.receipt.selected_arm_id

    @property
    def selected_arm_result_receipt_sha256(self) -> str:
        return self.receipt.selected_arm_result_receipt_sha256

    @property
    def selection_artifact_sha256(self) -> str:
        return self.receipt.selection_artifact_sha256

    @property
    def protocol_receipt_sha256(self) -> str:
        return self.receipt.protocol_receipt_sha256

    @property
    def foundation_feature_receipt_sha256(self) -> str:
        return self.receipt.foundation_feature_receipt_sha256

    @property
    def nested_dev_manifest_receipt_sha256(self) -> str:
        return self.receipt.nested_dev_manifest_receipt_sha256

    @property
    def source_patient_roster_sha256(self) -> str:
        return self.receipt.source_patient_roster_sha256

    def require_selected_arm(self, arm_id: str) -> None:
        self._capability.require_selected_arm(arm_id)

    def assert_unchanged(self) -> None:
        self._capability.assert_unchanged()
        expected = self._capability._producer_receipt(
            producer_kind=self.receipt.producer_kind,
            token_schema_version=self.receipt.token_schema_version,
        )
        if asdict(expected) != asdict(self.receipt):
            raise ValueError("Producer preprocessing authorization changed after issue")


@dataclass(frozen=True, init=False)
class VerifiedPreprocessingSelectionCapability:
    """Opaque capability issued only by the strict multi-file loader."""

    path: Path
    selection_artifact_sha256: str
    selection_bundle_receipt_sha256: str
    protocol_receipt_sha256: str
    nested_dev_manifest_receipt_sha256: str
    source_patient_roster_sha256: str
    foundation_feature_receipt_sha256: str
    raw_qc_intersection_receipt_sha256: str
    content_component_split_receipt_sha256: str
    selected_arm_id: str
    selected_arm_spec_receipt_sha256: str
    selected_arm_result_receipt_sha256: str
    arm_result_receipt_sha256_by_id: Mapping[str, str]

    def __init__(self, *, _issuer: object, loaded: _LoadedSelectionBundle) -> None:
        if _issuer is not _CAPABILITY_ISSUER:
            raise TypeError(
                "VerifiedPreprocessingSelectionCapability can only be issued "
                "by the strict selection loader"
            )
        if not isinstance(loaded, _LoadedSelectionBundle):
            raise TypeError("loaded selection has the wrong type")
        protocol = loaded.protocol
        selected_spec = FROZEN_PREPROCESSING_ARM_SPEC_BY_ID[loaded.selected_arm_id]
        object.__setattr__(self, "path", loaded.path)
        object.__setattr__(
            self, "selection_artifact_sha256", loaded.selection_artifact_sha256
        )
        object.__setattr__(
            self,
            "selection_bundle_receipt_sha256",
            loaded.bundle_receipt_sha256,
        )
        object.__setattr__(self, "protocol_receipt_sha256", protocol.receipt_sha256)
        object.__setattr__(
            self,
            "nested_dev_manifest_receipt_sha256",
            protocol.nested_dev_manifest_receipt_sha256,
        )
        object.__setattr__(
            self,
            "source_patient_roster_sha256",
            protocol.source_patient_roster_sha256,
        )
        object.__setattr__(
            self,
            "foundation_feature_receipt_sha256",
            protocol.foundation_feature_receipt_sha256,
        )
        object.__setattr__(
            self,
            "raw_qc_intersection_receipt_sha256",
            protocol.raw_qc_intersection_receipt_sha256,
        )
        object.__setattr__(
            self,
            "content_component_split_receipt_sha256",
            protocol.content_component_split_receipt_sha256,
        )
        object.__setattr__(self, "selected_arm_id", loaded.selected_arm_id)
        object.__setattr__(
            self, "selected_arm_spec_receipt_sha256", selected_spec.receipt_sha256
        )
        object.__setattr__(
            self,
            "selected_arm_result_receipt_sha256",
            loaded.selected_arm_result_receipt_sha256,
        )
        object.__setattr__(
            self,
            "arm_result_receipt_sha256_by_id",
            MappingProxyType(
                {
                    arm_id: loaded.arm_results[arm_id].receipt_sha256
                    for arm_id in PREPROCESSING_ARM_IDS
                }
            ),
        )

    def require_selected_arm(self, arm_id: str) -> None:
        normalized = _require_arm_id(arm_id)
        if normalized != self.selected_arm_id:
            raise ValueError(
                f"Producer requested arm {normalized}, but formal parity selected "
                f"{self.selected_arm_id}"
            )
        if normalized not in DEPLOYABLE_PREPROCESSING_ARM_IDS:
            raise ValueError("O-REF cannot authorize deployment")

    def assert_unchanged(self) -> None:
        loaded = _read_selection_bundle(
            self.path,
            expected_artifact_sha256=self.selection_artifact_sha256,
            expected_protocol_receipt_sha256=self.protocol_receipt_sha256,
        )
        replay = VerifiedPreprocessingSelectionCapability(
            _issuer=_CAPABILITY_ISSUER,
            loaded=loaded,
        )
        fields = (
            "selection_artifact_sha256",
            "selection_bundle_receipt_sha256",
            "protocol_receipt_sha256",
            "nested_dev_manifest_receipt_sha256",
            "source_patient_roster_sha256",
            "foundation_feature_receipt_sha256",
            "raw_qc_intersection_receipt_sha256",
            "content_component_split_receipt_sha256",
            "selected_arm_id",
            "selected_arm_spec_receipt_sha256",
            "selected_arm_result_receipt_sha256",
        )
        if any(getattr(self, field) != getattr(replay, field) for field in fields):
            raise ValueError("Verified preprocessing selection changed after load")
        if dict(self.arm_result_receipt_sha256_by_id) != dict(
            replay.arm_result_receipt_sha256_by_id
        ):
            raise ValueError("Verified preprocessing arm results changed after load")

    def _producer_receipt(
        self, *, producer_kind: str, token_schema_version: str
    ) -> PreprocessingProducerAuthorizationReceipt:
        return PreprocessingProducerAuthorizationReceipt(
            _issuer=_PRODUCER_AUTHORIZATION_ISSUER,
            producer_kind=producer_kind,
            token_schema_version=token_schema_version,
            selected_arm_id=self.selected_arm_id,
            selected_arm_spec_receipt_sha256=(
                self.selected_arm_spec_receipt_sha256
            ),
            selected_arm_result_receipt_sha256=(
                self.selected_arm_result_receipt_sha256
            ),
            selection_artifact_sha256=self.selection_artifact_sha256,
            selection_bundle_receipt_sha256=(
                self.selection_bundle_receipt_sha256
            ),
            protocol_receipt_sha256=self.protocol_receipt_sha256,
            nested_dev_manifest_receipt_sha256=(
                self.nested_dev_manifest_receipt_sha256
            ),
            source_patient_roster_sha256=self.source_patient_roster_sha256,
            foundation_feature_receipt_sha256=(
                self.foundation_feature_receipt_sha256
            ),
            raw_qc_intersection_receipt_sha256=(
                self.raw_qc_intersection_receipt_sha256
            ),
            content_component_split_receipt_sha256=(
                self.content_component_split_receipt_sha256
            ),
        )

    def authorize_producer(
        self,
        *,
        arm_id: str,
        expected_arm_result_receipt_sha256: str,
        producer_kind: str,
        token_schema_version: str,
    ) -> AuthorizedPreprocessingSelection:
        self.assert_unchanged()
        self.require_selected_arm(arm_id)
        expected = _require_sha256(
            expected_arm_result_receipt_sha256,
            field="expected_arm_result_receipt_sha256",
        )
        if expected != self.selected_arm_result_receipt_sha256:
            raise ValueError("Producer pinned the wrong selected-arm result receipt")
        receipt = self._producer_receipt(
            producer_kind=producer_kind,
            token_schema_version=token_schema_version,
        )
        return AuthorizedPreprocessingSelection(
            _issuer=_PRODUCER_AUTHORIZATION_ISSUER,
            receipt=receipt,
            capability=self,
        )


def load_preprocessing_selection_capability(
    directory: str | Path,
    *,
    expected_artifact_sha256: str,
    expected_protocol_receipt_sha256: str,
) -> VerifiedPreprocessingSelectionCapability:
    """Strictly replay a complete formal five-arm selection bundle."""

    loaded = _read_selection_bundle(
        directory,
        expected_artifact_sha256=expected_artifact_sha256,
        expected_protocol_receipt_sha256=expected_protocol_receipt_sha256,
    )
    return VerifiedPreprocessingSelectionCapability(
        _issuer=_CAPABILITY_ISSUER,
        loaded=loaded,
    )


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def materialize_preprocessing_selection_bundle(
    *,
    nested_dev_manifest: PreprocessingParityNestedDevManifest,
    protocol: PreprocessingParityProtocolReceipt,
    arm_results: Mapping[str, PreprocessingArmResultReceipt],
    decision: FormalPreprocessingSelectionDecision,
    output_directory: str | Path,
) -> VerifiedPreprocessingSelectionCapability:
    """Atomically publish a numerical-evaluator-issued formal selection."""

    if not isinstance(nested_dev_manifest, PreprocessingParityNestedDevManifest):
        raise TypeError(
            "nested_dev_manifest must be PreprocessingParityNestedDevManifest"
        )
    protocol.require_nested_dev_manifest(nested_dev_manifest)
    normalized_results = _validate_complete_result_set(protocol, arm_results)
    if not isinstance(decision, FormalPreprocessingSelectionDecision):
        raise TypeError("decision must be FormalPreprocessingSelectionDecision")
    if decision.protocol_receipt_sha256 != protocol.receipt_sha256:
        raise ValueError("Selection decision binds the wrong protocol")
    if (
        decision.selection_policy_receipt_sha256
        != protocol.selection_policy_receipt_sha256
    ):
        raise ValueError("Selection decision binds the wrong selection policy")
    result_hashes = {
        arm_id: normalized_results[arm_id].receipt_sha256
        for arm_id in PREPROCESSING_ARM_IDS
    }
    if dict(decision.arm_result_receipt_sha256_by_id) != result_hashes:
        raise ValueError("Selection decision binds the wrong arm-result set")
    if decision.selected_arm_id not in DEPLOYABLE_PREPROCESSING_ARM_IDS:
        raise ValueError("O-REF cannot be selected for deployment")
    replayed_decision = evaluate_preprocessing_arm_selection(
        protocol=protocol,
        arm_results=normalized_results,
        arm_metrics=decision._trace.arm_metrics_by_id,
    )
    if replayed_decision.to_payload() != decision.to_payload():
        raise ValueError("Selection decision differs from the locked-rule replay")
    if replayed_decision._trace.to_payload() != decision._trace.to_payload():
        raise ValueError("Selection trace differs from the locked-rule replay")

    target = _strict_absolute_directory(output_directory, output=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(target.parent))
    )
    published = False
    try:
        nested_raw = _canonical_json_bytes(nested_dev_manifest.to_payload())
        nested_path = staging / PREPROCESSING_NESTED_DEV_MANIFEST_FILENAME
        nested_path.write_bytes(nested_raw)
        nested_file_sha = hashlib.sha256(nested_raw).hexdigest()

        protocol_raw = _canonical_json_bytes(asdict(protocol))
        protocol_path = staging / PREPROCESSING_PROTOCOL_FILENAME
        protocol_path.write_bytes(protocol_raw)
        protocol_file_sha = hashlib.sha256(protocol_raw).hexdigest()

        policy_raw = _canonical_json_bytes(
            asdict(LOCKED_PREPROCESSING_SELECTION_POLICY)
        )
        policy_path = staging / PREPROCESSING_SELECTION_POLICY_FILENAME
        policy_path.write_bytes(policy_raw)
        policy_file_sha = hashlib.sha256(policy_raw).hexdigest()

        arm_file_hashes: dict[str, str] = {}
        for arm_id in PREPROCESSING_ARM_IDS:
            raw = _canonical_json_bytes(asdict(normalized_results[arm_id]))
            path = staging / _ARM_RESULT_FILENAME_BY_ID[arm_id]
            path.write_bytes(raw)
            arm_file_hashes[arm_id] = hashlib.sha256(raw).hexdigest()

        trace_raw = _canonical_json_bytes(decision._trace.to_payload())
        trace_path = staging / PREPROCESSING_SELECTION_TRACE_FILENAME
        trace_path.write_bytes(trace_raw)
        trace_file_sha = hashlib.sha256(trace_raw).hexdigest()

        selected_arm_id = decision.selected_arm_id
        selected_result = normalized_results[selected_arm_id]
        selection_payload = {
            "schema_version": PREPROCESSING_SELECTION_ARTIFACT_SCHEMA,
            "formal_complete": True,
            "run_tier": PREPROCESSING_FORMAL_RUN_TIER,
            "source_scope": PREPROCESSING_SOURCE_SCOPE,
            "forbidden_partitions": list(PREPROCESSING_FORBIDDEN_PARTITIONS),
            "label_policy": PREPROCESSING_LABEL_POLICY,
            "o_ref_role": PREPROCESSING_OREF_ROLE,
            "nested_dev_manifest_file_sha256": nested_file_sha,
            "protocol_file_sha256": protocol_file_sha,
            "protocol_receipt_sha256": protocol.receipt_sha256,
            "source_patient_roster_sha256": protocol.source_patient_roster_sha256,
            "content_component_split_receipt_sha256": (
                protocol.content_component_split_receipt_sha256
            ),
            "raw_qc_intersection_receipt_sha256": (
                protocol.raw_qc_intersection_receipt_sha256
            ),
            "foundation_feature_receipt_sha256": (
                protocol.foundation_feature_receipt_sha256
            ),
            "selection_policy_receipt_sha256": (
                protocol.selection_policy_receipt_sha256
            ),
            "selection_policy_file_sha256": policy_file_sha,
            "decision_trace_file_sha256": trace_file_sha,
            "decision_trace_receipt_sha256": decision._trace.receipt_sha256,
            "arm_specs_sha256": FROZEN_PREPROCESSING_ARM_SPECS_SHA256,
            "arm_result_file_sha256_by_id": arm_file_hashes,
            "arm_result_receipt_sha256_by_id": result_hashes,
            "decision": decision.to_payload(),
            "decision_receipt_sha256": decision.receipt_sha256,
            "selected_arm_id": selected_arm_id,
            "selected_arm_spec_receipt_sha256": (
                FROZEN_PREPROCESSING_ARM_SPEC_BY_ID[selected_arm_id].receipt_sha256
            ),
            "selected_arm_result_receipt_sha256": selected_result.receipt_sha256,
            "legacy_formal_v3_tokens_authorized": False,
            "smoke_is_formal": False,
        }
        selection_raw = _canonical_json_bytes(selection_payload)
        selection_sha = hashlib.sha256(selection_raw).hexdigest()
        selection_path = staging / PREPROCESSING_SELECTION_FILENAME
        selection_path.write_bytes(selection_raw)

        receipt_payload = {
            "schema_version": PREPROCESSING_SELECTION_BUNDLE_RECEIPT_SCHEMA,
            "selection_artifact_sha256": selection_sha,
            "nested_dev_manifest_file_sha256": nested_file_sha,
            "protocol_file_sha256": protocol_file_sha,
            "protocol_receipt_sha256": protocol.receipt_sha256,
            "selection_policy_file_sha256": policy_file_sha,
            "selection_policy_receipt_sha256": (
                LOCKED_PREPROCESSING_SELECTION_POLICY_RECEIPT_SHA256
            ),
            "decision_trace_file_sha256": trace_file_sha,
            "decision_trace_receipt_sha256": decision._trace.receipt_sha256,
            "arm_result_file_sha256_by_id": arm_file_hashes,
            "arm_result_receipt_sha256_by_id": result_hashes,
            "decision_receipt_sha256": decision.receipt_sha256,
            "selected_arm_id": selected_arm_id,
            "selected_arm_result_receipt_sha256": selected_result.receipt_sha256,
        }
        receipt_path = staging / PREPROCESSING_SELECTION_RECEIPT_FILENAME
        receipt_path.write_bytes(_canonical_json_bytes(receipt_payload))
        for path in staging.iterdir():
            _fsync_file(path)
        _fsync_directory(staging)

        load_preprocessing_selection_capability(
            staging,
            expected_artifact_sha256=selection_sha,
            expected_protocol_receipt_sha256=protocol.receipt_sha256,
        )
        if os.path.lexists(target):
            raise FileExistsError(f"Selection output already exists: {target}")
        os.rename(staging, target)
        published = True
        _fsync_directory(target.parent)
        return load_preprocessing_selection_capability(
            target,
            expected_artifact_sha256=selection_sha,
            expected_protocol_receipt_sha256=protocol.receipt_sha256,
        )
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


__all__ = [
    "AuthorizedPreprocessingSelection",
    "DEPLOYABLE_PREPROCESSING_ARM_IDS",
    "FROZEN_PREPROCESSING_ARM_SPEC_BY_ID",
    "FROZEN_PREPROCESSING_ARM_SPECS",
    "FROZEN_PREPROCESSING_ARM_SPECS_SHA256",
    "FormalPreprocessingSelectionDecision",
    "LEGACY_FORMAL_V3_TOKEN_SCHEMAS",
    "LOCKED_PREPROCESSING_SELECTION_POLICY",
    "LOCKED_PREPROCESSING_SELECTION_POLICY_RECEIPT_SHA256",
    "OFFICIAL_REF23_CHANNELS",
    "PREPROCESSING_ARM_IDS",
    "PREPROCESSING_ARM_RESULT_SCHEMA",
    "PREPROCESSING_ARM_SELECTION_METRICS_SCHEMA",
    "PREPROCESSING_DEPLOYABLE_TIE_BREAK",
    "PREPROCESSING_FORMAL_RUN_TIER",
    "PREPROCESSING_NESTED_DEV_MANIFEST_SCHEMA",
    "PREPROCESSING_NESTED_DEV_RECORD_SCHEMA",
    "PREPROCESSING_PARITY_PROTOCOL_SCHEMA",
    "PREPROCESSING_PRODUCER_AUTHORIZATION_SCHEMA",
    "PREPROCESSING_PRODUCER_KINDS",
    "PREPROCESSING_SELECTION_ARTIFACT_SCHEMA",
    "PREPROCESSING_SELECTION_BUNDLE_RECEIPT_SCHEMA",
    "PREPROCESSING_SELECTION_DECISION_SCHEMA",
    "PREPROCESSING_SELECTION_NO_GO_SCHEMA",
    "PREPROCESSING_SELECTION_POLICY_SCHEMA",
    "PREPROCESSING_SELECTION_TRACE_SCHEMA",
    "PreprocessingArmResultReceipt",
    "PreprocessingArmSelectionNoGoError",
    "PreprocessingArmSelectionMetrics",
    "PreprocessingArmSpec",
    "PreprocessingNestedDevSourceRecord",
    "PreprocessingParityNestedDevManifest",
    "PreprocessingParityProtocolReceipt",
    "PreprocessingProducerAuthorizationReceipt",
    "PreprocessingSelectionDecisionTrace",
    "PreprocessingSelectionPolicy",
    "VerifiedPreprocessingSelectionCapability",
    "build_preprocessing_parity_nested_dev_manifest",
    "evaluate_preprocessing_arm_selection",
    "load_preprocessing_selection_capability",
    "materialize_preprocessing_selection_bundle",
    "preprocessing_foundation_policy_receipt_sha256",
]
